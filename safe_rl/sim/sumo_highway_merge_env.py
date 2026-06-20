from __future__ import annotations

import json
import math
import os
import sys
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from safe_rl.prediction.actor_selector import select_merge_relevant_actors
from safe_rl.prediction.forecast_feature_augmentor import ForecastFeatureAugmentor
from safe_rl.risk.candidate_risk_ranker import CandidateRiskRanker
from safe_rl.risk.merge_local import is_candidate_legal, merge_local_stats
from safe_rl.shield.forecast_task_scorer import ForecastAwareTaskScorer
from safe_rl.shield.safety_shield import SafetyShield
from safe_rl.sim.action_space import ACTIONS, decode_action
from safe_rl.sim.gym_compat import gym, spaces
from safe_rl.sim.history_buffer import HistoryBuffer
from safe_rl.sim.metrics import SAFETY_METRIC_VERSION, INF_TTC, compute_step_metrics, explicit_risk_features
from safe_rl.sim.scenario_semantics import (
    auxiliary_lane_index,
    distance_to_taper,
    edge_role,
    is_auxiliary_edge,
    is_ramp_edge,
    is_taper_miss,
    is_target_lane,
    merge_target_lane,
    merge_zone_edges,
    target_lane_edges,
    target_lane_index,
    target_lane_mapping,
)
from safe_rl.sim.types import StepMetrics, VehicleState
from safe_rl.utils.performance import PerformanceTracker


def scheduled_episode_seed(
    base_seed: int,
    worker_rank: int,
    episode_index: int,
    num_envs: int,
) -> int:
    """Return the versioned incrementing-v1 training seed."""

    return (
        int(base_seed)
        + int(worker_rank)
        + int(episode_index) * max(1, int(num_envs))
    )


def configured_trajectory_actor_capacity(config: Any) -> int:
    """Return non-ego actor slots needed by enabled trajectory predictors."""

    scenario_top_k = int(config.scenario.get("top_k_neighbors", 5))
    prediction_cfg = config.get("prediction", {}) or {}
    capacities = [scenario_top_k]
    train_enabled = bool(prediction_cfg.get("train_enabled", True))
    if train_enabled and bool(prediction_cfg.get("wcdt_v1_train_enabled", False)):
        capacities.append(int(prediction_cfg.get("max_pred_num", scenario_top_k)))
    if train_enabled and bool(prediction_cfg.get("wcdt_v2_train_enabled", False)):
        capacities.append(int(prediction_cfg.get("wcdt_v2_max_agents", scenario_top_k)))
    if train_enabled and bool(prediction_cfg.get("wcdt_v3_train_enabled", False)):
        capacities.append(int(prediction_cfg.get("wcdt_v3_max_agents", scenario_top_k)))
    forecast_cfg = config.get("forecast_features", {}) or {}
    if bool(forecast_cfg.get("enabled", False)):
        source = str(forecast_cfg.get("source", "")).lower()
        if source == "wcdt_v2":
            capacities.append(int(prediction_cfg.get("wcdt_v2_max_agents", scenario_top_k)))
        elif source == "wcdt_v3":
            capacities.append(int(prediction_cfg.get("wcdt_v3_max_agents", scenario_top_k)))
        elif source == "wcdt":
            capacities.append(int(prediction_cfg.get("max_pred_num", scenario_top_k)))
    return max(1, max(int(value) for value in capacities))


def _actor_metadata_json(selection: Any, vehicle_ids: tuple[str, ...] | list[str]) -> str:
    rows: list[dict[str, Any]] = []
    for vehicle_id in vehicle_ids:
        metadata = selection.actor_metadata.get(str(vehicle_id))
        if metadata is None:
            continue
        rows.append(
            {
                "vehicle_id": str(vehicle_id),
                "role": str(metadata.role),
                "route_progress": metadata.route_progress,
                "signed_longitudinal_gap": metadata.signed_longitudinal_gap,
                "current_surface_gap": float(metadata.current_surface_gap),
                "closing_speed": float(metadata.closing_speed),
                "effective_gap": float(metadata.effective_gap),
                "ttc": float(metadata.ttc),
                "relevance_reasons": list(metadata.relevance_reasons),
                "relevance_class": str(metadata.relevance_class),
                "critical": bool(metadata.critical),
                "contextual": bool(metadata.contextual),
                "selection_priority": [float(item) for item in metadata.selection_priority],
            }
        )
    return json.dumps(rows, ensure_ascii=False, sort_keys=True, allow_nan=False)


class SumoHighwayMergeEnv(gym.Env):
    """Gymnasium-compatible SUMO highway-merge environment.

    The class intentionally imports TraCI lazily. Importing this module should work in
    environments used for static checks, while running the environment requires SUMO
    Python tools on PYTHONPATH.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        config: Any,
        seed: int | None = None,
        forecast_augmentor: ForecastFeatureAugmentor | None = None,
        shield: SafetyShield | None = None,
        reward_risk_model: Any | None = None,
        record_trajectory_samples: bool = False,
        sumo_step_delay_ms: float = 0.0,
        worker_rank: int = 0,
        num_envs: int = 1,
        advance_episode_seed: bool = False,
    ):
        self.config = config
        self._base_seed = int(seed if seed is not None else config.run.seed)
        self.seed_value = self._base_seed
        self.worker_rank = int(worker_rank)
        self.num_envs = max(1, int(num_envs))
        self.advance_episode_seed = bool(advance_episode_seed)
        self._reset_count = 0
        self._episode_index = 0
        self._active_episode_index = -1
        self.ego_id = config.scenario.ego_id
        self.step_length = float(config.scenario.step_length)
        self.control_interval_steps = int(config.scenario.control_interval_steps)
        self.episode_steps = int(float(config.scenario.episode_seconds) / self.step_length)
        self.top_k = int(config.scenario.top_k_neighbors)
        self.trajectory_actor_capacity = configured_trajectory_actor_capacity(config)
        self.history_steps = int(config.scenario.history_steps)
        self.forecast_enabled = bool(config.forecast_features.enabled or config.rl.use_wcdt_forecast_features)
        self.forecast_augmentor = forecast_augmentor
        self.shield = shield
        self.reward_risk_model = reward_risk_model
        self.reward_ranker = CandidateRiskRanker(config, reward_risk_model) if reward_risk_model is not None else None
        task_predictor = getattr(forecast_augmentor, "predictor", None) if forecast_augmentor is not None else None
        self.forecast_task_scorer = ForecastAwareTaskScorer(config, task_predictor)
        self.record_trajectory_samples = record_trajectory_samples
        self.sumo_step_delay_ms = float(sumo_step_delay_ms)

        self.action_space = spaces.Discrete(len(ACTIONS))
        self._base_obs_dim = 8 + self.top_k * 8 + 4
        self._forecast_dim = ForecastFeatureAugmentor.feature_dim(config) if self.forecast_enabled else 0
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self._base_obs_dim + self._forecast_dim,),
            dtype=np.float32,
        )

        self.history = HistoryBuffer(
            self.history_steps,
            max_agents=max(self.top_k, self.trajectory_actor_capacity) + 1,
        )
        self._traci_module = None
        self._traci = None
        self._conn_label = f"safe_rl_{uuid.uuid4().hex[:8]}"
        self._episode_step = 0
        self._decision_index = 0
        self._simulation_step_index = 0
        self._last_ego_speed = 0.0
        self._last_ego_x = 0.0
        self._episode_metrics: list[StepMetrics] = []
        self._ego_speeds: list[float] = []
        self._interventions: list[dict[str, Any]] = []
        self._action_execution_records: list[dict[str, Any]] = []
        self._reward_debug_records: list[dict[str, Any]] = []
        self._last_reward_debug: dict[str, Any] = {}
        self._reward_component_records: list[dict[str, float]] = []
        self._raw_action_lane_oob_count = 0
        self._final_action_lane_oob_count = 0
        self._prevented_lane_oob_count = 0
        self._trajectory_frames: list[dict[str, VehicleState]] = []
        self._trajectory_frame_metadata: list[dict[str, int]] = []
        self._last_trajectory_window_metadata: dict[str, np.ndarray] = {}
        self._last_done_reason = ""
        self._curriculum_profile = "disabled"
        self._curriculum_applied = False
        self._first_merge_request_step: int | None = None
        self._first_merge_request_distance_to_taper: float | None = None
        self._first_target_lane_entry_step: int | None = None
        self._first_target_lane_entry_distance_to_taper: float | None = None
        self._safe_merge_opportunity_count = 0
        self._missed_safe_merge_opportunity_count = 0
        self._task_merge_records: list[dict[str, Any]] = []
        self._last_task_merge_record: dict[str, Any] = {}
        self._task_missed_consecutive_count = 0
        self._task_backstop_consecutive_count = 0
        self._task_replacements: list[dict[str, Any]] = []
        self._forecast_ranking_replacements: list[dict[str, Any]] = []
        self.performance = PerformanceTracker()
        self._decision_context_cache: dict[str, Any] | None = None
        self._lane_count_cache: dict[str, int] = {}
        self._subscribed_vehicle_ids: set[str] = set()
        self._subscription_fallback_count = 0
        self._subscription_error_count = 0
        self._sumo_reload_count = 0
        self._sumo_restart_count = 0

    def _import_traci(self):
        if self._traci_module is not None:
            return self._traci_module
        self._add_sumo_tools_path()
        try:
            import traci
        except ImportError as exc:  # pragma: no cover - depends on SUMO install
            raise ImportError(
                "Running SumoHighwayMergeEnv requires SUMO Python tools. "
                "Install/configure traci and sumolib, or activate the SAFE_RL environment."
            ) from exc
        expected_path = str(
            dict(self.config.scenario.get("sumo_installation_fingerprint", {}) or {}).get(
                "traci_module_path",
                "",
            )
        )
        if expected_path:
            loaded_path = Path(str(getattr(traci, "__file__", ""))).resolve()
            if loaded_path != Path(expected_path).resolve():
                raise RuntimeError(
                    "TraCI module does not match the selected SUMO installation: "
                    f"loaded={loaded_path}, expected={Path(expected_path).resolve()}"
                )
        self._traci_module = traci
        return traci

    def _add_sumo_tools_path(self) -> None:
        candidates: list[Path] = []
        configured_tools = self.config.scenario.get("sumo_tools_directory")
        if configured_tools:
            candidates.append(Path(str(configured_tools)))
        if os.environ.get("SUMO_HOME"):
            candidates.append(Path(os.environ["SUMO_HOME"]) / "tools")
        sumo_binary = Path(str(self.config.scenario.get("sumo_binary", "sumo")))
        if sumo_binary.is_absolute() and sumo_binary.exists():
            candidates.append(sumo_binary.resolve().parents[1] / "tools")
        for candidate in candidates:
            if candidate.is_dir():
                resolved = str(candidate.resolve())
                sys.path[:] = [
                    item
                    for item in sys.path
                    if str(Path(item).resolve()) != resolved
                ]
                sys.path.insert(0, resolved)
                break

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        if seed is not None:
            self.seed_value = int(seed)
            self._active_episode_index = int(self._episode_index)
            if self.advance_episode_seed:
                self._episode_index += 1
        elif self.advance_episode_seed:
            self._active_episode_index = int(self._episode_index)
            self.seed_value = scheduled_episode_seed(
                self._base_seed,
                self.worker_rank,
                self._active_episode_index,
                self.num_envs,
            )
            self._episode_index += 1
        else:
            self._active_episode_index = 0
        persistent = bool(self.config.scenario.get("persistent_sumo_reset", False))
        restart_interval = max(1, int(self.config.scenario.get("persistent_sumo_restart_interval", 100)))
        should_reload = self._traci is not None and persistent and self._reset_count % restart_interval != 0
        if should_reload:
            try:
                self._reload_sumo()
            except Exception:
                self._close_sumo()
                self._start_sumo()
        else:
            self._close_sumo()
            self._start_sumo()
        self._reset_count += 1
        self.history.clear()
        self._invalidate_decision_cache()
        self._lane_count_cache.clear()
        self._episode_step = 0
        self._decision_index = 0
        self._simulation_step_index = 0
        self._episode_metrics.clear()
        self._ego_speeds.clear()
        self._interventions.clear()
        self._action_execution_records.clear()
        self._reward_debug_records.clear()
        self._last_reward_debug = {}
        self._reward_component_records.clear()
        self._raw_action_lane_oob_count = 0
        self._final_action_lane_oob_count = 0
        self._prevented_lane_oob_count = 0
        self._trajectory_frames.clear()
        self._trajectory_frame_metadata.clear()
        self._last_trajectory_window_metadata = {}
        self._last_done_reason = ""
        self._curriculum_profile = self._select_curriculum_profile()
        self._curriculum_applied = False
        self._first_merge_request_step = None
        self._first_merge_request_distance_to_taper = None
        self._first_target_lane_entry_step = None
        self._first_target_lane_entry_distance_to_taper = None
        self._safe_merge_opportunity_count = 0
        self._missed_safe_merge_opportunity_count = 0
        self._task_merge_records.clear()
        self._last_task_merge_record = {}
        self._task_missed_consecutive_count = 0
        self._task_backstop_consecutive_count = 0
        self._task_replacements.clear()
        self._forecast_ranking_replacements.clear()
        if self.shield is not None and hasattr(self.shield, "reset_episode_state"):
            self.shield.reset_episode_state()

        for _ in range(max(1, self.history_steps)):
            self._simulation_step()
            if not self._curriculum_applied:
                self._apply_curriculum_perturbation()
            self._configure_ego_control()
            states = self._collect_states()
            self.history.append(states)
            self._invalidate_decision_cache()
            self._append_trajectory_frame(states, decision_index=-1)
            if self.ego_id in self.history.latest():
                break

        ego = self._get_ego()
        self._last_ego_speed = ego.speed if ego else 0.0
        self._last_ego_x = ego.x if ego else 0.0
        return self._build_observation(), self._info()

    def step(self, action):
        decision_index = int(self._decision_index)
        raw_action = decode_action(int(action))
        final_action = raw_action
        intervention = None
        context = self.get_risk_context()
        self._record_merge_opportunity(context, raw_action)
        if self.shield is not None and self.shield.enabled:
            final_action, intervention = self.shield.select_action(raw_action, context)
            intervention["step"] = int(self._episode_step)
            intervention["decision_index"] = decision_index
            self._interventions.append(intervention)
        safety_shield_action = final_action
        forecast_replacement = self._maybe_forecast_aware_replacement(
            raw_action,
            final_action,
            context,
            intervention,
        )
        task_replacement = None
        forecast_ranking_replacement = None
        if forecast_replacement is not None:
            final_action = decode_action(int(forecast_replacement["final_action"]))
            if str(forecast_replacement.get("replacement_type", "")) == "task_backstop":
                task_replacement = forecast_replacement
                self._task_replacements.append(forecast_replacement)
            else:
                forecast_ranking_replacement = forecast_replacement
                self._forecast_ranking_replacements.append(forecast_replacement)

        safety_shield_replaced = bool(
            intervention is not None
            and int(intervention.get("final_action", raw_action.index)) != int(raw_action.index)
        )
        execution_path = "policy"
        if safety_shield_replaced:
            execution_path = "safety_shield"
        elif task_replacement is not None:
            execution_path = "task_backstop"
        elif forecast_ranking_replacement is not None:
            execution_path = "forecast_ranking"
        execution_record = {
            "step": int(self._episode_step),
            "decision_index": decision_index,
            "raw_action": int(raw_action.index),
            "raw_action_name": str(raw_action.name),
            "safety_shield_action": int(safety_shield_action.index),
            "safety_shield_action_name": str(safety_shield_action.name),
            "safety_shield_replaced": safety_shield_replaced,
            "safety_shield_replacement_reason": str(
                intervention.get("replacement_reason", "") if intervention is not None else ""
            ),
            "final_action": int(final_action.index),
            "final_action_name": str(final_action.name),
            "execution_path": execution_path,
            "task_replacement": bool(task_replacement is not None),
            "forecast_ranking_replacement": bool(forecast_ranking_replacement is not None),
        }
        self._action_execution_records.append(execution_record)

        raw_action_lane_oob = self._action_lane_oob(raw_action)
        final_action_lane_oob = self._action_lane_oob(final_action)
        self._raw_action_lane_oob_count += int(raw_action_lane_oob)
        self._final_action_lane_oob_count += int(final_action_lane_oob)
        self._prevented_lane_oob_count += int(raw_action_lane_oob and not final_action_lane_oob)
        lane_oob = self._apply_action(final_action)
        prev_ego = self._get_ego()
        prev_x = prev_ego.x if prev_ego else self._last_ego_x

        collision = False
        for _ in range(self.control_interval_steps):
            self._simulation_step()
            self._episode_step += 1
            collision = collision or self._ego_in_collision()
            states = self._collect_states()
            self.history.append(states)
            self._invalidate_decision_cache()
            if self.record_trajectory_samples:
                self._append_trajectory_frame(states, decision_index=decision_index)

        ego = self._get_ego()
        metrics = compute_step_metrics(
            ego,
            states,
            collision=collision,
            near_miss_threshold=float(self.config.risk_module.near_miss_distance_threshold),
            ttc_threshold=float(self.config.risk_module.ttc_threshold),
            drac_threshold=float(self.config.risk_module.drac_threshold),
            lane_oob=lane_oob,
            merge_ego_edges=merge_zone_edges(self.config),
            merge_target_edges=target_lane_edges(self.config),
            merge_target_lane=merge_target_lane(self.config),
            merge_target_lanes=target_lane_mapping(self.config),
        )
        self._episode_metrics.append(metrics)
        if intervention is not None:
            intervention.update(
                {
                    "min_distance": float(metrics.min_distance),
                    "min_ttc": float(metrics.min_ttc),
                    "max_drac": float(metrics.max_drac),
                    "geometric_overlap": bool(metrics.geometric_overlap),
                    "closest_vehicle_id": str(metrics.closest_vehicle_id),
                }
            )
        if forecast_replacement is not None:
            forecast_replacement.update(
                {
                    "min_distance": float(metrics.min_distance),
                    "min_ttc": float(metrics.min_ttc),
                    "max_drac": float(metrics.max_drac),
                    "geometric_overlap": bool(metrics.geometric_overlap),
                    "closest_vehicle_id": str(metrics.closest_vehicle_id),
                }
            )
        execution_record.update(
            {
                "min_distance": float(metrics.min_distance),
                "min_ttc": float(metrics.min_ttc),
                "max_drac": float(metrics.max_drac),
                "geometric_overlap": bool(metrics.geometric_overlap),
                "closest_vehicle_id": str(metrics.closest_vehicle_id),
            }
        )
        if ego is not None:
            self._ego_speeds.append(float(ego.speed))
            self._record_target_lane_entry(ego)

        terminated, done_reason = self._done(metrics)
        self._last_done_reason = done_reason
        truncated = self._episode_step >= self.episode_steps
        reward = self._reward(prev_x, ego, metrics, done_reason, raw_action=raw_action, risk_context=context)
        obs = self._build_observation()
        info = self._info(
            metrics=metrics,
            done_reason=done_reason,
            intervention=intervention,
            task_replacement=task_replacement,
            forecast_ranking_replacement=forecast_ranking_replacement,
        )
        info["raw_action"] = int(raw_action.index)
        info["final_action"] = int(final_action.index)
        info["raw_action_name"] = str(raw_action.name)
        info["final_action_name"] = str(final_action.name)
        info["safety_shield_action"] = int(safety_shield_action.index)
        info["safety_shield_action_name"] = str(safety_shield_action.name)
        info["safety_shield_replaced"] = safety_shield_replaced
        info["action_execution_path"] = execution_path
        info["raw_action_lane_oob"] = bool(raw_action_lane_oob)
        info["final_action_lane_oob"] = bool(final_action_lane_oob)
        info["prevented_lane_oob"] = bool(raw_action_lane_oob and not final_action_lane_oob)
        info["reward_components"] = dict(self._reward_component_records[-1]) if self._reward_component_records else {}
        info["decision_index"] = decision_index
        self._last_ego_speed = ego.speed if ego else 0.0
        self._last_ego_x = ego.x if ego else self._last_ego_x
        self._decision_index += 1
        return obs, float(reward), bool(terminated), bool(truncated), info

    def close(self):
        self._close_sumo()

    def _start_sumo(self) -> None:
        traci = self._import_traci()
        sumo_binary = self.config.scenario.get("sumo_binary", "sumo")
        cmd = [sumo_binary, *self._sumo_load_args()]
        retries = int(self.config.scenario.get("sumo_start_retries", 5))
        delay = float(self.config.scenario.get("sumo_start_retry_delay", 0.25))
        last_error: Exception | None = None
        with self.performance.measure("sumo_start_or_load_time"):
            for attempt in range(max(1, retries)):
                try:
                    self._conn_label = f"safe_rl_{uuid.uuid4().hex[:8]}"
                    traci.start(cmd, label=self._conn_label, numRetries=20)
                    self._traci = traci.getConnection(self._conn_label)
                    self._sumo_restart_count += 1
                    self.performance.increment("sumo_restarts")
                    self._reset_subscription_state()
                    return
                except Exception as exc:
                    last_error = exc
                    self._cleanup_failed_traci_start(traci)
                    time.sleep(delay * (attempt + 1))
        raise RuntimeError(f"Failed to start SUMO after {retries} attempts: {last_error}") from last_error

    def _sumo_load_args(self) -> list[str]:
        sumocfg = str(Path(self.config.scenario.sumocfg).resolve())
        return [
            "-c",
            sumocfg,
            "--seed",
            str(self.seed_value),
            "--step-length",
            str(self.step_length),
            "--no-step-log",
            "true",
            "--collision.action",
            "warn",
        ]

    def _reload_sumo(self) -> None:
        if self._traci is None or not hasattr(self._traci, "load"):
            raise RuntimeError("TraCI connection does not support load()")
        with self.performance.measure("sumo_start_or_load_time"):
            self._traci.load(self._sumo_load_args())
        self._sumo_reload_count += 1
        self.performance.increment("sumo_reloads")
        self._reset_subscription_state()

    def _cleanup_failed_traci_start(self, traci_module: Any) -> None:
        try:
            connection = traci_module.getConnection(self._conn_label)
        except Exception:
            connection = None
        if connection is not None:
            try:
                connection.close(wait=False)
            except Exception:
                pass
        try:
            if hasattr(traci_module, "close"):
                traci_module.close(False)
        except Exception:
            pass
        self._traci = None

    def _close_sumo(self) -> None:
        if self._traci is None:
            return
        try:
            self._traci.close()
        except Exception:
            pass
        self._traci = None
        self._reset_subscription_state()

    def _simulation_step(self) -> None:
        with self.performance.measure("simulation_step_time"):
            self._traci.simulationStep()
        self._simulation_step_index += 1
        self._refresh_vehicle_subscriptions()
        if self.sumo_step_delay_ms > 0:
            time.sleep(self.sumo_step_delay_ms / 1000.0)

    def _collect_states(self) -> list[VehicleState]:
        with self.performance.measure("state_collection_time"):
            return self._collect_states_impl()

    def _reset_subscription_state(self) -> None:
        self._subscribed_vehicle_ids.clear()

    def _subscription_variables(self) -> list[int]:
        constants = getattr(self._traci_module, "constants", None)
        names = (
            "VAR_POSITION",
            "VAR_ANGLE",
            "VAR_LANE_ID",
            "VAR_LANE_INDEX",
            "VAR_SPEED",
            "VAR_ACCELERATION",
            "VAR_LANEPOSITION",
            "VAR_ROAD_ID",
            "VAR_LENGTH",
            "VAR_WIDTH",
        )
        return [int(getattr(constants, name)) for name in names if constants is not None and hasattr(constants, name)]

    def _refresh_vehicle_subscriptions(self) -> None:
        if self._traci is None or not bool(self.config.scenario.get("traci_subscriptions_enabled", True)):
            return
        vehicle_api = self._traci.vehicle
        try:
            current_ids = set(str(item) for item in vehicle_api.getIDList())
            variables = self._subscription_variables()
            for vehicle_id in sorted(current_ids - self._subscribed_vehicle_ids):
                vehicle_api.subscribe(vehicle_id, variables)
            self._subscribed_vehicle_ids.intersection_update(current_ids)
            self._subscribed_vehicle_ids.update(current_ids)
        except Exception:
            self._subscription_error_count += 1
            self.performance.increment("traci_subscription_errors")

    def _state_from_getters(self, vehicle_id: str) -> VehicleState:
        vehicle_api = self._traci.vehicle
        x, y = vehicle_api.getPosition(vehicle_id)
        sumo_angle = vehicle_api.getAngle(vehicle_id)
        self._subscription_fallback_count += 1
        self.performance.increment("traci_getter_fallbacks")
        return VehicleState(
            vehicle_id=vehicle_id,
            x=float(x),
            y=float(y),
            heading=float(math.radians(90.0 - sumo_angle)),
            speed=float(vehicle_api.getSpeed(vehicle_id)),
            lane_index=int(vehicle_api.getLaneIndex(vehicle_id)),
            lane_id=str(vehicle_api.getLaneID(vehicle_id)),
            lane_pos=float(vehicle_api.getLanePosition(vehicle_id)),
            edge_id=str(vehicle_api.getRoadID(vehicle_id)),
            length=float(vehicle_api.getLength(vehicle_id)),
            width=float(vehicle_api.getWidth(vehicle_id)),
            accel=float(vehicle_api.getAcceleration(vehicle_id)),
        )

    def _collect_states_impl(self) -> list[VehicleState]:
        states: list[VehicleState] = []
        vehicle_api = self._traci.vehicle
        vehicle_ids = sorted(str(item) for item in vehicle_api.getIDList())
        use_subscriptions = bool(self.config.scenario.get("traci_subscriptions_enabled", True))
        if use_subscriptions:
            self._refresh_vehicle_subscriptions()
        all_results: dict[str, Any] = {}
        if use_subscriptions:
            try:
                all_results = vehicle_api.getAllSubscriptionResults() or {}
            except Exception:
                self._subscription_error_count += 1
                self.performance.increment("traci_subscription_errors")
        constants = getattr(self._traci_module, "constants", None)
        for vehicle_id in vehicle_ids:
            result = all_results.get(vehicle_id) if use_subscriptions else None
            if not result or constants is None:
                states.append(self._state_from_getters(vehicle_id))
                continue
            try:
                x, y = result[getattr(constants, "VAR_POSITION")]
                sumo_angle = result[getattr(constants, "VAR_ANGLE")]
                lane_id = str(result[getattr(constants, "VAR_LANE_ID")])
                lane_index = int(result[getattr(constants, "VAR_LANE_INDEX")])
                speed = float(result[getattr(constants, "VAR_SPEED")])
                accel = float(result[getattr(constants, "VAR_ACCELERATION")])
                lane_pos = float(result[getattr(constants, "VAR_LANEPOSITION")])
                edge_id = str(result[getattr(constants, "VAR_ROAD_ID")])
                length = float(result[getattr(constants, "VAR_LENGTH")])
                width = float(result[getattr(constants, "VAR_WIDTH")])
                heading = math.radians(90.0 - float(sumo_angle))
            except (KeyError, TypeError, ValueError):
                states.append(self._state_from_getters(vehicle_id))
                continue
            states.append(
                VehicleState(
                    vehicle_id=vehicle_id,
                    x=float(x),
                    y=float(y),
                    heading=float(heading),
                    speed=speed,
                    lane_index=lane_index,
                    lane_id=lane_id,
                    lane_pos=lane_pos,
                    edge_id=edge_id,
                    length=length,
                    width=width,
                    accel=accel,
                )
            )
        return states

    def _select_curriculum_profile(self) -> str:
        curriculum = self.config.stage1.get("curriculum", {})
        if not isinstance(curriculum, dict) or not bool(curriculum.get("enabled", False)):
            return "disabled"
        profiles = curriculum.get("profiles", {})
        if not isinstance(profiles, dict) or not profiles:
            return "disabled"
        names = list(profiles)
        probabilities = np.asarray(
            [max(0.0, float(profiles[name].get("probability", 0.0))) for name in names],
            dtype=np.float64,
        )
        if float(np.sum(probabilities)) <= 0.0:
            return str(names[0])
        probabilities /= np.sum(probabilities)
        rng = np.random.default_rng(self.seed_value)
        return str(rng.choice(names, p=probabilities))

    def _apply_curriculum_perturbation(self) -> None:
        self._curriculum_applied = True
        if self._curriculum_profile == "disabled" or self._traci is None:
            return
        curriculum = self.config.stage1.get("curriculum", {})
        profile = curriculum.get("profiles", {}).get(self._curriculum_profile, {})
        pos_jitter = float(profile.get("position_jitter", 0.0))
        speed_jitter = float(profile.get("speed_jitter", 0.0))
        rng = np.random.default_rng(self.seed_value + 104729)
        vehicle_api = self._traci.vehicle
        ids = set(vehicle_api.getIDList())
        configured = self.config.scenario.get(
            "curriculum_seed_vehicle_ids",
            [
                "ego",
                "target_lane_front_seed",
                "target_lane_gap_seed",
                "target_lane_rear_seed",
                "ramp_front_seed",
                "ramp_follow_seed",
                "auxiliary_front_seed",
                "auxiliary_rear_seed",
            ],
        )
        for vehicle_id in configured:
            vehicle_id = str(vehicle_id)
            if vehicle_id not in ids:
                continue
            try:
                lane_id = str(vehicle_api.getLaneID(vehicle_id))
                lane_length = float(self._traci.lane.getLength(lane_id))
                position = float(vehicle_api.getLanePosition(vehicle_id))
                target_position = float(np.clip(position + rng.uniform(-pos_jitter, pos_jitter), 0.0, max(0.0, lane_length - 5.0)))
                target_speed = max(0.0, float(vehicle_api.getSpeed(vehicle_id)) + rng.uniform(-speed_jitter, speed_jitter))
                vehicle_api.moveTo(vehicle_id, lane_id, target_position)
                vehicle_api.slowDown(vehicle_id, target_speed, max(self.step_length, 1.0))
            except Exception:
                continue

    def _configure_ego_control(self) -> None:
        if self._traci is None:
            return
        try:
            if self.ego_id in set(self._traci.vehicle.getIDList()):
                self._traci.vehicle.setLaneChangeMode(
                    self.ego_id,
                    int(self.config.scenario.get("ego_lane_change_mode", 512)),
                )
        except Exception:
            return

    def _get_ego(self) -> VehicleState | None:
        return self.history.latest().get(self.ego_id)

    def _ego_in_collision(self) -> bool:
        try:
            return self.ego_id in set(self._traci.simulation.getCollidingVehiclesIDList())
        except Exception:
            return False

    def _apply_action(self, action) -> bool:
        ego = self._get_ego()
        if ego is None:
            return True

        lane_oob = False
        target_speed = max(0.0, ego.speed + action.accel_cmd * 1.5 * self.control_interval_steps * self.step_length)
        self._traci.vehicle.setSpeed(self.ego_id, target_speed)

        if action.lateral_cmd != 0:
            target_lane = ego.lane_index + action.lateral_cmd
            lane_count = self._lane_count(ego.edge_id)
            if target_lane < 0 or target_lane >= lane_count:
                lane_oob = True
            else:
                self._traci.vehicle.changeLane(
                    self.ego_id,
                    target_lane,
                    max(self.step_length, self.control_interval_steps * self.step_length),
                )
        return lane_oob

    def _action_lane_oob(self, action) -> bool:
        ego = self._get_ego()
        if ego is None:
            return True
        target_lane = int(ego.lane_index) + int(action.lateral_cmd)
        return bool(target_lane < 0 or target_lane >= self._lane_count(ego.edge_id))

    def _lane_count(self, edge_id: str) -> int:
        if edge_id in self._lane_count_cache:
            return self._lane_count_cache[edge_id]
        try:
            count = int(self._traci.edge.getLaneNumber(edge_id))
        except Exception:
            latest = self.history.latest()
            same_edge = [state.lane_index for state in latest.values() if state.edge_id == edge_id]
            count = max(same_edge) + 1 if same_edge else 1
        self._lane_count_cache[edge_id] = count
        return count

    def _build_observation(self) -> np.ndarray:
        latest = self.history.latest()
        ego = latest.get(self.ego_id)
        if ego is None:
            base = np.zeros((self._base_obs_dim,), dtype=np.float32)
        else:
            base = self._base_observation(ego, latest)
        if not self.forecast_enabled:
            return base
        augmentor = self.forecast_augmentor or ForecastFeatureAugmentor(self.config)
        forecast = augmentor.extract(self.get_risk_context())
        return np.concatenate([base, forecast.astype(np.float32)], axis=0)

    def _base_observation(self, ego: VehicleState, latest: dict[str, VehicleState]) -> np.ndarray:
        local = merge_local_stats(ego, list(latest.values()), self.config)
        ego_vec = np.asarray(
            [
                ego.speed / 35.0,
                ego.accel / 5.0,
                ego.lane_index / 3.0,
                ego.lane_pos / 500.0,
                ego.x / 500.0,
                ego.y / 100.0,
                float(is_ramp_edge(self.config, ego.edge_id)),
                float(is_auxiliary_edge(self.config, ego.edge_id)),
            ],
            dtype=np.float32,
        )
        others = [state for state in latest.values() if state.vehicle_id != self.ego_id]
        others.sort(key=lambda state: abs(state.x - ego.x) + abs(state.y - ego.y))
        neighbor_features: list[float] = []
        for state in others[: self.top_k]:
            neighbor_features.extend(
                [
                    (state.x - ego.x) / 100.0,
                    (state.y - ego.y) / 25.0,
                    (state.speed - ego.speed) / 35.0,
                    (state.lane_index - ego.lane_index) / 3.0,
                    state.length / 10.0,
                    state.width / 4.0,
                    float(is_ramp_edge(self.config, state.edge_id)),
                    float(is_target_lane(self.config, state.edge_id, state.lane_index) or is_auxiliary_edge(self.config, state.edge_id)),
                ]
            )
        while len(neighbor_features) < self.top_k * 8:
            neighbor_features.append(0.0)
        merge_features = np.asarray(
            [
                distance_to_taper(self.config, ego) / 300.0,
                self._success_distance(ego) / 300.0,
                local.target_front_gap / 100.0,
                local.target_rear_gap / 100.0,
            ],
            dtype=np.float32,
        )
        return np.concatenate([ego_vec, np.asarray(neighbor_features, dtype=np.float32), merge_features], axis=0)

    def _success_distance(self, ego: VehicleState) -> float:
        if ego.edge_id == str(self.config.scenario.success_edge):
            return float(self.config.scenario.get("success_min_lane_position", 40.0)) - float(ego.lane_pos)
        return max(0.0, distance_to_taper(self.config, ego)) + float(
            self.config.scenario.get("success_min_lane_position", 40.0)
        )

    def _front_gap(self, ego: VehicleState, latest: dict[str, VehicleState]) -> float:
        gaps = [state.x - ego.x for state in latest.values() if state.vehicle_id != ego.vehicle_id and state.x >= ego.x]
        return float(min(gaps)) if gaps else 100.0

    def _rear_gap(self, ego: VehicleState, latest: dict[str, VehicleState]) -> float:
        gaps = [ego.x - state.x for state in latest.values() if state.vehicle_id != ego.vehicle_id and state.x < ego.x]
        return float(min(gaps)) if gaps else 100.0

    def _done(self, metrics: StepMetrics) -> tuple[bool, str]:
        ego = self._get_ego()
        if metrics.collision:
            return True, "collision"
        if ego is None:
            return True, "ego_missing"
        if is_taper_miss(self.config, ego):
            return True, "taper_miss"
        if (
            ego.edge_id == self.config.scenario.success_edge
            and ego.lane_pos >= float(self.config.scenario.get("success_min_lane_position", 40.0))
        ):
            return True, "merge_success"
        return False, ""

    def _reward(
        self,
        prev_x: float,
        ego: VehicleState | None,
        metrics: StepMetrics,
        done_reason: str,
        raw_action: Any | None = None,
        risk_context: dict[str, Any] | None = None,
    ) -> float:
        reward_cfg = self.config.rl.reward
        progress_reward = 0.0
        speed_reward = 0.0
        terminal_reward = 0.0
        lane_oob_penalty = 0.0
        safety_penalty = 0.0
        safety_forecast_shaping = 0.0
        shield_guided_shaping = 0.0
        merge_timing_shaping = 0.0
        self._last_reward_debug = {}
        if ego is not None:
            progress_reward = float(reward_cfg.progress * max(0.0, ego.x - prev_x))
            speed_reward = float(reward_cfg.speed * min(ego.speed, 33.33))
        if done_reason == "merge_success":
            terminal_reward += float(reward_cfg.merge_success)
        if metrics.collision:
            safety_penalty += float(reward_cfg.collision)
        if metrics.near_miss:
            safety_penalty += float(reward_cfg.near_miss)
        if metrics.low_ttc:
            safety_penalty += float(reward_cfg.low_ttc)
        if metrics.high_drac:
            safety_penalty += float(reward_cfg.high_drac)
        if metrics.hard_brake:
            safety_penalty += float(reward_cfg.hard_brake)
        if metrics.lane_oob:
            lane_oob_penalty = float(reward_cfg.lane_oob)
        reward_profile = str(self.config.rl.get("reward_profile", "default"))
        if reward_profile in {"safety_forecast", "shield_guided_forecast", "merge_timing_forecast"}:
            safety_forecast_shaping = self._safety_forecast_reward_adjustment(ego, metrics)
        if reward_profile in {"shield_guided_forecast", "merge_timing_forecast"}:
            shield_penalty, reward_debug = self._shield_guided_reward_adjustment(raw_action, risk_context)
            shield_guided_shaping = float(shield_penalty)
            self._last_reward_debug = reward_debug
            self._reward_debug_records.append(reward_debug)
        if reward_profile == "merge_timing_forecast":
            timing_adjustment, timing_debug = self._merge_timing_reward_adjustment(
                ego,
                done_reason,
                raw_action,
                risk_context,
            )
            merge_timing_shaping = float(timing_adjustment)
            self._last_reward_debug = {**self._last_reward_debug, **timing_debug}
        reward = float(
            progress_reward
            + speed_reward
            + terminal_reward
            + lane_oob_penalty
            + safety_penalty
            + safety_forecast_shaping
            + shield_guided_shaping
            + merge_timing_shaping
        )
        components = {
            "progress_reward": progress_reward,
            "speed_reward": speed_reward,
            "terminal_reward": terminal_reward,
            "lane_oob_penalty": lane_oob_penalty,
            "safety_penalty": safety_penalty,
            "safety_forecast_shaping": float(safety_forecast_shaping),
            "shield_guided_shaping": shield_guided_shaping,
            "merge_timing_shaping": merge_timing_shaping,
            "total_episode_reward": reward,
        }
        self._reward_component_records.append(components)
        self._last_reward_debug = {**self._last_reward_debug, **components}
        return float(reward)

    def _safety_forecast_reward_adjustment(self, ego: VehicleState | None, metrics: StepMetrics) -> float:
        if ego is None:
            return 0.0
        cfg = self.config.rl.get("safety_reward", {})
        distance_threshold = float(cfg.get("distance_threshold", 5.0))
        ttc_threshold = float(cfg.get("ttc_threshold", 2.0))
        drac_cap = float(cfg.get("drac_cap", 20.0))
        merge_gap_threshold = float(cfg.get("merge_gap_threshold", 8.0))
        merge_zone_margin = float(cfg.get("merge_zone_margin", 30.0))

        adjustment = 0.0
        if metrics.min_distance < distance_threshold:
            penalty = (distance_threshold - max(0.0, metrics.min_distance)) / max(distance_threshold, 1.0e-6)
            adjustment += float(cfg.get("distance_penalty_weight", -8.0)) * penalty
        if metrics.min_ttc < ttc_threshold:
            penalty = (ttc_threshold - max(0.0, metrics.min_ttc)) / max(ttc_threshold, 1.0e-6)
            adjustment += float(cfg.get("ttc_penalty_weight", -4.0)) * penalty
        if metrics.max_drac > float(self.config.risk_module.drac_threshold):
            penalty = min(max(0.0, metrics.max_drac), drac_cap) / max(drac_cap, 1.0e-6)
            adjustment += float(cfg.get("drac_penalty_weight", -3.0)) * penalty

        near_merge = distance_to_taper(self.config, ego) <= merge_zone_margin
        if near_merge:
            latest = self.history.latest()
            local = merge_local_stats(ego, list(latest.values()), self.config)
            if local.target_lane_gap < merge_gap_threshold:
                penalty = (merge_gap_threshold - max(0.0, local.target_lane_gap)) / max(merge_gap_threshold, 1.0e-6)
                adjustment += float(cfg.get("merge_gap_penalty_weight", -4.0)) * penalty
        return float(adjustment)

    def _shield_guided_reward_adjustment(
        self,
        raw_action: Any | None,
        context: dict[str, Any] | None,
    ) -> tuple[float, dict[str, Any]]:
        cfg = self.config.rl.get("shield_guided_reward", {})
        debug: dict[str, Any] = {
            "raw_action_risk": None,
            "best_candidate_risk": None,
            "risk_margin": None,
            "would_replace": False,
            "shield_guided_reward_penalty": 0.0,
            "available": False,
        }
        if raw_action is None or context is None or self.reward_risk_model is None or self.reward_ranker is None:
            return 0.0, debug

        ranked = self.reward_ranker.rank(raw_action, context)
        raw_prediction = next(
            (prediction for action, prediction, _score in ranked if action.index == raw_action.index),
            None,
        )
        if raw_prediction is None:
            raw_prediction = self.reward_risk_model.predict(raw_action, context)
        best_action, best_prediction = (raw_action, raw_prediction)
        if ranked:
            best_action, best_prediction, _score = min(ranked, key=lambda item: item[1].risk_score)

        raw_risk = float(raw_prediction.risk_score)
        best_risk = float(best_prediction.risk_score)
        risk_margin = raw_risk - best_risk
        raw_risk_threshold = float(cfg.get("raw_risk_threshold", 0.85))
        risk_margin_threshold = float(cfg.get("risk_margin_threshold", 0.15))
        uncertainty_threshold = float(cfg.get("uncertainty_threshold", 0.40))

        raw_penalty = max(0.0, raw_risk - raw_risk_threshold) * float(
            cfg.get("raw_risk_penalty_weight", -3.0)
        )
        margin_penalty = max(0.0, risk_margin - risk_margin_threshold) * float(
            cfg.get("risk_margin_penalty_weight", -4.0)
        )
        raw_legal = is_candidate_legal(raw_action, context)
        shield_shadow_action = raw_action
        shield_shadow_risk = raw_risk
        would_replace = False
        if raw_risk >= raw_risk_threshold:
            for candidate, prediction, _score in ranked:
                if candidate.index == raw_action.index:
                    continue
                improves_enough = (not raw_legal) or float(prediction.risk_score) <= raw_risk - risk_margin_threshold
                candidate_safe = (
                    float(prediction.risk_score) < float(self.config.shield.risk_threshold)
                    and float(prediction.risk_uncertainty) < uncertainty_threshold
                )
                if improves_enough and candidate_safe:
                    shield_shadow_action = candidate
                    shield_shadow_risk = float(prediction.risk_score)
                    would_replace = True
                    break
        replace_penalty = float(cfg.get("would_replace_penalty_weight", -2.0)) if would_replace else 0.0
        total_penalty = float(raw_penalty + margin_penalty + replace_penalty)

        debug = {
            "raw_action_risk": raw_risk,
            "best_candidate_risk": best_risk,
            "risk_margin": float(risk_margin),
            "would_replace": would_replace,
            "shield_guided_reward_penalty": total_penalty,
            "raw_risk_penalty": float(raw_penalty),
            "risk_margin_penalty": float(margin_penalty),
            "would_replace_penalty": float(replace_penalty),
            "best_candidate_action": int(best_action.index),
            "shield_shadow_action": int(shield_shadow_action.index),
            "shield_shadow_risk": float(shield_shadow_risk),
            "raw_candidate_legal": bool(raw_legal),
            "best_candidate_legal": bool(is_candidate_legal(best_action, context)),
            "available": True,
        }
        return total_penalty, debug

    def _forecast_aware_candidate_ranking_mode(self) -> str:
        raw_mode = self.config.shield.get("forecast_aware_candidate_ranking_mode", None)
        if raw_mode is None:
            if bool(self.config.shield.get("task_backstop_enabled", False)):
                return "task_backstop"
            if bool(self.config.shield.get("forecast_task_shadow_enabled", False)):
                return "shadow"
            return "off"
        mode = str(raw_mode).strip().lower()
        if mode not in {"off", "shadow", "task_backstop", "full_ranking"}:
            raise ValueError(
                "shield.forecast_aware_candidate_ranking_mode must be one of "
                "off, shadow, task_backstop, full_ranking"
            )
        return mode

    def _forecast_aware_scoring_enabled(self) -> bool:
        return self._forecast_aware_candidate_ranking_mode() != "off"

    def _task_merge_shadow(
        self,
        context: dict[str, Any] | None,
        raw_action: Any | None,
        *,
        update_counters: bool = True,
    ) -> dict[str, Any]:
        ranking_mode = self._forecast_aware_candidate_ranking_mode()
        debug: dict[str, Any] = {
            "available": False,
            "forecast_aware_candidate_ranking_mode": ranking_mode,
            "task_merge_opportunity": False,
            "task_would_merge": False,
            "task_missed_merge": False,
            "task_deadline_urgency": 0.0,
            "task_safe_merge_action": "",
            "task_safe_merge_action_index": None,
            "task_consecutive_missed_count": int(self._task_missed_consecutive_count),
            "distance_to_taper": None,
            "decision_distance_to_taper": None,
            "decision_target_front_gap": None,
            "decision_target_rear_gap": None,
            "decision_task_deadline_urgency": 0.0,
            "decision_ego_edge": "",
            "decision_ego_lane": -1,
            "forecast_aware_available": False,
            "forecast_aware_raw_task_cost": None,
            "forecast_aware_best_task_cost": None,
            "forecast_aware_task_improvement": None,
            "forecast_aware_raw_score": None,
            "forecast_aware_best_score": None,
            "forecast_aware_score_improvement": None,
            "forecast_aware_raw_task_risk": None,
            "forecast_aware_best_task_risk": None,
            "forecast_aware_best_action": None,
            "forecast_aware_best_action_name": "",
            "forecast_aware_would_merge": False,
            "forecast_aware_safety_risk": None,
            "forecast_actor_coverage_complete": False,
            "forecast_gap_consistency_pass": False,
            "forecast_gap_consistency_checkable": False,
            "task_backstop_watch_count": int(self._task_backstop_consecutive_count),
            "task_backstop_watch_eligible": False,
            "task_backstop_eligible": False,
            "task_backstop_risk_module_score": None,
            "task_backstop_risk_module_uncertainty": None,
            "task_backstop_risk_module_pass": False,
            "task_backstop_veto_reason": "",
            "forecast_ranking_eligible": False,
            "forecast_ranking_veto_reason": "",
            "forecast_ranking_risk_module_score": None,
            "forecast_ranking_risk_module_uncertainty": None,
            "forecast_ranking_risk_module_pass": False,
            "forecast_ranking_replacement": False,
            "forecast_ranking_replacement_reason": "",
        }
        if context is None or raw_action is None:
            return debug
        ego = context.get("ego")
        local = context.get("merge_local")
        merge_cmd = self._merge_lateral_cmd(ego)
        if ego is None or local is None or merge_cmd == 0:
            if update_counters:
                self._task_missed_consecutive_count = 0
            return debug

        deadline_distance = float(
            self.config.shield.get(
                "task_backstop_deadline_distance",
                self.config.rl.get("merge_timing_reward", {}).get("deadline_distance", 120.0),
            )
        )
        distance = max(0.0, float(local.merge_distance))
        urgency = float(np.clip((deadline_distance - distance) / max(deadline_distance, 1.0e-6), 0.0, 1.0))
        safe_action = next(
            (
                action
                for action in ACTIONS
                if action.lateral_cmd == merge_cmd and is_candidate_legal(action, context)
            ),
            None,
        )
        safe_gap = bool(
            safe_action is not None
            and float(local.target_front_gap)
            >= float(self.config.scenario.get("merge_opportunity_min_front_gap", 12.0))
            and float(local.target_rear_gap)
            >= float(self.config.scenario.get("merge_opportunity_min_rear_gap", 12.0))
        )
        missed = bool(safe_gap and int(raw_action.lateral_cmd) != merge_cmd)
        if update_counters:
            if missed:
                self._task_missed_consecutive_count += 1
            else:
                self._task_missed_consecutive_count = 0
        missed_count = (
            int(self._task_missed_consecutive_count)
            if update_counters
            else int(self._task_missed_consecutive_count + int(missed))
        )
        debug.update(
            {
                "available": True,
                "task_merge_opportunity": safe_gap,
                "task_would_merge": bool(missed and urgency > 0.0),
                "task_missed_merge": missed,
                "task_deadline_urgency": urgency,
                "task_safe_merge_action": str(safe_action.name) if safe_action is not None else "",
                "task_safe_merge_action_index": int(safe_action.index) if safe_action is not None else None,
                "task_consecutive_missed_count": missed_count,
                "distance_to_taper": distance,
                "decision_distance_to_taper": distance,
                "decision_target_front_gap": float(local.target_front_gap),
                "decision_target_rear_gap": float(local.target_rear_gap),
                "decision_task_deadline_urgency": urgency,
                "decision_ego_edge": str(ego.edge_id),
                "decision_ego_lane": int(ego.lane_index),
            }
        )
        if self._forecast_aware_scoring_enabled():
            debug.update(
                self.forecast_task_scorer.score(
                    context,
                    raw_action,
                    merge_cmd=merge_cmd,
                    deadline_distance=deadline_distance,
                    urgency=urgency,
                )
            )
        return debug

    def _maybe_forecast_aware_replacement(
        self,
        raw_action: Any,
        final_action: Any,
        context: dict[str, Any],
        intervention: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        mode = self._forecast_aware_candidate_ranking_mode()
        if mode in {"off", "shadow"}:
            self._task_backstop_consecutive_count = 0
            return None
        if mode == "task_backstop":
            return self._maybe_task_backstop(raw_action, final_action, context, intervention)
        if mode == "full_ranking":
            self._task_backstop_consecutive_count = 0
            return self._maybe_forecast_full_ranking(raw_action, final_action, context, intervention)
        return None

    def _mark_forecast_ranking_veto(
        self,
        reason: str,
        risk_check: dict[str, Any] | None = None,
        *,
        eligible: bool = False,
    ) -> None:
        if self._last_task_merge_record is None:
            self._last_task_merge_record = {}
        check = risk_check or {}
        self._last_task_merge_record.update(
            {
                "forecast_ranking_eligible": bool(eligible),
                "forecast_ranking_veto_reason": str(reason),
                "forecast_ranking_risk_module_score": check.get("risk_score"),
                "forecast_ranking_risk_module_uncertainty": check.get("risk_uncertainty"),
                "forecast_ranking_risk_module_pass": bool(check.get("safety_pass", False)),
                "forecast_ranking_replacement": False,
                "forecast_ranking_replacement_reason": "",
            }
        )

    def _maybe_forecast_full_ranking(
        self,
        raw_action: Any,
        final_action: Any,
        context: dict[str, Any],
        intervention: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        if self.shield is None or not self.shield.enabled:
            self._mark_forecast_ranking_veto("shield_disabled")
            return None
        if int(final_action.index) != int(raw_action.index):
            self._mark_forecast_ranking_veto("safety_shield_replaced")
            return None
        if intervention is not None and int(intervention.get("final_action", raw_action.index)) != int(
            intervention.get("raw_action", raw_action.index)
        ):
            self._mark_forecast_ranking_veto("safety_shield_replaced")
            return None

        record = self._last_task_merge_record or {}
        best_action_index = record.get("forecast_aware_best_action")
        best_action = decode_action(int(best_action_index)) if best_action_index is not None else None
        if best_action is None:
            self._mark_forecast_ranking_veto("candidate_unavailable")
            return None
        risk_check = self.shield.evaluate_candidate(best_action, context)

        def record_float(key: str, default: float) -> float:
            value = record.get(key)
            return float(default) if value is None else float(value)

        margin = float(self.config.shield.get("forecast_aware_ranking_improvement_margin", 0.05))
        uncertainty_threshold = float(self.config.shield.get("task_backstop_uncertainty_threshold", 0.40))
        improvement = record_float("forecast_aware_score_improvement", -INF_TTC)
        veto_reason = ""
        if not bool(record.get("forecast_aware_available", False)):
            veto_reason = "forecast_unavailable"
        elif int(best_action.index) == int(raw_action.index):
            veto_reason = "best_action_is_raw"
        elif not is_candidate_legal(best_action, context):
            veto_reason = "candidate_illegal"
        elif bool(record.get("actor_selector_overflow", False)):
            veto_reason = "actor_selector_overflow"
        elif bool(record.get("critical_actor_overflow", False)):
            veto_reason = "critical_actor_overflow"
        elif bool(record.get("cv_fallback_overflow", False)):
            veto_reason = "cv_fallback_overflow"
        elif not bool(record.get("forecast_safety_actor_coverage_complete", False)):
            veto_reason = "forecast_safety_actor_coverage"
        elif not bool(record.get("combined_critical_coverage_complete", False)):
            veto_reason = "combined_critical_coverage"
        elif bool(record.get("forecast_gap_consistency_checkable", False)) and not bool(
            record.get("forecast_gap_consistency_pass", False)
        ):
            veto_reason = "forecast_gap_consistency"
        elif not bool(record.get("forecast_gap_physical_consistency_pass", False)):
            veto_reason = "forecast_physical_consistency"
        elif record_float("forecast_aware_uncertainty", INF_TTC) > uncertainty_threshold:
            veto_reason = "forecast_uncertainty"
        elif improvement < margin:
            veto_reason = "forecast_score_margin"
        elif not bool(risk_check.get("safety_pass", False)):
            veto_reason = f"risk_module_{risk_check.get('veto_reason', 'veto')}"

        if veto_reason:
            self._mark_forecast_ranking_veto(veto_reason, risk_check)
            return None

        self._last_task_merge_record.update(
            {
                "forecast_ranking_eligible": True,
                "forecast_ranking_veto_reason": "",
                "forecast_ranking_risk_module_score": risk_check.get("risk_score"),
                "forecast_ranking_risk_module_uncertainty": risk_check.get("risk_uncertainty"),
                "forecast_ranking_risk_module_pass": bool(risk_check.get("safety_pass", False)),
                "forecast_ranking_replacement": True,
                "forecast_ranking_replacement_reason": "forecast_aware_full_ranking",
            }
        )
        return {
            "step": int(self._episode_step),
            "replacement_type": "forecast_ranking",
            "replacement_reason": "forecast_aware_full_ranking",
            "forecast_ranking_replacement_reason": "forecast_aware_full_ranking",
            "raw_action": int(raw_action.index),
            "final_action": int(best_action.index),
            "raw_action_name": str(raw_action.name),
            "final_action_name": str(best_action.name),
            "forecast_aware_candidate_ranking_mode": "full_ranking",
            "forecast_aware_raw_score": record.get("forecast_aware_raw_score"),
            "forecast_aware_best_score": record.get("forecast_aware_best_score"),
            "forecast_aware_score_improvement": float(improvement),
            "forecast_aware_raw_task_risk": record.get("forecast_aware_raw_task_risk"),
            "forecast_aware_best_task_risk": record.get("forecast_aware_best_task_risk"),
            "forecast_aware_safety_risk": record.get("forecast_aware_safety_risk"),
            "forecast_aware_uncertainty": record.get("forecast_aware_uncertainty"),
            "forecast_ranking_risk_module_score": risk_check.get("risk_score"),
            "forecast_ranking_risk_module_uncertainty": risk_check.get("risk_uncertainty"),
            "forecast_ranking_risk_module_pass": bool(risk_check.get("safety_pass", False)),
        }

    def _maybe_task_backstop(
        self,
        raw_action: Any,
        final_action: Any,
        context: dict[str, Any],
        intervention: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        if self.shield is None or not self.shield.enabled:
            self._task_backstop_consecutive_count = 0
            return None
        if int(final_action.index) != int(raw_action.index):
            self._task_backstop_consecutive_count = 0
            self._last_task_merge_record.update(
                {
                    "task_backstop_watch_count": 0,
                    "task_backstop_watch_eligible": False,
                    "task_backstop_eligible": False,
                    "task_backstop_veto_reason": "safety_shield_replaced",
                }
            )
            return None
        if intervention is not None and int(intervention.get("final_action", raw_action.index)) != int(
            intervention.get("raw_action", raw_action.index)
        ):
            self._task_backstop_consecutive_count = 0
            self._last_task_merge_record.update(
                {
                    "task_backstop_watch_count": 0,
                    "task_backstop_watch_eligible": False,
                    "task_backstop_eligible": False,
                    "task_backstop_veto_reason": "safety_shield_replaced",
                }
            )
            return None
        record = self._last_task_merge_record
        ego = context.get("ego")
        merge_cmd = self._merge_lateral_cmd(ego)
        best_action_index = record.get("forecast_aware_best_action")
        best_action = decode_action(int(best_action_index)) if best_action_index is not None else None
        deadline = float(self.config.shield.get("task_backstop_deadline_distance", 120.0))
        watch_urgency_threshold = float(
            self.config.shield.get(
                "task_backstop_watch_urgency_threshold",
                self.config.shield.get("task_backstop_urgency_threshold", 0.40),
            )
        )
        execute_urgency_threshold = float(
            self.config.shield.get(
                "task_backstop_execute_urgency_threshold",
                self.config.shield.get("task_backstop_urgency_threshold", 0.50),
            )
        )
        task_risk_margin = float(self.config.shield.get("task_backstop_task_risk_margin", 0.05))
        safety_threshold = float(self.config.shield.get("task_backstop_safety_risk_threshold", 0.35))
        uncertainty_threshold = float(self.config.shield.get("task_backstop_uncertainty_threshold", 0.40))

        def record_float(key: str, default: float) -> float:
            value = record.get(key)
            return float(default) if value is None else float(value)

        risk_check = {
            "candidate_legal": False,
            "risk_score": None,
            "risk_uncertainty": None,
            "safety_pass": False,
            "veto_reason": "candidate_unavailable",
        }
        if best_action is not None:
            risk_check = self.shield.evaluate_candidate(best_action, context)

        decision_distance = record_float("decision_distance_to_taper", record_float("distance_to_taper", INF_TTC))
        urgency = record_float(
            "decision_task_deadline_urgency",
            record_float("task_deadline_urgency", 0.0),
        )
        task_improvement = record_float("forecast_aware_task_improvement", -INF_TTC)
        veto_reason = ""
        if ego is None or not is_auxiliary_edge(self.config, ego.edge_id):
            veto_reason = "ego_not_auxiliary"
        elif merge_cmd == 0:
            veto_reason = "merge_direction_unavailable"
        elif decision_distance >= deadline:
            veto_reason = "outside_deadline"
        elif int(raw_action.lateral_cmd) == merge_cmd:
            veto_reason = "raw_requests_merge"
        elif not bool(record.get("forecast_aware_available", False)):
            veto_reason = "forecast_unavailable"
        elif not bool(record.get("forecast_actor_coverage_complete", False)):
            veto_reason = "forecast_actor_coverage"
        elif bool(record.get("actor_selector_overflow", False)):
            veto_reason = "actor_selector_overflow"
        elif bool(record.get("cv_fallback_overflow", False)):
            veto_reason = "cv_fallback_overflow"
        elif not bool(record.get("wcdt_required_actor_coverage_complete", False)):
            veto_reason = "wcdt_relevant_actor_coverage"
        elif not bool(record.get("forecast_safety_actor_coverage_complete", False)):
            veto_reason = "forecast_safety_actor_coverage"
        elif not bool(record.get("forecast_gap_consistency_pass", False)):
            veto_reason = "forecast_gap_consistency"
        elif not bool(record.get("forecast_gap_physical_consistency_pass", False)):
            veto_reason = "forecast_physical_consistency"
        elif best_action is None or int(best_action.lateral_cmd) != merge_cmd:
            veto_reason = "best_action_not_merge"
        elif not is_candidate_legal(best_action, context):
            veto_reason = "candidate_illegal"
        elif record_float("forecast_aware_safety_risk", INF_TTC) > safety_threshold:
            veto_reason = "forecast_safety_risk"
        elif record_float("forecast_aware_uncertainty", INF_TTC) > uncertainty_threshold:
            veto_reason = "forecast_uncertainty"
        elif task_improvement < task_risk_margin:
            veto_reason = "task_risk_margin"
        elif not bool(risk_check.get("safety_pass", False)):
            veto_reason = f"risk_module_{risk_check.get('veto_reason', 'veto')}"
        elif urgency < watch_urgency_threshold:
            veto_reason = "watch_urgency"

        watch_eligible = not veto_reason
        if not watch_eligible:
            self._task_backstop_consecutive_count = 0
        else:
            self._task_backstop_consecutive_count += 1
        required = max(1, int(self.config.shield.get("task_backstop_consecutive_steps", 2)))
        execute_eligible = bool(
            watch_eligible
            and self._task_backstop_consecutive_count >= required
            and urgency >= execute_urgency_threshold
        )
        if watch_eligible and not execute_eligible:
            veto_reason = (
                "execute_urgency"
                if self._task_backstop_consecutive_count >= required
                else "consecutive_steps"
            )
        record.update(
            {
                "task_backstop_watch_count": int(self._task_backstop_consecutive_count),
                "task_backstop_watch_eligible": bool(watch_eligible),
                "task_backstop_eligible": bool(execute_eligible),
                "task_backstop_risk_module_score": risk_check.get("risk_score"),
                "task_backstop_risk_module_uncertainty": risk_check.get("risk_uncertainty"),
                "task_backstop_risk_module_pass": bool(risk_check.get("safety_pass", False)),
                "task_backstop_veto_reason": str(veto_reason),
            }
        )
        if not execute_eligible or not bool(self.config.shield.get("task_backstop_enabled", False)):
            return None
        replacement = {
            "step": int(self._episode_step),
            "replacement_type": "task_backstop",
            "replacement_reason": "task_backstop",
            "raw_action": int(raw_action.index),
            "final_action": int(best_action.index),
            "raw_action_name": str(raw_action.name),
            "final_action_name": str(best_action.name),
            "forecast_aware_candidate_ranking_mode": "task_backstop",
            "task_deadline_urgency": urgency,
            "distance_to_taper": decision_distance,
            "decision_distance_to_taper": decision_distance,
            "forecast_aware_raw_score": record.get("forecast_aware_raw_score"),
            "forecast_aware_best_score": record.get("forecast_aware_best_score"),
            "forecast_aware_score_improvement": record.get("forecast_aware_score_improvement"),
            "forecast_aware_task_improvement": task_improvement,
            "forecast_aware_best_task_risk": record_float("forecast_aware_best_task_risk", 0.0),
            "forecast_aware_safety_risk": record_float("forecast_aware_safety_risk", 0.0),
            "forecast_aware_uncertainty": record_float("forecast_aware_uncertainty", 0.0),
            "forecast_aware_target_front_gap": record_float("forecast_aware_target_front_gap", INF_TTC),
            "forecast_aware_target_rear_gap": record_float("forecast_aware_target_rear_gap", INF_TTC),
            "task_backstop_consecutive_count": int(self._task_backstop_consecutive_count),
            "task_backstop_required": int(required),
            "task_backstop_risk_module_score": risk_check.get("risk_score"),
            "task_backstop_risk_module_uncertainty": risk_check.get("risk_uncertainty"),
            "task_backstop_risk_module_pass": bool(risk_check.get("safety_pass", False)),
        }
        self._task_backstop_consecutive_count = 0
        return replacement

    def _merge_timing_reward_adjustment(
        self,
        ego: VehicleState | None,
        done_reason: str,
        raw_action: Any | None,
        context: dict[str, Any] | None,
    ) -> tuple[float, dict[str, Any]]:
        cfg = self.config.rl.get("merge_timing_reward", {})
        # The task shadow mutates consecutive counters and must run exactly once
        # per control decision. Reward calculation only consumes that immutable
        # decision record after the action has been executed.
        record = self._last_task_merge_record or self._task_merge_shadow(
            context,
            raw_action,
            update_counters=False,
        )
        urgency = float(record.get("task_deadline_urgency", 0.0) or 0.0)
        grace = int(cfg.get("consecutive_missed_grace", 2))
        missed_count = int(record.get("task_consecutive_missed_count", 0) or 0)
        missed_penalty = 0.0
        if bool(record.get("task_missed_merge", False)) and missed_count > grace:
            missed_penalty = float(cfg.get("missed_opportunity_weight", -2.0)) * urgency

        deadline_penalty = 0.0
        if ego is not None and is_auxiliary_edge(self.config, ego.edge_id):
            deadline_penalty = float(cfg.get("taper_deadline_weight", -4.0)) * urgency

        taper_penalty = float(cfg.get("taper_miss_penalty", -35.0)) if done_reason == "taper_miss" else 0.0
        early_bonus = 0.0
        if (
            ego is not None
            and self._first_target_lane_entry_step == int(self._episode_step)
            and self._first_target_lane_entry_distance_to_taper is not None
            and float(self._first_target_lane_entry_distance_to_taper)
            >= float(cfg.get("bonus_min_distance_to_taper", 60.0))
        ):
            early_bonus = float(cfg.get("early_safe_merge_bonus", 1.5))
        total = float(missed_penalty + deadline_penalty + taper_penalty + early_bonus)
        debug = {
            "merge_timing_reward_adjustment": total,
            "merge_timing_missed_penalty": float(missed_penalty),
            "merge_timing_deadline_penalty": float(deadline_penalty),
            "merge_timing_taper_miss_penalty": float(taper_penalty),
            "merge_timing_early_safe_merge_bonus": float(early_bonus),
            "task_merge_opportunity": bool(record.get("task_merge_opportunity", False)),
            "task_would_merge": bool(record.get("task_would_merge", False)),
            "task_missed_merge": bool(record.get("task_missed_merge", False)),
            "task_deadline_urgency": urgency,
            "task_consecutive_missed_count": missed_count,
        }
        return total, debug

    def _merge_lateral_cmd(self, ego: VehicleState | None) -> int:
        if ego is None or not is_auxiliary_edge(self.config, ego.edge_id):
            return 0
        return int(target_lane_index(self.config, ego.edge_id) - auxiliary_lane_index(self.config, ego.edge_id))

    def _record_merge_opportunity(self, context: dict[str, Any], raw_action: Any) -> None:
        ego = context.get("ego")
        local = context.get("merge_local")
        merge_cmd = self._merge_lateral_cmd(ego)
        if ego is None or local is None or merge_cmd == 0:
            self._last_task_merge_record = {}
            return
        if int(raw_action.lateral_cmd) == merge_cmd and self._first_merge_request_step is None:
            self._first_merge_request_step = int(self._episode_step)
            self._first_merge_request_distance_to_taper = float(local.merge_distance)
        task_record = self._task_merge_shadow(context, raw_action)
        task_record["step"] = int(self._episode_step)
        task_record["decision_step"] = int(self._decision_index)
        task_record["decision_index"] = int(self._decision_index)
        task_record["trace_schema_version"] = 2
        self._last_task_merge_record = task_record
        self._task_merge_records.append(task_record)
        safe_opportunity = bool(
            task_record.get("task_merge_opportunity", False)
            and float(local.merge_distance)
            >= float(self.config.scenario.get("merge_opportunity_min_distance_to_taper", 60.0))
        )
        if not safe_opportunity:
            return
        self._safe_merge_opportunity_count += 1
        if int(raw_action.lateral_cmd) != merge_cmd:
            self._missed_safe_merge_opportunity_count += 1

    def _record_target_lane_entry(self, ego: VehicleState) -> None:
        if self._first_target_lane_entry_step is not None:
            return
        if not is_target_lane(self.config, ego.edge_id, ego.lane_index):
            return
        self._first_target_lane_entry_step = int(self._episode_step)
        self._first_target_lane_entry_distance_to_taper = float(distance_to_taper(self.config, ego))

    def _info(
        self,
        metrics: StepMetrics | None = None,
        done_reason: str = "",
        intervention: dict[str, Any] | None = None,
        task_replacement: dict[str, Any] | None = None,
        forecast_ranking_replacement: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        info: dict[str, Any] = {
            "seed": self.seed_value,
            "episode_seed": self.seed_value,
            "episode_index": int(self._active_episode_index),
            "episode_seed_schedule": str(
                self.config.get("run", {}).get("episode_seed_schedule", "fixed_legacy")
            ),
            "step": self._episode_step,
            "decision_index": int(self._decision_index),
            "done_reason": done_reason,
            "intervention": intervention,
            "safety_metric_version": str(
                self.config.risk_module.get("safety_metric_version", SAFETY_METRIC_VERSION)
            ),
        }
        if self._last_reward_debug:
            info["reward_debug"] = self._last_reward_debug
        if metrics is not None:
            latest = self.history.latest()
            ego_state = self._get_ego()
            local = merge_local_stats(ego_state, list(latest.values()), self.config)
            local_metrics = StepMetrics(
                min_distance=metrics.min_distance,
                min_ttc=metrics.min_ttc,
                max_drac=metrics.max_drac,
                collision=metrics.collision,
                near_miss=metrics.near_miss,
                low_ttc=metrics.low_ttc,
                high_drac=metrics.high_drac,
                merge_gap=local.target_lane_gap,
                lane_oob=metrics.lane_oob,
                hard_brake=metrics.hard_brake,
                geometric_overlap=metrics.geometric_overlap,
                closest_vehicle_id=metrics.closest_vehicle_id,
                closest_vehicle_edge=metrics.closest_vehicle_edge,
                closest_vehicle_lane=metrics.closest_vehicle_lane,
                ttc_vehicle_id=metrics.ttc_vehicle_id,
                drac_vehicle_id=metrics.drac_vehicle_id,
            )
            info.update(local_metrics.to_dict())
            info.update(
                {
                    "target_lane_id": local.target_lane_id,
                    "target_front_gap": local.target_front_gap,
                    "target_rear_gap": local.target_rear_gap,
                    "target_front_vehicle_id": local.target_front_vehicle_id,
                    "target_rear_vehicle_id": local.target_rear_vehicle_id,
                    "target_lane_gap": local.target_lane_gap,
                    "ramp_front_gap": local.ramp_front_gap,
                    "ramp_rear_gap": local.ramp_rear_gap,
                    "ramp_local_risk": local.ramp_local_risk,
                    "merge_zone_risk": local.merge_zone_risk,
                    "ego_on_auxiliary": local.ego_on_auxiliary,
                    "ego_edge": str(ego_state.edge_id) if ego_state is not None else "",
                    "ego_lane": int(ego_state.lane_index) if ego_state is not None else -1,
                    "distance_to_taper": local.merge_distance,
                    "post_action_step": int(self._episode_step),
                    "post_action_target_front_gap": local.target_front_gap,
                    "post_action_target_rear_gap": local.target_rear_gap,
                    "post_action_distance_to_taper": local.merge_distance,
                    "post_action_ego_edge": str(ego_state.edge_id) if ego_state is not None else "",
                    "post_action_ego_lane": int(ego_state.lane_index) if ego_state is not None else -1,
                    "taper_miss": local.taper_miss,
                    "first_merge_request_step": self._first_merge_request_step,
                    "first_merge_request_distance_to_taper": self._first_merge_request_distance_to_taper,
                    "first_target_lane_entry_step": self._first_target_lane_entry_step,
                    "first_target_lane_entry_distance_to_taper": self._first_target_lane_entry_distance_to_taper,
                    "safe_merge_opportunity_count": int(self._safe_merge_opportunity_count),
                    "missed_safe_merge_opportunity_count": int(self._missed_safe_merge_opportunity_count),
                    "task_merge_opportunity": bool(self._last_task_merge_record.get("task_merge_opportunity", False)),
                    "task_would_merge": bool(self._last_task_merge_record.get("task_would_merge", False)),
                    "task_missed_merge": bool(self._last_task_merge_record.get("task_missed_merge", False)),
                    "task_deadline_urgency": float(self._last_task_merge_record.get("task_deadline_urgency", 0.0)),
                    "trace_schema_version": int(self._last_task_merge_record.get("trace_schema_version", 2)),
                    "decision_step": self._last_task_merge_record.get("decision_step"),
                    "decision_distance_to_taper": self._last_task_merge_record.get("decision_distance_to_taper"),
                    "decision_target_front_gap": self._last_task_merge_record.get("decision_target_front_gap"),
                    "decision_target_rear_gap": self._last_task_merge_record.get("decision_target_rear_gap"),
                    "decision_task_deadline_urgency": self._last_task_merge_record.get(
                        "decision_task_deadline_urgency"
                    ),
                    "decision_ego_edge": str(self._last_task_merge_record.get("decision_ego_edge", "")),
                    "decision_ego_lane": int(self._last_task_merge_record.get("decision_ego_lane", -1)),
                    "task_safe_merge_action": str(self._last_task_merge_record.get("task_safe_merge_action", "")),
                    "forecast_aware_candidate_ranking_mode": str(
                        self._last_task_merge_record.get(
                            "forecast_aware_candidate_ranking_mode",
                            self._forecast_aware_candidate_ranking_mode(),
                        )
                    ),
                    "forecast_aware_raw_score": self._last_task_merge_record.get("forecast_aware_raw_score"),
                    "forecast_aware_best_score": self._last_task_merge_record.get("forecast_aware_best_score"),
                    "forecast_aware_score_improvement": self._last_task_merge_record.get(
                        "forecast_aware_score_improvement"
                    ),
                    "forecast_aware_raw_task_cost": self._last_task_merge_record.get(
                        "forecast_aware_raw_task_cost"
                    ),
                    "forecast_aware_best_task_cost": self._last_task_merge_record.get(
                        "forecast_aware_best_task_cost"
                    ),
                    "forecast_aware_task_improvement": self._last_task_merge_record.get(
                        "forecast_aware_task_improvement"
                    ),
                    "forecast_aware_raw_task_risk": self._last_task_merge_record.get("forecast_aware_raw_task_risk"),
                    "forecast_aware_best_task_risk": self._last_task_merge_record.get("forecast_aware_best_task_risk"),
                    "forecast_aware_best_action": self._last_task_merge_record.get("forecast_aware_best_action"),
                    "forecast_aware_best_action_name": str(
                        self._last_task_merge_record.get("forecast_aware_best_action_name", "")
                    ),
                    "forecast_aware_would_merge": bool(
                        self._last_task_merge_record.get("forecast_aware_would_merge", False)
                    ),
                    "forecast_aware_safety_risk": self._last_task_merge_record.get("forecast_aware_safety_risk"),
                    "forecast_aware_uncertainty": self._last_task_merge_record.get("forecast_aware_uncertainty"),
                    "forecast_aware_target_front_gap": self._last_task_merge_record.get(
                        "forecast_aware_target_front_gap"
                    ),
                    "forecast_aware_target_rear_gap": self._last_task_merge_record.get(
                        "forecast_aware_target_rear_gap"
                    ),
                    "forecast_first_step_target_front_gap": self._last_task_merge_record.get(
                        "forecast_first_step_target_front_gap"
                    ),
                    "forecast_first_step_target_rear_gap": self._last_task_merge_record.get(
                        "forecast_first_step_target_rear_gap"
                    ),
                    "forecast_gap_consistency_pass": bool(
                        self._last_task_merge_record.get("forecast_gap_consistency_pass", False)
                    ),
                    "forecast_gap_consistency_checkable": bool(
                        self._last_task_merge_record.get(
                            "forecast_gap_consistency_checkable",
                            False,
                        )
                    ),
                    "forecast_gap_consistency_failure_reason": str(
                        self._last_task_merge_record.get(
                            "forecast_gap_consistency_failure_reason",
                            "",
                        )
                    ),
                    "forecast_gap_physical_consistency_pass": bool(
                        self._last_task_merge_record.get("forecast_gap_physical_consistency_pass", False)
                    ),
                    "forecast_vehicle_identity_consistent": bool(
                        self._last_task_merge_record.get("forecast_vehicle_identity_consistent", False)
                    ),
                    "forecast_identity_turnover": bool(
                        self._last_task_merge_record.get(
                            "forecast_identity_turnover",
                            False,
                        )
                    ),
                    "forecast_identity_turnover_valid": bool(
                        self._last_task_merge_record.get(
                            "forecast_identity_turnover_valid",
                            False,
                        )
                    ),
                    "forecast_route_position_valid": bool(
                        self._last_task_merge_record.get(
                            "forecast_route_position_valid",
                            False,
                        )
                    ),
                    "forecast_projection_distance": self._last_task_merge_record.get(
                        "forecast_projection_distance"
                    ),
                    "forecast_projection_ambiguity_margin": self._last_task_merge_record.get(
                        "forecast_projection_ambiguity_margin"
                    ),
                    "forecast_front_first_step_progress_error": self._last_task_merge_record.get(
                        "forecast_front_first_step_progress_error"
                    ),
                    "forecast_rear_first_step_progress_error": self._last_task_merge_record.get(
                        "forecast_rear_first_step_progress_error"
                    ),
                    "forecast_selected_vehicle_ids": list(
                        self._last_task_merge_record.get("forecast_selected_vehicle_ids", [])
                    ),
                    "forecast_wcdt_selected_vehicle_ids": list(
                        self._last_task_merge_record.get("forecast_wcdt_selected_vehicle_ids", [])
                    ),
                    "forecast_cv_fallback_vehicle_ids": list(
                        self._last_task_merge_record.get("forecast_cv_fallback_vehicle_ids", [])
                    ),
                    "forecast_actor_sources": dict(
                        self._last_task_merge_record.get("forecast_actor_sources", {})
                    ),
                    "forecast_actor_relevance": dict(
                        self._last_task_merge_record.get("forecast_actor_relevance", {})
                    ),
                    "forecast_wcdt_uncertainty": self._last_task_merge_record.get(
                        "forecast_wcdt_uncertainty"
                    ),
                    "forecast_cv_fallback_uncertainty": self._last_task_merge_record.get(
                        "forecast_cv_fallback_uncertainty"
                    ),
                    "combined_forecast_uncertainty": self._last_task_merge_record.get(
                        "combined_forecast_uncertainty"
                    ),
                    "forecast_target_front_vehicle_id": str(
                        self._last_task_merge_record.get("forecast_target_front_vehicle_id", "")
                    ),
                    "forecast_target_rear_vehicle_id": str(
                        self._last_task_merge_record.get("forecast_target_rear_vehicle_id", "")
                    ),
                    "forecast_target_front_required": bool(
                        self._last_task_merge_record.get("forecast_target_front_required", False)
                    ),
                    "forecast_target_rear_required": bool(
                        self._last_task_merge_record.get("forecast_target_rear_required", False)
                    ),
                    "forecast_target_front_covered": bool(
                        self._last_task_merge_record.get("forecast_target_front_covered", False)
                    ),
                    "forecast_target_rear_covered": bool(
                        self._last_task_merge_record.get("forecast_target_rear_covered", False)
                    ),
                    "forecast_actor_coverage_complete": bool(
                        self._last_task_merge_record.get("forecast_actor_coverage_complete", False)
                    ),
                    "wcdt_required_actor_coverage_complete": bool(
                        self._last_task_merge_record.get(
                            "wcdt_required_actor_coverage_complete",
                            False,
                        )
                    ),
                    "forecast_safety_actor_coverage_complete": bool(
                        self._last_task_merge_record.get(
                            "forecast_safety_actor_coverage_complete",
                            False,
                        )
                    ),
                    "actor_selector_relevant_count": int(
                        self._last_task_merge_record.get("actor_selector_relevant_count", 0)
                    ),
                    "actor_selector_overflow": bool(
                        self._last_task_merge_record.get("actor_selector_overflow", False)
                    ),
                    "critical_actor_count": int(
                        self._last_task_merge_record.get("critical_actor_count", 0)
                    ),
                    "contextual_actor_count": int(
                        self._last_task_merge_record.get("contextual_actor_count", 0)
                    ),
                    "critical_actor_overflow": bool(
                        self._last_task_merge_record.get(
                            "critical_actor_overflow",
                            False,
                        )
                    ),
                    "critical_dropped_actor_ids": list(
                        self._last_task_merge_record.get(
                            "critical_dropped_actor_ids",
                            [],
                        )
                    ),
                    "contextual_actor_truncated_count": int(
                        self._last_task_merge_record.get(
                            "contextual_actor_truncated_count",
                            0,
                        )
                    ),
                    "critical_wcdt_coverage_complete": bool(
                        self._last_task_merge_record.get(
                            "critical_wcdt_coverage_complete",
                            False,
                        )
                    ),
                    "combined_critical_coverage_complete": bool(
                        self._last_task_merge_record.get(
                            "combined_critical_coverage_complete",
                            False,
                        )
                    ),
                    "actor_selector_dropped_relevant_ids": list(
                        self._last_task_merge_record.get(
                            "actor_selector_dropped_relevant_ids",
                            [],
                        )
                    ),
                    "cv_fallback_overflow": bool(
                        self._last_task_merge_record.get("cv_fallback_overflow", False)
                    ),
                    "cv_fallback_dropped_vehicle_ids": list(
                        self._last_task_merge_record.get(
                            "cv_fallback_dropped_vehicle_ids",
                            [],
                        )
                    ),
                    "forecast_closest_vehicle_id": str(
                        self._last_task_merge_record.get("forecast_closest_vehicle_id", "")
                    ),
                    "forecast_front_gap_vehicle_id": str(
                        self._last_task_merge_record.get("forecast_front_gap_vehicle_id", "")
                    ),
                    "forecast_rear_gap_vehicle_id": str(
                        self._last_task_merge_record.get("forecast_rear_gap_vehicle_id", "")
                    ),
                    "task_backstop_watch_count": int(
                        self._last_task_merge_record.get("task_backstop_watch_count", 0)
                    ),
                    "task_backstop_watch_eligible": bool(
                        self._last_task_merge_record.get("task_backstop_watch_eligible", False)
                    ),
                    "task_backstop_eligible": bool(
                        self._last_task_merge_record.get("task_backstop_eligible", False)
                    ),
                    "task_backstop_risk_module_score": self._last_task_merge_record.get(
                        "task_backstop_risk_module_score"
                    ),
                    "task_backstop_risk_module_uncertainty": self._last_task_merge_record.get(
                        "task_backstop_risk_module_uncertainty"
                    ),
                    "task_backstop_risk_module_pass": bool(
                        self._last_task_merge_record.get("task_backstop_risk_module_pass", False)
                    ),
                    "task_backstop_veto_reason": str(
                        self._last_task_merge_record.get("task_backstop_veto_reason", "")
                    ),
                    "forecast_ranking_eligible": bool(
                        self._last_task_merge_record.get("forecast_ranking_eligible", False)
                    ),
                    "forecast_ranking_veto_reason": str(
                        self._last_task_merge_record.get("forecast_ranking_veto_reason", "")
                    ),
                    "forecast_ranking_risk_module_score": self._last_task_merge_record.get(
                        "forecast_ranking_risk_module_score"
                    ),
                    "forecast_ranking_risk_module_uncertainty": self._last_task_merge_record.get(
                        "forecast_ranking_risk_module_uncertainty"
                    ),
                    "forecast_ranking_risk_module_pass": bool(
                        self._last_task_merge_record.get("forecast_ranking_risk_module_pass", False)
                    ),
                    "forecast_ranking_replacement": bool(forecast_ranking_replacement is not None),
                    "forecast_ranking_replacement_reason": str(
                        forecast_ranking_replacement.get("forecast_ranking_replacement_reason", "")
                    )
                    if forecast_ranking_replacement is not None
                    else "",
                    "task_replacement": bool(task_replacement is not None),
                    "task_replacement_reason": str(task_replacement.get("replacement_reason", ""))
                    if task_replacement is not None
                    else "",
                }
            )
            if intervention is not None:
                best_action = decode_action(int(intervention.get("best_candidate_action", intervention.get("raw_action", 0))))
                merge_cmd = self._merge_lateral_cmd(self._get_ego())
                info["best_merge_action"] = best_action.name if best_action.lateral_cmd == merge_cmd else ""
                info["best_merge_action_risk"] = float(intervention.get("best_candidate_risk", 0.0))
            else:
                info["best_merge_action"] = ""
                info["best_merge_action_risk"] = None
            info["explicit_risk_features"] = explicit_risk_features(local_metrics)
        return info

    def get_risk_context(self) -> dict[str, Any]:
        if self._decision_context_cache is not None:
            return self._decision_context_cache
        latest = self.history.latest()
        ego = latest.get(self.ego_id)
        vehicles = list(latest.values())
        local = merge_local_stats(ego, vehicles, self.config)
        context = {
            "ego": ego,
            "vehicles": vehicles,
            "history": self.history,
            "config": self.config,
            "lane_count": self._lane_count(ego.edge_id) if ego is not None else 1,
            "current_metrics": compute_step_metrics(
                ego,
                vehicles,
                collision=False,
                near_miss_threshold=float(self.config.risk_module.near_miss_distance_threshold),
                ttc_threshold=float(self.config.risk_module.ttc_threshold),
                drac_threshold=float(self.config.risk_module.drac_threshold),
                merge_ego_edges=merge_zone_edges(self.config),
                merge_target_edges=target_lane_edges(self.config),
                merge_target_lane=merge_target_lane(self.config),
                merge_target_lanes=target_lane_mapping(self.config),
            ) if ego is not None else None,
            "merge_local": local,
            "curriculum_profile": self._curriculum_profile,
            "performance_tracker": self.performance,
        }
        self._decision_context_cache = context
        return context

    def _invalidate_decision_cache(self) -> None:
        self._decision_context_cache = None

    def _append_trajectory_frame(
        self,
        states: list[VehicleState],
        *,
        decision_index: int,
    ) -> None:
        self._trajectory_frames.append({state.vehicle_id: state for state in states})
        self._trajectory_frame_metadata.append(
            {
                "simulation_step": int(self._simulation_step_index),
                "decision_index": int(decision_index),
                "episode_seed": int(self.seed_value),
            }
        )

    def episode_report(self) -> dict[str, Any]:
        collisions = [metric.collision for metric in self._episode_metrics]
        geometric_overlaps = [metric.geometric_overlap for metric in self._episode_metrics]
        near_misses = [metric.near_miss for metric in self._episode_metrics]
        min_distances = [metric.min_distance for metric in self._episode_metrics]
        ttcs = [metric.min_ttc for metric in self._episode_metrics if metric.min_ttc < INF_TTC]
        dracs = [metric.max_drac for metric in self._episode_metrics]
        hard_brake_count = sum(1 for metric in self._episode_metrics if metric.hard_brake)
        hard_brake_rate = float(hard_brake_count / len(self._episode_metrics)) if self._episode_metrics else 0.0
        ego_speed_mean = float(np.mean(self._ego_speeds)) if self._ego_speeds else 0.0
        ego_speed_p10 = float(np.percentile(self._ego_speeds, 10)) if self._ego_speeds else 0.0
        completion_time = float(self._episode_step * self.step_length)
        min_distance = float(min(min_distances)) if min_distances else INF_TTC
        ttc_p1 = float(np.percentile(ttcs, 1)) if ttcs else INF_TTC
        drac_raw = float(np.percentile(dracs, 99)) if dracs else 0.0
        drac_cap = float(self.config.rl.get("safety_reward", {}).get("drac_cap", 20.0))
        drac_capped = float(np.percentile(np.minimum(np.asarray(dracs, dtype=np.float32), drac_cap), 99)) if dracs else 0.0
        proxy_collision = min_distance <= float(self.config.risk_module.collision_distance_threshold)
        safety_violation = bool(any(collisions) or proxy_collision or any(near_misses) or ttc_p1 < 0.3)
        proxy_collision_count = int(bool(proxy_collision))
        safety_violation_count = int(bool(safety_violation))
        replacement_count = sum(
            1
            for item in self._interventions
            if int(item.get("final_action", item.get("raw_action", -1))) != int(item.get("raw_action", -1))
        )
        reason_counts = Counter(str(item.get("replacement_reason", "")) for item in self._interventions)
        if self._action_execution_records:
            raw_actions = Counter(
                str(item.get("raw_action", "")) for item in self._action_execution_records
            )
            safety_shield_actions = Counter(
                str(item.get("safety_shield_action", item.get("raw_action", "")))
                for item in self._action_execution_records
            )
            executed_actions = Counter(
                str(item.get("final_action", "")) for item in self._action_execution_records
            )
        else:
            # Backward-compatible fallback for synthetic reports and legacy callers.
            raw_actions = Counter(str(item.get("raw_action", "")) for item in self._interventions)
            safety_shield_actions = Counter(
                str(item.get("final_action", item.get("raw_action", "")))
                for item in self._interventions
            )
            executed_actions = Counter(safety_shield_actions)
        emergency_fallback_count = sum(1 for item in self._interventions if item.get("emergency_fallback"))
        task_available_records = [record for record in self._task_merge_records if record.get("available")]
        task_merge_count = sum(1 for record in task_available_records if record.get("task_merge_opportunity"))
        task_would_merge_count = sum(1 for record in task_available_records if record.get("task_would_merge"))
        task_missed_merge_count = sum(1 for record in task_available_records if record.get("task_missed_merge"))
        task_deadline_distance = float(
            self.config.shield.get(
                "task_backstop_deadline_distance",
                self.config.rl.get("merge_timing_reward", {}).get("deadline_distance", 120.0),
            )
        )
        task_deadline_records = [
            record
            for record in task_available_records
            if record.get("task_merge_opportunity")
            and float(
                record.get("decision_distance_to_taper")
                if record.get("decision_distance_to_taper") is not None
                else INF_TTC
            )
            < task_deadline_distance
        ]
        deadline_missed_count = sum(1 for record in task_deadline_records if record.get("task_missed_merge"))
        urgency_records = [
            record
            for record in task_available_records
            if record.get("task_merge_opportunity")
            and float(
                record.get("decision_task_deadline_urgency")
                if record.get("decision_task_deadline_urgency") is not None
                else 0.0
            )
            >= 0.5
        ]
        missed_after_urgency_count = sum(1 for record in urgency_records if record.get("task_missed_merge"))
        forecast_records = [record for record in task_available_records if record.get("forecast_aware_available")]
        forecast_coverage_complete_count = sum(
            1 for record in forecast_records if record.get("forecast_actor_coverage_complete")
        )
        forecast_gap_consistency_pass_count = sum(
            1 for record in forecast_records if record.get("forecast_gap_consistency_pass")
        )
        forecast_gap_consistency_checkable_count = sum(
            1
            for record in forecast_records
            if record.get("forecast_gap_consistency_checkable")
        )
        forecast_gap_failure_reason_counts = Counter(
            str(record.get("forecast_gap_consistency_failure_reason", ""))
            for record in forecast_records
            if str(record.get("forecast_gap_consistency_failure_reason", ""))
            not in {"", "ok"}
        )
        wcdt_relevant_coverage_count = sum(
            1
            for record in forecast_records
            if record.get("wcdt_required_actor_coverage_complete")
        )
        safety_actor_coverage_count = sum(
            1
            for record in forecast_records
            if record.get("forecast_safety_actor_coverage_complete")
        )
        selector_overflow_count = sum(
            1 for record in forecast_records if record.get("actor_selector_overflow")
        )
        critical_overflow_count = sum(
            1 for record in forecast_records if record.get("critical_actor_overflow")
        )
        critical_wcdt_coverage_count = sum(
            1
            for record in forecast_records
            if record.get("critical_wcdt_coverage_complete")
        )
        combined_critical_coverage_count = sum(
            1
            for record in forecast_records
            if record.get("combined_critical_coverage_complete")
        )
        cv_fallback_overflow_count = sum(
            1 for record in forecast_records if record.get("cv_fallback_overflow")
        )
        cv_fallback_usage_count = sum(
            1
            for record in forecast_records
            if record.get("forecast_cv_fallback_vehicle_ids")
        )
        task_backstop_watch_count = sum(
            1 for record in task_available_records if record.get("task_backstop_watch_eligible")
        )
        task_backstop_eligible_count = sum(
            1 for record in task_available_records if record.get("task_backstop_eligible")
        )
        task_backstop_veto_reason_counts = Counter(
            str(record.get("task_backstop_veto_reason", ""))
            for record in task_available_records
            if str(record.get("task_backstop_veto_reason", ""))
        )
        no_merge_request_before_taper = bool(
            self._first_merge_request_step is None and self._last_done_reason == "taper_miss"
        )
        task_replacement_count = len(self._task_replacements)
        forecast_ranking_replacement_count = len(self._forecast_ranking_replacements)
        task_replacement_reason_counts = Counter(
            str(item.get("replacement_reason", ""))
            for item in self._task_replacements
            if str(item.get("replacement_reason", ""))
        )
        forecast_ranking_replacement_reason_counts = Counter(
            str(item.get("forecast_ranking_replacement_reason", item.get("replacement_reason", "")))
            for item in self._forecast_ranking_replacements
            if str(item.get("forecast_ranking_replacement_reason", item.get("replacement_reason", "")))
        )
        task_replacement_records = [
            {
                "replacement_reason": str(item.get("replacement_reason", "")),
                "raw_action": int(item.get("raw_action", -1)),
                "final_action": int(item.get("final_action", -1)),
                "raw_action_name": str(item.get("raw_action_name", "")),
                "final_action_name": str(item.get("final_action_name", "")),
                "step": int(item.get("step", -1)),
                "distance_to_taper": float(item.get("distance_to_taper", INF_TTC)),
                "decision_distance_to_taper": float(item.get("decision_distance_to_taper", INF_TTC)),
                "task_deadline_urgency": float(item.get("task_deadline_urgency", 0.0)),
                "forecast_aware_task_improvement": float(item.get("forecast_aware_task_improvement", 0.0)),
                "forecast_aware_best_task_risk": float(item.get("forecast_aware_best_task_risk", 0.0)),
                "forecast_aware_safety_risk": float(item.get("forecast_aware_safety_risk", 0.0)),
                "forecast_aware_uncertainty": float(item.get("forecast_aware_uncertainty", 0.0)),
                "forecast_aware_target_front_gap": float(item.get("forecast_aware_target_front_gap", INF_TTC)),
                "forecast_aware_target_rear_gap": float(item.get("forecast_aware_target_rear_gap", INF_TTC)),
                "task_backstop_risk_module_score": item.get("task_backstop_risk_module_score"),
                "task_backstop_risk_module_uncertainty": item.get(
                    "task_backstop_risk_module_uncertainty"
                ),
                "task_backstop_risk_module_pass": bool(item.get("task_backstop_risk_module_pass", False)),
                "min_distance": float(item.get("min_distance", INF_TTC)),
                "min_ttc": float(item.get("min_ttc", INF_TTC)),
                "max_drac": float(item.get("max_drac", 0.0)),
                "geometric_overlap": bool(item.get("geometric_overlap", False)),
                "closest_vehicle_id": str(item.get("closest_vehicle_id", "")),
            }
            for item in self._task_replacements
        ]
        forecast_ranking_replacement_records = [
            {
                "replacement_reason": str(item.get("replacement_reason", "")),
                "forecast_ranking_replacement_reason": str(
                    item.get("forecast_ranking_replacement_reason", "")
                ),
                "raw_action": int(item.get("raw_action", -1)),
                "final_action": int(item.get("final_action", -1)),
                "raw_action_name": str(item.get("raw_action_name", "")),
                "final_action_name": str(item.get("final_action_name", "")),
                "step": int(item.get("step", -1)),
                "forecast_aware_raw_score": item.get("forecast_aware_raw_score"),
                "forecast_aware_best_score": item.get("forecast_aware_best_score"),
                "forecast_aware_score_improvement": item.get("forecast_aware_score_improvement"),
                "forecast_aware_best_task_risk": item.get("forecast_aware_best_task_risk"),
                "forecast_aware_safety_risk": item.get("forecast_aware_safety_risk"),
                "forecast_aware_uncertainty": item.get("forecast_aware_uncertainty"),
                "forecast_ranking_risk_module_score": item.get("forecast_ranking_risk_module_score"),
                "forecast_ranking_risk_module_uncertainty": item.get(
                    "forecast_ranking_risk_module_uncertainty"
                ),
                "forecast_ranking_risk_module_pass": bool(
                    item.get("forecast_ranking_risk_module_pass", False)
                ),
                "min_distance": float(item.get("min_distance", INF_TTC)),
                "min_ttc": float(item.get("min_ttc", INF_TTC)),
                "max_drac": float(item.get("max_drac", 0.0)),
                "geometric_overlap": bool(item.get("geometric_overlap", False)),
                "closest_vehicle_id": str(item.get("closest_vehicle_id", "")),
            }
            for item in self._forecast_ranking_replacements
        ]
        execution_by_decision = {
            int(item.get("decision_index", -1)): item
            for item in self._action_execution_records
            if int(item.get("decision_index", -1)) >= 0
        }
        score_records = []
        for item in self._interventions:
            execution = execution_by_decision.get(int(item.get("decision_index", -1)), {})
            safety_final_action = int(item.get("final_action", item.get("raw_action", -1)))
            executed_action = int(execution.get("final_action", safety_final_action))
            score_records.append(
                {
                "score_record_stage": "safety_shield_pre_forecast_ranking",
                "replacement_reason": str(item.get("replacement_reason", "")),
                "raw_risk_score": float(item.get("risk_before", 0.0)),
                "final_risk_score": float(item.get("risk_after", 0.0)),
                "best_candidate_risk_score": float(item.get("best_candidate_risk", item.get("risk_after", 0.0))),
                "replacement_risk_delta": float(item.get("replacement_risk_delta", 0.0)),
                "best_candidate_risk_delta": float(item.get("best_candidate_risk_delta", 0.0)),
                "raw_candidate_legal": bool(item.get("raw_candidate_legal", True)),
                "final_candidate_legal": bool(item.get("final_candidate_legal", True)),
                "emergency_fallback": bool(item.get("emergency_fallback", False)),
                "emergency_trigger": bool(item.get("emergency_trigger", False)),
                "emergency_reason": str(item.get("emergency_reason", "")),
                "emergency_saturated_count": int(item.get("emergency_saturated_count", 0)),
                "emergency_saturated_required": int(item.get("emergency_saturated_required", 0)),
                "raw_action": int(item.get("raw_action", -1)),
                "final_action": safety_final_action,
                "raw_action_name": str(item.get("raw_action_name", "")),
                "final_action_name": str(item.get("final_action_name", "")),
                "safety_shield_final_action": safety_final_action,
                "safety_shield_final_action_name": str(item.get("final_action_name", "")),
                "executed_action": executed_action,
                "executed_action_name": str(
                    execution.get("final_action_name", item.get("final_action_name", ""))
                ),
                "execution_path": str(execution.get("execution_path", "safety_shield")),
                "forecast_ranking_replaced": bool(
                    execution.get("forecast_ranking_replacement", False)
                ),
                "task_backstop_replaced": bool(execution.get("task_replacement", False)),
                "best_candidate_action": int(item.get("best_candidate_action", -1)),
                "best_candidate_action_name": str(item.get("best_candidate_action_name", "")),
                "step": int(item.get("step", -1)),
                "min_distance": float(item.get("min_distance", INF_TTC)),
                "min_ttc": float(item.get("min_ttc", INF_TTC)),
                "max_drac": float(item.get("max_drac", 0.0)),
                "geometric_overlap": bool(item.get("geometric_overlap", False)),
                "closest_vehicle_id": str(item.get("closest_vehicle_id", "")),
            }
            )
        reward_component_totals = {
            name: float(sum(record.get(name, 0.0) for record in self._reward_component_records))
            for name in (
                "progress_reward",
                "speed_reward",
                "terminal_reward",
                "lane_oob_penalty",
                "safety_penalty",
                "safety_forecast_shaping",
                "shield_guided_shaping",
                "merge_timing_shaping",
                "total_episode_reward",
            )
        }
        return {
            "seed": self.seed_value,
            "episode_seed": self.seed_value,
            "episode_index": int(self._active_episode_index),
            "episode_seed_schedule": str(
                self.config.get("run", {}).get("episode_seed_schedule", "fixed_legacy")
            ),
            "safety_metric_version": str(
                self.config.risk_module.get("safety_metric_version", SAFETY_METRIC_VERSION)
            ),
            "curriculum_profile": self._curriculum_profile,
            "done_reason": self._last_done_reason,
            "taper_miss": self._last_done_reason == "taper_miss",
            "steps": self._episode_step,
            "control_decisions": int(self._decision_index),
            "completion_time": completion_time,
            "collision": any(collisions),
            "geometric_overlap": any(geometric_overlaps),
            "near_miss": any(near_misses),
            "proxy_collision": bool(proxy_collision),
            "safety_violation": safety_violation,
            "proxy_collision_count": proxy_collision_count,
            "safety_violation_count": safety_violation_count,
            "min_distance_le_collision_threshold_count": proxy_collision_count,
            "min_distance": min_distance,
            "ttc_p1": ttc_p1,
            "drac_p99": drac_raw,
            "drac_p99_raw": drac_raw,
            "drac_p99_capped": drac_capped,
            "ego_speed_mean": ego_speed_mean,
            "ego_speed_p10": ego_speed_p10,
            "hard_brake_count": int(hard_brake_count),
            "hard_brake_rate": hard_brake_rate,
            "intervention_count": len(self._interventions),
            "shield_call_count": len(self._interventions),
            "actual_replacement_count": replacement_count,
            "actual_replacement_rate": float(replacement_count / len(self._interventions)) if self._interventions else 0.0,
            "actual_replacement_rate_semantics": "replacement_per_shield_call_rate",
            "episodes_with_replacement_rate": float(replacement_count > 0),
            "replacement_per_shield_call_rate": (
                float(replacement_count / len(self._interventions))
                if self._interventions
                else 0.0
            ),
            "mean_replacements_per_episode": float(replacement_count),
            "raw_action_lane_oob": bool(self._raw_action_lane_oob_count > 0),
            "final_action_lane_oob": bool(self._final_action_lane_oob_count > 0),
            "raw_action_lane_oob_count": int(self._raw_action_lane_oob_count),
            "final_action_lane_oob_count": int(self._final_action_lane_oob_count),
            "prevented_lane_oob_count": int(self._prevented_lane_oob_count),
            "reward_components": reward_component_totals,
            **reward_component_totals,
            "task_replacement_count": int(task_replacement_count),
            "task_replacement_rate": float(task_replacement_count / max(self._decision_index, 1)),
            "mean_task_replacements": float(task_replacement_count),
            "task_replacement_records": task_replacement_records,
            "task_replacement_reason_counts": dict(task_replacement_reason_counts),
            "forecast_aware_candidate_ranking_mode": self._forecast_aware_candidate_ranking_mode(),
            "forecast_ranking_replacement_count": int(forecast_ranking_replacement_count),
            "forecast_ranking_replacement_rate": float(
                forecast_ranking_replacement_count / max(self._decision_index, 1)
            ),
            "mean_forecast_ranking_replacements": float(forecast_ranking_replacement_count),
            "forecast_ranking_replacement_records": forecast_ranking_replacement_records,
            "forecast_ranking_replacement_reason_counts": dict(
                forecast_ranking_replacement_reason_counts
            ),
            "forecast_actor_coverage_complete_count": int(forecast_coverage_complete_count),
            "forecast_record_count": int(len(forecast_records)),
            "forecast_actor_coverage_complete_rate": (
                float(forecast_coverage_complete_count / len(forecast_records)) if forecast_records else 0.0
            ),
            "forecast_gap_consistency_pass_count": int(forecast_gap_consistency_pass_count),
            "forecast_gap_consistency_checkable_count": int(
                forecast_gap_consistency_checkable_count
            ),
            "forecast_gap_consistency_checkable_rate": (
                float(forecast_gap_consistency_checkable_count / len(forecast_records))
                if forecast_records
                else 0.0
            ),
            "forecast_gap_consistency_pass_rate": (
                float(
                    forecast_gap_consistency_pass_count
                    / forecast_gap_consistency_checkable_count
                )
                if forecast_gap_consistency_checkable_count
                else 0.0
            ),
            "forecast_gap_consistency_failure_reason_counts": dict(
                forecast_gap_failure_reason_counts
            ),
            "wcdt_relevant_actor_coverage_count": int(wcdt_relevant_coverage_count),
            "wcdt_relevant_actor_coverage_rate": (
                float(wcdt_relevant_coverage_count / len(forecast_records))
                if forecast_records
                else 0.0
            ),
            "combined_forecast_safety_coverage_count": int(safety_actor_coverage_count),
            "combined_forecast_safety_coverage_rate": (
                float(safety_actor_coverage_count / len(forecast_records))
                if forecast_records
                else 0.0
            ),
            "actor_selector_overflow_count": int(selector_overflow_count),
            "actor_selector_overflow_rate": (
                float(selector_overflow_count / len(forecast_records))
                if forecast_records
                else 0.0
            ),
            "critical_actor_overflow_count": int(critical_overflow_count),
            "critical_actor_overflow_rate": (
                float(critical_overflow_count / len(forecast_records))
                if forecast_records
                else 0.0
            ),
            "critical_wcdt_coverage_count": int(critical_wcdt_coverage_count),
            "critical_wcdt_coverage_rate": (
                float(critical_wcdt_coverage_count / len(forecast_records))
                if forecast_records
                else 0.0
            ),
            "combined_critical_coverage_count": int(
                combined_critical_coverage_count
            ),
            "combined_critical_coverage_rate": (
                float(combined_critical_coverage_count / len(forecast_records))
                if forecast_records
                else 0.0
            ),
            "cv_fallback_overflow_count": int(cv_fallback_overflow_count),
            "cv_fallback_overflow_rate": (
                float(cv_fallback_overflow_count / len(forecast_records))
                if forecast_records
                else 0.0
            ),
            "cv_fallback_usage_count": int(cv_fallback_usage_count),
            "cv_fallback_usage_rate": (
                float(cv_fallback_usage_count / len(forecast_records))
                if forecast_records
                else 0.0
            ),
            "task_backstop_watch_count": int(task_backstop_watch_count),
            "task_backstop_eligible_count": int(task_backstop_eligible_count),
            "task_backstop_veto_reason_counts": dict(task_backstop_veto_reason_counts),
            "fallback_count": sum(1 for item in self._interventions if item.get("fallback")),
            "emergency_fallback_count": int(emergency_fallback_count),
            "emergency_fallback_rate": (
                float(emergency_fallback_count / len(self._interventions)) if self._interventions else 0.0
            ),
            "replacement_reason_counts": dict(reason_counts),
            "raw_action_histogram": dict(raw_actions),
            "safety_shield_action_histogram": dict(safety_shield_actions),
            "executed_action_histogram": dict(executed_actions),
            "final_action_histogram": dict(executed_actions),
            "action_execution_records": list(self._action_execution_records),
            "safety_shield_score_records": score_records,
            "shield_score_records_semantics": "safety_shield_pre_forecast_ranking",
            # Legacy alias. Consumers needing executed actions must use
            # action_execution_records or executed_action_histogram.
            "shield_score_records": score_records,
            "shield_guided_reward_summary": self._shield_guided_reward_summary(),
            "first_merge_request_step": self._first_merge_request_step,
            "first_merge_request_distance_to_taper": self._first_merge_request_distance_to_taper,
            "first_target_lane_entry_step": self._first_target_lane_entry_step,
            "first_target_lane_entry_distance_to_taper": self._first_target_lane_entry_distance_to_taper,
            "safe_merge_opportunity_count": int(self._safe_merge_opportunity_count),
            "missed_safe_merge_opportunity_count": int(self._missed_safe_merge_opportunity_count),
            "missed_safe_merge_opportunity_rate": (
                float(self._missed_safe_merge_opportunity_count / self._safe_merge_opportunity_count)
                if self._safe_merge_opportunity_count
                else 0.0
            ),
            "task_merge_opportunity_count": int(task_merge_count),
            "task_would_merge_count": int(task_would_merge_count),
            "task_would_merge_rate": (
                float(task_would_merge_count / max(task_merge_count, 1)) if task_merge_count else 0.0
            ),
            "task_missed_merge_count": int(task_missed_merge_count),
            "task_missed_merge_rate": (
                float(task_missed_merge_count / max(task_merge_count, 1)) if task_merge_count else 0.0
            ),
            "deadline_safe_merge_opportunity_count": int(len(task_deadline_records)),
            "deadline_missed_safe_merge_count": int(deadline_missed_count),
            "deadline_missed_safe_merge_rate": (
                float(deadline_missed_count / len(task_deadline_records)) if task_deadline_records else 0.0
            ),
            "missed_safe_merge_after_urgency_0_5_count": int(missed_after_urgency_count),
            "safe_merge_after_urgency_0_5_count": int(len(urgency_records)),
            "missed_safe_merge_after_urgency_0_5_rate": (
                float(missed_after_urgency_count / len(urgency_records)) if urgency_records else 0.0
            ),
            "no_merge_request_before_taper": bool(no_merge_request_before_taper),
            "no_merge_request_before_taper_count": int(no_merge_request_before_taper),
            "performance": self.performance.summary(
                steps=int(self._episode_step),
                episodes=1,
                extra={
                    "sumo_restarts": int(self._sumo_restart_count),
                    "sumo_reloads": int(self._sumo_reload_count),
                    "subscription_fallback_count": int(self._subscription_fallback_count),
                    "subscription_error_count": int(self._subscription_error_count),
                },
            ),
        }

    def _shield_guided_reward_summary(self) -> dict[str, Any]:
        records = [record for record in self._reward_debug_records if record.get("available")]
        if not records:
            return {"available": False, "count": 0}
        penalties = [float(record.get("shield_guided_reward_penalty", 0.0)) for record in records]
        raw_risk = [float(record.get("raw_action_risk", 0.0)) for record in records]
        best_risk = [float(record.get("best_candidate_risk", 0.0)) for record in records]
        margins = [float(record.get("risk_margin", 0.0)) for record in records]
        would_replace_count = sum(1 for record in records if bool(record.get("would_replace", False)))
        return {
            "available": True,
            "count": len(records),
            "would_replace_count": int(would_replace_count),
            "would_replace_rate": float(would_replace_count / len(records)),
            "penalty_sum": float(np.sum(penalties)),
            "penalty_mean": float(np.mean(penalties)),
            "raw_action_risk_mean": float(np.mean(raw_risk)),
            "best_candidate_risk_mean": float(np.mean(best_risk)),
            "risk_margin_mean": float(np.mean(margins)),
        }

    def trajectory_window_samples(
        self,
        *,
        include_dimensions: bool = False,
    ) -> tuple[np.ndarray, ...]:
        """Return padded [sample, agent, time, state] windows from the episode."""

        hist = self.history_steps
        horizon = int(self.config.scenario.forecast_horizon_steps)
        trajectory_actor_capacity = int(getattr(self, "trajectory_actor_capacity", self.top_k))
        max_agents = trajectory_actor_capacity + 1
        frames = self._trajectory_frames
        frame_metadata = getattr(self, "_trajectory_frame_metadata", [])
        self._last_trajectory_window_metadata = {
            "trajectory_window_end_step": np.zeros((0,), dtype=np.int64),
            "trajectory_decision_index": np.zeros((0,), dtype=np.int64),
            "trajectory_episode_seed": np.zeros((0,), dtype=np.int64),
            "critical_actor_count": np.zeros((0,), dtype=np.int64),
            "contextual_actor_count": np.zeros((0,), dtype=np.int64),
            "critical_actor_overflow": np.zeros((0,), dtype=np.float32),
            "contextual_actor_truncated_count": np.zeros((0,), dtype=np.int64),
            "critical_actor_metadata_json": np.zeros((0,), dtype="<U2"),
            "dropped_critical_actor_metadata_json": np.zeros((0,), dtype="<U2"),
        }
        if len(frames) < hist + horizon:
            result = (
                np.zeros((0, max_agents, hist, 5), dtype=np.float32),
                np.zeros((0, max_agents, horizon, 5), dtype=np.float32),
                np.zeros((0, max_agents), dtype=np.float32),
                np.zeros((0, max_agents), dtype=np.int64),
                np.zeros((0, max_agents), dtype=np.int64),
                np.zeros((0, max_agents, hist), dtype=np.float32),
                np.zeros((0, max_agents, horizon), dtype=np.float32),
                np.full((0, max_agents, hist), -1, dtype=np.int64),
                np.zeros((0, max_agents, hist), dtype=np.int64),
                np.full((0, max_agents, horizon), -1, dtype=np.int64),
                np.zeros((0, max_agents, horizon), dtype=np.int64),
            )
            if include_dimensions:
                return (
                    *result,
                    np.full((0, max_agents), 4.8, dtype=np.float32),
                    np.full((0, max_agents), 1.8, dtype=np.float32),
                    np.zeros((0, max_agents), dtype=np.float32),
                    np.zeros((0, max_agents), dtype=np.float32),
                    np.zeros((0,), dtype=np.int64),
                    np.zeros((0,), dtype=np.float32),
                )
            return result
        history_samples: list[np.ndarray] = []
        future_samples: list[np.ndarray] = []
        masks: list[np.ndarray] = []
        lane_indices: list[np.ndarray] = []
        edge_roles: list[np.ndarray] = []
        history_valid_masks: list[np.ndarray] = []
        future_valid_masks: list[np.ndarray] = []
        history_lane_indices: list[np.ndarray] = []
        history_edge_roles: list[np.ndarray] = []
        future_lane_indices: list[np.ndarray] = []
        future_edge_roles: list[np.ndarray] = []
        agent_lengths: list[np.ndarray] = []
        agent_widths: list[np.ndarray] = []
        relevance_masks: list[np.ndarray] = []
        relevance_scores: list[np.ndarray] = []
        relevant_counts: list[int] = []
        selector_overflows: list[float] = []
        window_end_steps: list[int] = []
        window_decision_indices: list[int] = []
        window_episode_seeds: list[int] = []
        critical_actor_counts: list[int] = []
        contextual_actor_counts: list[int] = []
        critical_actor_overflows: list[float] = []
        contextual_actor_truncated_counts: list[int] = []
        critical_actor_metadata_json: list[str] = []
        dropped_critical_actor_metadata_json: list[str] = []
        for end_idx in range(hist, len(frames) - horizon + 1):
            latest = frames[end_idx - 1]
            if self.ego_id not in latest:
                continue
            ego = latest[self.ego_id]
            agent_ids = [self.ego_id]
            selection = select_merge_relevant_actors(
                self.config,
                ego,
                list(latest.values()),
                trajectory_actor_capacity,
            )
            agent_ids.extend(selection.selected_actor_ids)
            history = np.zeros((max_agents, hist, 5), dtype=np.float32)
            future = np.zeros((max_agents, horizon, 5), dtype=np.float32)
            mask = np.zeros((max_agents,), dtype=np.float32)
            sample_lane_indices = np.full((max_agents,), -1, dtype=np.int64)
            sample_edge_roles = np.zeros((max_agents,), dtype=np.int64)
            history_valid_mask = np.zeros((max_agents, hist), dtype=np.float32)
            future_valid_mask = np.zeros((max_agents, horizon), dtype=np.float32)
            sample_history_lane_indices = np.full((max_agents, hist), -1, dtype=np.int64)
            sample_history_edge_roles = np.zeros((max_agents, hist), dtype=np.int64)
            sample_future_lane_indices = np.full((max_agents, horizon), -1, dtype=np.int64)
            sample_future_edge_roles = np.zeros((max_agents, horizon), dtype=np.int64)
            sample_agent_lengths = np.full((max_agents,), 4.8, dtype=np.float32)
            sample_agent_widths = np.full((max_agents,), 1.8, dtype=np.float32)
            sample_relevance_mask = np.zeros((max_agents,), dtype=np.float32)
            sample_relevance_score = np.zeros((max_agents,), dtype=np.float32)
            for agent_idx, vehicle_id in enumerate(agent_ids[:max_agents]):
                mask[agent_idx] = 1.0
                latest_state = latest.get(vehicle_id)
                if latest_state is not None:
                    sample_lane_indices[agent_idx] = int(latest_state.lane_index)
                    sample_edge_roles[agent_idx] = int(edge_role(self.config, latest_state.edge_id, latest_state.lane_index))
                    sample_agent_lengths[agent_idx] = float(latest_state.length)
                    sample_agent_widths[agent_idx] = float(latest_state.width)
                    metadata = selection.actor_metadata.get(vehicle_id)
                    if metadata is not None:
                        sample_relevance_mask[agent_idx] = float(metadata.relevant)
                        urgency_gap = max(
                            0.0,
                            min(metadata.current_surface_gap, metadata.effective_gap),
                        )
                        ttc_score = 0.0 if metadata.ttc >= INF_TTC else 1.0 / (1.0 + metadata.ttc)
                        sample_relevance_score[agent_idx] = float(
                            max(1.0 / (1.0 + urgency_gap), ttc_score)
                        )
                last_state = None
                for step_idx, frame in enumerate(frames[end_idx - hist : end_idx]):
                    observed_state = frame.get(vehicle_id)
                    state = observed_state or last_state
                    if state is None:
                        continue
                    history[agent_idx, step_idx] = np.asarray(state.as_vector(), dtype=np.float32)
                    if observed_state is not None:
                        history_valid_mask[agent_idx, step_idx] = 1.0
                        sample_history_lane_indices[agent_idx, step_idx] = int(observed_state.lane_index)
                        sample_history_edge_roles[agent_idx, step_idx] = int(
                            edge_role(self.config, observed_state.edge_id, observed_state.lane_index)
                        )
                    last_state = state
                for step_idx, frame in enumerate(frames[end_idx : end_idx + horizon]):
                    state = frame.get(vehicle_id)
                    if state is None:
                        continue
                    future[agent_idx, step_idx] = np.asarray(state.as_vector(), dtype=np.float32)
                    future_valid_mask[agent_idx, step_idx] = 1.0
                    sample_future_lane_indices[agent_idx, step_idx] = int(state.lane_index)
                    sample_future_edge_roles[agent_idx, step_idx] = int(
                        edge_role(self.config, state.edge_id, state.lane_index)
                    )
            history_samples.append(history)
            future_samples.append(future)
            masks.append(mask)
            lane_indices.append(sample_lane_indices)
            edge_roles.append(sample_edge_roles)
            history_valid_masks.append(history_valid_mask)
            future_valid_masks.append(future_valid_mask)
            history_lane_indices.append(sample_history_lane_indices)
            history_edge_roles.append(sample_history_edge_roles)
            future_lane_indices.append(sample_future_lane_indices)
            future_edge_roles.append(sample_future_edge_roles)
            agent_lengths.append(sample_agent_lengths)
            agent_widths.append(sample_agent_widths)
            relevance_masks.append(sample_relevance_mask)
            relevance_scores.append(sample_relevance_score)
            relevant_counts.append(int(selection.relevant_count))
            selector_overflows.append(float(selection.overflow))
            critical_actor_counts.append(int(selection.critical_count))
            contextual_actor_counts.append(int(selection.contextual_count))
            critical_actor_overflows.append(float(selection.critical_overflow))
            contextual_actor_truncated_counts.append(
                int(len(selection.contextual_truncated_ids))
            )
            critical_actor_metadata_json.append(
                _actor_metadata_json(selection, selection.critical_actor_ids)
            )
            dropped_critical_actor_metadata_json.append(
                _actor_metadata_json(selection, selection.dropped_critical_ids)
            )
            metadata = frame_metadata[end_idx - 1] if end_idx - 1 < len(frame_metadata) else {}
            window_end_steps.append(int(metadata.get("simulation_step", -1)))
            window_decision_indices.append(int(metadata.get("decision_index", -1)))
            window_episode_seeds.append(
                int(metadata.get("episode_seed", getattr(self, "seed_value", -1)))
            )
        if not history_samples:
            result = (
                np.zeros((0, max_agents, hist, 5), dtype=np.float32),
                np.zeros((0, max_agents, horizon, 5), dtype=np.float32),
                np.zeros((0, max_agents), dtype=np.float32),
                np.zeros((0, max_agents), dtype=np.int64),
                np.zeros((0, max_agents), dtype=np.int64),
                np.zeros((0, max_agents, hist), dtype=np.float32),
                np.zeros((0, max_agents, horizon), dtype=np.float32),
                np.full((0, max_agents, hist), -1, dtype=np.int64),
                np.zeros((0, max_agents, hist), dtype=np.int64),
                np.full((0, max_agents, horizon), -1, dtype=np.int64),
                np.zeros((0, max_agents, horizon), dtype=np.int64),
            )
            if include_dimensions:
                return (
                    *result,
                    np.full((0, max_agents), 4.8, dtype=np.float32),
                    np.full((0, max_agents), 1.8, dtype=np.float32),
                    np.zeros((0, max_agents), dtype=np.float32),
                    np.zeros((0, max_agents), dtype=np.float32),
                    np.zeros((0,), dtype=np.int64),
                    np.zeros((0,), dtype=np.float32),
                )
            return result
        result = (
            np.stack(history_samples, axis=0),
            np.stack(future_samples, axis=0),
            np.stack(masks, axis=0),
            np.stack(lane_indices, axis=0),
            np.stack(edge_roles, axis=0),
            np.stack(history_valid_masks, axis=0),
            np.stack(future_valid_masks, axis=0),
            np.stack(history_lane_indices, axis=0),
            np.stack(history_edge_roles, axis=0),
            np.stack(future_lane_indices, axis=0),
            np.stack(future_edge_roles, axis=0),
        )
        self._last_trajectory_window_metadata = {
            "trajectory_window_end_step": np.asarray(window_end_steps, dtype=np.int64),
            "trajectory_decision_index": np.asarray(window_decision_indices, dtype=np.int64),
            "trajectory_episode_seed": np.asarray(window_episode_seeds, dtype=np.int64),
            "critical_actor_count": np.asarray(critical_actor_counts, dtype=np.int64),
            "contextual_actor_count": np.asarray(contextual_actor_counts, dtype=np.int64),
            "critical_actor_overflow": np.asarray(
                critical_actor_overflows,
                dtype=np.float32,
            ),
            "contextual_actor_truncated_count": np.asarray(
                contextual_actor_truncated_counts,
                dtype=np.int64,
            ),
            "critical_actor_metadata_json": np.asarray(
                critical_actor_metadata_json,
            ),
            "dropped_critical_actor_metadata_json": np.asarray(
                dropped_critical_actor_metadata_json,
            ),
        }
        if include_dimensions:
            return (
                *result,
                np.stack(agent_lengths, axis=0),
                np.stack(agent_widths, axis=0),
                np.stack(relevance_masks, axis=0),
                np.stack(relevance_scores, axis=0),
                np.asarray(relevant_counts, dtype=np.int64),
                np.asarray(selector_overflows, dtype=np.float32),
            )
        return result

    def trajectory_window_metadata(self) -> dict[str, np.ndarray]:
        return {
            key: np.asarray(value).copy()
            for key, value in self._last_trajectory_window_metadata.items()
        }
