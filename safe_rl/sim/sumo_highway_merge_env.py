from __future__ import annotations

import math
import os
import shutil
import sys
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from safe_rl.prediction.forecast_feature_augmentor import ForecastFeatureAugmentor
from safe_rl.risk.candidate_risk_ranker import CandidateRiskRanker
from safe_rl.risk.merge_local import is_candidate_legal, merge_local_stats
from safe_rl.shield.safety_shield import SafetyShield
from safe_rl.sim.action_space import ACTIONS, decode_action
from safe_rl.sim.gym_compat import gym, spaces
from safe_rl.sim.history_buffer import HistoryBuffer
from safe_rl.sim.metrics import INF_TTC, compute_step_metrics, explicit_risk_features
from safe_rl.sim.scenario_semantics import (
    distance_to_taper,
    edge_role,
    is_auxiliary_edge,
    is_ramp_edge,
    is_taper_miss,
    is_target_lane,
    merge_target_lane,
    merge_zone_edges,
    target_lane_edges,
    target_lane_mapping,
)
from safe_rl.sim.types import StepMetrics, VehicleState


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
    ):
        self.config = config
        self.seed_value = int(seed if seed is not None else config.run.seed)
        self.ego_id = config.scenario.ego_id
        self.step_length = float(config.scenario.step_length)
        self.control_interval_steps = int(config.scenario.control_interval_steps)
        self.episode_steps = int(float(config.scenario.episode_seconds) / self.step_length)
        self.top_k = int(config.scenario.top_k_neighbors)
        self.history_steps = int(config.scenario.history_steps)
        self.forecast_enabled = bool(config.forecast_features.enabled or config.rl.use_wcdt_forecast_features)
        self.forecast_augmentor = forecast_augmentor
        self.shield = shield
        self.reward_risk_model = reward_risk_model
        self.reward_ranker = CandidateRiskRanker(config, reward_risk_model) if reward_risk_model is not None else None
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

        self.history = HistoryBuffer(self.history_steps, max_agents=self.top_k + 1)
        self._traci_module = None
        self._traci = None
        self._conn_label = f"safe_rl_{uuid.uuid4().hex[:8]}"
        self._episode_step = 0
        self._last_ego_speed = 0.0
        self._last_ego_x = 0.0
        self._episode_metrics: list[StepMetrics] = []
        self._ego_speeds: list[float] = []
        self._interventions: list[dict[str, Any]] = []
        self._reward_debug_records: list[dict[str, Any]] = []
        self._last_reward_debug: dict[str, Any] = {}
        self._trajectory_frames: list[dict[str, VehicleState]] = []
        self._last_done_reason = ""
        self._curriculum_profile = "disabled"
        self._curriculum_applied = False

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
        self._traci_module = traci
        return traci

    def _add_sumo_tools_path(self) -> None:
        candidates: list[Path] = []
        if os.environ.get("SUMO_HOME"):
            candidates.append(Path(os.environ["SUMO_HOME"]) / "tools")
        sumo_binary = str(self.config.scenario.get("sumo_binary", "sumo"))
        resolved = shutil.which(sumo_binary) or (sumo_binary if Path(sumo_binary).exists() else "")
        if resolved:
            candidates.append(Path(resolved).resolve().parents[1] / "tools")
        candidates.append(Path(r"E:/Program Files/sumo-1.22.0/tools"))
        for candidate in candidates:
            if candidate.is_dir() and str(candidate) not in sys.path:
                sys.path.append(str(candidate))

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        if seed is not None:
            self.seed_value = int(seed)
        self._close_sumo()
        self._start_sumo()
        self.history.clear()
        self._episode_step = 0
        self._episode_metrics.clear()
        self._ego_speeds.clear()
        self._interventions.clear()
        self._reward_debug_records.clear()
        self._last_reward_debug = {}
        self._trajectory_frames.clear()
        self._last_done_reason = ""
        self._curriculum_profile = self._select_curriculum_profile()
        self._curriculum_applied = False
        if self.shield is not None and hasattr(self.shield, "reset_episode_state"):
            self.shield.reset_episode_state()

        for _ in range(max(1, self.history_steps)):
            self._simulation_step()
            if not self._curriculum_applied:
                self._apply_curriculum_perturbation()
            self._configure_ego_control()
            states = self._collect_states()
            self.history.append(states)
            self._trajectory_frames.append({state.vehicle_id: state for state in states})
            if self.ego_id in self.history.latest():
                break

        ego = self._get_ego()
        self._last_ego_speed = ego.speed if ego else 0.0
        self._last_ego_x = ego.x if ego else 0.0
        return self._build_observation(), self._info()

    def step(self, action):
        raw_action = decode_action(int(action))
        final_action = raw_action
        intervention = None
        context = self.get_risk_context()
        if self.shield is not None and self.shield.enabled:
            final_action, intervention = self.shield.select_action(raw_action, context)
            self._interventions.append(intervention)

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
            if self.record_trajectory_samples:
                self._trajectory_frames.append({state.vehicle_id: state for state in states})

        ego = self._get_ego()
        states = self._collect_states()
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
        if ego is not None:
            self._ego_speeds.append(float(ego.speed))

        terminated, done_reason = self._done(metrics)
        self._last_done_reason = done_reason
        truncated = self._episode_step >= self.episode_steps
        reward = self._reward(prev_x, ego, metrics, done_reason, raw_action=raw_action, risk_context=context)
        obs = self._build_observation()
        info = self._info(metrics=metrics, done_reason=done_reason, intervention=intervention)
        info["raw_action"] = int(raw_action.index)
        info["final_action"] = int(final_action.index)
        self._last_ego_speed = ego.speed if ego else 0.0
        self._last_ego_x = ego.x if ego else self._last_ego_x
        return obs, float(reward), bool(terminated), bool(truncated), info

    def close(self):
        self._close_sumo()

    def _start_sumo(self) -> None:
        traci = self._import_traci()
        sumocfg = str(Path(self.config.scenario.sumocfg).resolve())
        sumo_binary = self.config.scenario.get("sumo_binary", "sumo")
        cmd = [
            sumo_binary,
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
        retries = int(self.config.scenario.get("sumo_start_retries", 5))
        delay = float(self.config.scenario.get("sumo_start_retry_delay", 0.25))
        last_error: Exception | None = None
        for attempt in range(max(1, retries)):
            try:
                self._conn_label = f"safe_rl_{uuid.uuid4().hex[:8]}"
                traci.start(cmd, label=self._conn_label, numRetries=20)
                self._traci = traci.getConnection(self._conn_label)
                return
            except Exception as exc:
                last_error = exc
                self._cleanup_failed_traci_start(traci)
                time.sleep(delay * (attempt + 1))
        raise RuntimeError(f"Failed to start SUMO after {retries} attempts: {last_error}") from last_error

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

    def _simulation_step(self) -> None:
        self._traci.simulationStep()
        if self.sumo_step_delay_ms > 0:
            time.sleep(self.sumo_step_delay_ms / 1000.0)

    def _collect_states(self) -> list[VehicleState]:
        states: list[VehicleState] = []
        vehicle_api = self._traci.vehicle
        for vehicle_id in vehicle_api.getIDList():
            x, y = vehicle_api.getPosition(vehicle_id)
            sumo_angle = vehicle_api.getAngle(vehicle_id)
            heading = math.radians(90.0 - sumo_angle)
            lane_id = vehicle_api.getLaneID(vehicle_id)
            lane_index = int(vehicle_api.getLaneIndex(vehicle_id))
            speed = float(vehicle_api.getSpeed(vehicle_id))
            accel = float(vehicle_api.getAcceleration(vehicle_id))
            states.append(
                VehicleState(
                    vehicle_id=vehicle_id,
                    x=float(x),
                    y=float(y),
                    heading=float(heading),
                    speed=speed,
                    lane_index=lane_index,
                    lane_id=lane_id,
                    lane_pos=float(vehicle_api.getLanePosition(vehicle_id)),
                    edge_id=str(vehicle_api.getRoadID(vehicle_id)),
                    length=float(vehicle_api.getLength(vehicle_id)),
                    width=float(vehicle_api.getWidth(vehicle_id)),
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

    def _lane_count(self, edge_id: str) -> int:
        try:
            return int(self._traci.edge.getLaneNumber(edge_id))
        except Exception:
            latest = self.history.latest()
            same_edge = [state.lane_index for state in latest.values() if state.edge_id == edge_id]
            return max(same_edge) + 1 if same_edge else 1

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
        reward = 0.0
        self._last_reward_debug = {}
        if ego is not None:
            reward += reward_cfg.progress * max(0.0, ego.x - prev_x)
            reward += reward_cfg.speed * min(ego.speed, 33.33)
        if done_reason == "merge_success":
            reward += reward_cfg.merge_success
        if metrics.collision:
            reward += reward_cfg.collision
        if metrics.near_miss:
            reward += reward_cfg.near_miss
        if metrics.low_ttc:
            reward += reward_cfg.low_ttc
        if metrics.high_drac:
            reward += reward_cfg.high_drac
        if metrics.hard_brake:
            reward += reward_cfg.hard_brake
        if metrics.lane_oob:
            reward += reward_cfg.lane_oob
        reward_profile = str(self.config.rl.get("reward_profile", "default"))
        if reward_profile in {"safety_forecast", "shield_guided_forecast"}:
            reward += self._safety_forecast_reward_adjustment(ego, metrics)
        if reward_profile == "shield_guided_forecast":
            shield_penalty, reward_debug = self._shield_guided_reward_adjustment(raw_action, risk_context)
            reward += shield_penalty
            self._last_reward_debug = reward_debug
            self._reward_debug_records.append(reward_debug)
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

        raw_prediction = self.reward_risk_model.predict(raw_action, context)
        ranked = self.reward_ranker.rank(raw_action, context)
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

    def _info(
        self,
        metrics: StepMetrics | None = None,
        done_reason: str = "",
        intervention: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        info: dict[str, Any] = {
            "seed": self.seed_value,
            "step": self._episode_step,
            "done_reason": done_reason,
            "intervention": intervention,
        }
        if self._last_reward_debug:
            info["reward_debug"] = self._last_reward_debug
        if metrics is not None:
            latest = self.history.latest()
            local = merge_local_stats(self._get_ego(), list(latest.values()), self.config)
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
            )
            info.update(local_metrics.to_dict())
            info.update(
                {
                    "target_lane_id": local.target_lane_id,
                    "target_front_gap": local.target_front_gap,
                    "target_rear_gap": local.target_rear_gap,
                    "target_lane_gap": local.target_lane_gap,
                    "ramp_front_gap": local.ramp_front_gap,
                    "ramp_rear_gap": local.ramp_rear_gap,
                    "ramp_local_risk": local.ramp_local_risk,
                    "merge_zone_risk": local.merge_zone_risk,
                    "ego_on_auxiliary": local.ego_on_auxiliary,
                    "distance_to_taper": local.merge_distance,
                    "taper_miss": local.taper_miss,
                }
            )
            info["explicit_risk_features"] = explicit_risk_features(local_metrics)
        return info

    def get_risk_context(self) -> dict[str, Any]:
        latest = self.history.latest()
        ego = latest.get(self.ego_id)
        vehicles = list(latest.values())
        local = merge_local_stats(ego, vehicles, self.config)
        return {
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
        }

    def episode_report(self) -> dict[str, Any]:
        collisions = [metric.collision for metric in self._episode_metrics]
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
        raw_actions = Counter(str(item.get("raw_action", "")) for item in self._interventions)
        final_actions = Counter(str(item.get("final_action", "")) for item in self._interventions)
        emergency_fallback_count = sum(1 for item in self._interventions if item.get("emergency_fallback"))
        score_records = [
            {
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
            }
            for item in self._interventions
        ]
        return {
            "seed": self.seed_value,
            "curriculum_profile": self._curriculum_profile,
            "done_reason": self._last_done_reason,
            "taper_miss": self._last_done_reason == "taper_miss",
            "steps": self._episode_step,
            "completion_time": completion_time,
            "collision": any(collisions),
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
            "fallback_count": sum(1 for item in self._interventions if item.get("fallback")),
            "emergency_fallback_count": int(emergency_fallback_count),
            "emergency_fallback_rate": (
                float(emergency_fallback_count / len(self._interventions)) if self._interventions else 0.0
            ),
            "replacement_reason_counts": dict(reason_counts),
            "raw_action_histogram": dict(raw_actions),
            "final_action_histogram": dict(final_actions),
            "shield_score_records": score_records,
            "shield_guided_reward_summary": self._shield_guided_reward_summary(),
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

    def trajectory_window_samples(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return padded [sample, agent, time, state] windows from the episode."""

        hist = self.history_steps
        horizon = int(self.config.scenario.forecast_horizon_steps)
        max_agents = self.top_k + 1
        frames = self._trajectory_frames
        if len(frames) < hist + horizon:
            return (
                np.zeros((0, max_agents, hist, 5), dtype=np.float32),
                np.zeros((0, max_agents, horizon, 5), dtype=np.float32),
                np.zeros((0, max_agents), dtype=np.float32),
                np.zeros((0, max_agents), dtype=np.int64),
                np.zeros((0, max_agents), dtype=np.int64),
            )
        history_samples: list[np.ndarray] = []
        future_samples: list[np.ndarray] = []
        masks: list[np.ndarray] = []
        lane_indices: list[np.ndarray] = []
        edge_roles: list[np.ndarray] = []
        for end_idx in range(hist, len(frames) - horizon):
            latest = frames[end_idx - 1]
            if self.ego_id not in latest:
                continue
            ego = latest[self.ego_id]
            agent_ids = [self.ego_id]
            others = [state for vid, state in latest.items() if vid != self.ego_id]
            others.sort(key=lambda state: abs(state.x - ego.x) + abs(state.y - ego.y))
            agent_ids.extend(state.vehicle_id for state in others[: self.top_k])
            history = np.zeros((max_agents, hist, 5), dtype=np.float32)
            future = np.zeros((max_agents, horizon, 5), dtype=np.float32)
            mask = np.zeros((max_agents,), dtype=np.float32)
            sample_lane_indices = np.full((max_agents,), -1, dtype=np.int64)
            sample_edge_roles = np.zeros((max_agents,), dtype=np.int64)
            for agent_idx, vehicle_id in enumerate(agent_ids[:max_agents]):
                mask[agent_idx] = 1.0
                latest_state = latest.get(vehicle_id)
                if latest_state is not None:
                    sample_lane_indices[agent_idx] = int(latest_state.lane_index)
                    sample_edge_roles[agent_idx] = int(edge_role(self.config, latest_state.edge_id, latest_state.lane_index))
                last_state = None
                for step_idx, frame in enumerate(frames[end_idx - hist : end_idx]):
                    state = frame.get(vehicle_id) or last_state
                    if state is None:
                        continue
                    history[agent_idx, step_idx] = np.asarray(state.as_vector(), dtype=np.float32)
                    last_state = state
                for step_idx, frame in enumerate(frames[end_idx : end_idx + horizon]):
                    state = frame.get(vehicle_id) or last_state
                    if state is None:
                        continue
                    future[agent_idx, step_idx] = np.asarray(state.as_vector(), dtype=np.float32)
                    last_state = state
            history_samples.append(history)
            future_samples.append(future)
            masks.append(mask)
            lane_indices.append(sample_lane_indices)
            edge_roles.append(sample_edge_roles)
        if not history_samples:
            return (
                np.zeros((0, max_agents, hist, 5), dtype=np.float32),
                np.zeros((0, max_agents, horizon, 5), dtype=np.float32),
                np.zeros((0, max_agents), dtype=np.float32),
                np.zeros((0, max_agents), dtype=np.int64),
                np.zeros((0, max_agents), dtype=np.int64),
            )
        return (
            np.stack(history_samples, axis=0),
            np.stack(future_samples, axis=0),
            np.stack(masks, axis=0),
            np.stack(lane_indices, axis=0),
            np.stack(edge_roles, axis=0),
        )
