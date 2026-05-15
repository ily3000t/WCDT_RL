from __future__ import annotations

import math
import os
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from safe_rl.prediction.forecast_feature_augmentor import ForecastFeatureAugmentor
from safe_rl.shield.safety_shield import SafetyShield
from safe_rl.sim.action_space import ACTIONS, decode_action
from safe_rl.sim.gym_compat import gym, spaces
from safe_rl.sim.history_buffer import HistoryBuffer
from safe_rl.sim.metrics import INF_TTC, compute_step_metrics, explicit_risk_features
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
        record_trajectory_samples: bool = False,
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
        self.record_trajectory_samples = record_trajectory_samples

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
        self._traci = None
        self._conn_label = f"safe_rl_{uuid.uuid4().hex[:8]}"
        self._episode_step = 0
        self._last_ego_speed = 0.0
        self._last_ego_x = 0.0
        self._episode_metrics: list[StepMetrics] = []
        self._interventions: list[dict[str, Any]] = []
        self._trajectory_frames: list[dict[str, VehicleState]] = []

    def _import_traci(self):
        if self._traci is not None:
            return self._traci
        self._add_sumo_tools_path()
        try:
            import traci
        except ImportError as exc:  # pragma: no cover - depends on SUMO install
            raise ImportError(
                "Running SumoHighwayMergeEnv requires SUMO Python tools. "
                "Install/configure traci and sumolib, or activate the SAFE_RL environment."
            ) from exc
        self._traci = traci
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
        self._interventions.clear()
        self._trajectory_frames.clear()

        for _ in range(max(1, self.history_steps)):
            self._simulation_step()
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
        )
        self._episode_metrics.append(metrics)

        terminated, done_reason = self._done(metrics)
        truncated = self._episode_step >= self.episode_steps
        reward = self._reward(prev_x, ego, metrics, done_reason)
        obs = self._build_observation()
        info = self._info(metrics=metrics, done_reason=done_reason, intervention=intervention)
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
        traci.start(cmd, label=self._conn_label)
        self._traci = traci.getConnection(self._conn_label)

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
        ego_vec = np.asarray(
            [
                ego.speed / 35.0,
                ego.accel / 5.0,
                ego.lane_index / 3.0,
                ego.lane_pos / 500.0,
                ego.x / 500.0,
                ego.y / 100.0,
                float(ego.edge_id == "ramp_in"),
                float(ego.edge_id == "main_out"),
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
                    float(state.edge_id == "ramp_in"),
                    float(state.edge_id in ("main_in", "main_out")),
                ]
            )
        while len(neighbor_features) < self.top_k * 8:
            neighbor_features.append(0.0)
        merge_x = 220.0
        merge_features = np.asarray(
            [
                (merge_x - ego.x) / 300.0,
                (float(self.config.scenario.success_min_x) - ego.x) / 300.0,
                self._front_gap(ego, latest) / 100.0,
                self._rear_gap(ego, latest) / 100.0,
            ],
            dtype=np.float32,
        )
        return np.concatenate([ego_vec, np.asarray(neighbor_features, dtype=np.float32), merge_features], axis=0)

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
        if ego.edge_id == self.config.scenario.success_edge and ego.x >= float(self.config.scenario.success_min_x):
            return True, "merge_success"
        return False, ""

    def _reward(self, prev_x: float, ego: VehicleState | None, metrics: StepMetrics, done_reason: str) -> float:
        reward_cfg = self.config.rl.reward
        reward = 0.0
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
        return float(reward)

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
        if metrics is not None:
            info.update(metrics.to_dict())
            info["explicit_risk_features"] = explicit_risk_features(metrics)
        return info

    def get_risk_context(self) -> dict[str, Any]:
        latest = self.history.latest()
        ego = latest.get(self.ego_id)
        vehicles = list(latest.values())
        return {
            "ego": ego,
            "vehicles": vehicles,
            "history": self.history,
            "config": self.config,
            "lane_count": self._lane_count(ego.edge_id) if ego is not None else 1,
            "current_metrics": compute_step_metrics(ego, vehicles, collision=False) if ego is not None else None,
        }

    def episode_report(self) -> dict[str, Any]:
        collisions = [metric.collision for metric in self._episode_metrics]
        near_misses = [metric.near_miss for metric in self._episode_metrics]
        min_distances = [metric.min_distance for metric in self._episode_metrics]
        ttcs = [metric.min_ttc for metric in self._episode_metrics if metric.min_ttc < INF_TTC]
        dracs = [metric.max_drac for metric in self._episode_metrics]
        return {
            "seed": self.seed_value,
            "steps": self._episode_step,
            "collision": any(collisions),
            "near_miss": any(near_misses),
            "min_distance": float(min(min_distances)) if min_distances else INF_TTC,
            "ttc_p1": float(np.percentile(ttcs, 1)) if ttcs else INF_TTC,
            "drac_p99": float(np.percentile(dracs, 99)) if dracs else 0.0,
            "intervention_count": len(self._interventions),
            "fallback_count": sum(1 for item in self._interventions if item.get("fallback")),
        }

    def trajectory_window_samples(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
            )
        history_samples: list[np.ndarray] = []
        future_samples: list[np.ndarray] = []
        masks: list[np.ndarray] = []
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
            for agent_idx, vehicle_id in enumerate(agent_ids[:max_agents]):
                mask[agent_idx] = 1.0
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
        if not history_samples:
            return (
                np.zeros((0, max_agents, hist, 5), dtype=np.float32),
                np.zeros((0, max_agents, horizon, 5), dtype=np.float32),
                np.zeros((0, max_agents), dtype=np.float32),
            )
        return (
            np.stack(history_samples, axis=0),
            np.stack(future_samples, axis=0),
            np.stack(masks, axis=0),
        )
