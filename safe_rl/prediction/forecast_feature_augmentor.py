from __future__ import annotations

from typing import Any

import numpy as np

from safe_rl.risk.merge_local import (
    constant_velocity_rollout,
    merge_local_stats,
    merge_target_lane,
    nearest_future_gap,
)
from safe_rl.sim.metrics import INF_TTC, bbox_gap, drac, merge_gap, relative_ttc
from safe_rl.sim.scenario_semantics import lane_center


class ForecastFeatureAugmentor:
    """Extract low-dimensional, action-independent forecast features."""

    FEATURE_NAMES = (
        "forecast_min_distance",
        "forecast_min_ttc",
        "forecast_max_drac",
        "forecast_collision_probability",
        "forecast_uncertainty",
        "forecast_merge_gap",
        "forecast_nearest_vehicle_future_dx",
        "forecast_nearest_vehicle_future_dy",
        "forecast_risk_top1",
        "forecast_risk_top2",
        "forecast_risk_top3",
    )

    def __init__(self, config: Any, predictor: Any | None = None):
        self.config = config
        self.predictor = predictor

    @classmethod
    def feature_dim(cls, config: Any | None = None) -> int:
        return len(cls.FEATURE_NAMES)

    def extract(self, context: dict[str, Any]) -> np.ndarray:
        ego = context.get("ego")
        vehicles = context.get("vehicles") or []
        if ego is None:
            return np.zeros((self.feature_dim(self.config),), dtype=np.float32)

        prediction = None
        if self.predictor is not None:
            try:
                prediction = self.predictor.predict(context)
            except Exception:
                if not bool(self.config.forecast_features.get("allow_heuristic_fallback", False)):
                    raise
                prediction = None

        source = str(self.config.forecast_features.get("source", "heuristic")).lower()
        if source == "constant_velocity":
            features = self._constant_velocity_features(ego, vehicles)
        elif prediction is not None:
            features = self._from_prediction(ego, vehicles, prediction)
        else:
            features = self._heuristic_features(ego, vehicles)

        if bool(self.config.forecast_features.normalize):
            features = self._normalize(features)
        return features.astype(np.float32)

    def _heuristic_features(self, ego, vehicles) -> np.ndarray:
        others = [vehicle for vehicle in vehicles if vehicle.vehicle_id != ego.vehicle_id]
        if not others:
            return np.asarray([50.0, INF_TTC, 0.0, 0.0, 0.0, 50.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        gaps = np.asarray([bbox_gap(ego, other) for other in others], dtype=np.float32)
        ttcs = np.asarray([relative_ttc(ego, other) for other in others], dtype=np.float32)
        dracs = np.asarray([drac(ego, other) for other in others], dtype=np.float32)
        nearest = others[int(np.argmin(gaps))]
        risk_scores = 1.0 / (1.0 + np.maximum(gaps, 0.0))
        top = np.sort(risk_scores)[::-1]
        top = np.pad(top[:3], (0, max(0, 3 - len(top))), constant_values=0.0)
        return np.asarray(
            [
                float(np.min(gaps)),
                float(np.min(ttcs)),
                float(np.max(dracs)),
                float(np.max(gaps < 0.25)),
                0.0,
                float(merge_local_stats(ego, vehicles, self.config).target_lane_gap),
                float(nearest.x - ego.x),
                float(nearest.y - ego.y),
                float(top[0]),
                float(top[1]),
                float(top[2]),
            ],
            dtype=np.float32,
        )

    def _constant_velocity_features(self, ego, vehicles) -> np.ndarray:
        others = [vehicle for vehicle in vehicles if vehicle.vehicle_id != ego.vehicle_id]
        if not others:
            return np.asarray([50.0, INF_TTC, 0.0, 0.0, 0.0, 50.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        horizon = int(self.config.forecast_features.get("horizon_steps", self.config.scenario.forecast_horizon_steps))
        dt = float(self.config.scenario.step_length)
        ego_rollout = constant_velocity_rollout(ego, horizon, dt)
        other_rollouts = [constant_velocity_rollout(other, horizon, dt) for other in others]
        min_distance, min_ttc, max_drac, nearest_dx, nearest_dy = nearest_future_gap(ego_rollout, other_rollouts, dt)
        top_risks = []
        for rollout in other_rollouts:
            agent_min = min(bbox_gap(ego_rollout[idx], state) for idx, state in enumerate(rollout[: len(ego_rollout)]))
            top_risks.append(1.0 / (1.0 + max(0.0, agent_min)))
        top = np.sort(np.asarray(top_risks, dtype=np.float32))[::-1]
        top = np.pad(top[:3], (0, max(0, 3 - len(top))), constant_values=0.0)
        target_lane_gap = 50.0
        for step_idx, ego_state in enumerate(ego_rollout):
            step_vehicles = [
                rollout[step_idx]
                for rollout in other_rollouts
                if step_idx < len(rollout) and int(rollout[step_idx].lane_index) == merge_target_lane(self.config)
            ]
            stats = merge_local_stats(ego_state, step_vehicles, self.config)
            target_lane_gap = min(target_lane_gap, stats.target_lane_gap)
        collision_probability = float(min_distance < float(self.config.risk_module.collision_distance_threshold))
        return np.asarray(
            [
                float(min_distance),
                float(min_ttc),
                float(max_drac),
                collision_probability,
                0.0,
                float(target_lane_gap),
                float(nearest_dx),
                float(nearest_dy),
                float(top[0]),
                float(top[1]),
                float(top[2]),
            ],
            dtype=np.float32,
        )

    def _from_prediction(self, ego, vehicles, prediction: dict[str, Any]) -> np.ndarray:
        trajectories = prediction.get("future_trajectories")
        if trajectories is None:
            return self._heuristic_features(ego, vehicles)
        if hasattr(trajectories, "detach"):
            trajectories = trajectories.detach().cpu().numpy()
        trajectories = np.asarray(trajectories)
        if trajectories.ndim == 5:
            trajectories = trajectories[0]
        if trajectories.ndim == 4:
            trajectories = trajectories[:, 0]
        if trajectories.size == 0:
            return self._heuristic_features(ego, vehicles)

        min_distance = 50.0
        min_ttc = INF_TTC
        max_drac = 0.0
        nearest_dx = 0.0
        nearest_dy = 0.0
        top_risks: list[float] = []
        dt = float(self.config.scenario.step_length)
        horizon = int(min(trajectories.shape[-2], self.config.forecast_features.get("horizon_steps", trajectories.shape[-2])))
        ego_rollout = constant_velocity_rollout(ego, horizon, dt)
        target_lane_gap = forecast_target_lane_gap_from_trajectories(ego_rollout, trajectories, self.config)
        for traj in trajectories:
            agent_min = 50.0
            previous_distance = INF_TTC
            for step_idx, step in enumerate(traj[:horizon]):
                ego_future = ego_rollout[min(step_idx, len(ego_rollout) - 1)]
                dx = float(step[0] - ego_future.x)
                dy = float(step[1] - ego_future.y)
                distance = max(0.0, float(np.hypot(dx, dy)) - 3.0)
                if distance < min_distance:
                    min_distance = distance
                    nearest_dx = dx
                    nearest_dy = dy
                agent_min = min(agent_min, distance)
                if previous_distance < INF_TTC:
                    closing = max(0.0, (previous_distance - distance) / max(dt, 1.0e-6))
                    if closing > 1.0e-6:
                        min_ttc = min(min_ttc, distance / closing)
                        max_drac = max(max_drac, (closing * closing) / (2.0 * max(distance, 1.0e-6)))
                previous_distance = distance
            top_risks.append(1.0 / (1.0 + agent_min))
        top = np.sort(np.asarray(top_risks, dtype=np.float32))[::-1]
        top = np.pad(top[:3], (0, max(0, 3 - len(top))), constant_values=0.0)
        confidence = prediction.get("mode_confidence")
        uncertainty = prediction.get("uncertainty", 0.0)
        if hasattr(uncertainty, "detach"):
            uncertainty = float(uncertainty.detach().cpu().reshape(-1)[0])
        collision_probability = float(np.max(top > 0.8))
        if confidence is not None and hasattr(confidence, "detach"):
            probs = confidence.detach().cpu().numpy()
            collision_probability *= float(np.max(probs))
        return np.asarray(
            [
                min_distance,
                min_ttc,
                max_drac,
                collision_probability,
                float(uncertainty),
                target_lane_gap,
                nearest_dx,
                nearest_dy,
                float(top[0]),
                float(top[1]),
                float(top[2]),
            ],
            dtype=np.float32,
        )

    def _normalize(self, features: np.ndarray) -> np.ndarray:
        normalized = features.copy()
        normalized[0] = min(normalized[0], 50.0) / 50.0
        normalized[1] = min(normalized[1], 10.0) / 10.0
        normalized[2] = min(normalized[2], 10.0) / 10.0
        normalized[5] = min(normalized[5], 50.0) / 50.0
        normalized[6] = np.clip(normalized[6] / 100.0, -1.0, 1.0)
        normalized[7] = np.clip(normalized[7] / 25.0, -1.0, 1.0)
        return normalized


def _ego_future_xy(ego_rollout: list[Any] | np.ndarray) -> np.ndarray:
    if isinstance(ego_rollout, np.ndarray):
        arr = np.asarray(ego_rollout, dtype=np.float32)
        return arr[:, :2]
    return np.asarray([[float(state.x), float(state.y)] for state in ego_rollout], dtype=np.float32)


def forecast_target_lane_gap_from_trajectories(
    ego_rollout: list[Any] | np.ndarray,
    trajectories: np.ndarray,
    config: Any,
    *,
    default_gap: float = 50.0,
) -> float:
    """Estimate the future target-lane gap used by forecast feature index 5."""

    ego_xy = _ego_future_xy(ego_rollout)
    trajectories = np.asarray(trajectories, dtype=np.float32)
    if trajectories.ndim == 4:
        trajectories = trajectories[:, 0]
    if trajectories.ndim != 3 or trajectories.shape[0] == 0 or ego_xy.size == 0:
        return float(default_gap)
    horizon = min(int(ego_xy.shape[0]), int(trajectories.shape[1]))
    if horizon <= 0:
        return float(default_gap)
    target_y = float(lane_center(config, merge_target_lane(config)))
    min_gap = float(default_gap)
    for step_idx in range(horizon):
        ego_x = float(ego_xy[step_idx, 0])
        front_gap = float(default_gap)
        rear_gap = float(default_gap)
        saw_target_lane_vehicle = False
        for traj in trajectories:
            x = float(traj[step_idx, 0])
            y = float(traj[step_idx, 1])
            if abs(y - target_y) > 2.0:
                continue
            saw_target_lane_vehicle = True
            dx = x - ego_x
            gap = min(float(default_gap), max(0.0, abs(dx) - 4.8))
            if dx >= 0.0:
                front_gap = min(front_gap, gap)
            else:
                rear_gap = min(rear_gap, gap)
        if saw_target_lane_vehicle:
            min_gap = min(min_gap, front_gap, rear_gap)
    return float(min_gap)
