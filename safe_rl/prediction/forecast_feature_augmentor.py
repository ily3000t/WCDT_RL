from __future__ import annotations

from typing import Any

import numpy as np

from safe_rl.sim.metrics import INF_TTC, bbox_gap, drac, merge_gap, relative_ttc


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
                prediction = None

        if prediction is not None:
            features = self._from_prediction(ego, prediction)
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
                float(merge_gap(ego, others)),
                float(nearest.x - ego.x),
                float(nearest.y - ego.y),
                float(top[0]),
                float(top[1]),
                float(top[2]),
            ],
            dtype=np.float32,
        )

    def _from_prediction(self, ego, prediction: dict[str, Any]) -> np.ndarray:
        trajectories = prediction.get("future_trajectories")
        if trajectories is None:
            return self._heuristic_features(ego, [])
        if hasattr(trajectories, "detach"):
            trajectories = trajectories.detach().cpu().numpy()
        trajectories = np.asarray(trajectories)
        if trajectories.ndim == 5:
            trajectories = trajectories[0]
        if trajectories.ndim == 4:
            trajectories = trajectories[:, 0]
        if trajectories.size == 0:
            return self._heuristic_features(ego, [])

        min_distance = 50.0
        min_ttc = INF_TTC
        max_drac = 0.0
        nearest_dx = 0.0
        nearest_dy = 0.0
        top_risks: list[float] = []
        for traj in trajectories:
            agent_min = 50.0
            for step in traj:
                dx = float(step[0] - ego.x)
                dy = float(step[1] - ego.y)
                distance = max(0.0, float(np.hypot(dx, dy)) - 3.0)
                if distance < min_distance:
                    min_distance = distance
                    nearest_dx = dx
                    nearest_dy = dy
                agent_min = min(agent_min, distance)
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
                50.0,
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
