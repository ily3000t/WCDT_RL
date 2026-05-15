from __future__ import annotations

from typing import Any

import numpy as np

from safe_rl.sim.action_space import CandidateAction
from safe_rl.sim.metrics import compute_step_metrics, explicit_risk_features
from safe_rl.sim.types import VehicleState


def candidate_progress_score(action: CandidateAction) -> float:
    return float(action.accel_cmd + 1) / 2.0


def rollout_ego(ego: VehicleState, action: CandidateAction, horizon_steps: int, dt: float) -> list[VehicleState]:
    states: list[VehicleState] = []
    speed = ego.speed
    x = ego.x
    y = ego.y
    lateral_velocity = action.lateral_cmd * 0.6
    acceleration = action.accel_cmd * 1.5
    for step in range(horizon_steps):
        speed = max(0.0, speed + acceleration * dt)
        x += speed * np.cos(ego.heading) * dt
        y += speed * np.sin(ego.heading) * dt + lateral_velocity * dt
        states.append(
            VehicleState(
                vehicle_id=ego.vehicle_id,
                x=float(x),
                y=float(y),
                heading=ego.heading,
                speed=float(speed),
                lane_index=ego.lane_index + action.lateral_cmd,
                lane_id=ego.lane_id,
                lane_pos=ego.lane_pos + speed * dt * (step + 1),
                edge_id=ego.edge_id,
                length=ego.length,
                width=ego.width,
                accel=acceleration,
            )
        )
    return states


def extract_candidate_features(action: CandidateAction, context: dict[str, Any]) -> np.ndarray:
    ego = context.get("ego")
    vehicles = context.get("vehicles") or []
    if ego is None:
        return np.ones((8,), dtype=np.float32)
    lane_count = int(context.get("lane_count", 1))
    lane_oob = ego.lane_index + action.lateral_cmd < 0 or ego.lane_index + action.lateral_cmd >= lane_count
    metrics = compute_step_metrics(
        ego,
        vehicles,
        collision=False,
        near_miss_threshold=float(context["config"].risk_module.near_miss_distance_threshold),
        ttc_threshold=float(context["config"].risk_module.ttc_threshold),
        drac_threshold=float(context["config"].risk_module.drac_threshold),
        lane_oob=lane_oob,
    )
    features = explicit_risk_features(metrics)
    features[5] = float(lane_oob)
    features[6] = float(action.accel_cmd < 0)
    return features.astype(np.float32)
