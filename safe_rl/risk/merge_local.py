from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from safe_rl.sim.action_space import ACTIONS, CandidateAction
from safe_rl.sim.metrics import INF_TTC, bbox_gap, compute_step_metrics, explicit_risk_features
from safe_rl.sim.types import StepMetrics, VehicleState


@dataclass(frozen=True)
class MergeLocalStats:
    merge_distance: float
    in_merge_zone: bool
    ego_on_ramp: bool
    target_lane_id: int
    target_front_gap: float
    target_rear_gap: float
    target_front_rel_speed: float
    target_rear_rel_speed: float
    target_lane_gap: float
    ramp_front_gap: float
    ramp_rear_gap: float
    ramp_local_risk: bool
    merge_zone_risk: bool

    def to_dict(self) -> dict[str, float | bool | int]:
        return {
            "merge_distance": self.merge_distance,
            "in_merge_zone": self.in_merge_zone,
            "ego_on_ramp": self.ego_on_ramp,
            "target_lane_id": self.target_lane_id,
            "target_front_gap": self.target_front_gap,
            "target_rear_gap": self.target_rear_gap,
            "target_front_rel_speed": self.target_front_rel_speed,
            "target_rear_rel_speed": self.target_rear_rel_speed,
            "target_lane_gap": self.target_lane_gap,
            "ramp_front_gap": self.ramp_front_gap,
            "ramp_rear_gap": self.ramp_rear_gap,
            "ramp_local_risk": self.ramp_local_risk,
            "merge_zone_risk": self.merge_zone_risk,
        }


@dataclass(frozen=True)
class CandidateRiskSample:
    action: int
    features: np.ndarray
    overall_risk: float
    risk_types: np.ndarray
    local_stats: MergeLocalStats


def merge_x(config: Any) -> float:
    return float(config.scenario.get("merge_x", 220.0))


def merge_target_lane(config: Any) -> int:
    return int(config.scenario.get("merge_target_lane", 2))


def merge_conflict_gap(config: Any) -> float:
    return float(config.scenario.get("merge_conflict_gap", 8.0))


def merge_zone_distance(config: Any) -> float:
    return float(config.scenario.get("merge_zone_distance", 80.0))


def _half_length_sum(a: VehicleState, b: VehicleState) -> float:
    return 0.5 * max(a.length, 0.1) + 0.5 * max(b.length, 0.1)


def _longitudinal_gap(ego: VehicleState, other: VehicleState, value: float) -> float:
    return max(0.0, abs(value) - _half_length_sum(ego, other))


def target_lane_neighbors(
    ego: VehicleState | None,
    vehicles: list[VehicleState],
    config: Any,
) -> dict[str, float]:
    if ego is None:
        return {
            "front_gap": INF_TTC,
            "rear_gap": INF_TTC,
            "front_rel_speed": 0.0,
            "rear_rel_speed": 0.0,
        }
    lane = merge_target_lane(config)
    candidates = [
        vehicle
        for vehicle in vehicles
        if vehicle.vehicle_id != ego.vehicle_id
        and vehicle.edge_id in ("main_in", "main_out")
        and int(vehicle.lane_index) == lane
    ]
    front_gap = INF_TTC
    rear_gap = INF_TTC
    front_rel_speed = 0.0
    rear_rel_speed = 0.0
    for vehicle in candidates:
        dx = float(vehicle.x - ego.x)
        gap = _longitudinal_gap(ego, vehicle, dx)
        rel_speed = float(vehicle.speed - ego.speed)
        if dx >= 0.0 and gap < front_gap:
            front_gap = gap
            front_rel_speed = rel_speed
        if dx < 0.0 and gap < rear_gap:
            rear_gap = gap
            rear_rel_speed = rel_speed
    return {
        "front_gap": float(front_gap),
        "rear_gap": float(rear_gap),
        "front_rel_speed": float(front_rel_speed),
        "rear_rel_speed": float(rear_rel_speed),
    }


def ramp_neighbors(ego: VehicleState | None, vehicles: list[VehicleState]) -> dict[str, float]:
    if ego is None:
        return {"front_gap": INF_TTC, "rear_gap": INF_TTC}
    candidates = [
        vehicle
        for vehicle in vehicles
        if vehicle.vehicle_id != ego.vehicle_id
        and vehicle.edge_id == "ramp_in"
        and int(vehicle.lane_index) == int(ego.lane_index)
    ]
    front_gap = INF_TTC
    rear_gap = INF_TTC
    for vehicle in candidates:
        dpos = float(vehicle.lane_pos - ego.lane_pos)
        gap = _longitudinal_gap(ego, vehicle, dpos)
        if dpos >= 0.0:
            front_gap = min(front_gap, gap)
        else:
            rear_gap = min(rear_gap, gap)
    return {"front_gap": float(front_gap), "rear_gap": float(rear_gap)}


def merge_local_stats(
    ego: VehicleState | None,
    vehicles: list[VehicleState],
    config: Any,
) -> MergeLocalStats:
    if ego is None:
        lane = merge_target_lane(config)
        return MergeLocalStats(
            merge_distance=INF_TTC,
            in_merge_zone=False,
            ego_on_ramp=False,
            target_lane_id=lane,
            target_front_gap=INF_TTC,
            target_rear_gap=INF_TTC,
            target_front_rel_speed=0.0,
            target_rear_rel_speed=0.0,
            target_lane_gap=INF_TTC,
            ramp_front_gap=INF_TTC,
            ramp_rear_gap=INF_TTC,
            ramp_local_risk=False,
            merge_zone_risk=False,
        )
    target = target_lane_neighbors(ego, vehicles, config)
    ramp = ramp_neighbors(ego, vehicles)
    distance = float(merge_x(config) - ego.x)
    zone_distance = merge_zone_distance(config)
    in_zone = ego.edge_id in ("ramp_in", "main_out") and -10.0 <= distance <= zone_distance
    target_gap = min(float(target["front_gap"]), float(target["rear_gap"]))
    ramp_gap = min(float(ramp["front_gap"]), float(ramp["rear_gap"]))
    conflict_gap = merge_conflict_gap(config)
    return MergeLocalStats(
        merge_distance=distance,
        in_merge_zone=bool(in_zone),
        ego_on_ramp=bool(ego.edge_id == "ramp_in"),
        target_lane_id=merge_target_lane(config),
        target_front_gap=float(target["front_gap"]),
        target_rear_gap=float(target["rear_gap"]),
        target_front_rel_speed=float(target["front_rel_speed"]),
        target_rear_rel_speed=float(target["rear_rel_speed"]),
        target_lane_gap=float(target_gap),
        ramp_front_gap=float(ramp["front_gap"]),
        ramp_rear_gap=float(ramp["rear_gap"]),
        ramp_local_risk=bool(ego.edge_id == "ramp_in" and ramp_gap < conflict_gap),
        merge_zone_risk=bool(in_zone and target_gap < conflict_gap),
    )


def constant_velocity_rollout(vehicle: VehicleState, horizon_steps: int, dt: float) -> list[VehicleState]:
    states: list[VehicleState] = []
    x = float(vehicle.x)
    y = float(vehicle.y)
    lane_pos = float(vehicle.lane_pos)
    speed = max(0.0, float(vehicle.speed))
    vx = speed * math.cos(float(vehicle.heading))
    vy = speed * math.sin(float(vehicle.heading))
    for _step in range(max(1, horizon_steps)):
        x += vx * dt
        y += vy * dt
        lane_pos += speed * dt
        states.append(
            VehicleState(
                vehicle_id=vehicle.vehicle_id,
                x=float(x),
                y=float(y),
                heading=vehicle.heading,
                speed=float(speed),
                lane_index=vehicle.lane_index,
                lane_id=vehicle.lane_id,
                lane_pos=float(lane_pos),
                edge_id=vehicle.edge_id,
                length=vehicle.length,
                width=vehicle.width,
                accel=0.0,
            )
        )
    return states


def rollout_ego(ego: VehicleState, action: CandidateAction, horizon_steps: int, dt: float) -> list[VehicleState]:
    states: list[VehicleState] = []
    speed = float(ego.speed)
    x = float(ego.x)
    y = float(ego.y)
    lane_pos = float(ego.lane_pos)
    lateral_velocity = float(action.lateral_cmd) * 0.6
    acceleration = float(action.accel_cmd) * 1.5
    lane_index = int(ego.lane_index) + int(action.lateral_cmd)
    for _step in range(max(1, horizon_steps)):
        speed = max(0.0, speed + acceleration * dt)
        x += speed * math.cos(float(ego.heading)) * dt
        y += speed * math.sin(float(ego.heading)) * dt + lateral_velocity * dt
        lane_pos += speed * dt
        states.append(
            VehicleState(
                vehicle_id=ego.vehicle_id,
                x=float(x),
                y=float(y),
                heading=ego.heading,
                speed=float(speed),
                lane_index=lane_index,
                lane_id=ego.lane_id,
                lane_pos=float(lane_pos),
                edge_id=ego.edge_id,
                length=ego.length,
                width=ego.width,
                accel=acceleration,
            )
        )
    return states


def _rollout_dt(config: Any) -> float:
    return float(config.scenario.get("step_length", 0.1))


def evaluate_candidate_action_risk(action: CandidateAction, context: dict[str, Any]) -> CandidateRiskSample:
    ego = context.get("ego")
    config = context["config"]
    vehicles = list(context.get("vehicles") or [])
    if ego is None:
        stats = merge_local_stats(None, vehicles, config)
        return CandidateRiskSample(
            action=action.index,
            features=np.ones((int(config.risk_module.explicit_feature_dim),), dtype=np.float32),
            overall_risk=1.0,
            risk_types=np.asarray([0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            local_stats=stats,
        )

    lane_count = int(context.get("lane_count", 1))
    lane_oob = ego.lane_index + action.lateral_cmd < 0 or ego.lane_index + action.lateral_cmd >= lane_count
    horizon_steps = int(config.risk_module.get("collision_horizon_steps", 30))
    dt = _rollout_dt(config)
    ego_rollout = rollout_ego(ego, action, horizon_steps, dt)
    other_rollouts = [
        constant_velocity_rollout(vehicle, horizon_steps, dt)
        for vehicle in vehicles
        if vehicle.vehicle_id != ego.vehicle_id
    ]

    min_distance = INF_TTC
    min_ttc = INF_TTC
    max_drac = 0.0
    collision = False
    near_miss = False
    low_ttc = False
    high_drac = False
    merge_conflict = False
    best_stats = merge_local_stats(ego, vehicles, config)
    best_gap = best_stats.target_lane_gap

    for step_idx, ego_state in enumerate(ego_rollout):
        step_vehicles = [rollout[step_idx] for rollout in other_rollouts]
        metrics = compute_step_metrics(
            ego_state,
            [ego_state, *step_vehicles],
            collision=False,
            near_miss_threshold=float(config.risk_module.near_miss_distance_threshold),
            ttc_threshold=float(config.risk_module.ttc_threshold),
            drac_threshold=float(config.risk_module.drac_threshold),
            lane_oob=lane_oob,
        )
        stats = merge_local_stats(ego_state, step_vehicles, config)
        if stats.target_lane_gap < best_gap:
            best_gap = stats.target_lane_gap
            best_stats = stats
        min_distance = min(min_distance, metrics.min_distance)
        min_ttc = min(min_ttc, metrics.min_ttc)
        max_drac = max(max_drac, metrics.max_drac)
        collision = collision or metrics.collision
        near_miss = near_miss or metrics.near_miss
        low_ttc = low_ttc or metrics.low_ttc
        high_drac = high_drac or metrics.high_drac
        merge_conflict = merge_conflict or stats.merge_zone_risk

    worst = StepMetrics(
        min_distance=float(min_distance),
        min_ttc=float(min_ttc),
        max_drac=float(max_drac),
        collision=bool(collision),
        near_miss=bool(near_miss),
        low_ttc=bool(low_ttc),
        high_drac=bool(high_drac),
        merge_gap=float(best_gap),
        lane_oob=bool(lane_oob),
        hard_brake=bool(action.accel_cmd < 0),
    )
    features = explicit_risk_features(worst)
    if features.shape[0] != int(config.risk_module.explicit_feature_dim):
        padded = np.zeros((int(config.risk_module.explicit_feature_dim),), dtype=np.float32)
        limit = min(padded.shape[0], features.shape[0])
        padded[:limit] = features[:limit]
        features = padded
    risk_types = np.asarray(
        [float(collision), float(near_miss), float(low_ttc), float(high_drac), float(merge_conflict)],
        dtype=np.float32,
    )
    return CandidateRiskSample(
        action=action.index,
        features=features.astype(np.float32),
        overall_risk=float(max(float(np.max(risk_types)), float(lane_oob))),
        risk_types=risk_types,
        local_stats=best_stats,
    )


def candidate_action_risk_samples(context: dict[str, Any]) -> list[CandidateRiskSample]:
    return [evaluate_candidate_action_risk(action, context) for action in ACTIONS]


def nearest_future_gap(
    ego_rollout: list[VehicleState],
    other_rollouts: list[list[VehicleState]],
    dt: float = 0.1,
) -> tuple[float, float, float, float, float]:
    min_gap = INF_TTC
    min_ttc = INF_TTC
    max_drac = 0.0
    nearest_dx = 0.0
    nearest_dy = 0.0
    for step_idx, ego_state in enumerate(ego_rollout):
        for rollout in other_rollouts:
            if step_idx >= len(rollout):
                continue
            other = rollout[step_idx]
            gap = bbox_gap(ego_state, other)
            if gap < min_gap:
                min_gap = gap
                nearest_dx = float(other.x - ego_state.x)
                nearest_dy = float(other.y - ego_state.y)
            prev_gap = bbox_gap(ego_rollout[step_idx - 1], rollout[step_idx - 1]) if step_idx > 0 else INF_TTC
            closing = max(0.0, (prev_gap - gap) / max(1.0e-6, dt)) if prev_gap < INF_TTC else 0.0
            if closing > 1.0e-6:
                min_ttc = min(min_ttc, gap / closing)
                max_drac = max(max_drac, (closing * closing) / (2.0 * max(gap, 1.0e-6)))
    return float(min_gap), float(min_ttc), float(max_drac), float(nearest_dx), float(nearest_dy)
