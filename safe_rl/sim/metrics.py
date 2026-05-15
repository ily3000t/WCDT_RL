from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from safe_rl.sim.types import StepMetrics, VehicleState


INF_TTC = 1.0e6


def bbox_radius(state: VehicleState) -> float:
    return 0.5 * math.hypot(max(state.length, 0.1), max(state.width, 0.1))


def center_distance(a: VehicleState, b: VehicleState) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def bbox_gap(a: VehicleState, b: VehicleState) -> float:
    return max(0.0, center_distance(a, b) - bbox_radius(a) - bbox_radius(b))


def relative_ttc(ego: VehicleState, other: VehicleState) -> float:
    dx = other.x - ego.x
    dy = other.y - ego.y
    distance = math.hypot(dx, dy)
    if distance <= 1.0e-6:
        return 0.0
    ego_vx = ego.speed * math.cos(ego.heading)
    ego_vy = ego.speed * math.sin(ego.heading)
    other_vx = other.speed * math.cos(other.heading)
    other_vy = other.speed * math.sin(other.heading)
    rel_vx = ego_vx - other_vx
    rel_vy = ego_vy - other_vy
    closing_speed = (rel_vx * dx + rel_vy * dy) / distance
    if closing_speed <= 1.0e-6:
        return INF_TTC
    return max(0.0, bbox_gap(ego, other) / closing_speed)


def drac(ego: VehicleState, other: VehicleState) -> float:
    gap = bbox_gap(ego, other)
    if gap <= 1.0e-6:
        return INF_TTC
    ttc = relative_ttc(ego, other)
    if ttc >= INF_TTC:
        return 0.0
    closing_speed = gap / max(ttc, 1.0e-6)
    return (closing_speed * closing_speed) / (2.0 * gap)


def merge_gap(ego: VehicleState, vehicles: Iterable[VehicleState]) -> float:
    if ego.edge_id != "ramp_in" and ego.edge_id != "main_out":
        return INF_TTC
    same_target = [
        vehicle
        for vehicle in vehicles
        if vehicle.vehicle_id != ego.vehicle_id and vehicle.edge_id in ("main_in", "main_out")
    ]
    if not same_target:
        return INF_TTC
    gaps = [abs(vehicle.x - ego.x) for vehicle in same_target]
    return float(min(gaps))


def compute_step_metrics(
    ego: VehicleState | None,
    vehicles: Iterable[VehicleState],
    collision: bool,
    near_miss_threshold: float = 0.75,
    ttc_threshold: float = 1.5,
    drac_threshold: float = 3.35,
    hard_brake_threshold: float = -3.0,
    lane_oob: bool = False,
) -> StepMetrics:
    if ego is None:
        return StepMetrics(0.0, 0.0, INF_TTC, True, True, True, True, 0.0, lane_oob, False)

    others = [vehicle for vehicle in vehicles if vehicle.vehicle_id != ego.vehicle_id]
    if not others:
        min_gap = INF_TTC
        min_ttc = INF_TTC
        max_drac = 0.0
    else:
        min_gap = min(bbox_gap(ego, other) for other in others)
        min_ttc = min(relative_ttc(ego, other) for other in others)
        max_drac = max(drac(ego, other) for other in others)

    return StepMetrics(
        min_distance=float(min_gap),
        min_ttc=float(min_ttc),
        max_drac=float(max_drac),
        collision=bool(collision),
        near_miss=bool(min_gap < near_miss_threshold),
        low_ttc=bool(min_ttc < ttc_threshold),
        high_drac=bool(max_drac > drac_threshold),
        merge_gap=float(merge_gap(ego, others)),
        lane_oob=bool(lane_oob),
        hard_brake=bool(ego.accel < hard_brake_threshold),
    )


def explicit_risk_features(metrics: StepMetrics) -> np.ndarray:
    min_distance = min(metrics.min_distance, 50.0) / 50.0
    min_ttc = min(metrics.min_ttc, 10.0) / 10.0
    max_drac = min(metrics.max_drac, 10.0) / 10.0
    merge = min(metrics.merge_gap, 50.0) / 50.0
    return np.asarray(
        [
            1.0 - min_distance,
            1.0 - min_ttc,
            max_drac,
            float(metrics.collision),
            1.0 - merge,
            float(metrics.lane_oob),
            float(metrics.hard_brake),
            float(metrics.near_miss or metrics.low_ttc or metrics.high_drac),
        ],
        dtype=np.float32,
    )
