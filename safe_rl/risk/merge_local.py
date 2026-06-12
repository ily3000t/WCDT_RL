from __future__ import annotations

import math
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import numpy as np

from safe_rl.sim.action_space import ACTIONS, CandidateAction
from safe_rl.sim.metrics import INF_TTC, bbox_gap, compute_step_metrics, drac, explicit_risk_features, relative_ttc
from safe_rl.sim.scenario_semantics import (
    advance_route_state,
    auxiliary_lane_index,
    distance_to_taper,
    is_auxiliary_edge,
    is_mainline_edge,
    is_ramp_edge,
    is_taper_miss,
    is_target_lane_edge,
    is_target_lane,
    merge_target_lane,
    merge_zone_edges,
    target_lane_edges,
    target_lane_index,
    target_lane_mapping,
    taper_edge,
)
from safe_rl.sim.types import StepMetrics, VehicleState


@dataclass(frozen=True)
class MergeLocalStats:
    merge_distance: float
    in_merge_zone: bool
    ego_on_ramp: bool
    ego_on_auxiliary: bool
    target_lane_id: int
    target_front_gap: float
    target_rear_gap: float
    target_front_vehicle_id: str
    target_rear_vehicle_id: str
    target_front_rel_speed: float
    target_rear_rel_speed: float
    target_lane_gap: float
    ramp_front_gap: float
    ramp_rear_gap: float
    ramp_local_risk: bool
    merge_zone_risk: bool
    taper_miss: bool

    def to_dict(self) -> dict[str, float | bool | int]:
        return {
            "merge_distance": self.merge_distance,
            "in_merge_zone": self.in_merge_zone,
            "ego_on_ramp": self.ego_on_ramp,
            "ego_on_auxiliary": self.ego_on_auxiliary,
            "target_lane_id": self.target_lane_id,
            "target_front_gap": self.target_front_gap,
            "target_rear_gap": self.target_rear_gap,
            "target_front_vehicle_id": self.target_front_vehicle_id,
            "target_rear_vehicle_id": self.target_rear_vehicle_id,
            "target_front_rel_speed": self.target_front_rel_speed,
            "target_rear_rel_speed": self.target_rear_rel_speed,
            "target_lane_gap": self.target_lane_gap,
            "ramp_front_gap": self.ramp_front_gap,
            "ramp_rear_gap": self.ramp_rear_gap,
            "ramp_local_risk": self.ramp_local_risk,
            "merge_zone_risk": self.merge_zone_risk,
            "taper_miss": self.taper_miss,
        }


@dataclass(frozen=True)
class CandidateRiskSample:
    action: int
    features: np.ndarray
    overall_risk: float
    risk_types: np.ndarray
    local_stats: MergeLocalStats
    lane_oob: float
    candidate_legal: bool
    traffic_risk: float
    continuous_risk_target: float
    boundary_sample: bool
    distance_to_taper: float
    ego_on_auxiliary: bool


@dataclass
class CandidateRolloutContext:
    ego: VehicleState | None
    vehicles: list[VehicleState]
    config: Any
    horizon_steps: int
    dt: float
    current_stats: MergeLocalStats
    other_rollouts: list[list[VehicleState]]
    legality: dict[int, bool]
    ego_rollouts: dict[int, tuple[list[VehicleState], bool]]
    samples: dict[int, CandidateRiskSample]


def merge_x(config: Any) -> float:
    return float(config.scenario.get("merge_x", 220.0))


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
) -> dict[str, float | str]:
    if ego is None:
        return {
            "front_gap": INF_TTC,
            "rear_gap": INF_TTC,
            "front_vehicle_id": "",
            "rear_vehicle_id": "",
            "front_rel_speed": 0.0,
            "rear_rel_speed": 0.0,
        }
    candidates = [
        vehicle
        for vehicle in vehicles
        if vehicle.vehicle_id != ego.vehicle_id
        and is_target_lane(config, vehicle.edge_id, vehicle.lane_index)
    ]
    front_gap = INF_TTC
    rear_gap = INF_TTC
    front_rel_speed = 0.0
    rear_rel_speed = 0.0
    front_vehicle_id = ""
    rear_vehicle_id = ""
    for vehicle in candidates:
        dx = float(vehicle.x - ego.x)
        gap = _longitudinal_gap(ego, vehicle, dx)
        rel_speed = float(vehicle.speed - ego.speed)
        if dx >= 0.0 and gap < front_gap:
            front_gap = gap
            front_rel_speed = rel_speed
            front_vehicle_id = str(vehicle.vehicle_id)
        if dx < 0.0 and gap < rear_gap:
            rear_gap = gap
            rear_rel_speed = rel_speed
            rear_vehicle_id = str(vehicle.vehicle_id)
    return {
        "front_gap": float(front_gap),
        "rear_gap": float(rear_gap),
        "front_vehicle_id": front_vehicle_id,
        "rear_vehicle_id": rear_vehicle_id,
        "front_rel_speed": float(front_rel_speed),
        "rear_rel_speed": float(rear_rel_speed),
    }


def ramp_neighbors(ego: VehicleState | None, vehicles: list[VehicleState], config: Any) -> dict[str, float]:
    if ego is None:
        return {"front_gap": INF_TTC, "rear_gap": INF_TTC}
    candidates = [
        vehicle
        for vehicle in vehicles
        if vehicle.vehicle_id != ego.vehicle_id
        and vehicle.edge_id == ego.edge_id
        and (is_ramp_edge(config, vehicle.edge_id) or is_auxiliary_edge(config, vehicle.edge_id))
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
        lane = target_lane_index(config, taper_edge(config))
        return MergeLocalStats(
            merge_distance=INF_TTC,
            in_merge_zone=False,
            ego_on_ramp=False,
            ego_on_auxiliary=False,
            target_lane_id=lane,
            target_front_gap=INF_TTC,
            target_rear_gap=INF_TTC,
            target_front_vehicle_id="",
            target_rear_vehicle_id="",
            target_front_rel_speed=0.0,
            target_rear_rel_speed=0.0,
            target_lane_gap=INF_TTC,
            ramp_front_gap=INF_TTC,
            ramp_rear_gap=INF_TTC,
            ramp_local_risk=False,
            merge_zone_risk=False,
            taper_miss=False,
        )
    target = target_lane_neighbors(ego, vehicles, config)
    ramp = ramp_neighbors(ego, vehicles, config)
    distance = float(distance_to_taper(config, ego))
    zone_distance = merge_zone_distance(config)
    in_zone = ego.edge_id in set(merge_zone_edges(config)) and -10.0 <= distance <= zone_distance
    target_gap = min(float(target["front_gap"]), float(target["rear_gap"]))
    ramp_gap = min(float(ramp["front_gap"]), float(ramp["rear_gap"]))
    conflict_gap = merge_conflict_gap(config)
    return MergeLocalStats(
        merge_distance=distance,
        in_merge_zone=bool(in_zone),
        ego_on_ramp=bool(is_ramp_edge(config, ego.edge_id)),
        ego_on_auxiliary=bool(is_auxiliary_edge(config, ego.edge_id)),
        target_lane_id=target_lane_index(config, taper_edge(config)),
        target_front_gap=float(target["front_gap"]),
        target_rear_gap=float(target["rear_gap"]),
        target_front_vehicle_id=str(target["front_vehicle_id"]),
        target_rear_vehicle_id=str(target["rear_vehicle_id"]),
        target_front_rel_speed=float(target["front_rel_speed"]),
        target_rear_rel_speed=float(target["rear_rel_speed"]),
        target_lane_gap=float(target_gap),
        ramp_front_gap=float(ramp["front_gap"]),
        ramp_rear_gap=float(ramp["rear_gap"]),
        ramp_local_risk=bool(
            (is_ramp_edge(config, ego.edge_id) or is_auxiliary_edge(config, ego.edge_id))
            and ramp_gap < conflict_gap
        ),
        merge_zone_risk=bool(in_zone and target_gap < conflict_gap),
        taper_miss=bool(is_taper_miss(config, ego)),
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


def route_aware_constant_velocity_rollout(
    vehicle: VehicleState,
    horizon_steps: int,
    dt: float,
    config: Any,
) -> tuple[list[VehicleState], bool]:
    states: list[VehicleState] = []
    current = vehicle
    taper_miss = False
    for _step in range(max(1, horizon_steps)):
        current, step_taper_miss = advance_route_state(config, current, max(0.0, current.speed) * dt)
        taper_miss = taper_miss or step_taper_miss
        states.append(current)
        if step_taper_miss:
            break
    while len(states) < max(1, horizon_steps):
        states.append(states[-1])
    return states, taper_miss


def rollout_ego(
    ego: VehicleState,
    action: CandidateAction,
    horizon_steps: int,
    dt: float,
    config: Any | None = None,
) -> tuple[list[VehicleState], bool]:
    states: list[VehicleState] = []
    speed = float(ego.speed)
    x = float(ego.x)
    y = float(ego.y)
    lane_pos = float(ego.lane_pos)
    lateral_velocity = float(action.lateral_cmd) * 0.6
    acceleration = float(action.accel_cmd) * 1.5
    source_lane = int(ego.lane_index)
    target_lane = source_lane + int(action.lateral_cmd)
    current = VehicleState(**{**ego.to_dict(), "lane_index": source_lane})
    target_current = VehicleState(
        **{
            **ego.to_dict(),
            "lane_index": target_lane,
            "lane_id": f"{ego.edge_id}_{target_lane}",
        }
    )
    lane_change_duration = max(
        float(config.scenario.get("lane_change_duration", 1.0)) if config is not None else 1.0,
        dt,
    )
    taper_miss = False
    for step_idx in range(max(1, horizon_steps)):
        speed = max(0.0, speed + acceleration * dt)
        if config is None:
            x += speed * math.cos(float(ego.heading)) * dt
            y += speed * math.sin(float(ego.heading)) * dt + lateral_velocity * dt
            lane_pos += speed * dt
            next_state = VehicleState(
                vehicle_id=ego.vehicle_id,
                x=float(x),
                y=float(y),
                heading=ego.heading,
                speed=float(speed),
                lane_index=target_lane,
                lane_id=ego.lane_id,
                lane_pos=float(lane_pos),
                edge_id=ego.edge_id,
                length=ego.length,
                width=ego.width,
                accel=acceleration,
            )
        else:
            distance = max(0.0, speed) * dt
            source_next, source_taper_miss = advance_route_state(
                config, current, distance, lane_index=current.lane_index
            )
            if action.lateral_cmd == 0:
                next_state = source_next
                target_next = source_next
                progress = 1.0
            else:
                target_next, _target_taper_miss = advance_route_state(
                    config, target_current, distance, lane_index=target_current.lane_index
                )
                raw_progress = min(1.0, float((step_idx + 1) * dt / lane_change_duration))
                progress = raw_progress * raw_progress * (3.0 - 2.0 * raw_progress)
                if source_taper_miss and raw_progress < 1.0:
                    taper_miss = True
                    next_state = source_next
                elif raw_progress >= 1.0:
                    next_state = target_next
                else:
                    next_state = VehicleState(
                        **{
                            **source_next.to_dict(),
                            "x": float(source_next.x + progress * (target_next.x - source_next.x)),
                            "y": float(source_next.y + progress * (target_next.y - source_next.y)),
                            "lane_index": source_lane,
                            "lane_id": f"{source_next.edge_id}_{source_lane}",
                        }
                    )
                    dx = float(next_state.x - current.x)
                    dy = float(next_state.y - current.y)
                    if math.hypot(dx, dy) > 1.0e-9:
                        next_state.heading = float(math.atan2(dy, dx))
            next_state.speed = float(speed)
            next_state.accel = float(acceleration)
            taper_miss = taper_miss or (source_taper_miss and progress < 1.0)
            target_current = target_next
        states.append(next_state)
        current = next_state
        lane_pos = float(next_state.lane_pos)
        if taper_miss:
            break
    while len(states) < max(1, horizon_steps):
        states.append(states[-1])
    return states, taper_miss


def _rollout_dt(config: Any) -> float:
    return float(config.scenario.get("step_length", 0.1))


def is_candidate_legal(
    action: CandidateAction,
    context: dict[str, Any],
    *,
    missing_ego_is_legal: bool = True,
) -> bool:
    ego = context.get("ego")
    if ego is None:
        return bool(missing_ego_is_legal)
    lane_count = int(context.get("lane_count", 1))
    target_lane = int(ego.lane_index) + int(action.lateral_cmd)
    return 0 <= target_lane < lane_count


def candidate_legality_counts(context: dict[str, Any]) -> dict[str, int]:
    legal = sum(1 for action in ACTIONS if is_candidate_legal(action, context))
    total = len(ACTIONS)
    return {"legal": int(legal), "illegal": int(total - legal)}


def candidate_sample_weight(sample: CandidateRiskSample) -> float:
    return 1.0 if sample.candidate_legal else 0.0


def continuous_risk_target(
    metrics: StepMetrics,
    stats: MergeLocalStats,
    *,
    taper_miss: bool = False,
) -> float:
    distance_severity = float(np.clip((5.0 - metrics.min_distance) / 5.0, 0.0, 1.0))
    ttc_severity = float(np.clip((2.0 - metrics.min_ttc) / 2.0, 0.0, 1.0))
    drac_severity = float(np.clip(metrics.max_drac / 20.0, 0.0, 1.0))
    gap_severity = float(np.clip((12.0 - stats.target_lane_gap) / 12.0, 0.0, 1.0))
    taper_severity = (
        float(np.clip((40.0 - stats.merge_distance) / 40.0, 0.0, 1.0))
        if stats.ego_on_auxiliary
        else 0.0
    )
    score = float(
        np.clip(
            0.30 * distance_severity
            + 0.25 * ttc_severity
            + 0.15 * drac_severity
            + 0.20 * gap_severity
            + 0.10 * taper_severity,
            0.0,
            1.0,
        )
    )
    floors = (
        (metrics.collision, 1.00),
        (metrics.near_miss, 0.90),
        (metrics.low_ttc, 0.70),
        (metrics.high_drac, 0.60),
        (stats.merge_zone_risk, 0.55),
        (taper_miss or stats.taper_miss, 0.95),
    )
    for active, floor in floors:
        if active:
            score = max(score, float(floor))
    return float(np.clip(score, 0.0, 1.0))


def prepare_candidate_rollout_context(context: dict[str, Any]) -> CandidateRolloutContext:
    cached = context.get("_candidate_rollout_context")
    if isinstance(cached, CandidateRolloutContext):
        return cached
    ego = context.get("ego")
    config = context["config"]
    vehicles = list(context.get("vehicles") or [])
    horizon_steps = int(config.risk_module.get("collision_horizon_steps", 30))
    dt = _rollout_dt(config)
    tracker = context.get("performance_tracker")
    timer = tracker.measure("candidate_rollout_time") if tracker is not None else nullcontext()
    with timer:
        other_rollouts = (
            [
                route_aware_constant_velocity_rollout(vehicle, horizon_steps, dt, config)[0]
                for vehicle in vehicles
                if ego is None or vehicle.vehicle_id != ego.vehicle_id
            ]
            if ego is not None
            else []
        )
    prepared = CandidateRolloutContext(
        ego=ego,
        vehicles=vehicles,
        config=config,
        horizon_steps=horizon_steps,
        dt=dt,
        current_stats=merge_local_stats(ego, vehicles, config),
        other_rollouts=other_rollouts,
        legality={action.index: is_candidate_legal(action, context, missing_ego_is_legal=False) for action in ACTIONS},
        ego_rollouts={},
        samples={},
    )
    context["_candidate_rollout_context"] = prepared
    return prepared


def evaluate_candidate_action_risk(action: CandidateAction, context: dict[str, Any]) -> CandidateRiskSample:
    prepared = prepare_candidate_rollout_context(context)
    cached = prepared.samples.get(int(action.index))
    if cached is not None:
        return cached
    ego = prepared.ego
    config = prepared.config
    vehicles = prepared.vehicles
    if ego is None:
        sample = CandidateRiskSample(
            action=action.index,
            features=np.ones((int(config.risk_module.explicit_feature_dim),), dtype=np.float32),
            overall_risk=1.0,
            risk_types=np.asarray([0.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32),
            local_stats=prepared.current_stats,
            lane_oob=0.0,
            candidate_legal=False,
            traffic_risk=1.0,
            continuous_risk_target=1.0,
            boundary_sample=False,
            distance_to_taper=prepared.current_stats.merge_distance,
            ego_on_auxiliary=prepared.current_stats.ego_on_auxiliary,
        )
        prepared.samples[int(action.index)] = sample
        return sample

    candidate_legal = prepared.legality[int(action.index)]
    lane_oob = not candidate_legal
    tracker = context.get("performance_tracker")
    timer = tracker.measure("candidate_rollout_time") if tracker is not None else nullcontext()
    with timer:
        if int(action.index) not in prepared.ego_rollouts:
            prepared.ego_rollouts[int(action.index)] = rollout_ego(
                ego, action, prepared.horizon_steps, prepared.dt, config
            )
        ego_rollout, ego_taper_miss = prepared.ego_rollouts[int(action.index)]
    other_rollouts = prepared.other_rollouts

    min_distance = INF_TTC
    min_ttc = INF_TTC
    max_drac = 0.0
    collision = False
    near_miss = False
    low_ttc = False
    high_drac = False
    merge_conflict = False
    taper_miss = bool(ego_taper_miss)
    best_stats = prepared.current_stats
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
            merge_ego_edges=merge_zone_edges(config),
            merge_target_edges=target_lane_edges(config),
            merge_target_lane=merge_target_lane(config),
            merge_target_lanes=target_lane_mapping(config),
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
        taper_miss = taper_miss or stats.taper_miss

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
        [
            float(collision),
            float(near_miss),
            float(low_ttc),
            float(high_drac),
            float(merge_conflict),
            float(taper_miss),
        ],
        dtype=np.float32,
    )
    traffic_risk = float(np.max(risk_types))
    continuous_target = continuous_risk_target(worst, best_stats, taper_miss=taper_miss)
    sample = CandidateRiskSample(
        action=action.index,
        features=features.astype(np.float32),
        overall_risk=traffic_risk,
        risk_types=risk_types,
        local_stats=best_stats,
        lane_oob=float(lane_oob),
        candidate_legal=bool(candidate_legal),
        traffic_risk=traffic_risk,
        continuous_risk_target=continuous_target,
        boundary_sample=bool(0.20 <= continuous_target < 0.80),
        distance_to_taper=float(best_stats.merge_distance),
        ego_on_auxiliary=bool(best_stats.ego_on_auxiliary),
    )
    prepared.samples[int(action.index)] = sample
    return sample


def evaluate_candidate_actions(
    actions: list[CandidateAction] | tuple[CandidateAction, ...],
    context: dict[str, Any],
) -> list[CandidateRiskSample]:
    prepare_candidate_rollout_context(context)
    return [evaluate_candidate_action_risk(action, context) for action in actions]


def candidate_action_risk_samples(context: dict[str, Any]) -> list[CandidateRiskSample]:
    return evaluate_candidate_actions(ACTIONS, context)


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
            min_ttc = min(min_ttc, relative_ttc(ego_state, other))
            max_drac = max(max_drac, drac(ego_state, other))
    return float(min_gap), float(min_ttc), float(max_drac), float(nearest_dx), float(nearest_dy)
