"""Deterministic candidate-union conflict tubes for Selector-v3."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
from typing import Any

import numpy as np

from safe_rl.risk.merge_local import (
    route_aware_constant_velocity_rollout,
    rollout_ego,
)
from safe_rl.sim.action_space import ACTIONS, CandidateAction
from safe_rl.sim.metrics import (
    INF_TTC,
    batch_pairwise_obb_gap_overlap,
    bbox_gap,
    geometric_overlap,
)
from safe_rl.sim.scenario_semantics import (
    advance_route_state,
    lane_indices,
    with_cached_scenario_semantics,
)
from safe_rl.sim.types import VehicleState, copy_vehicle_state


CANDIDATE_CONFLICT_ORACLE_VERSION = "candidate_union_swept_obb_v1"


@dataclass(frozen=True)
class ActorConflictEvidence:
    vehicle_id: str
    candidate_conflict_eligible: bool
    conflict_candidate_ids: tuple[int, ...]
    conflict_hypothesis_ids: tuple[str, ...]
    conflict_surface_ids: tuple[str, ...]
    earliest_conflict_time_s: float
    earliest_overlap_time_s: float
    minimum_swept_obb_gap: float
    nearest_candidate_conflict: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _settings(
    config: Any,
    *,
    selector_scope: str = "prediction",
) -> dict[str, float]:
    if selector_scope == "prediction":
        configured = config.prediction.get("actor_relevance", {})
    elif selector_scope == "accvp":
        configured = config.accvp.get("actor_relevance")
        if configured is None:
            configured = config.prediction.get("actor_relevance", {})
    else:
        raise ValueError(
            "selector_scope must be 'prediction' or 'accvp': "
            f"{selector_scope!r}"
        )
    horizon = float(
        configured.get(
            "candidate_conflict_horizon_s",
            config.accvp.get("response_horizon_s", 3.0),
        )
    )
    return {
        "horizon_s": max(float(config.scenario.step_length), horizon),
        "surface_gap_m": float(
            configured.get("candidate_conflict_surface_gap", 30.0)
        ),
        "actor_longitudinal_accel_bound": max(
            0.0,
            float(
                configured.get(
                    "actor_longitudinal_accel_bound", 2.0
                )
            ),
        ),
        "unknown_topology_lateral_reach_m": max(
            0.0,
            float(
                configured.get(
                    "unknown_topology_lateral_reach_m", 3.5
                )
            ),
        ),
    }


def _legal_actions(
    config: Any,
    ego: VehicleState,
    vehicles: list[VehicleState],
) -> list[CandidateAction]:
    available = set(lane_indices(config, ego.edge_id))
    if not available:
        available = {
            int(vehicle.lane_index)
            for vehicle in vehicles
            if str(vehicle.edge_id) == str(ego.edge_id)
        }
    if not available:
        available = {int(ego.lane_index)}
    return [
        action
        for action in ACTIONS
        if int(ego.lane_index) + int(action.lateral_cmd) in available
    ]


def _commitment_rollout(
    config: Any,
    ego: VehicleState,
    action: CandidateAction,
    horizon_steps: int,
    dt: float,
) -> list[VehicleState]:
    """Route-aware 0.5 s command + 1.0 s lateral commitment rollout."""

    source_lane = int(ego.lane_index)
    target_lane = source_lane + int(action.lateral_cmd)
    current = copy_vehicle_state(ego)
    target_current = copy_vehicle_state(ego)
    target_current.lane_index = target_lane
    target_current.lane_id = f"{ego.edge_id}_{target_lane}"
    speed = max(0.0, float(ego.speed))
    command_accel = float(action.accel_cmd) * 1.5
    target_speed = max(0.0, speed + command_accel * 0.5)
    lane_change_duration = max(
        float(config.scenario.get("lane_change_duration", 1.0)),
        dt,
    )
    states: list[VehicleState] = []
    for step in range(max(1, int(horizon_steps))):
        elapsed = float(step) * dt
        if elapsed < 0.5:
            speed = max(0.0, speed + command_accel * dt)
            acceleration = command_accel
        else:
            speed = target_speed
            acceleration = 0.0
        distance = speed * dt
        source_next, _source_taper_miss = advance_route_state(
            config,
            current,
            distance,
            lane_index=current.lane_index,
        )
        if action.lateral_cmd == 0:
            next_state = source_next
            target_next = source_next
        else:
            target_next, _target_taper_miss = advance_route_state(
                config,
                target_current,
                distance,
                lane_index=target_current.lane_index,
            )
            raw = min(
                1.0,
                float((step + 1) * dt / lane_change_duration),
            )
            progress = raw * raw * (3.0 - 2.0 * raw)
            if raw >= 1.0:
                next_state = target_next
            else:
                next_state = copy_vehicle_state(source_next)
                next_state.x = float(
                    source_next.x
                    + progress * (target_next.x - source_next.x)
                )
                next_state.y = float(
                    source_next.y
                    + progress * (target_next.y - source_next.y)
                )
                next_state.lane_index = source_lane
                next_state.lane_id = f"{source_next.edge_id}_{source_lane}"
                dx = float(next_state.x - current.x)
                dy = float(next_state.y - current.y)
                if math.hypot(dx, dy) > 1.0e-9:
                    next_state.heading = float(math.atan2(dy, dx))
        next_state.speed = float(speed)
        next_state.accel = float(acceleration)
        states.append(next_state)
        current = next_state
        target_current = target_next
    return states


def _actor_hypotheses(
    config: Any,
    actor: VehicleState,
    horizon_steps: int,
    dt: float,
) -> dict[str, list[VehicleState]]:
    result = {
        "keep_lane": route_aware_constant_velocity_rollout(
            actor, horizon_steps, dt, config
        )[0]
    }
    available = set(lane_indices(config, actor.edge_id))
    for lateral, name in ((-1, "change_right"), (1, "change_left")):
        target_lane = int(actor.lane_index) + lateral
        if target_lane not in available:
            continue
        synthetic = CandidateAction(
            index=-10 + lateral,
            lateral_cmd=lateral,
            accel_cmd=0,
            name=name,
        )
        result[name] = rollout_ego(
            actor,
            synthetic,
            horizon_steps,
            dt,
            config,
        )[0]
    return result


def _expanded_actor(
    state: VehicleState,
    elapsed_s: float,
    acceleration_bound: float,
    lateral_reach_m: float,
) -> VehicleState:
    reach = 0.5 * float(acceleration_bound) * float(elapsed_s) ** 2
    expanded = copy_vehicle_state(state)
    expanded.length = float(state.length + 2.0 * reach)
    expanded.width = float(state.width + 2.0 * lateral_reach_m)
    return expanded


@with_cached_scenario_semantics
def candidate_union_conflict_oracle_reference(
    config: Any,
    ego: VehicleState,
    current_vehicles: list[VehicleState],
    *,
    selector_scope: str = "prediction",
) -> tuple[dict[str, ActorConflictEvidence], tuple[int, ...]]:
    """Scalar audit reference for the candidate-union contract.

    The actor tube is a conservative, intent-agnostic reachability envelope:
    route-aware keep-lane CV plus every feasible adjacent-lane transition,
    with a bounded longitudinal acceleration envelope.
    """

    settings = _settings(
        config,
        selector_scope=selector_scope,
    )
    dt = float(config.scenario.step_length)
    horizon_steps = max(1, int(math.ceil(settings["horizon_s"] / dt)))
    vehicles = [
        vehicle
        for vehicle in current_vehicles
        if vehicle.vehicle_id != ego.vehicle_id
    ]
    legal_actions = _legal_actions(config, ego, current_vehicles)
    ego_rollouts = {
        int(action.index): _commitment_rollout(
            config,
            ego,
            action,
            horizon_steps,
            dt,
        )
        for action in legal_actions
    }
    result: dict[str, ActorConflictEvidence] = {}
    for actor in vehicles:
        conflict_candidates: set[int] = set()
        conflict_hypotheses: set[str] = set()
        conflict_surfaces: set[str] = set()
        earliest_conflict = INF_TTC
        earliest_overlap = INF_TTC
        current_gap = float(bbox_gap(ego, actor))
        minimum_gap = current_gap
        current_overlap = bool(geometric_overlap(ego, actor))
        current_same_surface = bool(
            str(ego.edge_id) == str(actor.edge_id)
            and int(ego.lane_index) == int(actor.lane_index)
        )
        current_surface_conflict = bool(
            current_same_surface
            and current_gap <= settings["surface_gap_m"]
        )
        if current_overlap or current_surface_conflict:
            conflict_candidates.update(ego_rollouts)
            conflict_hypotheses.add("current_state")
            conflict_surfaces.add(
                f"{ego.edge_id}:{int(ego.lane_index)}"
            )
            earliest_conflict = 0.0
        if current_overlap:
            earliest_overlap = 0.0
        hypotheses = _actor_hypotheses(
            config,
            actor,
            horizon_steps,
            dt,
        )
        unknown_topology = not bool(lane_indices(config, actor.edge_id))
        for candidate_id, ego_rollout in ego_rollouts.items():
            for hypothesis_id, actor_rollout in hypotheses.items():
                for step, (ego_state, raw_actor_state) in enumerate(
                    zip(ego_rollout, actor_rollout)
                ):
                    elapsed = float(step + 1) * dt
                    actor_state = _expanded_actor(
                        raw_actor_state,
                        elapsed,
                        settings["actor_longitudinal_accel_bound"],
                        (
                            min(
                                settings[
                                    "unknown_topology_lateral_reach_m"
                                ],
                                settings[
                                    "unknown_topology_lateral_reach_m"
                                ]
                                * elapsed
                                / max(
                                    float(
                                        config.scenario.get(
                                            "lane_change_duration", 1.0
                                        )
                                    ),
                                    dt,
                                ),
                            )
                            if unknown_topology
                            else 0.0
                        ),
                    )
                    gap = float(bbox_gap(ego_state, actor_state))
                    minimum_gap = min(minimum_gap, gap)
                    overlap = bool(geometric_overlap(ego_state, actor_state))
                    same_surface = bool(
                        str(ego_state.edge_id) == str(actor_state.edge_id)
                        and int(ego_state.lane_index)
                        == int(actor_state.lane_index)
                    )
                    surface_conflict = bool(
                        same_surface
                        and gap <= settings["surface_gap_m"]
                    )
                    if overlap:
                        earliest_overlap = min(earliest_overlap, elapsed)
                    if overlap or surface_conflict:
                        earliest_conflict = min(
                            earliest_conflict, elapsed
                        )
                        conflict_candidates.add(candidate_id)
                        conflict_hypotheses.add(hypothesis_id)
                        conflict_surfaces.add(
                            f"{ego_state.edge_id}:{int(ego_state.lane_index)}"
                        )
        result[str(actor.vehicle_id)] = ActorConflictEvidence(
            vehicle_id=str(actor.vehicle_id),
            candidate_conflict_eligible=bool(conflict_candidates),
            conflict_candidate_ids=tuple(sorted(conflict_candidates)),
            conflict_hypothesis_ids=tuple(sorted(conflict_hypotheses)),
            conflict_surface_ids=tuple(sorted(conflict_surfaces)),
            earliest_conflict_time_s=float(earliest_conflict),
            earliest_overlap_time_s=float(earliest_overlap),
            minimum_swept_obb_gap=float(minimum_gap),
        )
    eligible = [
        evidence
        for evidence in result.values()
        if evidence.candidate_conflict_eligible
    ]
    if eligible:
        nearest = min(
            eligible,
            key=lambda item: (
                item.minimum_swept_obb_gap,
                item.earliest_conflict_time_s,
                item.vehicle_id,
            ),
        )
        result[nearest.vehicle_id] = replace(
            nearest,
            nearest_candidate_conflict=True,
        )
    return result, tuple(sorted(ego_rollouts))


@with_cached_scenario_semantics
def candidate_union_conflict_oracle(
    config: Any,
    ego: VehicleState,
    current_vehicles: list[VehicleState],
    *,
    selector_scope: str = "prediction",
) -> tuple[dict[str, ActorConflictEvidence], tuple[int, ...]]:
    """Vectorized action × time × actor-hypothesis Selector-v3 oracle."""

    settings = _settings(
        config,
        selector_scope=selector_scope,
    )
    dt = float(config.scenario.step_length)
    horizon_steps = max(1, int(math.ceil(settings["horizon_s"] / dt)))
    vehicles = [
        vehicle
        for vehicle in current_vehicles
        if vehicle.vehicle_id != ego.vehicle_id
    ]
    legal_actions = _legal_actions(config, ego, current_vehicles)
    ego_rollouts_by_id = {
        int(action.index): _commitment_rollout(
            config,
            ego,
            action,
            horizon_steps,
            dt,
        )
        for action in legal_actions
    }
    legal_ids = tuple(sorted(ego_rollouts_by_id))
    if not vehicles:
        return {}, legal_ids
    action_ids = [int(action.index) for action in legal_actions]
    ego_rollouts = [
        ego_rollouts_by_id[action_id] for action_id in action_ids
    ]

    flat_actor_rollouts: list[list[VehicleState]] = []
    flat_actor_ids: list[str] = []
    flat_hypothesis_ids: list[str] = []
    actor_flat_indices: dict[str, list[int]] = {}
    for actor in vehicles:
        actor_id = str(actor.vehicle_id)
        unknown_topology = not bool(lane_indices(config, actor.edge_id))
        hypotheses = _actor_hypotheses(
            config,
            actor,
            horizon_steps,
            dt,
        )
        for hypothesis_id, rollout in hypotheses.items():
            expanded: list[VehicleState] = []
            for step, state in enumerate(rollout):
                elapsed = float(step + 1) * dt
                lateral_reach = (
                    min(
                        settings["unknown_topology_lateral_reach_m"],
                        settings["unknown_topology_lateral_reach_m"]
                        * elapsed
                        / max(
                            float(
                                config.scenario.get(
                                    "lane_change_duration", 1.0
                                )
                            ),
                            dt,
                        ),
                    )
                    if unknown_topology
                    else 0.0
                )
                expanded.append(
                    _expanded_actor(
                        state,
                        elapsed,
                        settings["actor_longitudinal_accel_bound"],
                        lateral_reach,
                    )
                )
            flat_index = len(flat_actor_rollouts)
            flat_actor_rollouts.append(expanded)
            flat_actor_ids.append(actor_id)
            flat_hypothesis_ids.append(hypothesis_id)
            actor_flat_indices.setdefault(actor_id, []).append(flat_index)

    gap, overlap = batch_pairwise_obb_gap_overlap(
        ego_rollouts,
        flat_actor_rollouts,
    )
    ego_edges = np.asarray(
        [[str(state.edge_id) for state in rollout] for rollout in ego_rollouts],
        dtype=object,
    )
    ego_lanes = np.asarray(
        [
            [int(state.lane_index) for state in rollout]
            for rollout in ego_rollouts
        ],
        dtype=np.int64,
    )
    actor_edges = np.asarray(
        [
            [str(state.edge_id) for state in rollout]
            for rollout in flat_actor_rollouts
        ],
        dtype=object,
    )
    actor_lanes = np.asarray(
        [
            [int(state.lane_index) for state in rollout]
            for rollout in flat_actor_rollouts
        ],
        dtype=np.int64,
    )
    same_surface = (
        ego_edges[:, :, None] == actor_edges.T[None, :, :]
    ) & (
        ego_lanes[:, :, None] == actor_lanes.T[None, :, :]
    )
    surface_conflict = same_surface & (
        gap <= float(settings["surface_gap_m"])
    )
    conflict = overlap | surface_conflict

    result: dict[str, ActorConflictEvidence] = {}
    for actor in vehicles:
        actor_id = str(actor.vehicle_id)
        flat_indices = actor_flat_indices[actor_id]
        actor_conflict = conflict[:, :, flat_indices]
        actor_overlap = overlap[:, :, flat_indices]
        actor_gaps = gap[:, :, flat_indices]
        conflict_candidates = {
            action_ids[action_index]
            for action_index in range(len(action_ids))
            if bool(np.any(actor_conflict[action_index]))
        }
        conflict_hypotheses = {
            flat_hypothesis_ids[flat_index]
            for flat_index in flat_indices
            if bool(
                np.any(
                    conflict[
                        :,
                        :,
                        flat_index,
                    ]
                )
            )
        }
        conflict_surfaces: set[str] = set()
        conflict_action_time = np.any(actor_conflict, axis=2)
        for action_index, time_index in np.argwhere(
            conflict_action_time
        ):
            conflict_surfaces.add(
                f"{ego_edges[action_index, time_index]}:"
                f"{int(ego_lanes[action_index, time_index])}"
            )
        conflict_time_indices = np.argwhere(actor_conflict)
        overlap_time_indices = np.argwhere(actor_overlap)
        earliest_conflict = (
            float((int(np.min(conflict_time_indices[:, 1])) + 1) * dt)
            if conflict_time_indices.size
            else INF_TTC
        )
        earliest_overlap = (
            float((int(np.min(overlap_time_indices[:, 1])) + 1) * dt)
            if overlap_time_indices.size
            else INF_TTC
        )
        minimum_gap = min(
            float(bbox_gap(ego, actor)),
            float(np.min(actor_gaps)),
        )

        current_overlap = bool(geometric_overlap(ego, actor))
        current_same_surface = bool(
            str(ego.edge_id) == str(actor.edge_id)
            and int(ego.lane_index) == int(actor.lane_index)
        )
        current_surface_conflict = bool(
            current_same_surface
            and float(bbox_gap(ego, actor))
            <= settings["surface_gap_m"]
        )
        if current_overlap or current_surface_conflict:
            conflict_candidates.update(action_ids)
            conflict_hypotheses.add("current_state")
            conflict_surfaces.add(
                f"{ego.edge_id}:{int(ego.lane_index)}"
            )
            earliest_conflict = 0.0
        if current_overlap:
            earliest_overlap = 0.0

        result[actor_id] = ActorConflictEvidence(
            vehicle_id=actor_id,
            candidate_conflict_eligible=bool(conflict_candidates),
            conflict_candidate_ids=tuple(sorted(conflict_candidates)),
            conflict_hypothesis_ids=tuple(
                sorted(conflict_hypotheses)
            ),
            conflict_surface_ids=tuple(sorted(conflict_surfaces)),
            earliest_conflict_time_s=float(earliest_conflict),
            earliest_overlap_time_s=float(earliest_overlap),
            minimum_swept_obb_gap=float(minimum_gap),
        )

    eligible = [
        evidence
        for evidence in result.values()
        if evidence.candidate_conflict_eligible
    ]
    if eligible:
        nearest = min(
            eligible,
            key=lambda item: (
                item.minimum_swept_obb_gap,
                item.earliest_conflict_time_s,
                item.vehicle_id,
            ),
        )
        result[nearest.vehicle_id] = replace(
            nearest,
            nearest_candidate_conflict=True,
        )
    return result, legal_ids
