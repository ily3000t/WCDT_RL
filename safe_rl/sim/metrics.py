from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from safe_rl.sim.types import StepMetrics, VehicleState


INF_TTC = 1.0e6
SAFETY_METRIC_VERSION = "oriented_box_v1"
_EPS = 1.0e-9


@dataclass(frozen=True)
class PairwiseBatchMetrics:
    """Exact oriented-box metrics for an ``action x time x actor`` batch."""

    gap: np.ndarray
    ttc: np.ndarray
    drac: np.ndarray
    overlap: np.ndarray


def _state_batch(states: Sequence[Sequence[VehicleState]]) -> dict[str, np.ndarray]:
    rows = [list(row) for row in states]
    if not rows:
        return {
            name: np.zeros((0, 0), dtype=np.float64)
            for name in ("x", "y", "heading", "speed", "length", "width")
        }
    horizon = len(rows[0])
    if any(len(row) != horizon for row in rows):
        raise ValueError("vehicle-state batches must have a rectangular time dimension")
    return {
        name: np.asarray(
            [[float(getattr(state, name)) for state in row] for row in rows],
            dtype=np.float64,
        )
        for name in ("x", "y", "heading", "speed", "length", "width")
    }


def _batch_box_geometry(values: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    heading = values["heading"]
    cos_h = np.cos(heading)
    sin_h = np.sin(heading)
    half_length = 0.5 * np.maximum(values["length"], 0.1)
    half_width = 0.5 * np.maximum(values["width"], 0.1)
    center_x = values["x"] - half_length * cos_h
    center_y = values["y"] - half_length * sin_h
    forward = np.stack([cos_h, sin_h], axis=-1)
    lateral = np.stack([-sin_h, cos_h], axis=-1)
    axes = np.stack([forward, lateral], axis=-2)
    signs = np.asarray(
        ((1.0, 1.0), (1.0, -1.0), (-1.0, -1.0), (-1.0, 1.0)),
        dtype=np.float64,
    )
    offsets = (
        signs[:, 0] * half_length[..., None]
    )[..., None] * forward[..., None, :] + (
        signs[:, 1] * half_width[..., None]
    )[..., None] * lateral[..., None, :]
    centers = np.stack([center_x, center_y], axis=-1)
    return centers[..., None, :] + offsets, axes


def _batch_point_segment_minimum(
    points: np.ndarray,
    segment_points: np.ndarray,
) -> np.ndarray:
    starts = segment_points
    ends = np.roll(segment_points, shift=-1, axis=-2)
    point = points[..., :, None, :]
    start = starts[..., None, :, :]
    delta = (ends - starts)[..., None, :, :]
    length_sq = np.sum(delta * delta, axis=-1)
    numerator = np.sum((point - start) * delta, axis=-1)
    ratio = np.divide(
        numerator,
        length_sq,
        out=np.zeros_like(numerator),
        where=length_sq > _EPS,
    )
    ratio = np.clip(ratio, 0.0, 1.0)
    # Distance ordering is unchanged in squared space.  Reducing before the
    # square root avoids materialising the closest-point array and computes
    # one square root per box pair instead of one per corner/edge pair.
    residual = (point - start) - ratio[..., None] * delta
    distance_sq = np.sum(residual * residual, axis=-1)
    return np.sqrt(np.min(distance_sq, axis=(-2, -1)))


def batch_pairwise_obb_metrics(
    ego_rollouts: Sequence[Sequence[VehicleState]],
    other_rollouts: Sequence[Sequence[VehicleState]],
) -> PairwiseBatchMetrics:
    """Vectorise the scalar OBB gap/TTC/DRAC contract without changing it.

    ``ego_rollouts`` is action-major and ``other_rollouts`` is actor-major.
    Returned arrays have shape ``[action, time, actor]``. The function uses
    float64 because the scalar reference implementation operates on Python
    floats and downstream thresholds must retain the same boolean decisions.
    """

    ego_values = _state_batch(ego_rollouts)
    other_values = _state_batch(other_rollouts)
    action_count, horizon = ego_values["x"].shape
    actor_count = int(other_values["x"].shape[0])
    if actor_count and int(other_values["x"].shape[1]) != horizon:
        raise ValueError("ego and surrounding rollout horizons must match")
    shape = (action_count, horizon, actor_count)
    if actor_count == 0:
        return PairwiseBatchMetrics(
            gap=np.full(shape, INF_TTC, dtype=np.float64),
            ttc=np.full(shape, INF_TTC, dtype=np.float64),
            drac=np.zeros(shape, dtype=np.float64),
            overlap=np.zeros(shape, dtype=np.bool_),
        )

    ego_points, ego_axes = _batch_box_geometry(ego_values)
    other_points, other_axes = _batch_box_geometry(other_values)
    ego_points = np.broadcast_to(
        ego_points[:, :, None, :, :],
        (action_count, horizon, actor_count, 4, 2),
    )
    other_points = np.broadcast_to(
        other_points[None, :, :, :, :].transpose(0, 2, 1, 3, 4),
        (action_count, horizon, actor_count, 4, 2),
    )
    ego_axes = np.broadcast_to(
        ego_axes[:, :, None, :, :],
        (action_count, horizon, actor_count, 2, 2),
    )
    other_axes = np.broadcast_to(
        other_axes[None, :, :, :, :].transpose(0, 2, 1, 3, 4),
        (action_count, horizon, actor_count, 2, 2),
    )
    axes = np.concatenate([ego_axes, other_axes], axis=-2)

    ego_projection = np.einsum("...pi,...ki->...pk", ego_points, axes)
    other_projection = np.einsum("...pi,...ki->...pk", other_points, axes)
    ego_min = np.min(ego_projection, axis=-2)
    ego_max = np.max(ego_projection, axis=-2)
    other_min = np.min(other_projection, axis=-2)
    other_max = np.max(other_projection, axis=-2)
    separated = (ego_max < other_min - _EPS) | (other_max < ego_min - _EPS)
    overlap = ~np.any(separated, axis=-1)

    ego_to_other = _batch_point_segment_minimum(ego_points, other_points)
    other_to_ego = _batch_point_segment_minimum(other_points, ego_points)
    gap = np.where(overlap, 0.0, np.minimum(ego_to_other, other_to_ego))

    ego_heading = ego_values["heading"][:, :, None]
    other_heading = other_values["heading"].T[None, :, :]
    ego_vx = ego_values["speed"][:, :, None] * np.cos(ego_heading)
    ego_vy = ego_values["speed"][:, :, None] * np.sin(ego_heading)
    other_speed = other_values["speed"].T[None, :, :]
    rel_vx = other_speed * np.cos(other_heading) - ego_vx
    rel_vy = other_speed * np.sin(other_heading) - ego_vy
    projected_velocity = rel_vx[..., None] * axes[..., 0] + rel_vy[..., None] * axes[..., 1]
    moving = np.abs(projected_velocity) > _EPS
    static_separated = (~moving) & separated
    t1 = np.divide(
        ego_min - other_max,
        projected_velocity,
        out=np.zeros_like(projected_velocity),
        where=moving,
    )
    t2 = np.divide(
        ego_max - other_min,
        projected_velocity,
        out=np.zeros_like(projected_velocity),
        where=moving,
    )
    axis_entry = np.where(moving, np.minimum(t1, t2), -INF_TTC)
    axis_exit = np.where(moving, np.maximum(t1, t2), INF_TTC)
    entry_time = np.max(axis_entry, axis=-1)
    exit_time = np.min(axis_exit, axis=-1)
    no_intersection = (
        np.any(static_separated, axis=-1)
        | ((entry_time - exit_time) > _EPS)
        | (exit_time < np.maximum(0.0, entry_time) - _EPS)
    )
    ttc = np.where(no_intersection, INF_TTC, np.maximum(0.0, entry_time))
    ttc = np.where(overlap, 0.0, ttc)

    finite_ttc = ttc < INF_TTC
    positive_gap = gap > _EPS
    closing_speed = np.divide(
        gap,
        np.maximum(ttc, 1.0e-6),
        out=np.zeros_like(gap),
        where=finite_ttc & positive_gap,
    )
    drac_values = np.divide(
        closing_speed * closing_speed,
        2.0 * gap,
        out=np.zeros_like(gap),
        where=finite_ttc & positive_gap,
    )
    drac_values = np.where(positive_gap, drac_values, INF_TTC)
    return PairwiseBatchMetrics(gap=gap, ttc=ttc, drac=drac_values, overlap=overlap)


def batch_pairwise_obb_gap_overlap(
    ego_rollouts: Sequence[Sequence[VehicleState]],
    other_rollouts: Sequence[Sequence[VehicleState]],
    *,
    other_length_expansion: np.ndarray | None = None,
    other_width_expansion: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return only exact OBB gap/overlap for an action-time-actor batch.

    Selector conflict tubes do not consume TTC or DRAC.  Keeping this as a
    separate API avoids their velocity projections/divisions while retaining
    the exact float64 SAT and point-to-segment calculations used by
    :func:`batch_pairwise_obb_metrics`.
    """

    ego_values = _state_batch(ego_rollouts)
    other_values = _state_batch(other_rollouts)
    if other_length_expansion is not None:
        expansion = np.asarray(other_length_expansion, dtype=np.float64)
        if expansion.shape != other_values["length"].shape:
            raise ValueError(
                "other length expansion must match the actor-time batch"
            )
        other_values["length"] = other_values["length"] + expansion
    if other_width_expansion is not None:
        expansion = np.asarray(other_width_expansion, dtype=np.float64)
        if expansion.shape != other_values["width"].shape:
            raise ValueError(
                "other width expansion must match the actor-time batch"
            )
        other_values["width"] = other_values["width"] + expansion
    action_count, horizon = ego_values["x"].shape
    actor_count = int(other_values["x"].shape[0])
    if actor_count and int(other_values["x"].shape[1]) != horizon:
        raise ValueError("ego and surrounding rollout horizons must match")
    shape = (action_count, horizon, actor_count)
    if actor_count == 0:
        return (
            np.full(shape, INF_TTC, dtype=np.float64),
            np.zeros(shape, dtype=np.bool_),
        )

    ego_points, ego_axes = _batch_box_geometry(ego_values)
    other_points, other_axes = _batch_box_geometry(other_values)
    ego_points = np.broadcast_to(
        ego_points[:, :, None, :, :],
        (action_count, horizon, actor_count, 4, 2),
    )
    other_points = np.broadcast_to(
        other_points[None, :, :, :, :].transpose(0, 2, 1, 3, 4),
        (action_count, horizon, actor_count, 4, 2),
    )
    ego_axes = np.broadcast_to(
        ego_axes[:, :, None, :, :],
        (action_count, horizon, actor_count, 2, 2),
    )
    other_axes = np.broadcast_to(
        other_axes[None, :, :, :, :].transpose(0, 2, 1, 3, 4),
        (action_count, horizon, actor_count, 2, 2),
    )
    axes = np.concatenate([ego_axes, other_axes], axis=-2)

    ego_projection = np.einsum("...pi,...ki->...pk", ego_points, axes)
    other_projection = np.einsum("...pi,...ki->...pk", other_points, axes)
    ego_min = np.min(ego_projection, axis=-2)
    ego_max = np.max(ego_projection, axis=-2)
    other_min = np.min(other_projection, axis=-2)
    other_max = np.max(other_projection, axis=-2)
    separated = (ego_max < other_min - _EPS) | (other_max < ego_min - _EPS)
    overlap = ~np.any(separated, axis=-1)
    ego_to_other = _batch_point_segment_minimum(ego_points, other_points)
    other_to_ego = _batch_point_segment_minimum(other_points, ego_points)
    gap = np.where(overlap, 0.0, np.minimum(ego_to_other, other_to_ego))
    return gap, overlap


def bbox_radius(state: VehicleState) -> float:
    """Legacy helper retained for callers that only need a conservative radius."""

    return 0.5 * math.hypot(max(state.length, 0.1), max(state.width, 0.1))


def vehicle_box_center(state: VehicleState) -> tuple[float, float]:
    """Convert SUMO's front-bumper center position to the vehicle geometry center."""

    half_length = 0.5 * max(float(state.length), 0.1)
    return (
        float(state.x) - half_length * math.cos(float(state.heading)),
        float(state.y) - half_length * math.sin(float(state.heading)),
    )


def _box_axes(state: VehicleState) -> tuple[tuple[float, float], tuple[float, float]]:
    heading = float(state.heading)
    forward = (math.cos(heading), math.sin(heading))
    lateral = (-forward[1], forward[0])
    return forward, lateral


def vehicle_box_corners(state: VehicleState) -> tuple[tuple[float, float], ...]:
    center_x, center_y = vehicle_box_center(state)
    forward, lateral = _box_axes(state)
    half_length = 0.5 * max(float(state.length), 0.1)
    half_width = 0.5 * max(float(state.width), 0.1)
    corners: list[tuple[float, float]] = []
    for longitudinal, transverse in (
        (half_length, half_width),
        (half_length, -half_width),
        (-half_length, -half_width),
        (-half_length, half_width),
    ):
        corners.append(
            (
                center_x + longitudinal * forward[0] + transverse * lateral[0],
                center_y + longitudinal * forward[1] + transverse * lateral[1],
            )
        )
    return tuple(corners)


def center_distance(a: VehicleState, b: VehicleState) -> float:
    ax, ay = vehicle_box_center(a)
    bx, by = vehicle_box_center(b)
    return math.hypot(ax - bx, ay - by)


def _project(points: tuple[tuple[float, float], ...], axis: tuple[float, float]) -> tuple[float, float]:
    values = [point[0] * axis[0] + point[1] * axis[1] for point in points]
    return min(values), max(values)


def geometric_overlap(a: VehicleState, b: VehicleState) -> bool:
    a_points = vehicle_box_corners(a)
    b_points = vehicle_box_corners(b)
    for axis in (*_box_axes(a), *_box_axes(b)):
        a_min, a_max = _project(a_points, axis)
        b_min, b_max = _project(b_points, axis)
        if a_max < b_min - _EPS or b_max < a_min - _EPS:
            return False
    return True


def _point_segment_distance(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> float:
    dx = float(end[0] - start[0])
    dy = float(end[1] - start[1])
    length_sq = dx * dx + dy * dy
    if length_sq <= _EPS:
        return math.hypot(float(point[0] - start[0]), float(point[1] - start[1]))
    ratio = (
        (float(point[0] - start[0]) * dx + float(point[1] - start[1]) * dy) / length_sq
    )
    ratio = max(0.0, min(1.0, ratio))
    closest_x = float(start[0]) + ratio * dx
    closest_y = float(start[1]) + ratio * dy
    return math.hypot(float(point[0]) - closest_x, float(point[1]) - closest_y)


def bbox_gap(a: VehicleState, b: VehicleState) -> float:
    """Return the shortest surface distance between two oriented vehicle boxes."""

    if geometric_overlap(a, b):
        return 0.0
    a_points = vehicle_box_corners(a)
    b_points = vehicle_box_corners(b)
    a_edges = tuple(zip(a_points, (*a_points[1:], a_points[0])))
    b_edges = tuple(zip(b_points, (*b_points[1:], b_points[0])))
    distances = [
        _point_segment_distance(point, start, end)
        for point in a_points
        for start, end in b_edges
    ]
    distances.extend(
        _point_segment_distance(point, start, end)
        for point in b_points
        for start, end in a_edges
    )
    return float(min(distances)) if distances else 0.0


def _velocity(state: VehicleState) -> tuple[float, float]:
    return (
        float(state.speed) * math.cos(float(state.heading)),
        float(state.speed) * math.sin(float(state.heading)),
    )


def relative_ttc(ego: VehicleState, other: VehicleState) -> float:
    """Return constant-velocity time to OBB overlap using swept SAT."""

    if geometric_overlap(ego, other):
        return 0.0
    ego_points = vehicle_box_corners(ego)
    other_points = vehicle_box_corners(other)
    ego_vx, ego_vy = _velocity(ego)
    other_vx, other_vy = _velocity(other)
    rel_vx = other_vx - ego_vx
    rel_vy = other_vy - ego_vy
    entry_time = -INF_TTC
    exit_time = INF_TTC
    for axis in (*_box_axes(ego), *_box_axes(other)):
        ego_min, ego_max = _project(ego_points, axis)
        other_min, other_max = _project(other_points, axis)
        projected_velocity = rel_vx * axis[0] + rel_vy * axis[1]
        if abs(projected_velocity) <= _EPS:
            if ego_max < other_min - _EPS or other_max < ego_min - _EPS:
                return INF_TTC
            continue
        t1 = (ego_min - other_max) / projected_velocity
        t2 = (ego_max - other_min) / projected_velocity
        entry_time = max(entry_time, min(t1, t2))
        exit_time = min(exit_time, max(t1, t2))
        if entry_time - exit_time > _EPS:
            return INF_TTC
    if exit_time < max(0.0, entry_time) - _EPS:
        return INF_TTC
    return float(max(0.0, entry_time))


def drac(ego: VehicleState, other: VehicleState) -> float:
    gap = bbox_gap(ego, other)
    if gap <= _EPS:
        return INF_TTC
    ttc = relative_ttc(ego, other)
    if ttc >= INF_TTC:
        return 0.0
    closing_speed = gap / max(ttc, 1.0e-6)
    return float((closing_speed * closing_speed) / (2.0 * gap))


def trajectory_min_obb_gap(
    ego_future: np.ndarray,
    other_future: np.ndarray,
    other_mask: np.ndarray,
    future_valid_mask: np.ndarray | None = None,
    ego_future_valid_mask: np.ndarray | None = None,
    agent_length: np.ndarray | None = None,
    agent_width: np.ndarray | None = None,
    ego_length: float = 4.8,
    ego_width: float = 1.8,
) -> float:
    """Compute the minimum exact OBB gap for trajectory arrays."""

    minimum = INF_TTC
    horizon = min(int(ego_future.shape[0]), int(other_future.shape[1]))
    for agent_idx in range(other_future.shape[0]):
        if float(other_mask[agent_idx]) <= 0.0:
            continue
        length = float(agent_length[agent_idx]) if agent_length is not None else 4.8
        width = float(agent_width[agent_idx]) if agent_width is not None else 1.8
        for step_idx in range(horizon):
            if future_valid_mask is not None and float(future_valid_mask[agent_idx, step_idx]) <= 0.5:
                continue
            if ego_future_valid_mask is not None and float(ego_future_valid_mask[step_idx]) <= 0.5:
                continue
            ego_step = ego_future[step_idx]
            other_step = other_future[agent_idx, step_idx]
            ego_state = VehicleState(
                vehicle_id="ego_future",
                x=float(ego_step[0]),
                y=float(ego_step[1]),
                heading=float(ego_step[2]) if ego_step.shape[0] > 2 else 0.0,
                speed=float(ego_step[3]) if ego_step.shape[0] > 3 else 0.0,
                lane_index=0,
                lane_id="",
                lane_pos=0.0,
                edge_id="",
                length=float(ego_length),
                width=float(ego_width),
            )
            other_state = VehicleState(
                vehicle_id=f"other_future_{agent_idx}",
                x=float(other_step[0]),
                y=float(other_step[1]),
                heading=float(other_step[2]) if other_step.shape[0] > 2 else 0.0,
                speed=float(other_step[3]) if other_step.shape[0] > 3 else 0.0,
                lane_index=0,
                lane_id="",
                lane_pos=0.0,
                edge_id="",
                length=length,
                width=width,
            )
            minimum = min(minimum, bbox_gap(ego_state, other_state))
    return float(minimum)


def merge_gap(
    ego: VehicleState,
    vehicles: Iterable[VehicleState],
    *,
    ego_edges: Iterable[str] | None = None,
    target_edges: Iterable[str] | None = None,
    target_lane: int | None = None,
    target_lanes: dict[str, int] | None = None,
) -> float:
    configured_ego_edges = set(ego_edges or ("ramp_in", "main_out"))
    configured_target_edges = set(target_edges or ("main_in", "main_out"))
    if ego.edge_id not in configured_ego_edges:
        return INF_TTC
    same_target = [
        vehicle
        for vehicle in vehicles
        if vehicle.vehicle_id != ego.vehicle_id
        and vehicle.edge_id in configured_target_edges
        and (
            int(vehicle.lane_index) == int(target_lanes[vehicle.edge_id])
            if target_lanes is not None and vehicle.edge_id in target_lanes
            else target_lane is None or int(vehicle.lane_index) == int(target_lane)
        )
    ]
    if not same_target:
        return INF_TTC
    # Merge-gap is a longitudinal corridor feature, not a physical minimum-distance metric.
    gaps = [abs(float(vehicle.x) - float(ego.x)) for vehicle in same_target]
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
    merge_ego_edges: Iterable[str] | None = None,
    merge_target_edges: Iterable[str] | None = None,
    merge_target_lane: int | None = None,
    merge_target_lanes: dict[str, int] | None = None,
) -> StepMetrics:
    if ego is None:
        return StepMetrics(0.0, 0.0, INF_TTC, True, True, True, True, 0.0, lane_oob, False)

    others = [vehicle for vehicle in vehicles if vehicle.vehicle_id != ego.vehicle_id]
    if not others:
        min_gap = INF_TTC
        min_ttc = INF_TTC
        max_drac = 0.0
        closest = None
        ttc_vehicle = None
        drac_vehicle = None
        overlap = False
    else:
        pair_metrics = [
            (other, bbox_gap(ego, other), relative_ttc(ego, other), drac(ego, other), geometric_overlap(ego, other))
            for other in others
        ]
        closest, min_gap, _, _, _ = min(pair_metrics, key=lambda item: item[1])
        ttc_vehicle, _, min_ttc, _, _ = min(pair_metrics, key=lambda item: item[2])
        drac_vehicle, _, _, max_drac, _ = max(pair_metrics, key=lambda item: item[3])
        overlap = any(item[4] for item in pair_metrics)

    return StepMetrics(
        min_distance=float(min_gap),
        min_ttc=float(min_ttc),
        max_drac=float(max_drac),
        collision=bool(collision),
        near_miss=bool(min_gap < near_miss_threshold),
        low_ttc=bool(min_ttc < ttc_threshold),
        high_drac=bool(max_drac > drac_threshold),
        merge_gap=float(
            merge_gap(
                ego,
                others,
                ego_edges=merge_ego_edges,
                target_edges=merge_target_edges,
                target_lane=merge_target_lane,
                target_lanes=merge_target_lanes,
            )
        ),
        lane_oob=bool(lane_oob),
        hard_brake=bool(ego.accel < hard_brake_threshold),
        geometric_overlap=bool(overlap),
        closest_vehicle_id=str(closest.vehicle_id) if closest is not None else "",
        closest_vehicle_edge=str(closest.edge_id) if closest is not None else "",
        closest_vehicle_lane=int(closest.lane_index) if closest is not None else -1,
        ttc_vehicle_id=str(ttc_vehicle.vehicle_id) if ttc_vehicle is not None else "",
        drac_vehicle_id=str(drac_vehicle.vehicle_id) if drac_vehicle is not None else "",
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
