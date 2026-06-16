from __future__ import annotations

import math
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

from safe_rl.sim.metrics import INF_TTC
from safe_rl.sim.types import VehicleState


EDGE_ROLE_UNKNOWN = 0
EDGE_ROLE_RAMP = 1
EDGE_ROLE_AUXILIARY = 2
EDGE_ROLE_MAINLINE = 3
EDGE_ROLE_TARGET = 4


@dataclass(frozen=True)
class RouteProjection:
    edge_id: str
    lane_index: int
    lane_id: str
    lane_pos: float
    valid: bool
    projection_distance: float
    ambiguity_margin: float
    failure_reason: str = ""


def _scenario_list(config: Any, key: str, default: list[str]) -> list[str]:
    value = config.scenario.get(key, default)
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


def ramp_edges(config: Any) -> list[str]:
    return _scenario_list(config, "ramp_edges", ["ramp_in"])


def auxiliary_edges(config: Any) -> list[str]:
    return _scenario_list(config, "auxiliary_edges", ["main_aux"])


def mainline_edges(config: Any) -> list[str]:
    return _scenario_list(config, "mainline_edges", ["main_in", "main_aux", "main_out"])


def target_lane_edges(config: Any) -> list[str]:
    return _scenario_list(config, "target_lane_edges", mainline_edges(config))


def merge_zone_edges(config: Any) -> list[str]:
    return _scenario_list(config, "merge_zone_edges", [*ramp_edges(config), *auxiliary_edges(config)])


def merge_target_lane(config: Any) -> int:
    """Legacy fallback for callers without edge context."""

    return int(config.scenario.get("merge_target_lane", 2))


def auxiliary_lane(config: Any) -> int:
    """Legacy fallback for callers without edge context."""

    return int(config.scenario.get("auxiliary_lane", 3))


def _scenario_lane_mapping(config: Any, key: str) -> dict[str, int]:
    raw = config.scenario.get(key, {})
    if not isinstance(raw, dict):
        return {}
    return {str(edge_id): int(lane_index) for edge_id, lane_index in raw.items()}


def target_lane_mapping(config: Any) -> dict[str, int]:
    return _scenario_lane_mapping(config, "target_lane_by_edge")


def auxiliary_lane_mapping(config: Any) -> dict[str, int]:
    return _scenario_lane_mapping(config, "auxiliary_lane_by_edge")


def target_lane_index(config: Any, edge_id: str | None = None) -> int:
    mapping = target_lane_mapping(config)
    if edge_id is not None and str(edge_id) in mapping:
        return int(mapping[str(edge_id)])
    return merge_target_lane(config)


def auxiliary_lane_index(config: Any, edge_id: str | None = None) -> int:
    mapping = auxiliary_lane_mapping(config)
    if edge_id is not None and str(edge_id) in mapping:
        return int(mapping[str(edge_id)])
    return auxiliary_lane(config)


def merge_side(config: Any) -> str:
    return str(config.scenario.get("merge_side", "left")).strip().lower()


def is_ramp_side_y(config: Any, y: float, *, margin: float = 0.5) -> bool:
    reference = lane_centers(config).get(auxiliary_lane(config), 0.0)
    return (
        float(y) < float(reference) - abs(float(margin))
        if merge_side(config) == "right"
        else float(y) > float(reference) + abs(float(margin))
    )


def taper_edge(config: Any) -> str:
    return str(config.scenario.get("taper_edge", "main_aux"))


def edge_lengths(config: Any) -> dict[str, float]:
    configured = config.scenario.get("edge_lengths", {})
    if isinstance(configured, dict):
        return {str(key): float(value) for key, value in configured.items()}
    return {}


def _shape_points(value: str) -> tuple[tuple[float, float], ...]:
    points = []
    for item in str(value).split():
        x_text, y_text = item.split(",", 1)
        points.append((float(x_text), float(y_text)))
    return tuple(points)


@lru_cache(maxsize=8)
def _net_lane_geometry(net_file: str) -> dict[str, dict[int, dict[str, Any]]]:
    path = Path(net_file)
    if not path.is_file():
        return {}
    root = ET.parse(path).getroot()
    output: dict[str, dict[int, dict[str, Any]]] = {}
    for edge in root.findall("edge"):
        if edge.attrib.get("function") == "internal":
            continue
        edge_id = str(edge.attrib.get("id", ""))
        lanes: dict[int, dict[str, Any]] = {}
        for lane in edge.findall("lane"):
            lanes[int(lane.attrib.get("index", "0"))] = {
                "length": float(lane.attrib.get("length", "0")),
                "points": _shape_points(lane.attrib.get("shape", "")),
            }
        if lanes:
            output[edge_id] = lanes
    return output


@lru_cache(maxsize=8)
def _net_lane_connections(net_file: str) -> dict[tuple[str, int, str], int]:
    path = Path(net_file)
    if not path.is_file():
        return {}
    root = ET.parse(path).getroot()
    output: dict[tuple[str, int, str], int] = {}
    for connection in root.findall("connection"):
        from_edge = str(connection.attrib.get("from", ""))
        to_edge = str(connection.attrib.get("to", ""))
        if from_edge.startswith(":") or to_edge.startswith(":"):
            continue
        output[(from_edge, int(connection.attrib.get("fromLane", "0")), to_edge)] = int(
            connection.attrib.get("toLane", "0")
        )
    return output


def connected_lane_index(config: Any, from_edge: str, from_lane: int, to_edge: str) -> int | None:
    net_file = str(config.scenario.get("net_file", ""))
    if not net_file:
        return None
    return _net_lane_connections(net_file).get((str(from_edge), int(from_lane), str(to_edge)))


def _lane_geometry(config: Any, edge_id: str, lane_index: int | None = None) -> dict[str, Any] | None:
    net_file = str(config.scenario.get("net_file", ""))
    geometry = _net_lane_geometry(net_file) if net_file else {}
    lanes = geometry.get(str(edge_id), {})
    if not lanes:
        return None
    if lane_index is not None and int(lane_index) in lanes:
        return lanes[int(lane_index)]
    return lanes[min(lanes)]


def _point_at_distance(points: tuple[tuple[float, float], ...], distance: float) -> tuple[float, float] | None:
    if not points:
        return None
    if len(points) == 1:
        return points[0]
    remaining = max(0.0, float(distance))
    for start, end in zip(points, points[1:]):
        dx = float(end[0] - start[0])
        dy = float(end[1] - start[1])
        length = math.hypot(dx, dy)
        if length <= 1.0e-9:
            continue
        if remaining <= length:
            ratio = remaining / length
            return float(start[0] + ratio * dx), float(start[1] + ratio * dy)
        remaining -= length
    return points[-1]


def lane_point(config: Any, edge_id: str, lane_index: int, lane_pos: float) -> tuple[float, float] | None:
    geometry = _lane_geometry(config, edge_id, lane_index)
    if geometry is None:
        return None
    return _point_at_distance(geometry["points"], lane_pos)


def lane_heading(config: Any, edge_id: str, lane_index: int, lane_pos: float) -> float | None:
    """Return the route tangent heading at a lane position."""

    geometry = _lane_geometry(config, edge_id, lane_index)
    if geometry is None:
        return None
    length = max(float(geometry["length"]), 0.0)
    start = _point_at_distance(geometry["points"], max(0.0, float(lane_pos) - 0.25))
    end = _point_at_distance(geometry["points"], min(length, float(lane_pos) + 0.25))
    if start is None or end is None:
        return None
    dx = float(end[0] - start[0])
    dy = float(end[1] - start[1])
    if math.hypot(dx, dy) <= 1.0e-9:
        return None
    return float(math.atan2(dy, dx))


def _nearest_distance_on_shape(
    points: tuple[tuple[float, float], ...],
    x: float,
    y: float,
) -> tuple[float, float]:
    if not points:
        return 0.0, INF_TTC
    if len(points) == 1:
        return 0.0, math.hypot(float(x) - points[0][0], float(y) - points[0][1])
    best_along = 0.0
    best_distance = INF_TTC
    traversed = 0.0
    for start, end in zip(points, points[1:]):
        dx = float(end[0] - start[0])
        dy = float(end[1] - start[1])
        length_sq = dx * dx + dy * dy
        length = math.sqrt(length_sq)
        if length <= 1.0e-9:
            continue
        ratio = float(
            max(0.0, min(1.0, ((float(x) - start[0]) * dx + (float(y) - start[1]) * dy) / length_sq))
        )
        projected_x = float(start[0] + ratio * dx)
        projected_y = float(start[1] + ratio * dy)
        distance = math.hypot(float(x) - projected_x, float(y) - projected_y)
        if distance < best_distance:
            best_along = traversed + ratio * length
            best_distance = distance
        traversed += length
    return float(best_along), float(best_distance)


def edge_length(config: Any, edge_id: str, lane_index: int | None = None) -> float:
    geometry = _lane_geometry(config, edge_id, lane_index)
    if geometry is not None:
        return float(geometry["length"])
    return float(edge_lengths(config).get(str(edge_id), 0.0))


def lane_centers(config: Any) -> dict[int, float]:
    configured = config.scenario.get("lane_centers", {})
    if isinstance(configured, dict) and configured:
        return {int(key): float(value) for key, value in configured.items()}
    return {0: -8.0, 1: -4.8, 2: -1.6, 3: 1.6}


def lane_center(
    config: Any,
    lane_index: int,
    edge_id: str | None = None,
    lane_pos: float | None = None,
) -> float:
    if edge_id is not None:
        geometry = _lane_geometry(config, edge_id, lane_index)
        if geometry is not None:
            point = _point_at_distance(
                geometry["points"],
                0.5 * float(geometry["length"]) if lane_pos is None else float(lane_pos),
            )
            if point is not None:
                return float(point[1])
    centers = lane_centers(config)
    return float(centers.get(int(lane_index), centers.get(merge_target_lane(config), -1.6)))


def infer_lane_index(config: Any, y: float) -> int:
    centers = lane_centers(config)
    return min(centers, key=lambda lane: abs(float(y) - centers[lane]))


def infer_route_position(
    config: Any,
    x: float,
    y: float,
    lane_index: int | None = None,
    *,
    edge_ids: list[str] | None = None,
) -> tuple[str | None, float]:
    candidates = edge_ids or list(_net_lane_geometry(str(config.scenario.get("net_file", ""))))
    best_edge: str | None = None
    best_lane_pos = 0.0
    best_distance = INF_TTC
    for edge_id in candidates:
        geometry = _lane_geometry(config, edge_id, lane_index)
        if geometry is None:
            continue
        lane_pos, distance = _nearest_distance_on_shape(geometry["points"], x, y)
        if distance < best_distance:
            best_edge = str(edge_id)
            best_lane_pos = float(lane_pos)
            best_distance = float(distance)
    return best_edge, best_lane_pos


def _projection_candidates(
    config: Any,
    previous: VehicleState,
) -> list[tuple[str, int, bool]]:
    """Return current and route-reachable lane candidates.

    The boolean marks candidates that preserve the previous route/lane
    continuity and therefore may resolve a geometric projection tie.
    """

    net_file = str(config.scenario.get("net_file", ""))
    geometry = _net_lane_geometry(net_file) if net_file else {}
    current_edge = str(previous.edge_id)
    current_lane = int(previous.lane_index)
    candidates: list[tuple[str, int, bool]] = []

    def add(edge_id: str, lane_index: int, preferred: bool) -> None:
        item = (str(edge_id), int(lane_index), bool(preferred))
        if item not in candidates and int(lane_index) in geometry.get(str(edge_id), {}):
            candidates.append(item)

    add(current_edge, current_lane, True)
    for lane_index in sorted(geometry.get(current_edge, {})):
        if abs(int(lane_index) - current_lane) <= 1:
            add(current_edge, int(lane_index), int(lane_index) == current_lane)

    outgoing = next_edge(config, current_edge)
    if outgoing is not None:
        for edge_id, lane_index, _preferred in list(candidates):
            if edge_id != current_edge:
                continue
            connected = connected_lane_index(config, current_edge, lane_index, outgoing)
            if connected is not None:
                add(
                    outgoing,
                    int(connected),
                    lane_index == current_lane,
                )
        connected = connected_lane_index(config, current_edge, current_lane, outgoing)
        if connected is not None:
            add(outgoing, int(connected), True)
    return candidates


def project_route_position(
    config: Any,
    x: float,
    y: float,
    previous: VehicleState,
) -> RouteProjection:
    settings = config.prediction.get("route_projection", {})
    max_distance = float(settings.get("max_projection_distance", 2.5))
    ambiguity_threshold = float(settings.get("ambiguity_margin", 0.25))
    scored: list[tuple[float, str, int, float, bool]] = []
    for edge_id, lane_index, preferred in _projection_candidates(config, previous):
        geometry = _lane_geometry(config, edge_id, lane_index)
        if geometry is None:
            continue
        lane_pos, distance = _nearest_distance_on_shape(
            geometry["points"],
            float(x),
            float(y),
        )
        scored.append(
            (
                float(distance),
                str(edge_id),
                int(lane_index),
                float(lane_pos),
                bool(preferred),
            )
        )
    if not scored:
        return RouteProjection(
            edge_id=str(previous.edge_id),
            lane_index=int(previous.lane_index),
            lane_id=str(previous.lane_id),
            lane_pos=float(previous.lane_pos),
            valid=False,
            projection_distance=INF_TTC,
            ambiguity_margin=0.0,
            failure_reason="no_reachable_lane_candidate",
        )
    scored.sort(key=lambda item: (item[0], item[1], item[2]))
    best = scored[0]
    second_distance = scored[1][0] if len(scored) > 1 else INF_TTC
    ambiguity_margin = float(second_distance - best[0])
    if ambiguity_margin < ambiguity_threshold:
        tied = [item for item in scored if item[0] - best[0] < ambiguity_threshold]
        preferred = [item for item in tied if item[4]]
        if len(preferred) == 1:
            best = preferred[0]
            remaining = [item for item in scored if item != best]
            second_distance = remaining[0][0] if remaining else INF_TTC
            ambiguity_margin = float(abs(second_distance - best[0]))
        elif len(tied) > 1:
            return RouteProjection(
                edge_id=str(best[1]),
                lane_index=int(best[2]),
                lane_id=f"{best[1]}_{best[2]}",
                lane_pos=float(best[3]),
                valid=False,
                projection_distance=float(best[0]),
                ambiguity_margin=ambiguity_margin,
                failure_reason="ambiguous_lane_projection",
            )
    if best[0] > max_distance:
        return RouteProjection(
            edge_id=str(best[1]),
            lane_index=int(best[2]),
            lane_id=f"{best[1]}_{best[2]}",
            lane_pos=float(best[3]),
            valid=False,
            projection_distance=float(best[0]),
            ambiguity_margin=ambiguity_margin,
            failure_reason="projection_distance_exceeded",
        )
    return RouteProjection(
        edge_id=str(best[1]),
        lane_index=int(best[2]),
        lane_id=f"{best[1]}_{best[2]}",
        lane_pos=float(best[3]),
        valid=True,
        projection_distance=float(best[0]),
        ambiguity_margin=ambiguity_margin,
    )


def lane_center_at_x(
    config: Any,
    lane_index: int,
    x: float,
    *,
    edge_ids: list[str] | None = None,
) -> float:
    candidates = edge_ids or target_lane_edges(config)
    best_edge: str | None = None
    best_lane_pos = 0.0
    best_x_distance = INF_TTC
    for edge_id in candidates:
        geometry = _lane_geometry(config, edge_id, lane_index)
        if geometry is None or not geometry["points"]:
            continue
        xs = [float(point[0]) for point in geometry["points"]]
        left = min(xs)
        right = max(xs)
        x_distance = max(0.0, left - float(x), float(x) - right)
        if x_distance < best_x_distance:
            best_edge = str(edge_id)
            best_x_distance = float(x_distance)
            start_x = float(geometry["points"][0][0])
            end_x = float(geometry["points"][-1][0])
            ratio = 0.0 if abs(end_x - start_x) <= 1.0e-9 else (float(x) - start_x) / (end_x - start_x)
            best_lane_pos = max(0.0, min(float(geometry["length"]), ratio * float(geometry["length"])))
    return lane_center(config, lane_index, best_edge, best_lane_pos)


def target_lane_center_at_x(config: Any, x: float, *, edge_ids: list[str] | None = None) -> float:
    candidates = edge_ids or target_lane_edges(config)
    best_center: float | None = None
    best_x_distance = INF_TTC
    for edge_id in candidates:
        lane_index = target_lane_index(config, edge_id)
        geometry = _lane_geometry(config, edge_id, lane_index)
        if geometry is None or not geometry["points"]:
            continue
        xs = [float(point[0]) for point in geometry["points"]]
        left = min(xs)
        right = max(xs)
        x_distance = max(0.0, left - float(x), float(x) - right)
        if x_distance < best_x_distance:
            best_x_distance = float(x_distance)
            start_x = float(geometry["points"][0][0])
            end_x = float(geometry["points"][-1][0])
            ratio = 0.0 if abs(end_x - start_x) <= 1.0e-9 else (float(x) - start_x) / (end_x - start_x)
            lane_pos = max(0.0, min(float(geometry["length"]), ratio * float(geometry["length"])))
            best_center = lane_center(config, lane_index, edge_id, lane_pos)
    return float(best_center if best_center is not None else lane_center(config, merge_target_lane(config)))


def is_ramp_edge(config: Any, edge_id: str) -> bool:
    return str(edge_id) in set(ramp_edges(config))


def is_auxiliary_edge(config: Any, edge_id: str) -> bool:
    return str(edge_id) in set(auxiliary_edges(config))


def is_mainline_edge(config: Any, edge_id: str) -> bool:
    return str(edge_id) in set(mainline_edges(config))


def is_target_lane_edge(config: Any, edge_id: str) -> bool:
    return str(edge_id) in set(target_lane_edges(config))


def is_target_lane(config: Any, edge_id: str, lane_index: int) -> bool:
    return is_target_lane_edge(config, edge_id) and int(lane_index) == target_lane_index(config, edge_id)


def edge_role(config: Any, edge_id: str, lane_index: int) -> int:
    if is_ramp_edge(config, edge_id):
        return EDGE_ROLE_RAMP
    if is_auxiliary_edge(config, edge_id) and int(lane_index) == auxiliary_lane_index(config, edge_id):
        return EDGE_ROLE_AUXILIARY
    if is_target_lane(config, edge_id, lane_index):
        return EDGE_ROLE_TARGET
    if is_mainline_edge(config, edge_id):
        return EDGE_ROLE_MAINLINE
    return EDGE_ROLE_UNKNOWN


def distance_to_taper(config: Any, state: VehicleState | None) -> float:
    if state is None:
        return INF_TTC
    edge_id = str(state.edge_id)
    if is_ramp_edge(config, edge_id):
        return max(0.0, edge_length(config, edge_id, state.lane_index) - float(state.lane_pos)) + edge_length(
            config,
            taper_edge(config),
            auxiliary_lane_index(config, taper_edge(config)),
        )
    if edge_id == taper_edge(config):
        return max(0.0, edge_length(config, edge_id, state.lane_index) - float(state.lane_pos))
    if edge_id == str(config.scenario.get("success_edge", "main_out")):
        return -float(state.lane_pos)
    if edge_id == "main_in":
        return max(0.0, edge_length(config, edge_id, state.lane_index) - float(state.lane_pos)) + edge_length(
            config,
            taper_edge(config),
            target_lane_index(config, taper_edge(config)),
        )
    return float(config.scenario.get("merge_x", 220.0)) - float(state.x)


def merge_corridor_progress(config: Any, state: VehicleState | None) -> float | None:
    """Return signed longitudinal progress relative to the taper deadline."""

    if state is None:
        return None
    edge_id = str(state.edge_id)
    known_edges = {
        *ramp_edges(config),
        *mainline_edges(config),
        *auxiliary_edges(config),
        str(config.scenario.get("success_edge", "main_out")),
    }
    if edge_id not in known_edges:
        return None
    return -float(distance_to_taper(config, state))


def distance_to_taper_for_position(
    config: Any,
    x: float,
    y: float,
    lane_index: int | None = None,
) -> float:
    edge_id, lane_pos = infer_route_position(config, x, y, lane_index)
    if edge_id is None:
        return float(config.scenario.get("merge_x", 220.0)) - float(x)
    state = VehicleState(
        vehicle_id="_semantic",
        x=float(x),
        y=float(y),
        heading=0.0,
        speed=0.0,
        lane_index=int(lane_index if lane_index is not None else infer_lane_index(config, y)),
        lane_id=f"{edge_id}_{lane_index if lane_index is not None else 0}",
        lane_pos=float(lane_pos),
        edge_id=str(edge_id),
    )
    return distance_to_taper(config, state)


def is_taper_miss(config: Any, state: VehicleState | None, *, threshold: float | None = None) -> bool:
    if state is None or str(state.edge_id) != taper_edge(config):
        return False
    if int(state.lane_index) != auxiliary_lane_index(config, state.edge_id):
        return False
    miss_distance = float(
        config.scenario.get("taper_miss_distance", config.scenario.get("taper_warning_distance", 40.0))
        if threshold is None
        else threshold
    )
    return distance_to_taper(config, state) <= miss_distance


def next_edge(config: Any, edge_id: str) -> str | None:
    edge_id = str(edge_id)
    if edge_id in ramp_edges(config) or edge_id == "main_in":
        return taper_edge(config)
    if edge_id == taper_edge(config):
        return str(config.scenario.get("success_edge", "main_out"))
    return None


def advance_route_state(
    config: Any,
    state: VehicleState,
    distance: float,
    *,
    lane_index: int | None = None,
) -> tuple[VehicleState, bool]:
    """Advance a CV rollout across configured edges and flag an unrecoverable taper miss."""

    current = replace(state)
    current.lane_index = int(state.lane_index if lane_index is None else lane_index)
    current.lane_pos = float(state.lane_pos)
    remaining = max(0.0, float(distance))
    taper_miss = False
    while remaining > 0.0:
        length = edge_length(config, current.edge_id, current.lane_index)
        if length <= 0.0:
            current.lane_pos += remaining
            current.x = float(current.x) + remaining
            break
        available = max(0.0, length - current.lane_pos)
        if remaining <= available:
            current.lane_pos += remaining
            break
        remaining -= available
        outgoing = next_edge(config, current.edge_id)
        if outgoing is None:
            current.lane_pos = length
            break
        if current.edge_id == taper_edge(config) and current.lane_index == auxiliary_lane_index(config, current.edge_id):
            taper_miss = True
            current.lane_pos = length
            break
        connected_lane = connected_lane_index(config, current.edge_id, current.lane_index, outgoing)
        if connected_lane is not None:
            current.lane_index = int(connected_lane)
        elif is_ramp_edge(config, current.edge_id):
            current.lane_index = auxiliary_lane_index(config, outgoing)
        current.edge_id = outgoing
        current.lane_id = f"{outgoing}_{current.lane_index}"
        current.lane_pos = 0.0
    geometry = _lane_geometry(config, current.edge_id, current.lane_index)
    point = _point_at_distance(geometry["points"], current.lane_pos) if geometry is not None else None
    if point is not None:
        current.x, current.y = point
        heading = lane_heading(config, current.edge_id, current.lane_index, current.lane_pos)
        if heading is not None:
            current.heading = heading
    else:
        current.x = float(state.x) + max(0.0, float(distance))
        current.y = lane_center(config, current.lane_index, current.edge_id, current.lane_pos)
    return current, taper_miss
