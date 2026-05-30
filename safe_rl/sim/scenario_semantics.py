from __future__ import annotations

from dataclasses import replace
from typing import Any

from safe_rl.sim.metrics import INF_TTC
from safe_rl.sim.types import VehicleState


EDGE_ROLE_UNKNOWN = 0
EDGE_ROLE_RAMP = 1
EDGE_ROLE_AUXILIARY = 2
EDGE_ROLE_MAINLINE = 3
EDGE_ROLE_TARGET = 4


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
    return int(config.scenario.get("merge_target_lane", 2))


def auxiliary_lane(config: Any) -> int:
    return int(config.scenario.get("auxiliary_lane", 3))


def taper_edge(config: Any) -> str:
    return str(config.scenario.get("taper_edge", "main_aux"))


def edge_lengths(config: Any) -> dict[str, float]:
    configured = config.scenario.get("edge_lengths", {})
    if isinstance(configured, dict):
        return {str(key): float(value) for key, value in configured.items()}
    return {}


def edge_length(config: Any, edge_id: str) -> float:
    return float(edge_lengths(config).get(str(edge_id), 0.0))


def lane_centers(config: Any) -> dict[int, float]:
    configured = config.scenario.get("lane_centers", {})
    if isinstance(configured, dict) and configured:
        return {int(key): float(value) for key, value in configured.items()}
    return {0: -8.0, 1: -4.8, 2: -1.6, 3: 1.6}


def lane_center(config: Any, lane_index: int) -> float:
    centers = lane_centers(config)
    return float(centers.get(int(lane_index), centers.get(merge_target_lane(config), -1.6)))


def infer_lane_index(config: Any, y: float) -> int:
    centers = lane_centers(config)
    return min(centers, key=lambda lane: abs(float(y) - centers[lane]))


def is_ramp_edge(config: Any, edge_id: str) -> bool:
    return str(edge_id) in set(ramp_edges(config))


def is_auxiliary_edge(config: Any, edge_id: str) -> bool:
    return str(edge_id) in set(auxiliary_edges(config))


def is_mainline_edge(config: Any, edge_id: str) -> bool:
    return str(edge_id) in set(mainline_edges(config))


def is_target_lane_edge(config: Any, edge_id: str) -> bool:
    return str(edge_id) in set(target_lane_edges(config))


def edge_role(config: Any, edge_id: str, lane_index: int) -> int:
    if is_ramp_edge(config, edge_id):
        return EDGE_ROLE_RAMP
    if is_auxiliary_edge(config, edge_id) and int(lane_index) == auxiliary_lane(config):
        return EDGE_ROLE_AUXILIARY
    if is_target_lane_edge(config, edge_id) and int(lane_index) == merge_target_lane(config):
        return EDGE_ROLE_TARGET
    if is_mainline_edge(config, edge_id):
        return EDGE_ROLE_MAINLINE
    return EDGE_ROLE_UNKNOWN


def distance_to_taper(config: Any, state: VehicleState | None) -> float:
    if state is None:
        return INF_TTC
    edge_id = str(state.edge_id)
    if is_ramp_edge(config, edge_id):
        return max(0.0, edge_length(config, edge_id) - float(state.lane_pos)) + edge_length(config, taper_edge(config))
    if edge_id == taper_edge(config):
        return max(0.0, edge_length(config, edge_id) - float(state.lane_pos))
    if edge_id == str(config.scenario.get("success_edge", "main_out")):
        return -float(state.lane_pos)
    if edge_id == "main_in":
        return max(0.0, edge_length(config, edge_id) - float(state.lane_pos)) + edge_length(config, taper_edge(config))
    return float(config.scenario.get("merge_x", 220.0)) - float(state.x)


def is_taper_miss(config: Any, state: VehicleState | None, *, threshold: float | None = None) -> bool:
    if state is None or str(state.edge_id) != taper_edge(config):
        return False
    if int(state.lane_index) != auxiliary_lane(config):
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
    current.x = float(current.x) + remaining
    taper_miss = False
    while remaining > 0.0:
        length = edge_length(config, current.edge_id)
        if length <= 0.0:
            current.lane_pos += remaining
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
        if current.edge_id == taper_edge(config) and current.lane_index == auxiliary_lane(config):
            taper_miss = True
            current.lane_pos = length
            break
        if is_ramp_edge(config, current.edge_id):
            current.lane_index = auxiliary_lane(config)
        current.edge_id = outgoing
        current.lane_pos = 0.0
    current.y = lane_center(config, current.lane_index)
    return current, taper_miss
