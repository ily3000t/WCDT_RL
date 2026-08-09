from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class VehicleState:
    vehicle_id: str
    x: float
    y: float
    heading: float
    speed: float
    lane_index: int
    lane_id: str
    lane_pos: float
    edge_id: str
    length: float = 4.8
    width: float = 1.8
    accel: float = 0.0
    route_position_valid: bool = True
    projection_distance: float = 0.0
    projection_ambiguity_margin: float = float("inf")
    projection_failure_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def as_vector(self) -> list[float]:
        return [self.x, self.y, self.heading, self.speed, self.accel]


def copy_vehicle_state(state: VehicleState) -> VehicleState:
    """Return an exact, independent shallow copy without dataclass reflection.

    ``VehicleState`` contains only scalar/string fields.  Hot route rollouts can
    therefore use this explicit constructor instead of ``dataclasses.replace``;
    the latter repeatedly reflects over all dataclass fields and is materially
    slower in the selector's action x time x actor loop.
    """

    return VehicleState(
        vehicle_id=state.vehicle_id,
        x=state.x,
        y=state.y,
        heading=state.heading,
        speed=state.speed,
        lane_index=state.lane_index,
        lane_id=state.lane_id,
        lane_pos=state.lane_pos,
        edge_id=state.edge_id,
        length=state.length,
        width=state.width,
        accel=state.accel,
        route_position_valid=state.route_position_valid,
        projection_distance=state.projection_distance,
        projection_ambiguity_margin=state.projection_ambiguity_margin,
        projection_failure_reason=state.projection_failure_reason,
    )


@dataclass
class StepMetrics:
    min_distance: float
    min_ttc: float
    max_drac: float
    collision: bool
    near_miss: bool
    low_ttc: bool
    high_drac: bool
    merge_gap: float
    lane_oob: bool = False
    hard_brake: bool = False
    geometric_overlap: bool = False
    closest_vehicle_id: str = ""
    closest_vehicle_edge: str = ""
    closest_vehicle_lane: int = -1
    ttc_vehicle_id: str = ""
    drac_vehicle_id: str = ""

    def risk_label(self) -> float:
        return float(
            self.collision
            or self.near_miss
            or self.low_ttc
            or self.high_drac
            or self.lane_oob
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
