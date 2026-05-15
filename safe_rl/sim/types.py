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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def as_vector(self) -> list[float]:
        return [self.x, self.y, self.heading, self.speed, self.accel]


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
