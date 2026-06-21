from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from safe_rl.sim.types import VehicleState


@dataclass(frozen=True)
class RuleControlContext:
    """Current-state-only contract for deterministic comparison controllers."""

    ego: VehicleState | None
    current_lane_front: VehicleState | None
    target_front: VehicleState | None
    target_rear: VehicleState | None
    current_lane_front_gap: float
    target_front_gap: float
    target_rear_gap: float
    target_front_closing_speed: float
    target_rear_closing_speed: float
    target_front_ttc: float
    target_rear_ttc: float
    lane_speed_limit: float | None
    distance_to_taper: float
    ego_on_auxiliary: bool
    merge_lateral_cmd: int
    legal_action_indices: frozenset[int]


@dataclass(frozen=True)
class RuleDecision:
    action: int
    reason: str


class RulePolicy(Protocol):
    def act(self, context: RuleControlContext) -> RuleDecision: ...
