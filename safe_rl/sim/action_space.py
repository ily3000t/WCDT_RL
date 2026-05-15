from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CandidateAction:
    index: int
    lateral_cmd: int
    accel_cmd: int
    name: str

    @property
    def is_fallback(self) -> bool:
        return self.lateral_cmd == 0 and self.accel_cmd < 0


def _build_actions() -> tuple[CandidateAction, ...]:
    actions: list[CandidateAction] = []
    index = 0
    for lateral in (-1, 0, 1):
        for accel in (-1, 0, 1):
            lat_name = {-1: "left", 0: "keep", 1: "right"}[lateral]
            acc_name = {-1: "decelerate", 0: "hold", 1: "accelerate"}[accel]
            actions.append(CandidateAction(index, lateral, accel, f"{lat_name}_{acc_name}"))
            index += 1
    return tuple(actions)


ACTIONS: tuple[CandidateAction, ...] = _build_actions()
FALLBACK_ACTION = next(action for action in ACTIONS if action.name == "keep_decelerate")


def decode_action(action: int | CandidateAction) -> CandidateAction:
    if isinstance(action, CandidateAction):
        return action
    if int(action) < 0 or int(action) >= len(ACTIONS):
        raise ValueError(f"action index out of range: {action}")
    return ACTIONS[int(action)]


def action_distance(a: int | CandidateAction, b: int | CandidateAction) -> float:
    action_a = decode_action(a)
    action_b = decode_action(b)
    return abs(action_a.lateral_cmd - action_b.lateral_cmd) + abs(action_a.accel_cmd - action_b.accel_cmd)
