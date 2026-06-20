from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from safe_rl.risk.merge_local import is_candidate_legal
from safe_rl.sim.action_space import ACTIONS, CandidateAction
from safe_rl.sim.metrics import INF_TTC, bbox_gap
from safe_rl.sim.scenario_semantics import target_lane_index


@dataclass(frozen=True)
class RuleDecision:
    action: int
    reason: str


class RuleGapAcceptancePolicy:
    """IDM-style longitudinal control with current-state gap acceptance only."""

    def __init__(self, cfg: Any):
        self.cfg = cfg
        self.settings = dict(cfg.get("rule_gap_acceptance", {}) or {})

    def act(self, context: dict[str, Any]) -> RuleDecision:
        ego = context.get("ego")
        local = context.get("merge_local")
        if ego is None or local is None:
            return RuleDecision(self._action(0, -1, context), "missing_context")
        merge_cmd = int(target_lane_index(self.cfg, ego.edge_id) - ego.lane_index)
        desired_accel = self._idm_accel(ego, context.get("target_front"), context.get("lane_speed_limit"))
        accel_cmd = 1 if desired_accel > 0.25 else (-1 if desired_accel < -0.25 else 0)
        safe_merge = self._safe_merge(context)
        if bool(local.ego_on_auxiliary) and merge_cmd != 0 and safe_merge:
            return RuleDecision(self._action(merge_cmd, accel_cmd, context), "safe_gap_merge")
        if bool(local.ego_on_auxiliary) and float(local.merge_distance) < float(
            self.settings.get("deadline_distance", 120.0)
        ):
            accel_cmd = min(accel_cmd, -1)
            return RuleDecision(self._action(0, accel_cmd, context), "deadline_wait_safe_gap")
        return RuleDecision(self._action(0, accel_cmd, context), "idm_follow")

    def _safe_merge(self, context: dict[str, Any]) -> bool:
        ego = context["ego"]
        local = context["merge_local"]
        front_gap = float(local.target_front_gap)
        rear_gap = float(local.target_rear_gap)
        if front_gap < float(self.settings.get("merge_min_front_gap", 8.0)):
            return False
        if rear_gap < float(self.settings.get("merge_min_rear_gap", 8.0)):
            return False
        front_ttc = self._ttc(front_gap, max(0.0, float(ego.speed - local.target_front_rel_speed - ego.speed)))
        rear_ttc = self._ttc(rear_gap, max(0.0, float(local.target_rear_rel_speed)))
        if front_ttc < float(self.settings.get("merge_front_ttc_min", 3.0)):
            return False
        if rear_ttc < float(self.settings.get("merge_rear_ttc_min", 3.0)):
            return False
        for vehicle in (context.get("target_front"), context.get("target_rear")):
            if vehicle is not None and bbox_gap(ego, vehicle) <= 0.0:
                return False
        return True

    @staticmethod
    def _ttc(gap: float, closing_speed: float) -> float:
        if closing_speed <= 1.0e-6:
            return INF_TTC
        return max(0.0, gap) / closing_speed

    def _idm_accel(self, ego: Any, front: Any | None, lane_speed_limit: float | None) -> float:
        desired_speed = float(lane_speed_limit or 25.0)
        max_accel = float(self.settings.get("idm_max_acceleration", 1.5))
        comfortable_decel = float(self.settings.get("idm_comfortable_deceleration", 2.0))
        min_gap = float(self.settings.get("idm_min_gap", 2.0))
        headway = float(self.settings.get("idm_time_headway", 1.5))
        free = max_accel * (1.0 - (float(ego.speed) / max(desired_speed, 1.0e-6)) ** 4)
        if front is None:
            return free
        gap = max(0.1, float(front.x - ego.x) - 0.5 * (float(front.length) + float(ego.length)))
        closing = float(ego.speed - front.speed)
        desired_gap = min_gap + max(
            0.0,
            float(ego.speed) * headway + float(ego.speed) * closing / (2.0 * (max_accel * comfortable_decel) ** 0.5),
        )
        return free - max_accel * (desired_gap / gap) ** 2

    @staticmethod
    def _action(lateral_cmd: int, accel_cmd: int, context: dict[str, Any]) -> int:
        for action in ACTIONS:
            if (
                action.lateral_cmd == int(lateral_cmd)
                and action.accel_cmd == int(accel_cmd)
                and is_candidate_legal(action, context)
            ):
                return int(action.index)
        for action in ACTIONS:
            if action.lateral_cmd == 0 and action.accel_cmd == int(accel_cmd):
                return int(action.index)
        return next(action.index for action in ACTIONS if action.name == "keep_decelerate")
