from __future__ import annotations

from typing import Any

from safe_rl.baselines.api import RuleControlContext, RuleDecision
from safe_rl.sim.action_space import ACTIONS, CandidateAction
from safe_rl.sim.metrics import INF_TTC, bbox_gap


class RuleGapAcceptancePolicy:
    """IDM-style longitudinal control with current-state gap acceptance only."""

    def __init__(self, cfg: Any):
        self.cfg = cfg
        self.settings = dict(cfg.get("rule_gap_acceptance", {}) or {})

    def act(self, context: RuleControlContext) -> RuleDecision:
        ego = context.ego
        if ego is None:
            return RuleDecision(self._action(0, -1, context), "missing_context")
        merge_cmd = int(context.merge_lateral_cmd)
        desired_accel = self._idm_accel(ego, context.current_lane_front, context.lane_speed_limit, context.current_lane_front_gap)
        accel_cmd = 1 if desired_accel > 0.25 else (-1 if desired_accel < -0.25 else 0)
        safe_merge = self._safe_merge(context)
        merge_legal = any(
            action.lateral_cmd == merge_cmd and action.index in context.legal_action_indices
            for action in ACTIONS
        )
        if bool(context.ego_on_auxiliary) and merge_cmd != 0 and safe_merge and merge_legal:
            return RuleDecision(self._action(merge_cmd, accel_cmd, context), "safe_gap_merge")
        if bool(context.ego_on_auxiliary) and float(context.distance_to_taper) < float(
            self.settings.get("deadline_distance", 120.0)
        ):
            accel_cmd = min(accel_cmd, -1)
            return RuleDecision(self._action(0, accel_cmd, context), "deadline_wait_safe_gap")
        return RuleDecision(self._action(0, accel_cmd, context), "idm_follow")

    def _safe_merge(self, context: RuleControlContext) -> bool:
        ego = context.ego
        if ego is None:
            return False
        front_gap = float(context.target_front_gap)
        rear_gap = float(context.target_rear_gap)
        if front_gap < float(self.settings.get("merge_min_front_gap", 8.0)):
            return False
        if rear_gap < float(self.settings.get("merge_min_rear_gap", 8.0)):
            return False
        front_ttc = float(context.target_front_ttc)
        rear_ttc = float(context.target_rear_ttc)
        if front_ttc < float(self.settings.get("merge_front_ttc_min", 3.0)):
            return False
        if rear_ttc < float(self.settings.get("merge_rear_ttc_min", 3.0)):
            return False
        for vehicle in (context.target_front, context.target_rear):
            if vehicle is not None and bbox_gap(ego, vehicle) <= 0.0:
                return False
        return True

    @staticmethod
    def _ttc(gap: float, closing_speed: float) -> float:
        if closing_speed <= 1.0e-6:
            return INF_TTC
        return max(0.0, gap) / closing_speed

    def _idm_accel(
        self,
        ego: Any,
        front: Any | None,
        lane_speed_limit: float | None,
        front_gap: float,
    ) -> float:
        desired_speed = float(lane_speed_limit or 25.0)
        max_accel = float(self.settings.get("idm_max_acceleration", 1.5))
        comfortable_decel = float(self.settings.get("idm_comfortable_deceleration", 2.0))
        min_gap = float(self.settings.get("idm_min_gap", 2.0))
        headway = float(self.settings.get("idm_time_headway", 1.5))
        free = max_accel * (1.0 - (float(ego.speed) / max(desired_speed, 1.0e-6)) ** 4)
        if front is None:
            return free
        gap = max(0.1, float(front_gap))
        closing = float(ego.speed - front.speed)
        desired_gap = min_gap + max(
            0.0,
            float(ego.speed) * headway + float(ego.speed) * closing / (2.0 * (max_accel * comfortable_decel) ** 0.5),
        )
        return free - max_accel * (desired_gap / gap) ** 2

    @staticmethod
    def _action(lateral_cmd: int, accel_cmd: int, context: RuleControlContext) -> int:
        for action in ACTIONS:
            if (
                action.lateral_cmd == int(lateral_cmd)
                and action.accel_cmd == int(accel_cmd)
                and action.index in context.legal_action_indices
            ):
                return int(action.index)
        for action in ACTIONS:
            if (
                action.lateral_cmd == 0
                and action.accel_cmd == int(accel_cmd)
                and action.index in context.legal_action_indices
            ):
                return int(action.index)
        legal_keep = [
            action.index
            for action in ACTIONS
            if action.lateral_cmd == 0 and action.index in context.legal_action_indices
        ]
        if legal_keep:
            return int(legal_keep[0])
        return int(next(action.index for action in ACTIONS if action.name == "keep_decelerate"))
