from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from safe_rl.accvp.calibration import CalibrationBundle
from safe_rl.sim.action_space import ACTIONS, CandidateAction, action_distance


@dataclass
class LateralCommitment:
    action: CandidateAction
    expires_decision_index: int


class ACCVPController:
    """Post-Shield ACCVP controller with raw-action retention by construction."""

    def __init__(self, config: Any, predictor: Any, calibration: CalibrationBundle | None = None):
        self.config = config
        self.predictor = predictor
        self.calibration = calibration
        self._commitment: LateralCommitment | None = None

    @property
    def mode(self) -> str:
        return str(self.config.accvp.get("mode", "off")).strip().lower()

    def reset_episode_state(self) -> None:
        self._commitment = None

    def decide(
        self,
        *,
        context: dict[str, Any],
        raw_action: CandidateAction,
        safety_shield_action: CandidateAction,
        safety_shield_replaced: bool,
        shield: Any | None,
    ) -> tuple[CandidateAction, dict[str, Any]]:
        decision_index = int(context.get("decision_index", 0))
        debug = self._default_debug(raw_action, safety_shield_action)
        if self.mode == "off" or not bool(self.config.accvp.get("enabled", False)):
            return safety_shield_action, debug
        commitment_result = self._continue_commitment(context, safety_shield_action, shield, decision_index)
        if commitment_result is not None:
            action, commitment_debug = commitment_result
            debug.update(commitment_debug)
            return action, debug
        started = time.perf_counter()
        try:
            legal_actions = self._legal_actions(context)
            raw_scores = self.predictor.score_candidates(context, legal_actions)
            elapsed = time.perf_counter() - started
            debug["decision_latency_s"] = elapsed
            if elapsed > float(self.config.accvp.max_decision_latency_s):
                debug["accvp_bypass_reason"] = "timeout"
                return safety_shield_action, debug
            scored = self._apply_calibration(raw_scores)
            debug.update(self._shadow_debug(scored, raw_action))
        except ValueError:
            debug["accvp_bypass_reason"] = "invalid_bundle"
            return safety_shield_action, debug
        except Exception:
            debug["accvp_bypass_reason"] = "model_error"
            return safety_shield_action, debug

        if self.mode == "shadow":
            return safety_shield_action, debug
        if self.mode != "viability_branch":
            debug["accvp_bypass_reason"] = "invalid_bundle"
            return safety_shield_action, debug
        if safety_shield_replaced:
            debug["accvp_bypass_reason"] = "shield_replaced_raw"
            return safety_shield_action, debug
        if not self._deadline_auxiliary(context):
            debug["accvp_bypass_reason"] = "outside_deadline_window"
            return safety_shield_action, debug
        candidates = self._safe_viable_candidates(scored, context, shield)
        debug["candidate_set_available"] = bool(candidates)
        debug["accepted_action_ids"] = [int(row["action"].index) for row in candidates]
        raw_candidate = next((row for row in candidates if row["action"].index == raw_action.index), None)
        if raw_candidate is not None:
            # This invariant prevents the branch from becoming Full Ranking.
            debug["raw_feasible"] = True
            return safety_shield_action, debug
        debug["raw_feasible"] = False
        if not candidates:
            debug["accvp_no_feasible_action"] = True
            return safety_shield_action, debug
        best = min(
            candidates,
            key=lambda row: (
                -float(row["pL_merge_before_taper"]),
                float(row["pU_safety_violation"]),
                float(row.get("target_lane_entry_time_s", float("inf"))),
                abs(float(row["action"].accel_cmd)),
                action_distance(row["action"], raw_action),
            ),
        )
        selected = best["action"]
        debug.update(
            {
                "accvp_replacement": True,
                "accvp_replacement_reason": "raw_infeasible_viable_candidate",
                "accvp_selected_action": int(selected.index),
                "accvp_selected_action_name": str(selected.name),
                "selected_pU_proxy_collision": float(best["pU_proxy_collision"]),
                "selected_pU_safety_violation": float(best["pU_safety_violation"]),
                "selected_pL_merge_before_taper": float(best["pL_merge_before_taper"]),
            }
        )
        if selected.lateral_cmd != 0:
            interval = max(1, int(self.config.scenario.control_interval_steps))
            steps = max(1, round(float(self.config.accvp.lateral_commitment_s) / (interval * float(self.config.scenario.step_length))))
            self._commitment = LateralCommitment(selected, decision_index + steps)
            debug["accvp_commitment_started"] = True
            debug["accvp_lane_change_duration_s"] = float(self.config.accvp.lateral_commitment_s)
        return selected, debug

    def _default_debug(self, raw: CandidateAction, shield_action: CandidateAction) -> dict[str, Any]:
        return {
            "accvp_mode": self.mode,
            "accvp_replacement": False,
            "accvp_replacement_reason": "",
            "accvp_selected_action": int(shield_action.index),
            "accvp_selected_action_name": str(shield_action.name),
            "accvp_bypass_reason": "",
            "accvp_no_feasible_action": False,
            "accvp_commitment_started": False,
            "accvp_commitment_cancelled": False,
            "accvp_commitment_active": False,
            "candidate_set_available": False,
            "raw_feasible": False,
            "decision_latency_s": 0.0,
            "raw_action": int(raw.index),
            "safety_shield_action": int(shield_action.index),
        }

    def _continue_commitment(
        self,
        context: dict[str, Any],
        safety_shield_action: CandidateAction,
        shield: Any | None,
        decision_index: int,
    ) -> tuple[CandidateAction, dict[str, Any]] | None:
        commitment = self._commitment
        if commitment is None:
            return None
        if decision_index >= commitment.expires_decision_index:
            self._commitment = None
            return None
        check = shield.evaluate_candidate(commitment.action, context) if shield is not None else {"safety_pass": True}
        if not bool(check.get("safety_pass", False)):
            self._commitment = None
            return safety_shield_action, {
                "accvp_commitment_cancelled": True,
                "accvp_commitment_cancel_reason": str(check.get("veto_reason", "shield_veto")),
                "accvp_bypass_reason": "commitment_shield_veto",
                "accvp_replacement": False,
                "accvp_commitment_active": False,
            }
        return commitment.action, {
            "accvp_commitment_active": True,
            "accvp_replacement": True,
            "accvp_replacement_reason": "lateral_commitment",
            "accvp_selected_action": int(commitment.action.index),
            "accvp_selected_action_name": str(commitment.action.name),
            "accvp_lane_change_duration_s": max(
                float(self.config.scenario.step_length),
                (commitment.expires_decision_index - decision_index)
                * int(self.config.scenario.control_interval_steps)
                * float(self.config.scenario.step_length),
            ),
        }

    def _legal_actions(self, context: dict[str, Any]) -> list[CandidateAction]:
        from safe_rl.risk.merge_local import is_candidate_legal

        return [action for action in ACTIONS if is_candidate_legal(action, context)]

    def _apply_calibration(self, scores: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not scores:
            return []
        for score in scores:
            for name in ("p_proxy_collision", "p_safety_violation", "p_merge_before_taper"):
                if name not in score:
                    raise ValueError(f"ACCVP predictor score missing {name}")
        if self.calibration is None:
            for score in scores:
                score["pU_proxy_collision"] = float(score["p_proxy_collision"])
                score["pU_safety_violation"] = float(score["p_safety_violation"])
                score["pL_merge_before_taper"] = float(score["p_merge_before_taper"])
            return scores
        calibrated = self.calibration.score(
            {
                "p_proxy_collision": [row["p_proxy_collision"] for row in scores],
                "p_safety_violation": [row["p_safety_violation"] for row in scores],
                "p_merge_before_taper": [row["p_merge_before_taper"] for row in scores],
            }
        )
        for index, score in enumerate(scores):
            score["pU_proxy_collision"] = float(calibrated["pU_proxy_collision"][index])
            score["pU_safety_violation"] = float(calibrated["pU_safety_violation"][index])
            score["pL_merge_before_taper"] = float(calibrated["pL_merge_before_taper"][index])
        return scores

    def _safe_viable_candidates(self, scores: list[dict[str, Any]], context: dict[str, Any], shield: Any | None) -> list[dict[str, Any]]:
        accepted: list[dict[str, Any]] = []
        by_index = {action.index: action for action in ACTIONS}
        for score in scores:
            action = by_index.get(int(score.get("action_id", -1)))
            if action is None:
                continue
            risk_check = shield.evaluate_candidate(action, context) if shield is not None else {"safety_pass": True}
            if not bool(risk_check.get("safety_pass", False)):
                continue
            if float(score["pU_proxy_collision"]) > float(self.config.accvp.proxy_collision_upper_bound):
                continue
            if float(score["pU_safety_violation"]) > float(self.config.accvp.safety_violation_upper_bound):
                continue
            if float(score["pL_merge_before_taper"]) < float(self.config.accvp.merge_viability_lower_bound):
                continue
            score["action"] = action
            accepted.append(score)
        return accepted

    def _deadline_auxiliary(self, context: dict[str, Any]) -> bool:
        local = context.get("merge_local")
        return bool(local is not None and local.ego_on_auxiliary and float(local.merge_distance) <= float(self.config.accvp.deadline_distance))

    @staticmethod
    def _shadow_debug(scores: list[dict[str, Any]], raw_action: CandidateAction) -> dict[str, Any]:
        raw = next((row for row in scores if int(row.get("action_id", -1)) == raw_action.index), None)
        return {
            "accvp_shadow_scored_actions": len(scores),
            "accvp_shadow_raw_p_proxy_collision": None if raw is None else float(raw["p_proxy_collision"]),
            "accvp_shadow_raw_p_safety_violation": None if raw is None else float(raw["p_safety_violation"]),
            "accvp_shadow_raw_p_merge_before_taper": None if raw is None else float(raw["p_merge_before_taper"]),
        }
