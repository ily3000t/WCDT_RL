from __future__ import annotations

import time
import inspect
from dataclasses import dataclass
from typing import Any

from safe_rl.accvp.calibration import CalibrationBundle
from safe_rl.accvp.selection import select_viability_action
from safe_rl.accvp.protocol import effective_activation_distance
from safe_rl.sim.action_space import ACTIONS, CandidateAction


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

    def close(self) -> None:
        close = getattr(self.predictor, "close", None)
        if callable(close):
            close()

    def proposed_action(self, raw_action: CandidateAction, decision_index: int) -> CandidateAction:
        """Expose an active ACCVP commitment to the next Shield decision."""

        if self._commitment is None:
            return raw_action
        if int(decision_index) >= self._commitment.expires_decision_index:
            self._commitment = None
            return raw_action
        return self._commitment.action

    def decide(
        self,
        *,
        context: dict[str, Any],
        raw_action: CandidateAction,
        safety_shield_action: CandidateAction,
        safety_shield_replaced: bool,
        shield: Any | None,
        shield_input_action: CandidateAction | None = None,
    ) -> tuple[CandidateAction, dict[str, Any]]:
        decision_index = int(context.get("decision_index", 0))
        debug = self._default_debug(raw_action, safety_shield_action)
        if self.mode == "off" or not bool(self.config.accvp.get("enabled", False)):
            return safety_shield_action, debug
        shield_input_action = shield_input_action or raw_action
        if self._commitment is not None and shield_input_action.index == self._commitment.action.index:
            if safety_shield_replaced:
                self._commitment = None
                debug.update(
                    {
                        "accvp_commitment_cancelled": True,
                        "accvp_commitment_cancel_reason": "shield_veto",
                        "accvp_skip_reason": "commitment_shield_veto",
                    }
                )
                return safety_shield_action, debug
            if decision_index < self._commitment.expires_decision_index:
                debug.update(
                    {
                        "accvp_commitment_active": True,
                        "accvp_replacement": True,
                        "accvp_replacement_reason": "lateral_commitment",
                        "accvp_selected_action": int(self._commitment.action.index),
                        "accvp_selected_action_name": str(self._commitment.action.name),
                        "accvp_lane_change_duration_s": max(
                            float(self.config.scenario.step_length),
                            (self._commitment.expires_decision_index - decision_index)
                            * int(self.config.scenario.control_interval_steps)
                            * float(self.config.scenario.step_length),
                        ),
                    }
                )
                return self._commitment.action, debug
            self._commitment = None
        if self.mode == "viability_branch" and safety_shield_replaced:
            debug["accvp_skip_reason"] = "shield_replaced_raw"
            return safety_shield_action, debug
        if self.mode == "viability_branch" and not self._deadline_auxiliary(context):
            debug["accvp_skip_reason"] = "outside_deadline_window"
            return safety_shield_action, debug
        started = time.perf_counter()
        try:
            legal_actions = self._legal_actions(context)
            remaining = float(self.config.accvp.max_decision_latency_s) - (time.perf_counter() - started)
            if remaining <= 0.0:
                raise TimeoutError("ACCVP action legality exceeded the control budget")
            scorer = self.predictor.score_candidates
            if "timeout_s" in inspect.signature(scorer).parameters:
                raw_scores = scorer(context, legal_actions, timeout_s=remaining)
            else:
                raw_scores = scorer(context, legal_actions)
            scored = self._apply_calibration(raw_scores)
            self._attach_secondary_safety(scored, context, shield)
            decision = select_viability_action(scored, raw_action_id=raw_action.index, thresholds=self._thresholds())
            debug.update(self._shadow_debug(scored, raw_action))
            debug.update(self._selection_debug(scored, decision))
            elapsed = time.perf_counter() - started
            debug["decision_latency_s"] = elapsed
            if elapsed > float(self.config.accvp.max_decision_latency_s):
                debug["accvp_bypass_reason"] = "timeout"
                return safety_shield_action, debug
        except TimeoutError:
            debug["accvp_bypass_reason"] = "timeout"
            return safety_shield_action, debug
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
        if bool(decision["raw_feasible"]):
            return safety_shield_action, debug
        selected_row = decision["selected"]
        if selected_row is None:
            debug["accvp_no_feasible_action"] = True
            return safety_shield_action, debug
        selected = next(action for action in ACTIONS if action.index == int(selected_row["action_id"]))
        debug.update(
            {
                "accvp_replacement": True,
                "accvp_replacement_reason": str(decision["reason"]),
                "accvp_selected_action": int(selected.index),
                "accvp_selected_action_name": str(selected.name),
                "selected_pU_proxy_collision": float(selected_row["pU_proxy_collision"]),
                "selected_pU_safety_violation": float(selected_row["pU_safety_violation"]),
                "selected_pL_merge_before_taper": float(selected_row["pL_merge_before_taper"]),
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
            "accvp_skip_reason": "",
            "accvp_no_feasible_action": False,
            "accvp_commitment_started": False,
            "accvp_commitment_cancelled": False,
            "accvp_commitment_active": False,
            "candidate_set_available": False,
            "raw_feasible": False,
            "decision_latency_s": 0.0,
            "accvp_activation_distance_m": effective_activation_distance(self.config),
            "raw_action": int(raw.index),
            "safety_shield_action": int(shield_action.index),
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

    def _attach_secondary_safety(self, scores: list[dict[str, Any]], context: dict[str, Any], shield: Any | None) -> None:
        by_index = {action.index: action for action in ACTIONS}
        for score in scores:
            action = by_index.get(int(score.get("action_id", -1)))
            if action is None:
                score["candidate_legal"] = False
                score["secondary_safety_pass"] = False
                continue
            check = shield.evaluate_candidate(action, context) if shield is not None else {"candidate_legal": True, "safety_pass": True}
            score["candidate_legal"] = bool(check.get("candidate_legal", True))
            score["secondary_safety_pass"] = bool(check.get("safety_pass", False))
            score["secondary_risk_score"] = float(check.get("risk_score", 0.0))
            score["secondary_risk_uncertainty"] = float(check.get("risk_uncertainty", 0.0))
            score["secondary_veto_reason"] = str(check.get("veto_reason", ""))

    def _thresholds(self) -> dict[str, float]:
        return {
            "proxy_collision_upper_bound": float(self.config.accvp.proxy_collision_upper_bound),
            "safety_violation_upper_bound": float(self.config.accvp.safety_violation_upper_bound),
            "merge_viability_lower_bound": float(self.config.accvp.merge_viability_lower_bound),
        }

    def _deadline_auxiliary(self, context: dict[str, Any]) -> bool:
        local = context.get("merge_local")
        return bool(
            local is not None
            and local.ego_on_auxiliary
            and 0.0 < float(local.merge_distance) <= effective_activation_distance(self.config)
        )

    @staticmethod
    def _shadow_debug(scores: list[dict[str, Any]], raw_action: CandidateAction) -> dict[str, Any]:
        raw = next((row for row in scores if int(row.get("action_id", -1)) == raw_action.index), None)
        return {
            "accvp_shadow_scored_actions": len(scores),
            "accvp_shadow_raw_p_proxy_collision": None if raw is None else float(raw["p_proxy_collision"]),
            "accvp_shadow_raw_p_safety_violation": None if raw is None else float(raw["p_safety_violation"]),
            "accvp_shadow_raw_p_merge_before_taper": None if raw is None else float(raw["p_merge_before_taper"]),
        }

    @staticmethod
    def _selection_debug(scores: list[dict[str, Any]], decision: dict[str, Any]) -> dict[str, Any]:
        selected = decision.get("selected")
        return {
            "candidate_set_available": bool(decision.get("candidate_set_available", False)),
            "raw_feasible": bool(decision.get("raw_feasible", False)),
            "accepted_action_ids": [int(row["action_id"]) for row in decision.get("accepted", [])],
            "accvp_shadow_recommended_action": None if selected is None else int(selected["action_id"]),
            "accvp_shadow_candidates": [
                {
                    "action_id": int(row["action_id"]),
                    "p_proxy_collision": float(row["p_proxy_collision"]),
                    "p_safety_violation": float(row["p_safety_violation"]),
                    "p_merge_before_taper": float(row["p_merge_before_taper"]),
                    "pU_proxy_collision": float(row["pU_proxy_collision"]),
                    "pU_safety_violation": float(row["pU_safety_violation"]),
                    "pL_merge_before_taper": float(row["pL_merge_before_taper"]),
                    "secondary_safety_pass": bool(row.get("secondary_safety_pass", False)),
                    "candidate_legal": bool(row.get("candidate_legal", False)),
                    "gate_pass": bool(row.get("accvp_gate_pass", False)),
                    "ensemble_disagreement": float(row.get("ensemble_disagreement", 0.0)),
                }
                for row in scores
            ],
        }
