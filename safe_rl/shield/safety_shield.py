from __future__ import annotations

from typing import Any

from safe_rl.risk.merge_local import candidate_legality_counts, is_candidate_legal
from safe_rl.risk.candidate_risk_ranker import CandidateRiskRanker
from safe_rl.risk.risk_module import RiskModuleWrapper, RiskPrediction
from safe_rl.shield.fallback_policy import FallbackPolicy
from safe_rl.sim.action_space import ACTIONS, CandidateAction


class SafetyShield:
    def __init__(self, config: Any, risk_model: RiskModuleWrapper | None = None):
        self.config = config
        self.enabled = bool(config.shield.enabled)
        self.ranker = CandidateRiskRanker(config, risk_model)
        self.fallback_policy = FallbackPolicy()
        self._emergency_saturated_count = 0

    def reset_episode_state(self) -> None:
        self._emergency_saturated_count = 0

    def evaluate_candidate(self, action: CandidateAction, context: dict[str, Any]) -> dict[str, Any]:
        """Evaluate one candidate without mutating Shield episode state."""
        candidate_legal = bool(is_candidate_legal(action, context))
        prediction = self.ranker.risk_model.predict(action, context)
        risk_threshold = float(self.config.shield.risk_threshold)
        uncertainty_threshold = float(self.config.shield.uncertainty_threshold)
        if not candidate_legal:
            veto_reason = "candidate_illegal"
        elif float(prediction.risk_score) >= risk_threshold:
            veto_reason = "risk_score"
        elif float(prediction.risk_uncertainty) >= uncertainty_threshold:
            veto_reason = "risk_uncertainty"
        else:
            veto_reason = ""
        return {
            "candidate_legal": candidate_legal,
            "risk_score": float(prediction.risk_score),
            "risk_uncertainty": float(prediction.risk_uncertainty),
            "safety_pass": not veto_reason,
            "veto_reason": veto_reason,
        }

    def select_action(self, raw_action: CandidateAction, context: dict[str, Any]) -> tuple[CandidateAction, dict[str, Any]]:
        raw_prediction = self.ranker.risk_model.predict(raw_action, context)
        raw_legal = is_candidate_legal(raw_action, context)
        counts = candidate_legality_counts(context)
        ranked = self.ranker.rank(raw_action, context)
        best_candidate = ranked[0] if ranked else None
        best_prediction = best_candidate[1] if best_candidate is not None else raw_prediction
        activation_threshold = float(
            self.config.shield.get("activation_risk_threshold", self.config.shield.risk_threshold)
        )
        emergency_triggered, emergency_reason = self._emergency_trigger(
            raw_prediction,
            best_prediction,
            context,
        )
        if raw_legal and raw_prediction.risk_score < activation_threshold and not emergency_triggered:
            self._reset_saturated_count()
            return raw_action, self._record(
                raw_action,
                raw_action,
                raw_prediction,
                raw_prediction,
                "raw_safe",
                False,
                raw_legal,
                raw_legal,
                counts,
                best_candidate,
                emergency_trigger=False,
        )
        if raw_legal and raw_prediction.risk_uncertainty >= float(self.config.shield.uncertainty_threshold) and not emergency_triggered:
            self._reset_saturated_count()
            return raw_action, self._record(
                raw_action,
                raw_action,
                raw_prediction,
                raw_prediction,
                "raw_tolerated",
                False,
                raw_legal,
                raw_legal,
                counts,
                best_candidate,
                emergency_trigger=False,
            )

        margin = float(self.config.shield.get("replacement_margin", 0.15))
        for candidate, prediction, _score in ranked:
            if candidate.index == raw_action.index:
                continue
            improves_enough = (not raw_legal) or prediction.risk_score <= raw_prediction.risk_score - margin
            if improves_enough and self._safe(prediction):
                self._reset_saturated_count()
                return candidate, self._record(
                    raw_action,
                    candidate,
                    raw_prediction,
                    prediction,
                    "replacement",
                    False,
                    raw_legal,
                    is_candidate_legal(candidate, context),
                    counts,
                    best_candidate,
                    emergency_trigger=False,
                )

        consecutive_triggered, consecutive_reason = self._update_saturated_count(raw_prediction, best_prediction)
        if consecutive_triggered:
            emergency_triggered = True
            emergency_reason = consecutive_reason

        if emergency_triggered:
            emergency_action = self._select_emergency_action(context)
            if emergency_action is not None:
                emergency_prediction = self.ranker.risk_model.predict(emergency_action, context)
                return emergency_action, self._record(
                    raw_action,
                    emergency_action,
                    raw_prediction,
                    emergency_prediction,
                    "emergency_fallback",
                    False,
                    raw_legal,
                    is_candidate_legal(emergency_action, context),
                    counts,
                    best_candidate,
                    emergency_fallback=True,
                    emergency_trigger=True,
                    emergency_reason=emergency_reason,
                )
            return raw_action, self._record(
                raw_action,
                raw_action,
                raw_prediction,
                raw_prediction,
                "emergency_unavailable",
                False,
                raw_legal,
                raw_legal,
                counts,
                best_candidate,
                emergency_trigger=True,
                emergency_reason=emergency_reason,
            )

        if not self._fallback_allowed(context):
            if not raw_legal:
                reason = "raw_illegal"
            else:
                reason = "fallback_disabled" if not bool(self.config.shield.get("allow_fallback", False)) else "raw_tolerated"
            return raw_action, self._record(
                raw_action,
                raw_action,
                raw_prediction,
                raw_prediction,
                reason,
                False,
                raw_legal,
                raw_legal,
                counts,
                best_candidate,
                emergency_trigger=False,
            )

        fallback = self.fallback_policy.select()
        fallback_prediction = self.ranker.risk_model.predict(fallback, context)
        return fallback, self._record(
            raw_action,
            fallback,
            raw_prediction,
            fallback_prediction,
            "fallback",
            True,
            raw_legal,
            is_candidate_legal(fallback, context),
            counts,
            best_candidate,
            emergency_trigger=False,
        )

    def _safe(self, prediction: RiskPrediction) -> bool:
        return (
            prediction.risk_score < float(self.config.shield.risk_threshold)
            and prediction.risk_uncertainty < float(self.config.shield.uncertainty_threshold)
        )

    def _fallback_allowed(self, context: dict[str, Any]) -> bool:
        if not bool(self.config.shield.get("allow_fallback", False)):
            return False
        metrics = context.get("current_metrics")
        if metrics is None:
            return False
        min_ttc = float(getattr(metrics, "min_ttc", 1.0e6))
        min_distance = float(getattr(metrics, "min_distance", 1.0e6))
        return (
            min_ttc < float(self.config.shield.get("fallback_min_ttc", 0.30))
            or min_distance < float(self.config.shield.get("fallback_min_distance", 0.75))
        )

    def _emergency_trigger(
        self,
        raw_prediction: RiskPrediction,
        best_prediction: RiskPrediction,
        context: dict[str, Any],
    ) -> tuple[bool, str]:
        if not bool(self.config.shield.get("emergency_fallback_enabled", True)):
            return False, ""
        metrics = context.get("current_metrics")
        if metrics is None:
            return False, ""
        min_ttc = float(getattr(metrics, "min_ttc", 1.0e6))
        min_distance = float(getattr(metrics, "min_distance", 1.0e6))
        if min_ttc <= float(self.config.shield.get("emergency_min_ttc", 0.30)):
            return True, "min_ttc"
        if min_distance <= float(self.config.shield.get("emergency_min_distance", 1.0)):
            return True, "min_distance"

        saturated = float(self.config.shield.get("emergency_saturated_risk_threshold", 0.99))
        in_watch_zone = (
            min_ttc <= float(self.config.shield.get("emergency_watch_min_ttc", 0.75))
            or min_distance <= float(self.config.shield.get("emergency_watch_min_distance", 2.0))
        )
        risks_saturated = (
            float(raw_prediction.risk_score) >= saturated
            and float(best_prediction.risk_score) >= saturated
        )
        if in_watch_zone and risks_saturated:
            return True, "saturated_risk_watch_zone"
        return False, ""

    def _required_saturated_steps(self) -> int:
        return max(1, int(self.config.shield.get("emergency_saturated_consecutive_steps", 2)))

    def _risks_saturated(self, raw_prediction: RiskPrediction, best_prediction: RiskPrediction) -> bool:
        saturated = float(self.config.shield.get("emergency_saturated_risk_threshold", 0.99))
        return (
            float(raw_prediction.risk_score) >= saturated
            and float(best_prediction.risk_score) >= saturated
        )

    def _reset_saturated_count(self) -> None:
        self._emergency_saturated_count = 0

    def _update_saturated_count(
        self,
        raw_prediction: RiskPrediction,
        best_prediction: RiskPrediction,
    ) -> tuple[bool, str]:
        if (
            not bool(self.config.shield.get("emergency_fallback_enabled", True))
            or not bool(self.config.shield.get("emergency_saturated_consecutive_enabled", False))
        ):
            self._reset_saturated_count()
            return False, ""
        if not self._risks_saturated(raw_prediction, best_prediction):
            self._reset_saturated_count()
            return False, ""
        self._emergency_saturated_count += 1
        if self._emergency_saturated_count >= self._required_saturated_steps():
            return True, "saturated_risk_consecutive"
        return False, ""

    def _select_emergency_action(self, context: dict[str, Any]) -> CandidateAction | None:
        configured = self.config.shield.get("emergency_actions", ["keep_decelerate", "keep_hold"])
        names = [str(name) for name in configured]
        by_name = {action.name: action for action in ACTIONS}
        candidates: list[CandidateAction] = [by_name[name] for name in names if name in by_name]
        candidates.extend(
            action for action in ACTIONS if action.accel_cmd < 0 and action.name not in set(names)
        )
        for action in candidates:
            if is_candidate_legal(action, context):
                return action
        return None

    def _record(
        self,
        raw_action: CandidateAction,
        final_action: CandidateAction,
        raw_prediction: RiskPrediction,
        final_prediction: RiskPrediction,
        reason: str,
        fallback: bool,
        raw_candidate_legal: bool,
        final_candidate_legal: bool,
        candidate_counts: dict[str, int],
        best_candidate: tuple[CandidateAction, RiskPrediction, float] | None = None,
        emergency_fallback: bool = False,
        emergency_trigger: bool = False,
        emergency_reason: str = "",
    ) -> dict[str, Any]:
        best_action, best_prediction, best_score = best_candidate if best_candidate is not None else (None, None, None)
        best_risk = best_prediction.risk_score if best_prediction is not None else raw_prediction.risk_score
        best_uncertainty = (
            best_prediction.risk_uncertainty if best_prediction is not None else raw_prediction.risk_uncertainty
        )
        return {
            "raw_action": raw_action.index,
            "raw_action_name": raw_action.name,
            "final_action": final_action.index,
            "final_action_name": final_action.name,
            "best_candidate_action": best_action.index if best_action is not None else raw_action.index,
            "best_candidate_action_name": best_action.name if best_action is not None else raw_action.name,
            "raw_candidate_legal": bool(raw_candidate_legal),
            "final_candidate_legal": bool(final_candidate_legal),
            "legal_candidate_count": int(candidate_counts.get("legal", 0)),
            "illegal_candidate_count": int(candidate_counts.get("illegal", 0)),
            "replacement_reason": reason,
            "risk_before": raw_prediction.risk_score,
            "risk_after": final_prediction.risk_score,
            "best_candidate_risk": float(best_risk),
            "best_candidate_uncertainty": float(best_uncertainty),
            "best_candidate_score": float(best_score) if best_score is not None else float(raw_prediction.risk_score),
            "replacement_risk_delta": float(raw_prediction.risk_score - final_prediction.risk_score),
            "best_candidate_risk_delta": float(raw_prediction.risk_score - best_risk),
            "uncertainty_before": raw_prediction.risk_uncertainty,
            "uncertainty_after": final_prediction.risk_uncertainty,
            "fallback": fallback,
            "emergency_fallback": bool(emergency_fallback),
            "emergency_trigger": bool(emergency_trigger),
            "emergency_reason": str(emergency_reason),
            "emergency_saturated_count": int(self._emergency_saturated_count),
            "emergency_saturated_required": int(self._required_saturated_steps()),
        }
