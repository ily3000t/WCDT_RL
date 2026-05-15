from __future__ import annotations

from typing import Any

from safe_rl.risk.candidate_risk_ranker import CandidateRiskRanker
from safe_rl.risk.risk_module import RiskModuleWrapper, RiskPrediction
from safe_rl.shield.fallback_policy import FallbackPolicy
from safe_rl.sim.action_space import CandidateAction


class SafetyShield:
    def __init__(self, config: Any, risk_model: RiskModuleWrapper | None = None):
        self.config = config
        self.enabled = bool(config.shield.enabled)
        self.ranker = CandidateRiskRanker(config, risk_model)
        self.fallback_policy = FallbackPolicy()

    def select_action(self, raw_action: CandidateAction, context: dict[str, Any]) -> tuple[CandidateAction, dict[str, Any]]:
        raw_prediction = self.ranker.risk_model.predict(raw_action, context)
        if self._safe(raw_prediction):
            return raw_action, self._record(raw_action, raw_action, raw_prediction, raw_prediction, "raw_safe", False)

        ranked = self.ranker.rank(raw_action, context)
        for candidate, prediction, _score in ranked:
            if self._safe(prediction):
                return candidate, self._record(raw_action, candidate, raw_prediction, prediction, "replacement", False)

        fallback = self.fallback_policy.select()
        fallback_prediction = self.ranker.risk_model.predict(fallback, context)
        return fallback, self._record(raw_action, fallback, raw_prediction, fallback_prediction, "fallback", True)

    def _safe(self, prediction: RiskPrediction) -> bool:
        return (
            prediction.risk_score < float(self.config.shield.risk_threshold)
            and prediction.risk_uncertainty < float(self.config.shield.uncertainty_threshold)
        )

    def _record(
        self,
        raw_action: CandidateAction,
        final_action: CandidateAction,
        raw_prediction: RiskPrediction,
        final_prediction: RiskPrediction,
        reason: str,
        fallback: bool,
    ) -> dict[str, Any]:
        return {
            "raw_action": raw_action.index,
            "raw_action_name": raw_action.name,
            "final_action": final_action.index,
            "final_action_name": final_action.name,
            "replacement_reason": reason,
            "risk_before": raw_prediction.risk_score,
            "risk_after": final_prediction.risk_score,
            "uncertainty_before": raw_prediction.risk_uncertainty,
            "uncertainty_after": final_prediction.risk_uncertainty,
            "fallback": fallback,
        }
