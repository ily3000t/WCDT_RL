from __future__ import annotations

from typing import Any

from safe_rl.risk.merge_local import is_candidate_legal
from safe_rl.risk.risk_feature_extractor import candidate_progress_score
from safe_rl.risk.risk_module import RiskModuleWrapper, RiskPrediction
from safe_rl.sim.action_space import ACTIONS, CandidateAction, action_distance


class CandidateRiskRanker:
    def __init__(self, config: Any, risk_model: RiskModuleWrapper | None = None):
        self.config = config
        self.risk_model = risk_model or RiskModuleWrapper(config)

    def rank(self, raw_action: CandidateAction, context: dict[str, Any]) -> list[tuple[CandidateAction, RiskPrediction, float]]:
        weights = self.config.shield.score_weights
        ranked: list[tuple[CandidateAction, RiskPrediction, float]] = []
        for action in ACTIONS:
            if bool(self.config.shield.get("filter_illegal_candidates", True)) and not is_candidate_legal(action, context):
                continue
            prediction = self.risk_model.predict(action, context)
            score = (
                weights.risk * prediction.risk_score
                + weights.uncertainty * prediction.risk_uncertainty
                + weights.action_distance * action_distance(raw_action, action)
                + weights.progress * candidate_progress_score(action)
            )
            ranked.append((action, prediction, float(score)))
        return sorted(ranked, key=lambda item: item[2])
