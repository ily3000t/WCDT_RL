from __future__ import annotations

from typing import Any

import numpy as np

from safe_rl.risk.merge_local import evaluate_candidate_action_risk, rollout_ego
from safe_rl.sim.action_space import CandidateAction


def candidate_progress_score(action: CandidateAction) -> float:
    return float(action.accel_cmd + 1) / 2.0


def extract_candidate_features(action: CandidateAction, context: dict[str, Any]) -> np.ndarray:
    return evaluate_candidate_action_risk(action, context).features.astype(np.float32)
