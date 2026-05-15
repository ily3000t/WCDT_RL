from __future__ import annotations

from safe_rl.sim.action_space import FALLBACK_ACTION, CandidateAction


class FallbackPolicy:
    def select(self) -> CandidateAction:
        return FALLBACK_ACTION
