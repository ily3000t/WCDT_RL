from __future__ import annotations

from safe_rl.sim.action_space import ACTIONS, CandidateAction


class CandidateActionGenerator:
    def generate(self) -> tuple[CandidateAction, ...]:
        return ACTIONS
