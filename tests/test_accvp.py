from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from safe_rl.accvp.candidate_plan import ACCVP_COMMITMENT_PROFILE, build_commitment_plan
from safe_rl.accvp.calibration import CalibrationBundle, OneSidedBinnedCalibrator, selected_action_metrics
from safe_rl.accvp.controller import ACCVPController
from safe_rl.accvp.dataset import build_split_manifest
from safe_rl.accvp.oracle import counterfactual_oracle_report
from safe_rl.accvp.root_context import RootContext
from safe_rl.accvp.snapshot_store import CounterfactualSnapshotStore
from safe_rl.sim.action_space import decode_action
from safe_rl.sim.types import VehicleState
from safe_rl.utils.config import clone_with_overrides, load_config


class _Predictor:
    def __init__(self, scores):
        self.scores = scores
        self.calls = 0

    def score_candidates(self, _context, _actions):
        self.calls += 1
        return [dict(score) for score in self.scores]


class _Shield:
    def __init__(self, safe: bool = True):
        self.safe = safe

    def evaluate_candidate(self, _action, _context):
        return {"safety_pass": self.safe, "veto_reason": "risk_score" if not self.safe else ""}


def _cfg(mode: str):
    base = load_config()
    return clone_with_overrides(
        base,
        {
            "accvp": {
                "enabled": True,
                "mode": mode,
                "deadline_distance": 200.0,
                "proxy_collision_upper_bound": 0.2,
                "safety_violation_upper_bound": 0.2,
                "merge_viability_lower_bound": 0.5,
                "max_decision_latency_s": 1.0,
            }
        },
    )


def _context(decision: int = 0):
    return {
        "decision_index": decision,
        "merge_local": SimpleNamespace(ego_on_auxiliary=True, merge_distance=50.0),
    }


def _score(action_id: int, *, risk: float = 0.1, viability: float = 0.8):
    return {
        "action_id": action_id,
        "p_proxy_collision": risk,
        "p_safety_violation": risk,
        "p_taper_miss": 1.0 - viability,
        "p_merge_before_taper": viability,
        "target_lane_entry_time_s": 1.0,
    }


def test_raw_feasible_is_retained():
    raw = decode_action(4)
    controller = ACCVPController(_cfg("viability_branch"), _Predictor([_score(4), _score(7)]))
    action, debug = controller.decide(
        context=_context(), raw_action=raw, safety_shield_action=raw, safety_shield_replaced=False, shield=_Shield()
    )
    assert action == raw
    assert debug["raw_feasible"] is True
    assert debug["accvp_replacement"] is False


def test_only_raw_infeasible_allows_accvp_replacement_and_commitment():
    raw = decode_action(4)
    merge = decode_action(7)
    controller = ACCVPController(_cfg("viability_branch"), _Predictor([_score(4, risk=0.9), _score(7)]))
    action, debug = controller.decide(
        context=_context(), raw_action=raw, safety_shield_action=raw, safety_shield_replaced=False, shield=_Shield()
    )
    assert action == merge
    assert debug["accvp_replacement"] is True
    assert debug["accvp_commitment_started"] is True
    continued, continued_debug = controller.decide(
        context=_context(1), raw_action=raw, safety_shield_action=raw, safety_shield_replaced=False, shield=_Shield()
    )
    assert continued == merge
    assert continued_debug["accvp_commitment_active"] is True


def test_shield_veto_cancels_active_commitment():
    raw = decode_action(4)
    merge = decode_action(7)
    controller = ACCVPController(_cfg("viability_branch"), _Predictor([_score(4, risk=0.9), _score(7)]))
    controller.decide(context=_context(), raw_action=raw, safety_shield_action=raw, safety_shield_replaced=False, shield=_Shield())
    action, debug = controller.decide(
        context=_context(1), raw_action=raw, safety_shield_action=raw, safety_shield_replaced=False, shield=_Shield(False)
    )
    assert action == raw
    assert debug["accvp_commitment_cancelled"] is True
    assert debug["accvp_bypass_reason"] == "commitment_shield_veto"
    assert merge != action


def test_shadow_never_replaces():
    raw = decode_action(4)
    controller = ACCVPController(_cfg("shadow"), _Predictor([_score(4, risk=0.9), _score(7)]))
    action, debug = controller.decide(
        context=_context(), raw_action=raw, safety_shield_action=raw, safety_shield_replaced=False, shield=_Shield()
    )
    assert action == raw
    assert debug["accvp_replacement"] is False
    assert debug["accvp_shadow_scored_actions"] == 2


def test_candidate_plan_is_versioned_and_has_fixed_speed_continuation():
    ego = VehicleState("ego", 0.0, 0.0, 0.0, 10.0, 1, "lane", 0.0, "main_aux")
    plan = build_commitment_plan(ego, decode_action(7), step_length=0.1, horizon_steps=20)
    assert plan.profile == ACCVP_COMMITMENT_PROFILE
    assert plan.states.shape == (20, 5)
    assert np.isclose(plan.states[5, 3], plan.states[-1, 3])


def test_snapshot_is_deleted_only_after_all_expected_branches_complete(tmp_path: Path):
    snapshot = tmp_path / "root.xml"
    snapshot.write_text("snapshot", encoding="utf-8")
    root = RootContext(
        metadata={"root_id": "root", "snapshot_path": str(snapshot), "root_ego": {}, "history_frames": []},
        tensors={"history_features": np.zeros((1, 1, 1, 10), dtype=np.float32)},
    )
    store = CounterfactualSnapshotStore(tmp_path / "data")
    store.write_root(root, [0, 1])
    base = {
        "counterfactual_schema_version": 1,
        "root_id": "root",
        "snapshot_sha256": "hash",
        "candidate_plan_profile": ACCVP_COMMITMENT_PROFILE,
        "event_observed": False,
        "censor_time": 8.0,
        "censor_reason": "horizon_elapsed",
        "viability_observation_status": "censored",
        "branch_status": "completed",
    }
    store.write_branch({**base, "branch_id": "root_action0", "action_id": 0})
    assert store.finalise_root_if_complete("root") is False
    assert snapshot.exists()
    store.write_branch({**base, "branch_id": "root_action1", "action_id": 1})
    assert store.finalise_root_if_complete("root") is True
    assert not snapshot.exists()


def test_calibration_and_selected_action_metrics_are_decision_level():
    calibrator = OneSidedBinnedCalibrator.fit([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1], bins=2)
    bundle = CalibrationBundle(calibrator, calibrator, calibrator, {"split": "calibration"})
    bounds = bundle.score(
        {"p_proxy_collision": [0.1], "p_safety_violation": [0.9], "p_merge_before_taper": [0.8]}
    )
    assert set(bounds) == {"pU_proxy_collision", "pU_safety_violation", "pL_merge_before_taper"}
    metrics = selected_action_metrics(
        [
            {
                "root_id": "a",
                "selected": True,
                "candidate_set_available": True,
                "p_proxy_collision": 0.2,
                "proxy_collision": 0.0,
                "p_merge_before_taper": 0.8,
                "merge_before_taper": 1.0,
            }
        ]
    )
    assert metrics["selected_count"] == 1.0
    assert metrics["candidate_set_availability"] == 1.0


def test_split_keeps_all_roots_of_same_episode_seed_together(tmp_path: Path):
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    roots = [
        {"root_id": "a", "episode_seed": 1, "root_source": "mixed", "traffic_profile": "safe", "deadline_bin": "deadline", "complete": True},
        {"root_id": "b", "episode_seed": 1, "root_source": "mixed", "traffic_profile": "safe", "deadline_bin": "deadline", "complete": True},
        {"root_id": "c", "episode_seed": 2, "root_source": "rule", "traffic_profile": "hard", "deadline_bin": "pre_deadline", "complete": True},
    ]
    (manifests / "roots.jsonl").write_text("".join(__import__("json").dumps(row) + "\n" for row in roots), encoding="utf-8")
    rows = build_split_manifest(tmp_path, seed=7)
    assignments = {row["root_id"]: row["split"] for row in rows}
    assert assignments["a"] == assignments["b"]


def test_oracle_requires_safe_viable_counterfactual_for_each_failure_seed(tmp_path: Path):
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    roots = [
        {"root_id": "seed2", "episode_seed": 2, "deadline_bin": "deadline", "complete": True},
        {"root_id": "seed5", "episode_seed": 5, "deadline_bin": "deadline", "complete": True},
    ]
    branches = [
        {"root_id": "seed2", "branch_status": "completed", "action_id": 7, "proxy_collision_within_horizon": False, "safety_violation_within_horizon": False, "merge_before_taper_observed": True},
        {"root_id": "seed5", "branch_status": "completed", "action_id": 4, "proxy_collision_within_horizon": True, "safety_violation_within_horizon": True, "merge_before_taper_observed": False},
    ]
    (manifests / "roots.jsonl").write_text("".join(__import__("json").dumps(row) + "\n" for row in roots), encoding="utf-8")
    (manifests / "branches.jsonl").write_text("".join(__import__("json").dumps(row) + "\n" for row in branches), encoding="utf-8")
    report = counterfactual_oracle_report(tmp_path, required_seeds=[2, 5])
    assert report["required_failure_seed_results"]["2"]["go"] is True
    assert report["required_failure_seed_results"]["5"]["go"] is False
    assert report["go_for_training"] is False
