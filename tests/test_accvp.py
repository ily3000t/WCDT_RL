from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from safe_rl.accvp.candidate_plan import ACCVP_COMMITMENT_PROFILE, build_commitment_plan
from safe_rl.accvp.calibration import CalibrationBundle, OneSidedBinnedCalibrator, selected_action_metrics
from safe_rl.accvp.availability import audit_risk_secondary_false_negatives, diagnose_oracle_availability, model_gate_failure_diagnostics
from safe_rl.accvp.candidate_table_diagnostics import candidate_table_summary
from safe_rl.accvp.controller import ACCVPController
from safe_rl.accvp.dataset import build_split_manifest
from safe_rl.accvp.model import checkpoint_metadata
from safe_rl.accvp.oracle import counterfactual_oracle_report
from safe_rl.accvp.pilot import validate_pilot_dataset
from safe_rl.accvp.protocol import counterfactual_data_contract, data_contract_hash, effective_activation_distance
from safe_rl.accvp.root_context import RootContext
from safe_rl.accvp.runtime import ACCVPRuntimePredictor
from safe_rl.accvp.schema import COUNTERFACTUAL_SCHEMA_VERSION, stable_hash
from safe_rl.accvp.risk_secondary_audit import audit_risk_secondary
from safe_rl.accvp.online_trigger_audit import audit_online_triggers
from safe_rl.accvp.selection import select_viability_action, select_viability_lite_action
from safe_rl.accvp.shards import merge_counterfactual_shards
from safe_rl.accvp.snapshot_store import CounterfactualSnapshotStore
from safe_rl.accvp.viability_lite import evaluate_lite_thresholds, tune_viability_lite_operating_point
from safe_rl.accvp.viability_lite_audit import audit_lite_replacements
from safe_rl.pipeline.accvp_tune_viability_lite import _lite_acceptance_failures
from safe_rl.pipeline.stage1_collect_accvp_jobs import materialise_collection_job, validate_required_pilot
from safe_rl.stage1_counterfactual.collector import _cache_dir, _root_filter_matches, _seed_schedule
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


def test_explicit_activation_distance_overrides_legacy_deadline_without_mutating_it():
    cfg = clone_with_overrides(_cfg("shadow"), {"accvp": {"activation_distance": 240.0}})
    assert effective_activation_distance(cfg) == 240.0
    assert cfg.accvp.deadline_distance == 200.0
    controller = ACCVPController(cfg, _Predictor([_score(4), _score(7)]))
    action, debug = controller.decide(
        context={"decision_index": 0, "merge_local": SimpleNamespace(ego_on_auxiliary=True, merge_distance=220.0)},
        raw_action=decode_action(4),
        safety_shield_action=decode_action(4),
        safety_shield_replaced=False,
        shield=_Shield(),
    )
    assert action == decode_action(4)
    assert debug["accvp_activation_distance_m"] == 240.0


def test_checkpoint_metadata_tracks_counterfactual_schema_v2():
    metadata = checkpoint_metadata(load_config(), warm_start={})
    assert metadata["counterfactual_schema_version"] == COUNTERFACTUAL_SCHEMA_VERSION


def test_viability_branch_rejects_shadow_artifact_manifest(tmp_path: Path):
    manifest = tmp_path / "accvp_v1_shadow_artifact_manifest.json"
    manifest.write_text(
        __import__("json").dumps(
            {
                "artifact_kind": "accvp_v1_shadow_artifact_bundle",
                "deployable_artifact": False,
            }
        ),
        encoding="utf-8",
    )
    cfg = clone_with_overrides(
        load_config(),
        {"accvp": {"enabled": True, "mode": "viability_branch", "artifact_manifest": str(manifest)}},
    )
    predictor = ACCVPRuntimePredictor.__new__(ACCVPRuntimePredictor)
    predictor.config = cfg
    predictor.checkpoint_path = tmp_path / "accvp_v1_predictor.pt"
    with __import__("pytest").raises(ValueError, match="deployable_artifact=true"):
        predictor.validate_artifact_bundle(operating_point=tmp_path / "accvp_v1_operating_point.json")


def test_viability_lite_rejects_plain_shadow_artifact_manifest(tmp_path: Path):
    manifest = tmp_path / "accvp_v1_shadow_artifact_manifest.json"
    manifest.write_text(
        __import__("json").dumps(
            {
                "artifact_kind": "accvp_v1_shadow_artifact_bundle",
                "deployable_artifact": False,
            }
        ),
        encoding="utf-8",
    )
    cfg = clone_with_overrides(
        load_config(),
        {"accvp": {"enabled": True, "mode": "viability_lite", "artifact_manifest": str(manifest)}},
    )
    predictor = ACCVPRuntimePredictor.__new__(ACCVPRuntimePredictor)
    predictor.config = cfg
    predictor.checkpoint_path = tmp_path / "accvp_v1_predictor.pt"
    with __import__("pytest").raises(ValueError, match="viability_lite requires"):
        predictor.validate_artifact_bundle(operating_point=tmp_path / "accvp_v1_lite_operating_point.json")


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
        context=_context(1), raw_action=raw, safety_shield_action=merge, safety_shield_replaced=False, shield=_Shield(), shield_input_action=merge
    )
    assert continued == merge
    assert continued_debug["accvp_commitment_active"] is True


def test_shield_veto_cancels_active_commitment():
    raw = decode_action(4)
    merge = decode_action(7)
    controller = ACCVPController(_cfg("viability_branch"), _Predictor([_score(4, risk=0.9), _score(7)]))
    controller.decide(context=_context(), raw_action=raw, safety_shield_action=raw, safety_shield_replaced=False, shield=_Shield())
    action, debug = controller.decide(
        context=_context(1), raw_action=raw, safety_shield_action=raw, safety_shield_replaced=True, shield=_Shield(False), shield_input_action=merge
    )
    assert action == raw
    assert debug["accvp_commitment_cancelled"] is True
    assert debug["accvp_bypass_reason"] == ""
    assert debug["accvp_skip_reason"] == "commitment_shield_veto"
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


def test_viability_lite_shadow_never_replaces_but_recommends_left():
    raw = decode_action(4)
    controller = ACCVPController(
        _cfg("viability_lite_shadow"),
        _Predictor([_score(4, viability=0.2), _score(7, viability=0.95)]),
    )
    action, debug = controller.decide(
        context=_context(), raw_action=raw, safety_shield_action=raw, safety_shield_replaced=False, shield=_Shield()
    )
    assert action == raw
    assert debug["accvp_replacement"] is False
    assert debug["accvp_shadow_recommended_action"] == 7
    assert debug["accvp_lite_raw_task_feasible"] is False
    assert debug["accvp_lite_best_left_action"] == 7
    assert debug["accvp_lite_p_merge_improvement"] > 0.0


def test_viability_lite_active_replaces_and_starts_commitment():
    raw = decode_action(4)
    merge = decode_action(7)
    controller = ACCVPController(
        _cfg("viability_lite"),
        _Predictor([_score(4, viability=0.2), _score(7, viability=0.95)]),
    )
    action, debug = controller.decide(
        context=_context(), raw_action=raw, safety_shield_action=raw, safety_shield_replaced=False, shield=_Shield()
    )
    assert action == merge
    assert debug["accvp_replacement"] is True
    assert debug["accvp_replacement_reason"] == "raw_task_infeasible_lite_viable_left"
    assert debug["accvp_commitment_started"] is True


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
    store = CounterfactualSnapshotStore(tmp_path / "data", cache_dir=tmp_path / ".cache" / "accvp")
    assert store.snapshots_dir == tmp_path / ".cache" / "accvp" / "snapshots"
    store.write_root(root, [0, 1])
    base = {
        "counterfactual_schema_version": COUNTERFACTUAL_SCHEMA_VERSION,
        "root_id": "root",
        "snapshot_sha256": "hash",
        "candidate_plan_profile": ACCVP_COMMITMENT_PROFILE,
        "accvp_activation_distance_m": 240.0,
        "data_contract_hash": "contract",
        "risk_model_fingerprint": "risk_checkpoint:test",
        "secondary_safety_pass": True,
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
                "p_safety_violation": 0.2,
                "safety_violation": 0.0,
                "p_merge_before_taper": 0.8,
                "merge_before_taper": 1.0,
            }
        ]
    )
    assert metrics["selected_count"] == 1.0
    assert metrics["candidate_set_availability"] == 1.0
    assert "proxy_collision" in metrics
    assert "safety_violation" in metrics


def test_empty_calibration_bin_is_conservative_and_selector_retains_raw():
    calibrator = OneSidedBinnedCalibrator.fit([0.1], [0.0], bins=2)
    assert calibrator.transform_upper([0.9])[0] == 1.0
    assert calibrator.transform_lower([0.9])[0] == 0.0
    thresholds = {
        "proxy_collision_upper_bound": 0.2,
        "safety_violation_upper_bound": 0.2,
        "merge_viability_lower_bound": 0.5,
    }
    decision = select_viability_action(
        [
            {"action_id": 4, "pU_proxy_collision": 0.1, "pU_safety_violation": 0.1, "pL_merge_before_taper": 0.6, "secondary_safety_pass": True},
            {"action_id": 7, "pU_proxy_collision": 0.1, "pU_safety_violation": 0.1, "pL_merge_before_taper": 0.9, "secondary_safety_pass": True},
        ],
        raw_action_id=4,
        thresholds=thresholds,
    )
    assert decision["selected"]["action_id"] == 4
    assert decision["raw_feasible"] is True


def test_selector_replacement_requires_merge_intent():
    thresholds = {
        "proxy_collision_upper_bound": 0.2,
        "safety_violation_upper_bound": 0.2,
        "merge_viability_lower_bound": 0.5,
    }
    decision = select_viability_action(
        [
            {"action_id": 3, "pU_proxy_collision": 0.1, "pU_safety_violation": 0.1, "pL_merge_before_taper": 0.99, "secondary_safety_pass": True},
            {"action_id": 4, "pU_proxy_collision": 0.9, "pU_safety_violation": 0.9, "pL_merge_before_taper": 0.99, "secondary_safety_pass": True},
            {"action_id": 7, "pU_proxy_collision": 0.1, "pU_safety_violation": 0.1, "pL_merge_before_taper": 0.60, "secondary_safety_pass": True},
        ],
        raw_action_id=4,
        thresholds=thresholds,
    )
    assert decision["selected"]["action_id"] == 7
    assert decode_action(decision["selected"]["action_id"]).lateral_cmd > 0
    assert decision["replacement"] is True


def test_selector_keeps_shield_action_when_only_non_merge_actions_pass():
    thresholds = {
        "proxy_collision_upper_bound": 0.2,
        "safety_violation_upper_bound": 0.2,
        "merge_viability_lower_bound": 0.5,
    }
    decision = select_viability_action(
        [
            {"action_id": 1, "pU_proxy_collision": 0.1, "pU_safety_violation": 0.1, "pL_merge_before_taper": 0.99, "secondary_safety_pass": True},
            {"action_id": 3, "pU_proxy_collision": 0.1, "pU_safety_violation": 0.1, "pL_merge_before_taper": 0.99, "secondary_safety_pass": True},
            {"action_id": 4, "pU_proxy_collision": 0.9, "pU_safety_violation": 0.9, "pL_merge_before_taper": 0.99, "secondary_safety_pass": True},
        ],
        raw_action_id=4,
        thresholds=thresholds,
    )
    assert decision["selected"] is None
    assert decision["candidate_set_available"] is False
    assert decision["reason"] == "no_merge_intent_feasible_action"


def _lite_thresholds():
    return {
        "min_p_merge_before_taper": 0.75,
        "min_improvement_over_raw": 0.05,
        "max_target_entry_time_s": 8.0,
        "max_ensemble_disagreement": 0.2,
        "max_secondary_risk_score": 1.0,
    }


def test_viability_lite_retains_raw_when_task_feasible_and_ignores_accvp_safety_head():
    decision = select_viability_lite_action(
        [
            {"action_id": 4, "candidate_legal": True, "secondary_safety_pass": True, "secondary_risk_score": 0.1, "p_merge_before_taper": 0.80, "pU_proxy_collision": 1.0, "pU_safety_violation": 1.0, "target_lane_entry_time_s": 4.0, "ensemble_disagreement": 0.0},
            {"action_id": 7, "candidate_legal": True, "secondary_safety_pass": True, "secondary_risk_score": 0.1, "p_merge_before_taper": 0.95, "target_lane_entry_time_s": 2.0, "ensemble_disagreement": 0.0},
        ],
        raw_action_id=4,
        thresholds=_lite_thresholds(),
    )
    assert decision["selected"]["action_id"] == 4
    assert decision["raw_task_feasible"] is True
    assert decision["replacement"] is False


def test_viability_lite_replaces_only_with_risk_safe_left_action():
    decision = select_viability_lite_action(
        [
            {"action_id": 4, "candidate_legal": True, "secondary_safety_pass": True, "secondary_risk_score": 0.1, "p_merge_before_taper": 0.20, "target_lane_entry_time_s": 8.0, "ensemble_disagreement": 0.0},
            {"action_id": 7, "candidate_legal": True, "secondary_safety_pass": False, "secondary_risk_score": 0.9, "p_merge_before_taper": 0.99, "target_lane_entry_time_s": 1.0, "ensemble_disagreement": 0.0},
            {"action_id": 8, "candidate_legal": True, "secondary_safety_pass": True, "secondary_risk_score": 0.2, "p_merge_before_taper": 0.90, "target_lane_entry_time_s": 2.0, "ensemble_disagreement": 0.0},
        ],
        raw_action_id=4,
        thresholds=_lite_thresholds(),
    )
    assert decision["selected"]["action_id"] == 8
    assert decision["replacement"] is True
    assert decision["reason"] == "raw_task_infeasible_lite_viable_left"


def test_viability_lite_does_not_replace_with_keep_or_below_margin_left():
    keep_only = select_viability_lite_action(
        [
            {"action_id": 4, "candidate_legal": True, "secondary_safety_pass": True, "p_merge_before_taper": 0.2, "target_lane_entry_time_s": 8.0, "ensemble_disagreement": 0.0},
            {"action_id": 5, "candidate_legal": True, "secondary_safety_pass": True, "p_merge_before_taper": 0.99, "target_lane_entry_time_s": 1.0, "ensemble_disagreement": 0.0},
        ],
        raw_action_id=4,
        thresholds=_lite_thresholds(),
    )
    assert keep_only["selected"] is None
    assert keep_only["candidate_set_available"] is False
    margin = select_viability_lite_action(
        [
            {"action_id": 4, "candidate_legal": True, "secondary_safety_pass": True, "p_merge_before_taper": 0.74, "target_lane_entry_time_s": 8.0, "ensemble_disagreement": 0.0},
            {"action_id": 7, "candidate_legal": True, "secondary_safety_pass": True, "p_merge_before_taper": 0.78, "target_lane_entry_time_s": 1.0, "ensemble_disagreement": 0.0},
        ],
        raw_action_id=4,
        thresholds=_lite_thresholds(),
    )
    assert margin["replacement"] is False
    assert margin["reason"] == "best_left_below_improvement_margin"


def test_viability_lite_does_not_replace_when_raw_is_illegal_or_missing():
    decision = select_viability_lite_action(
        [
            {"action_id": 3, "raw_action_legal": False, "candidate_legal": True, "secondary_safety_pass": True, "p_merge_before_taper": 0.2, "target_lane_entry_time_s": 8.0, "ensemble_disagreement": 0.0},
            {"action_id": 8, "raw_action_legal": False, "candidate_legal": True, "secondary_safety_pass": True, "p_merge_before_taper": 0.95, "target_lane_entry_time_s": 1.0, "ensemble_disagreement": 0.0},
        ],
        raw_action_id=2,
        thresholds=_lite_thresholds(),
    )
    assert decision["replacement"] is False
    assert decision["reason"] == "raw_action_illegal_or_missing"


def _lite_record(root_id: str, seed: int, action_id: int, raw_id: int, *, p_merge: float, success: bool, safe: bool = True, risk_pass: bool = True, risk_score: float = 0.0):
    return {
        "root_id": root_id,
        "episode_seed": seed,
        "action_id": action_id,
        "raw_action_id": raw_id,
        "root_policy": "merge_timing",
        "collection_source": "merge_timing",
        "traffic_profile": "hard",
        "activation_bin": "activation_window",
        "candidate_legal": True,
        "secondary_safety_pass": risk_pass,
        "secondary_risk_score": risk_score,
        "secondary_risk_uncertainty": 0.0,
        "secondary_veto_reason": "" if risk_pass else "risk_score",
        "p_merge_before_taper": p_merge,
        "p_proxy_collision": 0.1,
        "p_safety_violation": 0.1,
        "target_lane_entry_time_s": 2.0,
        "ensemble_disagreement": 0.0,
        "merge_observed": True,
        "merge_before_taper": 1.0 if success else 0.0,
        "proxy_collision": 0.0 if safe else 1.0,
        "safety_violation": 0.0 if safe else 1.0,
        "oracle_min_obb_distance": 4.0 if safe else 0.1,
        "oracle_min_ttc": 10.0 if safe else 0.1,
        "oracle_max_drac": 1.0 if safe else 8.0,
    }


def test_viability_lite_audit_separates_replacement_failure_modes():
    thresholds = {
        "min_p_merge_before_taper": 0.8,
        "min_improvement_over_raw": 0.01,
        "max_target_entry_time_s": 6.0,
        "max_ensemble_disagreement": 0.1,
        "max_secondary_risk_score": 0.2,
    }
    records = [
        _lite_record("unsafe_replacement", 101, 4, 4, p_merge=0.1, success=False),
        _lite_record("unsafe_replacement", 101, 8, 4, p_merge=0.9, success=True, safe=False, risk_score=0.1),
        _lite_record("risk_false_negative", 102, 4, 4, p_merge=0.1, success=False),
        _lite_record("risk_false_negative", 102, 7, 4, p_merge=0.9, success=True, risk_pass=False, risk_score=0.9),
        _lite_record("unnecessary", 103, 4, 4, p_merge=0.1, success=True),
        _lite_record("unnecessary", 103, 8, 4, p_merge=0.9, success=True, risk_score=0.1),
        _lite_record("targeted", 104, 4, 4, p_merge=0.1, success=False),
        _lite_record("targeted", 104, 8, 4, p_merge=0.9, success=True, risk_score=0.1),
    ]
    report = audit_lite_replacements(records, thresholds, split="test")
    assert report["actual_replacement_count"] == 3
    assert report["replacement_safety_event_root_count"] == 1
    assert report["risk_failed_but_success_root_count"] == 1
    assert report["unnecessary_replacement_root_count"] == 1
    assert report["targeted_seeds"] == [104]
    assert report["replacement_action_safety_event_rate"] == __import__("pytest").approx(1 / 3)
    assert report["summary_from_tuning_metric"]["replacement_action_risk_pass_rate"] == 1.0


def test_viability_lite_tuning_uses_replacement_only_metrics_and_secondary_risk_grid():
    records = [
        _lite_record("low_risk_repair", 201, 4, 4, p_merge=0.1, success=False),
        _lite_record("low_risk_repair", 201, 8, 4, p_merge=0.9, success=True, risk_score=0.03),
    ]
    cfg = clone_with_overrides(
        load_config(),
        {
            "accvp": {
                "viability_lite": {
                    "min_left_p_merge_before_taper_grid": [0.8],
                    "min_improvement_over_raw_grid": [0.01],
                    "max_target_entry_time_s_grid": [6.0],
                    "max_ensemble_disagreement_grid": [0.1],
                    "max_secondary_risk_score_grid": [0.05, 1.0],
                    "max_replacement_rate": 0.5,
                }
            }
        },
    )
    report = tune_viability_lite_operating_point(records, cfg, split="operating_point")
    assert report["selected"]["max_secondary_risk_score"] == 0.05
    assert report["selected_metrics"]["replacement_action_risk_pass_rate"] == 1.0
    assert report["selected_metrics"]["replacement_action_safety_event_rate"] == 0.0
    assert report["selected_metrics"]["replacement_repairable_capture_rate"] == 1.0


def test_viability_lite_audited_profile_can_recover_risk_failed_left_only():
    thresholds = {
        "min_p_merge_before_taper": 0.8,
        "min_improvement_over_raw": 0.01,
        "max_target_entry_time_s": 6.0,
        "max_ensemble_disagreement": 0.1,
        "max_secondary_risk_score": 0.95,
        "secondary_safety_profile": "audited_merge_left_v1",
    }
    decision = select_viability_lite_action(
        [
            _lite_record("root", 301, 4, 4, p_merge=0.1, success=False, risk_pass=False, risk_score=0.1),
            _lite_record("root", 301, 8, 4, p_merge=0.9, success=True, risk_pass=False, risk_score=0.9),
        ],
        raw_action_id=4,
        thresholds=thresholds,
    )
    assert decision["replacement"] is True
    assert int(decision["selected"]["action_id"]) == 8
    assert decision["selected"]["accvp_lite_secondary_safety_profile"] == "audited_merge_left_v1"


def test_risk_secondary_audit_reports_false_negatives_and_safe_sweep():
    records = [
        _lite_record("risk_fn", 401, 4, 4, p_merge=0.1, success=False),
        _lite_record("risk_fn", 401, 8, 4, p_merge=0.9, success=True, risk_pass=False, risk_score=0.8),
    ]
    base = {
        "min_p_merge_before_taper": 0.8,
        "min_improvement_over_raw": 0.01,
        "max_target_entry_time_s": 6.0,
        "max_ensemble_disagreement": 0.1,
        "max_secondary_risk_score": 0.2,
        "max_replacement_rate": 1.0,
        "secondary_safety_profile": "strict",
    }
    report = audit_risk_secondary(records, base, split="test", risk_score_grid=[0.2, 0.9])
    assert report["confusion"]["risk_fail_clean_success"] == 1
    assert report["selected_audited_profile"]["secondary_safety_profile"] == "audited_merge_left_v1"
    assert report["selected_audited_profile"]["max_secondary_risk_score"] == 0.9
    assert report["threshold_sweep"][1]["safe_success_recovered_count"] == 1
    assert report["threshold_sweep"][1]["unsafe_pass_count"] == 0


def test_viability_lite_acceptance_detects_failed_final_test():
    cfg = clone_with_overrides(
        load_config(),
        {
            "accvp": {
                "viability_lite": {
                    "acceptance": {
                        "require_replacement": True,
                        "max_replacement_safety_event_rate": 0.0,
                        "min_replacement_merge_success_rate": 0.9,
                        "max_replacement_unnecessary_rate": 0.25,
                        "max_replacement_rate": 0.10,
                        "min_replacement_repairable_capture_rate": 0.05,
                    }
                }
            }
        },
    )
    failures = _lite_acceptance_failures(
        {
            "replacement_count": 1,
            "replacement_action_safety_event_rate": 0.0,
            "replacement_action_merge_success_rate": 0.8,
            "replacement_unnecessary_rate": 0.5,
            "replacement_rate": 0.02,
            "replacement_repairable_capture_rate": 0.05,
        },
        cfg,
    )
    assert "replacement_action_merge_success_rate<0.9" in failures
    assert "replacement_unnecessary_rate>0.25" in failures
    assert "replacement_repairable_capture_rate<=0.05" in failures


def test_online_trigger_audit_generates_deterministic_shadow_seed_file(tmp_path: Path):
    replay = tmp_path / "group_seed_12.json"
    replay.write_text(
        __import__("json").dumps(
            {
                "seed": 12,
                "group_name": "shield_accvp_lite_shadow_v3",
                "notes": {
                    "accvp_records": [
                        {
                            "accvp_mode": "viability_lite_shadow",
                            "accvp_skip_reason": "",
                            "accvp_bypass_reason": "",
                            "candidate_set_available": True,
                            "raw_feasible": False,
                            "raw_action": 4,
                            "safety_shield_action": 4,
                            "accvp_shadow_recommended_action": 8,
                            "accvp_lite_p_merge_improvement": 0.05,
                            "step": 10,
                            "decision_index": 2,
                        },
                        {
                            "accvp_mode": "viability_lite_shadow",
                            "accvp_skip_reason": "outside_deadline_window",
                            "accvp_bypass_reason": "",
                            "candidate_set_available": False,
                            "raw_action": 5,
                            "safety_shield_action": 5,
                        },
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    report = audit_online_triggers([tmp_path], group_contains="accvp_lite_shadow")
    assert report["online_targeted_seeds"] == [12]
    assert report["would_trigger_count"] == 1
    assert report["reason_counts"]["skip:outside_deadline_window"] == 1


def test_availability_diagnostic_separates_oracle_and_risk_ceilings(tmp_path: Path):
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    roots = []
    branches = []

    def add_root(root_id: str, raw_success: bool, left_rows: list[dict]):
        roots.append(
            {
                "root_id": root_id,
                "episode_seed": len(roots),
                "root_policy": "merge_timing",
                "collection_source": "merge_timing",
                "traffic_profile": "hard",
                "activation_bin": "activation_window",
                "deadline_bin": "deadline",
                "raw_action_id": 4,
                "complete": True,
            }
        )
        branches.append(
            {
                "root_id": root_id,
                "branch_status": "completed",
                "event_observed": True,
                "action_id": 4,
                "candidate_legal": True,
                "secondary_safety_pass": True,
                "proxy_collision_within_horizon": not raw_success,
                "safety_violation_within_horizon": not raw_success,
                "merge_before_taper_observed": raw_success,
                "viability_observation_status": "observed_success" if raw_success else "observed_failure",
            }
        )
        for row in left_rows:
            branches.append({"root_id": root_id, "branch_status": "completed", "event_observed": True, **row})

    add_root("raw_ok", True, [])
    add_root("no_left", False, [])
    add_root("risk_fail", False, [{"action_id": 7, "candidate_legal": True, "secondary_safety_pass": False, "proxy_collision_within_horizon": False, "safety_violation_within_horizon": False, "merge_before_taper_observed": True, "viability_observation_status": "observed_success"}])
    add_root("unsafe", False, [{"action_id": 7, "candidate_legal": True, "secondary_safety_pass": True, "proxy_collision_within_horizon": True, "safety_violation_within_horizon": True, "merge_before_taper_observed": False, "viability_observation_status": "observed_failure"}])
    add_root("left_ok", False, [{"action_id": 8, "candidate_legal": True, "secondary_safety_pass": True, "proxy_collision_within_horizon": False, "safety_violation_within_horizon": False, "merge_before_taper_observed": True, "viability_observation_status": "observed_success"}])

    (manifests / "roots.jsonl").write_text("".join(__import__("json").dumps(row) + "\n" for row in roots), encoding="utf-8")
    (manifests / "branches.jsonl").write_text("".join(__import__("json").dumps(row) + "\n" for row in branches), encoding="utf-8")
    (manifests / "split_manifest.jsonl").write_text(
        "".join(__import__("json").dumps({"root_id": row["root_id"], "split": "operating_point"}) + "\n" for row in roots),
        encoding="utf-8",
    )

    report = diagnose_oracle_availability(tmp_path, split="operating_point")
    assert report["decision_count"] == 5
    assert report["oracle_merge_intent_ceiling_availability"] == 0.4
    assert report["risk_secondary_pass_ceiling_availability"] == 0.6
    assert report["reason_counts"]["raw already feasible"] == 1
    assert report["reason_counts"]["no legal merge-left candidate"] == 1
    assert report["reason_counts"]["merge-left candidate Risk secondary failed"] == 1
    assert report["reason_counts"]["merge-left physically unsafe"] == 1
    assert report["reason_counts"]["oracle merge-left feasible"] == 1
    assert report["left_action_stats"]["7"]["count"] == 2
    assert report["left_action_stats"]["8"]["observed_success_count"] == 1


def test_risk_secondary_audit_reports_physical_ceiling_and_false_negatives(tmp_path: Path):
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    roots = [
        {"root_id": "raw_ok", "episode_seed": 1, "root_policy": "merge_timing", "activation_bin": "activation_window", "deadline_bin": "deadline", "raw_action_id": 4, "complete": True},
        {"root_id": "risk_fn", "episode_seed": 2, "root_policy": "merge_timing", "activation_bin": "activation_window", "deadline_bin": "deadline", "raw_action_id": 4, "complete": True},
        {"root_id": "unsafe", "episode_seed": 3, "root_policy": "merge_timing", "activation_bin": "activation_window", "deadline_bin": "deadline", "raw_action_id": 4, "complete": True},
    ]
    branches = [
        {"root_id": "raw_ok", "branch_status": "completed", "event_observed": True, "action_id": 4, "candidate_legal": True, "secondary_safety_pass": True, "proxy_collision_within_horizon": False, "safety_violation_within_horizon": False, "merge_before_taper_observed": True, "viability_observation_status": "observed_success"},
        {"root_id": "risk_fn", "branch_status": "completed", "event_observed": True, "action_id": 4, "candidate_legal": True, "secondary_safety_pass": True, "proxy_collision_within_horizon": True, "safety_violation_within_horizon": True, "merge_before_taper_observed": False, "viability_observation_status": "observed_failure"},
        {"root_id": "risk_fn", "branch_status": "completed", "event_observed": True, "action_id": 7, "candidate_legal": True, "secondary_safety_pass": False, "secondary_risk": {"risk_score": 0.9}, "proxy_collision_within_horizon": False, "safety_violation_within_horizon": False, "merge_before_taper_observed": True, "viability_observation_status": "observed_success"},
        {"root_id": "unsafe", "branch_status": "completed", "event_observed": True, "action_id": 4, "candidate_legal": True, "secondary_safety_pass": True, "proxy_collision_within_horizon": True, "safety_violation_within_horizon": True, "merge_before_taper_observed": False, "viability_observation_status": "observed_failure"},
        {"root_id": "unsafe", "branch_status": "completed", "event_observed": True, "action_id": 8, "candidate_legal": True, "secondary_safety_pass": False, "secondary_risk": {"risk_score": 0.8}, "proxy_collision_within_horizon": True, "safety_violation_within_horizon": True, "merge_before_taper_observed": False, "viability_observation_status": "observed_failure"},
    ]
    (manifests / "roots.jsonl").write_text("".join(__import__("json").dumps(row) + "\n" for row in roots), encoding="utf-8")
    (manifests / "branches.jsonl").write_text("".join(__import__("json").dumps(row) + "\n" for row in branches), encoding="utf-8")
    (manifests / "split_manifest.jsonl").write_text(
        "".join(__import__("json").dumps({"root_id": row["root_id"], "split": "operating_point"}) + "\n" for row in roots),
        encoding="utf-8",
    )

    report = audit_risk_secondary_false_negatives(tmp_path, split="operating_point")
    assert report["decision_count"] == 3
    assert report["physical_oracle_ceiling_ignore_risk"] == 2 / 3
    assert report["risk_gated_physical_ceiling"] == 1 / 3
    assert report["risk_false_negative_root_count"] == 1
    assert report["risk_false_negative_action_count"] == 1
    assert report["left_action_stats"]["7"]["risk_false_negative_count"] == 1
    assert report["false_negative_risk_score_summary"]["count"] == 1


def test_model_gate_diagnostic_identifies_viability_gate_failure():
    thresholds = {
        "proxy_collision_upper_bound": 0.2,
        "safety_violation_upper_bound": 0.2,
        "merge_viability_lower_bound": 0.5,
    }
    report = model_gate_failure_diagnostics(
        [
            {
                "root_id": "root",
                "raw_action_id": 4,
                "action_id": 4,
                "candidate_legal": True,
                "secondary_safety_pass": True,
                "pU_proxy_collision": 0.9,
                "pU_safety_violation": 0.9,
                "pL_merge_before_taper": 0.9,
                "proxy_collision": 1.0,
                "safety_violation": 1.0,
                "merge_before_taper": 0.0,
                "merge_observed": True,
            },
            {
                "root_id": "root",
                "raw_action_id": 4,
                "action_id": 7,
                "candidate_legal": True,
                "secondary_safety_pass": True,
                "pU_proxy_collision": 0.1,
                "pU_safety_violation": 0.1,
                "pL_merge_before_taper": 0.1,
                "proxy_collision": 0.0,
                "safety_violation": 0.0,
                "merge_before_taper": 1.0,
                "merge_observed": True,
            },
        ],
        thresholds,
        required_availability=0.95,
    )
    assert report["deployable_artifact"] is False
    assert report["model_gate_best_availability"] == 0.0
    assert report["reason_counts"]["model pL_viability gate failed"] == 1
    assert report["per_action_gate_pass_rate"]["7"]["viability_gate_pass_count"] == 0


def test_candidate_table_diagnostic_reports_viability_only_pass():
    thresholds = {
        "proxy_collision_upper_bound": 0.2,
        "safety_violation_upper_bound": 0.2,
        "merge_viability_lower_bound": 0.5,
    }
    records = [
        {
            "root_id": "repairable",
            "raw_action_id": 4,
            "action_id": 4,
            "candidate_legal": True,
            "secondary_safety_pass": True,
            "p_proxy_collision": 0.1,
            "p_safety_violation": 0.1,
            "p_taper_miss": 0.8,
            "p_merge_before_taper": 0.1,
            "pU_proxy_collision": 0.1,
            "pU_safety_violation": 0.1,
            "pL_merge_before_taper": 0.1,
            "proxy_collision": 0.0,
            "safety_violation": 0.0,
            "taper_miss": 1.0,
            "merge_before_taper": 0.0,
            "merge_observed": True,
        },
        {
            "root_id": "repairable",
            "raw_action_id": 4,
            "action_id": 7,
            "candidate_legal": True,
            "secondary_safety_pass": True,
            "p_proxy_collision": 0.1,
            "p_safety_violation": 0.1,
            "p_taper_miss": 0.1,
            "p_merge_before_taper": 0.9,
            "pU_proxy_collision": 0.1,
            "pU_safety_violation": 0.1,
            "pL_merge_before_taper": 0.9,
            "proxy_collision": 0.0,
            "safety_violation": 0.0,
            "taper_miss": 0.0,
            "merge_before_taper": 1.0,
            "merge_observed": True,
        },
    ]
    report = candidate_table_summary(records, split="test", thresholds=thresholds)
    assert report["verdict"]["deployable_claim"] is False
    assert report["verdict"]["step1_5_state"] == "viability_only_pass"
    assert report["pairwise_contrast"]["viability"]["accuracy"] == 1.0
    assert report["raw_fail_left_success"]["raw_fail_left_success_root_count"] == 1
    assert report["raw_fail_left_success"]["best_left_pmerge_gt_raw_rate"] == 1.0
    assert report["per_action"]["7"]["is_left_action"] is True
    assert report["raw_probability_recommendation"]["best_p_merge_action_counts"]["7"] == 1


def test_split_keeps_all_roots_of_same_episode_seed_together(tmp_path: Path):
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    roots = [
        {"root_id": "a", "episode_seed": 1, "root_source": "mixed", "traffic_profile": "safe", "deadline_bin": "deadline", "complete": True},
        {"root_id": "b", "episode_seed": 1, "root_source": "mixed", "traffic_profile": "safe", "deadline_bin": "deadline", "complete": True},
        {"root_id": "c", "episode_seed": 2, "root_source": "rule", "traffic_profile": "hard", "deadline_bin": "pre_deadline", "complete": True},
    ]
    (manifests / "roots.jsonl").write_text("".join(__import__("json").dumps(row) + "\n" for row in roots), encoding="utf-8")
    rows = build_split_manifest(tmp_path, seed=7, require_all_splits=False)
    assignments = {row["root_id"]: row["split"] for row in rows}
    assert assignments["a"] == assignments["b"]
    with __import__("pytest").raises(ValueError, match="at least 5"):
        build_split_manifest(tmp_path, seed=7)


def test_oracle_requires_safe_viable_counterfactual_for_each_failure_seed(tmp_path: Path):
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    roots = [
        {"root_id": "seed2", "root_episode_id": "ppo:2", "episode_seed": 2, "root_policy": "ppo", "deadline_bin": "deadline", "raw_action_id": 4, "raw_action_legal": True, "complete": True},
        {"root_id": "seed5", "root_episode_id": "ppo:5", "episode_seed": 5, "root_policy": "ppo", "deadline_bin": "deadline", "raw_action_id": 4, "raw_action_legal": True, "complete": True},
    ]
    branches = [
        {"root_id": "seed2", "branch_status": "completed", "action_id": 4, "proxy_collision_within_horizon": True, "safety_violation_within_horizon": True, "merge_before_taper_observed": False, "viability_observation_status": "observed_failure"},
        {"root_id": "seed2", "branch_status": "completed", "action_id": 7, "proxy_collision_within_horizon": False, "safety_violation_within_horizon": False, "merge_before_taper_observed": True, "viability_observation_status": "observed_success", "secondary_safety_pass": True},
        {"root_id": "seed5", "branch_status": "completed", "action_id": 4, "proxy_collision_within_horizon": True, "safety_violation_within_horizon": True, "merge_before_taper_observed": False, "viability_observation_status": "observed_failure"},
    ]
    (manifests / "roots.jsonl").write_text("".join(__import__("json").dumps(row) + "\n" for row in roots), encoding="utf-8")
    (manifests / "branches.jsonl").write_text("".join(__import__("json").dumps(row) + "\n" for row in branches), encoding="utf-8")
    report = counterfactual_oracle_report(tmp_path, required_seeds=[2, 5])
    assert report["required_failure_seed_results"]["2"]["state"] == "go"
    assert report["required_failure_seed_results"]["5"]["state"] == "no_safe_viable_alternative"
    assert report["oracle_state"] == "no_safe_viable_alternative"
    assert report["go_for_training"] is False


def test_oracle_rejects_physical_success_vetoed_by_secondary_risk(tmp_path: Path):
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    root = {"root_id": "seed2", "episode_seed": 2, "root_policy": "merge_timing", "deadline_bin": "deadline", "raw_action_id": 4, "raw_action_legal": True, "complete": True}
    branches = [
        {"root_id": "seed2", "branch_status": "completed", "action_id": 4, "proxy_collision_within_horizon": True, "safety_violation_within_horizon": True, "merge_before_taper_observed": False, "viability_observation_status": "observed_failure", "secondary_safety_pass": True},
        {"root_id": "seed2", "branch_status": "completed", "action_id": 7, "proxy_collision_within_horizon": False, "safety_violation_within_horizon": False, "merge_before_taper_observed": True, "viability_observation_status": "observed_success", "secondary_safety_pass": False},
    ]
    (manifests / "roots.jsonl").write_text(__import__("json").dumps(root) + "\n", encoding="utf-8")
    (manifests / "branches.jsonl").write_text("".join(__import__("json").dumps(row) + "\n" for row in branches), encoding="utf-8")
    report = counterfactual_oracle_report(tmp_path, required_seeds=[2], root_policy="merge_timing")
    assert report["oracle_state"] == "no_safe_viable_alternative"


def test_oracle_distinguishes_missing_deadline_coverage(tmp_path: Path):
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    (manifests / "roots.jsonl").write_text(
        __import__("json").dumps(
            {"root_id": "early", "episode_seed": 2, "deadline_bin": "pre_deadline", "raw_action_id": 4, "raw_action_legal": True, "complete": True}
        )
        + "\n",
        encoding="utf-8",
    )
    (manifests / "branches.jsonl").write_text("", encoding="utf-8")
    report = counterfactual_oracle_report(tmp_path, required_seeds=[2])
    assert report["oracle_state"] == "insufficient_coverage"
    assert report["go_for_training"] is False


def test_root_policy_filter_and_exact_seed_schedule_are_independent():
    cfg = load_config()
    assert _root_filter_matches("deadline", "deadline") is True
    assert _root_filter_matches("activation_window", "activation_window") is True
    assert _root_filter_matches("deadline", "pre_deadline") is False
    assert _seed_schedule(cfg, [2, 5], None) == [2, 5]


def test_default_counterfactual_cache_is_under_output_tree_not_repository_root(tmp_path: Path):
    cfg = clone_with_overrides(
        load_config(),
        {"run": {"output_root": str(tmp_path / "safe_rl_output" / "runs"), "run_id": "cache_test", "cache_root": None}},
    )
    cache = _cache_dir(cfg, "counterfactual", cfg.accvp.counterfactual)
    assert cache == tmp_path / "safe_rl_output" / ".cache" / "cache_test" / "stage1_counterfactual" / "counterfactual"


def test_immutable_shards_merge_without_overwriting_sources(tmp_path: Path):
    shards = []
    contract = {
        "protocol_version": "accvp_240_v1",
        "scenario_config_hash": "scenario",
        "scenario_route_hash": "route",
        "action_execution_profile": "current_v1",
        "candidate_plan_profile": ACCVP_COMMITMENT_PROFILE,
        "activation_distance_m": 240.0,
        "response_horizon_s": 3.0,
        "response_horizon_steps": 30,
        "viability_horizon_s": 8.0,
        "candidate_plan_horizon_steps": 80,
        "actor_count": 6,
        "actor_selection_config_hash": "actors",
        "safety_metric_version": "obb",
        "event_definition_version": "events",
        "risk_model_fingerprint": "risk_checkpoint:fixture",
    }
    for index in range(2):
        shard = tmp_path / f"shard_{index}"
        manifests = shard / "manifests"
        manifests.mkdir(parents=True)
        root_id = f"root_{index}"
        (manifests / "roots.jsonl").write_text(
            __import__("json").dumps(
                {
                    "root_id": root_id,
                    "complete": True,
                    "root_policy": "merge_timing",
                    "collection_source": "merge_timing",
                    "traffic_profile": "hard",
                    "deadline_bin": "deadline",
                    "activation_bin": "activation_window",
                    "data_contract_hash": stable_hash(contract),
                    "root_state_fingerprint": "same_physical_state",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        (manifests / "branches.jsonl").write_text(
            __import__("json").dumps(
                {
                    "root_id": root_id,
                    "action_id": 4,
                    "branch_status": "completed",
                    "secondary_safety_pass": True,
                    "risk_model_fingerprint": "risk_checkpoint:fixture",
                    "data_contract_hash": stable_hash(contract),
                }
            )
            + "\n",
            encoding="utf-8",
        )
        (manifests / "dataset_manifest.json").write_text(
            __import__("json").dumps(
                {
                    "artifact_kind": "counterfactual_shard_v2",
                    "counterfactual_schema_version": COUNTERFACTUAL_SCHEMA_VERSION,
                    "collection_id": f"job_{index}",
                    "collection_source": "merge_timing",
                    "scenario_config_hash": "scenario",
                    "action_execution_profile": "current_v1",
                    "candidate_plan_profile": ACCVP_COMMITMENT_PROFILE,
                    "risk_model_fingerprint": "risk_checkpoint:fixture",
                    "config_hash": f"config_{index}",
                    "data_contract": contract,
                    "data_contract_hash": stable_hash(contract),
                }
            ),
            encoding="utf-8",
        )
        shards.append(shard)
    output = merge_counterfactual_shards(shards, tmp_path / "formal")
    manifest = __import__("json").loads((output / "manifests" / "dataset_manifest.json").read_text(encoding="utf-8"))
    assert manifest["root_count"] == 2
    assert manifest["unique_root_state_fingerprint_count"] == 1
    assert manifest["duplicate_root_state_fingerprint_count"] == 1
    assert manifest["duplicate_root_state_fingerprints"][0]["first_root_id"] == "root_0"
    assert manifest["duplicate_root_state_fingerprints"][0]["duplicate_root_id"] == "root_1"
    assert (shards[0] / "manifests" / "roots.jsonl").exists()
    with __import__("pytest").raises(FileExistsError):
        merge_counterfactual_shards(shards, output)


def test_shard_merge_rejects_mismatched_protocol_contract(tmp_path: Path):
    base = {
        "artifact_kind": "counterfactual_shard_v2",
        "counterfactual_schema_version": COUNTERFACTUAL_SCHEMA_VERSION,
        "collection_id": "one",
        "scenario_config_hash": "scenario",
        "action_execution_profile": "current_v1",
        "candidate_plan_profile": ACCVP_COMMITMENT_PROFILE,
        "risk_model_fingerprint": "risk_checkpoint:fixture",
        "config_hash": "source_config",
    }
    shards = []
    for name, distance in (("one", 240.0), ("two", 120.0)):
        shard = tmp_path / name
        manifests = shard / "manifests"
        manifests.mkdir(parents=True)
        contract = {
            "protocol_version": "accvp_240_v1",
            "scenario_config_hash": "scenario",
            "scenario_route_hash": "route",
            "action_execution_profile": "current_v1",
            "candidate_plan_profile": ACCVP_COMMITMENT_PROFILE,
            "activation_distance_m": distance,
            "response_horizon_s": 3.0,
            "response_horizon_steps": 30,
            "viability_horizon_s": 8.0,
            "candidate_plan_horizon_steps": 80,
            "actor_count": 6,
            "actor_selection_config_hash": "actors",
            "safety_metric_version": "obb",
            "event_definition_version": "events",
            "risk_model_fingerprint": "risk_checkpoint:fixture",
        }
        (manifests / "roots.jsonl").write_text("", encoding="utf-8")
        (manifests / "branches.jsonl").write_text("", encoding="utf-8")
        (manifests / "dataset_manifest.json").write_text(
            __import__("json").dumps(
                {
                    **base,
                    "collection_id": name,
                    "data_contract": contract,
                    "data_contract_hash": stable_hash(contract),
                }
            ),
            encoding="utf-8",
        )
        shards.append(shard)
    with __import__("pytest").raises(ValueError, match="data contract"):
        merge_counterfactual_shards(shards, tmp_path / "formal")


def test_collection_job_can_override_policy_observation_config_without_mutating_parent():
    cfg = load_config()
    job_cfg, job = materialise_collection_job(
        cfg,
        {
            "name": "ppo_240",
            "root_policy": "ppo",
            "root_filter": "all",
            "root_budget": 100,
            "root_policy_checkpoint": "baseline.zip",
            "config_overrides": {
                "forecast_features": {"enabled": False},
                "rl": {"reward_profile": "default", "use_wcdt_forecast_features": False},
            },
        },
    )
    assert job["name"] == "ppo_240"
    assert job_cfg.accvp.counterfactual.root_budget == 100
    assert job_cfg.accvp.counterfactual.policy_checkpoints.ppo == "baseline.zip"
    assert job_cfg.forecast_features.enabled is False
    assert cfg.accvp.counterfactual.root_budget != 100


def test_formal_collection_requires_matching_pilot_report(tmp_path: Path):
    risk = tmp_path / "risk.pt"
    risk.write_bytes(b"risk")
    report_path = tmp_path / "pilot.json"
    cfg = clone_with_overrides(
        load_config(),
        {
            "accvp": {
                "activation_distance": 240.0,
                "counterfactual": {"risk_checkpoint": str(risk), "required_pilot_report": str(report_path)},
            }
        },
    )
    fingerprint = f"risk_checkpoint:{__import__('hashlib').sha256(risk.read_bytes()).hexdigest()}"
    report_path.write_text(
        __import__("json").dumps(
            {
                "pilot_state": "pass",
                "accvp_activation_distance_m": 240.0,
                "data_contract_hash": data_contract_hash(counterfactual_data_contract(cfg, fingerprint)),
            }
        ),
        encoding="utf-8",
    )
    validate_required_pilot(cfg)
    report_path.write_text(__import__("json").dumps({"pilot_state": "fail"}), encoding="utf-8")
    with __import__("pytest").raises(ValueError, match="pilot_state"):
        validate_required_pilot(cfg)


def test_pilot_validator_requires_source_quality_and_matching_seed_oracle(tmp_path: Path):
    source = tmp_path / "source"
    source_manifests = source / "manifests"
    source_manifests.mkdir(parents=True)
    (source_manifests / "dataset_manifest.json").write_text(
        __import__("json").dumps(
            {
                "collection_id": "mixed_240",
                "collection_source": "mixed",
                "branch_status_counts": {"completed": 1},
            }
        ),
        encoding="utf-8",
    )
    dataset = tmp_path / "dataset"
    manifests = dataset / "manifests"
    manifests.mkdir(parents=True)
    (manifests / "roots.jsonl").write_text(
        __import__("json").dumps(
            {"root_id": "root", "complete": True, "collection_source": "mixed", "activation_bin": "activation_window"}
        )
        + "\n",
        encoding="utf-8",
    )
    (manifests / "branches.jsonl").write_text(
        __import__("json").dumps(
            {"root_id": "root", "branch_status": "completed", "activation_bin": "activation_window", "event_observed": True}
        )
        + "\n",
        encoding="utf-8",
    )
    (manifests / "dataset_manifest.json").write_text(
        __import__("json").dumps(
            {
                "artifact_kind": "counterfactual_dataset_v2",
                "collection_phase": "pilot",
                "dataset_fingerprint": "dataset",
                "data_contract_hash": "contract",
                "accvp_activation_distance_m": 240.0,
                "source_shards": [{"path": str(source)}],
            }
        ),
        encoding="utf-8",
    )
    oracle = tmp_path / "oracle.json"
    oracle.write_text(
        __import__("json").dumps(
            {
                "oracle_state": "go",
                "required_seeds": [2, 5],
                "root_policy": "merge_timing",
                "dataset_provenance": {"dataset_fingerprint": "dataset"},
            }
        ),
        encoding="utf-8",
    )
    report = validate_pilot_dataset(
        dataset,
        expected_root_counts={"mixed": 1},
        oracle_report_path=oracle,
    )
    assert report["pilot_state"] == "pass"
