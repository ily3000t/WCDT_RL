from __future__ import annotations

import json
from pathlib import Path

import pytest

import safe_rl.pipeline.stage3_train_ppo as stage3_train_ppo
import safe_rl.pipeline.stage5_paired_eval as stage5_paired_eval
from safe_rl.accvp.serving.observation import RiskGatedACCVPCandidateTableAugmentor
from safe_rl.accvp.evaluation.risk_secondary import combine_audit_reports
from safe_rl.accvp.serving.predictor import ACCVPRuntimePredictor
from safe_rl.accvp.contracts.runtime_contract import (
    SIMULATION_BLOCKING_EXACT_CONTRACT,
    candidate_table_semantic_contract_sha256,
    canonical_formal_runtime_contract,
    closed_loop_execution_contract_sha256,
    formal_runtime_contract_sha256,
)
from safe_rl.accvp.contracts.schema import file_sha256, stable_hash
from safe_rl.accvp.planning.viability_lite import tune_viability_lite_operating_point
from safe_rl.artifact_revocation import (
    assert_artifact_not_revoked,
    write_artifact_revocation_manifest,
)
from safe_rl.evaluation_protocol import (
    EvidenceProtocolError,
    HoldoutAlreadyOpenedError,
    SeedCohortOverlapError,
    build_stage_lineage,
    claim_final_holdout,
    finalise_holdout_claim,
    protocol_snapshot,
    validate_seed_cohorts,
)
from safe_rl.rl.ppo import _validate_stage3_seed_preflight
from safe_rl.pipeline.accvp_final_holdout_eval import (
    _artifact_prefix,
    _canonical_seal_path,
    _validate_vnext_promotion_evidence,
    _verify_claim_inputs,
)
from safe_rl.utils.config import REPO_ROOT, load_config


class _Config(dict):
    def __getattr__(self, name):
        value = self[name]
        return _Config(value) if isinstance(value, dict) else value


def _protocol_config() -> _Config:
    return _Config(
        {
            "run": {"seed": 101},
            "stage3": {"eval_seeds": [201, 202]},
            "evaluation_protocol": {
                "strict": True,
                "protocol_id": "protocol-v1",
                "cohort_roles": {
                    "stage3_training": "ppo_training",
                    "stage3_selection": "ppo_selection",
                    "stage5_confirmatory": "natural_confirmatory",
                },
                "seed_cohorts": {
                    "ppo_training": [101, 102, 103],
                    "ppo_selection": [201, 202],
                    "natural_confirmatory": [301, 302],
                },
            },
        }
    )


def test_seed_ledger_rejects_cross_cohort_overlap():
    with pytest.raises(SeedCohortOverlapError, match="overlap"):
        validate_seed_cohorts({"train": [1, 2], "selection": [2, 3]})


def test_stage3_seed_preflight_uses_registered_disjoint_cohorts():
    cfg = _protocol_config()
    report = _validate_stage3_seed_preflight(cfg)
    assert report["selection_seeds"] == [201, 202]
    assert report["seed_audit"]["overlap_count"] == 0
    cfg["run"]["seed"] = 201
    with pytest.raises((SeedCohortOverlapError, EvidenceProtocolError)):
        _validate_stage3_seed_preflight(cfg)


def test_legacy_stage3_seed_overlap_is_recorded_but_not_enforced():
    cfg = _Config(
        {
            "run": {"seed": 1},
            "stage3": {"eval_enabled": True, "eval_seeds": [1, 2]},
        }
    )
    report = _validate_stage3_seed_preflight(cfg)
    assert report["protocol_enabled"] is False
    assert report["disjointness_enforced"] is False
    assert report["seed_audit"]["overlap_count"] == 1


def test_vnext_protocol_example_loads_seed_ranges_and_revocation_manifest():
    cfg = load_config("safe_rl/config/examples/vnext/evaluation_protocol_vnext.example.yaml")
    snapshot = protocol_snapshot(cfg, base_dir=REPO_ROOT)
    assert snapshot["strict"] is True
    assert snapshot["protocol_id"] == "accvp-vnext-correctness-v1"
    assert snapshot["seed_audit"]["overlap_count"] == 0
    assert snapshot["seed_audit"]["cohort_counts"]["natural_confirmatory"] == 300
    assert snapshot["revocation_manifest_sha256"]


def test_stage_lineage_rejects_seed_outside_registered_role():
    cfg = _protocol_config()
    with pytest.raises(EvidenceProtocolError, match="outside ledger cohort"):
        build_stage_lineage(
            cfg,
            stage="stage3",
            role_seeds={"stage3_training": [999], "stage3_selection": [201]},
        )
    snapshot = protocol_snapshot(cfg)
    assert snapshot["protocol_id"] == "protocol-v1"
    assert snapshot["seed_audit"]["overlap_count"] == 0


def test_final_holdout_claim_is_atomic_and_one_shot(tmp_path: Path):
    artifact = tmp_path / "artifact.json"
    split = tmp_path / "split.jsonl"
    result = tmp_path / "result.json"
    seal = tmp_path / "seal.json"
    artifact.write_text('{"holdout_state":"sealed"}', encoding="utf-8")
    split.write_text('{"root_id":"r1","split":"test"}\n', encoding="utf-8")
    result.write_text('{"decision":"go"}', encoding="utf-8")
    claim = claim_final_holdout(
        seal,
        protocol_id="protocol-v1",
        artifact_manifest=artifact,
        split_manifest=split,
    )
    assert claim["state"] == "opened"
    with pytest.raises(HoldoutAlreadyOpenedError):
        claim_final_holdout(
            seal,
            protocol_id="protocol-v1",
            artifact_manifest=artifact,
            split_manifest=split,
        )
    final = finalise_holdout_claim(seal, result_path=result, decision="go")
    assert final["state"] == "evaluated"
    assert final["decision"] == "go"
    with pytest.raises(HoldoutAlreadyOpenedError):
        finalise_holdout_claim(seal, result_path=result, decision="go")
    with pytest.raises(HoldoutAlreadyOpenedError):
        claim_final_holdout(
            seal,
            protocol_id="protocol-v1",
            artifact_manifest=artifact,
            split_manifest=split,
        )


def test_holdout_claim_binds_all_frozen_inputs_before_open(tmp_path: Path):
    artifact = tmp_path / "accvp_v1_candidate_artifact_manifest.json"
    split = tmp_path / "split.jsonl"
    checkpoint = tmp_path / "model.pt"
    seal = tmp_path / "seal.json"
    artifact.write_text("{}", encoding="utf-8")
    split.write_text(
        json.dumps({"root_id": "test-root", "split": "test"}) + "\n",
        encoding="utf-8",
    )
    checkpoint.write_bytes(b"frozen")
    with pytest.raises(EvidenceProtocolError, match="changed before claim"):
        claim_final_holdout(
            seal,
            protocol_id="protocol-v1",
            artifact_manifest=artifact,
            split_manifest=split,
            frozen_artifacts={"checkpoint": checkpoint},
            expected_sha256={"checkpoint": "not-the-real-hash"},
        )
    assert not seal.exists()
    assert _artifact_prefix(artifact) == "accvp_v1"
    first = _canonical_seal_path(
        dataset=tmp_path,
        protocol_id="protocol-v1",
        split_manifest_path=split,
    )
    second = _canonical_seal_path(
        dataset=tmp_path,
        protocol_id="protocol-v1",
        split_manifest_path=split,
    )
    assert first == second
    assert first.parent.name == "final_holdout_claims"


def test_final_holdout_claim_is_one_shot_per_protocol_and_test_cohort(
    tmp_path: Path,
):
    split = tmp_path / "split.jsonl"
    split.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "root_id": "test-root",
                        "root_observation_fingerprint": "fingerprint-a",
                        "scenario_episode_key": "scenario:101",
                        "episode_seed": 101,
                        "split": "test",
                    }
                ),
                json.dumps({"root_id": "train-root", "split": "train"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    first_artifact = tmp_path / "candidate-a.json"
    second_artifact = tmp_path / "candidate-b.json"
    first_artifact.write_text('{"candidate":"a"}', encoding="utf-8")
    second_artifact.write_text('{"candidate":"b"}', encoding="utf-8")
    seal = _canonical_seal_path(
        dataset=tmp_path,
        protocol_id="protocol-v1",
        split_manifest_path=split,
    )
    claim_final_holdout(
        seal,
        protocol_id="protocol-v1",
        artifact_manifest=first_artifact,
        split_manifest=split,
    )
    with pytest.raises(HoldoutAlreadyOpenedError):
        claim_final_holdout(
            seal,
            protocol_id="protocol-v1",
            artifact_manifest=second_artifact,
            split_manifest=split,
        )


def test_final_holdout_seal_ignores_non_test_split_changes(tmp_path: Path):
    first_split = tmp_path / "split-a.jsonl"
    second_split = tmp_path / "split-b.jsonl"
    test_row = {
        "root_id": "test-root",
        "root_observation_fingerprint": "fingerprint-a",
        "scenario_episode_key": "scenario:101",
        "episode_seed": 101,
        "split": "test",
    }
    first_split.write_text(
        json.dumps(test_row) + "\n" + json.dumps({"root_id": "train-a", "split": "train"}) + "\n",
        encoding="utf-8",
    )
    second_split.write_text(
        json.dumps(test_row) + "\n" + json.dumps({"root_id": "train-b", "split": "train"}) + "\n",
        encoding="utf-8",
    )
    first = _canonical_seal_path(
        dataset=tmp_path,
        protocol_id="protocol-v1",
        split_manifest_path=first_split,
    )
    second = _canonical_seal_path(
        dataset=tmp_path,
        protocol_id="protocol-v1",
        split_manifest_path=second_split,
    )
    assert first == second


def test_holdout_claim_detects_manifest_mutation_after_open(tmp_path: Path):
    artifact = tmp_path / "artifact.json"
    split = tmp_path / "split.jsonl"
    seal = tmp_path / "seal.json"
    artifact.write_text('{"holdout_state":"sealed"}', encoding="utf-8")
    split.write_text("{}\n", encoding="utf-8")
    claim = claim_final_holdout(
        seal,
        protocol_id="protocol-v1",
        artifact_manifest=artifact,
        split_manifest=split,
    )
    artifact.write_text('{"holdout_state":"tampered"}', encoding="utf-8")
    with pytest.raises(EvidenceProtocolError, match="changed after claim"):
        _verify_claim_inputs(claim)


def test_vnext_promotion_requires_bound_runtime_and_formal_five_seed_stage5(tmp_path: Path):
    manifest_path = tmp_path / "candidate_manifest.json"
    runtime_contract = canonical_formal_runtime_contract(
        observation={
            "enabled": True,
            "feature_version": "risk_gated_candidate_table_v3_bounded_stale",
            "activation_distance": 240.0,
            "include_risk_secondary": True,
            "secondary_safety_profile": "short_horizon_risk_v1",
            "risk_horizon_steps": 8,
            "invalid_table_strategy": "bounded_last_valid_v2",
            "fail_closed_defaults": True,
            "timeout_s": 0.5,
            "timeout_contract": "soft_realtime_post_return_v1",
            "full_table_hard_deadline_worker": False,
            "use_inference_worker": False,
            "profile_latency": True,
            "warmup_enabled": True,
            "warmup_max_attempts": 3,
            "invalid_table_dropout_rate": 0.0,
            "last_valid_max_decisions": 1,
            "last_valid_ttl_s": 0.5,
            "last_valid_max_merge_distance_delta_m": 15.0,
            "last_valid_max_ego_speed_delta_mps": 3.0,
            "last_valid_max_gap_delta_m": 8.0,
        },
        candidate_geometry_backend="vectorized",
        risk_checkpoint_sha256="2" * 64,
        risk_module_config_sha256="3" * 64,
    )
    runtime_contract_sha = formal_runtime_contract_sha256(runtime_contract)
    semantic_contract_sha = candidate_table_semantic_contract_sha256(
        runtime_contract
    )
    method_effect_execution_sha = closed_loop_execution_contract_sha256(
        {
            "execution_contract": SIMULATION_BLOCKING_EXACT_CONTRACT,
            "deployment_deadline_s": 0.5,
            "profile_latency": True,
            "use_inference_worker": False,
            "invalid_table_strategy": "fail_closed_v1",
            "fail_closed_defaults": True,
            "invalid_table_dropout_rate": 0.0,
        }
    )
    manifest = {
        "artifact_kind": "accvp_artifact_bundle",
        "artifact_variant": "viability_lite_task_v1",
        "artifact_fingerprint": "a" * 64,
        "evidence_protocol_id": "protocol-vnext",
        "formal_runtime_contract": runtime_contract,
        "formal_runtime_contract_sha256": runtime_contract_sha,
        "deployment_runtime_contract_sha256": runtime_contract_sha,
        "candidate_table_semantic_contract_sha256": semantic_contract_sha,
        "policy_method_effect_execution_contract_sha256": (
            method_effect_execution_sha
        ),
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    runtime = {
        "artifact_kind": "accvp_runtime_benchmark_v1",
        "schema_version": 2,
        "benchmark_scope": "policy_runtime",
        "policy_type": "sb3_ppo",
        "backend": "vectorized",
        "gate": {
            "pass": True,
            "profile": "bounded_stale_runtime_v3_strict",
            "checks": {"formal_runtime_contract_match": True},
        },
        "formal_runtime_contract": runtime_contract,
        "formal_runtime_contract_sha256": runtime_contract_sha,
        "deployment_runtime_contract_sha256": runtime_contract_sha,
        "candidate_table_semantic_contract_sha256": semantic_contract_sha,
        "policy_method_effect_execution_contract_sha256": (
            method_effect_execution_sha
        ),
        "metrics": {
            "accvp_table_unique_episode_seed_count": 30,
            "accvp_table_activation_window_decision_count": 1000,
            "accvp_table_seed_schedule_match": True,
        },
        "artifact_lineage": {
            "accvp_manifest": {"sha256": file_sha256(manifest_path)},
            "policy_model_sha256": f"{1:064x}",
        },
    }
    runtime["report_fingerprint"] = stable_hash(runtime)
    runtime_path = tmp_path / "runtime.json"
    runtime_path.write_text(json.dumps(runtime), encoding="utf-8")
    statistics = {
        "balanced_matrix": True,
        "method": "crossed_training_simulator_paired_percentile_bootstrap",
        "training_seed_count": 5,
        "training_seeds": [1, 2, 3, 4, 5],
        "simulator_seed_count": 3,
        "simulator_seeds": [101, 102, 103],
        "checkpoint_records": [
            {
                "training_seed": seed,
                "left_checkpoint_sha256": f"{seed:064x}",
                "right_checkpoint_sha256": f"{seed + 5:064x}",
            }
            for seed in [1, 2, 3, 4, 5]
        ],
    }
    statistics["statistics_fingerprint"] = stable_hash(statistics)
    candidate_checkpoint_records = [
        {
            "training_seed": seed,
            "group": "candidate",
            "checkpoint_sha256": f"{seed:064x}",
        }
        for seed in [1, 2, 3, 4, 5]
    ]
    candidate_binding = {
        "required": True,
        "candidate_manifest": {
            "sha256": file_sha256(manifest_path),
            "artifact_fingerprint": manifest["artifact_fingerprint"],
            "artifact_variant": manifest["artifact_variant"],
            "formal_runtime_contract_sha256": runtime_contract_sha,
        },
        "candidate_side": "left",
        "source_acceptance_key": "candidate_vs_baseline",
        "formal_runtime_contract_sha256": runtime_contract_sha,
        "deployment_runtime_contract_sha256": runtime_contract_sha,
        "candidate_table_semantic_contract_sha256": semantic_contract_sha,
        "closed_loop_execution_contract_sha256": method_effect_execution_sha,
        "candidate_checkpoint_records": candidate_checkpoint_records,
        "candidate_checkpoint_matrix_sha256": stable_hash(
            candidate_checkpoint_records
        ),
    }
    stage5 = {
        "artifact_kind": "stage5_replicated_paired_report_v1",
        "schema_version": 2,
        "formal_aggregation": True,
        "minimum_training_seed_count": 5,
        "statistics": statistics,
        "lineage": {"protocol_id": "protocol-vnext"},
        "candidate_binding": candidate_binding,
        "gate": {
            "profile": "formal_candidate_promotion_binding_v1",
            "pass": True,
            "checks": {
                "candidate_bundle_binding_satisfied": True,
                "candidate_side_bound": True,
                "formal_runtime_contract_bound": True,
                "candidate_semantic_contract_bound": True,
                "blocking_exact_execution_contract_bound": True,
                "all_source_acceptance_passed": True,
                "preregistered_metric_output_nonempty": True,
                "balanced_crossed_statistics": True,
            },
        },
    }
    stage5["report_fingerprint"] = stable_hash(stage5)
    stage5_path = tmp_path / "stage5.json"
    stage5_path.write_text(json.dumps(stage5), encoding="utf-8")

    evidence = _validate_vnext_promotion_evidence(
        manifest=manifest,
        manifest_path=manifest_path,
        runtime_benchmark_path=runtime_path,
        stage5_replicated_report_path=stage5_path,
        final_runtime_contract=runtime_contract,
    )
    assert evidence["runtime_benchmark"]["report_fingerprint"]
    assert evidence["stage5_replicated_report"]["training_seed_count"] == 5

    runtime_rows = []
    for seed in [1, 2, 3, 4, 5]:
        individual = json.loads(json.dumps(runtime))
        checkpoint_sha = f"{seed:064x}"
        individual["policy_model_sha256"] = checkpoint_sha
        individual["artifact_lineage"]["policy_model_sha256"] = checkpoint_sha
        individual["report_fingerprint"] = stable_hash(
            {
                key: value
                for key, value in individual.items()
                if key != "report_fingerprint"
            }
        )
        individual_path = tmp_path / f"runtime_{seed}.json"
        individual_path.write_text(json.dumps(individual), encoding="utf-8")
        runtime_rows.append(
            {
                "optimizer_seed": seed,
                "checkpoint_sha256": checkpoint_sha,
                "report": str(individual_path),
                "report_sha256": file_sha256(individual_path),
            }
        )
    replicated_runtime = {
        "artifact_kind": "accvp_runtime_benchmark_replicates_v1",
        "schema_version": 1,
        "replicates": runtime_rows,
        "gate": {"pass": True},
    }
    replicated_runtime["report_fingerprint"] = stable_hash(replicated_runtime)
    replicated_runtime_path = tmp_path / "runtime_replicated.json"
    replicated_runtime_path.write_text(
        json.dumps(replicated_runtime),
        encoding="utf-8",
    )
    replicated_evidence = _validate_vnext_promotion_evidence(
        manifest=manifest,
        manifest_path=manifest_path,
        runtime_benchmark_path=replicated_runtime_path,
        stage5_replicated_report_path=stage5_path,
        final_runtime_contract=runtime_contract,
    )
    assert replicated_evidence["runtime_benchmark"]["optimizer_replicate_count"] == 5

    wrong_checkpoint = f"{99:064x}"
    wrong_individual = json.loads((tmp_path / "runtime_5.json").read_text(encoding="utf-8"))
    wrong_individual["policy_model_sha256"] = wrong_checkpoint
    wrong_individual["artifact_lineage"]["policy_model_sha256"] = wrong_checkpoint
    wrong_individual["report_fingerprint"] = stable_hash(
        {
            key: value
            for key, value in wrong_individual.items()
            if key != "report_fingerprint"
        }
    )
    wrong_path = tmp_path / "runtime_wrong.json"
    wrong_path.write_text(json.dumps(wrong_individual), encoding="utf-8")
    replicated_runtime["replicates"][-1] = {
        "optimizer_seed": 5,
        "checkpoint_sha256": wrong_checkpoint,
        "report": str(wrong_path),
        "report_sha256": file_sha256(wrong_path),
    }
    replicated_runtime["report_fingerprint"] = stable_hash(
        {
            key: value
            for key, value in replicated_runtime.items()
            if key != "report_fingerprint"
        }
    )
    replicated_runtime_path.write_text(
        json.dumps(replicated_runtime),
        encoding="utf-8",
    )
    with pytest.raises(EvidenceProtocolError, match="does not exactly match"):
        _validate_vnext_promotion_evidence(
            manifest=manifest,
            manifest_path=manifest_path,
            runtime_benchmark_path=replicated_runtime_path,
            stage5_replicated_report_path=stage5_path,
            final_runtime_contract=runtime_contract,
        )

    invalid_stage5 = json.loads(json.dumps(stage5))
    invalid_stage5["gate"]["pass"] = False
    invalid_stage5["report_fingerprint"] = stable_hash(
        {
            key: value
            for key, value in invalid_stage5.items()
            if key != "report_fingerprint"
        }
    )
    stage5_path.write_text(json.dumps(invalid_stage5), encoding="utf-8")
    with pytest.raises(EvidenceProtocolError, match="promotion gate did not pass"):
        _validate_vnext_promotion_evidence(
            manifest=manifest,
            manifest_path=manifest_path,
            runtime_benchmark_path=runtime_path,
            stage5_replicated_report_path=stage5_path,
            final_runtime_contract=runtime_contract,
        )
    stage5_path.write_text(json.dumps(stage5), encoding="utf-8")

    runtime["gate"]["pass"] = False
    runtime["report_fingerprint"] = stable_hash(
        {key: value for key, value in runtime.items() if key != "report_fingerprint"}
    )
    runtime_path.write_text(json.dumps(runtime), encoding="utf-8")
    failed_runtime_evidence = _validate_vnext_promotion_evidence(
        manifest=manifest,
        manifest_path=manifest_path,
        runtime_benchmark_path=runtime_path,
        stage5_replicated_report_path=stage5_path,
        final_runtime_contract=runtime_contract,
    )
    assert failed_runtime_evidence["runtime_benchmark"][
        "deployment_runtime_gate_pass"
    ] is False


def test_risk_secondary_combiner_never_selects_test_profile():
    test_profile = {
        "secondary_safety_profile": "audited_merge_left_v1",
        "max_secondary_risk_score": 0.9,
        "split": "test",
    }
    report = combine_audit_reports(
        {
            "operating_point": {"selected_audited_profile": None},
            "test": {"selected_audited_profile": test_profile},
        }
    )
    assert report["selected_audited_profile"] is None
    assert report["test_used_for_selection"] is False
    operating_profile = {**test_profile, "split": "operating_point", "max_secondary_risk_score": 0.2}
    report = combine_audit_reports(
        {
            "operating_point": {"selected_audited_profile": operating_profile},
            "test": {"selected_audited_profile": test_profile},
        }
    )
    assert report["selected_audited_profile"] == operating_profile


def test_lite_tuning_rejects_test_split_before_search():
    with pytest.raises(ValueError, match="operating_point"):
        tune_viability_lite_operating_point([], _Config({"accvp": {}}), split="test")


def test_revocation_manifest_blocks_matching_artifact(tmp_path: Path):
    artifact = tmp_path / "model.pt"
    artifact.write_bytes(b"revoked-model")
    manifest = tmp_path / "revocations.json"
    write_artifact_revocation_manifest(manifest, [artifact], reason="leaked holdout")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["artifacts"][0]["sha256"]
    with pytest.raises(ValueError, match="revoked"):
        assert_artifact_not_revoked(artifact, manifest)


def test_formal_runtime_requires_successfully_evaluated_deployable_manifest(tmp_path: Path):
    checkpoint = tmp_path / "predictor.pt"
    calibration = tmp_path / "calibration.json"
    manifest_path = tmp_path / "manifest.json"
    checkpoint.write_bytes(b"predictor")
    calibration.write_text("{}", encoding="utf-8")

    class _RuntimeConfig:
        def __init__(self):
            self.accvp = {
                "mode": "viability_lite",
                "artifact_manifest": str(manifest_path),
                "calibration_bundle": str(calibration),
                "viability_lite": {"secondary_safety_profile": "strict"},
            }

    runtime = ACCVPRuntimePredictor.__new__(ACCVPRuntimePredictor)
    runtime.config = _RuntimeConfig()
    runtime.checkpoint_path = checkpoint
    base = {
        "artifact_kind": "accvp_v1_lite_task_artifact_bundle",
        "deployable_claim": "task_viability_only",
        "accvp_safety_head_hard_gate": False,
        "secondary_safety_profile": "strict",
    }
    manifest_path.write_text(
        json.dumps({**base, "deployable_artifact": False, "holdout_state": "sealed"}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="deployable_artifact=true"):
        runtime.validate_artifact_bundle(None)

    manifest_path.write_text(
        json.dumps(
            {
                **base,
                "deployable_artifact": True,
                "holdout_state": "evaluated",
                "holdout_decision": "no_go",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="holdout_decision='go'"):
        runtime.validate_artifact_bundle(None)

    manifest_path.write_text(
        json.dumps(
            {
                **base,
                "deployable_artifact": True,
                "holdout_state": "evaluated",
                "holdout_decision": "go",
                "predictor_sha256": "deliberately-wrong",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="bundle mismatch for predictor_sha256"):
        runtime.validate_artifact_bundle(None)


def test_stage3_and_stage5_bind_same_accvp_manifest_identity(tmp_path, monkeypatch):
    manifest_path = tmp_path / "candidate_manifest.json"
    predictor_path = tmp_path / "predictor.pt"
    manifest_path.write_text("{}", encoding="utf-8")
    predictor_path.write_bytes(b"predictor")
    bundle = {
        "artifact_fingerprint": "1" * 64,
        "artifact_variant": "full_candidate_gate_v1",
        "artifact_generation": "vnext_schema3",
        "bundle_schema_version": 2,
        "formal_runtime_contract_sha256": "2" * 64,
    }
    resolved = {"predictor": predictor_path}
    monkeypatch.setattr(
        stage3_train_ppo,
        "resolve_v2_bundle",
        lambda _path: (bundle, resolved),
    )
    monkeypatch.setattr(
        stage5_paired_eval,
        "resolve_v2_bundle",
        lambda _path: (bundle, resolved),
    )

    class Config:
        accvp = {
            "artifact_manifest": str(manifest_path),
            "observation": {"enabled": True},
        }

    stage3_binding = stage3_train_ppo._accvp_bundle_lineage(Config())
    stage5_binding = stage5_paired_eval._accvp_bundle_lineage(Config())
    assert stage3_binding == stage5_binding
    content = dict(stage3_binding)
    fingerprint = content.pop("binding_fingerprint")
    assert fingerprint == stable_hash(content)


def test_stage5_rejects_stage3_accvp_bundle_lineage_mismatch(tmp_path, monkeypatch):
    model_path = tmp_path / "ppo_model.zip"
    model_path.write_bytes(b"ppo")

    class Config:
        accvp = {
            "artifact_manifest": "candidate.json",
            "observation": {
                "enabled": True,
                "feature_version": RiskGatedACCVPCandidateTableAugmentor.FEATURE_VERSION,
            },
        }

    expected = {
        "required": True,
        "enabled": True,
        "manifest_path": str((tmp_path / "candidate.json").resolve()),
        "manifest_sha256": "3" * 64,
        "artifact_fingerprint": "4" * 64,
        "artifact_variant": "full_candidate_gate_v1",
        "artifact_generation": "vnext_schema3",
        "bundle_schema_version": 2,
        "formal_runtime_contract_sha256": "5" * 64,
        "predictor_sha256": "6" * 64,
    }
    expected["binding_fingerprint"] = stable_hash(expected)
    monkeypatch.setattr(
        stage5_paired_eval,
        "_accvp_bundle_lineage",
        lambda _cfg: dict(expected),
    )
    recorded = dict(expected)
    recorded["formal_runtime_contract_sha256"] = "7" * 64
    recorded.pop("binding_fingerprint")
    recorded["binding_fingerprint"] = stable_hash(recorded)
    parent = {"accvp_bundle": recorded}
    parent["lineage_fingerprint"] = stable_hash(parent)
    report = {
        "accvp_observation_feature_names_sha256": stable_hash(
            RiskGatedACCVPCandidateTableAugmentor.feature_names(Config())
        ),
        "accvp_bundle_lineage": recorded,
        "evidence_lineage": parent,
    }
    (tmp_path / "stage3_training_report.json").write_text(
        json.dumps(report), encoding="utf-8"
    )
    with pytest.raises(EvidenceProtocolError, match="bundle lineage mismatch"):
        stage5_paired_eval._validate_stage5_observation_contract(
            model_path=model_path,
            group_cfg=Config(),
            protocol_strict=True,
        )
