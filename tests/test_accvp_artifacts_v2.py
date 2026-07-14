from __future__ import annotations

import json
from pathlib import Path

import pytest

from safe_rl.accvp.contracts.artifacts import (
    ACCVP_ARTIFACT_GENERATION,
    ACCVP_ARTIFACT_KIND,
    ACCVP_BUNDLE_SCHEMA_VERSION,
    FINAL_HOLDOUT_LIFECYCLE_AUTHORIZATION,
    LIFECYCLE_HOLDOUT_GO,
    LIFECYCLE_HOLDOUT_NO_GO,
    LIFECYCLE_REVOKED,
    LIFECYCLE_SEALED_CANDIDATE,
    apply_v2_bundle_paths,
    artifact_filename,
    bundle_file_entry,
    resolve_v2_bundle,
    validate_lifecycle_for_mode,
)
from safe_rl.accvp.contracts.schema import file_sha256, stable_hash
from safe_rl.accvp.contracts.runtime_contract import (
    canonical_formal_runtime_contract,
    formal_runtime_contract_sha256,
)
from safe_rl.accvp.planning.viability_lite import write_lite_artifacts
from safe_rl.pipeline.accvp_final_holdout_eval import (
    _validate_operating_point_schema,
    _validated_manifest,
)
from safe_rl.utils.config import clone_with_overrides, load_config


def _bundle(tmp_path: Path) -> Path:
    files = {}
    for key, name in (
        ("predictor", "predictor.pt"),
        ("calibration", "calibration.json"),
        ("operating_point", "operating.json"),
        ("training_history", "history.json"),
    ):
        path = tmp_path / name
        if key == "operating_point":
            path.write_text(
                json.dumps(
                    {
                        "split": "operating_point",
                        "selected": {
                            "proxy_collision_upper_bound": 0.2,
                            "safety_violation_upper_bound": 0.2,
                            "merge_viability_lower_bound": 0.7,
                        },
                    }
                ),
                encoding="utf-8",
            )
        else:
            path.write_text(f"{key}\n", encoding="utf-8")
        files[key] = bundle_file_entry(path, manifest_dir=tmp_path)
    formal_runtime_contract = canonical_formal_runtime_contract(
        observation={
            "enabled": True,
            "feature_version": "risk_gated_candidate_table_v3_bounded_stale",
            "activation_distance": 240.0,
            "include_risk_secondary": True,
            "secondary_safety_profile": "risk_model_rollout_v1",
            "risk_horizon_steps": 30,
            "invalid_table_strategy": "bounded_last_valid_v2",
            "fail_closed_defaults": True,
            "timeout_s": 0.5,
            "timeout_contract": "soft_realtime_post_return_v1",
            "full_table_hard_deadline_worker": False,
            "use_inference_worker": False,
            "profile_latency": True,
            "warmup_enabled": True,
            "warmup_max_attempts": 1,
            "invalid_table_dropout_rate": 0.0,
            "last_valid_max_decisions": 1,
            "last_valid_ttl_s": 0.5,
            "last_valid_max_merge_distance_delta_m": 15.0,
            "last_valid_max_ego_speed_delta_mps": 3.0,
            "last_valid_max_gap_delta_m": 8.0,
        },
        candidate_geometry_backend="vectorized",
        risk_checkpoint_sha256="9" * 64,
        risk_module_config_sha256="8" * 64,
    )
    manifest = {
        "artifact_kind": ACCVP_ARTIFACT_KIND,
        "bundle_schema_version": ACCVP_BUNDLE_SCHEMA_VERSION,
        "artifact_generation": ACCVP_ARTIFACT_GENERATION,
        "artifact_variant": "full_candidate_gate_v1",
        "lifecycle_state": LIFECYCLE_SEALED_CANDIDATE,
        "capabilities": ["candidate_table_observation"],
        "deployable_artifact": False,
        "holdout_state": "sealed",
        "predictor_sha256": files["predictor"]["sha256"],
        "calibration_sha256": files["calibration"]["sha256"],
        "operating_point_sha256": files["operating_point"]["sha256"],
        "training_history_sha256": files["training_history"]["sha256"],
        "formal_runtime_contract": formal_runtime_contract,
        "formal_runtime_contract_sha256": formal_runtime_contract_sha256(
            formal_runtime_contract
        ),
        "files": files,
    }
    manifest["artifact_fingerprint"] = stable_hash(manifest)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def _promotion_evidence() -> dict:
    return {
        "runtime_benchmark": {
            "path": "runtime.json",
            "sha256": "1" * 64,
            "report_fingerprint": "2" * 64,
        },
        "stage5_replicated_report": {
            "path": "stage5.json",
            "sha256": "3" * 64,
            "report_fingerprint": "4" * 64,
        },
    }


def _authorise_holdout(manifest: dict, *, decision: str) -> None:
    manifest.update(
        {
            "lifecycle_authorization": FINAL_HOLDOUT_LIFECYCLE_AUTHORIZATION,
            "source_frozen_manifest": "source.json",
            "source_frozen_manifest_sha256": "5" * 64,
            "final_test_diagnostics_path": "result.json",
            "final_test_diagnostics_sha256": "6" * 64,
            "holdout_claim_fingerprint": "7" * 64,
            "promotion_evidence": _promotion_evidence(),
            "holdout_decision": decision,
        }
    )


def test_bundle_v2_is_manifest_relative_and_source_of_truth(tmp_path: Path):
    manifest_path = _bundle(tmp_path)
    manifest, files = resolve_v2_bundle(manifest_path)
    assert manifest["artifact_generation"] == ACCVP_ARTIFACT_GENERATION
    assert files["predictor"] == (tmp_path / "predictor.pt").resolve()

    cfg = clone_with_overrides(
        load_config(),
        {"accvp": {"artifact_manifest": str(manifest_path)}},
    )
    apply_v2_bundle_paths(cfg)
    assert Path(cfg.accvp.checkpoint) == files["predictor"]
    assert Path(cfg.accvp.calibration_bundle) == files["calibration"]
    assert Path(cfg.accvp.operating_point) == files["operating_point"]
    assert cfg.accvp.artifact_generation == ACCVP_ARTIFACT_GENERATION


def test_bundle_v2_rejects_configured_generation_mismatch(tmp_path: Path):
    manifest_path = _bundle(tmp_path)
    cfg = clone_with_overrides(
        load_config(),
        {
            "accvp": {
                "artifact_manifest": str(manifest_path),
                "artifact_generation": "legacy_generation",
            }
        },
    )
    with pytest.raises(ValueError, match="artifact_generation mismatch"):
        apply_v2_bundle_paths(cfg)


def test_bundle_v2_cannot_downgrade_its_kind_to_legacy(tmp_path: Path):
    manifest_path = _bundle(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifact_kind"] = "accvp_v1_shadow_artifact_bundle"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    cfg = clone_with_overrides(
        load_config(),
        {"accvp": {"artifact_manifest": str(manifest_path)}},
    )
    with pytest.raises(ValueError, match="cannot downgrade"):
        apply_v2_bundle_paths(cfg)


def test_bundle_v2_rejects_tampered_file(tmp_path: Path):
    manifest_path = _bundle(tmp_path)
    (tmp_path / "calibration.json").write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        resolve_v2_bundle(manifest_path)


def test_bundle_v2_rejects_tampered_formal_runtime_contract(tmp_path: Path):
    manifest_path = _bundle(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["formal_runtime_contract"]["risk_horizon_steps"] = 1
    manifest["artifact_fingerprint"] = stable_hash(
        {key: value for key, value in manifest.items() if key != "artifact_fingerprint"}
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="formal_runtime_contract_sha256 mismatch"):
        resolve_v2_bundle(manifest_path)


def test_bundle_v2_lifecycle_separates_shadow_from_deployment(tmp_path: Path):
    manifest_path = _bundle(tmp_path)
    manifest, _files = resolve_v2_bundle(manifest_path)
    validate_lifecycle_for_mode(manifest, "off")
    validate_lifecycle_for_mode(manifest, "shadow")
    with pytest.raises(ValueError, match="artifact_variant"):
        validate_lifecycle_for_mode(manifest, "viability_lite_shadow")
    with pytest.raises(ValueError, match="holdout_evaluated_go"):
        validate_lifecycle_for_mode(manifest, "viability_branch")
    with pytest.raises(ValueError, match="holdout_evaluated_go"):
        validate_lifecycle_for_mode(manifest, "viability_lite")

    manifest["lifecycle_state"] = LIFECYCLE_HOLDOUT_GO
    manifest["deployable_artifact"] = True
    manifest["holdout_state"] = "evaluated"
    manifest["holdout_decision"] = "go"
    _authorise_holdout(manifest, decision="go")
    validate_lifecycle_for_mode(manifest, "viability_branch")
    with pytest.raises(ValueError, match="artifact_variant"):
        validate_lifecycle_for_mode(manifest, "viability_lite")
    manifest["artifact_variant"] = "viability_lite_task_v1"
    manifest["deployable_claim"] = "task_viability_only"
    manifest["accvp_safety_head_hard_gate"] = False
    validate_lifecycle_for_mode(manifest, "viability_lite")


def test_bundle_v2_requires_frozen_training_history(tmp_path: Path):
    manifest_path = _bundle(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"].pop("training_history")
    manifest["artifact_fingerprint"] = stable_hash(
        {key: value for key, value in manifest.items() if key != "artifact_fingerprint"}
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="training_history"):
        resolve_v2_bundle(manifest_path)


def test_bundle_v2_rejects_self_declared_go_without_final_holdout_authorization(
    tmp_path: Path,
):
    manifest_path = _bundle(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "lifecycle_state": LIFECYCLE_HOLDOUT_GO,
            "deployable_artifact": True,
            "holdout_state": "evaluated",
            "holdout_decision": "go",
        }
    )
    manifest.pop("artifact_fingerprint")
    manifest["artifact_fingerprint"] = stable_hash(manifest)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="lifecycle_authorization"):
        resolve_v2_bundle(manifest_path)


def test_lite_shadow_bundle_cannot_omit_bound_operating_point(tmp_path: Path):
    manifest_path = _bundle(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "artifact_variant": "viability_lite_task_v1",
            "lifecycle_state": "shadow",
            "deployable_artifact": False,
            "holdout_state": "not_applicable",
            "operating_point_schema": "accvp_viability_lite_operating_point_v1",
            "deployable_claim": "task_viability_only",
            "accvp_safety_head_hard_gate": False,
            "decision_weighting": {
                "decision_weighting_version": (
                    "fingerprint_raw_action_total_weight_one_v2"
                )
            },
        }
    )
    manifest["files"].pop("operating_point")
    manifest.pop("operating_point_sha256")
    manifest.pop("artifact_fingerprint")
    manifest["artifact_fingerprint"] = stable_hash(manifest)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="operating_point|operating point"):
        resolve_v2_bundle(manifest_path)


def test_bundle_v2_rejects_manifest_tampering_even_when_files_are_unchanged(tmp_path: Path):
    manifest_path = _bundle(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["deployable_artifact"] = True
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        resolve_v2_bundle(manifest_path)


def test_bundle_v2_rejects_internally_consistent_manifest_with_wrong_top_level_hash(tmp_path: Path):
    manifest_path = _bundle(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["calibration_sha256"] = "0" * 64
    manifest.pop("artifact_fingerprint")
    manifest["artifact_fingerprint"] = stable_hash(manifest)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="top-level hash mismatch"):
        resolve_v2_bundle(manifest_path)


def test_bundle_v2_rejects_variant_operating_point_schema_mismatch(tmp_path: Path):
    manifest_path = _bundle(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifact_variant"] = "viability_lite_task_v1"
    manifest["operating_point_schema"] = "accvp_viability_lite_operating_point_v1"
    manifest["decision_weighting"] = {
        "decision_weighting_version": "fingerprint_raw_action_total_weight_one_v2"
    }
    manifest.pop("artifact_fingerprint")
    manifest["artifact_fingerprint"] = stable_hash(manifest)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="decision-weighting|missing thresholds"):
        resolve_v2_bundle(manifest_path)


@pytest.mark.parametrize("lifecycle", [LIFECYCLE_HOLDOUT_NO_GO, LIFECYCLE_REVOKED])
def test_bundle_v2_no_go_and_revoked_are_not_runnable(tmp_path: Path, lifecycle: str):
    manifest_path = _bundle(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["lifecycle_state"] = lifecycle
    manifest["deployable_artifact"] = False
    if lifecycle == LIFECYCLE_HOLDOUT_NO_GO:
        manifest["holdout_state"] = "evaluated"
        manifest["holdout_decision"] = "no_go"
        _authorise_holdout(manifest, decision="no_go")
    for mode in ("off", "shadow", "viability_lite"):
        with pytest.raises(ValueError, match="not runnable"):
            validate_lifecycle_for_mode(manifest, mode)


def test_legacy_v1_manifest_remains_read_only_and_does_not_supply_paths(tmp_path: Path):
    legacy = tmp_path / "legacy_manifest.json"
    legacy.write_text(json.dumps({"artifact_kind": "accvp_v1_shadow_artifact_bundle"}), encoding="utf-8")
    cfg = clone_with_overrides(
        load_config(),
        {"accvp": {"artifact_manifest": str(legacy), "checkpoint": "legacy.pt"}},
    )
    payload, resolved = apply_v2_bundle_paths(cfg)
    assert payload == {"artifact_kind": "accvp_v1_shadow_artifact_bundle"}
    assert resolved == {}
    assert cfg.accvp.checkpoint == "legacy.pt"


def test_generation_aware_vnext_config_rejects_legacy_v1_manifest(tmp_path: Path):
    legacy = tmp_path / "legacy_manifest.json"
    legacy.write_text(
        json.dumps({"artifact_kind": "accvp_v1_shadow_artifact_bundle"}),
        encoding="utf-8",
    )
    cfg = clone_with_overrides(
        load_config(),
        {
            "accvp": {
                "artifact_manifest": str(legacy),
                "artifact_generation": ACCVP_ARTIFACT_GENERATION,
                "checkpoint": "legacy.pt",
            }
        },
    )
    with pytest.raises(ValueError, match="cannot load a legacy"):
        apply_v2_bundle_paths(cfg)


def test_validated_bundle_rebases_relative_files_and_preserves_training_history(tmp_path: Path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source = _bundle(source_dir)
    result = tmp_path / "final_result.json"
    result.write_text('{"decision":"go"}', encoding="utf-8")
    target = tmp_path / "validated" / "validated_manifest.json"
    target.parent.mkdir()
    payload = _validated_manifest(
        json.loads(source.read_text(encoding="utf-8")),
        source_path=source,
        result_path=result,
        seal={"final_fingerprint": "8" * 64},
        decision="go",
        target_path=target,
        promotion_evidence=_promotion_evidence(),
    )
    target.write_text(json.dumps(payload), encoding="utf-8")
    manifest, resolved = resolve_v2_bundle(target)
    assert manifest["lifecycle_state"] == LIFECYCLE_HOLDOUT_GO
    assert manifest["holdout_decision"] == "go"
    assert (
        manifest["lifecycle_authorization"]
        == FINAL_HOLDOUT_LIFECYCLE_AUTHORIZATION
    )
    assert resolved["training_history"] == (source_dir / "history.json").resolve()
    assert not Path(manifest["files"]["training_history"]["path"]).is_absolute()


def test_vnext_lite_tuning_derives_a_separate_sealed_bundle(tmp_path: Path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source = _bundle(source_dir)
    dataset = tmp_path / "dataset"
    manifests = dataset / "manifests"
    manifests.mkdir(parents=True)
    dataset_manifest = manifests / "dataset_manifest.json"
    split_manifest = manifests / "split_manifest.jsonl"
    split_provenance = manifests / "split_provenance.json"
    dataset_manifest.write_text("{}", encoding="utf-8")
    split_manifest.write_text("{}\n", encoding="utf-8")
    split_provenance.write_text("{}", encoding="utf-8")
    source_payload = json.loads(source.read_text(encoding="utf-8"))
    source_payload.update(
        {
            "dataset_manifest_sha256": file_sha256(dataset_manifest),
            "split_manifest_sha256": file_sha256(split_manifest),
            "split_provenance_sha256": file_sha256(split_provenance),
        }
    )
    source_payload.pop("artifact_fingerprint")
    source_payload["artifact_fingerprint"] = stable_hash(source_payload)
    source.write_text(json.dumps(source_payload), encoding="utf-8")
    cfg = clone_with_overrides(
        load_config(),
        {
            "accvp": {
                "artifact_generation": ACCVP_ARTIFACT_GENERATION,
                "artifact_manifest": str(source),
                "viability_lite": {"secondary_safety_profile": "strict"},
            }
        },
    )
    operating_point = {
        "split": "operating_point",
        "decision_weighting": {
            "decision_weighting_version": "fingerprint_raw_action_total_weight_one_v2",
            "raw_candidate_row_count": 2,
            "effective_candidate_row_count": 1,
            "effective_decision_count": 1,
            "statistical_independence_claim": False,
        },
        "selected": {
            "min_p_merge_before_taper": 0.75,
            "min_improvement_over_raw": 0.01,
            "max_target_entry_time_s": 6.0,
            "max_ensemble_disagreement": 0.1,
            "max_secondary_risk_score": 0.2,
            "secondary_safety_profile": "strict",
        },
    }
    output = tmp_path / "lite"
    paths = write_lite_artifacts(
        output_dir=output,
        config=cfg,
        dataset_dir=dataset,
        checkpoint=source_dir / "predictor.pt",
        calibration=source_dir / "calibration.json",
        operating_point=operating_point,
    )
    assert paths["operating_point"].name == artifact_filename("lite_operating_point")
    assert paths["artifact_manifest"].name == artifact_filename("lite_candidate_manifest")
    lite_manifest, files = resolve_v2_bundle(paths["artifact_manifest"])
    assert lite_manifest["artifact_variant"] == "viability_lite_task_v1"
    assert lite_manifest["source_candidate_manifest_sha256"] == file_sha256(source)
    assert "source_candidate_manifest" not in lite_manifest
    assert lite_manifest["source_candidate_manifest_reference"] == {
        "reference_kind": "digest_only_v1",
        "sha256": file_sha256(source),
        "artifact_fingerprint": source_payload["artifact_fingerprint"],
        "artifact_variant": "full_candidate_gate_v1",
    }
    assert files["training_history"] == (source_dir / "history.json").resolve()
    _validate_operating_point_schema(lite_manifest, operating_point, mode="lite")
    with pytest.raises(ValueError, match="artifact_variant"):
        _validate_operating_point_schema(source_payload, operating_point, mode="lite")
