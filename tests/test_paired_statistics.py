from __future__ import annotations

import json
from pathlib import Path

import pytest

from safe_rl.accvp.artifacts import (
    ACCVP_ARTIFACT_GENERATION,
    ACCVP_ARTIFACT_KIND,
    ACCVP_BUNDLE_SCHEMA_VERSION,
    LIFECYCLE_SEALED_CANDIDATE,
    bundle_file_entry,
)
from safe_rl.accvp.runtime_contract import (
    canonical_formal_runtime_contract,
    formal_runtime_contract_sha256,
)
from safe_rl.accvp.schema import file_sha256, stable_hash
from safe_rl.analysis.paired_statistics import (
    build_pair_statistics,
    build_replicated_pair_statistics,
    crossed_paired_bootstrap_mean_ci,
    exact_binomial_upper_bound,
    exact_mcnemar_pvalue,
    hierarchical_paired_bootstrap_mean_ci,
    holm_adjust,
)
from safe_rl.pipeline.stage5_paired_eval import _build_paired_statistics
from safe_rl.pipeline.stage5_replicated_aggregate import (
    REQUEST_ARTIFACT_KIND,
    aggregate_manifest,
    run as run_replicated_aggregate,
)


def _report(rewards, collisions):
    return {
        "episodes": [
            {
                "seed": index + 1,
                "episode_reward": reward,
                "proxy_collision": collision,
                "merge_success": reward > 0,
            }
            for index, (reward, collision) in enumerate(zip(rewards, collisions))
        ]
    }


def _replicate_pair(training_seed, left_rewards, right_rewards, left_collisions, right_collisions, left_hash, right_hash):
    left = _report(left_rewards, left_collisions)
    right = _report(right_rewards, right_collisions)
    left["comparative"] = {"training_seed": training_seed}
    right["comparative"] = {"training_seed": training_seed}
    return {
        "training_seed": training_seed,
        "left_checkpoint_sha256": left_hash,
        "right_checkpoint_sha256": right_hash,
        "left_report": left,
        "right_report": right,
    }


def test_pair_statistics_preserve_seed_pairing_and_direction():
    baseline = _report([1.0, 2.0, 3.0, 4.0], [False, False, True, False])
    candidate = _report([2.0, 3.0, 4.0, 5.0], [False, True, False, False])
    result = build_pair_statistics(baseline, candidate, replicates=200, seed=7)
    reward = result["continuous"]["episode_reward"]
    assert result["paired_seed_count"] == 4
    assert reward["mean_delta"] == pytest.approx(1.0)
    assert reward["ci_low"] == pytest.approx(1.0)
    collision = result["binary"]["proxy_collision"]
    assert collision["candidate_only_events"] == 1
    assert collision["baseline_only_events"] == 1
    assert collision["risk_difference"] == pytest.approx(0.0)


def test_exact_binary_helpers_cover_zero_and_discordant_cases():
    assert exact_mcnemar_pvalue(0, 0) == 1.0
    assert exact_mcnemar_pvalue(0, 5) < 0.1
    assert exact_binomial_upper_bound(0, 300, confidence=0.95) < 0.01
    adjusted = holm_adjust({"a": 0.01, "b": 0.03, "c": 0.20})
    assert adjusted["a"] == pytest.approx(0.03)
    assert adjusted["b"] >= adjusted["a"]


def test_hierarchical_bootstrap_resamples_training_seed_clusters():
    rows = [
        {"training_seed": 11, "delta": 1.0},
        {"training_seed": 11, "delta": 1.0},
        {"training_seed": 22, "delta": 3.0},
        {"training_seed": 22, "delta": 3.0},
    ]
    result = hierarchical_paired_bootstrap_mean_ci(rows, replicates=200, seed=3)
    assert result["cluster_count"] == 2
    assert result["mean_delta"] == pytest.approx(2.0)
    assert result["ci_low"] <= 2.0 <= result["ci_high"]


def test_crossed_bootstrap_and_replicated_statistics_use_both_seed_axes():
    direct = crossed_paired_bootstrap_mean_ci(
        [[1.0, 1.0, 1.0], [3.0, 3.0, 3.0]],
        replicates=300,
        seed=5,
    )
    assert direct["training_seed_count"] == 2
    assert direct["simulator_seed_count"] == 3
    assert direct["mean_delta"] == pytest.approx(2.0)
    assert direct["method"] == "crossed_training_simulator_paired_percentile_bootstrap"

    pairs = [
        _replicate_pair(
            11,
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [False, False, True],
            [False, True, True],
            "a" * 64,
            "c" * 64,
        ),
        _replicate_pair(
            22,
            [0.0, 0.0, 0.0],
            [3.0, 3.0, 3.0],
            [False, False, True],
            [False, False, False],
            "b" * 64,
            "d" * 64,
        ),
    ]
    result = build_replicated_pair_statistics(
        pairs,
        continuous_metrics=["episode_reward"],
        binary_metrics=["proxy_collision"],
        replicates=300,
        seed=7,
    )
    assert result["balanced_matrix"] is True
    assert result["training_seeds"] == [11, 22]
    assert result["simulator_seeds"] == [1, 2, 3]
    assert result["cell_count"] == 6
    assert result["continuous"]["episode_reward"]["mean_delta"] == pytest.approx(2.0)
    collision = result["binary"]["proxy_collision"]
    assert collision["risk_difference"] == pytest.approx(0.0)
    assert collision["pooled_exact_mcnemar_performed"] is False
    assert "mcnemar_exact_pvalue" not in collision
    reversed_result = build_replicated_pair_statistics(
        list(reversed(pairs)),
        continuous_metrics=["episode_reward"],
        binary_metrics=["proxy_collision"],
        replicates=300,
        seed=7,
    )
    assert reversed_result == result


def test_replicated_statistics_reject_unbalanced_or_ambiguous_replicates():
    first = _replicate_pair(
        11,
        [0.0, 0.0],
        [1.0, 1.0],
        [False, False],
        [False, False],
        "a" * 64,
        "c" * 64,
    )
    second = _replicate_pair(
        22,
        [0.0, 0.0],
        [1.0, 1.0],
        [False, False],
        [False, False],
        "b" * 64,
        "d" * 64,
    )
    second["left_report"]["episodes"][1]["seed"] = 3
    second["right_report"]["episodes"][1]["seed"] = 3
    with pytest.raises(ValueError, match="same simulator seeds"):
        build_replicated_pair_statistics([first, second], replicates=20)

    duplicate_seed = dict(first)
    with pytest.raises(ValueError, match="duplicate replicated training seed"):
        build_replicated_pair_statistics([first, duplicate_seed], replicates=20)

    distinct_seed_reused_checkpoint = _replicate_pair(
        33,
        [0.0, 0.0],
        [1.0, 1.0],
        [False, False],
        [False, False],
        "a" * 64,
        "e" * 64,
    )
    with pytest.raises(ValueError, match="left checkpoints must be distinct"):
        build_replicated_pair_statistics([first, distinct_seed_reused_checkpoint], replicates=20)

    invalid_hash = dict(first)
    invalid_hash["left_checkpoint_sha256"] = "not-a-hash"
    with pytest.raises(ValueError, match="64-character SHA-256"):
        build_replicated_pair_statistics([invalid_hash], replicates=20)


def test_stage5_statistics_are_emitted_for_configured_pairs():
    reports = {
        "baseline": _report([1.0, 2.0], [False, False]),
        "candidate": _report([2.0, 3.0], [False, False]),
    }
    result = _build_paired_statistics(
        reports,
        [{"name": "candidate_vs_baseline", "left": "baseline", "right": "candidate"}],
        statistics_config={"bootstrap_replicates": 100, "bootstrap_seed": 9},
    )
    pair = result["pairs"]["candidate_vs_baseline"]
    assert pair["available"] is True
    assert pair["statistics"]["continuous"]["episode_reward"]["mean_delta"] == pytest.approx(1.0)


def test_pair_statistics_reject_duplicate_or_unpaired_episode_seeds():
    duplicate = _report([1.0, 2.0], [False, False])
    duplicate["episodes"][1]["seed"] = 1
    candidate = _report([2.0, 3.0], [False, False])
    with pytest.raises(ValueError, match="duplicate"):
        build_pair_statistics(duplicate, candidate, replicates=20)
    baseline = _report([1.0, 2.0], [False, False])
    candidate["episodes"][1]["seed"] = 3
    with pytest.raises(ValueError, match="identical episode seeds"):
        build_pair_statistics(baseline, candidate, replicates=20)


def test_statistics_validate_confidence_and_replicate_boundaries():
    report = _report([1.0], [False])
    with pytest.raises(ValueError, match="replicates"):
        build_pair_statistics(report, report, replicates=0)
    with pytest.raises(ValueError, match="confidence"):
        build_pair_statistics(report, report, confidence=1.0, replicates=20)
    assert exact_binomial_upper_bound(0, 100_000, confidence=0.95) < 0.001


def _stage5_checkpoint_hashes(training_seed):
    fixed = {
        11: ("a" * 64, "c" * 64),
        22: ("b" * 64, "d" * 64),
    }
    return fixed.get(
        int(training_seed),
        (f"{int(training_seed):064x}", f"{int(training_seed) + 1000:064x}"),
    )


def _replicated_stage5_source(training_seeds=(11, 22)):
    groups = {}
    artifacts = {}
    for training_seed in training_seeds:
        left_hash, right_hash = _stage5_checkpoint_hashes(training_seed)
        left_name = f"baseline_seed_{training_seed}"
        right_name = f"candidate_seed_{training_seed}"
        left = _report([0.0, 0.0, 0.0], [False, False, True])
        right = _report(
            [1.0 if training_seed == 11 else 3.0] * 3,
            [False, True, True] if training_seed == 11 else [False, False, False],
        )
        left["comparative"] = {
            "method": "baseline",
            "training_seed": training_seed,
            "evaluation_variant": "policy",
        }
        right["comparative"] = {
            "method": "candidate",
            "training_seed": training_seed,
            "evaluation_variant": "policy",
        }
        groups[left_name] = left
        groups[right_name] = right
        artifacts[f"ppo_model:{left_name}"] = {"sha256": left_hash, "exists": True}
        artifacts[f"ppo_model:{right_name}"] = {"sha256": right_hash, "exists": True}
    evidence_lineage = {
        "protocol_id": "replicated-protocol-v1",
        "protocol_strict": True,
        "artifacts": artifacts,
    }
    evidence_lineage["lineage_fingerprint"] = stable_hash(evidence_lineage)
    return {
        "stage": "stage5",
        "paired_eval": True,
        "safety_metric_version": "safety_v1",
        "seeds": [1, 2, 3],
        "groups": groups,
        "acceptance": {
            "baseline_vs_candidate": {
                "available": True,
                "regression": False,
                "checks": {"fixture_acceptance": True},
            }
        },
        "evidence_lineage": evidence_lineage,
    }


def _candidate_bundle(tmp_path: Path):
    bundle_dir = tmp_path / "candidate_bundle"
    bundle_dir.mkdir()
    files = {}
    for key, name in (
        ("predictor", "predictor.pt"),
        ("calibration", "calibration.json"),
        ("operating_point", "operating_point.json"),
        ("training_history", "training_history.json"),
    ):
        path = bundle_dir / name
        if key == "operating_point":
            value = {
                "split": "operating_point",
                "selected": {
                    "proxy_collision_upper_bound": 0.2,
                    "safety_violation_upper_bound": 0.2,
                    "merge_viability_lower_bound": 0.7,
                },
            }
            path.write_text(json.dumps(value), encoding="utf-8")
        elif key == "predictor":
            path.write_bytes(b"candidate-predictor")
        else:
            path.write_text("{}", encoding="utf-8")
        files[key] = bundle_file_entry(path, manifest_dir=bundle_dir)
    runtime_contract = canonical_formal_runtime_contract(
        observation={
            "enabled": True,
            "feature_version": "risk_gated_candidate_table_v3_bounded_stale",
            "activation_distance": 200.0,
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
        "formal_runtime_contract": runtime_contract,
        "formal_runtime_contract_sha256": formal_runtime_contract_sha256(
            runtime_contract
        ),
        "files": files,
    }
    manifest["artifact_fingerprint"] = stable_hash(manifest)
    path = bundle_dir / "candidate_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path, manifest


def _formal_replicated_request(tmp_path: Path, candidate_path: Path, candidate):
    training_seeds = (11, 22, 33, 44, 55)
    request = {
        "artifact_kind": REQUEST_ARTIFACT_KIND,
        "comparison_id": "baseline_vs_candidate_formal",
        "require_strict_lineage": True,
        "formal_aggregation": True,
        "minimum_training_seed_count": 5,
        "candidate_manifest": {
            "path": str(candidate_path.relative_to(tmp_path)),
            "sha256": file_sha256(candidate_path),
            "artifact_fingerprint": candidate["artifact_fingerprint"],
            "artifact_variant": candidate["artifact_variant"],
        },
        "candidate_side": "right",
        "source_acceptance_key": "baseline_vs_candidate",
        "formal_runtime_contract_sha256": candidate[
            "formal_runtime_contract_sha256"
        ],
        "statistics": {
            "continuous_metrics": ["episode_reward"],
            "binary_metrics": ["proxy_collision"],
            "bootstrap_replicates": 100,
            "bootstrap_seed": 31,
        },
        "replicates": [],
    }
    for training_seed in training_seeds:
        left_hash, right_hash = _stage5_checkpoint_hashes(training_seed)
        request["replicates"].append(
            {
                "training_seed": training_seed,
                "stage5_report": "stage5.json",
                "left_group": f"baseline_seed_{training_seed}",
                "right_group": f"candidate_seed_{training_seed}",
                "left_checkpoint_sha256": left_hash,
                "right_checkpoint_sha256": right_hash,
            }
        )
    return request


def _replicated_request(stage5_name="stage5.json"):
    return {
        "artifact_kind": REQUEST_ARTIFACT_KIND,
        "comparison_id": "baseline_vs_candidate",
        "require_strict_lineage": True,
        "formal_aggregation": False,
        "minimum_training_seed_count": 2,
        "statistics": {
            "continuous_metrics": ["episode_reward"],
            "binary_metrics": ["proxy_collision"],
            "bootstrap_replicates": 200,
            "bootstrap_seed": 17,
        },
        "replicates": [
            {
                "training_seed": 11,
                "stage5_report": stage5_name,
                "left_group": "baseline_seed_11",
                "right_group": "candidate_seed_11",
                "left_checkpoint_sha256": "a" * 64,
                "right_checkpoint_sha256": "c" * 64,
            },
            {
                "training_seed": 22,
                "stage5_report": stage5_name,
                "left_group": "baseline_seed_22",
                "right_group": "candidate_seed_22",
                "left_checkpoint_sha256": "b" * 64,
                "right_checkpoint_sha256": "d" * 64,
            },
        ],
    }


def test_replicated_stage5_pipeline_is_deterministic_and_binds_lineage(tmp_path):
    source_path = tmp_path / "stage5.json"
    manifest_path = tmp_path / "replicated_request.json"
    source_path.write_text(json.dumps(_replicated_stage5_source()), encoding="utf-8")
    manifest_path.write_text(json.dumps(_replicated_request()), encoding="utf-8")

    first = aggregate_manifest(manifest_path)
    second = aggregate_manifest(manifest_path)
    assert first == second
    assert first["statistics"]["training_seed_count"] == 2
    assert first["formal_aggregation"] is False
    assert first["minimum_training_seed_count"] == 2
    assert first["statistics"]["simulator_seed_count"] == 3
    assert first["lineage"]["protocol_id"] == "replicated-protocol-v1"
    assert len(first["lineage"]["source_reports"]) == 1
    assert first["lineage"]["lineage_fingerprint"]
    assert first["candidate_binding"]["required"] is False
    assert first["gate"]["pass"] is True
    assert first["gate"]["checks"]["balanced_crossed_statistics"] is True
    assert first["report_fingerprint"]

    output = tmp_path / "replicated_report.json"
    assert run_replicated_aggregate(manifest_path, output) == output
    assert json.loads(output.read_text(encoding="utf-8")) == first


def test_replicated_stage5_pipeline_rejects_checkpoint_lineage_mismatch(tmp_path):
    source_path = tmp_path / "stage5.json"
    manifest_path = tmp_path / "replicated_request.json"
    source_path.write_text(json.dumps(_replicated_stage5_source()), encoding="utf-8")
    request = _replicated_request()
    request["replicates"][0]["left_checkpoint_sha256"] = "f" * 64
    manifest_path.write_text(json.dumps(request), encoding="utf-8")
    with pytest.raises(ValueError, match="left checkpoint hash mismatch"):
        aggregate_manifest(manifest_path)


def test_replicated_stage5_rejects_tampered_source_lineage_fingerprint(tmp_path):
    source_path = tmp_path / "stage5.json"
    manifest_path = tmp_path / "replicated_request.json"
    source = _replicated_stage5_source()
    source["evidence_lineage"]["protocol_id"] = "tampered-protocol"
    source_path.write_text(json.dumps(source), encoding="utf-8")
    manifest_path.write_text(json.dumps(_replicated_request()), encoding="utf-8")

    with pytest.raises(ValueError, match="lineage_fingerprint mismatch"):
        aggregate_manifest(manifest_path)


def test_replicated_stage5_formal_mode_requires_five_optimizer_seeds(tmp_path):
    manifest_path = tmp_path / "replicated_request.json"
    request = _replicated_request()
    request.pop("formal_aggregation")
    request.pop("minimum_training_seed_count")
    manifest_path.write_text(json.dumps(request), encoding="utf-8")
    with pytest.raises(ValueError, match="too few optimizer training seeds"):
        aggregate_manifest(manifest_path)

    request["require_strict_lineage"] = False
    request["replicates"] = request["replicates"] * 3
    manifest_path.write_text(json.dumps(request), encoding="utf-8")
    with pytest.raises(ValueError, match="requires strict lineage"):
        aggregate_manifest(manifest_path)


def _write_formal_stage5_fixture(tmp_path: Path):
    source = _replicated_stage5_source((11, 22, 33, 44, 55))
    candidate_path, candidate = _candidate_bundle(tmp_path)
    bundle_lineage = {
        "required": True,
        "enabled": True,
        "manifest_path": str(candidate_path.resolve()),
        "manifest_sha256": file_sha256(candidate_path),
        "artifact_fingerprint": candidate["artifact_fingerprint"],
        "artifact_variant": candidate["artifact_variant"],
        "artifact_generation": candidate["artifact_generation"],
        "bundle_schema_version": candidate["bundle_schema_version"],
        "formal_runtime_contract_sha256": candidate[
            "formal_runtime_contract_sha256"
        ],
        "predictor_sha256": candidate["predictor_sha256"],
    }
    bundle_lineage["binding_fingerprint"] = stable_hash(bundle_lineage)
    source["evidence_lineage"]["accvp_group_bindings"] = {}
    for training_seed in (11, 22, 33, 44, 55):
        group_name = f"candidate_seed_{training_seed}"
        source["groups"][group_name]["accvp_bundle_lineage"] = dict(bundle_lineage)
        source["evidence_lineage"]["accvp_group_bindings"][group_name] = dict(
            bundle_lineage
        )
    source["evidence_lineage"].pop("lineage_fingerprint", None)
    source["evidence_lineage"]["lineage_fingerprint"] = stable_hash(
        source["evidence_lineage"]
    )
    source_path = tmp_path / "stage5.json"
    source_path.write_text(json.dumps(source), encoding="utf-8")
    request = _formal_replicated_request(tmp_path, candidate_path, candidate)
    request_path = tmp_path / "formal_request.json"
    return source, source_path, candidate_path, candidate, request, request_path


def test_formal_replicated_stage5_binds_candidate_bundle_side_acceptance_and_gate(tmp_path):
    _source, _source_path, candidate_path, candidate, request, request_path = (
        _write_formal_stage5_fixture(tmp_path)
    )
    request_path.write_text(json.dumps(request), encoding="utf-8")
    report = aggregate_manifest(request_path)
    binding = report["candidate_binding"]
    assert report["formal_aggregation"] is True
    assert binding["required"] is True
    assert binding["candidate_side"] == "right"
    assert binding["candidate_manifest"]["sha256"] == file_sha256(candidate_path)
    assert binding["candidate_manifest"]["artifact_fingerprint"] == candidate[
        "artifact_fingerprint"
    ]
    assert binding["candidate_manifest"]["artifact_variant"] == (
        "full_candidate_gate_v1"
    )
    assert binding["formal_runtime_contract_sha256"] == candidate[
        "formal_runtime_contract_sha256"
    ]
    assert len(binding["candidate_checkpoint_records"]) == 5
    assert all(
        record["checkpoint_sha256"]
        == _stage5_checkpoint_hashes(record["training_seed"])[1]
        for record in binding["candidate_checkpoint_records"]
    )
    assert report["gate"]["pass"] is True
    assert all(report["gate"]["checks"].values())
    assert report["gate"]["checks"]["candidate_groups_used_bound_bundle"] is True
    content = dict(report)
    fingerprint = content.pop("report_fingerprint")
    assert fingerprint == stable_hash(content)


@pytest.mark.parametrize(
    ("tamper", "match"),
    [
        ("side", "candidate_side"),
        ("manifest_sha256", "candidate manifest SHA-256 mismatch"),
        ("artifact_fingerprint", "artifact_fingerprint mismatch"),
        ("artifact_variant", "artifact_variant mismatch"),
        ("runtime_contract", "runtime contract SHA-256"),
    ],
)
def test_formal_replicated_stage5_rejects_tampered_candidate_binding(
    tmp_path,
    tamper,
    match,
):
    _source, _source_path, _candidate_path, _candidate, request, request_path = (
        _write_formal_stage5_fixture(tmp_path)
    )
    if tamper == "side":
        request["candidate_side"] = "middle"
    elif tamper == "manifest_sha256":
        request["candidate_manifest"]["sha256"] = "f" * 64
    elif tamper == "artifact_fingerprint":
        request["candidate_manifest"]["artifact_fingerprint"] = "f" * 64
    elif tamper == "artifact_variant":
        request["candidate_manifest"]["artifact_variant"] = "viability_lite_task_v1"
    elif tamper == "runtime_contract":
        request["formal_runtime_contract_sha256"] = "f" * 64
    request_path.write_text(json.dumps(request), encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        aggregate_manifest(request_path)


def test_formal_replicated_stage5_rejects_empty_preregistered_metric_output(tmp_path):
    _source, _source_path, _candidate_path, _candidate, request, request_path = (
        _write_formal_stage5_fixture(tmp_path)
    )
    request["statistics"]["continuous_metrics"] = ["not_recorded"]
    request["statistics"]["binary_metrics"] = []
    request_path.write_text(json.dumps(request), encoding="utf-8")
    with pytest.raises(ValueError, match="produced no preregistered"):
        aggregate_manifest(request_path)


def test_formal_replicated_stage5_rejects_candidate_group_using_other_bundle(tmp_path):
    source, source_path, _candidate_path, _candidate, request, request_path = (
        _write_formal_stage5_fixture(tmp_path)
    )
    group_name = "candidate_seed_33"
    tampered = dict(
        source["evidence_lineage"]["accvp_group_bindings"][group_name]
    )
    tampered["manifest_sha256"] = "e" * 64
    tampered.pop("binding_fingerprint", None)
    tampered["binding_fingerprint"] = stable_hash(tampered)
    source["evidence_lineage"]["accvp_group_bindings"][group_name] = dict(tampered)
    source["groups"][group_name]["accvp_bundle_lineage"] = dict(tampered)
    source["evidence_lineage"].pop("lineage_fingerprint", None)
    source["evidence_lineage"]["lineage_fingerprint"] = stable_hash(
        source["evidence_lineage"]
    )
    source_path.write_text(json.dumps(source), encoding="utf-8")
    request_path.write_text(json.dumps(request), encoding="utf-8")
    with pytest.raises(ValueError, match="did not use the request-bound ACCVP bundle"):
        aggregate_manifest(request_path)


@pytest.mark.parametrize(
    "acceptance_patch",
    [
        {"available": False, "regression": False},
        {"available": True, "regression": True},
    ],
)
def test_formal_replicated_stage5_rejects_source_acceptance_failure(
    tmp_path,
    acceptance_patch,
):
    source, source_path, _candidate_path, _candidate, request, request_path = (
        _write_formal_stage5_fixture(tmp_path)
    )
    source["acceptance"]["baseline_vs_candidate"].update(acceptance_patch)
    source_path.write_text(json.dumps(source), encoding="utf-8")
    request_path.write_text(json.dumps(request), encoding="utf-8")
    with pytest.raises(ValueError, match="acceptance.*(not available|regression)"):
        aggregate_manifest(request_path)
