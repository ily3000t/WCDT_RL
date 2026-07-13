from __future__ import annotations

import argparse
import json
import re
import traceback
from pathlib import Path
from typing import Any

from safe_rl.accvp.candidate_table_diagnostics import (
    candidate_records_from_dataset,
    load_calibration,
    load_models_from_checkpoint,
)
from safe_rl.accvp.artifacts import (
    ACCVP_ARTIFACT_KIND,
    FINAL_HOLDOUT_LIFECYCLE_AUTHORIZATION,
    LIFECYCLE_HOLDOUT_GO,
    LIFECYCLE_HOLDOUT_NO_GO,
    LIFECYCLE_SEALED_CANDIDATE,
    VNEXT_LITE_DECISION_WEIGHTING_VERSION,
    apply_v2_bundle_paths,
    artifact_filename,
    bundle_file_entry,
    resolve_v2_bundle,
)
from safe_rl.accvp.dataset import ACCVPBranchDataset
from safe_rl.accvp.diagnostics import final_test_diagnostics
from safe_rl.accvp.schema import file_sha256, read_json, stable_hash, write_json_atomic
from safe_rl.accvp.runtime_contract import (
    compare_formal_runtime_contracts,
    formal_runtime_contract_from_config,
    formal_runtime_contract_sha256,
    validate_manifest_runtime_contract,
)
from safe_rl.accvp.viability_lite import (
    collapse_vnext_lite_records,
    evaluate_lite_thresholds,
)
from safe_rl.evaluation_protocol import (
    EvidenceProtocolError,
    claim_final_holdout,
    finalise_holdout_claim,
    protocol_snapshot,
)
from safe_rl.pipeline.accvp_tune_viability_lite import _lite_acceptance_failures
from safe_rl.utils.config import REPO_ROOT, load_config


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else REPO_ROOT / value


def _artifact_dir(cfg: Any) -> Path:
    output_root = _resolve(cfg.run.output_root)
    return output_root / str(cfg.run.run_id) / "accvp"


def _validate_operating_point_schema(
    manifest: dict[str, Any],
    operating_point: dict[str, Any],
    *,
    mode: str,
) -> None:
    if str(operating_point.get("split", "")) != "operating_point":
        raise EvidenceProtocolError("operating-point artifact must declare split='operating_point'")
    selected = dict(operating_point.get("selected", {}) or {})
    if mode == "lite":
        required = {
            "min_p_merge_before_taper",
            "min_improvement_over_raw",
            "max_target_entry_time_s",
            "max_ensemble_disagreement",
        }
        expected_variant = "viability_lite_task_v1"
        expected_schema = "accvp_viability_lite_operating_point_v1"
    else:
        required = {
            "proxy_collision_upper_bound",
            "safety_violation_upper_bound",
            "merge_viability_lower_bound",
        }
        expected_variant = "full_candidate_gate_v1"
        expected_schema = ""
    missing = sorted(required.difference(selected))
    if missing:
        raise EvidenceProtocolError(
            f"{mode} final holdout operating point is missing thresholds: {missing}"
        )
    if str(manifest.get("artifact_kind", "")) == ACCVP_ARTIFACT_KIND:
        if str(manifest.get("artifact_variant", "")) != expected_variant:
            raise EvidenceProtocolError(
                f"{mode} final holdout requires artifact_variant={expected_variant!r}"
            )
        if expected_schema and str(manifest.get("operating_point_schema", "")) != expected_schema:
            raise EvidenceProtocolError(
                f"{mode} final holdout requires operating_point_schema={expected_schema!r}"
            )
        if mode == "lite":
            decision_weighting = dict(manifest.get("decision_weighting", {}) or {})
            operating_weighting = dict(
                operating_point.get("decision_weighting", {}) or {}
            )
            if (
                str(decision_weighting.get("decision_weighting_version", ""))
                != VNEXT_LITE_DECISION_WEIGHTING_VERSION
            ):
                raise EvidenceProtocolError(
                    "lite final holdout requires fingerprint decision-weighting provenance"
                )
            if decision_weighting != operating_weighting:
                raise EvidenceProtocolError(
                    "lite bundle and operating point disagree on decision weighting"
                )


def _validate_report_fingerprint(payload: dict[str, Any], *, name: str) -> str:
    expected = str(payload.get("report_fingerprint", ""))
    if not expected:
        raise EvidenceProtocolError(f"{name} is missing report_fingerprint")
    content = dict(payload)
    content.pop("report_fingerprint", None)
    if stable_hash(content) != expected:
        raise EvidenceProtocolError(f"{name} report_fingerprint mismatch")
    return expected


def _validate_vnext_promotion_evidence(
    *,
    manifest: dict[str, Any],
    manifest_path: Path,
    runtime_benchmark_path: Path | None,
    stage5_replicated_report_path: Path | None,
    final_runtime_contract: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    if str(manifest.get("artifact_kind", "")) != ACCVP_ARTIFACT_KIND:
        return {}
    if runtime_benchmark_path is None or stage5_replicated_report_path is None:
        raise EvidenceProtocolError(
            "VNext final holdout requires runtime benchmark and replicated Stage5 evidence"
        )
    try:
        bundle_runtime_contract, bundle_runtime_contract_sha = (
            validate_manifest_runtime_contract(manifest)
        )
    except ValueError as exc:
        raise EvidenceProtocolError(str(exc)) from exc
    if final_runtime_contract is None:
        raise EvidenceProtocolError(
            "VNext final holdout requires its canonical runtime contract"
        )
    final_contract_check = compare_formal_runtime_contracts(
        bundle_runtime_contract,
        final_runtime_contract,
    )
    if not bool(final_contract_check.get("pass", False)):
        raise EvidenceProtocolError(
            "final-holdout runtime contract does not match the frozen bundle"
        )
    runtime_path = runtime_benchmark_path.resolve()
    stage5_path = stage5_replicated_report_path.resolve()
    runtime = read_json(runtime_path)
    stage5 = read_json(stage5_path)
    runtime_evidence_path = runtime_path
    runtime_evidence_sha256 = file_sha256(runtime_path)
    runtime_evidence_fingerprint = ""
    runtime_replicate_count = 1
    if str(runtime.get("artifact_kind", "")) == "accvp_runtime_benchmark_replicates_v1":
        runtime_evidence_fingerprint = _validate_report_fingerprint(
            runtime, name="replicated runtime benchmark"
        )
        aggregate_gate = dict(runtime.get("gate", {}) or {})
        if not bool(aggregate_gate.get("pass", False)):
            raise EvidenceProtocolError("replicated VNext runtime benchmark gate did not pass")
        replicate_rows = list(runtime.get("replicates", []) or [])
        if len(replicate_rows) < 5:
            raise EvidenceProtocolError(
                "VNext promotion requires runtime coverage for at least five PPO optimizer seeds"
            )
        checkpoint_hashes: set[str] = set()
        individual_reports: list[tuple[Path, dict[str, Any]]] = []
        aggregate_allowed_manifest_hashes = {file_sha256(manifest_path)}
        aggregate_source_hash = str(manifest.get("source_candidate_manifest_sha256", ""))
        if aggregate_source_hash:
            aggregate_allowed_manifest_hashes.add(aggregate_source_hash)
        for row in replicate_rows:
            checkpoint_hash = str(row.get("checkpoint_sha256", ""))
            if not checkpoint_hash or checkpoint_hash in checkpoint_hashes:
                raise EvidenceProtocolError(
                    "replicated runtime benchmark contains a missing or duplicate checkpoint hash"
                )
            checkpoint_hashes.add(checkpoint_hash)
            individual_path = Path(str(row.get("report", "")))
            if not individual_path.is_absolute():
                individual_path = runtime_path.parent / individual_path
            individual_path = individual_path.resolve()
            if not individual_path.is_file() or file_sha256(individual_path) != str(
                row.get("report_sha256", "")
            ):
                raise EvidenceProtocolError(
                    "replicated runtime benchmark has a missing or changed individual report"
                )
            individual = read_json(individual_path)
            if (
                str(individual.get("artifact_kind", "")) != "accvp_runtime_benchmark_v1"
                or int(individual.get("schema_version", 0)) < 2
                or str(individual.get("benchmark_scope", "")) != "policy_runtime"
                or str(individual.get("policy_type", "")) != "sb3_ppo"
                or str(individual.get("backend", "")) != "vectorized"
                or not bool(dict(individual.get("gate", {}) or {}).get("pass", False))
                or str(dict(individual.get("gate", {}) or {}).get("profile", ""))
                != "bounded_stale_runtime_v3_strict"
                or str(individual.get("policy_model_sha256", "")) != checkpoint_hash
                or str(individual.get("formal_runtime_contract_sha256", ""))
                != bundle_runtime_contract_sha
                or str(
                    dict(
                        dict(individual.get("artifact_lineage", {}) or {}).get(
                            "accvp_manifest", {}
                        )
                        or {}
                    ).get("sha256", "")
                )
                not in aggregate_allowed_manifest_hashes
            ):
                raise EvidenceProtocolError(
                    "replicated runtime benchmark contains an invalid policy-runtime report"
                )
            _validate_report_fingerprint(individual, name="runtime benchmark replicate")
            individual_reports.append((individual_path, individual))
        runtime_replicate_count = len(individual_reports)
        # The common detailed checks below validate one member. The loop above
        # has already required every member to pass the same strict gate; the
        # aggregate report additionally records the worst value across members.
        runtime_path, runtime = individual_reports[0]
    if str(runtime.get("artifact_kind", "")) != "accvp_runtime_benchmark_v1":
        raise EvidenceProtocolError("invalid ACCVP runtime benchmark artifact_kind")
    if int(runtime.get("schema_version", 0)) < 2:
        raise EvidenceProtocolError("VNext promotion requires runtime benchmark schema version 2")
    if (
        str(runtime.get("benchmark_scope", "")) != "policy_runtime"
        or str(runtime.get("policy_type", "")) != "sb3_ppo"
    ):
        raise EvidenceProtocolError(
            "VNext promotion requires the post-training sb3_ppo policy runtime benchmark"
        )
    runtime_fingerprint = _validate_report_fingerprint(
        runtime, name="runtime benchmark"
    )
    if not runtime_evidence_fingerprint:
        runtime_evidence_fingerprint = runtime_fingerprint
    if str(runtime.get("backend", "")) != "vectorized":
        raise EvidenceProtocolError("VNext promotion requires vectorized runtime benchmark")
    if not bool(dict(runtime.get("gate", {}) or {}).get("pass", False)):
        raise EvidenceProtocolError("VNext runtime benchmark gate did not pass")
    if str(dict(runtime.get("gate", {}) or {}).get("profile", "")) != "bounded_stale_runtime_v3_strict":
        raise EvidenceProtocolError("VNext promotion requires the strict runtime gate profile")
    runtime_gate_checks = dict(dict(runtime.get("gate", {}) or {}).get("checks", {}) or {})
    if not bool(runtime_gate_checks.get("formal_runtime_contract_match", False)):
        raise EvidenceProtocolError("runtime benchmark gate did not verify the formal runtime contract")
    runtime_contract = dict(runtime.get("formal_runtime_contract", {}) or {})
    try:
        runtime_contract_sha = formal_runtime_contract_sha256(runtime_contract)
    except ValueError as exc:
        raise EvidenceProtocolError(str(exc)) from exc
    if str(runtime.get("formal_runtime_contract_sha256", "")) != runtime_contract_sha:
        raise EvidenceProtocolError("runtime benchmark formal runtime contract hash mismatch")
    runtime_contract_check = compare_formal_runtime_contracts(
        bundle_runtime_contract,
        runtime_contract,
    )
    if (
        not bool(runtime_contract_check.get("pass", False))
        or runtime_contract_sha != bundle_runtime_contract_sha
    ):
        raise EvidenceProtocolError(
            "runtime benchmark contract does not match the frozen bundle"
        )
    runtime_metrics = dict(runtime.get("metrics", {}) or {})
    if int(runtime_metrics.get("accvp_table_unique_episode_seed_count", 0)) < 30:
        raise EvidenceProtocolError("runtime benchmark used fewer than 30 episode seeds")
    if int(runtime_metrics.get("accvp_table_activation_window_decision_count", 0)) < 1000:
        raise EvidenceProtocolError("runtime benchmark used fewer than 1,000 activation decisions")
    if not bool(runtime_metrics.get("accvp_table_seed_schedule_match", False)):
        raise EvidenceProtocolError("runtime benchmark seed schedule did not match")
    runtime_manifest = dict(
        dict(runtime.get("artifact_lineage", {}) or {}).get("accvp_manifest", {})
        or {}
    )
    runtime_manifest_sha = str(runtime_manifest.get("sha256", ""))
    allowed_manifest_hashes = {file_sha256(manifest_path)}
    source_hash = str(manifest.get("source_candidate_manifest_sha256", ""))
    if source_hash:
        allowed_manifest_hashes.add(source_hash)
    if runtime_manifest_sha not in allowed_manifest_hashes:
        raise EvidenceProtocolError(
            "runtime benchmark does not bind the promoted or source candidate bundle"
        )

    if str(stage5.get("artifact_kind", "")) != "stage5_replicated_paired_report_v1":
        raise EvidenceProtocolError("invalid replicated Stage5 artifact_kind")
    stage5_fingerprint = _validate_report_fingerprint(stage5, name="replicated Stage5")
    if int(stage5.get("schema_version", 0)) < 2:
        raise EvidenceProtocolError("VNext promotion requires Stage5 report schema version 2")
    if not bool(stage5.get("formal_aggregation", False)):
        raise EvidenceProtocolError("VNext promotion requires formal replicated Stage5 aggregation")
    if int(stage5.get("minimum_training_seed_count", 0)) < 5:
        raise EvidenceProtocolError("formal replicated Stage5 report lowered its seed minimum")
    stage5_gate = dict(stage5.get("gate", {}) or {})
    stage5_gate_checks = dict(stage5_gate.get("checks", {}) or {})
    if (
        str(stage5_gate.get("profile", ""))
        != "formal_candidate_promotion_binding_v1"
        or not bool(stage5_gate.get("pass", False))
        or not stage5_gate_checks
        or not all(bool(value) for value in stage5_gate_checks.values())
    ):
        raise EvidenceProtocolError("replicated Stage5 candidate promotion gate did not pass")
    candidate_binding = dict(stage5.get("candidate_binding", {}) or {})
    candidate_manifest = dict(candidate_binding.get("candidate_manifest", {}) or {})
    if not bool(candidate_binding.get("required", False)):
        raise EvidenceProtocolError("replicated Stage5 report lacks required candidate binding")
    if str(candidate_manifest.get("sha256", "")) not in allowed_manifest_hashes:
        raise EvidenceProtocolError(
            "replicated Stage5 report does not bind the promoted or source candidate bundle"
        )
    allowed_candidate_fingerprints = {
        str(manifest.get("artifact_fingerprint", "")),
        str(manifest.get("source_candidate_fingerprint", "")),
    }
    allowed_candidate_fingerprints.discard("")
    if str(candidate_manifest.get("artifact_fingerprint", "")) not in allowed_candidate_fingerprints:
        raise EvidenceProtocolError("replicated Stage5 candidate fingerprint mismatch")
    allowed_candidate_variants = {str(manifest.get("artifact_variant", ""))}
    if source_hash:
        allowed_candidate_variants.add("full_candidate_gate_v1")
    if str(candidate_manifest.get("artifact_variant", "")) not in allowed_candidate_variants:
        raise EvidenceProtocolError("replicated Stage5 candidate artifact variant mismatch")
    if (
        str(candidate_binding.get("formal_runtime_contract_sha256", ""))
        != bundle_runtime_contract_sha
        or str(candidate_manifest.get("formal_runtime_contract_sha256", ""))
        != bundle_runtime_contract_sha
    ):
        raise EvidenceProtocolError("replicated Stage5 runtime contract mismatch")
    if str(candidate_binding.get("candidate_side", "")) not in {"left", "right"}:
        raise EvidenceProtocolError("replicated Stage5 candidate side is invalid")
    if not str(candidate_binding.get("source_acceptance_key", "")).strip():
        raise EvidenceProtocolError("replicated Stage5 source acceptance binding is missing")
    statistics = dict(stage5.get("statistics", {}) or {})
    if not bool(statistics.get("balanced_matrix", False)):
        raise EvidenceProtocolError("replicated Stage5 matrix is not balanced")
    if int(statistics.get("training_seed_count", 0)) < 5:
        raise EvidenceProtocolError("replicated Stage5 evidence requires at least five training seeds")
    if str(statistics.get("method", "")) != "crossed_training_simulator_paired_percentile_bootstrap":
        raise EvidenceProtocolError("replicated Stage5 evidence does not use crossed bootstrap")
    training_seeds = [int(value) for value in list(statistics.get("training_seeds", []) or [])]
    simulator_seeds = [int(value) for value in list(statistics.get("simulator_seeds", []) or [])]
    if (
        len(training_seeds) != int(statistics.get("training_seed_count", 0))
        or len(training_seeds) != len(set(training_seeds))
    ):
        raise EvidenceProtocolError("replicated Stage5 training seed ledger is inconsistent")
    if (
        len(simulator_seeds) != int(statistics.get("simulator_seed_count", 0))
        or len(simulator_seeds) != len(set(simulator_seeds))
    ):
        raise EvidenceProtocolError("replicated Stage5 simulator seed ledger is inconsistent")
    checkpoint_records = list(statistics.get("checkpoint_records", []) or [])
    if len(checkpoint_records) != len(training_seeds):
        raise EvidenceProtocolError("replicated Stage5 checkpoint matrix is incomplete")
    checkpoint_hashes: set[str] = set()
    checkpoint_training_seeds: set[int] = set()
    for record in checkpoint_records:
        if not isinstance(record, dict):
            raise EvidenceProtocolError("replicated Stage5 checkpoint record is invalid")
        checkpoint_training_seeds.add(int(record.get("training_seed", -1)))
        for key in ("left_checkpoint_sha256", "right_checkpoint_sha256"):
            digest = str(record.get(key, ""))
            if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
                raise EvidenceProtocolError("replicated Stage5 checkpoint hash is invalid")
            checkpoint_hashes.add(digest)
    if checkpoint_training_seeds != set(training_seeds):
        raise EvidenceProtocolError("replicated Stage5 checkpoint training seeds are inconsistent")
    candidate_checkpoint_records = list(
        candidate_binding.get("candidate_checkpoint_records", []) or []
    )
    if len(candidate_checkpoint_records) != len(training_seeds):
        raise EvidenceProtocolError("replicated Stage5 candidate checkpoint matrix is incomplete")
    candidate_checkpoint_hashes: set[str] = set()
    candidate_checkpoint_training_seeds: set[int] = set()
    for record in candidate_checkpoint_records:
        if not isinstance(record, dict):
            raise EvidenceProtocolError("replicated Stage5 candidate checkpoint record is invalid")
        candidate_checkpoint_training_seeds.add(int(record.get("training_seed", -1)))
        digest = str(record.get("checkpoint_sha256", ""))
        if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise EvidenceProtocolError("replicated Stage5 candidate checkpoint hash is invalid")
        candidate_checkpoint_hashes.add(digest)
    if candidate_checkpoint_training_seeds != set(training_seeds):
        raise EvidenceProtocolError(
            "replicated Stage5 candidate checkpoint training seeds are inconsistent"
        )
    candidate_checkpoint_content = [
        {
            "training_seed": int(record["training_seed"]),
            "group": str(record["group"]),
            "checkpoint_sha256": str(record["checkpoint_sha256"]),
        }
        for record in candidate_checkpoint_records
    ]
    if (
        stable_hash(candidate_checkpoint_content)
        != str(candidate_binding.get("candidate_checkpoint_matrix_sha256", ""))
    ):
        raise EvidenceProtocolError(
            "replicated Stage5 candidate checkpoint matrix fingerprint mismatch"
        )
    runtime_policy_sha = str(
        dict(runtime.get("artifact_lineage", {}) or {}).get(
            "policy_model_sha256", ""
        )
    )
    if runtime_policy_sha not in candidate_checkpoint_hashes:
        raise EvidenceProtocolError(
            "runtime benchmark policy is not part of the replicated Stage5 candidate side"
        )
    expected_statistics_fingerprint = str(statistics.get("statistics_fingerprint", ""))
    statistics_content = dict(statistics)
    statistics_content.pop("statistics_fingerprint", None)
    if (
        not expected_statistics_fingerprint
        or stable_hash(statistics_content) != expected_statistics_fingerprint
    ):
        raise EvidenceProtocolError("replicated Stage5 statistics_fingerprint mismatch")
    stage5_protocol = str(dict(stage5.get("lineage", {}) or {}).get("protocol_id", ""))
    bundle_protocol = str(manifest.get("evidence_protocol_id", ""))
    if not stage5_protocol or (bundle_protocol and stage5_protocol != bundle_protocol):
        raise EvidenceProtocolError("replicated Stage5 protocol_id does not match the bundle")
    return {
        "runtime_benchmark": {
            "path": str(runtime_evidence_path),
            "sha256": runtime_evidence_sha256,
            "report_fingerprint": runtime_evidence_fingerprint,
            "optimizer_replicate_count": runtime_replicate_count,
            "bound_candidate_manifest_sha256": runtime_manifest_sha,
            "formal_runtime_contract_sha256": runtime_contract_sha,
        },
        "stage5_replicated_report": {
            "path": str(stage5_path),
            "sha256": file_sha256(stage5_path),
            "report_fingerprint": stage5_fingerprint,
            "protocol_id": stage5_protocol,
            "candidate_manifest_sha256": str(candidate_manifest["sha256"]),
            "candidate_side": str(candidate_binding["candidate_side"]),
            "source_acceptance_key": str(candidate_binding["source_acceptance_key"]),
            "formal_runtime_contract_sha256": bundle_runtime_contract_sha,
            "training_seed_count": int(statistics["training_seed_count"]),
            "simulator_seed_count": int(statistics.get("simulator_seed_count", 0)),
        },
    }


def _validate_frozen_bundle(
    *,
    manifest_path: Path,
    manifest: dict[str, Any],
    checkpoint: Path,
    calibration: Path,
    operating_point: Path,
    dataset_manifest: Path,
    split_manifest: Path,
    protocol: dict[str, Any],
) -> None:
    resolved_files: dict[str, Path] = {}
    if str(manifest.get("artifact_kind", "")) == ACCVP_ARTIFACT_KIND:
        resolved_manifest, resolved_files = resolve_v2_bundle(manifest_path)
        if str(resolved_manifest.get("lifecycle_state", "")) != LIFECYCLE_SEALED_CANDIDATE:
            raise EvidenceProtocolError("final holdout requires bundle lifecycle_state='sealed_candidate'")
        expected_paths = {
            "predictor": checkpoint.resolve(),
            "calibration": calibration.resolve(),
            "operating_point": operating_point.resolve(),
        }
        mismatched_paths = {
            key: {"manifest": str(resolved_files.get(key)), "argument": str(value)}
            for key, value in expected_paths.items()
            if resolved_files.get(key) != value
        }
        if mismatched_paths:
            raise EvidenceProtocolError(f"frozen bundle file path mismatch: {mismatched_paths}")
    if str(manifest.get("holdout_state", "")) != "sealed":
        raise EvidenceProtocolError(
            f"final holdout requires a sealed artifact; found holdout_state={manifest.get('holdout_state')!r}"
        )
    if bool(manifest.get("deployable_artifact", False)):
        raise EvidenceProtocolError("final holdout input must be a non-deployable sealed candidate")
    if str(manifest.get("threshold_selection_split", "")) != "operating_point":
        raise EvidenceProtocolError("frozen artifact threshold provenance must be operating_point")
    if bool(manifest.get("test_used_for_threshold_selection", True)):
        raise EvidenceProtocolError("frozen artifact indicates test leakage during threshold selection")
    expected = {
        "predictor_sha256": file_sha256(checkpoint),
        "calibration_sha256": file_sha256(calibration),
        "operating_point_sha256": file_sha256(operating_point),
        "dataset_manifest_sha256": file_sha256(dataset_manifest),
        "split_manifest_sha256": file_sha256(split_manifest),
    }
    if resolved_files:
        expected["training_history_sha256"] = file_sha256(
            resolved_files["training_history"]
        )
    mismatches = {
        key: {"manifest": manifest.get(key), "actual": value}
        for key, value in expected.items()
        if str(manifest.get(key, "")) != value
    }
    if mismatches:
        raise EvidenceProtocolError(f"frozen artifact lineage mismatch: {mismatches}")
    if bool(protocol.get("strict", False)):
        if str(manifest.get("evidence_protocol_id", "")) != str(protocol.get("protocol_id", "")):
            raise EvidenceProtocolError("frozen artifact protocol_id does not match final-holdout protocol")
        if str(manifest.get("seed_ledger_sha256", "")) != str(protocol.get("seed_ledger_sha256", "")):
            raise EvidenceProtocolError("frozen artifact seed ledger does not match final-holdout protocol")
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)


def _artifact_prefix(path: Path) -> str:
    name = path.name
    for suffix in (
        "_candidate_manifest.json",
        "_shadow_manifest.json",
        "_task_artifact_manifest.json",
        "_candidate_artifact_manifest.json",
        "_artifact_manifest.json",
    ):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def _canonical_seal_path(
    *,
    dataset: Path,
    protocol_id: str,
    split_manifest_path: Path | None = None,
) -> Path:
    """Return the single claim path for one protocol/test cohort.

    Candidate identity is deliberately excluded: allowing the candidate
    manifest hash to select the seal would permit the same test cohort to be
    reopened by merely repackaging an otherwise identical bundle.
    """

    protocol = re.sub(r"[^A-Za-z0-9_.-]+", "_", protocol_id or "unversioned")
    split_path = (
        Path(split_manifest_path)
        if split_manifest_path is not None
        else dataset / "manifests" / "split_manifest.jsonl"
    )
    test_rows: list[dict[str, Any]] = []
    root_ids: set[str] = set()
    with split_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if str(row.get("split", "")) != "test":
                continue
            root_id = str(row.get("root_id", "")).strip()
            if not root_id:
                raise EvidenceProtocolError(
                    f"test split row {line_number} is missing root_id"
                )
            if root_id in root_ids:
                raise EvidenceProtocolError(
                    f"test split contains duplicate root_id={root_id!r}"
                )
            root_ids.add(root_id)
            semantic_identity = {
                "root_observation_fingerprint": str(
                    row.get("root_observation_fingerprint", "")
                ),
                "scenario_episode_key": str(row.get("scenario_episode_key", "")),
                "episode_seed": row.get("episode_seed"),
            }
            if not any(
                value not in {None, ""} for value in semantic_identity.values()
            ):
                semantic_identity = {"root_id": root_id}
            test_rows.append(semantic_identity)
    if not test_rows:
        raise EvidenceProtocolError("cannot claim an empty final test cohort")
    ordered_rows = sorted(test_rows, key=stable_hash)
    cohort_hash = stable_hash(
        {
            "identity_version": "accvp_final_test_cohort_v1",
            "test_rows": ordered_rows,
        }
    )
    return (
        dataset
        / "manifests"
        / "final_holdout_claims"
        / f"{protocol}_{cohort_hash[:24]}.json"
    )


def _verify_claim_inputs(claim: dict[str, Any]) -> None:
    changed = {}
    for name, path_key, hash_key in (
        ("artifact_manifest", "artifact_manifest", "artifact_manifest_sha256"),
        ("split_manifest", "split_manifest", "split_manifest_sha256"),
    ):
        path = Path(str(claim[path_key]))
        actual = file_sha256(path)
        if actual != str(claim.get(hash_key, "")):
            changed[name] = {"claimed": claim.get(hash_key), "actual": actual}
    for name, row in dict(claim.get("frozen_artifacts", {}) or {}).items():
        path = Path(str(row["path"]))
        actual = file_sha256(path)
        if actual != str(row.get("sha256", "")):
            changed[str(name)] = {"claimed": row.get("sha256"), "actual": actual}
    if changed:
        raise EvidenceProtocolError(f"frozen holdout inputs changed after claim: {changed}")


def _validated_manifest(
    source: dict[str, Any],
    *,
    source_path: Path,
    result_path: Path,
    seal: dict[str, Any],
    decision: str,
    target_path: Path,
    promotion_evidence: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    payload = dict(source)
    payload.pop("artifact_fingerprint", None)
    claimed_source_hash = str(seal.get("artifact_manifest_sha256", ""))
    actual_source_hash = file_sha256(source_path)
    if claimed_source_hash and actual_source_hash != claimed_source_hash:
        raise EvidenceProtocolError("frozen artifact manifest changed after holdout claim")
    if str(payload.get("artifact_kind", "")) == ACCVP_ARTIFACT_KIND:
        _source_manifest, source_files = resolve_v2_bundle(source_path)
        payload["files"] = {
            key: bundle_file_entry(path, manifest_dir=target_path.parent)
            for key, path in source_files.items()
        }
        payload["lifecycle_state"] = (
            LIFECYCLE_HOLDOUT_GO if decision == "go" else LIFECYCLE_HOLDOUT_NO_GO
        )
    payload.update(
        {
            "lifecycle_authorization": FINAL_HOLDOUT_LIFECYCLE_AUTHORIZATION,
            "deployable_artifact": decision == "go",
            "holdout_state": "evaluated",
            "holdout_decision": decision,
            "source_frozen_manifest": str(source_path.resolve()),
            "source_frozen_manifest_sha256": actual_source_hash,
            "final_test_diagnostics_path": str(result_path.resolve()),
            "final_test_diagnostics_sha256": file_sha256(result_path),
            "holdout_claim_fingerprint": seal.get("final_fingerprint", seal.get("claim_fingerprint")),
            "test_used_for_threshold_selection": False,
            "promotion_evidence": dict(promotion_evidence or {}),
        }
    )
    payload["artifact_fingerprint"] = stable_hash(payload)
    return payload


def run_final_holdout(
    *,
    cfg: Any,
    dataset: Path,
    checkpoint: Path,
    calibration_path: Path,
    operating_point_path: Path,
    artifact_manifest_path: Path,
    output_dir: Path,
    mode: str,
    runtime_benchmark_path: Path | None = None,
    stage5_replicated_report_path: Path | None = None,
) -> dict[str, Path]:
    if mode not in {"lite", "full"}:
        raise ValueError(f"unsupported final holdout mode={mode!r}")
    protocol = protocol_snapshot(cfg, base_dir=REPO_ROOT)
    if bool(protocol.get("strict", False)) and not protocol.get("protocol_id"):
        raise EvidenceProtocolError("strict final holdout requires protocol_id")
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_manifest = dataset / "manifests" / "dataset_manifest.json"
    split_manifest = dataset / "manifests" / "split_manifest.jsonl"
    manifest = read_json(artifact_manifest_path)
    final_runtime_contract: dict[str, Any] | None = None
    if str(manifest.get("artifact_kind", "")) == ACCVP_ARTIFACT_KIND:
        cfg.accvp["artifact_manifest"] = str(artifact_manifest_path.resolve())
        cfg.accvp["checkpoint"] = str(checkpoint.resolve())
        cfg.accvp["calibration_bundle"] = str(calibration_path.resolve())
        cfg.accvp["operating_point"] = str(operating_point_path.resolve())
        applied_manifest, _applied_files = apply_v2_bundle_paths(cfg)
        if applied_manifest is None:
            raise EvidenceProtocolError("final holdout could not apply bundle-v2 manifest")
        final_runtime_contract = formal_runtime_contract_from_config(
            cfg,
            base_dir=REPO_ROOT,
        )
    operating_point = read_json(operating_point_path)
    _validate_operating_point_schema(manifest, operating_point, mode=mode)
    _validate_frozen_bundle(
        manifest_path=artifact_manifest_path,
        manifest=manifest,
        checkpoint=checkpoint,
        calibration=calibration_path,
        operating_point=operating_point_path,
        dataset_manifest=dataset_manifest,
        split_manifest=split_manifest,
        protocol=protocol,
    )
    promotion_evidence = _validate_vnext_promotion_evidence(
        manifest=manifest,
        manifest_path=artifact_manifest_path,
        runtime_benchmark_path=runtime_benchmark_path,
        stage5_replicated_report_path=stage5_replicated_report_path,
        final_runtime_contract=final_runtime_contract,
    )
    prefix = _artifact_prefix(artifact_manifest_path)
    seal_path = _canonical_seal_path(
        dataset=dataset,
        protocol_id=str(protocol.get("protocol_id", "")),
        split_manifest_path=split_manifest,
    )
    result_path = output_dir / f"{prefix}_final_test_diagnostics.json"
    validated_manifest_path = (
        output_dir / artifact_filename("validated_manifest")
        if str(manifest.get("artifact_kind", "")) == ACCVP_ARTIFACT_KIND
        else output_dir / f"{prefix}_validated_task_artifact_manifest.json"
    )
    promoted_preflight_path = validated_manifest_path.with_suffix(
        validated_manifest_path.suffix + ".preflight"
    )
    if (
        result_path.exists()
        or validated_manifest_path.exists()
        or promoted_preflight_path.exists()
    ):
        raise FileExistsError(
            "final holdout output already exists; refusing to overwrite a one-shot result"
        )
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError("ACCVP final holdout evaluation requires torch") from exc
    # Validate all infrastructure before irreversibly opening the holdout.
    models = load_models_from_checkpoint(cfg, checkpoint, torch)
    calibration = load_calibration(calibration_path)
    if calibration is None:
        raise ValueError("final holdout evaluation requires a calibration bundle")
    extra_frozen_artifacts: dict[str, Path] = {}
    if str(manifest.get("artifact_kind", "")) == ACCVP_ARTIFACT_KIND:
        _resolved_manifest, resolved_files = resolve_v2_bundle(artifact_manifest_path)
        training_history = resolved_files.get("training_history")
        if training_history is not None:
            extra_frozen_artifacts["training_history"] = training_history
    if runtime_benchmark_path is not None:
        extra_frozen_artifacts["runtime_benchmark"] = runtime_benchmark_path
    if stage5_replicated_report_path is not None:
        extra_frozen_artifacts["stage5_replicated_report"] = (
            stage5_replicated_report_path
        )
    expected_hashes = {
        "checkpoint": str(manifest["predictor_sha256"]),
        "calibration": str(manifest["calibration_sha256"]),
        "operating_point": str(manifest["operating_point_sha256"]),
        "dataset_manifest": str(manifest["dataset_manifest_sha256"]),
        "split_manifest": str(manifest["split_manifest_sha256"]),
        **{
            name: file_sha256(path)
            for name, path in extra_frozen_artifacts.items()
        },
    }
    claim = claim_final_holdout(
        seal_path,
        protocol_id=str(protocol.get("protocol_id", "")),
        artifact_manifest=artifact_manifest_path,
        split_manifest=split_manifest,
        metadata={"mode": mode, "dataset": str(dataset.resolve())},
        frozen_artifacts={
            "checkpoint": checkpoint,
            "calibration": calibration_path,
            "operating_point": operating_point_path,
            "dataset_manifest": dataset_manifest,
            "split_manifest": split_manifest,
            **extra_frozen_artifacts,
        },
        expected_sha256=expected_hashes,
    )
    try:
        # The test split is deliberately instantiated only after the atomic claim.
        test_set = ACCVPBranchDataset(dataset, "test")
        if mode == "lite":
            records = candidate_records_from_dataset(models, test_set, calibration, torch)
            if str(manifest.get("artifact_kind", "")) == ACCVP_ARTIFACT_KIND:
                records, test_decision_weighting = collapse_vnext_lite_records(records)
            else:
                test_decision_weighting = {
                    "decision_weighting_version": "legacy_root_id_v1",
                    "raw_candidate_row_count": len(records),
                    "effective_candidate_row_count": len(records),
                    "statistical_independence_claim": False,
                }
            result = evaluate_lite_thresholds(records, dict(operating_point["selected"]), split="test")
            result["decision_weighting"] = test_decision_weighting
            failures = _lite_acceptance_failures(result, cfg)
            if not dict((cfg.accvp.get("viability_lite", {}) or {}).get("acceptance", {}) or {}):
                failures.append("acceptance_profile_missing")
        elif mode == "full":
            result = final_test_diagnostics(models, test_set, calibration, operating_point, torch)
            failures = ["full_viability_branch_not_authorized_for_deployment"]
        decision = "go" if not failures else "no_go"
        report = {
            "artifact_kind": "accvp_one_shot_final_holdout_report_v1",
            "mode": mode,
            "split": "test",
            "threshold_selection_split": "operating_point",
            "test_used_for_threshold_selection": False,
            "decision": decision,
            "failures": failures,
            "result": result,
            "protocol": protocol,
            "promotion_evidence": promotion_evidence,
            "frozen_artifact_manifest_sha256": file_sha256(artifact_manifest_path),
        }
        write_json_atomic(result_path, report)
        _verify_claim_inputs(claim)
    except Exception as exc:
        failure = {
            "artifact_kind": "accvp_one_shot_final_holdout_report_v1",
            "mode": mode,
            "split": "test",
            "decision": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "test_used_for_threshold_selection": False,
        }
        write_json_atomic(result_path, failure)
        finalise_holdout_claim(
            seal_path,
            result_path=result_path,
            decision="error",
            expected_claim_fingerprint=str(claim.get("claim_fingerprint", "")),
        )
        raise
    # Build and validate the promoted manifest while the claim is still open;
    # a path/hash/lifecycle failure must not finalise a false GO transition.
    try:
        validated = _validated_manifest(
            manifest,
            source_path=artifact_manifest_path,
            result_path=result_path,
            seal=claim,
            decision=decision,
            target_path=validated_manifest_path,
            promotion_evidence=promotion_evidence,
        )
        write_json_atomic(promoted_preflight_path, validated)
        if str(validated.get("artifact_kind", "")) == ACCVP_ARTIFACT_KIND:
            resolve_v2_bundle(promoted_preflight_path)
        else:
            read_json(promoted_preflight_path)
        promoted_preflight_path.unlink()
    except Exception as exc:
        if promoted_preflight_path.exists():
            promoted_preflight_path.unlink()
        failure = {
            "artifact_kind": "accvp_one_shot_final_holdout_report_v1",
            "mode": mode,
            "split": "test",
            "decision": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "test_used_for_threshold_selection": False,
        }
        write_json_atomic(result_path, failure)
        finalise_holdout_claim(
            seal_path,
            result_path=result_path,
            decision="error",
            expected_claim_fingerprint=str(claim.get("claim_fingerprint", "")),
        )
        raise
    seal = finalise_holdout_claim(
        seal_path,
        result_path=result_path,
        decision=decision,
        expected_claim_fingerprint=str(claim.get("claim_fingerprint", "")),
    )
    validated["holdout_claim_fingerprint"] = seal.get(
        "final_fingerprint", seal.get("claim_fingerprint")
    )
    validated.pop("artifact_fingerprint", None)
    validated["artifact_fingerprint"] = stable_hash(validated)
    write_json_atomic(validated_manifest_path, validated)
    return {
        "result": result_path,
        "seal": seal_path,
        "validated_manifest": validated_manifest_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Open a frozen ACCVP final holdout exactly once")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--calibration", default=None)
    parser.add_argument("--operating-point", default=None)
    parser.add_argument("--artifact-manifest", required=True)
    parser.add_argument("--runtime-benchmark", default=None)
    parser.add_argument("--stage5-replicated-report", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--mode", choices=["lite", "full"], default="lite")
    args = parser.parse_args()
    cfg = load_config(args.config)
    artifact_manifest_path = _resolve(args.artifact_manifest)
    artifact_payload = read_json(artifact_manifest_path)
    resolved_bundle_files: dict[str, Path] = {}
    if str(artifact_payload.get("artifact_kind", "")) == ACCVP_ARTIFACT_KIND:
        _resolved_manifest, resolved_bundle_files = resolve_v2_bundle(
            artifact_manifest_path
        )
    output_dir = _resolve(args.output_dir) if args.output_dir else _artifact_dir(cfg)
    paths = run_final_holdout(
        cfg=cfg,
        dataset=_resolve(args.dataset or cfg.accvp.dataset_dir),
        checkpoint=(
            _resolve(args.checkpoint)
            if args.checkpoint
            else resolved_bundle_files["predictor"]
            if "predictor" in resolved_bundle_files
            else _resolve(cfg.accvp.checkpoint)
        ),
        calibration_path=(
            _resolve(args.calibration)
            if args.calibration
            else resolved_bundle_files["calibration"]
            if "calibration" in resolved_bundle_files
            else _resolve(cfg.accvp.calibration_bundle)
        ),
        operating_point_path=(
            _resolve(args.operating_point)
            if args.operating_point
            else resolved_bundle_files["operating_point"]
            if "operating_point" in resolved_bundle_files
            else _resolve(cfg.accvp.operating_point)
        ),
        artifact_manifest_path=artifact_manifest_path,
        output_dir=output_dir,
        mode=args.mode,
        runtime_benchmark_path=(
            None if args.runtime_benchmark is None else _resolve(args.runtime_benchmark)
        ),
        stage5_replicated_report_path=(
            None
            if args.stage5_replicated_report is None
            else _resolve(args.stage5_replicated_report)
        ),
    )
    print(
        "accvp_final_holdout "
        f"result={paths['result']} seal={paths['seal']} validated_manifest={paths['validated_manifest']}"
    )


if __name__ == "__main__":
    main()
