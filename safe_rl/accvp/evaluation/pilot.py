"""Deterministic ACCVP-240 pilot acceptance before formal collection."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

from safe_rl.accvp.contracts.schema import file_sha256, read_json, write_json_atomic
from safe_rl.accvp.contracts.protocol import (
    ACCVP_SELECTOR3_DATA_CONTRACT_VERSION,
    ACCVP_SELECTOR4_DATA_CONTRACT_VERSION,
    is_strict_selector_data_contract,
)
from safe_rl.accvp.evaluation.dataset_integrity import (
    audit_dataset_actor_contract,
)


PILOT_VALIDATION_IMPLEMENTATION_VERSION = "pilot_gate_selector4_strict_v2"


def _jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def validate_pilot_dataset(
    dataset_dir: str | Path,
    *,
    expected_root_counts: Mapping[str, int],
    min_source_fraction: float = 0.90,
    min_branch_success_rate: float = 0.99,
    min_observed_viability_fraction: float = 0.70,
    oracle_report_path: str | Path | None = None,
    oracle_required_seeds: Iterable[int] | None = None,
    oracle_cohort_role: str | None = None,
    oracle_exclude_from_model_splits: bool | None = None,
) -> dict[str, Any]:
    """Validate fixed pilot criteria without treating model loss as a gate."""

    dataset = Path(dataset_dir).resolve()
    manifests = dataset / "manifests"
    manifest = read_json(manifests / "dataset_manifest.json")
    if str(manifest.get("artifact_kind", "")) != "counterfactual_dataset_v2":
        raise ValueError("ACCVP pilot validation requires a merged counterfactual_dataset_v2")
    if str(manifest.get("collection_phase", "")) != "pilot":
        raise ValueError("ACCVP pilot validation requires a dataset merged from pilot shards")
    roots = [row for row in _jsonl(manifests / "roots.jsonl") if bool(row.get("complete", False))]
    branches = [row for row in _jsonl(manifests / "branches.jsonl") if row.get("branch_status") == "completed"]
    protocol_version = str(
        dict(manifest.get("data_contract", {}) or {}).get(
            "protocol_version", ""
        )
    )
    selector3_contract = protocol_version == ACCVP_SELECTOR3_DATA_CONTRACT_VERSION
    selector4_contract = protocol_version == ACCVP_SELECTOR4_DATA_CONTRACT_VERSION
    strict_selector_contract = is_strict_selector_data_contract(protocol_version)
    coverage_incomplete_root_ids = [
        str(row.get("root_id", ""))
        for row in roots
        if (
            not bool(row.get("task_actor_coverage_complete", False))
            or bool(row.get("critical_actor_overflow", False))
            or bool(list(row.get("dropped_critical_actor_ids", []) or []))
        )
    ]
    counts = Counter(str(row.get("collection_source", "unknown")) for row in roots)
    source_coverage = {
        name: {
            "target": int(target),
            "actual": int(counts.get(name, 0)),
            "fraction": float(counts.get(name, 0)) / max(1, int(target)),
            "pass": float(counts.get(name, 0)) >= float(target) * float(min_source_fraction),
        }
        for name, target in expected_root_counts.items()
    }
    source_manifests = []
    completed_branches = 0
    failed_branches = 0
    rejected_root_count = 0
    critical_actor_overflow_count = 0
    source_coverage_incomplete_count = 0
    source_risk_coverage_incomplete_count = 0
    for source in manifest.get("source_shards", []):
        path = Path(str(source["path"])) / "manifests" / "dataset_manifest.json"
        source_manifest = read_json(path)
        status = Counter({str(key): int(value) for key, value in dict(source_manifest.get("branch_status_counts", {})).items()})
        completed_branches += int(status.get("completed", 0))
        failed_branches += sum(value for key, value in status.items() if key != "completed")
        rejected_root_count += int(source_manifest.get("rejected_root_count", 0))
        critical_actor_overflow_count += int(
            source_manifest.get("critical_actor_overflow_count", 0)
        )
        source_coverage_incomplete_count += int(
            source_manifest.get("task_actor_coverage_incomplete_count", 0)
        )
        source_risk_coverage_incomplete_count += int(
            source_manifest.get(
                "risk_safety_actor_coverage_incomplete_count", 0
            )
        )
        source_manifests.append(
            {
                "collection_id": str(source_manifest["collection_id"]),
                "collection_source": str(source_manifest.get("collection_source", "unknown")),
                "manifest_path": str(path),
                "branch_status_counts": dict(status),
                "rejected_root_count": int(
                    source_manifest.get("rejected_root_count", 0)
                ),
                "critical_actor_overflow_count": int(
                    source_manifest.get("critical_actor_overflow_count", 0)
                ),
                "safety_actor_coverage_incomplete_count": int(
                    source_manifest.get(
                        "safety_actor_coverage_incomplete_count", 0
                    )
                ),
                "task_actor_coverage_incomplete_count": int(
                    source_manifest.get(
                        "task_actor_coverage_incomplete_count", 0
                    )
                ),
                "risk_safety_actor_coverage_incomplete_count": int(
                    source_manifest.get(
                        "risk_safety_actor_coverage_incomplete_count", 0
                    )
                ),
            }
        )
    branch_success_rate = float(completed_branches) / max(1, completed_branches + failed_branches)
    activation_branches = [
        row
        for row in branches
        if str(row.get("activation_bin", row.get("deadline_bin", ""))) in {"activation_window", "deadline"}
    ]
    observed_viability_fraction = float(sum(bool(row.get("event_observed", False)) for row in activation_branches)) / max(
        1, len(activation_branches)
    )
    conditions = {
        "source_coverage": all(item["pass"] for item in source_coverage.values()),
        "branch_success_rate": branch_success_rate >= float(min_branch_success_rate),
        "observed_viability_fraction": observed_viability_fraction >= float(min_observed_viability_fraction),
    }
    actor_contract_audit = (
        audit_dataset_actor_contract(dataset)
        if strict_selector_contract
        else None
    )
    if strict_selector_contract:
        assert actor_contract_audit is not None
        conditions.update(
            {
                "rejected_root_count_zero": (
                    rejected_root_count == 0
                    and int(manifest.get("rejected_root_count", -1)) == 0
                ),
                "critical_actor_overflow_zero": (
                    critical_actor_overflow_count == 0
                    and int(manifest.get("critical_actor_overflow_count", -1))
                    == 0
                ),
                "task_actor_coverage_complete": (
                    not coverage_incomplete_root_ids
                    and source_coverage_incomplete_count == 0
                    and int(
                        manifest.get(
                            "task_actor_coverage_incomplete_count", -1
                        )
                    )
                    == 0
                ),
                "risk_safety_actor_coverage_complete": (
                    source_risk_coverage_incomplete_count == 0
                    and int(
                        manifest.get(
                            "risk_safety_actor_coverage_incomplete_count", -1
                        )
                    )
                    == 0
                    and all(
                        bool(
                            row.get(
                                "risk_safety_actor_coverage_complete",
                                False,
                            )
                        )
                        for row in roots
                    )
                ),
                "actor_mapping_mismatch_zero": int(
                    actor_contract_audit["actor_mapping_mismatch_count"]
                )
                == 0,
                "root_observation_fingerprint_mismatch_zero": int(
                    actor_contract_audit[
                        "root_observation_fingerprint_mismatch_count"
                    ]
                )
                == 0,
                "protected_actor_coverage_complete": (
                    int(
                        actor_contract_audit[
                            "selector_metadata_missing_count"
                        ]
                    )
                    == 0
                    and int(
                        actor_contract_audit[
                            "protected_actor_coverage_incomplete_count"
                        ]
                    )
                    == 0
                    and float(
                        actor_contract_audit[
                            "protected_actor_coverage_rate"
                        ]
                    )
                    == 1.0
                ),
            }
        )
    oracle = None
    if oracle_report_path is not None:
        oracle = read_json(oracle_report_path)
        expected_seeds = [
            int(value)
            for value in (
                oracle_required_seeds
                if oracle_required_seeds is not None
                else oracle.get("required_seeds", [])
            )
        ]
        report_dataset = str(oracle.get("dataset_dir", ""))
        strict_oracle_contract = oracle_cohort_role is not None
        oracle_dataset_path = Path(report_dataset).resolve() if report_dataset else None
        oracle_dataset_exists = bool(
            oracle_dataset_path is not None and oracle_dataset_path.is_dir()
        ) if strict_oracle_contract else (
            not report_dataset or bool(oracle_dataset_path and oracle_dataset_path.is_dir())
        )
        oracle_provenance_matches = not strict_oracle_contract
        oracle_contract_matches = not strict_oracle_contract
        oracle_scope_matches = not strict_oracle_contract
        if strict_oracle_contract and oracle_dataset_path is not None and oracle_dataset_path.is_dir():
            oracle_manifests = oracle_dataset_path / "manifests"
            oracle_manifest = read_json(oracle_manifests / "dataset_manifest.json")
            expected_provenance = dict(oracle.get("dataset_provenance", {}))
            oracle_provenance_matches = all(
                str(expected_provenance.get(key, "")) == actual
                for key, actual in {
                    "dataset_manifest_sha256": file_sha256(oracle_manifests / "dataset_manifest.json"),
                    "roots_manifest_sha256": file_sha256(oracle_manifests / "roots.jsonl"),
                    "branches_manifest_sha256": file_sha256(oracle_manifests / "branches.jsonl"),
                }.items()
            )
            oracle_contract_matches = all(
                oracle_manifest.get(key) == manifest.get(key)
                for key in (
                    "counterfactual_schema_version",
                    "data_contract_hash",
                    "accvp_activation_distance_m",
                    "risk_model_fingerprint",
                    "scenario_route_hash",
                )
            )
            oracle_roots = [
                row
                for row in _jsonl(oracle_manifests / "roots.jsonl")
                if bool(row.get("complete", False))
                and int(row.get("episode_seed", -1)) in set(expected_seeds)
                and str(row.get("root_policy", row.get("root_source", ""))) == "merge_timing"
            ]
            oracle_scope_matches = (
                {int(row.get("episode_seed", -1)) for row in oracle_roots} == set(expected_seeds)
                and all(
                    bool(row.get("oracle_only", False))
                    and str(row.get("cohort_role", "")) == str(oracle_cohort_role)
                    and bool(row.get("exclude_from_model_splits", False))
                    for row in oracle_roots
                )
            )
        role_matches = oracle_cohort_role is None or (
            str(oracle.get("cohort_role", "")) == str(oracle_cohort_role)
            and bool(oracle.get("oracle_only", False))
        )
        exclusion_matches = oracle_exclude_from_model_splits is None or (
            bool(oracle.get("exclude_from_model_splits", False))
            == bool(oracle_exclude_from_model_splits)
        )
        if strict_oracle_contract:
            forbidden_pilot_roots = [
                str(row.get("root_id", ""))
                for row in roots
                if int(row.get("episode_seed", -1)) in set(expected_seeds)
                or bool(row.get("oracle_only", False))
                or bool(row.get("exclude_from_model_splits", False))
                or str(row.get("cohort_role", "")) == str(oracle_cohort_role)
            ]
            conditions["oracle_excluded_from_pilot_dataset"] = not forbidden_pilot_roots
        conditions["oracle_regression"] = bool(
            oracle_dataset_exists
            and oracle_provenance_matches
            and oracle_contract_matches
            and oracle_scope_matches
            and str(oracle.get("oracle_state", "")) == "go"
            and [int(value) for value in oracle.get("required_seeds", [])] == expected_seeds
            and str(oracle.get("root_policy", "")) == "merge_timing"
            and role_matches
            and exclusion_matches
        )
    return {
        "artifact_kind": "accvp_pilot_validation_v2",
        "validation_implementation_version": PILOT_VALIDATION_IMPLEMENTATION_VERSION,
        "dataset_dir": str(dataset),
        "dataset_fingerprint": str(manifest.get("dataset_fingerprint", "")),
        "data_contract_hash": str(manifest.get("data_contract_hash", "")),
        "accvp_activation_distance_m": float(manifest.get("accvp_activation_distance_m", -1.0)),
        "source_coverage": source_coverage,
        "branch_success_rate": branch_success_rate,
        "observed_viability_fraction": observed_viability_fraction,
        "activation_branch_count": len(activation_branches),
        "data_contract_version": protocol_version,
        "selector3_contract": selector3_contract,
        "selector4_contract": selector4_contract,
        "strict_selector_contract": strict_selector_contract,
        "rejected_root_count": rejected_root_count,
        "critical_actor_overflow_count": critical_actor_overflow_count,
        "task_actor_coverage_incomplete_count": (
            len(coverage_incomplete_root_ids)
        ),
        "task_actor_coverage_incomplete_root_ids": (
            coverage_incomplete_root_ids[:20]
        ),
        "coverage_incomplete_count": sum(
            1
            for row in roots
            if not bool(row.get("task_actor_coverage_complete", False))
            or not bool(
                row.get("risk_safety_actor_coverage_complete", False)
            )
        ),
        "actor_mapping_mismatch_count": (
            None
            if actor_contract_audit is None
            else int(actor_contract_audit["actor_mapping_mismatch_count"])
        ),
        "root_observation_fingerprint_mismatch_count": (
            None
            if actor_contract_audit is None
            else int(
                actor_contract_audit[
                    "root_observation_fingerprint_mismatch_count"
                ]
            )
        ),
        "protected_actor_coverage_rate": (
            None
            if actor_contract_audit is None
            else float(actor_contract_audit["protected_actor_coverage_rate"])
        ),
        "actor_contract_audit": actor_contract_audit,
        "source_manifests": source_manifests,
        "oracle_report": None if oracle is None else str(Path(oracle_report_path).resolve()),
        "conditions": conditions,
        "pilot_state": "pass" if all(conditions.values()) else "fail",
    }


def write_pilot_report(
    dataset_dir: str | Path,
    output_path: str | Path,
    **kwargs: Any,
) -> dict[str, Any]:
    report = validate_pilot_dataset(dataset_dir, **kwargs)
    write_json_atomic(output_path, report)
    return report
