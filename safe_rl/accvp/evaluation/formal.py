"""Fail-closed validation for a merged strict-selector formal dataset."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from safe_rl.accvp.contracts.protocol import (
    ACCVP_SELECTOR3_DATA_CONTRACT_VERSION,
    counterfactual_data_contract_candidates,
    data_contract_hash,
    is_strict_selector_data_contract,
    is_selector4_data_contract,
)
from safe_rl.accvp.contracts.schema import (
    file_sha256,
    read_json,
    stable_hash,
    write_json_atomic,
)
from safe_rl.accvp.data.dataset import (
    SPLIT_ALGORITHM_VERSION,
    build_split_manifest,
)
from safe_rl.accvp.evaluation.dataset_integrity import (
    audit_dataset_actor_contract,
)


FORMAL_VALIDATION_KIND = "accvp_selector3_formal_validation_v1"


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _configured_contract_matches(
    config: Any,
    dataset_manifest: dict[str, Any],
) -> bool:
    contract = dict(dataset_manifest.get("data_contract", {}) or {})
    expected_contracts = counterfactual_data_contract_candidates(
        config,
        str(dataset_manifest.get("risk_model_fingerprint", "")),
    )
    return any(
        contract == expected_contract
        and str(dataset_manifest.get("data_contract_hash", ""))
        == data_contract_hash(expected_contract)
        for expected_contract in expected_contracts
    )


def validate_formal_dataset(
    config: Any,
    dataset_dir: str | Path,
    *,
    minimum_root_count: int,
    minimum_branch_success_rate: float = 0.99,
    excluded_episode_seeds: Iterable[int] = (),
    excluded_cohort_roles: Iterable[str] = (),
) -> dict[str, Any]:
    dataset = Path(dataset_dir).resolve()
    manifests = dataset / "manifests"
    dataset_manifest_path = manifests / "dataset_manifest.json"
    dataset_manifest = read_json(dataset_manifest_path)
    roots = [
        row
        for row in _jsonl(manifests / "roots.jsonl")
        if bool(row.get("complete", False))
    ]
    branches = [
        row
        for row in _jsonl(manifests / "branches.jsonl")
        if str(row.get("branch_status", "")) == "completed"
    ]
    contract = dict(dataset_manifest.get("data_contract", {}) or {})
    protocol_version = str(contract.get("protocol_version", ""))
    selector3_contract = protocol_version == ACCVP_SELECTOR3_DATA_CONTRACT_VERSION
    selector4_contract = is_selector4_data_contract(protocol_version)
    strict_selector_contract = is_strict_selector_data_contract(protocol_version)
    contract_matches = _configured_contract_matches(
        config,
        dataset_manifest,
    )

    source_shards = list(dataset_manifest.get("source_shards", []) or [])
    rejected_root_count = sum(
        int(row.get("rejected_root_count", 0)) for row in source_shards
    )
    source_overflow_count = sum(
        int(row.get("critical_actor_overflow_count", 0))
        for row in source_shards
    )
    source_incomplete_count = sum(
        int(row.get("task_actor_coverage_incomplete_count", 0))
        for row in source_shards
    )
    completed_branches = 0
    failed_branches = 0
    for source in source_shards:
        status = Counter(
            {
                str(key): int(value)
                for key, value in dict(
                    source.get("branch_status_counts", {}) or {}
                ).items()
            }
        )
        completed_branches += int(status.get("completed", 0))
        failed_branches += sum(
            count for name, count in status.items() if name != "completed"
        )
    branch_success_rate = float(completed_branches) / max(
        1, completed_branches + failed_branches
    )

    incomplete_roots = [
        str(row.get("root_id", ""))
        for row in roots
        if (
            not bool(row.get("task_actor_coverage_complete", False))
            or bool(row.get("critical_actor_overflow", False))
            or bool(list(row.get("dropped_critical_actor_ids", []) or []))
        )
    ]
    incomplete_branches = [
        str(row.get("branch_id", ""))
        for row in branches
        if (
            not bool(row.get("task_actor_coverage_complete", False))
            or bool(row.get("critical_actor_overflow", False))
            or bool(list(row.get("dropped_critical_actor_ids", []) or []))
        )
    ]
    actor_contract_audit = (
        audit_dataset_actor_contract(dataset)
        if strict_selector_contract
        else None
    )
    preliminary = {
        "artifact_kind": (
            str(dataset_manifest.get("artifact_kind", ""))
            == "counterfactual_dataset_v2"
        ),
        "formal_collection_phase": (
            str(dataset_manifest.get("collection_phase", "")) == "formal"
        ),
        # Compatibility key retained for existing report consumers. Its gate
        # semantics now mean any strict selector generation (v3 or v4).
        "selector3_data_contract": strict_selector_contract,
        "strict_selector_data_contract": strict_selector_contract,
        "configured_data_contract_match": contract_matches,
        "minimum_root_count": len(roots) >= int(minimum_root_count),
        "rejected_root_count_zero": (
            rejected_root_count == 0
            and int(dataset_manifest.get("rejected_root_count", -1)) == 0
        ),
        "critical_actor_overflow_zero": (
            source_overflow_count == 0
            and int(dataset_manifest.get("critical_actor_overflow_count", -1))
            == 0
        ),
        "task_actor_coverage_complete": (
            source_incomplete_count == 0
            and not incomplete_roots
            and not incomplete_branches
            and int(
                dataset_manifest.get(
                    "task_actor_coverage_incomplete_count", -1
                )
            )
            == 0
        ),
        "risk_safety_actor_coverage_complete": (
            int(
                dataset_manifest.get(
                    "risk_safety_actor_coverage_incomplete_count", -1
                )
            )
            == 0
            and all(
                bool(
                    row.get(
                        "risk_safety_actor_coverage_complete", False
                    )
                )
                for row in roots
            )
            and all(
                bool(
                    row.get(
                        "risk_safety_actor_coverage_complete", False
                    )
                )
                for row in branches
            )
        ),
        "branch_success_rate": (
            branch_success_rate >= float(minimum_branch_success_rate)
        ),
        "actor_mapping_mismatch_zero": bool(
            actor_contract_audit is not None
            and int(actor_contract_audit["actor_mapping_mismatch_count"]) == 0
        ),
        "root_observation_fingerprint_mismatch_zero": bool(
            actor_contract_audit is not None
            and int(
                actor_contract_audit[
                    "root_observation_fingerprint_mismatch_count"
                ]
            )
            == 0
        ),
        "protected_actor_coverage_complete": bool(
            actor_contract_audit is not None
            and int(actor_contract_audit["selector_metadata_missing_count"]) == 0
            and int(
                actor_contract_audit[
                    "protected_actor_coverage_incomplete_count"
                ]
            )
            == 0
            and float(actor_contract_audit["protected_actor_coverage_rate"])
            == 1.0
        ),
    }

    split_conditions = {
        "split_manifest_present": False,
        "split_algorithm_current": False,
        "cross_split_episode_overlap_zero": False,
        "cross_split_fingerprint_overlap_zero": False,
        "cross_split_scenario_episode_overlap_zero": False,
        "split_coverage_complete": False,
    }
    if all(preliminary.values()):
        split_path = manifests / "split_manifest.jsonl"
        if not split_path.exists():
            build_split_manifest(
                dataset,
                seed=int(config.run.seed),
                require_all_splits=True,
                excluded_episode_seeds=excluded_episode_seeds,
                excluded_cohort_roles=excluded_cohort_roles,
            )
        provenance = read_json(manifests / "split_provenance.json")
        split_rows = _jsonl(split_path)
        split_conditions = {
            "split_manifest_present": bool(split_rows),
            "split_algorithm_current": (
                str(provenance.get("split_algorithm_version", ""))
                == SPLIT_ALGORITHM_VERSION
            ),
            "cross_split_episode_overlap_zero": (
                int(provenance.get("cross_split_episode_overlap_count", -1))
                == 0
            ),
            "cross_split_fingerprint_overlap_zero": (
                int(
                    provenance.get(
                        "cross_split_observation_fingerprint_overlap_count",
                        -1,
                    )
                )
                == 0
            ),
            "cross_split_scenario_episode_overlap_zero": (
                int(
                    provenance.get(
                        "cross_split_scenario_episode_overlap_count", -1
                    )
                )
                == 0
            ),
            "split_coverage_complete": (
                int(
                    provenance.get(
                        "task_actor_coverage_incomplete_root_count", -1
                    )
                )
                == 0
            ),
        }

    conditions = {**preliminary, **split_conditions}
    report = {
        "artifact_kind": FORMAL_VALIDATION_KIND,
        "dataset_dir": str(dataset),
        "dataset_manifest_sha256": file_sha256(dataset_manifest_path),
        "dataset_fingerprint": str(
            dataset_manifest.get("dataset_fingerprint", "")
        ),
        "data_contract_hash": str(
            dataset_manifest.get("data_contract_hash", "")
        ),
        "data_contract_version": protocol_version,
        "selector3_contract": selector3_contract,
        "selector4_contract": selector4_contract,
        "strict_selector_contract": strict_selector_contract,
        "root_count": len(roots),
        "branch_count": len(branches),
        "minimum_root_count": int(minimum_root_count),
        "completed_branch_count": completed_branches,
        "failed_branch_count": failed_branches,
        "branch_success_rate": branch_success_rate,
        "rejected_root_count": rejected_root_count,
        "critical_actor_overflow_count": source_overflow_count,
        "task_actor_coverage_incomplete_root_count": len(
            incomplete_roots
        ),
        "task_actor_coverage_incomplete_branch_count": len(
            incomplete_branches
        ),
        "task_actor_coverage_incomplete_root_ids": incomplete_roots[:20],
        "task_actor_coverage_incomplete_branch_ids": incomplete_branches[:20],
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
        "conditions": conditions,
        "formal_state": "pass" if all(conditions.values()) else "fail",
    }
    report["report_fingerprint"] = stable_hash(report)
    return report


def write_formal_report(
    config: Any,
    dataset_dir: str | Path,
    output_path: str | Path,
    **kwargs: Any,
) -> dict[str, Any]:
    report = validate_formal_dataset(config, dataset_dir, **kwargs)
    write_json_atomic(output_path, report)
    return report
