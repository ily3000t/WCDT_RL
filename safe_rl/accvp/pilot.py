"""Deterministic ACCVP-240 pilot acceptance checks before formal collection."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from safe_rl.accvp.schema import read_json, write_json_atomic


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
    for source in manifest.get("source_shards", []):
        path = Path(str(source["path"])) / "manifests" / "dataset_manifest.json"
        source_manifest = read_json(path)
        status = Counter({str(key): int(value) for key, value in dict(source_manifest.get("branch_status_counts", {})).items()})
        completed_branches += int(status.get("completed", 0))
        failed_branches += sum(value for key, value in status.items() if key != "completed")
        source_manifests.append(
            {
                "collection_id": str(source_manifest["collection_id"]),
                "collection_source": str(source_manifest.get("collection_source", "unknown")),
                "manifest_path": str(path),
                "branch_status_counts": dict(status),
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
    oracle = None
    if oracle_report_path is not None:
        oracle = read_json(oracle_report_path)
        oracle_matches_dataset = str(oracle.get("dataset_provenance", {}).get("dataset_fingerprint", "")) == str(
            manifest.get("dataset_fingerprint", "")
        )
        conditions["seed2_5_oracle"] = bool(
            oracle_matches_dataset
            and str(oracle.get("oracle_state", "")) == "go"
            and [int(value) for value in oracle.get("required_seeds", [])] == [2, 5]
            and str(oracle.get("root_policy", "")) == "merge_timing"
        )
    return {
        "dataset_dir": str(dataset),
        "dataset_fingerprint": str(manifest.get("dataset_fingerprint", "")),
        "data_contract_hash": str(manifest.get("data_contract_hash", "")),
        "accvp_activation_distance_m": float(manifest.get("accvp_activation_distance_m", -1.0)),
        "source_coverage": source_coverage,
        "branch_success_rate": branch_success_rate,
        "observed_viability_fraction": observed_viability_fraction,
        "activation_branch_count": len(activation_branches),
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
