"""Read-only feasibility audit for legacy ACCVP counterfactual datasets."""

# Migration is deliberately isolated from formal schema-v3 loading.

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from safe_rl.accvp.contracts.schema import COUNTERFACTUAL_SCHEMA_VERSION, file_sha256, read_json, stable_hash
from safe_rl.prediction.wcdt_v3_predictor import selected_vehicle_ids_from_indices


MIGRATION_AUDIT_VERSION = "accvp_schema3_migration_audit_v1"
MIGRATION_CLASSES = ("full_repairable", "task_only", "recollect_required")
_SCALAR_FIELDS = (
    "action_id",
    "event_observed",
    "censor_time",
    "viability_observation_status",
    "proxy_collision_within_horizon",
    "safety_violation_within_horizon",
    "taper_miss_observed",
    "merge_before_taper_observed",
    "min_obb_distance",
    "max_drac",
    "target_front_gap",
    "target_rear_gap",
)


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"required legacy ACCVP manifest is missing: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _artifact_path(dataset: Path, value: Any) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else (dataset / path).resolve()


def _manifest_hashes(manifests: Path) -> dict[str, str]:
    return {
        name: file_sha256(manifests / name)
        for name in ("dataset_manifest.json", "roots.jsonl", "branches.jsonl")
    }


def _scalar_label_errors(root: dict[str, Any], branches: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    expected = [int(value) for value in root.get("expected_action_ids", [])]
    completed = [row for row in branches if str(row.get("branch_status", "")) == "completed"]
    action_counts = Counter(int(row.get("action_id", -1)) for row in completed)
    if not completed:
        errors.append("no_completed_branches")
    if expected:
        missing = sorted(set(expected) - set(action_counts))
        duplicate = sorted(action for action, count in action_counts.items() if action in expected and count != 1)
        if missing:
            errors.append(f"missing_expected_actions:{','.join(map(str, missing))}")
        if duplicate:
            errors.append(f"duplicate_expected_actions:{','.join(map(str, duplicate))}")
    for row in completed:
        branch_id = str(row.get("branch_id", f"action{row.get('action_id', '?')}"))
        missing_fields = [name for name in _SCALAR_FIELDS if name not in row]
        if missing_fields:
            errors.append(f"{branch_id}:missing_scalar_fields:{','.join(missing_fields)}")
            continue
        status = str(row["viability_observation_status"])
        if status not in {"observed_success", "observed_failure", "censored"}:
            errors.append(f"{branch_id}:invalid_viability_status")
            continue
        if bool(row["event_observed"]) != (status != "censored"):
            errors.append(f"{branch_id}:event_observed_mismatch")
        target_entry = row.get("target_lane_entry_time_s")
        if (target_entry is not None) != (status == "observed_success"):
            errors.append(f"{branch_id}:entry_time_semantics_invalid")
        numeric = ["censor_time", "min_obb_distance", "max_drac", "target_front_gap", "target_rear_gap"]
        if target_entry is not None:
            numeric.append("target_lane_entry_time_s")
        for name in numeric:
            try:
                value = float(row[name])
            except (TypeError, ValueError):
                errors.append(f"{branch_id}:{name}_not_numeric")
                continue
            if not math.isfinite(value) or (name in {"censor_time", "target_lane_entry_time_s"} and value < 0.0):
                errors.append(f"{branch_id}:{name}_invalid")
    return errors


def _mapping_evidence(
    dataset: Path,
    root: dict[str, Any],
    branches: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    evidence: dict[str, Any] = {}
    try:
        metadata_path = _artifact_path(dataset, root["metadata_path"])
        tensor_path = _artifact_path(dataset, root["tensor_path"])
        metadata = read_json(metadata_path)
        with np.load(tensor_path, allow_pickle=False) as values:
            if "selected_indices" not in values.files or "mask" not in values.files:
                raise ValueError("root tensor lacks selected_indices or mask")
            selected_indices = np.asarray(values["selected_indices"], dtype=np.int64).reshape(-1)
            actor_mask = np.asarray(values["mask"], dtype=np.float32).reshape(-1)
        if str(metadata.get("root_id", "")) != str(root.get("root_id", "")):
            raise ValueError("root metadata ID mismatch")
        legacy_ids = [str(value) for value in metadata.get("selected_actor_ids", [])]
        selector_ids = [str(value) for value in dict(metadata.get("selector", {})).get("selected_actor_ids", [])]
        if not legacy_ids or len(legacy_ids) != len(set(legacy_ids)):
            raise ValueError("legacy selected_actor_ids are missing or non-unique")
        if selector_ids and selector_ids != legacy_ids:
            raise ValueError("legacy selector IDs disagree with response-row IDs")
        ego_id = str(metadata.get("ego_id", "ego"))
        actor_row_ids = selected_vehicle_ids_from_indices(
            [ego_id, *legacy_ids], selected_indices, actor_mask
        )
        permutation = [legacy_ids.index(vehicle_id) if vehicle_id else -1 for vehicle_id in actor_row_ids]
        evidence.update(
            {
                "legacy_response_row_ids": legacy_ids,
                "actor_row_ids": actor_row_ids,
                "actor_row_source_indices": selected_indices.tolist(),
                "legacy_to_actor_row_permutation": permutation,
                "mapping_evidence": "legacy_selected_actor_ids_plus_root_selected_indices",
            }
        )
    except (KeyError, OSError, ValueError, TypeError) as exc:
        return evidence, [f"root_mapping_unrecoverable:{exc}"]

    legacy_ids = evidence["legacy_response_row_ids"]
    actor_rows = len(evidence["actor_row_ids"])
    for row in branches:
        if str(row.get("branch_status", "")) != "completed":
            continue
        branch_id = str(row.get("branch_id", f"action{row.get('action_id', '?')}"))
        if [str(value) for value in row.get("selected_actor_ids", [])] != legacy_ids:
            errors.append(f"{branch_id}:branch_response_row_ids_mismatch")
            continue
        try:
            tensor_path = _artifact_path(dataset, row["tensor_path"])
            with np.load(tensor_path, allow_pickle=False) as values:
                response = np.asarray(values["actor_response"])
                valid = np.asarray(values["actor_valid_mask"])
            if response.ndim != 3 or response.shape[-1] != 5 or response.shape[0] != actor_rows:
                raise ValueError(f"invalid actor_response shape {response.shape}")
            if valid.shape != response.shape[:2]:
                raise ValueError(f"invalid actor_valid_mask shape {valid.shape}")
        except (KeyError, OSError, ValueError, TypeError) as exc:
            errors.append(f"{branch_id}:response_tensor_unrecoverable:{exc}")
    return evidence, errors


def audit_legacy_dataset_migration(dataset_dir: str | Path) -> dict[str, Any]:
    """Classify every legacy root without modifying any source artifact."""

    dataset = Path(dataset_dir).resolve()
    manifests = dataset / "manifests"
    before = _manifest_hashes(manifests)
    dataset_manifest = read_json(manifests / "dataset_manifest.json")
    source_schema = int(dataset_manifest.get("counterfactual_schema_version", -1))
    if source_schema < 1 or source_schema >= COUNTERFACTUAL_SCHEMA_VERSION:
        raise ValueError(
            f"migration audit requires a legacy schema below {COUNTERFACTUAL_SCHEMA_VERSION}; got {source_schema}"
        )
    roots = _jsonl(manifests / "roots.jsonl")
    branches = _jsonl(manifests / "branches.jsonl")
    branches_by_root: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in branches:
        branches_by_root[str(row.get("root_id", ""))].append(row)
    root_reports: list[dict[str, Any]] = []
    for root in roots:
        root_id = str(root.get("root_id", ""))
        scalar_errors = _scalar_label_errors(root, branches_by_root[root_id])
        evidence, mapping_errors = _mapping_evidence(dataset, root, branches_by_root[root_id])
        if scalar_errors or not bool(root.get("complete", False)):
            classification = "recollect_required"
            reasons = (["root_not_complete"] if not bool(root.get("complete", False)) else []) + scalar_errors
        elif mapping_errors:
            classification = "task_only"
            reasons = mapping_errors
        else:
            classification = "full_repairable"
            reasons = ["unique_actor_row_mapping_verified"]
        root_reports.append(
            {
                "root_id": root_id,
                "classification": classification,
                "reasons": reasons,
                "completed_branch_count": sum(
                    str(row.get("branch_status", "")) == "completed" for row in branches_by_root[root_id]
                ),
                **evidence,
            }
        )
    after = _manifest_hashes(manifests)
    counts = Counter(row["classification"] for row in root_reports)
    derivation_allowed = bool(root_reports) and counts["full_repairable"] == len(root_reports) and before == after
    report = {
        "artifact_kind": "accvp_schema3_migration_feasibility_audit",
        "audit_version": MIGRATION_AUDIT_VERSION,
        "read_only": True,
        "source_dataset": str(dataset),
        "source_schema_version": source_schema,
        "target_schema_version": COUNTERFACTUAL_SCHEMA_VERSION,
        "source_manifest_hashes_before": before,
        "source_manifest_hashes_after": after,
        "source_files_unchanged": before == after,
        "root_count": len(root_reports),
        "classification_counts": {name: int(counts[name]) for name in MIGRATION_CLASSES},
        "unique_mapping_rate": (
            float(counts["full_repairable"]) / len(root_reports) if root_reports else 0.0
        ),
        "schema3_derivation_allowed": derivation_allowed,
        "derivation_blockers": [] if derivation_allowed else [
            "schema3 derivation requires every root to have a unique, fully verified actor-row mapping"
        ],
        "roots": root_reports,
    }
    report["audit_fingerprint"] = stable_hash(report)
    return report


def assert_schema3_derivation_allowed(report: dict[str, Any]) -> None:
    if not bool(report.get("schema3_derivation_allowed", False)):
        raise ValueError("schema-v3 derivation is blocked: migration audit is not 100% full_repairable")
