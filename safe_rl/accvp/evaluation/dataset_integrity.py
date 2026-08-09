"""Read-only integrity audits shared by strict selector dataset gates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from safe_rl.accvp.contracts.schema import (
    ACTOR_ROW_MAPPING_VERSION,
    ROOT_OBSERVATION_FINGERPRINT_VERSION,
    actor_row_mapping_hash,
    root_observation_fingerprint,
    validate_branch_row,
)


AUDIT_SAMPLE_LIMIT = 20


def _jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _protected_actor_ids(selector: dict[str, Any]) -> set[str]:
    actor_metadata = dict(selector.get("actor_metadata", {}) or {})
    protected: set[str] = set()
    for vehicle_id, raw in actor_metadata.items():
        metadata = dict(raw or {})
        reasons = {str(value) for value in metadata.get("relevance_reasons", []) or []}
        if (
            str(metadata.get("role", "")) in {"target_front", "target_rear"}
            or bool(metadata.get("candidate_conflict_eligible", False))
            or bool(metadata.get("nearest_candidate_conflict", False))
            or "lowest_ttc" in reasons
        ):
            protected.add(str(vehicle_id))
    return protected


def audit_dataset_actor_contract(dataset_dir: str | Path) -> dict[str, Any]:
    """Audit every root/branch mapping and protected-actor selection.

    A completed merge normally proves the same mapping invariant.  Rechecking
    it here makes that evidence explicit in pilot/formal reports and also
    protects datasets created by older merge implementations.
    """

    dataset = Path(dataset_dir).resolve()
    manifests = dataset / "manifests"
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
    mapping_by_root: dict[str, tuple[str, list[str], list[int]]] = {}
    mismatch_examples: list[dict[str, str]] = []
    mapping_mismatch_count = 0
    root_fingerprint_mismatch_count = 0
    root_fingerprint_mismatch_examples: list[dict[str, str]] = []
    protected_required = 0
    protected_covered = 0
    protected_incomplete_root_ids: list[str] = []
    selector_metadata_missing_root_ids: list[str] = []

    def mismatch(kind: str, record_id: str, error: Exception | str) -> None:
        nonlocal mapping_mismatch_count
        mapping_mismatch_count += 1
        if len(mismatch_examples) < AUDIT_SAMPLE_LIMIT:
            mismatch_examples.append(
                {"kind": kind, "record_id": record_id, "error": str(error)}
            )

    for root in roots:
        root_id = str(root.get("root_id", ""))
        try:
            metadata_path = Path(str(root["metadata_path"]))
            with metadata_path.open("r", encoding="utf-8") as handle:
                metadata = json.load(handle)
            actor_ids = [str(value) for value in metadata.get("actor_row_ids", [])]
            source_indices = [
                int(value) for value in metadata.get("actor_row_source_indices", [])
            ]
            with np.load(Path(str(root["tensor_path"])), allow_pickle=False) as values:
                root_tensors = {
                    name: np.asarray(values[name]) for name in values.files
                }
                mask = np.asarray(root_tensors["mask"], dtype=np.float32).reshape(-1)
                tensor_indices = np.asarray(
                    root_tensors["selected_indices"], dtype=np.int64
                ).reshape(-1).tolist()
            if tensor_indices != source_indices:
                raise ValueError("root selected_indices mismatch")
            computed = actor_row_mapping_hash(actor_ids, source_indices, mask)
            recorded = str(metadata.get("actor_row_mapping_hash", ""))
            if computed != recorded:
                raise ValueError("root actor-row mapping hash mismatch")
            if str(metadata.get("actor_row_mapping_version", "")) != ACTOR_ROW_MAPPING_VERSION:
                raise ValueError("root actor-row mapping version mismatch")
            if root.get("actor_row_mapping_hash") not in {None, "", recorded}:
                raise ValueError("merged root actor-row mapping hash mismatch")
            merged_actor_ids = root.get("actor_row_ids")
            if merged_actor_ids not in (None, []) and [
                str(value) for value in merged_actor_ids
            ] != actor_ids:
                raise ValueError("merged root actor-row IDs mismatch")
            mapping_by_root[root_id] = (recorded, actor_ids, source_indices)

            try:
                if str(
                    metadata.get("root_observation_fingerprint_version", "")
                ) != ROOT_OBSERVATION_FINGERPRINT_VERSION:
                    raise ValueError("root observation fingerprint version mismatch")
                expected_fingerprint = root_observation_fingerprint(
                    actor_row_ids=actor_ids,
                    root_ego=dict(metadata.get("root_ego", {})),
                    data_contract_hash=str(metadata.get("data_contract_hash", "")),
                    tensors={
                        name: (
                            np.asarray(value)[0]
                            if np.asarray(value).ndim > 0
                            and np.asarray(value).shape[0] == 1
                            else np.asarray(value)
                        )
                        for name, value in root_tensors.items()
                    },
                )
                recorded_fingerprint = str(
                    metadata.get("root_observation_fingerprint", "")
                )
                if (
                    not recorded_fingerprint
                    or expected_fingerprint != recorded_fingerprint
                    or str(root.get("root_observation_fingerprint", ""))
                    != recorded_fingerprint
                ):
                    raise ValueError("root observation fingerprint mismatch")
            except Exception as exc:
                root_fingerprint_mismatch_count += 1
                if len(root_fingerprint_mismatch_examples) < AUDIT_SAMPLE_LIMIT:
                    root_fingerprint_mismatch_examples.append(
                        {"root_id": root_id, "error": str(exc)}
                    )

            selector = dict(metadata.get("selector", {}) or {})
            actor_metadata = dict(selector.get("actor_metadata", {}) or {})
            if not selector or not actor_metadata:
                selector_metadata_missing_root_ids.append(root_id)
            else:
                protected = _protected_actor_ids(selector)
                selected = {
                    str(value)
                    for value in selector.get("selected_actor_ids", []) or []
                    if str(value)
                }
                protected_required += len(protected)
                protected_covered += len(protected.intersection(selected))
                if not protected.issubset(selected):
                    protected_incomplete_root_ids.append(root_id)
        except Exception as exc:  # report all corrupt records, not only the first
            mismatch("root", root_id, exc)

    for branch in branches:
        branch_id = str(branch.get("branch_id", ""))
        root_id = str(branch.get("root_id", ""))
        try:
            validate_branch_row(branch)
            mapping_hash, actor_ids, source_indices = mapping_by_root[root_id]
            if str(branch.get("actor_row_mapping_hash", "")) != mapping_hash:
                raise ValueError("branch actor-row mapping hash mismatch")
            if [str(value) for value in branch.get("actor_row_ids", [])] != actor_ids:
                raise ValueError("branch actor-row IDs mismatch")
            if [int(value) for value in branch.get("actor_row_source_indices", [])] != source_indices:
                raise ValueError("branch actor-row source indices mismatch")
            with np.load(Path(str(branch["tensor_path"])), allow_pickle=False) as values:
                tensor_ids = [
                    str(value)
                    for value in np.asarray(values["actor_row_ids"]).reshape(-1).tolist()
                ]
                response = np.asarray(values["actor_response"])
                response_mask = np.asarray(values["actor_valid_mask"])
            if tensor_ids != actor_ids:
                raise ValueError("branch tensor actor-row IDs mismatch")
            if response.ndim != 3 or response.shape[-1] != 5:
                raise ValueError("branch response shape mismatch")
            if response.shape[0] != len(actor_ids):
                raise ValueError("branch response row count mismatch")
            if response_mask.shape != response.shape[:2]:
                raise ValueError("branch response mask shape mismatch")
        except Exception as exc:  # report all corrupt records, not only the first
            mismatch("branch", branch_id, exc)

    protected_rate = (
        float(protected_covered) / float(protected_required)
        if protected_required
        else (0.0 if selector_metadata_missing_root_ids else 1.0)
    )
    return {
        "audit_kind": "accvp_dataset_actor_contract_audit_v1",
        "root_count": len(roots),
        "completed_branch_count": len(branches),
        "actor_mapping_mismatch_count": int(mapping_mismatch_count),
        "actor_mapping_mismatch_examples": mismatch_examples,
        "actor_mapping_sample_limit": AUDIT_SAMPLE_LIMIT,
        "root_observation_fingerprint_mismatch_count": int(
            root_fingerprint_mismatch_count
        ),
        "root_observation_fingerprint_mismatch_examples": (
            root_fingerprint_mismatch_examples
        ),
        "selector_metadata_missing_count": len(selector_metadata_missing_root_ids),
        "selector_metadata_missing_root_ids": selector_metadata_missing_root_ids[
            :AUDIT_SAMPLE_LIMIT
        ],
        "protected_actor_required_count": int(protected_required),
        "protected_actor_covered_count": int(protected_covered),
        "protected_actor_coverage_rate": float(protected_rate),
        "protected_actor_coverage_incomplete_count": len(
            protected_incomplete_root_ids
        ),
        "protected_actor_coverage_incomplete_root_ids": (
            protected_incomplete_root_ids[:AUDIT_SAMPLE_LIMIT]
        ),
    }
