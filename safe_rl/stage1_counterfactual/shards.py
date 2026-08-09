"""Immutable counterfactual shard and formal-dataset assembly helpers."""

# Shards belong to data generation; model datasets consume their merged form.

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from safe_rl.accvp.contracts.protocol import (
    ACCVP_DATA_CONTRACT_VERSION,
    ACCVP_SELECTOR3_DATA_CONTRACT_VERSION,
)
from safe_rl.accvp.contracts.schema import (
    ACTOR_ROW_MAPPING_VERSION,
    COUNTERFACTUAL_DATASET_MANIFEST_VERSION,
    COUNTERFACTUAL_SCHEMA_VERSION,
    ROOT_OBSERVATION_FINGERPRINT_VERSION,
    SCENARIO_EPISODE_KEY_VERSION,
    actor_row_mapping_hash,
    canonical_json,
    file_sha256,
    jsonl_sha256,
    read_json,
    root_observation_fingerprint,
    scenario_episode_key,
    stable_hash,
    validate_branch_row,
    write_json_atomic,
)


def _jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")
    temporary.replace(output)
    return output


def immutable_shard_dir(stage_dir: str | Path, output_name: str, collection_id: str) -> Path:
    """Return the deterministic location of one non-overwritable collection shard."""

    return Path(stage_dir) / str(output_name) / "shards" / str(collection_id)


def assert_new_shard(path: str | Path) -> Path:
    shard = Path(path)
    manifest = shard / "manifests" / "dataset_manifest.json"
    if manifest.exists():
        raise FileExistsError(
            f"counterfactual shard already exists and is immutable: {shard}; choose a new collection_id"
        )
    if shard.exists() and any(shard.iterdir()):
        raise FileExistsError(f"counterfactual shard path is not empty: {shard}")
    return shard


def shard_fingerprints(shard_dir: str | Path) -> dict[str, str]:
    shard = Path(shard_dir)
    manifests = shard / "manifests"
    rejected = manifests / "rejected_roots.jsonl"
    return {
        "dataset_manifest_sha256": file_sha256(manifests / "dataset_manifest.json"),
        "roots_manifest_sha256": jsonl_sha256(manifests / "roots.jsonl"),
        "branches_manifest_sha256": jsonl_sha256(manifests / "branches.jsonl"),
        "rejected_roots_manifest_sha256": (
            jsonl_sha256(rejected) if rejected.exists() else ""
        ),
    }


def _required_shard_manifest(shard: Path) -> dict[str, Any]:
    manifest_path = shard / "manifests" / "dataset_manifest.json"
    manifest = read_json(manifest_path)
    if int(manifest.get("counterfactual_schema_version", -1)) != COUNTERFACTUAL_SCHEMA_VERSION:
        raise ValueError(f"unsupported counterfactual schema in shard {shard}")
    if str(manifest.get("artifact_kind", "")) != "counterfactual_shard_v2":
        raise ValueError(f"not an immutable ACCVP shard: {shard}")
    required = (
        "collection_id",
        "scenario_config_hash",
        "action_execution_profile",
        "candidate_plan_profile",
        "risk_model_fingerprint",
        "config_hash",
        "data_contract",
        "data_contract_hash",
    )
    missing = [name for name in required if name not in manifest]
    if missing:
        raise ValueError(f"ACCVP shard manifest missing {missing}: {shard}")
    return manifest


def merge_counterfactual_shards(
    shard_dirs: Iterable[str | Path],
    output_dir: str | Path,
    *,
    require_frozen_risk_model: bool = True,
    expected_collection_phase: str | None = None,
) -> Path:
    """Assemble immutable shards into one formal, manifest-only dataset.

    Root and branch tensors stay in their shards. The formal dataset only owns
    immutable manifests with absolute references, avoiding copies and accidental
    overwrites of the source collection.
    """

    shards = [Path(value).resolve() for value in shard_dirs]
    if not shards:
        raise ValueError("at least one counterfactual shard is required")
    destination = Path(output_dir).resolve()
    manifest_dir = destination / "manifests"
    if (manifest_dir / "dataset_manifest.json").exists():
        raise FileExistsError(f"formal counterfactual dataset already exists: {destination}")
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"formal counterfactual dataset path is not empty: {destination}")

    manifests = [_required_shard_manifest(shard) for shard in shards]
    if expected_collection_phase is not None:
        invalid = [
            (str(shard), str(manifest.get("collection_phase", "ad_hoc")))
            for shard, manifest in zip(shards, manifests)
            if str(manifest.get("collection_phase", "ad_hoc")) != str(expected_collection_phase)
        ]
        if invalid:
            raise ValueError(f"counterfactual shards do not match collection phase {expected_collection_phase!r}: {invalid}")
    # Full collection config hashes intentionally differ: a frozen baseline
    # PPO and a WcDT-v3 merge-timing PPO need different observation settings.
    # Only the formal counterfactual data contract is required to match.
    baseline = dict(manifests[0]["data_contract"])
    baseline_hash = str(manifests[0]["data_contract_hash"])
    protocol_version = str(baseline.get("protocol_version", ""))
    strict_actor_rows = protocol_version in {
        ACCVP_DATA_CONTRACT_VERSION,
        ACCVP_SELECTOR3_DATA_CONTRACT_VERSION,
    }
    strict_selector_coverage = (
        protocol_version == ACCVP_SELECTOR3_DATA_CONTRACT_VERSION
    )
    if stable_hash(baseline) != baseline_hash:
        raise ValueError(f"invalid counterfactual data_contract hash in shard {shards[0]}")
    for shard, manifest in zip(shards[1:], manifests[1:]):
        contract = dict(manifest["data_contract"])
        contract_hash = str(manifest["data_contract_hash"])
        if stable_hash(contract) != contract_hash:
            raise ValueError(f"invalid counterfactual data_contract hash in shard {shard}")
        if contract != baseline or contract_hash != baseline_hash:
            mismatch = {
                name: (baseline.get(name), contract.get(name))
                for name in sorted(set(baseline) | set(contract))
                if baseline.get(name) != contract.get(name)
            }
            raise ValueError(f"incompatible counterfactual data contract in shard {shard}: {mismatch}")
    if require_frozen_risk_model and str(baseline["risk_model_fingerprint"]).startswith("heuristic:"):
        raise ValueError("formal ACCVP dataset requires a frozen Risk Module checkpoint, not heuristic risk")

    root_rows: list[dict[str, Any]] = []
    rejected_root_rows: list[dict[str, Any]] = []
    branch_rows: list[dict[str, Any]] = []
    root_ids: set[str] = set()
    root_state_fingerprints: dict[str, dict[str, str]] = {}
    root_mapping_hashes: dict[str, str] = {}
    root_actor_row_ids: dict[str, list[str]] = {}
    duplicate_root_state_fingerprints: list[dict[str, str]] = []
    shard_records: list[dict[str, Any]] = []
    for shard, manifest in zip(shards, manifests):
        roots_path = shard / "manifests" / "roots.jsonl"
        branches_path = shard / "manifests" / "branches.jsonl"
        fingerprints = shard_fingerprints(shard)
        shard_records.append(
            {
                "collection_id": str(manifest["collection_id"]),
                "collection_source": str(manifest.get("collection_source", manifest["collection_id"])),
                "collection_phase": str(manifest.get("collection_phase", "ad_hoc")),
                "path": str(shard),
                "config_hash": str(manifest["config_hash"]),
                "root_policy_checkpoint_fingerprint": str(manifest.get("root_policy_checkpoint_fingerprint", "")),
                "collection_job": dict(manifest.get("collection_job", {})),
                "complete_roots": int(manifest.get("complete_roots", 0)),
                "rejected_root_count": int(manifest.get("rejected_root_count", 0)),
                "critical_actor_overflow_count": int(
                    manifest.get("critical_actor_overflow_count", 0)
                ),
                "safety_actor_coverage_incomplete_count": int(
                    manifest.get("safety_actor_coverage_incomplete_count", 0)
                ),
                "task_actor_coverage_incomplete_count": int(
                    manifest.get(
                        "task_actor_coverage_incomplete_count", 0
                    )
                ),
                "risk_safety_actor_coverage_incomplete_count": int(
                    manifest.get(
                        "risk_safety_actor_coverage_incomplete_count", 0
                    )
                ),
                "failed_branches": int(manifest.get("failed_branches", 0)),
                "branch_status_counts": dict(manifest.get("branch_status_counts", {})),
                **fingerprints,
            }
        )
        shard_roots = _jsonl(roots_path)
        shard_branches = _jsonl(branches_path)
        rejected_path = shard / "manifests" / "rejected_roots.jsonl"
        shard_rejected = _jsonl(rejected_path) if rejected_path.exists() else []
        if strict_selector_coverage and len(shard_rejected) != int(
            manifest.get("rejected_root_count", -1)
        ):
            raise ValueError(
                f"selector3 shard rejected-root count mismatch: {shard}"
            )
        for rejected in shard_rejected:
            enriched_rejected = dict(rejected)
            enriched_rejected["source_shard_id"] = str(manifest["collection_id"])
            enriched_rejected["source_shard_path"] = str(shard)
            rejected_root_rows.append(enriched_rejected)
        completed = {str(row["root_id"]) for row in shard_roots if bool(row.get("complete", False))}
        for root in shard_roots:
            root_id = str(root["root_id"])
            if root_id in root_ids:
                raise ValueError(f"duplicate root_id across ACCVP shards: {root_id}")
            root_ids.add(root_id)
            if root_id not in completed:
                continue
            if str(root.get("data_contract_hash", "")) != baseline_hash:
                raise ValueError(f"root data-contract mismatch in shard {shard}: {root_id}")
            actor_row_ids: list[str] = []
            source_indices: list[int] = []
            mapping_hash = ""
            root_scenario_episode_key = ""
            if strict_actor_rows:
                root_scenario_episode_key = scenario_episode_key(
                    scenario_route_hash=str(baseline.get("scenario_route_hash", "")),
                    traffic_profile=str(root.get("traffic_profile", "")),
                    episode_seed=int(root["episode_seed"]),
                )
                root_metadata = read_json(root["metadata_path"])
                if str(root_metadata.get("traffic_profile", "")) != str(root.get("traffic_profile", "")):
                    raise ValueError(f"root traffic-profile mismatch in shard {shard}: {root_id}")
                if int(root_metadata.get("episode_seed", -1)) != int(root["episode_seed"]):
                    raise ValueError(f"root episode-seed mismatch in shard {shard}: {root_id}")
                for source_name, recorded_key in (
                    ("root manifest", str(root.get("scenario_episode_key", ""))),
                    ("root metadata", str(root_metadata.get("scenario_episode_key", ""))),
                ):
                    if recorded_key and recorded_key != root_scenario_episode_key:
                        raise ValueError(f"{source_name} scenario-episode key mismatch in shard {shard}: {root_id}")
                for source_name, recorded_route_hash in (
                    ("root manifest", str(root.get("scenario_route_hash", ""))),
                    ("root metadata", str(root_metadata.get("scenario_route_hash", ""))),
                    (
                        "root metadata data contract",
                        str(dict(root_metadata.get("data_contract", {})).get("scenario_route_hash", "")),
                    ),
                ):
                    if recorded_route_hash and recorded_route_hash != str(baseline["scenario_route_hash"]):
                        raise ValueError(f"{source_name} scenario-route hash mismatch in shard {shard}: {root_id}")
                actor_row_ids = [str(value) for value in root_metadata.get("actor_row_ids", [])]
                source_indices = [int(value) for value in root_metadata.get("actor_row_source_indices", [])]
                with np.load(root["tensor_path"], allow_pickle=False) as tensors:
                    actor_mask = np.asarray(tensors["mask"], dtype=np.float32)[0].reshape(-1)
                    tensor_indices = np.asarray(tensors["selected_indices"], dtype=np.int64)[0].reshape(-1).tolist()
                    root_tensor_values = {name: np.asarray(tensors[name]) for name in tensors.files}
                if tensor_indices != source_indices:
                    raise ValueError(f"root selected_indices mismatch in shard {shard}: {root_id}")
                mapping_hash = actor_row_mapping_hash(actor_row_ids, source_indices, actor_mask)
                if mapping_hash != str(root_metadata.get("actor_row_mapping_hash", "")):
                    raise ValueError(f"root actor-row mapping mismatch in shard {shard}: {root_id}")
                if str(root_metadata.get("actor_row_mapping_version", "")) != ACTOR_ROW_MAPPING_VERSION:
                    raise ValueError(f"root actor-row mapping version mismatch in shard {shard}: {root_id}")
                observation_fingerprint = str(root_metadata.get("root_observation_fingerprint", ""))
                if not observation_fingerprint:
                    raise ValueError(f"root observation fingerprint missing in shard {shard}: {root_id}")
                if str(root_metadata.get("root_observation_fingerprint_version", "")) != ROOT_OBSERVATION_FINGERPRINT_VERSION:
                    raise ValueError(f"root observation fingerprint version mismatch in shard {shard}: {root_id}")
                recomputed_fingerprint = root_observation_fingerprint(
                    actor_row_ids=actor_row_ids,
                    root_ego=dict(root_metadata.get("root_ego", {})),
                    data_contract_hash=str(root_metadata.get("data_contract_hash", "")),
                    tensors=root_tensor_values,
                )
                if recomputed_fingerprint != observation_fingerprint:
                    raise ValueError(f"root observation fingerprint content mismatch in shard {shard}: {root_id}")
                root_mapping_hashes[root_id] = mapping_hash
                root_actor_row_ids[root_id] = actor_row_ids
                if strict_selector_coverage:
                    selector = dict(root_metadata.get("selector", {}) or {})
                    required_coverage = {
                        "critical_actor_count": int(
                            root_metadata.get(
                                "critical_actor_count",
                                selector.get("critical_count", -1),
                            )
                        ),
                        "contextual_actor_count": int(
                            root_metadata.get(
                                "contextual_actor_count",
                                selector.get("contextual_count", -1),
                            )
                        ),
                        "critical_actor_overflow": bool(
                            root_metadata.get(
                                "critical_actor_overflow",
                                selector.get("critical_overflow", True),
                            )
                        ),
                        "dropped_critical_actor_ids": [
                            str(value)
                            for value in root_metadata.get(
                                "dropped_critical_actor_ids",
                                selector.get("dropped_critical_ids", []),
                            )
                        ],
                        "safety_actor_coverage_complete": bool(
                            root_metadata.get(
                                "safety_actor_coverage_complete",
                                False,
                            )
                        ),
                        "task_actor_coverage_complete": bool(
                            root_metadata.get(
                                "task_actor_coverage_complete", False
                            )
                        ),
                        "risk_safety_actor_coverage_complete": bool(
                            root_metadata.get(
                                "risk_safety_actor_coverage_complete", False
                            )
                        ),
                    }
                    if (
                        required_coverage["critical_actor_overflow"]
                        or required_coverage["dropped_critical_actor_ids"]
                        or not required_coverage[
                            "task_actor_coverage_complete"
                        ]
                        or not required_coverage[
                            "risk_safety_actor_coverage_complete"
                        ]
                    ):
                        raise ValueError(
                            "selector3 completed root has incomplete task/Risk actor "
                            f"coverage: {root_id}"
                        )
                    for key, expected in required_coverage.items():
                        if root.get(key) != expected:
                            raise ValueError(
                                f"selector3 root manifest {key} mismatch: {root_id}"
                            )
            else:
                observation_fingerprint = str(root.get("root_state_fingerprint", ""))
            root_state_fingerprint = observation_fingerprint
            if root_state_fingerprint:
                current = {
                    "root_id": root_id,
                    "collection_source": str(root.get("collection_source", manifest.get("collection_source", ""))),
                    "root_policy": str(root.get("root_policy", root.get("root_source", ""))),
                    "source_shard_id": str(manifest["collection_id"]),
                }
                previous = root_state_fingerprints.get(root_state_fingerprint)
                if previous is None:
                    root_state_fingerprints[root_state_fingerprint] = current
                else:
                    duplicate_root_state_fingerprints.append(
                        {
                            "root_state_fingerprint": root_state_fingerprint,
                            "first_root_id": previous["root_id"],
                            "first_collection_source": previous["collection_source"],
                            "first_root_policy": previous["root_policy"],
                            "first_source_shard_id": previous["source_shard_id"],
                            "duplicate_root_id": current["root_id"],
                            "duplicate_collection_source": current["collection_source"],
                            "duplicate_root_policy": current["root_policy"],
                            "duplicate_source_shard_id": current["source_shard_id"],
                        }
                    )
            enriched = dict(root)
            if strict_actor_rows:
                enriched.update(
                    {
                        "actor_row_ids": actor_row_ids,
                        "actor_row_source_indices": source_indices,
                        "actor_row_mapping_version": ACTOR_ROW_MAPPING_VERSION,
                        "actor_row_mapping_hash": mapping_hash,
                        "root_observation_fingerprint_version": ROOT_OBSERVATION_FINGERPRINT_VERSION,
                        "root_observation_fingerprint": observation_fingerprint,
                        "root_state_fingerprint": observation_fingerprint,
                        "scenario_episode_key_version": SCENARIO_EPISODE_KEY_VERSION,
                        "scenario_episode_key": root_scenario_episode_key,
                        "scenario_route_hash": str(baseline["scenario_route_hash"]),
                        "critical_actor_count": int(
                            root_metadata.get("critical_actor_count", 0)
                        ),
                        "contextual_actor_count": int(
                            root_metadata.get("contextual_actor_count", 0)
                        ),
                        "critical_actor_overflow": bool(
                            root_metadata.get("critical_actor_overflow", False)
                        ),
                        "dropped_critical_actor_ids": [
                            str(value)
                            for value in root_metadata.get(
                                "dropped_critical_actor_ids", []
                            )
                        ],
                        "safety_actor_coverage_complete": bool(
                            root_metadata.get(
                                "safety_actor_coverage_complete", False
                            )
                        ),
                        "task_actor_coverage_complete": bool(
                            root_metadata.get(
                                "task_actor_coverage_complete", False
                            )
                        ),
                        "risk_safety_actor_coverage_complete": bool(
                            root_metadata.get(
                                "risk_safety_actor_coverage_complete", False
                            )
                        ),
                    }
                )
            enriched["source_shard_id"] = str(manifest["collection_id"])
            enriched["source_shard_path"] = str(shard)
            root_rows.append(enriched)
        for branch in shard_branches:
            if str(branch.get("root_id", "")) not in completed:
                continue
            if str(branch.get("branch_status", "")) != "completed":
                continue
            if str(branch.get("data_contract_hash", "")) != baseline_hash:
                raise ValueError(f"branch data-contract mismatch in shard {shard}: {branch.get('branch_id')}")
            if "secondary_safety_pass" not in branch or not branch.get("risk_model_fingerprint"):
                raise ValueError(f"counterfactual branch is missing frozen secondary-risk metadata: {branch.get('branch_id')}")
            root_id = str(branch.get("root_id", ""))
            if strict_actor_rows:
                validate_branch_row(branch)
                if strict_selector_coverage and (
                    bool(branch.get("critical_actor_overflow", False))
                    or list(branch.get("dropped_critical_actor_ids", []) or [])
                    or not bool(
                        branch.get("task_actor_coverage_complete", False)
                    )
                    or not bool(
                        branch.get(
                            "risk_safety_actor_coverage_complete", False
                        )
                    )
                ):
                    raise ValueError(
                        "selector3 branch has incomplete task/Risk actor coverage: "
                        f"{branch.get('branch_id')}"
                    )
                if str(branch.get("actor_row_mapping_hash", "")) != root_mapping_hashes.get(root_id, ""):
                    raise ValueError(f"counterfactual branch actor-row mapping mismatch: {branch.get('branch_id')}")
                expected_actor_ids = root_actor_row_ids.get(root_id, [])
                if [str(value) for value in branch.get("actor_row_ids", [])] != expected_actor_ids:
                    raise ValueError(f"counterfactual branch actor-row IDs mismatch: {branch.get('branch_id')}")
                with np.load(branch["tensor_path"], allow_pickle=False) as tensors:
                    tensor_ids = [str(value) for value in np.asarray(tensors["actor_row_ids"]).tolist()]
                    response = np.asarray(tensors["actor_response"])
                    response_mask = np.asarray(tensors["actor_valid_mask"])
                if tensor_ids != expected_actor_ids:
                    raise ValueError(f"counterfactual branch tensor actor-row IDs mismatch: {branch.get('branch_id')}")
                if response.ndim != 3 or response.shape[-1] != 5 or response.shape[0] != len(tensor_ids):
                    raise ValueError(f"counterfactual branch actor_response shape mismatch: {branch.get('branch_id')}")
                if response_mask.shape != response.shape[:2]:
                    raise ValueError(f"counterfactual branch actor_valid_mask shape mismatch: {branch.get('branch_id')}")
            enriched = dict(branch)
            enriched["source_shard_id"] = str(manifest["collection_id"])
            enriched["source_shard_path"] = str(shard)
            branch_rows.append(enriched)

    if not root_rows or not branch_rows:
        raise ValueError("formal ACCVP dataset requires completed roots and branches")
    root_rows.sort(key=lambda row: str(row["root_id"]))
    branch_rows.sort(key=lambda row: (str(row["root_id"]), int(row["action_id"])))
    roots_path = _write_jsonl(manifest_dir / "roots.jsonl", root_rows)
    branches_path = _write_jsonl(manifest_dir / "branches.jsonl", branch_rows)
    rejected_roots_path = _write_jsonl(
        manifest_dir / "rejected_roots.jsonl",
        rejected_root_rows,
    )
    coverage = {
        "collection_source": dict(Counter(str(row.get("collection_source", "unknown")) for row in root_rows)),
        "root_policy": dict(Counter(str(row.get("root_policy", row.get("root_source", "unknown"))) for row in root_rows)),
        "traffic_profile": dict(Counter(str(row.get("traffic_profile", "unknown")) for row in root_rows)),
        "activation_bin": dict(Counter(str(row.get("activation_bin", "unknown")) for row in root_rows)),
        "deadline_bin": dict(Counter(str(row.get("deadline_bin", "unknown")) for row in root_rows)),
    }
    source_config_hashes = sorted({str(manifest["config_hash"]) for manifest in manifests})
    manifest = {
        "artifact_kind": "counterfactual_dataset_v2",
        "counterfactual_dataset_manifest_version": COUNTERFACTUAL_DATASET_MANIFEST_VERSION,
        "counterfactual_schema_version": COUNTERFACTUAL_SCHEMA_VERSION,
        "collection_phase": expected_collection_phase or "mixed_or_ad_hoc",
        "data_contract": baseline,
        "data_contract_hash": baseline_hash,
        "scenario_config_hash": str(baseline["scenario_config_hash"]),
        "scenario_route_hash": str(baseline["scenario_route_hash"]),
        "scenario_episode_key_version": SCENARIO_EPISODE_KEY_VERSION,
        "action_execution_profile": str(baseline["action_execution_profile"]),
        "candidate_plan_profile": str(baseline["candidate_plan_profile"]),
        "actor_row_mapping_version": str(baseline.get("actor_row_mapping_version", "")),
        "root_observation_fingerprint_version": str(baseline.get("root_observation_fingerprint_version", "")),
        "entry_time_label_version": str(baseline.get("entry_time_label_version", "")),
        "risk_model_fingerprint": str(baseline["risk_model_fingerprint"]),
        "accvp_activation_distance_m": float(baseline["activation_distance_m"]),
        "source_config_hashes": source_config_hashes,
        "config_hash": stable_hash({"source_config_hashes": source_config_hashes}),
        "source_shards": shard_records,
        "root_count": len(root_rows),
        "rejected_root_count": len(rejected_root_rows),
        "critical_actor_overflow_count": sum(
            bool(row.get("critical_actor_overflow", False))
            for row in [*root_rows, *rejected_root_rows]
        ),
        "safety_actor_coverage_incomplete_count": sum(
            not bool(row.get("task_actor_coverage_complete", False))
            for row in [*root_rows, *rejected_root_rows]
        ),
        "task_actor_coverage_incomplete_count": sum(
            not bool(row.get("task_actor_coverage_complete", False))
            for row in [*root_rows, *rejected_root_rows]
        ),
        "risk_safety_actor_coverage_incomplete_count": sum(
            not bool(
                row.get("risk_safety_actor_coverage_complete", False)
            )
            for row in [*root_rows, *rejected_root_rows]
        ),
        "critical_actor_count_histogram": dict(
            Counter(
                str(int(row.get("critical_actor_count", 0)))
                for row in [*root_rows, *rejected_root_rows]
            )
        ),
        "contextual_actor_count_histogram": dict(
            Counter(
                str(int(row.get("contextual_actor_count", 0)))
                for row in [*root_rows, *rejected_root_rows]
            )
        ),
        "branch_count": len(branch_rows),
        "unique_root_state_fingerprint_count": len(root_state_fingerprints),
        "duplicate_root_state_fingerprint_count": len(duplicate_root_state_fingerprints),
        "duplicate_root_state_fingerprints": duplicate_root_state_fingerprints[:100],
        "coverage": coverage,
        "roots_manifest_sha256": file_sha256(roots_path),
        "branches_manifest_sha256": file_sha256(branches_path),
        "rejected_roots_manifest_sha256": file_sha256(rejected_roots_path),
    }
    manifest["dataset_fingerprint"] = stable_hash(manifest)
    write_json_atomic(manifest_dir / "dataset_manifest.json", manifest)
    return destination
