"""Immutable counterfactual shard and formal-dataset assembly helpers."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from safe_rl.accvp.schema import (
    COUNTERFACTUAL_DATASET_MANIFEST_VERSION,
    COUNTERFACTUAL_SCHEMA_VERSION,
    canonical_json,
    file_sha256,
    jsonl_sha256,
    read_json,
    stable_hash,
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
    return {
        "dataset_manifest_sha256": file_sha256(manifests / "dataset_manifest.json"),
        "roots_manifest_sha256": jsonl_sha256(manifests / "roots.jsonl"),
        "branches_manifest_sha256": jsonl_sha256(manifests / "branches.jsonl"),
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
    branch_rows: list[dict[str, Any]] = []
    root_ids: set[str] = set()
    root_state_fingerprints: set[str] = set()
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
                "failed_branches": int(manifest.get("failed_branches", 0)),
                "branch_status_counts": dict(manifest.get("branch_status_counts", {})),
                **fingerprints,
            }
        )
        shard_roots = _jsonl(roots_path)
        shard_branches = _jsonl(branches_path)
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
            root_state_fingerprint = str(root.get("root_state_fingerprint", ""))
            if root_state_fingerprint:
                if root_state_fingerprint in root_state_fingerprints:
                    raise ValueError(f"duplicate root state across ACCVP shards: {root_state_fingerprint}")
                root_state_fingerprints.add(root_state_fingerprint)
            enriched = dict(root)
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
        "action_execution_profile": str(baseline["action_execution_profile"]),
        "candidate_plan_profile": str(baseline["candidate_plan_profile"]),
        "risk_model_fingerprint": str(baseline["risk_model_fingerprint"]),
        "accvp_activation_distance_m": float(baseline["activation_distance_m"]),
        "source_config_hashes": source_config_hashes,
        "config_hash": stable_hash({"source_config_hashes": source_config_hashes}),
        "source_shards": shard_records,
        "root_count": len(root_rows),
        "branch_count": len(branch_rows),
        "coverage": coverage,
        "roots_manifest_sha256": file_sha256(roots_path),
        "branches_manifest_sha256": file_sha256(branches_path),
    }
    manifest["dataset_fingerprint"] = stable_hash(manifest)
    write_json_atomic(manifest_dir / "dataset_manifest.json", manifest)
    return destination
