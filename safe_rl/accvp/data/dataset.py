"""Leakage-safe split construction and ACCVP training dataset loading."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from safe_rl.accvp.planning.candidate_plan import build_commitment_plan
from safe_rl.accvp.contracts.protocol import is_strict_selector_data_contract
from safe_rl.accvp.contracts.schema import (
    COUNTERFACTUAL_SCHEMA_VERSION,
    ENTRY_TIME_LABEL_VERSION,
    ROOT_OBSERVATION_FINGERPRINT_VERSION,
    SCENARIO_EPISODE_KEY_VERSION,
    actor_row_mapping_hash,
    read_json,
    root_observation_fingerprint,
    scenario_episode_key,
)
from safe_rl.sim.action_space import decode_action
from safe_rl.sim.types import VehicleState


SPLIT_RATIOS = {
    "train": 0.55,
    "validation": 0.15,
    "calibration": 0.10,
    "operating_point": 0.10,
    "test": 0.10,
}
SPLIT_ALGORITHM_VERSION = "fingerprint_scenario_component_v3"


def _is_activation_window(row: dict[str, Any]) -> bool:
    """Use v2 terminology while retaining read-only support for v1 diagnostics."""

    return str(row.get("activation_bin", row.get("deadline_bin", ""))) in {"activation_window", "deadline"}


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _root_group_id(row: dict[str, Any]) -> str:
    return str(row.get("root_episode_id") or f"{row.get('root_policy', row.get('root_source', 'unknown'))}:{row['episode_seed']}")


def _root_observation_fingerprint(row: dict[str, Any]) -> str:
    return str(row.get("root_observation_fingerprint") or row.get("root_state_fingerprint") or "")


def _dataset_scenario_route_hash(dataset_manifest: dict[str, Any]) -> str:
    contract = dataset_manifest.get("data_contract", {})
    contract_hash = str(contract.get("scenario_route_hash") or "") if isinstance(contract, dict) else ""
    manifest_hash = str(dataset_manifest.get("scenario_route_hash") or "")
    if manifest_hash and contract_hash and manifest_hash != contract_hash:
        raise ValueError("ACCVP dataset manifest scenario_route_hash disagrees with its data contract")
    return manifest_hash or contract_hash


def _root_scenario_episode_key(row: dict[str, Any], dataset_scenario_route_hash: str = "") -> str:
    contract = row.get("data_contract", {})
    contract_hash = str(contract.get("scenario_route_hash") or "") if isinstance(contract, dict) else ""
    root_hash = str(row.get("scenario_route_hash") or "")
    route_hashes = {value for value in (str(dataset_scenario_route_hash), contract_hash, root_hash) if value}
    if len(route_hashes) > 1:
        raise ValueError(f"ACCVP root scenario_route_hash mismatch: {row.get('root_id')}")
    route_hash = next(iter(route_hashes), "")
    traffic_profile = str(row.get("traffic_profile") or "").strip()
    if not route_hash or not traffic_profile or row.get("episode_seed") is None:
        return ""
    expected = scenario_episode_key(
        scenario_route_hash=route_hash,
        traffic_profile=traffic_profile,
        episode_seed=int(row["episode_seed"]),
    )
    recorded = str(row.get("scenario_episode_key", ""))
    if recorded and recorded != expected:
        raise ValueError(f"ACCVP root scenario_episode_key mismatch: {row.get('root_id')}")
    recorded_version = str(row.get("scenario_episode_key_version", ""))
    if recorded_version and recorded_version != SCENARIO_EPISODE_KEY_VERSION:
        raise ValueError(f"ACCVP root scenario_episode_key version mismatch: {row.get('root_id')}")
    return expected


def _connected_root_groups(
    roots: list[dict[str, Any]],
    *,
    scenario_route_hash: str = "",
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, str]]:
    """Union roots sharing an episode, model input, or traffic realization."""

    root_ids = [str(row["root_id"]) for row in roots]
    if len(root_ids) != len(set(root_ids)):
        raise ValueError("ACCVP roots manifest contains duplicate root_id values")
    parent = {root_id: root_id for root_id in root_ids}

    def find(value: str) -> str:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        first, second = sorted((left_root, right_root))
        parent[second] = first

    relation_owner: dict[tuple[str, str], str] = {}
    for row in roots:
        root_id = str(row["root_id"])
        relations = [("episode", _root_group_id(row))]
        fingerprint = _root_observation_fingerprint(row)
        if fingerprint:
            relations.append(("observation", fingerprint))
        scenario_key = _root_scenario_episode_key(row, scenario_route_hash)
        if scenario_key:
            relations.append(("scenario_episode", scenario_key))
        for relation in relations:
            previous = relation_owner.get(relation)
            if previous is None:
                relation_owner[relation] = root_id
            else:
                union(root_id, previous)

    members_by_representative: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in roots:
        members_by_representative[find(str(row["root_id"]))].append(row)
    grouped: dict[str, list[dict[str, Any]]] = {}
    component_by_root: dict[str, str] = {}
    for members in members_by_representative.values():
        ids = sorted(str(row["root_id"]) for row in members)
        component_id = f"component:{hashlib.sha256('|'.join(ids).encode('utf-8')).hexdigest()}"
        grouped[component_id] = members
        for root_id in ids:
            component_by_root[root_id] = component_id
    return grouped, component_by_root


def _split_quotas(group_count: int, require_all_splits: bool) -> dict[str, int]:
    names = list(SPLIT_RATIOS)
    if require_all_splits and group_count < len(names):
        raise ValueError(
            f"ACCVP requires at least {len(names)} grouped root episodes for train/validation/calibration/"
            f"operating-point/test separation; found {group_count}"
        )
    raw = {name: float(ratio) * group_count for name, ratio in SPLIT_RATIOS.items()}
    quotas = {name: int(np.floor(value)) for name, value in raw.items()}
    if require_all_splits:
        for name in names:
            quotas[name] = max(1, quotas[name])
    while sum(quotas.values()) > group_count:
        removable = [name for name in names if quotas[name] > (1 if require_all_splits else 0)]
        name = min(removable, key=lambda item: (raw[item] - quotas[item], SPLIT_RATIOS[item]))
        quotas[name] -= 1
    while sum(quotas.values()) < group_count:
        name = max(names, key=lambda item: (raw[item] - quotas[item], SPLIT_RATIOS[item]))
        quotas[name] += 1
    return quotas


def build_split_manifest(
    dataset_dir: str | Path,
    *,
    seed: int = 0,
    require_all_splits: bool = True,
    excluded_episode_seeds: Iterable[int] = (),
    excluded_cohort_roles: Iterable[str] = (),
) -> list[dict[str, Any]]:
    """Build model splits while fail-closing explicitly oracle-only roots.

    Oracle-regression observations establish a historical premise; they are
    not model samples.  Explicit root markers are always honoured.  The
    configured seed/role exclusions additionally protect formal datasets that
    predate propagation of those markers.
    """

    dataset = Path(dataset_dir)
    complete_roots = [
        row
        for row in _jsonl(dataset / "manifests" / "roots.jsonl")
        if bool(row.get("complete", False))
    ]
    excluded_seeds = {int(value) for value in excluded_episode_seeds}
    excluded_roles = {str(value) for value in excluded_cohort_roles if str(value)}

    def excluded_reason(row: dict[str, Any]) -> str | None:
        if bool(row.get("oracle_only", False)):
            return "oracle_only"
        if bool(row.get("exclude_from_model_splits", False)):
            return "exclude_from_model_splits"
        if str(row.get("cohort_role", "")) in excluded_roles:
            return "cohort_role"
        if int(row.get("episode_seed", -1)) in excluded_seeds:
            return "episode_seed"
        return None

    excluded_roots = [row for row in complete_roots if excluded_reason(row) is not None]
    roots = [row for row in complete_roots if excluded_reason(row) is None]
    if not roots:
        raise ValueError("ACCVP model split has no roots after excluding oracle-only cohorts")
    dataset_manifest_path = dataset / "manifests" / "dataset_manifest.json"
    dataset_manifest = read_json(dataset_manifest_path) if dataset_manifest_path.exists() else {}
    strict_selector_contract = is_strict_selector_data_contract(
        dict(dataset_manifest.get("data_contract", {}) or {}).get(
            "protocol_version", ""
        )
    )
    incomplete_coverage_roots = [
        str(row.get("root_id", ""))
        for row in complete_roots
        if (
            not bool(row.get("task_actor_coverage_complete", False))
            or bool(row.get("critical_actor_overflow", False))
            or bool(list(row.get("dropped_critical_actor_ids", []) or []))
        )
    ]
    if strict_selector_contract and incomplete_coverage_roots:
        raise ValueError(
            "selector3 ACCVP roots with incomplete task actor coverage cannot "
            f"enter any split: {incomplete_coverage_roots[:10]}"
        )
    strict_fingerprints = int(dataset_manifest.get("counterfactual_schema_version", -1)) >= COUNTERFACTUAL_SCHEMA_VERSION
    dataset_scenario_route_hash = _dataset_scenario_route_hash(dataset_manifest)
    missing_fingerprints = [str(row.get("root_id", "")) for row in roots if not _root_observation_fingerprint(row)]
    scenario_keys = {
        str(row["root_id"]): _root_scenario_episode_key(row, dataset_scenario_route_hash)
        for row in roots
    }
    missing_scenario_keys = [root_id for root_id, key in scenario_keys.items() if not key]
    if strict_fingerprints and missing_fingerprints:
        raise ValueError(
            "schema-v3 ACCVP split requires a root_observation_fingerprint for every complete root; "
            f"missing={missing_fingerprints[:10]}"
        )
    if strict_fingerprints and missing_scenario_keys:
        raise ValueError(
            "schema-v3 ACCVP split requires scenario_route_hash, traffic_profile, and episode_seed "
            f"for every complete root; missing={missing_scenario_keys[:10]}"
        )
    grouped, component_by_root = _connected_root_groups(
        roots,
        scenario_route_hash=dataset_scenario_route_hash,
    )
    quotas = _split_quotas(len(grouped), require_all_splits)
    group_items = []
    for group_id, members in grouped.items():
        # Balance independent marginals without creating a sparse
        # combinatorial signature that would produce singleton strata.
        # A connected component may span policies, collection sources and
        # activation bins. Preserve every represented marginal rather than
        # allowing the first root's provenance to stand in for the component.
        marginals = Counter(
            value
            for member in members
            for value in (
                f"policy:{member.get('root_policy', member.get('root_source', 'unknown'))}",
                f"source:{member.get('collection_source', member.get('root_policy', member.get('root_source', 'unknown')))}",
                f"traffic:{member.get('traffic_profile', 'unknown')}",
                f"activation:{member.get('activation_bin', member.get('deadline_bin', 'unknown'))}",
            )
        )
        digest = hashlib.sha256(f"{seed}:{group_id}".encode("utf-8")).hexdigest()
        group_items.append((marginals, digest, group_id, members))
    marginal_sizes: Counter[str] = Counter()
    for component_marginals, _digest, _group_id, _members in group_items:
        marginal_sizes.update(component_marginals)
    group_items.sort(
        key=lambda item: (
            min(marginal_sizes[value] for value in item[0]),
            -sum(item[0].values()),
            item[1],
        )
    )
    assigned: Counter[str] = Counter()
    marginal_counts: dict[str, Counter[str]] = {name: Counter() for name in SPLIT_RATIOS}
    assignments: dict[str, str] = {}
    for marginals, _digest, group_id, _members in group_items:
        available = [name for name in SPLIT_RATIOS if assigned[name] < quotas[name]]

        def marginal_balance_delta(name: str) -> float:
            delta = 0.0
            for value, count in marginals.items():
                target = float(SPLIT_RATIOS[name]) * float(marginal_sizes[value])
                scale = max(1.0, target)
                before = (float(marginal_counts[name][value]) - target) / scale
                after = (float(marginal_counts[name][value] + count) - target) / scale
                delta += after * after - before * before
            return delta

        split = min(
            available,
            key=lambda name: (
                marginal_balance_delta(name),
                assigned[name] / max(1, quotas[name]),
                hashlib.sha256(f"{seed}:{group_id}:{name}".encode("utf-8")).hexdigest(),
            ),
        )
        assignments[group_id] = split
        assigned[split] += 1
        marginal_counts[split].update(marginals)
    if require_all_splits and any(assigned[name] == 0 for name in SPLIT_RATIOS):
        raise RuntimeError(f"ACCVP split assignment left an empty split: {dict(assigned)}")
    rows: list[dict[str, Any]] = []
    for group_id, members in grouped.items():
        for root in members:
            rows.append(
                {
                    "root_id": root["root_id"],
                    "root_episode_id": _root_group_id(root),
                    "root_observation_fingerprint": _root_observation_fingerprint(root),
                    "scenario_episode_key_version": SCENARIO_EPISODE_KEY_VERSION,
                    "scenario_episode_key": scenario_keys[str(root["root_id"])],
                    "scenario_route_hash": dataset_scenario_route_hash,
                    "split_component_id": group_id,
                    "episode_seed": int(root["episode_seed"]),
                    "primary_stratum": (
                        f"{root.get('root_policy', root.get('root_source', 'unknown'))}:"
                        f"{root.get('activation_bin', root.get('deadline_bin', 'unknown'))}"
                    ),
                    "split": assignments[group_id],
                }
            )
    provenance = {
        "split_algorithm_version": SPLIT_ALGORITHM_VERSION,
        "component_marginal_balance_version": "root_count_target_deviation_v1",
        "split_ratios": SPLIT_RATIOS,
        "component_count": len(grouped),
        "root_count": len(roots),
        "complete_root_count_before_oracle_exclusion": len(complete_roots),
        "oracle_exclusion_contract_version": "oracle_cohort_exclusion_v1",
        "configured_excluded_episode_seeds": sorted(excluded_seeds),
        "configured_excluded_cohort_roles": sorted(excluded_roles),
        "excluded_oracle_root_count": len(excluded_roots),
        "excluded_oracle_root_ids": sorted(str(row.get("root_id", "")) for row in excluded_roots),
        "excluded_oracle_root_reason_counts": dict(
            Counter(str(excluded_reason(row)) for row in excluded_roots)
        ),
        "task_actor_coverage_incomplete_root_count": len(
            incomplete_coverage_roots
        ),
        "missing_observation_fingerprint_count": len(missing_fingerprints),
        "scenario_episode_key_version": SCENARIO_EPISODE_KEY_VERSION,
        "scenario_route_hash": dataset_scenario_route_hash,
        "scenario_episode_count": len({key for key in scenario_keys.values() if key}),
        "missing_scenario_episode_key_count": len(missing_scenario_keys),
        "group_quotas": quotas,
        "group_counts": dict(assigned),
        "root_counts": {
            name: sum(
                len(grouped[component_id])
                for component_id, assigned_name in assignments.items()
                if assigned_name == name
            )
            for name in SPLIT_RATIOS
        },
        "source_counts": {
            name: dict(Counter(str(root.get("root_policy", root.get("root_source", "unknown"))) for root in roots if assignments[component_by_root[str(root["root_id"])]] == name))
            for name in SPLIT_RATIOS
        },
        "collection_source_counts": {
            name: dict(
                Counter(
                    str(root.get("collection_source", root.get("root_policy", root.get("root_source", "unknown"))))
                    for root in roots
                    if assignments[component_by_root[str(root["root_id"])]] == name
                )
            )
            for name in SPLIT_RATIOS
        },
        "traffic_profile_counts": {
            name: dict(Counter(str(root.get("traffic_profile", "unknown")) for root in roots if assignments[component_by_root[str(root["root_id"])]] == name))
            for name in SPLIT_RATIOS
        },
        "deadline_bin_counts": {
            name: dict(Counter(str(root.get("deadline_bin", "unknown")) for root in roots if assignments[component_by_root[str(root["root_id"])]] == name))
            for name in SPLIT_RATIOS
        },
        "activation_bin_counts": {
            name: dict(
                Counter(
                    str(root.get("activation_bin", root.get("deadline_bin", "unknown")))
                    for root in roots
                    if assignments[component_by_root[str(root["root_id"])]] == name
                )
            )
            for name in SPLIT_RATIOS
        },
        "unsupported_marginals": {
            marginal: count
            for marginal, count in sorted(marginal_sizes.items())
            if count < len(SPLIT_RATIOS)
        },
    }
    episode_splits: dict[str, set[str]] = defaultdict(set)
    fingerprint_splits: dict[str, set[str]] = defaultdict(set)
    scenario_episode_splits: dict[str, set[str]] = defaultdict(set)
    for root in roots:
        split = assignments[component_by_root[str(root["root_id"])]]
        episode_splits[_root_group_id(root)].add(split)
        fingerprint = _root_observation_fingerprint(root)
        if fingerprint:
            fingerprint_splits[fingerprint].add(split)
        scenario_key = scenario_keys[str(root["root_id"])]
        if scenario_key:
            scenario_episode_splits[scenario_key].add(split)
    provenance["cross_split_episode_overlap_count"] = sum(len(values) > 1 for values in episode_splits.values())
    provenance["cross_split_observation_fingerprint_overlap_count"] = sum(
        len(values) > 1 for values in fingerprint_splits.values()
    )
    provenance["cross_split_scenario_episode_overlap_count"] = sum(
        len(values) > 1 for values in scenario_episode_splits.values()
    )
    rows.sort(key=lambda row: str(row["root_id"]))
    manifest_dir = dataset / "manifests"
    output = manifest_dir / "split_manifest.jsonl"
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    with (manifest_dir / "split_provenance.json").open("w", encoding="utf-8") as handle:
        json.dump(provenance, handle, indent=2, sort_keys=True)
    return rows


def entry_time_supervision(
    *,
    target_lane_entry_time_s: float | None,
    censor_time_s: float,
    viability_status: str,
) -> tuple[float, float, bool, float, str]:
    """Return conditional-entry regression target and explicit censor metadata."""

    status = str(viability_status)
    censor_value = float(censor_time_s)
    if not np.isfinite(censor_value) or censor_value < 0.0:
        raise ValueError("entry-time censor value must be finite and non-negative")
    if target_lane_entry_time_s is not None:
        if status != "observed_success":
            raise ValueError("target_lane_entry_time_s requires observed_success viability status")
        value = float(target_lane_entry_time_s)
        if not np.isfinite(value) or value < 0.0:
            raise ValueError("target_lane_entry_time_s must be finite and non-negative")
        return value, 1.0, True, value, ""
    if status == "observed_failure":
        reason = "taper_miss"
    elif status == "censored":
        reason = "horizon_elapsed"
    else:
        raise ValueError(
            "observed-success ACCVP branch is missing target_lane_entry_time_s"
        )
    return 0.0, 0.0, False, censor_value, reason


def event_supervision_mask(
    row: dict[str, Any],
    root_row: dict[str, Any],
) -> np.ndarray:
    """Return event-head supervision without turning censoring into a label."""

    observed = float(bool(row["event_observed"]))
    viability_eligible = float(bool(observed) and _is_activation_window(root_row))
    # Collision and safety are observed over every completed rollout. Taper
    # miss and merge viability require a terminal success/failure observation;
    # a horizon-elapsed branch is censored, not a negative taper-miss example.
    return np.asarray([1.0, 1.0, observed, viability_eligible], dtype=np.float32)


def _validate_actor_row_mapping(
    root_metadata: dict[str, Any],
    branch_row: dict[str, Any],
    root_tensors: dict[str, np.ndarray],
    branch_tensors: dict[str, np.ndarray],
) -> None:
    schema_version = int(root_metadata.get("counterfactual_schema_version", -1))
    if schema_version < 3:
        return
    actor_row_ids = [str(value) for value in root_metadata.get("actor_row_ids", [])]
    selected_indices = [int(value) for value in root_metadata.get("actor_row_source_indices", [])]
    tensor_indices = [int(value) for value in np.asarray(root_tensors.get("selected_indices", [])).reshape(-1)]
    if tensor_indices != selected_indices:
        raise ValueError(f"ACCVP root selected_indices mismatch: {root_metadata.get('root_id')}")
    actor_mask = np.asarray(root_tensors["mask"], dtype=np.float32).reshape(-1)
    computed = actor_row_mapping_hash(actor_row_ids, selected_indices, actor_mask)
    expected = str(root_metadata.get("actor_row_mapping_hash", ""))
    if not expected or computed != expected:
        raise ValueError(f"ACCVP root actor-row mapping hash mismatch: {root_metadata.get('root_id')}")
    if str(branch_row.get("actor_row_mapping_hash", "")) != expected:
        raise ValueError(f"ACCVP branch actor-row mapping hash mismatch: {branch_row.get('branch_id')}")
    branch_ids = [str(value) for value in branch_row.get("actor_row_ids", [])]
    if branch_ids != actor_row_ids:
        raise ValueError(f"ACCVP branch manifest actor-row IDs mismatch: {branch_row.get('branch_id')}")
    tensor_ids = [str(value) for value in np.asarray(branch_tensors.get("actor_row_ids", [])).tolist()]
    if tensor_ids != actor_row_ids:
        raise ValueError(f"ACCVP branch actor-row IDs mismatch: {branch_row.get('branch_id')}")
    response = np.asarray(branch_tensors.get("actor_response", []))
    response_mask = np.asarray(branch_tensors.get("actor_valid_mask", []))
    if response.ndim != 3 or response.shape[-1] != 5:
        raise ValueError(f"ACCVP branch actor_response must have shape [actor,time,5]: {branch_row.get('branch_id')}")
    if response_mask.shape != response.shape[:2]:
        raise ValueError(f"ACCVP branch actor_valid_mask shape mismatch: {branch_row.get('branch_id')}")
    if response.shape[0] != len(actor_row_ids):
        raise ValueError(f"ACCVP branch actor response row count mismatch: {branch_row.get('branch_id')}")


def _validate_root_fingerprint(
    root_metadata: dict[str, Any],
    root_row: dict[str, Any],
    root_tensors: dict[str, np.ndarray],
) -> None:
    if int(root_metadata.get("counterfactual_schema_version", -1)) < COUNTERFACTUAL_SCHEMA_VERSION:
        return
    if str(root_metadata.get("root_observation_fingerprint_version", "")) != ROOT_OBSERVATION_FINGERPRINT_VERSION:
        raise ValueError(f"ACCVP root observation fingerprint version mismatch: {root_metadata.get('root_id')}")
    expected = root_observation_fingerprint(
        actor_row_ids=[str(value) for value in root_metadata.get("actor_row_ids", [])],
        root_ego=dict(root_metadata.get("root_ego", {})),
        data_contract_hash=str(root_metadata.get("data_contract_hash", "")),
        tensors=root_tensors,
    )
    recorded = str(root_metadata.get("root_observation_fingerprint", ""))
    if expected != recorded or str(root_row.get("root_observation_fingerprint", "")) != recorded:
        raise ValueError(f"ACCVP root observation fingerprint mismatch: {root_metadata.get('root_id')}")


def _validate_entry_time_row(row: dict[str, Any]) -> tuple[float, float, bool, float, str]:
    result = entry_time_supervision(
        target_lane_entry_time_s=row.get("target_lane_entry_time_s"),
        censor_time_s=float(row.get("entry_time_censor_time_s", row["censor_time"])),
        viability_status=str(row["viability_observation_status"]),
    )
    if int(row.get("counterfactual_schema_version", -1)) >= COUNTERFACTUAL_SCHEMA_VERSION:
        _target, _mask, observed, censor_time, reason = result
        if str(row.get("entry_time_label_version", "")) != ENTRY_TIME_LABEL_VERSION:
            raise ValueError(f"ACCVP branch entry-time label version mismatch: {row.get('branch_id')}")
        if bool(row.get("entry_time_observed")) != observed:
            raise ValueError(f"ACCVP branch entry-time observed flag mismatch: {row.get('branch_id')}")
        if float(row.get("entry_time_censor_time_s", -1.0)) != censor_time:
            raise ValueError(f"ACCVP branch entry-time censor value mismatch: {row.get('branch_id')}")
        if str(row.get("entry_time_censor_reason", "")) != reason:
            raise ValueError(f"ACCVP branch entry-time censor reason mismatch: {row.get('branch_id')}")
    return result


class ACCVPBranchDataset:
    """Numpy dataset for one split; torch conversion remains in the trainer."""

    def __init__(self, dataset_dir: str | Path, split: str):
        self.dataset_dir = Path(dataset_dir)
        manifest_dir = self.dataset_dir / "manifests"
        split_rows = _jsonl(manifest_dir / "split_manifest.jsonl")
        if not split_rows:
            raise FileNotFoundError("missing ACCVP split_manifest.jsonl; call build_split_manifest first")
        if len(split_rows) != len({str(row["root_id"]) for row in split_rows}):
            raise ValueError("ACCVP split manifest contains duplicate root_id values")
        self.split_records = {str(row["root_id"]): row for row in split_rows}
        splits = {root_id: str(row["split"]) for root_id, row in self.split_records.items()}
        complete_roots = [
            row
            for row in _jsonl(manifest_dir / "roots.jsonl")
            if bool(row.get("complete", False))
        ]
        dataset_manifest = read_json(manifest_dir / "dataset_manifest.json")
        strict_selector_contract = is_strict_selector_data_contract(
            dict(dataset_manifest.get("data_contract", {}) or {}).get(
                "protocol_version", ""
            )
        )
        selected_incomplete_roots = [
            str(row.get("root_id", ""))
            for row in complete_roots
            if splits.get(str(row.get("root_id", ""))) == split
            and (
                not bool(row.get("task_actor_coverage_complete", False))
                or bool(row.get("critical_actor_overflow", False))
                or bool(list(row.get("dropped_critical_actor_ids", []) or []))
            )
        ]
        if strict_selector_contract and selected_incomplete_roots:
            raise ValueError(
                "selector3 ACCVP split contains roots with incomplete safety "
                f"actor coverage: {selected_incomplete_roots[:10]}"
            )
        selected_oracle_roots = [
            str(row.get("root_id", ""))
            for row in complete_roots
            if splits.get(str(row.get("root_id", ""))) == split
            and (
                bool(row.get("oracle_only", False))
                or bool(row.get("exclude_from_model_splits", False))
            )
        ]
        if selected_oracle_roots:
            raise ValueError(
                "ACCVP split manifest illegally includes oracle-only roots: "
                f"{selected_oracle_roots[:10]}"
            )
        self.roots = {
            row["root_id"]: row
            for row in complete_roots
            if bool(row.get("complete", False)) and splits.get(row["root_id"]) == split
        }
        self.split_component_by_root = {
            str(root_id): str(self.split_records[str(root_id)].get("split_component_id", ""))
            for root_id in self.roots
        }
        self.observation_fingerprint_by_root = {
            str(root_id): str(
                self.split_records[str(root_id)].get("root_observation_fingerprint", "")
                or root.get("root_observation_fingerprint", root.get("root_state_fingerprint", ""))
            )
            for root_id, root in self.roots.items()
        }
        self.component_metadata_complete = all(self.split_component_by_root.values())
        self.rows = [
            row
            for row in _jsonl(manifest_dir / "branches.jsonl")
            if row.get("branch_status") == "completed" and row.get("root_id") in self.roots
        ]
        # Repeated model-visible states collected under multiple root policies
        # are correlated replicates, not additional independent state mass.
        # Retain every stochastic outcome while making each
        # (observation-fingerprint, action) group contribute total weight one.
        duplicate_groups: dict[tuple[str, int], list[int]] = defaultdict(list)
        for index, row in enumerate(self.rows):
            root_id = str(row["root_id"])
            fingerprint = self.observation_fingerprint_by_root.get(root_id, "") or f"root:{root_id}"
            duplicate_groups[(fingerprint, int(row["action_id"]))].append(index)
        self.sample_weight_by_index: dict[int, float] = {}
        for indices in duplicate_groups.values():
            weight = 1.0 / float(len(indices))
            for index in indices:
                self.sample_weight_by_index[index] = weight
        self.duplicate_weighting_version = "fingerprint_action_total_weight_one_v1"
        self.duplicate_group_count = len(duplicate_groups)
        self.duplicate_row_count = sum(max(0, len(indices) - 1) for indices in duplicate_groups.values())
        self.branch_indices_by_fingerprint_action = {
            key: tuple(int(index) for index in indices)
            for key, indices in duplicate_groups.items()
        }
        self.fingerprint_action_group_by_index = {
            int(index): key
            for key, indices in duplicate_groups.items()
            for index in indices
        }
        self.branch_indices_by_component: dict[str, list[int]] = defaultdict(list)
        for index, row in enumerate(self.rows):
            component_id = self.split_component_by_root.get(str(row["root_id"]), "")
            if component_id:
                self.branch_indices_by_component[component_id].append(index)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        row = self.rows[index]
        root_row = self.roots[row["root_id"]]
        with Path(root_row["metadata_path"]).open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        with np.load(root_row["tensor_path"], allow_pickle=False) as values:
            root = {key: np.asarray(values[key])[0] for key in values.files}
        with np.load(row["tensor_path"], allow_pickle=False) as values:
            branch = {key: np.asarray(values[key]) for key in values.files}
        _validate_actor_row_mapping(metadata, row, root, branch)
        _validate_root_fingerprint(metadata, root_row, root)
        ego = VehicleState(**metadata["root_ego"])
        plan = build_commitment_plan(
            ego,
            decode_action(int(row["action_id"])),
            step_length=float(metadata.get("step_length", 0.1)),
            horizon_steps=int(metadata.get("candidate_plan_horizon_steps", 80)),
        ).states
        event_mask = event_supervision_mask(row, root_row)
        viability_eligible = float(event_mask[3])
        entry_target, entry_mask, entry_observed, entry_censor_time, _entry_censor_reason = _validate_entry_time_row(row)
        return {
            "history_features": root["history_features"].astype(np.float32),
            "history_valid_mask": root["history_valid_mask"].astype(np.float32),
            "history_lane_ids": root["history_lane_ids"].astype(np.int64),
            "history_edge_role_ids": root["history_edge_role_ids"].astype(np.int64),
            "role_ids": root["role_ids"].astype(np.int64),
            "lane_ids": root["lane_ids"].astype(np.int64),
            "edge_role_ids": root["edge_role_ids"].astype(np.int64),
            "actor_mask": root["mask"].astype(np.float32),
            "candidate_plan": plan.astype(np.float32),
            "candidate_action_ids": np.asarray(int(row["action_id"]), dtype=np.int64),
            "sample_weight": np.asarray(self.sample_weight_by_index[index], dtype=np.float32),
            "actor_response": branch["actor_response"].astype(np.float32),
            "actor_response_mask": branch["actor_valid_mask"].astype(np.float32),
            "event_targets": np.asarray(
                [
                    float(row["proxy_collision_within_horizon"]),
                    float(row["safety_violation_within_horizon"]),
                    float(row["taper_miss_observed"]),
                    float(row["merge_before_taper_observed"]),
                ],
                dtype=np.float32,
            ),
            "event_mask": event_mask,
            "geometry_targets": np.asarray(
                [
                    float(row["min_obb_distance"]),
                    float(row["max_drac"]),
                    float(row["target_front_gap"]),
                    float(row["target_rear_gap"]),
                    entry_target,
                ],
                dtype=np.float32,
            ),
            "geometry_mask": np.asarray([1.0, 1.0, 1.0, 1.0, entry_mask], dtype=np.float32),
            "entry_time_observed": np.asarray(float(entry_observed), dtype=np.float32),
            "entry_time_censor_time_s": np.asarray(entry_censor_time, dtype=np.float32),
            "viability_eligible": np.asarray(viability_eligible, dtype=np.float32),
        }


def collate_numpy(items: Iterable[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    rows = list(items)
    if not rows:
        raise ValueError("cannot collate an empty ACCVP batch")
    return {key: np.stack([row[key] for row in rows], axis=0) for key in rows[0]}
