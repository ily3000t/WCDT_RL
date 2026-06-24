from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from safe_rl.accvp.candidate_plan import build_commitment_plan
from safe_rl.sim.action_space import decode_action
from safe_rl.sim.types import VehicleState


SPLIT_RATIOS = {
    "train": 0.60,
    "validation": 0.15,
    "calibration": 0.10,
    "operating_point": 0.05,
    "test": 0.10,
}


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _root_group_id(row: dict[str, Any]) -> str:
    return str(row.get("root_episode_id") or f"{row.get('root_policy', row.get('root_source', 'unknown'))}:{row['episode_seed']}")


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
) -> list[dict[str, Any]]:
    """Grouped, quota-constrained split with marginal rather than combinatorial strata."""

    dataset = Path(dataset_dir)
    roots = [row for row in _jsonl(dataset / "manifests" / "roots.jsonl") if bool(row.get("complete", False))]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in roots:
        grouped[_root_group_id(row)].append(row)
    quotas = _split_quotas(len(grouped), require_all_splits)
    group_items = []
    for group_id, members in grouped.items():
        # Balance the three meaningful marginals without creating the sparse
        # combinatorial signature that previously produced singleton strata.
        policy = str(members[0].get("root_policy", members[0].get("root_source", "unknown")))
        traffic = str(members[0].get("traffic_profile", "unknown"))
        deadline = str(members[0].get("deadline_bin", "unknown"))
        marginals = (f"policy:{policy}", f"traffic:{traffic}", f"deadline:{deadline}")
        digest = hashlib.sha256(f"{seed}:{group_id}".encode("utf-8")).hexdigest()
        group_items.append((marginals, digest, group_id, members))
    marginal_sizes = Counter(marginal for item in group_items for marginal in item[0])
    group_items.sort(key=lambda item: (min(marginal_sizes[value] for value in item[0]), item[1]))
    assigned: Counter[str] = Counter()
    marginal_counts: dict[str, Counter[str]] = {name: Counter() for name in SPLIT_RATIOS}
    assignments: dict[str, str] = {}
    for marginals, _digest, group_id, _members in group_items:
        available = [name for name in SPLIT_RATIOS if assigned[name] < quotas[name]]
        split = min(
            available,
            key=lambda name: (
                sum(marginal_counts[name][value] for value in marginals),
                assigned[name] / max(1, quotas[name]),
                hashlib.sha256(f"{seed}:{group_id}:{name}".encode("utf-8")).hexdigest(),
            ),
        )
        assignments[group_id] = split
        assigned[split] += 1
        for marginal in marginals:
            marginal_counts[split][marginal] += 1
    if require_all_splits and any(assigned[name] == 0 for name in SPLIT_RATIOS):
        raise RuntimeError(f"ACCVP split assignment left an empty split: {dict(assigned)}")
    rows: list[dict[str, Any]] = []
    for group_id, members in grouped.items():
        for root in members:
            rows.append(
                {
                    "root_id": root["root_id"],
                    "root_episode_id": group_id,
                    "episode_seed": int(root["episode_seed"]),
                    "primary_stratum": f"{root.get('root_policy', root.get('root_source', 'unknown'))}:{root.get('deadline_bin', 'unknown')}",
                    "split": assignments[group_id],
                }
            )
    provenance = {
        "split_ratios": SPLIT_RATIOS,
        "group_quotas": quotas,
        "group_counts": dict(assigned),
        "source_counts": {
            name: dict(Counter(str(root.get("root_policy", root.get("root_source", "unknown"))) for root in roots if assignments[_root_group_id(root)] == name))
            for name in SPLIT_RATIOS
        },
        "traffic_profile_counts": {
            name: dict(Counter(str(root.get("traffic_profile", "unknown")) for root in roots if assignments[_root_group_id(root)] == name))
            for name in SPLIT_RATIOS
        },
        "deadline_bin_counts": {
            name: dict(Counter(str(root.get("deadline_bin", "unknown")) for root in roots if assignments[_root_group_id(root)] == name))
            for name in SPLIT_RATIOS
        },
        "unsupported_marginals": {
            marginal: count
            for marginal, count in sorted(marginal_sizes.items())
            if count < len(SPLIT_RATIOS)
        },
    }
    manifest_dir = dataset / "manifests"
    output = manifest_dir / "split_manifest.jsonl"
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    with (manifest_dir / "split_provenance.json").open("w", encoding="utf-8") as handle:
        json.dump(provenance, handle, indent=2, sort_keys=True)
    return rows


class ACCVPBranchDataset:
    """Numpy dataset for one split; torch conversion remains in the trainer."""

    def __init__(self, dataset_dir: str | Path, split: str):
        self.dataset_dir = Path(dataset_dir)
        manifest_dir = self.dataset_dir / "manifests"
        splits = {row["root_id"]: row["split"] for row in _jsonl(manifest_dir / "split_manifest.jsonl")}
        if not splits:
            raise FileNotFoundError("missing ACCVP split_manifest.jsonl; call build_split_manifest first")
        self.roots = {
            row["root_id"]: row
            for row in _jsonl(manifest_dir / "roots.jsonl")
            if bool(row.get("complete", False)) and splits.get(row["root_id"]) == split
        }
        self.rows = [
            row
            for row in _jsonl(manifest_dir / "branches.jsonl")
            if row.get("branch_status") == "completed" and row.get("root_id") in self.roots
        ]

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
        ego = VehicleState(**metadata["root_ego"])
        plan = build_commitment_plan(
            ego,
            decode_action(int(row["action_id"])),
            step_length=float(metadata.get("step_length", 0.1)),
            horizon_steps=int(metadata.get("candidate_plan_horizon_steps", 80)),
        ).states
        observed = float(bool(row["event_observed"]))
        viability_eligible = float(
            bool(row["event_observed"])
            and str(root_row.get("deadline_bin", row.get("deadline_bin", ""))) == "deadline"
        )
        target_entry = row.get("target_lane_entry_time_s")
        entry_target = float(target_entry) if target_entry is not None else float(row["censor_time"])
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
            "event_mask": np.asarray([1.0, 1.0, 1.0, viability_eligible], dtype=np.float32),
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
            "geometry_mask": np.asarray([1.0, 1.0, 1.0, 1.0, viability_eligible], dtype=np.float32),
            "viability_eligible": np.asarray(viability_eligible, dtype=np.float32),
        }


def collate_numpy(items: Iterable[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    rows = list(items)
    if not rows:
        raise ValueError("cannot collate an empty ACCVP batch")
    return {key: np.stack([row[key] for row in rows], axis=0) for key in rows[0]}
