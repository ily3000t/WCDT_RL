from __future__ import annotations

import hashlib
import json
from collections import defaultdict
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


def build_split_manifest(dataset_dir: str | Path, *, seed: int = 0) -> list[dict[str, Any]]:
    """Split by episode seed, never by action branch or individual root row."""

    dataset = Path(dataset_dir)
    roots = [row for row in _jsonl(dataset / "manifests" / "roots.jsonl") if bool(row.get("complete", False))]
    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in roots:
        groups[int(row["episode_seed"])].append(row)
    strata: dict[str, list[int]] = defaultdict(list)
    for episode_seed, grouped_roots in groups.items():
        signature = "|".join(
            sorted(
                f"{row.get('root_source', '')}:{row.get('traffic_profile', '')}:{row.get('deadline_bin', '')}"
                for row in grouped_roots
            )
        )
        strata[signature].append(episode_seed)
    rows: list[dict[str, Any]] = []
    for signature, group_seeds in sorted(strata.items()):
        rng_seed = int.from_bytes(hashlib.sha256(f"{seed}:{signature}".encode("utf-8")).digest()[:8], "little")
        rng = np.random.default_rng(rng_seed)
        group_seeds = list(sorted(group_seeds))
        rng.shuffle(group_seeds)
        boundaries: dict[str, int] = {}
        start = 0
        for name, ratio in list(SPLIT_RATIOS.items())[:-1]:
            start += int(round(len(group_seeds) * ratio))
            boundaries[name] = min(start, len(group_seeds))
        for index, episode_seed in enumerate(group_seeds):
            if index < boundaries["train"]:
                split = "train"
            elif index < boundaries["validation"]:
                split = "validation"
            elif index < boundaries["calibration"]:
                split = "calibration"
            elif index < boundaries["operating_point"]:
                split = "operating_point"
            else:
                split = "test"
            for root in groups[episode_seed]:
                rows.append(
                    {
                        "root_id": root["root_id"],
                        "episode_seed": episode_seed,
                        "stratum": signature,
                        "split": split,
                    }
                )
    seen: dict[str, str] = {}
    for row in rows:
        previous = seen.setdefault(str(row["root_id"]), str(row["split"]))
        if previous != row["split"]:
            raise RuntimeError("ACCVP split leakage: root assigned to multiple splits")
    output = dataset / "manifests" / "split_manifest.jsonl"
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
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
            "event_mask": np.asarray([1.0, 1.0, 1.0, observed], dtype=np.float32),
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
            "geometry_mask": np.asarray([1.0, 1.0, 1.0, 1.0, observed], dtype=np.float32),
        }


def collate_numpy(items: Iterable[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    rows = list(items)
    if not rows:
        raise ValueError("cannot collate an empty ACCVP batch")
    return {key: np.stack([row[key] for row in rows], axis=0) for key in rows[0]}
