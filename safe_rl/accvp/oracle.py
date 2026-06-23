from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


def _jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def counterfactual_oracle_report(dataset_dir: str | Path, required_seeds: Iterable[int] = (2, 5)) -> dict[str, Any]:
    """Go/No-Go report before training: does SUMO offer a safe viable action?"""

    dataset = Path(dataset_dir)
    roots = {
        str(row["root_id"]): row
        for row in _jsonl(dataset / "manifests" / "roots.jsonl")
        if bool(row.get("complete", False))
    }
    branches = [
        row
        for row in _jsonl(dataset / "manifests" / "branches.jsonl")
        if row.get("branch_status") == "completed" and str(row.get("root_id")) in roots
    ]
    by_root: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for branch in branches:
        by_root[str(branch["root_id"])].append(branch)
    root_rows: list[dict[str, Any]] = []
    for root_id, candidates in by_root.items():
        root = roots[root_id]
        viable = [
            candidate
            for candidate in candidates
            if not bool(candidate.get("proxy_collision_within_horizon", False))
            and not bool(candidate.get("safety_violation_within_horizon", False))
            and bool(candidate.get("merge_before_taper_observed", False))
        ]
        root_rows.append(
            {
                "root_id": root_id,
                "episode_seed": int(root["episode_seed"]),
                "root_source": str(root.get("root_source", "")),
                "deadline_bin": str(root.get("deadline_bin", "")),
                "safe_viable_action_exists": bool(viable),
                "safe_viable_action_ids": [int(row["action_id"]) for row in viable],
                "candidate_count": len(candidates),
            }
        )
    required = [int(seed) for seed in required_seeds]
    failure_seed_rows = {
        seed: [row for row in root_rows if row["episode_seed"] == seed and row["deadline_bin"] == "deadline"]
        for seed in required
    }
    required_results = {
        str(seed): {
            "deadline_roots": len(rows),
            "safe_viable_root_count": sum(bool(row["safe_viable_action_exists"]) for row in rows),
            "go": any(bool(row["safe_viable_action_exists"]) for row in rows),
        }
        for seed, rows in failure_seed_rows.items()
    }
    return {
        "dataset_dir": str(dataset.resolve()),
        "root_count": len(root_rows),
        "safe_viable_root_count": sum(bool(row["safe_viable_action_exists"]) for row in root_rows),
        "required_failure_seed_results": required_results,
        "go_for_training": bool(required_results) and all(row["go"] for row in required_results.values()),
        "roots": root_rows,
    }


def write_oracle_report(dataset_dir: str | Path, output_path: str | Path, required_seeds: Iterable[int] = (2, 5)) -> dict[str, Any]:
    report = counterfactual_oracle_report(dataset_dir, required_seeds)
    with Path(output_path).open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    return report
