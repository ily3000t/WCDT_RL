from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from safe_rl.accvp.schema import file_sha256, read_json, write_json_atomic


ORACLE_STATES = frozenset({"insufficient_coverage", "no_safe_viable_alternative", "go"})


def _jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _dataset_provenance(dataset: Path) -> dict[str, Any]:
    """Capture immutable dataset inputs so a report cannot be reused elsewhere."""

    manifests = dataset / "manifests"
    dataset_manifest_path = manifests / "dataset_manifest.json"
    roots_path = manifests / "roots.jsonl"
    branches_path = manifests / "branches.jsonl"
    if not dataset_manifest_path.exists():
        return {"formal_dataset": False}
    manifest = read_json(dataset_manifest_path)
    return {
        "formal_dataset": str(manifest.get("artifact_kind", "")) == "counterfactual_dataset_v1",
        "dataset_manifest_sha256": file_sha256(dataset_manifest_path),
        "roots_manifest_sha256": file_sha256(roots_path),
        "branches_manifest_sha256": file_sha256(branches_path),
        "dataset_fingerprint": str(manifest.get("dataset_fingerprint", "")),
        "config_hash": str(manifest.get("config_hash", "")),
        "risk_model_fingerprint": str(manifest.get("risk_model_fingerprint", "")),
    }


def _safe_viable(candidate: dict[str, Any]) -> bool:
    return (
        not bool(candidate.get("proxy_collision_within_horizon", False))
        and not bool(candidate.get("safety_violation_within_horizon", False))
        and str(candidate.get("viability_observation_status", "")) == "observed_success"
        and bool(candidate.get("merge_before_taper_observed", False))
    )


def _raw_infeasible(root: dict[str, Any], candidates: list[dict[str, Any]]) -> tuple[bool | None, str]:
    if not bool(root.get("raw_action_legal", False)):
        return True, "raw_illegal"
    raw_action = root.get("raw_action_id")
    if raw_action is None:
        return None, "raw_action_missing"
    raw = next((row for row in candidates if int(row.get("action_id", -1)) == int(raw_action)), None)
    if raw is None:
        return None, "raw_branch_missing"
    if bool(raw.get("proxy_collision_within_horizon", False)) or bool(raw.get("safety_violation_within_horizon", False)):
        return True, "raw_safety_failure"
    status = str(raw.get("viability_observation_status", ""))
    if status == "observed_failure" or bool(raw.get("taper_miss_observed", False)):
        return True, "raw_taper_failure"
    if status == "observed_success" and bool(raw.get("merge_before_taper_observed", False)):
        return False, "raw_already_viable"
    return None, "raw_outcome_censored"


def counterfactual_oracle_report(
    dataset_dir: str | Path,
    required_seeds: Iterable[int] = (2, 5),
    *,
    min_deadline_roots_per_seed: int = 1,
    root_policy: str | None = None,
) -> dict[str, Any]:
    """Pre-training ACCVP repairability oracle with explicit coverage semantics.

    ``go`` requires the actual frozen raw action to be infeasible and a
    different legal candidate to be safety-safe and observed to merge before
    taper. ``false`` is never overloaded: callers receive one of the three
    named states in :data:`ORACLE_STATES`.
    """

    seed_list = [int(value) for value in required_seeds]
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
    for root_id, root in roots.items():
        candidates = by_root.get(root_id, [])
        raw_infeasible, raw_reason = _raw_infeasible(root, candidates)
        raw_action = root.get("raw_action_id")
        alternatives = [
            candidate
            for candidate in candidates
            if raw_action is None or int(candidate.get("action_id", -1)) != int(raw_action)
            if _safe_viable(candidate)
        ]
        repairable = bool(raw_infeasible is True and alternatives)
        root_rows.append(
            {
                "root_id": root_id,
                "episode_seed": int(root["episode_seed"]),
                "root_policy": str(root.get("root_policy", root.get("root_source", ""))),
                "root_filter": str(root.get("root_filter", "all")),
                "deadline_bin": str(root.get("deadline_bin", "")),
                "raw_action_id": raw_action,
                "raw_action_legal": bool(root.get("raw_action_legal", False)),
                "raw_infeasible": raw_infeasible,
                "raw_infeasible_reason": raw_reason,
                "safe_viable_alternative_action_ids": [int(row["action_id"]) for row in alternatives],
                "repairable": repairable,
                "candidate_count": len(candidates),
            }
        )
    per_seed: dict[str, dict[str, Any]] = {}
    for seed in seed_list:
        deadline_roots = [
            row
            for row in root_rows
            if row["episode_seed"] == seed
            and row["deadline_bin"] == "deadline"
            and (root_policy is None or row["root_policy"] == root_policy)
        ]
        evaluated = [row for row in deadline_roots if row["raw_infeasible"] is not None]
        if len(deadline_roots) < int(min_deadline_roots_per_seed) or len(evaluated) < int(min_deadline_roots_per_seed):
            state = "insufficient_coverage"
        elif any(bool(row["repairable"]) for row in evaluated):
            state = "go"
        else:
            state = "no_safe_viable_alternative"
        per_seed[str(seed)] = {
            "state": state,
            "deadline_roots": len(deadline_roots),
            "raw_outcome_evaluated_roots": len(evaluated),
            "repairable_roots": sum(bool(row["repairable"]) for row in evaluated),
            "roots": deadline_roots,
        }
    states = [row["state"] for row in per_seed.values()]
    if any(state == "insufficient_coverage" for state in states):
        state = "insufficient_coverage"
    elif states and all(item == "go" for item in states):
        state = "go"
    else:
        state = "no_safe_viable_alternative"
    return {
        "dataset_dir": str(dataset.resolve()),
        "oracle_state": state,
        "go_for_training": state == "go",
        "required_seeds": seed_list,
        "required_min_deadline_roots_per_seed": int(min_deadline_roots_per_seed),
        "root_policy": root_policy,
        "dataset_provenance": _dataset_provenance(dataset),
        "root_count": len(root_rows),
        "required_failure_seed_results": per_seed,
        "roots": root_rows,
    }


def write_oracle_report(
    dataset_dir: str | Path,
    output_path: str | Path,
    required_seeds: Iterable[int] = (2, 5),
    *,
    min_deadline_roots_per_seed: int = 1,
    root_policy: str | None = None,
) -> dict[str, Any]:
    report = counterfactual_oracle_report(
        dataset_dir,
        required_seeds,
        min_deadline_roots_per_seed=min_deadline_roots_per_seed,
        root_policy=root_policy,
    )
    write_json_atomic(output_path, report)
    return report


def validate_oracle_for_training(config: Any, dataset_dir: str | Path) -> dict[str, Any]:
    """Enforce the non-bypassable ACCVP-v1 repairability gate."""

    report_path = config.accvp.get("oracle_report")
    if not report_path:
        raise FileNotFoundError("formal ACCVP training requires accvp.oracle_report with oracle_state='go'")
    report = read_json(report_path)
    if str(report.get("oracle_state", "")) != "go" or not bool(report.get("go_for_training", False)):
        raise ValueError(f"ACCVP training blocked by oracle_state={report.get('oracle_state')!r}")
    if str(report.get("root_policy", "")) != "merge_timing":
        raise ValueError("ACCVP training requires a merge_timing-PPO oracle report")
    if [int(value) for value in report.get("required_seeds", [])] != [2, 5]:
        raise ValueError("ACCVP training requires the pre-registered oracle seeds [2, 5]")
    dataset = Path(dataset_dir).resolve()
    if Path(str(report.get("dataset_dir", ""))).resolve() != dataset:
        raise ValueError("ACCVP oracle report belongs to a different dataset directory")
    current = _dataset_provenance(dataset)
    expected = dict(report.get("dataset_provenance", {}))
    if not bool(current.get("formal_dataset", False)):
        raise ValueError("ACCVP training requires a merged formal counterfactual dataset")
    for key in ("dataset_manifest_sha256", "roots_manifest_sha256", "branches_manifest_sha256", "dataset_fingerprint"):
        if not expected.get(key) or expected.get(key) != current.get(key):
            raise ValueError(f"ACCVP oracle report provenance mismatch for {key}")
    risk_fingerprint = str(current.get("risk_model_fingerprint", ""))
    if risk_fingerprint.startswith("heuristic:") or not risk_fingerprint:
        raise ValueError("ACCVP formal dataset is not bound to a frozen Risk Module checkpoint")
    configured_risk = config.accvp.get("risk_checkpoint")
    if configured_risk:
        expected_fingerprint = f"risk_checkpoint:{file_sha256(configured_risk)}"
        if expected_fingerprint != risk_fingerprint:
            raise ValueError("ACCVP Risk Module checkpoint does not match the counterfactual dataset")
    return report
