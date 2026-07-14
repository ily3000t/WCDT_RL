"""Seed-2/5 repairability oracle regression and training preflight."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from safe_rl.accvp.contracts.schema import file_sha256, read_json, write_json_atomic
from safe_rl.accvp.contracts.protocol import effective_activation_distance
from safe_rl.evaluation_protocol import seeds_for_role


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
        "formal_dataset": str(manifest.get("artifact_kind", "")) == "counterfactual_dataset_v2",
        "counterfactual_schema_version": int(manifest.get("counterfactual_schema_version", -1)),
        "collection_phase": str(manifest.get("collection_phase", "")),
        "dataset_manifest_sha256": file_sha256(dataset_manifest_path),
        "roots_manifest_sha256": file_sha256(roots_path),
        "branches_manifest_sha256": file_sha256(branches_path),
        "dataset_fingerprint": str(manifest.get("dataset_fingerprint", "")),
        "config_hash": str(manifest.get("config_hash", "")),
        "data_contract_hash": str(manifest.get("data_contract_hash", "")),
        "data_contract_protocol_version": str(
            dict(manifest.get("data_contract", {})).get("protocol_version", "")
        ),
        "scenario_route_hash": str(
            manifest.get("scenario_route_hash")
            or dict(manifest.get("data_contract", {})).get("scenario_route_hash", "")
        ),
        "accvp_activation_distance_m": manifest.get("accvp_activation_distance_m"),
        "risk_model_fingerprint": str(manifest.get("risk_model_fingerprint", "")),
    }


def _safe_viable(candidate: dict[str, Any]) -> bool:
    """Match the ACV-Shield feasible set, not only physical branch outcomes."""

    return (
        bool(candidate.get("secondary_safety_pass", False))
        and not bool(candidate.get("proxy_collision_within_horizon", False))
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
    cohort_role: str = "oracle_regression",
    oracle_only: bool = True,
    exclude_from_model_splits: bool = True,
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
                "activation_bin": str(root.get("activation_bin", root.get("deadline_bin", ""))),
                "raw_action_id": raw_action,
                "raw_action_legal": bool(root.get("raw_action_legal", False)),
                "raw_infeasible": raw_infeasible,
                "raw_infeasible_reason": raw_reason,
                "safe_viable_alternative_action_ids": [int(row["action_id"]) for row in alternatives],
                "repairable": repairable,
                "candidate_count": len(candidates),
                "oracle_only": bool(root.get("oracle_only", False)),
                "cohort_role": str(root.get("cohort_role", "")),
                "exclude_from_model_splits": bool(root.get("exclude_from_model_splits", False)),
            }
        )
    per_seed: dict[str, dict[str, Any]] = {}
    for seed in seed_list:
        deadline_roots = [
            row
            for row in root_rows
            if row["episode_seed"] == seed
            and row["activation_bin"] in {"activation_window", "deadline"}
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
        "cohort_role": str(cohort_role),
        "oracle_only": bool(oracle_only),
        "exclude_from_model_splits": bool(exclude_from_model_splits),
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
    cohort_role: str = "oracle_regression",
    oracle_only: bool = True,
    exclude_from_model_splits: bool = True,
) -> dict[str, Any]:
    report = counterfactual_oracle_report(
        dataset_dir,
        required_seeds,
        min_deadline_roots_per_seed=min_deadline_roots_per_seed,
        root_policy=root_policy,
        cohort_role=cohort_role,
        oracle_only=oracle_only,
        exclude_from_model_splits=exclude_from_model_splits,
    )
    write_json_atomic(output_path, report)
    return report


def validate_oracle_for_training(config: Any, dataset_dir: str | Path) -> dict[str, Any]:
    """Validate the independent oracle-regression premise for formal training.

    The oracle cohort may live in a separate schema-v3 dataset.  Its report is
    bound to that dataset, while compatibility-critical collection semantics
    are compared with the formal model dataset.  Oracle seeds and explicitly
    oracle-only roots are forbidden from every model split.
    """

    report_path = config.accvp.get("oracle_report")
    if not report_path:
        raise FileNotFoundError("formal ACCVP training requires accvp.oracle_report with oracle_state='go'")
    report = read_json(report_path)
    if str(report.get("oracle_state", "")) != "go" or not bool(report.get("go_for_training", False)):
        raise ValueError(f"ACCVP training blocked by oracle_state={report.get('oracle_state')!r}")
    if str(report.get("root_policy", "")) != "merge_timing":
        raise ValueError("ACCVP training requires a merge_timing-PPO oracle report")

    oracle_cfg = dict(config.accvp.get("oracle", {}) or {})
    configured_seeds = oracle_cfg.get("required_seeds")
    if configured_seeds is None:
        configured_seeds = report.get("required_seeds", [])
    required_seeds = [int(value) for value in configured_seeds]
    if not required_seeds or len(required_seeds) != len(set(required_seeds)):
        raise ValueError("accvp.oracle.required_seeds must be a non-empty unique seed list")
    cohort_role = str(oracle_cfg.get("cohort_role", report.get("cohort_role", "")))
    if not cohort_role:
        raise ValueError("accvp.oracle.cohort_role is required")
    exclude_from_splits = bool(
        oracle_cfg.get(
            "exclude_from_model_splits",
            report.get("exclude_from_model_splits", False),
        )
    )
    if not exclude_from_splits:
        raise ValueError("formal ACCVP training requires oracle exclusion from model splits")
    registered_seeds = seeds_for_role(config, cohort_role, fallback=required_seeds)
    if registered_seeds != required_seeds:
        raise ValueError(
            "accvp.oracle.required_seeds do not match the registered oracle cohort: "
            f"configured={required_seeds} registered={registered_seeds}"
        )
    if [int(value) for value in report.get("required_seeds", [])] != required_seeds:
        raise ValueError("ACCVP oracle report seeds do not match accvp.oracle.required_seeds")
    if str(report.get("cohort_role", "")) != cohort_role:
        raise ValueError("ACCVP oracle report cohort_role does not match configuration")
    if not bool(report.get("oracle_only", False)) or not bool(
        report.get("exclude_from_model_splits", False)
    ):
        raise ValueError("ACCVP oracle report must declare oracle-only split exclusion")

    dataset = Path(dataset_dir).resolve()
    oracle_dataset = Path(str(report.get("dataset_dir", ""))).resolve()
    if not oracle_dataset.is_dir():
        raise ValueError("ACCVP oracle report dataset directory does not exist")
    current = _dataset_provenance(oracle_dataset)
    expected = dict(report.get("dataset_provenance", {}))
    if not bool(current.get("formal_dataset", False)):
        raise ValueError("ACCVP oracle report requires a merged counterfactual dataset")
    for key in (
        "dataset_manifest_sha256",
        "roots_manifest_sha256",
        "branches_manifest_sha256",
        "dataset_fingerprint",
        "data_contract_hash",
    ):
        if not expected.get(key) or expected.get(key) != current.get(key):
            raise ValueError(f"ACCVP oracle report provenance mismatch for {key}")

    oracle_roots = [
        row
        for row in _jsonl(oracle_dataset / "manifests" / "roots.jsonl")
        if bool(row.get("complete", False))
        and int(row.get("episode_seed", -1)) in set(required_seeds)
        and str(row.get("root_policy", row.get("root_source", ""))) == "merge_timing"
    ]
    covered_seeds = {int(row.get("episode_seed", -1)) for row in oracle_roots}
    if covered_seeds != set(required_seeds):
        raise ValueError("ACCVP oracle dataset does not contain every required oracle seed")
    incorrectly_scoped = [
        str(row.get("root_id", ""))
        for row in oracle_roots
        if not bool(row.get("oracle_only", False))
        or str(row.get("cohort_role", "")) != cohort_role
        or not bool(row.get("exclude_from_model_splits", False))
    ]
    if incorrectly_scoped:
        raise ValueError(
            "ACCVP oracle roots lack oracle-only cohort metadata: "
            f"{incorrectly_scoped[:10]}"
        )

    model_provenance = _dataset_provenance(dataset)
    if not bool(model_provenance.get("formal_dataset", False)):
        raise ValueError("ACCVP training requires a merged formal counterfactual dataset")
    if str(model_provenance.get("collection_phase", "")) != "formal":
        raise ValueError("ACCVP training requires a dataset merged from formal collection shards")
    for key in (
        "counterfactual_schema_version",
        "data_contract_hash",
        "data_contract_protocol_version",
        "scenario_route_hash",
        "risk_model_fingerprint",
        "accvp_activation_distance_m",
    ):
        if current.get(key) != model_provenance.get(key):
            raise ValueError(f"ACCVP oracle/model dataset compatibility mismatch for {key}")

    model_roots = [
        row
        for row in _jsonl(dataset / "manifests" / "roots.jsonl")
        if bool(row.get("complete", False))
    ]
    forbidden_model_roots = [
        str(row.get("root_id", ""))
        for row in model_roots
        if int(row.get("episode_seed", -1)) in set(required_seeds)
        or bool(row.get("oracle_only", False))
        or bool(row.get("exclude_from_model_splits", False))
        or str(row.get("cohort_role", "")) == cohort_role
    ]
    if forbidden_model_roots:
        raise ValueError(
            "formal ACCVP model dataset contains oracle-regression roots: "
            f"{forbidden_model_roots[:10]}"
        )

    risk_fingerprint = str(model_provenance.get("risk_model_fingerprint", ""))
    if risk_fingerprint.startswith("heuristic:") or not risk_fingerprint:
        raise ValueError("ACCVP formal dataset is not bound to a frozen Risk Module checkpoint")
    configured_risk = config.accvp.get("risk_checkpoint")
    if configured_risk:
        expected_fingerprint = f"risk_checkpoint:{file_sha256(configured_risk)}"
        if expected_fingerprint != risk_fingerprint:
            raise ValueError("ACCVP Risk Module checkpoint does not match the counterfactual dataset")
    expected_activation = float(model_provenance.get("accvp_activation_distance_m", -1.0))
    if expected_activation <= 0.0 or abs(effective_activation_distance(config) - expected_activation) > 1.0e-9:
        raise ValueError("ACCVP activation window does not match the formal counterfactual dataset")
    validated = dict(report)
    validated["training_exclusion_audit"] = {
        "cohort_role": cohort_role,
        "required_seeds": required_seeds,
        "exclude_from_model_splits": True,
        "oracle_dataset_dir": str(oracle_dataset),
        "model_dataset_dir": str(dataset),
        "model_root_count": len(model_roots),
        "forbidden_model_root_count": 0,
    }
    return validated
