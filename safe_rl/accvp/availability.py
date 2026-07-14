from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from safe_rl.accvp.selection import select_viability_action
from safe_rl.sim.action_space import decode_action


LEFT_ACTION_IDS = (6, 7, 8)
AVAILABILITY_DENOMINATOR_VERSION = "risk_eligible_raw_or_merge_left_v1"

NO_LEGAL_LEFT = "no legal merge-left candidate"
RISK_LEFT_FAILED = "merge-left candidate Risk secondary failed"
PHYSICAL_LEFT_UNSAFE = "merge-left physically unsafe"
LEFT_TAPER_MISS_OR_CENSORED = "merge-left taper miss / censored"
MODEL_PROXY_FAILED = "model pU_proxy gate failed"
MODEL_SAFETY_FAILED = "model pU_safety gate failed"
MODEL_VIABILITY_FAILED = "model pL_viability gate failed"
RAW_ALREADY_FEASIBLE = "raw already feasible"
ORACLE_LEFT_FEASIBLE = "oracle merge-left feasible"
MODEL_LEFT_SELECTED = "model merge-left selected"


class OperatingPointAvailabilityError(RuntimeError):
    """Raised when no operating point satisfies the pre-registered availability gate."""

    def __init__(self, message: str, diagnostics: dict[str, Any]):
        super().__init__(message)
        self.diagnostics = diagnostics


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _is_activation_window(row: dict[str, Any]) -> bool:
    return str(row.get("activation_bin", row.get("deadline_bin", ""))) in {"activation_window", "deadline"}


def _candidate_legal(row: dict[str, Any]) -> bool:
    secondary = dict(row.get("secondary_risk", {}))
    return bool(row.get("candidate_legal", secondary.get("candidate_legal", True)))


def _secondary_safety_pass(row: dict[str, Any]) -> bool:
    secondary = dict(row.get("secondary_risk", {}))
    return bool(row.get("secondary_safety_pass", secondary.get("secondary_safety_pass", False)))


def _physically_safe(row: dict[str, Any]) -> bool:
    return not bool(row.get("proxy_collision_within_horizon", row.get("proxy_collision", False))) and not bool(
        row.get("safety_violation_within_horizon", row.get("safety_violation", False))
    )


def _observed_success(row: dict[str, Any]) -> bool:
    return str(row.get("viability_observation_status", "")) == "observed_success" and bool(
        row.get("merge_before_taper_observed", row.get("merge_before_taper", False))
    )


def _oracle_feasible(row: dict[str, Any]) -> bool:
    return _candidate_legal(row) and _secondary_safety_pass(row) and _physically_safe(row) and _observed_success(row)


def _left_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if int(row.get("action_id", -1)) in LEFT_ACTION_IDS]


def _split_root_ids(dataset_dir: Path, split: str) -> set[str]:
    if split == "all":
        return {
            str(row["root_id"])
            for row in _jsonl(dataset_dir / "manifests" / "roots.jsonl")
            if bool(row.get("complete", False))
        }
    split_path = dataset_dir / "manifests" / "split_manifest.jsonl"
    if not split_path.exists():
        raise FileNotFoundError(f"missing ACCVP split manifest: {split_path}")
    root_ids = {str(row["root_id"]) for row in _jsonl(split_path) if str(row.get("split", "")) == split}
    if not root_ids:
        raise ValueError(f"split {split!r} contains no roots")
    return root_ids


def _decision_roots(dataset_dir: Path, split: str) -> tuple[dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    root_ids = _split_root_ids(dataset_dir, split)
    roots = {
        str(row["root_id"]): row
        for row in _jsonl(dataset_dir / "manifests" / "roots.jsonl")
        if bool(row.get("complete", False)) and str(row["root_id"]) in root_ids and _is_activation_window(row)
    }
    by_root: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in _jsonl(dataset_dir / "manifests" / "branches.jsonl"):
        root_id = str(row.get("root_id", ""))
        if root_id not in roots or row.get("branch_status") != "completed" or not bool(row.get("event_observed", False)):
            continue
        by_root[root_id].append(row)
    roots = {root_id: root for root_id, root in roots.items() if by_root.get(root_id)}
    by_root = {root_id: rows for root_id, rows in by_root.items() if root_id in roots}
    return roots, by_root


def _root_oracle_record(root: dict[str, Any], candidates: list[dict[str, Any]]) -> dict[str, Any]:
    raw_action_id = root.get("raw_action_id")
    raw = next((row for row in candidates if raw_action_id is not None and int(row.get("action_id", -1)) == int(raw_action_id)), None)
    left = _left_rows(candidates)
    legal_left = [row for row in left if _candidate_legal(row)]
    risk_left = [row for row in legal_left if _secondary_safety_pass(row)]
    safe_left = [row for row in risk_left if _physically_safe(row)]
    oracle_left = [row for row in safe_left if _observed_success(row)]
    raw_feasible = bool(raw is not None and _oracle_feasible(raw))
    risk_ceiling_available = bool(raw_feasible or risk_left)
    oracle_available = bool(raw_feasible or oracle_left)
    if raw_feasible:
        reason = RAW_ALREADY_FEASIBLE
    elif oracle_left:
        reason = ORACLE_LEFT_FEASIBLE
    elif not legal_left:
        reason = NO_LEGAL_LEFT
    elif not risk_left:
        reason = RISK_LEFT_FAILED
    elif not safe_left:
        reason = PHYSICAL_LEFT_UNSAFE
    else:
        reason = LEFT_TAPER_MISS_OR_CENSORED
    return {
        "root_id": str(root["root_id"]),
        "episode_seed": int(root.get("episode_seed", -1)),
        "root_policy": str(root.get("root_policy", root.get("root_source", "unknown"))),
        "collection_source": str(root.get("collection_source", root.get("root_policy", root.get("root_source", "unknown")))),
        "traffic_profile": str(root.get("traffic_profile", "unknown")),
        "activation_bin": str(root.get("activation_bin", root.get("deadline_bin", "unknown"))),
        "deadline_bin": str(root.get("deadline_bin", "unknown")),
        "raw_action_id": int(raw_action_id) if raw_action_id is not None else None,
        "raw_oracle_feasible": raw_feasible,
        "risk_secondary_pass_ceiling_available": risk_ceiling_available,
        "oracle_merge_intent_ceiling_available": oracle_available,
        "reason": reason,
        "legal_left_action_ids": [int(row["action_id"]) for row in legal_left],
        "secondary_pass_left_action_ids": [int(row["action_id"]) for row in risk_left],
        "oracle_safe_viable_left_action_ids": [int(row["action_id"]) for row in oracle_left],
    }


def _left_action_stats(candidates_by_root: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, int]]:
    stats: dict[str, Counter[str]] = {str(action_id): Counter() for action_id in LEFT_ACTION_IDS}
    for rows in candidates_by_root.values():
        for row in _left_rows(rows):
            action_id = str(int(row["action_id"]))
            stats[action_id]["count"] += 1
            stats[action_id]["legal_count"] += int(_candidate_legal(row))
            stats[action_id]["secondary_pass_count"] += int(_secondary_safety_pass(row))
            stats[action_id]["physically_safe_count"] += int(_physically_safe(row))
            stats[action_id]["observed_success_count"] += int(_observed_success(row))
            stats[action_id]["observed_failure_count"] += int(str(row.get("viability_observation_status", "")) == "observed_failure")
            stats[action_id]["censored_count"] += int(str(row.get("viability_observation_status", "")) == "censored")
    return {action_id: dict(counter) for action_id, counter in stats.items()}


def diagnose_oracle_availability(dataset_dir: str | Path, split: str = "operating_point") -> dict[str, Any]:
    dataset = Path(dataset_dir)
    roots, by_root = _decision_roots(dataset, split)
    if not roots:
        raise ValueError(f"dataset split {split!r} has no observed activation-window roots")
    root_records = [_root_oracle_record(roots[root_id], by_root[root_id]) for root_id in sorted(roots)]
    decision_count = len(root_records)
    oracle_available = sum(int(row["oracle_merge_intent_ceiling_available"]) for row in root_records)
    risk_available = sum(int(row["risk_secondary_pass_ceiling_available"]) for row in root_records)
    return {
        "diagnostic_kind": "accvp_availability_attribution",
        "dataset_dir": str(dataset.resolve()),
        "split": split,
        "decision_count": decision_count,
        "oracle_merge_intent_ceiling_availability": float(oracle_available / max(1, decision_count)),
        "risk_secondary_pass_ceiling_availability": float(risk_available / max(1, decision_count)),
        "model_gate_best_availability": None,
        "model_gate_source": "not_evaluated_by_dataset_only_diagnostic",
        "reason_counts": dict(Counter(str(row["reason"]) for row in root_records)),
        "left_action_stats": _left_action_stats(by_root),
        "roots": root_records,
    }


def audit_risk_secondary_false_negatives(dataset_dir: str | Path, split: str = "operating_point") -> dict[str, Any]:
    """Audit whether frozen Risk secondary pass rejects SUMO-safe merge-left actions."""

    dataset = Path(dataset_dir)
    roots, by_root = _decision_roots(dataset, split)
    if not roots:
        raise ValueError(f"dataset split {split!r} has no observed activation-window roots")

    root_records: list[dict[str, Any]] = []
    action_stats: dict[str, Counter[str]] = {str(action_id): Counter() for action_id in LEFT_ACTION_IDS}
    risk_scores: list[float] = []
    false_negative_scores: list[float] = []

    for root_id in sorted(roots):
        root = roots[root_id]
        candidates = by_root[root_id]
        raw_action_id = root.get("raw_action_id")
        raw = next(
            (row for row in candidates if raw_action_id is not None and int(row.get("action_id", -1)) == int(raw_action_id)),
            None,
        )
        raw_physical_success = bool(raw is not None and _candidate_legal(raw) and _physically_safe(raw) and _observed_success(raw))
        left = [row for row in _left_rows(candidates) if _candidate_legal(row)]
        left_physical_success = [row for row in left if _physically_safe(row) and _observed_success(row)]
        left_risk_pass_physical_success = [row for row in left_physical_success if _secondary_safety_pass(row)]
        risk_false_negative_rows = [row for row in left_physical_success if not _secondary_safety_pass(row)]

        for row in left:
            action_id = str(int(row["action_id"]))
            counter = action_stats[action_id]
            counter["count"] += 1
            counter["secondary_pass_count"] += int(_secondary_safety_pass(row))
            counter["secondary_fail_count"] += int(not _secondary_safety_pass(row))
            counter["physically_safe_count"] += int(_physically_safe(row))
            counter["observed_success_count"] += int(_observed_success(row))
            counter["physical_success_count"] += int(_physically_safe(row) and _observed_success(row))
            counter["risk_false_negative_count"] += int(row in risk_false_negative_rows)
            secondary = dict(row.get("secondary_risk", {}))
            if "risk_score" in secondary:
                risk_scores.append(float(secondary["risk_score"]))
                if row in risk_false_negative_rows:
                    false_negative_scores.append(float(secondary["risk_score"]))

        root_records.append(
            {
                "root_id": root_id,
                "episode_seed": int(root.get("episode_seed", -1)),
                "root_policy": str(root.get("root_policy", root.get("root_source", "unknown"))),
                "collection_source": str(root.get("collection_source", root.get("root_policy", root.get("root_source", "unknown")))),
                "traffic_profile": str(root.get("traffic_profile", "unknown")),
                "raw_action_id": int(raw_action_id) if raw_action_id is not None else None,
                "raw_physical_success": raw_physical_success,
                "left_physical_success_action_ids": [int(row["action_id"]) for row in left_physical_success],
                "left_risk_pass_physical_success_action_ids": [int(row["action_id"]) for row in left_risk_pass_physical_success],
                "risk_false_negative_action_ids": [int(row["action_id"]) for row in risk_false_negative_rows],
                "has_risk_false_negative": bool(risk_false_negative_rows),
                "physical_oracle_available_ignore_risk": bool(raw_physical_success or left_physical_success),
                "risk_gated_physical_available": bool(raw_physical_success or left_risk_pass_physical_success),
            }
        )

    def _score_summary(values: list[float]) -> dict[str, float | int]:
        if not values:
            return {"count": 0}
        array = np.asarray(values, dtype=np.float64)
        return {
            "count": int(array.size),
            "min": float(array.min()),
            "p10": float(np.percentile(array, 10)),
            "p50": float(np.percentile(array, 50)),
            "p90": float(np.percentile(array, 90)),
            "max": float(array.max()),
            "mean": float(array.mean()),
        }

    decision_count = len(root_records)
    physical_available = sum(int(row["physical_oracle_available_ignore_risk"]) for row in root_records)
    risk_gated_available = sum(int(row["risk_gated_physical_available"]) for row in root_records)
    false_negative_roots = sum(int(row["has_risk_false_negative"]) for row in root_records)
    return {
        "diagnostic_kind": "accvp_risk_secondary_false_negative_audit",
        "dataset_dir": str(dataset.resolve()),
        "split": split,
        "decision_count": decision_count,
        "physical_oracle_ceiling_ignore_risk": float(physical_available / max(1, decision_count)),
        "risk_gated_physical_ceiling": float(risk_gated_available / max(1, decision_count)),
        "risk_false_negative_root_count": int(false_negative_roots),
        "risk_false_negative_root_rate": float(false_negative_roots / max(1, decision_count)),
        "risk_false_negative_action_count": int(
            sum(counter["risk_false_negative_count"] for counter in action_stats.values())
        ),
        "left_action_stats": {
            action_id: {
                **dict(counter),
                "risk_false_negative_rate": float(counter["risk_false_negative_count"] / max(1, counter["count"])),
            }
            for action_id, counter in action_stats.items()
        },
        "all_rejected_risk_score_summary": _score_summary(risk_scores),
        "false_negative_risk_score_summary": _score_summary(false_negative_scores),
        "roots": root_records,
    }


def _gate_pass(row: dict[str, Any], thresholds: dict[str, float]) -> bool:
    return (
        bool(row.get("candidate_legal", True))
        and bool(row.get("secondary_safety_pass", True))
        and float(row["pU_proxy_collision"]) <= float(thresholds["proxy_collision_upper_bound"])
        and float(row["pU_safety_violation"]) <= float(thresholds["safety_violation_upper_bound"])
        and float(row["pL_merge_before_taper"]) >= float(thresholds["merge_viability_lower_bound"])
    )


def _risk_eligible_for_model_gate(candidates: list[dict[str, Any]]) -> bool:
    if not candidates:
        return False

    def risk_pass(row: dict[str, Any]) -> bool:
        return bool(row.get("candidate_legal", True)) and bool(
            row.get("secondary_safety_pass", False)
        )

    raw_action_id = int(candidates[0]["raw_action_id"])
    raw = next(
        (row for row in candidates if int(row["action_id"]) == raw_action_id),
        None,
    )
    return bool(raw is not None and risk_pass(raw)) or any(
        int(row["action_id"]) in LEFT_ACTION_IDS and risk_pass(row)
        for row in candidates
    )


def _model_failure_reason(candidates: list[dict[str, Any]], thresholds: dict[str, float]) -> str:
    raw_action_id = int(candidates[0]["raw_action_id"])
    raw = next((row for row in candidates if int(row["action_id"]) == raw_action_id), None)
    if raw is not None and _gate_pass(raw, thresholds):
        return RAW_ALREADY_FEASIBLE
    left = _left_rows(candidates)
    legal_left = [row for row in left if bool(row.get("candidate_legal", True))]
    risk_left = [row for row in legal_left if bool(row.get("secondary_safety_pass", False))]
    safe_left = [row for row in risk_left if not bool(row.get("proxy_collision", False)) and not bool(row.get("safety_violation", False))]
    viable_left = [row for row in safe_left if bool(row.get("merge_before_taper", False)) and bool(row.get("merge_observed", True))]
    if not legal_left:
        return NO_LEGAL_LEFT
    if not risk_left:
        return RISK_LEFT_FAILED
    if not safe_left:
        return PHYSICAL_LEFT_UNSAFE
    if not viable_left:
        return LEFT_TAPER_MISS_OR_CENSORED
    if not any(float(row["pU_proxy_collision"]) <= float(thresholds["proxy_collision_upper_bound"]) for row in viable_left):
        return MODEL_PROXY_FAILED
    proxy_pass = [row for row in viable_left if float(row["pU_proxy_collision"]) <= float(thresholds["proxy_collision_upper_bound"])]
    if not any(float(row["pU_safety_violation"]) <= float(thresholds["safety_violation_upper_bound"]) for row in proxy_pass):
        return MODEL_SAFETY_FAILED
    safety_pass = [row for row in proxy_pass if float(row["pU_safety_violation"]) <= float(thresholds["safety_violation_upper_bound"])]
    if not any(float(row["pL_merge_before_taper"]) >= float(thresholds["merge_viability_lower_bound"]) for row in safety_pass):
        return MODEL_VIABILITY_FAILED
    return "no selected action despite passing merge-intent gates"


def model_gate_failure_diagnostics(
    rows: list[dict[str, Any]],
    thresholds: dict[str, float],
    *,
    required_availability: float,
    split: str = "operating_point",
    evaluated_points: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    by_root: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_root[str(row["root_id"])].append(row)
    decisions = [
        select_viability_action(candidates, raw_action_id=int(candidates[0]["raw_action_id"]), thresholds=thresholds)
        for candidates in by_root.values()
    ]
    selected_count = sum(int(decision["selected"] is not None) for decision in decisions)
    risk_eligible_decision_count = sum(
        _risk_eligible_for_model_gate(candidates)
        for candidates in by_root.values()
    )
    decision_count = len(by_root)
    root_records: list[dict[str, Any]] = []
    for root_id, candidates in sorted(by_root.items()):
        decision = select_viability_action(candidates, raw_action_id=int(candidates[0]["raw_action_id"]), thresholds=thresholds)
        if decision["selected"] is not None:
            reason = RAW_ALREADY_FEASIBLE if bool(decision["raw_feasible"]) else MODEL_LEFT_SELECTED
        else:
            reason = _model_failure_reason(candidates, thresholds)
        root_records.append(
            {
                "root_id": root_id,
                "raw_action_id": int(candidates[0]["raw_action_id"]),
                "selected_action_id": int(decision["selected"]["action_id"]) if decision["selected"] is not None else None,
                "candidate_set_available": bool(decision["selected"] is not None),
                "raw_feasible": bool(decision["raw_feasible"]),
                "reason": reason,
            }
        )
    action_stats: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        action_id = str(int(row["action_id"]))
        action_stats[action_id]["count"] += 1
        action_stats[action_id]["gate_pass_count"] += int(_gate_pass(row, thresholds))
        action_stats[action_id]["candidate_legal_count"] += int(bool(row.get("candidate_legal", True)))
        action_stats[action_id]["secondary_pass_count"] += int(bool(row.get("secondary_safety_pass", False)))
        action_stats[action_id]["proxy_gate_pass_count"] += int(float(row["pU_proxy_collision"]) <= float(thresholds["proxy_collision_upper_bound"]))
        action_stats[action_id]["safety_gate_pass_count"] += int(float(row["pU_safety_violation"]) <= float(thresholds["safety_violation_upper_bound"]))
        action_stats[action_id]["viability_gate_pass_count"] += int(float(row["pL_merge_before_taper"]) >= float(thresholds["merge_viability_lower_bound"]))
    return {
        "diagnostic_kind": "accvp_tuning_failure",
        "deployable_artifact": False,
        "split": split,
        "required_availability": float(required_availability),
        "best_thresholds": dict(thresholds),
        "availability_denominator_version": AVAILABILITY_DENOMINATOR_VERSION,
        "model_gate_best_availability": float(
            selected_count / max(1, risk_eligible_decision_count)
        ),
        "model_conditional_availability": float(
            selected_count / max(1, risk_eligible_decision_count)
        ),
        "unconditional_candidate_set_availability": float(
            selected_count / max(1, decision_count)
        ),
        "risk_eligible_decision_fraction": float(
            risk_eligible_decision_count / max(1, decision_count)
        ),
        "risk_eligible_decision_count": int(risk_eligible_decision_count),
        "risk_ineligible_decision_count": int(
            decision_count - risk_eligible_decision_count
        ),
        "selected_count": int(selected_count),
        "decision_count": int(decision_count),
        "reason_counts": dict(Counter(str(row["reason"]) for row in root_records)),
        "per_root_gate_failure_reason": root_records,
        "per_action_gate_pass_rate": {
            action_id: {
                **dict(counter),
                "gate_pass_rate": float(counter["gate_pass_count"] / max(1, counter["count"])),
            }
            for action_id, counter in sorted(action_stats.items(), key=lambda item: int(item[0]))
        },
        "evaluated_points": list(evaluated_points or []),
    }
