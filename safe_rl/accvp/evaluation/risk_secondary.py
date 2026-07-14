"""Secondary Risk false-negative and coverage audit."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from safe_rl.accvp.contracts.schema import write_json_atomic
from safe_rl.accvp.planning.selection import LEFT_ACTION_IDS, lite_secondary_safety_pass
from safe_rl.accvp.planning.viability_lite import (
    evaluate_lite_thresholds,
    outcome_merge_observation_rate,
    outcome_merge_success_rate,
    outcome_safety_event_rate,
)


def _rate(values: list[bool]) -> float | None:
    return float(np.mean([bool(value) for value in values])) if values else None


def _quantiles(values: list[float]) -> dict[str, float | None]:
    clean = [float(value) for value in values if np.isfinite(float(value))]
    if not clean:
        return {"count": 0, "min": None, "p10": None, "p50": None, "p90": None, "p95": None, "max": None, "mean": None}
    arr = np.asarray(clean, dtype=np.float64)
    return {
        "count": int(arr.size),
        "min": float(np.min(arr)),
        "p10": float(np.quantile(arr, 0.10)),
        "p50": float(np.quantile(arr, 0.50)),
        "p90": float(np.quantile(arr, 0.90)),
        "p95": float(np.quantile(arr, 0.95)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
    }


def _risk_score(row: dict[str, Any]) -> float:
    if row.get("secondary_risk_score") is not None:
        return float(row.get("secondary_risk_score", 0.0))
    secondary = dict(row.get("secondary_risk", {}) or {})
    return float(secondary.get("risk_score", 0.0))


def _strict_pass(row: dict[str, Any]) -> bool:
    return bool(row.get("candidate_legal", True)) and bool(row.get("secondary_safety_pass", True))


def _success(row: dict[str, Any]) -> bool:
    tolerance = np.finfo(np.float64).eps
    return (
        outcome_merge_observation_rate(row) >= 1.0 - tolerance
        and outcome_merge_success_rate(row) >= 1.0 - tolerance
    )


def _safety_event(row: dict[str, Any]) -> bool:
    return (
        outcome_safety_event_rate(row) > np.finfo(np.float64).eps
        or bool(row.get("oracle_geometric_overlap", False))
    )


def _clean_success(row: dict[str, Any]) -> bool:
    return _success(row) and not _safety_event(row)


def _row_summary(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "root_id": str(row.get("root_id", "")),
        "episode_seed": int(row.get("episode_seed", -1)),
        "split": str(row.get("split", "")),
        "root_policy": str(row.get("root_policy", "")),
        "collection_source": str(row.get("collection_source", "")),
        "traffic_profile": str(row.get("traffic_profile", "")),
        "activation_bin": str(row.get("activation_bin", "")),
        "action_id": int(row.get("action_id", -1)),
        "candidate_legal": bool(row.get("candidate_legal", True)),
        "secondary_safety_pass": bool(row.get("secondary_safety_pass", False)),
        "secondary_risk_score": _risk_score(row),
        "secondary_risk_uncertainty": float(row.get("secondary_risk_uncertainty", 0.0)),
        "secondary_veto_reason": str(row.get("secondary_veto_reason", "")),
        "merge_observed": bool(row.get("merge_observed", False)),
        "merge_success": _success(row),
        "merge_observation_rate": outcome_merge_observation_rate(row),
        "merge_success_rate": outcome_merge_success_rate(row),
        "safety_event_rate": outcome_safety_event_rate(row),
        "clean_success": _clean_success(row),
        "safety_event": _safety_event(row),
        "proxy_collision": float(row.get("proxy_collision", 0.0)) >= 0.5,
        "safety_violation": float(row.get("safety_violation", 0.0)) >= 0.5,
        "oracle_geometric_overlap": bool(row.get("oracle_geometric_overlap", False)),
        "oracle_min_obb_distance": row.get("oracle_min_obb_distance"),
        "oracle_min_ttc": row.get("oracle_min_ttc"),
        "oracle_max_drac": row.get("oracle_max_drac"),
        "p_merge_before_taper": float(row.get("p_merge_before_taper", 0.0)),
        "target_lane_entry_time_s": row.get("target_lane_entry_time_s"),
        "ensemble_disagreement": float(row.get("ensemble_disagreement", 0.0)),
    }


def _group_counts(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    counter: Counter[str] = Counter(str(row.get(field, "")) for row in rows)
    return dict(sorted(counter.items()))


def _split_by_root(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[str(row.get("root_id", ""))].append(row)
    return grouped


def audit_risk_secondary(
    records: list[dict[str, Any]],
    base_thresholds: dict[str, Any],
    *,
    split: str,
    risk_score_grid: list[float],
) -> dict[str, Any]:
    selection_eligible = str(split) == "operating_point"
    rows = [{**row, "split": split} for row in records if int(row.get("action_id", -1)) in LEFT_ACTION_IDS]
    strict_pass_rows = [row for row in rows if _strict_pass(row)]
    strict_fail_rows = [row for row in rows if not _strict_pass(row)]
    clean_success_rows = [row for row in rows if _clean_success(row)]
    unsafe_rows = [row for row in rows if _safety_event(row)]
    clean_success_failed = [row for row in rows if _clean_success(row) and not _strict_pass(row)]
    unsafe_passed = [row for row in rows if _safety_event(row) and _strict_pass(row)]
    confusion = {
        "risk_pass_clean_success": len([row for row in rows if _strict_pass(row) and _clean_success(row)]),
        "risk_fail_clean_success": len(clean_success_failed),
        "risk_pass_unsafe": len(unsafe_passed),
        "risk_fail_unsafe": len([row for row in rows if (not _strict_pass(row)) and _safety_event(row)]),
        "risk_pass_other": len([row for row in rows if _strict_pass(row) and not _clean_success(row) and not _safety_event(row)]),
        "risk_fail_other": len([row for row in rows if (not _strict_pass(row)) and not _clean_success(row) and not _safety_event(row)]),
    }
    clean_success_failed_total = max(1, len(clean_success_failed))
    unsafe_total = max(1, len(unsafe_rows))
    sweep = []
    # A non-selection split may only evaluate the already frozen threshold.
    # In particular, test must never expose a threshold grid or a "best"
    # profile that could feed a subsequent tuning decision.
    effective_grid = (
        list(risk_score_grid)
        if selection_eligible
        else [float(base_thresholds.get("max_secondary_risk_score", 1.0))]
    )
    for threshold in effective_grid:
        thresholds = {
            **base_thresholds,
            "max_secondary_risk_score": float(threshold),
            "secondary_safety_profile": "audited_merge_left_v1",
        }
        threshold_pass_rows = [row for row in rows if lite_secondary_safety_pass(row, thresholds)]
        recovered = [row for row in clean_success_failed if lite_secondary_safety_pass(row, thresholds)]
        unsafe_pass = [row for row in unsafe_rows if lite_secondary_safety_pass(row, thresholds)]
        replacement_metrics = evaluate_lite_thresholds(records, thresholds, split=split)
        sweep.append(
            {
                "max_secondary_risk_score": float(threshold),
                "threshold_pass_count": int(len(threshold_pass_rows)),
                "safe_success_recovered_count": int(len(recovered)),
                "safe_success_recovery_rate": float(len(recovered) / clean_success_failed_total),
                "unsafe_pass_count": int(len(unsafe_pass)),
                "unsafe_pass_rate": float(len(unsafe_pass) / unsafe_total),
                "merge_success_rate": _rate([_success(row) for row in threshold_pass_rows if bool(row.get("merge_observed", False))]),
                "clean_success_rate": _rate([_clean_success(row) for row in threshold_pass_rows if bool(row.get("merge_observed", False))]),
                "replacement_count": replacement_metrics["replacement_count"],
                "replacement_rate": replacement_metrics["replacement_rate"],
                "replacement_action_safety_event_rate": replacement_metrics["replacement_action_safety_event_rate"],
                "replacement_action_merge_success_rate": replacement_metrics["replacement_action_merge_success_rate"],
                "replacement_unnecessary_rate": replacement_metrics["replacement_unnecessary_rate"],
                "replacement_repairable_capture_rate": replacement_metrics["replacement_repairable_capture_rate"],
            }
        )
    safe_sweep = [
        row
        for row in sweep
        if int(row["replacement_count"]) > 0
        and float(row["unsafe_pass_rate"]) == 0.0
        and float(row["replacement_action_safety_event_rate"]) == 0.0
        and (row["replacement_action_merge_success_rate"] is not None and float(row["replacement_action_merge_success_rate"]) >= 0.90)
        and float(row["replacement_unnecessary_rate"]) <= 0.25
        and float(row["replacement_rate"]) <= float(base_thresholds.get("max_replacement_rate", 0.10))
    ]
    diagnostic_best = (
        max(
            safe_sweep,
            key=lambda row: (
                float(row["replacement_repairable_capture_rate"]),
                float(row["safe_success_recovery_rate"]),
                -float(row["max_secondary_risk_score"]),
            ),
            default=None,
        )
        if selection_eligible
        else None
    )
    selected = diagnostic_best if selection_eligible else None
    return {
        "split": split,
        "selection_eligible": selection_eligible,
        "selection_note": (
            "threshold selection is permitted only on the operating_point split"
            if not selection_eligible
            else "operating_point is the sole threshold-selection split"
        ),
        "left_action_count": int(len(rows)),
        "strict_pass_count": int(len(strict_pass_rows)),
        "strict_fail_count": int(len(strict_fail_rows)),
        "clean_success_count": int(len(clean_success_rows)),
        "unsafe_count": int(len(unsafe_rows)),
        "confusion": confusion,
        "risk_score_distribution": {
            "all_left": _quantiles([_risk_score(row) for row in rows]),
            "strict_pass": _quantiles([_risk_score(row) for row in strict_pass_rows]),
            "strict_fail": _quantiles([_risk_score(row) for row in strict_fail_rows]),
            "clean_success_failed": _quantiles([_risk_score(row) for row in clean_success_failed]),
            "unsafe_passed": _quantiles([_risk_score(row) for row in unsafe_passed]),
        },
        "by_action": {
            str(action_id): {
                "count": len([row for row in rows if int(row["action_id"]) == action_id]),
                "risk_fail_clean_success": len(
                    [row for row in clean_success_failed if int(row["action_id"]) == action_id]
                ),
                "risk_pass_unsafe": len([row for row in unsafe_passed if int(row["action_id"]) == action_id]),
            }
            for action_id in sorted(LEFT_ACTION_IDS)
        },
        "by_source": _group_counts(clean_success_failed, "collection_source"),
        "by_root_policy": _group_counts(clean_success_failed, "root_policy"),
        "threshold_sweep": sweep,
        "threshold_sweep_is_selection": selection_eligible,
        "selected_audited_profile": None
        if selected is None
        else {
            "secondary_safety_profile": "audited_merge_left_v1",
            "max_secondary_risk_score": float(selected["max_secondary_risk_score"]),
            "source": "accvp_risk_secondary_audit",
            "split": split,
        },
        "diagnostic_best_profile": None
        if diagnostic_best is None
        else {
            "secondary_safety_profile": "audited_merge_left_v1",
            "max_secondary_risk_score": float(diagnostic_best["max_secondary_risk_score"]),
            "source": "accvp_risk_secondary_audit",
            "split": split,
            "selection_eligible": selection_eligible,
        },
        "clean_success_but_risk_failed_roots": [_row_summary(row) for row in clean_success_failed],
        "unsafe_but_risk_passed_roots": [_row_summary(row) for row in unsafe_passed],
    }


def combine_audit_reports(split_reports: dict[str, dict[str, Any]]) -> dict[str, Any]:
    operating_point = split_reports.get("operating_point")
    selected = None if operating_point is None else operating_point.get("selected_audited_profile")
    if selected is not None and str(selected.get("split", "")) != "operating_point":
        raise ValueError("Risk-secondary threshold provenance must be operating_point")
    return {
        "artifact_kind": "accvp_risk_secondary_audit_v1",
        "controller": "acv_shield_lite",
        "safety_authority": "risk_module_safety_shield",
        "accvp_safety_head_hard_gate": False,
        "splits": split_reports,
        "selection_split": "operating_point",
        "test_used_for_selection": False,
        "selected_audited_profile": selected,
        "audit_state": "go" if selected is not None else "no_safe_operating_point_threshold",
    }


def write_risk_secondary_audit(*, output_dir: str | Path, report: dict[str, Any]) -> dict[str, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    combined_clean = []
    combined_unsafe = []
    for split, split_report in dict(report.get("splits", {})).items():
        combined_clean.extend({**row, "split": split} for row in split_report.get("clean_success_but_risk_failed_roots", []))
        combined_unsafe.extend({**row, "split": split} for row in split_report.get("unsafe_but_risk_passed_roots", []))
    return {
        "report": write_json_atomic(output / "accvp_risk_secondary_audit.json", report),
        "clean_success_but_risk_failed_roots": write_json_atomic(
            output / "risk_clean_success_failed_roots.json",
            {"roots": combined_clean},
        ),
        "unsafe_but_risk_passed_roots": write_json_atomic(
            output / "risk_unsafe_passed_roots.json",
            {"roots": combined_unsafe},
        ),
        "audited_profile": write_json_atomic(
            output / "accvp_v1_lite_v3_audited_secondary_profile.json",
            {
                "audit_state": report.get("audit_state"),
                "selected_audited_profile": report.get("selected_audited_profile"),
            },
        ),
    }
