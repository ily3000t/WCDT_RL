from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from safe_rl.accvp.schema import write_json_atomic
from safe_rl.accvp.selection import LEFT_ACTION_IDS, lite_secondary_safety_pass, select_viability_lite_action
from safe_rl.accvp.viability_lite import evaluate_lite_thresholds


def _by_root(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[str(row["root_id"])].append(row)
    return grouped


def _strict_risk_pass(row: dict[str, Any]) -> bool:
    return bool(row.get("candidate_legal", True)) and bool(row.get("secondary_safety_pass", True))


def _risk_pass(row: dict[str, Any], thresholds: dict[str, Any]) -> bool:
    return lite_secondary_safety_pass(row, thresholds)


def _safe(row: dict[str, Any]) -> bool:
    return float(row.get("proxy_collision", 0.0)) < 0.5 and float(row.get("safety_violation", 0.0)) < 0.5


def _success(row: dict[str, Any]) -> bool:
    return bool(row.get("merge_observed", False)) and float(row.get("merge_before_taper", 0.0)) >= 0.5


def _rate(values: list[bool]) -> float | None:
    return float(np.mean([bool(value) for value in values])) if values else None


def _action_summary(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return {
        "action_id": int(row["action_id"]),
        "p_merge_before_taper": float(row.get("p_merge_before_taper", 0.0)),
        "p_proxy_collision": float(row.get("p_proxy_collision", 0.0)),
        "p_safety_violation": float(row.get("p_safety_violation", 0.0)),
        "target_lane_entry_time_s": (
            None if row.get("target_lane_entry_time_s") is None else float(row.get("target_lane_entry_time_s", 0.0))
        ),
        "ensemble_disagreement": float(row.get("ensemble_disagreement", 0.0)),
        "candidate_legal": bool(row.get("candidate_legal", True)),
        "secondary_safety_pass": bool(row.get("secondary_safety_pass", True)),
        "lite_secondary_pass": bool(row.get("accvp_lite_secondary_pass", False)),
        "lite_secondary_safety_profile": str(row.get("accvp_lite_secondary_safety_profile", "")),
        "secondary_risk_score": float(row.get("secondary_risk_score", 0.0)),
        "secondary_risk_uncertainty": float(row.get("secondary_risk_uncertainty", 0.0)),
        "secondary_veto_reason": str(row.get("secondary_veto_reason", "")),
        "merge_observed": bool(row.get("merge_observed", False)),
        "merge_success": _success(row),
        "proxy_collision": bool(float(row.get("proxy_collision", 0.0)) >= 0.5),
        "safety_violation": bool(float(row.get("safety_violation", 0.0)) >= 0.5),
        "safe": _safe(row),
        "oracle_min_obb_distance": row.get("oracle_min_obb_distance"),
        "oracle_min_ttc": row.get("oracle_min_ttc"),
        "oracle_max_drac": row.get("oracle_max_drac"),
        "oracle_geometric_overlap": bool(row.get("oracle_geometric_overlap", False)),
    }


def _root_summary(
    *,
    root_id: str,
    candidates: list[dict[str, Any]],
    raw: dict[str, Any] | None,
    selected: dict[str, Any] | None,
    decision: dict[str, Any],
) -> dict[str, Any]:
    first = candidates[0]
    return {
        "root_id": root_id,
        "episode_seed": int(first.get("episode_seed", -1)),
        "root_policy": str(first.get("root_policy", "")),
        "collection_source": str(first.get("collection_source", "")),
        "traffic_profile": str(first.get("traffic_profile", "")),
        "activation_bin": str(first.get("activation_bin", "")),
        "raw_action_id": None if raw is None else int(raw["action_id"]),
        "selected_action_id": None if selected is None else int(selected["action_id"]),
        "replacement": bool(decision.get("replacement", False)),
        "reason": str(decision.get("reason", "")),
        "raw": _action_summary(raw),
        "selected": _action_summary(selected),
        "p_merge_improvement": float(decision.get("p_merge_improvement", 0.0)),
        "raw_task_feasible": bool(decision.get("raw_task_feasible", False)),
        "candidate_count": int(len(candidates)),
    }


def audit_lite_replacements(
    records: list[dict[str, Any]],
    thresholds: dict[str, float],
    *,
    split: str,
    max_targeted_seeds: int = 20,
) -> dict[str, Any]:
    grouped = _by_root(records)
    selected_rows: list[dict[str, Any]] = []
    replacement_rows: list[dict[str, Any]] = []
    raw_retained_rows: list[dict[str, Any]] = []
    replacement_safety_event_roots: list[dict[str, Any]] = []
    unnecessary_replacement_roots: list[dict[str, Any]] = []
    risk_failed_but_success_roots: list[dict[str, Any]] = []
    targeted_seed_candidates: list[tuple[str, int]] = []
    reason_counts: Counter[str] = Counter()
    replacement_action_counts: Counter[str] = Counter()

    for root_id, candidates in grouped.items():
        raw_action_id = int(candidates[0].get("raw_action_id", -1))
        raw = next((row for row in candidates if int(row["action_id"]) == raw_action_id), None)
        decision = select_viability_lite_action(candidates, raw_action_id=raw_action_id, thresholds=thresholds)
        reason_counts[str(decision.get("reason", ""))] += 1
        selected = decision.get("selected")
        if selected is not None:
            selected_rows.append(selected)
            if bool(decision.get("replacement", False)):
                replacement_rows.append(selected)
                replacement_action_counts[str(int(selected["action_id"]))] += 1
            else:
                raw_retained_rows.append(selected)

        root_report = _root_summary(root_id=root_id, candidates=candidates, raw=raw, selected=selected, decision=decision)
        if bool(decision.get("replacement", False)) and selected is not None:
            if not _safe(selected):
                replacement_safety_event_roots.append(root_report)
            if raw is not None and _success(raw):
                unnecessary_replacement_roots.append(root_report)
            if raw is not None and not _success(raw) and _safe(selected) and _success(selected):
                targeted_seed_candidates.append((root_id, int(candidates[0].get("episode_seed", -1))))

        for row in candidates:
            if int(row["action_id"]) in LEFT_ACTION_IDS and not _strict_risk_pass(row) and _safe(row) and _success(row):
                risk_failed_but_success_roots.append(
                    {
                        **_root_summary(root_id=root_id, candidates=candidates, raw=raw, selected=row, decision=decision),
                        "risk_failed_action": _action_summary(row),
                    }
                )

    selected_observed = [row for row in selected_rows if bool(row.get("merge_observed", False))]
    replacement_observed = [row for row in replacement_rows if bool(row.get("merge_observed", False))]
    targeted_seeds: list[int] = []
    seen: set[int] = set()
    for _root_id, seed in sorted(targeted_seed_candidates, key=lambda item: (item[0], item[1])):
        if seed >= 0 and seed not in seen:
            targeted_seeds.append(seed)
            seen.add(seed)
        if len(targeted_seeds) >= max_targeted_seeds:
            break

    replacement_count = len(replacement_rows)
    unnecessary_count = len(unnecessary_replacement_roots)

    return {
        "split": split,
        "thresholds": dict(thresholds),
        "decision_count": int(len(grouped)),
        "selected_count": int(len(selected_rows)),
        "all_selected_count": int(len(selected_rows)),
        "raw_retained_selected_count": int(len(raw_retained_rows)),
        "actual_replacement_count": int(replacement_count),
        "replacement_count": int(replacement_count),
        "replacement_rate": float(replacement_count / max(1, len(grouped))),
        "reason_counts": dict(sorted(reason_counts.items())),
        "replacement_action_histogram": dict(sorted(replacement_action_counts.items(), key=lambda item: int(item[0]))),
        "all_selected_action_risk_pass_rate": _rate([_risk_pass(row, thresholds) for row in selected_rows]),
        "all_selected_action_safety_event_rate": _rate([not _safe(row) for row in selected_rows]),
        "all_selected_action_merge_success_rate": _rate([_success(row) for row in selected_observed]),
        "replacement_action_risk_pass_rate": _rate([_risk_pass(row, thresholds) for row in replacement_rows]),
        "replacement_action_safety_event_rate": _rate([not _safe(row) for row in replacement_rows]),
        "replacement_action_merge_success_rate": _rate([_success(row) for row in replacement_observed]),
        "replacement_safety_event_root_count": int(len(replacement_safety_event_roots)),
        "risk_failed_but_success_root_count": int(len(risk_failed_but_success_roots)),
        "unnecessary_replacement_root_count": int(unnecessary_count),
        "replacement_unnecessary_rate": float(unnecessary_count / replacement_count) if replacement_count else None,
        "replacement_safety_event_roots": replacement_safety_event_roots,
        "risk_failed_but_success_roots": risk_failed_but_success_roots,
        "unnecessary_replacement_roots": unnecessary_replacement_roots,
        "targeted_seeds": targeted_seeds,
        "targeted_seed_source_count": int(len(targeted_seed_candidates)),
        "summary_from_tuning_metric": evaluate_lite_thresholds(records, thresholds, split=split),
    }


def write_lite_replacement_audit(
    *,
    output_dir: str | Path,
    report: dict[str, Any],
) -> dict[str, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "report": write_json_atomic(output / "accvp_lite_replacement_audit.json", report),
        "replacement_safety_event_roots": write_json_atomic(
            output / "replacement_safety_event_roots.json",
            {"roots": report.get("combined", {}).get("replacement_safety_event_roots", [])},
        ),
        "risk_failed_but_success_roots": write_json_atomic(
            output / "risk_failed_but_success_roots.json",
            {"roots": report.get("combined", {}).get("risk_failed_but_success_roots", [])},
        ),
        "unnecessary_replacement_roots": write_json_atomic(
            output / "unnecessary_replacement_roots.json",
            {"roots": report.get("combined", {}).get("unnecessary_replacement_roots", [])},
        ),
        "targeted_seeds": write_json_atomic(
            output / "targeted_seeds.json",
            {"seeds": report.get("targeted_seeds", [])},
        ),
    }
    return paths
