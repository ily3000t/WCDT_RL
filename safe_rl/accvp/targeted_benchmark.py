from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from safe_rl.accvp.online_trigger_audit import _as_int
from safe_rl.accvp.schema import file_sha256, write_json_atomic
from safe_rl.accvp.selection import LEFT_ACTION_IDS


CASE_TABLE_FIELDS = [
    "seed",
    "group_name",
    "step",
    "decision_index",
    "raw_action",
    "shield_action",
    "selected_action",
    "selection_reason",
    "raw_p_merge_before_taper",
    "selected_p_merge_before_taper",
    "p_merge_improvement",
    "target_entry_time_s",
    "secondary_risk_score",
    "secondary_safety_profile",
    "secondary_safety_pass",
    "lite_secondary_pass",
    "min_distance",
    "ttc_p1",
    "drac_p99",
    "proxy_collision",
    "safety_violation",
    "done_reason",
    "first_target_lane_entry_distance_to_taper",
]


def _load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _replay_files(replay_dirs: list[str | Path]) -> list[Path]:
    files: list[Path] = []
    for replay_dir in replay_dirs:
        root = Path(replay_dir)
        files.extend(sorted(root.rglob("*.json") if root.is_dir() else [root]))
    return files


def _episode_payload(path: Path) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    payload = _load_json(path)
    notes = dict(payload.get("notes", {}) or {})
    episode = dict(notes.get("episode_report", {}) or {})
    records = list(notes.get("accvp_records", episode.get("accvp_records", [])) or [])
    meta = {
        "seed": _as_int(payload.get("seed", episode.get("seed", -1))),
        "group_name": str(payload.get("group_name", "")),
        "path": str(path),
    }
    return meta, episode, records


def _is_action_change(record: dict[str, Any]) -> bool:
    reason = str(record.get("accvp_replacement_reason", ""))
    if not bool(record.get("accvp_replacement", False)) or reason == "lateral_commitment":
        return False
    if "accvp_action_change" in record:
        return bool(record.get("accvp_action_change", False))
    selected = record.get("accvp_selected_action")
    shield_action = record.get("safety_shield_action")
    return selected is not None and shield_action is not None and _as_int(selected) != _as_int(shield_action)


def _candidate(record: dict[str, Any], action_id: int) -> dict[str, Any] | None:
    for row in list(record.get("accvp_shadow_candidates", []) or []):
        if _as_int(row.get("action_id")) == int(action_id):
            return dict(row)
    return None


def _value(row: dict[str, Any] | None, key: str) -> Any:
    if row is None:
        return None
    return row.get(key)


def build_replacement_case_table(
    replay_dirs: list[str | Path],
    *,
    group_contains: str | None = None,
) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for path in _replay_files(replay_dirs):
        meta, episode, records = _episode_payload(path)
        if group_contains and group_contains not in meta["group_name"]:
            continue
        for record in records:
            if not _is_action_change(record):
                continue
            raw_action = _as_int(record.get("raw_action"))
            selected_action = _as_int(record.get("accvp_selected_action"))
            raw = _candidate(record, raw_action)
            selected = _candidate(record, selected_action)
            cases.append(
                {
                    "seed": meta["seed"],
                    "group_name": meta["group_name"],
                    "step": _as_int(record.get("step")),
                    "decision_index": _as_int(record.get("decision_index")),
                    "raw_action": raw_action,
                    "shield_action": _as_int(record.get("safety_shield_action")),
                    "selected_action": selected_action,
                    "selection_reason": str(record.get("accvp_replacement_reason", "")),
                    "raw_p_merge_before_taper": _value(raw, "p_merge_before_taper"),
                    "selected_p_merge_before_taper": _value(selected, "p_merge_before_taper"),
                    "p_merge_improvement": record.get("accvp_lite_p_merge_improvement"),
                    "target_entry_time_s": _value(selected, "target_lane_entry_time_s"),
                    "secondary_risk_score": _value(selected, "secondary_risk_score"),
                    "secondary_safety_profile": _value(selected, "lite_secondary_safety_profile"),
                    "secondary_safety_pass": _value(selected, "secondary_safety_pass"),
                    "lite_secondary_pass": _value(selected, "lite_secondary_pass"),
                    "min_distance": episode.get("min_distance"),
                    "ttc_p1": episode.get("ttc_p1"),
                    "drac_p99": episode.get("drac_p99"),
                    "proxy_collision": bool(episode.get("proxy_collision", False)),
                    "safety_violation": bool(episode.get("safety_violation", False)),
                    "done_reason": str(episode.get("done_reason", "")),
                    "first_target_lane_entry_distance_to_taper": episode.get(
                        "first_target_lane_entry_distance_to_taper"
                    ),
                }
            )
    return sorted(cases, key=lambda row: (int(row["seed"]), int(row["decision_index"]), int(row["step"])))


def _metric(metrics: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = metrics.get(key, default)
    return default if value is None else float(value)


def _summary_metric_block(metrics: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "terminal_success_rate",
        "timely_merge_success_rate",
        "first_target_lane_entry_distance_to_taper_p10",
        "first_target_lane_entry_distance_to_taper_p50",
        "first_target_lane_entry_distance_to_taper_mean",
        "deadline_opportunity_capture_rate",
        "late_merge_request_rate",
        "taper_miss_rate",
        "proxy_collision_rate",
        "safety_violation_rate",
        "fallback_rate",
        "accvp_active_action_change_count",
        "accvp_active_action_change_per_decision_rate",
        "accvp_active_same_action_confirm_count",
        "accvp_shadow_latency_p95",
    ]
    return {key: metrics.get(key) for key in keys if key in metrics}


def build_targeted_benchmark_summary(
    *,
    stage5_report: str | Path,
    cases: list[dict[str, Any]],
    baseline_group: str,
    accvp_group: str,
    online_audit: str | Path | None = None,
) -> dict[str, Any]:
    report = _load_json(stage5_report)
    groups = dict(report.get("groups", {}) or {})
    baseline_metrics = dict(groups.get(baseline_group, {}).get("metrics", {}) or {})
    accvp_metrics = dict(groups.get(accvp_group, {}).get("metrics", {}) or {})
    online = _load_json(online_audit) if online_audit else {}
    action_change_count = int(accvp_metrics.get("accvp_active_action_change_count", len(cases)) or 0)
    selected_actions = [int(row["selected_action"]) for row in cases]
    case_safety_events = [
        bool(row.get("proxy_collision", False)) or bool(row.get("safety_violation", False))
        for row in cases
    ]
    all_left = all(action in LEFT_ACTION_IDS for action in selected_actions)
    all_audited_risk_pass = all(
        bool(row.get("lite_secondary_pass", False))
        and str(row.get("secondary_safety_profile", "")) == "audited_merge_left_v1"
        for row in cases
    )
    task_improvements = {
        "first_target_lane_entry_distance_to_taper_p50_increased": _metric(
            accvp_metrics, "first_target_lane_entry_distance_to_taper_p50", -1.0
        )
        > _metric(baseline_metrics, "first_target_lane_entry_distance_to_taper_p50", -1.0),
        "deadline_opportunity_capture_rate_increased": _metric(
            accvp_metrics, "deadline_opportunity_capture_rate", -1.0
        )
        > _metric(baseline_metrics, "deadline_opportunity_capture_rate", -1.0),
        "late_merge_request_rate_decreased": _metric(accvp_metrics, "late_merge_request_rate", 1.0)
        < _metric(baseline_metrics, "late_merge_request_rate", 1.0),
    }
    safety_checks = {
        "collision_not_worse": _metric(accvp_metrics, "collision_rate") <= _metric(baseline_metrics, "collision_rate"),
        "proxy_collision_not_worse": _metric(accvp_metrics, "proxy_collision_rate")
        <= _metric(baseline_metrics, "proxy_collision_rate"),
        "safety_violation_not_worse": _metric(accvp_metrics, "safety_violation_rate")
        <= _metric(baseline_metrics, "safety_violation_rate"),
        "fallback_rate_zero": _metric(accvp_metrics, "fallback_rate") == 0.0,
        "replacement_level_safety_event_rate_zero": not any(case_safety_events),
        "action_changes_are_left_actions": all_left,
        "action_changes_pass_audited_risk_profile": all_audited_risk_pass,
    }
    task_checks = {
        "action_change_count_positive": action_change_count > 0,
        "taper_miss_not_worse": _metric(accvp_metrics, "taper_miss_rate")
        <= _metric(baseline_metrics, "taper_miss_rate"),
        "timely_merge_success_not_worse": _metric(accvp_metrics, "timely_merge_success_rate")
        >= _metric(baseline_metrics, "timely_merge_success_rate"),
        "at_least_one_task_quality_metric_improved": any(task_improvements.values()),
        **task_improvements,
    }
    task_gate_required = {
        key: task_checks[key]
        for key in (
            "action_change_count_positive",
            "taper_miss_not_worse",
            "timely_merge_success_not_worse",
            "at_least_one_task_quality_metric_improved",
        )
    }
    latency_checks = {
        "p95_latency_le_0_5s": _metric(accvp_metrics, "accvp_shadow_latency_p95") <= 0.5,
    }
    engineering_checks = {
        "stage5_report_available": bool(groups),
        "online_audit_matches_action_change_count": (
            online.get("actual_action_change_count") is None
            or int(online.get("actual_action_change_count", -1)) == action_change_count
        ),
    }
    return {
        "artifact_kind": "accvp_lite_v3_targeted_benchmark_summary_v1",
        "stage5_report": str(stage5_report),
        "stage5_report_sha256": file_sha256(stage5_report),
        "baseline_group": baseline_group,
        "accvp_group": accvp_group,
        "baseline_metrics": _summary_metric_block(baseline_metrics),
        "accvp_metrics": _summary_metric_block(accvp_metrics),
        "action_change_case_count": len(cases),
        "action_change_selected_action_counts": {
            str(action): selected_actions.count(action) for action in sorted(set(selected_actions))
        },
        "safety_checks": safety_checks,
        "task_checks": task_checks,
        "latency_checks": latency_checks,
        "engineering_checks": engineering_checks,
        "safety_gate_pass": all(safety_checks.values()),
        "task_gate_pass": all(task_gate_required.values()),
        "latency_gate_pass": all(latency_checks.values()),
        "engineering_gate_pass": all(engineering_checks.values()),
        "performance_benefit_claim_allowed": all(safety_checks.values()) and all(task_gate_required.values()),
    }


def _write_csv(path: str | Path, rows: list[dict[str, Any]]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CASE_TABLE_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in CASE_TABLE_FIELDS})
    tmp.replace(output)
    return output


def write_targeted_benchmark_outputs(
    *,
    output_dir: str | Path,
    cases: list[dict[str, Any]],
    summary: dict[str, Any],
) -> dict[str, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    return {
        "summary": write_json_atomic(output / "accvp_lite_v3_targeted_benchmark_summary.json", summary),
        "case_table_json": write_json_atomic(output / "accvp_lite_v3_replacement_case_table.json", cases),
        "case_table_csv": _write_csv(output / "accvp_lite_v3_replacement_case_table.csv", cases),
    }
