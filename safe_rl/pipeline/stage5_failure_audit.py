from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from safe_rl.pipeline.common import run_root, write_report
from safe_rl.utils.config import load_config
from safe_rl.utils.progress import stage_log


DEFAULT_GROUPS = ("cv_prediction_shield", "wcdt_v3_prediction_shield")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _report_path(base_dir: Path) -> Path:
    return base_dir / "formal_paired_eval_report.json"


def _is_failure(episode: dict[str, Any], collision_threshold: float) -> bool:
    return bool(
        episode.get("proxy_collision", False)
        or episode.get("geometric_overlap", False)
        or episode.get("safety_violation", False)
        or _safe_float(episode.get("min_distance"), 1.0e9) <= collision_threshold
        or _safe_float(episode.get("drac_p99_raw"), 0.0) >= 1.0e5
    )


def _is_task_failure(episode: dict[str, Any]) -> bool:
    done_reason = str(episode.get("done_reason", "") or "")
    return bool(
        episode.get("taper_miss", False)
        or done_reason == "taper_miss"
        or episode.get("merge_success") is False
    )


def _compact_record(record: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "raw_action",
        "final_action",
        "raw_risk",
        "raw_risk_score",
        "best_candidate_risk",
        "best_candidate_risk_score",
        "replacement_risk_delta",
        "replacement_reason",
        "emergency_fallback",
        "emergency_trigger",
        "emergency_reason",
        "raw_candidate_legal",
        "final_candidate_legal",
        "legal_candidate_count",
        "illegal_candidate_count",
        "emergency_saturated_count",
        "emergency_saturated_required",
        "raw_action_name",
        "final_action_name",
        "best_candidate_action_name",
        "step",
        "min_distance",
        "min_ttc",
        "max_drac",
        "geometric_overlap",
        "closest_vehicle_id",
    )
    return {key: record.get(key) for key in keys if key in record}


def _load_replay(replay_path: Path) -> dict[str, Any] | None:
    if not replay_path.exists():
        return None
    with replay_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _powershell_quote(value: str | Path) -> str:
    text = str(value).replace("`", "``").replace('"', '`"')
    return f'"{text}"'


def _step_trace_from_replay(replay: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not replay:
        return []
    for key in ("step_records", "step_safety_records", "safety_trace"):
        value = replay.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    notes = replay.get("notes", {})
    if isinstance(notes, dict):
        for key in ("step_records", "step_safety_records", "safety_trace"):
            value = notes.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    return []


def _first_failure_step(step_trace: list[dict[str, Any]], collision_threshold: float) -> int | None:
    for index, item in enumerate(step_trace):
        if (
            bool(item.get("proxy_collision", False))
            or bool(item.get("geometric_overlap", False))
            or bool(item.get("safety_violation", False))
        ):
            return int(item.get("step", index))
        if _safe_float(item.get("min_distance"), 1.0e9) <= collision_threshold:
            return int(item.get("step", index))
        if _safe_float(item.get("drac"), _safe_float(item.get("drac_raw"), 0.0)) >= 1.0e5:
            return int(item.get("step", index))
    return None


def _records_near_failure(records: list[dict[str, Any]], first_step: int | None, window: int = 6) -> list[dict[str, Any]]:
    if not records:
        return []
    if first_step is None:
        return [_compact_record(item) for item in records[-min(len(records), window):]]
    lower = max(0, first_step - window)
    upper = first_step + window
    selected = [
        record
        for index, record in enumerate(records)
        if lower <= int(record.get("step", record.get("control_step", index))) <= upper
    ]
    if not selected:
        selected = records[-min(len(records), window):]
    return [_compact_record(item) for item in selected]


def _compact_step_trace(item: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "step",
        "control_step",
        "raw_action",
        "final_action",
        "raw_action_name",
        "final_action_name",
        "trace_schema_version",
        "decision_step",
        "decision_ego_edge",
        "decision_ego_lane",
        "decision_distance_to_taper",
        "decision_target_front_gap",
        "decision_target_rear_gap",
        "decision_task_deadline_urgency",
        "post_action_step",
        "post_action_ego_edge",
        "post_action_ego_lane",
        "post_action_distance_to_taper",
        "post_action_target_front_gap",
        "post_action_target_rear_gap",
        "ego_edge",
        "ego_lane",
        "distance_to_taper",
        "target_front_gap",
        "target_rear_gap",
        "task_merge_opportunity",
        "task_would_merge",
        "task_missed_merge",
        "task_deadline_urgency",
        "forecast_aware_raw_task_risk",
        "forecast_aware_best_task_risk",
        "forecast_aware_raw_task_cost",
        "forecast_aware_best_task_cost",
        "forecast_aware_task_improvement",
        "forecast_aware_best_action",
        "forecast_aware_best_action_name",
        "forecast_aware_would_merge",
        "forecast_aware_safety_risk",
        "forecast_aware_uncertainty",
        "forecast_aware_target_front_gap",
        "forecast_aware_target_rear_gap",
        "forecast_first_step_target_front_gap",
        "forecast_first_step_target_rear_gap",
        "forecast_gap_consistency_pass",
        "forecast_selected_vehicle_ids",
        "forecast_target_front_vehicle_id",
        "forecast_target_rear_vehicle_id",
        "forecast_target_front_required",
        "forecast_target_rear_required",
        "forecast_target_front_covered",
        "forecast_target_rear_covered",
        "forecast_actor_coverage_complete",
        "forecast_closest_vehicle_id",
        "forecast_front_gap_vehicle_id",
        "forecast_rear_gap_vehicle_id",
        "task_backstop_watch_count",
        "task_backstop_watch_eligible",
        "task_backstop_eligible",
        "task_backstop_risk_module_score",
        "task_backstop_risk_module_uncertainty",
        "task_backstop_risk_module_pass",
        "task_backstop_veto_reason",
        "task_replacement",
        "task_replacement_reason",
        "taper_miss",
        "done_reason",
        "min_distance",
        "min_ttc",
        "drac",
    )
    return {key: item.get(key) for key in keys if key in item}


def _trace_alignment_summary(step_trace: list[dict[str, Any]], deadline_distance: float) -> dict[str, Any]:
    errors: list[float] = []
    for item in step_trace:
        distance = item.get("decision_distance_to_taper")
        urgency = item.get("decision_task_deadline_urgency")
        if distance is None or urgency is None:
            continue
        expected = float(
            min(
                max(
                    (float(deadline_distance) - float(distance)) / max(float(deadline_distance), 1.0e-6),
                    0.0,
                ),
                1.0,
            )
        )
        errors.append(abs(float(urgency) - expected))
    return {
        "available": bool(errors),
        "count": len(errors),
        "max_abs_error": max(errors) if errors else None,
        "passed": bool(errors) and max(errors) <= 1.0e-6,
    }


def _task_failure_step(step_trace: list[dict[str, Any]], episode: dict[str, Any]) -> int | None:
    for index, item in enumerate(step_trace):
        if bool(item.get("taper_miss", False)) or str(item.get("done_reason", "") or "") == "taper_miss":
            return int(item.get("step", index))
    if step_trace and _is_task_failure(episode):
        return int(step_trace[-1].get("step", len(step_trace) - 1))
    return None


def _step_trace_near_failure(
    step_trace: list[dict[str, Any]],
    first_step: int | None,
    *,
    window: int = 10,
) -> list[dict[str, Any]]:
    if not step_trace:
        return []
    if first_step is None:
        return [_compact_step_trace(item) for item in step_trace[-min(len(step_trace), window):]]
    lower = max(0, first_step - window)
    upper = first_step + 1
    selected = [
        item
        for index, item in enumerate(step_trace)
        if lower <= int(item.get("step", item.get("control_step", index))) <= upper
    ]
    if not selected:
        selected = step_trace[-min(len(step_trace), window):]
    return [_compact_step_trace(item) for item in selected]


def _classify_failure(
    episode: dict[str, Any],
    records: list[dict[str, Any]],
    first_failure_step: int | None,
    step_trace_available: bool,
) -> list[str]:
    labels: list[str] = []
    if not step_trace_available:
        labels.append("missing_step_trace")
    emergency_count = int(_safe_float(episode.get("emergency_fallback_count"), 0.0))
    replacement_count = int(_safe_float(episode.get("actual_replacement_count"), 0.0))
    if bool(episode.get("geometric_overlap", False)):
        labels.append("geometric_metric_overlap")
    if bool(episode.get("taper_miss", False)) or str(episode.get("done_reason", "") or "") == "taper_miss":
        labels.append("taper_miss")
    if episode.get("merge_success") is False:
        labels.append("merge_task_failure")
    if episode.get("first_merge_request_step") is None and (
        bool(episode.get("taper_miss", False)) or episode.get("merge_success") is False
    ):
        labels.append("no_merge_request_before_taper")
    if float(episode.get("missed_safe_merge_opportunity_rate", 0.0) or 0.0) > 0.0:
        labels.append("missed_safe_merge_opportunity")
    if (
        episode.get("first_merge_request_distance_to_taper") is not None
        and _safe_float(episode.get("first_merge_request_distance_to_taper"), 1.0e9) < 60.0
    ):
        labels.append("late_merge_request")
    if emergency_count > 0 and _safe_float(episode.get("min_distance"), 1.0e9) <= 0.0:
        labels.append("late_emergency")
    if records:
        saturated = [
            item
            for item in records
            if _safe_float(item.get("raw_risk", item.get("raw_risk_score")), 0.0) >= 0.99
            and _safe_float(item.get("best_candidate_risk", item.get("best_candidate_risk_score")), 0.0) >= 0.99
        ]
        if saturated and replacement_count == 0:
            labels.append("no_safe_candidate")
        if replacement_count > 0 and emergency_count > 0 and _safe_float(episode.get("min_distance"), 1.0e9) <= 0.0:
            labels.append("policy_entered_unrecoverable_state")
        if replacement_count == 0 and any(
            str(item.get("replacement_reason", "")) in {"raw_safe", "raw_tolerated", "fallback_disabled"}
            for item in records
        ):
            labels.append("ranker_or_margin_mismatch")
    if first_failure_step is not None and records and any(
        str(item.get("replacement_reason", "")) in {"raw_safe", "raw_tolerated"}
        for item in records[-min(len(records), 6):]
    ):
        labels.append("risk_missed_before_failure")
    if not labels:
        labels.append("unclassified")
    return labels


def _episode_summary(
    *,
    run_id: str,
    eval_stage: str,
    replay_dir: Path,
    group: str,
    episode: dict[str, Any],
    collision_threshold: float,
    deadline_distance: float,
) -> dict[str, Any]:
    seed = int(episode.get("seed", -1))
    replay_path = replay_dir / f"{group}_seed_{seed}.json"
    replay = _load_replay(replay_path)
    step_trace = _step_trace_from_replay(replay)
    first_step = _first_failure_step(step_trace, collision_threshold)
    task_step = _task_failure_step(step_trace, episode)
    trace_focus_step = first_step if first_step is not None else task_step
    records = episode.get("shield_score_records", []) or []
    records = [item for item in records if isinstance(item, dict)]
    replay_command = (
        f"& {_powershell_quote(sys.executable)} -m safe_rl.tools.replay_episode "
        f"--replay {_powershell_quote(replay_path)}"
    )
    classification = _classify_failure(episode, records, first_step, bool(step_trace))
    step_trace_near_failure = _step_trace_near_failure(step_trace, trace_focus_step)
    if _is_task_failure(episode) and any(
        str(item.get("forecast_aware_best_action_name", "")).startswith("left_")
        for item in step_trace_near_failure
    ):
        classification.append("forecast_merge_candidate_observed")
    if _is_task_failure(episode) and any(
        bool(item.get("task_backstop_eligible", False))
        for item in step_trace_near_failure
    ):
        classification.append("forecast_backstop_eligible")
    return {
        "run_id": run_id,
        "eval_stage": eval_stage,
        "group": group,
        "seed": seed,
        "replay_path": str(replay_path),
        "replay_exists": replay is not None,
        "replay_command": replay_command,
        "failure_classification": classification,
        "first_failure_step": first_step if first_step is not None else "unavailable",
        "step_trace_available": bool(step_trace),
        "trace_schema_version": int(replay.get("trace_schema_version", 1)) if replay else None,
        "trace_alignment_summary": _trace_alignment_summary(step_trace, deadline_distance),
        "metrics": {
            "episode_reward": _safe_float(episode.get("episode_reward"), 0.0),
            "merge_success": bool(episode.get("merge_success", False)),
            "done_reason": episode.get("done_reason"),
            "taper_miss": bool(episode.get("taper_miss", False)),
            "collision": bool(episode.get("collision", False)),
            "geometric_overlap": bool(episode.get("geometric_overlap", False)),
            "near_miss": bool(episode.get("near_miss", False)),
            "proxy_collision": bool(episode.get("proxy_collision", False)),
            "safety_violation": bool(episode.get("safety_violation", False)),
            "min_distance": _safe_float(episode.get("min_distance"), 1.0e9),
            "ttc_p1": _safe_float(episode.get("ttc_p1"), 0.0),
            "drac_p99_raw": _safe_float(episode.get("drac_p99_raw"), _safe_float(episode.get("drac_p99"), 0.0)),
            "actual_replacement_count": int(_safe_float(episode.get("actual_replacement_count"), 0.0)),
            "task_replacement_count": int(_safe_float(episode.get("task_replacement_count"), 0.0)),
            "emergency_fallback_count": int(_safe_float(episode.get("emergency_fallback_count"), 0.0)),
            "fallback_count": int(_safe_float(episode.get("fallback_count"), 0.0)),
            "steps": int(_safe_float(episode.get("steps"), 0.0)),
            "first_merge_request_step": episode.get("first_merge_request_step"),
            "first_merge_request_distance_to_taper": episode.get("first_merge_request_distance_to_taper"),
            "first_target_lane_entry_step": episode.get("first_target_lane_entry_step"),
            "first_target_lane_entry_distance_to_taper": episode.get("first_target_lane_entry_distance_to_taper"),
            "safe_merge_opportunity_count": int(_safe_float(episode.get("safe_merge_opportunity_count"), 0.0)),
            "missed_safe_merge_opportunity_count": int(
                _safe_float(episode.get("missed_safe_merge_opportunity_count"), 0.0)
            ),
            "missed_safe_merge_opportunity_rate": _safe_float(
                episode.get("missed_safe_merge_opportunity_rate"), 0.0
            ),
            "deadline_missed_safe_merge_rate": _safe_float(
                episode.get("deadline_missed_safe_merge_rate"), 0.0
            ),
            "missed_safe_merge_after_urgency_0_5_rate": _safe_float(
                episode.get("missed_safe_merge_after_urgency_0_5_rate"), 0.0
            ),
            "safe_merge_after_urgency_0_5_count": int(
                _safe_float(episode.get("safe_merge_after_urgency_0_5_count"), 0.0)
            ),
            "missed_safe_merge_after_urgency_0_5_count": int(
                _safe_float(episode.get("missed_safe_merge_after_urgency_0_5_count"), 0.0)
            ),
            "no_merge_request_before_taper_count": int(
                _safe_float(episode.get("no_merge_request_before_taper_count"), 0.0)
            ),
        },
        "replacement_reason_counts": episode.get("replacement_reason_counts", {}) or {},
        "raw_action_histogram": episode.get("raw_action_histogram", {}) or {},
        "final_action_histogram": episode.get("final_action_histogram", {}) or {},
        "shield_record_count": len(records),
        "shield_records_near_failure": _records_near_failure(records, first_step),
        "step_trace_near_failure": step_trace_near_failure,
    }


def build_failure_audit(
    run_id: str,
    *,
    groups: list[str] | tuple[str, ...] = DEFAULT_GROUPS,
    eval_stage: str = "stage5",
    collision_threshold: float = 0.25,
    include_task_failures: bool = False,
) -> dict[str, Any]:
    cfg = load_config()
    cfg.run["run_id"] = run_id
    base_dir = run_root(cfg) / eval_stage
    report_path = _report_path(base_dir)
    if not report_path.exists():
        raise FileNotFoundError(f"Stage5 report not found: {report_path}")
    with report_path.open("r", encoding="utf-8") as file:
        report = json.load(file)
    replay_dir = base_dir / "replay"
    deadline_distance = float(cfg.shield.get("task_backstop_deadline_distance", 120.0))
    failures: dict[str, list[dict[str, Any]]] = {}
    safety_failures: dict[str, list[dict[str, Any]]] = {}
    task_failures: dict[str, list[dict[str, Any]]] = {}
    for group in groups:
        group_report = report.get("groups", {}).get(group)
        if not group_report:
            failures[group] = []
            safety_failures[group] = []
            task_failures[group] = []
            continue
        episodes = group_report.get("episodes", []) or []
        safety_failures[group] = [
            _episode_summary(
                run_id=run_id,
                eval_stage=eval_stage,
                replay_dir=replay_dir,
                group=group,
                episode=episode,
                collision_threshold=collision_threshold,
                deadline_distance=deadline_distance,
            )
            for episode in episodes
            if isinstance(episode, dict) and _is_failure(episode, collision_threshold)
        ]
        task_failures[group] = [
            _episode_summary(
                run_id=run_id,
                eval_stage=eval_stage,
                replay_dir=replay_dir,
                group=group,
                episode=episode,
                collision_threshold=collision_threshold,
                deadline_distance=deadline_distance,
            )
            for episode in episodes
            if isinstance(episode, dict) and _is_task_failure(episode)
        ]
        if include_task_failures:
            by_seed = {int(item["seed"]): item for item in safety_failures[group]}
            for item in task_failures[group]:
                by_seed.setdefault(int(item["seed"]), item)
            failures[group] = list(by_seed.values())
        else:
            failures[group] = list(safety_failures[group])
    classification_counts: dict[str, int] = {}
    for items in failures.values():
        for item in items:
            for label in item["failure_classification"]:
                classification_counts[label] = classification_counts.get(label, 0) + 1
    return {
        "run_id": run_id,
        "eval_stage": eval_stage,
        "report_path": str(report_path),
        "replay_dir": str(replay_dir),
        "groups": list(groups),
        "collision_threshold": float(collision_threshold),
        "include_task_failures": bool(include_task_failures),
        "safety_failure_counts": {group: len(items) for group, items in safety_failures.items()},
        "task_failure_counts": {group: len(items) for group, items in task_failures.items()},
        "failure_counts": {group: len(items) for group, items in failures.items()},
        "classification_counts": classification_counts,
        "safety_failures": safety_failures,
        "task_failures": task_failures,
        "failures": failures,
    }


def write_replay_commands(path: str | Path, audit: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Auto-generated by safe_rl.pipeline.stage5_failure_audit",
        "# Review the listed failure seeds in SUMO GUI.",
        "",
    ]
    for group, items in audit.get("failures", {}).items():
        lines.append(f"# {group}")
        for item in items:
            lines.append(str(item.get("replay_command", "")) + " --gui --delay-ms 200")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def run(
    run_id: str,
    groups: list[str],
    eval_stage: str = "stage5",
    collision_threshold: float = 0.25,
    include_task_failures: bool = False,
) -> Path:
    cfg = load_config()
    cfg.run["run_id"] = run_id
    stage_dir = run_root(cfg) / eval_stage / "failure_audit"
    audit = build_failure_audit(
        run_id,
        groups=groups,
        eval_stage=eval_stage,
        collision_threshold=collision_threshold,
        include_task_failures=include_task_failures,
    )
    report_path = stage_dir / "failure_audit_report.json"
    commands_path = stage_dir / "failure_replay_commands.ps1"
    write_report(report_path, audit)
    write_replay_commands(commands_path, audit)
    stage_log("stage5_failure_audit", f"report={report_path}")
    stage_log("stage5_failure_audit", f"replay_commands={commands_path}")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Stage5 failure seeds and Shield records.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--groups",
        default=",".join(DEFAULT_GROUPS),
        help="Comma-separated Stage5 group names to audit.",
    )
    parser.add_argument("--eval-stage", default="stage5", help="Evaluation stage directory, e.g. stage5 or stage5_confirmatory.")
    parser.add_argument("--collision-threshold", type=float, default=0.25)
    parser.add_argument(
        "--include-task-failures",
        action="store_true",
        help="Also audit merge task failures such as taper_miss or merge_success=false.",
    )
    args = parser.parse_args()
    groups = [item.strip() for item in str(args.groups).split(",") if item.strip()]
    run(
        args.run_id,
        groups=groups,
        eval_stage=args.eval_stage,
        collision_threshold=float(args.collision_threshold),
        include_task_failures=bool(args.include_task_failures),
    )


if __name__ == "__main__":
    main()
