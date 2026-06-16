from __future__ import annotations

from collections import Counter

import numpy as np


DRAC_REPORT_CAP_DEFAULT = 20.0


def _report_safety_violation(report: dict) -> float:
    if "safety_violation" in report:
        return float(bool(report.get("safety_violation", False)))
    return float(
        bool(report.get("collision", False))
        or bool(report.get("proxy_collision", False))
        or bool(report.get("near_miss", False))
        or float(report.get("ttc_p1", 1.0e6)) < 0.3
    )


def _report_proxy_collision(report: dict) -> float:
    if "proxy_collision" in report:
        return float(bool(report.get("proxy_collision", False)))
    return float(float(report.get("min_distance", 1.0e6)) <= 0.25)


def aggregate_episode_reports(reports: list[dict]) -> dict:
    if not reports:
        return {}
    collisions = np.asarray([float(report.get("collision", False)) for report in reports], dtype=np.float32)
    near_misses = np.asarray([float(report.get("near_miss", False)) for report in reports], dtype=np.float32)
    proxy_collisions = np.asarray([_report_proxy_collision(report) for report in reports], dtype=np.float32)
    safety_violations = np.asarray([_report_safety_violation(report) for report in reports], dtype=np.float32)
    proxy_collision_counts = np.asarray(
        [float(report.get("proxy_collision_count", _report_proxy_collision(report))) for report in reports],
        dtype=np.float32,
    )
    safety_violation_counts = np.asarray(
        [float(report.get("safety_violation_count", _report_safety_violation(report))) for report in reports],
        dtype=np.float32,
    )
    min_distance_collision_counts = np.asarray(
        [
            float(
                report.get(
                    "min_distance_le_collision_threshold_count",
                    report.get("proxy_collision_count", _report_proxy_collision(report)),
                )
            )
            for report in reports
        ],
        dtype=np.float32,
    )
    min_distances = np.asarray([float(report.get("min_distance", 0.0)) for report in reports], dtype=np.float32)
    ttc = np.asarray([float(report.get("ttc_p1", 1.0e6)) for report in reports], dtype=np.float32)
    drac_raw = np.asarray(
        [float(report.get("drac_p99_raw", report.get("drac_p99", 0.0))) for report in reports],
        dtype=np.float32,
    )
    drac_capped = np.asarray(
        [
            float(report.get("drac_p99_capped", min(float(report.get("drac_p99", 0.0)), DRAC_REPORT_CAP_DEFAULT)))
            for report in reports
        ],
        dtype=np.float32,
    )
    steps = np.asarray([float(report.get("steps", 0.0)) for report in reports], dtype=np.float32)
    completion_time = np.asarray([float(report.get("completion_time", 0.0)) for report in reports], dtype=np.float32)
    ego_speed_mean = np.asarray([float(report.get("ego_speed_mean", 0.0)) for report in reports], dtype=np.float32)
    ego_speed_p10 = np.asarray([float(report.get("ego_speed_p10", 0.0)) for report in reports], dtype=np.float32)
    hard_brake_rates = np.asarray([float(report.get("hard_brake_rate", 0.0)) for report in reports], dtype=np.float32)
    interventions = np.asarray([float(report.get("intervention_count", 0)) for report in reports], dtype=np.float32)
    shield_calls = np.asarray([float(report.get("shield_call_count", report.get("intervention_count", 0))) for report in reports], dtype=np.float32)
    replacements = np.asarray([float(report.get("actual_replacement_count", 0)) for report in reports], dtype=np.float32)
    task_replacements = np.asarray(
        [float(report.get("task_replacement_count", 0)) for report in reports],
        dtype=np.float32,
    )
    fallbacks = np.asarray([float(report.get("fallback_count", 0)) for report in reports], dtype=np.float32)
    emergency_fallbacks = np.asarray(
        [float(report.get("emergency_fallback_count", 0)) for report in reports],
        dtype=np.float32,
    )
    taper_misses = np.asarray([float(bool(report.get("taper_miss", False))) for report in reports], dtype=np.float32)
    geometric_overlaps = np.asarray(
        [float(bool(report.get("geometric_overlap", False))) for report in reports],
        dtype=np.float32,
    )
    first_merge_request_distance = np.asarray(
        [
            float(report["first_merge_request_distance_to_taper"])
            for report in reports
            if report.get("first_merge_request_distance_to_taper") is not None
        ],
        dtype=np.float32,
    )
    first_target_entry_distance = np.asarray(
        [
            float(report["first_target_lane_entry_distance_to_taper"])
            for report in reports
            if report.get("first_target_lane_entry_distance_to_taper") is not None
        ],
        dtype=np.float32,
    )
    safe_merge_opportunities = int(sum(int(report.get("safe_merge_opportunity_count", 0)) for report in reports))
    missed_safe_merge_opportunities = int(
        sum(int(report.get("missed_safe_merge_opportunity_count", 0)) for report in reports)
    )
    task_merge_opportunities = int(sum(int(report.get("task_merge_opportunity_count", 0)) for report in reports))
    task_would_merges = int(sum(int(report.get("task_would_merge_count", 0)) for report in reports))
    task_missed_merges = int(sum(int(report.get("task_missed_merge_count", 0)) for report in reports))
    deadline_opportunities = int(
        sum(int(report.get("deadline_safe_merge_opportunity_count", 0)) for report in reports)
    )
    deadline_missed = int(sum(int(report.get("deadline_missed_safe_merge_count", 0)) for report in reports))
    urgency_missed = int(
        sum(int(report.get("missed_safe_merge_after_urgency_0_5_count", 0)) for report in reports)
    )
    urgency_opportunities = int(
        sum(int(report.get("safe_merge_after_urgency_0_5_count", 0)) for report in reports)
    )
    no_merge_before_taper = np.asarray(
        [float(report.get("no_merge_request_before_taper_count", 0)) for report in reports],
        dtype=np.float32,
    )
    forecast_record_count = int(
        sum(int(report.get("forecast_record_count", 0)) for report in reports)
    )
    forecast_coverage_complete_count = int(
        sum(int(report.get("forecast_actor_coverage_complete_count", 0)) for report in reports)
    )
    forecast_gap_checkable_count = int(
        sum(
            int(report.get("forecast_gap_consistency_checkable_count", 0))
            for report in reports
        )
    )
    forecast_gap_pass_count = int(
        sum(
            int(report.get("forecast_gap_consistency_pass_count", 0))
            for report in reports
        )
    )
    wcdt_relevant_coverage_count = int(
        sum(int(report.get("wcdt_relevant_actor_coverage_count", 0)) for report in reports)
    )
    combined_safety_coverage_count = int(
        sum(int(report.get("combined_forecast_safety_coverage_count", 0)) for report in reports)
    )
    selector_overflow_count = int(
        sum(int(report.get("actor_selector_overflow_count", 0)) for report in reports)
    )
    critical_overflow_count = int(
        sum(int(report.get("critical_actor_overflow_count", report.get("actor_selector_overflow_count", 0))) for report in reports)
    )
    critical_wcdt_coverage_count = int(
        sum(int(report.get("critical_wcdt_coverage_count", 0)) for report in reports)
    )
    combined_critical_coverage_count = int(
        sum(int(report.get("combined_critical_coverage_count", 0)) for report in reports)
    )
    cv_fallback_overflow_count = int(
        sum(int(report.get("cv_fallback_overflow_count", 0)) for report in reports)
    )
    cv_fallback_usage_count = int(
        sum(int(report.get("cv_fallback_usage_count", 0)) for report in reports)
    )
    reward_component_names = (
        "progress_reward",
        "speed_reward",
        "terminal_reward",
        "lane_oob_penalty",
        "safety_penalty",
        "safety_forecast_shaping",
        "shield_guided_shaping",
        "merge_timing_shaping",
        "total_episode_reward",
    )
    reward_component_means = {
        name: float(np.mean([float(report.get(name, 0.0)) for report in reports]))
        for name in reward_component_names
    }
    raw_lane_oob_count = int(sum(int(report.get("raw_action_lane_oob_count", 0)) for report in reports))
    final_lane_oob_count = int(sum(int(report.get("final_action_lane_oob_count", 0)) for report in reports))
    prevented_lane_oob_count = int(sum(int(report.get("prevented_lane_oob_count", 0)) for report in reports))
    task_backstop_watch_count = int(sum(int(report.get("task_backstop_watch_count", 0)) for report in reports))
    task_backstop_eligible_count = int(
        sum(int(report.get("task_backstop_eligible_count", 0)) for report in reports)
    )
    task_backstop_veto_reason_counts: Counter[str] = Counter()
    for report in reports:
        task_backstop_veto_reason_counts.update(report.get("task_backstop_veto_reason_counts", {}) or {})
    return {
        "episodes": len(reports),
        "collision_rate": float(np.mean(collisions)),
        "near_miss_rate": float(np.mean(near_misses)),
        "geometric_overlap_rate": float(np.mean(geometric_overlaps)),
        "geometric_overlap_count": int(np.sum(geometric_overlaps)),
        "proxy_collision_rate": float(np.mean(proxy_collisions)),
        "safety_violation_rate": float(np.mean(safety_violations)),
        "proxy_collision_count": int(np.sum(proxy_collision_counts)),
        "safety_violation_count": int(np.sum(safety_violation_counts)),
        "min_distance_le_collision_threshold_count": int(np.sum(min_distance_collision_counts)),
        "min_distance_p1": float(np.percentile(min_distances, 1)),
        "ttc_p1": float(np.percentile(ttc, 1)),
        "drac_p99": float(np.percentile(drac_raw, 99)),
        "drac_p99_raw": float(np.percentile(drac_raw, 99)),
        "drac_p99_capped": float(np.percentile(drac_capped, 99)),
        "steps_mean": float(np.mean(steps)),
        "steps_p95": float(np.percentile(steps, 95)),
        "completion_time_mean": float(np.mean(completion_time)),
        "completion_time_p95": float(np.percentile(completion_time, 95)),
        "ego_speed_mean": float(np.mean(ego_speed_mean)),
        "ego_speed_p10": float(np.percentile(ego_speed_p10, 10)),
        "hard_brake_rate": float(np.mean(hard_brake_rates)),
        "intervention_rate": float(np.mean(interventions > 0)),
        "shield_call_rate": float(np.mean(shield_calls > 0)),
        "mean_shield_calls": float(np.mean(shield_calls)),
        "actual_replacement_rate": float(np.mean(replacements > 0)),
        "mean_actual_replacements": float(np.mean(replacements)),
        "actual_replacement_rate_semantics": "episodes_with_replacement_rate",
        "episodes_with_replacement_rate": float(np.mean(replacements > 0)),
        "replacement_per_shield_call_rate": (
            float(np.sum(replacements) / np.sum(shield_calls))
            if np.sum(shield_calls) > 0
            else 0.0
        ),
        "mean_replacements_per_episode": float(np.mean(replacements)),
        "task_replacement_rate": float(np.mean(task_replacements > 0)),
        "mean_task_replacements": float(np.mean(task_replacements)),
        "task_replacement_count": int(np.sum(task_replacements)),
        "fallback_rate": float(np.mean(fallbacks > 0)),
        "emergency_fallback_rate": float(np.mean(emergency_fallbacks > 0)),
        "mean_emergency_fallbacks": float(np.mean(emergency_fallbacks)),
        "emergency_fallback_count": int(np.sum(emergency_fallbacks)),
        "taper_miss_rate": float(np.mean(taper_misses)) if taper_misses.size else 0.0,
        "taper_miss_count": int(np.sum(taper_misses)),
        "first_merge_request_distance_to_taper_mean": (
            float(np.mean(first_merge_request_distance)) if first_merge_request_distance.size else None
        ),
        "first_merge_request_distance_to_taper_p50": (
            float(np.percentile(first_merge_request_distance, 50)) if first_merge_request_distance.size else None
        ),
        "first_target_lane_entry_distance_to_taper_mean": (
            float(np.mean(first_target_entry_distance)) if first_target_entry_distance.size else None
        ),
        "first_target_lane_entry_distance_to_taper_p50": (
            float(np.percentile(first_target_entry_distance, 50)) if first_target_entry_distance.size else None
        ),
        "safe_merge_opportunity_count": safe_merge_opportunities,
        "missed_safe_merge_opportunity_count": missed_safe_merge_opportunities,
        "missed_safe_merge_opportunity_rate": (
            float(missed_safe_merge_opportunities / safe_merge_opportunities)
            if safe_merge_opportunities
            else 0.0
        ),
        "task_merge_opportunity_count": task_merge_opportunities,
        "task_would_merge_count": task_would_merges,
        "task_would_merge_rate": (
            float(task_would_merges / task_merge_opportunities) if task_merge_opportunities else 0.0
        ),
        "task_missed_merge_count": task_missed_merges,
        "task_missed_merge_rate": (
            float(task_missed_merges / task_merge_opportunities) if task_merge_opportunities else 0.0
        ),
        "deadline_safe_merge_opportunity_count": deadline_opportunities,
        "deadline_missed_safe_merge_count": deadline_missed,
        "deadline_missed_safe_merge_rate": (
            float(deadline_missed / deadline_opportunities) if deadline_opportunities else 0.0
        ),
        "missed_safe_merge_after_urgency_0_5_count": urgency_missed,
        "safe_merge_after_urgency_0_5_count": urgency_opportunities,
        "missed_safe_merge_after_urgency_0_5_rate": (
            float(urgency_missed / urgency_opportunities) if urgency_opportunities else 0.0
        ),
        "no_merge_request_before_taper_count": int(np.sum(no_merge_before_taper)),
        "no_merge_request_before_taper_rate": float(np.mean(no_merge_before_taper > 0)),
        "forecast_actor_coverage_complete_count": forecast_coverage_complete_count,
        "forecast_actor_coverage_complete_rate": (
            float(forecast_coverage_complete_count / forecast_record_count)
            if forecast_record_count
            else 0.0
        ),
        "forecast_record_count": forecast_record_count,
        "forecast_gap_consistency_checkable_count": forecast_gap_checkable_count,
        "forecast_gap_consistency_pass_count": forecast_gap_pass_count,
        "forecast_gap_consistency_checkable_rate": (
            float(forecast_gap_checkable_count / forecast_record_count)
            if forecast_record_count
            else 0.0
        ),
        "forecast_gap_consistency_pass_rate": (
            float(forecast_gap_pass_count / forecast_gap_checkable_count)
            if forecast_gap_checkable_count
            else 0.0
        ),
        "wcdt_relevant_actor_coverage_count": wcdt_relevant_coverage_count,
        "wcdt_relevant_actor_coverage_rate": (
            float(wcdt_relevant_coverage_count / forecast_record_count)
            if forecast_record_count
            else 0.0
        ),
        "combined_forecast_safety_coverage_count": combined_safety_coverage_count,
        "combined_forecast_safety_coverage_rate": (
            float(combined_safety_coverage_count / forecast_record_count)
            if forecast_record_count
            else 0.0
        ),
        "actor_selector_overflow_count": selector_overflow_count,
        "actor_selector_overflow_rate": (
            float(selector_overflow_count / forecast_record_count)
            if forecast_record_count
            else 0.0
        ),
        "critical_actor_overflow_count": critical_overflow_count,
        "critical_actor_overflow_rate": (
            float(critical_overflow_count / forecast_record_count)
            if forecast_record_count
            else 0.0
        ),
        "critical_wcdt_coverage_count": critical_wcdt_coverage_count,
        "critical_wcdt_coverage_rate": (
            float(critical_wcdt_coverage_count / forecast_record_count)
            if forecast_record_count
            else 0.0
        ),
        "combined_critical_coverage_count": combined_critical_coverage_count,
        "combined_critical_coverage_rate": (
            float(combined_critical_coverage_count / forecast_record_count)
            if forecast_record_count
            else 0.0
        ),
        "cv_fallback_overflow_count": cv_fallback_overflow_count,
        "cv_fallback_overflow_rate": (
            float(cv_fallback_overflow_count / forecast_record_count)
            if forecast_record_count
            else 0.0
        ),
        "cv_fallback_usage_count": cv_fallback_usage_count,
        "cv_fallback_usage_rate": (
            float(cv_fallback_usage_count / forecast_record_count)
            if forecast_record_count
            else 0.0
        ),
        "raw_action_lane_oob_count": raw_lane_oob_count,
        "final_action_lane_oob_count": final_lane_oob_count,
        "prevented_lane_oob_count": prevented_lane_oob_count,
        **reward_component_means,
        "task_backstop_watch_count": task_backstop_watch_count,
        "task_backstop_eligible_count": task_backstop_eligible_count,
        "task_backstop_veto_reason_counts": dict(task_backstop_veto_reason_counts),
    }
