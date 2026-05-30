from __future__ import annotations

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
    fallbacks = np.asarray([float(report.get("fallback_count", 0)) for report in reports], dtype=np.float32)
    emergency_fallbacks = np.asarray(
        [float(report.get("emergency_fallback_count", 0)) for report in reports],
        dtype=np.float32,
    )
    taper_misses = np.asarray([float(bool(report.get("taper_miss", False))) for report in reports], dtype=np.float32)
    return {
        "episodes": len(reports),
        "collision_rate": float(np.mean(collisions)),
        "near_miss_rate": float(np.mean(near_misses)),
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
        "fallback_rate": float(np.mean(fallbacks > 0)),
        "emergency_fallback_rate": float(np.mean(emergency_fallbacks > 0)),
        "mean_emergency_fallbacks": float(np.mean(emergency_fallbacks)),
        "emergency_fallback_count": int(np.sum(emergency_fallbacks)),
        "taper_miss_rate": float(np.mean(taper_misses)) if taper_misses.size else 0.0,
        "taper_miss_count": int(np.sum(taper_misses)),
    }
