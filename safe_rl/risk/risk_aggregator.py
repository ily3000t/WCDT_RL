from __future__ import annotations

import numpy as np


def aggregate_episode_reports(reports: list[dict]) -> dict:
    if not reports:
        return {}
    collisions = np.asarray([float(report.get("collision", False)) for report in reports], dtype=np.float32)
    near_misses = np.asarray([float(report.get("near_miss", False)) for report in reports], dtype=np.float32)
    min_distances = np.asarray([float(report.get("min_distance", 0.0)) for report in reports], dtype=np.float32)
    ttc = np.asarray([float(report.get("ttc_p1", 1.0e6)) for report in reports], dtype=np.float32)
    drac = np.asarray([float(report.get("drac_p99", 0.0)) for report in reports], dtype=np.float32)
    interventions = np.asarray([float(report.get("intervention_count", 0)) for report in reports], dtype=np.float32)
    fallbacks = np.asarray([float(report.get("fallback_count", 0)) for report in reports], dtype=np.float32)
    return {
        "episodes": len(reports),
        "collision_rate": float(np.mean(collisions)),
        "near_miss_rate": float(np.mean(near_misses)),
        "min_distance_p1": float(np.percentile(min_distances, 1)),
        "ttc_p1": float(np.percentile(ttc, 1)),
        "drac_p99": float(np.percentile(drac, 99)),
        "intervention_rate": float(np.mean(interventions > 0)),
        "fallback_rate": float(np.mean(fallbacks > 0)),
    }
