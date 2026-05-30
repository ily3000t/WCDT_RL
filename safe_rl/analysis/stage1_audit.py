from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

from safe_rl.utils.io import write_json


RISK_TYPE_NAMES = ["collision", "near_miss", "low_ttc", "high_drac", "merge_conflict", "taper_miss"]


def _quantiles(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {}
    return {
        "min": float(np.min(values)),
        "p01": float(np.percentile(values, 1)),
        "p05": float(np.percentile(values, 5)),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
    }


def _risk_label_arrays(data: np.lib.npyio.NpzFile, risk_features: np.ndarray, risk_types: np.ndarray) -> dict[str, np.ndarray]:
    risk = data["overall_risk"].astype(np.float32)
    if "traffic_risk" in data:
        traffic_risk = data["traffic_risk"].astype(np.float32)
    elif risk_types.ndim == 2 and risk_types.shape[1] > 0:
        traffic_risk = np.max(risk_types, axis=1).astype(np.float32)
    else:
        traffic_risk = risk
    if "lane_oob_risk" in data:
        lane_oob = data["lane_oob_risk"].astype(np.float32)
    elif risk_features.ndim == 2 and risk_features.shape[1] > 5:
        lane_oob = (risk_features[:, 5] > 0.5).astype(np.float32)
    else:
        lane_oob = np.zeros_like(traffic_risk, dtype=np.float32)
    if "candidate_legal" in data:
        candidate_legal = data["candidate_legal"].astype(np.float32) > 0.5
    else:
        candidate_legal = lane_oob <= 0.5
    return {
        "risk": risk,
        "traffic_risk": traffic_risk,
        "lane_oob": lane_oob,
        "candidate_legal": candidate_legal,
    }


def audit_stage1_buffer(buffer_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    buffer_path = Path(buffer_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data = np.load(buffer_path, allow_pickle=False)

    actions = data["actions"].astype(np.int64)
    rewards = data["rewards"].astype(np.float32)
    risk_types = data["risk_types"].astype(np.float32)
    episode_ids = data["episode_id"].astype(np.int64)
    risk_features = data["risk_features"].astype(np.float32)
    label_arrays = _risk_label_arrays(data, risk_features, risk_types)
    risk = label_arrays["traffic_risk"]
    lane_oob = label_arrays["lane_oob"]
    candidate_legal = label_arrays["candidate_legal"].astype(bool)
    legal_risk = risk[candidate_legal]
    continuous_risk = (
        data["continuous_risk_target"].astype(np.float32)
        if "continuous_risk_target" in data
        else risk.astype(np.float32)
    )
    legal_continuous = continuous_risk[candidate_legal]
    boundary = (legal_continuous >= 0.20) & (legal_continuous < 0.80)

    action_hist = {str(idx): int(np.sum(actions == idx)) for idx in range(9)}
    per_action_risk_rate = {
        str(idx): float(np.mean(risk[actions == idx])) if np.any(actions == idx) else 0.0
        for idx in range(9)
    }
    lane_oob_by_action = {
        str(idx): float(np.mean(lane_oob[actions == idx])) if np.any(actions == idx) else 0.0
        for idx in range(9)
    }
    legal_candidate_action_risk_rate = {
        str(idx): (
            float(np.mean(risk[(actions == idx) & candidate_legal]))
            if np.any((actions == idx) & candidate_legal)
            else 0.0
        )
        for idx in range(9)
    }
    episode_transition_counts = {
        str(int(episode)): int(np.sum(episode_ids == episode)) for episode in sorted(set(episode_ids.tolist()))
    }
    risk_type_rates = {
        name: float(np.mean(risk_types[:, idx])) if risk_types.size else 0.0
        for idx, name in enumerate(RISK_TYPE_NAMES[: risk_types.shape[1] if risk_types.ndim == 2 else 0])
    }
    feature_stats = {
        f"feature_{idx}": _quantiles(risk_features[:, idx])
        for idx in range(risk_features.shape[1] if risk_features.ndim == 2 else 0)
    }
    report = {
        "buffer": str(buffer_path),
        "transition_count": int(data["executed_actions"].shape[0]) if "executed_actions" in data else int(actions.shape[0]),
        "candidate_risk_sample_count": int(actions.shape[0]),
        "episode_count": int(len(set(episode_ids.tolist()))) if episode_ids.size else 0,
        "observation_shape": list(data["observations"].shape),
        "action_histogram": action_hist,
        "candidate_action_histogram": action_hist,
        "per_action_risk_rate": per_action_risk_rate,
        "traffic_risk_by_action": per_action_risk_rate,
        "lane_oob_by_action": lane_oob_by_action,
        "legal_candidate_action_risk_rate": legal_candidate_action_risk_rate,
        "episode_transition_counts": episode_transition_counts,
        "overall_risk_semantics": "traffic_risk_only",
        "overall_risk_rate": float(np.mean(risk)) if risk.size else 0.0,
        "traffic_risk_rate": float(np.mean(risk)) if risk.size else 0.0,
        "lane_oob_risk_rate": float(np.mean(lane_oob)) if lane_oob.size else 0.0,
        "illegal_candidate_rate": float(np.mean(~candidate_legal)) if candidate_legal.size else 0.0,
        "legal_candidate_risk_rate": float(np.mean(legal_risk)) if legal_risk.size else 0.0,
        "risk_type_rates": risk_type_rates,
        "reward": _quantiles(rewards),
        "risk_features": feature_stats,
        "trajectory_sample_count": int(data["agent_history"].shape[0]) if "agent_history" in data else 0,
        "continuous_risk": {
            "summary": _quantiles(continuous_risk),
            "legal_summary": _quantiles(legal_continuous),
            "easy_safe_rate": float(np.mean(legal_continuous < 0.20)) if legal_continuous.size else 0.0,
            "boundary_rate": float(np.mean(boundary)) if legal_continuous.size else 0.0,
            "extreme_risk_rate": float(np.mean(legal_continuous >= 0.80)) if legal_continuous.size else 0.0,
            "boundary_sample_count": int(np.sum(boundary)),
        },
    }
    if "executed_actions" in data:
        executed_actions = data["executed_actions"].astype(np.int64)
        report["executed_action_histogram"] = {
            str(idx): int(np.sum(executed_actions == idx)) for idx in range(9)
        }
    if "sampling_modes" in data:
        modes = data["sampling_modes"].astype(str)
        report["action_sampling"] = {
            "counts": {mode: int(np.sum(modes == mode)) for mode in sorted(set(modes.tolist()))},
            "proportions": {
                mode: float(np.mean(modes == mode)) for mode in sorted(set(modes.tolist()))
            },
        }
    if "target_lane_gap" in data:
        report["target_lane_gap"] = _quantiles(data["target_lane_gap"].astype(np.float32))
    if "candidate_target_lane_gap" in data:
        report["candidate_target_lane_gap"] = _quantiles(data["candidate_target_lane_gap"].astype(np.float32))
    if "ramp_local_risk" in data:
        report["ramp_local_risk_rate"] = float(np.mean(data["ramp_local_risk"].astype(np.float32)))
    if "merge_zone_risk" in data:
        report["merge_zone_risk_rate"] = float(np.mean(data["merge_zone_risk"].astype(np.float32)))
    if "candidate_ramp_local_risk" in data:
        report["candidate_ramp_local_risk_rate"] = float(np.mean(data["candidate_ramp_local_risk"].astype(np.float32)))
    if "candidate_merge_zone_risk" in data:
        report["candidate_merge_zone_risk_rate"] = float(np.mean(data["candidate_merge_zone_risk"].astype(np.float32)))
    if "candidate_taper_miss" in data:
        report["candidate_taper_miss_rate"] = float(np.mean(data["candidate_taper_miss"].astype(np.float32)))
    if "distance_to_taper" in data:
        report["distance_to_taper"] = _quantiles(data["distance_to_taper"].astype(np.float32))
    if "curriculum_profiles" in data:
        profiles = data["curriculum_profiles"].astype(str)
        report["curriculum_profile_counts"] = {
            profile: int(np.sum(profiles == profile)) for profile in sorted(set(profiles.tolist()))
        }

    write_json(output_dir / "stage1_data_audit.json", report)
    with (output_dir / "stage1_action_histogram.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["action", "count"])
        for action, count in action_hist.items():
            writer.writerow([action, count])

    _try_write_plots(output_dir, actions, continuous_risk, rewards)
    return report


def _try_write_plots(output_dir: Path, actions: np.ndarray, risk: np.ndarray, rewards: np.ndarray) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    fig, axis = plt.subplots(figsize=(7, 4))
    axis.bar(range(9), [int(np.sum(actions == idx)) for idx in range(9)])
    axis.set_xlabel("action index")
    axis.set_ylabel("count")
    axis.set_title("Stage1 action histogram")
    fig.tight_layout()
    fig.savefig(output_dir / "stage1_action_histogram.png")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7, 4))
    axis.hist(rewards, bins=30)
    axis.set_xlabel("reward")
    axis.set_ylabel("count")
    axis.set_title("Stage1 reward distribution")
    fig.tight_layout()
    fig.savefig(output_dir / "stage1_reward_distribution.png")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7, 4))
    axis.hist(risk, bins=2)
    axis.set_xlabel("overall risk label")
    axis.set_ylabel("count")
    axis.set_title("Stage1 risk label distribution")
    fig.tight_layout()
    fig.savefig(output_dir / "stage1_risk_distribution.png")
    plt.close(fig)
