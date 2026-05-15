from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

from safe_rl.utils.io import write_json


RISK_TYPE_NAMES = ["collision", "near_miss", "low_ttc", "high_drac", "merge_conflict"]


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


def audit_stage1_buffer(buffer_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    buffer_path = Path(buffer_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data = np.load(buffer_path, allow_pickle=False)

    actions = data["actions"].astype(np.int64)
    rewards = data["rewards"].astype(np.float32)
    risk = data["overall_risk"].astype(np.float32)
    risk_types = data["risk_types"].astype(np.float32)
    episode_ids = data["episode_id"].astype(np.int64)
    risk_features = data["risk_features"].astype(np.float32)

    action_hist = {str(idx): int(np.sum(actions == idx)) for idx in range(9)}
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
        "transition_count": int(actions.shape[0]),
        "episode_count": int(len(set(episode_ids.tolist()))) if episode_ids.size else 0,
        "observation_shape": list(data["observations"].shape),
        "action_histogram": action_hist,
        "episode_transition_counts": episode_transition_counts,
        "overall_risk_rate": float(np.mean(risk)) if risk.size else 0.0,
        "risk_type_rates": risk_type_rates,
        "reward": _quantiles(rewards),
        "risk_features": feature_stats,
        "trajectory_sample_count": int(data["agent_history"].shape[0]) if "agent_history" in data else 0,
    }

    write_json(output_dir / "stage1_data_audit.json", report)
    with (output_dir / "stage1_action_histogram.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["action", "count"])
        for action, count in action_hist.items():
            writer.writerow([action, count])

    _try_write_plots(output_dir, actions, risk, rewards)
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
