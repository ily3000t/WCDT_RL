from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from safe_rl.accvp.observation import RiskGatedACCVPCandidateTableAugmentor
from safe_rl.pipeline.common import make_env
from safe_rl.risk.risk_aggregator import aggregate_episode_reports
from safe_rl.rl.evaluation import validate_model_env_observation_shape
from safe_rl.rl.ppo import load_ppo
from safe_rl.utils.config import REPO_ROOT, load_config
from safe_rl.utils.io import write_json


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Preflight Risk-gated ACCVP candidate-table observation availability")
    parser.add_argument("--config", required=True, help="Config with accvp.observation.enabled=true")
    parser.add_argument("--policy-model", required=True, help="PPO model used only to visit realistic states")
    parser.add_argument("--seeds", nargs="+", type=int, required=True, help="Episode seeds")
    parser.add_argument("--output", required=True, help="Output JSON report")
    parser.add_argument("--device", default="auto", help="SB3 device")
    return parser.parse_args()


def _gate(metrics: dict[str, Any]) -> dict[str, Any]:
    hard_fail_closed = int(
        metrics.get(
            "accvp_table_hard_fail_closed_count",
            metrics.get("accvp_table_fail_closed_count", 0),
        )
    )
    checks = {
        "valid_rate_activation_window": float(metrics.get("accvp_table_valid_rate_activation_window", 0.0)) >= 0.95,
        "hard_fail_closed_count_zero": hard_fail_closed == 0,
        "latency_p95_within_0_5s": (
            metrics.get("accvp_table_latency_p95") is not None
            and float(metrics.get("accvp_table_latency_p95", 1.0e9)) <= 0.5
        ),
    }
    return {
        "pass": bool(all(checks.values())),
        "checks": checks,
        "required": {
            "valid_rate_activation_window": ">= 0.95",
            "hard_fail_closed_count": "0",
            "latency_p95_s": "<= 0.5",
        },
    }


def run(
    *,
    config_path: str | Path,
    policy_model: str | Path,
    seeds: list[int],
    output: str | Path,
    device: str = "auto",
) -> Path:
    cfg = load_config(config_path)
    if not RiskGatedACCVPCandidateTableAugmentor.enabled(cfg):
        raise ValueError("preflight requires accvp.observation.enabled=true")
    model_path = _resolve(policy_model)
    model = load_ppo(model_path, device=device)
    reports: list[dict[str, Any]] = []
    rewards: list[float] = []
    shape_env = make_env(cfg, seed=int(seeds[0]), shield_enabled=False)
    try:
        validate_model_env_observation_shape(model, shape_env, model_path)
        model_observation_shape = list(model.observation_space.shape)
        env_observation_shape = list(shape_env.observation_space.shape)
    finally:
        shape_env.close()
    for seed in seeds:
        env = make_env(cfg, seed=int(seed), shield_enabled=False)
        total_reward = 0.0
        try:
            obs, _info = env.reset(seed=int(seed))
            terminated = truncated = False
            while not (terminated or truncated):
                action, _state = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _info = env.step(int(action))
                total_reward += float(reward)
            report = env.episode_report()
            report["episode_reward"] = total_reward
            reports.append(report)
            rewards.append(total_reward)
        finally:
            env.close()
    metrics = aggregate_episode_reports(reports)
    metrics["average_reward"] = float(np.mean(rewards)) if rewards else 0.0
    gate = _gate(metrics)
    payload = {
        "stage": "accvp_observation_preflight",
        "config": str(config_path),
        "policy_model": str(model_path),
        "seeds": [int(seed) for seed in seeds],
        "model_observation_shape": model_observation_shape,
        "env_observation_shape": env_observation_shape,
        "metrics": metrics,
        "gate": gate,
        "episodes": reports,
    }
    output_path = _resolve(output)
    write_json(output_path, payload)
    print(f"[accvp_observation_preflight] report={output_path}")
    print(f"[accvp_observation_preflight] gate_pass={gate['pass']} metrics={metrics}")
    return output_path


def main() -> None:
    args = parse_args()
    run(
        config_path=args.config,
        policy_model=args.policy_model,
        seeds=[int(seed) for seed in args.seeds],
        output=args.output,
        device=str(args.device),
    )


if __name__ == "__main__":
    main()
