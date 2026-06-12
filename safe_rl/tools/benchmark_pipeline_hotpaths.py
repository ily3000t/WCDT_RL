from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from safe_rl.pipeline import stage1_risk_probe, stage3_train_ppo
from safe_rl.utils.config import REPO_ROOT, clone_with_overrides, load_config
from safe_rl.utils.sumo_installation import resolve_sumo_installation


def _profile_config(profile: str):
    path = None
    if profile == "performance":
        path = REPO_ROOT / "safe_rl" / "config" / "advanced" / "pipeline_performance.yaml"
    cfg = load_config(path)
    installation = resolve_sumo_installation(cfg.scenario)
    cfg.scenario.update(
        {
            "sumo_binary": installation.sumo_binary,
            "sumo_gui_binary": installation.sumo_gui_binary,
            "netconvert_binary": installation.netconvert_binary,
            "sumo_tools_directory": installation.tools_directory,
            "sumo_home": installation.sumo_home,
            "sumo_version": installation.sumo_version,
        }
    )
    return cfg


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def benchmark_profile(profile: str, stage1_episodes: int, ppo_timesteps: int, stamp: str) -> dict[str, Any]:
    run_id = f"pipeline_hotpaths_{stamp}_{profile}"
    cfg = clone_with_overrides(
        _profile_config(profile),
        {
            "run": {"run_id": run_id, "tensorboard": False, "replay": False},
            "stage1": {
                "episodes": int(stage1_episodes),
                "audit_enabled": False,
                "audit_gate": {"enabled": False},
                "replay_enabled": False,
            },
            "stage3": {"eval_enabled": False},
            "rl": {"total_timesteps": int(ppo_timesteps), "reward_profile": "default"},
            "forecast_features": {"enabled": False},
            "shield": {"forecast_task_shadow_enabled": False, "task_backstop_enabled": False},
        },
    )
    stage1_risk_probe.run(cfg)
    stage3_train_ppo.run(cfg)
    run_root = Path(cfg.run.output_root)
    if not run_root.is_absolute():
        run_root = REPO_ROOT / run_root
    stage1_report = _read_json(run_root / run_id / "stage1" / "stage1_report.json")
    stage3_report = _read_json(run_root / run_id / "stage3" / "stage3_training_report.json")
    return {
        "profile": profile,
        "run_id": run_id,
        "stage1": stage1_report.get("performance", {}),
        "ppo": {
            "wall_time": stage3_report.get("wall_time", 0.0),
            "fps": stage3_report.get("steps_per_second", 0.0),
            "num_envs": stage3_report.get("ppo_num_envs", 1),
            "rollout_size": stage3_report.get("ppo_rollout_size", 0),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark SAFE_RL Stage1 and PPO hot paths.")
    parser.add_argument("--stage1-episodes", type=int, default=50)
    parser.add_argument("--ppo-timesteps", type=int, default=4096)
    parser.add_argument("--profiles", default="default,performance")
    parser.add_argument("--output", default="safe_rl_output/benchmarks/pipeline_hotpaths.json")
    args = parser.parse_args()
    profiles = [item.strip() for item in args.profiles.split(",") if item.strip()]
    invalid = [item for item in profiles if item not in {"default", "performance"}]
    if invalid:
        parser.error(f"unsupported profiles: {invalid}")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = [
        benchmark_profile(profile, int(args.stage1_episodes), int(args.ppo_timesteps), stamp)
        for profile in profiles
    ]
    by_profile = {item["profile"]: item for item in results}
    baseline = by_profile.get("default")
    performance = by_profile.get("performance")
    speedup = {}
    if baseline and performance:
        baseline_stage1 = float(baseline["stage1"].get("wall_time", 0.0))
        performance_stage1 = float(performance["stage1"].get("wall_time", 0.0))
        baseline_ppo = float(baseline["ppo"].get("wall_time", 0.0))
        performance_ppo = float(performance["ppo"].get("wall_time", 0.0))
        speedup = {
            "stage1": baseline_stage1 / performance_stage1 if performance_stage1 > 0.0 else 0.0,
            "ppo": baseline_ppo / performance_ppo if performance_ppo > 0.0 else 0.0,
        }
    payload = {"profiles": results, "speedup_ratio": speedup}
    output = Path(args.output)
    if not output.is_absolute():
        output = REPO_ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
