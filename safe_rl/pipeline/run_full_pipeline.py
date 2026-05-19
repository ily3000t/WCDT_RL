from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from safe_rl.pipeline import (
    stage1_risk_probe,
    stage2_train_prediction_risk,
    stage3_train_ppo,
    stage4_collect_failures,
    stage5_paired_eval,
)
from safe_rl.utils.config import REPO_ROOT, load_config, prepare_run_dir
from safe_rl.utils.progress import stage_log


def _relative_run_path(run_id: str, stage: str, name: str) -> str:
    return (Path("safe_rl_output") / "runs" / run_id / stage / name).as_posix()


def _write_yaml(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(payload, file, sort_keys=False, allow_unicode=True)
    return path


def build_generated_configs(
    run_id: str,
    generated_dir: str | Path,
    stage1_episodes: int | None = None,
    ppo_timesteps: int | None = None,
    forecast_source: str = "constant_velocity",
) -> dict[str, Path]:
    generated_dir = Path(generated_dir)
    forecast_run_id = f"{run_id}_forecast"
    forecast_source = str(forecast_source).lower()
    if forecast_source not in ("constant_velocity", "wcdt"):
        raise ValueError("forecast_source must be 'constant_velocity' or 'wcdt'")
    forecast_group_name = "ppo_wcdt_features" if forecast_source == "wcdt" else "ppo_cv_features"

    main_payload: dict[str, Any] = {"run": {"run_id": run_id}}
    if stage1_episodes is not None:
        main_payload["stage1"] = {"episodes": int(stage1_episodes)}
    if ppo_timesteps is not None:
        main_payload["rl"] = {"total_timesteps": int(ppo_timesteps)}

    forecast_payload: dict[str, Any] = {
        "run": {"run_id": forecast_run_id},
        "forecast_features": {
            "enabled": True,
            "use_for_ppo_observation": True,
            "source": forecast_source,
            "checkpoint": _relative_run_path(run_id, "stage2", "wcdt_predictor.pt") if forecast_source == "wcdt" else None,
            "allow_heuristic_fallback": False,
        },
        "rl": {"use_wcdt_forecast_features": True},
    }
    if ppo_timesteps is not None:
        forecast_payload["rl"]["total_timesteps"] = int(ppo_timesteps)

    stage2_stage4_payload: dict[str, Any] = {
        "run": {"run_id": run_id},
        "stage2": {"input_stage4": "auto"},
        "prediction": {"train_enabled": True},
    }

    stage5_payload: dict[str, Any] = {
        "run": {"run_id": run_id},
        "stage5": {
            "episodes_per_group": 20,
            "seeds": list(range(1, 21)),
            "groups": [
                {
                    "name": "ppo",
                    "forecast_features": False,
                    "shield": False,
                    "model_path": _relative_run_path(run_id, "stage3", "ppo_model.zip"),
                },
                {
                    "name": forecast_group_name,
                    "forecast_features": True,
                    "shield": False,
                    "model_path": _relative_run_path(forecast_run_id, "stage3", "ppo_model.zip"),
                    "forecast_source": forecast_source,
                },
                {
                    "name": "ppo_shield",
                    "forecast_features": False,
                    "shield": True,
                    "model_path": _relative_run_path(run_id, "stage3", "ppo_model.zip"),
                },
                {
                    "name": "full_prediction_shield",
                    "forecast_features": True,
                    "shield": True,
                    "model_path": _relative_run_path(forecast_run_id, "stage3", "ppo_model.zip"),
                    "forecast_source": forecast_source,
                },
            ],
        },
    }
    if forecast_source == "wcdt":
        stage5_payload["stage5"]["groups"][1]["forecast_checkpoint"] = _relative_run_path(run_id, "stage2", "wcdt_predictor.pt")
        stage5_payload["stage5"]["groups"][3]["forecast_checkpoint"] = _relative_run_path(run_id, "stage2", "wcdt_predictor.pt")

    return {
        "main": _write_yaml(generated_dir / "main_overrides.yaml", main_payload),
        "stage2_with_stage4": _write_yaml(generated_dir / "stage2_with_stage4.yaml", stage2_stage4_payload),
        "forecast_ppo": _write_yaml(generated_dir / "forecast_ppo.yaml", forecast_payload),
        "stage5_four_groups": _write_yaml(generated_dir / "stage5_four_groups.yaml", stage5_payload),
    }


def _load_stage_cfg(config_path: Path, run_id: str):
    cfg = load_config(config_path)
    cfg.run["run_id"] = run_id
    return cfg


def _run_subprocess(command: list[str], label: str) -> None:
    stage_log("full", f"{label}: {' '.join(command)}")
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def _sumo_smoke_check(config_path: Path, run_id: str) -> None:
    cfg = _load_stage_cfg(config_path, run_id)
    _run_subprocess(
        [
            str(cfg.scenario.get("sumo_binary", "sumo")),
            "-c",
            str(cfg.scenario.sumocfg),
            "--end",
            "1",
            "--no-step-log",
            "true",
            "--duration-log.disable",
            "true",
            "--seed",
            str(cfg.run.seed),
        ],
        "SUMO smoke check",
    )


def _print_stage5_summary(run_id: str) -> None:
    report_path = REPO_ROOT / "safe_rl_output" / "runs" / run_id / "stage5" / "formal_paired_eval_report.json"
    if not report_path.exists():
        stage_log("full", f"Stage5 report not found: {report_path}")
        return
    with report_path.open("r", encoding="utf-8") as file:
        report = json.load(file)
    stage_log("full", "Stage5 acceptance:")
    print(json.dumps(report.get("acceptance", {}), ensure_ascii=False, indent=2))
    metrics = {name: item.get("metrics", {}) for name, item in report.get("groups", {}).items()}
    stage_log("full", "Stage5 metrics:")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def run_full_pipeline(
    run_id: str,
    stage1_episodes: int | None = None,
    ppo_timesteps: int | None = None,
    forecast_source: str = "constant_velocity",
) -> Path:
    bootstrap_cfg = load_config()
    bootstrap_cfg.run["run_id"] = run_id
    run_dir = prepare_run_dir(bootstrap_cfg)
    generated_dir = run_dir / "generated_configs"
    configs = build_generated_configs(run_id, generated_dir, stage1_episodes, ppo_timesteps, forecast_source=forecast_source)
    forecast_run_id = f"{run_id}_forecast"

    stage_log("full", f"run_id={run_id}")
    stage_log("full", f"forecast_run_id={forecast_run_id}")
    stage_log("full", f"forecast_source={forecast_source}")
    stage_log("full", f"generated_configs={generated_dir}")

    _run_subprocess([sys.executable, str(REPO_ROOT / "scenarios" / "highway_merge" / "build_network.py")], "build network")
    _sumo_smoke_check(configs["main"], run_id)

    stage_log("full", "Stage1 risk probe")
    stage1_risk_probe.run(_load_stage_cfg(configs["main"], run_id))
    stage_log("full", "Stage2 initial prediction + risk")
    stage2_train_prediction_risk.run(_load_stage_cfg(configs["main"], run_id))
    stage_log("full", "Stage3 baseline PPO")
    stage3_train_ppo.run(_load_stage_cfg(configs["main"], run_id))
    stage_log("full", "Stage4 shadow collection")
    stage4_collect_failures.run(_load_stage_cfg(configs["main"], run_id))
    stage_log("full", "Stage2 risk retraining with Stage4 buffer")
    stage2_train_prediction_risk.run(_load_stage_cfg(configs["stage2_with_stage4"], run_id))
    stage_log("full", "Stage3 forecast PPO")
    stage3_train_ppo.run(_load_stage_cfg(configs["forecast_ppo"], forecast_run_id))
    stage_log("full", "Stage5 four-group paired evaluation")
    stage5_paired_eval.run(_load_stage_cfg(configs["stage5_four_groups"], run_id))
    _print_stage5_summary(run_id)
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full SAFE_RL Stage1-Stage5 pipeline.")
    parser.add_argument("--run-id", required=True, help="Baseline run id. Forecast run id is '<run-id>_forecast'.")
    parser.add_argument("--stage1-episodes", type=int, default=None, help="Optional override for Stage1 episodes.")
    parser.add_argument("--ppo-timesteps", type=int, default=None, help="Optional override for baseline and forecast PPO timesteps.")
    parser.add_argument(
        "--forecast-source",
        choices=["constant_velocity", "wcdt"],
        default="constant_velocity",
        help="Forecast feature source for the forecast PPO branch.",
    )
    args = parser.parse_args()
    run_full_pipeline(
        args.run_id,
        stage1_episodes=args.stage1_episodes,
        ppo_timesteps=args.ppo_timesteps,
        forecast_source=args.forecast_source,
    )


if __name__ == "__main__":
    main()
