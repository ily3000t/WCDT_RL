from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from safe_rl.pipeline import (
    forecast_diagnostics,
    stage1_risk_probe,
    stage2_train_prediction_risk,
    stage3_train_ppo,
    stage4_collect_failures,
    stage5_paired_eval,
)
from safe_rl.utils.config import REPO_ROOT, load_config, prepare_run_dir
from safe_rl.utils.progress import stage_log


VALID_FORECAST_SOURCES = ("constant_velocity", "wcdt", "wcdt_v2")
DEFAULT_FORECAST_SOURCES = ("constant_velocity", "wcdt")
VALID_FORECAST_PPO_PROFILES = ("default", "safety")


def _relative_run_path(run_id: str, stage: str, name: str) -> str:
    return (Path("safe_rl_output") / "runs" / run_id / stage / name).as_posix()


def _write_yaml(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(payload, file, sort_keys=False, allow_unicode=True)
    return path


def _source_suffix(source: str) -> str:
    if source == "constant_velocity":
        return "cv"
    if source == "wcdt_v2":
        return "wcdt_v2"
    return "wcdt"


def _forecast_run_id(run_id: str, source: str) -> str:
    return f"{run_id}_forecast_{_source_suffix(source)}"


def resolve_forecast_sources(
    forecast_sources: str | list[str] | tuple[str, ...] | None = None,
    forecast_source: str | None = None,
) -> list[str]:
    if forecast_sources is not None and forecast_source is not None:
        raise ValueError("Use either --forecast-source or --forecast-sources, not both.")
    raw = forecast_source if forecast_source is not None else forecast_sources
    if raw is None:
        values = list(DEFAULT_FORECAST_SOURCES)
    elif isinstance(raw, str):
        values = [item.strip().lower() for item in raw.split(",") if item.strip()]
    else:
        values = [str(item).strip().lower() for item in raw if str(item).strip()]
    if not values:
        raise ValueError("forecast sources cannot be empty")
    invalid = [item for item in values if item not in VALID_FORECAST_SOURCES]
    if invalid:
        raise ValueError(f"forecast sources must be one of {VALID_FORECAST_SOURCES}; invalid={invalid}")
    deduped: list[str] = []
    for item in values:
        if item not in deduped:
            deduped.append(item)
    return deduped


def _forecast_group_name(source: str) -> str:
    if source == "wcdt":
        return "ppo_wcdt_features"
    if source == "wcdt_v2":
        return "ppo_wcdt_v2_features"
    return "ppo_cv_features"


def _forecast_shield_group_name(source: str) -> str:
    if source == "wcdt":
        return "wcdt_prediction_shield"
    if source == "wcdt_v2":
        return "wcdt_v2_prediction_shield"
    return "cv_prediction_shield"


def _forecast_checkpoint_name(source: str) -> str | None:
    if source == "wcdt":
        return "wcdt_predictor.pt"
    if source == "wcdt_v2":
        return "wcdt_v2_predictor.pt"
    return None


def _forecast_payload(
    run_id: str,
    source: str,
    ppo_timesteps: int | None,
    forecast_ppo_profile: str = "default",
) -> dict[str, Any]:
    forecast_run_id = _forecast_run_id(run_id, source)
    payload: dict[str, Any] = {
        "run": {"run_id": forecast_run_id},
        "forecast_features": {
            "enabled": True,
            "use_for_ppo_observation": True,
            "source": source,
            "checkpoint": (
                _relative_run_path(run_id, "stage2", _forecast_checkpoint_name(source))
                if _forecast_checkpoint_name(source)
                else None
            ),
            "allow_heuristic_fallback": False,
        },
        "rl": {"use_wcdt_forecast_features": True},
    }
    profile = str(forecast_ppo_profile or "default").lower()
    if profile not in VALID_FORECAST_PPO_PROFILES:
        raise ValueError(f"forecast PPO profile must be one of {VALID_FORECAST_PPO_PROFILES}; got {profile!r}")
    if profile == "safety":
        payload["rl"]["reward_profile"] = "safety_forecast"
    if ppo_timesteps is not None:
        payload["rl"]["total_timesteps"] = int(ppo_timesteps)
    return payload


def _forecast_stage5_groups(run_id: str, source: str) -> list[dict[str, Any]]:
    forecast_run_id = _forecast_run_id(run_id, source)
    base = {
        "name": _forecast_group_name(source),
        "forecast_features": True,
        "shield": False,
        "model_path": _relative_run_path(forecast_run_id, "stage3", "ppo_model.zip"),
        "forecast_source": source,
    }
    shield = {
        "name": _forecast_shield_group_name(source),
        "forecast_features": True,
        "shield": True,
        "model_path": _relative_run_path(forecast_run_id, "stage3", "ppo_model.zip"),
        "forecast_source": source,
    }
    checkpoint_name = _forecast_checkpoint_name(source)
    if checkpoint_name:
        checkpoint = _relative_run_path(run_id, "stage2", checkpoint_name)
        base["forecast_checkpoint"] = checkpoint
        shield["forecast_checkpoint"] = checkpoint
    return [base, shield]


def build_generated_configs(
    run_id: str,
    generated_dir: str | Path,
    stage1_episodes: int | None = None,
    ppo_timesteps: int | None = None,
    forecast_ppo_timesteps: int | None = None,
    forecast_ppo_profile: str = "default",
    forecast_sources: str | list[str] | tuple[str, ...] | None = None,
    forecast_source: str | None = None,
) -> dict[str, Path]:
    generated_dir = Path(generated_dir)
    sources = resolve_forecast_sources(forecast_sources=forecast_sources, forecast_source=forecast_source)

    main_payload: dict[str, Any] = {"run": {"run_id": run_id}}
    if stage1_episodes is not None:
        main_payload["stage1"] = {"episodes": int(stage1_episodes)}
    if ppo_timesteps is not None:
        main_payload["rl"] = {"total_timesteps": int(ppo_timesteps)}

    stage2_stage4_payload: dict[str, Any] = {
        "run": {"run_id": run_id},
        "stage2": {"input_stage4": "auto"},
        "prediction": {"train_enabled": False},
    }

    groups: list[dict[str, Any]] = [
        {
            "name": "ppo",
            "forecast_features": False,
            "shield": False,
            "model_path": _relative_run_path(run_id, "stage3", "ppo_model.zip"),
        },
        {
            "name": "ppo_shield",
            "forecast_features": False,
            "shield": True,
            "model_path": _relative_run_path(run_id, "stage3", "ppo_model.zip"),
        },
    ]
    for source in sources:
        groups.extend(_forecast_stage5_groups(run_id, source))

    stage5_payload: dict[str, Any] = {
        "run": {"run_id": run_id},
        "stage5": {
            "episodes_per_group": 20,
            "seeds": list(range(1, 21)),
            "groups": groups,
        },
    }

    configs = {
        "main": _write_yaml(generated_dir / "main_overrides.yaml", main_payload),
        "stage2_with_stage4": _write_yaml(generated_dir / "stage2_with_stage4.yaml", stage2_stage4_payload),
        "stage5_multi_groups": _write_yaml(generated_dir / "stage5_multi_groups.yaml", stage5_payload),
    }
    configs["stage5_four_groups"] = configs["stage5_multi_groups"]
    for source in sources:
        key = f"forecast_{_source_suffix(source)}_ppo"
        configs[key] = _write_yaml(
            generated_dir / f"{key}.yaml",
            _forecast_payload(
                run_id,
                source,
                forecast_ppo_timesteps if forecast_ppo_timesteps is not None else ppo_timesteps,
                forecast_ppo_profile=forecast_ppo_profile,
            ),
        )
    if sources:
        configs["forecast_ppo"] = configs[f"forecast_{_source_suffix(sources[0])}_ppo"]
    return configs


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
    forecast_ppo_timesteps: int | None = None,
    forecast_ppo_profile: str = "default",
    forecast_sources: str | list[str] | tuple[str, ...] | None = None,
    forecast_source: str | None = None,
) -> Path:
    sources = resolve_forecast_sources(forecast_sources=forecast_sources, forecast_source=forecast_source)
    bootstrap_cfg = load_config()
    bootstrap_cfg.run["run_id"] = run_id
    run_dir = prepare_run_dir(bootstrap_cfg)
    generated_dir = run_dir / "generated_configs"
    configs = build_generated_configs(
        run_id,
        generated_dir,
        stage1_episodes=stage1_episodes,
        ppo_timesteps=ppo_timesteps,
        forecast_ppo_timesteps=forecast_ppo_timesteps,
        forecast_ppo_profile=forecast_ppo_profile,
        forecast_sources=sources,
    )

    stage_log("full", f"run_id={run_id}")
    stage_log("full", f"forecast_sources={sources}")
    stage_log("full", f"forecast_ppo_profile={forecast_ppo_profile}")
    if forecast_ppo_timesteps is not None:
        stage_log("full", f"forecast_ppo_timesteps={forecast_ppo_timesteps}")
    for source in sources:
        stage_log("full", f"forecast_run_id[{source}]={_forecast_run_id(run_id, source)}")
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
    for source in sources:
        forecast_run_id = _forecast_run_id(run_id, source)
        stage_log("full", f"Stage3 forecast PPO ({source})")
        stage3_train_ppo.run(_load_stage_cfg(configs[f"forecast_{_source_suffix(source)}_ppo"], forecast_run_id))
    stage_log("full", "Stage5 multi-group paired evaluation")
    stage5_paired_eval.run(_load_stage_cfg(configs["stage5_multi_groups"], run_id))
    stage_log("full", "Forecast diagnostics")
    forecast_diagnostics.run_forecast_diagnostics(_load_stage_cfg(configs["main"], run_id))
    _print_stage5_summary(run_id)
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full SAFE_RL Stage1-Stage5 pipeline.")
    parser.add_argument("--run-id", required=True, help="Baseline run id. Forecast run id is '<run-id>_forecast'.")
    parser.add_argument("--stage1-episodes", type=int, default=None, help="Optional override for Stage1 episodes.")
    parser.add_argument("--ppo-timesteps", type=int, default=None, help="Optional override for baseline and forecast PPO timesteps.")
    parser.add_argument(
        "--forecast-ppo-timesteps",
        type=int,
        default=None,
        help="Optional override for forecast PPO timesteps only.",
    )
    parser.add_argument(
        "--forecast-ppo-profile",
        choices=list(VALID_FORECAST_PPO_PROFILES),
        default="default",
        help="Forecast PPO reward profile. 'safety' writes rl.reward_profile=safety_forecast for forecast PPO only.",
    )
    parser.add_argument(
        "--forecast-source",
        choices=list(VALID_FORECAST_SOURCES),
        default=None,
        help="Legacy single forecast feature source for the forecast PPO branch.",
    )
    parser.add_argument(
        "--forecast-sources",
        default=None,
        help="Comma-separated forecast sources. Default: constant_velocity,wcdt.",
    )
    args = parser.parse_args()
    if args.forecast_source and args.forecast_sources:
        parser.error("Use either --forecast-source or --forecast-sources, not both.")
    run_full_pipeline(
        args.run_id,
        stage1_episodes=args.stage1_episodes,
        ppo_timesteps=args.ppo_timesteps,
        forecast_ppo_timesteps=args.forecast_ppo_timesteps,
        forecast_ppo_profile=args.forecast_ppo_profile,
        forecast_sources=args.forecast_sources,
        forecast_source=args.forecast_source,
    )


if __name__ == "__main__":
    main()
