from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

from safe_rl.prediction.forecast_feature_augmentor import ForecastFeatureAugmentor
from safe_rl.prediction.wcdt_predictor import WcDTPredictor
from safe_rl.risk.risk_module import RiskModuleWrapper
from safe_rl.accvp.schema import file_sha256
from safe_rl.shield.safety_shield import SafetyShield
from safe_rl.sim.sumo_highway_merge_env import SumoHighwayMergeEnv
from safe_rl.utils.config import ConfigDict, REPO_ROOT, load_config, prepare_run_dir
from safe_rl.utils.sumo_installation import (
    configure_sumo_python,
    resolve_sumo_installation,
    sumo_installation_from_config,
    sumo_subprocess_environment,
)


def parse_config_arg(description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default=None, help="Optional YAML config overlay.")
    parser.add_argument("--run-id", default=None, help="Existing or new run id.")
    return parser.parse_args()


def load_stage_config(args: argparse.Namespace) -> ConfigDict:
    cfg = load_config(args.config)
    if args.run_id:
        cfg.run["run_id"] = args.run_id
    return cfg


def run_root(cfg: ConfigDict) -> Path:
    root = Path(cfg.run.output_root)
    if not root.is_absolute():
        root = REPO_ROOT / root
    if not cfg.run.get("run_id"):
        return prepare_run_dir(cfg)
    return root / cfg.run.run_id


def stage_file(cfg: ConfigDict, stage: str, name: str) -> Path:
    return run_root(cfg) / stage / name


def latest_stage_file(cfg: ConfigDict, stage: str, name: str) -> Path:
    candidate = stage_file(cfg, stage, name)
    if candidate.exists():
        return candidate
    root = Path(cfg.run.output_root)
    if not root.is_absolute():
        root = REPO_ROOT / root
    matches = sorted(root.glob(f"*/{stage}/{name}"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not matches:
        raise FileNotFoundError(f"could not find {stage}/{name}; set run.run_id or stage input path")
    return matches[0]


def _resolve_repo_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def _forecast_checkpoint_name(source: str) -> str:
    if source == "wcdt_v2":
        return "wcdt_v2_predictor.pt"
    if source == "wcdt_v3":
        return "wcdt_v3_predictor.pt"
    return "wcdt_predictor.pt"


def _forecast_checkpoint_path(cfg: ConfigDict, source: str = "wcdt") -> Path:
    checkpoint = cfg.forecast_features.get("checkpoint")
    if checkpoint:
        path = _resolve_repo_path(checkpoint)
        if path.exists():
            return path
        raise FileNotFoundError(f"forecast_features.checkpoint does not exist: {path}")
    checkpoint_name = _forecast_checkpoint_name(source)
    candidate = stage_file(cfg, "stage2", checkpoint_name)
    if candidate.exists():
        return candidate
    return latest_stage_file(cfg, "stage2", checkpoint_name)


def make_forecast_augmentor(cfg: ConfigDict) -> ForecastFeatureAugmentor | None:
    forecast_enabled = bool(cfg.forecast_features.enabled or cfg.rl.use_wcdt_forecast_features)
    if not forecast_enabled:
        return None
    source = str(cfg.forecast_features.get("source", "heuristic")).lower()
    if source == "constant_velocity":
        return ForecastFeatureAugmentor(cfg)
    try:
        checkpoint = _forecast_checkpoint_path(cfg, source)
    except FileNotFoundError:
        if bool(cfg.forecast_features.get("allow_heuristic_fallback", False)):
            return ForecastFeatureAugmentor(cfg)
        raise
    if source == "wcdt_v2":
        from safe_rl.prediction.wcdt_v2_predictor import WcDTV2Predictor

        return ForecastFeatureAugmentor(cfg, predictor=WcDTV2Predictor(cfg, checkpoint))
    if source == "wcdt_v3":
        from safe_rl.prediction.wcdt_v3_predictor import WcDTV3Predictor

        return ForecastFeatureAugmentor(cfg, predictor=WcDTV3Predictor(cfg, checkpoint))
    if source == "wcdt":
        return ForecastFeatureAugmentor(cfg, predictor=WcDTPredictor(cfg, checkpoint))
    return ForecastFeatureAugmentor(cfg)


def make_env(
    cfg: ConfigDict,
    seed: int,
    shield_enabled: bool | None = None,
    risk_checkpoint: str | None = None,
    reward_risk_checkpoint: str | None = None,
    record_trajectory_samples: bool = False,
    sumo_step_delay_ms: float = 0.0,
    worker_rank: int = 0,
    num_envs: int = 1,
    advance_episode_seed: bool = False,
) -> SumoHighwayMergeEnv:
    installation = (
        sumo_installation_from_config(cfg.scenario)
        if cfg.scenario.get("sumo_installation_fingerprint")
        else resolve_sumo_installation(cfg.scenario)
    )
    configure_sumo_python(installation)
    environment = sumo_subprocess_environment(installation)
    for key in ("SUMO_HOME", "PATH", "PYTHONPATH"):
        os.environ[key] = environment[key]
    cfg.scenario["sumo_binary"] = installation.sumo_binary
    cfg.scenario["sumo_gui_binary"] = installation.sumo_gui_binary
    cfg.scenario["netconvert_binary"] = installation.netconvert_binary
    cfg.scenario["sumo_tools_directory"] = installation.tools_directory
    cfg.scenario["sumo_home"] = installation.sumo_home
    cfg.scenario["sumo_version"] = installation.sumo_version
    cfg.scenario["sumo_installation_fingerprint"] = installation.to_dict()
    shield = None
    if shield_enabled if shield_enabled is not None else bool(cfg.shield.enabled):
        risk_model = RiskModuleWrapper(cfg, checkpoint=risk_checkpoint)
        shield = SafetyShield(cfg, risk_model)
        shield.enabled = True
    reward_risk_model = None
    if str(cfg.rl.get("reward_profile", "default")) in {"shield_guided_forecast", "merge_timing_forecast"}:
        reward_cfg = cfg.rl.get("shield_guided_reward", {})
        configured_checkpoint = reward_risk_checkpoint or reward_cfg.get("risk_checkpoint")
        if not configured_checkpoint:
            raise FileNotFoundError(
                "rl.reward_profile=shield_guided_forecast or merge_timing_forecast requires "
                "rl.shield_guided_reward.risk_checkpoint or make_env(..., reward_risk_checkpoint=...)."
            )
        checkpoint_path = _resolve_repo_path(configured_checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"shield-guided reward risk checkpoint does not exist: {checkpoint_path}")
        reward_risk_model = RiskModuleWrapper(cfg, checkpoint=str(checkpoint_path))
        if bool(reward_cfg.get("use_calibrated_risk", False)):
            reward_risk_model.apply_temperature = True
    forecast_augmentor = make_forecast_augmentor(cfg)
    accvp_controller = None
    if bool(cfg.accvp.get("enabled", False)) and str(cfg.accvp.get("mode", "off")) != "off":
        from safe_rl.accvp.runtime import build_accvp_controller

        configured_risk = cfg.accvp.get("risk_checkpoint")
        if configured_risk and risk_checkpoint and file_sha256(configured_risk) != file_sha256(risk_checkpoint):
            raise ValueError("ACCVP risk_checkpoint must match the Risk Module used by Safety Shield")

        accvp_controller = build_accvp_controller(cfg)
    return SumoHighwayMergeEnv(
        cfg,
        seed=seed,
        forecast_augmentor=forecast_augmentor,
        shield=shield,
        accvp_controller=accvp_controller,
        reward_risk_model=reward_risk_model,
        record_trajectory_samples=record_trajectory_samples,
        sumo_step_delay_ms=sumo_step_delay_ms,
        worker_rank=worker_rank,
        num_envs=num_envs,
        advance_episode_seed=advance_episode_seed,
    )


def json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return json_ready(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: json_ready(val) for key, val in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    return value


def write_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(json_ready(payload), file, ensure_ascii=False, indent=2, allow_nan=False)
