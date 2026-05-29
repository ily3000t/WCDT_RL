from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from safe_rl.prediction.forecast_feature_augmentor import ForecastFeatureAugmentor
from safe_rl.prediction.wcdt_predictor import WcDTPredictor
from safe_rl.risk.risk_module import RiskModuleWrapper
from safe_rl.shield.safety_shield import SafetyShield
from safe_rl.sim.sumo_highway_merge_env import SumoHighwayMergeEnv
from safe_rl.utils.config import ConfigDict, REPO_ROOT, load_config, prepare_run_dir


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
    return "wcdt_v2_predictor.pt" if source == "wcdt_v2" else "wcdt_predictor.pt"


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
) -> SumoHighwayMergeEnv:
    shield = None
    if shield_enabled if shield_enabled is not None else bool(cfg.shield.enabled):
        risk_model = RiskModuleWrapper(cfg, checkpoint=risk_checkpoint)
        shield = SafetyShield(cfg, risk_model)
        shield.enabled = True
    reward_risk_model = None
    if str(cfg.rl.get("reward_profile", "default")) == "shield_guided_forecast":
        reward_cfg = cfg.rl.get("shield_guided_reward", {})
        configured_checkpoint = reward_risk_checkpoint or reward_cfg.get("risk_checkpoint")
        if not configured_checkpoint:
            raise FileNotFoundError(
                "rl.reward_profile=shield_guided_forecast requires "
                "rl.shield_guided_reward.risk_checkpoint or make_env(..., reward_risk_checkpoint=...)."
            )
        checkpoint_path = _resolve_repo_path(configured_checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"shield-guided reward risk checkpoint does not exist: {checkpoint_path}")
        reward_risk_model = RiskModuleWrapper(cfg, checkpoint=str(checkpoint_path))
        if bool(reward_cfg.get("use_calibrated_risk", False)):
            reward_risk_model.apply_temperature = True
    forecast_augmentor = make_forecast_augmentor(cfg)
    return SumoHighwayMergeEnv(
        cfg,
        seed=seed,
        forecast_augmentor=forecast_augmentor,
        shield=shield,
        reward_risk_model=reward_risk_model,
        record_trajectory_samples=record_trajectory_samples,
        sumo_step_delay_ms=sumo_step_delay_ms,
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
