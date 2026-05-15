from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

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


def make_env(
    cfg: ConfigDict,
    seed: int,
    shield_enabled: bool | None = None,
    risk_checkpoint: str | None = None,
    record_trajectory_samples: bool = False,
) -> SumoHighwayMergeEnv:
    shield = None
    if shield_enabled if shield_enabled is not None else bool(cfg.shield.enabled):
        risk_model = RiskModuleWrapper(cfg, checkpoint=risk_checkpoint)
        shield = SafetyShield(cfg, risk_model)
        shield.enabled = True
    return SumoHighwayMergeEnv(
        cfg,
        seed=seed,
        shield=shield,
        record_trajectory_samples=record_trajectory_samples,
    )


def json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {key: json_ready(val) for key, val in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    return value


def write_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(json_ready(payload), file, ensure_ascii=False, indent=2)
