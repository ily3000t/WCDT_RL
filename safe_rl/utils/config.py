from __future__ import annotations

import copy
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "safe_rl" / "config" / "default_safe_rl.yaml"


class ConfigDict(dict):
    """Dict with attribute access for read-heavy experiment configs."""

    def __getattr__(self, item: str) -> Any:
        try:
            return self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc

    def copy(self) -> "ConfigDict":
        return ConfigDict(super().copy())


def _to_config_dict(value: Any) -> Any:
    if isinstance(value, Mapping):
        return ConfigDict({key: _to_config_dict(val) for key, val in value.items()})
    if isinstance(value, list):
        return [_to_config_dict(item) for item in value]
    return value


def _deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(dict(merged[key]), value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_config(config_path: str | os.PathLike[str] | None = None) -> ConfigDict:
    """Load the default config and overlay an optional YAML config."""

    with DEFAULT_CONFIG_PATH.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}

    if config_path:
        path = Path(config_path)
        with path.open("r", encoding="utf-8") as file:
            override = yaml.safe_load(file) or {}
        data = _deep_merge(data, override)

    cfg = _to_config_dict(data)
    return resolve_paths(cfg)


def resolve_paths(cfg: ConfigDict) -> ConfigDict:
    """Resolve repo-relative scenario paths without mutating unrelated values."""

    for key in ("root", "sumocfg", "net_file", "route_file"):
        value = cfg.scenario.get(key)
        if value and not Path(value).is_absolute():
            cfg.scenario[key] = str((REPO_ROOT / value).resolve())
    return cfg


def make_run_id(prefix: str = "safe_rl") -> str:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{prefix}_{stamp}"


def prepare_run_dir(cfg: ConfigDict, stage_name: str | None = None) -> Path:
    run_id = cfg.run.get("run_id") or make_run_id()
    cfg.run["run_id"] = run_id
    output_root = Path(cfg.run.output_root)
    if not output_root.is_absolute():
        output_root = REPO_ROOT / output_root
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    if stage_name:
        stage_dir = run_dir / stage_name
        stage_dir.mkdir(parents=True, exist_ok=True)
        return stage_dir
    return run_dir


def clone_with_overrides(cfg: ConfigDict, overrides: Mapping[str, Any]) -> ConfigDict:
    merged = _deep_merge(dict(cfg), overrides)
    return _to_config_dict(merged)
