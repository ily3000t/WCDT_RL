from __future__ import annotations

import copy
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "safe_rl" / "config" / "default_safe_rl.yaml"

# These keys existed in the default file but were never consumed by runtime
# code. Rejecting them in overlays is safer than silently pretending a no-op
# switch changed an experiment. See safe_rl/config/README.md for migrations.
RETIRED_NOOP_CONFIG_PATHS: dict[tuple[str, ...], str] = {
    ("scenario", "merge_opportunity_min_distance_to_taper"): "no runtime replacement",
    ("prediction", "model_type"): "select a maintained WcDT config/checkpoint",
    ("prediction", "freeze"): "runtime predictors are inference-only",
    ("prediction", "num_modes"): "WcDT-v1 mode count is checkpoint-versioned",
    ("forecast_features", "freeze_predictor"): "runtime predictors are inference-only",
    ("forecast_features", "top_k_agents"): "use prediction.wcdt_v*_max_agents",
    ("forecast_features", "include_min_distance"): "feature vector is versioned",
    ("forecast_features", "include_ttc"): "feature vector is versioned",
    ("forecast_features", "include_drac"): "feature vector is versioned",
    ("forecast_features", "include_collision_probability"): "feature vector is versioned",
    ("forecast_features", "include_uncertainty"): "feature vector is versioned",
    ("forecast_features", "include_merge_gap"): "feature vector is versioned",
    ("forecast_features", "normalize_features"): "use forecast_features.normalize",
    ("forecast_features", "detach_gradient"): "runtime path has no gradient graph",
    ("rule_gap_acceptance", "desired_speed_source"): "lane speed limit is the fixed contract",
    ("risk_module", "use_wcdt_latent"): "Risk architecture is checkpoint-versioned",
    ("risk_module", "use_explicit_risk_features"): "Risk architecture is checkpoint-versioned",
    ("risk_module", "action_conditioned"): "Risk architecture is checkpoint-versioned",
    ("risk_module", "calibration_enabled"): "use risk_module.calibration settings",
    ("shield", "coarse_to_fine"): "candidate evaluation order is fixed",
    ("shield", "prefer_closest_safe_action"): "replacement ranking is versioned",
    ("shield", "fallback_action"): "fallback behavior is versioned",
    ("accvp", "inference_worker", "restart_on_timeout"): "worker recovery is versioned",
    ("stage1", "random_action_probability"): "use stage1.sampling_probs",
}


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


def _reject_retired_noop_keys(override: Mapping[str, Any]) -> None:
    found: list[str] = []
    for path, replacement in RETIRED_NOOP_CONFIG_PATHS.items():
        current: Any = override
        for part in path:
            if not isinstance(current, Mapping) or part not in current:
                break
            current = current[part]
        else:
            found.append(f"{'.'.join(path)} ({replacement})")
    if found:
        raise ValueError(
            "configuration contains retired no-op keys: " + "; ".join(found)
        )


def load_config(config_path: str | os.PathLike[str] | None = None) -> ConfigDict:
    """Load the default config and overlay an optional YAML config."""

    with DEFAULT_CONFIG_PATH.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}

    if config_path:
        path = Path(config_path)
        with path.open("r", encoding="utf-8") as file:
            override = yaml.safe_load(file) or {}
        _reject_retired_noop_keys(override)
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
    _reject_retired_noop_keys(overrides)
    merged = _deep_merge(dict(cfg), overrides)
    return _to_config_dict(merged)
