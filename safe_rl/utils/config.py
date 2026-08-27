from __future__ import annotations

import copy
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "safe_rl" / "config" / "default_safe_rl.yaml"
STANDALONE_CONFIG_KEY = "standalone_config"

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


def _resolve_extended_config(
    source: Path,
    *,
    active: tuple[Path, ...] = (),
) -> dict[str, Any]:
    resolved = source.resolve()
    if resolved in active:
        cycle = " -> ".join(str(path) for path in (*active, resolved))
        raise ValueError(f"configuration extends cycle: {cycle}")
    with resolved.open("r", encoding="utf-8") as file:
        payload = yaml.safe_load(file) or {}
    if not isinstance(payload, Mapping):
        raise ValueError(f"configuration overlay must be a mapping: {resolved}")
    payload = dict(payload)
    parent_value = payload.pop("extends", None)
    _reject_retired_noop_keys(payload)
    if parent_value is None:
        return payload
    parent = Path(str(parent_value))
    if not parent.is_absolute():
        repo_candidate = (REPO_ROOT / parent).resolve()
        parent = (
            repo_candidate
            if repo_candidate.is_file()
            else (resolved.parent / parent).resolve()
        )
    inherited = _resolve_extended_config(
        parent,
        active=(*active, resolved),
    )
    return _deep_merge(inherited, payload)


def load_config(config_path: str | os.PathLike[str] | None = None) -> ConfigDict:
    """Load defaults plus an optional recursively inherited YAML overlay.

    ``extends`` is resolved before the final merge and is intentionally
    removed from the resolved configuration. This lets canonical protocol
    generations override only their contract/path changes without copying
    hundreds of unrelated defaults.
    """

    override: dict[str, Any] = {}
    if config_path:
        path = Path(config_path)
        override = _resolve_extended_config(path)
    standalone = override.pop(STANDALONE_CONFIG_KEY, False)
    if standalone not in {False, True}:
        raise ValueError(f"{STANDALONE_CONFIG_KEY} must be a boolean")
    if standalone:
        data: dict[str, Any] = {}
    else:
        with DEFAULT_CONFIG_PATH.open("r", encoding="utf-8") as file:
            data = yaml.safe_load(file) or {}
    data = _deep_merge(data, override)

    cfg = resolve_paths(_to_config_dict(data))
    return _apply_selector_capacity_lock(cfg)


def _apply_selector_capacity_lock(cfg: ConfigDict) -> ConfigDict:
    """Apply an immutable selector audit capacity to every downstream stage."""

    selector_contract = cfg.accvp.get("selector_contract", {}) or {}
    if not bool(selector_contract.get("require_capacity_lock", False)):
        return cfg
    configured = selector_contract.get("audit_report")
    if not configured:
        raise ValueError(
            "accvp.selector_contract.require_capacity_lock requires audit_report"
        )
    path = Path(str(configured))
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"selector capacity audit report does not exist: {path}"
        )
    with path.open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    report_kind = str(report.get("artifact_kind", ""))
    protocol_id = str(report.get("protocol_id", ""))
    supported_reports = {
        "accvp_selector_contract_audit_v1": {
            "protocol_id": "accvp-vnext-correctness-v2-selector3",
            "capacities": {6, 8},
        },
        "accvp_selector_capacity_audit_v2": {
            "protocol_id": "accvp-vnext-correctness-v3-selector4",
            "capacities": {8, 10, 12},
        },
    }
    if report_kind not in supported_reports:
        raise ValueError("unsupported selector capacity audit report")
    expected = supported_reports[report_kind]
    if protocol_id != str(expected["protocol_id"]):
        raise ValueError("selector capacity audit protocol_id mismatch")
    from safe_rl.accvp.contracts.schema import stable_hash

    declared_fingerprint = str(report.get("report_fingerprint", ""))
    recomputed_fingerprint = stable_hash(
        {
            key: value
            for key, value in report.items()
            if key != "report_fingerprint"
        }
    )
    if (
        not declared_fingerprint
        or declared_fingerprint != recomputed_fingerprint
    ):
        raise ValueError("selector capacity audit report fingerprint mismatch")
    if str(report.get("audit_state", "")) != "pass":
        raise ValueError("selector capacity audit is blocked")
    capacity = int(report.get("selected_capacity", -1))
    if capacity not in set(expected["capacities"]):
        raise ValueError(
            "selector capacity audit selected an unsupported capacity"
        )
    # The audit freezes ACCVP task rows only. WcDT keeps the independent
    # selector/max-agent contract stored in its checkpoint.
    cfg.accvp["actor_count"] = capacity
    cfg.accvp.selector_contract["resolved_capacity"] = capacity
    cfg.accvp.selector_contract["audit_report"] = str(path)
    cfg.accvp.selector_contract["audit_report_fingerprint"] = str(
        declared_fingerprint
    )
    return cfg


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


def clone_with_overrides(
    cfg: ConfigDict,
    overrides: Mapping[str, Any],
    *,
    replace_blocks: tuple[str, ...] = (),
) -> ConfigDict:
    _reject_retired_noop_keys(overrides)
    merged = _deep_merge(dict(cfg), overrides)
    for key in replace_blocks:
        if key in overrides:
            merged[key] = copy.deepcopy(overrides[key])
    return _to_config_dict(merged)
