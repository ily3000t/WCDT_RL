"""Versioned ACCVP protocol semantics shared by collection and runtime."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from safe_rl.accvp.schema import file_sha256, stable_hash
from safe_rl.prediction.actor_selector import actor_selection_config_hash
from safe_rl.sim.metrics import SAFETY_METRIC_VERSION


ACCVP_EVENT_DEFINITION_VERSION = "accvp_v2_proxy_safety_taper_viability"


def effective_activation_distance(config: Any) -> float:
    """Return the ACV-Shield activation window without altering legacy configs."""

    configured = config.accvp.get("activation_distance")
    value = config.accvp.get("deadline_distance", 120.0) if configured is None else configured
    distance = float(value)
    if distance <= 0.0:
        raise ValueError("accvp.activation_distance must be positive")
    return distance


def activation_bin(context: dict[str, Any], activation_distance: float) -> str:
    """Classify a root against the ACV-Shield window, excluding past taper."""

    local = context.get("merge_local")
    if local is None or not bool(local.ego_on_auxiliary):
        return "not_auxiliary"
    distance = float(local.merge_distance)
    if distance <= 0.0:
        return "past_taper"
    return "activation_window" if distance <= float(activation_distance) else "pre_activation"


def legacy_deadline_bin(value: str) -> str:
    """Keep schema-v1 consumers readable while v2 uses ``activation_bin``."""

    return {
        "activation_window": "deadline",
        "pre_activation": "pre_deadline",
    }.get(str(value), str(value))


def _scenario_file_fingerprint(value: Any) -> dict[str, str | None]:
    if not value:
        return {"path": None, "sha256": None}
    path = Path(str(value))
    return {
        "path": str(path.resolve()),
        "sha256": file_sha256(path) if path.exists() and path.is_file() else None,
    }


def scenario_route_fingerprint(config: Any) -> str:
    """Hash the actual scenario and route inputs, not only their config paths."""

    scenario = config.scenario
    files = {
        key: _scenario_file_fingerprint(scenario.get(key))
        for key in ("sumocfg", "net_file", "route_file", "additional_file")
    }
    return stable_hash({"scenario": dict(scenario), "files": files})


def counterfactual_data_contract(config: Any, risk_model_fingerprint: str) -> dict[str, Any]:
    """Fields that must match before ACCVP shards may be merged or deployed."""

    return {
        "protocol_version": "accvp_240_v1",
        "scenario_config_hash": stable_hash(dict(config.scenario)),
        "scenario_route_hash": scenario_route_fingerprint(config),
        "action_execution_profile": str(config.scenario.get("action_execution_profile", "current_v1")),
        "candidate_plan_profile": str(config.accvp.candidate_plan_profile),
        "activation_distance_m": effective_activation_distance(config),
        "response_horizon_s": float(config.accvp.response_horizon_s),
        "response_horizon_steps": int(config.accvp.response_horizon_steps),
        "viability_horizon_s": float(config.accvp.viability_horizon_s),
        "candidate_plan_horizon_steps": int(config.accvp.candidate_plan_horizon_steps),
        "actor_count": int(config.accvp.actor_count),
        "actor_selection_config_hash": actor_selection_config_hash(config),
        "safety_metric_version": str(config.risk_module.get("safety_metric_version", SAFETY_METRIC_VERSION)),
        "event_definition_version": ACCVP_EVENT_DEFINITION_VERSION,
        "risk_model_fingerprint": str(risk_model_fingerprint),
    }


def data_contract_hash(contract: dict[str, Any]) -> str:
    return stable_hash(contract)
