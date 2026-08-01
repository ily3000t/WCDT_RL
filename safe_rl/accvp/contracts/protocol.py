"""Versioned ACCVP data semantics shared by collection and runtime."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from safe_rl.accvp.contracts.schema import (
    ACTOR_ROW_MAPPING_VERSION,
    ENTRY_TIME_LABEL_VERSION,
    ROOT_OBSERVATION_FINGERPRINT_VERSION,
    file_sha256,
    stable_hash,
)
from safe_rl.prediction.actor_selector import (
    ACTOR_SELECTION_VERSION_V3,
    actor_relevance_config,
    actor_selection_config_hash,
)
from safe_rl.prediction.wcdt_v3_predictor import TRAJECTORY_SCHEMA_VERSION
from safe_rl.sim.metrics import SAFETY_METRIC_VERSION
from safe_rl.utils.sumo_installation import (
    resolve_sumo_installation,
    sumo_installation_from_config,
)


ACCVP_EVENT_DEFINITION_VERSION = "accvp_v2_proxy_safety_taper_viability"
ACCVP_DATA_CONTRACT_VERSION = "accvp_240_v2"
ACCVP_SELECTOR3_DATA_CONTRACT_VERSION = "accvp_240_v3_conflict_selector"
SUPPORTED_ACCVP_DATA_CONTRACT_VERSIONS = frozenset(
    {ACCVP_DATA_CONTRACT_VERSION, ACCVP_SELECTOR3_DATA_CONTRACT_VERSION}
)

# ``accvp_240_v2`` was frozen while this retired, runtime-inert key still
# existed in the default scenario mapping.  Removing the no-op configuration
# surface must not invalidate already collected schema-v3 data or its trained
# bundle.  Keep the value only in the versioned contract payload; overlays are
# still rejected by ``utils.config.RETIRED_NOOP_CONFIG_PATHS`` and simulation
# code never reads it.  A future data-contract version may drop this shim.
_ACCVP_240_V2_RETIRED_SCENARIO_DEFAULTS = {
    "merge_opportunity_min_distance_to_taper": 60.0,
}


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


def scenario_contract_payload(config: Any) -> dict[str, Any]:
    """Return the scenario mapping frozen by the current data contract.

    This is intentionally distinct from the executable configuration: retired
    no-op fields remain hash-compatible without becoming supported controls.
    """

    scenario = dict(config.scenario)
    configured_version = str(
        config.accvp.get("data_contract_version", ACCVP_DATA_CONTRACT_VERSION)
    )
    if configured_version == ACCVP_DATA_CONTRACT_VERSION:
        for key, value in _ACCVP_240_V2_RETIRED_SCENARIO_DEFAULTS.items():
            scenario.setdefault(key, value)
    return scenario


def scenario_config_hash(config: Any) -> str:
    """Hash scenario semantics using the versioned compatibility payload."""

    return stable_hash(scenario_contract_payload(config))


def scenario_route_fingerprint(config: Any) -> str:
    """Hash the actual scenario and route inputs, not only their config paths."""

    scenario = config.scenario
    files = {
        key: _scenario_file_fingerprint(scenario.get(key))
        for key in ("sumocfg", "net_file", "route_file", "additional_file")
    }
    return stable_hash({"scenario": scenario_contract_payload(config), "files": files})


def counterfactual_data_contract(config: Any, risk_model_fingerprint: str) -> dict[str, Any]:
    """Fields that must match before ACCVP shards may be merged or deployed."""

    configured_version = str(
        config.accvp.get("data_contract_version", ACCVP_DATA_CONTRACT_VERSION)
    )
    if configured_version not in SUPPORTED_ACCVP_DATA_CONTRACT_VERSIONS:
        raise ValueError(
            "unsupported accvp.data_contract_version: "
            f"configured={configured_version!r} "
            f"supported={sorted(SUPPORTED_ACCVP_DATA_CONTRACT_VERSIONS)!r}"
        )
    selection_version = str(
        actor_relevance_config(
            config,
            selector_scope="accvp",
        )["version"]
    )
    if (
        configured_version == ACCVP_SELECTOR3_DATA_CONTRACT_VERSION
        and selection_version != ACTOR_SELECTION_VERSION_V3
    ):
        raise ValueError(
            f"{ACCVP_SELECTOR3_DATA_CONTRACT_VERSION} requires "
            f"accvp.actor_relevance.version={ACTOR_SELECTION_VERSION_V3!r}"
        )
    selector_contract = dict(config.accvp.get("selector_contract", {}) or {})
    contract = {
        "protocol_version": configured_version,
        "scenario_config_hash": scenario_config_hash(config),
        "scenario_route_hash": scenario_route_fingerprint(config),
        "action_execution_profile": str(config.scenario.get("action_execution_profile", "current_v1")),
        "candidate_plan_profile": str(config.accvp.candidate_plan_profile),
        "activation_distance_m": effective_activation_distance(config),
        "response_horizon_s": float(config.accvp.response_horizon_s),
        "response_horizon_steps": int(config.accvp.response_horizon_steps),
        "viability_horizon_s": float(config.accvp.viability_horizon_s),
        "candidate_plan_horizon_steps": int(config.accvp.candidate_plan_horizon_steps),
        "actor_count": int(config.accvp.actor_count),
        "actor_row_mapping_version": ACTOR_ROW_MAPPING_VERSION,
        "root_observation_fingerprint_version": ROOT_OBSERVATION_FINGERPRINT_VERSION,
        "entry_time_label_version": ENTRY_TIME_LABEL_VERSION,
        "response_feature_order": ["x", "y", "heading", "speed", "accel"],
        "wcdt_trajectory_schema_version": int(TRAJECTORY_SCHEMA_VERSION),
        "actor_selection_version": selection_version,
        "actor_selection_config_hash": actor_selection_config_hash(
            config,
            selector_scope="accvp",
        ),
        "vehicle_state_ordering_version": str(
            config.scenario.get("vehicle_state_ordering_version", "unspecified_legacy")
        ),
        "safety_metric_version": str(config.risk_module.get("safety_metric_version", SAFETY_METRIC_VERSION)),
        "event_definition_version": ACCVP_EVENT_DEFINITION_VERSION,
        "risk_model_fingerprint": str(risk_model_fingerprint),
    }
    if configured_version == ACCVP_SELECTOR3_DATA_CONTRACT_VERSION:
        if not bool(selector_contract.get("require_capacity_lock", False)):
            raise ValueError(
                "selector3 data contract requires a frozen selector capacity audit"
            )
        audit_report = selector_contract.get("audit_report")
        if not audit_report:
            raise ValueError("selector3 data contract requires selector audit_report")
        audit_path = Path(str(audit_report))
        if not audit_path.is_file():
            raise FileNotFoundError(audit_path)
        contract["selector_capacity_audit_report_sha256"] = file_sha256(
            audit_path
        )
        contract["selector_capacity_audit_report_fingerprint"] = str(
            selector_contract.get("audit_report_fingerprint", "")
        )
    return contract


def config_with_sumo_installation_fingerprint(config: Any) -> Any:
    """Mirror the collector's runtime scenario fingerprint mutation.

    Collection resolves the concrete SUMO installation before freezing the
    data contract. Offline validators must reproduce that exact mutation
    rather than compare the dataset against an unresolved YAML scenario.
    """

    candidate = copy.deepcopy(config)
    installation = (
        sumo_installation_from_config(candidate.scenario)
        if candidate.scenario.get("sumo_installation_fingerprint")
        else resolve_sumo_installation(candidate.scenario)
    )
    candidate.scenario["sumo_binary"] = installation.sumo_binary
    candidate.scenario["sumo_gui_binary"] = installation.sumo_gui_binary
    candidate.scenario["netconvert_binary"] = installation.netconvert_binary
    candidate.scenario[
        "sumo_tools_directory"
    ] = installation.tools_directory
    candidate.scenario["sumo_home"] = installation.sumo_home
    candidate.scenario["sumo_version"] = installation.sumo_version
    candidate.scenario[
        "sumo_installation_fingerprint"
    ] = installation.to_dict()
    return candidate


def counterfactual_data_contract_candidates(
    config: Any,
    risk_model_fingerprint: str,
) -> tuple[dict[str, Any], ...]:
    """Return exact offline and runtime-resolved contract candidates.

    A candidate is never field-relaxed: callers still require full mapping and
    hash equality. Resolution failure leaves only the raw contract, preserving
    fail-closed behavior on machines without the configured SUMO installation.
    """

    candidates = [
        counterfactual_data_contract(config, risk_model_fingerprint)
    ]
    try:
        resolved = counterfactual_data_contract(
            config_with_sumo_installation_fingerprint(config),
            risk_model_fingerprint,
        )
    except Exception:
        resolved = None
    if resolved is not None and resolved not in candidates:
        candidates.append(resolved)
    return tuple(candidates)


def data_contract_hash(contract: dict[str, Any]) -> str:
    return stable_hash(contract)
