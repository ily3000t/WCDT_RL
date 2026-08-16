"""Canonical online runtime contract and manifest comparison helpers."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping

from safe_rl.accvp.contracts.schema import file_sha256, stable_hash
from safe_rl.utils.config import REPO_ROOT


FORMAL_RUNTIME_CONTRACT_KIND = "accvp_formal_runtime_contract_v1"
FORMAL_RUNTIME_CONTRACT_SCHEMA_VERSION = 1
FORMAL_RUNTIME_FEATURE_VERSION = "risk_gated_candidate_table_v3_bounded_stale"
CANDIDATE_TABLE_SEMANTIC_CONTRACT_KIND = "accvp_candidate_table_semantic_contract_v1"
CLOSED_LOOP_EXECUTION_CONTRACT_KIND = "accvp_closed_loop_execution_contract_v1"
SOFT_REALTIME_POST_RETURN_CONTRACT = "soft_realtime_post_return_v1"
SIMULATION_BLOCKING_EXACT_CONTRACT = "simulation_blocking_exact_v1"
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def _plain_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    return dict(value)


def _resolve(path: str | Path, *, base_dir: str | Path | None = None) -> Path:
    value = Path(path)
    if value.is_absolute():
        return value.resolve()
    return (Path(base_dir or REPO_ROOT) / value).resolve()


def canonical_formal_runtime_contract(
    *,
    observation: Mapping[str, Any],
    candidate_geometry_backend: str,
    risk_checkpoint_sha256: str,
    risk_module_config_sha256: str,
) -> dict[str, Any]:
    """Build and validate the single formal Candidate-Table runtime contract."""

    obs = dict(observation)
    contract = {
        "artifact_kind": FORMAL_RUNTIME_CONTRACT_KIND,
        "schema_version": FORMAL_RUNTIME_CONTRACT_SCHEMA_VERSION,
        "candidate_geometry_backend": str(candidate_geometry_backend).strip().lower(),
        "observation_enabled": bool(obs.get("enabled", False)),
        "feature_version": str(obs.get("feature_version", "")),
        "activation_distance_m": float(obs.get("activation_distance", 0.0) or 0.0),
        "include_risk_secondary": bool(obs.get("include_risk_secondary", False)),
        "secondary_safety_profile": str(obs.get("secondary_safety_profile", "")).strip(),
        "risk_horizon_steps": int(obs.get("risk_horizon_steps", 0) or 0),
        "invalid_table_strategy": str(obs.get("invalid_table_strategy", "")),
        "fail_closed_defaults": bool(obs.get("fail_closed_defaults", False)),
        "timeout_s": float(obs.get("timeout_s", 0.0) or 0.0),
        "timeout_contract": str(obs.get("timeout_contract", "")),
        "full_table_hard_deadline_worker": bool(
            obs.get("full_table_hard_deadline_worker", False)
        ),
        "use_inference_worker": bool(obs.get("use_inference_worker", False)),
        "profile_latency": bool(obs.get("profile_latency", False)),
        "warmup_enabled": bool(obs.get("warmup_enabled", False)),
        "warmup_max_attempts": int(obs.get("warmup_max_attempts", 0) or 0),
        "invalid_table_dropout_rate": float(
            obs.get("invalid_table_dropout_rate", 0.0) or 0.0
        ),
        "last_valid_max_decisions": int(obs.get("last_valid_max_decisions", -1)),
        "last_valid_ttl_s": float(obs.get("last_valid_ttl_s", -1.0)),
        "last_valid_max_merge_distance_delta_m": float(
            obs.get("last_valid_max_merge_distance_delta_m", -1.0)
        ),
        "last_valid_max_ego_speed_delta_mps": float(
            obs.get("last_valid_max_ego_speed_delta_mps", -1.0)
        ),
        "last_valid_max_gap_delta_m": float(
            obs.get("last_valid_max_gap_delta_m", -1.0)
        ),
        "risk_checkpoint_sha256": str(risk_checkpoint_sha256).strip().lower(),
        "risk_module_config_sha256": str(risk_module_config_sha256).strip().lower(),
    }
    validate_formal_runtime_contract(contract)
    return contract


def validate_formal_runtime_contract(contract: Mapping[str, Any]) -> None:
    value = dict(contract)
    if str(value.get("artifact_kind", "")) != FORMAL_RUNTIME_CONTRACT_KIND:
        raise ValueError("formal runtime contract artifact_kind mismatch")
    if int(value.get("schema_version", -1)) != FORMAL_RUNTIME_CONTRACT_SCHEMA_VERSION:
        raise ValueError("formal runtime contract schema_version mismatch")
    if str(value.get("candidate_geometry_backend", "")) != "vectorized":
        raise ValueError("formal runtime contract requires vectorized candidate geometry")
    if not bool(value.get("observation_enabled", False)):
        raise ValueError("formal runtime contract requires observation enabled")
    if str(value.get("feature_version", "")) != FORMAL_RUNTIME_FEATURE_VERSION:
        raise ValueError("formal runtime contract requires the bounded-stale v3 feature version")
    if float(value.get("activation_distance_m", 0.0)) <= 0.0:
        raise ValueError("formal runtime contract requires a positive activation distance")
    if not bool(value.get("include_risk_secondary", False)):
        raise ValueError("formal runtime contract requires include_risk_secondary=true")
    if not str(value.get("secondary_safety_profile", "")).strip():
        raise ValueError("formal runtime contract requires secondary_safety_profile")
    if int(value.get("risk_horizon_steps", 0)) <= 0:
        raise ValueError("formal runtime contract requires a positive Risk-secondary horizon")
    if str(value.get("invalid_table_strategy", "")) != "bounded_last_valid_v2":
        raise ValueError("formal runtime contract requires bounded_last_valid_v2")
    if not bool(value.get("fail_closed_defaults", False)):
        raise ValueError("formal runtime contract requires fail_closed_defaults=true")
    timeout_s = float(value.get("timeout_s", 0.0))
    if timeout_s <= 0.0 or timeout_s > 0.5:
        raise ValueError("formal runtime contract timeout_s must be in (0, 0.5]")
    if str(value.get("timeout_contract", "")) != SOFT_REALTIME_POST_RETURN_CONTRACT:
        raise ValueError("formal runtime contract timeout_contract mismatch")
    if bool(value.get("full_table_hard_deadline_worker", True)):
        raise ValueError("formal runtime contract must not claim a hard-deadline worker")
    if bool(value.get("use_inference_worker", True)):
        raise ValueError("formal runtime contract requires the audited in-process backend")
    if not bool(value.get("profile_latency", False)):
        raise ValueError("formal runtime contract requires latency profiling")
    if not bool(value.get("warmup_enabled", False)):
        raise ValueError("formal runtime contract requires warmup_enabled=true")
    if int(value.get("warmup_max_attempts", 0)) <= 0:
        raise ValueError("formal runtime contract requires positive warmup_max_attempts")
    if float(value.get("invalid_table_dropout_rate", -1.0)) != 0.0:
        raise ValueError("formal runtime contract requires invalid_table_dropout_rate=0")
    if int(value.get("last_valid_max_decisions", -1)) != 1:
        raise ValueError("formal runtime contract requires exactly one bounded-stale decision")
    if not 0.0 < float(value.get("last_valid_ttl_s", -1.0)) <= 0.5:
        raise ValueError("formal runtime contract last_valid_ttl_s must be in (0, 0.5]")
    for key, upper_bound in (
        ("last_valid_max_merge_distance_delta_m", 15.0),
        ("last_valid_max_ego_speed_delta_mps", 3.0),
        ("last_valid_max_gap_delta_m", 8.0),
    ):
        numeric = float(value.get(key, -1.0))
        if numeric < 0.0 or numeric > upper_bound:
            raise ValueError(f"formal runtime contract {key} is outside its safe bound")
    if _SHA256_PATTERN.fullmatch(str(value.get("risk_checkpoint_sha256", ""))) is None:
        raise ValueError("formal runtime contract requires a valid Risk checkpoint SHA256")
    if _SHA256_PATTERN.fullmatch(str(value.get("risk_module_config_sha256", ""))) is None:
        raise ValueError("formal runtime contract requires a valid Risk-module config SHA256")


def formal_runtime_contract_sha256(contract: Mapping[str, Any]) -> str:
    validate_formal_runtime_contract(contract)
    return stable_hash(dict(contract))


def formal_runtime_contract_from_config(
    config: Any,
    *,
    declared: bool = False,
    base_dir: str | Path | None = None,
) -> dict[str, Any]:
    accvp = _plain_mapping(config.accvp)
    observation = _plain_mapping(
        accvp.get("formal_runtime_contract") if declared else accvp.get("observation")
    )
    if not observation:
        source = "accvp.formal_runtime_contract" if declared else "accvp.observation"
        raise ValueError(f"formal runtime contract source {source} is missing")
    risk_checkpoint = accvp.get("risk_checkpoint")
    if not risk_checkpoint:
        raise ValueError("formal runtime contract requires accvp.risk_checkpoint")
    checkpoint_path = _resolve(str(risk_checkpoint), base_dir=base_dir)
    backend = observation.pop(
        "candidate_geometry_backend",
        _plain_mapping(config.risk_module).get("candidate_geometry_backend", ""),
    )
    return canonical_formal_runtime_contract(
        observation=observation,
        candidate_geometry_backend=str(backend),
        risk_checkpoint_sha256=file_sha256(checkpoint_path),
        risk_module_config_sha256=stable_hash(_plain_mapping(config.risk_module)),
    )


def validate_manifest_runtime_contract(manifest: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    contract = dict(manifest.get("formal_runtime_contract", {}) or {})
    digest = formal_runtime_contract_sha256(contract)
    claimed = str(manifest.get("formal_runtime_contract_sha256", ""))
    if claimed != digest:
        raise ValueError("bundle formal_runtime_contract_sha256 mismatch")
    return contract, digest


def compare_formal_runtime_contracts(
    expected: Mapping[str, Any], actual: Mapping[str, Any]
) -> dict[str, Any]:
    expected_value = dict(expected)
    actual_value = dict(actual)
    expected_sha = formal_runtime_contract_sha256(expected_value)
    actual_sha = formal_runtime_contract_sha256(actual_value)
    differing_fields = sorted(
        key
        for key in set(expected_value) | set(actual_value)
        if expected_value.get(key) != actual_value.get(key)
    )
    return {
        "pass": expected_sha == actual_sha,
        "expected_sha256": expected_sha,
        "actual_sha256": actual_sha,
        "differing_fields": differing_fields,
    }


def candidate_table_semantic_contract(
    deployment_runtime_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Extract CPU-independent Candidate-Table semantics from a deployment contract.

    Deadline handling and bounded-stale policy are deliberately excluded.  The
    resulting identity can therefore be shared by blocking closed-loop method
    evaluation and a separate soft-realtime deployment benchmark.
    """

    value = dict(deployment_runtime_contract)
    validate_formal_runtime_contract(value)
    contract = {
        "artifact_kind": CANDIDATE_TABLE_SEMANTIC_CONTRACT_KIND,
        "schema_version": 1,
        "candidate_geometry_backend": str(value["candidate_geometry_backend"]),
        "observation_enabled": bool(value["observation_enabled"]),
        "feature_version": str(value["feature_version"]),
        "activation_distance_m": float(value["activation_distance_m"]),
        "include_risk_secondary": bool(value["include_risk_secondary"]),
        "secondary_safety_profile": str(value["secondary_safety_profile"]),
        "risk_horizon_steps": int(value["risk_horizon_steps"]),
        "fail_closed_defaults": bool(value["fail_closed_defaults"]),
        "risk_checkpoint_sha256": str(value["risk_checkpoint_sha256"]),
        "risk_module_config_sha256": str(value["risk_module_config_sha256"]),
    }
    return contract


def candidate_table_semantic_contract_sha256(
    deployment_runtime_contract: Mapping[str, Any],
) -> str:
    return stable_hash(candidate_table_semantic_contract(deployment_runtime_contract))


def observation_execution_contract(observation: Mapping[str, Any]) -> str:
    """Return the effective closed-loop execution contract with legacy fallback."""

    value = dict(observation)
    return str(
        value.get("execution_contract")
        or value.get("timeout_contract")
        or SOFT_REALTIME_POST_RETURN_CONTRACT
    )


def closed_loop_execution_contract(observation: Mapping[str, Any]) -> dict[str, Any]:
    """Build the wall-clock/observation contract used by PPO and Stage5."""

    value = dict(observation)
    execution = observation_execution_contract(value)
    if execution not in {
        SOFT_REALTIME_POST_RETURN_CONTRACT,
        SIMULATION_BLOCKING_EXACT_CONTRACT,
    }:
        raise ValueError(f"unsupported ACCVP execution_contract={execution!r}")
    deadline_s = float(
        value.get("deployment_deadline_s", value.get("timeout_s", 0.0)) or 0.0
    )
    if deadline_s <= 0.0:
        raise ValueError("ACCVP execution contract requires a positive deployment deadline")
    contract = {
        "artifact_kind": CLOSED_LOOP_EXECUTION_CONTRACT_KIND,
        "schema_version": 1,
        "execution_contract": execution,
        "deployment_deadline_s": deadline_s,
        "deadline_exceedance_changes_observation": (
            execution == SOFT_REALTIME_POST_RETURN_CONTRACT
        ),
        "profile_latency": bool(value.get("profile_latency", False)),
        "use_inference_worker": bool(value.get("use_inference_worker", False)),
        "invalid_table_strategy": str(value.get("invalid_table_strategy", "")),
        "fail_closed_defaults": bool(value.get("fail_closed_defaults", False)),
        "invalid_table_dropout_rate": float(
            value.get("invalid_table_dropout_rate", 0.0) or 0.0
        ),
    }
    if execution == SIMULATION_BLOCKING_EXACT_CONTRACT:
        if contract["deadline_exceedance_changes_observation"]:
            raise ValueError("blocking-exact execution cannot discard a completed table")
        if contract["invalid_table_strategy"] != "fail_closed_v1":
            raise ValueError("blocking-exact execution requires fail_closed_v1")
        if contract["use_inference_worker"]:
            raise ValueError("blocking-exact execution requires the synchronous in-process backend")
        if not contract["profile_latency"]:
            raise ValueError("blocking-exact execution requires latency profiling")
        if not contract["fail_closed_defaults"]:
            raise ValueError("blocking-exact execution requires fail_closed defaults")
        if contract["invalid_table_dropout_rate"] != 0.0:
            raise ValueError("blocking-exact formal evaluation forbids observation dropout")
    else:
        contract.update(
            {
                "last_valid_max_decisions": int(
                    value.get("last_valid_max_decisions", -1)
                ),
                "last_valid_ttl_s": float(value.get("last_valid_ttl_s", -1.0)),
                "last_valid_max_merge_distance_delta_m": float(
                    value.get("last_valid_max_merge_distance_delta_m", -1.0)
                ),
                "last_valid_max_ego_speed_delta_mps": float(
                    value.get("last_valid_max_ego_speed_delta_mps", -1.0)
                ),
                "last_valid_max_gap_delta_m": float(
                    value.get("last_valid_max_gap_delta_m", -1.0)
                ),
            }
        )
    return contract


def closed_loop_execution_contract_sha256(observation: Mapping[str, Any]) -> str:
    return stable_hash(closed_loop_execution_contract(observation))


def deployment_observation_from_formal_runtime_contract(
    contract: Mapping[str, Any],
    *,
    current_observation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Materialise the frozen deployment path without changing PPO config files."""

    value = dict(contract)
    validate_formal_runtime_contract(value)
    observation = dict(current_observation or {})
    observation.update(
        {
            "enabled": bool(value["observation_enabled"]),
            "feature_version": str(value["feature_version"]),
            "activation_distance": float(value["activation_distance_m"]),
            "include_risk_secondary": bool(value["include_risk_secondary"]),
            "secondary_safety_profile": str(value["secondary_safety_profile"]),
            "risk_horizon_steps": int(value["risk_horizon_steps"]),
            "invalid_table_strategy": str(value["invalid_table_strategy"]),
            "fail_closed_defaults": bool(value["fail_closed_defaults"]),
            "timeout_s": float(value["timeout_s"]),
            "deployment_deadline_s": float(value["timeout_s"]),
            "timeout_contract": str(value["timeout_contract"]),
            "execution_contract": SOFT_REALTIME_POST_RETURN_CONTRACT,
            "full_table_hard_deadline_worker": bool(
                value["full_table_hard_deadline_worker"]
            ),
            "use_inference_worker": bool(value["use_inference_worker"]),
            "profile_latency": bool(value["profile_latency"]),
            "warmup_enabled": bool(value["warmup_enabled"]),
            "warmup_max_attempts": int(value["warmup_max_attempts"]),
            "invalid_table_dropout_rate": float(value["invalid_table_dropout_rate"]),
            "last_valid_max_decisions": int(value["last_valid_max_decisions"]),
            "last_valid_ttl_s": float(value["last_valid_ttl_s"]),
            "last_valid_max_merge_distance_delta_m": float(
                value["last_valid_max_merge_distance_delta_m"]
            ),
            "last_valid_max_ego_speed_delta_mps": float(
                value["last_valid_max_ego_speed_delta_mps"]
            ),
            "last_valid_max_gap_delta_m": float(value["last_valid_max_gap_delta_m"]),
        }
    )
    return observation
