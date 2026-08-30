from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml

from safe_rl.evaluation_protocol import protocol_snapshot
from safe_rl.ppo_replicates import observation_contract, plain, validate_reward_semantics
from safe_rl.utils.config import REPO_ROOT, load_config


PROTOCOL_KIND = "accvp_main_method_experiment_v1"
PROTOCOL_SCHEMA_VERSION = 1

RULE_METHOD_ID = "rule_gap_acceptance_idm_style_v1"
NO_FORECAST_METHOD_ID = "ppo_no_forecast_reward_v2"
CV_METHOD_ID = "ppo_constant_velocity_reward_v2"
WCDT_V1_METHOD_ID = "ppo_wcdt_v1_adapted_reward_v2"
WCDT_V3_METHOD_ID = "wcdt_reward_v2"
FINAL_METHOD_ID = "candidate_table_reward_v3_1_commitment"

EXPECTED_METHOD_ROLES = {
    RULE_METHOD_ID: "deterministic_rule_baseline",
    NO_FORECAST_METHOD_ID: "no_prediction_control",
    CV_METHOD_ID: "constant_velocity_prediction_control",
    WCDT_V1_METHOD_ID: "published_architecture_adapted_baseline",
    WCDT_V3_METHOD_ID: "matched_backbone_direct_baseline",
    FINAL_METHOD_ID: "final_complete_method",
}

FORECAST_ATTRIBUTION_METHODS = (
    NO_FORECAST_METHOD_ID,
    CV_METHOD_ID,
    WCDT_V1_METHOD_ID,
    WCDT_V3_METHOD_ID,
)


def resolve_path(path: str | Path, *, relative_to: Path | None = None) -> Path:
    value = Path(path)
    if value.is_absolute():
        return value.resolve()
    return ((relative_to or REPO_ROOT) / value).resolve()


def _mapping(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping")
    return plain(value)


def _positive_int(value: Any, *, field: str) -> int:
    result = int(value)
    if result <= 0:
        raise ValueError(f"{field} must be positive")
    return result


def executor_contract(config: Any) -> dict[str, Any]:
    training = config.get("training", {}) or {}
    rl = config.get("rl", {}) or {}
    stage3 = config.get("stage3", {}) or {}
    num_envs = _positive_int(training.get("ppo_num_envs", 1), field="training.ppo_num_envs")
    n_steps = _positive_int(rl.get("n_steps", 0), field="rl.n_steps")
    return {
        "ppo_num_envs": num_envs,
        "n_steps": n_steps,
        "rollout_size": num_envs * n_steps,
        "batch_size": _positive_int(rl.get("batch_size", 0), field="rl.batch_size"),
        "total_timesteps": _positive_int(
            rl.get("total_timesteps", 0), field="rl.total_timesteps"
        ),
        "ppo_worker_torch_threads": _positive_int(
            training.get("ppo_worker_torch_threads", 1),
            field="training.ppo_worker_torch_threads",
        ),
        "ppo_main_torch_threads": _positive_int(
            training.get("ppo_main_torch_threads", 1),
            field="training.ppo_main_torch_threads",
        ),
        "checkpoint_selection_workers": _positive_int(
            stage3.get("checkpoint_selection_workers", 1),
            field="stage3.checkpoint_selection_workers",
        ),
        "checkpoint_selection_worker_torch_threads": _positive_int(
            stage3.get("checkpoint_selection_worker_torch_threads", 1),
            field="stage3.checkpoint_selection_worker_torch_threads",
        ),
        "checkpoint_selection_start_method": str(
            stage3.get("checkpoint_selection_start_method", "spawn")
        ),
    }


def forecast_identity(config: Any) -> dict[str, Any]:
    forecast = config.get("forecast_features", {}) or {}
    enabled = bool(forecast.get("enabled", False))
    return {
        "enabled": enabled,
        "source": str(forecast.get("source", "")) if enabled else "",
        "checkpoint": str(forecast.get("checkpoint", "")) if enabled else "",
        "allow_heuristic_fallback": bool(
            forecast.get("allow_heuristic_fallback", False)
        ),
    }


def load_protocol(path: str | Path, *, verify_artifacts: bool = False) -> dict[str, Any]:
    source = resolve_path(path)
    with source.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    protocol = _mapping(payload, field="main-method protocol")
    if protocol.get("artifact_kind") != PROTOCOL_KIND:
        raise ValueError("unsupported main-method experiment protocol")
    if int(protocol.get("schema_version", -1)) != PROTOCOL_SCHEMA_VERSION:
        raise ValueError("unsupported main-method experiment protocol schema")
    protocol_id = str(protocol.get("protocol_id", "")).strip()
    if not protocol_id:
        raise ValueError("main-method protocol_id is required")

    seeds = [int(seed) for seed in protocol.get("optimizer_seeds", [])]
    if len(seeds) < 5 or len(seeds) != len(set(seeds)):
        raise ValueError("main-method protocol requires at least five unique optimizer seeds")
    if seeds != sorted(seeds):
        raise ValueError("main-method optimizer seeds must be sorted")

    methods = _mapping(protocol.get("methods", {}), field="methods")
    roles = {str(method_id): str(item.get("role", "")) for method_id, item in methods.items()}
    if roles != EXPECTED_METHOD_ROLES:
        raise ValueError(
            "main-method protocol must declare the frozen six-method table: "
            f"declared={roles!r} expected={EXPECTED_METHOD_ROLES!r}"
        )

    preferred_rollout = _positive_int(
        protocol.get("ppo", {}).get("rollout_size", 0), field="ppo.rollout_size"
    )
    reward_hashes: set[str] = set()
    configs: dict[str, dict[str, Any]] = {}
    executor_contracts: dict[str, dict[str, Any]] = {}
    observation_contracts: dict[str, dict[str, Any]] = {}
    for method_id, raw in methods.items():
        method = _mapping(raw, field=f"methods.{method_id}")
        policy_type = str(method.get("policy_type", "sb3_ppo"))
        if method_id == RULE_METHOD_ID:
            if policy_type != "rule_gap_acceptance" or bool(method.get("train", False)):
                raise ValueError("rule method must be deterministic and non-training")
            continue
        if policy_type != "sb3_ppo":
            raise ValueError(f"{method_id}: learning method must use sb3_ppo")
        config_value = str(method.get("config", "")).strip()
        manifest_value = str(method.get("manifest", "")).strip()
        if not config_value or not manifest_value:
            raise ValueError(f"{method_id}: config and manifest are required")
        config_path = resolve_path(config_value)
        cfg = load_config(config_path)
        if str(cfg.evaluation_protocol.get("protocol_id", "")) != protocol_id and bool(
            method.get("train", False)
        ):
            raise ValueError(f"{method_id}: training config protocol_id mismatch")
        if str(cfg.get("experiment", {}).get("method_id", "")) != method_id:
            raise ValueError(f"{method_id}: config experiment.method_id mismatch")
        identity = forecast_identity(cfg)
        expected_source = str(method.get("forecast_source", ""))
        if identity["source"] != expected_source:
            raise ValueError(
                f"{method_id}: forecast source mismatch: "
                f"config={identity['source']!r} protocol={expected_source!r}"
            )
        if identity["allow_heuristic_fallback"]:
            raise ValueError(f"{method_id}: heuristic forecast fallback is forbidden")
        executor = executor_contract(cfg)
        if executor["rollout_size"] != preferred_rollout:
            raise ValueError(f"{method_id}: PPO rollout budget differs from the protocol")
        reward = validate_reward_semantics(cfg)
        observation = observation_contract(cfg, require_artifacts=verify_artifacts)
        configs[method_id] = {
            "path": str(config_path),
            "config": cfg,
            "reward": reward,
            "observation": observation,
        }
        executor_contracts[method_id] = executor
        observation_contracts[method_id] = observation
        if method_id in FORECAST_ATTRIBUTION_METHODS:
            reward_hashes.add(str(reward["sha256"]))
            if bool(cfg.get("accvp", {}).get("observation", {}).get("enabled", False)):
                raise ValueError(f"{method_id}: forecast-attribution baseline enables ACCVP")
            if bool(
                cfg.get("rl", {})
                .get("policy_lateral_commitment", {})
                .get("enabled", False)
            ):
                raise ValueError(f"{method_id}: forecast-attribution baseline enables commitment")
    if len(reward_hashes) != 1:
        raise ValueError("forecast-attribution methods must share one reward contract")

    evaluation = _mapping(protocol.get("evaluation", {}), field="evaluation")
    secondary = _positive_int(
        evaluation.get("secondary_simulator_seed_count", 0),
        field="evaluation.secondary_simulator_seed_count",
    )
    primary = _positive_int(
        evaluation.get("primary_simulator_seed_count", 0),
        field="evaluation.primary_simulator_seed_count",
    )
    if secondary > primary:
        raise ValueError("secondary Stage5 seed count exceeds primary seed count")
    protocol_config = load_config(resolve_path(str(protocol["evaluation_config"])))
    snapshot = protocol_snapshot(protocol_config)
    if snapshot["protocol_id"] != protocol_id or not snapshot["strict"]:
        raise ValueError("evaluation_config does not bind the strict main-method protocol")
    role = str(evaluation.get("seed_role", "stage5_confirmatory"))
    cohort = snapshot["cohort_roles"].get(role, role)
    available = list(snapshot["cohorts"].get(cohort, []))
    if len(available) < primary:
        raise ValueError("evaluation protocol does not contain enough Stage5 seeds")

    protocol["_source"] = str(source)
    protocol["_configs"] = configs
    protocol["_executor_contracts"] = executor_contracts
    protocol["_observation_contracts"] = observation_contracts
    protocol["_protocol_snapshot"] = snapshot
    protocol["_stage5_seeds"] = [int(seed) for seed in available[:primary]]
    return protocol


def public_protocol(protocol: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: plain(value)
        for key, value in protocol.items()
        if not str(key).startswith("_")
    }
