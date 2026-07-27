from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from safe_rl.accvp.contracts.artifacts import apply_v2_bundle_paths
from safe_rl.accvp.serving.observation import RiskGatedACCVPCandidateTableAugmentor
from safe_rl.accvp.contracts.runtime_contract import (
    compare_formal_runtime_contracts,
    formal_runtime_contract_from_config,
    validate_manifest_runtime_contract,
)
from safe_rl.accvp.contracts.schema import file_sha256, stable_hash
from safe_rl.pipeline.common import make_env
from safe_rl.risk.risk_aggregator import aggregate_episode_reports
from safe_rl.rl.evaluation import validate_model_env_observation_shape
from safe_rl.rl.ppo import load_ppo
from safe_rl.utils.config import REPO_ROOT, load_config
from safe_rl.utils.io import write_json


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Preflight Risk-gated ACCVP candidate-table observation availability")
    parser.add_argument("--config", required=True, help="Config with accvp.observation.enabled=true")
    parser.add_argument("--policy-model", required=True, help="PPO model used only to visit realistic states")
    parser.add_argument("--seeds", nargs="+", type=int, required=True, help="Episode seeds")
    parser.add_argument("--output", required=True, help="Output JSON report")
    parser.add_argument("--device", default="auto", help="SB3 device")
    return parser.parse_args()


def _gate(
    metrics: dict[str, Any],
    *,
    require_vnext: bool = False,
    runtime_contract_check: dict[str, Any] | None = None,
) -> dict[str, Any]:
    hard_fail_closed = int(
        metrics.get(
            "accvp_table_hard_fail_closed_count",
            metrics.get("accvp_table_fail_closed_count", 0),
        )
    )
    # Keep a compatibility path for callers that pass the historical minimal
    # metric fixture. Real aggregated reports always carry the vNext fields and
    # therefore use the strict gate below.
    strict = require_vnext or any(
        key in metrics
        for key in (
            "accvp_table_latency_p99",
            "accvp_table_timeout_rate_activation_window",
            "accvp_table_max_consecutive_timeout_count",
        )
    )
    if not strict:
        checks = {
            "valid_rate_activation_window": float(
                metrics.get("accvp_table_valid_rate_activation_window", 0.0)
            )
            >= 0.95,
            "hard_fail_closed_count_zero": hard_fail_closed == 0,
            "latency_p95_within_0_5s": (
                metrics.get("accvp_table_latency_p95") is not None
                and float(metrics.get("accvp_table_latency_p95", 1.0e9)) <= 0.5
            ),
        }
        return {
            "profile": "legacy_compatibility_v1",
            "pass": bool(all(checks.values())),
            "checks": checks,
            "required": {
                "valid_rate_activation_window": ">= 0.95",
                "hard_fail_closed_count": "0",
                "latency_p95_s": "<= 0.5",
            },
        }

    risk_secondary = dict(metrics.get("accvp_table_latency_per_stage", {}) or {}).get(
        "risk_secondary", {}
    )
    required_metric_fields = (
        "accvp_table_unique_episode_seed_count",
        "accvp_table_missing_episode_seed_count",
        "accvp_table_seed_schedule_match",
        "accvp_table_activation_window_decision_count",
        "accvp_table_valid_rate_activation_window",
        "accvp_table_timeout_rate_activation_window",
        "accvp_table_last_valid_fallback_rate_activation_window",
        "accvp_table_hard_fail_closed_count",
        "accvp_table_max_consecutive_timeout_count",
        "accvp_table_model_error_count",
        "accvp_table_invalid_bundle_count",
        "accvp_table_invalid_output_count",
        "accvp_table_runtime_context_error_count",
        "accvp_table_critical_actor_overflow_count",
        "accvp_table_unexpected_value_error_count",
        "accvp_table_runtime_error_reasons",
        "accvp_table_warmup_error_count",
        "accvp_table_warmup_ready_rate",
        "accvp_table_latency_p95",
        "accvp_table_latency_p99",
        "accvp_table_latency_max",
        "accvp_table_latency_per_stage",
    )
    checks = {
        "formal_runtime_contract_match": bool(
            runtime_contract_check is None
            or runtime_contract_check.get("pass", False)
        ),
        "required_metric_fields_present": all(
            field in metrics for field in required_metric_fields
        ),
        "minimum_unique_episode_seeds": int(
            metrics.get("accvp_table_unique_episode_seed_count", 0)
        )
        >= 30,
        "episode_seed_metadata_complete": int(
            metrics.get("accvp_table_missing_episode_seed_count", 1_000_000)
        )
        == 0,
        "episode_seed_schedule_matches": bool(
            metrics.get("accvp_table_seed_schedule_match", False)
        ),
        "minimum_activation_decisions": int(
            metrics.get("accvp_table_activation_window_decision_count", 0)
        )
        >= 1000,
        "fresh_valid_rate_activation_window": float(
            metrics.get("accvp_table_valid_rate_activation_window", 0.0)
        )
        >= 0.995,
        "timeout_rate_activation_window": float(
            metrics.get("accvp_table_timeout_rate_activation_window", 1.0)
        )
        <= 0.005,
        "bounded_stale_rate_activation_window": float(
            metrics.get("accvp_table_last_valid_fallback_rate_activation_window", 1.0)
        )
        <= 0.005,
        "hard_fail_closed_count_zero": hard_fail_closed == 0,
        "max_consecutive_timeouts": int(
            metrics.get("accvp_table_max_consecutive_timeout_count", 1_000_000)
        )
        <= 1,
        "model_error_count_zero": int(metrics.get("accvp_table_model_error_count", 0)) == 0,
        "invalid_bundle_count_zero": int(metrics.get("accvp_table_invalid_bundle_count", 0)) == 0,
        "invalid_output_count_zero": int(metrics.get("accvp_table_invalid_output_count", 0)) == 0,
        "runtime_context_error_count_zero": int(
            metrics.get("accvp_table_runtime_context_error_count", 0)
        )
        == 0,
        "critical_actor_overflow_count_zero": int(
            metrics.get("accvp_table_critical_actor_overflow_count", 0)
        )
        == 0,
        "unexpected_value_error_count_zero": int(
            metrics.get("accvp_table_unexpected_value_error_count", 0)
        )
        == 0,
        "warmup_error_count_zero": int(metrics.get("accvp_table_warmup_error_count", 0)) == 0,
        "warmup_ready": float(metrics.get("accvp_table_warmup_ready_rate", 0.0)) >= 1.0,
        "latency_p95_within_0_30s": (
            metrics.get("accvp_table_latency_p95") is not None
            and float(metrics.get("accvp_table_latency_p95", 1.0e9)) <= 0.30
        ),
        "latency_p99_within_0_40s": (
            metrics.get("accvp_table_latency_p99") is not None
            and float(metrics.get("accvp_table_latency_p99", 1.0e9)) <= 0.40
        ),
        "latency_max_within_0_50s": (
            metrics.get("accvp_table_latency_max") is not None
            and float(metrics.get("accvp_table_latency_max", 1.0e9)) <= 0.50
        ),
        "risk_secondary_p95_within_0_15s": (
            isinstance(risk_secondary, dict)
            and risk_secondary.get("p95") is not None
            and float(risk_secondary.get("p95", 1.0e9)) <= 0.15
        ),
    }
    return {
        "profile": "bounded_stale_runtime_v3_strict",
        "pass": bool(all(checks.values())),
        "checks": checks,
        "required": {
            "unique_episode_seeds": ">= 30",
            "missing_episode_seed_count": "0",
            "seed_schedule_match": "true",
            "activation_window_decisions": ">= 1000",
            "valid_rate_activation_window": ">= 0.995",
            "timeout_rate_activation_window": "<= 0.005",
            "last_valid_fallback_rate_activation_window": "<= 0.005",
            "hard_fail_closed_count": "0",
            "max_consecutive_timeouts": "<= 1",
            "model/bundle/output/context/overflow/unexpected/warmup_errors": "0",
            "warmup_ready_rate": "1.0",
            "latency_p95_s": "<= 0.30",
            "latency_p99_s": "<= 0.40",
            "latency_max_s": "<= 0.50",
            "risk_secondary_p95_s": "<= 0.15",
            "formal_runtime_contract": "bundle == benchmark/preflight config",
        },
    }


def run(
    *,
    config_path: str | Path,
    policy_model: str | Path,
    seeds: list[int],
    output: str | Path,
    device: str = "auto",
) -> Path:
    cfg = load_config(config_path)
    bundle_manifest, _bundle_files = apply_v2_bundle_paths(cfg)
    if not RiskGatedACCVPCandidateTableAugmentor.enabled(cfg):
        raise ValueError("preflight requires accvp.observation.enabled=true")
    observation_feature_version = str(
        cfg.accvp.get("observation", {}).get(
            "feature_version",
            RiskGatedACCVPCandidateTableAugmentor.FEATURE_VERSION,
        )
    )
    require_vnext = bool(
        observation_feature_version
        == RiskGatedACCVPCandidateTableAugmentor.BOUNDED_STALE_FEATURE_VERSION
    )
    requested_seeds = [int(seed) for seed in seeds]
    if not requested_seeds:
        raise ValueError("preflight requires at least one episode seed")
    if len(set(requested_seeds)) != len(requested_seeds):
        raise ValueError("preflight episode seeds must be distinct")
    if require_vnext and len(requested_seeds) < 30:
        raise ValueError("VNext preflight requires at least 30 distinct episode seeds")
    if require_vnext and str(
        cfg.risk_module.get("candidate_geometry_backend", "vectorized")
    ) != "vectorized":
        raise ValueError("VNext preflight requires vectorized candidate geometry backend")
    runtime_contract: dict[str, Any] | None = None
    runtime_contract_check: dict[str, Any] | None = None
    if require_vnext:
        if bundle_manifest is None:
            raise ValueError("VNext preflight requires a bundle-v2 artifact manifest")
        expected_contract, _expected_contract_sha = validate_manifest_runtime_contract(
            bundle_manifest
        )
        runtime_contract = formal_runtime_contract_from_config(
            cfg,
            base_dir=REPO_ROOT,
        )
        runtime_contract_check = compare_formal_runtime_contracts(
            expected_contract,
            runtime_contract,
        )
    seeds = requested_seeds
    feature_names = RiskGatedACCVPCandidateTableAugmentor.feature_names(cfg)
    feature_contract = {
        "feature_version": observation_feature_version,
        "feature_dim": int(RiskGatedACCVPCandidateTableAugmentor.feature_dim(cfg)),
        "feature_names": feature_names,
    }
    config_file = _resolve(config_path)
    model_path = _resolve(policy_model)
    configured_artifacts: dict[str, tuple[str, str]] = {}
    for payload_key, config_key in (
        ("accvp_checkpoint", "checkpoint"),
        ("accvp_calibration_bundle", "calibration_bundle"),
        ("accvp_operating_point", "operating_point"),
        ("accvp_artifact_manifest", "artifact_manifest"),
        ("risk_checkpoint", "risk_checkpoint"),
    ):
        configured_value = cfg.accvp.get(config_key)
        if not configured_value:
            if require_vnext:
                raise ValueError(
                    f"VNext preflight requires configured accvp.{config_key}"
                )
            configured_artifacts[payload_key] = ("", "")
            continue
        configured_path = _resolve(configured_value)
        configured_artifacts[payload_key] = (
            str(configured_path),
            file_sha256(configured_path),
        )
    model = load_ppo(model_path, device=device)
    reports: list[dict[str, Any]] = []
    rewards: list[float] = []
    shape_env = make_env(cfg, seed=int(seeds[0]), shield_enabled=False)
    try:
        validate_model_env_observation_shape(model, shape_env, model_path)
        model_observation_shape = list(model.observation_space.shape)
        env_observation_shape = list(shape_env.observation_space.shape)
    finally:
        shape_env.close()
    for seed in seeds:
        env = make_env(cfg, seed=int(seed), shield_enabled=False)
        total_reward = 0.0
        try:
            obs, _info = env.reset(seed=int(seed))
            terminated = truncated = False
            while not (terminated or truncated):
                action, _state = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _info = env.step(int(action))
                total_reward += float(reward)
            report = env.episode_report()
            report["episode_reward"] = total_reward
            reports.append(report)
            rewards.append(total_reward)
        finally:
            env.close()
    metrics = aggregate_episode_reports(reports)
    observed_seeds: list[int] = []
    for report in reports:
        raw_seed = report.get("seed", report.get("episode_seed"))
        try:
            if raw_seed is None or isinstance(raw_seed, bool):
                raise ValueError
            observed_seeds.append(int(raw_seed))
        except (TypeError, ValueError, OverflowError):
            continue
    metrics["accvp_table_seed_schedule_match"] = bool(
        sorted(observed_seeds) == sorted(requested_seeds)
    )
    metrics["average_reward"] = float(np.mean(rewards)) if rewards else 0.0
    gate = _gate(
        metrics,
        require_vnext=require_vnext,
        runtime_contract_check=runtime_contract_check,
    )
    payload = {
        "stage": "accvp_observation_preflight",
        "config": str(config_path),
        "config_file_sha256": file_sha256(config_file),
        "config_hash": stable_hash(dict(cfg)),
        "policy_model": str(model_path),
        "policy_model_sha256": file_sha256(model_path),
        "accvp_observation_feature_version": observation_feature_version,
        "accvp_observation_dim": int(feature_contract["feature_dim"]),
        "accvp_observation_feature_names": feature_names,
        "accvp_observation_feature_names_sha256": stable_hash(
            {"feature_names": feature_names}
        ),
        "accvp_observation_feature_contract_hash": stable_hash(feature_contract),
        "accvp_timeout_contract": str(
            cfg.accvp.get("observation", {}).get(
                "timeout_contract", "soft_realtime_post_return_v1"
            )
        ),
        "accvp_full_table_hard_deadline_worker": bool(
            cfg.accvp.get("observation", {}).get(
                "full_table_hard_deadline_worker", False
            )
        ),
        "formal_runtime_contract": runtime_contract,
        "formal_runtime_contract_sha256": (
            ""
            if runtime_contract_check is None
            else str(runtime_contract_check["actual_sha256"])
        ),
        "formal_runtime_contract_check": runtime_contract_check,
        "accvp_checkpoint": configured_artifacts["accvp_checkpoint"][0],
        "accvp_checkpoint_sha256": configured_artifacts["accvp_checkpoint"][1],
        "accvp_calibration_bundle": configured_artifacts["accvp_calibration_bundle"][0],
        "accvp_calibration_bundle_sha256": configured_artifacts["accvp_calibration_bundle"][1],
        "accvp_operating_point": configured_artifacts["accvp_operating_point"][0],
        "accvp_operating_point_sha256": configured_artifacts["accvp_operating_point"][1],
        "accvp_artifact_manifest": configured_artifacts["accvp_artifact_manifest"][0],
        "accvp_artifact_manifest_sha256": configured_artifacts["accvp_artifact_manifest"][1],
        "risk_checkpoint": configured_artifacts["risk_checkpoint"][0],
        "risk_checkpoint_sha256": configured_artifacts["risk_checkpoint"][1],
        "seeds": requested_seeds,
        "requested_seed_count": len(requested_seeds),
        "requested_seed_sha256": stable_hash({"episode_seeds": requested_seeds}),
        "observed_seed_count": int(metrics.get("accvp_table_unique_episode_seed_count", 0)),
        "observed_seed_sha256": stable_hash({"episode_seeds": sorted(observed_seeds)}),
        "model_observation_shape": model_observation_shape,
        "env_observation_shape": env_observation_shape,
        "metrics": metrics,
        "gate": gate,
        "episodes": reports,
    }
    output_path = _resolve(output)
    write_json(output_path, payload)
    print(f"[accvp_observation_preflight] report={output_path}")
    print(f"[accvp_observation_preflight] gate_pass={gate['pass']} metrics={metrics}")
    return output_path


def main() -> None:
    args = parse_args()
    run(
        config_path=args.config,
        policy_model=args.policy_model,
        seeds=[int(seed) for seed in args.seeds],
        output=args.output,
        device=str(args.device),
    )


if __name__ == "__main__":
    main()
