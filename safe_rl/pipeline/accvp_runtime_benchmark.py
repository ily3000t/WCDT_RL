from __future__ import annotations

import argparse
import os
import platform
import sys
import time
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
from safe_rl.accvp.contracts.schema import (
    file_sha256,
    read_json,
    stable_hash,
    write_json_atomic,
)
from safe_rl.pipeline.accvp_observation_preflight import _gate
from safe_rl.pipeline.common import make_env
from safe_rl.risk.risk_aggregator import aggregate_episode_reports
from safe_rl.rl.evaluation import validate_model_env_observation_shape
from safe_rl.rl.ppo import load_ppo
from safe_rl.utils.config import REPO_ROOT, load_config


RUNTIME_IMPLEMENTATION_VERSION = (
    "accvp_runtime_conflict_selector_cached_geometry_v5"
)


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (REPO_ROOT / value).resolve()


def _software_hardware() -> dict[str, Any]:
    torch_payload: dict[str, Any] = {}
    try:
        import torch

        torch_payload = {
            "torch_version": str(torch.__version__),
            "cuda_version": None if torch.version.cuda is None else str(torch.version.cuda),
            "cuda_available": bool(torch.cuda.is_available()),
            "torch_threads": int(torch.get_num_threads()),
            "torch_interop_threads": int(torch.get_num_interop_threads()),
        }
        if torch.cuda.is_available():
            torch_payload["cuda_device"] = str(torch.cuda.get_device_name(0))
    except ImportError:
        torch_payload = {"torch_version": None, "cuda_available": False}
    return {
        "python_version": sys.version,
        "numpy_version": str(np.__version__),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "process_id": os.getpid(),
        **torch_payload,
    }


def _artifact_lineage(
    cfg: Any,
    policy_model: Path | None,
    *,
    policy_type: str,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "policy_type": str(policy_type),
        "policy_identity": (
            "rule_gap_acceptance_v1"
            if policy_type == "rule_gap_acceptance"
            else "sb3_ppo_checkpoint_v1"
        ),
        "policy_model": "" if policy_model is None else str(policy_model),
        "policy_model_sha256": (
            "" if policy_model is None else file_sha256(policy_model)
        ),
    }
    for output_key, config_key in (
        ("accvp_predictor", "checkpoint"),
        ("accvp_calibration", "calibration_bundle"),
        ("accvp_operating_point", "operating_point"),
        ("accvp_manifest", "artifact_manifest"),
        ("risk_checkpoint", "risk_checkpoint"),
    ):
        configured = cfg.accvp.get(config_key)
        if not configured:
            result[output_key] = {"path": "", "sha256": ""}
            continue
        path = _resolve(configured)
        result[output_key] = {"path": str(path), "sha256": file_sha256(path)}
    return result


def _report_episode_seeds(reports: list[dict[str, Any]]) -> list[int]:
    seeds: list[int] = []
    for report in reports:
        value = report.get("seed", report.get("episode_seed"))
        if value is None:
            raise ValueError("runtime benchmark extension report has an episode without a seed")
        seeds.append(int(value))
    return seeds


def _stable_software_hardware(payload: dict[str, Any]) -> dict[str, Any]:
    """Remove the process id while retaining the latency-comparability host."""

    return {
        key: value
        for key, value in dict(payload).items()
        if key != "process_id"
    }


def _validate_failed_report_extension(
    payload: dict[str, Any],
    *,
    requested_seeds: list[int],
    expected_identity: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return reusable episodes only for an exact failed-report prefix.

    Runtime latency reports may be extended solely to meet the preregistered
    sample-size gate.  Policy, bundle, config, backend and runtime-contract
    lineage must remain identical, and a passing report is immutable.
    """

    if str(payload.get("artifact_kind", "")) != "accvp_runtime_benchmark_v1":
        raise ValueError("runtime benchmark extension requires a benchmark-v1 report")
    if bool(dict(payload.get("gate", {}) or {}).get("pass", False)):
        raise ValueError("a passing runtime benchmark report is immutable and cannot be extended")
    mismatches: list[str] = []
    for key, expected in expected_identity.items():
        actual = payload.get(key)
        if key == "software_hardware":
            actual = _stable_software_hardware(dict(actual or {}))
            expected = _stable_software_hardware(dict(expected or {}))
        if actual != expected:
            mismatches.append(key)
    if mismatches:
        raise ValueError(
            "runtime benchmark extension lineage mismatch: " + ", ".join(sorted(mismatches))
        )
    reports = [dict(report) for report in list(payload.get("episodes", []) or [])]
    reused_seeds = _report_episode_seeds(reports)
    if len(reused_seeds) != len(set(reused_seeds)):
        raise ValueError("runtime benchmark extension report contains duplicate episode seeds")
    if requested_seeds[: len(reused_seeds)] != reused_seeds:
        raise ValueError(
            "runtime benchmark extension seeds must preserve the existing report as an exact prefix"
        )
    workload = dict(payload.get("workload", {}) or {})
    expected_seed_hash = stable_hash({"episode_seeds": reused_seeds})
    if str(workload.get("requested_episode_seed_sha256", "")) != expected_seed_hash:
        raise ValueError("runtime benchmark extension requested-seed hash mismatch")
    if str(workload.get("observed_episode_seed_sha256", "")) != expected_seed_hash:
        raise ValueError("runtime benchmark extension observed-seed hash mismatch")
    if any("episode_reward" not in report for report in reports):
        raise ValueError("runtime benchmark extension report lacks per-episode rewards")
    if len(reused_seeds) == len(requested_seeds):
        raise ValueError("runtime benchmark extension has no new episode seeds to execute")
    return reports


def _failed_report_archive_path(output_path: Path, payload: dict[str, Any]) -> Path:
    fingerprint = str(payload.get("report_fingerprint", ""))
    if not fingerprint:
        fingerprint = stable_hash(payload)
    return (
        output_path.parent
        / "failed_attempts"
        / f"{output_path.stem}.{fingerprint[:16]}.json"
    )


def run(
    *,
    config_path: str | Path,
    policy_model: str | Path | None,
    seeds: list[int],
    output: str | Path,
    backend: str = "vectorized",
    device: str = "auto",
    policy_type: str = "sb3_ppo",
    extend_failed_report: bool = False,
    diagnostic_smoke: bool = False,
) -> Path:
    backend = str(backend).strip().lower()
    if backend not in {"reference", "vectorized"}:
        raise ValueError("runtime benchmark backend must be reference or vectorized")
    policy_type = str(policy_type).strip().lower()
    if policy_type not in {"sb3_ppo", "rule_gap_acceptance"}:
        raise ValueError(
            "runtime benchmark policy_type must be sb3_ppo or rule_gap_acceptance"
        )
    if policy_type == "sb3_ppo" and policy_model is None:
        raise ValueError("policy runtime benchmark requires --policy-model")
    if policy_type == "rule_gap_acceptance" and policy_model is not None:
        raise ValueError(
            "scorer preflight uses rule_gap_acceptance and must not receive a policy model"
        )
    requested_seeds = [int(seed) for seed in seeds]
    minimum_seed_count = 3 if diagnostic_smoke else 30
    if (
        len(requested_seeds) < minimum_seed_count
        or len(set(requested_seeds)) != len(requested_seeds)
    ):
        scope = "diagnostic smoke" if diagnostic_smoke else "formal runtime benchmark"
        raise ValueError(
            f"{scope} requires at least {minimum_seed_count} distinct episode seeds"
        )
    cfg = load_config(config_path)
    output_path = _resolve(output)
    bundle_manifest, _bundle_files = apply_v2_bundle_paths(cfg)
    if bundle_manifest is None:
        raise ValueError("formal runtime benchmark requires a bundle-v2 manifest")
    expected_runtime_contract, expected_runtime_contract_sha = (
        validate_manifest_runtime_contract(bundle_manifest)
    )
    if not RiskGatedACCVPCandidateTableAugmentor.enabled(cfg):
        raise ValueError("runtime benchmark requires accvp.observation.enabled=true")
    cfg.risk_module["candidate_geometry_backend"] = backend
    runtime_contract: dict[str, Any] | None = None
    if backend == "vectorized":
        runtime_contract = formal_runtime_contract_from_config(
            cfg,
            base_dir=REPO_ROOT,
        )
        runtime_contract_check = compare_formal_runtime_contracts(
            expected_runtime_contract,
            runtime_contract,
        )
    else:
        runtime_contract_check = {
            "pass": False,
            "expected_sha256": expected_runtime_contract_sha,
            "actual_sha256": "",
            "differing_fields": ["candidate_geometry_backend"],
            "diagnostic_backend": backend,
        }
    if policy_type == "sb3_ppo":
        model_path = _resolve(policy_model)  # type: ignore[arg-type]
        model = load_ppo(model_path, device=device)
        controller = None
        benchmark_scope = "policy_runtime"
    else:
        from safe_rl.baselines import RuleGapAcceptancePolicy

        model_path = None
        model = None
        controller = RuleGapAcceptancePolicy(cfg)
        benchmark_scope = (
            "scorer_latency_feasibility_smoke"
            if diagnostic_smoke
            else "scorer_preflight"
        )
    config_file = _resolve(config_path)
    feature_names_sha = stable_hash(
        {"feature_names": RiskGatedACCVPCandidateTableAugmentor.feature_names(cfg)}
    )
    feature_contract_hash = stable_hash(
        {
            "feature_names": RiskGatedACCVPCandidateTableAugmentor.feature_names(cfg),
            "observation": dict(cfg.accvp.get("observation", {}) or {}),
        }
    )
    artifact_lineage = _artifact_lineage(
        cfg,
        model_path,
        policy_type=policy_type,
    )
    software_hardware = _software_hardware()
    extension_source: dict[str, Any] | None = None
    expected_extension_identity = {
        "runtime_implementation_version": RUNTIME_IMPLEMENTATION_VERSION,
        "benchmark_scope": benchmark_scope,
        "policy_type": policy_type,
        "backend": backend,
        "formal_runtime_contract_sha256": str(
            runtime_contract_check.get("actual_sha256", "")
        ),
        "config": str(config_file),
        "config_file_sha256": file_sha256(config_file),
        "policy_model_sha256": (
            "" if model_path is None else file_sha256(model_path)
        ),
        "accvp_observation_feature_names_sha256": feature_names_sha,
        "accvp_observation_feature_contract_hash": feature_contract_hash,
        "artifact_lineage": artifact_lineage,
        "software_hardware": software_hardware,
    }
    prior_failed_implementation: dict[str, Any] | None = None
    if output_path.exists():
        if not extend_failed_report:
            raise FileExistsError(f"runtime benchmark report already exists: {output_path}")
        existing = read_json(output_path)
        existing_version = str(
            existing.get("runtime_implementation_version", "")
        )
        if existing_version != RUNTIME_IMPLEMENTATION_VERSION:
            if bool(dict(existing.get("gate", {}) or {}).get("pass", False)):
                raise ValueError(
                    "a passing report from another runtime implementation is "
                    "immutable and cannot be reused by Selector-v3"
                )
            declared = str(existing.get("report_fingerprint", ""))
            expected_fingerprint = stable_hash(
                {
                    key: value
                    for key, value in existing.items()
                    if key != "report_fingerprint"
                }
            )
            if not declared or declared != expected_fingerprint:
                raise ValueError(
                    "refusing to archive a failed runtime report with an "
                    "invalid fingerprint"
                )
            archive_path = _failed_report_archive_path(output_path, existing)
            if archive_path.exists():
                if read_json(archive_path) != existing:
                    raise FileExistsError(
                        "runtime benchmark failed-report archive collision: "
                        f"{archive_path}"
                    )
            else:
                write_json_atomic(archive_path, existing)
            prior_failed_implementation = {
                "runtime_implementation_version": (
                    existing_version or "legacy"
                ),
                "report_fingerprint": declared,
                "report_sha256": file_sha256(archive_path),
                "archived_source_report": str(archive_path),
                "episodes_reused": 0,
            }
            reports = []
            print(
                "[accvp_runtime_benchmark] "
                "action=archive_failed_implementation "
                f"from={existing_version or 'legacy'} "
                f"to={RUNTIME_IMPLEMENTATION_VERSION} "
                f"archive={archive_path}",
                flush=True,
            )
        else:
            extension_source = existing
            reports = _validate_failed_report_extension(
                extension_source,
                requested_seeds=requested_seeds,
                expected_identity=expected_extension_identity,
            )
    else:
        reports = []
    reused_episode_count = len(reports)
    rewards = [float(report["episode_reward"]) for report in reports]
    pending_seeds = requested_seeds[reused_episode_count:]
    benchmark_started = time.perf_counter()
    shape_env = make_env(cfg, seed=requested_seeds[0], shield_enabled=False)
    try:
        if model is not None:
            validate_model_env_observation_shape(model, shape_env, model_path)
            model_observation_shape = list(model.observation_space.shape)
        else:
            model_observation_shape = []
        env_observation_shape = list(shape_env.observation_space.shape)
    finally:
        shape_env.close()
    for seed in pending_seeds:
        env = make_env(cfg, seed=seed, shield_enabled=False)
        episode_reward = 0.0
        try:
            observation, _info = env.reset(seed=seed)
            terminated = truncated = False
            while not (terminated or truncated):
                if model is not None:
                    action, _state = model.predict(observation, deterministic=True)
                else:
                    action = int(controller.act(env.get_rule_control_context()).action)
                observation, reward, terminated, truncated, _info = env.step(int(action))
                episode_reward += float(reward)
            report = env.episode_report()
            report["seed"] = int(report.get("seed", report.get("episode_seed", seed)))
            report["episode_reward"] = episode_reward
            reports.append(report)
            rewards.append(episode_reward)
        finally:
            env.close()
    metrics = aggregate_episode_reports(reports)
    observed_seeds = _report_episode_seeds(reports)
    metrics["accvp_table_seed_schedule_match"] = observed_seeds == requested_seeds
    metrics["average_reward"] = float(np.mean(rewards)) if rewards else 0.0
    gate = _gate(
        metrics,
        require_vnext=True,
        runtime_contract_check=runtime_contract_check,
    )
    payload = {
        "artifact_kind": "accvp_runtime_benchmark_v1",
        "schema_version": 2,
        "evidence_role": "diagnostic_only" if diagnostic_smoke else "formal_gate",
        "hard_realtime_claim": False,
        "runtime_implementation_version": RUNTIME_IMPLEMENTATION_VERSION,
        "benchmark_scope": benchmark_scope,
        "policy_type": policy_type,
        "backend": backend,
        "timeout_contract": str(
            cfg.accvp.get("observation", {}).get(
                "timeout_contract", "soft_realtime_post_return_v1"
            )
        ),
        "soft_realtime_contract": not bool(
            cfg.accvp.get("observation", {}).get("full_table_hard_deadline_worker", False)
        ),
        "formal_runtime_contract": runtime_contract,
        "formal_runtime_contract_sha256": str(
            runtime_contract_check.get("actual_sha256", "")
        ),
        "formal_runtime_contract_check": runtime_contract_check,
        "config": str(config_file),
        "config_file_sha256": expected_extension_identity["config_file_sha256"],
        "config_hash": stable_hash(dict(cfg)),
        "policy_model_sha256": expected_extension_identity["policy_model_sha256"],
        "accvp_observation_feature_names_sha256": feature_names_sha,
        "accvp_observation_feature_contract_hash": feature_contract_hash,
        "artifact_lineage": artifact_lineage,
        "software_hardware": software_hardware,
        "workload": {
            "requested_episode_seed_count": len(requested_seeds),
            "requested_episode_seed_sha256": stable_hash({"episode_seeds": requested_seeds}),
            "observed_episode_seed_sha256": stable_hash({"episode_seeds": observed_seeds}),
            "actor_count": int(cfg.accvp.actor_count),
            "candidate_action_count": 9,
            "policy_type": policy_type,
            "risk_horizon_steps": int(
                cfg.accvp.get("observation", {}).get(
                    "risk_horizon_steps", cfg.risk_module.get("rollout_horizon_steps", 0)
                )
            ),
            "activation_decision_count": int(
                metrics.get("accvp_table_activation_window_decision_count", 0)
            ),
            "reused_episode_seed_count": reused_episode_count,
            "new_episode_seed_count": len(pending_seeds),
        },
        "model_observation_shape": model_observation_shape,
        "env_observation_shape": env_observation_shape,
        "wall_time_s": float(time.perf_counter() - benchmark_started),
        "metrics": metrics,
        "gate": gate,
        "episodes": reports,
    }
    if extension_source is not None:
        archive_path = _failed_report_archive_path(output_path, extension_source)
        payload["extension"] = {
            "mode": "exact_failed_report_prefix_v1",
            "source_report_fingerprint": str(
                extension_source.get("report_fingerprint", "")
            ),
            "source_gate_pass": False,
            "reused_episode_seed_count": reused_episode_count,
            "new_episode_seed_count": len(pending_seeds),
            "archived_source_report": str(archive_path),
        }
    if prior_failed_implementation is not None:
        payload["prior_failed_implementation"] = prior_failed_implementation
    payload["report_fingerprint"] = stable_hash(payload)
    if extension_source is not None:
        archive_path = _failed_report_archive_path(output_path, extension_source)
        if archive_path.exists():
            if read_json(archive_path) != extension_source:
                raise FileExistsError(
                    f"runtime benchmark failed-report archive collision: {archive_path}"
                )
        else:
            write_json_atomic(archive_path, extension_source)
    write_json_atomic(output_path, payload)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Formal ACCVP Candidate Table runtime benchmark")
    parser.add_argument("--config", required=True)
    parser.add_argument("--policy-model", default=None)
    parser.add_argument(
        "--policy-type",
        choices=("sb3_ppo", "rule_gap_acceptance"),
        default="sb3_ppo",
    )
    parser.add_argument(
        "--diagnostic-smoke",
        action="store_true",
        help=(
            "Run a small development-only latency feasibility sample. This "
            "never satisfies or replaces the formal runtime gate."
        ),
    )
    parser.add_argument("--seeds", nargs="+", required=True, type=int)
    parser.add_argument("--output", required=True)
    parser.add_argument("--backend", choices=("reference", "vectorized"), default="vectorized")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--extend-failed-report",
        action="store_true",
        help=(
            "Reuse an existing gate-failed report only when it is an exact "
            "lineage-matched prefix of the requested seed schedule."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = run(
        config_path=args.config,
        policy_model=args.policy_model,
        seeds=list(args.seeds),
        output=args.output,
        backend=args.backend,
        device=args.device,
        policy_type=args.policy_type,
        extend_failed_report=args.extend_failed_report,
        diagnostic_smoke=args.diagnostic_smoke,
    )
    print(f"[accvp_runtime_benchmark] report={path}")


if __name__ == "__main__":
    main()
