from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from safe_rl.accvp.contracts.schema import file_sha256, read_json
from safe_rl.evaluation_protocol import stable_hash
from safe_rl.pipeline import accvp_runtime_benchmark
from safe_rl.ppo_replicates import (
    MIN_FORMAL_OPTIMIZER_REPLICATES,
    REPLICATE_MANIFEST_KIND,
    observation_contract,
    validate_reward_semantics,
    write_json_new,
)
from safe_rl.utils.config import REPO_ROOT, load_config


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (REPO_ROOT / value).resolve()


def _number(metrics: dict[str, Any], key: str, *, default: float) -> float:
    value = metrics.get(key)
    return default if value is None else float(value)


def aggregate_runtime_reports(reports: list[dict[str, Any]]) -> dict[str, Any]:
    if not reports:
        raise ValueError("runtime replicate aggregation requires at least one report")
    metrics = [dict(report.get("metrics", {}) or {}) for report in reports]
    risk_p95 = [
        float(dict(item.get("accvp_table_latency_per_stage", {}) or {}).get("risk_secondary", {}).get("p95", 1.0e9))
        for item in metrics
    ]
    worst = {
        "min_fresh_valid_rate_activation_window": min(
            _number(item, "accvp_table_valid_rate_activation_window", default=0.0) for item in metrics
        ),
        "max_timeout_rate_activation_window": max(
            _number(item, "accvp_table_timeout_rate_activation_window", default=1.0) for item in metrics
        ),
        "max_bounded_stale_rate_activation_window": max(
            _number(item, "accvp_table_last_valid_fallback_rate_activation_window", default=1.0)
            for item in metrics
        ),
        "max_consecutive_timeout_count": max(
            int(item.get("accvp_table_max_consecutive_timeout_count", 1_000_000)) for item in metrics
        ),
        "max_risk_secondary_p95_s": max(risk_p95),
        "max_total_p95_s": max(
            _number(item, "accvp_table_latency_p95", default=1.0e9) for item in metrics
        ),
        "max_total_p99_s": max(
            _number(item, "accvp_table_latency_p99", default=1.0e9) for item in metrics
        ),
        "max_total_max_s": max(
            _number(item, "accvp_table_latency_max", default=1.0e9) for item in metrics
        ),
    }
    checks = {
        "all_individual_runtime_gates_pass": all(
            bool(report.get("gate", {}).get("pass", False)) for report in reports
        ),
        "minimum_five_optimizer_replicates": len(reports) >= MIN_FORMAL_OPTIMIZER_REPLICATES,
        "fresh_valid_rate": worst["min_fresh_valid_rate_activation_window"] >= 0.995,
        "timeout_rate": worst["max_timeout_rate_activation_window"] <= 0.005,
        "bounded_stale_rate": worst["max_bounded_stale_rate_activation_window"] <= 0.005,
        "consecutive_timeouts": worst["max_consecutive_timeout_count"] <= 1,
        "risk_secondary_p95": worst["max_risk_secondary_p95_s"] <= 0.15,
        "total_p95": worst["max_total_p95_s"] <= 0.30,
        "total_p99": worst["max_total_p99_s"] <= 0.40,
        "total_max": worst["max_total_max_s"] <= 0.50,
    }
    return {"worst_case": worst, "checks": checks, "pass": all(checks.values())}


def run(
    *,
    config_template: str | Path,
    replicate_manifest: str | Path,
    seeds: list[int],
    backend: str,
    output: str | Path,
    device: str = "auto",
) -> Path:
    source = _resolve(replicate_manifest)
    manifest = read_json(source)
    if str(manifest.get("artifact_kind", "")) != REPLICATE_MANIFEST_KIND:
        raise ValueError("unsupported PPO replicate manifest")
    rows = list(manifest.get("records", []) or [])
    if manifest.get("status") != "complete" or len(rows) < MIN_FORMAL_OPTIMIZER_REPLICATES:
        raise ValueError("policy runtime benchmark requires a complete five-replicate PPO manifest")
    requested_seeds = [int(seed) for seed in seeds]
    if len(requested_seeds) < 30 or len(requested_seeds) != len(set(requested_seeds)):
        raise ValueError("replicated runtime benchmark requires at least 30 distinct simulator seeds")
    template = load_config(_resolve(config_template))
    template_reward = validate_reward_semantics(template)["sha256"]
    template_observation = observation_contract(template, require_artifacts=True)["sha256"]
    checkpoint_hashes = [str(row.get("checkpoint_sha256", "")) for row in rows]
    if len(checkpoint_hashes) != len(set(checkpoint_hashes)):
        raise ValueError("runtime replicates must use distinct checkpoints")
    output_path = _resolve(output)
    report_dir = output_path.parent / "replicates"
    produced: list[dict[str, Any]] = []
    lineage: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda item: int(item["optimizer_seed"])):
        if str(row.get("reward_semantics_hash", "")) != template_reward:
            raise ValueError("replicate reward semantics disagree with runtime config template")
        if str(row.get("observation_contract_hash", "")) != template_observation:
            raise ValueError("replicate observation contract disagrees with runtime config template")
        seed = int(row["optimizer_seed"])
        report_path = report_dir / f"runtime_seed_{seed}.json"
        accvp_runtime_benchmark.run(
            config_path=row["resolved_config"],
            policy_model=row["checkpoint"],
            seeds=requested_seeds,
            output=report_path,
            backend=backend,
            device=device,
            policy_type="sb3_ppo",
        )
        report = read_json(report_path)
        if str(report.get("policy_model_sha256", "")) != str(row["checkpoint_sha256"]):
            raise ValueError("runtime report checkpoint hash disagrees with replicate manifest")
        produced.append(report)
        lineage.append(
            {
                "optimizer_seed": seed,
                "checkpoint_sha256": str(row["checkpoint_sha256"]),
                "report": str(report_path),
                "report_sha256": file_sha256(report_path),
            }
        )
    gate = aggregate_runtime_reports(produced)
    payload = {
        "artifact_kind": "accvp_runtime_benchmark_replicates_v1",
        "schema_version": 1,
        "backend": backend,
        "hard_real_time_claim": all(not bool(item.get("soft_realtime_contract", True)) for item in produced),
        "replicate_manifest": str(source),
        "replicate_manifest_sha256": file_sha256(source),
        "simulator_seeds": requested_seeds,
        "replicates": lineage,
        "gate": gate,
    }
    payload["report_fingerprint"] = stable_hash(payload)
    return write_json_new(output_path, payload)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the formal runtime gate over every PPO optimizer replicate")
    parser.add_argument("--config-template", required=True)
    parser.add_argument("--replicate-manifest", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--backend", choices=("reference", "vectorized"), default="vectorized")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    path = run(
        config_template=args.config_template,
        replicate_manifest=args.replicate_manifest,
        seeds=args.seeds,
        backend=args.backend,
        output=args.output,
        device=args.device,
    )
    print(f"accvp_runtime_benchmark_replicates={path}")


if __name__ == "__main__":
    main()
