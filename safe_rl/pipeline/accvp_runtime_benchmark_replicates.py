from __future__ import annotations

import argparse
from collections import Counter
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


RUNTIME_REPLICATE_REPORT_KIND = "accvp_runtime_benchmark_replicates_v1"
RUNTIME_REPLICATE_REPORT_SCHEMA_VERSION = 1
SUPPORTED_REPLICATE_MANIFEST_KINDS = {
    REPLICATE_MANIFEST_KIND,
    # The main-method suite preserves the original training rows and their
    # hashes in a protocol-specific manifest.  Accepting it here avoids
    # manufacturing a second artifact that pretends the reused rows were
    # retrained under the new protocol.
    "main_method_ppo_method_manifest_v1",
}


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
    overflow_count = int(
        sum(
            int(item.get("accvp_table_critical_actor_overflow_count", 0))
            for item in metrics
        )
    )
    risk_coverage_incomplete_count = int(
        sum(
            int(
                item.get(
                    "accvp_table_risk_safety_actor_coverage_incomplete_count",
                    0,
                )
            )
            for item in metrics
        )
    )
    overflow_histogram: Counter[str] = Counter()
    overflow_examples: list[dict[str, Any]] = []
    for item in metrics:
        overflow_histogram.update(
            dict(item.get("accvp_table_critical_actor_overflow_histogram", {}) or {})
        )
        remaining = 20 - len(overflow_examples)
        if remaining > 0:
            overflow_examples.extend(
                dict(value)
                for value in list(
                    item.get("accvp_table_critical_actor_overflow_examples", []) or []
                )[:remaining]
            )
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
        "critical_actor_overflow_zero": overflow_count == 0,
        "risk_safety_actor_coverage_complete": (
            risk_coverage_incomplete_count == 0
        ),
    }
    return {
        "worst_case": worst,
        "critical_actor_overflow_count": overflow_count,
        "critical_actor_overflow_histogram": dict(overflow_histogram),
        "critical_actor_overflow_examples": overflow_examples,
        "critical_actor_overflow_sample_limit": 20,
        "risk_safety_actor_coverage_incomplete_count": (
            risk_coverage_incomplete_count
        ),
        "checks": checks,
        "pass": all(checks.values()),
    }


def _report_fingerprint(payload: dict[str, Any]) -> str:
    return stable_hash({key: value for key, value in payload.items() if key != "report_fingerprint"})


def runtime_replicate_request_fingerprint(
    *,
    config_template: Path,
    replicate_manifest: Path,
    manifest: dict[str, Any],
    seeds: list[int],
    backend: str,
    device: str,
) -> str:
    rows = list(manifest.get("records", []) or [])
    return stable_hash(
        {
            "artifact_kind": RUNTIME_REPLICATE_REPORT_KIND,
            "schema_version": RUNTIME_REPLICATE_REPORT_SCHEMA_VERSION,
            "runtime_implementation_version": (
                accvp_runtime_benchmark.RUNTIME_IMPLEMENTATION_VERSION
            ),
            "method_id": str(manifest.get("method_id", "")),
            "config_template": str(config_template),
            "config_template_sha256": file_sha256(config_template),
            "replicate_manifest": str(replicate_manifest),
            "replicate_manifest_sha256": file_sha256(replicate_manifest),
            "checkpoint_sha256s": [str(row.get("checkpoint_sha256", "")) for row in rows],
            "resolved_config_sha256s": [
                str(row.get("resolved_config_sha256", "")) for row in rows
            ],
            "simulator_seeds": [int(seed) for seed in seeds],
            "backend": str(backend),
            "device": str(device),
        }
    )


def _validate_single_runtime_report(
    report: dict[str, Any],
    *,
    row: dict[str, Any],
    requested_seeds: list[int],
    backend: str,
) -> None:
    if str(report.get("artifact_kind", "")) != "accvp_runtime_benchmark_v1":
        raise ValueError("existing runtime replicate has an unsupported artifact kind")
    if str(report.get("policy_type", "")) != "sb3_ppo":
        raise ValueError("existing runtime replicate is not an SB3 PPO policy benchmark")
    if str(report.get("runtime_implementation_version", "")) != (
        accvp_runtime_benchmark.RUNTIME_IMPLEMENTATION_VERSION
    ):
        raise ValueError("existing runtime replicate implementation version mismatch")
    if str(report.get("backend", "")) != str(backend):
        raise ValueError("existing runtime replicate backend disagrees with the request")
    if str(report.get("policy_model_sha256", "")) != str(row.get("checkpoint_sha256", "")):
        raise ValueError("existing runtime replicate checkpoint hash disagrees with the manifest")
    config_path = _resolve(str(row.get("resolved_config", "")))
    if str(report.get("config_file_sha256", "")) != file_sha256(config_path):
        raise ValueError("existing runtime replicate resolved-config hash disagrees with the manifest")
    for report_field, row_field in (
        (
            "candidate_table_semantic_contract_sha256",
            "candidate_table_semantic_contract_sha256",
        ),
        (
            "deployment_runtime_contract_sha256",
            "deployment_runtime_contract_sha256",
        ),
        (
            "policy_method_effect_execution_contract_sha256",
            "closed_loop_execution_contract_sha256",
        ),
    ):
        if str(report.get(report_field, "")) != str(row.get(row_field, "")):
            raise ValueError(
                f"existing runtime replicate {report_field} disagrees with the manifest"
            )
    if str(report.get("conclusion_scope", "")) != "deployment_runtime_only":
        raise ValueError("existing runtime replicate has an invalid conclusion scope")
    workload = dict(report.get("workload", {}) or {})
    expected_seed_hash = stable_hash({"episode_seeds": requested_seeds})
    if int(workload.get("requested_episode_seed_count", -1)) != len(requested_seeds):
        raise ValueError("existing runtime replicate seed count disagrees with the request")
    if str(workload.get("requested_episode_seed_sha256", "")) != expected_seed_hash:
        raise ValueError("existing runtime replicate seed schedule disagrees with the request")
    if str(workload.get("observed_episode_seed_sha256", "")) != expected_seed_hash:
        raise ValueError("existing runtime replicate did not observe the requested seed schedule")
    declared = str(report.get("report_fingerprint", ""))
    if not declared or declared != _report_fingerprint(report):
        raise ValueError("existing runtime replicate report fingerprint mismatch")


def validate_runtime_replicate_report(
    report: dict[str, Any],
    *,
    expected_request_fingerprint: str,
    replicate_manifest: Path,
    manifest: dict[str, Any],
    requested_seeds: list[int],
    backend: str,
    verify_child_reports: bool = True,
) -> dict[str, Any]:
    if str(report.get("artifact_kind", "")) != RUNTIME_REPLICATE_REPORT_KIND:
        raise ValueError("unsupported replicated runtime benchmark report")
    if int(report.get("schema_version", -1)) != RUNTIME_REPLICATE_REPORT_SCHEMA_VERSION:
        raise ValueError("unsupported replicated runtime benchmark report schema")
    if str(report.get("status", "")) != "complete":
        raise ValueError("replicated runtime benchmark report is not complete")
    if str(report.get("runtime_implementation_version", "")) != (
        accvp_runtime_benchmark.RUNTIME_IMPLEMENTATION_VERSION
    ):
        raise ValueError("replicated runtime benchmark implementation version mismatch")
    if str(report.get("request_fingerprint", "")) != expected_request_fingerprint:
        raise ValueError("replicated runtime benchmark request fingerprint mismatch")
    if str(report.get("replicate_manifest_sha256", "")) != file_sha256(replicate_manifest):
        raise ValueError("replicated runtime benchmark manifest hash mismatch")
    if [int(seed) for seed in report.get("simulator_seeds", [])] != requested_seeds:
        raise ValueError("replicated runtime benchmark seed schedule mismatch")
    rows = list(manifest.get("records", []) or [])
    checkpoint_hashes = [str(row.get("checkpoint_sha256", "")) for row in rows]
    if list(report.get("checkpoint_sha256s", []) or []) != checkpoint_hashes:
        raise ValueError("replicated runtime benchmark checkpoint set mismatch")
    expected_layered = {
        report_field: {
            str(row.get(row_field, "")) for row in rows
        }
        for report_field, row_field in (
            (
                "candidate_table_semantic_contract_sha256",
                "candidate_table_semantic_contract_sha256",
            ),
            (
                "deployment_runtime_contract_sha256",
                "deployment_runtime_contract_sha256",
            ),
            (
                "policy_method_effect_execution_contract_sha256",
                "closed_loop_execution_contract_sha256",
            ),
        )
    }
    if str(report.get("conclusion_scope", "")) != "deployment_runtime_only":
        raise ValueError("replicated runtime benchmark has an invalid conclusion scope")
    for field, values in expected_layered.items():
        if len(values) != 1 or "" in values or str(report.get(field, "")) != next(
            iter(values)
        ):
            raise ValueError(
                f"replicated runtime benchmark {field} disagrees with the manifest"
            )
    lineage = list(report.get("replicates", []) or [])
    if len(lineage) != len(rows):
        raise ValueError("replicated runtime benchmark lineage is incomplete")
    if verify_child_reports:
        rows_by_seed = {int(row["optimizer_seed"]): row for row in rows}
        for entry in lineage:
            seed = int(entry.get("optimizer_seed", -1))
            row = rows_by_seed.get(seed)
            if row is None:
                raise ValueError("replicated runtime benchmark contains an unknown optimizer seed")
            report_path = _resolve(str(entry.get("report", "")))
            if not report_path.is_file():
                raise FileNotFoundError(report_path)
            if str(entry.get("report_sha256", "")) != file_sha256(report_path):
                raise ValueError("replicated runtime child report hash mismatch")
            _validate_single_runtime_report(
                read_json(report_path),
                row=row,
                requested_seeds=requested_seeds,
                backend=backend,
            )
    declared = str(report.get("report_fingerprint", ""))
    if not declared or declared != _report_fingerprint(report):
        raise ValueError("replicated runtime benchmark report fingerprint mismatch")
    return report


def run(
    *,
    config_template: str | Path,
    replicate_manifest: str | Path,
    seeds: list[int],
    backend: str,
    output: str | Path,
    device: str = "auto",
    resume: bool = False,
) -> Path:
    source = _resolve(replicate_manifest)
    manifest = read_json(source)
    if str(manifest.get("artifact_kind", "")) not in SUPPORTED_REPLICATE_MANIFEST_KINDS:
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
    request_fingerprint = runtime_replicate_request_fingerprint(
        config_template=_resolve(config_template),
        replicate_manifest=source,
        manifest=manifest,
        seeds=requested_seeds,
        backend=backend,
        device=device,
    )
    if output_path.exists():
        if not resume:
            raise FileExistsError(output_path)
        validate_runtime_replicate_report(
            read_json(output_path),
            expected_request_fingerprint=request_fingerprint,
            replicate_manifest=source,
            manifest=manifest,
            requested_seeds=requested_seeds,
            backend=backend,
        )
        print(f"[accvp_runtime_replicates] status=complete action=skip report={output_path}", flush=True)
        return output_path
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
        if report_path.exists():
            if not resume:
                raise FileExistsError(report_path)
            _validate_single_runtime_report(
                read_json(report_path),
                row=row,
                requested_seeds=requested_seeds,
                backend=backend,
            )
            print(
                f"[accvp_runtime_replicates] optimizer_seed={seed} action=skip",
                flush=True,
            )
        else:
            print(
                f"[accvp_runtime_replicates] optimizer_seed={seed} action=run",
                flush=True,
            )
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
    semantic_hashes = {
        str(item.get("candidate_table_semantic_contract_sha256", ""))
        for item in produced
    }
    deployment_hashes = {
        str(item.get("deployment_runtime_contract_sha256", ""))
        for item in produced
    }
    method_effect_execution_hashes = {
        str(item.get("policy_method_effect_execution_contract_sha256", ""))
        for item in produced
    }
    if any(len(values) != 1 or "" in values for values in (
        semantic_hashes,
        deployment_hashes,
        method_effect_execution_hashes,
    )):
        raise ValueError("runtime replicates do not share the layered ACCVP contracts")
    payload = {
        "artifact_kind": RUNTIME_REPLICATE_REPORT_KIND,
        "schema_version": RUNTIME_REPLICATE_REPORT_SCHEMA_VERSION,
        "runtime_implementation_version": (
            accvp_runtime_benchmark.RUNTIME_IMPLEMENTATION_VERSION
        ),
        "status": "complete",
        "method_id": str(manifest.get("method_id", "")),
        "backend": backend,
        "device": str(device),
        "hard_real_time_claim": all(not bool(item.get("soft_realtime_contract", True)) for item in produced),
        "conclusion_scope": "deployment_runtime_only",
        "candidate_table_semantic_contract_sha256": next(iter(semantic_hashes)),
        "deployment_runtime_contract_sha256": next(iter(deployment_hashes)),
        "policy_method_effect_execution_contract_sha256": next(
            iter(method_effect_execution_hashes)
        ),
        "config_template": str(_resolve(config_template)),
        "config_template_sha256": file_sha256(_resolve(config_template)),
        "replicate_manifest": str(source),
        "replicate_manifest_sha256": file_sha256(source),
        "checkpoint_sha256s": checkpoint_hashes,
        "simulator_seeds": requested_seeds,
        "replicates": lineage,
        "gate": gate,
        "request_fingerprint": request_fingerprint,
    }
    payload["report_fingerprint"] = _report_fingerprint(payload)
    return write_json_new(output_path, payload)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the formal runtime gate over every PPO optimizer replicate")
    parser.add_argument("--config-template", required=True)
    parser.add_argument("--replicate-manifest", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--backend", choices=("reference", "vectorized"), default="vectorized")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse only immutable per-checkpoint reports whose complete lineage matches the request",
    )
    args = parser.parse_args()
    path = run(
        config_template=args.config_template,
        replicate_manifest=args.replicate_manifest,
        seeds=args.seeds,
        backend=args.backend,
        output=args.output,
        device=args.device,
        resume=args.resume,
    )
    print(f"accvp_runtime_benchmark_replicates={path}")


if __name__ == "__main__":
    main()
