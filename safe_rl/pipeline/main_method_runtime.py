from __future__ import annotations

import argparse
import os
import platform
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from safe_rl.accvp.contracts.schema import (
    file_sha256,
    read_json,
    stable_hash,
    write_json_atomic,
)
from safe_rl.main_method_protocol import (
    FINAL_METHOD_ID,
    NO_FORECAST_METHOD_ID,
    RULE_METHOD_ID,
    load_protocol,
    resolve_path,
)
from safe_rl.pipeline import accvp_runtime_benchmark_replicates
from safe_rl.pipeline.main_method_ppo_suite import METHOD_MANIFEST_KIND, SUITE_MANIFEST_KIND
from safe_rl.rl.evaluation import evaluate_policy
from safe_rl.utils.config import load_config


REQUEST_KIND = "main_method_deployment_runtime_request_v1"
METHOD_REPORT_KIND = "main_method_deployment_runtime_method_v1"
REPORT_KIND = "main_method_deployment_runtime_aggregate_v1"


def _fingerprint(payload: Mapping[str, Any], field: str) -> str:
    return stable_hash({key: value for key, value in payload.items() if key != field})


def _validate_fingerprint(payload: Mapping[str, Any], field: str, name: str) -> None:
    declared = str(payload.get(field, ""))
    if not declared or declared != _fingerprint(payload, field):
        raise ValueError(f"{name} fingerprint mismatch")


def _write_new_or_same(path: Path, payload: Mapping[str, Any]) -> Path:
    ready = dict(payload)
    if path.exists():
        if read_json(path) != ready:
            raise FileExistsError(f"refusing to replace a different artifact: {path}")
        return path.resolve()
    write_json_atomic(path, ready)
    return path.resolve()


def _load_suite(path: str | Path) -> tuple[Path, dict[str, Any], dict[str, dict[str, Any]]]:
    source = resolve_path(path)
    suite = read_json(source)
    _validate_fingerprint(suite, "manifest_fingerprint", "main-method PPO suite")
    if suite.get("artifact_kind") != SUITE_MANIFEST_KIND or suite.get("status") != "complete":
        raise ValueError("main-method PPO suite is not complete")
    manifests: dict[str, dict[str, Any]] = {}
    for method_id, binding in dict(suite.get("methods", {}) or {}).items():
        if str(binding.get("policy_type", "")) == "rule_gap_acceptance":
            continue
        manifest_path = resolve_path(str(binding["manifest"]))
        if file_sha256(manifest_path) != str(binding["manifest_sha256"]):
            raise ValueError(f"{method_id}: method manifest hash changed")
        manifest = read_json(manifest_path)
        _validate_fingerprint(manifest, "manifest_fingerprint", f"{method_id} manifest")
        if (
            manifest.get("artifact_kind") != METHOD_MANIFEST_KIND
            or manifest.get("status") != "complete"
            or str(manifest.get("method_id", "")) != str(method_id)
        ):
            raise ValueError(f"{method_id}: invalid method manifest")
        manifests[str(method_id)] = manifest
    return source, suite, manifests


def _software_hardware() -> dict[str, Any]:
    result: dict[str, Any] = {
        "python_version": sys.version,
        "numpy_version": str(np.__version__),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "process_id": os.getpid(),
    }
    try:
        import torch

        result.update(
            {
                "torch_version": str(torch.__version__),
                "cuda_version": None if torch.version.cuda is None else str(torch.version.cuda),
                "cuda_available": bool(torch.cuda.is_available()),
                "torch_threads": int(torch.get_num_threads()),
                "torch_interop_threads": int(torch.get_num_interop_threads()),
            }
        )
        if torch.cuda.is_available():
            result["cuda_device"] = str(torch.cuda.get_device_name(0))
    except ImportError:
        result.update({"torch_version": None, "cuda_available": False})
    return result


def _summary(values: list[float]) -> dict[str, float | int | None]:
    finite = np.asarray([float(value) for value in values if np.isfinite(value)], dtype=np.float64)
    if finite.size == 0:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "p99": None, "max": None}
    return {
        "count": int(finite.size),
        "mean": float(np.mean(finite)),
        "p50": float(np.percentile(finite, 50)),
        "p95": float(np.percentile(finite, 95)),
        "p99": float(np.percentile(finite, 99)),
        "max": float(np.max(finite)),
    }


def _generic_metrics(episodes: list[Mapping[str, Any]]) -> dict[str, Any]:
    episode_wall: list[float] = []
    per_step_wall: list[float] = []
    steps_per_second: list[float] = []
    operation_samples: dict[str, list[float]] = {}
    for episode in episodes:
        performance = dict(episode.get("performance", {}) or {})
        wall = float(performance.get("wall_time", 0.0))
        steps = int(episode.get("steps", 0))
        episode_wall.append(wall)
        if steps > 0:
            per_step_wall.append(wall / steps)
        steps_per_second.append(float(performance.get("steps_per_second", 0.0)))
        for name, samples in dict(performance.get("operation_samples_s", {}) or {}).items():
            operation_samples.setdefault(str(name), []).extend(float(value) for value in samples)
    return {
        "episode_wall_time_s": _summary(episode_wall),
        "amortized_wall_time_per_simulation_step_s": _summary(per_step_wall),
        "steps_per_second": _summary(steps_per_second),
        "operation_latency_s": {
            name: _summary(values) for name, values in sorted(operation_samples.items())
        },
        "interpretation": {
            "operation_samples": "per instrumented call",
            "amortized_wall_time_per_simulation_step": (
                "episode wall time divided by episode simulation steps; not a hard-deadline sample"
            ),
            "gate": "descriptive_only_unless_a_method_specific_threshold_is_preregistered",
        },
    }


def _risk_checkpoint(config: Mapping[str, Any]) -> str:
    value = str(
        dict(dict(config.get("rl", {}) or {}).get("shield_guided_reward", {}) or {}).get(
            "risk_checkpoint", ""
        )
    )
    path = resolve_path(value)
    if not value or not path.is_file():
        raise FileNotFoundError(path)
    return str(path)


def _runtime_seeds(protocol: Mapping[str, Any]) -> list[int]:
    runtime = dict(protocol.get("deployment_runtime", {}) or {})
    role = str(runtime.get("seed_role", "runtime_confirmatory"))
    snapshot = dict(protocol["_protocol_snapshot"])
    cohort_name = str(dict(snapshot.get("cohort_roles", {}) or {}).get(role, role))
    values = [
        int(value)
        for value in list(dict(snapshot.get("cohorts", {}) or {}).get(cohort_name, []) or [])
    ]
    expected = int(runtime.get("simulator_seed_count", 60))
    if len(values) != expected or len(values) != len(set(values)):
        raise ValueError("runtime_confirmatory seed cohort does not match the frozen runtime request")
    return values


def prepare(*, protocol_path: str | Path, suite_manifest: str | Path, output_root: str | Path) -> Path:
    protocol = load_protocol(protocol_path, verify_artifacts=True)
    suite_path, suite, manifests = _load_suite(suite_manifest)
    if str(suite.get("protocol_id", "")) != str(protocol["protocol_id"]):
        raise ValueError("PPO suite and deployment-runtime protocol disagree")
    output = resolve_path(output_root)
    methods: dict[str, Any] = {}
    for method_id, method in protocol["methods"].items():
        if method_id == RULE_METHOD_ID:
            rule_config = resolve_path(
                str(protocol["methods"][NO_FORECAST_METHOD_ID]["config"])
            )
            methods[method_id] = {
                "runner": "generic_closed_loop_workload",
                "policy_type": "rule_gap_acceptance",
                "config": str(rule_config),
                "config_sha256": file_sha256(rule_config),
                "manifest": "",
                "manifest_sha256": "",
            }
            continue
        manifest_path = resolve_path(str(method["manifest"]))
        methods[method_id] = {
            "runner": (
                "accvp_candidate_table_runtime_gate"
                if method_id == FINAL_METHOD_ID
                else "generic_closed_loop_workload"
            ),
            "policy_type": "sb3_ppo",
            "config": str(resolve_path(str(method["config"]))),
            "config_sha256": file_sha256(resolve_path(str(method["config"]))),
            "manifest": str(manifest_path),
            "manifest_sha256": file_sha256(manifest_path),
            "checkpoint_sha256s": [
                str(row["checkpoint_sha256"])
                for row in list(manifests[method_id].get("records", []) or [])
            ],
        }
    payload: dict[str, Any] = {
        "artifact_kind": REQUEST_KIND,
        "schema_version": 1,
        "status": "ready",
        "protocol_id": str(protocol["protocol_id"]),
        "protocol": str(resolve_path(protocol_path)),
        "protocol_sha256": file_sha256(resolve_path(protocol_path)),
        "suite_manifest": str(suite_path),
        "suite_manifest_sha256": file_sha256(suite_path),
        "conclusion_scope": "deployment_runtime_only",
        "shield_enabled": bool(dict(protocol["deployment_runtime"]).get("shield_enabled", True)),
        "simulator_seeds": _runtime_seeds(protocol),
        "methods": methods,
        "output_root": str(output),
    }
    payload["request_fingerprint"] = _fingerprint(payload, "request_fingerprint")
    return _write_new_or_same(output / "deployment_runtime_request.json", payload)


def _validate_child(path: Path, expected: str) -> dict[str, Any]:
    report = read_json(path)
    if report.get("artifact_kind") != METHOD_REPORT_KIND or report.get("status") != "complete":
        raise ValueError(f"invalid generic deployment-runtime report: {path}")
    _validate_fingerprint(report, "report_fingerprint", str(path))
    if str(report.get("request_fingerprint", "")) != expected:
        raise ValueError(f"deployment-runtime child request mismatch: {path}")
    return report


def _run_generic(
    *,
    method_id: str,
    method: Mapping[str, Any],
    manifest: Mapping[str, Any] | None,
    seeds: list[int],
    shield_enabled: bool,
    output: Path,
    resume: bool,
) -> Path:
    rows = (
        [{"optimizer_seed": None, "checkpoint": "", "checkpoint_sha256": ""}]
        if manifest is None
        else sorted(list(manifest.get("records", []) or []), key=lambda row: int(row["optimizer_seed"]))
    )
    children: list[dict[str, Any]] = []
    for row in rows:
        optimizer_seed = row.get("optimizer_seed")
        config_path = resolve_path(
            str(method["config"] if manifest is None else row["resolved_config"])
        )
        checkpoint = "" if manifest is None else str(resolve_path(str(row["checkpoint"])))
        identity = {
            "method_id": method_id,
            "optimizer_seed": optimizer_seed,
            "config_sha256": file_sha256(config_path),
            "checkpoint_sha256": "" if manifest is None else str(row["checkpoint_sha256"]),
            "simulator_seeds": seeds,
            "shield_enabled": shield_enabled,
            "policy_type": str(method["policy_type"]),
            "runtime_instrumentation": "operation_samples_v1",
        }
        request_fingerprint = stable_hash(identity)
        suffix = "deterministic" if optimizer_seed is None else f"optimizer_seed_{int(optimizer_seed)}"
        child_path = output / "replicates" / f"{suffix}.json"
        if child_path.exists():
            if not resume:
                raise FileExistsError(child_path)
            _validate_child(child_path, request_fingerprint)
        else:
            cfg = load_config(config_path)
            cfg["runtime_profiling"] = {"record_operation_samples": True}
            risk_checkpoint = _risk_checkpoint(cfg) if shield_enabled else None
            result = evaluate_policy(
                cfg,
                None if manifest is None else checkpoint,
                seeds,
                shield_enabled,
                risk_checkpoint=risk_checkpoint,
                group_name=f"runtime_{method_id}_{suffix}",
                policy_type=str(method["policy_type"]),
            )
            episodes = list(result.get("episodes", []) or [])
            if [int(item.get("seed", item.get("episode_seed", -1))) for item in episodes] != seeds:
                raise ValueError(f"{method_id}: deployment-runtime seed schedule was not observed")
            payload: dict[str, Any] = {
                "artifact_kind": METHOD_REPORT_KIND,
                "schema_version": 1,
                "status": "complete",
                "conclusion_scope": "deployment_runtime_only",
                "method_id": method_id,
                "optimizer_seed": optimizer_seed,
                "policy_type": str(method["policy_type"]),
                "shield_enabled": shield_enabled,
                "config": str(config_path),
                "config_sha256": file_sha256(config_path),
                "checkpoint": checkpoint,
                "checkpoint_sha256": "" if manifest is None else str(row["checkpoint_sha256"]),
                "simulator_seeds": seeds,
                "request_fingerprint": request_fingerprint,
                "environment": _software_hardware(),
                "runtime_metrics": _generic_metrics(episodes),
                "evaluation_metrics": dict(result.get("metrics", {}) or {}),
                "episodes": episodes,
                "gate": {"registered": False, "pass": None, "reason": "descriptive_runtime_only"},
            }
            payload["report_fingerprint"] = _fingerprint(payload, "report_fingerprint")
            _write_new_or_same(child_path, payload)
        children.append(
            {
                "optimizer_seed": optimizer_seed,
                "report": str(child_path.resolve()),
                "report_sha256": file_sha256(child_path),
            }
        )
    child_reports = [read_json(resolve_path(str(item["report"]))) for item in children]
    pooled_operations: dict[str, list[float]] = {}
    episode_wall: list[float] = []
    per_step_wall: list[float] = []
    for report in child_reports:
        for episode in report.get("episodes", []) or []:
            performance = dict(episode.get("performance", {}) or {})
            wall = float(performance.get("wall_time", 0.0))
            steps = int(episode.get("steps", 0))
            episode_wall.append(wall)
            if steps > 0:
                per_step_wall.append(wall / steps)
            for name, values in dict(performance.get("operation_samples_s", {}) or {}).items():
                pooled_operations.setdefault(str(name), []).extend(float(value) for value in values)
    aggregate: dict[str, Any] = {
        "artifact_kind": METHOD_REPORT_KIND,
        "schema_version": 1,
        "status": "complete",
        "conclusion_scope": "deployment_runtime_only",
        "method_id": method_id,
        "optimizer_replicate_count": len(children),
        "shield_enabled": shield_enabled,
        "simulator_seeds": seeds,
        "replicates": children,
        "runtime_metrics": {
            "episode_wall_time_s": _summary(episode_wall),
            "amortized_wall_time_per_simulation_step_s": _summary(per_step_wall),
            "operation_latency_s": {
                name: _summary(values) for name, values in sorted(pooled_operations.items())
            },
        },
        "gate": {"registered": False, "pass": None, "reason": "descriptive_runtime_only"},
        "request_fingerprint": stable_hash(
            {
                "method_id": method_id,
                "manifest_sha256": str(method.get("manifest_sha256", "")),
                "simulator_seeds": seeds,
                "shield_enabled": shield_enabled,
            }
        ),
    }
    aggregate["report_fingerprint"] = _fingerprint(aggregate, "report_fingerprint")
    return _write_new_or_same(output / "method_report.json", aggregate)


def run(*, request_path: str | Path, resume: bool = False) -> Path:
    source = resolve_path(request_path)
    request = read_json(source)
    _validate_fingerprint(request, "request_fingerprint", "deployment-runtime request")
    if request.get("artifact_kind") != REQUEST_KIND or request.get("status") != "ready":
        raise ValueError("invalid deployment-runtime request")
    if file_sha256(resolve_path(str(request["suite_manifest"]))) != str(request["suite_manifest_sha256"]):
        raise ValueError("deployment-runtime PPO suite changed after preparation")
    _, _, manifests = _load_suite(str(request["suite_manifest"]))
    seeds = [int(value) for value in request["simulator_seeds"]]
    root = resolve_path(str(request["output_root"]))
    lineage: dict[str, Any] = {}
    for method_id, method in dict(request["methods"]).items():
        method_output = root / "methods" / str(method_id)
        if str(method["runner"]) == "accvp_candidate_table_runtime_gate":
            report_path = method_output / "accvp_runtime_report.json"
            accvp_runtime_benchmark_replicates.run(
                config_template=str(method["config"]),
                replicate_manifest=str(method["manifest"]),
                seeds=seeds,
                backend="vectorized",
                output=report_path,
                device="auto",
                resume=resume,
            )
        else:
            report_path = _run_generic(
                method_id=str(method_id),
                method=method,
                manifest=None if method_id == RULE_METHOD_ID else manifests[str(method_id)],
                seeds=seeds,
                shield_enabled=bool(request["shield_enabled"] and method_id != RULE_METHOD_ID),
                output=method_output,
                resume=resume,
            )
        lineage[str(method_id)] = {
            "runner": str(method["runner"]),
            "report": str(report_path.resolve()),
            "report_sha256": file_sha256(report_path),
        }
    payload: dict[str, Any] = {
        "artifact_kind": REPORT_KIND,
        "schema_version": 1,
        "status": "complete",
        "protocol_id": str(request["protocol_id"]),
        "conclusion_scope": "deployment_runtime_only",
        "method_effect_gate_independent": True,
        "system_effect_gate_independent": True,
        "request": str(source),
        "request_sha256": file_sha256(source),
        "request_fingerprint": str(request["request_fingerprint"]),
        "methods": lineage,
    }
    payload["report_fingerprint"] = _fingerprint(payload, "report_fingerprint")
    return _write_new_or_same(root / "deployment_runtime_report.json", payload)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare or execute the main-method deployment-runtime suite")
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--protocol", required=True)
    prepare_parser.add_argument("--suite-manifest", required=True)
    prepare_parser.add_argument("--output-root", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--request", required=True)
    run_parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        path = prepare(
            protocol_path=args.protocol,
            suite_manifest=args.suite_manifest,
            output_root=args.output_root,
        )
    else:
        path = run(request_path=args.request, resume=args.resume)
    print(f"main_method_runtime={path}")


if __name__ == "__main__":
    main()
