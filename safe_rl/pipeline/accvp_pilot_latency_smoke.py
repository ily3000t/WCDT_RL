"""Development-only 12-actor latency feasibility check before formal collection."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from safe_rl.accvp.contracts.schema import file_sha256, read_json, stable_hash, write_json_atomic
from safe_rl.utils.config import REPO_ROOT, load_config


SMOKE_ARTIFACT_KIND = "accvp_pilot_latency_feasibility_smoke_v1"
SMOKE_IMPLEMENTATION_VERSION = "selector4_actor12_shadow_full_path_v1"


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _valid_runtime_detail(
    path: Path,
    *,
    runtime_config: Path,
    seeds: list[int],
) -> bool:
    if not path.is_file():
        return False
    try:
        payload = read_json(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    workload = dict(payload.get("workload", {}) or {})
    return bool(
        str(payload.get("artifact_kind", "")) == "accvp_runtime_benchmark_v1"
        and str(payload.get("evidence_role", "")) == "diagnostic_only"
        and str(payload.get("benchmark_scope", ""))
        == "scorer_latency_feasibility_smoke"
        and Path(str(payload.get("config", ""))).resolve()
        == runtime_config.resolve()
        and int(workload.get("actor_count", -1)) == 12
        and int(workload.get("requested_episode_seed_count", -1)) == len(seeds)
        and str(workload.get("requested_episode_seed_sha256", ""))
        == stable_hash({"episode_seeds": seeds})
    )


def _structure_from_checkpoint(
    checkpoint: Path,
    *,
    runtime_actor_count: int,
) -> dict[str, Any]:
    import torch

    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    metadata = dict(payload.get("metadata", {}) or {})
    state_dicts = list(payload.get("model_state_dicts", []) or [])
    return {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_sha256(checkpoint),
        "actor_count": int(runtime_actor_count),
        "actor_capacity_source": "frozen_runtime_and_data_contract_config",
        "variable_set_encoder": True,
        "ensemble_member_count": len(state_dicts),
        "architecture_version": str(metadata.get("architecture_version", "")),
        "pass": int(runtime_actor_count) == 12 and len(state_dicts) == 3,
    }


def run(
    *,
    train_config: str | Path,
    runtime_config: str | Path,
    seeds: list[int],
    output: str | Path,
) -> Path:
    train_path = _resolve(train_config)
    runtime_path = _resolve(runtime_config)
    output_path = _resolve(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    detail_path = output_path.parent / "runtime_detail.json"

    if len(seeds) < 3 or len(seeds) != len(set(seeds)):
        raise ValueError("pilot latency smoke requires at least three distinct development seeds")
    runtime_cfg = load_config(runtime_path)
    checkpoint = _resolve(str(runtime_cfg.accvp.checkpoint))
    manifest = _resolve(str(runtime_cfg.accvp.artifact_manifest))
    if not checkpoint.is_file() or not manifest.is_file():
        print("[accvp_pilot_latency_smoke] training 12-actor three-member shadow model")
        subprocess.run(
            [
                sys.executable,
                "-m",
                "safe_rl.pipeline.stage2_train_accvp",
                "--config",
                str(train_path),
                "--mode",
                "diagnostic_latency_smoke",
            ],
            cwd=REPO_ROOT,
            check=True,
        )
    structure = _structure_from_checkpoint(
        checkpoint,
        runtime_actor_count=int(runtime_cfg.accvp.actor_count),
    )
    if not bool(structure["pass"]):
        raise ValueError(
            "pilot latency smoke checkpoint is not the required 12-actor "
            "three-member architecture"
        )

    if not _valid_runtime_detail(
        detail_path,
        runtime_config=runtime_path,
        seeds=seeds,
    ):
        if detail_path.exists():
            superseded = output_path.parent / "_superseded"
            superseded.mkdir(parents=True, exist_ok=True)
            archived = superseded / (
                f"runtime_detail_{file_sha256(detail_path)[:12]}.json"
            )
            suffix = 1
            while archived.exists():
                archived = superseded / (
                    f"runtime_detail_{file_sha256(detail_path)[:12]}_{suffix}.json"
                )
                suffix += 1
            detail_path.replace(archived)
        print(
            "[accvp_pilot_latency_smoke] running full ACCVP+Risk+table path "
            f"seeds={seeds}"
        )
        subprocess.run(
            [
                sys.executable,
                "-m",
                "safe_rl.pipeline.accvp_runtime_benchmark",
                "--config",
                str(runtime_path),
                "--policy-type",
                "rule_gap_acceptance",
                "--seeds",
                *[str(seed) for seed in seeds],
                "--backend",
                "vectorized",
                "--diagnostic-smoke",
                "--output",
                str(detail_path),
            ],
            cwd=REPO_ROOT,
            check=True,
        )

    detail = read_json(detail_path)
    metrics = dict(detail.get("metrics", {}) or {})
    stages = dict(metrics.get("accvp_table_latency_per_stage", {}) or {})
    risk = dict(stages.get("risk_secondary", {}) or {})
    thresholds = {
        "total_p95_s": 0.30,
        "total_p99_s": 0.40,
        "total_max_s": 0.50,
        "risk_p95_s": 0.15,
    }
    observed = {
        "activation_decision_count": int(
            metrics.get("accvp_table_activation_window_decision_count", 0)
        ),
        "latency_sample_count": int(metrics.get("accvp_table_latency_count", 0)),
        "total_p95_s": float(metrics.get("accvp_table_latency_p95", float("inf"))),
        "total_p99_s": float(metrics.get("accvp_table_latency_p99", float("inf"))),
        "total_max_s": float(metrics.get("accvp_table_latency_max", float("inf"))),
        "risk_p95_s": float(risk.get("p95", float("inf"))),
        "critical_actor_overflow_count": int(
            metrics.get("accvp_table_critical_actor_overflow_count", 0)
        ),
        "task_actor_overflow_count": int(
            metrics.get("accvp_table_task_actor_overflow_count", 0)
        ),
        "model_error_count": int(metrics.get("accvp_table_model_error_count", 0)),
        "runtime_context_error_count": int(
            metrics.get("accvp_table_runtime_context_error_count", 0)
        ),
        "unexpected_value_error_count": int(
            metrics.get("accvp_table_unexpected_value_error_count", 0)
        ),
    }
    conditions = {
        "model_structure_12_actor_three_member": bool(structure["pass"]),
        "activation_decisions_observed": observed["activation_decision_count"] > 0,
        "latency_samples_observed": observed["latency_sample_count"] > 0,
        "total_p95_within_budget": observed["total_p95_s"] <= thresholds["total_p95_s"],
        "total_p99_within_budget": observed["total_p99_s"] <= thresholds["total_p99_s"],
        "total_max_within_budget": observed["total_max_s"] <= thresholds["total_max_s"],
        "risk_p95_within_budget": observed["risk_p95_s"] <= thresholds["risk_p95_s"],
        "critical_actor_overflow_zero": observed["critical_actor_overflow_count"] == 0,
        "task_actor_overflow_zero": observed["task_actor_overflow_count"] == 0,
        "model_and_runtime_errors_zero": (
            observed["model_error_count"] == 0
            and observed["runtime_context_error_count"] == 0
            and observed["unexpected_value_error_count"] == 0
        ),
    }
    report: dict[str, Any] = {
        "artifact_kind": SMOKE_ARTIFACT_KIND,
        "implementation_version": SMOKE_IMPLEMENTATION_VERSION,
        "evidence_role": "diagnostic_only_pre_formal_feasibility",
        "formal_runtime_evidence": False,
        "hard_realtime_claim": False,
        "training_semantics": "one_epoch_shadow_latency_only_not_accuracy_evidence",
        "train_config": str(train_path),
        "runtime_config": str(runtime_path),
        "development_seeds": seeds,
        "model_structure": structure,
        "runtime_detail": str(detail_path),
        "runtime_detail_sha256": file_sha256(detail_path),
        "thresholds": thresholds,
        "observed": observed,
        "conditions": conditions,
        "smoke_state": "pass" if all(conditions.values()) else "fail",
    }
    report["report_fingerprint"] = stable_hash(report)
    write_json_atomic(output_path, report)
    print(
        "[accvp_pilot_latency_smoke] "
        f"state={report['smoke_state']} p95={observed['total_p95_s']:.6f}s "
        f"p99={observed['total_p99_s']:.6f}s max={observed['total_max_s']:.6f}s "
        f"risk_p95={observed['risk_p95_s']:.6f}s report={output_path}"
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a diagnostic 12-actor full-path latency smoke before formal collection"
    )
    parser.add_argument("--train-config", required=True)
    parser.add_argument("--runtime-config", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(
        train_config=args.train_config,
        runtime_config=args.runtime_config,
        seeds=[int(value) for value in args.seeds],
        output=args.output,
    )


if __name__ == "__main__":
    main()
