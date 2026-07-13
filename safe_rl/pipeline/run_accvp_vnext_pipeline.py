from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from safe_rl.accvp.schema import read_json
from safe_rl.utils.config import REPO_ROOT
from safe_rl.utils.io import write_json


PILOT_CONFIG = "safe_rl/config/active/accvp_vnext/pilot.yaml"
ORACLE_CONFIG = "safe_rl/config/active/accvp_vnext/oracle_regression.yaml"
FORMAL_CONFIG = "safe_rl/config/active/accvp_vnext/formal.yaml"
TRAIN_CONFIG = "safe_rl/config/active/accvp_vnext/train.yaml"
PPO_CONFIG = "safe_rl/config/active/accvp_vnext/ppo_candidate_table_full.yaml"
BASELINE_PPO_CONFIG = "safe_rl/config/baselines/wcdt/ppo_wcdt_v3_reward_v2.yaml"
MATRIX_CONFIG = "safe_rl/config/active/accvp_vnext/ppo_ablation_matrix.yaml"
PROTOCOL_CONFIG = "safe_rl/config/examples/vnext/evaluation_protocol_vnext.example.yaml"
OPTIMIZER_SEEDS = [1001, 1002, 1003, 1004, 1005]
RUNTIME_SEEDS = list(range(50001, 50031))


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (REPO_ROOT / value).resolve()


def _complete_shards(root: Path) -> list[Path]:
    result: list[Path] = []
    for manifest_path in sorted(root.glob("*/manifests/dataset_manifest.json")):
        try:
            manifest = read_json(manifest_path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if (
            int(manifest.get("complete_roots", -1)) == int(manifest.get("root_budget", -2))
            and int(manifest.get("failed_branches", 1)) == 0
            and int(manifest.get("counterfactual_schema_version", -1)) == 3
        ):
            result.append(manifest_path.parents[1])
    return result


def _artifact_ok(
    path: Path,
    *,
    state_field: str | None = None,
    artifact_kind: str | None = None,
    status: str | None = None,
    gate_pass: bool = False,
) -> bool:
    if not path.is_file():
        return False
    try:
        payload = read_json(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    if state_field is not None and str(payload.get(state_field, "")).lower() not in {
        "pass",
        "go",
        "complete",
    }:
        return False
    if artifact_kind is not None and str(payload.get("artifact_kind", "")) != artifact_kind:
        return False
    if status is not None and str(payload.get("status", "")) != status:
        return False
    if gate_pass and not bool(dict(payload.get("gate", {}) or {}).get("pass", False)):
        return False
    return True


def _module_command(module: str, *args: Any) -> list[str]:
    return [sys.executable, "-m", module, *(str(value) for value in args)]


def workflow_status(
    *,
    baseline_manifest: str | Path = "safe_rl_output/runs/wcdt_vnext_replicates/ppo_replicate_manifest.json",
) -> dict[str, Any]:
    pilot_shard_root = _resolve(
        "safe_rl_output/runs/accvp_vnext_pilot/stage1_counterfactual/accvp_vnext_schema3_pilot/shards"
    )
    oracle_shard_root = _resolve(
        "safe_rl_output/runs/accvp_vnext_oracle_regression/stage1_counterfactual/accvp_vnext_schema3_oracle_regression/shards"
    )
    formal_shard_root = _resolve(
        "safe_rl_output/runs/accvp_vnext_formal/stage1_counterfactual/accvp_vnext_schema3_formal/shards"
    )
    pilot_shards = _complete_shards(pilot_shard_root)
    oracle_shards = _complete_shards(oracle_shard_root)
    formal_shards = _complete_shards(formal_shard_root)
    pilot_dataset = _resolve("safe_rl_output/runs/accvp_vnext_pilot_dataset")
    oracle_dataset = _resolve("safe_rl_output/runs/accvp_vnext_oracle_regression_dataset")
    formal_dataset = _resolve("safe_rl_output/runs/accvp_vnext_formal_dataset")
    oracle_report = _resolve("safe_rl_output/runs/accvp_vnext_oracle_regression/oracle_report.json")
    pilot_report = _resolve("safe_rl_output/runs/accvp_vnext_pilot/pilot_report.json")
    predictor_manifest = _resolve(
        "safe_rl_output/runs/accvp_vnext_train/accvp/accvp_vnext_schema3_candidate_manifest.json"
    )
    scorer_report = _resolve("safe_rl_output/runs/accvp_vnext_runtime/scorer_preflight.json")
    candidate_manifest = _resolve(
        "safe_rl_output/runs/accvp_vnext_replicates/ppo_replicate_manifest.json"
    )
    baseline_path = _resolve(baseline_manifest)
    runtime_replicates = _resolve(
        "safe_rl_output/runs/accvp_vnext_runtime/replicated_runtime_report.json"
    )
    stage5_request = _resolve(
        "safe_rl_output/runs/accvp_vnext_stage5/generated/replicated_request.json"
    )
    stage5_report = _resolve("safe_rl_output/runs/accvp_vnext_stage5/replicated_report.json")
    holdout_report = _resolve(
        "safe_rl_output/runs/accvp_vnext_final_holdout/accvp_vnext_schema3_final_test_diagnostics.json"
    )

    phases: list[dict[str, Any]] = []

    def add(name: str, complete: bool, command: list[str] | None, artifact: str | Path) -> None:
        phases.append(
            {
                "name": name,
                "complete": bool(complete),
                "artifact": str(artifact),
                "command": command,
            }
        )

    add(
        "pilot_collection",
        len(pilot_shards) >= 10,
        _module_command("safe_rl.pipeline.stage1_collect_accvp_jobs", "--config", PILOT_CONFIG),
        pilot_shard_root,
    )
    merge_pilot = _module_command(
        "safe_rl.pipeline.stage1_merge_counterfactual",
        "--config",
        PILOT_CONFIG,
        *[value for shard in pilot_shards for value in ("--shard", shard)],
        "--output",
        pilot_dataset,
    )
    add(
        "pilot_merge",
        (pilot_dataset / "manifests" / "dataset_manifest.json").is_file(),
        merge_pilot if pilot_shards else None,
        pilot_dataset,
    )
    add(
        "oracle_collection",
        bool(oracle_shards),
        _module_command("safe_rl.pipeline.stage1_collect_accvp_jobs", "--config", ORACLE_CONFIG),
        oracle_shard_root,
    )
    add(
        "oracle_merge",
        (oracle_dataset / "manifests" / "dataset_manifest.json").is_file(),
        (
            _module_command(
                "safe_rl.pipeline.stage1_merge_counterfactual",
                "--config",
                ORACLE_CONFIG,
                *[value for shard in oracle_shards for value in ("--shard", shard)],
                "--output",
                oracle_dataset,
            )
            if oracle_shards
            else None
        ),
        oracle_dataset,
    )
    add(
        "oracle_regression",
        _artifact_ok(oracle_report, state_field="oracle_state"),
        _module_command(
            "safe_rl.pipeline.accvp_oracle_smoke",
            "--dataset",
            oracle_dataset,
            "--output",
            oracle_report,
            "--seeds",
            2,
            5,
            "--cohort-role",
            "oracle_regression",
        ),
        oracle_report,
    )
    add(
        "pilot_validation",
        _artifact_ok(pilot_report, state_field="pilot_state"),
        _module_command(
            "safe_rl.pipeline.stage1_validate_accvp_pilot",
            "--config",
            PILOT_CONFIG,
            "--dataset",
            pilot_dataset,
            "--oracle-report",
            oracle_report,
            "--output",
            pilot_report,
        ),
        pilot_report,
    )
    add(
        "formal_collection",
        len(formal_shards) >= 50,
        _module_command("safe_rl.pipeline.stage1_collect_accvp_jobs", "--config", FORMAL_CONFIG),
        formal_shard_root,
    )
    add(
        "formal_merge",
        (formal_dataset / "manifests" / "dataset_manifest.json").is_file(),
        (
            _module_command(
                "safe_rl.pipeline.stage1_merge_counterfactual",
                "--config",
                FORMAL_CONFIG,
                *[value for shard in formal_shards for value in ("--shard", shard)],
                "--output",
                formal_dataset,
            )
            if formal_shards
            else None
        ),
        formal_dataset,
    )
    add(
        "accvp_training",
        predictor_manifest.is_file(),
        _module_command("safe_rl.pipeline.stage2_train_accvp", "--config", TRAIN_CONFIG),
        predictor_manifest,
    )
    add(
        "scorer_runtime_preflight",
        _artifact_ok(
            scorer_report,
            artifact_kind="accvp_runtime_benchmark_v1",
            gate_pass=True,
        ),
        _module_command(
            "safe_rl.pipeline.accvp_runtime_benchmark",
            "--config",
            PPO_CONFIG,
            "--policy-type",
            "rule_gap_acceptance",
            "--seeds",
            *RUNTIME_SEEDS,
            "--backend",
            "vectorized",
            "--output",
            scorer_report,
        ),
        scorer_report,
    )
    add(
        "candidate_ppo_replicates",
        _artifact_ok(
            candidate_manifest,
            artifact_kind="ppo_optimizer_replicate_manifest_v1",
            status="complete",
        ),
        _module_command(
            "safe_rl.pipeline.stage3_train_ppo_replicates",
            "--config",
            PPO_CONFIG,
            "--matrix",
            MATRIX_CONFIG,
            "--method-id",
            "candidate_table_reward_v2",
            "--optimizer-seeds",
            *OPTIMIZER_SEEDS,
            "--run-id-prefix",
            "ppo_accvp_vnext",
            "--output-root",
            candidate_manifest.parent,
        ),
        candidate_manifest,
    )
    add(
        "baseline_ppo_replicates",
        _artifact_ok(
            baseline_path,
            artifact_kind="ppo_optimizer_replicate_manifest_v1",
            status="complete",
        ),
        _module_command(
            "safe_rl.pipeline.stage3_train_ppo_replicates",
            "--config",
            BASELINE_PPO_CONFIG,
            "--matrix",
            MATRIX_CONFIG,
            "--method-id",
            "wcdt_reward_v2",
            "--optimizer-seeds",
            *OPTIMIZER_SEEDS,
            "--run-id-prefix",
            "ppo_wcdt_vnext",
            "--output-root",
            baseline_path.parent,
        ),
        baseline_path,
    )
    add(
        "policy_runtime_replicates",
        _artifact_ok(
            runtime_replicates,
            artifact_kind="accvp_runtime_benchmark_replicates_v1",
            gate_pass=True,
        ),
        _module_command(
            "safe_rl.pipeline.accvp_runtime_benchmark_replicates",
            "--config-template",
            PPO_CONFIG,
            "--replicate-manifest",
            candidate_manifest,
            "--seeds",
            *RUNTIME_SEEDS,
            "--backend",
            "vectorized",
            "--output",
            runtime_replicates,
        ),
        runtime_replicates,
    )
    add(
        "stage5_generate",
        stage5_request.is_file(),
        _module_command(
            "safe_rl.pipeline.stage5_generate_replicated_configs",
            "--baseline-manifest",
            baseline_path,
            "--candidate-manifest",
            candidate_manifest,
            "--protocol",
            PROTOCOL_CONFIG,
            "--seed-role",
            "natural_confirmatory",
            "--runtime-replicate-report",
            runtime_replicates,
            "--output-dir",
            stage5_request.parent,
        ),
        stage5_request,
    )
    add(
        "stage5_replicates_and_aggregate",
        _artifact_ok(
            stage5_report,
            artifact_kind="stage5_replicated_paired_report_v1",
            gate_pass=True,
        ),
        _module_command(
            "safe_rl.pipeline.stage5_run_replicates",
            "--generated-dir",
            stage5_request.parent,
            "--aggregate-output",
            stage5_report,
        ),
        stage5_report,
    )
    add(
        "one_shot_final_holdout",
        holdout_report.is_file(),
        _module_command(
            "safe_rl.pipeline.accvp_final_holdout_eval",
            "--config",
            TRAIN_CONFIG,
            "--artifact-manifest",
            predictor_manifest,
            "--runtime-benchmark",
            runtime_replicates,
            "--stage5-replicated-report",
            stage5_report,
            "--output-dir",
            holdout_report.parent,
            "--mode",
            "full",
        ),
        holdout_report,
    )
    first_incomplete = next((phase for phase in phases if not phase["complete"]), None)
    return {
        "artifact_kind": "accvp_vnext_pipeline_status_v1",
        "schema_version": 1,
        "complete": first_incomplete is None,
        "next_phase": None if first_incomplete is None else first_incomplete["name"],
        "next_command": None if first_incomplete is None else first_incomplete["command"],
        "phases": phases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fail-closed ACCVP VNext workflow coordinator; executes at most one gated phase per invocation"
    )
    parser.add_argument("--baseline-manifest", default="safe_rl_output/runs/wcdt_vnext_replicates/ppo_replicate_manifest.json")
    parser.add_argument("--status-output")
    parser.add_argument("--execute-next", action="store_true")
    parser.add_argument("--allow-final-holdout", action="store_true")
    args = parser.parse_args()
    status = workflow_status(baseline_manifest=args.baseline_manifest)
    if args.status_output:
        write_json(_resolve(args.status_output), status)
    print(json.dumps(status, indent=2, ensure_ascii=False))
    if not args.execute_next or status["complete"]:
        return
    if status["next_phase"] == "one_shot_final_holdout" and not args.allow_final_holdout:
        raise RuntimeError(
            "final holdout is sealed and cannot be opened without --allow-final-holdout"
        )
    command = status.get("next_command")
    if not command:
        raise RuntimeError(
            f"phase {status['next_phase']!r} requires an external baseline manifest or explicit manual input"
        )
    subprocess.run(command, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
