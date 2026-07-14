from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from safe_rl.accvp.contracts.schema import file_sha256, read_json
from safe_rl.evaluation_protocol import protocol_snapshot
from safe_rl.pipeline.audit_ppo_replicate_lineage import audit_manifest
from safe_rl.pipeline.stage5_replicated_aggregate import REQUEST_ARTIFACT_KIND
from safe_rl.ppo_replicates import REPLICATE_MANIFEST_KIND, plain, write_json_new, write_yaml_atomic
from safe_rl.utils.config import REPO_ROOT, load_config


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (REPO_ROOT / value).resolve()


def _load_complete_manifest(path: str | Path) -> tuple[Path, dict[str, Any]]:
    source = _resolve(path)
    payload = read_json(source)
    if str(payload.get("artifact_kind", "")) != REPLICATE_MANIFEST_KIND:
        raise ValueError(f"unsupported PPO replicate manifest: {source}")
    if payload.get("status") != "complete":
        raise ValueError(f"replicate manifest is not complete: {source}")
    return source, payload


def _row_map(manifest: dict[str, Any]) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for item in manifest.get("records", []) or []:
        row = dict(item)
        seed = int(row.get("optimizer_seed", row.get("training_seed", -1)))
        if seed in result:
            raise ValueError(f"duplicate optimizer seed in replicate manifest: {seed}")
        result[seed] = row
    return result


def _runtime_report_map(path: str | Path) -> tuple[Path, dict[int, str]]:
    source = _resolve(path)
    payload = read_json(source)
    if str(payload.get("artifact_kind", "")) != "accvp_runtime_benchmark_replicates_v1":
        raise ValueError("Stage5 generation requires a replicated policy-runtime report")
    if not bool(payload.get("gate", {}).get("pass", False)):
        raise ValueError("replicated policy runtime gate did not pass")
    reports = {
        int(row["optimizer_seed"]): str(row["report"])
        for row in payload.get("replicates", []) or []
    }
    return source, reports


def _semantic_rl(config: dict[str, Any]) -> dict[str, Any]:
    rl = config.get("rl", {}) or {}
    return {
        "reward_profile": rl.get("reward_profile"),
        "training_semantics_version": rl.get("training_semantics_version"),
        "merge_timing_reward": plain(rl.get("merge_timing_reward", {}) or {}),
        "policy_lateral_commitment": plain(rl.get("policy_lateral_commitment", {}) or {}),
    }


def _group(
    *,
    name: str,
    row: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    forecast = config.get("forecast_features", {}) or {}
    enabled = bool(forecast.get("enabled", False))
    accvp = plain(config.get("accvp", {}) or {})
    return {
        "name": name,
        "forecast_features": enabled,
        "forecast_source": str(forecast.get("source", "")) if enabled else "",
        "forecast_checkpoint": str(forecast.get("checkpoint", "")) if enabled else "",
        "shield": True,
        "model_path": str(row["checkpoint"]),
        "raw_policy": str(row["method_id"]),
        "shield_overrides": {
            "forecast_aware_candidate_ranking_mode": "off",
            "forecast_task_shadow_enabled": False,
            "task_backstop_enabled": False,
        },
        "rl_overrides": _semantic_rl(config),
        "accvp": accvp,
        "comparative": {
            "training_seed": int(row["optimizer_seed"]),
            "optimizer_seed": int(row["optimizer_seed"]),
            "checkpoint_sha256": str(row["checkpoint_sha256"]),
            "reward_semantics_hash": str(row["reward_semantics_hash"]),
            "observation_contract_hash": str(row["observation_contract_hash"]),
        },
    }


def generate(
    *,
    baseline_manifest: str | Path,
    candidate_manifest: str | Path,
    protocol: str | Path,
    seed_role: str,
    runtime_replicate_report: str | Path,
    output_dir: str | Path,
    comparison_id: str = "baseline_vs_accvp_vnext",
) -> Path:
    baseline_path, baseline = _load_complete_manifest(baseline_manifest)
    candidate_path, candidate = _load_complete_manifest(candidate_manifest)
    baseline_rows = _row_map(baseline)
    candidate_rows = _row_map(candidate)
    seeds = sorted(set(baseline_rows).intersection(candidate_rows))
    if set(baseline_rows) != set(candidate_rows) or len(seeds) < 5:
        raise ValueError("baseline and candidate manifests must share at least five identical optimizer seeds")
    for source in (baseline_path, candidate_path):
        audit = audit_manifest(source, required_seeds=seeds)
        if audit["status"] != "reusable":
            raise ValueError(f"replicate manifest is not reusable: {source}: {audit}")
    runtime_path, runtime_reports = _runtime_report_map(runtime_replicate_report)
    if set(runtime_reports) != set(seeds):
        raise ValueError("runtime replicate report does not cover the exact optimizer-seed set")
    reward_hashes = {
        str(row["reward_semantics_hash"])
        for row in [*baseline_rows.values(), *candidate_rows.values()]
    }
    if len(reward_hashes) != 1:
        raise ValueError(
            "formal Candidate Table attribution requires identical reward semantics on both sides"
        )
    protocol_cfg = load_config(_resolve(protocol))
    snapshot = protocol_snapshot(protocol_cfg)
    role = str(seed_role)
    cohort = snapshot["cohort_roles"].get(role, role)
    if cohort not in snapshot["cohorts"]:
        raise ValueError(f"unknown simulator seed role/cohort={seed_role!r}")
    simulator_seeds = list(snapshot["cohorts"][cohort])
    if not simulator_seeds:
        raise ValueError("Stage5 simulator seed cohort is empty")
    if role == cohort:
        matching_roles = [key for key, value in snapshot["cohort_roles"].items() if value == cohort]
        role = "stage5_confirmatory" if "stage5_confirmatory" in matching_roles else matching_roles[0]
    output = _resolve(output_dir)
    request_rows: list[dict[str, Any]] = []
    candidate_binding: dict[str, Any] | None = None
    for seed in seeds:
        left = baseline_rows[seed]
        right = candidate_rows[seed]
        with Path(left["resolved_config"]).open("r", encoding="utf-8") as handle:
            import yaml

            left_cfg = yaml.safe_load(handle) or {}
        with Path(right["resolved_config"]).open("r", encoding="utf-8") as handle:
            import yaml

            right_cfg = yaml.safe_load(handle) or {}
        resolved = plain(right_cfg)
        run_id = f"stage5_{comparison_id}_seed_{seed}"
        resolved["run"]["run_id"] = run_id
        resolved["evaluation_protocol"] = plain(protocol_cfg.evaluation_protocol)
        resolved["evaluation_protocol"]["stage5_role"] = role
        left_name = f"baseline_seed_{seed}"
        right_name = f"candidate_seed_{seed}"
        pair_name = comparison_id
        resolved["experiment"] = {
            "purpose": "formal_replicated_candidate_table_attribution",
            "comparison_id": comparison_id,
            "optimizer_seed": seed,
            "deployable_claim": False,
        }
        resolved["stage5"] = {
            "paired_eval": True,
            "same_seed": True,
            "compare_shield_off_on": False,
            "replay_enabled": True,
            "episodes_per_group": len(simulator_seeds),
            "seeds": simulator_seeds,
            "require_accvp_observation_runtime_gate": True,
            "accvp_observation_preflight_report": runtime_reports[seed],
            "statistics": {
                "confidence": 0.95,
                "bootstrap_replicates": 10000,
                "bootstrap_seed": 42001,
            },
            "acceptance": {
                "mainline_reward_v2": {
                    "max_actual_replacement_rate": 0.05,
                    "reward_tolerance": 1.0e-6,
                }
            },
            "pairs": [
                {
                    "name": pair_name,
                    "left": left_name,
                    "right": right_name,
                    "acceptance_profile": "mainline_reward_v2",
                }
            ],
            "groups": [
                _group(name=left_name, row=left, config=left_cfg),
                _group(name=right_name, row=right, config=right_cfg),
            ],
        }
        config_path = output / f"stage5_seed_{seed}.yaml"
        write_yaml_atomic(config_path, resolved)
        run_root = _resolve(resolved["run"]["output_root"]) / run_id
        report_path = run_root / "stage5" / "formal_paired_eval_report.json"
        request_rows.append(
            {
                "training_seed": seed,
                "stage5_config": str(config_path),
                "stage5_config_sha256": file_sha256(config_path),
                "stage5_report": str(report_path),
                "left_group": left_name,
                "right_group": right_name,
                "left_checkpoint_sha256": str(left["checkpoint_sha256"]),
                "right_checkpoint_sha256": str(right["checkpoint_sha256"]),
            }
        )
        observation = dict(right.get("observation_contract", {}) or {})
        current_binding = {
            "path": str(observation.get("accvp_artifact_manifest", "")),
            "sha256": str(observation.get("accvp_artifact_manifest_sha256", "")),
            "artifact_fingerprint": str(observation.get("accvp_artifact_fingerprint", "")),
            "artifact_variant": str(observation.get("accvp_artifact_variant", "")),
            "formal_runtime_contract_sha256": str(
                observation.get("formal_runtime_contract_sha256", "")
            ),
        }
        if candidate_binding is not None and current_binding != candidate_binding:
            raise ValueError("candidate optimizer replicates do not bind one frozen ACCVP artifact")
        candidate_binding = current_binding
    if candidate_binding is None or not all(candidate_binding.values()):
        raise ValueError("candidate replicate manifest lacks complete ACCVP bundle binding")
    request = {
        "artifact_kind": REQUEST_ARTIFACT_KIND,
        "comparison_id": comparison_id,
        "formal_aggregation": True,
        "minimum_training_seed_count": 5,
        "require_strict_lineage": True,
        "candidate_manifest": {
            key: candidate_binding[key]
            for key in ("path", "sha256", "artifact_fingerprint", "artifact_variant")
        },
        "candidate_side": "right",
        "source_acceptance_key": comparison_id,
        "formal_runtime_contract_sha256": candidate_binding["formal_runtime_contract_sha256"],
        "statistics": {
            "continuous_metrics": ["episode_reward"],
            "binary_metrics": ["proxy_collision", "safety_violation", "taper_miss", "merge_success"],
            "confidence": 0.95,
            "bootstrap_replicates": 10000,
            "bootstrap_seed": 42001,
            "require_distinct_checkpoints": True,
        },
        "baseline_replicate_manifest": str(baseline_path),
        "candidate_replicate_manifest": str(candidate_path),
        "runtime_replicate_report": str(runtime_path),
        "replicates": request_rows,
    }
    return write_json_new(output / "replicated_request.json", request)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paired Stage5 configs for crossed PPO replicates")
    parser.add_argument("--baseline-manifest", required=True)
    parser.add_argument("--candidate-manifest", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--seed-role", required=True)
    parser.add_argument("--runtime-replicate-report", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--comparison-id", default="baseline_vs_accvp_vnext")
    args = parser.parse_args()
    path = generate(
        baseline_manifest=args.baseline_manifest,
        candidate_manifest=args.candidate_manifest,
        protocol=args.protocol,
        seed_role=args.seed_role,
        runtime_replicate_report=args.runtime_replicate_report,
        output_dir=args.output_dir,
        comparison_id=args.comparison_id,
    )
    print(f"stage5_replicated_request={path}")


if __name__ == "__main__":
    main()
