from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping

from safe_rl.accvp.contracts.schema import file_sha256, read_json, stable_hash, write_json_atomic
from safe_rl.main_method_protocol import (
    RULE_METHOD_ID,
    executor_contract,
    load_protocol,
    public_protocol,
    resolve_path,
)
from safe_rl.pipeline import stage3_train_ppo
from safe_rl.pipeline.audit_ppo_replicate_lineage import audit_manifest
from safe_rl.ppo_replicates import (
    REPLICATE_MANIFEST_KIND,
    observation_contract,
    plain,
    validate_reward_semantics,
    write_yaml_atomic,
)
from safe_rl.utils.config import load_config


PLAN_KIND = "main_method_ppo_suite_plan_v1"
METHOD_MANIFEST_KIND = "main_method_ppo_method_manifest_v1"
SUITE_MANIFEST_KIND = "main_method_ppo_suite_manifest_v1"
EQUIVALENCE_REPORT_KIND = "main_method_acceleration_equivalence_report_v1"


def _write_or_validate_json(path: Path, payload: Mapping[str, Any]) -> Path:
    ready = plain(payload)
    if path.exists():
        if read_json(path) != ready:
            raise FileExistsError(f"refusing to replace a different artifact: {path}")
        return path.resolve()
    write_json_atomic(path, ready)
    return path.resolve()


def _load_equivalence(path: str | Path | None) -> tuple[Path | None, dict[str, Any] | None]:
    if path is None:
        return None, None
    source = resolve_path(path)
    report = read_json(source)
    declared = str(report.get("report_fingerprint", ""))
    content = {key: value for key, value in report.items() if key != "report_fingerprint"}
    if (
        report.get("artifact_kind") != EQUIVALENCE_REPORT_KIND
        or report.get("status") != "complete"
        or not declared
        or stable_hash(content) != declared
    ):
        raise ValueError("invalid acceleration equivalence report")
    return source, report


def _executor_key(contract: Mapping[str, Any]) -> tuple[int, int, int]:
    return (
        int(contract.get("ppo_num_envs", -1)),
        int(contract.get("n_steps", -1)),
        int(contract.get("checkpoint_selection_workers", -1)),
    )


def _equivalence_authorization(
    *,
    method_id: str,
    source_contract: Mapping[str, Any],
    target_contract: Mapping[str, Any],
    report_path: Path | None,
    report: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if _executor_key(source_contract) == _executor_key(target_contract):
        return {
            "kind": "identical_executor_contract",
            "source_executor": plain(source_contract),
            "target_executor": plain(target_contract),
        }
    if report is None or report_path is None:
        return None
    summary = dict((report.get("methods", {}) or {}).get(method_id, {}) or {})
    rows = list(summary.get("comparisons", []) or [])
    by_key = {
        (
            int(row.get("ppo_num_envs", -1)),
            int(row.get("n_steps", -1)),
            int(row.get("checkpoint_selection_workers", -1)),
        ): dict(row)
        for row in rows
    }
    source = by_key.get(_executor_key(source_contract))
    target = by_key.get(_executor_key(target_contract))
    if source is None or target is None:
        return None
    if not bool(source.get("equivalent", False)) or not bool(target.get("equivalent", False)):
        return None
    return {
        "kind": "exact_state_and_selection_equivalence",
        "report": str(report_path),
        "report_sha256": file_sha256(report_path),
        "report_fingerprint": str(report["report_fingerprint"]),
        "source_executor": plain(source_contract),
        "target_executor": plain(target_contract),
        "source_comparison": source,
        "target_comparison": target,
    }


def _source_row(
    *,
    source: Path,
    method_id: str,
    seed: int,
    method_config: Path,
) -> tuple[dict[str, Any] | None, str | None]:
    payload = read_json(source)
    if payload.get("artifact_kind") != REPLICATE_MANIFEST_KIND:
        return None, "unsupported_source_manifest"
    if payload.get("status") != "complete" or str(payload.get("method_id", "")) != method_id:
        return None, "source_manifest_identity_or_status_mismatch"
    audit = audit_manifest(
        source,
        required_seeds=[int(seed)],
        method_config=method_config,
    )
    if audit.get("status") != "reusable":
        return None, f"source_lineage_audit_failed:{audit}"
    rows = [
        dict(row)
        for row in payload.get("records", []) or []
        if int(row.get("optimizer_seed", row.get("training_seed", -1))) == int(seed)
    ]
    if len(rows) != 1:
        return None, "source_seed_is_missing_or_ambiguous"
    return rows[0], None


def prepare(
    *,
    protocol_path: str | Path,
    output_root: str | Path,
    acceleration_equivalence: str | Path | None = None,
) -> Path:
    protocol = load_protocol(protocol_path, verify_artifacts=True)
    equivalence_path, equivalence = _load_equivalence(acceleration_equivalence)
    output = resolve_path(output_root)
    method_plans: dict[str, Any] = {}
    for method_id, method in protocol["methods"].items():
        if method_id == RULE_METHOD_ID:
            method_plans[method_id] = {
                "policy_type": "rule_gap_acceptance",
                "action": "evaluate_without_training",
            }
            continue
        config_path = resolve_path(str(method["config"]))
        base = plain(load_config(config_path))
        target_executor = executor_contract(base)
        method_output = resolve_path(str(method["manifest"])).parent
        records: list[dict[str, Any]] = []
        for seed in protocol["optimizer_seeds"]:
            rejected: list[dict[str, str]] = []
            selected_row: dict[str, Any] | None = None
            authorization: dict[str, Any] | None = None
            selected_source: Path | None = None
            for source_value in method.get("reuse_manifests", []) or []:
                source = resolve_path(str(source_value))
                if not source.is_file():
                    rejected.append({"manifest": str(source), "reason": "missing"})
                    continue
                row, reason = _source_row(
                    source=source,
                    method_id=method_id,
                    seed=int(seed),
                    method_config=config_path,
                )
                if row is None:
                    rejected.append({"manifest": str(source), "reason": str(reason)})
                    continue
                source_cfg = load_config(resolve_path(str(row["resolved_config"])))
                source_executor = executor_contract(source_cfg)
                authorization = _equivalence_authorization(
                    method_id=method_id,
                    source_contract=source_executor,
                    target_contract=target_executor,
                    report_path=equivalence_path,
                    report=equivalence,
                )
                if authorization is None:
                    rejected.append(
                        {
                            "manifest": str(source),
                            "reason": "cross_executor_reuse_lacks_exact_equivalence",
                        }
                    )
                    continue
                selected_row = row
                selected_source = source
                break
            if selected_row is not None and selected_source is not None:
                records.append(
                    {
                        "optimizer_seed": int(seed),
                        "action": "reuse",
                        "source_manifest": str(selected_source),
                        "source_manifest_sha256": file_sha256(selected_source),
                        "source_record": selected_row,
                        "executor_authorization": authorization,
                        "rejected_reuse_candidates": rejected,
                    }
                )
                continue
            if not bool(method.get("train", False)):
                raise RuntimeError(
                    f"{method_id} seed {seed} has no reusable formal checkpoint and is not trainable"
                )
            resolved = plain(base)
            run_id = f"{method['run_id_prefix']}_seed_{seed}"
            resolved["run"]["run_id"] = run_id
            resolved["rl"]["optimizer_seed"] = int(seed)
            resolved.setdefault("experiment", {})["method_id"] = method_id
            resolved["experiment"]["optimizer_seed"] = int(seed)
            config_file = method_output / "generated_configs" / f"optimizer_seed_{seed}.yaml"
            if config_file.exists():
                import yaml

                with config_file.open("r", encoding="utf-8") as handle:
                    existing = yaml.safe_load(handle) or {}
                if plain(existing) != resolved:
                    raise FileExistsError(f"generated PPO config changed: {config_file}")
            else:
                write_yaml_atomic(config_file, resolved)
            run_root = resolve_path(resolved["run"]["output_root"]) / run_id / "stage3"
            records.append(
                {
                    "optimizer_seed": int(seed),
                    "action": "train",
                    "config": str(config_file.resolve()),
                    "config_sha256": file_sha256(config_file),
                    "checkpoint": str(
                        (run_root / str(resolved["stage3"]["model_name"])).resolve()
                    ),
                    "stage3_report": str((run_root / "stage3_training_report.json").resolve()),
                    "executor_authorization": {
                        "kind": "target_executor_training",
                        "target_executor": target_executor,
                    },
                    "rejected_reuse_candidates": rejected,
                }
            )
        method_plans[method_id] = {
            "policy_type": "sb3_ppo",
            "train": bool(method.get("train", False)),
            "config": str(config_path),
            "config_sha256": file_sha256(config_path),
            "target_executor": target_executor,
            "reward_semantics": protocol["_configs"][method_id]["reward"],
            "observation_contract": protocol["_configs"][method_id]["observation"],
            "output_manifest": str(resolve_path(str(method["manifest"]))),
            "records": records,
        }
    plan: dict[str, Any] = {
        "artifact_kind": PLAN_KIND,
        "schema_version": 1,
        "status": "prepared",
        "protocol": public_protocol(protocol),
        "protocol_path": str(resolve_path(protocol_path)),
        "protocol_sha256": file_sha256(resolve_path(protocol_path)),
        "acceleration_equivalence": (
            None
            if equivalence_path is None
            else {
                "path": str(equivalence_path),
                "sha256": file_sha256(equivalence_path),
                "report_fingerprint": str(equivalence["report_fingerprint"]),
            }
        ),
        "methods": method_plans,
    }
    plan["plan_fingerprint"] = stable_hash(plan)
    return _write_or_validate_json(output / "ppo_suite_plan.json", plan)


def _validate_reused_record(record: Mapping[str, Any]) -> dict[str, Any]:
    row = dict(record["source_record"])
    source = resolve_path(str(record["source_manifest"]))
    if file_sha256(source) != str(record["source_manifest_sha256"]):
        raise ValueError("reused PPO source manifest changed after planning")
    for path_field, hash_field in (
        ("resolved_config", "resolved_config_sha256"),
        ("checkpoint", "checkpoint_sha256"),
        ("stage3_report", "stage3_report_sha256"),
    ):
        path = resolve_path(str(row[path_field]))
        if not path.is_file() or file_sha256(path) != str(row[hash_field]):
            raise ValueError(f"reused PPO {path_field} binding changed")
    row["provenance"] = {
        "action": "reused",
        "source_manifest": str(source),
        "source_manifest_sha256": file_sha256(source),
        "executor_authorization": plain(record["executor_authorization"]),
    }
    return row


def _train_or_recover(method_id: str, record: Mapping[str, Any]) -> dict[str, Any]:
    config_path = resolve_path(str(record["config"]))
    if file_sha256(config_path) != str(record["config_sha256"]):
        raise ValueError("PPO generated config changed after planning")
    checkpoint = resolve_path(str(record["checkpoint"]))
    report_path = resolve_path(str(record["stage3_report"]))
    if checkpoint.is_file() and report_path.is_file():
        action = "recovered_complete"
    else:
        if checkpoint.parent.exists() and any(checkpoint.parent.iterdir()):
            raise RuntimeError(
                "PPO run contains partial Stage3 output without its final checkpoint/report: "
                f"{checkpoint.parent}"
            )
        produced = Path(stage3_train_ppo.run(load_config(config_path))).resolve()
        if produced != checkpoint:
            raise RuntimeError(f"PPO output path mismatch: expected={checkpoint} actual={produced}")
        action = "trained"
    report = read_json(report_path)
    seed = int(record["optimizer_seed"])
    if int(report.get("optimizer_seed", -1)) != seed:
        raise ValueError("PPO Stage3 report optimizer seed mismatch")
    cfg = load_config(config_path)
    reward = validate_reward_semantics(cfg)
    observation = observation_contract(cfg, require_artifacts=True)
    for field, expected in (
        ("reward_semantics_hash", reward["sha256"]),
        ("observation_contract_hash", observation["sha256"]),
    ):
        if str(report.get(field, "")) != str(expected):
            raise ValueError(f"PPO Stage3 report {field} mismatch")
    return {
        "method_id": method_id,
        "training_seed": seed,
        "optimizer_seed": seed,
        "simulator_training_start_seed": int(report["simulator_training_start_seed"]),
        "run_id": str(cfg.run.run_id),
        "resolved_config": str(config_path),
        "resolved_config_sha256": file_sha256(config_path),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_sha256(checkpoint),
        "stage3_report": str(report_path),
        "stage3_report_sha256": file_sha256(report_path),
        "training_budget": {
            "total_timesteps": int(cfg.rl.total_timesteps),
            "n_steps": int(cfg.rl.n_steps),
            "batch_size": int(cfg.rl.batch_size),
            "ppo_num_envs": int(cfg.training.ppo_num_envs),
        },
        "actual_total_timesteps": int(report.get("actual_total_timesteps", 0)),
        "selection_seed_sha256": str(
            report.get("checkpoint_selection_seed_sha256", "")
        ),
        "reward_semantics_hash": str(reward["sha256"]),
        "reward_semantics": reward["payload"],
        "observation_contract_hash": str(observation["sha256"]),
        "observation_contract": observation["payload"],
        "candidate_table_semantic_contract_sha256": str(
            observation["payload"].get("candidate_table_semantic_contract_sha256", "")
        ),
        "closed_loop_execution_contract_sha256": str(
            observation["payload"].get("closed_loop_execution_contract_sha256", "")
        ),
        "deployment_runtime_contract_sha256": str(
            observation["payload"].get("deployment_runtime_contract_sha256", "")
        ),
        "accvp_artifact_fingerprint": str(observation["accvp_artifact_fingerprint"]),
        "provenance": {
            "action": action,
            "executor_authorization": plain(record["executor_authorization"]),
        },
    }


def execute(plan_path: str | Path) -> Path:
    source = resolve_path(plan_path)
    plan = read_json(source)
    declared = str(plan.get("plan_fingerprint", ""))
    content = {key: value for key, value in plan.items() if key != "plan_fingerprint"}
    if plan.get("artifact_kind") != PLAN_KIND or declared != stable_hash(content):
        raise ValueError("invalid main-method PPO suite plan")
    method_bindings: dict[str, Any] = {}
    for method_id, method in plan["methods"].items():
        if str(method.get("policy_type", "")) == "rule_gap_acceptance":
            method_bindings[method_id] = plain(method)
            continue
        completed = []
        for record in method["records"]:
            if record["action"] == "reuse":
                completed.append(_validate_reused_record(record))
            elif record["action"] == "train":
                completed.append(_train_or_recover(method_id, record))
            else:
                raise ValueError(f"unknown PPO suite action: {record['action']!r}")
        seeds = [int(row["optimizer_seed"]) for row in completed]
        expected = [int(seed) for seed in plan["protocol"]["optimizer_seeds"]]
        if seeds != expected:
            raise ValueError(f"{method_id}: completed PPO seed order differs from protocol")
        if len({str(row["checkpoint_sha256"]) for row in completed}) != len(completed):
            raise ValueError(f"{method_id}: optimizer replicates reuse checkpoint bytes")
        manifest: dict[str, Any] = {
            "artifact_kind": METHOD_MANIFEST_KIND,
            "schema_version": 1,
            "status": "complete",
            "protocol_id": str(plan["protocol"]["protocol_id"]),
            "method_id": method_id,
            "optimizer_seeds": seeds,
            "target_config": str(method["config"]),
            "target_config_sha256": str(method["config_sha256"]),
            "target_executor": plain(method["target_executor"]),
            "reward_semantics": plain(method["reward_semantics"]),
            "observation_contract": plain(method["observation_contract"]),
            "records": completed,
        }
        manifest["manifest_fingerprint"] = stable_hash(manifest)
        output = resolve_path(str(method["output_manifest"]))
        _write_or_validate_json(output, manifest)
        method_bindings[method_id] = {
            "policy_type": "sb3_ppo",
            "manifest": str(output),
            "manifest_sha256": file_sha256(output),
            "manifest_fingerprint": str(manifest["manifest_fingerprint"]),
        }
    suite: dict[str, Any] = {
        "artifact_kind": SUITE_MANIFEST_KIND,
        "schema_version": 1,
        "status": "complete",
        "protocol_id": str(plan["protocol"]["protocol_id"]),
        "protocol": plain(plan["protocol"]),
        "source_plan": str(source),
        "source_plan_sha256": file_sha256(source),
        "optimizer_seeds": [int(seed) for seed in plan["protocol"]["optimizer_seeds"]],
        "methods": method_bindings,
    }
    suite["manifest_fingerprint"] = stable_hash(suite)
    output = source.parent / "ppo_suite_manifest.json"
    return _write_or_validate_json(output, suite)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare or execute the six-method PPO suite with audited checkpoint reuse"
    )
    sub = parser.add_subparsers(dest="action", required=True)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--protocol", required=True)
    prepare_parser.add_argument("--output-root", required=True)
    prepare_parser.add_argument("--acceleration-equivalence")
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--plan", required=True)
    args = parser.parse_args()
    if args.action == "prepare":
        output = prepare(
            protocol_path=args.protocol,
            output_root=args.output_root,
            acceleration_equivalence=args.acceleration_equivalence,
        )
    else:
        output = execute(args.plan)
    print(f"main_method_ppo_suite_artifact={output}")


if __name__ == "__main__":
    main()
