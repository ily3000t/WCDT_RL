from __future__ import annotations

import argparse
import hashlib
import zipfile
from pathlib import Path
from typing import Any, Mapping

from safe_rl.accvp.contracts.schema import file_sha256, read_json, stable_hash, write_json_atomic
from safe_rl.main_method_protocol import (
    FORECAST_ATTRIBUTION_METHODS,
    executor_contract,
    load_protocol,
    public_protocol,
    resolve_path,
)
from safe_rl.pipeline import stage3_train_ppo
from safe_rl.ppo_replicates import plain, write_yaml_atomic
from safe_rl.utils.config import load_config


PLAN_KIND = "main_method_acceleration_plan_v1"
REPORT_KIND = "main_method_acceleration_equivalence_report_v1"
STATE_MEMBERS = ("policy.pth", "policy.optimizer.pth", "pytorch_variables.pth")


def _write_or_validate(path: Path, payload: Mapping[str, Any]) -> Path:
    ready = plain(payload)
    if path.exists():
        if read_json(path) != ready:
            raise FileExistsError(f"refusing to replace a different artifact: {path}")
        return path.resolve()
    write_json_atomic(path, ready)
    return path.resolve()


def _zip_member_hashes(path: Path) -> dict[str, str]:
    with zipfile.ZipFile(path, "r") as archive:
        names = set(archive.namelist())
        missing = sorted(set(STATE_MEMBERS) - names)
        if missing:
            raise ValueError(f"PPO checkpoint is missing state members {missing}: {path}")
        return {
            name: hashlib.sha256(archive.read(name)).hexdigest()
            for name in STATE_MEMBERS
        }


def _selection_semantics(report: Mapping[str, Any]) -> dict[str, Any]:
    selection = dict(report.get("checkpoint_selection", {}) or {})
    rows = []
    for raw in selection.get("records", []) or []:
        row = dict(raw)
        row.pop("checkpoint_path", None)
        rows.append(row)
    best = dict(selection.get("best_record", {}) or {})
    best.pop("checkpoint_path", None)
    return {
        "enabled": bool(selection.get("enabled", False)),
        "best_record": best,
        "records": rows,
        "selection_profile": selection.get("selection_profile"),
        "selection_weights": selection.get("selection_weights"),
        "selection_metric": selection.get("selection_metric"),
    }


def prepare(
    *,
    protocol_path: str | Path,
    output_root: str | Path,
    methods: list[str] | None = None,
) -> Path:
    protocol = load_protocol(protocol_path, verify_artifacts=False)
    selected = list(methods or FORECAST_ATTRIBUTION_METHODS)
    unknown = sorted(set(selected) - set(FORECAST_ATTRIBUTION_METHODS))
    if unknown:
        raise ValueError(f"unsupported acceleration benchmark methods: {unknown}")
    block = dict(protocol["ppo"]["acceleration_benchmark"])
    seed = int(block["optimizer_seed"])
    total_timesteps = int(block["total_timesteps"])
    candidates = [dict(item) for item in block["candidates"]]
    output = resolve_path(output_root)
    rows: list[dict[str, Any]] = []
    for method_id in selected:
        source = protocol["_configs"][method_id]["path"]
        base = plain(load_config(source))
        for candidate in candidates:
            num_envs = int(candidate["ppo_num_envs"])
            n_steps = int(candidate["n_steps"])
            selection_workers = int(candidate["checkpoint_selection_workers"])
            if num_envs * n_steps != int(protocol["ppo"]["rollout_size"]):
                raise ValueError("acceleration candidate changes the PPO rollout budget")
            resolved = plain(base)
            run_id = (
                f"main_table_accel_{method_id}_env{num_envs}_sel{selection_workers}"
            )
            resolved["run"]["run_id"] = run_id
            resolved["rl"]["optimizer_seed"] = seed
            resolved["rl"]["total_timesteps"] = total_timesteps
            resolved["rl"]["n_steps"] = n_steps
            resolved["training"]["ppo_num_envs"] = num_envs
            resolved["training"]["ppo_expected_rollout_size"] = int(
                protocol["ppo"]["rollout_size"]
            )
            resolved["stage3"]["checkpoint_selection_workers"] = selection_workers
            resolved["evaluation_protocol"]["cohort_roles"][
                "ppo_optimizer_replicates"
            ] = "ppo_acceleration_development"
            resolved.setdefault("experiment", {})["purpose"] = (
                "development_only_executor_equivalence"
            )
            resolved["experiment"]["deployable_claim"] = False
            config_path = (
                output
                / "configs"
                / method_id
                / f"env_{num_envs}_selection_{selection_workers}.yaml"
            )
            if config_path.exists():
                import yaml

                with config_path.open("r", encoding="utf-8") as handle:
                    existing = yaml.safe_load(handle) or {}
                if plain(existing) != resolved:
                    raise FileExistsError(
                        f"refusing to replace a different acceleration config: {config_path}"
                    )
            else:
                write_yaml_atomic(config_path, resolved)
            run_root = resolve_path(resolved["run"]["output_root"]) / run_id / "stage3"
            rows.append(
                {
                    "method_id": method_id,
                    "ppo_num_envs": num_envs,
                    "n_steps": n_steps,
                    "checkpoint_selection_workers": selection_workers,
                    "config": str(config_path.resolve()),
                    "config_sha256": file_sha256(config_path),
                    "checkpoint": str((run_root / str(resolved["stage3"]["model_name"])).resolve()),
                    "stage3_report": str((run_root / "stage3_training_report.json").resolve()),
                    "executor_contract": executor_contract(resolved),
                }
            )
    plan: dict[str, Any] = {
        "artifact_kind": PLAN_KIND,
        "schema_version": 1,
        "status": "prepared",
        "protocol": public_protocol(protocol),
        "protocol_path": str(resolve_path(protocol_path)),
        "protocol_sha256": file_sha256(resolve_path(protocol_path)),
        "optimizer_seed": seed,
        "development_only": True,
        "rows": rows,
    }
    plan["plan_fingerprint"] = stable_hash(plan)
    return _write_or_validate(output / "acceleration_plan.json", plan)


def _validate_completed_row(row: Mapping[str, Any]) -> dict[str, Any]:
    checkpoint = resolve_path(str(row["checkpoint"]))
    report_path = resolve_path(str(row["stage3_report"]))
    if not checkpoint.is_file() or not report_path.is_file():
        raise FileNotFoundError(f"acceleration run is incomplete: {checkpoint.parent}")
    report = read_json(report_path)
    config_path = resolve_path(str(row["config"]))
    if file_sha256(config_path) != str(row["config_sha256"]):
        raise ValueError("acceleration config changed after planning")
    if int(report.get("optimizer_seed", -1)) < 0:
        raise ValueError("acceleration Stage3 report lacks optimizer seed")
    actual = {
        "ppo_num_envs": int(report.get("ppo_num_envs", -1)),
        "n_steps": int(report.get("ppo_n_steps_per_env", -1)),
        "rollout_size": int(report.get("ppo_rollout_size", -1)),
    }
    expected = dict(row["executor_contract"])
    if any(actual[key] != int(expected[key]) for key in actual):
        raise ValueError("acceleration Stage3 report executor contract mismatch")
    return {
        **plain(row),
        "checkpoint_sha256": file_sha256(checkpoint),
        "stage3_report_sha256": file_sha256(report_path),
        "policy_state_sha256s": _zip_member_hashes(checkpoint),
        "selection_semantics": _selection_semantics(report),
        "wall_time_s": float(report.get("wall_time", 0.0)),
        "steps_per_second": float(report.get("steps_per_second", 0.0)),
        "actual_total_timesteps": int(report.get("actual_total_timesteps", 0)),
        "reward_semantics_hash": str(report.get("reward_semantics_hash", "")),
        "observation_contract_hash": str(report.get("observation_contract_hash", "")),
    }


def execute(plan_path: str | Path) -> Path:
    source = resolve_path(plan_path)
    plan = read_json(source)
    declared = str(plan.get("plan_fingerprint", ""))
    core = {key: value for key, value in plan.items() if key != "plan_fingerprint"}
    if plan.get("artifact_kind") != PLAN_KIND or declared != stable_hash(core):
        raise ValueError("invalid acceleration plan fingerprint")
    completed: list[dict[str, Any]] = []
    for row in plan.get("rows", []) or []:
        checkpoint = resolve_path(str(row["checkpoint"]))
        report_path = resolve_path(str(row["stage3_report"]))
        if not checkpoint.is_file() or not report_path.is_file():
            run_dir = checkpoint.parent
            if run_dir.exists() and any(run_dir.iterdir()):
                raise RuntimeError(
                    "acceleration run has partial output without a complete checkpoint/report: "
                    f"{run_dir}"
                )
            produced = Path(stage3_train_ppo.run(load_config(row["config"]))).resolve()
            if produced != checkpoint:
                raise RuntimeError(
                    f"acceleration output path mismatch: expected={checkpoint} actual={produced}"
                )
        completed.append(_validate_completed_row(row))

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in completed:
        grouped.setdefault(str(row["method_id"]), []).append(row)
    summaries: dict[str, Any] = {}
    for method_id, rows in grouped.items():
        ordered = sorted(
            rows,
            key=lambda item: (
                int(item["ppo_num_envs"]),
                int(item["checkpoint_selection_workers"]),
            ),
        )
        reference = next(
            (
                item
                for item in ordered
                if int(item["ppo_num_envs"]) == 1
                and int(item["checkpoint_selection_workers"]) == 1
            ),
            ordered[0],
        )
        comparisons = []
        eligible = []
        for row in ordered:
            state_equal = row["policy_state_sha256s"] == reference["policy_state_sha256s"]
            selection_equal = row["selection_semantics"] == reference["selection_semantics"]
            contract_equal = (
                row["reward_semantics_hash"] == reference["reward_semantics_hash"]
                and row["observation_contract_hash"]
                == reference["observation_contract_hash"]
                and row["actual_total_timesteps"] == reference["actual_total_timesteps"]
            )
            equivalent = bool(state_equal and selection_equal and contract_equal)
            comparison = {
                "ppo_num_envs": int(row["ppo_num_envs"]),
                "n_steps": int(row["n_steps"]),
                "checkpoint_selection_workers": int(
                    row["checkpoint_selection_workers"]
                ),
                "state_equal": state_equal,
                "checkpoint_selection_equal": selection_equal,
                "semantic_contract_equal": contract_equal,
                "equivalent": equivalent,
                "wall_time_s": float(row["wall_time_s"]),
                "steps_per_second": float(row["steps_per_second"]),
            }
            comparisons.append(comparison)
            if equivalent:
                eligible.append(comparison)
        selected = max(eligible, key=lambda item: item["steps_per_second"]) if eligible else None
        summaries[method_id] = {
            "reference_executor": {
                "ppo_num_envs": int(reference["ppo_num_envs"]),
                "n_steps": int(reference["n_steps"]),
                "checkpoint_selection_workers": int(
                    reference["checkpoint_selection_workers"]
                ),
            },
            "comparisons": comparisons,
            "equivalence_pass": bool(eligible),
            "selected_executor": selected,
        }
    report: dict[str, Any] = {
        "artifact_kind": REPORT_KIND,
        "schema_version": 1,
        "status": "complete",
        "conclusion_scope": "executor_semantics_and_throughput_only",
        "plan": str(source),
        "plan_sha256": file_sha256(source),
        "rows": completed,
        "methods": summaries,
    }
    report["report_fingerprint"] = stable_hash(report)
    return _write_or_validate(source.parent / "acceleration_equivalence_report.json", report)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare or run 1/2/4-environment PPO executor equivalence benchmarks"
    )
    sub = parser.add_subparsers(dest="action", required=True)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--protocol", required=True)
    prepare_parser.add_argument("--output-root", required=True)
    prepare_parser.add_argument("--methods", nargs="+")
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--plan", required=True)
    args = parser.parse_args()
    if args.action == "prepare":
        output = prepare(
            protocol_path=args.protocol,
            output_root=args.output_root,
            methods=args.methods,
        )
    else:
        output = execute(args.plan)
    print(f"main_method_acceleration_artifact={output}")


if __name__ == "__main__":
    main()
