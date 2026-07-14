from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any, Mapping

import yaml

from safe_rl.accvp.contracts.schema import file_sha256
from safe_rl.evaluation_protocol import stable_hash
from safe_rl.pipeline import stage3_train_ppo
from safe_rl.pipeline.stage3_train_ppo_replicates import build_replicate_plan
from safe_rl.ppo_factorial import (
    EXPECTED_CANDIDATE_METHOD_ROLES,
    FACTORIAL_MANIFEST_KIND,
    FACTORIAL_PLAN_KIND,
    FACTORIAL_PLAN_SCHEMA_VERSION,
    atomic_write_json,
    build_factorial_manifest,
    fingerprint_payload,
    load_factorial_contract,
    read_json_mapping,
    resolve_path,
    validate_factorial_manifest,
    validate_replicate_manifest,
)
from safe_rl.ppo_replicates import (
    observation_contract,
    plain,
    validate_reward_semantics,
    write_yaml_atomic,
)
from safe_rl.utils.config import load_config


PLAN_FILENAME = "factorial_plan.json"
MANIFEST_FILENAME = "ppo_factorial_manifest.json"
CHILD_MANIFEST_FILENAME = "ppo_replicate_manifest.json"

_DYNAMIC_RECORD_FIELDS = {
    "resolved_config_sha256",
    "checkpoint",
    "checkpoint_sha256",
    "stage3_report",
    "stage3_report_sha256",
    "actual_total_timesteps",
    "selection_seed_sha256",
}


def _validated_seeds(seeds: list[int], minimum: int) -> list[int]:
    values = [int(seed) for seed in seeds]
    if len(values) < int(minimum) or len(values) != len(set(values)):
        raise ValueError(f"factorial PPO requires at least {minimum} unique optimizer seeds")
    return values


def _config_payload_hash(payload: Mapping[str, Any]) -> str:
    return stable_hash(plain(payload))


def _plan_without_fingerprint(plan: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in plain(plan).items() if key != "plan_fingerprint"}


def build_factorial_plan(
    *,
    config_path: str | Path,
    matrix_path: str | Path,
    workflow_path: str | Path,
    optimizer_seeds: list[int],
    output_root: str | Path,
    require_artifacts: bool = True,
) -> dict[str, Any]:
    contract = load_factorial_contract(workflow_path, matrix_path)
    seeds = _validated_seeds(optimizer_seeds, contract["minimum_optimizer_replicates"])
    root = resolve_path(output_root)
    config_source = resolve_path(config_path)
    methods: dict[str, Any] = {}
    run_ids: set[str] = set()
    config_paths: set[str] = set()
    budget_hashes: set[str] = set()
    observation_hashes: set[str] = set()

    for method_id in contract["method_ids"]:
        method_root = root / "methods" / method_id
        replicate_plan, configs = build_replicate_plan(
            config_path=config_source,
            matrix_path=contract["matrix_path"],
            method_id=method_id,
            optimizer_seeds=seeds,
            output_root=method_root,
            run_id_prefix=f"ppo_accvp_vnext_{method_id}",
            require_artifacts=require_artifacts,
        )
        config_records: list[dict[str, Any]] = []
        for config_file, resolved in configs:
            payload = plain(resolved)
            config_path_value = str(Path(config_file).resolve())
            if config_path_value in config_paths:
                raise ValueError("factorial plan generated duplicate resolved-config paths")
            config_paths.add(config_path_value)
            config_records.append(
                {
                    "path": config_path_value,
                    "payload_hash": _config_payload_hash(payload),
                    "payload": payload,
                }
            )
        for record in replicate_plan["records"]:
            run_id = str(record["run_id"])
            if run_id in run_ids:
                raise ValueError("factorial plan generated duplicate PPO run ids")
            run_ids.add(run_id)
            budget_hashes.add(stable_hash(record["training_budget"]))
            observation_hashes.add(str(record["observation_contract_hash"]))
        methods[method_id] = {
            "role": contract["method_roles"][method_id],
            "method_root": str(method_root.resolve()),
            "replicate_manifest": str((method_root / CHILD_MANIFEST_FILENAME).resolve()),
            "replicate_plan": replicate_plan,
            "configs": config_records,
        }
    if len(budget_hashes) != 1:
        raise ValueError("Candidate factorial methods must use an identical PPO training budget")
    if len(observation_hashes) != 1:
        raise ValueError("Candidate factorial methods must use an identical observation contract")

    plan = {
        "artifact_kind": FACTORIAL_PLAN_KIND,
        "schema_version": FACTORIAL_PLAN_SCHEMA_VERSION,
        "status": "planned",
        "protocol_id": contract["protocol_id"],
        "output_root": str(root),
        "template_config": str(config_source),
        "template_config_sha256": file_sha256(config_source),
        "workflow_config": contract["workflow_path"],
        "workflow_config_sha256": contract["workflow_sha256"],
        "ablation_matrix": contract["matrix_path"],
        "ablation_matrix_sha256": contract["matrix_sha256"],
        "optimizer_seed_role": contract["optimizer_seed_role"],
        "optimizer_seeds": seeds,
        "method_roles": contract["method_roles"],
        "final_method_id": contract["final_method_id"],
        "training_budget_hash": next(iter(budget_hashes)),
        "observation_contract_hash": next(iter(observation_hashes)),
        "methods": methods,
    }
    plan["plan_fingerprint"] = stable_hash(_plan_without_fingerprint(plan))
    return plan


def _validate_frozen_plan(
    plan: Mapping[str, Any],
    *,
    config_path: str | Path,
    matrix_path: str | Path,
    workflow_path: str | Path,
    optimizer_seeds: list[int],
    output_root: str | Path,
) -> dict[str, Any]:
    payload = plain(plan)
    if payload.get("artifact_kind") != FACTORIAL_PLAN_KIND:
        raise ValueError("resume target does not contain a PPO factorial plan")
    if int(payload.get("schema_version", -1)) != FACTORIAL_PLAN_SCHEMA_VERSION:
        raise ValueError("unsupported PPO factorial plan schema")
    declared_fingerprint = str(payload.get("plan_fingerprint", ""))
    if declared_fingerprint != stable_hash(_plan_without_fingerprint(payload)):
        raise ValueError("PPO factorial plan fingerprint mismatch")
    contract = load_factorial_contract(workflow_path, matrix_path)
    seeds = _validated_seeds(optimizer_seeds, contract["minimum_optimizer_replicates"])
    expected_inputs = {
        "output_root": str(resolve_path(output_root)),
        "template_config": str(resolve_path(config_path)),
        "template_config_sha256": file_sha256(resolve_path(config_path)),
        "workflow_config": contract["workflow_path"],
        "workflow_config_sha256": contract["workflow_sha256"],
        "ablation_matrix": contract["matrix_path"],
        "ablation_matrix_sha256": contract["matrix_sha256"],
        "optimizer_seeds": seeds,
        "method_roles": contract["method_roles"],
        "final_method_id": contract["final_method_id"],
    }
    for field, expected in expected_inputs.items():
        if payload.get(field) != expected:
            raise ValueError(f"resume request changes frozen factorial input {field!r}")
    methods = payload.get("methods", {}) or {}
    if set(methods) != set(EXPECTED_CANDIDATE_METHOD_ROLES):
        raise ValueError("frozen PPO factorial plan has an incomplete Candidate method set")
    for method_id, expected_role in EXPECTED_CANDIDATE_METHOD_ROLES.items():
        entry = methods[method_id]
        if str(entry.get("role", "")) != expected_role:
            raise ValueError(f"{method_id}: frozen plan method role mismatch")
        replicate_plan = entry.get("replicate_plan", {}) or {}
        if str(replicate_plan.get("method_id", "")) != method_id:
            raise ValueError(f"{method_id}: frozen child plan method_id mismatch")
        if [int(seed) for seed in replicate_plan.get("optimizer_seeds", [])] != seeds:
            raise ValueError(f"{method_id}: frozen child seed schedule mismatch")
        summary = validate_replicate_manifest(
            replicate_plan,
            method_id=method_id,
            expected_seeds=seeds,
            verify_files=False,
        )
        if summary["training_budget_hash"] != str(payload.get("training_budget_hash", "")):
            raise ValueError(f"{method_id}: frozen plan training budget summary mismatch")
        if summary["observation_contract_hash"] != str(payload.get("observation_contract_hash", "")):
            raise ValueError(f"{method_id}: frozen plan observation contract summary mismatch")
        configs = entry.get("configs", []) or []
        if len(configs) != len(seeds):
            raise ValueError(f"{method_id}: frozen plan has the wrong resolved-config count")
        for config in configs:
            if str(config.get("payload_hash", "")) != _config_payload_hash(config.get("payload", {}) or {}):
                raise ValueError(f"{method_id}: frozen resolved-config payload hash mismatch")
    return payload


def _load_or_create_plan(
    *,
    config_path: str | Path,
    matrix_path: str | Path,
    workflow_path: str | Path,
    optimizer_seeds: list[int],
    output_root: str | Path,
    resume: bool,
    require_artifacts: bool,
) -> tuple[Path, dict[str, Any]]:
    root = resolve_path(output_root)
    plan_path = root / PLAN_FILENAME
    if plan_path.is_file():
        if not resume:
            raise FileExistsError(f"factorial plan already exists and --no-resume was selected: {plan_path}")
        plan = _validate_frozen_plan(
            read_json_mapping(plan_path),
            config_path=config_path,
            matrix_path=matrix_path,
            workflow_path=workflow_path,
            optimizer_seeds=optimizer_seeds,
            output_root=root,
        )
        return plan_path, plan
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(
            f"refusing to infer resume state without {PLAN_FILENAME}: {root}"
        )
    plan = build_factorial_plan(
        config_path=config_path,
        matrix_path=matrix_path,
        workflow_path=workflow_path,
        optimizer_seeds=optimizer_seeds,
        output_root=root,
        require_artifacts=require_artifacts,
    )
    atomic_write_json(plan_path, plan, replace=False)
    return plan_path, plan


def _materialise_configs(method: Mapping[str, Any]) -> list[tuple[Path, dict[str, Any]]]:
    result: list[tuple[Path, dict[str, Any]]] = []
    for entry in method.get("configs", []) or []:
        path = Path(str(entry["path"])).resolve()
        payload = plain(entry["payload"])
        if _config_payload_hash(payload) != str(entry["payload_hash"]):
            raise ValueError(f"frozen config payload fingerprint mismatch: {path}")
        if path.is_file():
            with path.open("r", encoding="utf-8") as handle:
                existing = yaml.safe_load(handle) or {}
            if _config_payload_hash(existing) != str(entry["payload_hash"]):
                raise ValueError(f"existing generated config disagrees with frozen plan: {path}")
        else:
            write_yaml_atomic(path, payload)
        result.append((path, payload))
    return result


def _immutable_record(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: plain(value)
        for key, value in record.items()
        if key not in _DYNAMIC_RECORD_FIELDS
    }


def _write_child_manifest(path: Path, manifest: dict[str, Any]) -> Path:
    manifest.pop("manifest_fingerprint", None)
    manifest["manifest_fingerprint"] = fingerprint_payload(manifest, "manifest_fingerprint")
    return atomic_write_json(path, manifest, replace=path.exists())


def _load_or_create_child_manifest(
    method_id: str,
    method: Mapping[str, Any],
    configs: list[tuple[Path, dict[str, Any]]],
    seeds: list[int],
) -> tuple[Path, dict[str, Any]]:
    child_path = Path(str(method["replicate_manifest"])).resolve()
    planned = copy.deepcopy(plain(method["replicate_plan"]))
    for index, (config_path, _payload) in enumerate(configs):
        planned["records"][index]["resolved_config_sha256"] = file_sha256(config_path)
    if not child_path.is_file():
        planned["status"] = "planned"
        _write_child_manifest(child_path, planned)
        return child_path, planned
    existing = read_json_mapping(child_path)
    if str(existing.get("plan_fingerprint", "")) != str(planned.get("plan_fingerprint", "")):
        raise ValueError(f"{method_id}: existing child manifest belongs to another plan")
    if len(existing.get("records", []) or []) != len(planned["records"]):
        raise ValueError(f"{method_id}: existing child manifest has the wrong record count")
    for expected, actual in zip(planned["records"], existing["records"]):
        if _immutable_record(actual) != _immutable_record(expected):
            raise ValueError(f"{method_id}: existing child manifest changes a frozen replicate field")
        if str(actual.get("resolved_config_sha256", "")) != str(expected["resolved_config_sha256"]):
            raise ValueError(f"{method_id}: generated config hash changed after planning")
    validate_replicate_manifest(
        existing,
        method_id=method_id,
        expected_seeds=seeds,
        verify_files=True,
    )
    return child_path, existing


def _expected_run_artifacts(config: Mapping[str, Any]) -> tuple[Path, Path, Path]:
    output_root = resolve_path(str(config["run"]["output_root"]))
    run_dir = output_root / str(config["run"]["run_id"])
    stage_dir = run_dir / "stage3"
    return run_dir, stage_dir / str(config["stage3"]["model_name"]), stage_dir / "stage3_training_report.json"


def _capture_stage3_record(
    record: dict[str, Any],
    config: Mapping[str, Any],
    checkpoint: str | Path,
    report_path: str | Path,
) -> None:
    checkpoint_source = Path(checkpoint).resolve()
    report_source = Path(report_path).resolve()
    if not checkpoint_source.is_file():
        raise FileNotFoundError(f"Stage3 did not produce checkpoint: {checkpoint_source}")
    if not report_source.is_file():
        raise FileNotFoundError(f"Stage3 did not produce training report: {report_source}")
    report = read_json_mapping(report_source)
    if int(report.get("optimizer_seed", -1)) != int(record["optimizer_seed"]):
        raise ValueError("Stage3 report optimizer seed disagrees with factorial plan")
    for field in ("reward_semantics_hash", "observation_contract_hash"):
        if str(report.get(field, "")) != str(record[field]):
            raise ValueError(f"Stage3 report {field} disagrees with factorial plan")
    expected_checkpoint = _expected_run_artifacts(config)[1]
    if checkpoint_source != expected_checkpoint.resolve():
        raise ValueError(
            "Stage3 checkpoint path disagrees with the frozen generated config: "
            f"expected={expected_checkpoint} actual={checkpoint_source}"
        )
    record.update(
        {
            "checkpoint": str(checkpoint_source),
            "checkpoint_sha256": file_sha256(checkpoint_source),
            "stage3_report": str(report_source),
            "stage3_report_sha256": file_sha256(report_source),
            "actual_total_timesteps": int(report.get("actual_total_timesteps", 0)),
            "selection_seed_sha256": str(report.get("checkpoint_selection_seed_sha256", "")),
        }
    )


def _remove_empty_run_tree(run_dir: Path) -> bool:
    """Remove a run-directory shell only when it contains no files or links.

    Stage3 creates ``run/stage3`` before constructing the environment.  If the
    process exits at that point, a later factorial resume sees an existing run
    directory even though there is no training state to preserve.  Treat that
    exact case as safely restartable while continuing to fail closed for every
    non-empty partial run.
    """

    if not run_dir.is_dir() or run_dir.is_symlink():
        return False
    descendants = list(run_dir.rglob("*"))
    if any(path.is_symlink() or not path.is_dir() for path in descendants):
        return False
    try:
        for directory in sorted(descendants, key=lambda path: len(path.parts), reverse=True):
            directory.rmdir()
        run_dir.rmdir()
    except OSError:
        return False
    return True


def _recover_or_train_record(
    *,
    method_id: str,
    record: dict[str, Any],
    config_path: Path,
    config_payload: Mapping[str, Any],
) -> str:
    # Revalidate formal contracts at the last point before an expensive SUMO run.
    cfg = load_config(config_path)
    reward = validate_reward_semantics(cfg)
    observation = observation_contract(cfg, require_artifacts=True)
    if reward["sha256"] != str(record["reward_semantics_hash"]):
        raise ValueError(f"{method_id}: generated config reward semantics changed after planning")
    if observation["sha256"] != str(record["observation_contract_hash"]):
        raise ValueError(f"{method_id}: generated config observation contract changed after planning")

    run_dir, expected_checkpoint, expected_report = _expected_run_artifacts(config_payload)
    if record.get("checkpoint") or record.get("stage3_report"):
        _capture_stage3_record(
            record,
            config_payload,
            str(record.get("checkpoint", "")),
            str(record.get("stage3_report", "")),
        )
        return "reused_manifest"
    if expected_checkpoint.is_file() and expected_report.is_file():
        _capture_stage3_record(record, config_payload, expected_checkpoint, expected_report)
        return "recovered_artifacts"
    if run_dir.exists():
        if _remove_empty_run_tree(run_dir):
            print(
                f"[ppo_factorial] empty_run_shell_removed path={run_dir}",
                flush=True,
            )
        else:
            raise FileExistsError(
                "partial PPO replicate run directory contains files or links and cannot be "
                "overwritten automatically; inspect/archive it before resuming: "
                f"{run_dir}"
            )
    checkpoint = Path(stage3_train_ppo.run(cfg)).resolve()
    _capture_stage3_record(record, config_payload, checkpoint, checkpoint.parent / "stage3_training_report.json")
    return "trained"


def _write_total_manifest(
    *,
    output: Path,
    plan_path: Path,
    plan: Mapping[str, Any],
    child_paths: Mapping[str, Path],
    status: str,
) -> Path:
    payload = build_factorial_manifest(
        protocol_id=str(plan["protocol_id"]),
        final_method_id=str(plan["final_method_id"]),
        method_roles=plan["method_roles"],
        optimizer_seeds=[int(seed) for seed in plan["optimizer_seeds"]],
        plan_path=plan_path,
        child_manifests=child_paths,
        status=status,
        verify_files=True,
    )
    return atomic_write_json(output, payload, replace=output.exists())


def run(
    *,
    config_path: str | Path,
    matrix_path: str | Path,
    workflow_path: str | Path,
    optimizer_seeds: list[int],
    output_root: str | Path,
    prepare_only: bool = False,
    resume: bool = True,
) -> Path:
    root = resolve_path(output_root)
    plan_path, plan = _load_or_create_plan(
        config_path=config_path,
        matrix_path=matrix_path,
        workflow_path=workflow_path,
        optimizer_seeds=optimizer_seeds,
        output_root=root,
        resume=resume,
        require_artifacts=not prepare_only,
    )
    seeds = [int(seed) for seed in plan["optimizer_seeds"]]
    child_paths: dict[str, Path] = {}
    children: dict[str, dict[str, Any]] = {}
    configs_by_method: dict[str, list[tuple[Path, dict[str, Any]]]] = {}
    for method_id in EXPECTED_CANDIDATE_METHOD_ROLES:
        method = plan["methods"][method_id]
        configs = _materialise_configs(method)
        child_path, child = _load_or_create_child_manifest(method_id, method, configs, seeds)
        child_paths[method_id] = child_path
        children[method_id] = child
        configs_by_method[method_id] = configs

    output = root / MANIFEST_FILENAME
    initial_status = "prepared" if prepare_only else "planned"
    if prepare_only:
        for method_id, child in children.items():
            if child.get("status") != "complete":
                child["status"] = "prepared"
                _write_child_manifest(child_paths[method_id], child)
        if all(child.get("status") == "complete" for child in children.values()):
            initial_status = "complete"
        _write_total_manifest(
            output=output,
            plan_path=plan_path,
            plan=plan,
            child_paths=child_paths,
            status=initial_status,
        )
        print(f"[ppo_factorial] prepared methods={len(children)} seeds_per_method={len(seeds)}", flush=True)
        return output

    _write_total_manifest(
        output=output,
        plan_path=plan_path,
        plan=plan,
        child_paths=child_paths,
        status="planned" if not all(child.get("status") == "complete" for child in children.values()) else "complete",
    )
    for method_id in EXPECTED_CANDIDATE_METHOD_ROLES:
        child = children[method_id]
        configs = configs_by_method[method_id]
        if child.get("status") == "complete":
            print(f"[ppo_factorial] method={method_id} status=complete action=skip", flush=True)
            continue
        child["status"] = "planned"
        _write_child_manifest(child_paths[method_id], child)
        for index, (config_path, config_payload) in enumerate(configs):
            seed = int(child["records"][index]["optimizer_seed"])
            print(f"[ppo_factorial] method={method_id} optimizer_seed={seed} start", flush=True)
            action = _recover_or_train_record(
                method_id=method_id,
                record=child["records"][index],
                config_path=config_path,
                config_payload=config_payload,
            )
            _write_child_manifest(child_paths[method_id], child)
            _write_total_manifest(
                output=output,
                plan_path=plan_path,
                plan=plan,
                child_paths=child_paths,
                status="planned",
            )
            print(
                f"[ppo_factorial] method={method_id} optimizer_seed={seed} done action={action}",
                flush=True,
            )
        child["status"] = "complete"
        _write_child_manifest(child_paths[method_id], child)
        print(f"[ppo_factorial] method={method_id} status=complete", flush=True)

    _write_total_manifest(
        output=output,
        plan_path=plan_path,
        plan=plan,
        child_paths=child_paths,
        status="complete",
    )
    validate_factorial_manifest(output, require_complete=True, verify_files=True)
    print(f"[ppo_factorial] status=complete manifest={output}", flush=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the frozen 2x2 Candidate PPO Reward x commitment factorial with safe resume"
    )
    parser.add_argument("--config", required=True, help="Canonical Candidate PPO template")
    parser.add_argument(
        "--matrix",
        default="safe_rl/config/active/accvp_vnext/ppo_ablation_matrix.yaml",
    )
    parser.add_argument(
        "--workflow-config",
        default="safe_rl/config/active/accvp_vnext/workflow.yaml",
    )
    parser.add_argument("--optimizer-seeds", "--training-seeds", nargs="+", type=int, required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--prepare-only", action="store_true")
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument("--resume", dest="resume", action="store_true", default=True)
    resume_group.add_argument("--no-resume", dest="resume", action="store_false")
    args = parser.parse_args()
    output = run(
        config_path=args.config,
        matrix_path=args.matrix,
        workflow_path=args.workflow_config,
        optimizer_seeds=args.optimizer_seeds,
        output_root=args.output_root,
        prepare_only=args.prepare_only,
        resume=args.resume,
    )
    print(f"ppo_factorial_manifest={output}")


if __name__ == "__main__":
    main()
