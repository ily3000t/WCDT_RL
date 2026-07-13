from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from safe_rl.accvp.schema import file_sha256, read_json
from safe_rl.evaluation_protocol import seeds_for_role, stable_hash
from safe_rl.pipeline import stage3_train_ppo
from safe_rl.ppo_replicates import (
    MIN_FORMAL_OPTIMIZER_REPLICATES,
    REPLICATE_MANIFEST_KIND,
    REPLICATE_MANIFEST_SCHEMA_VERSION,
    apply_variant,
    observation_contract,
    plain,
    validate_reward_semantics,
    write_json_new,
    write_yaml_atomic,
)
from safe_rl.utils.config import REPO_ROOT, load_config


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (REPO_ROOT / value).resolve()


def _load_matrix(path: str | Path, method_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    source = _resolve(path)
    with source.open("r", encoding="utf-8") as handle:
        matrix = yaml.safe_load(handle) or {}
    if str(matrix.get("artifact_kind", "")) != "accvp_vnext_ppo_ablation_matrix_v1":
        raise ValueError("unsupported PPO ablation matrix")
    variants = matrix.get("variants", {}) or {}
    if method_id not in variants:
        raise ValueError(f"unknown method_id={method_id!r}; available={sorted(variants)}")
    return matrix, dict(variants[method_id])


def build_replicate_plan(
    *,
    config_path: str | Path,
    matrix_path: str | Path,
    method_id: str,
    optimizer_seeds: list[int],
    output_root: str | Path,
    run_id_prefix: str | None = None,
    require_artifacts: bool = True,
) -> tuple[dict[str, Any], list[tuple[Path, dict[str, Any]]]]:
    seeds = [int(seed) for seed in optimizer_seeds]
    if len(seeds) < MIN_FORMAL_OPTIMIZER_REPLICATES or len(seeds) != len(set(seeds)):
        raise ValueError(
            f"formal PPO replication requires at least {MIN_FORMAL_OPTIMIZER_REPLICATES} unique optimizer seeds"
        )
    base_cfg = load_config(_resolve(config_path))
    matrix, variant = _load_matrix(matrix_path, method_id)
    expected = seeds_for_role(base_cfg, str(matrix.get("optimizer_seed_role", "ppo_optimizer_replicates")))
    if expected and set(seeds) != set(expected):
        raise ValueError(
            "optimizer seeds must exactly match the frozen seed-ledger role: "
            f"requested={sorted(seeds)} expected={sorted(expected)}"
        )
    base = apply_variant(plain(base_cfg), variant)
    simulator_seed = int(base["run"]["seed"])
    prefix = str(run_id_prefix or f"ppo_{method_id}")
    root = _resolve(output_root)
    generated_dir = root / "generated_configs"
    records: list[dict[str, Any]] = []
    configs: list[tuple[Path, dict[str, Any]]] = []
    run_output_root = _resolve(base["run"]["output_root"])
    for seed in seeds:
        resolved = plain(base)
        run_id = f"{prefix}_seed_{seed}"
        resolved["run"]["run_id"] = run_id
        resolved["rl"]["optimizer_seed"] = seed
        resolved.setdefault("experiment", {})["method_id"] = method_id
        resolved["experiment"]["optimizer_seed"] = seed
        if int(resolved["run"]["seed"]) != simulator_seed:
            raise AssertionError("replicate construction changed the simulator seed")
        reward = validate_reward_semantics(resolved)
        observation = observation_contract(resolved, require_artifacts=require_artifacts)
        config_file = generated_dir / f"{run_id}.yaml"
        run_dir = run_output_root / run_id
        if run_dir.exists():
            raise FileExistsError(f"refusing to reuse PPO replicate run directory: {run_dir}")
        records.append(
            {
                "method_id": method_id,
                "training_seed": seed,
                "optimizer_seed": seed,
                "simulator_training_start_seed": simulator_seed,
                "run_id": run_id,
                "resolved_config": str(config_file),
                "resolved_config_sha256": "",
                "checkpoint": "",
                "checkpoint_sha256": "",
                "stage3_report": "",
                "stage3_report_sha256": "",
                "training_budget": {
                    "total_timesteps": int(resolved["rl"]["total_timesteps"]),
                    "n_steps": int(resolved["rl"]["n_steps"]),
                    "batch_size": int(resolved["rl"]["batch_size"]),
                    "ppo_num_envs": int(resolved.get("training", {}).get("ppo_num_envs", 1)),
                },
                "reward_semantics_hash": reward["sha256"],
                "reward_semantics": reward["payload"],
                "observation_contract_hash": observation["sha256"],
                "observation_contract": observation["payload"],
                "accvp_artifact_fingerprint": observation["accvp_artifact_fingerprint"],
            }
        )
        configs.append((config_file, resolved))
    manifest = {
        "artifact_kind": REPLICATE_MANIFEST_KIND,
        "schema_version": REPLICATE_MANIFEST_SCHEMA_VERSION,
        "status": "planned",
        "method_id": method_id,
        "minimum_optimizer_replicates": int(matrix.get("minimum_optimizer_replicates", 5)),
        "optimizer_seed_role": str(matrix.get("optimizer_seed_role", "ppo_optimizer_replicates")),
        "optimizer_seeds": seeds,
        "simulator_training_start_seed": simulator_seed,
        "template_config": str(_resolve(config_path)),
        "template_config_sha256": file_sha256(_resolve(config_path)),
        "ablation_matrix": str(_resolve(matrix_path)),
        "ablation_matrix_sha256": file_sha256(_resolve(matrix_path)),
        "records": records,
    }
    manifest["plan_fingerprint"] = stable_hash(manifest)
    return manifest, configs


def run(
    *,
    config_path: str | Path,
    matrix_path: str | Path,
    method_id: str,
    optimizer_seeds: list[int],
    output_root: str | Path,
    run_id_prefix: str | None = None,
    prepare_only: bool = False,
) -> Path:
    manifest, configs = build_replicate_plan(
        config_path=config_path,
        matrix_path=matrix_path,
        method_id=method_id,
        optimizer_seeds=optimizer_seeds,
        output_root=output_root,
        run_id_prefix=run_id_prefix,
        require_artifacts=not prepare_only,
    )
    records = manifest["records"]
    for index, (config_file, resolved) in enumerate(configs):
        write_yaml_atomic(config_file, resolved)
        records[index]["resolved_config_sha256"] = file_sha256(config_file)
        if prepare_only:
            continue
        cfg = load_config(config_file)
        checkpoint = Path(stage3_train_ppo.run(cfg)).resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Stage3 did not produce checkpoint: {checkpoint}")
        report_path = checkpoint.parent / "stage3_training_report.json"
        report = read_json(report_path)
        if int(report.get("optimizer_seed", -1)) != int(records[index]["optimizer_seed"]):
            raise ValueError("Stage3 report optimizer seed disagrees with replicate plan")
        for field in ("reward_semantics_hash", "observation_contract_hash"):
            if str(report.get(field, "")) != str(records[index][field]):
                raise ValueError(f"Stage3 report {field} disagrees with replicate plan")
        records[index].update(
            {
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": file_sha256(checkpoint),
                "stage3_report": str(report_path),
                "stage3_report_sha256": file_sha256(report_path),
                "actual_total_timesteps": int(report.get("actual_total_timesteps", 0)),
                "selection_seed_sha256": str(report.get("checkpoint_selection_seed_sha256", "")),
            }
        )
    manifest["status"] = "prepared" if prepare_only else "complete"
    manifest["manifest_fingerprint"] = stable_hash(
        {key: value for key, value in manifest.items() if key != "manifest_fingerprint"}
    )
    output = _resolve(output_root) / "ppo_replicate_manifest.json"
    return write_json_new(output, manifest)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train independent PPO optimizer-seed replicates without changing simulator cohorts"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--matrix",
        default="safe_rl/config/active/accvp_vnext/ppo_ablation_matrix.yaml",
    )
    parser.add_argument("--method-id", required=True)
    parser.add_argument("--training-seeds", "--optimizer-seeds", nargs="+", type=int, required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--run-id-prefix")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    output = run(
        config_path=args.config,
        matrix_path=args.matrix,
        method_id=args.method_id,
        optimizer_seeds=args.training_seeds,
        output_root=args.output_root,
        run_id_prefix=args.run_id_prefix,
        prepare_only=args.prepare_only,
    )
    print(f"ppo_replicate_manifest={output}")


if __name__ == "__main__":
    main()
