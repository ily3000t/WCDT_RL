from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from safe_rl.accvp.contracts.schema import (
    file_sha256,
    read_json,
    stable_hash,
    write_json_atomic,
)
from safe_rl.accvp.evaluation.selector_contract import (
    SELECTOR_INPUT_COVERAGE_KIND,
    run_selector_contract_audit,
    selector_audit_input_coverage,
)
from safe_rl.pipeline.common import make_env
from safe_rl.ppo_factorial import (
    EXPECTED_FINAL_METHOD_ID,
    read_json_mapping,
    resolve_manifest_path,
    validate_factorial_manifest,
)
from safe_rl.rl.ppo import load_ppo
from safe_rl.sim.types import VehicleState
from safe_rl.utils.config import REPO_ROOT, load_config


DEFAULT_OPTIMIZER_SEEDS = (1002, 1004)
DEFAULT_SIMULATOR_SEEDS = (50021, 50027)


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (REPO_ROOT / value).resolve()


def _final_method_records(
    manifest_path: str | Path,
    optimizer_seeds: list[int],
) -> list[dict[str, Any]]:
    source = _resolve(manifest_path)
    factorial = validate_factorial_manifest(
        source,
        require_complete=True,
        verify_files=True,
    )
    final_entry = dict(factorial["methods"][EXPECTED_FINAL_METHOD_ID])
    child_path = resolve_manifest_path(
        source,
        str(final_entry["replicate_manifest"]),
    )
    child = read_json_mapping(child_path)
    by_seed = {
        int(row["optimizer_seed"]): dict(row)
        for row in list(child.get("records", []) or [])
    }
    missing = sorted(set(optimizer_seeds).difference(by_seed))
    if missing:
        raise ValueError(
            "selector targeted replay is missing final-method PPO replicas: "
            f"{missing}"
        )
    return [by_seed[seed] for seed in optimizer_seeds]


def _capture_episode_states(
    record: dict[str, Any],
    simulator_seed: int,
) -> list[tuple[str, list[VehicleState], dict[str, Any]]]:
    optimizer_seed = int(record["optimizer_seed"])
    config_path = _resolve(str(record["resolved_config"]))
    checkpoint_path = _resolve(str(record["checkpoint"]))
    if file_sha256(config_path) != str(record["resolved_config_sha256"]):
        raise ValueError("selector replay resolved-config SHA-256 mismatch")
    if file_sha256(checkpoint_path) != str(record["checkpoint_sha256"]):
        raise ValueError("selector replay PPO checkpoint SHA-256 mismatch")
    cfg = load_config(config_path)
    env = make_env(
        cfg,
        seed=int(simulator_seed),
        shield_enabled=bool(cfg.shield.enabled),
    )
    model = load_ppo(checkpoint_path, device="cpu")
    records: list[tuple[str, list[VehicleState], dict[str, Any]]] = []
    try:
        observation, _info = env.reset(seed=int(simulator_seed))
        terminated = False
        truncated = False
        decision_index = 0
        while not (terminated or truncated):
            latest = list(env.history.latest().values())
            context = env.get_risk_context()
            merge_local = context.get("merge_local")
            taper_distance = (
                merge_local.get("merge_distance", float("inf"))
                if isinstance(merge_local, dict)
                else getattr(merge_local, "merge_distance", float("inf"))
            )
            if latest:
                records.append(
                    (
                        (
                            f"optimizer_{optimizer_seed}:"
                            f"simulator_{simulator_seed}:decision_{decision_index}"
                        ),
                        [
                            VehicleState(**vehicle.to_dict())
                            for vehicle in latest
                        ],
                        {
                            "scope": "selector_targeted_replay",
                            "optimizer_seed": optimizer_seed,
                            "episode_seed": int(simulator_seed),
                            "decision_index": int(decision_index),
                            "taper_distance_m": float(taper_distance),
                            "method_id": EXPECTED_FINAL_METHOD_ID,
                            "checkpoint_sha256": str(
                                record["checkpoint_sha256"]
                            ),
                            "seed_role": "selector_diagnostic_only",
                        },
                    )
                )
            action, _state = model.predict(observation, deterministic=True)
            observation, _reward, terminated, truncated, _info = env.step(
                int(action)
            )
            decision_index += 1
    finally:
        env.close()
    return records


def run(
    *,
    config_path: str | Path,
    dataset_dir: str | Path,
    factorial_manifest: str | Path,
    optimizer_seeds: list[int],
    simulator_seeds: list[int],
    output_path: str | Path,
    input_coverage_output: str | Path | None = None,
) -> Path:
    if len(optimizer_seeds) != 2 or len(set(optimizer_seeds)) != 2:
        raise ValueError("selector audit requires exactly two optimizer seeds")
    if len(simulator_seeds) != 2 or len(set(simulator_seeds)) != 2:
        raise ValueError("selector audit requires exactly two simulator seeds")
    if set(simulator_seeds) != set(DEFAULT_SIMULATOR_SEEDS):
        raise ValueError(
            "selector diagnostic simulator seeds are frozen to 50021/50027"
        )
    output = _resolve(output_path)
    if output.exists():
        raise FileExistsError(output)
    coverage_path = (
        _resolve(input_coverage_output)
        if input_coverage_output
        else output.with_name("selector_audit_input_coverage.json")
    )
    if coverage_path.exists():
        raise FileExistsError(coverage_path)

    cfg = load_config(_resolve(config_path))
    coverage = selector_audit_input_coverage(cfg, dataset_dir)
    write_json_atomic(coverage_path, coverage)
    if (
        str(coverage.get("artifact_kind", "")) != SELECTOR_INPUT_COVERAGE_KIND
        or str(coverage.get("input_coverage_state", "")) != "pass"
    ):
        raise RuntimeError(
            "selector audit input coverage failed; run selector-only root "
            f"recollection before capacity audit: {coverage_path}"
        )

    ppo_records = _final_method_records(factorial_manifest, optimizer_seeds)
    targeted_states: list[
        tuple[str, list[VehicleState], dict[str, Any]]
    ] = []
    for record in ppo_records:
        for simulator_seed in simulator_seeds:
            optimizer_seed = int(record["optimizer_seed"])
            print(
                "[selector_contract_audit] targeted replay "
                f"optimizer_seed={optimizer_seed} "
                f"simulator_seed={simulator_seed} start",
                flush=True,
            )
            captured = _capture_episode_states(record, simulator_seed)
            targeted_states.extend(captured)
            print(
                "[selector_contract_audit] targeted replay "
                f"optimizer_seed={optimizer_seed} "
                f"simulator_seed={simulator_seed} "
                f"decisions={len(captured)} end",
                flush=True,
            )
    report = run_selector_contract_audit(
        cfg,
        dataset_dir,
        targeted_states=targeted_states,
        minimum_formal_roots=5000,
    )
    report["input_coverage_artifact"] = {
        "path": str(coverage_path),
        "sha256": file_sha256(coverage_path),
        "report_fingerprint": str(coverage["report_fingerprint"]),
    }
    report["source_factorial_manifest"] = {
        "path": str(_resolve(factorial_manifest)),
        "sha256": file_sha256(_resolve(factorial_manifest)),
        "protocol_id": str(
            read_json(_resolve(factorial_manifest)).get("protocol_id", "")
        ),
        "usage": "diagnostic_state_source_only",
    }
    report["report_fingerprint"] = stable_hash(
        {
            key: value
            for key, value in report.items()
            if key != "report_fingerprint"
        }
    )
    write_json_atomic(output, report)
    print(
        "[selector_contract_audit] "
        f"state={report['audit_state']} "
        f"selected_capacity={report['selected_capacity']} "
        f"report={output}",
        flush=True,
    )
    if str(report["audit_state"]) != "pass":
        raise RuntimeError(
            "Selector-v3 capacity audit is blocked; do not collect pilot data"
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Audit Selector-v3 on all V1 formal root histories and four "
            "diagnostic final-policy replays, then freeze actor capacity 6 or 8."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--factorial-manifest", required=True)
    parser.add_argument(
        "--optimizer-seeds",
        nargs="+",
        type=int,
        default=list(DEFAULT_OPTIMIZER_SEEDS),
    )
    parser.add_argument(
        "--simulator-seeds",
        nargs="+",
        type=int,
        default=list(DEFAULT_SIMULATOR_SEEDS),
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--input-coverage-output")
    args = parser.parse_args()
    run(
        config_path=args.config,
        dataset_dir=args.dataset,
        factorial_manifest=args.factorial_manifest,
        optimizer_seeds=list(args.optimizer_seeds),
        simulator_seeds=list(args.simulator_seeds),
        output_path=args.output,
        input_coverage_output=args.input_coverage_output,
    )


if __name__ == "__main__":
    main()
