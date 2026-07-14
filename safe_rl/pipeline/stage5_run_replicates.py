from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from safe_rl.accvp.contracts.schema import file_sha256, read_json
from safe_rl.pipeline import stage5_paired_eval, stage5_replicated_aggregate
from safe_rl.pipeline.stage5_replicated_aggregate import REQUEST_ARTIFACT_KIND
from safe_rl.utils.config import REPO_ROOT, load_config


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (REPO_ROOT / value).resolve()


def run(
    *,
    generated_dir: str | Path,
    aggregate_output: str | Path | None = None,
) -> list[Path]:
    directory = _resolve(generated_dir)
    request_path = directory / "replicated_request.json"
    request = read_json(request_path)
    if str(request.get("artifact_kind", "")) != REQUEST_ARTIFACT_KIND:
        raise ValueError("unsupported Stage5 replicated request")
    rows = list(request.get("replicates", []) or [])
    if len(rows) < 5:
        raise ValueError("formal Stage5 execution requires at least five optimizer seeds")
    training_seeds = [int(row["training_seed"]) for row in rows]
    if len(training_seeds) != len(set(training_seeds)):
        raise ValueError("Stage5 replicated request contains duplicate optimizer seeds")
    left_hashes = [str(row["left_checkpoint_sha256"]) for row in rows]
    right_hashes = [str(row["right_checkpoint_sha256"]) for row in rows]
    if len(left_hashes) != len(set(left_hashes)) or len(right_hashes) != len(set(right_hashes)):
        raise ValueError("Stage5 formal replicates reuse a checkpoint within one method")
    prepared: list[tuple[dict[str, Any], Path, dict[str, Any]]] = []
    simulator_schedule: list[int] | None = None
    for row in sorted(rows, key=lambda item: int(item["training_seed"])):
        config_path = _resolve(row["stage5_config"])
        if file_sha256(config_path) != str(row["stage5_config_sha256"]):
            raise ValueError(f"generated Stage5 config hash mismatch: {config_path}")
        with config_path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        seeds = [int(seed) for seed in raw.get("stage5", {}).get("seeds", [])]
        if simulator_schedule is None:
            simulator_schedule = seeds
        elif seeds != simulator_schedule:
            raise ValueError("replicated Stage5 configs do not share one simulator seed schedule")
        groups = list(raw.get("stage5", {}).get("groups", []) or [])
        recorded = {str(group.get("name")): group for group in groups}
        left = recorded.get(str(row["left_group"]), {})
        right = recorded.get(str(row["right_group"]), {})
        if file_sha256(_resolve(left.get("model_path", ""))) != str(row["left_checkpoint_sha256"]):
            raise ValueError("left Stage5 checkpoint hash mismatch")
        if file_sha256(_resolve(right.get("model_path", ""))) != str(row["right_checkpoint_sha256"]):
            raise ValueError("right Stage5 checkpoint hash mismatch")
        preflight = _resolve(raw.get("stage5", {}).get("accvp_observation_preflight_report", ""))
        runtime = read_json(preflight)
        if not bool(runtime.get("gate", {}).get("pass", False)):
            raise ValueError(f"candidate policy runtime gate failed: {preflight}")
        report_path = _resolve(row["stage5_report"])
        if report_path.exists():
            raise FileExistsError(f"refusing to overwrite formal Stage5 report: {report_path}")
        prepared.append((row, config_path, raw))
    produced: list[Path] = []
    for row, config_path, _raw in prepared:
        stage_dir = Path(stage5_paired_eval.run(load_config(config_path))).resolve()
        report_path = stage_dir / "formal_paired_eval_report.json"
        expected = _resolve(row["stage5_report"])
        if report_path != expected or not report_path.is_file():
            raise RuntimeError(
                f"Stage5 output disagrees with frozen request: actual={report_path} expected={expected}"
            )
        produced.append(report_path)
    if aggregate_output is not None:
        produced.append(
            stage5_replicated_aggregate.run(request_path, _resolve(aggregate_output)).resolve()
        )
    return produced


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute frozen paired Stage5 optimizer replicates")
    parser.add_argument("--generated-dir", required=True)
    parser.add_argument("--aggregate-output")
    args = parser.parse_args()
    paths = run(generated_dir=args.generated_dir, aggregate_output=args.aggregate_output)
    print(f"stage5_replicate_reports={len(paths)}")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
