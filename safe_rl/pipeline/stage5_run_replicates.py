from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from safe_rl.accvp.contracts.schema import file_sha256, read_json
from safe_rl.evaluation_protocol import stable_hash
from safe_rl.pipeline import stage5_paired_eval, stage5_replicated_aggregate
from safe_rl.pipeline.stage5_replicated_aggregate import (
    REPORT_ARTIFACT_KIND,
    REQUEST_ARTIFACT_KIND,
)
from safe_rl.ppo_replicates import write_json_new
from safe_rl.utils.config import REPO_ROOT, load_config


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (REPO_ROOT / value).resolve()


def _receipt_path(report_path: Path) -> Path:
    return report_path.with_name(f"{report_path.stem}.execution_receipt.json")


def _receipt_payload(row: dict[str, Any], report_path: Path) -> dict[str, Any]:
    payload = {
        "artifact_kind": "stage5_replicate_execution_receipt_v1",
        "schema_version": 1,
        "training_seed": int(row["training_seed"]),
        "stage5_config_sha256": str(row["stage5_config_sha256"]),
        "left_group": str(row["left_group"]),
        "right_group": str(row["right_group"]),
        "left_checkpoint_sha256": str(row["left_checkpoint_sha256"]),
        "right_checkpoint_sha256": str(row["right_checkpoint_sha256"]),
        "stage5_report": str(report_path.resolve()),
        "stage5_report_sha256": file_sha256(report_path),
    }
    payload["receipt_fingerprint"] = stable_hash(payload)
    return payload


def _validate_resume_receipt(row: dict[str, Any], report_path: Path) -> None:
    receipt_path = _receipt_path(report_path)
    if not receipt_path.is_file():
        raise FileExistsError(
            "refusing to reuse a formal Stage5 report without an execution receipt: "
            f"{report_path}"
        )
    receipt = read_json(receipt_path)
    expected = _receipt_payload(row, report_path)
    if receipt != expected:
        raise ValueError(f"Stage5 execution receipt mismatch: {receipt_path}")
    report = read_json(report_path)
    if not bool(report.get("paired_eval", False)) or str(report.get("stage", "")) != "stage5":
        raise ValueError(f"invalid resumed Stage5 report: {report_path}")


def _validate_runtime_preflights(raw: dict[str, Any], groups: dict[str, dict[str, Any]]) -> None:
    stage5 = dict(raw.get("stage5", {}) or {})
    single = stage5.get("accvp_observation_preflight_report")
    per_group = dict(stage5.get("accvp_observation_preflight_reports", {}) or {})
    per_group_hashes = dict(
        stage5.get("accvp_observation_preflight_report_sha256s", {}) or {}
    )
    if single and per_group:
        raise ValueError("Stage5 config mixes legacy and per-group runtime preflights")
    if per_group:
        if set(per_group_hashes) != set(per_group):
            raise ValueError("per-group runtime preflight reports/hashes do not match")
        for group_name, source in sorted(per_group.items()):
            group = groups.get(str(group_name))
            if group is None:
                raise ValueError(f"runtime preflight refers to unknown Stage5 group: {group_name}")
            path = _resolve(source)
            if file_sha256(path) != str(per_group_hashes[group_name]):
                raise ValueError(
                    f"runtime preflight report hash mismatch for Stage5 group {group_name!r}"
                )
            runtime = read_json(path)
            if not bool(runtime.get("gate", {}).get("pass", False)):
                raise ValueError(f"candidate policy runtime gate failed: {path}")
            model_path = _resolve(group.get("model_path", ""))
            if str(runtime.get("policy_model_sha256", "")) != file_sha256(model_path):
                raise ValueError(
                    f"runtime preflight checkpoint hash mismatch for Stage5 group {group_name!r}"
                )
        return
    if not single:
        raise ValueError("formal Stage5 config is missing its policy runtime preflight")
    preflight = _resolve(single)
    runtime = read_json(preflight)
    if not bool(runtime.get("gate", {}).get("pass", False)):
        raise ValueError(f"candidate policy runtime gate failed: {preflight}")


def _validate_existing_aggregate(request_path: Path, output: Path) -> None:
    report = read_json(output)
    if str(report.get("artifact_kind", "")) != REPORT_ARTIFACT_KIND:
        raise ValueError(f"invalid resumed Stage5 aggregate: {output}")
    if not bool(dict(report.get("gate", {}) or {}).get("pass", False)):
        raise ValueError(f"resumed Stage5 aggregate gate is closed: {output}")
    lineage = dict(report.get("lineage", {}) or {})
    source = dict(lineage.get("source_manifest", {}) or {})
    if str(source.get("sha256", "")) != file_sha256(request_path):
        raise ValueError("resumed Stage5 aggregate does not bind the frozen request")
    fingerprint = str(report.get("report_fingerprint", ""))
    content = dict(report)
    content.pop("report_fingerprint", None)
    if not fingerprint or stable_hash(content) != fingerprint:
        raise ValueError(f"resumed Stage5 aggregate fingerprint mismatch: {output}")


def run(
    *,
    generated_dir: str | Path,
    aggregate_output: str | Path | None = None,
    resume: bool = False,
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
        recorded = {str(group.get("name")): dict(group) for group in groups}
        left = recorded.get(str(row["left_group"]), {})
        right = recorded.get(str(row["right_group"]), {})
        if file_sha256(_resolve(left.get("model_path", ""))) != str(row["left_checkpoint_sha256"]):
            raise ValueError("left Stage5 checkpoint hash mismatch")
        if file_sha256(_resolve(right.get("model_path", ""))) != str(row["right_checkpoint_sha256"]):
            raise ValueError("right Stage5 checkpoint hash mismatch")
        _validate_runtime_preflights(raw, recorded)
        report_path = _resolve(row["stage5_report"])
        if report_path.exists():
            if not resume:
                raise FileExistsError(f"refusing to overwrite formal Stage5 report: {report_path}")
            _validate_resume_receipt(row, report_path)
        elif _receipt_path(report_path).exists():
            raise FileNotFoundError(
                f"Stage5 execution receipt exists without its report: {_receipt_path(report_path)}"
            )
        prepared.append((row, config_path, raw))
    produced: list[Path] = []
    for row, config_path, _raw in prepared:
        expected = _resolve(row["stage5_report"])
        if expected.is_file():
            produced.append(expected)
            continue
        stage_dir = Path(stage5_paired_eval.run(load_config(config_path))).resolve()
        report_path = stage_dir / "formal_paired_eval_report.json"
        if report_path != expected or not report_path.is_file():
            raise RuntimeError(
                f"Stage5 output disagrees with frozen request: actual={report_path} expected={expected}"
            )
        write_json_new(_receipt_path(report_path), _receipt_payload(row, report_path))
        produced.append(report_path)
    if aggregate_output is not None:
        output = _resolve(aggregate_output)
        if output.exists():
            if not resume:
                raise FileExistsError(f"replicated Stage5 report already exists: {output}")
            _validate_existing_aggregate(request_path, output)
            produced.append(output.resolve())
        else:
            produced.append(stage5_replicated_aggregate.run(request_path, output).resolve())
    return produced


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute frozen paired Stage5 optimizer replicates")
    parser.add_argument("--generated-dir", required=True)
    parser.add_argument("--aggregate-output")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    paths = run(
        generated_dir=args.generated_dir,
        aggregate_output=args.aggregate_output,
        resume=args.resume,
    )
    print(f"stage5_replicate_reports={len(paths)}")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
