from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from safe_rl.accvp.contracts.schema import file_sha256, stable_hash, write_json_atomic
from safe_rl.pipeline import stage2_train_prediction_risk
from safe_rl.pipeline.train_comparative_suite import (
    _require_schema9,
    _validate_existing_wcdt_v1_checkpoint,
)
from safe_rl.tools.audit_wcdt_upstream import run as audit_upstream
from safe_rl.utils.config import REPO_ROOT, load_config


REPORT_KIND = "main_method_wcdt_v1_predictor_report_v1"


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (REPO_ROOT / value).resolve()


def _paths(cfg: Any) -> tuple[Path, Path, Path, Path]:
    output_root = _resolve(str(cfg.run.output_root))
    run_root = output_root / str(cfg.run.run_id)
    stage1 = _resolve(str(cfg.stage2.input_stage1))
    risk = _resolve(str(cfg.stage2.risk_checkpoint_reference))
    checkpoint = run_root / "stage2" / "wcdt_predictor.pt"
    return run_root, stage1, risk, checkpoint


def run(
    *,
    config_path: str | Path,
    upstream_root: str | Path,
    upstream_commit: str,
    allowed_differences: set[str] | None = None,
    prepare_only: bool = False,
) -> Path:
    config_source = _resolve(config_path)
    cfg = load_config(config_source)
    run_root, stage1, risk, checkpoint = _paths(cfg)
    if not stage1.is_dir() or not (stage1 / "manifest.json").is_file():
        raise FileNotFoundError(stage1)
    if not risk.is_file():
        raise FileNotFoundError(risk)
    _require_schema9(stage1)
    if bool(cfg.stage2.get("train_risk_module", True)):
        raise ValueError("main-method WcDT-v1 preparation must not retrain Risk")
    if not bool(cfg.prediction.get("wcdt_v1_train_enabled", False)):
        raise ValueError("main-method WcDT-v1 predictor config does not enable v1")
    if bool(cfg.prediction.get("wcdt_v2_train_enabled", False)) or bool(
        cfg.prediction.get("wcdt_v3_train_enabled", False)
    ):
        raise ValueError("main-method WcDT-v1 predictor run must train only v1")

    manifest_dir = run_root / "manifests"
    source_diff = manifest_dir / "source_diff_manifest.json"
    audit = audit_upstream(
        upstream_root=_resolve(upstream_root),
        output=source_diff,
        upstream_commit=str(upstream_commit),
        allowed_differences=allowed_differences,
    )
    if str(audit.get("source_fidelity", "")) != "verified":
        raise ValueError("formal WcDT-v1 adapted baseline requires verified upstream fidelity")

    if prepare_only:
        status = "source_verified_predictor_not_trained"
        checkpoint_summary: dict[str, Any] | None = None
    elif checkpoint.is_file():
        checkpoint_summary = _validate_existing_wcdt_v1_checkpoint(checkpoint, cfg)
        status = "reused_existing_verified_predictor"
    else:
        stage2_dir = run_root / "stage2"
        if stage2_dir.exists() and any(stage2_dir.iterdir()):
            raise RuntimeError(
                "WcDT-v1 output contains partial Stage2 artifacts without a valid "
                f"checkpoint; use a new run_id: {stage2_dir}"
            )
        stage2_train_prediction_risk.run(cfg)
        checkpoint_summary = _validate_existing_wcdt_v1_checkpoint(checkpoint, cfg)
        status = "trained"

    report: dict[str, Any] = {
        "artifact_kind": REPORT_KIND,
        "schema_version": 1,
        "status": status,
        "config": str(config_source),
        "config_sha256": file_sha256(config_source),
        "stage1_manifest": str((stage1 / "manifest.json").resolve()),
        "stage1_manifest_sha256": file_sha256(stage1 / "manifest.json"),
        "risk_checkpoint": str(risk),
        "risk_checkpoint_sha256": file_sha256(risk),
        "source_diff_manifest": str(source_diff.resolve()),
        "source_diff_manifest_sha256": file_sha256(source_diff),
        "upstream_commit": str(upstream_commit),
        "source_fidelity": "verified",
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_available": checkpoint.is_file(),
        "checkpoint_sha256": file_sha256(checkpoint) if checkpoint.is_file() else "",
        "checkpoint_summary": checkpoint_summary,
    }
    report["report_fingerprint"] = stable_hash(report)
    output = manifest_dir / "wcdt_v1_predictor_report.json"
    if output.exists():
        from safe_rl.accvp.contracts.schema import read_json

        if read_json(output) != report:
            raise FileExistsError(f"refusing to replace a different WcDT-v1 report: {output}")
        return output.resolve()
    write_json_atomic(output, report)
    return output.resolve()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit the pinned WcDT-v1 source and train only its SUMO adapter predictor"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--upstream-root", required=True)
    parser.add_argument(
        "--upstream-commit",
        default="6baa2330fc3f620863d358b5d7f36323b4bfccae",
    )
    parser.add_argument("--allowed-difference", action="append", default=[])
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    output = run(
        config_path=args.config,
        upstream_root=args.upstream_root,
        upstream_commit=str(args.upstream_commit),
        allowed_differences=set(args.allowed_difference) or None,
        prepare_only=bool(args.prepare_only),
    )
    print(f"main_method_wcdt_v1_predictor_report={output}")


if __name__ == "__main__":
    main()
