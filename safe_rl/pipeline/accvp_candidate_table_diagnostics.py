from __future__ import annotations

import argparse
from pathlib import Path

from safe_rl.accvp.contracts.artifacts import (
    ACCVP_ARTIFACT_GENERATION,
    apply_v2_bundle_paths,
    artifact_filename,
)
from safe_rl.accvp.evaluation.candidate_table import candidate_table_diagnostics
from safe_rl.pipeline.common import write_report
from safe_rl.utils.config import REPO_ROOT, load_config


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else REPO_ROOT / value


def _default_artifact_path(cfg, name: str) -> Path:
    output_root = Path(cfg.run.output_root)
    if not output_root.is_absolute():
        output_root = REPO_ROOT / output_root
    run_id = str(cfg.run.run_id)
    if not run_id:
        raise ValueError("run.run_id is required when --checkpoint/--calibration/--output are omitted")
    return output_root / run_id / "accvp" / name


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose ACCVP candidate-table action contrast quality")
    parser.add_argument("--config", required=True, help="ACCVP shadow/deployable training config")
    parser.add_argument("--dataset", default=None, help="Merged counterfactual dataset directory")
    parser.add_argument("--checkpoint", default=None, help="ACCVP predictor checkpoint")
    parser.add_argument("--calibration", default=None, help="ACCVP calibration bundle")
    parser.add_argument("--splits", nargs="+", default=["operating_point", "test"], help="Splits to diagnose")
    parser.add_argument("--output", default=None, help="Path to write the JSON diagnostic report")
    args = parser.parse_args()

    cfg = load_config(args.config)
    apply_v2_bundle_paths(cfg)
    dataset = _resolve(args.dataset or cfg.accvp.dataset_dir)
    vnext = str(cfg.accvp.get("artifact_generation") or "") == ACCVP_ARTIFACT_GENERATION
    checkpoint = (
        _resolve(args.checkpoint)
        if args.checkpoint
        else _resolve(cfg.accvp.checkpoint)
        if cfg.accvp.get("checkpoint")
        else _default_artifact_path(
            cfg, artifact_filename("predictor") if vnext else "accvp_v1_predictor.pt"
        )
    )
    calibration = (
        _resolve(args.calibration)
        if args.calibration
        else _resolve(cfg.accvp.calibration_bundle)
        if cfg.accvp.get("calibration_bundle")
        else _default_artifact_path(
            cfg, artifact_filename("calibration") if vnext else "accvp_v1_calibration.json"
        )
    )
    output = _resolve(args.output) if args.output else _default_artifact_path(cfg, "accvp_candidate_table_diagnostics.json")
    report = candidate_table_diagnostics(
        config=cfg,
        dataset_dir=dataset,
        splits=[str(split) for split in args.splits],
        checkpoint=checkpoint,
        calibration_path=calibration,
        output=output,
    )
    write_report(output, report)
    primary = report["primary_split"]
    primary_report = report["splits"][primary]
    verdict = primary_report["verdict"]
    print(
        "accvp_candidate_table_diagnostics "
        f"primary_split={primary} "
        f"state={verdict['step1_5_state']} "
        f"viability_pass={verdict['viability_signal_pass']} "
        f"safety_pass={verdict['safety_signal_pass']} "
        f"gate_pass={verdict['gate_availability_pass']} "
        f"report={output}"
    )


if __name__ == "__main__":
    main()
