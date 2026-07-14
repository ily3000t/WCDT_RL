from __future__ import annotations

import argparse
import json
from pathlib import Path

from safe_rl.accvp.evaluation.candidate_table import (
    candidate_records_from_dataset,
    load_calibration,
    load_models_from_checkpoint,
)
from safe_rl.accvp.data.dataset import ACCVPBranchDataset
from safe_rl.accvp.evaluation.risk_secondary import audit_risk_secondary, combine_audit_reports, write_risk_secondary_audit
from safe_rl.accvp.planning.viability_lite import lite_thresholds_from_config
from safe_rl.utils.config import REPO_ROOT, load_config


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else REPO_ROOT / value


def _artifact_dir(cfg) -> Path:
    output_root = Path(cfg.run.output_root)
    if not output_root.is_absolute():
        output_root = REPO_ROOT / output_root
    return output_root / str(cfg.run.run_id) / "accvp"


def _thresholds_from_operating_point(path: Path | None, cfg) -> dict[str, float | str]:
    thresholds = lite_thresholds_from_config(cfg)
    if path is None:
        return thresholds
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    selected = dict(payload.get("selected", payload))
    thresholds.update(
        {
            "min_p_merge_before_taper": float(selected["min_p_merge_before_taper"]),
            "min_improvement_over_raw": float(selected["min_improvement_over_raw"]),
            "max_target_entry_time_s": float(selected["max_target_entry_time_s"]),
            "max_ensemble_disagreement": float(selected["max_ensemble_disagreement"]),
            "max_secondary_risk_score": float(selected.get("max_secondary_risk_score", 1.0)),
            "secondary_safety_profile": str(selected.get("secondary_safety_profile", thresholds["secondary_safety_profile"])),
        }
    )
    return thresholds


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Risk secondary false negatives for ACCVP-lite left actions")
    parser.add_argument("--config", required=True, help="ACCVP lite tuning config")
    parser.add_argument("--dataset", default=None, help="Merged counterfactual dataset directory")
    parser.add_argument("--checkpoint", default=None, help="ACCVP predictor checkpoint")
    parser.add_argument("--calibration", default=None, help="ACCVP calibration bundle")
    parser.add_argument("--operating-point", default=None, help="Lite operating point JSON")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["operating_point"],
        help=(
            "Dataset splits to audit. Only operating_point may select a threshold; "
            "all other splits are diagnostic-only."
        ),
    )
    parser.add_argument("--risk-score-grid", nargs="+", type=float, default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    if "operating_point" not in {str(split) for split in args.splits}:
        raise SystemExit("Risk-secondary tuning requires the operating_point split")
    if "test" in {str(split) for split in args.splits}:
        raise SystemExit(
            "The test split is sealed; evaluate its frozen profile only through "
            "accvp_final_holdout_eval"
        )

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError("ACCVP Risk secondary audit requires torch.") from exc

    cfg = load_config(args.config)
    output_dir = _resolve(args.output_dir) if args.output_dir else _artifact_dir(cfg)
    dataset = _resolve(args.dataset or cfg.accvp.dataset_dir)
    checkpoint = _resolve(args.checkpoint or cfg.accvp.checkpoint)
    calibration_path = _resolve(args.calibration or cfg.accvp.calibration_bundle)
    operating_point = _resolve(args.operating_point or cfg.accvp.operating_point) if cfg.accvp.get("operating_point") or args.operating_point else None
    thresholds = _thresholds_from_operating_point(operating_point, cfg)
    lite_cfg = cfg.accvp.get("viability_lite", {}) or {}
    grid = args.risk_score_grid or [float(value) for value in lite_cfg.get("audit_secondary_risk_score_grid", lite_cfg.get("max_secondary_risk_score_grid", [0.05, 0.10, 0.20, 0.40, 0.60, 1.0]))]
    models = load_models_from_checkpoint(cfg, checkpoint, torch)
    calibration = load_calibration(calibration_path)
    split_reports = {}
    for split in args.splits:
        dataset_split = ACCVPBranchDataset(dataset, split)
        records = candidate_records_from_dataset(models, dataset_split, calibration, torch)
        split_reports[split] = audit_risk_secondary(
            records,
            thresholds,
            split=split,
            risk_score_grid=grid,
        )
    report = combine_audit_reports(split_reports)
    report.update(
        {
            "dataset_dir": str(dataset.resolve()),
            "checkpoint": str(checkpoint.resolve()),
            "calibration": str(calibration_path.resolve()),
            "operating_point": None if operating_point is None else str(operating_point.resolve()),
            "base_thresholds": thresholds,
            "risk_score_grid": grid,
        }
    )
    paths = write_risk_secondary_audit(output_dir=output_dir, report=report)
    print(
        "accvp_risk_secondary_audit "
        f"audit_state={report['audit_state']} "
        f"profile={report.get('selected_audited_profile')} "
        f"report={paths['report']}"
    )


if __name__ == "__main__":
    main()
