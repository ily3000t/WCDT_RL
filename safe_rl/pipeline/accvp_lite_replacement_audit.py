from __future__ import annotations

import argparse
import json
from pathlib import Path

from safe_rl.accvp.candidate_table_diagnostics import (
    candidate_records_from_dataset,
    load_calibration,
    load_models_from_checkpoint,
)
from safe_rl.accvp.dataset import ACCVPBranchDataset
from safe_rl.accvp.viability_lite import lite_thresholds_from_config
from safe_rl.accvp.viability_lite_audit import audit_lite_replacements, write_lite_replacement_audit
from safe_rl.utils.config import REPO_ROOT, load_config


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else REPO_ROOT / value


def _artifact_dir(cfg) -> Path:
    output_root = Path(cfg.run.output_root)
    if not output_root.is_absolute():
        output_root = REPO_ROOT / output_root
    return output_root / str(cfg.run.run_id) / "accvp"


def _thresholds_from_operating_point(path: Path | None, cfg) -> dict[str, float]:
    if path is None:
        return lite_thresholds_from_config(cfg)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    selected = dict(payload.get("selected", payload))
    return {
        "min_p_merge_before_taper": float(selected["min_p_merge_before_taper"]),
        "min_improvement_over_raw": float(selected["min_improvement_over_raw"]),
        "max_target_entry_time_s": float(selected["max_target_entry_time_s"]),
        "max_ensemble_disagreement": float(selected["max_ensemble_disagreement"]),
        "max_secondary_risk_score": float(selected.get("max_secondary_risk_score", 1.0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit ACV-Shield-lite replacement safety and task viability")
    parser.add_argument("--config", required=True, help="ACCVP lite tuning or shadow config")
    parser.add_argument("--dataset", default=None, help="Merged counterfactual dataset directory")
    parser.add_argument("--checkpoint", default=None, help="ACCVP predictor checkpoint")
    parser.add_argument("--calibration", default=None, help="ACCVP calibration bundle")
    parser.add_argument("--operating-point", default=None, help="Lite operating point JSON")
    parser.add_argument("--splits", nargs="+", default=["operating_point", "test"], help="Dataset splits to audit")
    parser.add_argument("--output-dir", default=None, help="Directory to write audit artifacts")
    parser.add_argument("--max-targeted-seeds", type=int, default=20)
    args = parser.parse_args()

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError("ACCVP-lite replacement audit requires torch.") from exc

    cfg = load_config(args.config)
    output_dir = _resolve(args.output_dir) if args.output_dir else _artifact_dir(cfg)
    dataset = _resolve(args.dataset or cfg.accvp.dataset_dir)
    checkpoint = (
        _resolve(args.checkpoint)
        if args.checkpoint
        else _resolve(cfg.accvp.checkpoint)
        if cfg.accvp.get("checkpoint")
        else output_dir / "accvp_v1_predictor.pt"
    )
    calibration_path = (
        _resolve(args.calibration)
        if args.calibration
        else _resolve(cfg.accvp.calibration_bundle)
        if cfg.accvp.get("calibration_bundle")
        else output_dir / "accvp_v1_calibration.json"
    )
    operating_point = (
        _resolve(args.operating_point)
        if args.operating_point
        else _resolve(cfg.accvp.operating_point)
        if cfg.accvp.get("operating_point")
        else None
    )
    thresholds = _thresholds_from_operating_point(operating_point, cfg)
    models = load_models_from_checkpoint(cfg, checkpoint, torch)
    calibration = load_calibration(calibration_path)
    split_reports = {}
    combined = {
        "replacement_safety_event_roots": [],
        "risk_failed_but_success_roots": [],
        "unnecessary_replacement_roots": [],
    }
    targeted_by_split = {}
    for split in args.splits:
        dataset_split = ACCVPBranchDataset(dataset, split)
        records = candidate_records_from_dataset(models, dataset_split, calibration, torch)
        split_report = audit_lite_replacements(
            records,
            thresholds,
            split=split,
            max_targeted_seeds=int(args.max_targeted_seeds),
        )
        split_reports[split] = split_report
        targeted_by_split[split] = list(split_report["targeted_seeds"])
        for key in combined:
            combined[key].extend({**row, "split": split} for row in split_report[key])
    targeted_source = "test" if "test" in targeted_by_split else str(args.splits[0])
    report = {
        "artifact_kind": "accvp_lite_replacement_audit_v1",
        "controller": "acv_shield_lite",
        "deployable_claim": "task_viability_only",
        "thresholds": thresholds,
        "dataset_dir": str(dataset.resolve()),
        "checkpoint": str(checkpoint.resolve()),
        "calibration": str(calibration_path.resolve()),
        "operating_point": None if operating_point is None else str(operating_point.resolve()),
        "splits": split_reports,
        "combined": combined,
        "targeted_seed_source_split": targeted_source,
        "targeted_seeds_by_split": targeted_by_split,
        "targeted_seeds": targeted_by_split.get(targeted_source, [])[: int(args.max_targeted_seeds)],
    }
    paths = write_lite_replacement_audit(output_dir=output_dir, report=report)
    print(
        "accvp_lite_replacement_audit "
        f"targeted_seed_source={targeted_source} "
        f"targeted_seed_count={len(report['targeted_seeds'])} "
        f"report={paths['report']}"
    )


if __name__ == "__main__":
    main()
