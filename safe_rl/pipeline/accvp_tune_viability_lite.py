from __future__ import annotations

import argparse
from pathlib import Path

from safe_rl.accvp.candidate_table_diagnostics import (
    candidate_records_from_dataset,
    load_calibration,
    load_models_from_checkpoint,
)
from safe_rl.accvp.dataset import ACCVPBranchDataset
from safe_rl.accvp.viability_lite import (
    evaluate_lite_thresholds,
    tune_viability_lite_operating_point,
    write_lite_artifacts,
)
from safe_rl.pipeline.common import write_report
from safe_rl.utils.config import REPO_ROOT, load_config


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else REPO_ROOT / value


def _artifact_dir(cfg) -> Path:
    output_root = Path(cfg.run.output_root)
    if not output_root.is_absolute():
        output_root = REPO_ROOT / output_root
    return output_root / str(cfg.run.run_id) / "accvp"


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune ACV-Shield-lite task-viability operating point")
    parser.add_argument("--config", required=True, help="ACCVP shadow training config")
    parser.add_argument("--dataset", default=None, help="Merged counterfactual dataset directory")
    parser.add_argument("--checkpoint", default=None, help="ACCVP predictor checkpoint")
    parser.add_argument("--calibration", default=None, help="ACCVP calibration bundle")
    parser.add_argument("--output-dir", default=None, help="Directory to write lite artifacts")
    args = parser.parse_args()

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError("ACCVP-lite tuning requires torch.") from exc

    cfg = load_config(args.config)
    output_dir = _resolve(args.output_dir) if args.output_dir else _artifact_dir(cfg)
    dataset = _resolve(args.dataset or cfg.accvp.dataset_dir)
    checkpoint = _resolve(args.checkpoint) if args.checkpoint else output_dir / "accvp_v1_predictor.pt"
    calibration_path = _resolve(args.calibration) if args.calibration else output_dir / "accvp_v1_calibration.json"
    models = load_models_from_checkpoint(cfg, checkpoint, torch)
    calibration = load_calibration(calibration_path)
    operating_set = ACCVPBranchDataset(dataset, "operating_point")
    test_set = ACCVPBranchDataset(dataset, "test")
    operating_records = candidate_records_from_dataset(models, operating_set, calibration, torch)
    test_records = candidate_records_from_dataset(models, test_set, calibration, torch)
    operating_point = tune_viability_lite_operating_point(operating_records, cfg, split="operating_point")
    final_test = evaluate_lite_thresholds(test_records, operating_point["selected"], split="test")
    artifacts = write_lite_artifacts(
        output_dir=output_dir,
        config=cfg,
        dataset_dir=dataset,
        checkpoint=checkpoint,
        calibration=calibration_path,
        operating_point=operating_point,
        final_test=final_test,
    )
    summary = {
        "artifact_kind": "accvp_viability_lite_tuning_summary",
        "controller": "acv_shield_lite",
        "deployable_claim": "task_viability_only",
        "operating_point": operating_point,
        "final_test": final_test,
        "artifacts": {key: str(value.resolve()) for key, value in artifacts.items()},
    }
    write_report(output_dir / "accvp_v1_lite_tuning_summary.json", summary)
    print(
        "accvp_viability_lite_tuning "
        f"repair_capture={operating_point['selected_metrics']['repairable_root_capture_rate']:.6f} "
        f"replacement_rate={operating_point['selected_metrics']['replacement_rate']:.6f} "
        f"test_repair_capture={final_test['repairable_root_capture_rate']:.6f} "
        f"manifest={artifacts['artifact_manifest']}"
    )


if __name__ == "__main__":
    main()
