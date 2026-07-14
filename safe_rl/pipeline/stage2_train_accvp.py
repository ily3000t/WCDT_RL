from __future__ import annotations

import argparse
from pathlib import Path

from safe_rl.accvp.training.trainer import train_accvp
from safe_rl.pipeline.common import load_stage_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ACCVP-v1 conditional predictor")
    parser.add_argument("--config", default=None, help="Optional YAML config overlay.")
    parser.add_argument("--run-id", default=None, help="Existing or new run id.")
    parser.add_argument(
        "--mode",
        choices=["shadow", "deployable"],
        default="deployable",
        help="shadow writes a non-deployable shadow artifact; deployable requires operating-point tuning.",
    )
    args = parser.parse_args()
    cfg = load_stage_config(args)
    dataset_dir = cfg.accvp.get("dataset_dir")
    if not dataset_dir:
        raise ValueError("accvp.dataset_dir must point at a complete counterfactual dataset")
    train_accvp(cfg, Path(str(dataset_dir)), mode=str(args.mode))


if __name__ == "__main__":
    main()
