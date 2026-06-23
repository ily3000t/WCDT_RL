from __future__ import annotations

from pathlib import Path

from safe_rl.accvp.train import train_accvp
from safe_rl.pipeline.common import load_stage_config, parse_config_arg


def main() -> None:
    args = parse_config_arg("Train ACCVP-v1 conditional predictor")
    cfg = load_stage_config(args)
    dataset_dir = cfg.accvp.get("dataset_dir")
    if not dataset_dir:
        raise ValueError("accvp.dataset_dir must point at a complete counterfactual dataset")
    train_accvp(cfg, Path(str(dataset_dir)))


if __name__ == "__main__":
    main()
