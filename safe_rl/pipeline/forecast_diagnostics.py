from __future__ import annotations

import argparse

from safe_rl.analysis.forecast_diagnostics import run_forecast_diagnostics
from safe_rl.pipeline.common import load_stage_config
from safe_rl.utils.progress import stage_log


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze CV vs WcDT forecast features and WcDT prediction errors.")
    parser.add_argument("--config", default=None, help="Optional YAML config overlay.")
    parser.add_argument("--run-id", required=True, help="Existing run id to analyze.")
    parser.add_argument("--max-samples", type=int, default=512, help="Maximum Stage1 trajectory windows to sample.")
    parser.add_argument("--batch-size", type=int, default=32, help="WcDT diagnostic batch size.")
    parser.add_argument("--low-seed-count", type=int, default=5, help="Number of low-min-distance CV seeds to report.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_stage_config(args)
    output = run_forecast_diagnostics(
        cfg,
        max_samples=int(args.max_samples),
        batch_size=int(args.batch_size),
        low_seed_count=int(args.low_seed_count),
    )
    stage_log("diagnostics", f"report={output}")


if __name__ == "__main__":
    main()
