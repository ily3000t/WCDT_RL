from __future__ import annotations

import argparse
from pathlib import Path

from safe_rl.accvp.shards import merge_counterfactual_shards
from safe_rl.pipeline.common import load_stage_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge immutable ACCVP counterfactual shards")
    parser.add_argument("--config", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--shard", action="append", required=True, help="Shard directory; repeat for each input")
    parser.add_argument("--output", required=True, help="New formal dataset directory")
    args = parser.parse_args()
    cfg = load_stage_config(args)
    output = merge_counterfactual_shards(
        args.shard,
        args.output,
        require_frozen_risk_model=bool(cfg.accvp.counterfactual.get("require_frozen_risk_model", True)),
        expected_collection_phase=(
            str(cfg.accvp.counterfactual.get("collection_phase"))
            if str(cfg.accvp.counterfactual.get("collection_phase", "ad_hoc")) in {"pilot", "formal"}
            else None
        ),
    )
    print(output)


if __name__ == "__main__":
    main()
