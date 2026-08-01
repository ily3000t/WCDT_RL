from __future__ import annotations

import argparse
from pathlib import Path

from safe_rl.accvp.evaluation.formal import write_formal_report
from safe_rl.pipeline.common import load_stage_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate selector-v3 formal data before ACCVP training"
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cfg = load_stage_config(args)
    criteria = dict(
        cfg.accvp.counterfactual.get("formal_acceptance", {}) or {}
    )
    minimum_root_count = int(
        criteria.pop(
            "minimum_root_count",
            cfg.accvp.counterfactual.get("root_budget", 0),
        )
    )
    if minimum_root_count <= 0:
        raise ValueError(
            "formal validation requires a positive minimum_root_count"
        )
    oracle_cfg = dict(cfg.accvp.get("oracle", {}) or {})
    report = write_formal_report(
        cfg,
        Path(args.dataset),
        Path(args.output),
        minimum_root_count=minimum_root_count,
        excluded_episode_seeds=[
            int(value)
            for value in list(oracle_cfg.get("required_seeds", []) or [])
        ],
        excluded_cohort_roles=[
            str(oracle_cfg.get("cohort_role", ""))
        ],
        **criteria,
    )
    print(report["formal_state"])


if __name__ == "__main__":
    main()
