from __future__ import annotations

import argparse
from pathlib import Path

from safe_rl.accvp.evaluation.pilot import write_pilot_report
from safe_rl.pipeline.common import load_stage_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate ACCVP-240 pilot collection before formal collection")
    parser.add_argument("--config", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--oracle-report", required=True)
    args = parser.parse_args()
    cfg = load_stage_config(args)
    criteria = dict(cfg.accvp.counterfactual.get("pilot_acceptance", {}))
    expected = dict(criteria.pop("expected_root_counts", {}))
    if not expected:
        raise ValueError("accvp.counterfactual.pilot_acceptance.expected_root_counts must not be empty")
    oracle_cfg = dict(cfg.accvp.get("oracle", {}) or {})
    required_oracle_seeds = oracle_cfg.get("required_seeds")
    if not required_oracle_seeds:
        raise ValueError("pilot validation requires accvp.oracle.required_seeds")
    oracle_cohort_role = str(oracle_cfg.get("cohort_role", ""))
    if not oracle_cohort_role:
        raise ValueError("pilot validation requires accvp.oracle.cohort_role")
    if not bool(oracle_cfg.get("exclude_from_model_splits", False)):
        raise ValueError("pilot validation requires oracle exclusion from model splits")
    report = write_pilot_report(
        Path(args.dataset),
        Path(args.output),
        expected_root_counts=expected,
        oracle_report_path=Path(args.oracle_report),
        oracle_required_seeds=required_oracle_seeds,
        oracle_cohort_role=oracle_cohort_role,
        oracle_exclude_from_model_splits=True,
        **criteria,
    )
    print(report["pilot_state"])


if __name__ == "__main__":
    main()
