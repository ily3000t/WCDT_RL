from __future__ import annotations

import argparse
from pathlib import Path

from safe_rl.accvp.evaluation.oracle import write_oracle_report


def main() -> None:
    parser = argparse.ArgumentParser(description="ACCVP counterfactual oracle Go/No-Go report")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--root-policy", default=None)
    parser.add_argument("--cohort-role", default="oracle_regression")
    parser.add_argument("--min-deadline-roots-per-seed", type=int, default=1)
    args = parser.parse_args()
    dataset = Path(args.dataset)
    output = Path(args.output) if args.output else dataset / "manifests" / "oracle_smoke_report.json"
    report = write_oracle_report(
        dataset,
        output,
        args.seeds,
        min_deadline_roots_per_seed=args.min_deadline_roots_per_seed,
        root_policy=args.root_policy,
        cohort_role=args.cohort_role,
        oracle_only=True,
        exclude_from_model_splits=True,
    )
    print(f"oracle_state={report['oracle_state']} go_for_training={report['go_for_training']} report={output}")


if __name__ == "__main__":
    main()
