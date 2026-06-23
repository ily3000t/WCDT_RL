from __future__ import annotations

import argparse
from pathlib import Path

from safe_rl.accvp.oracle import write_oracle_report


def main() -> None:
    parser = argparse.ArgumentParser(description="ACCVP counterfactual oracle Go/No-Go report")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--seeds", nargs="*", type=int, default=[2, 5])
    args = parser.parse_args()
    dataset = Path(args.dataset)
    output = Path(args.output) if args.output else dataset / "manifests" / "oracle_smoke_report.json"
    report = write_oracle_report(dataset, output, args.seeds)
    print(f"go_for_training={report['go_for_training']} report={output}")


if __name__ == "__main__":
    main()
