from __future__ import annotations

import argparse
import json
from pathlib import Path

from safe_rl.accvp.availability import diagnose_oracle_availability


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose ACCVP merge-intent availability ceilings")
    parser.add_argument("--dataset", required=True, help="Merged counterfactual dataset directory")
    parser.add_argument("--split", default="operating_point", help="Dataset split to diagnose, or 'all'")
    parser.add_argument("--output", required=True, help="Path to write the JSON diagnostic report")
    args = parser.parse_args()

    report = diagnose_oracle_availability(Path(args.dataset), split=str(args.split))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    print(
        "availability_diagnostic "
        f"split={report['split']} decisions={report['decision_count']} "
        f"oracle_ceiling={report['oracle_merge_intent_ceiling_availability']:.6f} "
        f"risk_ceiling={report['risk_secondary_pass_ceiling_availability']:.6f} "
        f"report={output}"
    )


if __name__ == "__main__":
    main()
