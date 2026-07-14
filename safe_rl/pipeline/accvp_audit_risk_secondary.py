from __future__ import annotations

import argparse
import json
from pathlib import Path

from safe_rl.accvp.training.availability import audit_risk_secondary_false_negatives


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit ACCVP merge-left Risk secondary false negatives")
    parser.add_argument("--dataset", required=True, help="Merged counterfactual dataset directory")
    parser.add_argument("--split", default="operating_point", help="Dataset split to audit, or 'all'")
    parser.add_argument("--output", required=True, help="Path to write the JSON audit report")
    args = parser.parse_args()

    report = audit_risk_secondary_false_negatives(Path(args.dataset), split=str(args.split))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    print(
        "risk_secondary_audit "
        f"split={report['split']} decisions={report['decision_count']} "
        f"physical_ceiling={report['physical_oracle_ceiling_ignore_risk']:.6f} "
        f"risk_gated_ceiling={report['risk_gated_physical_ceiling']:.6f} "
        f"false_negative_roots={report['risk_false_negative_root_count']} "
        f"false_negative_actions={report['risk_false_negative_action_count']} "
        f"report={output}"
    )


if __name__ == "__main__":
    main()
