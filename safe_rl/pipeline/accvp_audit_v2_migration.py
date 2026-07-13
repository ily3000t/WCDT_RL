from __future__ import annotations

import argparse
import json
from pathlib import Path

from safe_rl.accvp.migration import audit_legacy_dataset_migration


def main() -> None:
    parser = argparse.ArgumentParser(description="Dry-run schema-v3 feasibility audit for a legacy ACCVP dataset")
    parser.add_argument("--dataset", required=True, help="Legacy merged dataset or shard directory")
    parser.add_argument("--output", required=True, help="JSON report path outside the legacy dataset")
    args = parser.parse_args()

    dataset = Path(args.dataset).resolve()
    output = Path(args.output).resolve()
    try:
        output.relative_to(dataset)
    except ValueError:
        pass
    else:
        raise ValueError("migration audit output must be outside the immutable legacy dataset")
    report = audit_legacy_dataset_migration(dataset)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    print(
        "accvp_migration_audit "
        f"roots={report['root_count']} classes={report['classification_counts']} "
        f"derivation_allowed={report['schema3_derivation_allowed']} report={output}"
    )


if __name__ == "__main__":
    main()

