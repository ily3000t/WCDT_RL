from __future__ import annotations

import argparse
from pathlib import Path

from safe_rl.accvp.targeted_benchmark import (
    build_replacement_case_table,
    build_targeted_benchmark_summary,
    write_targeted_benchmark_outputs,
)
from safe_rl.utils.config import REPO_ROOT


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else REPO_ROOT / value


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ACV-Shield-lite v3 targeted benchmark evidence pack")
    parser.add_argument("--stage5-report", required=True)
    parser.add_argument("--replay-dir", nargs="+", required=True)
    parser.add_argument("--group-contains", default="accvp_lite_v3")
    parser.add_argument("--baseline-group", required=True)
    parser.add_argument("--accvp-group", required=True)
    parser.add_argument("--online-audit", default=None)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    replay_dirs = [_resolve(path) for path in args.replay_dir]
    cases = build_replacement_case_table(replay_dirs, group_contains=args.group_contains)
    summary = build_targeted_benchmark_summary(
        stage5_report=_resolve(args.stage5_report),
        cases=cases,
        baseline_group=args.baseline_group,
        accvp_group=args.accvp_group,
        online_audit=_resolve(args.online_audit) if args.online_audit else None,
    )
    paths = write_targeted_benchmark_outputs(output_dir=_resolve(args.output_dir), cases=cases, summary=summary)
    print(
        "accvp_targeted_benchmark_report "
        f"cases={len(cases)} "
        f"safety_gate_pass={summary['safety_gate_pass']} "
        f"task_gate_pass={summary['task_gate_pass']} "
        f"summary={paths['summary']}"
    )


if __name__ == "__main__":
    main()
