from __future__ import annotations

import argparse
from pathlib import Path

from safe_rl.accvp.online_trigger_audit import audit_online_triggers, write_online_trigger_audit
from safe_rl.utils.config import REPO_ROOT, load_config


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else REPO_ROOT / value


def _artifact_dir(cfg) -> Path:
    output_root = Path(cfg.run.output_root)
    if not output_root.is_absolute():
        output_root = REPO_ROOT / output_root
    return output_root / str(cfg.run.run_id) / "accvp"


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit ACCVP online shadow/active trigger alignment from Stage5 replay")
    parser.add_argument("--config", default=None, help="Optional config used only to choose default output dir")
    parser.add_argument("--replay-dir", nargs="+", required=True, help="Stage5 replay directory or replay JSON files")
    parser.add_argument("--group-contains", default=None, help="Only audit replay files whose group_name contains this text")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--artifact-manifest", default=None, help="Optional ACCVP artifact manifest to hash into targeted_benchmark_seeds.json")
    parser.add_argument("--risk-checkpoint", default=None, help="Optional Risk Module checkpoint to hash into targeted_benchmark_seeds.json")
    args = parser.parse_args()

    cfg = load_config(args.config) if args.config else None
    output_dir = _resolve(args.output_dir) if args.output_dir else (_artifact_dir(cfg) if cfg is not None else REPO_ROOT / "safe_rl_output" / "runs" / "accvp_online_trigger_audit")
    report = audit_online_triggers(
        [_resolve(path) for path in args.replay_dir],
        group_contains=args.group_contains,
    )
    replay_paths = [_resolve(path) for path in args.replay_dir]
    paths = write_online_trigger_audit(
        output_dir=output_dir,
        report=report,
        source_replay_dirs=replay_paths,
        artifact_manifest=_resolve(args.artifact_manifest) if args.artifact_manifest else None,
        risk_checkpoint=_resolve(args.risk_checkpoint) if args.risk_checkpoint else None,
    )
    print(
        "accvp_online_trigger_audit "
        f"would_trigger_seed_count={report['would_trigger_seed_count']} "
        f"actual_action_change_count={report['actual_action_change_count']} "
        f"report={paths['report']}"
    )


if __name__ == "__main__":
    main()
