from __future__ import annotations

from safe_rl.stage1_counterfactual.collector import collect
from safe_rl.utils.config import load_config


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="ACCVP isolated counterfactual collection")
    parser.add_argument("--config", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--root-policy", choices=["mixed", "ppo", "merge_timing", "rule"], default=None)
    parser.add_argument("--root-filter", choices=["all", "deadline"], default=None)
    parser.add_argument("--episode-seeds", nargs="*", type=int, default=None)
    parser.add_argument("--root-source", default=None, choices=["mixed", "ppo", "merge_timing", "rule", "deadline_hard"], help="Deprecated compatibility alias for --root-policy.")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.run_id:
        cfg.run["run_id"] = args.run_id
    collect(
        cfg,
        root_policy=args.root_policy,
        root_filter=args.root_filter,
        episode_seeds=args.episode_seeds,
        root_source=args.root_source,
    )


if __name__ == "__main__":
    main()
