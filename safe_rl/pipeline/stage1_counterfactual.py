from __future__ import annotations

from safe_rl.stage1_counterfactual.collector import collect
from safe_rl.utils.config import load_config


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="ACCVP isolated counterfactual collection")
    parser.add_argument("--config", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--root-source", default="mixed", choices=["mixed", "ppo", "merge_timing", "rule", "deadline_hard"])
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.run_id:
        cfg.run["run_id"] = args.run_id
    collect(cfg, root_source=args.root_source)


if __name__ == "__main__":
    main()
