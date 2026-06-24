from __future__ import annotations

from safe_rl.pipeline.common import load_stage_config, parse_config_arg
from safe_rl.stage1_counterfactual.collector import collect


def main() -> None:
    args = parse_config_arg("Collect configured immutable ACCVP counterfactual shards")
    cfg = load_stage_config(args)
    jobs = list(cfg.accvp.counterfactual.get("collection_jobs", []))
    if not jobs:
        raise ValueError("accvp.counterfactual.collection_jobs must not be empty")
    for job in jobs:
        name = str(job["name"])
        collect(
            cfg,
            root_policy=str(job["root_policy"]),
            root_filter=str(job["root_filter"]),
            episode_seeds=job.get("episode_seeds"),
            episodes=job.get("episodes"),
            collection_id=name,
        )


if __name__ == "__main__":
    main()
