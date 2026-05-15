from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from safe_rl.pipeline.common import make_env
from safe_rl.risk.risk_aggregator import aggregate_episode_reports
from safe_rl.rl.ppo import load_ppo
from safe_rl.utils.progress import TensorboardLogger, progress_iter, stage_log
from safe_rl.utils.replay import write_replay_file


def evaluate_ppo(
    cfg: Any,
    model_path: str | Path,
    seeds: list[int],
    shield_enabled: bool,
    risk_checkpoint: str | None = None,
    replay_dir: str | Path | None = None,
    group_name: str | None = None,
    tensorboard: TensorboardLogger | None = None,
    tensorboard_step_offset: int = 0,
) -> dict:
    model = load_ppo(model_path)
    reports: list[dict] = []
    rewards: list[float] = []
    for episode_idx, seed in enumerate(progress_iter(seeds, desc=f"Eval {group_name or 'ppo'} seeds")):
        env = make_env(cfg, seed=seed, shield_enabled=shield_enabled, risk_checkpoint=risk_checkpoint)
        total_reward = 0.0
        actions: list[int] = []
        try:
            obs, _info = env.reset(seed=seed)
            terminated = truncated = False
            while not (terminated or truncated):
                action, _state = model.predict(obs, deterministic=True)
                actions.append(int(action))
                obs, reward, terminated, truncated, _info = env.step(int(action))
                total_reward += float(reward)
            report = env.episode_report()
            report["episode_reward"] = total_reward
            report["merge_success"] = _info.get("done_reason") == "merge_success"
            reports.append(report)
            rewards.append(total_reward)
            if tensorboard is not None:
                step = tensorboard_step_offset + episode_idx
                prefix = f"stage5/{group_name or 'ppo'}"
                tensorboard.scalar(f"{prefix}/episode_reward", total_reward, step)
                tensorboard.scalar(f"{prefix}/collision", float(report.get("collision", False)), step)
                tensorboard.scalar(f"{prefix}/near_miss", float(report.get("near_miss", False)), step)
                tensorboard.scalar(f"{prefix}/merge_success", float(report.get("merge_success", False)), step)
                tensorboard.scalar(f"{prefix}/intervention_count", float(report.get("intervention_count", 0)), step)
            if replay_dir is not None:
                replay_path = Path(replay_dir) / f"{group_name or 'ppo'}_seed_{seed}.json"
                write_replay_file(
                    replay_path,
                    run_id=str(cfg.run.run_id),
                    stage="stage5",
                    episode=episode_idx,
                    seed=int(seed),
                    actions=actions,
                    shield_enabled=shield_enabled,
                    risk_checkpoint=risk_checkpoint if shield_enabled else None,
                    model_path=str(model_path),
                    group_name=group_name,
                    notes={"episode_report": report},
                )
        finally:
            env.close()
    metrics = aggregate_episode_reports(reports)
    metrics["average_reward"] = float(np.mean(rewards)) if rewards else 0.0
    metrics["merge_success_rate"] = float(np.mean([float(item.get("merge_success", False)) for item in reports])) if reports else 0.0
    stage_log("stage5", f"group={group_name or 'ppo'} metrics={metrics}")
    return {"episodes": reports, "metrics": metrics}
