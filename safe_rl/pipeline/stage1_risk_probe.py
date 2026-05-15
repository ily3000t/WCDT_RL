from __future__ import annotations

from pathlib import Path

import numpy as np

from safe_rl.analysis.stage1_audit import audit_stage1_buffer
from safe_rl.pipeline.common import json_ready, load_stage_config, make_env, parse_config_arg, write_report
from safe_rl.risk.risk_aggregator import aggregate_episode_reports
from safe_rl.utils.config import prepare_run_dir
from safe_rl.utils.io import append_jsonl
from safe_rl.utils.progress import TensorboardLogger, progress_iter, stage_log
from safe_rl.utils.replay import write_replay_file


def run(cfg) -> Path:
    stage_dir = prepare_run_dir(cfg, "stage1")
    stage_log("stage1", f"run_id={cfg.run.run_id}")
    stage_log("stage1", f"SUMO config={cfg.scenario.sumocfg}")
    stage_log("stage1", f"SUMO binary={cfg.scenario.sumo_binary}, episodes={cfg.stage1.episodes}")
    stage_log("stage1", f"output_dir={stage_dir}")
    tb = TensorboardLogger(stage_dir / "tensorboard", enabled=bool(cfg.run.get("tensorboard", True)))
    rng = np.random.default_rng(int(cfg.run.seed))
    transitions: dict[str, list] = {
        "observations": [],
        "actions": [],
        "next_observations": [],
        "rewards": [],
        "dones": [],
        "risk_features": [],
        "overall_risk": [],
        "risk_types": [],
        "episode_id": [],
    }
    history_samples: list[np.ndarray] = []
    future_samples: list[np.ndarray] = []
    agent_masks: list[np.ndarray] = []
    reports: list[dict] = []
    events_path = stage_dir / "risk_events.jsonl"
    replay_dir = stage_dir / "replay"
    if events_path.exists():
        events_path.unlink()

    env = make_env(cfg, seed=int(cfg.run.seed), shield_enabled=False, record_trajectory_samples=True)
    try:
        for episode in progress_iter(range(int(cfg.stage1.episodes)), desc="Stage1 episodes"):
            episode_seed = int(cfg.run.seed) + episode
            stage_log("stage1", f"episode={episode} seed={episode_seed} reset SUMO")
            obs, _info = env.reset(seed=episode_seed)
            terminated = truncated = False
            episode_actions: list[int] = []
            episode_reward = 0.0
            while not (terminated or truncated):
                action = int(rng.integers(0, env.action_space.n))
                next_obs, reward, terminated, truncated, info = env.step(action)
                episode_actions.append(action)
                episode_reward += float(reward)
                risk_features = np.asarray(info.get("explicit_risk_features"), dtype=np.float32)
                if risk_features.size == 0:
                    risk_features = np.zeros((int(cfg.risk_module.explicit_feature_dim),), dtype=np.float32)
                risk_types = np.asarray(
                    [
                        float(info.get("collision", False)),
                        float(info.get("near_miss", False)),
                        float(info.get("low_ttc", False)),
                        float(info.get("high_drac", False)),
                        float(info.get("merge_gap", 1.0e6) < 8.0),
                    ],
                    dtype=np.float32,
                )
                overall = float(np.max(risk_types))
                transitions["observations"].append(obs)
                transitions["actions"].append(action)
                transitions["next_observations"].append(next_obs)
                transitions["rewards"].append(reward)
                transitions["dones"].append(float(terminated or truncated))
                transitions["risk_features"].append(risk_features)
                transitions["overall_risk"].append(overall)
                transitions["risk_types"].append(risk_types)
                transitions["episode_id"].append(episode)
                if overall > 0:
                    append_jsonl(
                        events_path,
                        json_ready(
                            {
                                "episode": episode,
                                "step": info.get("step"),
                                "action": action,
                                "collision": info.get("collision"),
                                "near_miss": info.get("near_miss"),
                                "min_distance": info.get("min_distance"),
                                "min_ttc": info.get("min_ttc"),
                                "max_drac": info.get("max_drac"),
                                "merge_gap": info.get("merge_gap"),
                                "done_reason": info.get("done_reason"),
                            }
                        ),
                    )
                obs = next_obs
            episode_report = env.episode_report()
            episode_report["episode_reward"] = episode_reward
            reports.append(episode_report)
            tb.scalar("stage1/episode_reward", episode_reward, episode)
            tb.scalar("stage1/collision", float(episode_report.get("collision", False)), episode)
            tb.scalar("stage1/near_miss", float(episode_report.get("near_miss", False)), episode)
            tb.scalar("stage1/min_distance", float(episode_report.get("min_distance", 0.0)), episode)
            if bool(cfg.stage1.get("replay_enabled", True)) and bool(cfg.run.get("replay", True)):
                write_replay_file(
                    replay_dir / f"episode_{episode:04d}.json",
                    run_id=str(cfg.run.run_id),
                    stage="stage1",
                    episode=episode,
                    seed=episode_seed,
                    actions=episode_actions,
                    shield_enabled=False,
                    notes={"episode_report": episode_report},
                )
                stage_log("stage1", f"episode={episode} replay={replay_dir / f'episode_{episode:04d}.json'}")
            hist, fut, mask = env.trajectory_window_samples()
            if hist.shape[0] > 0:
                history_samples.append(hist)
                future_samples.append(fut)
                agent_masks.append(mask)
    finally:
        env.close()

    output = stage_dir / str(cfg.stage1.output_name)
    np.savez_compressed(
        output,
        **{key: np.asarray(value) for key, value in transitions.items()},
        agent_history=np.concatenate(history_samples, axis=0) if history_samples else np.zeros((0, 1, 1, 5)),
        agent_future=np.concatenate(future_samples, axis=0) if future_samples else np.zeros((0, 1, 1, 5)),
        agent_mask=np.concatenate(agent_masks, axis=0) if agent_masks else np.zeros((0, 1)),
    )
    audit_report = None
    if bool(cfg.stage1.get("audit_enabled", True)):
        audit_report = audit_stage1_buffer(output, stage_dir / "audit")
        stage_log("stage1", f"audit={stage_dir / 'audit' / 'stage1_data_audit.json'}")
    report = {
        "stage": "stage1",
        "run_id": cfg.run.run_id,
        "buffer": str(output),
        "events": str(events_path),
        "replay_dir": str(replay_dir),
        "audit": str(stage_dir / "audit" / "stage1_data_audit.json") if audit_report else None,
        "tensorboard": str(stage_dir / "tensorboard"),
        "transition_count": len(transitions["actions"]),
        "trajectory_sample_count": int(sum(item.shape[0] for item in history_samples)),
        "metrics": aggregate_episode_reports(reports),
    }
    write_report(stage_dir / "stage1_report.json", report)
    tb.close()
    stage_log("stage1", f"buffer={output}")
    stage_log("stage1", f"report={stage_dir / 'stage1_report.json'}")
    return output


def main() -> None:
    args = parse_config_arg("Stage1 SUMO risk prior collection")
    cfg = load_stage_config(args)
    run(cfg)


if __name__ == "__main__":
    main()
