from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np

from safe_rl.pipeline.common import latest_stage_file, load_stage_config, make_env, parse_config_arg, write_report
from safe_rl.risk.risk_module import RiskModuleWrapper
from safe_rl.rl.ppo import load_ppo
from safe_rl.shield.safety_shield import SafetyShield
from safe_rl.sim.action_space import decode_action
from safe_rl.utils.config import prepare_run_dir
from safe_rl.utils.io import append_jsonl
from safe_rl.utils.progress import TensorboardLogger, progress_iter, stage_log
from safe_rl.utils.replay import write_replay_file


def _model_path(cfg) -> Path:
    return latest_stage_file(cfg, "stage3", str(cfg.stage3.model_name))


def _risk_path(cfg) -> Path:
    return latest_stage_file(cfg, "stage2", "risk_module.pt")


def _array_summary(values: list[float]) -> dict:
    if not values:
        return {"count": 0}
    arr = np.asarray(values, dtype=np.float32)
    return {
        "count": int(arr.shape[0]),
        "min": float(np.min(arr)),
        "mean": float(np.mean(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
    }


def _shadow_summary(records: list[dict]) -> dict:
    if not records:
        return {
            "count": 0,
            "would_replace_rate": 0.0,
            "fallback_rate": 0.0,
        }
    replacement_deltas = [
        float(record["risk_before"]) - float(record["risk_after"])
        for record in records
        if bool(record.get("would_replace", False))
    ]
    return {
        "count": len(records),
        "would_replace_rate": float(np.mean([bool(record.get("would_replace", False)) for record in records])),
        "fallback_rate": float(np.mean([bool(record.get("fallback", False)) for record in records])),
        "reason_counts": dict(Counter(str(record.get("replacement_reason", "")) for record in records)),
        "raw_action_counts": dict(Counter(str(record.get("raw_action_name", "")) for record in records)),
        "final_action_counts": dict(Counter(str(record.get("final_action_name", "")) for record in records)),
        "raw_risk": _array_summary([float(record["risk_before"]) for record in records]),
        "final_risk": _array_summary([float(record["risk_after"]) for record in records]),
        "replacement_risk_delta": _array_summary(replacement_deltas),
    }


def run(cfg) -> Path:
    stage_dir = prepare_run_dir(cfg, "stage4")
    model_path = _model_path(cfg)
    risk_path = _risk_path(cfg)
    stage_log("stage4", f"run_id={cfg.run.run_id}")
    stage_log("stage4", f"mode={cfg.stage4.mode}")
    stage_log("stage4", f"ppo_model={model_path}")
    stage_log("stage4", f"risk_checkpoint={risk_path}")
    stage_log("stage4", f"output_dir={stage_dir}")
    tb = TensorboardLogger(stage_dir / "tensorboard", enabled=bool(cfg.run.get("tensorboard", True)))
    model = load_ppo(model_path)
    risk_model = RiskModuleWrapper(cfg, checkpoint=str(risk_path))
    shadow_shield = SafetyShield(cfg, risk_model)
    shadow_shield.enabled = True

    mode = str(cfg.stage4.mode)
    intervention_env = mode == "intervention"
    env = make_env(
        cfg,
        seed=int(cfg.run.seed),
        shield_enabled=intervention_env,
        risk_checkpoint=str(risk_path) if intervention_env else None,
        record_trajectory_samples=True,
    )
    transitions = {
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
    reports: list[dict] = []
    shadow_records: list[dict] = []
    events_path = stage_dir / "intervention_buffer.jsonl"
    replay_dir = stage_dir / "replay"
    if events_path.exists():
        events_path.unlink()
    try:
        for episode in progress_iter(range(int(cfg.stage4.episodes)), desc="Stage4 episodes"):
            episode_seed = int(cfg.run.seed) + episode
            stage_log("stage4", f"episode={episode} seed={episode_seed} reset SUMO")
            obs, _info = env.reset(seed=episode_seed)
            terminated = truncated = False
            episode_reward = 0.0
            episode_actions: list[int] = []
            while not (terminated or truncated):
                action, _state = model.predict(obs, deterministic=True)
                action = int(action)
                episode_actions.append(action)
                shadow_record = None
                if not intervention_env:
                    raw_action = decode_action(action)
                    final_action, shadow_record = shadow_shield.select_action(raw_action, env.get_risk_context())
                    shadow_record["would_replace"] = final_action.index != raw_action.index
                    shadow_records.append(shadow_record)
                next_obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += float(reward)
                risk_features = np.asarray(info.get("explicit_risk_features"), dtype=np.float32)
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
                if overall > 0 or shadow_record or info.get("intervention"):
                    append_jsonl(
                        events_path,
                        {
                            "episode": episode,
                            "step": info.get("step"),
                            "mode": mode,
                            "raw_action": action,
                            "shadow": shadow_record,
                            "intervention": info.get("intervention"),
                            "outcome": {
                                "collision": info.get("collision"),
                                "near_miss": info.get("near_miss"),
                                "min_distance": info.get("min_distance"),
                                "min_ttc": info.get("min_ttc"),
                                "max_drac": info.get("max_drac"),
                                "done_reason": info.get("done_reason"),
                            },
                        },
                    )
                obs = next_obs
            episode_report = env.episode_report()
            episode_report["episode_reward"] = episode_reward
            reports.append(episode_report)
            tb.scalar("stage4/episode_reward", episode_reward, episode)
            tb.scalar("stage4/intervention_count", float(episode_report.get("intervention_count", 0)), episode)
            tb.scalar("stage4/fallback_count", float(episode_report.get("fallback_count", 0)), episode)
            tb.scalar("stage4/collision", float(episode_report.get("collision", False)), episode)
            if bool(cfg.stage4.get("replay_enabled", True)) and bool(cfg.run.get("replay", True)):
                replay_path = replay_dir / f"episode_{episode:04d}.json"
                write_replay_file(
                    replay_path,
                    run_id=str(cfg.run.run_id),
                    stage="stage4",
                    episode=episode,
                    seed=episode_seed,
                    actions=episode_actions,
                    shield_enabled=intervention_env,
                    risk_checkpoint=str(risk_path) if intervention_env else None,
                    model_path=str(model_path),
                    notes={"mode": mode, "episode_report": episode_report},
                )
                stage_log("stage4", f"episode={episode} replay={replay_path}")
    finally:
        env.close()

    output = stage_dir / "on_policy_failure_buffer.npz"
    np.savez_compressed(output, **{key: np.asarray(value) for key, value in transitions.items()})
    actions = np.asarray(transitions["actions"], dtype=np.int64)
    risk_types = np.asarray(transitions["risk_types"], dtype=np.float32)
    overall_risk = np.asarray(transitions["overall_risk"], dtype=np.float32)
    report = {
        "stage": "stage4",
        "mode": mode,
        "buffer": str(output),
        "interventions": str(events_path),
        "replay_dir": str(replay_dir),
        "tensorboard": str(stage_dir / "tensorboard"),
        "transition_count": len(transitions["actions"]),
        "action_histogram": {
            str(index): int(count)
            for index, count in enumerate(np.bincount(actions, minlength=9))
        } if actions.size else {},
        "overall_risk_rate": float(np.mean(overall_risk)) if overall_risk.size else 0.0,
        "risk_type_rates": {
            "collision": float(np.mean(risk_types[:, 0])) if risk_types.size else 0.0,
            "near_miss": float(np.mean(risk_types[:, 1])) if risk_types.size else 0.0,
            "low_ttc": float(np.mean(risk_types[:, 2])) if risk_types.size else 0.0,
            "high_drac": float(np.mean(risk_types[:, 3])) if risk_types.size else 0.0,
            "merge_conflict": float(np.mean(risk_types[:, 4])) if risk_types.size else 0.0,
        },
        "shadow_summary": _shadow_summary(shadow_records),
        "episodes": reports,
    }
    write_report(stage_dir / "stage4_report.json", report)
    tb.close()
    stage_log("stage4", f"buffer={output}")
    stage_log("stage4", f"report={stage_dir / 'stage4_report.json'}")
    return output


def main() -> None:
    args = parse_config_arg("Stage4 on-policy failure/intervention collection")
    cfg = load_stage_config(args)
    run(cfg)


if __name__ == "__main__":
    main()
