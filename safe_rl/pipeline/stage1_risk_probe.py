from __future__ import annotations

from pathlib import Path

import numpy as np

from safe_rl.analysis.stage1_audit import audit_stage1_buffer
from safe_rl.pipeline.common import json_ready, load_stage_config, make_env, parse_config_arg, write_report
from safe_rl.risk.merge_local import candidate_action_risk_samples, candidate_sample_weight, merge_local_stats
from safe_rl.risk.risk_aggregator import aggregate_episode_reports
from safe_rl.risk.stage1_sampling import configured_sampling_probs, sampling_summary, select_stage1_action
from safe_rl.utils.config import prepare_run_dir
from safe_rl.utils.io import append_jsonl
from safe_rl.utils.progress import TensorboardLogger, progress_iter, stage_log
from safe_rl.utils.replay import write_replay_file


def _array_summary(values: list[float]) -> dict:
    if not values:
        return {"count": 0}
    arr = np.asarray(values, dtype=np.float32)
    return {
        "count": int(arr.shape[0]),
        "p1": float(np.percentile(arr, 1)),
        "p5": float(np.percentile(arr, 5)),
        "mean": float(np.mean(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
    }


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
        "executed_actions": [],
        "next_observations": [],
        "rewards": [],
        "dones": [],
        "risk_features": [],
        "overall_risk": [],
        "risk_types": [],
        "lane_oob_risk": [],
        "candidate_legal": [],
        "traffic_risk": [],
        "continuous_risk_target": [],
        "boundary_sample": [],
        "risk_sample_weight": [],
        "episode_id": [],
        "candidate_transition_id": [],
        "candidate_raw_action": [],
        "transition_episode_id": [],
        "candidate_target_lane_gap": [],
        "candidate_ramp_local_risk": [],
        "candidate_merge_zone_risk": [],
        "candidate_taper_miss": [],
        "candidate_distance_to_taper": [],
        "candidate_ego_on_auxiliary": [],
        "target_lane_gap": [],
        "ramp_local_risk": [],
        "merge_zone_risk": [],
        "taper_miss": [],
        "distance_to_taper": [],
        "ego_on_auxiliary": [],
        "curriculum_profiles": [],
        "sampling_modes": [],
    }
    history_samples: list[np.ndarray] = []
    future_samples: list[np.ndarray] = []
    agent_masks: list[np.ndarray] = []
    agent_lane_indices: list[np.ndarray] = []
    agent_edge_roles: list[np.ndarray] = []
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
                context = env.get_risk_context()
                action, sampling_mode = select_stage1_action(cfg, rng, context)
                candidate_samples = candidate_action_risk_samples(context)
                candidate_by_action = {sample.action: sample for sample in candidate_samples}
                local = merge_local_stats(context.get("ego"), list(context.get("vehicles") or []), cfg)
                next_obs, reward, terminated, truncated, info = env.step(action)
                episode_actions.append(action)
                episode_reward += float(reward)
                transitions["observations"].append(obs)
                transitions["executed_actions"].append(action)
                transitions["next_observations"].append(next_obs)
                transitions["rewards"].append(reward)
                transitions["dones"].append(float(terminated or truncated))
                transitions["transition_episode_id"].append(episode)
                transitions["target_lane_gap"].append(local.target_lane_gap)
                transitions["ramp_local_risk"].append(float(local.ramp_local_risk))
                transitions["merge_zone_risk"].append(float(local.merge_zone_risk))
                transitions["taper_miss"].append(float(local.taper_miss))
                transitions["distance_to_taper"].append(float(local.merge_distance))
                transitions["ego_on_auxiliary"].append(float(local.ego_on_auxiliary))
                transitions["curriculum_profiles"].append(str(context.get("curriculum_profile", "disabled")))
                transitions["sampling_modes"].append(sampling_mode)
                transition_id = len(transitions["executed_actions"]) - 1
                for sample in candidate_samples:
                    transitions["actions"].append(sample.action)
                    transitions["risk_features"].append(sample.features)
                    transitions["overall_risk"].append(sample.overall_risk)
                    transitions["risk_types"].append(sample.risk_types)
                    transitions["lane_oob_risk"].append(sample.lane_oob)
                    transitions["candidate_legal"].append(float(sample.candidate_legal))
                    transitions["traffic_risk"].append(sample.traffic_risk)
                    transitions["continuous_risk_target"].append(sample.continuous_risk_target)
                    transitions["boundary_sample"].append(float(sample.boundary_sample))
                    transitions["risk_sample_weight"].append(candidate_sample_weight(sample))
                    transitions["episode_id"].append(episode)
                    transitions["candidate_transition_id"].append(transition_id)
                    transitions["candidate_raw_action"].append(action)
                    transitions["candidate_target_lane_gap"].append(sample.local_stats.target_lane_gap)
                    transitions["candidate_ramp_local_risk"].append(float(sample.local_stats.ramp_local_risk))
                    transitions["candidate_merge_zone_risk"].append(float(sample.local_stats.merge_zone_risk))
                    transitions["candidate_taper_miss"].append(float(sample.local_stats.taper_miss))
                    transitions["candidate_distance_to_taper"].append(float(sample.distance_to_taper))
                    transitions["candidate_ego_on_auxiliary"].append(float(sample.ego_on_auxiliary))
                executed_sample = candidate_by_action.get(action)
                executed_candidate_risk = float(executed_sample.overall_risk) if executed_sample is not None else 0.0
                executed_candidate_legal = (
                    bool(executed_sample.candidate_legal) if executed_sample is not None else True
                )
                executed_lane_oob_risk = float(executed_sample.lane_oob) if executed_sample is not None else 0.0
                actual_risk_types = np.asarray(
                    [
                        float(info.get("collision", False)),
                        float(info.get("near_miss", False)),
                        float(info.get("low_ttc", False)),
                        float(info.get("high_drac", False)),
                        float(local.merge_zone_risk),
                    ],
                    dtype=np.float32,
                )
                actual_overall = float(np.max(actual_risk_types))
                if actual_overall > 0 or executed_candidate_risk > 0:
                    append_jsonl(
                        events_path,
                        json_ready(
                            {
                                "episode": episode,
                                "step": info.get("step"),
                                "action": action,
                                "sampling_mode": sampling_mode,
                                "executed_candidate_risk": executed_candidate_risk,
                                "executed_candidate_legal": executed_candidate_legal,
                                "executed_lane_oob_risk": executed_lane_oob_risk,
                                "collision": info.get("collision"),
                                "near_miss": info.get("near_miss"),
                                "min_distance": info.get("min_distance"),
                                "min_ttc": info.get("min_ttc"),
                                "max_drac": info.get("max_drac"),
                                "merge_gap": local.target_lane_gap,
                                "target_front_gap": local.target_front_gap,
                                "target_rear_gap": local.target_rear_gap,
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
            hist, fut, mask, lane_indices, edge_roles = env.trajectory_window_samples()
            if hist.shape[0] > 0:
                history_samples.append(hist)
                future_samples.append(fut)
                agent_masks.append(mask)
                agent_lane_indices.append(lane_indices)
                agent_edge_roles.append(edge_roles)
    finally:
        env.close()

    output = stage_dir / str(cfg.stage1.output_name)
    np.savez_compressed(
        output,
        **{key: np.asarray(value) for key, value in transitions.items()},
        agent_history=np.concatenate(history_samples, axis=0) if history_samples else np.zeros((0, 1, 1, 5)),
        agent_future=np.concatenate(future_samples, axis=0) if future_samples else np.zeros((0, 1, 1, 5)),
        agent_mask=np.concatenate(agent_masks, axis=0) if agent_masks else np.zeros((0, 1)),
        agent_lane_index=np.concatenate(agent_lane_indices, axis=0) if agent_lane_indices else np.zeros((0, 1), dtype=np.int64),
        agent_edge_role=np.concatenate(agent_edge_roles, axis=0) if agent_edge_roles else np.zeros((0, 1), dtype=np.int64),
    )
    audit_report = None
    if bool(cfg.stage1.get("audit_enabled", True)):
        audit_report = audit_stage1_buffer(output, stage_dir / "audit")
        stage_log("stage1", f"audit={stage_dir / 'audit' / 'stage1_data_audit.json'}")
    candidate_actions = np.asarray(transitions["actions"], dtype=np.int64)
    traffic_risk = np.asarray(transitions["traffic_risk"], dtype=np.float32)
    lane_oob_risk = np.asarray(transitions["lane_oob_risk"], dtype=np.float32)
    candidate_legal = np.asarray(transitions["candidate_legal"], dtype=np.float32) > 0.5
    legal_risk = traffic_risk[candidate_legal]
    continuous_risk = np.asarray(transitions["continuous_risk_target"], dtype=np.float32)
    legal_continuous = continuous_risk[candidate_legal]
    legal_boundary = legal_continuous[(legal_continuous >= 0.20) & (legal_continuous < 0.80)]
    coverage = _continuous_risk_coverage(legal_continuous)
    audit_gate = _stage1_audit_gate(cfg, len(transitions["actions"]), legal_boundary.shape[0], coverage)
    report = {
        "stage": "stage1",
        "run_id": cfg.run.run_id,
        "buffer": str(output),
        "events": str(events_path),
        "replay_dir": str(replay_dir),
        "audit": str(stage_dir / "audit" / "stage1_data_audit.json") if audit_report else None,
        "tensorboard": str(stage_dir / "tensorboard"),
        "transition_count": len(transitions["executed_actions"]),
        "candidate_risk_sample_count": len(transitions["actions"]),
        "trajectory_sample_count": int(sum(item.shape[0] for item in history_samples)),
        "action_sampling": {
            "mode": str(cfg.stage1.get("action_sampling", "random")),
            "configured_probs": configured_sampling_probs(cfg),
            "actual": sampling_summary([str(item) for item in transitions["sampling_modes"]]),
        },
        "merge_local": {
            "target_lane_gap": _array_summary([float(item) for item in transitions["target_lane_gap"]]),
            "candidate_target_lane_gap": _array_summary(
                [float(item) for item in transitions["candidate_target_lane_gap"]]
            ),
            "ramp_local_risk_rate": (
                float(np.mean(np.asarray(transitions["ramp_local_risk"], dtype=np.float32)))
                if transitions["ramp_local_risk"]
                else 0.0
            ),
            "merge_zone_risk_rate": (
                float(np.mean(np.asarray(transitions["merge_zone_risk"], dtype=np.float32)))
                if transitions["merge_zone_risk"]
                else 0.0
            ),
            "candidate_ramp_local_risk_rate": (
                float(np.mean(np.asarray(transitions["candidate_ramp_local_risk"], dtype=np.float32)))
                if transitions["candidate_ramp_local_risk"]
                else 0.0
            ),
            "candidate_merge_zone_risk_rate": (
                float(np.mean(np.asarray(transitions["candidate_merge_zone_risk"], dtype=np.float32)))
                if transitions["candidate_merge_zone_risk"]
                else 0.0
            ),
            "candidate_taper_miss_rate": (
                float(np.mean(np.asarray(transitions["candidate_taper_miss"], dtype=np.float32)))
                if transitions["candidate_taper_miss"]
                else 0.0
            ),
        },
        "risk_labels": {
            "overall_risk_semantics": "traffic_risk_only",
            "overall_risk_rate": float(np.mean(traffic_risk)) if traffic_risk.size else 0.0,
            "traffic_risk_rate": float(np.mean(traffic_risk)) if traffic_risk.size else 0.0,
            "lane_oob_risk_rate": float(np.mean(lane_oob_risk)) if lane_oob_risk.size else 0.0,
            "illegal_candidate_rate": float(np.mean(~candidate_legal)) if candidate_legal.size else 0.0,
            "legal_candidate_risk_rate": float(np.mean(legal_risk)) if legal_risk.size else 0.0,
            "traffic_risk_by_action": {
                str(index): (
                    float(np.mean(traffic_risk[candidate_actions == index]))
                    if np.any(candidate_actions == index)
                    else 0.0
                )
                for index in range(9)
            },
            "lane_oob_by_action": {
                str(index): (
                    float(np.mean(lane_oob_risk[candidate_actions == index]))
                    if np.any(candidate_actions == index)
                    else 0.0
                )
                for index in range(9)
            },
            "legal_candidate_action_risk_rate": {
                str(index): (
                    float(np.mean(traffic_risk[(candidate_actions == index) & candidate_legal]))
                    if np.any((candidate_actions == index) & candidate_legal)
                    else 0.0
                )
                for index in range(9)
            },
        },
        "continuous_risk": {
            **coverage,
            "summary": _array_summary([float(item) for item in continuous_risk.tolist()]),
            "legal_summary": _array_summary([float(item) for item in legal_continuous.tolist()]),
            "distance_to_taper": _array_summary([float(item) for item in transitions["distance_to_taper"]]),
            "taper_miss_rate": (
                float(np.mean(np.asarray(transitions["taper_miss"], dtype=np.float32)))
                if transitions["taper_miss"]
                else 0.0
            ),
            "curriculum_profile_counts": {
                str(profile): int(sum(1 for item in transitions["curriculum_profiles"] if item == profile))
                for profile in sorted(set(transitions["curriculum_profiles"]))
            },
        },
        "audit_gate": audit_gate,
        "metrics": aggregate_episode_reports(reports),
    }
    write_report(stage_dir / "stage1_report.json", report)
    tb.close()
    stage_log("stage1", f"buffer={output}")
    stage_log("stage1", f"report={stage_dir / 'stage1_report.json'}")
    if not bool(audit_gate.get("passed", True)):
        raise RuntimeError(f"Stage1 audit gate failed: {audit_gate}")
    return output


def _continuous_risk_coverage(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return {
            "sample_count": 0,
            "easy_safe_rate": 0.0,
            "boundary_rate": 0.0,
            "extreme_risk_rate": 0.0,
            "boundary_sample_count": 0,
        }
    boundary = (values >= 0.20) & (values < 0.80)
    return {
        "sample_count": int(values.size),
        "easy_safe_rate": float(np.mean(values < 0.20)),
        "boundary_rate": float(np.mean(boundary)),
        "extreme_risk_rate": float(np.mean(values >= 0.80)),
        "boundary_sample_count": int(np.sum(boundary)),
    }


def _stage1_audit_gate(cfg, candidate_count: int, legal_boundary_count: int, coverage: dict) -> dict:
    gate_cfg = cfg.stage1.get("audit_gate", {})
    enabled = bool(gate_cfg.get("enabled", False))
    smoke_skip = int(gate_cfg.get("smoke_skip_below_episodes", 0))
    skipped = int(cfg.stage1.episodes) <= smoke_skip
    checks = {
        "candidate_samples": int(candidate_count) >= int(gate_cfg.get("min_candidate_samples", 0)),
        "legal_boundary_samples": int(legal_boundary_count) >= int(gate_cfg.get("min_legal_boundary_samples", 0)),
        "boundary_rate": float(coverage.get("boundary_rate", 0.0)) >= float(gate_cfg.get("min_boundary_rate", 0.0)),
        "easy_safe_non_empty": float(coverage.get("easy_safe_rate", 0.0)) > 0.0,
        "boundary_non_empty": int(coverage.get("boundary_sample_count", 0)) > 0,
        "extreme_risk_non_empty": float(coverage.get("extreme_risk_rate", 0.0)) > 0.0,
    }
    return {
        "enabled": enabled,
        "skipped_for_smoke": skipped,
        "passed": bool(not enabled or skipped or all(checks.values())),
        "checks": checks,
        "candidate_sample_count": int(candidate_count),
        "legal_boundary_sample_count": int(legal_boundary_count),
    }


def main() -> None:
    args = parse_config_arg("Stage1 SUMO risk prior collection")
    cfg = load_stage_config(args)
    run(cfg)


if __name__ == "__main__":
    main()
