from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np

from safe_rl.pipeline.common import latest_stage_file, load_stage_config, make_env, parse_config_arg, write_report
from safe_rl.pipeline.stage1_risk_probe import _continuous_risk_coverage
from safe_rl.risk.merge_local import candidate_action_risk_samples, candidate_sample_weight, merge_local_stats
from safe_rl.risk.risk_module import RiskModuleWrapper
from safe_rl.rl.ppo import _training_device, load_ppo
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


def _shadow_candidate_ranking(candidate_samples, raw_action, final_action, risk_model, context) -> dict:
    legal_samples = [sample for sample in candidate_samples if bool(sample.candidate_legal)]
    if not legal_samples:
        return {
            "raw_action_rank": None,
            "chosen_action_rank": None,
            "oracle_action_rank": None,
            "model_vs_oracle_risk_delta": None,
        }
    predictions = {
        sample.action: float(risk_model.predict(decode_action(sample.action), context).risk_score)
        for sample in legal_samples
    }
    label_risks = {sample.action: float(sample.traffic_risk) for sample in legal_samples}
    model_order = sorted(label_risks, key=lambda action: (predictions[action], action))
    oracle_order = sorted(label_risks, key=lambda action: (label_risks[action], action))
    model_rank = {action: rank + 1 for rank, action in enumerate(model_order)}
    oracle_rank = {action: rank + 1 for rank, action in enumerate(oracle_order)}
    oracle_action = oracle_order[0]
    final_index = int(final_action.index)
    return {
        "raw_action_rank": model_rank.get(int(raw_action.index)),
        "chosen_action_rank": model_rank.get(final_index),
        "oracle_action_rank": oracle_rank.get(final_index),
        "oracle_action": int(oracle_action),
        "model_best_action": int(model_order[0]),
        "model_vs_oracle_risk_delta": (
            float(label_risks[final_index] - label_risks[oracle_action])
            if final_index in label_risks
            else None
        ),
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
    legal_replacement_deltas = [
        float(record["risk_before"]) - float(record["risk_after"])
        for record in records
        if bool(record.get("would_replace", False)) and bool(record.get("final_candidate_legal", True))
    ]
    raw_action_ranks = [float(record["raw_action_rank"]) for record in records if record.get("raw_action_rank") is not None]
    chosen_action_ranks = [
        float(record["chosen_action_rank"]) for record in records if record.get("chosen_action_rank") is not None
    ]
    oracle_action_ranks = [
        float(record["oracle_action_rank"]) for record in records if record.get("oracle_action_rank") is not None
    ]
    model_vs_oracle = [
        float(record["model_vs_oracle_risk_delta"])
        for record in records
        if record.get("model_vs_oracle_risk_delta") is not None
    ]
    return {
        "count": len(records),
        "would_replace_rate": float(np.mean([bool(record.get("would_replace", False)) for record in records])),
        "legal_candidate_would_replace_rate": float(
            np.mean(
                [
                    bool(record.get("would_replace", False)) and bool(record.get("final_candidate_legal", True))
                    for record in records
                ]
            )
        ),
        "fallback_rate": float(np.mean([bool(record.get("fallback", False)) for record in records])),
        "raw_illegal_rate": float(np.mean([not bool(record.get("raw_candidate_legal", True)) for record in records])),
        "mean_legal_candidate_count": float(
            np.mean([int(record.get("legal_candidate_count", 9)) for record in records])
        ),
        "mean_illegal_candidate_count": float(
            np.mean([int(record.get("illegal_candidate_count", 0)) for record in records])
        ),
        "reason_counts": dict(Counter(str(record.get("replacement_reason", "")) for record in records)),
        "raw_action_counts": dict(Counter(str(record.get("raw_action_name", "")) for record in records)),
        "final_action_counts": dict(Counter(str(record.get("final_action_name", "")) for record in records)),
        "raw_risk": _array_summary([float(record["risk_before"]) for record in records]),
        "final_risk": _array_summary([float(record["risk_after"]) for record in records]),
        "replacement_risk_delta": _array_summary(replacement_deltas),
        "legal_replacement_risk_delta": _array_summary(legal_replacement_deltas),
        "raw_action_rank": _array_summary(raw_action_ranks),
        "chosen_action_rank": _array_summary(chosen_action_ranks),
        "oracle_action_rank": _array_summary(oracle_action_ranks),
        "model_vs_oracle_risk_delta": _array_summary(model_vs_oracle),
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
    model = load_ppo(model_path, device=_training_device(cfg))
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
                context = env.get_risk_context()
                candidate_samples = candidate_action_risk_samples(context)
                candidate_by_action = {sample.action: sample for sample in candidate_samples}
                local = merge_local_stats(context.get("ego"), list(context.get("vehicles") or []), cfg)
                shadow_record = None
                if not intervention_env:
                    raw_action = decode_action(action)
                    final_action, shadow_record = shadow_shield.select_action(raw_action, context)
                    shadow_record["would_replace"] = final_action.index != raw_action.index
                    shadow_record.update(
                        _shadow_candidate_ranking(candidate_samples, raw_action, final_action, risk_model, context)
                    )
                    shadow_records.append(shadow_record)
                next_obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += float(reward)
                actual_risk_types = np.asarray(
                    [
                        float(info.get("collision", False)),
                        float(info.get("near_miss", False)),
                        float(info.get("low_ttc", False)),
                        float(info.get("high_drac", False)),
                        float(local.merge_zone_risk),
                        float(local.taper_miss),
                    ],
                    dtype=np.float32,
                )
                actual_overall = float(np.max(actual_risk_types))
                executed_sample = candidate_by_action.get(action)
                executed_candidate_risk = float(executed_sample.overall_risk) if executed_sample is not None else 0.0
                executed_candidate_legal = (
                    bool(executed_sample.candidate_legal) if executed_sample is not None else True
                )
                executed_lane_oob_risk = float(executed_sample.lane_oob) if executed_sample is not None else 0.0
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
                if actual_overall > 0 or executed_candidate_risk > 0 or shadow_record or info.get("intervention"):
                    append_jsonl(
                        events_path,
                        {
                            "episode": episode,
                            "step": info.get("step"),
                            "mode": mode,
                            "raw_action": action,
                            "executed_candidate_risk": executed_candidate_risk,
                            "executed_candidate_legal": executed_candidate_legal,
                            "executed_lane_oob_risk": executed_lane_oob_risk,
                            "shadow": shadow_record,
                            "intervention": info.get("intervention"),
                            "outcome": {
                                "collision": info.get("collision"),
                                "near_miss": info.get("near_miss"),
                                "min_distance": info.get("min_distance"),
                                "min_ttc": info.get("min_ttc"),
                                "max_drac": info.get("max_drac"),
                                "target_lane_gap": local.target_lane_gap,
                                "target_front_gap": local.target_front_gap,
                                "target_rear_gap": local.target_rear_gap,
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
                    safety_metric_version=str(cfg.risk_module.get("safety_metric_version", "")),
                    notes={"mode": mode, "episode_report": episode_report},
                )
                stage_log("stage4", f"episode={episode} replay={replay_path}")
    finally:
        env.close()

    output = stage_dir / "on_policy_failure_buffer.npz"
    np.savez_compressed(output, **{key: np.asarray(value) for key, value in transitions.items()})
    actions = np.asarray(transitions["actions"], dtype=np.int64)
    executed_actions = np.asarray(transitions["executed_actions"], dtype=np.int64)
    risk_types = np.asarray(transitions["risk_types"], dtype=np.float32)
    overall_risk = np.asarray(transitions["overall_risk"], dtype=np.float32)
    traffic_risk = np.asarray(transitions["traffic_risk"], dtype=np.float32)
    lane_oob_risk = np.asarray(transitions["lane_oob_risk"], dtype=np.float32)
    candidate_legal = np.asarray(transitions["candidate_legal"], dtype=np.float32) > 0.5
    legal_risk = traffic_risk[candidate_legal]
    continuous_risk = np.asarray(transitions["continuous_risk_target"], dtype=np.float32)
    legal_continuous = continuous_risk[candidate_legal]
    report = {
        "stage": "stage4",
        "mode": mode,
        "buffer": str(output),
        "interventions": str(events_path),
        "replay_dir": str(replay_dir),
        "tensorboard": str(stage_dir / "tensorboard"),
        "transition_count": len(transitions["executed_actions"]),
        "candidate_risk_sample_count": len(transitions["actions"]),
        "action_histogram": {
            str(index): int(count)
            for index, count in enumerate(np.bincount(executed_actions, minlength=9))
        } if executed_actions.size else {},
        "candidate_action_histogram": {
            str(index): int(count)
            for index, count in enumerate(np.bincount(actions, minlength=9))
        } if actions.size else {},
        "overall_risk_rate": float(np.mean(overall_risk)) if overall_risk.size else 0.0,
        "risk_labels": {
            "overall_risk_semantics": "traffic_risk_only",
            "overall_risk_rate": float(np.mean(overall_risk)) if overall_risk.size else 0.0,
            "traffic_risk_rate": float(np.mean(traffic_risk)) if traffic_risk.size else 0.0,
            "lane_oob_risk_rate": float(np.mean(lane_oob_risk)) if lane_oob_risk.size else 0.0,
            "illegal_candidate_rate": float(np.mean(~candidate_legal)) if candidate_legal.size else 0.0,
            "legal_candidate_risk_rate": float(np.mean(legal_risk)) if legal_risk.size else 0.0,
            "traffic_risk_by_action": {
                str(index): float(np.mean(traffic_risk[actions == index])) if np.any(actions == index) else 0.0
                for index in range(9)
            },
            "lane_oob_by_action": {
                str(index): float(np.mean(lane_oob_risk[actions == index])) if np.any(actions == index) else 0.0
                for index in range(9)
            },
            "legal_candidate_action_risk_rate": {
                str(index): (
                    float(np.mean(traffic_risk[(actions == index) & candidate_legal]))
                    if np.any((actions == index) & candidate_legal)
                    else 0.0
                )
                for index in range(9)
            },
        },
        "risk_type_rates": {
            "collision": float(np.mean(risk_types[:, 0])) if risk_types.size else 0.0,
            "near_miss": float(np.mean(risk_types[:, 1])) if risk_types.size else 0.0,
            "low_ttc": float(np.mean(risk_types[:, 2])) if risk_types.size else 0.0,
            "high_drac": float(np.mean(risk_types[:, 3])) if risk_types.size else 0.0,
            "merge_conflict": float(np.mean(risk_types[:, 4])) if risk_types.size else 0.0,
            "taper_miss": float(np.mean(risk_types[:, 5])) if risk_types.size and risk_types.shape[1] > 5 else 0.0,
        },
        "candidate_action_risk_rate": {
            str(index): float(np.mean(overall_risk[actions == index])) if np.any(actions == index) else 0.0
            for index in range(9)
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
        "continuous_risk": {
            **_continuous_risk_coverage(legal_continuous),
            "distance_to_taper": _array_summary([float(item) for item in transitions["distance_to_taper"]]),
            "taper_miss_rate": (
                float(np.mean(np.asarray(transitions["taper_miss"], dtype=np.float32)))
                if transitions["taper_miss"]
                else 0.0
            ),
            "curriculum_profile_counts": dict(Counter(str(item) for item in transitions["curriculum_profiles"])),
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
