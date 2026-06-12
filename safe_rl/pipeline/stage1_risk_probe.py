from __future__ import annotations

import json
import multiprocessing as mp
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from safe_rl.analysis.stage1_audit import audit_stage1_buffer
from safe_rl.prediction.actor_selector import (
    ACTOR_SELECTION_VERSION,
    actor_selection_config_hash,
)
from safe_rl.pipeline.common import json_ready, load_stage_config, make_env, parse_config_arg, write_report
from safe_rl.risk.merge_local import candidate_action_risk_samples, candidate_sample_weight, merge_local_stats
from safe_rl.risk.risk_aggregator import aggregate_episode_reports
from safe_rl.risk.stage1_sampling import configured_sampling_probs, sampling_summary, select_stage1_action
from safe_rl.sim.metrics import SAFETY_METRIC_VERSION
from safe_rl.utils.config import clone_with_overrides, prepare_run_dir
from safe_rl.utils.progress import TensorboardLogger, progress_iter, stage_log
from safe_rl.utils.replay import write_replay_file
from safe_rl.utils.performance import PerformanceTracker


TRANSITION_KEYS = {
    "observations",
    "executed_actions",
    "next_observations",
    "rewards",
    "dones",
    "transition_episode_id",
    "transition_episode_step",
    "target_lane_gap",
    "ramp_local_risk",
    "merge_zone_risk",
    "taper_miss",
    "distance_to_taper",
    "ego_on_auxiliary",
    "curriculum_profiles",
    "sampling_modes",
}
CANDIDATE_KEYS = {
    "actions",
    "risk_features",
    "overall_risk",
    "risk_types",
    "lane_oob_risk",
    "candidate_legal",
    "traffic_risk",
    "continuous_risk_target",
    "boundary_sample",
    "risk_sample_weight",
    "episode_id",
    "candidate_transition_id",
    "candidate_episode_step",
    "candidate_raw_action",
    "candidate_target_lane_gap",
    "candidate_ramp_local_risk",
    "candidate_merge_zone_risk",
    "candidate_taper_miss",
    "candidate_distance_to_taper",
    "candidate_ego_on_auxiliary",
}
TRAJECTORY_KEYS = {
    "agent_history",
    "agent_future",
    "agent_mask",
    "agent_lane_index",
    "agent_edge_role",
    "agent_length",
    "agent_width",
    "agent_history_valid_mask",
    "agent_future_valid_mask",
    "agent_history_lane_index",
    "agent_history_edge_role",
    "agent_future_lane_index",
    "agent_future_edge_role",
    "agent_relevance_mask",
    "agent_relevance_score",
    "actor_selector_relevant_count",
    "actor_selector_overflow",
    "trajectory_episode_id",
    "trajectory_window_end_step",
}


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


def _should_write_replay(cfg, episode_report: dict) -> bool:
    if not bool(cfg.stage1.get("replay_enabled", True)) or not bool(cfg.run.get("replay", True)):
        return False
    mode = str(cfg.stage1.get("replay_mode", "risk_or_failure")).strip().lower()
    if mode == "all":
        return True
    failed = bool(
        episode_report.get("collision")
        or episode_report.get("near_miss")
        or episode_report.get("taper_miss")
        or episode_report.get("safety_violation")
    )
    if mode == "failures_only":
        return failed
    return bool(failed or episode_report.get("_stage1_boundary_or_extreme", False))


def _stage1_worker_entry(cfg, worker_id: int, shard_index: int, episode_ids: list[int], worker_root: str) -> str:
    worker_cfg = clone_with_overrides(
        cfg,
        {
            "run": {
                "run_id": f"worker_{worker_id:03d}_shard_{shard_index:04d}",
                "output_root": str(Path(worker_root).resolve()),
                "tensorboard": False,
            },
            "stage1": {
                "workers": 1,
                "_worker_mode": True,
                "episode_ids": [int(item) for item in episode_ids],
                "episodes": len(episode_ids),
                "audit_enabled": False,
                "audit_gate": {"enabled": False},
            },
        },
    )
    return str(_run_serial(worker_cfg))


def _concatenate_shards(shard_paths: list[Path], output: Path) -> dict[str, np.ndarray]:
    loaded = [dict(np.load(path, allow_pickle=False)) for path in shard_paths]
    keys = sorted(set().union(*(payload.keys() for payload in loaded)))
    merged: dict[str, np.ndarray] = {}
    for key in keys:
        arrays = [payload[key] for payload in loaded if key in payload]
        if key in TRANSITION_KEYS | CANDIDATE_KEYS | TRAJECTORY_KEYS:
            merged[key] = np.concatenate(arrays, axis=0) if arrays else np.zeros((0,), dtype=np.float32)
        else:
            merged[key] = np.asarray(arrays[0])

    transition_count = int(np.asarray(merged.get("transition_episode_id", [])).shape[0])
    if transition_count:
        transition_order = np.lexsort(
            (
                np.asarray(merged["transition_episode_step"], dtype=np.int64),
                np.asarray(merged["transition_episode_id"], dtype=np.int64),
            )
        )
        for key in TRANSITION_KEYS:
            if key in merged and merged[key].shape[0] == transition_count:
                merged[key] = merged[key][transition_order]

    candidate_count = int(np.asarray(merged.get("episode_id", [])).shape[0])
    if candidate_count:
        candidate_order = np.lexsort(
            (
                np.asarray(merged["actions"], dtype=np.int64),
                np.asarray(merged["candidate_episode_step"], dtype=np.int64),
                np.asarray(merged["episode_id"], dtype=np.int64),
            )
        )
        for key in CANDIDATE_KEYS:
            if key in merged and merged[key].shape[0] == candidate_count:
                merged[key] = merged[key][candidate_order]
        transition_lookup = {
            (int(episode), int(step)): index
            for index, (episode, step) in enumerate(
                zip(merged["transition_episode_id"], merged["transition_episode_step"])
            )
        }
        merged["candidate_transition_id"] = np.asarray(
            [
                transition_lookup[(int(episode), int(step))]
                for episode, step in zip(merged["episode_id"], merged["candidate_episode_step"])
            ],
            dtype=np.int64,
        )

    trajectory_count = int(np.asarray(merged.get("trajectory_episode_id", [])).shape[0])
    if trajectory_count:
        trajectory_order = np.lexsort(
            (
                np.asarray(merged["trajectory_window_end_step"], dtype=np.int64),
                np.asarray(merged["trajectory_episode_id"], dtype=np.int64),
            )
        )
        for key in TRAJECTORY_KEYS:
            if key in merged and merged[key].shape[0] == trajectory_count:
                merged[key] = merged[key][trajectory_order]
    np.savez_compressed(output, **merged)
    return merged


def _aggregate_worker_performance(
    worker_reports: list[dict],
    parent_summary: dict,
    *,
    episodes: int,
    transition_count: int,
) -> dict:
    worker_performance = [dict(item.get("performance", {}) or {}) for item in worker_reports]
    result = dict(parent_summary)
    operation_counts: dict[str, int] = {}
    timing_keys = (
        "sumo_start_or_load_time",
        "simulation_step_time",
        "state_collection_time",
        "candidate_rollout_time",
        "risk_forward_time",
        "forecast_inference_time",
        "replay_io_time",
    )
    for key in timing_keys:
        result[key] = float(sum(float(item.get(key, 0.0)) for item in worker_performance))
    for item in worker_performance:
        for key, value in dict(item.get("operation_counts", {}) or {}).items():
            operation_counts[key] = operation_counts.get(key, 0) + int(value)
    result["operation_counts"] = operation_counts
    wall_time = float(result.get("wall_time", 0.0))
    result["steps_per_second"] = float(transition_count / wall_time) if wall_time > 0.0 else 0.0
    result["episodes_per_hour"] = float(episodes * 3600.0 / wall_time) if wall_time > 0.0 else 0.0
    result["worker_count"] = len(worker_performance)
    result["worker_cpu_time_is_aggregate"] = True
    return result


def _run_parallel(cfg, workers: int) -> Path:
    stage_dir = prepare_run_dir(cfg, "stage1")
    tracker = PerformanceTracker()
    episode_ids = list(range(int(cfg.stage1.episodes)))
    worker_count = min(max(1, int(workers)), max(1, len(episode_ids)))
    shard_size = max(1, int(cfg.stage1.get("shard_episodes", 25)))
    assignments = [episode_ids[rank::worker_count] for rank in range(worker_count)]
    tasks: list[tuple[int, int, list[int]]] = []
    for worker_id, assigned in enumerate(assignments):
        for shard_index, start in enumerate(range(0, len(assigned), shard_size)):
            tasks.append((worker_id, shard_index, assigned[start : start + shard_size]))
    worker_root = stage_dir / "_worker_runs"
    worker_root.mkdir(parents=True, exist_ok=True)
    stage_log(
        "stage1",
        f"parallel workers={worker_count} shards={len(tasks)} shard_episodes={shard_size}",
    )
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=worker_count, mp_context=context) as executor:
        futures = [
            executor.submit(
                _stage1_worker_entry,
                cfg,
                worker_id,
                shard_index,
                shard_episodes,
                str(worker_root),
            )
            for worker_id, shard_index, shard_episodes in tasks
        ]
        shard_paths = [Path(future.result()) for future in futures]
    shard_paths.sort(key=lambda path: str(path))
    output = stage_dir / str(cfg.stage1.output_name)
    merged = _concatenate_shards(shard_paths, output)

    reports: list[dict] = []
    events: list[dict] = []
    replay_dir = stage_dir / "replay"
    replay_dir.mkdir(parents=True, exist_ok=True)
    worker_reports: list[dict] = []
    for shard_path in shard_paths:
        shard_stage = shard_path.parent
        episode_report_path = shard_stage / "stage1_episode_reports.json"
        if episode_report_path.exists():
            with episode_report_path.open("r", encoding="utf-8") as file:
                reports.extend(json.load(file).get("episodes", []))
        worker_report_path = shard_stage / "stage1_report.json"
        if worker_report_path.exists():
            with worker_report_path.open("r", encoding="utf-8") as file:
                worker_reports.append(json.load(file))
        event_path = shard_stage / "risk_events.jsonl"
        if event_path.exists():
            with event_path.open("r", encoding="utf-8") as file:
                events.extend(json.loads(line) for line in file if line.strip())
        for replay in (shard_stage / "replay").glob("*.json"):
            shutil.copy2(replay, replay_dir / replay.name)
    reports.sort(key=lambda item: int(item.get("seed", 0)))
    events.sort(key=lambda item: (int(item.get("episode", 0)), int(item.get("step", 0)), int(item.get("action", 0))))
    events_path = stage_dir / "risk_events.jsonl"
    with events_path.open("w", encoding="utf-8") as file:
        for event in events:
            file.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")

    audit_report = audit_stage1_buffer(output, stage_dir / "audit") if bool(cfg.stage1.get("audit_enabled", True)) else None
    candidate_legal = np.asarray(merged.get("candidate_legal", []), dtype=np.float32) > 0.5
    continuous = np.asarray(merged.get("continuous_risk_target", []), dtype=np.float32)
    legal_continuous = continuous[candidate_legal] if continuous.size else continuous
    coverage = _continuous_risk_coverage(legal_continuous)
    legal_boundary_count = int(np.sum((legal_continuous >= 0.20) & (legal_continuous < 0.80)))
    audit_gate = _stage1_audit_gate(cfg, int(continuous.size), legal_boundary_count, coverage)
    transition_count = int(np.asarray(merged.get("executed_actions", [])).shape[0])
    parent_performance = tracker.summary(episodes=len(episode_ids))
    performance = _aggregate_worker_performance(
        worker_reports,
        parent_performance,
        episodes=len(episode_ids),
        transition_count=transition_count,
    )
    report = {
        "stage": "stage1",
        "run_id": cfg.run.run_id,
        "sumo_installation": {
            "binary": str(cfg.scenario.get("sumo_binary", "")),
            "version": str(cfg.scenario.get("sumo_version", "")),
            "home": str(cfg.scenario.get("sumo_home", "")),
        },
        "parallel_workers": int(worker_count),
        "shard_count": int(len(shard_paths)),
        "shards": [str(path) for path in shard_paths],
        "buffer": str(output),
        "events": str(events_path),
        "replay_dir": str(replay_dir),
        "audit": str(stage_dir / "audit" / "stage1_data_audit.json") if audit_report else None,
        "transition_count": transition_count,
        "candidate_risk_sample_count": int(np.asarray(merged.get("actions", [])).shape[0]),
        "trajectory_sample_count": int(np.asarray(merged.get("agent_history", [])).shape[0]),
        "continuous_risk": {**coverage, "legal_summary": _array_summary(legal_continuous.tolist())},
        "audit_gate": audit_gate,
        "metrics": aggregate_episode_reports(reports),
        "performance": performance,
        "worker_performance": [item.get("performance", {}) for item in worker_reports],
    }
    tb = TensorboardLogger(stage_dir / "tensorboard", enabled=bool(cfg.run.get("tensorboard", True)))
    for episode_report in reports:
        episode_index = int(episode_report.get("seed", int(cfg.run.seed))) - int(cfg.run.seed)
        tb.scalar("stage1/episode_reward", float(episode_report.get("episode_reward", 0.0)), episode_index)
        tb.scalar("stage1/collision", float(episode_report.get("collision", False)), episode_index)
        tb.scalar("stage1/near_miss", float(episode_report.get("near_miss", False)), episode_index)
        tb.scalar("stage1/min_distance", float(episode_report.get("min_distance", 0.0)), episode_index)
    tb.close()
    write_report(stage_dir / "stage1_episode_reports.json", {"episodes": reports})
    write_report(stage_dir / "stage1_report.json", report)
    if not bool(audit_gate.get("passed", True)):
        raise RuntimeError(f"Stage1 audit gate failed: {audit_gate}")
    return output


def run(cfg) -> Path:
    cfg.shield["forecast_task_shadow_enabled"] = False
    cfg.shield["task_backstop_enabled"] = False
    workers = max(1, int(cfg.stage1.get("workers", 1)))
    if workers > 1 and not bool(cfg.stage1.get("_worker_mode", False)):
        return _run_parallel(cfg, workers)
    return _run_serial(cfg)


def _run_serial(cfg) -> Path:
    stage_dir = prepare_run_dir(cfg, "stage1")
    stage_log("stage1", f"run_id={cfg.run.run_id}")
    stage_log("stage1", f"SUMO config={cfg.scenario.sumocfg}")
    stage_log("stage1", f"SUMO binary={cfg.scenario.sumo_binary}, episodes={cfg.stage1.episodes}")
    stage_log("stage1", f"output_dir={stage_dir}")
    tb = TensorboardLogger(stage_dir / "tensorboard", enabled=bool(cfg.run.get("tensorboard", True)))
    tracker = PerformanceTracker()
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
        "candidate_episode_step": [],
        "candidate_raw_action": [],
        "transition_episode_id": [],
        "transition_episode_step": [],
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
    history_valid_masks: list[np.ndarray] = []
    future_valid_masks: list[np.ndarray] = []
    history_lane_indices: list[np.ndarray] = []
    history_edge_roles: list[np.ndarray] = []
    future_lane_indices: list[np.ndarray] = []
    future_edge_roles: list[np.ndarray] = []
    agent_lengths: list[np.ndarray] = []
    agent_widths: list[np.ndarray] = []
    relevance_masks: list[np.ndarray] = []
    relevance_scores: list[np.ndarray] = []
    selector_relevant_counts: list[np.ndarray] = []
    selector_overflows: list[np.ndarray] = []
    trajectory_episode_ids: list[np.ndarray] = []
    trajectory_window_end_steps: list[np.ndarray] = []
    reports: list[dict] = []
    events_path = stage_dir / "risk_events.jsonl"
    replay_dir = stage_dir / "replay"
    if events_path.exists():
        events_path.unlink()

    episode_ids = [
        int(item)
        for item in cfg.stage1.get("episode_ids", list(range(int(cfg.stage1.episodes))))
    ]
    env = make_env(cfg, seed=int(cfg.run.seed), shield_enabled=False, record_trajectory_samples=True)
    events_file = events_path.open("a", encoding="utf-8", buffering=1024 * 1024)
    try:
        for episode in progress_iter(episode_ids, desc="Stage1 episodes"):
            episode_seed = int(cfg.run.seed) + episode
            if episode % max(1, int(cfg.stage1.get("log_every_episodes", 20))) == 0:
                stage_log("stage1", f"episode={episode} seed={episode_seed} reset SUMO")
            rng = np.random.default_rng(np.random.SeedSequence([int(cfg.run.seed), int(episode)]))
            obs, _info = env.reset(seed=episode_seed)
            terminated = truncated = False
            episode_actions: list[int] = []
            episode_reward = 0.0
            episode_boundary_or_extreme = False
            while not (terminated or truncated):
                context = env.get_risk_context()
                action, sampling_mode = select_stage1_action(cfg, rng, context)
                candidate_samples = candidate_action_risk_samples(context)
                episode_boundary_or_extreme = episode_boundary_or_extreme or any(
                    float(sample.continuous_risk_target) >= 0.20 for sample in candidate_samples
                )
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
                transitions["transition_episode_step"].append(int(info.get("step", 0)))
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
                    transitions["candidate_episode_step"].append(int(info.get("step", 0)))
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
                        float(local.taper_miss),
                    ],
                    dtype=np.float32,
                )
                actual_overall = float(np.max(actual_risk_types))
                if actual_overall > 0 or executed_candidate_risk > 0:
                    events_file.write(
                        json.dumps(
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
                                "geometric_overlap": info.get("geometric_overlap"),
                                "closest_vehicle_id": info.get("closest_vehicle_id"),
                                "merge_gap": local.target_lane_gap,
                                "target_front_gap": local.target_front_gap,
                                "target_rear_gap": local.target_rear_gap,
                                "done_reason": info.get("done_reason"),
                                }
                            ),
                            ensure_ascii=False,
                            allow_nan=False,
                        )
                        + "\n"
                    )
                obs = next_obs
            episode_report = env.episode_report()
            episode_report["episode_reward"] = episode_reward
            episode_report["_stage1_boundary_or_extreme"] = bool(episode_boundary_or_extreme)
            reports.append(episode_report)
            tb.scalar("stage1/episode_reward", episode_reward, episode)
            tb.scalar("stage1/collision", float(episode_report.get("collision", False)), episode)
            tb.scalar("stage1/near_miss", float(episode_report.get("near_miss", False)), episode)
            tb.scalar("stage1/min_distance", float(episode_report.get("min_distance", 0.0)), episode)
            if _should_write_replay(cfg, episode_report):
                with tracker.measure("replay_io_time"):
                    write_replay_file(
                        replay_dir / f"episode_{episode:04d}.json",
                        run_id=str(cfg.run.run_id),
                        stage="stage1",
                        episode=episode,
                        seed=episode_seed,
                        actions=episode_actions,
                        shield_enabled=False,
                        safety_metric_version=SAFETY_METRIC_VERSION,
                        notes={"episode_report": episode_report},
                    )
            (
                hist,
                fut,
                mask,
                lane_indices,
                edge_roles,
                history_valid_mask,
                future_valid_mask,
                sample_history_lane_indices,
                sample_history_edge_roles,
                sample_future_lane_indices,
                sample_future_edge_roles,
                sample_agent_lengths,
                sample_agent_widths,
                sample_relevance_mask,
                sample_relevance_score,
                sample_selector_relevant_count,
                sample_selector_overflow,
            ) = env.trajectory_window_samples(include_dimensions=True)
            if hist.shape[0] > 0:
                history_samples.append(hist)
                future_samples.append(fut)
                agent_masks.append(mask)
                agent_lane_indices.append(lane_indices)
                agent_edge_roles.append(edge_roles)
                history_valid_masks.append(history_valid_mask)
                future_valid_masks.append(future_valid_mask)
                history_lane_indices.append(sample_history_lane_indices)
                history_edge_roles.append(sample_history_edge_roles)
                future_lane_indices.append(sample_future_lane_indices)
                future_edge_roles.append(sample_future_edge_roles)
                agent_lengths.append(sample_agent_lengths)
                agent_widths.append(sample_agent_widths)
                relevance_masks.append(sample_relevance_mask)
                relevance_scores.append(sample_relevance_score)
                selector_relevant_counts.append(sample_selector_relevant_count)
                selector_overflows.append(sample_selector_overflow)
                sample_count = int(hist.shape[0])
                trajectory_episode_ids.append(np.full((sample_count,), episode, dtype=np.int64))
                trajectory_window_end_steps.append(
                    np.arange(sample_count, dtype=np.int64) + int(cfg.scenario.history_steps)
                )
    finally:
        events_file.close()
        env.close()

    output = stage_dir / str(cfg.stage1.output_name)
    save_npz = np.savez if bool(cfg.stage1.get("_worker_mode", False)) else np.savez_compressed
    save_npz(
        output,
        **{key: np.asarray(value) for key, value in transitions.items()},
        agent_history=np.concatenate(history_samples, axis=0) if history_samples else np.zeros((0, 1, 1, 5)),
        agent_future=np.concatenate(future_samples, axis=0) if future_samples else np.zeros((0, 1, 1, 5)),
        agent_mask=np.concatenate(agent_masks, axis=0) if agent_masks else np.zeros((0, 1)),
        agent_lane_index=np.concatenate(agent_lane_indices, axis=0) if agent_lane_indices else np.zeros((0, 1), dtype=np.int64),
        agent_edge_role=np.concatenate(agent_edge_roles, axis=0) if agent_edge_roles else np.zeros((0, 1), dtype=np.int64),
        trajectory_schema_version=np.asarray(4, dtype=np.int64),
        safety_metric_version=np.asarray(SAFETY_METRIC_VERSION),
        actor_selection_version=np.asarray(ACTOR_SELECTION_VERSION),
        actor_selection_config_hash=np.asarray(actor_selection_config_hash(cfg)),
        agent_length=np.concatenate(agent_lengths, axis=0) if agent_lengths else np.full((0, 1), 4.8),
        agent_width=np.concatenate(agent_widths, axis=0) if agent_widths else np.full((0, 1), 1.8),
        agent_history_valid_mask=(
            np.concatenate(history_valid_masks, axis=0) if history_valid_masks else np.zeros((0, 1, 1))
        ),
        agent_future_valid_mask=(
            np.concatenate(future_valid_masks, axis=0) if future_valid_masks else np.zeros((0, 1, 1))
        ),
        agent_history_lane_index=(
            np.concatenate(history_lane_indices, axis=0)
            if history_lane_indices
            else np.full((0, 1, 1), -1, dtype=np.int64)
        ),
        agent_history_edge_role=(
            np.concatenate(history_edge_roles, axis=0)
            if history_edge_roles
            else np.zeros((0, 1, 1), dtype=np.int64)
        ),
        agent_future_lane_index=(
            np.concatenate(future_lane_indices, axis=0)
            if future_lane_indices
            else np.full((0, 1, 1), -1, dtype=np.int64)
        ),
        agent_future_edge_role=(
            np.concatenate(future_edge_roles, axis=0)
            if future_edge_roles
            else np.zeros((0, 1, 1), dtype=np.int64)
        ),
        agent_relevance_mask=(
            np.concatenate(relevance_masks, axis=0)
            if relevance_masks
            else np.zeros((0, 1), dtype=np.float32)
        ),
        agent_relevance_score=(
            np.concatenate(relevance_scores, axis=0)
            if relevance_scores
            else np.zeros((0, 1), dtype=np.float32)
        ),
        actor_selector_relevant_count=(
            np.concatenate(selector_relevant_counts, axis=0)
            if selector_relevant_counts
            else np.zeros((0,), dtype=np.int64)
        ),
        actor_selector_overflow=(
            np.concatenate(selector_overflows, axis=0)
            if selector_overflows
            else np.zeros((0,), dtype=np.float32)
        ),
        trajectory_episode_id=(
            np.concatenate(trajectory_episode_ids, axis=0)
            if trajectory_episode_ids
            else np.zeros((0,), dtype=np.int64)
        ),
        trajectory_window_end_step=(
            np.concatenate(trajectory_window_end_steps, axis=0)
            if trajectory_window_end_steps
            else np.zeros((0,), dtype=np.int64)
        ),
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
    trajectory_coverage = _trajectory_coverage_summary(
        np.concatenate(agent_masks, axis=0) if agent_masks else np.zeros((0, 1)),
        np.concatenate(history_valid_masks, axis=0) if history_valid_masks else np.zeros((0, 1, 1)),
        np.concatenate(future_valid_masks, axis=0) if future_valid_masks else np.zeros((0, 1, 1)),
    )
    report = {
        "stage": "stage1",
        "run_id": cfg.run.run_id,
        "sumo_installation": {
            "binary": str(cfg.scenario.get("sumo_binary", "")),
            "version": str(cfg.scenario.get("sumo_version", "")),
            "home": str(cfg.scenario.get("sumo_home", "")),
        },
        "buffer": str(output),
        "events": str(events_path),
        "replay_dir": str(replay_dir),
        "audit": str(stage_dir / "audit" / "stage1_data_audit.json") if audit_report else None,
        "tensorboard": str(stage_dir / "tensorboard"),
        "transition_count": len(transitions["executed_actions"]),
        "candidate_risk_sample_count": len(transitions["actions"]),
        "trajectory_sample_count": int(sum(item.shape[0] for item in history_samples)),
        "trajectory_schema": {
            "version": 4,
            "safety_metric_version": SAFETY_METRIC_VERSION,
            "actor_selection_version": ACTOR_SELECTION_VERSION,
            "actor_selection_config_hash": actor_selection_config_hash(cfg),
            "actor_selector_overflow_rate": (
                float(np.mean(np.concatenate(selector_overflows, axis=0)))
                if selector_overflows
                else 0.0
            ),
            **trajectory_coverage,
        },
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
        "performance": {
            **tracker.summary(episodes=len(episode_ids)),
            **env.performance.summary(
                steps=int(sum(int(item.get("steps", 0)) for item in reports)),
                episodes=len(episode_ids),
            ),
        },
    }
    write_report(stage_dir / "stage1_episode_reports.json", {"episodes": reports})
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


def _trajectory_coverage_summary(
    actor_mask: np.ndarray,
    history_valid_mask: np.ndarray,
    future_valid_mask: np.ndarray,
) -> dict:
    selected = np.asarray(actor_mask, dtype=np.float32) > 0.5
    history_valid = np.asarray(history_valid_mask, dtype=np.float32) > 0.5
    future_valid = np.asarray(future_valid_mask, dtype=np.float32) > 0.5
    if not np.any(selected):
        return {
            "selected_actor_count": 0,
            "history_valid_rate": 0.0,
            "future_valid_rate": 0.0,
            "departed_actor_rate": 0.0,
            "valid_future_horizon": _array_summary([]),
        }
    selected_history = history_valid[selected]
    selected_future = future_valid[selected]
    future_horizon = np.sum(selected_future, axis=-1).astype(np.float32)
    return {
        "selected_actor_count": int(np.sum(selected)),
        "history_valid_rate": float(np.mean(selected_history)),
        "future_valid_rate": float(np.mean(selected_future)),
        "departed_actor_rate": float(np.mean(future_horizon < future_valid.shape[-1])),
        "valid_future_horizon": _array_summary([float(item) for item in future_horizon.tolist()]),
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
