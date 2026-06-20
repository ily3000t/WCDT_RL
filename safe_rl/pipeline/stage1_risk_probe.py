from __future__ import annotations

import json
import multiprocessing as mp
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from safe_rl.analysis.stage1_audit import audit_stage1_buffer
from safe_rl.prediction.actor_selector import (
    ACTOR_SELECTION_VERSION,
    actor_selection_config_hash,
)
from safe_rl.prediction.forecast_rollout_bundle import FORECAST_ROLLOUT_BUNDLE_VERSION
from safe_rl.prediction.trajectory_postprocess import TRAJECTORY_POSTPROCESS_VERSION
from safe_rl.pipeline.common import json_ready, load_stage_config, make_env, parse_config_arg, write_report
from safe_rl.risk.merge_local import candidate_action_risk_samples, candidate_sample_weight, merge_local_stats
from safe_rl.risk.risk_aggregator import aggregate_episode_reports
from safe_rl.risk.stage1_sampling import configured_sampling_probs, sampling_summary, select_stage1_action
from safe_rl.sim.metrics import SAFETY_METRIC_VERSION
from safe_rl.sim.sumo_highway_merge_env import configured_trajectory_actor_capacity
from safe_rl.utils.config import clone_with_overrides, prepare_run_dir
from safe_rl.utils.progress import TensorboardLogger, progress_iter, stage_log
from safe_rl.utils.replay import write_replay_file
from safe_rl.utils.performance import PerformanceTracker
from safe_rl.utils.stage1_dataset import (
    STAGE1_BUFFER_SCHEMA_VERSION,
    STAGE1_FORMAT_VERSION,
    merge_stage1_shards,
    open_stage1_dataset,
    write_stage1_dataset,
)


TRANSITION_KEYS = {
    "observations",
    "executed_actions",
    "next_observations",
    "rewards",
    "dones",
    "transition_episode_id",
    "transition_episode_step",
    "transition_episode_seed",
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
    "candidate_episode_seed",
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
    "critical_actor_count",
    "contextual_actor_count",
    "critical_actor_overflow",
    "contextual_actor_truncated_count",
    "critical_actor_metadata_json",
    "dropped_critical_actor_metadata_json",
    "trajectory_agent_vehicle_id_index",
    "trajectory_selector_selected_count",
    "trajectory_episode_id",
    "trajectory_window_end_step",
    "trajectory_decision_index",
    "trajectory_episode_seed",
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


def _stage1_dataset_metadata(cfg) -> dict:
    trajectory_actor_capacity = configured_trajectory_actor_capacity(cfg)
    return {
        "trajectory_schema_version": 4,
        "trajectory_actor_capacity": trajectory_actor_capacity,
        "trajectory_max_agent_count": trajectory_actor_capacity + 1,
        "wcdt_v1_max_agents": int(cfg.prediction.get("wcdt_v1_max_agents", 0)),
        "wcdt_v2_max_agents": int(cfg.prediction.get("wcdt_v2_max_agents", 0)),
        "wcdt_v3_max_agents": int(cfg.prediction.get("wcdt_v3_max_agents", 0)),
        "episode_seed_schedule": str(
            cfg.get("run", {}).get("episode_seed_schedule", "fixed_legacy")
        ),
        "vehicle_state_ordering_version": str(
            cfg.get("scenario", {}).get(
                "vehicle_state_ordering_version",
                "unspecified_legacy",
            )
        ),
        "safety_metric_version": SAFETY_METRIC_VERSION,
        "actor_selection_version": ACTOR_SELECTION_VERSION,
        "actor_selection_config_hash": actor_selection_config_hash(cfg),
        "trajectory_postprocess_version": TRAJECTORY_POSTPROCESS_VERSION,
        "forecast_rollout_bundle_version": FORECAST_ROLLOUT_BUNDLE_VERSION,
        "trajectory_actor_row_alignment": "selector_v2_vehicle_id_verified",
        "trajectory_selector_order_version": "selector_v2_vehicle_id_order_v1",
        "route_projection_config": dict(cfg.prediction.get("route_projection", {}) or {}),
        "sumo_installation_fingerprint": dict(
            cfg.get("scenario", {}).get("sumo_installation_fingerprint", {}) or {}
        ),
    }


def _encode_trajectory_vehicle_ids(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Encode fixed-width row IDs as a compact, shard-mergeable ID table."""

    values = np.asarray(rows, dtype=str)
    non_empty = sorted({str(value) for value in values.reshape(-1) if str(value)})
    width = max(1, max((len(value) for value in non_empty), default=1))
    table = np.asarray(non_empty, dtype=f"<U{width}")
    lookup = {value: index for index, value in enumerate(non_empty)}
    indices = np.full(values.shape, -1, dtype=np.int32)
    for index in np.ndindex(values.shape):
        vehicle_id = str(values[index])
        if vehicle_id:
            indices[index] = int(lookup[vehicle_id])
    return table, indices


def _stage1_worker_entry(
    cfg,
    worker_id: int,
    episode_ids: list[int],
    shard_size: int,
    worker_root: str,
    worker_count: int,
) -> dict:
    started = time.perf_counter()
    env = make_env(
        cfg,
        seed=int(cfg.run.seed),
        shield_enabled=False,
        record_trajectory_samples=True,
        worker_rank=int(worker_id),
        num_envs=int(worker_count),
        advance_episode_seed=False,
    )
    shard_paths: list[str] = []
    shard_reports: list[str] = []
    episode_count = 0
    control_decisions = 0
    simulation_steps = 0
    try:
        for shard_index, start in enumerate(range(0, len(episode_ids), shard_size)):
            shard_episode_ids = episode_ids[start : start + shard_size]
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
                        "episode_ids": [int(item) for item in shard_episode_ids],
                        "episodes": len(shard_episode_ids),
                        "audit_enabled": False,
                        "audit_gate": {"enabled": False},
                    },
                },
            )
            shard_path = _run_serial(worker_cfg, env=env, close_env=False)
            shard_paths.append(str(shard_path))
            shard_report_path = shard_path.parent / "stage1_report.json"
            shard_reports.append(str(shard_report_path))
            if shard_report_path.exists():
                with shard_report_path.open("r", encoding="utf-8") as file:
                    report = json.load(file)
                episode_count += len(shard_episode_ids)
                control_decisions += int(report.get("transition_count", 0))
                episode_report_path = shard_path.parent / "stage1_episode_reports.json"
                with episode_report_path.open("r", encoding="utf-8") as file:
                    episode_reports = json.load(file).get("episodes", [])
                simulation_steps += int(
                    sum(int(item.get("steps", 0)) for item in episode_reports)
                )
    finally:
        env.close()
    wall_time = time.perf_counter() - started
    return {
        "worker_id": int(worker_id),
        "shard_paths": shard_paths,
        "shard_reports": shard_reports,
        "wall_time": float(wall_time),
        "episode_count": int(episode_count),
        "control_decisions": int(control_decisions),
        "simulation_steps": int(simulation_steps),
        "performance": env.performance.summary(
            steps=simulation_steps,
            episodes=episode_count,
        ),
    }


def _concatenate_shards(shard_paths: list[Path], output: Path, cfg=None):
    metadata = _stage1_dataset_metadata(cfg) if cfg is not None else {
        "trajectory_schema_version": 4,
        "episode_seed_schedule": "fixed_legacy",
        "vehicle_state_ordering_version": "unspecified_legacy",
    }
    merge_stage1_shards(
        shard_paths,
        output,
        transition_keys=TRANSITION_KEYS,
        candidate_keys=CANDIDATE_KEYS,
        trajectory_keys=TRAJECTORY_KEYS,
        metadata=metadata,
    )
    return open_stage1_dataset(output)


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
        summed = float(
            sum(float(item.get(key, 0.0)) for item in worker_performance)
        )
        result[f"worker_{key}_sum"] = summed
        result[key] = summed
    for item in worker_performance:
        for key, value in dict(item.get("operation_counts", {}) or {}).items():
            operation_counts[key] = operation_counts.get(key, 0) + int(value)
    result["operation_counts"] = operation_counts
    wall_time = float(result.get("wall_time", 0.0))
    worker_times = [float(item.get("wall_time", 0.0)) for item in worker_reports]
    worker_time_sum = float(sum(worker_times))
    worker_max_time = float(max(worker_times, default=0.0))
    simulation_steps = int(sum(int(item.get("simulation_steps", 0)) for item in worker_reports))
    result["worker_time_sum"] = worker_time_sum
    result["worker_max_time"] = worker_max_time
    result["aggregate_control_decisions_per_second"] = (
        float(transition_count / wall_time) if wall_time > 0.0 else 0.0
    )
    result["aggregate_simulation_steps_per_second"] = (
        float(simulation_steps / wall_time) if wall_time > 0.0 else 0.0
    )
    result["per_worker_control_decisions_per_second"] = [
        (
            float(item.get("control_decisions", 0)) / float(item.get("wall_time", 1.0))
            if float(item.get("wall_time", 0.0)) > 0.0
            else 0.0
        )
        for item in worker_reports
    ]
    result["per_worker_episodes_per_hour"] = [
        (
            float(item.get("episode_count", 0)) * 3600.0 / float(item.get("wall_time", 1.0))
            if float(item.get("wall_time", 0.0)) > 0.0
            else 0.0
        )
        for item in worker_reports
    ]
    result["steps_per_second"] = result["aggregate_control_decisions_per_second"]
    result["episodes_per_hour"] = (
        float(episodes * 3600.0 / wall_time) if wall_time > 0.0 else 0.0
    )
    result["worker_count"] = len(worker_performance)
    result["worker_hotpath_time_denominator"] = "worker_time_sum"
    return result


def _run_parallel(cfg, workers: int) -> Path:
    stage_dir = prepare_run_dir(cfg, "stage1")
    tracker = PerformanceTracker()
    episode_ids = list(range(int(cfg.stage1.episodes)))
    worker_count = min(max(1, int(workers)), max(1, len(episode_ids)))
    shard_size = max(1, int(cfg.stage1.get("shard_episodes", 25)))
    assignments = [episode_ids[rank::worker_count] for rank in range(worker_count)]
    worker_root = stage_dir / "_worker_runs"
    worker_root.mkdir(parents=True, exist_ok=True)
    stage_log(
        "stage1",
        f"parallel workers={worker_count} shard_episodes={shard_size}",
    )
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=worker_count, mp_context=context) as executor:
        futures = [
            executor.submit(
                _stage1_worker_entry,
                cfg,
                worker_id,
                assigned,
                shard_size,
                str(worker_root),
                worker_count,
            )
            for worker_id, assigned in enumerate(assignments)
        ]
        worker_results = [future.result() for future in futures]
        shard_paths = [
            Path(path)
            for result in worker_results
            for path in result.get("shard_paths", [])
        ]
    shard_paths.sort(key=lambda path: str(path))
    output = stage_dir / str(cfg.stage1.output_name)
    merge_started = time.perf_counter()
    merged = _concatenate_shards(shard_paths, output, cfg)
    merge_io_time = time.perf_counter() - merge_started

    reports: list[dict] = []
    events: list[dict] = []
    replay_dir = stage_dir / "replay"
    replay_dir.mkdir(parents=True, exist_ok=True)
    for shard_path in shard_paths:
        shard_stage = shard_path.parent
        episode_report_path = shard_stage / "stage1_episode_reports.json"
        if episode_report_path.exists():
            with episode_report_path.open("r", encoding="utf-8") as file:
                reports.extend(json.load(file).get("episodes", []))
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
        worker_results,
        parent_performance,
        episodes=len(episode_ids),
        transition_count=transition_count,
    )
    performance["merge_io_time"] = float(merge_io_time)
    trajectory_actor_capacity = configured_trajectory_actor_capacity(cfg)
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
        "worker_performance": worker_results,
        "stage1_buffer_schema_version": STAGE1_BUFFER_SCHEMA_VERSION,
        "trajectory_actor_capacity": trajectory_actor_capacity,
        "trajectory_max_agent_count": trajectory_actor_capacity + 1,
        "wcdt_v1_max_agents": int(cfg.prediction.get("wcdt_v1_max_agents", 0)),
        "wcdt_v2_max_agents": int(cfg.prediction.get("wcdt_v2_max_agents", 0)),
        "wcdt_v3_max_agents": int(cfg.prediction.get("wcdt_v3_max_agents", 0)),
        "episode_seed_schedule": str(
            cfg.get("run", {}).get("episode_seed_schedule", "fixed_legacy")
        ),
        "vehicle_state_ordering_version": str(
            cfg.get("scenario", {}).get(
                "vehicle_state_ordering_version",
                "unspecified_legacy",
            )
        ),
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
    merged.close()
    if bool(cfg.stage1.get("cleanup_worker_shards", True)):
        shutil.rmtree(worker_root, ignore_errors=True)
    if not bool(audit_gate.get("passed", True)):
        raise RuntimeError(f"Stage1 audit gate failed: {audit_gate}")
    return output


def run(cfg) -> Path:
    output_format = str(cfg.stage1.get("output_format", "")).strip()
    if output_format != STAGE1_FORMAT_VERSION:
        raise ValueError(
            "stage1.output_format must be "
            f"{STAGE1_FORMAT_VERSION!r}; received {output_format!r}"
        )
    cfg.shield["forecast_task_shadow_enabled"] = False
    cfg.shield["task_backstop_enabled"] = False
    cfg.shield["forecast_aware_candidate_ranking_mode"] = "off"
    workers = max(1, int(cfg.stage1.get("workers", 1)))
    if not bool(cfg.stage1.get("_worker_mode", False)):
        return _run_parallel(cfg, workers)
    return _run_serial(cfg)


def _run_serial(
    cfg,
    *,
    env=None,
    close_env: bool = True,
) -> Path:
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
        "transition_episode_seed": [],
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
        "candidate_episode_seed": [],
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
    critical_actor_counts: list[np.ndarray] = []
    contextual_actor_counts: list[np.ndarray] = []
    critical_actor_overflows: list[np.ndarray] = []
    contextual_actor_truncated_counts: list[np.ndarray] = []
    critical_actor_metadata_json: list[np.ndarray] = []
    dropped_critical_actor_metadata_json: list[np.ndarray] = []
    trajectory_agent_vehicle_ids: list[np.ndarray] = []
    trajectory_selector_selected_counts: list[np.ndarray] = []
    trajectory_episode_ids: list[np.ndarray] = []
    trajectory_window_end_steps: list[np.ndarray] = []
    trajectory_decision_indices: list[np.ndarray] = []
    trajectory_episode_seeds: list[np.ndarray] = []
    reports: list[dict] = []
    events_path = stage_dir / "risk_events.jsonl"
    replay_dir = stage_dir / "replay"
    if events_path.exists():
        events_path.unlink()

    episode_ids = [
        int(item)
        for item in cfg.stage1.get("episode_ids", list(range(int(cfg.stage1.episodes))))
    ]
    owns_env = env is None
    if env is None:
        env = make_env(
            cfg,
            seed=int(cfg.run.seed),
            shield_enabled=False,
            record_trajectory_samples=True,
        )
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
                transitions["transition_episode_seed"].append(episode_seed)
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
                    transitions["candidate_episode_seed"].append(episode_seed)
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
            episode_report["episode_index"] = int(episode)
            episode_report["episode_seed"] = int(episode_seed)
            episode_report["episode_seed_schedule"] = str(
                cfg.get("run", {}).get("episode_seed_schedule", "fixed_legacy")
            )
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
            trajectory_metadata = env.trajectory_window_metadata()
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
                critical_actor_counts.append(
                    np.asarray(
                        trajectory_metadata.get("critical_actor_count"),
                        dtype=np.int64,
                    )
                )
                contextual_actor_counts.append(
                    np.asarray(
                        trajectory_metadata.get("contextual_actor_count"),
                        dtype=np.int64,
                    )
                )
                critical_actor_overflows.append(
                    np.asarray(
                        trajectory_metadata.get("critical_actor_overflow"),
                        dtype=np.float32,
                    )
                )
                contextual_actor_truncated_counts.append(
                    np.asarray(
                        trajectory_metadata.get("contextual_actor_truncated_count"),
                        dtype=np.int64,
                    )
                )
                critical_actor_metadata_json.append(
                    np.asarray(
                        trajectory_metadata.get("critical_actor_metadata_json"),
                    )
                )
                dropped_critical_actor_metadata_json.append(
                    np.asarray(
                        trajectory_metadata.get("dropped_critical_actor_metadata_json"),
                    )
                )
                trajectory_agent_vehicle_ids.append(
                    np.asarray(trajectory_metadata.get("trajectory_agent_vehicle_ids"), dtype=str)
                )
                trajectory_selector_selected_counts.append(
                    np.asarray(
                        trajectory_metadata.get("trajectory_selector_selected_count"),
                        dtype=np.int64,
                    )
                )
                sample_count = int(hist.shape[0])
                trajectory_episode_ids.append(np.full((sample_count,), episode, dtype=np.int64))
                trajectory_window_end_steps.append(
                    np.asarray(
                        trajectory_metadata.get("trajectory_window_end_step"),
                        dtype=np.int64,
                    )
                )
                trajectory_decision_indices.append(
                    np.asarray(
                        trajectory_metadata.get("trajectory_decision_index"),
                        dtype=np.int64,
                    )
                )
                trajectory_episode_seeds.append(
                    np.asarray(
                        trajectory_metadata.get("trajectory_episode_seed"),
                        dtype=np.int64,
                    )
                )
    finally:
        events_file.close()
        if owns_env or close_env:
            env.close()

    output = stage_dir / str(cfg.stage1.output_name)
    raw_trajectory_agent_vehicle_ids = (
        np.concatenate(trajectory_agent_vehicle_ids, axis=0)
        if trajectory_agent_vehicle_ids
        else np.zeros((0, configured_trajectory_actor_capacity(cfg) + 1), dtype="<U1")
    )
    trajectory_vehicle_id_table, trajectory_agent_vehicle_id_index = _encode_trajectory_vehicle_ids(
        raw_trajectory_agent_vehicle_ids
    )
    output_arrays = {
        **{key: np.asarray(value) for key, value in transitions.items()},
        "agent_history": (
            np.concatenate(history_samples, axis=0)
            if history_samples
            else np.zeros((0, 1, 1, 5), dtype=np.float32)
        ),
        "agent_future": (
            np.concatenate(future_samples, axis=0)
            if future_samples
            else np.zeros((0, 1, 1, 5), dtype=np.float32)
        ),
        "agent_mask": (
            np.concatenate(agent_masks, axis=0)
            if agent_masks
            else np.zeros((0, 1), dtype=np.float32)
        ),
        "agent_lane_index": (
            np.concatenate(agent_lane_indices, axis=0)
            if agent_lane_indices
            else np.zeros((0, 1), dtype=np.int64)
        ),
        "agent_edge_role": (
            np.concatenate(agent_edge_roles, axis=0)
            if agent_edge_roles
            else np.zeros((0, 1), dtype=np.int64)
        ),
        "stage1_buffer_schema_version": np.asarray(
            STAGE1_BUFFER_SCHEMA_VERSION,
            dtype=np.int64,
        ),
        "trajectory_schema_version": np.asarray(4, dtype=np.int64),
        "trajectory_actor_capacity": np.asarray(
            configured_trajectory_actor_capacity(cfg),
            dtype=np.int64,
        ),
        "trajectory_max_agent_count": np.asarray(
            configured_trajectory_actor_capacity(cfg) + 1,
            dtype=np.int64,
        ),
        "wcdt_v1_max_agents": np.asarray(
            int(cfg.prediction.get("wcdt_v1_max_agents", 0)),
            dtype=np.int64,
        ),
        "wcdt_v2_max_agents": np.asarray(
            int(cfg.prediction.get("wcdt_v2_max_agents", 0)),
            dtype=np.int64,
        ),
        "wcdt_v3_max_agents": np.asarray(
            int(cfg.prediction.get("wcdt_v3_max_agents", 0)),
            dtype=np.int64,
        ),
        "safety_metric_version": np.asarray(SAFETY_METRIC_VERSION),
        "actor_selection_version": np.asarray(ACTOR_SELECTION_VERSION),
        "actor_selection_config_hash": np.asarray(actor_selection_config_hash(cfg)),
        "trajectory_postprocess_version": np.asarray(TRAJECTORY_POSTPROCESS_VERSION),
        "forecast_rollout_bundle_version": np.asarray(FORECAST_ROLLOUT_BUNDLE_VERSION),
        "trajectory_actor_row_alignment": np.asarray("selector_v2_vehicle_id_verified"),
        "trajectory_selector_order_version": np.asarray("selector_v2_vehicle_id_order_v1"),
        "trajectory_vehicle_id_table": trajectory_vehicle_id_table,
        "trajectory_agent_vehicle_id_index": trajectory_agent_vehicle_id_index,
        "trajectory_selector_selected_count": (
            np.concatenate(trajectory_selector_selected_counts, axis=0)
            if trajectory_selector_selected_counts
            else np.zeros((0,), dtype=np.int64)
        ),
        "episode_seed_schedule": np.asarray(
            str(cfg.get("run", {}).get("episode_seed_schedule", "fixed_legacy"))
        ),
        "vehicle_state_ordering_version": np.asarray(
            str(
                cfg.get("scenario", {}).get(
                    "vehicle_state_ordering_version",
                    "unspecified_legacy",
                )
            )
        ),
        "agent_length": (
            np.concatenate(agent_lengths, axis=0)
            if agent_lengths
            else np.full((0, 1), 4.8, dtype=np.float32)
        ),
        "agent_width": (
            np.concatenate(agent_widths, axis=0)
            if agent_widths
            else np.full((0, 1), 1.8, dtype=np.float32)
        ),
        "agent_history_valid_mask": (
            np.concatenate(history_valid_masks, axis=0)
            if history_valid_masks
            else np.zeros((0, 1, 1), dtype=np.float32)
        ),
        "agent_future_valid_mask": (
            np.concatenate(future_valid_masks, axis=0)
            if future_valid_masks
            else np.zeros((0, 1, 1), dtype=np.float32)
        ),
        "agent_history_lane_index": (
            np.concatenate(history_lane_indices, axis=0)
            if history_lane_indices
            else np.full((0, 1, 1), -1, dtype=np.int64)
        ),
        "agent_history_edge_role": (
            np.concatenate(history_edge_roles, axis=0)
            if history_edge_roles
            else np.zeros((0, 1, 1), dtype=np.int64)
        ),
        "agent_future_lane_index": (
            np.concatenate(future_lane_indices, axis=0)
            if future_lane_indices
            else np.full((0, 1, 1), -1, dtype=np.int64)
        ),
        "agent_future_edge_role": (
            np.concatenate(future_edge_roles, axis=0)
            if future_edge_roles
            else np.zeros((0, 1, 1), dtype=np.int64)
        ),
        "agent_relevance_mask": (
            np.concatenate(relevance_masks, axis=0)
            if relevance_masks
            else np.zeros((0, 1), dtype=np.float32)
        ),
        "agent_relevance_score": (
            np.concatenate(relevance_scores, axis=0)
            if relevance_scores
            else np.zeros((0, 1), dtype=np.float32)
        ),
        "actor_selector_relevant_count": (
            np.concatenate(selector_relevant_counts, axis=0)
            if selector_relevant_counts
            else np.zeros((0,), dtype=np.int64)
        ),
        "actor_selector_overflow": (
            np.concatenate(selector_overflows, axis=0)
            if selector_overflows
            else np.zeros((0,), dtype=np.float32)
        ),
        "critical_actor_count": (
            np.concatenate(critical_actor_counts, axis=0)
            if critical_actor_counts
            else np.zeros((0,), dtype=np.int64)
        ),
        "contextual_actor_count": (
            np.concatenate(contextual_actor_counts, axis=0)
            if contextual_actor_counts
            else np.zeros((0,), dtype=np.int64)
        ),
        "critical_actor_overflow": (
            np.concatenate(critical_actor_overflows, axis=0)
            if critical_actor_overflows
            else np.zeros((0,), dtype=np.float32)
        ),
        "contextual_actor_truncated_count": (
            np.concatenate(contextual_actor_truncated_counts, axis=0)
            if contextual_actor_truncated_counts
            else np.zeros((0,), dtype=np.int64)
        ),
        "critical_actor_metadata_json": (
            np.concatenate(critical_actor_metadata_json, axis=0)
            if critical_actor_metadata_json
            else np.zeros((0,), dtype="<U2")
        ),
        "dropped_critical_actor_metadata_json": (
            np.concatenate(dropped_critical_actor_metadata_json, axis=0)
            if dropped_critical_actor_metadata_json
            else np.zeros((0,), dtype="<U2")
        ),
        "trajectory_episode_id": (
            np.concatenate(trajectory_episode_ids, axis=0)
            if trajectory_episode_ids
            else np.zeros((0,), dtype=np.int64)
        ),
        "trajectory_window_end_step": (
            np.concatenate(trajectory_window_end_steps, axis=0)
            if trajectory_window_end_steps
            else np.zeros((0,), dtype=np.int64)
        ),
        "trajectory_decision_index": (
            np.concatenate(trajectory_decision_indices, axis=0)
            if trajectory_decision_indices
            else np.zeros((0,), dtype=np.int64)
        ),
        "trajectory_episode_seed": (
            np.concatenate(trajectory_episode_seeds, axis=0)
            if trajectory_episode_seeds
            else np.zeros((0,), dtype=np.int64)
        ),
    }
    write_stage1_dataset(
        output,
        output_arrays,
        metadata=_stage1_dataset_metadata(cfg),
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
        "stage1_buffer_schema_version": STAGE1_BUFFER_SCHEMA_VERSION,
        "trajectory_actor_capacity": configured_trajectory_actor_capacity(cfg),
        "trajectory_max_agent_count": configured_trajectory_actor_capacity(cfg) + 1,
        "wcdt_v1_max_agents": int(cfg.prediction.get("wcdt_v1_max_agents", 0)),
        "wcdt_v2_max_agents": int(cfg.prediction.get("wcdt_v2_max_agents", 0)),
        "wcdt_v3_max_agents": int(cfg.prediction.get("wcdt_v3_max_agents", 0)),
        "episode_seed_schedule": str(
            cfg.get("run", {}).get("episode_seed_schedule", "fixed_legacy")
        ),
        "vehicle_state_ordering_version": str(
            cfg.get("scenario", {}).get(
                "vehicle_state_ordering_version",
                "unspecified_legacy",
            )
        ),
        "trajectory_schema": {
            "version": 4,
            "safety_metric_version": SAFETY_METRIC_VERSION,
            "actor_selection_version": ACTOR_SELECTION_VERSION,
            "actor_selection_config_hash": actor_selection_config_hash(cfg),
            "trajectory_postprocess_version": TRAJECTORY_POSTPROCESS_VERSION,
            "forecast_rollout_bundle_version": FORECAST_ROLLOUT_BUNDLE_VERSION,
            "actor_selector_overflow_rate": (
                float(np.mean(np.concatenate(selector_overflows, axis=0)))
                if selector_overflows
                else 0.0
            ),
            "critical_actor_overflow_rate": (
                float(np.mean(np.concatenate(critical_actor_overflows, axis=0)))
                if critical_actor_overflows
                else 0.0
            ),
            "critical_wcdt_coverage": (
                float(
                    np.mean(
                        np.concatenate(critical_actor_overflows, axis=0)
                        <= 0
                    )
                )
                if critical_actor_overflows
                else 0.0
            ),
            "critical_actor_count": _array_summary(
                np.concatenate(critical_actor_counts, axis=0).astype(np.float32).tolist()
                if critical_actor_counts
                else []
            ),
            "contextual_actor_count": _array_summary(
                np.concatenate(contextual_actor_counts, axis=0).astype(np.float32).tolist()
                if contextual_actor_counts
                else []
            ),
            "contextual_actor_truncated_count": int(
                np.sum(np.concatenate(contextual_actor_truncated_counts, axis=0))
            )
            if contextual_actor_truncated_counts
            else 0,
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
