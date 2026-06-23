from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any

import numpy as np

from safe_rl.accvp.candidate_plan import ACCVP_COMMITMENT_PROFILE, apply_branch_command
from safe_rl.accvp.root_context import load_root_context, restore_root_context
from safe_rl.accvp.schema import COUNTERFACTUAL_SCHEMA_VERSION, file_sha256
from safe_rl.risk.merge_local import merge_local_stats
from safe_rl.sim.action_space import decode_action
from safe_rl.sim.metrics import compute_step_metrics
from safe_rl.sim.scenario_semantics import is_target_lane, merge_zone_edges, target_lane_edges, target_lane_mapping, merge_target_lane
from safe_rl.utils.config import ConfigDict


def _config_dict(value: Any) -> Any:
    if isinstance(value, dict):
        return ConfigDict({key: _config_dict(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_config_dict(item) for item in value]
    return value


def _write_npz_atomic(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def _branch_outcome(job: dict[str, Any]) -> dict[str, Any]:
    """Run exactly one branch in a worker-owned SUMO process.

    This is deliberately a top-level function so ProcessPoolExecutor/spawn creates
    a new Python process. It never receives the root collector environment or its
    TraCI connection.
    """

    root = load_root_context(job["root_metadata_path"], job["root_tensor_path"])
    config = _config_dict(job["config"])
    action = decode_action(int(job["action_id"]))
    output_dir = Path(job["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    branch_id = f"{root.root_id}_action{int(action.index)}"
    env = None
    try:
        from safe_rl.pipeline.common import make_env

        config.accvp["enabled"] = False
        config.accvp["mode"] = "off"
        env = make_env(config, seed=int(root.metadata["episode_seed"]), shield_enabled=False)
        env._start_sumo()
        # This loadState occurs only on the worker's own TraCI connection.
        env._traci.simulation.loadState(str(root.metadata["snapshot_path"]))
        restore_root_context(env, root)
        ego = env._get_ego()
        if ego is None:
            raise RuntimeError("worker has no ego after loadState")
        env._accvp_branch_target_speed = max(0.0, float(ego.speed) + float(action.accel_cmd) * 1.5 * 0.5)
        response_steps = max(1, int(config.accvp.response_horizon_steps))
        horizon_steps = max(1, int(config.accvp.candidate_plan_horizon_steps))
        actor_ids = [str(value) for value in root.metadata.get("selected_actor_ids", [])]
        actor_count = int(config.accvp.actor_count)
        actor_response = np.zeros((actor_count, response_steps, 5), dtype=np.float32)
        actor_valid = np.zeros((actor_count, response_steps), dtype=np.float32)
        min_distance = float("inf")
        min_ttc = float("inf")
        max_drac = 0.0
        proxy_collision = False
        safety_violation = False
        geometric_overlap = False
        collision = False
        target_front_gap = float("inf")
        target_rear_gap = float("inf")
        target_lane_entry_time: float | None = None
        taper_miss_time: float | None = None
        for step in range(horizon_steps):
            elapsed = float(step) * float(env.step_length)
            lane_oob = apply_branch_command(env, action, elapsed)
            env._simulation_step()
            env._episode_step += 1
            states = env._collect_states()
            env.history.append(states)
            env._invalidate_decision_cache()
            ego = env._get_ego()
            metrics = compute_step_metrics(
                ego,
                states,
                collision=env._ego_in_collision(),
                near_miss_threshold=float(config.risk_module.near_miss_distance_threshold),
                ttc_threshold=float(config.risk_module.ttc_threshold),
                drac_threshold=float(config.risk_module.drac_threshold),
                lane_oob=lane_oob,
                merge_ego_edges=merge_zone_edges(config),
                merge_target_edges=target_lane_edges(config),
                merge_target_lane=merge_target_lane(config),
                merge_target_lanes=target_lane_mapping(config),
            )
            local = merge_local_stats(ego, states, config)
            min_distance = min(min_distance, float(metrics.min_distance))
            min_ttc = min(min_ttc, float(metrics.min_ttc))
            max_drac = max(max_drac, float(metrics.max_drac))
            collision = collision or bool(metrics.collision)
            geometric_overlap = geometric_overlap or bool(metrics.geometric_overlap)
            proxy_collision = proxy_collision or bool(metrics.min_distance <= float(config.shield.fallback_min_distance))
            safety_violation = safety_violation or bool(
                metrics.collision or metrics.near_miss or metrics.low_ttc or metrics.high_drac or lane_oob
            )
            target_front_gap = min(target_front_gap, float(local.target_front_gap))
            target_rear_gap = min(target_rear_gap, float(local.target_rear_gap))
            observed_time = float(step + 1) * float(env.step_length)
            if ego is not None and target_lane_entry_time is None and is_target_lane(config, ego.edge_id, ego.lane_index):
                target_lane_entry_time = observed_time
            if taper_miss_time is None and bool(local.taper_miss):
                taper_miss_time = observed_time
            if step < response_steps:
                by_id = {str(state.vehicle_id): state for state in states}
                for actor_idx, actor_id in enumerate(actor_ids[:actor_count]):
                    actor = by_id.get(actor_id)
                    if actor is not None:
                        actor_response[actor_idx, step] = np.asarray(actor.as_vector(), dtype=np.float32)
                        actor_valid[actor_idx, step] = 1.0
        if target_lane_entry_time is not None and (taper_miss_time is None or target_lane_entry_time < taper_miss_time):
            viability_status = "observed_success"
            censor_reason = ""
            censor_time = target_lane_entry_time
        elif taper_miss_time is not None:
            viability_status = "observed_failure"
            censor_reason = ""
            censor_time = taper_miss_time
        else:
            viability_status = "censored"
            censor_reason = "horizon_elapsed"
            censor_time = float(horizon_steps) * float(env.step_length)
        tensor_path = output_dir / f"{branch_id}.npz"
        _write_npz_atomic(tensor_path, actor_response=actor_response, actor_valid_mask=actor_valid)
        return {
            "counterfactual_schema_version": COUNTERFACTUAL_SCHEMA_VERSION,
            "root_id": root.root_id,
            "branch_id": branch_id,
            "action_id": int(action.index),
            "action_name": str(action.name),
            "snapshot_sha256": str(root.metadata["snapshot_sha256"]),
            "candidate_plan_profile": ACCVP_COMMITMENT_PROFILE,
            "root_source": str(root.metadata["root_source"]),
            "traffic_profile": str(root.metadata["traffic_profile"]),
            "deadline_bin": str(root.metadata["deadline_bin"]),
            "episode_seed": int(root.metadata["episode_seed"]),
            "selected_actor_ids": actor_ids,
            "selected_actor_coverage_complete": bool(root.metadata.get("selected_actor_coverage_complete", False)),
            "safety_actor_coverage_complete": bool(root.metadata.get("safety_actor_coverage_complete", False)),
            "event_observed": viability_status != "censored",
            "censor_time": float(censor_time),
            "censor_reason": censor_reason,
            "viability_observation_status": viability_status,
            "collision_within_horizon": bool(collision),
            "proxy_collision_within_horizon": bool(proxy_collision),
            "safety_violation_within_horizon": bool(safety_violation),
            "geometric_overlap_within_horizon": bool(geometric_overlap),
            "taper_miss_observed": bool(taper_miss_time is not None),
            "merge_before_taper_observed": bool(viability_status == "observed_success"),
            "target_lane_entry_time_s": target_lane_entry_time,
            "min_obb_distance": float(min_distance),
            "min_ttc": float(min_ttc),
            "max_drac": float(max_drac),
            "target_front_gap": float(target_front_gap),
            "target_rear_gap": float(target_rear_gap),
            "tensor_path": str(tensor_path.resolve()),
            "tensor_sha256": file_sha256(tensor_path),
            "branch_status": "completed",
        }
    finally:
        if env is not None:
            env.close()


def run_branch_job(job: dict[str, Any]) -> dict[str, Any]:
    try:
        return {"ok": True, "row": _branch_outcome(job)}
    except Exception as exc:  # pragma: no cover - requires SUMO failure injection
        return {
            "ok": False,
            "root_id": str(job.get("root_id", "")),
            "action_id": int(job.get("action_id", -1)),
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
