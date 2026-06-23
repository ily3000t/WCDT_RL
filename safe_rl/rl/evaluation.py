from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from safe_rl.pipeline.common import make_env
from safe_rl.risk.risk_aggregator import aggregate_episode_reports
from safe_rl.rl.ppo import _training_device, load_ppo
from safe_rl.utils.progress import TensorboardLogger, progress_iter, stage_log
from safe_rl.utils.replay import write_replay_file


def validate_model_env_observation_shape(model: Any, env: Any, model_path: str | Path) -> None:
    model_shape = tuple(getattr(model.observation_space, "shape", ()) or ())
    env_shape = tuple(getattr(env.observation_space, "shape", ()) or ())
    if model_shape != env_shape:
        raise ValueError(
            f"PPO model observation shape {model_shape} does not match environment observation shape "
            f"{env_shape}; model={model_path}"
        )


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _step_safety_record(
    *,
    step_index: int,
    raw_action: int,
    final_action: int,
    reward: float,
    terminated: bool,
    truncated: bool,
    info: dict[str, Any],
    collision_threshold: float,
    shield_enabled: bool,
) -> dict[str, Any]:
    min_distance = _safe_float(info.get("min_distance"), 1.0e9)
    min_ttc = _safe_float(info.get("min_ttc"), 1.0e9)
    max_drac = _safe_float(info.get("max_drac"), 0.0)
    collision = bool(info.get("collision", False))
    geometric_overlap = bool(info.get("geometric_overlap", False))
    near_miss = bool(info.get("near_miss", False))
    proxy_collision = min_distance <= float(collision_threshold)
    safety_violation = bool(collision or proxy_collision or near_miss or min_ttc < 0.30)
    return {
        "step": int(info.get("step", step_index)),
        "control_step": int(step_index),
        "shield_record_index": int(step_index) if shield_enabled else None,
        "raw_action": int(raw_action),
        "final_action": int(final_action),
        "raw_action_name": str(info.get("raw_action_name", "")),
        "final_action_name": str(info.get("final_action_name", "")),
        "safety_shield_action": int(info.get("safety_shield_action", final_action)),
        "safety_shield_action_name": str(info.get("safety_shield_action_name", "")),
        "safety_shield_replaced": bool(info.get("safety_shield_replaced", False)),
        "action_execution_path": str(info.get("action_execution_path", "policy")),
        "accvp_mode": str(info.get("accvp_mode", "off")),
        "accvp_replacement": bool(info.get("accvp_replacement", False)),
        "accvp_replacement_reason": str(info.get("accvp_replacement_reason", "")),
        "accvp_bypass_reason": str(info.get("accvp_bypass_reason", "")),
        "accvp_no_feasible_action": bool(info.get("accvp_no_feasible_action", False)),
        "accvp_commitment_cancelled": bool(info.get("accvp_commitment_cancelled", False)),
        "accvp_decision_latency_s": _safe_float(info.get("decision_latency_s"), 0.0),
        "reward": float(reward),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "done_reason": info.get("done_reason", ""),
        "min_distance": min_distance,
        "min_ttc": min_ttc,
        "drac": max_drac,
        "drac_raw": max_drac,
        "collision": collision,
        "geometric_overlap": geometric_overlap,
        "near_miss": near_miss,
        "proxy_collision": proxy_collision,
        "safety_violation": safety_violation,
        "low_ttc": bool(info.get("low_ttc", False)),
        "high_drac": bool(info.get("high_drac", False)),
        "target_front_gap": _safe_float(info.get("target_front_gap"), 1.0e9),
        "target_rear_gap": _safe_float(info.get("target_rear_gap"), 1.0e9),
        "target_lane_gap": _safe_float(info.get("target_lane_gap"), 1.0e9),
        "distance_to_taper": _safe_float(info.get("distance_to_taper"), 1.0e9),
        "taper_miss": bool(info.get("taper_miss", False)),
        "ego_edge": str(info.get("ego_edge", "")),
        "ego_lane": int(info.get("ego_lane", -1)),
        "decision_step": info.get("decision_step"),
        "decision_distance_to_taper": info.get("decision_distance_to_taper"),
        "decision_target_front_gap": info.get("decision_target_front_gap"),
        "decision_target_rear_gap": info.get("decision_target_rear_gap"),
        "decision_task_deadline_urgency": info.get("decision_task_deadline_urgency"),
        "decision_ego_edge": str(info.get("decision_ego_edge", "")),
        "decision_ego_lane": int(info.get("decision_ego_lane", -1)),
        "post_action_step": info.get("post_action_step"),
        "post_action_distance_to_taper": info.get("post_action_distance_to_taper"),
        "post_action_target_front_gap": info.get("post_action_target_front_gap"),
        "post_action_target_rear_gap": info.get("post_action_target_rear_gap"),
        "post_action_ego_edge": str(info.get("post_action_ego_edge", "")),
        "post_action_ego_lane": int(info.get("post_action_ego_lane", -1)),
        "closest_vehicle_id": str(info.get("closest_vehicle_id", "")),
        "closest_vehicle_edge": str(info.get("closest_vehicle_edge", "")),
        "closest_vehicle_lane": int(info.get("closest_vehicle_lane", -1)),
        "ttc_vehicle_id": str(info.get("ttc_vehicle_id", "")),
        "drac_vehicle_id": str(info.get("drac_vehicle_id", "")),
        "ego_on_auxiliary": bool(info.get("ego_on_auxiliary", False)),
        "best_merge_action": str(info.get("best_merge_action", "")),
        "best_merge_action_risk": info.get("best_merge_action_risk"),
        "safe_merge_opportunity_count": int(info.get("safe_merge_opportunity_count", 0)),
        "missed_safe_merge_opportunity_count": int(info.get("missed_safe_merge_opportunity_count", 0)),
        "task_merge_opportunity": bool(info.get("task_merge_opportunity", False)),
        "task_would_merge": bool(info.get("task_would_merge", False)),
        "task_missed_merge": bool(info.get("task_missed_merge", False)),
        "task_deadline_urgency": _safe_float(info.get("task_deadline_urgency"), 0.0),
        "task_safe_merge_action": str(info.get("task_safe_merge_action", "")),
        "forecast_aware_candidate_ranking_mode": str(
            info.get("forecast_aware_candidate_ranking_mode", "")
        ),
        "forecast_aware_raw_score": info.get("forecast_aware_raw_score"),
        "forecast_aware_best_score": info.get("forecast_aware_best_score"),
        "forecast_aware_score_improvement": info.get("forecast_aware_score_improvement"),
        "forecast_aware_raw_task_risk": info.get("forecast_aware_raw_task_risk"),
        "forecast_aware_best_task_risk": info.get("forecast_aware_best_task_risk"),
        "forecast_aware_raw_task_cost": info.get("forecast_aware_raw_task_cost"),
        "forecast_aware_best_task_cost": info.get("forecast_aware_best_task_cost"),
        "forecast_aware_task_improvement": info.get("forecast_aware_task_improvement"),
        "forecast_aware_best_action": info.get("forecast_aware_best_action"),
        "forecast_aware_best_action_name": str(info.get("forecast_aware_best_action_name", "")),
        "forecast_aware_would_merge": bool(info.get("forecast_aware_would_merge", False)),
        "forecast_aware_safety_risk": info.get("forecast_aware_safety_risk"),
        "forecast_aware_uncertainty": info.get("forecast_aware_uncertainty"),
        "forecast_aware_target_front_gap": info.get("forecast_aware_target_front_gap"),
        "forecast_aware_target_rear_gap": info.get("forecast_aware_target_rear_gap"),
        "forecast_first_step_target_front_gap": info.get("forecast_first_step_target_front_gap"),
        "forecast_first_step_target_rear_gap": info.get("forecast_first_step_target_rear_gap"),
        "forecast_gap_consistency_pass": bool(info.get("forecast_gap_consistency_pass", False)),
        "forecast_gap_physical_consistency_pass": bool(
            info.get("forecast_gap_physical_consistency_pass", False)
        ),
        "forecast_vehicle_identity_consistent": bool(
            info.get("forecast_vehicle_identity_consistent", False)
        ),
        "forecast_front_first_step_progress_error": info.get(
            "forecast_front_first_step_progress_error"
        ),
        "forecast_rear_first_step_progress_error": info.get(
            "forecast_rear_first_step_progress_error"
        ),
        "forecast_selected_vehicle_ids": list(info.get("forecast_selected_vehicle_ids", []) or []),
        "forecast_wcdt_selected_vehicle_ids": list(
            info.get("forecast_wcdt_selected_vehicle_ids", []) or []
        ),
        "forecast_cv_fallback_vehicle_ids": list(
            info.get("forecast_cv_fallback_vehicle_ids", []) or []
        ),
        "forecast_actor_sources": dict(info.get("forecast_actor_sources", {}) or {}),
        "forecast_actor_relevance": dict(info.get("forecast_actor_relevance", {}) or {}),
        "forecast_wcdt_uncertainty": info.get("forecast_wcdt_uncertainty"),
        "forecast_cv_fallback_uncertainty": info.get("forecast_cv_fallback_uncertainty"),
        "combined_forecast_uncertainty": info.get("combined_forecast_uncertainty"),
        "forecast_target_front_vehicle_id": str(info.get("forecast_target_front_vehicle_id", "")),
        "forecast_target_rear_vehicle_id": str(info.get("forecast_target_rear_vehicle_id", "")),
        "forecast_target_front_required": bool(info.get("forecast_target_front_required", False)),
        "forecast_target_rear_required": bool(info.get("forecast_target_rear_required", False)),
        "forecast_target_front_covered": bool(info.get("forecast_target_front_covered", False)),
        "forecast_target_rear_covered": bool(info.get("forecast_target_rear_covered", False)),
        "forecast_actor_coverage_complete": bool(info.get("forecast_actor_coverage_complete", False)),
        "wcdt_required_actor_coverage_complete": bool(
            info.get("wcdt_required_actor_coverage_complete", False)
        ),
        "forecast_safety_actor_coverage_complete": bool(
            info.get("forecast_safety_actor_coverage_complete", False)
        ),
        "actor_selector_relevant_count": int(info.get("actor_selector_relevant_count", 0)),
        "actor_selector_overflow": bool(info.get("actor_selector_overflow", False)),
        "actor_selector_dropped_relevant_ids": list(
            info.get("actor_selector_dropped_relevant_ids", []) or []
        ),
        "cv_fallback_overflow": bool(info.get("cv_fallback_overflow", False)),
        "cv_fallback_dropped_vehicle_ids": list(
            info.get("cv_fallback_dropped_vehicle_ids", []) or []
        ),
        "forecast_closest_vehicle_id": str(info.get("forecast_closest_vehicle_id", "")),
        "forecast_front_gap_vehicle_id": str(info.get("forecast_front_gap_vehicle_id", "")),
        "forecast_rear_gap_vehicle_id": str(info.get("forecast_rear_gap_vehicle_id", "")),
        "task_backstop_watch_count": int(info.get("task_backstop_watch_count", 0)),
        "task_backstop_watch_eligible": bool(info.get("task_backstop_watch_eligible", False)),
        "task_backstop_eligible": bool(info.get("task_backstop_eligible", False)),
        "task_backstop_risk_module_score": info.get("task_backstop_risk_module_score"),
        "task_backstop_risk_module_uncertainty": info.get("task_backstop_risk_module_uncertainty"),
        "task_backstop_risk_module_pass": bool(info.get("task_backstop_risk_module_pass", False)),
        "task_backstop_veto_reason": str(info.get("task_backstop_veto_reason", "")),
        "task_replacement": bool(info.get("task_replacement", False)),
        "task_replacement_reason": str(info.get("task_replacement_reason", "")),
        "forecast_ranking_eligible": bool(info.get("forecast_ranking_eligible", False)),
        "forecast_ranking_veto_reason": str(info.get("forecast_ranking_veto_reason", "")),
        "forecast_ranking_risk_module_score": info.get("forecast_ranking_risk_module_score"),
        "forecast_ranking_risk_module_uncertainty": info.get(
            "forecast_ranking_risk_module_uncertainty"
        ),
        "forecast_ranking_risk_module_pass": bool(
            info.get("forecast_ranking_risk_module_pass", False)
        ),
        "forecast_ranking_replacement": bool(info.get("forecast_ranking_replacement", False)),
        "forecast_ranking_replacement_reason": str(
            info.get("forecast_ranking_replacement_reason", "")
        ),
        "safety_metric_version": str(info.get("safety_metric_version", "")),
    }


def evaluate_policy(
    cfg: Any,
    model_path: str | Path | None,
    seeds: list[int],
    shield_enabled: bool,
    risk_checkpoint: str | None = None,
    replay_dir: str | Path | None = None,
    group_name: str | None = None,
    tensorboard: TensorboardLogger | None = None,
    tensorboard_step_offset: int = 0,
    policy_type: str = "sb3_ppo",
) -> dict:
    policy_type = str(policy_type).strip().lower()
    if policy_type not in {"sb3_ppo", "rule_gap_acceptance"}:
        raise ValueError(f"Unsupported policy_type={policy_type!r}")
    model = load_ppo(model_path, device=_training_device(cfg)) if policy_type == "sb3_ppo" else None
    controller = None
    if policy_type == "rule_gap_acceptance":
        from safe_rl.baselines import RuleGapAcceptancePolicy

        controller = RuleGapAcceptancePolicy(cfg)
    shape_env = make_env(cfg, seed=int(seeds[0]) if seeds else int(cfg.run.seed), shield_enabled=shield_enabled, risk_checkpoint=risk_checkpoint)
    try:
        if model is not None:
            validate_model_env_observation_shape(model, shape_env, model_path or "")
        model_observation_shape = tuple(model.observation_space.shape) if model is not None else []
        env_observation_shape = tuple(shape_env.observation_space.shape)
    finally:
        shape_env.close()
    reports: list[dict] = []
    rewards: list[float] = []
    collision_threshold = float(cfg.risk_module.collision_distance_threshold)
    for episode_idx, seed in enumerate(progress_iter(seeds, desc=f"Eval {group_name or 'ppo'} seeds")):
        env = make_env(cfg, seed=seed, shield_enabled=shield_enabled, risk_checkpoint=risk_checkpoint)
        total_reward = 0.0
        actions: list[int] = []
        executed_actions: list[int] = []
        step_safety_records: list[dict[str, Any]] = []
        try:
            obs, _info = env.reset(seed=seed)
            terminated = truncated = False
            while not (terminated or truncated):
                if model is not None:
                    action, _state = model.predict(obs, deterministic=True)
                else:
                    decision = controller.act(env.get_rule_control_context())
                    action = int(decision.action)
                actions.append(int(action))
                obs, reward, terminated, truncated, _info = env.step(int(action))
                final_action = int(_info.get("final_action", action))
                executed_actions.append(final_action)
                step_safety_records.append(
                    _step_safety_record(
                        step_index=len(actions) - 1,
                        raw_action=int(action),
                        final_action=final_action,
                        reward=float(reward),
                        terminated=bool(terminated),
                        truncated=bool(truncated),
                        info=_info,
                        collision_threshold=collision_threshold,
                        shield_enabled=shield_enabled,
                    )
                )
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
                    executed_actions=executed_actions,
                    shield_enabled=shield_enabled,
                    risk_checkpoint=risk_checkpoint if shield_enabled else None,
                    model_path=str(model_path or ""),
                    group_name=group_name,
                    safety_metric_version=str(cfg.risk_module.get("safety_metric_version", "")),
                    trace_schema_version=2,
                    notes={"episode_report": report, "step_safety_records": step_safety_records},
                )
        finally:
            env.close()
    metrics = aggregate_episode_reports(reports)
    metrics["average_reward"] = float(np.mean(rewards)) if rewards else 0.0
    metrics["merge_success_rate"] = float(np.mean([float(item.get("merge_success", False)) for item in reports])) if reports else 0.0
    stage_log("stage5", f"group={group_name or 'ppo'} metrics={metrics}")
    return {
        "episodes": reports,
        "metrics": metrics,
        "model_observation_shape": list(model_observation_shape),
        "env_observation_shape": list(env_observation_shape),
        "policy_type": policy_type,
    }


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
    """Backward-compatible PPO-only entrypoint."""

    return evaluate_policy(
        cfg,
        model_path,
        seeds,
        shield_enabled,
        risk_checkpoint=risk_checkpoint,
        replay_dir=replay_dir,
        group_name=group_name,
        tensorboard=tensorboard,
        tensorboard_step_offset=tensorboard_step_offset,
        policy_type="sb3_ppo",
    )
