from __future__ import annotations

from typing import Any

import numpy as np

from safe_rl.prediction.forecast_rollout_bundle import (
    ForecastRolloutBundle,
    get_or_build_forecast_rollout_bundle,
)
from safe_rl.prediction.trajectory_postprocess import trajectory_to_states
from safe_rl.risk.merge_local import (
    get_cached_ego_rollout,
    is_candidate_legal,
    merge_local_stats,
)
from safe_rl.sim.action_space import ACTIONS, CandidateAction, decode_action
from safe_rl.sim.metrics import INF_TTC, bbox_gap, drac, relative_ttc
from safe_rl.sim.history_buffer import HistoryBuffer
from safe_rl.sim.scenario_semantics import is_target_lane, merge_corridor_progress
from safe_rl.sim.types import VehicleState


class ForecastAwareTaskScorer:
    """Rule-based task risk scorer for taper-deadline merge decisions.

    The scorer intentionally does not depend on the learned Risk Module. It uses the
    current forecast source when available and falls back to route-aware constant
    velocity rollouts, so it can be used for diagnostics across CV/WcDT branches.
    """

    def __init__(self, config: Any, predictor: Any | None = None):
        self.config = config
        self.predictor = predictor

    def score(
        self,
        context: dict[str, Any],
        raw_action: int | CandidateAction,
        *,
        merge_cmd: int,
        deadline_distance: float,
        urgency: float,
    ) -> dict[str, Any]:
        ego = context.get("ego")
        if ego is None or int(merge_cmd) == 0:
            return self._empty()
        raw = decode_action(raw_action)
        horizon = int(self.config.forecast_features.get("horizon_steps", self.config.scenario.forecast_horizon_steps))
        dt = float(self.config.scenario.step_length)
        bundle = get_or_build_forecast_rollout_bundle(
            self.config,
            context,
            self.predictor,
        )
        other_rollouts = bundle.rollout_lists()
        uncertainty = float(bundle.combined_uncertainty)
        source = "hybrid" if self.predictor is not None else "constant_velocity"
        selected_vehicle_ids = [actor.vehicle_id for actor in bundle.actors]
        scores = [
            self._candidate_score(
                action,
                context,
                other_rollouts,
                uncertainty,
                merge_cmd=int(merge_cmd),
                urgency=float(urgency),
                deadline_distance=float(deadline_distance),
                dt=dt,
            )
            for action in ACTIONS
            if is_candidate_legal(action, context)
        ]
        scores = [item for item in scores if item is not None]
        if not scores:
            return self._empty(source=source, uncertainty=uncertainty)
        raw_score = next((item for item in scores if int(item["action"]) == int(raw.index)), None)
        if raw_score is None:
            raw_score = self._candidate_score(
                raw,
                context,
                other_rollouts,
                uncertainty,
                merge_cmd=int(merge_cmd),
                urgency=float(urgency),
                deadline_distance=float(deadline_distance),
                dt=dt,
            )
        best = min(scores, key=lambda item: float(item["task_cost"]))
        best_action = decode_action(int(best["action"]))
        safety_threshold = float(self.config.shield.get("task_backstop_safety_risk_threshold", 0.35))
        uncertainty_threshold = float(self.config.shield.get("task_backstop_uncertainty_threshold", 0.40))
        front_threshold = float(self.config.scenario.get("merge_opportunity_min_front_gap", 12.0))
        rear_threshold = float(self.config.scenario.get("merge_opportunity_min_rear_gap", 12.0))
        local = context.get("merge_local")
        target_front_vehicle_id = str(getattr(local, "target_front_vehicle_id", "") or "")
        target_rear_vehicle_id = str(getattr(local, "target_rear_vehicle_id", "") or "")
        target_front_required = bool(target_front_vehicle_id)
        target_rear_required = bool(target_rear_vehicle_id)
        selected_vehicle_id_set = {
            str(value)
            for value in bundle.wcdt_selected_vehicle_ids
            if str(value)
        }
        combined_vehicle_id_set = {
            str(value)
            for value in selected_vehicle_ids
            if str(value)
        }
        target_front_covered = not target_front_required or target_front_vehicle_id in selected_vehicle_id_set
        target_rear_covered = not target_rear_required or target_rear_vehicle_id in selected_vehicle_id_set
        target_front_safety_covered = (
            not target_front_required or target_front_vehicle_id in combined_vehicle_id_set
        )
        target_rear_safety_covered = (
            not target_rear_required or target_rear_vehicle_id in combined_vehicle_id_set
        )
        coverage_complete = bool(bundle.forecast_safety_actor_coverage_complete)
        max_gap_jump = float(self.config.shield.get("task_backstop_max_first_step_gap_jump", 20.0))
        current_front_gap = float(getattr(local, "target_front_gap", INF_TTC))
        current_rear_gap = float(getattr(local, "target_rear_gap", INF_TTC))
        first_front_gap = float(best["first_step_target_front_gap"])
        first_rear_gap = float(best["first_step_target_rear_gap"])
        first_front_vehicle_id = str(best["first_step_target_front_vehicle_id"])
        first_rear_vehicle_id = str(best["first_step_target_rear_vehicle_id"])
        front_consistent = bool(
            not target_front_required
            or (
                first_front_vehicle_id == target_front_vehicle_id
                and abs(first_front_gap - current_front_gap) <= max_gap_jump
            )
        )
        rear_consistent = bool(
            not target_rear_required
            or (
                first_rear_vehicle_id == target_rear_vehicle_id
                and abs(first_rear_gap - current_rear_gap) <= max_gap_jump
            )
        )
        front_progress_error, front_progress_pass = self._first_step_progress_consistency(
            bundle,
            target_front_vehicle_id,
            dt,
        )
        rear_progress_error, rear_progress_pass = self._first_step_progress_consistency(
            bundle,
            target_rear_vehicle_id,
            dt,
        )
        identity_consistent = bool(
            front_consistent
            and rear_consistent
            and (not target_front_required or first_front_vehicle_id == target_front_vehicle_id)
            and (not target_rear_required or first_rear_vehicle_id == target_rear_vehicle_id)
        )
        physical_consistency_pass = bool(
            identity_consistent and front_progress_pass and rear_progress_pass
        )
        gap_consistency_pass = bool(
            coverage_complete
            and front_consistent
            and rear_consistent
            and physical_consistency_pass
        )
        would_merge = bool(
            int(best_action.lateral_cmd) == int(merge_cmd)
            and int(raw.lateral_cmd) != int(merge_cmd)
            and coverage_complete
            and bundle.wcdt_required_actor_coverage_complete
            and not bundle.actor_selector_overflow
            and not bundle.cv_fallback_overflow
            and gap_consistency_pass
            and float(best["safety_risk"]) <= safety_threshold
            and float(best["target_front_gap"]) >= front_threshold
            and float(best["target_rear_gap"]) >= rear_threshold
            and float(uncertainty) <= uncertainty_threshold
        )
        raw_score = raw_score or best
        task_improvement = float(raw_score["task_cost"] - best["task_cost"])
        return {
            "forecast_aware_available": True,
            "forecast_aware_source": source,
            "forecast_aware_raw_task_cost": float(raw_score["task_cost"]),
            "forecast_aware_best_task_cost": float(best["task_cost"]),
            "forecast_aware_task_improvement": task_improvement,
            "forecast_aware_raw_task_risk": float(raw_score["task_risk"]),
            "forecast_aware_raw_safety_risk": float(raw_score["safety_risk"]),
            "forecast_aware_best_task_risk": float(best["task_risk"]),
            "forecast_aware_best_action": int(best_action.index),
            "forecast_aware_best_action_name": str(best_action.name),
            "forecast_aware_would_merge": would_merge,
            "forecast_aware_safety_risk": float(best["safety_risk"]),
            "forecast_aware_best_safety_risk": float(best["safety_risk"]),
            "forecast_aware_uncertainty": float(uncertainty),
            "forecast_aware_future_min_distance": float(best["future_min_distance"]),
            "forecast_aware_future_min_ttc": float(best["future_min_ttc"]),
            "forecast_aware_future_max_drac": float(best["future_max_drac"]),
            "forecast_aware_target_front_gap": float(best["target_front_gap"]),
            "forecast_aware_target_rear_gap": float(best["target_rear_gap"]),
            "forecast_first_step_target_front_gap": first_front_gap,
            "forecast_first_step_target_rear_gap": first_rear_gap,
            "forecast_gap_consistency_pass": gap_consistency_pass,
            "forecast_gap_physical_consistency_pass": physical_consistency_pass,
            "forecast_vehicle_identity_consistent": identity_consistent,
            "forecast_front_first_step_progress_error": front_progress_error,
            "forecast_rear_first_step_progress_error": rear_progress_error,
            "forecast_selected_vehicle_ids": list(selected_vehicle_ids),
            "forecast_target_front_vehicle_id": target_front_vehicle_id,
            "forecast_target_rear_vehicle_id": target_rear_vehicle_id,
            "forecast_target_front_required": target_front_required,
            "forecast_target_rear_required": target_rear_required,
            "forecast_target_front_covered": target_front_covered,
            "forecast_target_rear_covered": target_rear_covered,
            "forecast_target_front_safety_covered": target_front_safety_covered,
            "forecast_target_rear_safety_covered": target_rear_safety_covered,
            "forecast_actor_coverage_complete": bool(
                bundle.wcdt_required_actor_coverage_complete
            ),
            "forecast_closest_vehicle_id": str(best["closest_vehicle_id"]),
            "forecast_front_gap_vehicle_id": str(best["front_gap_vehicle_id"]),
            "forecast_rear_gap_vehicle_id": str(best["rear_gap_vehicle_id"]),
            "forecast_aware_taper_miss_risk": float(best["taper_miss_risk"]),
            "forecast_aware_merge_progress_bonus": float(best["merge_progress_bonus"]),
            **bundle.trace_fields(),
        }

    def _empty(self, *, source: str = "unavailable", uncertainty: float = 0.0) -> dict[str, Any]:
        return {
            "forecast_aware_available": False,
            "forecast_aware_source": source,
            "forecast_aware_raw_task_cost": None,
            "forecast_aware_best_task_cost": None,
            "forecast_aware_task_improvement": None,
            "forecast_aware_raw_task_risk": None,
            "forecast_aware_raw_safety_risk": None,
            "forecast_aware_best_task_risk": None,
            "forecast_aware_best_action": None,
            "forecast_aware_best_action_name": "",
            "forecast_aware_would_merge": False,
            "forecast_aware_safety_risk": None,
            "forecast_aware_best_safety_risk": None,
            "forecast_aware_uncertainty": float(uncertainty),
            "forecast_aware_future_min_distance": None,
            "forecast_aware_future_min_ttc": None,
            "forecast_aware_future_max_drac": None,
            "forecast_aware_target_front_gap": None,
            "forecast_aware_target_rear_gap": None,
            "forecast_first_step_target_front_gap": None,
            "forecast_first_step_target_rear_gap": None,
            "forecast_gap_consistency_pass": False,
            "forecast_gap_physical_consistency_pass": False,
            "forecast_vehicle_identity_consistent": False,
            "forecast_front_first_step_progress_error": None,
            "forecast_rear_first_step_progress_error": None,
            "forecast_selected_vehicle_ids": [],
            "forecast_target_front_vehicle_id": "",
            "forecast_target_rear_vehicle_id": "",
            "forecast_target_front_required": False,
            "forecast_target_rear_required": False,
            "forecast_target_front_covered": False,
            "forecast_target_rear_covered": False,
            "forecast_target_front_safety_covered": False,
            "forecast_target_rear_safety_covered": False,
            "forecast_actor_coverage_complete": False,
            "wcdt_required_actor_coverage_complete": False,
            "forecast_safety_actor_coverage_complete": False,
            "actor_selector_relevant_count": 0,
            "actor_selector_overflow": False,
            "actor_selector_dropped_relevant_ids": [],
            "cv_fallback_overflow": False,
            "cv_fallback_dropped_vehicle_ids": [],
            "forecast_wcdt_selected_vehicle_ids": [],
            "forecast_cv_fallback_vehicle_ids": [],
            "forecast_actor_sources": {},
            "forecast_actor_relevance": {},
            "forecast_wcdt_uncertainty": 0.0,
            "forecast_cv_fallback_uncertainty": 0.0,
            "combined_forecast_uncertainty": float(uncertainty),
            "forecast_closest_vehicle_id": "",
            "forecast_front_gap_vehicle_id": "",
            "forecast_rear_gap_vehicle_id": "",
            "forecast_aware_taper_miss_risk": None,
            "forecast_aware_merge_progress_bonus": None,
        }

    def _first_step_progress_consistency(
        self,
        bundle: ForecastRolloutBundle,
        vehicle_id: str,
        dt: float,
    ) -> tuple[float | None, bool]:
        if not vehicle_id:
            return None, True
        actor = bundle.actor_by_id(vehicle_id)
        if actor is None or actor.current_state is None or not actor.trajectory:
            return None, False
        current_progress = merge_corridor_progress(self.config, actor.current_state)
        first_progress = merge_corridor_progress(self.config, actor.trajectory[0])
        if current_progress is None or first_progress is None:
            return None, False
        expected = (
            float(actor.current_state.speed) * float(dt)
            + 0.5 * float(actor.current_state.accel) * float(dt) * float(dt)
        )
        actual = float(first_progress - current_progress)
        error = abs(actual - expected)
        tolerance = max(2.0, 0.5 * abs(expected))
        return float(error), bool(error <= tolerance)

    def _prediction_rollouts(
        self,
        context: dict[str, Any],
        trajectories: np.ndarray,
        prediction: dict[str, Any],
        horizon: int,
        dt: float,
    ) -> tuple[list[list[VehicleState]], list[str]]:
        """Compatibility helper using explicit vehicle IDs, never row-position inference."""

        history = context.get("history")
        latest = history.latest() if isinstance(history, HistoryBuffer) else {
            str(vehicle.vehicle_id): vehicle
            for vehicle in context.get("vehicles", [])
        }
        selected_vehicle_ids = [
            str(value or "")
            for value in prediction.get("selected_vehicle_ids", [])
        ]
        rollouts: list[list[VehicleState]] = []
        used_vehicle_ids: list[str] = []
        for actor_idx, trajectory in enumerate(np.asarray(trajectories)):
            vehicle_id = (
                selected_vehicle_ids[actor_idx]
                if actor_idx < len(selected_vehicle_ids)
                else ""
            )
            reference = latest.get(vehicle_id)
            if reference is None:
                continue
            states = trajectory_to_states(
                trajectory[:horizon],
                reference=reference,
                dt=dt,
                vehicle_id=vehicle_id,
            )
            if states:
                rollouts.append(states)
                used_vehicle_ids.append(vehicle_id)
        return rollouts, used_vehicle_ids

    def _candidate_score(
        self,
        action: CandidateAction,
        context: dict[str, Any],
        other_rollouts: list[list[VehicleState]],
        uncertainty: float,
        *,
        merge_cmd: int,
        urgency: float,
        deadline_distance: float,
        dt: float,
    ) -> dict[str, Any] | None:
        ego = context.get("ego")
        if ego is None:
            return None
        horizon = int(self.config.forecast_features.get("horizon_steps", self.config.scenario.forecast_horizon_steps))
        ego_rollout, taper_miss = get_cached_ego_rollout(
            context,
            action,
            horizon_steps=horizon,
            dt=dt,
        )
        min_distance = INF_TTC
        min_ttc = INF_TTC
        max_drac = 0.0
        front_gap = INF_TTC
        rear_gap = INF_TTC
        closest_vehicle_id = ""
        front_gap_vehicle_id = ""
        rear_gap_vehicle_id = ""
        first_step_front_gap = INF_TTC
        first_step_rear_gap = INF_TTC
        first_step_front_vehicle_id = ""
        first_step_rear_vehicle_id = ""
        for step_idx, ego_future in enumerate(ego_rollout):
            step_target_vehicles: list[VehicleState] = []
            for rollout in other_rollouts:
                if not rollout:
                    continue
                other = rollout[min(step_idx, len(rollout) - 1)]
                candidate_distance = bbox_gap(ego_future, other)
                if candidate_distance < min_distance:
                    min_distance = candidate_distance
                    closest_vehicle_id = str(other.vehicle_id)
                min_ttc = min(min_ttc, relative_ttc(ego_future, other))
                max_drac = max(max_drac, drac(ego_future, other))
                if is_target_lane(self.config, other.edge_id, other.lane_index):
                    step_target_vehicles.append(other)
            if step_target_vehicles:
                stats = merge_local_stats(ego_future, [ego_future, *step_target_vehicles], self.config)
                if float(stats.target_front_gap) < front_gap:
                    front_gap = float(stats.target_front_gap)
                    front_gap_vehicle_id = str(stats.target_front_vehicle_id)
                if float(stats.target_rear_gap) < rear_gap:
                    rear_gap = float(stats.target_rear_gap)
                    rear_gap_vehicle_id = str(stats.target_rear_vehicle_id)
                if step_idx == 0:
                    first_step_front_gap = float(stats.target_front_gap)
                    first_step_rear_gap = float(stats.target_rear_gap)
                    first_step_front_vehicle_id = str(stats.target_front_vehicle_id)
                    first_step_rear_vehicle_id = str(stats.target_rear_vehicle_id)
        if front_gap >= INF_TTC:
            front_gap = float(getattr(context.get("merge_local"), "target_front_gap", INF_TTC))
        if rear_gap >= INF_TTC:
            rear_gap = float(getattr(context.get("merge_local"), "target_rear_gap", INF_TTC))
        front_threshold = float(self.config.scenario.get("merge_opportunity_min_front_gap", 12.0))
        rear_threshold = float(self.config.scenario.get("merge_opportunity_min_rear_gap", 12.0))
        distance_risk = float(np.clip((5.0 - min_distance) / 5.0, 0.0, 1.0))
        ttc_risk = float(np.clip((2.0 - min_ttc) / 2.0, 0.0, 1.0)) if min_ttc < INF_TTC else 0.0
        drac_risk = float(np.clip(max_drac / 20.0, 0.0, 1.0))
        front_risk = float(np.clip((front_threshold - front_gap) / max(front_threshold, 1.0e-6), 0.0, 1.0))
        rear_risk = float(np.clip((rear_threshold - rear_gap) / max(rear_threshold, 1.0e-6), 0.0, 1.0))
        unsafe_gap_risk = max(front_risk, rear_risk)
        taper_miss_risk = 1.0 if taper_miss else (float(urgency) if int(action.lateral_cmd) != int(merge_cmd) else 0.0)
        safety_risk = max(distance_risk, ttc_risk, drac_risk, unsafe_gap_risk)
        uncertainty_risk = float(np.clip(float(uncertainty) / 0.40, 0.0, 1.0))
        merge_progress_bonus = (
            0.20
            if int(action.lateral_cmd) == int(merge_cmd)
            and unsafe_gap_risk <= 0.0
            and not taper_miss
            else 0.0
        )
        task_cost = float(
            0.35 * taper_miss_risk
            + 0.25 * unsafe_gap_risk
            + 0.25 * safety_risk
            + 0.15 * uncertainty_risk
            - merge_progress_bonus
        )
        task_risk = float(np.clip(task_cost, 0.0, 1.0))
        return {
            "action": int(action.index),
            "task_cost": task_cost,
            "task_risk": task_risk,
            "safety_risk": float(safety_risk),
            "future_min_distance": float(min_distance),
            "future_min_ttc": float(min_ttc),
            "future_max_drac": float(max_drac),
            "target_front_gap": float(front_gap),
            "target_rear_gap": float(rear_gap),
            "first_step_target_front_gap": float(first_step_front_gap),
            "first_step_target_rear_gap": float(first_step_rear_gap),
            "first_step_target_front_vehicle_id": first_step_front_vehicle_id,
            "first_step_target_rear_vehicle_id": first_step_rear_vehicle_id,
            "closest_vehicle_id": closest_vehicle_id,
            "front_gap_vehicle_id": front_gap_vehicle_id,
            "rear_gap_vehicle_id": rear_gap_vehicle_id,
            "taper_miss_risk": float(taper_miss_risk),
            "merge_progress_bonus": float(merge_progress_bonus),
        }
