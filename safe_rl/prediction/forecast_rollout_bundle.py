from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from safe_rl.prediction.actor_selector import (
    ActorSelectionResult,
    actor_relevance_config,
    select_merge_relevant_actors,
)
from safe_rl.prediction.trajectory_postprocess import trajectory_to_states
from safe_rl.risk.merge_local import merge_local_stats, route_aware_constant_velocity_rollout
from safe_rl.sim.metrics import INF_TTC, bbox_gap, drac, relative_ttc
from safe_rl.sim.scenario_semantics import is_auxiliary_edge, is_ramp_edge
from safe_rl.sim.types import VehicleState


FORECAST_ROLLOUT_BUNDLE_VERSION = "forecast_rollout_bundle_v1"


def _prediction_array(prediction: dict[str, Any]) -> np.ndarray | None:
    trajectories = prediction.get("future_trajectories")
    if trajectories is None:
        return None
    if hasattr(trajectories, "detach"):
        trajectories = trajectories.detach().cpu().numpy()
    trajectories = np.asarray(trajectories, dtype=np.float32)
    if trajectories.ndim == 5:
        trajectories = trajectories[0]
    if trajectories.ndim == 4:
        trajectories = trajectories[:, 0]
    if trajectories.ndim != 3 or trajectories.size == 0:
        return None
    return trajectories


@dataclass
class ForecastActorRollout:
    vehicle_id: str
    source: str
    trajectory: list[VehicleState]
    uncertainty: float
    relevance_reasons: tuple[str, ...] = ()
    current_state: VehicleState | None = None


@dataclass
class ForecastRolloutBundle:
    actors: list[ForecastActorRollout]
    selection_result: ActorSelectionResult
    wcdt_uncertainty: float
    cv_fallback_uncertainty: float
    combined_uncertainty: float
    wcdt_selected_vehicle_ids: list[str]
    cv_fallback_vehicle_ids: list[str]
    safety_required_vehicle_ids: list[str]
    wcdt_required_actor_coverage_complete: bool
    forecast_safety_actor_coverage_complete: bool
    actor_selector_overflow: bool
    cv_fallback_overflow: bool
    cv_fallback_dropped_vehicle_ids: list[str]
    version: str = FORECAST_ROLLOUT_BUNDLE_VERSION
    actor_sources: dict[str, str] = field(default_factory=dict)

    def rollout_lists(self) -> list[list[VehicleState]]:
        return [actor.trajectory for actor in self.actors if actor.trajectory]

    def actor_by_id(self, vehicle_id: str) -> ForecastActorRollout | None:
        return next((actor for actor in self.actors if actor.vehicle_id == vehicle_id), None)

    def trace_fields(self) -> dict[str, Any]:
        return {
            "forecast_rollout_bundle_version": self.version,
            "forecast_wcdt_selected_vehicle_ids": list(self.wcdt_selected_vehicle_ids),
            "forecast_cv_fallback_vehicle_ids": list(self.cv_fallback_vehicle_ids),
            "forecast_actor_sources": dict(self.actor_sources),
            "forecast_wcdt_uncertainty": float(self.wcdt_uncertainty),
            "forecast_cv_fallback_uncertainty": float(self.cv_fallback_uncertainty),
            "combined_forecast_uncertainty": float(self.combined_uncertainty),
            "wcdt_required_actor_coverage_complete": bool(
                self.wcdt_required_actor_coverage_complete
            ),
            "forecast_safety_actor_coverage_complete": bool(
                self.forecast_safety_actor_coverage_complete
            ),
            "actor_selector_relevant_count": int(self.selection_result.relevant_count),
            "actor_selector_overflow": bool(self.actor_selector_overflow),
            "actor_selector_dropped_relevant_ids": list(
                self.selection_result.dropped_relevant_ids
            ),
            "cv_fallback_overflow": bool(self.cv_fallback_overflow),
            "cv_fallback_dropped_vehicle_ids": list(
                self.cv_fallback_dropped_vehicle_ids
            ),
            "forecast_actor_relevance": {
                vehicle_id: metadata.to_dict()
                for vehicle_id, metadata in self.selection_result.actor_metadata.items()
            },
        }


def _cv_uncertainty(cfg: Any, ego: VehicleState, actor: VehicleState) -> float:
    settings = actor_relevance_config(cfg)
    metadata = select_merge_relevant_actors(cfg, ego, [ego, actor], 1).actor_metadata.get(
        actor.vehicle_id
    )
    closing_speed = float(metadata.closing_speed if metadata is not None else 0.0)
    merge_penalty = (
        settings["cv_uncertainty_merge_corridor_penalty"]
        if is_auxiliary_edge(cfg, actor.edge_id) or is_ramp_edge(cfg, actor.edge_id)
        else 0.0
    )
    return float(
        np.clip(
            settings["cv_uncertainty_base"]
            + settings["cv_uncertainty_accel_scale"] * abs(float(actor.accel))
            + settings["cv_uncertainty_closing_speed_scale"] * closing_speed
            + merge_penalty,
            0.0,
            1.0,
        )
    )


def _safety_required_ids(
    cfg: Any,
    ego: VehicleState,
    vehicles: list[VehicleState],
    selection: ActorSelectionResult,
) -> list[str]:
    others = [vehicle for vehicle in vehicles if vehicle.vehicle_id != ego.vehicle_id]
    local = merge_local_stats(ego, [ego, *others], cfg)
    required: list[str] = []

    def add(vehicle_id: str) -> None:
        if vehicle_id and vehicle_id not in required:
            required.append(vehicle_id)

    add(str(local.target_front_vehicle_id or ""))
    add(str(local.target_rear_vehicle_id or ""))
    if others:
        add(min(others, key=lambda actor: bbox_gap(ego, actor)).vehicle_id)
        finite_ttc = [
            (relative_ttc(ego, actor), actor.vehicle_id)
            for actor in others
            if relative_ttc(ego, actor) < INF_TTC
        ]
        if finite_ttc:
            add(min(finite_ttc)[1])
    for vehicle_id in selection.relevant_actor_ids:
        add(vehicle_id)
    for vehicle_id, metadata in selection.actor_metadata.items():
        if metadata.relevant and metadata.role in {"auxiliary_local", "ramp_local"}:
            add(vehicle_id)
    return required


def _selected_prediction_rollouts(
    cfg: Any,
    vehicles_by_id: dict[str, VehicleState],
    prediction: dict[str, Any],
    horizon: int,
    dt: float,
) -> tuple[list[ForecastActorRollout], list[str], float]:
    trajectories = _prediction_array(prediction)
    selected_ids = [str(value or "") for value in prediction.get("selected_vehicle_ids", [])]
    if trajectories is None:
        return [], [], 0.0
    uncertainty = float(prediction.get("uncertainty", 0.0))
    actors: list[ForecastActorRollout] = []
    used_ids: list[str] = []
    for row, trajectory in enumerate(trajectories):
        vehicle_id = selected_ids[row] if row < len(selected_ids) else ""
        reference = vehicles_by_id.get(vehicle_id)
        if reference is None:
            continue
        states = trajectory_to_states(
            trajectory[:horizon],
            reference=reference,
            dt=dt,
            vehicle_id=vehicle_id,
        )
        if not states:
            continue
        actors.append(
            ForecastActorRollout(
                vehicle_id=vehicle_id,
                source=str(prediction.get("forecast_source", "wcdt")),
                trajectory=states,
                uncertainty=uncertainty,
                current_state=reference,
            )
        )
        used_ids.append(vehicle_id)
    return actors, used_ids, uncertainty


def build_forecast_rollout_bundle(
    cfg: Any,
    context: dict[str, Any],
    predictor: Any | None = None,
) -> ForecastRolloutBundle:
    ego = context.get("ego")
    if ego is None:
        raise ValueError("ForecastRolloutBundle requires an ego state.")
    vehicles = list(context.get("vehicles") or [])
    vehicles_by_id = {str(vehicle.vehicle_id): vehicle for vehicle in vehicles}
    forecast_source = str(cfg.forecast_features.get("source", "constant_velocity")).lower()
    max_actor_key = "wcdt_v2_max_agents" if forecast_source == "wcdt_v2" else "wcdt_v3_max_agents"
    max_actors = int(cfg.prediction.get(max_actor_key, cfg.prediction.max_pred_num))
    selection = select_merge_relevant_actors(cfg, ego, vehicles, max_actors)
    horizon = int(
        cfg.forecast_features.get(
            "horizon_steps",
            cfg.scenario.forecast_horizon_steps,
        )
    )
    dt = float(cfg.scenario.step_length)
    predicted_actors: list[ForecastActorRollout] = []
    selected_ids: list[str] = []
    wcdt_uncertainty = 0.0
    if predictor is not None:
        prediction = predictor.predict(context)
        predicted_actors, selected_ids, wcdt_uncertainty = _selected_prediction_rollouts(
            cfg,
            vehicles_by_id,
            prediction,
            horizon,
            dt,
        )

    if predictor is None:
        selected_ids = []
        safety_required = [
            vehicle.vehicle_id
            for vehicle in vehicles
            if vehicle.vehicle_id != ego.vehicle_id
        ]
    else:
        safety_required = _safety_required_ids(cfg, ego, vehicles, selection)
    fallback_candidates = [
        vehicle_id
        for vehicle_id in safety_required
        if vehicle_id not in selected_ids and vehicle_id in vehicles_by_id
    ]
    settings = actor_relevance_config(cfg)
    fallback_limit = max(0, int(settings["cv_fallback_max_actors"]))
    cv_fallback_overflow = len(fallback_candidates) > fallback_limit
    fallback_ids = fallback_candidates[:fallback_limit]
    dropped_fallback = fallback_candidates[fallback_limit:]
    fallback_actors: list[ForecastActorRollout] = []
    for vehicle_id in fallback_ids:
        actor = vehicles_by_id[vehicle_id]
        uncertainty = _cv_uncertainty(cfg, ego, actor)
        fallback_actors.append(
            ForecastActorRollout(
                vehicle_id=vehicle_id,
                source="constant_velocity",
                trajectory=route_aware_constant_velocity_rollout(
                    actor,
                    horizon,
                    dt,
                    cfg,
                )[0],
                uncertainty=uncertainty,
                relevance_reasons=tuple(
                    selection.actor_metadata.get(vehicle_id).relevance_reasons
                    if vehicle_id in selection.actor_metadata
                    else ()
                ),
                current_state=actor,
            )
        )
    if predictor is None:
        fallback_ids = safety_required
        dropped_fallback = []
        cv_fallback_overflow = False
        fallback_actors = []
        for vehicle_id in fallback_ids:
            actor = vehicles_by_id[vehicle_id]
            fallback_actors.append(
                ForecastActorRollout(
                    vehicle_id=vehicle_id,
                    source="constant_velocity",
                    trajectory=route_aware_constant_velocity_rollout(
                        actor,
                        horizon,
                        dt,
                        cfg,
                    )[0],
                    uncertainty=_cv_uncertainty(cfg, ego, actor),
                    current_state=actor,
                )
            )

    metadata = selection.actor_metadata
    for actor in predicted_actors:
        item = metadata.get(actor.vehicle_id)
        actor.relevance_reasons = tuple(item.relevance_reasons if item else ())
    actors = [*predicted_actors, *fallback_actors]
    actor_ids = {actor.vehicle_id for actor in actors}
    wcdt_required_complete = bool(
        not selection.overflow
        and all(vehicle_id in selected_ids for vehicle_id in selection.relevant_actor_ids)
    )
    safety_complete = bool(
        not cv_fallback_overflow
        and all(vehicle_id in actor_ids for vehicle_id in safety_required)
    )
    cv_uncertainty = max(
        (actor.uncertainty for actor in fallback_actors),
        default=0.0,
    )
    combined_uncertainty = max(
        (actor.uncertainty for actor in actors),
        default=0.0,
    )
    return ForecastRolloutBundle(
        actors=actors,
        selection_result=selection,
        wcdt_uncertainty=wcdt_uncertainty,
        cv_fallback_uncertainty=float(cv_uncertainty),
        combined_uncertainty=float(combined_uncertainty),
        wcdt_selected_vehicle_ids=selected_ids,
        cv_fallback_vehicle_ids=list(fallback_ids),
        safety_required_vehicle_ids=list(safety_required),
        wcdt_required_actor_coverage_complete=wcdt_required_complete,
        forecast_safety_actor_coverage_complete=safety_complete,
        actor_selector_overflow=bool(selection.overflow),
        cv_fallback_overflow=bool(cv_fallback_overflow),
        cv_fallback_dropped_vehicle_ids=list(dropped_fallback),
        actor_sources={actor.vehicle_id: actor.source for actor in actors},
    )


def get_or_build_forecast_rollout_bundle(
    cfg: Any,
    context: dict[str, Any],
    predictor: Any | None = None,
) -> ForecastRolloutBundle:
    key = (
        getattr(predictor, "checkpoint_path", None),
        str(cfg.forecast_features.get("source", "constant_velocity")),
    )
    cache = context.setdefault("_forecast_rollout_bundle_cache", {})
    if key not in cache:
        started = time.perf_counter()
        cache[key] = build_forecast_rollout_bundle(cfg, context, predictor)
        tracker = context.get("performance_tracker")
        if tracker is not None:
            tracker.add_time("forecast_inference_time", time.perf_counter() - started)
            tracker.increment("forecast_forwards")
    return cache[key]
