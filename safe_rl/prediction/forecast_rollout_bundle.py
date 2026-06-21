from __future__ import annotations

import time
import hashlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from safe_rl.prediction.actor_selector import (
    ActorSelectionResult,
    actor_relevance_config,
    select_merge_relevant_actors,
)
from safe_rl.prediction.trajectory_postprocess import (
    TRAJECTORY_POSTPROCESS_VERSION,
    trajectory_to_states,
)
from safe_rl.risk.merge_local import merge_local_stats, route_aware_constant_velocity_rollout
from safe_rl.sim.metrics import INF_TTC, bbox_gap, drac, relative_ttc
from safe_rl.sim.scenario_semantics import is_auxiliary_edge, is_ramp_edge
from safe_rl.sim.types import VehicleState


FORECAST_ROLLOUT_BUNDLE_VERSION = "hybrid_bundle_v2"


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


def _prediction_modes(prediction: dict[str, Any]) -> np.ndarray | None:
    """Return [mode, actor, horizon, state] without averaging trajectories."""

    trajectories = prediction.get("future_trajectories")
    if trajectories is None:
        return None
    if hasattr(trajectories, "detach"):
        trajectories = trajectories.detach().cpu().numpy()
    values = np.asarray(trajectories, dtype=np.float32)
    if values.ndim == 5:
        values = values[0]
    if values.ndim == 4:  # [actor, mode, horizon, state]
        return np.transpose(values, (1, 0, 2, 3))
    if values.ndim == 3:
        return values[None, ...]
    return None


@dataclass
class ForecastActorRollout:
    vehicle_id: str
    source: str
    trajectory: list[VehicleState]
    uncertainty: float
    relevance_reasons: tuple[str, ...] = ()
    current_state: VehicleState | None = None


@dataclass
class ActorModeForecast:
    vehicle_id: str
    source: str
    trajectories: list[list[VehicleState]]
    probabilities: np.ndarray
    uncertainty: float
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
    critical_wcdt_coverage_complete: bool
    combined_critical_coverage_complete: bool
    actor_selector_overflow: bool
    cv_fallback_overflow: bool
    cv_fallback_dropped_vehicle_ids: list[str]
    version: str = FORECAST_ROLLOUT_BUNDLE_VERSION
    actor_sources: dict[str, str] = field(default_factory=dict)
    actor_mode_forecasts: list[ActorModeForecast] = field(default_factory=list)
    cv_fallback_actors: list[ForecastActorRollout] = field(default_factory=list)
    joint_world_count: int = 0
    joint_world_seed: tuple[int, int, int] = (0, 0, 0)
    _joint_world_cache: list[list[ForecastActorRollout]] | None = field(default=None, init=False, repr=False)

    def rollout_lists(self) -> list[list[VehicleState]]:
        return [actor.trajectory for actor in self.actors if actor.trajectory]

    def actor_by_id(self, vehicle_id: str) -> ForecastActorRollout | None:
        return next((actor for actor in self.actors if actor.vehicle_id == vehicle_id), None)

    def joint_world_actor_sets(self) -> list[list[ForecastActorRollout]]:
        """Return deterministic independent-mode world samples for safety metrics."""

        if self._joint_world_cache is not None:
            return self._joint_world_cache
        if not self.actor_mode_forecasts:
            self._joint_world_cache = [list(self.actors)]
            return self._joint_world_cache
        worlds: list[list[ForecastActorRollout]] = []
        world_count = max(1, int(self.joint_world_count))
        run_seed, episode_seed, decision_index = self.joint_world_seed
        for world_index in range(world_count):
            actors = list(self.cv_fallback_actors)
            for forecast in self.actor_mode_forecasts:
                digest = hashlib.sha256(
                    f"{run_seed}:{episode_seed}:{decision_index}:{forecast.vehicle_id}:{world_index}".encode("utf-8")
                ).digest()
                random_value = int.from_bytes(digest[:8], "little") / float(2**64)
                cumulative = np.cumsum(forecast.probabilities)
                mode_index = int(np.searchsorted(cumulative, random_value, side="right"))
                mode_index = min(max(mode_index, 0), len(forecast.trajectories) - 1)
                actors.append(
                    ForecastActorRollout(
                        vehicle_id=forecast.vehicle_id,
                        source=forecast.source,
                        trajectory=forecast.trajectories[mode_index],
                        uncertainty=forecast.uncertainty,
                        current_state=forecast.current_state,
                    )
                )
            worlds.append(actors)
        self._joint_world_cache = worlds
        return worlds

    def trace_fields(self) -> dict[str, Any]:
        return {
            "forecast_rollout_bundle_version": self.version,
            "trajectory_postprocess_version": TRAJECTORY_POSTPROCESS_VERSION,
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
            "critical_actor_count": int(self.selection_result.critical_count),
            "contextual_actor_count": int(self.selection_result.contextual_count),
            "critical_actor_overflow": bool(self.selection_result.critical_overflow),
            "critical_dropped_actor_ids": list(
                self.selection_result.dropped_critical_ids
            ),
            "contextual_actor_truncated_count": int(
                len(self.selection_result.contextual_truncated_ids)
            ),
            "critical_wcdt_coverage_complete": bool(
                self.critical_wcdt_coverage_complete
            ),
            "combined_critical_coverage_complete": bool(
                self.combined_critical_coverage_complete
            ),
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
            "forecast_mode_aggregation": "per_actor_joint_world_v1",
            "forecast_joint_world_count": int(self.joint_world_count),
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
) -> tuple[list[ForecastActorRollout], list[str], float, list[ActorModeForecast]]:
    trajectories = _prediction_array(prediction)
    modes = _prediction_modes(prediction)
    selected_ids = [str(value or "") for value in prediction.get("selected_vehicle_ids", [])]
    if trajectories is None:
        return [], [], 0.0, []
    uncertainty = float(prediction.get("uncertainty", 0.0))
    actor_uncertainty = prediction.get("actor_uncertainty", [])
    if hasattr(actor_uncertainty, "detach"):
        actor_uncertainty = actor_uncertainty.detach().cpu().numpy()
    actor_uncertainty = np.asarray(actor_uncertainty, dtype=np.float32).reshape(-1)
    actor_mode_probabilities = prediction.get("actor_mode_probabilities")
    if hasattr(actor_mode_probabilities, "detach"):
        actor_mode_probabilities = actor_mode_probabilities.detach().cpu().numpy()
    actor_mode_probabilities = np.asarray(actor_mode_probabilities, dtype=np.float32)
    if actor_mode_probabilities.ndim == 3:
        actor_mode_probabilities = actor_mode_probabilities[0]
    actor_mode_forecasts: list[ActorModeForecast] = []
    actors: list[ForecastActorRollout] = []
    used_ids: list[str] = []
    if modes is not None and modes.shape[0] > 1:
        for row, vehicle_id in enumerate(selected_ids[: modes.shape[1]]):
            reference = vehicles_by_id.get(vehicle_id)
            if reference is None:
                continue
            trajectories_for_actor: list[list[VehicleState]] = []
            for mode_index in range(modes.shape[0]):
                states = trajectory_to_states(
                    modes[mode_index, row, :horizon],
                    reference=reference,
                    dt=dt,
                    vehicle_id=vehicle_id,
                    config=cfg,
                )
                if states:
                    trajectories_for_actor.append(states)
            if not trajectories_for_actor:
                continue
            probabilities = (
                actor_mode_probabilities[row, : len(trajectories_for_actor)]
                if actor_mode_probabilities.ndim == 2 and row < actor_mode_probabilities.shape[0]
                else np.full((len(trajectories_for_actor),), 1.0 / len(trajectories_for_actor), dtype=np.float32)
            )
            probabilities = probabilities / max(float(np.sum(probabilities)), 1.0e-8)
            actor_forecast = ActorModeForecast(
                vehicle_id=vehicle_id,
                source=str(prediction.get("forecast_source", "wcdt")),
                trajectories=trajectories_for_actor,
                probabilities=probabilities.astype(np.float32),
                uncertainty=float(actor_uncertainty[row]) if row < actor_uncertainty.size else uncertainty,
                current_state=reference,
            )
            actor_mode_forecasts.append(actor_forecast)
            top_mode = int(np.argmax(probabilities))
            actors.append(
                ForecastActorRollout(
                    vehicle_id=vehicle_id,
                    source=actor_forecast.source,
                    trajectory=actor_forecast.trajectories[top_mode],
                    uncertainty=actor_forecast.uncertainty,
                    current_state=reference,
                )
            )
            used_ids.append(vehicle_id)
        return actors, used_ids, uncertainty, actor_mode_forecasts
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
            config=cfg,
        )
        if not states:
            continue
        actors.append(
            ForecastActorRollout(
                vehicle_id=vehicle_id,
                source=str(prediction.get("forecast_source", "wcdt")),
                trajectory=states,
                uncertainty=float(actor_uncertainty[row]) if row < actor_uncertainty.size else uncertainty,
                current_state=reference,
            )
        )
        used_ids.append(vehicle_id)
    return actors, used_ids, uncertainty, actor_mode_forecasts


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
    if forecast_source == "wcdt_v2":
        max_actors = int(cfg.prediction.get("wcdt_v2_max_agents", cfg.prediction.max_pred_num))
    elif forecast_source == "wcdt_v3":
        max_actors = int(cfg.prediction.get("wcdt_v3_max_agents", cfg.prediction.max_pred_num))
    elif forecast_source == "wcdt":
        max_actors = int(cfg.prediction.get("wcdt_v1_max_agents", cfg.prediction.max_pred_num))
    else:
        max_actors = int(cfg.prediction.max_pred_num)
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
    actor_mode_forecasts: list[ActorModeForecast] = []
    if predictor is not None:
        prediction = predictor.predict(context)
        predicted_actors, selected_ids, wcdt_uncertainty, actor_mode_forecasts = _selected_prediction_rollouts(
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
    critical_wcdt_complete = bool(
        not selection.critical_overflow
        and all(vehicle_id in selected_ids for vehicle_id in selection.critical_actor_ids)
    )
    wcdt_required_complete = critical_wcdt_complete
    safety_complete = bool(
        not cv_fallback_overflow
        and all(vehicle_id in actor_ids for vehicle_id in safety_required)
    )
    combined_critical_complete = bool(
        not cv_fallback_overflow
        and all(vehicle_id in actor_ids for vehicle_id in selection.critical_actor_ids)
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
        critical_wcdt_coverage_complete=critical_wcdt_complete,
        combined_critical_coverage_complete=combined_critical_complete,
        actor_selector_overflow=bool(selection.overflow),
        cv_fallback_overflow=bool(cv_fallback_overflow),
        cv_fallback_dropped_vehicle_ids=list(dropped_fallback),
        actor_sources={actor.vehicle_id: actor.source for actor in actors},
        actor_mode_forecasts=actor_mode_forecasts,
        cv_fallback_actors=fallback_actors,
        joint_world_count=int(
            cfg.prediction.get("wcdt_v1_mode_aggregation", {}).get("joint_world_count", 32)
        ) if actor_mode_forecasts else 0,
        joint_world_seed=(
            int(cfg.run.get("seed", 0)),
            int(context.get("episode_seed", cfg.run.get("seed", 0))),
            int(context.get("decision_index", 0)),
        ),
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
