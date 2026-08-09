from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
from typing import Any

from safe_rl.risk.merge_local import merge_local_stats
from safe_rl.prediction.candidate_conflict import (
    CANDIDATE_CONFLICT_ORACLE_VERSION,
    candidate_union_conflict_oracle,
)
from safe_rl.sim.metrics import INF_TTC, bbox_gap
from safe_rl.sim.scenario_semantics import (
    auxiliary_lane_index,
    distance_to_taper,
    is_auxiliary_edge,
    is_ramp_edge,
    is_target_lane,
    merge_corridor_progress,
    with_cached_scenario_semantics,
)
from safe_rl.sim.types import VehicleState


ACTOR_SELECTION_VERSION = "merge_relevance_v2"
ACTOR_SELECTION_VERSION_V2 = ACTOR_SELECTION_VERSION
ACTOR_SELECTION_VERSION_V3 = "candidate_union_conflict_relevance_v3"
ACTOR_SELECTION_VERSION_V4 = "merge_conflict_relevance_v4_lane_aware"
SUPPORTED_ACTOR_SELECTION_VERSIONS = frozenset(
    {
        ACTOR_SELECTION_VERSION_V2,
        ACTOR_SELECTION_VERSION_V3,
        ACTOR_SELECTION_VERSION_V4,
    }
)


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if hasattr(value, "items"):
        return {str(key): _plain(item) for key, item in value.items()}
    return value


def actor_relevance_config(
    cfg: Any,
    *,
    selector_scope: str = "prediction",
) -> dict[str, Any]:
    """Resolve a selector without conflating forecast and ACCVP contracts.

    ``prediction`` is frozen by WcDT checkpoint lineage. ``accvp`` selects
    task actors for counterfactual data and Candidate Table inference. ACCVP
    falls back to the prediction selector for legacy configurations.
    """

    if selector_scope == "prediction":
        configured = cfg.prediction.get("actor_relevance", {})
    elif selector_scope == "accvp":
        configured = cfg.accvp.get("actor_relevance")
        if configured is None:
            configured = cfg.prediction.get("actor_relevance", {})
    else:
        raise ValueError(
            "selector_scope must be 'prediction' or 'accvp': "
            f"{selector_scope!r}"
        )
    version = str(configured.get("version", ACTOR_SELECTION_VERSION))
    if version not in SUPPORTED_ACTOR_SELECTION_VERSIONS:
        raise ValueError(
            f"unsupported {selector_scope} actor_relevance.version: "
            f"{version!r}; supported={sorted(SUPPORTED_ACTOR_SELECTION_VERSIONS)}"
        )
    result = {
        "version": version,
        "current_gap_distance": float(configured.get("current_gap_distance", 45.0)),
        "effective_gap_distance": float(configured.get("effective_gap_distance", 35.0)),
        "ttc_threshold": float(configured.get("ttc_threshold", 5.0)),
        "local_actor_distance": float(configured.get("local_actor_distance", 45.0)),
        "nearest_conflict_distance": float(configured.get("nearest_conflict_distance", 30.0)),
        "critical_taper_distance": float(configured.get("critical_taper_distance", 120.0)),
        "cv_fallback_max_actors": int(configured.get("cv_fallback_max_actors", 12)),
        "cv_uncertainty_base": float(configured.get("cv_uncertainty_base", 0.25)),
        "cv_uncertainty_accel_scale": float(configured.get("cv_uncertainty_accel_scale", 0.05)),
        "cv_uncertainty_closing_speed_scale": float(
            configured.get("cv_uncertainty_closing_speed_scale", 0.02)
        ),
        "cv_uncertainty_merge_corridor_penalty": float(
            configured.get("cv_uncertainty_merge_corridor_penalty", 0.10)
        ),
    }
    if version in {
        ACTOR_SELECTION_VERSION_V3,
        ACTOR_SELECTION_VERSION_V4,
    }:
        result.update(
            {
                "candidate_conflict_horizon_s": float(
                    configured.get(
                        "candidate_conflict_horizon_s",
                        cfg.accvp.get("response_horizon_s", 3.0),
                    )
                ),
                "candidate_conflict_surface_gap": float(
                    configured.get(
                        "candidate_conflict_surface_gap", 30.0
                    )
                ),
                "actor_longitudinal_accel_bound": float(
                    configured.get(
                        "actor_longitudinal_accel_bound", 2.0
                    )
                ),
                "unknown_topology_lateral_reach_m": float(
                    configured.get(
                        "unknown_topology_lateral_reach_m", 3.5
                    )
                ),
            }
        )
    return result


def actor_selection_config_hash(
    cfg: Any,
    *,
    selector_scope: str = "prediction",
) -> str:
    payload = json.dumps(
        _plain(
            actor_relevance_config(
                cfg,
                selector_scope=selector_scope,
            )
        ),
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ActorRelevance:
    vehicle_id: str
    role: str
    edge_id: str
    lane_id: str
    lane_index: int
    route_progress: float | None
    signed_longitudinal_gap: float | None
    current_surface_gap: float
    closing_speed: float
    effective_gap: float
    ttc: float
    relevance_reasons: tuple[str, ...]
    selection_priority: tuple[float, ...]
    relevant: bool
    relevance_class: str = "non_relevant"
    critical: bool = False
    contextual: bool = False
    candidate_conflict_eligible: bool = False
    conflict_candidate_ids: tuple[int, ...] = ()
    conflict_hypothesis_ids: tuple[str, ...] = ()
    conflict_surface_ids: tuple[str, ...] = ()
    earliest_conflict_time_s: float = INF_TTC
    earliest_overlap_time_s: float = INF_TTC
    minimum_swept_obb_gap: float = INF_TTC
    nearest_candidate_conflict: bool = False
    conflict_oracle_version: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ActorSelectionResult:
    selected_actor_ids: tuple[str, ...]
    relevant_actor_ids: tuple[str, ...]
    dropped_relevant_ids: tuple[str, ...]
    relevant_count: int
    overflow: bool
    actor_metadata: dict[str, ActorRelevance]
    version: str
    config_hash: str
    critical_actor_ids: tuple[str, ...] = ()
    contextual_actor_ids: tuple[str, ...] = ()
    dropped_critical_ids: tuple[str, ...] = ()
    contextual_truncated_ids: tuple[str, ...] = ()
    critical_count: int = 0
    contextual_count: int = 0
    critical_overflow: bool = False
    legal_candidate_ids: tuple[int, ...] = ()
    conflict_oracle_version: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_actor_ids": list(self.selected_actor_ids),
            "relevant_actor_ids": list(self.relevant_actor_ids),
            "dropped_relevant_ids": list(self.dropped_relevant_ids),
            "relevant_count": int(self.relevant_count),
            "overflow": bool(self.overflow),
            "critical_actor_ids": list(self.critical_actor_ids),
            "contextual_actor_ids": list(self.contextual_actor_ids),
            "dropped_critical_ids": list(self.dropped_critical_ids),
            "contextual_truncated_ids": list(self.contextual_truncated_ids),
            "critical_count": int(self.critical_count),
            "contextual_count": int(self.contextual_count),
            "critical_overflow": bool(self.critical_overflow),
            "legal_candidate_ids": list(self.legal_candidate_ids),
            "conflict_oracle_version": self.conflict_oracle_version,
            "actor_metadata": {
                vehicle_id: metadata.to_dict()
                for vehicle_id, metadata in self.actor_metadata.items()
            },
            "version": self.version,
            "config_hash": self.config_hash,
        }


def _role(
    cfg: Any,
    actor: VehicleState,
    target_front_id: str,
    target_rear_id: str,
    *,
    lane_aware_auxiliary: bool = False,
) -> str:
    if actor.vehicle_id == target_front_id:
        return "target_front"
    if actor.vehicle_id == target_rear_id:
        return "target_rear"
    if is_auxiliary_edge(cfg, actor.edge_id) and (
        not lane_aware_auxiliary
        or int(actor.lane_index)
        == int(auxiliary_lane_index(cfg, actor.edge_id))
    ):
        return "auxiliary_local"
    if is_ramp_edge(cfg, actor.edge_id):
        return "ramp_local"
    if is_target_lane(cfg, actor.edge_id, actor.lane_index):
        return "target_lane_other"
    return "other"


def _priority(metadata: ActorRelevance) -> tuple[float, ...]:
    role_priority = {
        "target_front": 0.0,
        "target_rear": 0.0,
        "nearest_conflict": 1.0,
        "auxiliary_local": 2.0,
        "ramp_local": 2.0,
        "target_lane_other": 3.0,
        "other": 4.0,
    }
    role_rank = (
        1.0
        if metadata.nearest_candidate_conflict
        else role_priority.get(metadata.role, 4.0)
    )
    return (
        {"critical": 0.0, "contextual": 1.0}.get(
            metadata.relevance_class, 2.0
        ),
        role_rank,
        (
            metadata.earliest_conflict_time_s
            if metadata.earliest_conflict_time_s < INF_TTC
            else INF_TTC
        ),
        metadata.ttc if metadata.ttc < INF_TTC else INF_TTC,
        metadata.effective_gap,
        metadata.current_surface_gap,
        metadata.minimum_swept_obb_gap,
    )


def reclassify_v3_telemetry_for_selector_v4(
    cfg: Any,
    selector_payload: dict[str, Any],
    *,
    ego_taper_distance: float,
    max_actors: int = 12,
) -> ActorSelectionResult:
    """Exactly replay Selector-v4 from full Selector-v3 actor telemetry.

    Candidate tubes, TTC and geometric evidence are unchanged between v3 and
    v4. This path is valid only when the v3 payload contains metadata for the
    full current vehicle set; callers must verify that input coverage and the
    frozen v3 selector hash match before using it.
    """

    if str(selector_payload.get("version", "")) != ACTOR_SELECTION_VERSION_V3:
        raise ValueError("Selector-v4 telemetry replay requires Selector-v3 input")
    if str(selector_payload.get("conflict_oracle_version", "")) != (
        CANDIDATE_CONFLICT_ORACLE_VERSION
    ):
        raise ValueError("Selector-v4 telemetry replay conflict oracle mismatch")
    raw_metadata = dict(selector_payload.get("actor_metadata", {}) or {})
    if not raw_metadata:
        raise ValueError("Selector-v4 telemetry replay requires full actor metadata")

    settings = actor_relevance_config(cfg, selector_scope="accvp")
    metadata_map: dict[str, ActorRelevance] = {}
    for vehicle_id, raw in raw_metadata.items():
        values = dict(raw)
        for field_name in (
            "relevance_reasons",
            "selection_priority",
            "conflict_candidate_ids",
            "conflict_hypothesis_ids",
            "conflict_surface_ids",
        ):
            values[field_name] = tuple(values.get(field_name, ()) or ())
        item = ActorRelevance(**values)
        role = item.role
        if role not in {"target_front", "target_rear"} and is_auxiliary_edge(
            cfg, item.edge_id
        ):
            if int(item.lane_index) == int(
                auxiliary_lane_index(cfg, item.edge_id)
            ):
                role = "auxiliary_local"
            elif is_target_lane(cfg, item.edge_id, item.lane_index):
                role = "target_lane_other"
            else:
                role = "other"
        reasons = list(item.relevance_reasons)
        if role not in {"auxiliary_local", "ramp_local"}:
            reasons = [reason for reason in reasons if reason != "merge_local"]
        reasons = list(dict.fromkeys(reasons))
        critical = bool(
            role in {"target_front", "target_rear"}
            or item.candidate_conflict_eligible
            or item.nearest_candidate_conflict
            or "lowest_ttc" in reasons
            or (
                role in {"auxiliary_local", "ramp_local"}
                and item.current_surface_gap
                <= float(settings["local_actor_distance"])
                and float(ego_taper_distance)
                <= float(settings["critical_taper_distance"])
            )
        )
        relevant = bool(critical or reasons)
        relevance_class = (
            "critical"
            if critical
            else ("contextual" if relevant else "non_relevant")
        )
        normalized = replace(
            item,
            role=role,
            relevance_reasons=tuple(reasons),
            relevant=relevant,
            relevance_class=relevance_class,
            critical=critical,
            contextual=relevance_class == "contextual",
            selection_priority=(),
        )
        metadata_map[str(vehicle_id)] = replace(
            normalized,
            selection_priority=_priority(normalized),
        )
    ordered = sorted(
        metadata_map.values(),
        key=lambda item: (*item.selection_priority, item.vehicle_id),
    )
    critical = [item for item in ordered if item.critical]
    contextual = [item for item in ordered if item.contextual]
    relevant = [*critical, *contextual]
    selected = ordered[: max(0, int(max_actors))]
    selected_ids = tuple(item.vehicle_id for item in selected)
    dropped_critical = tuple(
        item.vehicle_id
        for item in critical
        if item.vehicle_id not in selected_ids
    )
    dropped_relevant = tuple(
        item.vehicle_id
        for item in relevant
        if item.vehicle_id not in selected_ids
    )
    return ActorSelectionResult(
        selected_actor_ids=selected_ids,
        relevant_actor_ids=tuple(item.vehicle_id for item in relevant),
        dropped_relevant_ids=dropped_relevant,
        relevant_count=len(relevant),
        overflow=bool(dropped_relevant),
        actor_metadata=metadata_map,
        version=ACTOR_SELECTION_VERSION_V4,
        config_hash=actor_selection_config_hash(
            cfg, selector_scope="accvp"
        ),
        critical_actor_ids=tuple(item.vehicle_id for item in critical),
        contextual_actor_ids=tuple(item.vehicle_id for item in contextual),
        dropped_critical_ids=dropped_critical,
        contextual_truncated_ids=tuple(
            item.vehicle_id
            for item in contextual
            if item.vehicle_id not in selected_ids
        ),
        critical_count=len(critical),
        contextual_count=len(contextual),
        critical_overflow=bool(dropped_critical),
        legal_candidate_ids=tuple(
            int(value)
            for value in selector_payload.get("legal_candidate_ids", ())
        ),
        conflict_oracle_version=CANDIDATE_CONFLICT_ORACLE_VERSION,
    )


@with_cached_scenario_semantics
def select_merge_relevant_actors(
    cfg: Any,
    ego: VehicleState,
    current_vehicles: list[VehicleState],
    max_actors: int,
    *,
    selector_scope: str = "prediction",
) -> ActorSelectionResult:
    """Select merge-relevant actors using only the current decision state.

    ``merge_relevance_v2`` is intentionally retained byte-for-byte at the
    semantic decision boundary for historical reproduction. Selector-v3
    restricts the critical class to the merge conflict surface; global
    gap/TTC heuristics may make an unrelated actor contextual, but never
    critical unless it is the true nearest geometric conflict. Selector-v4
    preserves that candidate-union rule while restricting
    ``auxiliary_local`` to the configured auxiliary lane.
    """

    settings = actor_relevance_config(
        cfg,
        selector_scope=selector_scope,
    )
    horizon_seconds = float(cfg.scenario.forecast_horizon_steps) * float(cfg.scenario.step_length)
    vehicles = [
        vehicle
        for vehicle in current_vehicles
        if vehicle.vehicle_id != ego.vehicle_id
    ]
    local = merge_local_stats(ego, [ego, *vehicles], cfg)
    target_front_id = str(local.target_front_vehicle_id or "")
    target_rear_id = str(local.target_rear_vehicle_id or "")
    ego_progress = merge_corridor_progress(cfg, ego)

    base: dict[str, ActorRelevance] = {}
    nearest_id = ""
    nearest_gap = INF_TTC
    lowest_ttc_id = ""
    lowest_ttc = INF_TTC
    ego_taper_distance = float(distance_to_taper(cfg, ego))
    conflict_evidence = {}
    legal_candidate_ids: tuple[int, ...] = ()
    if settings["version"] in {
        ACTOR_SELECTION_VERSION_V3,
        ACTOR_SELECTION_VERSION_V4,
    }:
        conflict_evidence, legal_candidate_ids = (
            candidate_union_conflict_oracle(
                cfg,
                ego,
                current_vehicles,
                selector_scope=selector_scope,
            )
        )
    for actor in vehicles:
        actor_progress = merge_corridor_progress(cfg, actor)
        signed_gap = (
            None
            if ego_progress is None or actor_progress is None
            else float(actor_progress - ego_progress)
        )
        geometric_gap = float(bbox_gap(ego, actor))
        if geometric_gap < nearest_gap:
            nearest_gap = geometric_gap
            nearest_id = str(actor.vehicle_id)
        surface_gap = (
            geometric_gap
            if signed_gap is None
            else max(
                0.0,
                abs(float(signed_gap)) - 0.5 * (float(ego.length) + float(actor.length)),
            )
        )
        if signed_gap is None:
            closing_speed = 0.0
        elif signed_gap >= 0.0:
            closing_speed = max(0.0, float(ego.speed - actor.speed))
        else:
            closing_speed = max(0.0, float(actor.speed - ego.speed))
        effective_gap = float(surface_gap - closing_speed * horizon_seconds)
        ttc = (
            float(surface_gap / closing_speed)
            if closing_speed > 1.0e-6
            else INF_TTC
        )
        if ttc < lowest_ttc:
            lowest_ttc = float(ttc)
            lowest_ttc_id = str(actor.vehicle_id)
        role = _role(
            cfg,
            actor,
            target_front_id,
            target_rear_id,
            lane_aware_auxiliary=(
                settings["version"] == ACTOR_SELECTION_VERSION_V4
            ),
        )
        conflict = conflict_evidence.get(str(actor.vehicle_id))
        reasons: list[str] = []
        if surface_gap <= settings["current_gap_distance"]:
            reasons.append("current_gap")
        if effective_gap <= settings["effective_gap_distance"]:
            reasons.append("effective_gap")
        if ttc < INF_TTC and ttc <= settings["ttc_threshold"]:
            reasons.append("ttc")
        if role in {"auxiliary_local", "ramp_local"} and surface_gap <= settings["local_actor_distance"]:
            reasons.append("merge_local")
        if bool(
            getattr(conflict, "candidate_conflict_eligible", False)
        ):
            reasons.append("candidate_union_conflict")
        if bool(
            getattr(conflict, "nearest_candidate_conflict", False)
        ):
            reasons.append("nearest_candidate_conflict")
        relevant = bool(reasons)
        metadata = ActorRelevance(
            vehicle_id=str(actor.vehicle_id),
            role=role,
            edge_id=str(actor.edge_id),
            lane_id=str(actor.lane_id),
            lane_index=int(actor.lane_index),
            route_progress=actor_progress,
            signed_longitudinal_gap=signed_gap,
            current_surface_gap=surface_gap,
            closing_speed=float(closing_speed),
            effective_gap=effective_gap,
            ttc=ttc,
            relevance_reasons=tuple(reasons),
            selection_priority=(),
            relevant=relevant,
            relevance_class="contextual" if relevant else "non_relevant",
            contextual=relevant,
            candidate_conflict_eligible=bool(
                getattr(conflict, "candidate_conflict_eligible", False)
            ),
            conflict_candidate_ids=tuple(
                getattr(conflict, "conflict_candidate_ids", ())
            ),
            conflict_hypothesis_ids=tuple(
                getattr(conflict, "conflict_hypothesis_ids", ())
            ),
            conflict_surface_ids=tuple(
                getattr(conflict, "conflict_surface_ids", ())
            ),
            earliest_conflict_time_s=float(
                getattr(conflict, "earliest_conflict_time_s", INF_TTC)
            ),
            earliest_overlap_time_s=float(
                getattr(conflict, "earliest_overlap_time_s", INF_TTC)
            ),
            minimum_swept_obb_gap=float(
                getattr(conflict, "minimum_swept_obb_gap", INF_TTC)
            ),
            nearest_candidate_conflict=bool(
                getattr(conflict, "nearest_candidate_conflict", False)
            ),
            conflict_oracle_version=(
                CANDIDATE_CONFLICT_ORACLE_VERSION
                if conflict is not None
                else ""
            ),
        )
        base[str(actor.vehicle_id)] = metadata

    if (
        settings["version"] == ACTOR_SELECTION_VERSION_V2
        and nearest_id
        and nearest_gap <= settings["nearest_conflict_distance"]
    ):
        item = base[nearest_id]
        reasons = tuple(dict.fromkeys([*item.relevance_reasons, "nearest_conflict"]))
        base[nearest_id] = ActorRelevance(
            **{
                **item.to_dict(),
                "role": "nearest_conflict" if item.role == "other" else item.role,
                "relevance_reasons": reasons,
                "relevant": True,
                "relevance_class": "critical",
                "critical": True,
                "contextual": False,
            }
        )

    if settings["version"] in {
        ACTOR_SELECTION_VERSION_V3,
        ACTOR_SELECTION_VERSION_V4,
    }:
        lowest_ttc_id = ""
        lowest_ttc = INF_TTC
        conflict_surface = [
            item
            for item in base.values()
            if item.candidate_conflict_eligible
        ]
        finite_surface = [item for item in conflict_surface if item.ttc < INF_TTC]
        if finite_surface:
            lowest_ttc_id = min(
                finite_surface,
                key=lambda item: (item.ttc, item.effective_gap, item.vehicle_id),
            ).vehicle_id
            lowest_ttc = float(base[lowest_ttc_id].ttc)

    if lowest_ttc_id and lowest_ttc < INF_TTC:
        item = base[lowest_ttc_id]
        reasons = tuple(dict.fromkeys([*item.relevance_reasons, "lowest_ttc"]))
        base[lowest_ttc_id] = ActorRelevance(
            **{
                **item.to_dict(),
                "relevance_reasons": reasons,
                "relevant": True,
                "relevance_class": "critical",
                "critical": True,
                "contextual": False,
            }
        )

    metadata_map: dict[str, ActorRelevance] = {}
    for vehicle_id, item in base.items():
        reasons = set(item.relevance_reasons)
        if settings["version"] in {
            ACTOR_SELECTION_VERSION_V3,
            ACTOR_SELECTION_VERSION_V4,
        }:
            critical = bool(
                item.critical
                or item.role in {"target_front", "target_rear"}
                or item.candidate_conflict_eligible
                or "lowest_ttc" in reasons
                or (
                    item.role in {"auxiliary_local", "ramp_local"}
                    and item.current_surface_gap
                    <= settings["local_actor_distance"]
                    and ego_taper_distance <= settings["critical_taper_distance"]
                )
            )
        else:
            critical = bool(
                item.critical
                or (item.role in {"target_front", "target_rear"} and item.relevant)
                or "ttc" in reasons
                or "effective_gap" in reasons
                or "nearest_conflict" in reasons
                or "lowest_ttc" in reasons
                or (
                    item.role in {"auxiliary_local", "ramp_local"}
                    and item.relevant
                    and ego_taper_distance <= settings["critical_taper_distance"]
                )
            )
        relevance_class = (
            "critical"
            if critical
            else ("contextual" if item.relevant else "non_relevant")
        )
        normalized = ActorRelevance(
            **{
                **item.to_dict(),
                "critical": critical,
                "contextual": relevance_class == "contextual",
                "relevance_class": relevance_class,
                "relevant": bool(critical or item.relevant),
            }
        )
        priority = _priority(normalized)
        metadata_map[vehicle_id] = ActorRelevance(
            **{**normalized.to_dict(), "selection_priority": priority}
        )
    ordered = sorted(
        metadata_map.values(),
        key=lambda item: (*item.selection_priority, item.vehicle_id),
    )
    critical = [item for item in ordered if item.critical]
    contextual = [item for item in ordered if item.contextual]
    relevant = [*critical, *contextual]
    selected = ordered[: max(0, int(max_actors))]
    selected_ids = tuple(item.vehicle_id for item in selected)
    dropped = tuple(item.vehicle_id for item in relevant if item.vehicle_id not in selected_ids)
    dropped_critical = tuple(
        item.vehicle_id for item in critical if item.vehicle_id not in selected_ids
    )
    contextual_truncated = tuple(
        item.vehicle_id for item in contextual if item.vehicle_id not in selected_ids
    )
    critical_overflow = len(critical) > int(max_actors)
    return ActorSelectionResult(
        selected_actor_ids=selected_ids,
        relevant_actor_ids=tuple(item.vehicle_id for item in relevant),
        dropped_relevant_ids=dropped,
        relevant_count=len(relevant),
        overflow=critical_overflow,
        actor_metadata=metadata_map,
        version=str(settings["version"]),
        config_hash=actor_selection_config_hash(
            cfg,
            selector_scope=selector_scope,
        ),
        critical_actor_ids=tuple(item.vehicle_id for item in critical),
        contextual_actor_ids=tuple(item.vehicle_id for item in contextual),
        dropped_critical_ids=dropped_critical,
        contextual_truncated_ids=contextual_truncated,
        critical_count=len(critical),
        contextual_count=len(contextual),
        critical_overflow=critical_overflow,
        legal_candidate_ids=legal_candidate_ids,
        conflict_oracle_version=(
            CANDIDATE_CONFLICT_ORACLE_VERSION
            if settings["version"]
            in {
                ACTOR_SELECTION_VERSION_V3,
                ACTOR_SELECTION_VERSION_V4,
            }
            else ""
        ),
    )
