"""Selector-v3 input coverage and fixed-capacity contract audits."""

from __future__ import annotations

import json
import copy
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from safe_rl.accvp.contracts.schema import (
    read_json,
    stable_hash,
    write_json_atomic,
)
from safe_rl.prediction.actor_selector import (
    ACTOR_SELECTION_VERSION_V2,
    ACTOR_SELECTION_VERSION_V3,
    ActorSelectionResult,
    select_merge_relevant_actors,
)
from safe_rl.sim.scenario_semantics import lane_indices
from safe_rl.sim.scenario_semantics import distance_to_taper
from safe_rl.sim.types import VehicleState
from safe_rl.utils.config import REPO_ROOT, clone_with_overrides


SELECTOR_INPUT_COVERAGE_KIND = "accvp_selector_audit_input_coverage_v1"
SELECTOR_CONTRACT_AUDIT_KIND = "accvp_selector_contract_audit_v1"
SELECTOR_AUDIT_MODEL = "conservative_reachable_tube_v1"
SELECTOR_AUDIT_SAMPLE_LIMIT = 20


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (REPO_ROOT / value).resolve()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _metadata_rows(
    dataset_dir: str | Path,
) -> Iterable[tuple[dict[str, Any], dict[str, Any]]]:
    dataset = _resolve(dataset_dir)
    roots = [
        row
        for row in _jsonl(dataset / "manifests" / "roots.jsonl")
        if bool(row.get("complete", False))
    ]
    for root in roots:
        yield root, read_json(_resolve(str(root["metadata_path"])))


def selector_audit_input_coverage(
    config: Any,
    dataset_dir: str | Path,
) -> dict[str, Any]:
    rows = _metadata_rows(dataset_dir)
    root_count = 0
    missing_metadata = 0
    missing_history = 0
    selected_only_roots = 0
    missing_kinematic_fields = 0
    invalid_route_projection = 0
    missing_network_topology = 0
    internal_connector_fallback_roots = 0
    internal_connector_vehicle_rows = 0
    explicit_route_intent_roots = 0
    vehicle_rows = 0
    history_vehicle_rows = 0
    latest_counts: list[int] = []
    required = {
        "vehicle_id",
        "x",
        "y",
        "heading",
        "speed",
        "accel",
        "edge_id",
        "lane_id",
        "lane_index",
        "lane_pos",
        "length",
        "width",
        "route_position_valid",
    }
    for root, metadata in rows:
        root_count += 1
        if not metadata:
            missing_metadata += 1
            continue
        history = list(metadata.get("history_frames", []) or [])
        if not history:
            missing_history += 1
            continue
        latest = list(history[-1] or [])
        latest_counts.append(len(latest))
        if len(latest) <= int(metadata.get("selected_actor_count", 0)) + 1:
            selected_only_roots += 1
        root_has_intent = False
        root_missing_topology = False
        root_has_internal_connector = False
        for frame in history:
            for vehicle in list(frame or []):
                history_vehicle_rows += 1
                if not required.issubset(set(vehicle)):
                    missing_kinematic_fields += 1
        for vehicle in latest:
            vehicle_rows += 1
            if not bool(vehicle.get("route_position_valid", False)):
                invalid_route_projection += 1
            edge_id = str(vehicle.get("edge_id", ""))
            if not lane_indices(config, edge_id):
                if edge_id.startswith(":"):
                    root_has_internal_connector = True
                    internal_connector_vehicle_rows += 1
                else:
                    root_missing_topology = True
            if any(
                key in vehicle
                for key in ("route_id", "route_edges", "next_edge_id")
            ):
                root_has_intent = True
        missing_network_topology += int(root_missing_topology)
        internal_connector_fallback_roots += int(
            root_has_internal_connector
        )
        explicit_route_intent_roots += int(root_has_intent)
    full_state_pass = bool(
        root_count > 0
        and missing_metadata == 0
        and missing_history == 0
        and selected_only_roots == 0
        and missing_kinematic_fields == 0
        and invalid_route_projection == 0
        and missing_network_topology == 0
    )
    report = {
        "artifact_kind": SELECTOR_INPUT_COVERAGE_KIND,
        "dataset_dir": str(_resolve(dataset_dir)),
        "root_count": root_count,
        "latest_vehicle_row_count": vehicle_rows,
        "history_vehicle_row_count": history_vehicle_rows,
        "minimum_latest_vehicle_count": min(latest_counts) if latest_counts else 0,
        "maximum_latest_vehicle_count": max(latest_counts) if latest_counts else 0,
        "missing_metadata_root_count": missing_metadata,
        "missing_history_root_count": missing_history,
        "selected_only_latest_state_root_count": selected_only_roots,
        "missing_current_kinematic_field_row_count": missing_kinematic_fields,
        "invalid_route_projection_row_count": invalid_route_projection,
        "missing_network_topology_root_count": missing_network_topology,
        "internal_connector_swept_obb_fallback_root_count": (
            internal_connector_fallback_roots
        ),
        "internal_connector_latest_vehicle_row_count": (
            internal_connector_vehicle_rows
        ),
        "explicit_future_route_intent_root_count": explicit_route_intent_roots,
        "full_current_vehicle_state_available": full_state_pass,
        "unselected_actor_state_available": selected_only_roots == 0,
        "route_projection_complete": invalid_route_projection == 0,
        "network_topology_available": missing_network_topology == 0,
        "external_edge_topology_complete": missing_network_topology == 0,
        "internal_connector_fallback_model": (
            "current_obb_cv_longitudinal_and_lateral_reach_envelope_v1"
        ),
        "explicit_future_route_intent_available": (
            root_count > 0 and explicit_route_intent_roots == root_count
        ),
        "audit_model": SELECTOR_AUDIT_MODEL,
        "exact_route_intent_claim": False,
        "conservative_reachable_tube_audit_allowed": full_state_pass,
        "selector_only_recollection_required": not full_state_pass,
        "input_coverage_state": "pass" if full_state_pass else "fail",
    }
    report["report_fingerprint"] = stable_hash(report)
    return report


@dataclass
class _SelectionAccumulator:
    version: str
    state_count: int = 0
    critical_count_histogram: Counter[str] = field(default_factory=Counter)
    contextual_count_histogram: Counter[str] = field(default_factory=Counter)
    overflow_counts: dict[int, int] = field(
        default_factory=lambda: {6: 0, 8: 0}
    )
    dropped_counts: dict[int, int] = field(
        default_factory=lambda: {6: 0, 8: 0}
    )
    target_required: int = 0
    target_covered: dict[int, int] = field(
        default_factory=lambda: {6: 0, 8: 0}
    )
    conflict_required: int = 0
    conflict_covered: dict[int, int] = field(
        default_factory=lambda: {6: 0, 8: 0}
    )
    nearest_required: int = 0
    nearest_covered: dict[int, int] = field(
        default_factory=lambda: {6: 0, 8: 0}
    )
    overflow_examples: dict[int, list[dict[str, Any]]] = field(
        default_factory=lambda: {6: [], 8: []}
    )
    mandatory_target_not_critical_count: int = 0
    candidate_conflict_not_critical_count: int = 0

    def add(
        self,
        selection: ActorSelectionResult,
        *,
        state_id: str,
        provenance: dict[str, Any],
    ) -> None:
        self.state_count += 1
        self.critical_count_histogram[str(selection.critical_count)] += 1
        self.contextual_count_histogram[str(selection.contextual_count)] += 1
        critical_ids = list(selection.critical_actor_ids)
        target_ids = [
            vehicle_id
            for vehicle_id, metadata in selection.actor_metadata.items()
            if metadata.role in {"target_front", "target_rear"}
        ]
        conflict_ids = [
            vehicle_id
            for vehicle_id, metadata in selection.actor_metadata.items()
            if metadata.candidate_conflict_eligible
        ]
        nearest_ids = [
            vehicle_id
            for vehicle_id, metadata in selection.actor_metadata.items()
            if metadata.nearest_candidate_conflict
        ]
        self.target_required += len(target_ids)
        self.conflict_required += len(conflict_ids)
        self.nearest_required += len(nearest_ids)
        critical_set = set(critical_ids)
        self.mandatory_target_not_critical_count += sum(
            vehicle_id not in critical_set for vehicle_id in target_ids
        )
        self.candidate_conflict_not_critical_count += sum(
            vehicle_id not in critical_set for vehicle_id in conflict_ids
        )
        ordered = [
            *selection.critical_actor_ids,
            *selection.contextual_actor_ids,
        ]
        for capacity in (6, 8):
            selected = set(ordered[:capacity])
            dropped = [
                vehicle_id
                for vehicle_id in critical_ids
                if vehicle_id not in selected
            ]
            self.overflow_counts[capacity] += int(
                selection.critical_count > capacity
            )
            self.dropped_counts[capacity] += len(dropped)
            self.target_covered[capacity] += sum(
                vehicle_id in selected for vehicle_id in target_ids
            )
            self.conflict_covered[capacity] += sum(
                vehicle_id in selected for vehicle_id in conflict_ids
            )
            self.nearest_covered[capacity] += sum(
                vehicle_id in selected for vehicle_id in nearest_ids
            )
            if (
                dropped
                and len(self.overflow_examples[capacity])
                < SELECTOR_AUDIT_SAMPLE_LIMIT
            ):
                self.overflow_examples[capacity].append(
                    {
                        "state_id": state_id,
                        **provenance,
                        "capacity": capacity,
                        "critical_count": selection.critical_count,
                        "selected_actor_ids": ordered[:capacity],
                        "dropped_critical_actor_ids": dropped,
                        "dropped_critical_actors": [
                            selection.actor_metadata[
                                vehicle_id
                            ].to_dict()
                            for vehicle_id in dropped
                        ],
                    }
                )

    def capacity_report(self, capacity: int) -> dict[str, Any]:
        def rate(covered: int, required: int) -> float:
            return float(covered / required) if required else 1.0

        return {
            "capacity": int(capacity),
            "state_count": self.state_count,
            "critical_overflow_count": self.overflow_counts[capacity],
            "dropped_critical_count": self.dropped_counts[capacity],
            "target_front_rear_required_count": self.target_required,
            "mandatory_target_not_critical_count": (
                self.mandatory_target_not_critical_count
            ),
            "target_front_rear_coverage_rate": rate(
                self.target_covered[capacity], self.target_required
            ),
            "candidate_conflict_required_count": self.conflict_required,
            "candidate_conflict_not_critical_count": (
                self.candidate_conflict_not_critical_count
            ),
            "candidate_conflict_coverage_rate": rate(
                self.conflict_covered[capacity], self.conflict_required
            ),
            "nearest_conflict_required_count": self.nearest_required,
            "nearest_conflict_coverage_rate": rate(
                self.nearest_covered[capacity], self.nearest_required
            ),
            "critical_count_histogram": dict(
                self.critical_count_histogram
            ),
            "contextual_count_histogram": dict(
                self.contextual_count_histogram
            ),
            "overflow_examples": self.overflow_examples[capacity],
            "overflow_sample_limit": SELECTOR_AUDIT_SAMPLE_LIMIT,
        }


def _selection_configs(config: Any) -> tuple[Any, Any]:
    v2 = clone_with_overrides(
        config,
        {
            "accvp": {
                "actor_relevance": {
                    "version": ACTOR_SELECTION_VERSION_V2
                }
            }
        },
    )
    v3 = clone_with_overrides(
        config,
        {
            "accvp": {
                "actor_relevance": {
                    "version": ACTOR_SELECTION_VERSION_V3
                }
            }
        },
    )
    return v2, v3


def _states(metadata: dict[str, Any]) -> list[VehicleState]:
    history = list(metadata.get("history_frames", []) or [])
    if not history:
        raise ValueError("selector audit root has no full history")
    return [VehicleState(**row) for row in history[-1]]


def _audit_states(
    records: Iterable[tuple[str, list[VehicleState], dict[str, Any]]],
    *,
    v2_config: Any,
    v3_config: Any,
    v2_accumulator: _SelectionAccumulator,
    v3_accumulator: _SelectionAccumulator,
) -> None:
    for state_id, vehicles, provenance in records:
        ego = next(
            (vehicle for vehicle in vehicles if vehicle.vehicle_id == "ego"),
            None,
        )
        if ego is None:
            raise ValueError(f"selector audit state has no ego: {state_id}")
        state_provenance = dict(provenance)
        state_provenance.setdefault(
            "taper_distance_m",
            float(distance_to_taper(v3_config, ego)),
        )
        v2_accumulator.add(
            select_merge_relevant_actors(
                v2_config,
                ego,
                vehicles,
                max_actors=8,
                selector_scope="accvp",
            ),
            state_id=state_id,
            provenance=state_provenance,
        )
        v3_accumulator.add(
            select_merge_relevant_actors(
                v3_config,
                ego,
                vehicles,
                max_actors=8,
                selector_scope="accvp",
            ),
            state_id=state_id,
            provenance=state_provenance,
        )


def _capacity_pass(report: dict[str, Any]) -> bool:
    return bool(
        int(report["critical_overflow_count"]) == 0
        and int(report["dropped_critical_count"]) == 0
        and int(report["mandatory_target_not_critical_count"]) == 0
        and int(report["candidate_conflict_not_critical_count"]) == 0
        and float(report["target_front_rear_coverage_rate"]) == 1.0
        and float(report["candidate_conflict_coverage_rate"]) == 1.0
        and float(report["nearest_conflict_coverage_rate"]) == 1.0
    )


def run_selector_contract_audit(
    config: Any,
    dataset_dir: str | Path,
    *,
    targeted_states: Iterable[
        tuple[str, list[VehicleState], dict[str, Any]]
    ] = (),
    minimum_formal_roots: int = 5000,
) -> dict[str, Any]:
    input_coverage = selector_audit_input_coverage(config, dataset_dir)
    v2_config, v3_config = _selection_configs(config)
    formal_v2 = _SelectionAccumulator(ACTOR_SELECTION_VERSION_V2)
    formal_v3 = _SelectionAccumulator(ACTOR_SELECTION_VERSION_V3)
    formal_records = []
    for root, metadata in _metadata_rows(dataset_dir):
        formal_records.append(
            (
                str(root["root_id"]),
                _states(metadata),
                {
                    "scope": "formal_root_history",
                    "episode_seed": int(root.get("episode_seed", -1)),
                    "decision_index": int(
                        metadata.get("decision_index", -1)
                    ),
                },
            )
        )
    if input_coverage["input_coverage_state"] == "pass":
        _audit_states(
            formal_records,
            v2_config=v2_config,
            v3_config=v3_config,
            v2_accumulator=formal_v2,
            v3_accumulator=formal_v3,
        )
    targeted_v2 = _SelectionAccumulator(ACTOR_SELECTION_VERSION_V2)
    targeted_v3 = _SelectionAccumulator(ACTOR_SELECTION_VERSION_V3)
    targeted_records = list(targeted_states)
    if targeted_records:
        _audit_states(
            targeted_records,
            v2_config=v2_config,
            v3_config=v3_config,
            v2_accumulator=targeted_v2,
            v3_accumulator=targeted_v3,
        )

    combined_v2 = copy.deepcopy(formal_v2)
    combined_v3 = copy.deepcopy(formal_v3)
    if (
        input_coverage["input_coverage_state"] == "pass"
        and targeted_records
    ):
        _audit_states(
            targeted_records,
            v2_config=v2_config,
            v3_config=v3_config,
            v2_accumulator=combined_v2,
            v3_accumulator=combined_v3,
        )
    capacity6 = combined_v3.capacity_report(6)
    capacity8 = combined_v3.capacity_report(8)
    targeted_episode_keys = {
        (
            int(provenance.get("optimizer_seed", -1)),
            int(provenance.get("episode_seed", -1)),
        )
        for _state_id, _vehicles, provenance in targeted_records
    }
    prerequisites = {
        "input_coverage_pass": (
            input_coverage["input_coverage_state"] == "pass"
        ),
        "minimum_5000_formal_roots": (
            len(formal_records) >= int(minimum_formal_roots)
        ),
        "four_targeted_replays": len(targeted_episode_keys) == 4,
    }
    if all(prerequisites.values()) and _capacity_pass(capacity6):
        selected_capacity = 6
        audit_state = "pass"
    elif all(prerequisites.values()) and _capacity_pass(capacity8):
        selected_capacity = 8
        audit_state = "pass"
    else:
        selected_capacity = None
        audit_state = "blocked"
    report = {
        "artifact_kind": SELECTOR_CONTRACT_AUDIT_KIND,
        "protocol_id": "accvp-vnext-correctness-v2-selector3",
        "audit_state": audit_state,
        "selected_capacity": selected_capacity,
        "capacity_decision_rule": "strict_6_then_8_fail_closed_v1",
        "input_coverage": input_coverage,
        "prerequisites": prerequisites,
        "formal_root_count": len(formal_records),
        "targeted_replay_keys": [
            {
                "optimizer_seed": optimizer_seed,
                "episode_seed": episode_seed,
                "seed_role": "selector_diagnostic_only",
            }
            for optimizer_seed, episode_seed in sorted(
                targeted_episode_keys
            )
        ],
        "selector_v2_shadow": {
            "formal": formal_v2.capacity_report(6),
            "targeted": targeted_v2.capacity_report(6),
            "combined": combined_v2.capacity_report(6),
        },
        "selector_v3": {
            "formal_capacity_6": formal_v3.capacity_report(6),
            "formal_capacity_8": formal_v3.capacity_report(8),
            "targeted_capacity_6": targeted_v3.capacity_report(6),
            "targeted_capacity_8": targeted_v3.capacity_report(8),
            "combined_capacity_6": capacity6,
            "combined_capacity_8": capacity8,
        },
    }
    report["report_fingerprint"] = stable_hash(report)
    return report


def write_selector_contract_audit(
    config: Any,
    dataset_dir: str | Path,
    output_path: str | Path,
    **kwargs: Any,
) -> Path:
    output = _resolve(output_path)
    if output.exists():
        raise FileExistsError(output)
    report = run_selector_contract_audit(
        config,
        dataset_dir,
        **kwargs,
    )
    write_json_atomic(output, report)
    return output
