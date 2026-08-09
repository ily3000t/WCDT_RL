"""Strict fixed-capacity audit for lane-aware ACCVP Selector-v4."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
import math
from typing import Any, Iterable

import numpy as np

from safe_rl.accvp.contracts.schema import stable_hash


SELECTOR4_PROTOCOL_ID = "accvp-vnext-correctness-v3-selector4"
SELECTOR4_AUDIT_KIND = "accvp_selector_capacity_audit_v2"
SELECTOR4_AUDIT_IMPLEMENTATION = (
    "lane_aware_capacity_sweep_v4_recorded_overflow_capacity12"
)
SELECTOR4_CAPACITIES = (8, 10, 12)
SELECTOR4_MINIMUM_HEADROOM = 2
SELECTOR4_SAMPLE_LIMIT = 20


def _quantile(values: list[int] | list[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.quantile(np.asarray(values, dtype=np.float64), q))


def _latency_summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "p50": None,
            "p95": None,
            "p99": None,
            "max": None,
        }
    return {
        "count": len(values),
        "p50": _quantile(values, 0.50),
        "p95": _quantile(values, 0.95),
        "p99": _quantile(values, 0.99),
        "max": float(max(values)),
    }


def selection_audit_row(
    selection: Any,
    *,
    state_id: str,
    provenance: dict[str, Any],
    selector_latency_s: float,
    latency_semantics: str = "full_selector_runtime",
) -> dict[str, Any]:
    """Convert one full-state selector result into a compact immutable row."""

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
    lowest_ttc_ids = [
        vehicle_id
        for vehicle_id, metadata in selection.actor_metadata.items()
        if "lowest_ttc" in set(metadata.relevance_reasons)
    ]
    protected_ids = list(
        dict.fromkeys(
            [*target_ids, *conflict_ids, *nearest_ids, *lowest_ttc_ids]
        )
    )
    return {
        "state_id": str(state_id),
        "provenance": dict(provenance),
        "selector_latency_s": float(selector_latency_s),
        "latency_semantics": str(latency_semantics),
        "critical_count": int(selection.critical_count),
        "contextual_count": int(selection.contextual_count),
        "critical_actor_ids": critical_ids,
        "target_front_rear_ids": target_ids,
        "candidate_conflict_ids": conflict_ids,
        "nearest_conflict_ids": nearest_ids,
        "lowest_conflict_ttc_ids": lowest_ttc_ids,
        "protected_actor_ids": protected_ids,
        "critical_actors": [
            selection.actor_metadata[vehicle_id].to_dict()
            for vehicle_id in critical_ids
        ],
    }


@dataclass
class _Accumulator:
    state_count: int = 0
    critical_counts: list[int] = field(default_factory=list)
    mandatory_counts: list[int] = field(default_factory=list)
    contextual_counts: Counter[str] = field(default_factory=Counter)
    contextual_observed_state_count: int = 0
    selector_latencies: list[float] = field(default_factory=list)
    telemetry_reclassification_latencies: list[float] = field(
        default_factory=list
    )
    target_required: int = 0
    conflict_required: int = 0
    protected_required: int = 0
    overflow: Counter[int] = field(default_factory=Counter)
    dropped_mandatory: Counter[int] = field(default_factory=Counter)
    target_covered: Counter[int] = field(default_factory=Counter)
    conflict_covered: Counter[int] = field(default_factory=Counter)
    protected_covered: Counter[int] = field(default_factory=Counter)
    overflow_examples: dict[int, list[dict[str, Any]]] = field(
        default_factory=lambda: {
            capacity: [] for capacity in SELECTOR4_CAPACITIES
        }
    )

    def add(self, row: dict[str, Any]) -> None:
        self.state_count += 1
        critical_ids = [str(value) for value in row["critical_actor_ids"]]
        target_ids = [str(value) for value in row["target_front_rear_ids"]]
        conflict_ids = [str(value) for value in row["candidate_conflict_ids"]]
        protected_ids = [str(value) for value in row["protected_actor_ids"]]
        critical_count = int(row["critical_count"])
        self.critical_counts.append(critical_count)
        self.mandatory_counts.append(len(protected_ids))
        if bool(row.get("contextual_count_observed", True)):
            self.contextual_counts[str(int(row["contextual_count"]))] += 1
            self.contextual_observed_state_count += 1
        latency = float(row["selector_latency_s"])
        latency_semantics = str(
            row.get("latency_semantics", "full_selector_runtime")
        )
        if latency_semantics == "full_selector_runtime":
            self.selector_latencies.append(latency)
        elif latency_semantics == "v3_telemetry_reclassification":
            self.telemetry_reclassification_latencies.append(latency)
        elif latency_semantics == "recorded_runtime_telemetry":
            pass
        else:
            raise ValueError(
                f"unknown Selector-v4 latency semantics: {latency_semantics}"
            )
        self.target_required += len(target_ids)
        self.conflict_required += len(conflict_ids)
        self.protected_required += len(protected_ids)
        for capacity in SELECTOR4_CAPACITIES:
            selected = set(critical_ids[:capacity])
            dropped = [
                vehicle_id
                for vehicle_id in protected_ids
                if vehicle_id not in selected
            ]
            self.overflow[capacity] += int(critical_count > capacity)
            self.dropped_mandatory[capacity] += len(dropped)
            self.target_covered[capacity] += sum(
                vehicle_id in selected for vehicle_id in target_ids
            )
            self.conflict_covered[capacity] += sum(
                vehicle_id in selected for vehicle_id in conflict_ids
            )
            self.protected_covered[capacity] += sum(
                vehicle_id in selected for vehicle_id in protected_ids
            )
            if (
                (critical_count > capacity or dropped)
                and len(self.overflow_examples[capacity])
                < SELECTOR4_SAMPLE_LIMIT
            ):
                self.overflow_examples[capacity].append(
                    {
                        "state_id": str(row["state_id"]),
                        **dict(row["provenance"]),
                        "capacity": int(capacity),
                        "critical_count": critical_count,
                        "selected_critical_actor_ids": critical_ids[:capacity],
                        "dropped_critical_actor_ids": critical_ids[capacity:],
                        "dropped_mandatory_actor_ids": dropped,
                        "critical_actors": list(row["critical_actors"]),
                    }
                )

    @staticmethod
    def _rate(covered: int, required: int) -> float:
        return float(covered / required) if required else 1.0

    def capacity_report(self, capacity: int) -> dict[str, Any]:
        maximum = max(self.critical_counts) if self.critical_counts else 0
        headroom = int(capacity) - int(maximum)
        overflow = int(self.overflow[capacity])
        dropped = int(self.dropped_mandatory[capacity])
        target_rate = self._rate(
            self.target_covered[capacity], self.target_required
        )
        conflict_rate = self._rate(
            self.conflict_covered[capacity], self.conflict_required
        )
        protected_rate = self._rate(
            self.protected_covered[capacity], self.protected_required
        )
        gate = bool(
            self.state_count > 0
            and overflow == 0
            and dropped == 0
            and target_rate == 1.0
            and conflict_rate == 1.0
            and protected_rate == 1.0
            and headroom >= SELECTOR4_MINIMUM_HEADROOM
        )
        return {
            "capacity": int(capacity),
            "state_count": int(self.state_count),
            "critical_overflow_count": overflow,
            "dropped_mandatory_actor_count": dropped,
            "protected_actor_required_count": self.protected_required,
            "protected_actor_coverage_rate": protected_rate,
            "target_front_rear_required_count": self.target_required,
            "target_front_rear_coverage_rate": target_rate,
            "candidate_conflict_required_count": self.conflict_required,
            "candidate_conflict_coverage_rate": conflict_rate,
            "maximum_critical_actor_count": int(maximum),
            "capacity_headroom": int(headroom),
            "minimum_required_headroom": SELECTOR4_MINIMUM_HEADROOM,
            "gate_pass": gate,
            "overflow_examples": self.overflow_examples[capacity],
            "overflow_sample_limit": SELECTOR4_SAMPLE_LIMIT,
        }

    def distribution_report(self) -> dict[str, Any]:
        return {
            "state_count": int(self.state_count),
            "critical_count_histogram": dict(
                sorted(Counter(map(str, self.critical_counts)).items())
            ),
            "contextual_count_histogram": dict(
                sorted(self.contextual_counts.items())
            ),
            "contextual_count_observed_state_count": int(
                self.contextual_observed_state_count
            ),
            "mandatory_actor_count": {
                "max": (
                    int(max(self.mandatory_counts))
                    if self.mandatory_counts
                    else 0
                ),
                "p99": _quantile(self.mandatory_counts, 0.99),
                "p99_9": _quantile(self.mandatory_counts, 0.999),
            },
            "critical_actor_count": {
                "max": (
                    int(max(self.critical_counts))
                    if self.critical_counts
                    else 0
                ),
                "p99": _quantile(self.critical_counts, 0.99),
                "p99_9": _quantile(self.critical_counts, 0.999),
            },
            "selector_only_latency_s": _latency_summary(
                self.selector_latencies
            ),
            "v3_telemetry_reclassification_latency_s": _latency_summary(
                self.telemetry_reclassification_latencies
            ),
        }


def _stratum_keys(provenance: dict[str, Any]) -> Iterable[tuple[str, str]]:
    for field in (
        "scope",
        "method_id",
        "optimizer_seed",
        "traffic_profile",
        "stress_profile",
    ):
        value = provenance.get(field)
        if value is not None and str(value):
            yield field, str(value)


def build_selector4_capacity_report(
    rows: Iterable[dict[str, Any]],
    *,
    source_coverage: dict[str, Any],
    selector_config: dict[str, Any],
    source_lineage: dict[str, Any],
) -> dict[str, Any]:
    overall = _Accumulator()
    strata: dict[str, dict[str, _Accumulator]] = defaultdict(dict)
    for row in rows:
        overall.add(row)
        provenance = dict(row.get("provenance", {}) or {})
        for field, value in _stratum_keys(provenance):
            accumulator = strata[field].setdefault(value, _Accumulator())
            accumulator.add(row)

    capacity_reports = {
        str(capacity): overall.capacity_report(capacity)
        for capacity in SELECTOR4_CAPACITIES
    }
    # Capacities 8 and 10 remain diagnostic counterfactuals. The formal V4
    # contract deliberately freezes 12 when (and only when) 12 passes. It must
    # not retreat to a just-sufficient smaller set if the later scorer runtime
    # gate fails; that failure requires model/runtime optimisation instead.
    selected_capacity = (
        12 if bool(capacity_reports["12"]["gate_pass"]) else None
    )
    prerequisites_pass = bool(
        source_coverage
        and all(bool(value) for value in source_coverage.values())
    )
    audit_state = (
        "pass"
        if prerequisites_pass and selected_capacity is not None
        else "blocked"
    )
    report = {
        "artifact_kind": SELECTOR4_AUDIT_KIND,
        "implementation_version": SELECTOR4_AUDIT_IMPLEMENTATION,
        "protocol_id": SELECTOR4_PROTOCOL_ID,
        "audit_state": audit_state,
        "selected_capacity": selected_capacity,
        "capacity_decision_rule": (
            "audit_8_10_12_freeze_12_only_with_zero_overflow_"
            "full_coverage_and_two_actor_headroom_v2"
        ),
        "diagnostic_capacity_results_only": [8, 10],
        "source_coverage": dict(source_coverage),
        "selector_config": dict(selector_config),
        "source_lineage": dict(source_lineage),
        "overall_distribution": overall.distribution_report(),
        "selector_latency_interpretation": {
            "state": "diagnostic_only",
            "gate_eligible": False,
            "reason": (
                "per-state selector latencies are read from resumable replay "
                "cache and may span different worker-concurrency runs; they "
                "are not the controlled capacity-specific scorer benchmark"
            ),
            "formal_telemetry_reclassification_separated": True,
        },
        "capacity_reports": capacity_reports,
        "strata": {
            field: {
                value: {
                    "distribution": accumulator.distribution_report(),
                    "capacities": {
                        str(capacity): accumulator.capacity_report(capacity)
                        for capacity in SELECTOR4_CAPACITIES
                    },
                }
                for value, accumulator in sorted(values.items())
            }
            for field, values in sorted(strata.items())
        },
        "scorer_runtime_gate": {
            "state": "deferred_until_capacity_specific_model_exists",
            "reason": (
                "a Selector-v3 8-actor predictor cannot measure the latency "
                "of a new 10/12-actor Selector-v4 model"
            ),
            "required_after_training": {
                "p95_s_max": 0.30,
                "p99_s_max": 0.40,
                "max_s_max": 0.50,
            },
            "capacity_fallback_after_failure_allowed": False,
        },
        "freeze_contract": {
            "selector_definition": "merge_conflict_relevance_v4_lane_aware",
            "actor_capacity": selected_capacity,
            "role_priority_frozen": selected_capacity is not None,
            "candidate_union_definition_frozen": selected_capacity is not None,
            "reward_frozen": "factorial_matrix_only",
            "commitment_frozen": "factorial_matrix_only",
            "risk_threshold_frozen": True,
            "mutable_before_stage5": False,
        },
        "limitations": [
            (
                "Development replays cover the selected checkpoints on the "
                "frozen stage3-selection cohort; transient training-rollout "
                "states were not persisted by the historical trainer."
            ),
            (
                "Selector-only latency is reported for diagnostics but is "
                "not a substitute for the post-training scorer runtime gate."
            ),
        ],
    }
    report["report_fingerprint"] = stable_hash(report)
    return report


def validate_selector4_capacity_report(report: dict[str, Any]) -> None:
    if str(report.get("artifact_kind", "")) != SELECTOR4_AUDIT_KIND:
        raise ValueError("unexpected Selector-v4 audit artifact kind")
    if str(report.get("protocol_id", "")) != SELECTOR4_PROTOCOL_ID:
        raise ValueError("unexpected Selector-v4 audit protocol")
    if str(report.get("implementation_version", "")) != (
        SELECTOR4_AUDIT_IMPLEMENTATION
    ):
        raise ValueError("unexpected Selector-v4 audit implementation")
    fingerprint = str(report.get("report_fingerprint", ""))
    expected = stable_hash(
        {
            key: value
            for key, value in report.items()
            if key != "report_fingerprint"
        }
    )
    if not fingerprint or fingerprint != expected:
        raise ValueError("Selector-v4 audit fingerprint mismatch")
    if str(report.get("audit_state", "")) != "pass":
        raise ValueError("Selector-v4 capacity audit is blocked")
    capacity = int(report.get("selected_capacity", -1))
    if capacity not in SELECTOR4_CAPACITIES:
        raise ValueError("Selector-v4 audit selected unsupported capacity")
    if capacity != 12:
        raise ValueError("Selector-v4 formal contract must freeze capacity 12")
    selected = dict(report["capacity_reports"][str(capacity)])
    if not bool(selected.get("gate_pass", False)):
        raise ValueError("Selector-v4 selected capacity did not pass")
    if int(selected.get("capacity_headroom", -1)) < 2:
        raise ValueError("Selector-v4 selected capacity lacks headroom")
    latency = dict(report.get("selector_latency_interpretation", {}) or {})
    if bool(latency.get("gate_eligible", True)):
        raise ValueError("Selector-v4 mixed-concurrency latency cannot be a gate")
