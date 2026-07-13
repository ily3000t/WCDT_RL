from __future__ import annotations

from collections import Counter, defaultdict
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np

from safe_rl.accvp.artifacts import (
    ACCVP_ARTIFACT_GENERATION,
    ACCVP_ARTIFACT_KIND,
    LIFECYCLE_SEALED_CANDIDATE,
    VNEXT_LITE_DECISION_WEIGHTING_VERSION,
    artifact_filename,
    bundle_file_entry,
    resolve_v2_bundle,
)
from safe_rl.accvp.schema import file_sha256, read_json, stable_hash, write_json_atomic
from safe_rl.accvp.selection import LEFT_ACTION_IDS, lite_secondary_safety_pass, select_viability_lite_action
from safe_rl.evaluation_protocol import protocol_snapshot


def lite_thresholds_from_config(config: Any) -> dict[str, float]:
    lite = config.accvp.get("viability_lite", {}) or {}
    return {
        "min_p_merge_before_taper": float(lite.get("min_p_merge_before_taper", 0.75)),
        "min_improvement_over_raw": float(lite.get("min_improvement_over_raw", 0.01)),
        "max_target_entry_time_s": float(lite.get("max_target_entry_time_s", 8.0)),
        "max_ensemble_disagreement": float(lite.get("max_ensemble_disagreement", 0.20)),
        "max_secondary_risk_score": float(lite.get("max_secondary_risk_score", 1.0)),
        "secondary_safety_profile": str(lite.get("secondary_safety_profile", "strict")),
    }


def collapse_vnext_lite_records(
    records: list[dict[str, Any]],
    *,
    score_tolerance: float = 1.0e-6,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Give repeated model-equivalent decisions total weight one."""

    tolerance = float(score_tolerance)
    if tolerance < 0.0:
        raise ValueError("lite duplicate score tolerance must be non-negative")
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    decision_roots: dict[str, set[str]] = defaultdict(set)
    group_roots: dict[tuple[str, int], set[str]] = defaultdict(set)
    decision_components: dict[str, set[str]] = defaultdict(set)
    decision_raw_actions: dict[str, set[int]] = defaultdict(set)
    decision_raw_legal_values: dict[str, set[bool]] = defaultdict(set)
    for position, row in enumerate(records):
        fingerprint = str(row.get("root_observation_fingerprint", "")).strip()
        component_id = str(row.get("split_component_id", "")).strip()
        if not fingerprint or not component_id:
            raise ValueError(
                f"VNext lite record {position} requires fingerprint and split_component_id"
            )
        if row.get("raw_action_id") is None:
            raise ValueError(f"VNext lite record {position} requires raw_action_id")
        decision_id = f"{fingerprint}|raw:{int(row['raw_action_id'])}"
        group_key = (decision_id, int(row["action_id"]))
        source_root_id = str(row.get("root_id", ""))
        if not source_root_id or source_root_id in group_roots[group_key]:
            raise ValueError(
                "VNext lite records contain a missing or duplicate root/action row"
            )
        decision_roots[decision_id].add(source_root_id)
        group_roots[group_key].add(source_root_id)
        decision_components[decision_id].add(component_id)
        decision_raw_actions[decision_id].add(int(row["raw_action_id"]))
        decision_raw_legal_values[decision_id].add(
            bool(row.get("raw_action_legal", False))
        )
        grouped[group_key].append(dict(row))

    decision_raw_coverage: dict[str, bool] = {}
    decision_raw_legality: dict[str, bool] = {}
    for decision_id, source_roots in decision_roots.items():
        components = decision_components[decision_id]
        raw_actions = decision_raw_actions[decision_id]
        if len(components) != 1:
            raise ValueError(
                "VNext lite decision spans multiple split components: "
                f"decision={decision_id} components={sorted(components)}"
            )
        if len(raw_actions) != 1:
            raise ValueError(
                f"VNext lite decision has inconsistent raw actions: {decision_id}"
            )
        if len(decision_raw_legal_values[decision_id]) != 1:
            raise ValueError(
                f"VNext lite decision has inconsistent raw-action legality: {decision_id}"
            )
        raw_action_id = next(iter(raw_actions))
        raw_group_key = (decision_id, raw_action_id)
        raw_members = grouped.get(raw_group_key, [])
        raw_complete = bool(raw_members) and group_roots.get(raw_group_key, set()) == source_roots
        decision_raw_coverage[decision_id] = raw_complete
        decision_raw_legality[decision_id] = bool(
            raw_complete
            and decision_raw_legal_values[decision_id] == {True}
        )

    collapsed: list[dict[str, Any]] = []
    model_equivalent_fields = (
        "p_proxy_collision",
        "p_safety_violation",
        "p_taper_miss",
        "p_merge_before_taper",
        "pU_proxy_collision",
        "pU_safety_violation",
        "pL_merge_before_taper",
        "target_lane_entry_time_s",
        "ensemble_disagreement",
    )
    for (decision_id, action_id), members in sorted(grouped.items()):
        complete_decision_coverage = (
            group_roots[(decision_id, action_id)] == decision_roots[decision_id]
        )
        components = {str(row["split_component_id"]) for row in members}
        raw_actions = {int(row["raw_action_id"]) for row in members}
        if len(components) != 1 or len(raw_actions) != 1:
            raise ValueError("VNext lite duplicate group has inconsistent component or raw action")
        for field in model_equivalent_fields:
            values = np.asarray([float(row[field]) for row in members], dtype=np.float64)
            if not np.isfinite(values).all():
                raise ValueError(f"VNext lite duplicate group has non-finite {field}")
            if float(values.max() - values.min()) > tolerance:
                raise ValueError(
                    "VNext lite model-equivalent scores diverged beyond tolerance: "
                    f"decision={decision_id} action={action_id} field={field}"
                )
        observed = [row for row in members if bool(row.get("merge_observed", False))]
        secondary_veto_reasons = {
            str(row.get("secondary_veto_reason", ""))
            for row in members
            if str(row.get("secondary_veto_reason", ""))
        }
        if not complete_decision_coverage:
            secondary_veto_reasons.add("incomplete_duplicate_coverage")
        replicate_outcomes = [
            {
                "root_id": str(row["root_id"]),
                "merge_observed": bool(row.get("merge_observed", False)),
                "merge_success": bool(
                    row.get("merge_observed", False)
                    and float(row.get("merge_before_taper", 0.0)) >= 0.5
                ),
                "safety_event": bool(
                    float(row.get("proxy_collision", 0.0)) >= 0.5
                    or float(row.get("safety_violation", 0.0)) >= 0.5
                ),
            }
            for row in sorted(members, key=lambda item: str(item["root_id"]))
        ]
        merged = dict(members[0])
        merged.update(
            {
                "root_id": decision_id,
                "decision_unit_id": decision_id,
                "action_id": action_id,
                "root_policy": "collapsed_fingerprint_action_group",
                "root_policies": sorted({str(row.get("root_policy", "")) for row in members}),
                "episode_seeds": sorted(
                    {int(row.get("episode_seed", -1)) for row in members}
                ),
                "replicate_count": len(members),
                "expected_replicate_count": len(decision_roots[decision_id]),
                "complete_decision_coverage": complete_decision_coverage,
                "p_proxy_collision": max(float(row["p_proxy_collision"]) for row in members),
                "p_safety_violation": max(float(row["p_safety_violation"]) for row in members),
                "p_taper_miss": max(float(row["p_taper_miss"]) for row in members),
                "p_merge_before_taper": min(float(row["p_merge_before_taper"]) for row in members),
                "pU_proxy_collision": max(float(row["pU_proxy_collision"]) for row in members),
                "pU_safety_violation": max(float(row["pU_safety_violation"]) for row in members),
                "pL_merge_before_taper": min(float(row["pL_merge_before_taper"]) for row in members),
                "target_lane_entry_time_s": max(
                    float(row["target_lane_entry_time_s"]) for row in members
                ),
                "ensemble_disagreement": max(
                    float(row["ensemble_disagreement"]) for row in members
                ),
                "proxy_collision": float(
                    np.mean([float(row["proxy_collision"]) for row in members])
                ),
                "safety_violation": float(
                    np.mean([float(row["safety_violation"]) for row in members])
                ),
                "taper_miss": float(
                    np.mean([float(row["taper_miss"]) for row in members])
                ),
                "outcome_safety_event_rate": float(
                    np.mean(
                        [
                            float(
                                float(row.get("proxy_collision", 0.0)) >= 0.5
                                or float(row.get("safety_violation", 0.0)) >= 0.5
                            )
                            for row in members
                        ]
                    )
                ),
                "outcome_merge_observation_rate": float(
                    len(observed) / len(members)
                ),
                "outcome_merge_success_rate": (
                    float(
                        np.mean(
                            [
                                float(float(row["merge_before_taper"]) >= 0.5)
                                for row in observed
                            ]
                        )
                    )
                    if observed
                    else 0.0
                ),
                "outcome_merge_success_mass": float(
                    sum(float(row["merge_success"]) for row in replicate_outcomes)
                    / len(replicate_outcomes)
                ),
                "replicate_outcomes": replicate_outcomes,
                "merge_before_taper": (
                    float(np.mean([float(row["merge_before_taper"]) for row in observed]))
                    if observed
                    else 0.0
                ),
                "merge_observed": bool(observed),
                "raw_action_complete_decision_coverage": decision_raw_coverage[
                    decision_id
                ],
                "raw_action_legal": decision_raw_legality[decision_id],
                "candidate_legal": all(
                    bool(row.get("candidate_legal", True)) for row in members
                )
                and complete_decision_coverage,
                "secondary_safety_pass": all(
                    bool(row.get("secondary_safety_pass", True)) for row in members
                )
                and complete_decision_coverage,
                "secondary_risk_score": max(
                    float(row.get("secondary_risk_score", 1.0)) for row in members
                ),
                "secondary_risk_uncertainty": max(
                    float(row.get("secondary_risk_uncertainty", 1.0)) for row in members
                ),
                "secondary_veto_reason": "|".join(sorted(secondary_veto_reasons)),
            }
        )
        collapsed.append(merged)
    decision_ids = sorted({str(row["decision_unit_id"]) for row in collapsed})
    component_ids = sorted({str(row["split_component_id"]) for row in collapsed})
    provenance = {
        "decision_weighting_version": VNEXT_LITE_DECISION_WEIGHTING_VERSION,
        "candidate_weighting_unit": "model_input_fingerprint_x_raw_action_x_action",
        "statistical_independence_claim": False,
        "raw_candidate_row_count": len(records),
        "effective_candidate_row_count": len(collapsed),
        "effective_decision_count": len(decision_ids),
        "effective_split_component_count": len(component_ids),
        "incomplete_candidate_group_count": sum(
            not bool(row["complete_decision_coverage"]) for row in collapsed
        ),
        "raw_incomplete_decision_count": sum(
            not complete for complete in decision_raw_coverage.values()
        ),
        "raw_illegal_decision_count": sum(
            not legal for legal in decision_raw_legality.values()
        ),
        "decision_ids_sha256": stable_hash({"decision_ids": decision_ids}),
        "group_label_aggregation": "replicate_mass_and_conditional_rate_v2",
        "joint_repairable_aggregation": "paired_root_outcome_mass_v1",
        "risk_score_aggregation": "maximum",
        "viability_score_aggregation": "minimum",
        "score_tolerance": tolerance,
    }
    return collapsed, provenance


def _by_root(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[str(row["root_id"])].append(row)
    return grouped


def _episode_seed_values(rows: list[dict[str, Any]]) -> set[int]:
    seeds: set[int] = set()
    for row in rows:
        values = row.get("episode_seeds")
        if isinstance(values, list):
            seeds.update(int(value) for value in values if int(value) >= 0)
        elif row.get("episode_seed") is not None and int(row["episode_seed"]) >= 0:
            seeds.add(int(row["episode_seed"]))
    return seeds


def _split_component_values(rows: list[dict[str, Any]]) -> set[str]:
    return {
        str(row.get("split_component_id", ""))
        for row in rows
        if str(row.get("split_component_id", ""))
    }


def outcome_safety_event_rate(row: dict[str, Any]) -> float:
    if row.get("outcome_safety_event_rate") is not None:
        return float(row["outcome_safety_event_rate"])
    return float(
        float(row.get("proxy_collision", 0.0)) >= 0.5
        or float(row.get("safety_violation", 0.0)) >= 0.5
    )


def outcome_merge_observation_rate(row: dict[str, Any]) -> float:
    if row.get("outcome_merge_observation_rate") is not None:
        return float(row["outcome_merge_observation_rate"])
    return float(bool(row.get("merge_observed", False)))


def outcome_merge_success_rate(row: dict[str, Any]) -> float:
    if row.get("outcome_merge_success_rate") is not None:
        return float(row["outcome_merge_success_rate"])
    return float(
        bool(row.get("merge_observed", False))
        and float(row.get("merge_before_taper", 0.0)) >= 0.5
    )


def outcome_merge_success_mass(row: dict[str, Any]) -> float:
    if row.get("outcome_merge_success_mass") is not None:
        return float(row["outcome_merge_success_mass"])
    return outcome_merge_observation_rate(row) * outcome_merge_success_rate(row)


def conditional_merge_success_rate(rows: list[dict[str, Any]]) -> float | None:
    observed_mass = sum(outcome_merge_observation_rate(row) for row in rows)
    if observed_mass <= np.finfo(np.float64).eps:
        return None
    return float(
        sum(outcome_merge_success_mass(row) for row in rows) / observed_mass
    )


def _replicate_outcome_map(row: dict[str, Any]) -> dict[str, dict[str, bool]]:
    payload = row.get("replicate_outcomes")
    if isinstance(payload, list) and payload:
        result: dict[str, dict[str, bool]] = {}
        for record in payload:
            root_id = str(record.get("root_id", ""))
            if not root_id or root_id in result:
                raise ValueError("invalid duplicate replicate outcome provenance")
            result[root_id] = {
                "merge_observed": bool(record.get("merge_observed", False)),
                "merge_success": bool(record.get("merge_success", False)),
                "safety_event": bool(record.get("safety_event", False)),
            }
        return result
    root_id = str(row.get("root_id", ""))
    return {
        root_id: {
            "merge_observed": bool(row.get("merge_observed", False)),
            "merge_success": bool(
                row.get("merge_observed", False)
                and float(row.get("merge_before_taper", 0.0)) >= 0.5
            ),
            "safety_event": bool(
                float(row.get("proxy_collision", 0.0)) >= 0.5
                or float(row.get("safety_violation", 0.0)) >= 0.5
            ),
        }
    }


def _safe(row: dict[str, Any]) -> bool:
    return outcome_safety_event_rate(row) <= np.finfo(np.float64).eps


def _success(row: dict[str, Any]) -> bool:
    tolerance = np.finfo(np.float64).eps
    return (
        outcome_merge_observation_rate(row) >= 1.0 - tolerance
        and outcome_merge_success_rate(row) >= 1.0 - tolerance
    )


def _risk_pass(row: dict[str, Any], thresholds: dict[str, Any] | None = None) -> bool:
    if thresholds is None:
        return bool(row.get("candidate_legal", True)) and bool(row.get("secondary_safety_pass", True))
    return lite_secondary_safety_pass(row, thresholds)


def _rate_or_none(values: list[float | bool]) -> float | None:
    return float(np.mean([float(value) for value in values])) if values else None


def _finite_metric(value: Any, *, default: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if np.isfinite(result) else float(default)


def _joint_repairable_mass(
    candidates: list[dict[str, Any]],
    thresholds: dict[str, Any],
    *,
    selected: dict[str, Any] | None,
    replacement: bool,
) -> tuple[float, float]:
    raw_action_id = int(candidates[0].get("raw_action_id", -1))
    raw = next(
        (row for row in candidates if int(row["action_id"]) == raw_action_id),
        None,
    )
    if raw is None:
        return 0.0, 0.0
    raw_outcomes = _replicate_outcome_map(raw)
    if not raw_outcomes:
        return 0.0, 0.0
    left = [
        row
        for row in candidates
        if int(row["action_id"]) in LEFT_ACTION_IDS and _risk_pass(row, thresholds)
    ]
    left_outcomes = {
        int(row["action_id"]): _replicate_outcome_map(row) for row in left
    }
    selected_outcomes = (
        _replicate_outcome_map(selected)
        if replacement
        and selected is not None
        and int(selected["action_id"]) in LEFT_ACTION_IDS
        else {}
    )
    repairable = 0.0
    captured = 0.0
    for root_id, raw_outcome in raw_outcomes.items():
        raw_failure = bool(raw_outcome["merge_observed"]) and not bool(
            raw_outcome["merge_success"]
        )
        if not raw_failure:
            continue
        has_safe_success = any(
            root_id in outcomes
            and bool(outcomes[root_id]["merge_observed"])
            and bool(outcomes[root_id]["merge_success"])
            and not bool(outcomes[root_id]["safety_event"])
            for outcomes in left_outcomes.values()
        )
        if not has_safe_success:
            continue
        repairable += 1.0
        selected_outcome = selected_outcomes.get(root_id)
        if (
            selected_outcome is not None
            and bool(selected_outcome["merge_observed"])
            and bool(selected_outcome["merge_success"])
            and not bool(selected_outcome["safety_event"])
        ):
            captured += 1.0
    denominator = float(len(raw_outcomes))
    return repairable / denominator, captured / denominator


def evaluate_lite_thresholds(
    records: list[dict[str, Any]],
    thresholds: dict[str, float],
    *,
    split: str = "operating_point",
) -> dict[str, Any]:
    grouped = _by_root(records)
    selected: list[dict[str, Any]] = []
    replacements: list[dict[str, Any]] = []
    raw_retained = 0
    repairable_decision_count = 0
    repairable_mass = 0.0
    repairable_captured_mass = 0.0
    unnecessary_replacements = 0.0
    reasons: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()
    examples: list[dict[str, Any]] = []
    for root_id, candidates in grouped.items():
        raw_action_id = int(candidates[0].get("raw_action_id", -1))
        raw = next((row for row in candidates if int(row["action_id"]) == raw_action_id), None)
        decision = select_viability_lite_action(candidates, raw_action_id=raw_action_id, thresholds=thresholds)
        reasons[str(decision.get("reason", ""))] += 1
        chosen = decision.get("selected")
        if chosen is not None:
            selected.append(chosen)
            action_counts[str(int(chosen["action_id"]))] += 1
        if bool(decision.get("replacement", False)) and chosen is not None:
            replacements.append(chosen)
            if raw is not None and outcome_merge_observation_rate(raw) > 0.0:
                unnecessary_replacements += outcome_merge_success_rate(raw)
            examples.append(
                {
                    "root_id": root_id,
                    "raw_action_id": raw_action_id,
                    "selected_action_id": int(chosen["action_id"]),
                    "raw_p_merge_before_taper": None if raw is None else float(raw.get("p_merge_before_taper", 0.0)),
                    "selected_p_merge_before_taper": float(chosen.get("p_merge_before_taper", 0.0)),
                    "p_merge_improvement": float(decision.get("p_merge_improvement", 0.0)),
                    "selected_merge_success": _success(chosen),
                    "selected_merge_success_rate": outcome_merge_success_rate(chosen),
                    "selected_safety_event": not _safe(chosen),
                    "selected_safety_event_rate": outcome_safety_event_rate(chosen),
                }
            )
        else:
            raw_retained += 1
        decision_repairable_mass, decision_captured_mass = _joint_repairable_mass(
            candidates,
            thresholds,
            selected=chosen,
            replacement=bool(decision.get("replacement", False)),
        )
        if decision_repairable_mass > 0.0:
            repairable_decision_count += 1
        repairable_mass += decision_repairable_mass
        repairable_captured_mass += decision_captured_mass
    selected_observed = [
        row for row in selected if outcome_merge_observation_rate(row) > 0.0
    ]
    replacement_observed = [
        row for row in replacements if outcome_merge_observation_rate(row) > 0.0
    ]
    replacement_risk_pass_rate = _rate_or_none([_risk_pass(row, thresholds) for row in replacements])
    replacement_safety_event_rate = _rate_or_none(
        [outcome_safety_event_rate(row) for row in replacements]
    )
    replacement_merge_success_rate = conditional_merge_success_rate(
        replacement_observed
    )
    replacement_unnecessary_rate = float(unnecessary_replacements / max(1, len(replacements)))
    replacement_repairable_capture_rate = float(
        repairable_captured_mass / max(np.finfo(np.float64).eps, repairable_mass)
    )
    return {
        "split": split,
        "thresholds": dict(thresholds),
        "decision_count": int(len(grouped)),
        "selected_count": int(len(selected)),
        "replacement_count": int(len(replacements)),
        "effective_decision_count": int(len(grouped)),
        "unique_episode_seed_count": len(_episode_seed_values(records)),
        "effective_split_component_count": len(
            _split_component_values(records)
        ),
        "replacement_unique_episode_seed_count": len(
            _episode_seed_values(replacements)
        ),
        "replacement_effective_split_component_count": len(
            _split_component_values(replacements)
        ),
        "replacement_observed_mass": float(
            sum(outcome_merge_observation_rate(row) for row in replacements)
        ),
        "replacement_rate": float(len(replacements) / max(1, len(grouped))),
        "raw_retention_rate": float(raw_retained / max(1, len(grouped))),
        "repairable_root_count": int(repairable_decision_count),
        "repairable_decision_mass": float(repairable_mass),
        "repairable_captured_mass": float(repairable_captured_mass),
        "repairable_root_capture_rate": replacement_repairable_capture_rate,
        "replacement_repairable_capture_rate": replacement_repairable_capture_rate,
        "selected_action_risk_pass_rate": (
            float(np.mean([_risk_pass(row, thresholds) for row in selected])) if selected else None
        ),
        "selected_action_safety_event_rate": (
            float(np.mean([outcome_safety_event_rate(row) for row in selected]))
            if selected
            else None
        ),
        "selected_action_merge_success_rate": (
            conditional_merge_success_rate(selected_observed)
        ),
        "replacement_merge_success_rate": (
            replacement_merge_success_rate
        ),
        "replacement_action_risk_pass_rate": replacement_risk_pass_rate,
        "replacement_action_safety_event_rate": replacement_safety_event_rate,
        "replacement_action_merge_success_rate": replacement_merge_success_rate,
        "replacement_unnecessary_rate": replacement_unnecessary_rate,
        "unnecessary_replacement_rate": replacement_unnecessary_rate,
        "left_action_selection_histogram": {
            key: value for key, value in sorted(action_counts.items(), key=lambda item: int(item[0])) if int(key) in LEFT_ACTION_IDS
        },
        "selected_action_histogram": dict(sorted(action_counts.items(), key=lambda item: int(item[0]))),
        "reason_counts": dict(sorted(reasons.items())),
        "replacement_examples": examples[:20],
    }


def tune_viability_lite_operating_point(
    records: list[dict[str, Any]],
    config: Any,
    *,
    split: str = "operating_point",
) -> dict[str, Any]:
    if str(split) != "operating_point":
        raise ValueError("ACCVP-lite thresholds may be selected only on the operating_point split")
    if str(config.accvp.get("artifact_generation") or "").strip() == ACCVP_ARTIFACT_GENERATION:
        records, decision_weighting = collapse_vnext_lite_records(records)
    else:
        decision_weighting = {
            "decision_weighting_version": "legacy_root_id_v1",
            "raw_candidate_row_count": len(records),
            "effective_candidate_row_count": len(records),
            "effective_decision_count": len(_by_root(records)),
            "statistical_independence_claim": False,
        }
    lite = config.accvp.get("viability_lite", {}) or {}
    max_replacement_rate = float(lite.get("max_replacement_rate", 0.50))
    secondary_safety_profile = str(lite.get("secondary_safety_profile", "strict"))
    evaluated = []
    for p_merge, improvement, entry, disagreement, risk_score in product(
        lite.get("min_left_p_merge_before_taper_grid", [0.70, 0.75, 0.78, 0.80]),
        lite.get("min_improvement_over_raw_grid", [0.005, 0.01, 0.02, 0.03]),
        lite.get("max_target_entry_time_s_grid", [4.0, 6.0, 8.0]),
        lite.get("max_ensemble_disagreement_grid", [0.05, 0.10, 0.20]),
        lite.get("max_secondary_risk_score_grid", [1.0]),
    ):
        thresholds = {
            "min_p_merge_before_taper": float(p_merge),
            "min_improvement_over_raw": float(improvement),
            "max_target_entry_time_s": float(entry),
            "max_ensemble_disagreement": float(disagreement),
            "max_secondary_risk_score": float(risk_score),
            "secondary_safety_profile": secondary_safety_profile,
        }
        evaluated.append(evaluate_lite_thresholds(records, thresholds, split=split))
    feasible = [
        row
        for row in evaluated
        if int(row["replacement_count"]) > 0
        and _finite_metric(row["replacement_action_risk_pass_rate"], default=-1.0) == 1.0
        and _finite_metric(row["replacement_action_safety_event_rate"], default=1.0) == 0.0
        and float(row["replacement_rate"]) <= max_replacement_rate
    ]
    candidates = feasible or evaluated
    selected = max(
        candidates,
        key=lambda row: (
            float(row["replacement_repairable_capture_rate"]),
            _finite_metric(row["replacement_action_merge_success_rate"], default=-1.0),
            -_finite_metric(row["replacement_action_safety_event_rate"], default=1.0),
            -float(row["replacement_unnecessary_rate"]),
            -float(row["replacement_rate"]),
            -float(row["thresholds"].get("max_secondary_risk_score", 1.0)),
            -float(row["thresholds"]["min_improvement_over_raw"]),
        ),
    )
    return {
        "split": split,
        "controller": "acv_shield_lite",
        "safety_authority": "risk_module_safety_shield",
        "secondary_safety_profile": secondary_safety_profile,
        "accvp_safety_head_hard_gate": False,
        "deployable_claim": "task_viability_only",
        "max_replacement_rate": max_replacement_rate,
        "selected": selected["thresholds"],
        "selected_metrics": selected,
        "evaluated_points": evaluated,
        "decision_weighting": decision_weighting,
    }


def _write_vnext_lite_artifacts(
    *,
    output: Path,
    config: Any,
    dataset_dir: Path,
    checkpoint: Path,
    calibration: Path,
    operating_point: dict[str, Any],
) -> dict[str, Path]:
    if str(operating_point.get("split", "")) != "operating_point":
        raise ValueError("VNext lite operating point must originate from split='operating_point'")
    source_manifest_value = config.accvp.get("artifact_manifest")
    if not source_manifest_value:
        raise FileNotFoundError(
            "VNext lite tuning requires the source sealed bundle in accvp.artifact_manifest"
        )
    source_manifest_path = Path(str(source_manifest_value)).resolve()
    source_manifest, source_files = resolve_v2_bundle(source_manifest_path)
    if str(source_manifest.get("lifecycle_state", "")) != LIFECYCLE_SEALED_CANDIDATE:
        raise ValueError("VNext lite tuning requires a sealed_candidate source bundle")
    if str(source_manifest.get("artifact_variant", "")) != "full_candidate_gate_v1":
        raise ValueError(
            "VNext lite tuning requires artifact_variant='full_candidate_gate_v1'"
        )
    if source_files.get("predictor") != checkpoint.resolve():
        raise ValueError("VNext lite tuning checkpoint does not match the source bundle")
    if source_files.get("calibration") != calibration.resolve():
        raise ValueError("VNext lite tuning calibration does not match the source bundle")
    training_history = source_files.get("training_history")
    if training_history is None:
        raise ValueError("VNext lite tuning source bundle is missing training history")

    selected = dict(operating_point.get("selected", {}) or {})
    required_thresholds = {
        "min_p_merge_before_taper",
        "min_improvement_over_raw",
        "max_target_entry_time_s",
        "max_ensemble_disagreement",
    }
    missing_thresholds = sorted(required_thresholds.difference(selected))
    if missing_thresholds:
        raise ValueError(
            f"VNext lite operating point is missing task-viability thresholds: {missing_thresholds}"
        )
    decision_weighting = dict(operating_point.get("decision_weighting", {}) or {})
    if (
        str(decision_weighting.get("decision_weighting_version", ""))
        != VNEXT_LITE_DECISION_WEIGHTING_VERSION
    ):
        raise ValueError("VNext lite operating point is missing decision-weighting provenance")

    dataset_manifest = dataset_dir / "manifests" / "dataset_manifest.json"
    split_manifest = dataset_dir / "manifests" / "split_manifest.jsonl"
    split_provenance = dataset_dir / "manifests" / "split_provenance.json"
    expected_dataset_hashes = {
        "dataset_manifest_sha256": file_sha256(dataset_manifest),
        "split_manifest_sha256": file_sha256(split_manifest),
    }
    if split_provenance.is_file():
        expected_dataset_hashes["split_provenance_sha256"] = file_sha256(split_provenance)
    mismatches = {
        key: {"source_bundle": source_manifest.get(key), "actual": value}
        for key, value in expected_dataset_hashes.items()
        if str(source_manifest.get(key, "")) != value
    }
    if mismatches:
        raise ValueError(f"VNext lite tuning dataset lineage mismatch: {mismatches}")

    evidence = protocol_snapshot(config)
    source_protocol_id = str(source_manifest.get("evidence_protocol_id", ""))
    if source_protocol_id and source_protocol_id != str(evidence.get("protocol_id", "")):
        raise ValueError("VNext lite tuning protocol_id does not match the source bundle")
    source_ledger = str(source_manifest.get("seed_ledger_sha256", ""))
    if source_ledger and source_ledger != str(evidence.get("seed_ledger_sha256", "")):
        raise ValueError("VNext lite tuning seed ledger does not match the source bundle")

    operating_path = output / artifact_filename("lite_operating_point")
    manifest_path = output / artifact_filename("lite_candidate_manifest")
    existing = [path for path in (operating_path, manifest_path) if path.exists()]
    if existing:
        raise FileExistsError(f"VNext lite artifact already exists: {existing[0]}")
    write_json_atomic(operating_path, operating_point)

    files = {
        "predictor": bundle_file_entry(checkpoint, manifest_dir=output),
        "calibration": bundle_file_entry(calibration, manifest_dir=output),
        "operating_point": bundle_file_entry(operating_path, manifest_dir=output),
        "training_history": bundle_file_entry(training_history, manifest_dir=output),
    }
    manifest = dict(source_manifest)
    manifest.pop("artifact_fingerprint", None)
    manifest.pop("holdout_decision", None)
    source_training_config_hash = str(manifest.get("config_hash", ""))
    manifest.update(
        {
            "artifact_kind": ACCVP_ARTIFACT_KIND,
            "artifact_generation": ACCVP_ARTIFACT_GENERATION,
            "artifact_variant": "viability_lite_task_v1",
            "controller": "acv_shield_lite",
            "safety_authority": "risk_module_safety_shield",
            "secondary_safety_profile": str(
                selected.get(
                    "secondary_safety_profile",
                    config.accvp.get("viability_lite", {}).get(
                        "secondary_safety_profile", "strict"
                    ),
                )
            ),
            "accvp_safety_head_hard_gate": False,
            "deployable_claim": "task_viability_only",
            "deployable_artifact": False,
            "holdout_state": "sealed",
            "lifecycle_state": LIFECYCLE_SEALED_CANDIDATE,
            "threshold_selection_split": "operating_point",
            "test_used_for_threshold_selection": False,
            "operating_point_schema": "accvp_viability_lite_operating_point_v1",
            "decision_weighting": decision_weighting,
            "source_candidate_manifest_sha256": file_sha256(source_manifest_path),
            "source_candidate_fingerprint": str(
                source_manifest.get("artifact_fingerprint", "")
            ),
            "source_candidate_manifest_reference": {
                "reference_kind": "digest_only_v1",
                "sha256": file_sha256(source_manifest_path),
                "artifact_fingerprint": str(
                    source_manifest.get("artifact_fingerprint", "")
                ),
                "artifact_variant": str(
                    source_manifest.get("artifact_variant", "")
                ),
            },
            "source_training_config_hash": source_training_config_hash,
            "lite_tuning_config_hash": stable_hash(dict(config)),
            "files": files,
            "predictor_sha256": files["predictor"]["sha256"],
            "calibration_sha256": files["calibration"]["sha256"],
            "operating_point_sha256": files["operating_point"]["sha256"],
            "training_history_sha256": files["training_history"]["sha256"],
        }
    )
    manifest["artifact_fingerprint"] = stable_hash(manifest)
    write_json_atomic(manifest_path, manifest)
    resolved_manifest, resolved_files = resolve_v2_bundle(manifest_path)
    if resolved_files.get("operating_point") != operating_path.resolve():
        raise ValueError("VNext lite bundle did not resolve its operating point")
    if str(resolved_manifest.get("artifact_variant", "")) != "viability_lite_task_v1":
        raise ValueError("VNext lite bundle artifact variant mismatch")
    return {
        "operating_point": operating_path,
        "artifact_manifest": manifest_path,
    }


def write_lite_artifacts(
    *,
    output_dir: str | Path,
    config: Any,
    dataset_dir: str | Path,
    checkpoint: str | Path,
    calibration: str | Path,
    operating_point: dict[str, Any],
    artifact_prefix: str = "accvp_v1_lite",
) -> dict[str, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    dataset_path = Path(dataset_dir)
    checkpoint_path = Path(checkpoint)
    calibration_path = Path(calibration)
    if str(config.accvp.get("artifact_generation") or "").strip() == ACCVP_ARTIFACT_GENERATION:
        return _write_vnext_lite_artifacts(
            output=output,
            config=config,
            dataset_dir=dataset_path,
            checkpoint=checkpoint_path,
            calibration=calibration_path,
            operating_point=operating_point,
        )
    prefix = str(artifact_prefix).strip() or "accvp_v1_lite"
    operating_path = output / f"{prefix}_operating_point.json"
    if str(operating_point.get("split", "")) != "operating_point":
        raise ValueError("lite artifact operating point must originate from split='operating_point'")
    write_json_atomic(operating_path, operating_point)
    dataset_manifest = dataset_path / "manifests" / "dataset_manifest.json"
    split_manifest = dataset_path / "manifests" / "split_manifest.jsonl"
    manifest_payload = read_json(dataset_manifest)
    evidence = protocol_snapshot(config)
    manifest = {
        "artifact_kind": "accvp_v1_lite_task_artifact_bundle",
        "controller": "acv_shield_lite",
        "safety_authority": "risk_module_safety_shield",
        "secondary_safety_profile": str(config.accvp.get("viability_lite", {}).get("secondary_safety_profile", "strict")),
        "accvp_safety_head_hard_gate": False,
        "deployable_claim": "task_viability_only",
        "deployable_artifact": False,
        "holdout_state": "sealed",
        "threshold_selection_split": "operating_point",
        "test_used_for_threshold_selection": False,
        "predictor_sha256": file_sha256(checkpoint),
        "calibration_sha256": file_sha256(calibration),
        "operating_point_sha256": file_sha256(operating_path),
        "dataset_manifest_sha256": file_sha256(dataset_manifest),
        "split_manifest_sha256": file_sha256(split_manifest),
        "dataset_fingerprint": str(manifest_payload.get("dataset_fingerprint", "")),
        "risk_model_fingerprint": str(manifest_payload.get("risk_model_fingerprint", "")),
        "counterfactual_schema_version": int(manifest_payload.get("counterfactual_schema_version", 2)),
        "accvp_activation_distance_m": float(manifest_payload.get("accvp_activation_distance_m", -1.0)),
        "data_contract_hash": str(manifest_payload.get("data_contract_hash", "")),
        "config_hash": stable_hash(dict(config)),
        "evidence_protocol_id": str(evidence.get("protocol_id", "")),
        "seed_ledger_sha256": evidence.get("seed_ledger_sha256"),
    }
    manifest["artifact_fingerprint"] = stable_hash(manifest)
    manifest_path = write_json_atomic(output / f"{prefix}_task_artifact_manifest.json", manifest)
    paths = {
        "operating_point": operating_path,
        "artifact_manifest": manifest_path,
    }
    return paths
