from __future__ import annotations

from collections import Counter

import numpy as np


DRAC_REPORT_CAP_DEFAULT = 20.0


def _report_safety_violation(report: dict) -> float:
    if "safety_violation" in report:
        return float(bool(report.get("safety_violation", False)))
    return float(
        bool(report.get("collision", False))
        or bool(report.get("proxy_collision", False))
        or bool(report.get("near_miss", False))
        or float(report.get("ttc_p1", 1.0e6)) < 0.3
    )


def _report_proxy_collision(report: dict) -> float:
    if "proxy_collision" in report:
        return float(bool(report.get("proxy_collision", False)))
    return float(float(report.get("min_distance", 1.0e6)) <= 0.25)


def _task_quality_thresholds(task_quality: dict | None) -> dict[str, float]:
    task_quality = dict(task_quality or {})
    result = {
        "timely_entry_distance_threshold_m": float(
            task_quality.get("timely_entry_distance_threshold_m", 80.0)
        ),
        "late_merge_request_threshold_m": float(
            task_quality.get("late_merge_request_threshold_m", 60.0)
        ),
    }
    return result


def _quantile_summary(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "p50": None, "p95": None}
    arr = np.asarray(values, dtype=np.float32)
    return {
        "mean": float(np.mean(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
    }


def _latency_quantiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"p50": None, "p95": None, "max": None}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
    }


def _accvp_observation_metrics(reports: list[dict]) -> dict:
    decision_count = int(sum(int(report.get("accvp_table_decision_count", 0)) for report in reports))
    activation_count = int(sum(int(report.get("accvp_table_activation_window_decision_count", 0)) for report in reports))
    valid_count = int(sum(int(report.get("accvp_table_valid_decision_count", 0)) for report in reports))
    activation_valid_count = int(
        sum(int(report.get("accvp_table_activation_window_valid_decision_count", 0)) for report in reports)
    )
    timeout_count = int(sum(int(report.get("accvp_table_timeout_count", 0)) for report in reports))
    fail_closed_count = int(sum(int(report.get("accvp_table_fail_closed_count", 0)) for report in reports))
    model_error_count = int(sum(int(report.get("accvp_table_model_error_count", 0)) for report in reports))
    invalid_bundle_count = int(sum(int(report.get("accvp_table_invalid_bundle_count", 0)) for report in reports))
    latencies: list[float] = []
    stage_latencies: dict[str, list[float]] = {
        "context_legality": [],
        "accvp_prepare_candidates": [],
        "accvp_score": [],
        "risk_secondary": [],
        "table_pack": [],
    }
    for report in reports:
        latencies.extend(float(value) for value in list(report.get("accvp_table_latency_total_s", []) or []))
        stage_payload = dict(report.get("accvp_table_latency_stage_s", {}) or {})
        for stage_name in stage_latencies:
            stage_latencies[stage_name].extend(
                float(value) for value in list(stage_payload.get(stage_name, []) or [])
            )
    latency_summary = _latency_quantiles(latencies)
    return {
        "accvp_table_decision_count": decision_count,
        "accvp_table_activation_window_decision_count": activation_count,
        "accvp_table_valid_decision_count": valid_count,
        "accvp_table_activation_window_valid_decision_count": activation_valid_count,
        "accvp_table_valid_rate_all_decisions": float(valid_count / decision_count) if decision_count else 0.0,
        "accvp_table_valid_rate_activation_window": (
            float(activation_valid_count / activation_count) if activation_count else 0.0
        ),
        "accvp_table_timeout_count": timeout_count,
        "accvp_table_timeout_rate": float(timeout_count / decision_count) if decision_count else 0.0,
        "accvp_table_fail_closed_count": fail_closed_count,
        "accvp_table_fail_closed_rate": float(fail_closed_count / decision_count) if decision_count else 0.0,
        "accvp_table_model_error_count": model_error_count,
        "accvp_table_invalid_bundle_count": invalid_bundle_count,
        "accvp_table_latency_count": int(len(latencies)),
        "accvp_table_latency_p50": latency_summary["p50"],
        "accvp_table_latency_p95": latency_summary["p95"],
        "accvp_table_latency_max": latency_summary["max"],
        "accvp_table_latency_per_stage": {
            name: _latency_quantiles(values) for name, values in stage_latencies.items()
        },
    }


def _policy_commitment_metrics(reports: list[dict]) -> dict:
    decision_count = int(sum(len(list(report.get("action_execution_records", []) or [])) for report in reports))
    started = int(sum(int(report.get("policy_commitment_started_count", 0) or 0) for report in reports))
    active = int(sum(int(report.get("policy_commitment_active_count", 0) or 0) for report in reports))
    action_change = int(sum(int(report.get("policy_commitment_action_change_count", 0) or 0) for report in reports))
    veto = int(sum(int(report.get("policy_commitment_veto_count", 0) or 0) for report in reports))
    target_entry_cancel = int(
        sum(int(report.get("policy_commitment_cancelled_by_target_entry_count", 0) or 0) for report in reports)
    )
    released_by_risk = int(
        sum(int(report.get("policy_commitment_released_by_risk_count", 0) or 0) for report in reports)
    )
    abort_penalty_applied = int(
        sum(int(report.get("persistence_abort_penalty_applied_count", 0) or 0) for report in reports)
    )
    continue_bonus_applied = int(
        sum(int(report.get("persistence_continue_bonus_applied_count", 0) or 0) for report in reports)
    )
    suppressed_by_risk = int(
        sum(int(report.get("persistence_suppressed_by_risk_count", 0) or 0) for report in reports)
    )
    abort_count = int(sum(int(report.get("raw_left_abort_before_entry_count", 0) or 0) for report in reports))
    left_request_count = int(sum(int(report.get("left_request_count_before_entry", 0) or 0) for report in reports))
    first_left_success_values = [
        float(report.get("first_left_request_to_target_entry_success_rate", 0.0) or 0.0)
        for report in reports
        if report.get("first_merge_request_step") is not None
    ]
    seed12_statuses = [
        str(report.get("seed12_repair_status", ""))
        for report in reports
        if str(report.get("seed12_repair_status", ""))
    ]
    case_records: list[dict] = []
    for report in reports:
        seed = report.get("seed", report.get("episode_seed", ""))
        final_done_reason = str(report.get("done_reason", ""))
        entry_distance = report.get("first_target_lane_entry_distance_to_taper")
        for record in list(report.get("action_execution_records", []) or []):
            involved = bool(
                record.get("policy_commitment_started", False)
                or record.get("policy_commitment_action_changed", False)
                or record.get("policy_commitment_vetoed", False)
                or record.get("policy_commitment_released_by_risk", False)
            )
            if not involved:
                continue
            case_records.append(
                {
                    "seed": seed,
                    "step": int(record.get("step", -1)),
                    "decision_index": int(record.get("decision_index", -1)),
                    "raw_action": int(record.get("raw_action", -1)),
                    "shield_input_action": int(record.get("shield_input_action", -1)),
                    "safety_shield_action": int(record.get("safety_shield_action", -1)),
                    "final_action": int(record.get("final_action", -1)),
                    "policy_commitment_started": bool(record.get("policy_commitment_started", False)),
                    "policy_commitment_action_changed": bool(record.get("policy_commitment_action_changed", False)),
                    "policy_commitment_vetoed": bool(record.get("policy_commitment_vetoed", False)),
                    "policy_commitment_released_by_risk": bool(record.get("policy_commitment_released_by_risk", False)),
                    "safety_shield_replaced": bool(record.get("safety_shield_replaced", False)),
                    "safety_shield_replacement_reason": str(record.get("safety_shield_replacement_reason", "")),
                    "merge_timing_persistence_abort_penalty": float(
                        record.get("merge_timing_persistence_abort_penalty", 0.0) or 0.0
                    ),
                    "merge_timing_persistence_continue_bonus": float(
                        record.get("merge_timing_persistence_continue_bonus", 0.0) or 0.0
                    ),
                    "merge_timing_persistence_suppressed_by_risk": bool(
                        record.get("merge_timing_persistence_suppressed_by_risk", False)
                    ),
                    "min_distance": float(record.get("min_distance", 0.0) or 0.0),
                    "min_ttc": float(record.get("min_ttc", 0.0) or 0.0),
                    "max_drac": float(record.get("max_drac", 0.0) or 0.0),
                    "final_done_reason": final_done_reason,
                    "first_target_lane_entry_distance_to_taper": entry_distance,
                }
            )
    return {
        "policy_commitment_started_count": started,
        "policy_commitment_active_count": active,
        "policy_commitment_action_change_count": action_change,
        "policy_commitment_veto_count": veto,
        "policy_commitment_cancelled_by_target_entry_count": target_entry_cancel,
        "policy_commitment_released_by_risk_count": released_by_risk,
        "policy_commitment_action_change_per_decision_rate": float(action_change / max(1, decision_count)),
        "persistence_abort_penalty_applied_count": abort_penalty_applied,
        "persistence_continue_bonus_applied_count": continue_bonus_applied,
        "persistence_suppressed_by_risk_count": suppressed_by_risk,
        "raw_left_abort_before_entry_count": abort_count,
        "left_request_count_before_entry": left_request_count,
        "merge_intent_persistence_rate": float(left_request_count / max(1, left_request_count + abort_count)),
        "first_left_request_to_target_entry_success_rate": (
            float(np.mean(first_left_success_values)) if first_left_success_values else 0.0
        ),
        "seed12_repair_status": seed12_statuses[0] if seed12_statuses else "",
        "policy_commitment_case_records": case_records,
    }


def _accvp_shadow_metrics(reports: list[dict]) -> dict:
    records = [record for report in reports for record in list(report.get("accvp_records", []) or [])]
    accvp_record_count = int(
        sum(int(report.get("accvp_record_count", len(list(report.get("accvp_records", []) or [])))) for report in reports)
    )
    accvp_replacement_counts: list[int] = []
    accvp_action_change_counts: list[int] = []
    accvp_same_action_confirm_counts: list[int] = []
    accvp_commitment_counts: list[int] = []
    accvp_replacement_reason_counts: Counter[str] = Counter()
    accvp_replacement_selected_counts: Counter[str] = Counter()
    accvp_action_change_reason_counts: Counter[str] = Counter()
    accvp_action_change_selected_counts: Counter[str] = Counter()
    for report in reports:
        report_records = list(report.get("accvp_records", []) or [])
        if "accvp_replacement_count" in report:
            replacement_count = int(report.get("accvp_replacement_count", 0) or 0)
        else:
            replacement_count = 0
            for record in report_records:
                reason = str(record.get("accvp_replacement_reason", ""))
                if bool(record.get("accvp_replacement", False)) and reason != "lateral_commitment":
                    replacement_count += 1
        action_change_count = 0
        same_action_confirm_count = 0
        commitment_count = 0
        for record in report_records:
            reason = str(record.get("accvp_replacement_reason", ""))
            selected = record.get("accvp_selected_action")
            shield_action = record.get("safety_shield_action")
            if bool(record.get("accvp_replacement", False)) and reason == "lateral_commitment":
                commitment_count += 1
            elif bool(record.get("accvp_replacement", False)) and reason:
                accvp_replacement_reason_counts[reason] += 1
                accvp_replacement_selected_counts["None" if selected is None else str(int(selected))] += 1
                action_change = bool(record.get("accvp_action_change", False))
                if "accvp_action_change" not in record and selected is not None and shield_action is not None:
                    action_change = int(selected) != int(shield_action)
                if action_change:
                    action_change_count += 1
                    accvp_action_change_reason_counts[reason] += 1
                    accvp_action_change_selected_counts[
                        "None" if selected is None else str(int(selected))
                    ] += 1
                else:
                    same_action_confirm_count += 1
        accvp_replacement_counts.append(replacement_count)
        accvp_action_change_counts.append(action_change_count)
        accvp_same_action_confirm_counts.append(same_action_confirm_count)
        accvp_commitment_counts.append(commitment_count)
    total_accvp_replacements = int(sum(accvp_replacement_counts))
    total_accvp_action_changes = int(sum(accvp_action_change_counts))
    total_accvp_same_action_confirms = int(sum(accvp_same_action_confirm_counts))
    total_accvp_commitments = int(sum(accvp_commitment_counts))
    accvp_episode_denominator = max(1, len(reports))
    accvp_decision_denominator = max(1, accvp_record_count)
    accvp_active_defaults = {
        "accvp_active_replacement_count": total_accvp_replacements,
        "accvp_active_replacement_count_semantics": (
            "legacy_count_of_non_commitment_accvp_replacement_flags; "
            "use accvp_active_action_change_count for true action changes"
        ),
        "accvp_active_replacement_episode_rate": float(
            sum(1 for item in accvp_replacement_counts if item > 0) / accvp_episode_denominator
        ),
        "accvp_active_replacement_per_decision_rate": float(
            total_accvp_replacements / accvp_decision_denominator
        ),
        "accvp_active_action_change_count": total_accvp_action_changes,
        "accvp_active_action_change_episode_rate": float(
            sum(1 for item in accvp_action_change_counts if item > 0) / accvp_episode_denominator
        ),
        "accvp_active_action_change_per_decision_rate": float(
            total_accvp_action_changes / accvp_decision_denominator
        ),
        "accvp_active_same_action_confirm_count": total_accvp_same_action_confirms,
        "accvp_active_same_action_confirm_episode_rate": float(
            sum(1 for item in accvp_same_action_confirm_counts if item > 0) / accvp_episode_denominator
        ),
        "accvp_active_commitment_count": total_accvp_commitments,
        "accvp_active_commitment_replacement_count": total_accvp_commitments,
        "accvp_active_replacement_reason_counts": dict(sorted(accvp_replacement_reason_counts.items())),
        "accvp_active_replacement_selected_action_counts": dict(
            sorted(
                accvp_replacement_selected_counts.items(),
                key=lambda item: 999 if item[0] == "None" else int(item[0]),
            )
        ),
        "accvp_active_action_change_reason_counts": dict(sorted(accvp_action_change_reason_counts.items())),
        "accvp_active_action_change_selected_action_counts": dict(
            sorted(
                accvp_action_change_selected_counts.items(),
                key=lambda item: 999 if item[0] == "None" else int(item[0]),
            )
        ),
    }
    if not records:
        return {
            "accvp_shadow_record_count": 0,
            "accvp_shadow_candidate_set_available_rate": 0.0,
            "accvp_shadow_raw_feasible_rate": 0.0,
            "accvp_shadow_recommended_diff_rate": 0.0,
            "accvp_shadow_recommended_merge_intent_rate": 0.0,
            "accvp_shadow_bypass_rate": 0.0,
            "accvp_shadow_timeout_rate": 0.0,
            "accvp_shadow_latency_p50": 0.0,
            "accvp_shadow_latency_p95": 0.0,
            "accvp_shadow_per_action_gate_pass_rate": {},
            "accvp_shadow_per_action_summary": {},
            "accvp_shadow_recommended_action_counts": {},
            "accvp_shadow_raw_probabilities": {},
            "accvp_shadow_raw_bounds": {},
            "accvp_lite_raw_task_feasible_rate": 0.0,
            "accvp_lite_replacement_would_trigger_rate": 0.0,
            "accvp_lite_best_left_action_counts": {},
            "accvp_lite_p_merge_improvement": {},
            "accvp_lite_per_action_gate_pass_rate": {},
            **accvp_active_defaults,
        }
    merge_intent = {6, 7, 8}
    candidate_available = [bool(record.get("candidate_set_available", False)) for record in records]
    raw_feasible = [bool(record.get("raw_feasible", False)) for record in records]
    recommended = [record.get("accvp_shadow_recommended_action") for record in records]
    raw_actions = [record.get("raw_action") for record in records]
    bypass = [bool(str(record.get("accvp_bypass_reason", ""))) for record in records]
    timeout = [str(record.get("accvp_bypass_reason", "")) == "timeout" for record in records]
    latency = [float(record.get("decision_latency_s", 0.0)) for record in records]
    per_action_counts: dict[str, int] = {}
    per_action_pass: dict[str, int] = {}
    per_action_values: dict[str, dict[str, list[float]]] = {}
    recommended_counts: Counter[str] = Counter()
    raw_p_proxy: list[float] = []
    raw_p_safety: list[float] = []
    raw_p_viability: list[float] = []
    raw_pu_proxy: list[float] = []
    raw_pu_safety: list[float] = []
    raw_pl_viability: list[float] = []
    lite_task_feasible = [bool(record.get("accvp_lite_raw_task_feasible", False)) for record in records]
    lite_best_left = [record.get("accvp_lite_best_left_action") for record in records]
    lite_improvement = [float(record.get("accvp_lite_p_merge_improvement", 0.0)) for record in records]
    lite_best_left_counts: Counter[str] = Counter()
    per_action_lite_pass: dict[str, int] = {}
    for item in lite_best_left:
        lite_best_left_counts["None" if item is None else str(int(item))] += 1
    for item in recommended:
        recommended_counts["None" if item is None else str(int(item))] += 1
    for record in records:
        raw_action = int(record.get("raw_action", -1))
        for candidate in list(record.get("accvp_shadow_candidates", []) or []):
            action_id = str(int(candidate.get("action_id", -1)))
            per_action_counts[action_id] = per_action_counts.get(action_id, 0) + 1
            per_action_values.setdefault(
                action_id,
                {
                    "p_proxy_collision": [],
                    "p_safety_violation": [],
                    "p_merge_before_taper": [],
                    "pU_proxy_collision": [],
                    "pU_safety_violation": [],
                    "pL_merge_before_taper": [],
                    "ensemble_disagreement": [],
                },
            )
            for key, values in per_action_values[action_id].items():
                if candidate.get(key) is not None:
                    values.append(float(candidate.get(key, 0.0)))
            if bool(candidate.get("gate_pass", False)):
                per_action_pass[action_id] = per_action_pass.get(action_id, 0) + 1
            if bool(candidate.get("lite_gate_pass", False)):
                per_action_lite_pass[action_id] = per_action_lite_pass.get(action_id, 0) + 1
            if int(candidate.get("action_id", -999)) == raw_action:
                raw_p_proxy.append(float(candidate.get("p_proxy_collision", 0.0)))
                raw_p_safety.append(float(candidate.get("p_safety_violation", 0.0)))
                raw_p_viability.append(float(candidate.get("p_merge_before_taper", 0.0)))
                raw_pu_proxy.append(float(candidate.get("pU_proxy_collision", 0.0)))
                raw_pu_safety.append(float(candidate.get("pU_safety_violation", 0.0)))
                raw_pl_viability.append(float(candidate.get("pL_merge_before_taper", 0.0)))
    pass_rate = {
        action_id: float(per_action_pass.get(action_id, 0) / max(1, count))
        for action_id, count in sorted(per_action_counts.items(), key=lambda item: int(item[0]))
    }
    lite_pass_rate = {
        action_id: float(per_action_lite_pass.get(action_id, 0) / max(1, count))
        for action_id, count in sorted(per_action_counts.items(), key=lambda item: int(item[0]))
    }
    per_action_summary = {
        action_id: {
            "count": int(per_action_counts.get(action_id, 0)),
            "gate_pass_rate": pass_rate.get(action_id, 0.0),
            **{key: _quantile_summary(values) for key, values in values_by_key.items()},
        }
        for action_id, values_by_key in sorted(per_action_values.items(), key=lambda item: int(item[0]))
    }
    return {
        "accvp_shadow_record_count": int(len(records)),
        "accvp_shadow_candidate_set_available_rate": float(np.mean(candidate_available)),
        "accvp_shadow_raw_feasible_rate": float(np.mean(raw_feasible)),
        "accvp_shadow_recommended_diff_rate": float(
            np.mean([
                item is not None and raw is not None and int(item) != int(raw)
                for item, raw in zip(recommended, raw_actions)
            ])
        ),
        "accvp_shadow_recommended_merge_intent_rate": float(
            np.mean([item is not None and int(item) in merge_intent for item in recommended])
        ),
        "accvp_shadow_bypass_rate": float(np.mean(bypass)),
        "accvp_shadow_timeout_rate": float(np.mean(timeout)),
        "accvp_shadow_latency_p50": float(np.percentile(latency, 50)),
        "accvp_shadow_latency_p95": float(np.percentile(latency, 95)),
        "accvp_shadow_per_action_gate_pass_rate": pass_rate,
        "accvp_shadow_per_action_summary": per_action_summary,
        "accvp_shadow_recommended_action_counts": dict(
            sorted(recommended_counts.items(), key=lambda item: 999 if item[0] == "None" else int(item[0]))
        ),
        "accvp_shadow_raw_probabilities": {
            "p_proxy_collision": _quantile_summary(raw_p_proxy),
            "p_safety_violation": _quantile_summary(raw_p_safety),
            "p_merge_before_taper": _quantile_summary(raw_p_viability),
        },
        "accvp_shadow_raw_bounds": {
            "pU_proxy_collision": _quantile_summary(raw_pu_proxy),
            "pU_safety_violation": _quantile_summary(raw_pu_safety),
            "pL_merge_before_taper": _quantile_summary(raw_pl_viability),
        },
        "accvp_lite_raw_task_feasible_rate": float(np.mean(lite_task_feasible)),
        "accvp_lite_replacement_would_trigger_rate": float(
            np.mean([
                item is not None and raw is not None and int(item) != int(raw)
                for item, raw in zip(recommended, raw_actions)
            ])
        ),
        "accvp_lite_best_left_action_counts": dict(
            sorted(lite_best_left_counts.items(), key=lambda item: 999 if item[0] == "None" else int(item[0]))
        ),
        "accvp_lite_p_merge_improvement": _quantile_summary(lite_improvement),
        "accvp_lite_per_action_gate_pass_rate": lite_pass_rate,
        **accvp_active_defaults,
    }


def aggregate_episode_reports(reports: list[dict], task_quality: dict | None = None) -> dict:
    if not reports:
        return {}
    thresholds = _task_quality_thresholds(task_quality)
    collisions = np.asarray([float(report.get("collision", False)) for report in reports], dtype=np.float32)
    near_misses = np.asarray([float(report.get("near_miss", False)) for report in reports], dtype=np.float32)
    proxy_collisions = np.asarray([_report_proxy_collision(report) for report in reports], dtype=np.float32)
    safety_violations = np.asarray([_report_safety_violation(report) for report in reports], dtype=np.float32)
    proxy_collision_counts = np.asarray(
        [float(report.get("proxy_collision_count", _report_proxy_collision(report))) for report in reports],
        dtype=np.float32,
    )
    safety_violation_counts = np.asarray(
        [float(report.get("safety_violation_count", _report_safety_violation(report))) for report in reports],
        dtype=np.float32,
    )
    min_distance_collision_counts = np.asarray(
        [
            float(
                report.get(
                    "min_distance_le_collision_threshold_count",
                    report.get("proxy_collision_count", _report_proxy_collision(report)),
                )
            )
            for report in reports
        ],
        dtype=np.float32,
    )
    min_distances = np.asarray([float(report.get("min_distance", 0.0)) for report in reports], dtype=np.float32)
    ttc = np.asarray([float(report.get("ttc_p1", 1.0e6)) for report in reports], dtype=np.float32)
    drac_raw = np.asarray(
        [float(report.get("drac_p99_raw", report.get("drac_p99", 0.0))) for report in reports],
        dtype=np.float32,
    )
    drac_capped = np.asarray(
        [
            float(report.get("drac_p99_capped", min(float(report.get("drac_p99", 0.0)), DRAC_REPORT_CAP_DEFAULT)))
            for report in reports
        ],
        dtype=np.float32,
    )
    steps = np.asarray([float(report.get("steps", 0.0)) for report in reports], dtype=np.float32)
    completion_time = np.asarray([float(report.get("completion_time", 0.0)) for report in reports], dtype=np.float32)
    ego_speed_mean = np.asarray([float(report.get("ego_speed_mean", 0.0)) for report in reports], dtype=np.float32)
    ego_speed_p10 = np.asarray([float(report.get("ego_speed_p10", 0.0)) for report in reports], dtype=np.float32)
    hard_brake_rates = np.asarray([float(report.get("hard_brake_rate", 0.0)) for report in reports], dtype=np.float32)
    interventions = np.asarray([float(report.get("intervention_count", 0)) for report in reports], dtype=np.float32)
    shield_calls = np.asarray([float(report.get("shield_call_count", report.get("intervention_count", 0))) for report in reports], dtype=np.float32)
    replacements = np.asarray([float(report.get("actual_replacement_count", 0)) for report in reports], dtype=np.float32)
    task_replacements = np.asarray(
        [float(report.get("task_replacement_count", 0)) for report in reports],
        dtype=np.float32,
    )
    forecast_ranking_replacements = np.asarray(
        [float(report.get("forecast_ranking_replacement_count", 0)) for report in reports],
        dtype=np.float32,
    )
    fallbacks = np.asarray([float(report.get("fallback_count", 0)) for report in reports], dtype=np.float32)
    emergency_fallbacks = np.asarray(
        [float(report.get("emergency_fallback_count", 0)) for report in reports],
        dtype=np.float32,
    )
    taper_misses = np.asarray([float(bool(report.get("taper_miss", False))) for report in reports], dtype=np.float32)
    geometric_overlaps = np.asarray(
        [float(bool(report.get("geometric_overlap", False))) for report in reports],
        dtype=np.float32,
    )
    first_merge_request_distance = np.asarray(
        [
            float(report["first_merge_request_distance_to_taper"])
            for report in reports
            if report.get("first_merge_request_distance_to_taper") is not None
        ],
        dtype=np.float32,
    )
    first_target_entry_distance = np.asarray(
        [
            float(report["first_target_lane_entry_distance_to_taper"])
            for report in reports
            if report.get("first_target_lane_entry_distance_to_taper") is not None
        ],
        dtype=np.float32,
    )
    terminal_successes = np.asarray(
        [
            float(
                bool(report.get("merge_success", False))
                or str(report.get("done_reason", "")) == "merge_success"
            )
            for report in reports
        ],
        dtype=np.float32,
    )
    timely_merge_successes = np.asarray(
        [
            float(
                (
                    bool(report.get("merge_success", False))
                    or str(report.get("done_reason", "")) == "merge_success"
                )
                and report.get("first_target_lane_entry_distance_to_taper") is not None
                and float(report.get("first_target_lane_entry_distance_to_taper", -1.0))
                >= thresholds["timely_entry_distance_threshold_m"]
            )
            for report in reports
        ],
        dtype=np.float32,
    )
    timely_merge_requests = np.asarray(
        [
            float(
                report.get("first_merge_request_distance_to_taper") is not None
                and float(report.get("first_merge_request_distance_to_taper", -1.0))
                >= thresholds["late_merge_request_threshold_m"]
            )
            for report in reports
        ],
        dtype=np.float32,
    )
    late_merge_requests = np.asarray(
        [
            float(
                report.get("first_merge_request_distance_to_taper") is None
                or float(report.get("first_merge_request_distance_to_taper", -1.0))
                < thresholds["late_merge_request_threshold_m"]
            )
            for report in reports
        ],
        dtype=np.float32,
    )
    safety_clean_successes = np.asarray(
        [
            float(
                (
                    bool(report.get("merge_success", False))
                    or str(report.get("done_reason", "")) == "merge_success"
                )
                and not bool(report.get("proxy_collision", False))
                and not bool(report.get("safety_violation", False))
                and not bool(report.get("geometric_overlap", False))
                and int(report.get("fallback_count", 0)) == 0
            )
            for report in reports
        ],
        dtype=np.float32,
    )
    safe_merge_opportunities = int(sum(int(report.get("safe_merge_opportunity_count", 0)) for report in reports))
    missed_safe_merge_opportunities = int(
        sum(int(report.get("missed_safe_merge_opportunity_count", 0)) for report in reports)
    )
    task_merge_opportunities = int(sum(int(report.get("task_merge_opportunity_count", 0)) for report in reports))
    task_would_merges = int(sum(int(report.get("task_would_merge_count", 0)) for report in reports))
    task_missed_merges = int(sum(int(report.get("task_missed_merge_count", 0)) for report in reports))
    deadline_opportunities = int(
        sum(int(report.get("deadline_safe_merge_opportunity_count", 0)) for report in reports)
    )
    deadline_missed = int(sum(int(report.get("deadline_missed_safe_merge_count", 0)) for report in reports))
    pre_entry_deadline_opportunities = int(
        sum(int(report.get("pre_entry_deadline_safe_merge_opportunity_count", 0)) for report in reports)
    )
    pre_entry_deadline_missed = int(
        sum(int(report.get("pre_entry_deadline_missed_safe_merge_count", 0)) for report in reports)
    )
    post_entry_opportunity_excluded = int(
        sum(int(report.get("post_entry_opportunity_excluded_count", 0)) for report in reports)
    )
    early_merge_success_before_deadline = int(
        sum(int(report.get("early_merge_success_before_deadline_count", 0)) for report in reports)
    )
    urgency_missed = int(
        sum(int(report.get("missed_safe_merge_after_urgency_0_5_count", 0)) for report in reports)
    )
    urgency_opportunities = int(
        sum(int(report.get("safe_merge_after_urgency_0_5_count", 0)) for report in reports)
    )
    no_merge_before_taper = np.asarray(
        [float(report.get("no_merge_request_before_taper_count", 0)) for report in reports],
        dtype=np.float32,
    )
    forecast_record_count = int(
        sum(int(report.get("forecast_record_count", 0)) for report in reports)
    )
    forecast_coverage_complete_count = int(
        sum(int(report.get("forecast_actor_coverage_complete_count", 0)) for report in reports)
    )
    forecast_gap_checkable_count = int(
        sum(
            int(report.get("forecast_gap_consistency_checkable_count", 0))
            for report in reports
        )
    )
    forecast_gap_pass_count = int(
        sum(
            int(report.get("forecast_gap_consistency_pass_count", 0))
            for report in reports
        )
    )
    wcdt_relevant_coverage_count = int(
        sum(int(report.get("wcdt_relevant_actor_coverage_count", 0)) for report in reports)
    )
    combined_safety_coverage_count = int(
        sum(int(report.get("combined_forecast_safety_coverage_count", 0)) for report in reports)
    )
    selector_overflow_count = int(
        sum(int(report.get("actor_selector_overflow_count", 0)) for report in reports)
    )
    critical_overflow_count = int(
        sum(int(report.get("critical_actor_overflow_count", report.get("actor_selector_overflow_count", 0))) for report in reports)
    )
    critical_wcdt_coverage_count = int(
        sum(int(report.get("critical_wcdt_coverage_count", 0)) for report in reports)
    )
    combined_critical_coverage_count = int(
        sum(int(report.get("combined_critical_coverage_count", 0)) for report in reports)
    )
    cv_fallback_overflow_count = int(
        sum(int(report.get("cv_fallback_overflow_count", 0)) for report in reports)
    )
    cv_fallback_usage_count = int(
        sum(int(report.get("cv_fallback_usage_count", 0)) for report in reports)
    )
    reward_component_names = (
        "progress_reward",
        "speed_reward",
        "terminal_reward",
        "lane_oob_penalty",
        "safety_penalty",
        "safety_forecast_shaping",
        "shield_guided_shaping",
        "merge_timing_shaping",
        "total_episode_reward",
    )
    reward_component_means = {
        name: float(np.mean([float(report.get(name, 0.0)) for report in reports]))
        for name in reward_component_names
    }
    raw_lane_oob_count = int(sum(int(report.get("raw_action_lane_oob_count", 0)) for report in reports))
    final_lane_oob_count = int(sum(int(report.get("final_action_lane_oob_count", 0)) for report in reports))
    prevented_lane_oob_count = int(sum(int(report.get("prevented_lane_oob_count", 0)) for report in reports))
    task_backstop_watch_count = int(sum(int(report.get("task_backstop_watch_count", 0)) for report in reports))
    task_backstop_eligible_count = int(
        sum(int(report.get("task_backstop_eligible_count", 0)) for report in reports)
    )
    task_backstop_veto_reason_counts: Counter[str] = Counter()
    task_replacement_reason_counts: Counter[str] = Counter()
    forecast_ranking_replacement_reason_counts: Counter[str] = Counter()
    for report in reports:
        task_backstop_veto_reason_counts.update(report.get("task_backstop_veto_reason_counts", {}) or {})
        task_replacement_reason_counts.update(report.get("task_replacement_reason_counts", {}) or {})
        forecast_ranking_replacement_reason_counts.update(
            report.get("forecast_ranking_replacement_reason_counts", {}) or {}
        )
    result = {
        "episodes": len(reports),
        "collision_rate": float(np.mean(collisions)),
        "near_miss_rate": float(np.mean(near_misses)),
        "geometric_overlap_rate": float(np.mean(geometric_overlaps)),
        "geometric_overlap_count": int(np.sum(geometric_overlaps)),
        "proxy_collision_rate": float(np.mean(proxy_collisions)),
        "safety_violation_rate": float(np.mean(safety_violations)),
        "proxy_collision_count": int(np.sum(proxy_collision_counts)),
        "safety_violation_count": int(np.sum(safety_violation_counts)),
        "min_distance_le_collision_threshold_count": int(np.sum(min_distance_collision_counts)),
        "min_distance_p1": float(np.percentile(min_distances, 1)),
        "ttc_p1": float(np.percentile(ttc, 1)),
        "drac_p99": float(np.percentile(drac_raw, 99)),
        "drac_p99_raw": float(np.percentile(drac_raw, 99)),
        "drac_p99_capped": float(np.percentile(drac_capped, 99)),
        "steps_mean": float(np.mean(steps)),
        "steps_p95": float(np.percentile(steps, 95)),
        "completion_time_mean": float(np.mean(completion_time)),
        "completion_time_p95": float(np.percentile(completion_time, 95)),
        "ego_speed_mean": float(np.mean(ego_speed_mean)),
        "ego_speed_p10": float(np.percentile(ego_speed_p10, 10)),
        "hard_brake_rate": float(np.mean(hard_brake_rates)),
        "intervention_rate": float(np.mean(interventions > 0)),
        "shield_call_rate": float(np.mean(shield_calls > 0)),
        "mean_shield_calls": float(np.mean(shield_calls)),
        "actual_replacement_rate": float(np.mean(replacements > 0)),
        "mean_actual_replacements": float(np.mean(replacements)),
        "actual_replacement_rate_semantics": "episodes_with_replacement_rate",
        "episodes_with_replacement_rate": float(np.mean(replacements > 0)),
        "replacement_per_shield_call_rate": (
            float(np.sum(replacements) / np.sum(shield_calls))
            if np.sum(shield_calls) > 0
            else 0.0
        ),
        "mean_replacements_per_episode": float(np.mean(replacements)),
        "task_replacement_rate": float(np.mean(task_replacements > 0)),
        "mean_task_replacements": float(np.mean(task_replacements)),
        "task_replacement_count": int(np.sum(task_replacements)),
        "forecast_ranking_replacement_rate": float(np.mean(forecast_ranking_replacements > 0)),
        "mean_forecast_ranking_replacements": float(np.mean(forecast_ranking_replacements)),
        "forecast_ranking_replacement_count": int(np.sum(forecast_ranking_replacements)),
        "fallback_rate": float(np.mean(fallbacks > 0)),
        "emergency_fallback_rate": float(np.mean(emergency_fallbacks > 0)),
        "mean_emergency_fallbacks": float(np.mean(emergency_fallbacks)),
        "emergency_fallback_count": int(np.sum(emergency_fallbacks)),
        "taper_miss_rate": float(np.mean(taper_misses)) if taper_misses.size else 0.0,
        "taper_miss_count": int(np.sum(taper_misses)),
        "task_quality_thresholds": thresholds,
        "terminal_success_rate": float(np.mean(terminal_successes)),
        "terminal_success_count": int(np.sum(terminal_successes)),
        "timely_merge_success_rate": float(np.mean(timely_merge_successes)),
        "timely_merge_success_count": int(np.sum(timely_merge_successes)),
        "timely_merge_request_rate": float(np.mean(timely_merge_requests)),
        "timely_merge_request_count": int(np.sum(timely_merge_requests)),
        "safety_clean_success_rate": float(np.mean(safety_clean_successes)),
        "safety_clean_success_count": int(np.sum(safety_clean_successes)),
        "late_merge_request_rate": float(np.mean(late_merge_requests)),
        "late_merge_request_count": int(np.sum(late_merge_requests)),
        "first_merge_request_distance_to_taper_mean": (
            float(np.mean(first_merge_request_distance)) if first_merge_request_distance.size else None
        ),
        "first_merge_request_distance_to_taper_p10": (
            float(np.percentile(first_merge_request_distance, 10)) if first_merge_request_distance.size else None
        ),
        "first_merge_request_distance_to_taper_p50": (
            float(np.percentile(first_merge_request_distance, 50)) if first_merge_request_distance.size else None
        ),
        "first_merge_request_distance_to_taper_p90": (
            float(np.percentile(first_merge_request_distance, 90)) if first_merge_request_distance.size else None
        ),
        "first_target_lane_entry_distance_to_taper_mean": (
            float(np.mean(first_target_entry_distance)) if first_target_entry_distance.size else None
        ),
        "first_target_lane_entry_distance_to_taper_p10": (
            float(np.percentile(first_target_entry_distance, 10)) if first_target_entry_distance.size else None
        ),
        "first_target_lane_entry_distance_to_taper_p50": (
            float(np.percentile(first_target_entry_distance, 50)) if first_target_entry_distance.size else None
        ),
        "first_target_lane_entry_distance_to_taper_p90": (
            float(np.percentile(first_target_entry_distance, 90)) if first_target_entry_distance.size else None
        ),
        "safe_merge_opportunity_count": safe_merge_opportunities,
        "missed_safe_merge_opportunity_count": missed_safe_merge_opportunities,
        "missed_safe_merge_opportunity_rate": (
            float(missed_safe_merge_opportunities / safe_merge_opportunities)
            if safe_merge_opportunities
            else 0.0
        ),
        "opportunity_capture_step_rate": (
            float(1.0 - missed_safe_merge_opportunities / safe_merge_opportunities)
            if safe_merge_opportunities
            else 0.0
        ),
        "task_merge_opportunity_count": task_merge_opportunities,
        "task_would_merge_count": task_would_merges,
        "task_would_merge_rate": (
            float(task_would_merges / task_merge_opportunities) if task_merge_opportunities else 0.0
        ),
        "task_missed_merge_count": task_missed_merges,
        "task_missed_merge_rate": (
            float(task_missed_merges / task_merge_opportunities) if task_merge_opportunities else 0.0
        ),
        "deadline_safe_merge_opportunity_count": deadline_opportunities,
        "deadline_missed_safe_merge_count": deadline_missed,
        "deadline_missed_safe_merge_rate": (
            float(deadline_missed / deadline_opportunities) if deadline_opportunities else 0.0
        ),
        "deadline_opportunity_capture_rate": (
            float(1.0 - deadline_missed / deadline_opportunities) if deadline_opportunities else 0.0
        ),
        "pre_entry_deadline_safe_merge_opportunity_count": pre_entry_deadline_opportunities,
        "pre_entry_deadline_missed_safe_merge_count": pre_entry_deadline_missed,
        "pre_entry_deadline_missed_safe_merge_rate": (
            float(pre_entry_deadline_missed / pre_entry_deadline_opportunities)
            if pre_entry_deadline_opportunities
            else 0.0
        ),
        "pre_entry_deadline_opportunity_capture_rate": (
            float(1.0 - pre_entry_deadline_missed / pre_entry_deadline_opportunities)
            if pre_entry_deadline_opportunities
            else 0.0
        ),
        "post_entry_opportunity_excluded_count": post_entry_opportunity_excluded,
        "early_merge_success_before_deadline_count": early_merge_success_before_deadline,
        "early_merge_success_before_deadline_rate": (
            float(early_merge_success_before_deadline / len(reports)) if reports else 0.0
        ),
        "missed_safe_merge_after_urgency_0_5_count": urgency_missed,
        "safe_merge_after_urgency_0_5_count": urgency_opportunities,
        "missed_safe_merge_after_urgency_0_5_rate": (
            float(urgency_missed / urgency_opportunities) if urgency_opportunities else 0.0
        ),
        "no_merge_request_before_taper_count": int(np.sum(no_merge_before_taper)),
        "no_merge_request_before_taper_rate": float(np.mean(no_merge_before_taper > 0)),
        "forecast_actor_coverage_complete_count": forecast_coverage_complete_count,
        "forecast_actor_coverage_complete_rate": (
            float(forecast_coverage_complete_count / forecast_record_count)
            if forecast_record_count
            else 0.0
        ),
        "forecast_record_count": forecast_record_count,
        "forecast_gap_consistency_checkable_count": forecast_gap_checkable_count,
        "forecast_gap_consistency_pass_count": forecast_gap_pass_count,
        "forecast_gap_consistency_checkable_rate": (
            float(forecast_gap_checkable_count / forecast_record_count)
            if forecast_record_count
            else 0.0
        ),
        "forecast_gap_consistency_pass_rate": (
            float(forecast_gap_pass_count / forecast_gap_checkable_count)
            if forecast_gap_checkable_count
            else 0.0
        ),
        "wcdt_relevant_actor_coverage_count": wcdt_relevant_coverage_count,
        "wcdt_relevant_actor_coverage_rate": (
            float(wcdt_relevant_coverage_count / forecast_record_count)
            if forecast_record_count
            else 0.0
        ),
        "combined_forecast_safety_coverage_count": combined_safety_coverage_count,
        "combined_forecast_safety_coverage_rate": (
            float(combined_safety_coverage_count / forecast_record_count)
            if forecast_record_count
            else 0.0
        ),
        "actor_selector_overflow_count": selector_overflow_count,
        "actor_selector_overflow_rate": (
            float(selector_overflow_count / forecast_record_count)
            if forecast_record_count
            else 0.0
        ),
        "critical_actor_overflow_count": critical_overflow_count,
        "critical_actor_overflow_rate": (
            float(critical_overflow_count / forecast_record_count)
            if forecast_record_count
            else 0.0
        ),
        "critical_wcdt_coverage_count": critical_wcdt_coverage_count,
        "critical_wcdt_coverage_rate": (
            float(critical_wcdt_coverage_count / forecast_record_count)
            if forecast_record_count
            else 0.0
        ),
        "combined_critical_coverage_count": combined_critical_coverage_count,
        "combined_critical_coverage_rate": (
            float(combined_critical_coverage_count / forecast_record_count)
            if forecast_record_count
            else 0.0
        ),
        "cv_fallback_overflow_count": cv_fallback_overflow_count,
        "cv_fallback_overflow_rate": (
            float(cv_fallback_overflow_count / forecast_record_count)
            if forecast_record_count
            else 0.0
        ),
        "cv_fallback_usage_count": cv_fallback_usage_count,
        "cv_fallback_usage_rate": (
            float(cv_fallback_usage_count / forecast_record_count)
            if forecast_record_count
            else 0.0
        ),
        "raw_action_lane_oob_count": raw_lane_oob_count,
        "final_action_lane_oob_count": final_lane_oob_count,
        "prevented_lane_oob_count": prevented_lane_oob_count,
        **reward_component_means,
        "task_backstop_watch_count": task_backstop_watch_count,
        "task_backstop_eligible_count": task_backstop_eligible_count,
        "task_backstop_veto_reason_counts": dict(task_backstop_veto_reason_counts),
        "task_replacement_reason_counts": dict(task_replacement_reason_counts),
        "forecast_ranking_replacement_reason_counts": dict(
            forecast_ranking_replacement_reason_counts
        ),
    }
    result.update(_accvp_observation_metrics(reports))
    result.update(_policy_commitment_metrics(reports))
    result.update(_accvp_shadow_metrics(reports))
    return result
