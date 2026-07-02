"""One deterministic ACCVP selection rule shared by runtime and offline reports."""

from __future__ import annotations

from typing import Any

from safe_rl.sim.action_space import action_distance, decode_action


LEFT_ACTION_IDS = {6, 7, 8}


def _candidate_action(row: dict[str, Any]):
    return row.get("action") or decode_action(int(row["action_id"]))


def _secondary_risk_score(row: dict[str, Any]) -> float:
    if row.get("secondary_risk_score") is not None:
        return float(row.get("secondary_risk_score", 0.0))
    secondary = dict(row.get("secondary_risk", {}) or {})
    return float(secondary.get("risk_score", 0.0))


def _candidate_legal(row: dict[str, Any]) -> bool:
    return bool(row.get("candidate_legal", True))


def _strict_secondary_pass(row: dict[str, Any]) -> bool:
    return _candidate_legal(row) and bool(row.get("secondary_safety_pass", True))


def lite_secondary_safety_pass(row: dict[str, Any], thresholds: dict[str, Any]) -> bool:
    """Versioned secondary safety predicate for ACV-Shield-lite.

    ``strict`` preserves the existing Risk secondary pass semantics.  The
    audited profile is intentionally narrow: it can only affect merge-left
    candidates and still relies on the Risk score rather than ACCVP safety
    heads.
    """

    if not _candidate_legal(row):
        return False
    max_score = float(thresholds.get("max_secondary_risk_score", float("inf")))
    profile = str(thresholds.get("secondary_safety_profile", "strict"))
    action_id = int(row.get("action_id", -1))
    if profile == "audited_merge_left_v1" and action_id in LEFT_ACTION_IDS:
        return _secondary_risk_score(row) <= max_score
    return _strict_secondary_pass(row) and _secondary_risk_score(row) <= max_score


def gate_candidates(candidates: list[dict[str, Any]], thresholds: dict[str, float]) -> list[dict[str, Any]]:
    accepted: list[dict[str, Any]] = []
    for source in candidates:
        row = source
        passed = (
            bool(row.get("candidate_legal", True))
            and bool(row.get("secondary_safety_pass", True))
            and float(row["pU_proxy_collision"]) <= float(thresholds["proxy_collision_upper_bound"])
            and float(row["pU_safety_violation"]) <= float(thresholds["safety_violation_upper_bound"])
            and float(row["pL_merge_before_taper"]) >= float(thresholds["merge_viability_lower_bound"])
        )
        row["accvp_gate_pass"] = bool(passed)
        if passed:
            accepted.append(row)
    return accepted


def select_viability_action(
    candidates: list[dict[str, Any]],
    *,
    raw_action_id: int,
    thresholds: dict[str, float],
) -> dict[str, Any]:
    """Apply ACCVP's raw-retention rule and deterministic viability tie-breaks."""

    accepted = gate_candidates(candidates, thresholds)
    raw = next((row for row in accepted if int(row["action_id"]) == int(raw_action_id)), None)
    if raw is not None:
        return {
            "selected": raw,
            "accepted": accepted,
            "raw_feasible": True,
            "candidate_set_available": True,
            "replacement": False,
            "reason": "raw_feasible",
        }
    # In the highway-merge task, ACCVP is allowed to rescue an infeasible raw
    # action only by choosing a merge-intent action.  Replacing a left/merge
    # request with keep/right may look safer over the short horizon but can
    # consume the remaining taper window and recreate the Full-Ranking failure
    # mode that ACCVP is meant to avoid.
    replacement_candidates = [row for row in accepted if int(_candidate_action(row).lateral_cmd) > 0]
    if not accepted:
        return {
            "selected": None,
            "accepted": accepted,
            "raw_feasible": False,
            "candidate_set_available": False,
            "replacement": False,
            "reason": "no_feasible_action",
        }
    if not replacement_candidates:
        return {
            "selected": None,
            "accepted": accepted,
            "raw_feasible": False,
            "candidate_set_available": False,
            "replacement": False,
            "reason": "no_merge_intent_feasible_action",
        }
    raw_action = decode_action(int(raw_action_id))
    selected = min(
        replacement_candidates,
        key=lambda row: (
            -float(row["pL_merge_before_taper"]),
            float(row["pU_safety_violation"]),
            float(row.get("target_lane_entry_time_s", float("inf"))),
            abs(float(_candidate_action(row).accel_cmd)),
            action_distance(_candidate_action(row), raw_action),
            int(row["action_id"]),
        ),
    )
    return {
        "selected": selected,
        "accepted": accepted,
        "raw_feasible": False,
        "candidate_set_available": True,
        "replacement": True,
        "reason": "raw_infeasible_viable_candidate",
    }


def _lite_task_pass(row: dict[str, Any], thresholds: dict[str, float]) -> bool:
    return (
        lite_secondary_safety_pass(row, thresholds)
        and float(row.get("p_merge_before_taper", 0.0)) >= float(thresholds["min_p_merge_before_taper"])
        and float(row.get("target_lane_entry_time_s", float("inf"))) <= float(thresholds["max_target_entry_time_s"])
        and float(row.get("ensemble_disagreement", 0.0)) <= float(thresholds["max_ensemble_disagreement"])
    )


def gate_viability_lite_candidates(candidates: list[dict[str, Any]], thresholds: dict[str, float]) -> list[dict[str, Any]]:
    """Task-only gate: Risk/Shield owns safety; ACCVP owns merge viability."""

    accepted: list[dict[str, Any]] = []
    for row in candidates:
        task_pass = _lite_task_pass(row, thresholds)
        row["accvp_lite_secondary_pass"] = bool(lite_secondary_safety_pass(row, thresholds))
        row["accvp_lite_secondary_safety_profile"] = str(thresholds.get("secondary_safety_profile", "strict"))
        row["accvp_lite_task_pass"] = bool(task_pass)
        row["accvp_lite_gate_pass"] = bool(task_pass)
        if task_pass:
            accepted.append(row)
    return accepted


def select_viability_lite_action(
    candidates: list[dict[str, Any]],
    *,
    raw_action_id: int,
    thresholds: dict[str, float],
) -> dict[str, Any]:
    """Select a merge-intent rescue using only Risk safety and ACCVP task viability.

    Unlike ``select_viability_action``, this function intentionally ignores
    ACCVP pU safety heads.  It is the selector for ACV-Shield-lite where the
    Risk Module / Safety Shield remains the only safety authority.
    """

    accepted = gate_viability_lite_candidates(candidates, thresholds)
    raw = next((row for row in candidates if int(row["action_id"]) == int(raw_action_id)), None)
    raw_action_legal = bool(candidates[0].get("raw_action_legal", True)) if candidates else False
    if raw is None or not raw_action_legal:
        return {
            "selected": None,
            "accepted": accepted,
            "raw_feasible": False,
            "raw_task_feasible": False,
            "candidate_set_available": False,
            "replacement": False,
            "reason": "raw_action_illegal_or_missing",
            "best_left": None,
            "p_merge_improvement": 0.0,
        }
    raw_task_feasible = bool(raw is not None and _lite_task_pass(raw, thresholds))
    replacement_candidates = [row for row in accepted if int(row["action_id"]) in LEFT_ACTION_IDS]
    raw_p_merge = float(raw.get("p_merge_before_taper", 0.0)) if raw is not None else 0.0
    if raw_task_feasible:
        return {
            "selected": raw,
            "accepted": accepted,
            "raw_feasible": True,
            "raw_task_feasible": True,
            "candidate_set_available": True,
            "replacement": False,
            "reason": "raw_task_feasible",
            "best_left": None,
            "p_merge_improvement": 0.0,
        }
    if not replacement_candidates:
        return {
            "selected": None,
            "accepted": accepted,
            "raw_feasible": False,
            "raw_task_feasible": False,
            "candidate_set_available": False,
            "replacement": False,
            "reason": "no_risk_safe_task_viable_left_action",
            "best_left": None,
            "p_merge_improvement": 0.0,
        }
    raw_action = decode_action(int(raw_action_id))
    best_left = min(
        replacement_candidates,
        key=lambda row: (
            -float(row["p_merge_before_taper"]),
            float(row.get("target_lane_entry_time_s", float("inf"))),
            _secondary_risk_score(row),
            float(row.get("ensemble_disagreement", 0.0)),
            abs(float(_candidate_action(row).accel_cmd)),
            action_distance(_candidate_action(row), raw_action),
            int(row["action_id"]),
        ),
    )
    improvement = float(best_left["p_merge_before_taper"]) - raw_p_merge
    if improvement < float(thresholds["min_improvement_over_raw"]):
        return {
            "selected": raw,
            "accepted": accepted,
            "raw_feasible": False,
            "raw_task_feasible": False,
            "candidate_set_available": False,
            "replacement": False,
            "reason": "best_left_below_improvement_margin",
            "best_left": best_left,
            "p_merge_improvement": improvement,
        }
    return {
        "selected": best_left,
        "accepted": accepted,
        "raw_feasible": False,
        "raw_task_feasible": False,
        "candidate_set_available": True,
        "replacement": True,
        "reason": "raw_task_infeasible_lite_viable_left",
        "best_left": best_left,
        "p_merge_improvement": improvement,
    }
