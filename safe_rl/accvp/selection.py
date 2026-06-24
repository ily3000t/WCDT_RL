"""One deterministic ACCVP selection rule shared by runtime and offline reports."""

from __future__ import annotations

from typing import Any

from safe_rl.sim.action_space import action_distance, decode_action


def _candidate_action(row: dict[str, Any]):
    return row.get("action") or decode_action(int(row["action_id"]))


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
    if not accepted:
        return {
            "selected": None,
            "accepted": accepted,
            "raw_feasible": False,
            "candidate_set_available": False,
            "replacement": False,
            "reason": "no_feasible_action",
        }
    raw_action = decode_action(int(raw_action_id))
    selected = min(
        accepted,
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
