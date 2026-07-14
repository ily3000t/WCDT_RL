"""Leakage-free operating-point threshold selection."""

from __future__ import annotations

from collections import defaultdict
from itertools import product
from typing import Any

import numpy as np

from safe_rl.accvp.training.calibration import CalibrationBundle
from safe_rl.accvp.training.availability import OperatingPointAvailabilityError, model_gate_failure_diagnostics
from safe_rl.accvp.data.dataset import ACCVPBranchDataset, collate_numpy
from safe_rl.accvp.planning.selection import LEFT_ACTION_IDS, select_viability_action
from safe_rl.accvp.training.trainer import _model_output, _tensor_batch


AVAILABILITY_DENOMINATOR_VERSION = "risk_eligible_raw_or_merge_left_v1"


def _risk_eligible_for_viability_selection(candidates: list[dict[str, Any]]) -> bool:
    """Return whether Risk permits raw retention or a merge-left rescue."""

    if not candidates:
        return False

    def risk_pass(row: dict[str, Any]) -> bool:
        return bool(row.get("candidate_legal", True)) and bool(
            row.get("secondary_safety_pass", False)
        )

    raw_action_id = int(candidates[0]["raw_action_id"])
    raw = next(
        (row for row in candidates if int(row["action_id"]) == raw_action_id),
        None,
    )
    return bool(raw is not None and risk_pass(raw)) or any(
        int(row["action_id"]) in LEFT_ACTION_IDS and risk_pass(row)
        for row in candidates
    )


def tune_operating_point(models: list[Any], dataset: ACCVPBranchDataset, calibration: CalibrationBundle, torch: Any, tuning: Any) -> dict[str, Any]:
    """Select gates on the dedicated operating-point split at decision level."""

    if not len(dataset):
        raise ValueError("operating-point split is empty")
    rows: list[dict[str, Any]] = []
    for model in models:
        model.eval()
    with torch.no_grad():
        for index in range(len(dataset)):
            batch_np = collate_numpy([dataset[index]])
            if not bool(batch_np["viability_eligible"][0]):
                continue
            batch = _tensor_batch(batch_np, torch)
            event_members = [torch.sigmoid(_model_output(model, batch)["event_logits"]).cpu().numpy()[0] for model in models]
            events = np.stack(event_members, axis=0)
            raw = {
                "p_proxy_collision": [float(events[:, 0].max())],
                "p_safety_violation": [float(events[:, 1].max())],
                "p_merge_before_taper": [float(events[:, 3].min())],
            }
            bounds = calibration.score(raw)
            manifest = dataset.rows[index]
            root = dataset.roots[str(manifest["root_id"])]
            root_id = str(manifest["root_id"])
            fingerprint = dataset.observation_fingerprint_by_root.get(root_id, "")
            decision_unit_id = (
                f"{fingerprint}|raw:{int(root['raw_action_id'])}"
                if fingerprint
                else root_id
            )
            secondary = dict(manifest.get("secondary_risk", {}))
            rows.append(
                {
                    "root_id": root_id,
                    "decision_unit_id": decision_unit_id,
                    "root_observation_fingerprint": fingerprint,
                    "pU_proxy_collision": float(bounds["pU_proxy_collision"][0]),
                    "pU_safety_violation": float(bounds["pU_safety_violation"][0]),
                    "pL_merge_before_taper": float(bounds["pL_merge_before_taper"][0]),
                    "proxy_collision": float(batch_np["event_targets"][0, 0]),
                    "safety_violation": float(batch_np["event_targets"][0, 1]),
                    "merge_before_taper": float(batch_np["event_targets"][0, 3]),
                    "merge_observed": bool(batch_np["event_mask"][0, 3]),
                    "action_id": int(manifest["action_id"]),
                    "raw_action_id": int(root["raw_action_id"]),
                    "candidate_legal": bool(secondary.get("candidate_legal", True)),
                    "secondary_safety_pass": bool(manifest.get("secondary_safety_pass", secondary.get("secondary_safety_pass", False))),
                }
            )
    if not rows:
        raise ValueError("operating-point split has no observed deadline viability rows")
    raw_row_count = len(rows)
    grouped_rows: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped_rows[(str(row["decision_unit_id"]), int(row["action_id"]))].append(row)
    collapsed_rows: list[dict[str, Any]] = []
    for (decision_unit_id, action_id), members in sorted(grouped_rows.items()):
        raw_actions = {int(row["raw_action_id"]) for row in members}
        if len(raw_actions) != 1:
            raise ValueError("duplicate operating-point decision unit has inconsistent raw actions")
        collapsed_rows.append(
            {
                **members[0],
                "root_id": decision_unit_id,
                "decision_unit_id": decision_unit_id,
                "action_id": action_id,
                "pU_proxy_collision": max(float(row["pU_proxy_collision"]) for row in members),
                "pU_safety_violation": max(float(row["pU_safety_violation"]) for row in members),
                "pL_merge_before_taper": min(float(row["pL_merge_before_taper"]) for row in members),
                "proxy_collision": float(np.mean([float(row["proxy_collision"]) for row in members])),
                "safety_violation": float(np.mean([float(row["safety_violation"]) for row in members])),
                "merge_before_taper": float(np.mean([float(row["merge_before_taper"]) for row in members])),
                "merge_observed": bool(all(bool(row["merge_observed"]) for row in members)),
                "candidate_legal": bool(all(bool(row["candidate_legal"]) for row in members)),
                "secondary_safety_pass": bool(all(bool(row["secondary_safety_pass"]) for row in members)),
                "replicate_count": len(members),
            }
        )
    rows = collapsed_rows
    required = float(tuning.required_availability)
    denominator_version = str(
        tuning.get("availability_denominator", AVAILABILITY_DENOMINATOR_VERSION)
    )
    if denominator_version != AVAILABILITY_DENOMINATOR_VERSION:
        raise ValueError(
            "unsupported ACCVP tuning availability denominator: "
            f"{denominator_version!r}"
        )
    by_root: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_root[row["root_id"]].append(row)
    risk_eligible_decision_count = sum(
        _risk_eligible_for_viability_selection(root_rows)
        for root_rows in by_root.values()
    )
    decision_count = len(by_root)
    candidates: list[dict[str, Any]] = []
    evaluated_points: list[dict[str, Any]] = []
    for collision_bound, safety_bound, viability_bound in product(
        tuning.proxy_collision_upper_bounds,
        tuning.safety_violation_upper_bounds,
        tuning.merge_viability_lower_bounds,
    ):
        thresholds = {
            "proxy_collision_upper_bound": float(collision_bound),
            "safety_violation_upper_bound": float(safety_bound),
            "merge_viability_lower_bound": float(viability_bound),
        }
        decisions = [
            select_viability_action(candidates, raw_action_id=int(candidates[0]["raw_action_id"]), thresholds=thresholds)
            for candidates in by_root.values()
        ]
        selected = [decision["selected"] for decision in decisions if decision["selected"] is not None]
        availability = float(len(selected) / max(1, risk_eligible_decision_count))
        unconditional_availability = float(len(selected) / max(1, decision_count))
        risk_eligible_fraction = float(
            risk_eligible_decision_count / max(1, decision_count)
        )
        observed = [row for row in selected if row["merge_observed"]]
        point = {
            "proxy_collision_upper_bound": float(collision_bound),
            "safety_violation_upper_bound": float(safety_bound),
            "merge_viability_lower_bound": float(viability_bound),
            "candidate_set_availability": availability,
            "model_conditional_availability": availability,
            "unconditional_candidate_set_availability": unconditional_availability,
            "risk_eligible_decision_fraction": risk_eligible_fraction,
            "availability_denominator_version": denominator_version,
            "risk_eligible_decision_count": int(risk_eligible_decision_count),
            "risk_ineligible_decision_count": int(
                decision_count - risk_eligible_decision_count
            ),
            "selected_count": int(len(selected)),
            "decision_count": int(decision_count),
            "selected_safety_ucb": float(np.mean([row["pU_safety_violation"] for row in selected])) if selected else float("inf"),
            "selected_viability_lcb": float(np.mean([row["pL_merge_before_taper"] for row in selected])) if selected else float("-inf"),
            "selected_observed_safety_rate": float(np.mean([row["safety_violation"] for row in selected])) if selected else float("nan"),
            "selected_observed_viability_rate": float(np.mean([row["merge_before_taper"] for row in observed])) if observed else float("nan"),
        }
        evaluated_points.append(point)
        if availability >= required:
            candidates.append(point)
    if not candidates:
        best = max(
            evaluated_points,
            key=lambda row: (
                row["candidate_set_availability"],
                -row["selected_safety_ucb"],
                row["selected_viability_lcb"],
            ),
        )
        best_thresholds = {
            "proxy_collision_upper_bound": float(best["proxy_collision_upper_bound"]),
            "safety_violation_upper_bound": float(best["safety_violation_upper_bound"]),
            "merge_viability_lower_bound": float(best["merge_viability_lower_bound"]),
        }
        diagnostics = model_gate_failure_diagnostics(
            rows,
            best_thresholds,
            required_availability=required,
            split="operating_point",
            evaluated_points=evaluated_points,
        )
        raise OperatingPointAvailabilityError(
            "no operating point satisfies required ACCVP risk-conditional model availability; "
            f"required={required:.6f} best_conditional_availability={best['candidate_set_availability']:.6f} "
            f"unconditional_coverage={best['unconditional_candidate_set_availability']:.6f} "
            f"selected={best['selected_count']}/{best['decision_count']} "
            f"best_thresholds={{'proxy_collision_upper_bound': {best['proxy_collision_upper_bound']}, "
            f"'safety_violation_upper_bound': {best['safety_violation_upper_bound']}, "
            f"'merge_viability_lower_bound': {best['merge_viability_lower_bound']}}}",
            diagnostics=diagnostics,
        )
    selected = min(
        candidates,
        key=lambda row: (
            row["selected_safety_ucb"],
            -row["selected_viability_lcb"],
            -row["candidate_set_availability"],
        ),
    )
    return {
        "split": "operating_point",
        "required_availability": required,
        "availability_denominator_version": denominator_version,
        "risk_eligible_decision_count": int(risk_eligible_decision_count),
        "risk_ineligible_decision_count": int(
            decision_count - risk_eligible_decision_count
        ),
        "risk_eligible_decision_fraction": float(
            risk_eligible_decision_count / max(1, decision_count)
        ),
        "selected": selected,
        "evaluated_points": candidates,
        "decision_weighting_version": "fingerprint_raw_action_risk_eligible_total_weight_one_v2",
        "raw_candidate_row_count": raw_row_count,
        "effective_candidate_row_count": len(rows),
        "effective_decision_count": int(decision_count),
    }
