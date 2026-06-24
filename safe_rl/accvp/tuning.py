from __future__ import annotations

from collections import defaultdict
from itertools import product
from typing import Any

import numpy as np

from safe_rl.accvp.calibration import CalibrationBundle
from safe_rl.accvp.dataset import ACCVPBranchDataset, collate_numpy
from safe_rl.accvp.selection import select_viability_action
from safe_rl.accvp.train import _model_output, _tensor_batch


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
            secondary = dict(manifest.get("secondary_risk", {}))
            rows.append(
                {
                    "root_id": str(manifest["root_id"]),
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
    required = float(tuning.required_availability)
    candidates: list[dict[str, Any]] = []
    for collision_bound, safety_bound, viability_bound in product(
        tuning.proxy_collision_upper_bounds,
        tuning.safety_violation_upper_bounds,
        tuning.merge_viability_lower_bounds,
    ):
        by_root: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_root[row["root_id"]].append(row)
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
        availability = float(len(selected) / max(1, len(by_root)))
        if availability < required:
            continue
        observed = [row for row in selected if row["merge_observed"]]
        candidates.append(
            {
                "proxy_collision_upper_bound": float(collision_bound),
                "safety_violation_upper_bound": float(safety_bound),
                "merge_viability_lower_bound": float(viability_bound),
                "candidate_set_availability": availability,
                "selected_safety_ucb": float(np.mean([row["pU_safety_violation"] for row in selected])),
                "selected_viability_lcb": float(np.mean([row["pL_merge_before_taper"] for row in selected])),
                "selected_observed_safety_rate": float(np.mean([row["safety_violation"] for row in selected])),
                "selected_observed_viability_rate": float(np.mean([row["merge_before_taper"] for row in observed])) if observed else float("nan"),
            }
        )
    if not candidates:
        raise RuntimeError("no operating point satisfies required ACCVP candidate-set availability")
    selected = min(
        candidates,
        key=lambda row: (
            row["selected_safety_ucb"],
            -row["selected_viability_lcb"],
            -row["candidate_set_availability"],
        ),
    )
    return {"split": "operating_point", "required_availability": required, "selected": selected, "evaluated_points": candidates}
