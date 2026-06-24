from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from safe_rl.accvp.calibration import CalibrationBundle, brier_score, expected_calibration_error, selected_action_metrics
from safe_rl.accvp.dataset import ACCVPBranchDataset, collate_numpy
from safe_rl.accvp.selection import select_viability_action


def _tensor_batch(batch: dict[str, np.ndarray], torch: Any) -> dict[str, Any]:
    integer = {"history_lane_ids", "history_edge_role_ids", "role_ids", "lane_ids", "edge_role_ids", "candidate_action_ids"}
    return {key: torch.as_tensor(value, dtype=torch.long if key in integer else torch.float32) for key, value in batch.items()}


def _model_output(model: Any, batch: dict[str, Any]) -> dict[str, Any]:
    return model(
        batch["history_features"],
        batch["history_valid_mask"],
        batch["history_lane_ids"],
        batch["history_edge_role_ids"],
        batch["role_ids"],
        batch["lane_ids"],
        batch["edge_role_ids"],
        batch["actor_mask"],
        batch["candidate_plan"],
        batch["candidate_action_ids"],
    )


def _candidate_records(models: list[Any], dataset: ACCVPBranchDataset, calibration: CalibrationBundle, torch: Any) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for model in models:
        model.eval()
    with torch.no_grad():
        for index, row in enumerate(dataset.rows):
            batch_np = collate_numpy([dataset[index]])
            if not bool(batch_np["viability_eligible"][0]):
                continue
            batch = _tensor_batch(batch_np, torch)
            outputs = [_model_output(model, batch) for model in models]
            events = np.stack([torch.sigmoid(output["event_logits"]).cpu().numpy()[0] for output in outputs], axis=0)
            geometry = np.stack([output["geometry"].cpu().numpy()[0] for output in outputs], axis=0)
            raw = {
                "p_proxy_collision": [float(events[:, 0].max())],
                "p_safety_violation": [float(events[:, 1].max())],
                "p_merge_before_taper": [float(events[:, 3].min())],
            }
            bounds = calibration.score(raw)
            root = dataset.roots[str(row["root_id"])]
            records.append(
                {
                    "root_id": str(row["root_id"]),
                    "action_id": int(row["action_id"]),
                    "raw_action_id": root.get("raw_action_id"),
                    "raw_action_legal": bool(root.get("raw_action_legal", False)),
                    "p_proxy_collision": raw["p_proxy_collision"][0],
                    "p_safety_violation": raw["p_safety_violation"][0],
                    "p_merge_before_taper": raw["p_merge_before_taper"][0],
                    "pU_proxy_collision": float(bounds["pU_proxy_collision"][0]),
                    "pU_safety_violation": float(bounds["pU_safety_violation"][0]),
                    "pL_merge_before_taper": float(bounds["pL_merge_before_taper"][0]),
                    "target_lane_entry_time_s": float(max(0.0, np.median(geometry[:, 4]))),
                    "proxy_collision": float(batch_np["event_targets"][0, 0]),
                    "safety_violation": float(batch_np["event_targets"][0, 1]),
                    "merge_before_taper": float(batch_np["event_targets"][0, 3]),
                    "merge_observed": bool(batch_np["event_mask"][0, 3]),
                    "candidate_legal": bool(dict(row.get("secondary_risk", {})).get("candidate_legal", True)),
                    "secondary_safety_pass": bool(
                        row.get("secondary_safety_pass", dict(row.get("secondary_risk", {})).get("secondary_safety_pass", False))
                    ),
                }
            )
    return records


def final_test_diagnostics(
    models: list[Any],
    dataset: ACCVPBranchDataset,
    calibration: CalibrationBundle,
    operating_point: dict[str, Any],
    torch: Any,
) -> dict[str, Any]:
    """Frozen final-test diagnostics, including the post-selection policy."""

    if not len(dataset):
        raise ValueError("ACCVP final test split is empty")
    thresholds = dict(operating_point["selected"])
    records = _candidate_records(models, dataset, calibration, torch)
    if not records:
        raise ValueError("ACCVP final test split has no observed deadline viability rows")
    by_root: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_root[record["root_id"]].append(record)
    selected: list[dict[str, Any]] = []
    availability = 0
    for root_id, candidates in by_root.items():
        decision = select_viability_action(
            candidates,
            raw_action_id=int(candidates[0]["raw_action_id"]),
            thresholds=thresholds,
        )
        chosen = decision["selected"]
        if chosen is None:
            continue
        availability += 1
        chosen = dict(chosen)
        chosen["selected"] = True
        chosen["candidate_set_available"] = bool(decision["candidate_set_available"])
        selected.append(chosen)
    candidate_proxy = np.asarray([row["p_proxy_collision"] for row in records])
    candidate_proxy_y = np.asarray([row["proxy_collision"] for row in records])
    candidate_safety = np.asarray([row["p_safety_violation"] for row in records])
    candidate_safety_y = np.asarray([row["safety_violation"] for row in records])
    candidate_viability = np.asarray([row["p_merge_before_taper"] for row in records if row["merge_observed"]])
    candidate_viability_y = np.asarray([row["merge_before_taper"] for row in records if row["merge_observed"]])
    return {
        "split": "test",
        "sample_count": len(records),
        "decision_count": len(by_root),
        "candidate_set_availability": float(availability / max(1, len(by_root))),
        "candidate_level": {
            "proxy_collision_brier": brier_score(candidate_proxy, candidate_proxy_y),
            "proxy_collision_ece": expected_calibration_error(candidate_proxy, candidate_proxy_y),
            "safety_violation_brier": brier_score(candidate_safety, candidate_safety_y),
            "safety_violation_ece": expected_calibration_error(candidate_safety, candidate_safety_y),
            "viability_brier": brier_score(candidate_viability, candidate_viability_y),
            "viability_ece": expected_calibration_error(candidate_viability, candidate_viability_y),
        },
        "post_selection": selected_action_metrics(selected, total_decision_count=len(by_root)),
        "operating_point": thresholds,
    }
