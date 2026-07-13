from __future__ import annotations

import json
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from safe_rl.accvp.calibration import CalibrationBundle
from safe_rl.accvp.dataset import ACCVPBranchDataset, collate_numpy
from safe_rl.accvp.model import ACCVPPredictor
from safe_rl.accvp.schema import file_sha256, read_json, stable_hash
from safe_rl.accvp.selection import select_viability_action


LEFT_ACTION_IDS = {6, 7, 8}

DEFAULT_STEP1_5_THRESHOLDS = {
    "viability_pair_contrast": 0.90,
    "raw_fail_left_success_contrast": 0.90,
    "safety_pair_contrast": 0.60,
}


def _tensor_batch(batch: dict[str, np.ndarray], torch: Any) -> dict[str, Any]:
    integer = {"history_lane_ids", "history_edge_role_ids", "role_ids", "lane_ids", "edge_role_ids", "candidate_action_ids"}
    return {
        key: torch.as_tensor(value, dtype=torch.long if key in integer else torch.float32)
        for key, value in batch.items()
    }


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


def _activation_window(row: dict[str, Any]) -> bool:
    return str(row.get("activation_bin", row.get("deadline_bin", ""))) in {"activation_window", "deadline"}


def _candidate_legal(row: dict[str, Any]) -> bool:
    secondary = dict(row.get("secondary_risk", {}) or {})
    return bool(row.get("candidate_legal", secondary.get("candidate_legal", True)))


def _secondary_safety_pass(row: dict[str, Any]) -> bool:
    secondary = dict(row.get("secondary_risk", {}) or {})
    return bool(row.get("secondary_safety_pass", secondary.get("secondary_safety_pass", False)))


def _secondary_risk_score(row: dict[str, Any]) -> float:
    secondary = dict(row.get("secondary_risk", {}) or {})
    return float(row.get("secondary_risk_score", secondary.get("risk_score", 0.0)))


def _secondary_risk_uncertainty(row: dict[str, Any]) -> float:
    secondary = dict(row.get("secondary_risk", {}) or {})
    return float(row.get("secondary_risk_uncertainty", secondary.get("risk_uncertainty", 0.0)))


def _secondary_veto_reason(row: dict[str, Any]) -> str:
    secondary = dict(row.get("secondary_risk", {}) or {})
    return str(row.get("secondary_veto_reason", secondary.get("veto_reason", "")))


def _finite(values: list[float]) -> list[float]:
    return [float(value) for value in values if np.isfinite(float(value))]


def _mean(values: list[float]) -> float | None:
    clean = _finite(values)
    return float(np.mean(clean)) if clean else None


def _rate(values: list[bool]) -> float | None:
    return float(np.mean([bool(value) for value in values])) if values else None


def _summary(values: list[float]) -> dict[str, float | int | None]:
    clean = _finite(values)
    if not clean:
        return {"count": 0, "mean": None, "p10": None, "p50": None, "p90": None}
    arr = np.asarray(clean, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
    }


def _by_root(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[str(row["root_id"])].append(row)
    return grouped


def _pair_contrast(
    grouped: dict[str, list[dict[str, Any]]],
    *,
    label: str,
    score: str,
    positive_higher: bool,
) -> dict[str, float | int | None]:
    pair_count = 0
    correct = 0.0
    tie_count = 0
    for rows in grouped.values():
        comparable = rows
        if label == "merge_before_taper":
            comparable = [row for row in rows if bool(row.get("merge_observed", False))]
        positives = [row for row in comparable if float(row.get(label, 0.0)) >= 0.5]
        negatives = [row for row in comparable if float(row.get(label, 0.0)) < 0.5]
        for positive in positives:
            for negative in negatives:
                pair_count += 1
                delta = float(positive.get(score, 0.0)) - float(negative.get(score, 0.0))
                if abs(delta) <= 1.0e-12:
                    tie_count += 1
                    correct += 0.5
                elif (delta > 0.0) == bool(positive_higher):
                    correct += 1.0
    return {
        "pair_count": int(pair_count),
        "tie_count": int(tie_count),
        "accuracy": None if pair_count == 0 else float(correct / pair_count),
    }


def _action_stats(records: list[dict[str, Any]], thresholds: dict[str, float]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    action_ids = sorted({int(row["action_id"]) for row in records}.union(LEFT_ACTION_IDS))
    for action_id in action_ids:
        rows = [row for row in records if int(row["action_id"]) == action_id]
        result[str(action_id)] = {
            "count": int(len(rows)),
            "is_left_action": bool(action_id in LEFT_ACTION_IDS),
            "candidate_legal_rate": _rate([bool(row.get("candidate_legal", False)) for row in rows]),
            "secondary_safety_pass_rate": _rate([bool(row.get("secondary_safety_pass", False)) for row in rows]),
            "observed_success_rate": _rate(
                [
                    float(row.get("merge_before_taper", 0.0)) >= 0.5
                    for row in rows
                    if bool(row.get("merge_observed", False))
                ]
            ),
            "taper_miss_rate": _rate([float(row.get("taper_miss", 0.0)) >= 0.5 for row in rows]),
            "safety_event_rate": _rate(
                [
                    bool(float(row.get("proxy_collision", 0.0)) >= 0.5 or float(row.get("safety_violation", 0.0)) >= 0.5)
                    for row in rows
                ]
            ),
            "mean_p_proxy_collision": _mean([float(row.get("p_proxy_collision", 0.0)) for row in rows]),
            "mean_p_safety_violation": _mean([float(row.get("p_safety_violation", 0.0)) for row in rows]),
            "mean_p_taper_miss": _mean([float(row.get("p_taper_miss", 0.0)) for row in rows]),
            "mean_p_merge_before_taper": _mean([float(row.get("p_merge_before_taper", 0.0)) for row in rows]),
            "mean_pU_proxy_collision": _mean([float(row.get("pU_proxy_collision", 0.0)) for row in rows]),
            "mean_pU_safety_violation": _mean([float(row.get("pU_safety_violation", 0.0)) for row in rows]),
            "mean_pL_merge_before_taper": _mean([float(row.get("pL_merge_before_taper", 0.0)) for row in rows]),
            "proxy_gate_pass_rate": _rate(
                [float(row.get("pU_proxy_collision", 1.0)) <= float(thresholds["proxy_collision_upper_bound"]) for row in rows]
            ),
            "safety_gate_pass_rate": _rate(
                [float(row.get("pU_safety_violation", 1.0)) <= float(thresholds["safety_violation_upper_bound"]) for row in rows]
            ),
            "viability_gate_pass_rate": _rate(
                [float(row.get("pL_merge_before_taper", 0.0)) >= float(thresholds["merge_viability_lower_bound"]) for row in rows]
            ),
        }
    return result


def _gate_selection_summary(grouped: dict[str, list[dict[str, Any]]], thresholds: dict[str, float]) -> dict[str, Any]:
    selected_actions: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    candidate_available: list[bool] = []
    raw_feasible: list[bool] = []
    for candidates in grouped.values():
        raw_action_id = int(candidates[0].get("raw_action_id", -1))
        decision = select_viability_action(deepcopy(candidates), raw_action_id=raw_action_id, thresholds=thresholds)
        selected = decision.get("selected")
        if selected is not None:
            selected_actions[str(int(selected["action_id"]))] += 1
        reasons[str(decision.get("reason", ""))] += 1
        candidate_available.append(bool(decision.get("candidate_set_available", False)))
        raw_feasible.append(bool(decision.get("raw_feasible", False)))
    selected = [int(action_id) for action_id, count in selected_actions.items() for _ in range(count)]
    return {
        "decision_count": int(len(grouped)),
        "candidate_set_available_rate": _rate(candidate_available),
        "raw_feasible_rate": _rate(raw_feasible),
        "selected_action_counts": dict(sorted(selected_actions.items(), key=lambda item: int(item[0]))),
        "selected_merge_intent_rate": _rate([action_id in LEFT_ACTION_IDS for action_id in selected]),
        "reason_counts": dict(sorted(reasons.items())),
    }


def _raw_probability_recommendation_summary(grouped: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    diff: list[bool] = []
    left: list[bool] = []
    for rows in grouped.values():
        if not rows:
            continue
        raw_action_id = int(rows[0].get("raw_action_id", -1))
        best = max(rows, key=lambda row: (float(row.get("p_merge_before_taper", 0.0)), -int(row["action_id"])))
        action_id = int(best["action_id"])
        counts[str(action_id)] += 1
        diff.append(action_id != raw_action_id)
        left.append(action_id in LEFT_ACTION_IDS)
    return {
        "decision_count": int(sum(counts.values())),
        "best_p_merge_action_counts": dict(sorted(counts.items(), key=lambda item: int(item[0]))),
        "best_p_merge_differs_from_raw_rate": _rate(diff),
        "best_p_merge_is_left_action_rate": _rate(left),
    }


def _raw_action_summary(grouped: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    raw_rows = []
    for rows in grouped.values():
        raw_action_id = int(rows[0].get("raw_action_id", -1))
        raw = next((row for row in rows if int(row["action_id"]) == raw_action_id), None)
        if raw is not None:
            raw_rows.append(raw)
    return {
        "count": int(len(raw_rows)),
        "p_proxy_collision": _summary([float(row["p_proxy_collision"]) for row in raw_rows]),
        "p_safety_violation": _summary([float(row["p_safety_violation"]) for row in raw_rows]),
        "p_merge_before_taper": _summary([float(row["p_merge_before_taper"]) for row in raw_rows]),
        "pU_proxy_collision": _summary([float(row["pU_proxy_collision"]) for row in raw_rows]),
        "pU_safety_violation": _summary([float(row["pU_safety_violation"]) for row in raw_rows]),
        "pL_merge_before_taper": _summary([float(row["pL_merge_before_taper"]) for row in raw_rows]),
    }


def _raw_fail_left_success_summary(grouped: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for root_id, candidates in grouped.items():
        raw_action_id = int(candidates[0].get("raw_action_id", -1))
        raw = next((row for row in candidates if int(row["action_id"]) == raw_action_id), None)
        if raw is None or not bool(raw.get("merge_observed", False)) or float(raw.get("merge_before_taper", 0.0)) >= 0.5:
            continue
        left_success = [
            row
            for row in candidates
            if int(row["action_id"]) in LEFT_ACTION_IDS
            and bool(row.get("merge_observed", False))
            and float(row.get("merge_before_taper", 0.0)) >= 0.5
        ]
        if not left_success:
            continue
        safe_left_success = [
            row
            for row in left_success
            if float(row.get("proxy_collision", 0.0)) < 0.5 and float(row.get("safety_violation", 0.0)) < 0.5
        ]
        best_left = max(left_success, key=lambda row: (float(row["p_merge_before_taper"]), -int(row["action_id"])))
        pmerge_gt_raw = float(best_left["p_merge_before_taper"]) > float(raw["p_merge_before_taper"])
        psafety_lt_raw = float(best_left["p_safety_violation"]) < float(raw["p_safety_violation"])
        rows.append(
            {
                "root_id": root_id,
                "raw_action_id": raw_action_id,
                "best_left_action_id": int(best_left["action_id"]),
                "has_safe_left_success": bool(safe_left_success),
                "best_left_pmerge_gt_raw": bool(pmerge_gt_raw),
                "best_left_psafety_lt_raw": bool(psafety_lt_raw),
                "best_left_both": bool(pmerge_gt_raw and psafety_lt_raw),
            }
        )
    return {
        "raw_fail_left_success_root_count": int(len(rows)),
        "raw_fail_safe_left_success_root_count": int(sum(1 for row in rows if bool(row["has_safe_left_success"]))),
        "best_left_pmerge_gt_raw_rate": _rate([bool(row["best_left_pmerge_gt_raw"]) for row in rows]),
        "best_left_psafety_lt_raw_rate": _rate([bool(row["best_left_psafety_lt_raw"]) for row in rows]),
        "best_left_both_rate": _rate([bool(row["best_left_both"]) for row in rows]),
        "examples": rows[:20],
    }


def _verdict(summary: dict[str, Any], thresholds: dict[str, float]) -> dict[str, Any]:
    repair = summary["raw_fail_left_success"]
    viability = summary["pairwise_contrast"]["viability"]
    safety = summary["pairwise_contrast"]["safety"]
    gate = summary["calibrated_gate_selection"]
    enough = bool(
        int(viability["pair_count"]) > 0
        and int(repair["raw_fail_left_success_root_count"]) > 0
        and repair["best_left_pmerge_gt_raw_rate"] is not None
    )
    viability_pass = bool(
        enough
        and float(viability["accuracy"]) >= float(thresholds["viability_pair_contrast"])
        and float(repair["best_left_pmerge_gt_raw_rate"]) >= float(thresholds["raw_fail_left_success_contrast"])
    )
    safety_pass = bool(
        int(safety["pair_count"]) > 0
        and safety["accuracy"] is not None
        and float(safety["accuracy"]) >= float(thresholds["safety_pair_contrast"])
    )
    gate_pass = bool((gate.get("candidate_set_available_rate") or 0.0) > 0.0)
    if not enough:
        state = "insufficient"
    elif viability_pass and safety_pass and gate_pass:
        state = "diagnostic_pass"
    elif viability_pass:
        state = "viability_only_pass"
    else:
        state = "fail"
    return {
        "step1_5_state": state,
        "viability_signal_pass": viability_pass,
        "safety_signal_pass": safety_pass,
        "gate_availability_pass": gate_pass,
        "thresholds": dict(thresholds),
        "deployable_claim": False,
    }


def candidate_table_summary(
    records: list[dict[str, Any]],
    *,
    split: str,
    thresholds: dict[str, float],
    diagnostic_thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    diagnostic_thresholds = {**DEFAULT_STEP1_5_THRESHOLDS, **dict(diagnostic_thresholds or {})}
    grouped = _by_root(records)
    safety_records = []
    for row in records:
        item = dict(row)
        item["safety_event"] = float(
            float(item.get("proxy_collision", 0.0)) >= 0.5 or float(item.get("safety_violation", 0.0)) >= 0.5
        )
        safety_records.append(item)
    safety_grouped = _by_root(safety_records)
    successes = [row for row in records if bool(row.get("merge_observed", False)) and float(row.get("merge_before_taper", 0.0)) >= 0.5]
    failures = [row for row in records if bool(row.get("merge_observed", False)) and float(row.get("merge_before_taper", 0.0)) < 0.5]
    summary = {
        "split": str(split),
        "sample_count": int(len(records)),
        "decision_count": int(len(grouped)),
        "observed_viability_rows": int(sum(1 for row in records if bool(row.get("merge_observed", False)))),
        "observed_success_count": int(len(successes)),
        "observed_failure_count": int(len(failures)),
        "mean_p_merge_success": _mean([float(row["p_merge_before_taper"]) for row in successes]),
        "mean_p_merge_failure": _mean([float(row["p_merge_before_taper"]) for row in failures]),
        "mean_p_safety_safe": _mean(
            [
                float(row["p_safety_violation"])
                for row in records
                if float(row.get("proxy_collision", 0.0)) < 0.5 and float(row.get("safety_violation", 0.0)) < 0.5
            ]
        ),
        "mean_p_safety_unsafe": _mean(
            [
                float(row["p_safety_violation"])
                for row in records
                if float(row.get("proxy_collision", 0.0)) >= 0.5 or float(row.get("safety_violation", 0.0)) >= 0.5
            ]
        ),
        "pairwise_contrast": {
            "viability": _pair_contrast(grouped, label="merge_before_taper", score="p_merge_before_taper", positive_higher=True),
            "safety": _pair_contrast(safety_grouped, label="safety_event", score="p_safety_violation", positive_higher=True),
            "taper_miss": _pair_contrast(grouped, label="taper_miss", score="p_taper_miss", positive_higher=True),
        },
        "raw_fail_left_success": _raw_fail_left_success_summary(grouped),
        "per_action": _action_stats(records, thresholds),
        "raw_action": _raw_action_summary(grouped),
        "raw_probability_recommendation": _raw_probability_recommendation_summary(grouped),
        "calibrated_gate_selection": _gate_selection_summary(grouped, thresholds),
    }
    summary["verdict"] = _verdict(summary, diagnostic_thresholds)
    return summary


def candidate_records_from_dataset(
    models: list[Any],
    dataset: ACCVPBranchDataset,
    calibration: CalibrationBundle | None,
    torch: Any,
    *,
    batch_size: int = 64,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for model in models:
        model.eval()
    eligible_indices = [
        index
        for index, row in enumerate(dataset.rows)
        if _activation_window(dataset.roots[str(row["root_id"])])
    ]
    with torch.no_grad():
        for start in range(0, len(eligible_indices), max(1, int(batch_size))):
            indices = eligible_indices[start : start + max(1, int(batch_size))]
            batch_np = collate_numpy(dataset[index] for index in indices)
            batch = _tensor_batch(batch_np, torch)
            outputs = [_model_output(model, batch) for model in models]
            events = np.stack([torch.sigmoid(output["event_logits"]).cpu().numpy() for output in outputs], axis=0)
            geometry = np.stack([output["geometry"].cpu().numpy() for output in outputs], axis=0)
            raw_scores = {
                "p_proxy_collision": events[:, :, 0].max(axis=0).tolist(),
                "p_safety_violation": events[:, :, 1].max(axis=0).tolist(),
                "p_merge_before_taper": events[:, :, 3].min(axis=0).tolist(),
            }
            if calibration is not None:
                bounds = calibration.score(raw_scores)
            else:
                bounds = {
                    "pU_proxy_collision": np.asarray(raw_scores["p_proxy_collision"], dtype=np.float64),
                    "pU_safety_violation": np.asarray(raw_scores["p_safety_violation"], dtype=np.float64),
                    "pL_merge_before_taper": np.asarray(raw_scores["p_merge_before_taper"], dtype=np.float64),
                }
            p_taper_miss = events[:, :, 2].max(axis=0)
            geometry_q = np.median(geometry, axis=0)
            disagreement = events.std(axis=0).mean(axis=1)
            for local, index in enumerate(indices):
                row = dataset.rows[index]
                root = dataset.roots[str(row["root_id"])]
                records.append(
                    {
                        "root_id": str(row["root_id"]),
                        "root_observation_fingerprint": str(
                            dataset.observation_fingerprint_by_root.get(
                                str(row["root_id"]), ""
                            )
                        ),
                        "split_component_id": str(
                            dataset.split_component_by_root.get(
                                str(row["root_id"]), ""
                            )
                        ),
                        "episode_seed": int(root.get("episode_seed", row.get("episode_seed", -1))),
                        "action_id": int(row["action_id"]),
                        "raw_action_id": root.get("raw_action_id", row.get("raw_action_id")),
                        "raw_action_legal": bool(root.get("raw_action_legal", row.get("raw_action_legal", False))),
                        "root_policy": str(root.get("root_policy", root.get("root_source", ""))),
                        "collection_source": str(root.get("collection_source", root.get("root_policy", ""))),
                        "traffic_profile": str(root.get("traffic_profile", "unknown")),
                        "activation_bin": str(root.get("activation_bin", root.get("deadline_bin", ""))),
                        "p_proxy_collision": float(raw_scores["p_proxy_collision"][local]),
                        "p_safety_violation": float(raw_scores["p_safety_violation"][local]),
                        "p_taper_miss": float(p_taper_miss[local]),
                        "p_merge_before_taper": float(raw_scores["p_merge_before_taper"][local]),
                        "pU_proxy_collision": float(bounds["pU_proxy_collision"][local]),
                        "pU_safety_violation": float(bounds["pU_safety_violation"][local]),
                        "pL_merge_before_taper": float(bounds["pL_merge_before_taper"][local]),
                        "q10_min_distance": float(max(0.0, geometry_q[local, 0])),
                        "q90_drac": float(max(0.0, geometry_q[local, 1])),
                        "target_front_gap": float(max(0.0, geometry_q[local, 2])),
                        "target_rear_gap": float(max(0.0, geometry_q[local, 3])),
                        "target_lane_entry_time_s": float(max(0.0, geometry_q[local, 4])),
                        "ensemble_disagreement": float(disagreement[local]),
                        "proxy_collision": float(batch_np["event_targets"][local, 0]),
                        "safety_violation": float(batch_np["event_targets"][local, 1]),
                        "taper_miss": float(batch_np["event_targets"][local, 2]),
                        "merge_before_taper": float(batch_np["event_targets"][local, 3]),
                        "merge_observed": bool(batch_np["event_mask"][local, 3]),
                        "candidate_legal": _candidate_legal(row),
                        "secondary_safety_pass": _secondary_safety_pass(row),
                        "secondary_risk_score": _secondary_risk_score(row),
                        "secondary_risk_uncertainty": _secondary_risk_uncertainty(row),
                        "secondary_veto_reason": _secondary_veto_reason(row),
                        "oracle_min_obb_distance": (
                            None if row.get("min_obb_distance") is None else float(row.get("min_obb_distance", 0.0))
                        ),
                        "oracle_min_ttc": None if row.get("min_ttc") is None else float(row.get("min_ttc", 0.0)),
                        "oracle_max_drac": None if row.get("max_drac") is None else float(row.get("max_drac", 0.0)),
                        "oracle_geometric_overlap": bool(row.get("geometric_overlap_within_horizon", False)),
                    }
                )
    return records


def load_models_from_checkpoint(config: Any, checkpoint: str | Path, torch: Any) -> list[Any]:
    payload = torch.load(checkpoint, map_location="cpu")
    metadata = dict(payload.get("metadata", {}) or {})
    kwargs = dict(metadata.get("model_kwargs", {}) or {})
    if not kwargs:
        from safe_rl.accvp.model import model_kwargs_from_config

        kwargs = model_kwargs_from_config(config)
    states = payload.get("model_state_dicts")
    if not states:
        raise ValueError("ACCVP checkpoint has no model_state_dicts")
    models = []
    for state in states:
        model = ACCVPPredictor(**kwargs)
        model.load_state_dict(state)
        model.eval()
        models.append(model)
    return models


def load_calibration(path: str | Path | None) -> CalibrationBundle | None:
    if path is None:
        return None
    value = Path(path)
    if not value.exists():
        raise FileNotFoundError(f"ACCVP calibration bundle does not exist: {value}")
    with value.open("r", encoding="utf-8") as handle:
        return CalibrationBundle.from_dict(json.load(handle))


def candidate_table_diagnostics(
    *,
    config: Any,
    dataset_dir: str | Path,
    splits: list[str],
    checkpoint: str | Path,
    calibration_path: str | Path | None,
    output: str | Path | None = None,
) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError("ACCVP candidate-table diagnostics require torch.") from exc

    dataset_dir = Path(dataset_dir)
    split_manifest = dataset_dir / "manifests" / "split_manifest.jsonl"
    if not split_manifest.exists():
        raise FileNotFoundError("missing ACCVP split_manifest.jsonl; run diagnostics on the formal training split")
    models = load_models_from_checkpoint(config, checkpoint, torch)
    calibration = load_calibration(calibration_path)
    thresholds = {
        "proxy_collision_upper_bound": float(config.accvp.proxy_collision_upper_bound),
        "safety_violation_upper_bound": float(config.accvp.safety_violation_upper_bound),
        "merge_viability_lower_bound": float(config.accvp.merge_viability_lower_bound),
    }
    split_reports = {}
    for split in splits:
        dataset = ACCVPBranchDataset(dataset_dir, split)
        records = candidate_records_from_dataset(models, dataset, calibration, torch)
        split_reports[split] = candidate_table_summary(records, split=split, thresholds=thresholds)
        split_reports[split]["dataset_row_count"] = int(len(dataset.rows))
    primary_split = "test" if "test" in split_reports else str(splits[0])
    report = {
        "artifact_kind": "accvp_candidate_table_diagnostics_v1",
        "deployable_claim": False,
        "primary_split": primary_split,
        "step1_5_state": split_reports[primary_split]["verdict"]["step1_5_state"],
        "splits": split_reports,
        "thresholds": {
            "selection": thresholds,
            "diagnostic": dict(DEFAULT_STEP1_5_THRESHOLDS),
        },
        "dataset_dir": str(dataset_dir.resolve()),
        "split_manifest_sha256": file_sha256(split_manifest),
        "checkpoint": str(Path(checkpoint).resolve()),
        "checkpoint_sha256": file_sha256(checkpoint),
        "config_hash": stable_hash(dict(config)),
    }
    dataset_manifest = dataset_dir / "manifests" / "dataset_manifest.json"
    if dataset_manifest.exists():
        report["dataset_manifest_sha256"] = file_sha256(dataset_manifest)
        report["dataset_manifest"] = read_json(dataset_manifest)
    if calibration_path is not None:
        report["calibration"] = str(Path(calibration_path).resolve())
        report["calibration_sha256"] = file_sha256(calibration_path)
    if output is not None:
        report["output"] = str(Path(output).resolve())
    return report
