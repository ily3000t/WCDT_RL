from __future__ import annotations

from collections import Counter, defaultdict
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np

from safe_rl.accvp.schema import file_sha256, read_json, stable_hash, write_json_atomic
from safe_rl.accvp.selection import LEFT_ACTION_IDS, select_viability_lite_action


def lite_thresholds_from_config(config: Any) -> dict[str, float]:
    lite = config.accvp.get("viability_lite", {}) or {}
    return {
        "min_p_merge_before_taper": float(lite.get("min_p_merge_before_taper", 0.75)),
        "min_improvement_over_raw": float(lite.get("min_improvement_over_raw", 0.01)),
        "max_target_entry_time_s": float(lite.get("max_target_entry_time_s", 8.0)),
        "max_ensemble_disagreement": float(lite.get("max_ensemble_disagreement", 0.20)),
        "max_secondary_risk_score": float(lite.get("max_secondary_risk_score", 1.0)),
    }


def _by_root(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[str(row["root_id"])].append(row)
    return grouped


def _safe(row: dict[str, Any]) -> bool:
    return float(row.get("proxy_collision", 0.0)) < 0.5 and float(row.get("safety_violation", 0.0)) < 0.5


def _success(row: dict[str, Any]) -> bool:
    return bool(row.get("merge_observed", False)) and float(row.get("merge_before_taper", 0.0)) >= 0.5


def _risk_pass(row: dict[str, Any]) -> bool:
    return bool(row.get("candidate_legal", True)) and bool(row.get("secondary_safety_pass", True))


def _rate_or_nan(values: list[bool]) -> float:
    return float(np.mean([bool(value) for value in values])) if values else float("nan")


def _repairable(candidates: list[dict[str, Any]]) -> bool:
    raw_action_id = int(candidates[0].get("raw_action_id", -1))
    raw = next((row for row in candidates if int(row["action_id"]) == raw_action_id), None)
    return bool(
        raw is not None
        and bool(raw.get("merge_observed", False))
        and not _success(raw)
        and any(int(row["action_id"]) in LEFT_ACTION_IDS and _risk_pass(row) and _safe(row) and _success(row) for row in candidates)
    )


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
    repairable_count = 0
    repairable_captured = 0
    unnecessary_replacements = 0
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
            if raw is not None and _success(raw):
                unnecessary_replacements += 1
            examples.append(
                {
                    "root_id": root_id,
                    "raw_action_id": raw_action_id,
                    "selected_action_id": int(chosen["action_id"]),
                    "raw_p_merge_before_taper": None if raw is None else float(raw.get("p_merge_before_taper", 0.0)),
                    "selected_p_merge_before_taper": float(chosen.get("p_merge_before_taper", 0.0)),
                    "p_merge_improvement": float(decision.get("p_merge_improvement", 0.0)),
                    "selected_merge_success": _success(chosen),
                    "selected_safety_event": not _safe(chosen),
                }
            )
        else:
            raw_retained += 1
        if _repairable(candidates):
            repairable_count += 1
            if bool(decision.get("replacement", False)) and chosen is not None and int(chosen["action_id"]) in LEFT_ACTION_IDS:
                repairable_captured += 1
    selected_observed = [row for row in selected if bool(row.get("merge_observed", False))]
    replacement_observed = [row for row in replacements if bool(row.get("merge_observed", False))]
    replacement_risk_pass_rate = _rate_or_nan([_risk_pass(row) for row in replacements])
    replacement_safety_event_rate = _rate_or_nan([not _safe(row) for row in replacements])
    replacement_merge_success_rate = _rate_or_nan([_success(row) for row in replacement_observed])
    replacement_unnecessary_rate = float(unnecessary_replacements / max(1, len(replacements)))
    replacement_repairable_capture_rate = float(repairable_captured / max(1, repairable_count))
    return {
        "split": split,
        "thresholds": dict(thresholds),
        "decision_count": int(len(grouped)),
        "selected_count": int(len(selected)),
        "replacement_count": int(len(replacements)),
        "replacement_rate": float(len(replacements) / max(1, len(grouped))),
        "raw_retention_rate": float(raw_retained / max(1, len(grouped))),
        "repairable_root_count": int(repairable_count),
        "repairable_root_capture_rate": float(repairable_captured / max(1, repairable_count)),
        "replacement_repairable_capture_rate": replacement_repairable_capture_rate,
        "selected_action_risk_pass_rate": (
            float(np.mean([_risk_pass(row) for row in selected])) if selected else float("nan")
        ),
        "selected_action_safety_event_rate": (
            float(np.mean([not _safe(row) for row in selected])) if selected else float("nan")
        ),
        "selected_action_merge_success_rate": (
            float(np.mean([_success(row) for row in selected_observed])) if selected_observed else float("nan")
        ),
        "replacement_merge_success_rate": (
            float(np.mean([_success(row) for row in replacement_observed])) if replacement_observed else float("nan")
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
    lite = config.accvp.get("viability_lite", {}) or {}
    max_replacement_rate = float(lite.get("max_replacement_rate", 0.50))
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
        }
        evaluated.append(evaluate_lite_thresholds(records, thresholds, split=split))
    feasible = [
        row
        for row in evaluated
        if int(row["replacement_count"]) > 0
        and float(row["replacement_action_risk_pass_rate"]) == 1.0
        and float(row["replacement_action_safety_event_rate"]) == 0.0
        and float(row["replacement_rate"]) <= max_replacement_rate
    ]
    candidates = feasible or evaluated
    selected = max(
        candidates,
        key=lambda row: (
            float(row["replacement_repairable_capture_rate"]),
            float(row["replacement_action_merge_success_rate"]) if np.isfinite(float(row["replacement_action_merge_success_rate"])) else -1.0,
            -float(row["replacement_action_safety_event_rate"]) if np.isfinite(float(row["replacement_action_safety_event_rate"])) else -1.0,
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
        "accvp_safety_head_hard_gate": False,
        "deployable_claim": "task_viability_only",
        "max_replacement_rate": max_replacement_rate,
        "selected": selected["thresholds"],
        "selected_metrics": selected,
        "evaluated_points": evaluated,
    }


def write_lite_artifacts(
    *,
    output_dir: str | Path,
    config: Any,
    dataset_dir: str | Path,
    checkpoint: str | Path,
    calibration: str | Path,
    operating_point: dict[str, Any],
    final_test: dict[str, Any],
    artifact_prefix: str = "accvp_v1_lite",
) -> dict[str, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    prefix = str(artifact_prefix).strip() or "accvp_v1_lite"
    operating_path = output / f"{prefix}_operating_point.json"
    final_path = output / f"{prefix}_final_test_diagnostics.json"
    write_json_atomic(operating_path, operating_point)
    write_json_atomic(final_path, final_test)
    dataset_dir = Path(dataset_dir)
    dataset_manifest = dataset_dir / "manifests" / "dataset_manifest.json"
    split_manifest = dataset_dir / "manifests" / "split_manifest.jsonl"
    manifest_payload = read_json(dataset_manifest)
    manifest = {
        "artifact_kind": "accvp_v1_lite_task_artifact_bundle",
        "controller": "acv_shield_lite",
        "safety_authority": "risk_module_safety_shield",
        "accvp_safety_head_hard_gate": False,
        "deployable_claim": "task_viability_only",
        "predictor_sha256": file_sha256(checkpoint),
        "calibration_sha256": file_sha256(calibration),
        "operating_point_sha256": file_sha256(operating_path),
        "final_test_diagnostics_sha256": file_sha256(final_path),
        "dataset_manifest_sha256": file_sha256(dataset_manifest),
        "split_manifest_sha256": file_sha256(split_manifest),
        "dataset_fingerprint": str(manifest_payload.get("dataset_fingerprint", "")),
        "risk_model_fingerprint": str(manifest_payload.get("risk_model_fingerprint", "")),
        "counterfactual_schema_version": int(manifest_payload.get("counterfactual_schema_version", 2)),
        "accvp_activation_distance_m": float(manifest_payload.get("accvp_activation_distance_m", -1.0)),
        "data_contract_hash": str(manifest_payload.get("data_contract_hash", "")),
        "config_hash": stable_hash(dict(config)),
    }
    manifest["artifact_fingerprint"] = stable_hash(manifest)
    manifest_path = write_json_atomic(output / f"{prefix}_task_artifact_manifest.json", manifest)
    return {
        "operating_point": operating_path,
        "final_test": final_path,
        "artifact_manifest": manifest_path,
    }
