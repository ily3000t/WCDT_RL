from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

import numpy as np

from safe_rl.pipeline.common import load_stage_config, run_root
from safe_rl.sim.metrics import INF_TTC
from safe_rl.utils.stage1_dataset import open_stage1_dataset


EDGE_ROLE_NAMES = {
    0: "unknown",
    1: "ramp",
    2: "auxiliary",
    3: "mainline",
    4: "target",
}


def _summary(values: np.ndarray | list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"count": 0}
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "p50": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def _histogram(values: np.ndarray | list[int]) -> dict[str, int]:
    array = np.asarray(values).reshape(-1)
    if array.size == 0:
        return {}
    unique, counts = np.unique(array, return_counts=True)
    return {str(int(key)): int(value) for key, value in zip(unique, counts)}


def _parse_metadata_array(values: np.ndarray, mask: np.ndarray) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    selected = np.asarray(values)[mask]
    for raw in selected.tolist():
        try:
            items = json.loads(str(raw)) if str(raw).strip() else []
        except json.JSONDecodeError:
            continue
        if isinstance(items, dict):
            items = [items]
        if not isinstance(items, list):
            continue
        for item in items:
            if isinstance(item, dict):
                rows.append(item)
    return rows


def _metadata_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "available": False,
            "count": 0,
            "reason": "no metadata rows",
        }
    role_counts = Counter(str(item.get("role", "")) for item in rows)
    class_counts = Counter(str(item.get("relevance_class", "")) for item in rows)
    reason_counts: Counter[str] = Counter()
    reason_set_counts: Counter[str] = Counter()
    low_risk_like = 0
    hard_critical_like = 0
    for item in rows:
        reasons = [str(reason) for reason in item.get("relevance_reasons", []) or []]
        reason_counts.update(reasons)
        reason_set_counts["+".join(sorted(reasons)) or "none"] += 1
        role = str(item.get("role", ""))
        ttc = float(item.get("ttc", INF_TTC))
        effective_gap = float(item.get("effective_gap", INF_TTC))
        is_hard = bool(
            role in {"target_front", "target_rear"}
            or "ttc" in reasons
            or "effective_gap" in reasons
            or "lowest_ttc" in reasons
        )
        hard_critical_like += int(is_hard)
        low_risk_like += int(
            not is_hard
            and ttc >= 5.0
            and effective_gap > 35.0
            and set(reasons).issubset({"current_gap", "nearest_conflict", "merge_local"})
        )
    return {
        "available": True,
        "count": int(len(rows)),
        "role_counts": dict(sorted(role_counts.items())),
        "relevance_class_counts": dict(sorted(class_counts.items())),
        "relevance_reason_counts": dict(sorted(reason_counts.items())),
        "relevance_reason_set_counts": dict(sorted(reason_set_counts.items())),
        "current_surface_gap": _summary([float(item.get("current_surface_gap", np.nan)) for item in rows]),
        "effective_gap": _summary([float(item.get("effective_gap", np.nan)) for item in rows]),
        "ttc": _summary([float(item.get("ttc", np.nan)) for item in rows]),
        "closing_speed": _summary([float(item.get("closing_speed", np.nan)) for item in rows]),
        "hard_critical_like_count": int(hard_critical_like),
        "hard_critical_like_rate": float(hard_critical_like / len(rows)),
        "low_risk_like_count": int(low_risk_like),
        "low_risk_like_rate": float(low_risk_like / len(rows)),
    }


def _rate(mask: np.ndarray) -> float:
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    return float(np.mean(mask)) if mask.size else 0.0


def _transition_lookup(dataset: Any) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    if "transition_episode_id" not in dataset or "transition_episode_step" not in dataset:
        return {}
    episode_ids = np.asarray(dataset["transition_episode_id"], dtype=np.int64)
    steps = np.asarray(dataset["transition_episode_step"], dtype=np.int64)
    grouped: dict[int, list[tuple[int, int]]] = {}
    for index, (episode, step) in enumerate(zip(episode_ids, steps)):
        grouped.setdefault(int(episode), []).append((int(step), int(index)))
    output: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for episode, rows in grouped.items():
        rows.sort(key=lambda item: item[0])
        output[episode] = (
            np.asarray([item[0] for item in rows], dtype=np.int64),
            np.asarray([item[1] for item in rows], dtype=np.int64),
        )
    return output


def _transition_values_for_trajectories(dataset: Any, key: str, default: float = np.nan) -> np.ndarray:
    if key not in dataset or "trajectory_episode_id" not in dataset or "trajectory_window_end_step" not in dataset:
        count = int(dataset["critical_actor_count"].shape[0]) if "critical_actor_count" in dataset else 0
        return np.full((count,), default, dtype=np.float64)
    lookup = _transition_lookup(dataset)
    values = np.asarray(dataset[key])
    episode_ids = np.asarray(dataset["trajectory_episode_id"], dtype=np.int64)
    end_steps = np.asarray(dataset["trajectory_window_end_step"], dtype=np.int64)
    output = np.full((episode_ids.shape[0],), default, dtype=np.float64)
    for index, item in enumerate(zip(episode_ids, end_steps)):
        episode_steps = lookup.get(int(item[0]))
        if episode_steps is None:
            continue
        steps, transition_indices = episode_steps
        local_index = int(np.searchsorted(steps, int(item[1]), side="right") - 1)
        if local_index < 0:
            continue
        transition_index = int(transition_indices[local_index])
        if transition_index < values.shape[0]:
            try:
                output[index] = float(values[transition_index])
            except (TypeError, ValueError):
                output[index] = default
    return output


def _selected_actor_role_summary(dataset: Any, mask: np.ndarray) -> dict[str, Any]:
    required = {"agent_mask", "agent_edge_role", "agent_relevance_mask", "agent_relevance_score"}
    if not required.issubset(set(dataset.files)):
        return {"available": False, "reason": "Stage1 dataset lacks selected actor role/relevance arrays"}
    agent_mask = np.asarray(dataset["agent_mask"])[mask]
    edge_roles = np.asarray(dataset["agent_edge_role"])[mask]
    relevance_mask = np.asarray(dataset["agent_relevance_mask"])[mask]
    relevance_score = np.asarray(dataset["agent_relevance_score"])[mask]
    actor_slots = agent_mask[:, 1:] > 0.5
    selected_edge_roles = edge_roles[:, 1:][actor_slots]
    selected_relevance = relevance_mask[:, 1:][actor_slots]
    selected_scores = relevance_score[:, 1:][actor_slots]
    role_counts: dict[str, int] = {}
    for role_id in selected_edge_roles.astype(np.int64).tolist():
        name = EDGE_ROLE_NAMES.get(int(role_id), f"role_{int(role_id)}")
        role_counts[name] = role_counts.get(name, 0) + 1
    return {
        "available": True,
        "selected_actor_slot_count": int(actor_slots.sum()),
        "selected_edge_role_counts": role_counts,
        "selected_relevance_mask_rate": float(np.mean(selected_relevance > 0.5)) if selected_relevance.size else 0.0,
        "selected_relevance_score": _summary(selected_scores),
    }


def _example_rows(dataset: Any, mask: np.ndarray, *, max_examples: int) -> list[dict[str, Any]]:
    indices = np.flatnonzero(mask)[: max(0, int(max_examples))]
    distance_to_taper = _transition_values_for_trajectories(dataset, "distance_to_taper")
    target_lane_gap = _transition_values_for_trajectories(dataset, "target_lane_gap")
    rows: list[dict[str, Any]] = []
    for index in indices.tolist():
        rows.append(
            {
                "trajectory_index": int(index),
                "episode_id": int(dataset["trajectory_episode_id"][index]) if "trajectory_episode_id" in dataset else None,
                "episode_seed": int(dataset["trajectory_episode_seed"][index]) if "trajectory_episode_seed" in dataset else None,
                "window_end_step": int(dataset["trajectory_window_end_step"][index]) if "trajectory_window_end_step" in dataset else None,
                "distance_to_taper": float(distance_to_taper[index]) if np.isfinite(distance_to_taper[index]) else None,
                "target_lane_gap": float(target_lane_gap[index]) if np.isfinite(target_lane_gap[index]) else None,
                "critical_actor_count": int(dataset["critical_actor_count"][index]) if "critical_actor_count" in dataset else None,
                "contextual_actor_count": int(dataset["contextual_actor_count"][index]) if "contextual_actor_count" in dataset else None,
                "contextual_actor_truncated_count": int(dataset["contextual_actor_truncated_count"][index]) if "contextual_actor_truncated_count" in dataset else None,
                "actor_selector_relevant_count": int(dataset["actor_selector_relevant_count"][index]) if "actor_selector_relevant_count" in dataset else None,
            }
        )
    return rows


def build_actor_selector_overflow_audit(cfg: Any, *, max_examples: int = 20) -> dict[str, Any]:
    dataset_path = run_root(cfg) / "stage1" / "risk_probe_buffer"
    with open_stage1_dataset(dataset_path) as dataset:
        if "critical_actor_count" not in dataset or "critical_actor_overflow" not in dataset:
            return {
                "available": False,
                "dataset": str(dataset_path),
                "reason": "Stage1 dataset does not contain selector v2 critical overflow fields",
            }
        critical_count = np.asarray(dataset["critical_actor_count"], dtype=np.int64)
        contextual_count = (
            np.asarray(dataset["contextual_actor_count"], dtype=np.int64)
            if "contextual_actor_count" in dataset
            else np.zeros_like(critical_count)
        )
        contextual_truncated = (
            np.asarray(dataset["contextual_actor_truncated_count"], dtype=np.int64)
            if "contextual_actor_truncated_count" in dataset
            else np.zeros_like(critical_count)
        )
        relevant_count = (
            np.asarray(dataset["actor_selector_relevant_count"], dtype=np.int64)
            if "actor_selector_relevant_count" in dataset
            else critical_count + contextual_count
        )
        overflow = np.asarray(dataset["critical_actor_overflow"], dtype=np.float32) > 0.5
        distance_to_taper = _transition_values_for_trajectories(dataset, "distance_to_taper")
        target_gap = _transition_values_for_trajectories(dataset, "target_lane_gap")
        near_taper = np.isfinite(distance_to_taper) & (distance_to_taper <= 120.0)

        overflow_critical = critical_count[overflow]
        overflow_contextual = contextual_count[overflow]
        overflow_truncated = contextual_truncated[overflow]
        likely_capacity = bool(overflow_critical.size and np.percentile(overflow_critical, 50) >= 6.0)
        contextual_only = bool(np.any((contextual_truncated > 0) & ~overflow))
        dropped_metadata_available = "dropped_critical_actor_metadata_json" in dataset
        critical_metadata_available = "critical_actor_metadata_json" in dataset
        dropped_rows = (
            _parse_metadata_array(dataset["dropped_critical_actor_metadata_json"], overflow)
            if dropped_metadata_available
            else []
        )
        critical_rows = (
            _parse_metadata_array(dataset["critical_actor_metadata_json"], overflow)
            if critical_metadata_available
            else []
        )
        dropped_summary = _metadata_summary(dropped_rows)
        critical_summary = _metadata_summary(critical_rows)
        schema_has_dropped_metadata = bool(dropped_metadata_available)
        evidence_limitations = []
        if not schema_has_dropped_metadata:
            evidence_limitations.append(
                "Current Stage1 schema stores overflow counts but not dropped actor IDs/reasons/TTC/effective_gap; "
                "this audit cannot identify the exact actor that exceeded capacity."
            )
        interpretation = (
            "capacity_or_critical_threshold_issue"
            if likely_capacity
            else "no_strong_capacity_signal"
        )
        if likely_capacity and float(np.mean(overflow_critical == 6)) >= 0.5:
            interpretation = "mostly_one_extra_critical_actor"
        if dropped_summary.get("available"):
            if float(dropped_summary.get("low_risk_like_rate", 0.0)) >= 0.5:
                interpretation = "likely_overbroad_critical_definition"
            elif float(dropped_summary.get("hard_critical_like_rate", 0.0)) >= 0.5:
                interpretation = "likely_true_capacity_pressure"

        return {
            "available": True,
            "run_id": str(cfg.run.run_id),
            "dataset": str(dataset_path),
            "trajectory_count": int(critical_count.shape[0]),
            "selector_version": str(dataset["actor_selection_version"].item()) if "actor_selection_version" in dataset else None,
            "actor_selection_config_hash": str(dataset["actor_selection_config_hash"].item()) if "actor_selection_config_hash" in dataset else None,
            "critical_actor_overflow_count": int(np.sum(overflow)),
            "critical_actor_overflow_rate": _rate(overflow),
            "critical_actor_count": _summary(critical_count),
            "critical_actor_count_histogram": _histogram(critical_count),
            "overflow_critical_actor_count": _summary(overflow_critical),
            "overflow_critical_actor_count_histogram": _histogram(overflow_critical),
            "contextual_actor_count": _summary(contextual_count),
            "contextual_actor_truncated_total": int(np.sum(contextual_truncated)),
            "contextual_actor_truncated_rate": _rate(contextual_truncated > 0),
            "contextual_truncation_without_critical_overflow_rate": _rate((contextual_truncated > 0) & ~overflow),
            "overflow_contextual_actor_count": _summary(overflow_contextual),
            "overflow_contextual_truncated_count": _summary(overflow_truncated),
            "relevant_actor_count": _summary(relevant_count),
            "near_taper_overflow_rate": _rate(overflow & near_taper),
            "overflow_distance_to_taper": _summary(distance_to_taper[overflow]),
            "overflow_target_lane_gap": _summary(target_gap[overflow]),
            "selected_actor_roles_all": _selected_actor_role_summary(dataset, np.ones_like(overflow, dtype=bool)),
            "selected_actor_roles_overflow": _selected_actor_role_summary(dataset, overflow),
            "dropped_critical_actor_metadata": dropped_summary,
            "overflow_critical_actor_metadata": critical_summary,
            "example_overflow_windows": _example_rows(dataset, overflow, max_examples=max_examples),
            "interpretation": interpretation,
            "schema_has_dropped_actor_metadata": schema_has_dropped_metadata,
            "evidence_limitations": evidence_limitations,
            "recommended_next_step": (
                "Rerun Stage1 or Stage5 with metadata-enabled code; old buffers cannot reconstruct dropped actor reasons."
                if evidence_limitations
                else "Tune selector thresholds if dropped actors are low-risk-like; increase max agents only if hard-critical-like actors dominate."
            ),
            "notes": {
                "contextual_truncation_blocks_backstop": False,
                "critical_overflow_blocks_backstop": True,
                "contextual_only_truncation_observed": contextual_only,
            },
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit SAFE_RL actor selector critical overflow samples.")
    parser.add_argument("--config", default=None, help="Optional YAML config overlay.")
    parser.add_argument("--run-id", required=True, help="Existing run id to audit.")
    parser.add_argument("--max-examples", type=int, default=20, help="Maximum overflow windows to include.")
    parser.add_argument("--output", default=None, help="Optional output JSON path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_stage_config(args)
    report = build_actor_selector_overflow_audit(cfg, max_examples=int(args.max_examples))
    output = Path(args.output) if args.output else run_root(cfg) / "stage5" / "diagnostics" / "actor_selector_overflow_audit.json"
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as file:
            json.dump(report, file, indent=2, ensure_ascii=False, sort_keys=True)
            file.write("\n")
        report["output"] = str(output)
    except OSError as exc:
        report["output"] = str(output)
        report["output_error"] = str(exc)
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
