from __future__ import annotations

from collections import Counter
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from safe_rl.pipeline.common import latest_stage_file, load_stage_config, parse_config_arg, write_report
from safe_rl.prediction.forecast_feature_augmentor import forecast_target_lane_gap_from_trajectories
from safe_rl.sim.scenario_semantics import (
    EDGE_ROLE_AUXILIARY,
    EDGE_ROLE_RAMP,
    EDGE_ROLE_TARGET,
    distance_to_taper_for_position,
    infer_lane_index,
    is_ramp_side_y,
    target_lane_center_at_x,
)
from safe_rl.utils.config import prepare_run_dir
from safe_rl.utils.progress import TensorboardLogger, progress_iter, stage_log


def _require_torch():
    try:
        import torch
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Stage2 requires torch. Activate the SAFE_RL training environment.") from exc
    return torch, DataLoader, TensorDataset


def _resolve_device(cfg: Any, torch: Any):
    training_cfg = cfg.get("training", {})
    requested = str(training_cfg.get("stage2_device", training_cfg.get("device", "auto"))).strip().lower()
    if requested in ("auto", ""):
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "gpu":
        requested = "cuda"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("training.device requests CUDA, but torch.cuda.is_available() is false.")
    if requested == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        raise RuntimeError("training.device requests MPS, but PyTorch MPS is unavailable.")
    return torch.device(requested)


def _configure_torch_backend(cfg: Any, torch: Any, device: Any) -> None:
    training_cfg = cfg.get("training", {})
    if device.type == "cuda":
        allow_tf32 = bool(training_cfg.get("cuda_tf32", True))
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32
        if allow_tf32 and hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")


def _loader_kwargs(cfg: Any, device: Any) -> dict[str, Any]:
    training_cfg = cfg.get("training", {})
    num_workers = int(training_cfg.get("num_workers", 0))
    pin_memory = bool(training_cfg.get("pin_memory", True)) and device.type == "cuda"
    kwargs: dict[str, Any] = {"num_workers": num_workers, "pin_memory": pin_memory}
    if num_workers > 0:
        kwargs["persistent_workers"] = bool(training_cfg.get("persistent_workers", True))
    return kwargs


def _to_device(batch, device: Any, non_blocking: bool):
    return tuple(tensor.to(device, non_blocking=non_blocking) for tensor in batch)


def _cpu_state_dict(model: Any) -> dict[str, Any]:
    return {key: value.detach().cpu() for key, value in model.state_dict().items()}


def _prediction_loss_summary(history: list[float] | None) -> dict | None:
    if not history:
        return None
    return {
        "epochs": len(history),
        "first": float(history[0]),
        "last": float(history[-1]),
        "min": float(min(history)),
    }


def _prediction_val_score(metrics: dict[str, Any], cfg: Any) -> float:
    fde = float(metrics.get("fde", {}).get("mean", 0.0))
    gap = float(metrics.get("target_lane_gap_abs_error", {}).get("mean", 0.0))
    min_distance = float(metrics.get("future_min_distance_abs_error", {}).get("mean", 0.0))
    return (
        fde
        + float(cfg.prediction.get("target_lane_gap_metric_weight", 0.5)) * gap
        + float(cfg.prediction.get("future_min_distance_metric_weight", 0.5)) * min_distance
    )


def _summary(values: list[float] | np.ndarray) -> dict[str, float | int]:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"count": 0}
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "p01": float(np.percentile(arr, 1)),
        "p05": float(np.percentile(arr, 5)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _correlation(a_values: list[float], b_values: list[float]) -> float:
    a = np.asarray(a_values, dtype=np.float32)
    b = np.asarray(b_values, dtype=np.float32)
    mask = np.isfinite(a) & np.isfinite(b)
    a = a[mask]
    b = b[mask]
    if a.size < 2 or float(np.std(a)) <= 1.0e-8 or float(np.std(b)) <= 1.0e-8:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _ordered_prediction_indices(
    cfg: Any,
    sample_history: np.ndarray,
    sample_mask: np.ndarray,
    lane_indices: np.ndarray | None = None,
    edge_roles: np.ndarray | None = None,
) -> list[int]:
    if sample_history.shape[0] <= 1:
        return []
    ego = sample_history[0, -1]
    ego_x = float(ego[0])
    def _priority(agent_idx: int) -> tuple[float, float, float, int]:
        latest = sample_history[agent_idx, -1]
        x = float(latest[0])
        y = float(latest[1])
        dx = x - ego_x
        lane = infer_lane_index(cfg, y) if lane_indices is None else int(lane_indices[agent_idx])
        role = 0 if edge_roles is None else int(edge_roles[agent_idx])
        target_center = target_lane_center_at_x(cfg, x)
        is_ramp_local = (
            role in (EDGE_ROLE_RAMP, EDGE_ROLE_AUXILIARY)
            if role > 0
            else is_ramp_side_y(cfg, y) and distance_to_taper_for_position(cfg, x, y, lane) > 0.0
        )
        is_target_lane = (
            role == EDGE_ROLE_TARGET
            if role > 0
            else (not is_ramp_local) and abs(y - target_center) <= 2.0
        )
        if is_target_lane and dx >= 0.0:
            group = 0
        elif is_target_lane and dx < 0.0:
            group = 1
        elif is_target_lane:
            group = 2
        elif is_ramp_local:
            group = 3
        else:
            group = 4
        return (float(group), abs(dx), abs(y - target_center), int(agent_idx))

    valid = [idx for idx in range(1, sample_history.shape[0]) if float(sample_mask[idx]) > 0.0]
    return sorted(valid, key=_priority)


def _stage1_path(cfg) -> Path:
    if cfg.stage2.input_stage1:
        return Path(cfg.stage2.input_stage1)
    return latest_stage_file(cfg, "stage1", str(cfg.stage1.output_name))


def _stage4_path(cfg) -> Path | None:
    input_stage4 = cfg.stage2.get("input_stage4")
    if not input_stage4:
        return None
    if str(input_stage4).lower() == "auto":
        return latest_stage_file(cfg, "stage4", "on_policy_failure_buffer.npz")
    return Path(input_stage4)


def _merge_risk_buffers(stage1_data: Any, stage4_data: Any | None) -> dict[str, np.ndarray] | Any:
    if stage4_data is None:
        return stage1_data
    risk_keys = (
        "risk_features",
        "actions",
        "overall_risk",
        "risk_types",
        "lane_oob_risk",
        "candidate_legal",
        "traffic_risk",
        "continuous_risk_target",
        "boundary_sample",
        "risk_sample_weight",
        "candidate_transition_id",
        "candidate_raw_action",
    )
    stage1 = _risk_training_arrays(stage1_data)
    stage4 = _risk_training_arrays(stage4_data)
    if stage1["candidate_transition_id"].size and stage4["candidate_transition_id"].size:
        offset = int(np.max(stage1["candidate_transition_id"])) + 1
        stage4["candidate_transition_id"] = stage4["candidate_transition_id"] + offset
    merged = {key: np.concatenate([stage1[key], stage4[key]], axis=0) for key in risk_keys}
    return merged


def _has_key(data: Any, key: str) -> bool:
    return key in data.files if hasattr(data, "files") else key in data


def _trajectory_schema_version(data: Any) -> int:
    if not _has_key(data, "trajectory_schema_version"):
        return 1
    return int(np.asarray(data["trajectory_schema_version"]).reshape(-1)[0])


def _require_trajectory_schema_v2(data: Any, consumer: str) -> None:
    version = _trajectory_schema_version(data)
    required = {
        "agent_history_valid_mask",
        "agent_future_valid_mask",
        "agent_history_lane_index",
        "agent_history_edge_role",
        "agent_future_lane_index",
        "agent_future_edge_role",
    }
    missing = sorted(key for key in required if not _has_key(data, key))
    if version < 2 or missing:
        raise ValueError(
            f"{consumer} training requires trajectory_schema_version>=2 with timestep masks and route metadata; "
            f"found version={version}, missing={missing}. Re-run Stage1 with a new run id."
        )


def _risk_training_arrays(data: Any, risk_type_count: int = 6) -> dict[str, np.ndarray]:
    risk_features = np.asarray(data["risk_features"], dtype=np.float32)
    actions = np.asarray(data["actions"], dtype=np.int64)
    risk_types = np.asarray(data["risk_types"], dtype=np.float32)
    if risk_types.ndim == 2 and risk_types.shape[1] < int(risk_type_count):
        padded_types = np.zeros((risk_types.shape[0], int(risk_type_count)), dtype=np.float32)
        padded_types[:, : risk_types.shape[1]] = risk_types
        risk_types = padded_types
    if _has_key(data, "traffic_risk"):
        traffic_risk = np.asarray(data["traffic_risk"], dtype=np.float32)
    elif risk_types.ndim == 2 and risk_types.shape[1] > 0:
        traffic_risk = np.max(risk_types, axis=1).astype(np.float32)
    else:
        traffic_risk = np.asarray(data["overall_risk"], dtype=np.float32)
    if _has_key(data, "lane_oob_risk"):
        lane_oob = np.asarray(data["lane_oob_risk"], dtype=np.float32)
    elif risk_features.ndim == 2 and risk_features.shape[1] > 5:
        lane_oob = (risk_features[:, 5] > 0.5).astype(np.float32)
    else:
        lane_oob = np.zeros_like(traffic_risk, dtype=np.float32)
    if _has_key(data, "candidate_legal"):
        candidate_legal = (np.asarray(data["candidate_legal"], dtype=np.float32) > 0.5).astype(np.float32)
    else:
        candidate_legal = (lane_oob <= 0.5).astype(np.float32)
    if _has_key(data, "risk_sample_weight"):
        sample_weight = np.asarray(data["risk_sample_weight"], dtype=np.float32)
    else:
        sample_weight = candidate_legal.astype(np.float32)
    if _has_key(data, "continuous_risk_target"):
        continuous_risk = np.asarray(data["continuous_risk_target"], dtype=np.float32)
    else:
        continuous_risk = traffic_risk.astype(np.float32)
    if _has_key(data, "boundary_sample"):
        boundary_sample = np.asarray(data["boundary_sample"], dtype=np.float32)
    else:
        boundary_sample = ((continuous_risk >= 0.20) & (continuous_risk < 0.80)).astype(np.float32)
    if _has_key(data, "candidate_transition_id"):
        transition_id = np.asarray(data["candidate_transition_id"], dtype=np.int64)
        transition_id_source = "explicit"
    else:
        transition_id = (np.arange(actions.shape[0], dtype=np.int64) // 9).astype(np.int64)
        transition_id_source = "inferred_by_9_rows"
    if _has_key(data, "candidate_raw_action"):
        raw_actions = np.asarray(data["candidate_raw_action"], dtype=np.int64)
    elif _has_key(data, "executed_actions"):
        executed = np.asarray(data["executed_actions"], dtype=np.int64)
        raw_actions = np.full_like(actions, -1, dtype=np.int64)
        valid = (transition_id >= 0) & (transition_id < executed.shape[0])
        raw_actions[valid] = executed[transition_id[valid]]
    else:
        raw_actions = np.full_like(actions, -1, dtype=np.int64)
    return {
        "risk_features": risk_features,
        "actions": actions,
        "overall_risk": traffic_risk.astype(np.float32),
        "risk_types": risk_types,
        "lane_oob_risk": lane_oob.astype(np.float32),
        "candidate_legal": candidate_legal.astype(np.float32),
        "traffic_risk": traffic_risk.astype(np.float32),
        "continuous_risk_target": continuous_risk.astype(np.float32),
        "boundary_sample": boundary_sample.astype(np.float32),
        "risk_sample_weight": sample_weight.astype(np.float32),
        "candidate_transition_id": transition_id.astype(np.int64),
        "candidate_raw_action": raw_actions.astype(np.int64),
        "candidate_transition_id_source": np.asarray([transition_id_source]),
    }


def _configured_sample_weight(cfg: Any, arrays: dict[str, np.ndarray]) -> np.ndarray:
    legal = arrays["candidate_legal"] > 0.5
    if bool(cfg.risk_module.get("legal_candidates_only_for_training", True)):
        weights = np.where(
            legal,
            np.maximum(arrays["risk_sample_weight"], 1.0),
            float(cfg.risk_module.get("illegal_candidate_sample_weight", 0.0)),
        ).astype(np.float32)
    else:
        weights = np.ones_like(arrays["traffic_risk"], dtype=np.float32)
    weights = weights * np.where(
        arrays["traffic_risk"] > 0.5,
        float(cfg.risk_module.get("positive_traffic_risk_weight", 1.0)),
        1.0,
    ).astype(np.float32)
    if arrays["risk_types"].ndim == 2 and arrays["risk_types"].shape[1] > 4:
        weights = weights * np.where(
            arrays["risk_types"][:, 4] > 0.5,
            float(cfg.risk_module.get("merge_conflict_weight", 1.0)),
            1.0,
        ).astype(np.float32)
    return weights.astype(np.float32)


def _split_indices(count: int, val_split: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(count, dtype=np.int64)
    if count <= 1 or val_split <= 0.0:
        return indices, np.asarray([], dtype=np.int64)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    val_count = int(round(count * val_split))
    val_count = min(max(val_count, 1), count - 1)
    return indices[val_count:], indices[:val_count]


def _split_risk_indices(arrays: dict[str, np.ndarray], val_split: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    transition_ids = np.asarray(arrays.get("candidate_transition_id", []), dtype=np.int64)
    if transition_ids.shape[0] != arrays["traffic_risk"].shape[0] or transition_ids.size == 0:
        return _split_indices(int(arrays["traffic_risk"].shape[0]), val_split, seed)
    unique_ids = np.unique(transition_ids)
    if unique_ids.shape[0] <= 1 or val_split <= 0.0:
        return np.arange(transition_ids.shape[0], dtype=np.int64), np.asarray([], dtype=np.int64)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_ids)
    val_count = int(round(unique_ids.shape[0] * val_split))
    val_count = min(max(val_count, 1), unique_ids.shape[0] - 1)
    val_ids = set(int(item) for item in unique_ids[:val_count])
    val_mask = np.asarray([int(item) in val_ids for item in transition_ids], dtype=bool)
    return np.where(~val_mask)[0].astype(np.int64), np.where(val_mask)[0].astype(np.int64)


def _risk_data_summary(arrays: dict[str, np.ndarray], sample_weight: np.ndarray) -> dict:
    actions = arrays["actions"]
    legal = arrays["candidate_legal"] > 0.5
    risk = arrays["traffic_risk"]
    continuous = arrays["continuous_risk_target"]
    lane_oob = arrays["lane_oob_risk"]
    weighted = sample_weight > 0.0
    return {
        "sample_count": int(risk.shape[0]),
        "traffic_risk_rate": float(np.mean(risk)) if risk.size else 0.0,
        "lane_oob_risk_rate": float(np.mean(lane_oob)) if lane_oob.size else 0.0,
        "illegal_candidate_rate": float(np.mean(~legal)) if legal.size else 0.0,
        "legal_candidate_risk_rate": float(np.mean(risk[legal])) if np.any(legal) else 0.0,
        "weighted_sample_rate": float(np.mean(weighted)) if weighted.size else 0.0,
        "weighted_positive_risk_rate": float(np.mean(risk[weighted])) if np.any(weighted) else 0.0,
        "continuous_risk": _summary(continuous[legal] if np.any(legal) else continuous),
        "continuous_risk_coverage": _continuous_risk_coverage(continuous[legal] if np.any(legal) else continuous),
        "traffic_risk_by_action": {
            str(index): float(np.mean(risk[actions == index])) if np.any(actions == index) else 0.0
            for index in range(9)
        },
        "legal_candidate_action_risk_rate": {
            str(index): (
                float(np.mean(risk[(actions == index) & legal])) if np.any((actions == index) & legal) else 0.0
            )
            for index in range(9)
        },
    }


def _risk_ranking_summary(arrays: dict[str, np.ndarray], indices: np.ndarray, predictions: np.ndarray) -> dict:
    if indices.size == 0 or predictions.size == 0:
        return {"available": False, "reason": "no validation samples"}
    actions = arrays["actions"][indices]
    risk = arrays["continuous_risk_target"][indices]
    legal = arrays["candidate_legal"][indices] > 0.5
    transition_ids = arrays["candidate_transition_id"][indices]
    raw_actions = arrays["candidate_raw_action"][indices]
    grouped: dict[int, list[int]] = {}
    for local_idx, transition_id in enumerate(transition_ids.tolist()):
        grouped.setdefault(int(transition_id), []).append(local_idx)

    top1_matches = []
    top3_matches = []
    raw_ranks = []
    oracle_best_risks = []
    model_best_label_risks = []
    model_minus_oracle = []
    oracle_hist: Counter[str] = Counter()
    model_hist: Counter[str] = Counter()
    skipped_incomplete = 0
    skipped_no_legal = 0
    skipped_raw_missing = 0

    for positions in grouped.values():
        group_actions = actions[positions]
        if set(int(item) for item in group_actions.tolist()) != set(range(9)):
            skipped_incomplete += 1
            continue
        legal_positions = [pos for pos in positions if bool(legal[pos])]
        if not legal_positions:
            skipped_no_legal += 1
            continue
        pred_order = sorted(legal_positions, key=lambda pos: (float(predictions[pos]), int(actions[pos])))
        label_order = sorted(legal_positions, key=lambda pos: (float(risk[pos]), int(actions[pos])))
        oracle_pos = label_order[0]
        model_pos = pred_order[0]
        oracle_best = float(risk[oracle_pos])
        model_label = float(risk[model_pos])
        top3 = pred_order[: min(3, len(pred_order))]
        top1_matches.append(float(model_label <= oracle_best + 1.0e-6))
        top3_matches.append(float(any(float(risk[pos]) <= oracle_best + 1.0e-6 for pos in top3)))
        oracle_best_risks.append(oracle_best)
        model_best_label_risks.append(model_label)
        model_minus_oracle.append(model_label - oracle_best)
        oracle_hist[str(int(actions[oracle_pos]))] += 1
        model_hist[str(int(actions[model_pos]))] += 1

        raw_action = int(raw_actions[positions[0]])
        rank = next((rank_idx + 1 for rank_idx, pos in enumerate(pred_order) if int(actions[pos]) == raw_action), None)
        if rank is None:
            skipped_raw_missing += 1
        else:
            raw_ranks.append(float(rank))

    evaluated = len(top1_matches)
    return {
        "available": evaluated > 0,
        "transition_group_count": int(len(grouped)),
        "evaluated_group_count": int(evaluated),
        "skipped_incomplete_group_count": int(skipped_incomplete),
        "skipped_no_legal_group_count": int(skipped_no_legal),
        "skipped_raw_missing_group_count": int(skipped_raw_missing),
        "top1_match_rate": float(np.mean(top1_matches)) if top1_matches else 0.0,
        "top3_match_rate": float(np.mean(top3_matches)) if top3_matches else 0.0,
        "raw_action_rank_mean": float(np.mean(raw_ranks)) if raw_ranks else 0.0,
        "oracle_best_action_histogram": dict(oracle_hist),
        "model_best_action_histogram": dict(model_hist),
        "mean_oracle_best_risk": float(np.mean(oracle_best_risks)) if oracle_best_risks else 0.0,
        "mean_model_best_label_risk": float(np.mean(model_best_label_risks)) if model_best_label_risks else 0.0,
        "mean_model_best_minus_oracle_risk": float(np.mean(model_minus_oracle)) if model_minus_oracle else 0.0,
    }


def _risk_validation_summary(pred: np.ndarray, target: np.ndarray, sample_weight: np.ndarray, legal: np.ndarray) -> dict:
    if pred.size == 0:
        return {"sample_count": 0}
    active = sample_weight > 0.0
    legal_mask = legal > 0.5
    legal_active = active & legal_mask
    pred_label = pred >= 0.5
    target_label = target >= 0.5
    weighted_abs = np.abs(pred - target)
    return {
        "sample_count": int(pred.shape[0]),
        "active_sample_count": int(np.sum(active)),
        "positive_risk_rate": float(np.mean(target[active])) if np.any(active) else 0.0,
        "predicted_positive_rate": float(np.mean(pred_label[active])) if np.any(active) else 0.0,
        "accuracy": float(np.mean(pred_label[active] == target_label[active])) if np.any(active) else 0.0,
        "legal_candidate_accuracy": (
            float(np.mean(pred_label[legal_active] == target_label[legal_active])) if np.any(legal_active) else 0.0
        ),
        "mean_abs_calibration_error": float(np.mean(weighted_abs[active])) if np.any(active) else 0.0,
        "prediction_distribution": _prediction_distribution(pred[active]),
        "boundary": _boundary_validation_summary(pred[active], target[active]),
    }


def _continuous_risk_coverage(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if values.size == 0:
        return {
            "sample_count": 0,
            "easy_safe_rate": 0.0,
            "boundary_rate": 0.0,
            "extreme_risk_rate": 0.0,
            "boundary_sample_count": 0,
        }
    boundary = (values >= 0.20) & (values < 0.80)
    return {
        "sample_count": int(values.size),
        "easy_safe_rate": float(np.mean(values < 0.20)),
        "boundary_rate": float(np.mean(boundary)),
        "extreme_risk_rate": float(np.mean(values >= 0.80)),
        "boundary_sample_count": int(np.sum(boundary)),
    }


def _prediction_distribution(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if values.size == 0:
        return {"sample_count": 0}
    return {
        "sample_count": int(values.size),
        "lt_0_01_rate": float(np.mean(values < 0.01)),
        "between_0_01_0_99_rate": float(np.mean((values >= 0.01) & (values <= 0.99))),
        "gt_0_99_rate": float(np.mean(values > 0.99)),
    }


def _boundary_validation_summary(pred: np.ndarray, target: np.ndarray) -> dict[str, float | int]:
    pred = np.asarray(pred, dtype=np.float32).reshape(-1)
    target = np.asarray(target, dtype=np.float32).reshape(-1)
    boundary = (target >= 0.20) & (target < 0.80)
    if not np.any(boundary):
        return {"sample_count": 0}
    pred = np.clip(pred[boundary], 1.0e-6, 1.0 - 1.0e-6)
    target = target[boundary]
    return {
        "sample_count": int(target.size),
        "ece": float(np.mean(np.abs(pred - target))),
        "brier": float(np.mean(np.square(pred - target))),
        "nll": float(np.mean(-(target * np.log(pred) + (1.0 - target) * np.log(1.0 - pred)))),
    }


def _probability_logit(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(probabilities, dtype=np.float32), 1.0e-6, 1.0 - 1.0e-6)
    return np.log(clipped / (1.0 - clipped))


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def _binary_nll(probabilities: np.ndarray, target: np.ndarray) -> float:
    clipped = np.clip(np.asarray(probabilities, dtype=np.float32), 1.0e-6, 1.0 - 1.0e-6)
    labels = np.asarray(target, dtype=np.float32)
    return float(np.mean(-(labels * np.log(clipped) + (1.0 - labels) * np.log(1.0 - clipped))))


def _binary_calibration_summary(
    pred: np.ndarray,
    target: np.ndarray,
    sample_weight: np.ndarray,
    legal: np.ndarray,
    *,
    bin_count: int = 10,
) -> dict[str, Any]:
    active = (np.asarray(sample_weight, dtype=np.float32) > 0.0) & (np.asarray(legal, dtype=np.float32) > 0.5)
    probabilities = np.asarray(pred, dtype=np.float32)[active]
    labels = np.asarray(target, dtype=np.float32)[active]
    if probabilities.size == 0:
        return {
            "sample_count": 0,
            "ece": 0.0,
            "brier": 0.0,
            "nll": 0.0,
            "reliability_bins": [],
        }
    bin_count = max(1, int(bin_count))
    edges = np.linspace(0.0, 1.0, bin_count + 1, dtype=np.float32)
    bins = []
    ece = 0.0
    for index in range(bin_count):
        left = float(edges[index])
        right = float(edges[index + 1])
        if index == bin_count - 1:
            mask = (probabilities >= left) & (probabilities <= right)
        else:
            mask = (probabilities >= left) & (probabilities < right)
        count = int(np.sum(mask))
        confidence = float(np.mean(probabilities[mask])) if count else 0.0
        accuracy = float(np.mean(labels[mask])) if count else 0.0
        gap = abs(confidence - accuracy)
        ece += (count / max(probabilities.size, 1)) * gap
        bins.append(
            {
                "bin": int(index),
                "left": left,
                "right": right,
                "count": count,
                "confidence": confidence,
                "empirical_risk": accuracy,
                "abs_gap": float(gap),
            }
        )
    return {
        "sample_count": int(probabilities.size),
        "positive_rate": float(np.mean(labels)),
        "predicted_mean": float(np.mean(probabilities)),
        "ece": float(ece),
        "brier": float(np.mean(np.square(probabilities - labels))),
        "nll": _binary_nll(probabilities, labels),
        "reliability_bins": bins,
    }


def _temperature_scaled_probabilities(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    temperature = max(float(temperature), 1.0e-6)
    return _sigmoid(_probability_logit(probabilities) / temperature).astype(np.float32)


def _temperature_scaling_diagnostics(
    pred: np.ndarray,
    target: np.ndarray,
    sample_weight: np.ndarray,
    legal: np.ndarray,
    cfg: Any,
) -> dict[str, Any]:
    calibration_cfg = cfg.risk_module.get("calibration", {})
    if not isinstance(calibration_cfg, dict):
        calibration_cfg = {}
    grid = calibration_cfg.get("temperature_grid")
    if not grid:
        grid = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0]
    active = (np.asarray(sample_weight, dtype=np.float32) > 0.0) & (np.asarray(legal, dtype=np.float32) > 0.5)
    probabilities = np.asarray(pred, dtype=np.float32)[active]
    labels = np.asarray(target, dtype=np.float32)[active]
    if probabilities.size == 0:
        return {"available": False, "reason": "no legal validation samples", "temperature": 1.0}
    candidates = []
    for raw_temperature in grid:
        temperature = max(float(raw_temperature), 1.0e-6)
        scaled = _temperature_scaled_probabilities(probabilities, temperature)
        candidates.append(
            {
                "temperature": temperature,
                "nll": _binary_nll(scaled, labels),
                "brier": float(np.mean(np.square(scaled - labels))),
            }
        )
    best = min(candidates, key=lambda item: (float(item["nll"]), float(item["temperature"])))
    scaled_all = _temperature_scaled_probabilities(np.asarray(pred, dtype=np.float32), float(best["temperature"]))
    return {
        "available": True,
        "temperature": float(best["temperature"]),
        "enabled_for_runtime": bool(calibration_cfg.get("temperature_scaling_enabled", False)),
        "candidate_count": int(len(candidates)),
        "candidates": candidates,
        "calibrated_summary": _binary_calibration_summary(
            scaled_all,
            target,
            sample_weight,
            legal,
            bin_count=int(calibration_cfg.get("reliability_bins", 10)),
        ),
    }


def _train_risk_module(
    cfg: Any,
    data: np.lib.npyio.NpzFile,
    stage_dir: Path,
    tb: TensorboardLogger,
    device: Any,
) -> dict:
    torch, DataLoader, TensorDataset = _require_torch()
    from safe_rl.risk.risk_module import RiskModule, risk_loss

    arrays = _risk_training_arrays(data)
    sample_weight = _configured_sample_weight(cfg, arrays)
    train_indices, val_indices = _split_risk_indices(
        arrays,
        float(cfg.risk_module.get("validation_split", 0.0)),
        int(cfg.run.seed),
    )

    def _dataset(indices: np.ndarray):
        return TensorDataset(
            torch.tensor(arrays["risk_features"][indices], dtype=torch.float32),
            torch.tensor(arrays["actions"][indices], dtype=torch.long),
            torch.tensor(arrays["continuous_risk_target"][indices], dtype=torch.float32),
            torch.tensor(arrays["risk_types"][indices], dtype=torch.float32),
            torch.tensor(sample_weight[indices], dtype=torch.float32),
            torch.tensor(arrays["candidate_legal"][indices], dtype=torch.float32),
        )

    train_dataset = _dataset(train_indices)
    val_dataset = _dataset(val_indices) if val_indices.size else None
    loader_kwargs = _loader_kwargs(cfg, device)
    loader = DataLoader(train_dataset, batch_size=int(cfg.risk_module.batch_size), shuffle=True, **loader_kwargs)
    val_loader = (
        DataLoader(val_dataset, batch_size=int(cfg.risk_module.batch_size), shuffle=False, **loader_kwargs)
        if val_dataset is not None
        else None
    )
    model = RiskModule(
        explicit_dim=int(cfg.risk_module.explicit_feature_dim),
        latent_dim=int(cfg.risk_module.latent_dim),
        action_embedding_dim=int(cfg.risk_module.action_embedding_dim),
        hidden_dim=int(cfg.risk_module.hidden_dim),
        risk_type_count=int(cfg.risk_module.get("risk_type_count", 6)),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.risk_module.learning_rate))
    weights = dict(cfg.risk_module.loss_weights)
    history: list[float] = []
    val_history: list[float] = []
    stage_log(
        "stage2",
        f"risk module samples={len(train_dataset)}, val_samples={len(val_dataset) if val_dataset is not None else 0}, "
        f"batch_size={cfg.risk_module.batch_size}, "
        f"pin_memory={loader_kwargs['pin_memory']}",
    )
    for epoch in progress_iter(range(int(cfg.risk_module.epochs)), desc="Stage2 risk epochs"):
        losses = []
        model.train()
        for batch in loader:
            batch_x, batch_actions, batch_y, batch_types, batch_weights, _batch_legal = _to_device(
                batch,
                device,
                non_blocking=bool(loader_kwargs["pin_memory"]),
            )
            output = model(batch_x, batch_actions)
            loss = risk_loss(
                output,
                {"risk_score": batch_y, "risk_types": batch_types, "sample_weight": batch_weights},
                {"risk": weights.get("risk", 1.0), "calibration": weights.get("calibration", 0.1)},
            )
            if bool(cfg.risk_module.ranking_loss_enabled) and batch_y.numel() > 1:
                active = batch_weights > 0.0
                scores = output["risk_score"][active]
                active_y = batch_y[active]
                pos = active_y.view(-1, 1)
                label_diff = pos - pos.t()
                score_diff = scores.view(-1, 1) - scores.view(1, -1)
                mask = label_diff > 0
                if scores.numel() > 1 and torch.any(mask):
                    rank_loss = torch.relu(0.05 - score_diff[mask]).mean()
                    loss = loss + weights.get("ranking", 0.5) * rank_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        epoch_loss = float(np.mean(losses)) if losses else 0.0
        history.append(epoch_loss)
        val_loss = 0.0
        if val_loader is not None:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for batch in val_loader:
                    batch_x, batch_actions, batch_y, batch_types, batch_weights, _batch_legal = _to_device(
                        batch,
                        device,
                        non_blocking=bool(loader_kwargs["pin_memory"]),
                    )
                    output = model(batch_x, batch_actions)
                    loss = risk_loss(
                        output,
                        {"risk_score": batch_y, "risk_types": batch_types, "sample_weight": batch_weights},
                        {"risk": weights.get("risk", 1.0), "calibration": weights.get("calibration", 0.1)},
                    )
                    val_losses.append(float(loss.detach().cpu()))
            val_loss = float(np.mean(val_losses)) if val_losses else 0.0
            val_history.append(val_loss)
        tb.scalar("stage2/risk_loss", epoch_loss, epoch)
        if val_loader is not None:
            tb.scalar("stage2/risk_val_loss", val_loss, epoch)
            stage_log(
                "stage2",
                f"risk epoch={epoch + 1}/{cfg.risk_module.epochs} loss={epoch_loss:.6f} val_loss={val_loss:.6f}",
            )
        else:
            stage_log("stage2", f"risk epoch={epoch + 1}/{cfg.risk_module.epochs} loss={epoch_loss:.6f}")
    checkpoint = stage_dir / "risk_module.pt"
    validation_summary = {"sample_count": 0}
    calibration_summary = {"raw": {"sample_count": 0}, "temperature_scaling": {"available": False}}
    ranking_summary = {"available": False, "reason": "no validation samples"}
    runtime_temperature = 1.0
    apply_runtime_temperature = False
    if val_indices.size:
        model.eval()
        val_x = torch.tensor(arrays["risk_features"][val_indices], dtype=torch.float32, device=device)
        val_actions = torch.tensor(arrays["actions"][val_indices], dtype=torch.long, device=device)
        with torch.no_grad():
            val_pred = model(val_x, val_actions)["risk_score"].detach().cpu().numpy()
        validation_summary = _risk_validation_summary(
            val_pred,
            arrays["continuous_risk_target"][val_indices],
            sample_weight[val_indices],
            arrays["candidate_legal"][val_indices],
        )
        calibration_cfg = cfg.risk_module.get("calibration", {})
        if not isinstance(calibration_cfg, dict):
            calibration_cfg = {}
        raw_calibration = _binary_calibration_summary(
            val_pred,
            arrays["continuous_risk_target"][val_indices],
            sample_weight[val_indices],
            arrays["candidate_legal"][val_indices],
            bin_count=int(calibration_cfg.get("reliability_bins", 10)),
        )
        temperature_report = _temperature_scaling_diagnostics(
            val_pred,
            arrays["continuous_risk_target"][val_indices],
            sample_weight[val_indices],
            arrays["candidate_legal"][val_indices],
            cfg,
        )
        runtime_temperature = float(temperature_report.get("temperature", 1.0))
        apply_runtime_temperature = bool(temperature_report.get("enabled_for_runtime", False))
        calibration_summary = {
            "raw": raw_calibration,
            "temperature_scaling": temperature_report,
        }
        ranking_summary = _risk_ranking_summary(arrays, val_indices, val_pred)
    training_summary = {
        "data": _risk_data_summary(arrays, sample_weight),
        "train_sample_count": int(train_indices.shape[0]),
        "validation_sample_count": int(val_indices.shape[0]),
        "validation": validation_summary,
        "calibration": calibration_summary,
        "ranking": ranking_summary,
        "config": {
            "legal_candidates_only_for_training": bool(cfg.risk_module.get("legal_candidates_only_for_training", True)),
            "illegal_candidate_sample_weight": float(cfg.risk_module.get("illegal_candidate_sample_weight", 0.0)),
            "positive_traffic_risk_weight": float(cfg.risk_module.get("positive_traffic_risk_weight", 1.0)),
            "merge_conflict_weight": float(cfg.risk_module.get("merge_conflict_weight", 1.0)),
            "validation_split": float(cfg.risk_module.get("validation_split", 0.0)),
        },
    }
    torch.save(
        {
            "model_state_dict": _cpu_state_dict(model),
            "loss_history": history,
            "val_loss_history": val_history,
            "training_summary": training_summary,
            "temperature": float(runtime_temperature),
            "apply_temperature": bool(apply_runtime_temperature),
        },
        checkpoint,
    )
    return {
        "risk_checkpoint": str(checkpoint),
        "risk_loss_history": history,
        "risk_val_loss_history": val_history,
        "risk_training_summary": training_summary,
        "risk_ranking_summary": ranking_summary,
        "risk_calibration_summary": calibration_summary,
    }


def _build_wcdt_batch(
    cfg: Any,
    data: np.lib.npyio.NpzFile,
    device: Any | None = None,
    indices: np.ndarray | None = None,
    *,
    shuffle: bool = True,
):
    torch, DataLoader, TensorDataset = _require_torch()
    history = data["agent_history"]
    future = data["agent_future"]
    mask = data["agent_mask"]
    if history.shape[0] == 0 or history.ndim != 4:
        return None
    sample_indices = np.asarray(indices if indices is not None else np.arange(history.shape[0]), dtype=np.int64)
    if sample_indices.size == 0:
        return None
    max_pred = int(cfg.prediction.max_pred_num)
    max_other = int(cfg.prediction.max_other_num)
    hist_steps = int(cfg.scenario.history_steps)
    horizon = future.shape[2]
    padded_future = np.zeros((sample_indices.shape[0], max_pred, 80, 5), dtype=np.float32)
    ego_future = np.zeros((sample_indices.shape[0], horizon, 5), dtype=np.float32)
    predicted_his = np.zeros((sample_indices.shape[0], max_pred, hist_steps, 5), dtype=np.float32)
    other_his = np.zeros((sample_indices.shape[0], max_other, hist_steps, 5), dtype=np.float32)
    predicted_mask = np.zeros((sample_indices.shape[0], max_pred), dtype=np.float32)
    other_mask = np.zeros((sample_indices.shape[0], max_other), dtype=np.float32)

    for output_idx, sample_idx in enumerate(sample_indices):
        ordered_agents = _ordered_prediction_indices(
            cfg,
            history[sample_idx],
            mask[sample_idx],
            None if "agent_lane_index" not in data else data["agent_lane_index"][sample_idx],
            None if "agent_edge_role" not in data else data["agent_edge_role"][sample_idx],
        )
        pred_agents = ordered_agents[:max_pred]
        leftover = [agent_idx for agent_idx in ordered_agents if agent_idx not in set(pred_agents)]
        leftover.sort(
            key=lambda agent_idx: (
                abs(float(history[sample_idx, agent_idx, -1, 0] - history[sample_idx, 0, -1, 0]))
                + abs(float(history[sample_idx, agent_idx, -1, 1] - history[sample_idx, 0, -1, 1])),
                int(agent_idx),
            )
        )
        other_agents = [0] + leftover
        ego_future[output_idx] = future[sample_idx, 0, :horizon]
        for row, agent_idx in enumerate(pred_agents[:max_pred]):
            predicted_his[output_idx, row] = history[sample_idx, agent_idx]
            padded_future[output_idx, row, :horizon] = future[sample_idx, agent_idx]
            if horizon < 80:
                padded_future[output_idx, row, horizon:] = future[sample_idx, agent_idx, -1]
            predicted_mask[output_idx, row] = mask[sample_idx, agent_idx]
        for row, agent_idx in enumerate(other_agents[:max_other]):
            other_his[output_idx, row] = history[sample_idx, agent_idx]
            other_mask[output_idx, row] = mask[sample_idx, agent_idx]

    predicted_feature = np.zeros((sample_indices.shape[0], max_pred, 7), dtype=np.float32)
    other_feature = np.zeros((sample_indices.shape[0], max_other, 7), dtype=np.float32)
    predicted_feature[..., 0] = 1.8
    predicted_feature[..., 1] = 4.8
    predicted_feature[..., 3] = 1.0
    other_feature[..., 0] = 1.8
    other_feature[..., 1] = 4.8
    other_feature[..., 3] = 1.0

    from safe_rl.prediction.sumo_wcdt_adapter import SumoWcDTAdapter

    lane_list = SumoWcDTAdapter(cfg).lane_list
    lane_batch = np.repeat(lane_list[None, ...], sample_indices.shape[0], axis=0)
    dataset = TensorDataset(
        torch.tensor(predicted_his, dtype=torch.float32),
        torch.tensor(padded_future, dtype=torch.float32),
        torch.tensor(predicted_mask, dtype=torch.float32),
        torch.tensor(ego_future, dtype=torch.float32),
        torch.tensor(predicted_feature, dtype=torch.float32),
        torch.tensor(other_his, dtype=torch.float32),
        torch.tensor(other_feature, dtype=torch.float32),
        torch.tensor(other_mask, dtype=torch.float32),
        torch.tensor(lane_batch, dtype=torch.float32),
    )
    loader_kwargs = _loader_kwargs(cfg, device or torch.device("cpu"))
    return DataLoader(dataset, batch_size=_wcdt_v1_batch_size(cfg), shuffle=shuffle, **loader_kwargs)


def _wcdt_v1_batch_size(cfg: Any) -> int:
    return int(cfg.prediction.get("wcdt_v1_batch_size", cfg.prediction.batch_size))


def _wcdt_v2_batch_size(cfg: Any) -> int:
    return int(cfg.prediction.get("wcdt_v2_batch_size", cfg.prediction.batch_size))


def _wcdt_v3_batch_size(cfg: Any) -> int:
    return int(cfg.prediction.get("wcdt_v3_batch_size", cfg.prediction.batch_size))


def _wcdt_v2_early_stopping_config(cfg: Any, *, has_validation: bool) -> dict[str, Any]:
    configured = dict(cfg.prediction.get("wcdt_v2_early_stopping", {}) or {})
    return {
        "enabled": bool(configured.get("enabled", True)) and bool(has_validation),
        "patience": max(1, int(configured.get("patience", 10))),
        "min_delta": max(0.0, float(configured.get("min_delta", 0.0001))),
        "warmup_epochs": max(0, int(configured.get("warmup_epochs", 5))),
        "disabled_reason": None if has_validation else "validation_unavailable",
    }


def _wcdt_v3_early_stopping_config(cfg: Any, *, has_validation: bool) -> dict[str, Any]:
    configured = dict(cfg.prediction.get("wcdt_v3_early_stopping", {}) or {})
    return {
        "enabled": bool(configured.get("enabled", True)) and bool(has_validation),
        "patience": max(1, int(configured.get("patience", 10))),
        "min_delta": max(0.0, float(configured.get("min_delta", 0.0001))),
        "warmup_epochs": max(0, int(configured.get("warmup_epochs", 5))),
        "disabled_reason": None if has_validation else "validation_unavailable",
    }


def _wcdt_v2_early_stopping_step(
    *,
    best_score: float | None,
    score: float,
    epoch: int,
    stale_epochs: int,
    config: dict[str, Any],
) -> tuple[bool, int, bool]:
    min_delta = float(config.get("min_delta", 0.0)) if bool(config.get("enabled", False)) else 0.0
    improved = best_score is None or float(score) < float(best_score) - min_delta
    stale_epochs = 0 if improved else int(stale_epochs) + 1
    should_stop = bool(
        config.get("enabled", False)
        and int(epoch) >= int(config.get("warmup_epochs", 0))
        and stale_epochs >= int(config.get("patience", 10))
    )
    return bool(improved), int(stale_epochs), should_stop


def _wcdt_v2_vs_cv_summary(cv_metrics: dict[str, Any] | None, v2_metrics: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(cv_metrics, dict) or not isinstance(v2_metrics, dict):
        return {"available": False}

    def _stat(metrics: dict[str, Any], name: str, stat: str = "mean") -> float | None:
        summary = metrics.get(name, {})
        return float(summary[stat]) if isinstance(summary, dict) and stat in summary else None

    metric_names = (
        "ade",
        "fde",
        "future_min_distance_abs_error",
        "target_lane_gap_abs_error",
        "target_lane_front_gap_abs_error",
        "target_lane_rear_gap_abs_error",
    )
    comparisons = {}
    for name in metric_names:
        cv_value = _stat(cv_metrics, name)
        v2_value = _stat(v2_metrics, name)
        comparisons[name] = {
            "cv": cv_value,
            "wcdt_v2": v2_value,
            "delta": float(v2_value - cv_value) if cv_value is not None and v2_value is not None else None,
        }
    return {
        "available": True,
        "metrics": comparisons,
        "uncertainty_std": _stat(v2_metrics, "uncertainty", "std"),
        "uncertainty_fde_correlation": float(v2_metrics.get("uncertainty_fde_correlation", 0.0)),
        "uncertainty_future_min_distance_abs_error_correlation": float(
            v2_metrics.get("uncertainty_future_min_distance_abs_error_correlation", 0.0)
        ),
    }


def _wcdt_data_dict(cfg: Any, pred_his, pred_future, pred_mask, pred_feat, other_his, other_feat, other_mask, lane_list, device):
    return {
        "predicted_feature": pred_feat,
        "other_his_pos": other_his[:, :, -1, :2],
        "other_his_traj_delt": other_his[:, :, 1:] - other_his[:, :, :-1],
        "other_feature": other_feat,
        "other_traj_mask": other_mask,
        "predicted_his_pos": pred_his[:, :, -1, :2],
        "predicted_his_traj_delt": pred_his[:, :, 1:] - pred_his[:, :, :-1],
        "predicted_his_traj": pred_his,
        "predicted_future_traj": pred_future,
        "predicted_traj_mask": pred_mask,
        "traffic_light": torch_zeros_like(
            pred_his,
            (pred_his.shape[0], int(cfg.prediction.max_traffic_light), int(cfg.scenario.history_steps)),
            device,
        ),
        "traffic_light_pos": torch_zeros_like(
            pred_his,
            (pred_his.shape[0], int(cfg.prediction.max_traffic_light), 2),
            device,
        ),
        "lane_list": lane_list,
    }


def torch_zeros_like(reference, shape: tuple[int, ...], device):
    return reference.new_zeros(shape).to(device)


def _select_best_mode(trajectories: np.ndarray, confidence: np.ndarray | None) -> np.ndarray:
    if trajectories.ndim == 4:
        return trajectories
    if trajectories.ndim != 5:
        raise ValueError(f"unexpected WcDT trajectory shape: {trajectories.shape}")
    if confidence is None:
        mode_idx = np.zeros((trajectories.shape[0], trajectories.shape[1]), dtype=np.int64)
    else:
        mode_idx = np.argmax(confidence, axis=-1)
    selected = np.zeros((trajectories.shape[0], trajectories.shape[1], trajectories.shape[3], trajectories.shape[4]), dtype=np.float32)
    for batch_idx in range(trajectories.shape[0]):
        for agent_idx in range(trajectories.shape[1]):
            selected[batch_idx, agent_idx] = trajectories[batch_idx, agent_idx, mode_idx[batch_idx, agent_idx]]
    return selected


def _future_min_distance(
    ego_future: np.ndarray,
    other_future: np.ndarray,
    other_mask: np.ndarray,
    future_valid_mask: np.ndarray | None = None,
    ego_future_valid_mask: np.ndarray | None = None,
) -> float:
    min_distance = 1.0e6
    horizon = min(ego_future.shape[0], other_future.shape[1])
    for agent_idx in range(other_future.shape[0]):
        if float(other_mask[agent_idx]) <= 0.0:
            continue
        valid = np.ones((horizon,), dtype=bool)
        if future_valid_mask is not None:
            valid &= np.asarray(future_valid_mask[agent_idx, :horizon]) > 0.5
        if ego_future_valid_mask is not None:
            valid &= np.asarray(ego_future_valid_mask[:horizon]) > 0.5
        if not np.any(valid):
            continue
        distances = np.linalg.norm(
            other_future[agent_idx, :horizon, :2][valid] - ego_future[:horizon, :2][valid],
            axis=-1,
        ) - 3.0
        min_distance = min(min_distance, float(np.min(np.maximum(0.0, distances))))
    return float(min_distance)


def _target_role_gap_abs_errors(
    ego_future: np.ndarray,
    pred_future: np.ndarray,
    actual_future: np.ndarray,
    other_mask: np.ndarray,
    role_ids: np.ndarray,
    future_valid_mask: np.ndarray | None = None,
    ego_future_valid_mask: np.ndarray | None = None,
) -> dict[str, float | None]:
    horizon = min(ego_future.shape[0], pred_future.shape[1], actual_future.shape[1])
    output: dict[str, float | None] = {}
    for name, role_id in (("target_lane_front_gap_abs_error", 0), ("target_lane_rear_gap_abs_error", 1)):
        selected = (np.asarray(other_mask) > 0.0) & (np.asarray(role_ids) == role_id)
        if not np.any(selected):
            output[name] = None
            continue
        pred_dx = pred_future[selected, :horizon, 0] - ego_future[None, :horizon, 0]
        actual_dx = actual_future[selected, :horizon, 0] - ego_future[None, :horizon, 0]
        valid = np.ones(pred_dx.shape, dtype=bool)
        if future_valid_mask is not None:
            valid &= np.asarray(future_valid_mask)[selected, :horizon] > 0.5
        if ego_future_valid_mask is not None:
            valid &= np.asarray(ego_future_valid_mask)[None, :horizon] > 0.5
        output[name] = float(np.mean(np.abs(pred_dx - actual_dx)[valid])) if np.any(valid) else None
    return output


def _target_lane_gap(
    ego_future: np.ndarray,
    other_future: np.ndarray,
    other_mask: np.ndarray,
    cfg: Any,
    future_valid_mask: np.ndarray | None = None,
    ego_future_valid_mask: np.ndarray | None = None,
) -> float:
    selected = np.asarray(other_mask) > 0.0
    trajectories = other_future[selected].copy()
    selected_valid = None
    if future_valid_mask is not None:
        selected_valid = np.asarray(future_valid_mask)[selected] > 0.5
        if ego_future_valid_mask is not None:
            selected_valid &= np.asarray(ego_future_valid_mask)[None, :] > 0.5
    return forecast_target_lane_gap_from_trajectories(
        ego_future,
        trajectories,
        cfg,
        default_gap=1.0e6,
        valid_mask=selected_valid,
        ego_valid_mask=ego_future_valid_mask,
    )


def _masked_trajectory_errors(
    pred_future: np.ndarray,
    actual_future: np.ndarray,
    actor_mask: np.ndarray,
    future_valid_mask: np.ndarray,
    ego_future_valid_mask: np.ndarray,
) -> tuple[float, float] | None:
    valid = (
        (np.asarray(actor_mask) > 0.5)[:, None]
        & (np.asarray(future_valid_mask) > 0.5)
        & (np.asarray(ego_future_valid_mask) > 0.5)[None, :]
    )
    if not np.any(valid):
        return None
    per_step = np.linalg.norm(pred_future[..., :2] - actual_future[..., :2], axis=-1)
    ade = float(np.mean(per_step[valid]))
    last_errors: list[float] = []
    for actor_idx in range(valid.shape[0]):
        indices = np.flatnonzero(valid[actor_idx])
        if indices.size:
            last_errors.append(float(per_step[actor_idx, indices[-1]]))
    return ade, float(np.mean(last_errors)) if last_errors else ade


def _wcdt_validation_metrics(cfg: Any, model: Any, loader: Any, device: Any, pin_memory: bool) -> dict[str, Any]:
    torch, _DataLoader, _TensorDataset = _require_torch()
    ade: list[float] = []
    fde: list[float] = []
    future_min_distance_errors: list[float] = []
    future_min_distance_abs_errors: list[float] = []
    target_gap_errors: list[float] = []
    target_gap_abs_errors: list[float] = []
    uncertainty_values: list[float] = []
    confidence_values: list[float] = []
    confidence_fde_values: list[float] = []
    val_losses: list[float] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            pred_his, pred_future, pred_mask, ego_future, pred_feat, other_his, other_feat, other_mask, lane_list = _to_device(
                batch,
                device,
                non_blocking=pin_memory,
            )
            data_dict = _wcdt_data_dict(
                cfg,
                pred_his,
                pred_future,
                pred_mask,
                pred_feat,
                other_his,
                other_feat,
                other_mask,
                lane_list,
                device,
            )
            diffusion_loss, traj_loss, confidence_loss, _min_loss_traj = model(data_dict)
            val_losses.append(float((diffusion_loss.mean() + traj_loss.mean() + confidence_loss.mean()).detach().cpu()))
            horizon = min(int(cfg.forecast_features.get("horizon_steps", cfg.scenario.forecast_horizon_steps)), ego_future.shape[1])
            output = model.predict(data_dict, horizon_steps=horizon)
            traj = output["future_trajectories"].detach().cpu().numpy()
            confidence = output.get("mode_confidence")
            confidence_np = confidence.detach().cpu().numpy() if confidence is not None else None
            selected = _select_best_mode(traj, confidence_np)
            actual_future = pred_future[:, :, :horizon].detach().cpu().numpy()
            ego_future_np = ego_future[:, :horizon].detach().cpu().numpy()
            pred_mask_np = pred_mask.detach().cpu().numpy()
            uncertainty = output.get("uncertainty")
            uncertainty_np = uncertainty.detach().cpu().numpy() if uncertainty is not None else np.zeros(pred_mask_np.shape)
            max_confidence_np = (
                np.max(confidence_np, axis=-1) if confidence_np is not None else np.ones(pred_mask_np.shape, dtype=np.float32)
            )
            for row in range(selected.shape[0]):
                valid = pred_mask_np[row] > 0.0
                if not np.any(valid):
                    continue
                diff = selected[row, valid, :, :2] - actual_future[row, valid, :, :2]
                per_step = np.linalg.norm(diff, axis=-1)
                row_ade = float(np.mean(per_step))
                row_fde = float(np.mean(per_step[:, -1]))
                ade.append(row_ade)
                fde.append(row_fde)
                confidence_values.append(float(np.mean(max_confidence_np[row][valid])))
                confidence_fde_values.append(row_fde)
                uncertainty_values.append(float(np.mean(uncertainty_np[row][valid])))
                pred_min = _future_min_distance(ego_future_np[row], selected[row], pred_mask_np[row])
                actual_min = _future_min_distance(ego_future_np[row], actual_future[row], pred_mask_np[row])
                future_min_distance_errors.append(float(pred_min - actual_min))
                future_min_distance_abs_errors.append(abs(float(pred_min - actual_min)))
                pred_gap = _target_lane_gap(ego_future_np[row], selected[row], pred_mask_np[row], cfg)
                actual_gap = _target_lane_gap(ego_future_np[row], actual_future[row], pred_mask_np[row], cfg)
                if pred_gap < 1.0e6 and actual_gap < 1.0e6:
                    target_gap_errors.append(float(pred_gap - actual_gap))
                    target_gap_abs_errors.append(abs(float(pred_gap - actual_gap)))
    return {
        "sample_count": int(len(ade)),
        "loss": float(np.mean(val_losses)) if val_losses else 0.0,
        "ade": _summary(ade),
        "fde": _summary(fde),
        "future_min_distance_error": _summary(future_min_distance_errors),
        "future_min_distance_abs_error": _summary(future_min_distance_abs_errors),
        "target_lane_gap_error": _summary(target_gap_errors),
        "target_lane_gap_abs_error": _summary(target_gap_abs_errors),
        "uncertainty": _summary(uncertainty_values),
        "confidence": _summary(confidence_values),
        "confidence_fde_correlation": _correlation(confidence_values, confidence_fde_values),
    }


def _train_wcdt_predictor(
    cfg: Any,
    data: np.lib.npyio.NpzFile,
    stage_dir: Path,
    tb: TensorboardLogger,
    device: Any,
) -> dict:
    sample_count = int(data["agent_history"].shape[0]) if _has_key(data, "agent_history") else 0
    train_indices, val_indices = _split_indices(
        sample_count,
        float(cfg.prediction.get("validation_split", 0.15)),
        int(cfg.run.seed),
    )
    loader = _build_wcdt_batch(cfg, data, device=device, indices=train_indices, shuffle=True)
    if loader is None:
        return {"prediction_skipped": True, "prediction_skip_reason": "no trajectory samples in Stage1 buffer"}
    val_loader = _build_wcdt_batch(cfg, data, device=device, indices=val_indices, shuffle=False) if val_indices.size else None
    torch, _DataLoader, _TensorDataset = _require_torch()
    from net_works import BackBone
    from utils import MathUtil

    betas = MathUtil.generate_linear_schedule(50, 1e-4, 0.008)
    model = BackBone(betas).to(device)
    if cfg.prediction.checkpoint:
        state = torch.load(cfg.prediction.checkpoint, map_location=device)
        model.load_state_dict(state.get("model_state_dict", state), strict=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.prediction.learning_rate))
    loss_history: list[float] = []
    val_loss_history: list[float] = []
    validation_history: list[dict[str, Any]] = []
    best_payload: dict[str, Any] | None = None
    best_score: float | None = None
    best_epoch = 0
    pin_memory = bool(getattr(loader, "pin_memory", False))
    stage_log(
        "stage2",
        f"WcDT predictor train_samples={train_indices.shape[0]}, val_samples={val_indices.shape[0]}, "
        f"batches={len(loader)}, batch_size={_wcdt_v1_batch_size(cfg)}, pin_memory={pin_memory}",
    )
    for epoch in progress_iter(range(int(cfg.prediction.epochs)), desc="Stage2 prediction epochs"):
        losses = []
        model.train()
        for batch in loader:
            pred_his, pred_future, pred_mask, _ego_future, pred_feat, other_his, other_feat, other_mask, lane_list = _to_device(
                batch,
                device,
                non_blocking=pin_memory,
            )
            data_dict = _wcdt_data_dict(
                cfg,
                pred_his,
                pred_future,
                pred_mask,
                pred_feat,
                other_his,
                other_feat,
                other_mask,
                lane_list,
                device,
            )
            diffusion_loss, traj_loss, confidence_loss, _min_loss_traj = model(data_dict)
            loss = diffusion_loss.mean() + traj_loss.mean() + confidence_loss.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        epoch_loss = float(np.mean(losses)) if losses else 0.0
        loss_history.append(epoch_loss)
        tb.scalar("stage2/prediction_loss", epoch_loss, epoch)
        validation_metrics: dict[str, Any] | None = None
        val_loss = 0.0
        val_score = epoch_loss
        if val_loader is not None:
            validation_metrics = _wcdt_validation_metrics(cfg, model, val_loader, device, pin_memory)
            validation_metrics["epoch"] = int(epoch + 1)
            val_loss = float(validation_metrics.get("loss", 0.0))
            val_loss_history.append(val_loss)
            val_score = _prediction_val_score(validation_metrics, cfg)
            validation_metrics["val_score"] = float(val_score)
            validation_history.append(validation_metrics)
            tb.scalar("stage2/prediction_val_loss", val_loss, epoch)
            tb.scalar("stage2/prediction_val_score", val_score, epoch)
            tb.scalar("stage2/prediction_val_ade", float(validation_metrics["ade"].get("mean", 0.0)), epoch)
            tb.scalar("stage2/prediction_val_fde", float(validation_metrics["fde"].get("mean", 0.0)), epoch)
        if best_score is None or val_score < best_score:
            best_score = float(val_score)
            best_epoch = int(epoch + 1)
            best_payload = {
                "model_state_dict": _cpu_state_dict(model),
                "loss_history": list(loss_history),
                "val_loss_history": list(val_loss_history),
                "validation_history": list(validation_history),
                "best_epoch": best_epoch,
                "best_val_score": float(best_score),
                "best_metric": "val_score" if val_loader is not None else "train_loss",
                "train_sample_count": int(train_indices.shape[0]),
                "validation_sample_count": int(val_indices.shape[0]),
            }
        if validation_metrics is not None:
            stage_log(
                "stage2",
                f"prediction epoch={epoch + 1}/{cfg.prediction.epochs} loss={epoch_loss:.6f} "
                f"val_loss={val_loss:.6f} val_score={val_score:.6f} "
                f"ade={float(validation_metrics['ade'].get('mean', 0.0)):.3f} "
                f"fde={float(validation_metrics['fde'].get('mean', 0.0)):.3f}",
            )
        else:
            stage_log("stage2", f"prediction epoch={epoch + 1}/{cfg.prediction.epochs} loss={epoch_loss:.6f}")
    checkpoint = stage_dir / "wcdt_predictor.pt"
    best_checkpoint = stage_dir / "wcdt_predictor_best.pt"
    if best_payload is None:
        best_payload = {
            "model_state_dict": _cpu_state_dict(model),
            "loss_history": list(loss_history),
            "val_loss_history": list(val_loss_history),
            "validation_history": list(validation_history),
            "best_epoch": int(len(loss_history)),
            "best_val_score": float(loss_history[-1]) if loss_history else 0.0,
            "best_metric": "train_loss",
            "train_sample_count": int(train_indices.shape[0]),
            "validation_sample_count": int(val_indices.shape[0]),
        }
    torch.save(best_payload, best_checkpoint)
    torch.save(best_payload, checkpoint)
    return {
        "prediction_checkpoint": str(checkpoint),
        "prediction_best_checkpoint": str(best_checkpoint),
        "prediction_loss_history": loss_history,
        "prediction_val_loss_history": val_loss_history,
        "prediction_validation_history": validation_history,
        "prediction_best_epoch": int(best_payload["best_epoch"]),
        "prediction_best_val_score": float(best_payload["best_val_score"]),
    }


def _build_wcdt_v2_loader(
    cfg: Any,
    data: np.lib.npyio.NpzFile,
    indices: np.ndarray,
    device: Any,
    *,
    shuffle: bool,
):
    torch, DataLoader, TensorDataset = _require_torch()
    from safe_rl.prediction.wcdt_v2_predictor import build_v2_numpy_batch

    if not _has_key(data, "agent_history") or data["agent_history"].shape[0] == 0:
        return None
    if indices.size == 0:
        return None
    _require_trajectory_schema_v2(data, "WcDT v2")
    batch = build_v2_numpy_batch(
        cfg,
        data["agent_history"],
        data["agent_future"],
        data["agent_mask"],
        indices,
        lane_indices=data["agent_lane_index"] if "agent_lane_index" in data else None,
        edge_roles=data["agent_edge_role"] if "agent_edge_role" in data else None,
        future_valid_mask=data["agent_future_valid_mask"],
    )
    dataset = TensorDataset(
        torch.tensor(batch["features"], dtype=torch.float32),
        torch.tensor(batch["baseline"], dtype=torch.float32),
        torch.tensor(batch["target"], dtype=torch.float32),
        torch.tensor(batch["mask"], dtype=torch.float32),
        torch.tensor(batch["role_ids"], dtype=torch.long),
        torch.tensor(batch["ego_future"], dtype=torch.float32),
        torch.tensor(batch["future_valid_mask"], dtype=torch.float32),
        torch.tensor(batch["ego_future_valid_mask"], dtype=torch.float32),
    )
    return DataLoader(dataset, batch_size=_wcdt_v2_batch_size(cfg), shuffle=shuffle, **_loader_kwargs(cfg, device))


def _wcdt_v2_validation_metrics(cfg: Any, models: list[Any], loader: Any, device: Any, pin_memory: bool) -> dict[str, Any]:
    torch, _DataLoader, _TensorDataset = _require_torch()
    from safe_rl.prediction.wcdt_v2_predictor import ensemble_predict, tensorize_batch, v2_loss

    ade: list[float] = []
    fde: list[float] = []
    future_min_distance_errors: list[float] = []
    future_min_distance_abs_errors: list[float] = []
    target_gap_errors: list[float] = []
    target_gap_abs_errors: list[float] = []
    target_front_gap_abs_errors: list[float] = []
    target_rear_gap_abs_errors: list[float] = []
    uncertainty_values: list[float] = []
    uncertainty_fde_values: list[float] = []
    uncertainty_min_distance_error_values: list[float] = []
    val_losses: list[float] = []
    component_losses: dict[str, list[float]] = {}
    for model in models:
        model.eval()
    with torch.no_grad():
        for features, baseline, target, mask, role_ids, ego_future, future_valid_mask, ego_future_valid_mask in loader:
            batch = {
                "features": features.to(device, non_blocking=pin_memory),
                "baseline": baseline.to(device, non_blocking=pin_memory),
                "target": target.to(device, non_blocking=pin_memory),
                "mask": mask.to(device, non_blocking=pin_memory),
                "role_ids": role_ids.to(device, non_blocking=pin_memory),
                "ego_future": ego_future.to(device, non_blocking=pin_memory),
                "future_valid_mask": future_valid_mask.to(device, non_blocking=pin_memory),
                "ego_future_valid_mask": ego_future_valid_mask.to(device, non_blocking=pin_memory),
            }
            pred, uncertainty = ensemble_predict(models, batch)
            loss, components = v2_loss(
                pred,
                batch["target"],
                batch["mask"],
                batch["ego_future"],
                batch["role_ids"],
                dict(cfg.prediction.get("wcdt_v2_loss_weights", {})),
                future_valid_mask=batch["future_valid_mask"],
                ego_future_valid_mask=batch["ego_future_valid_mask"],
            )
            val_losses.append(float(loss.detach().cpu()))
            for name, value in components.items():
                component_losses.setdefault(name, []).append(float(value.detach().cpu()))
            pred_np = pred.detach().cpu().numpy()
            target_np = batch["target"].detach().cpu().numpy()
            mask_np = batch["mask"].detach().cpu().numpy()
            ego_np = batch["ego_future"].detach().cpu().numpy()
            uncertainty_np = uncertainty.detach().cpu().numpy()
            role_ids_np = batch["role_ids"].detach().cpu().numpy()
            future_valid_np = batch["future_valid_mask"].detach().cpu().numpy()
            ego_future_valid_np = batch["ego_future_valid_mask"].detach().cpu().numpy()
            for row in range(pred_np.shape[0]):
                errors = _masked_trajectory_errors(
                    pred_np[row],
                    target_np[row],
                    mask_np[row],
                    future_valid_np[row],
                    ego_future_valid_np[row],
                )
                if errors is None:
                    continue
                row_ade, row_fde = errors
                ade.append(row_ade)
                fde.append(row_fde)
                uncertainty_values.append(float(uncertainty_np[row]))
                uncertainty_fde_values.append(row_fde)
                pred_min = _future_min_distance(
                    ego_np[row], pred_np[row], mask_np[row], future_valid_np[row], ego_future_valid_np[row]
                )
                actual_min = _future_min_distance(
                    ego_np[row], target_np[row], mask_np[row], future_valid_np[row], ego_future_valid_np[row]
                )
                future_min_distance_errors.append(float(pred_min - actual_min))
                min_distance_abs_error = abs(float(pred_min - actual_min))
                future_min_distance_abs_errors.append(min_distance_abs_error)
                uncertainty_min_distance_error_values.append(min_distance_abs_error)
                role_gap_errors = _target_role_gap_abs_errors(
                    ego_np[row],
                    pred_np[row],
                    target_np[row],
                    mask_np[row],
                    role_ids_np[row],
                    future_valid_np[row],
                    ego_future_valid_np[row],
                )
                if role_gap_errors["target_lane_front_gap_abs_error"] is not None:
                    target_front_gap_abs_errors.append(float(role_gap_errors["target_lane_front_gap_abs_error"]))
                if role_gap_errors["target_lane_rear_gap_abs_error"] is not None:
                    target_rear_gap_abs_errors.append(float(role_gap_errors["target_lane_rear_gap_abs_error"]))
                pred_gap = _target_lane_gap(
                    ego_np[row], pred_np[row], mask_np[row], cfg, future_valid_np[row], ego_future_valid_np[row]
                )
                actual_gap = _target_lane_gap(
                    ego_np[row], target_np[row], mask_np[row], cfg, future_valid_np[row], ego_future_valid_np[row]
                )
                if pred_gap < 1.0e6 and actual_gap < 1.0e6:
                    target_gap_errors.append(float(pred_gap - actual_gap))
                    target_gap_abs_errors.append(abs(float(pred_gap - actual_gap)))
    return {
        "sample_count": int(len(ade)),
        "loss": float(np.mean(val_losses)) if val_losses else 0.0,
        "ade": _summary(ade),
        "fde": _summary(fde),
        "future_min_distance_error": _summary(future_min_distance_errors),
        "future_min_distance_abs_error": _summary(future_min_distance_abs_errors),
        "target_lane_gap_error": _summary(target_gap_errors),
        "target_lane_gap_abs_error": _summary(target_gap_abs_errors),
        "target_lane_front_gap_abs_error": _summary(target_front_gap_abs_errors),
        "target_lane_rear_gap_abs_error": _summary(target_rear_gap_abs_errors),
        "uncertainty": _summary(uncertainty_values),
        "uncertainty_fde_correlation": _correlation(uncertainty_values, uncertainty_fde_values),
        "uncertainty_future_min_distance_abs_error_correlation": _correlation(
            uncertainty_values,
            uncertainty_min_distance_error_values,
        ),
        "loss_components": {name: _summary(values) for name, values in component_losses.items()},
    }


def _wcdt_v2_cv_baseline_metrics(cfg: Any, loader: Any, device: Any, pin_memory: bool) -> dict[str, Any]:
    torch, _DataLoader, _TensorDataset = _require_torch()
    from safe_rl.prediction.wcdt_v2_predictor import WcDTV2ResidualPredictor

    horizon = int(cfg.prediction.get("wcdt_v2_horizon_steps", cfg.scenario.forecast_horizon_steps))
    dummy = WcDTV2ResidualPredictor(horizon_steps=horizon)
    models: list[Any] = []
    ade: list[float] = []
    fde: list[float] = []
    future_min_distance_abs_errors: list[float] = []
    target_gap_abs_errors: list[float] = []
    target_front_gap_abs_errors: list[float] = []
    target_rear_gap_abs_errors: list[float] = []
    for features, baseline, target, mask, role_ids, ego_future, future_valid_mask, ego_future_valid_mask in loader:
        pred_np = baseline.numpy()
        target_np = target.numpy()
        mask_np = mask.numpy()
        ego_np = ego_future.numpy()
        role_ids_np = role_ids.numpy()
        future_valid_np = future_valid_mask.numpy()
        ego_future_valid_np = ego_future_valid_mask.numpy()
        for row in range(pred_np.shape[0]):
            errors = _masked_trajectory_errors(
                pred_np[row], target_np[row], mask_np[row], future_valid_np[row], ego_future_valid_np[row]
            )
            if errors is None:
                continue
            row_ade, row_fde = errors
            ade.append(row_ade)
            fde.append(row_fde)
            pred_min = _future_min_distance(
                ego_np[row], pred_np[row], mask_np[row], future_valid_np[row], ego_future_valid_np[row]
            )
            actual_min = _future_min_distance(
                ego_np[row], target_np[row], mask_np[row], future_valid_np[row], ego_future_valid_np[row]
            )
            future_min_distance_abs_errors.append(abs(float(pred_min - actual_min)))
            role_gap_errors = _target_role_gap_abs_errors(
                ego_np[row],
                pred_np[row],
                target_np[row],
                mask_np[row],
                role_ids_np[row],
                future_valid_np[row],
                ego_future_valid_np[row],
            )
            if role_gap_errors["target_lane_front_gap_abs_error"] is not None:
                target_front_gap_abs_errors.append(float(role_gap_errors["target_lane_front_gap_abs_error"]))
            if role_gap_errors["target_lane_rear_gap_abs_error"] is not None:
                target_rear_gap_abs_errors.append(float(role_gap_errors["target_lane_rear_gap_abs_error"]))
            pred_gap = _target_lane_gap(
                ego_np[row], pred_np[row], mask_np[row], cfg, future_valid_np[row], ego_future_valid_np[row]
            )
            actual_gap = _target_lane_gap(
                ego_np[row], target_np[row], mask_np[row], cfg, future_valid_np[row], ego_future_valid_np[row]
            )
            if pred_gap < 1.0e6 and actual_gap < 1.0e6:
                target_gap_abs_errors.append(abs(float(pred_gap - actual_gap)))
    return {
        "ade": _summary(ade),
        "fde": _summary(fde),
        "future_min_distance_abs_error": _summary(future_min_distance_abs_errors),
        "target_lane_gap_abs_error": _summary(target_gap_abs_errors),
        "target_lane_front_gap_abs_error": _summary(target_front_gap_abs_errors),
        "target_lane_rear_gap_abs_error": _summary(target_rear_gap_abs_errors),
    }


def _train_wcdt_v2_predictor(
    cfg: Any,
    data: np.lib.npyio.NpzFile,
    stage_dir: Path,
    tb: TensorboardLogger,
    device: Any,
) -> dict[str, Any]:
    sample_count = int(data["agent_history"].shape[0]) if _has_key(data, "agent_history") else 0
    train_indices, val_indices = _split_indices(
        sample_count,
        float(cfg.prediction.get("validation_split", 0.15)),
        int(cfg.run.seed),
    )
    train_loader = _build_wcdt_v2_loader(cfg, data, train_indices, device, shuffle=True)
    val_loader = _build_wcdt_v2_loader(cfg, data, val_indices, device, shuffle=False) if val_indices.size else None
    if train_loader is None:
        return {"wcdt_v2_prediction_skipped": True, "wcdt_v2_prediction_skip_reason": "no trajectory samples"}
    torch, _DataLoader, _TensorDataset = _require_torch()
    from safe_rl.prediction.wcdt_v2_predictor import (
        ARCHITECTURE_VERSION,
        INPUT_DIM,
        LOSS_VERSION,
        WcDTV2ResidualPredictor,
        v2_loss,
    )

    ensemble_size = int(cfg.prediction.get("wcdt_v2_ensemble_size", 3))
    hidden_dim = int(cfg.prediction.get("wcdt_v2_hidden_dim", 128))
    horizon = int(cfg.prediction.get("wcdt_v2_horizon_steps", cfg.scenario.forecast_horizon_steps))
    epochs = int(cfg.prediction.get("wcdt_v2_epochs", cfg.prediction.epochs))
    loss_weights = dict(cfg.prediction.get("wcdt_v2_loss_weights", {}))
    early_stopping = _wcdt_v2_early_stopping_config(cfg, has_validation=val_loader is not None)
    model_states: list[dict[str, Any]] = []
    member_histories: list[dict[str, Any]] = []
    validation_history: list[dict[str, Any]] = []
    cv_baseline = _wcdt_v2_cv_baseline_metrics(cfg, val_loader, device, False) if val_loader is not None else None
    pin_memory = bool(getattr(train_loader, "pin_memory", False))
    stage_log(
        "stage2",
        f"WcDT v2 train_samples={train_indices.shape[0]}, val_samples={val_indices.shape[0]}, "
        f"ensemble={ensemble_size}, epochs={epochs}, batch_size={_wcdt_v2_batch_size(cfg)}",
    )
    for member_idx in range(ensemble_size):
        torch.manual_seed(int(cfg.run.seed) + 1000 + member_idx)
        model = WcDTV2ResidualPredictor(INPUT_DIM, horizon, hidden_dim).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.prediction.get("wcdt_v2_learning_rate", cfg.prediction.learning_rate)))
        losses: list[float] = []
        loss_component_history: list[dict[str, float]] = []
        best_state: dict[str, Any] | None = None
        best_score: float | None = None
        best_epoch = 0
        stale_epochs = 0
        stopped_early = False
        early_stopping_reason: str | None = None
        member_validation: list[dict[str, Any]] = []
        for epoch in progress_iter(range(epochs), desc=f"Stage2 WcDT v2 member {member_idx + 1}/{ensemble_size}"):
            epoch_losses = []
            epoch_components: dict[str, list[float]] = {}
            model.train()
            for (
                features,
                baseline,
                target,
                mask,
                role_ids,
                ego_future,
                future_valid_mask,
                ego_future_valid_mask,
            ) in train_loader:
                features = features.to(device, non_blocking=pin_memory)
                baseline = baseline.to(device, non_blocking=pin_memory)
                target = target.to(device, non_blocking=pin_memory)
                mask = mask.to(device, non_blocking=pin_memory)
                role_ids = role_ids.to(device, non_blocking=pin_memory)
                ego_future = ego_future.to(device, non_blocking=pin_memory)
                future_valid_mask = future_valid_mask.to(device, non_blocking=pin_memory)
                ego_future_valid_mask = ego_future_valid_mask.to(device, non_blocking=pin_memory)
                pred = model(features, baseline)
                loss, components = v2_loss(
                    pred,
                    target,
                    mask,
                    ego_future,
                    role_ids,
                    loss_weights,
                    future_valid_mask=future_valid_mask,
                    ego_future_valid_mask=ego_future_valid_mask,
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_losses.append(float(loss.detach().cpu()))
                for name, value in components.items():
                    epoch_components.setdefault(name, []).append(float(value.detach().cpu()))
            epoch_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
            losses.append(epoch_loss)
            component_summary = {
                name: float(np.mean(values)) if values else 0.0
                for name, values in epoch_components.items()
            }
            loss_component_history.append(component_summary)
            val_score = epoch_loss
            validation_metrics: dict[str, Any] | None = None
            if val_loader is not None:
                validation_metrics = _wcdt_v2_validation_metrics(cfg, [model], val_loader, device, pin_memory)
                validation_metrics["epoch"] = int(epoch + 1)
                validation_metrics["member"] = int(member_idx)
                val_score = _prediction_val_score(validation_metrics, cfg)
                validation_metrics["val_score"] = float(val_score)
                member_validation.append(validation_metrics)
                tb.scalar(f"stage2/wcdt_v2_member_{member_idx}/val_score", val_score, epoch)
            tb.scalar(f"stage2/wcdt_v2_member_{member_idx}/loss", epoch_loss, epoch)
            for name, value in component_summary.items():
                tb.scalar(f"stage2/wcdt_v2_member_{member_idx}/loss_{name}", value, epoch)
            improved, stale_epochs, should_stop = _wcdt_v2_early_stopping_step(
                best_score=best_score,
                score=val_score,
                epoch=epoch + 1,
                stale_epochs=stale_epochs,
                config=early_stopping,
            )
            if improved:
                best_score = float(val_score)
                best_epoch = int(epoch + 1)
                best_state = _cpu_state_dict(model)
            if validation_metrics is not None:
                stage_log(
                    "stage2",
                    f"wcdt_v2 member={member_idx + 1}/{ensemble_size} epoch={epoch + 1}/{epochs} "
                    f"loss={epoch_loss:.6f} val_score={val_score:.6f} "
                    f"ade={float(validation_metrics['ade'].get('mean', 0.0)):.3f} "
                    f"fde={float(validation_metrics['fde'].get('mean', 0.0)):.3f} "
                    f"minD={float(validation_metrics['future_min_distance_abs_error'].get('mean', 0.0)):.3f} "
                    f"front_gap={float(validation_metrics['target_lane_front_gap_abs_error'].get('mean', 0.0)):.3f} "
                    f"rear_gap={float(validation_metrics['target_lane_rear_gap_abs_error'].get('mean', 0.0)):.3f}",
                )
            else:
                stage_log(
                    "stage2",
                    f"wcdt_v2 member={member_idx + 1}/{ensemble_size} epoch={epoch + 1}/{epochs} "
                    f"loss={epoch_loss:.6f}",
                )
            if should_stop:
                stopped_early = True
                early_stopping_reason = (
                    f"no val_score improvement >= {early_stopping['min_delta']} "
                    f"for {early_stopping['patience']} epochs"
                )
                stage_log(
                    "stage2",
                    f"wcdt_v2 member={member_idx + 1}/{ensemble_size} early_stop epoch={epoch + 1} "
                    f"best_epoch={best_epoch} best_val_score={float(best_score):.6f}",
                )
                break
        if best_state is None:
            best_state = _cpu_state_dict(model)
        model_states.append(best_state)
        member_histories.append(
            {
                "member": int(member_idx),
                "loss_history": losses,
                "loss_component_history": loss_component_history,
                "validation_history": member_validation,
                "trained_epochs": int(len(losses)),
                "best_epoch": int(best_epoch),
                "best_val_score": float(best_score if best_score is not None else (losses[-1] if losses else 0.0)),
                "stopped_early": bool(stopped_early),
                "early_stopping_reason": early_stopping_reason,
            }
        )
    ensemble_models = []
    for state in model_states:
        model = WcDTV2ResidualPredictor(INPUT_DIM, horizon, hidden_dim).to(device)
        model.load_state_dict(state)
        model.eval()
        ensemble_models.append(model)
    ensemble_validation = (
        _wcdt_v2_validation_metrics(cfg, ensemble_models, val_loader, device, pin_memory)
        if val_loader is not None
        else {"sample_count": 0}
    )
    if val_loader is not None:
        ensemble_validation["val_score"] = _prediction_val_score(ensemble_validation, cfg)
        validation_history.append(ensemble_validation)
    vs_cv_summary = _wcdt_v2_vs_cv_summary(cv_baseline, ensemble_validation)
    checkpoint = stage_dir / "wcdt_v2_predictor.pt"
    best_checkpoint = stage_dir / "wcdt_v2_predictor_best.pt"
    payload = {
        "model_state_dicts": model_states,
        "member_histories": member_histories,
        "validation_history": validation_history,
        "ensemble_validation": ensemble_validation,
        "cv_baseline_validation": cv_baseline,
        "wcdt_v2_vs_cv_summary": vs_cv_summary,
        "architecture_version": ARCHITECTURE_VERSION,
        "loss_version": LOSS_VERSION,
        "trajectory_schema_version": _trajectory_schema_version(data),
        "horizon_steps": int(horizon),
        "history_steps": int(cfg.scenario.history_steps),
        "input_dim": int(INPUT_DIM),
        "hidden_dim": int(hidden_dim),
        "ensemble_size": int(ensemble_size),
        "loss_weights": loss_weights,
        "early_stopping_config": early_stopping,
        "best_metric": "ensemble_val_score",
        "best_val_score": float(ensemble_validation.get("val_score", 0.0)) if isinstance(ensemble_validation, dict) else 0.0,
        "train_sample_count": int(train_indices.shape[0]),
        "validation_sample_count": int(val_indices.shape[0]),
    }
    torch.save(payload, best_checkpoint)
    torch.save(payload, checkpoint)
    return {
        "wcdt_v2_prediction_checkpoint": str(checkpoint),
        "wcdt_v2_prediction_best_checkpoint": str(best_checkpoint),
        "wcdt_v2_member_histories": member_histories,
        "wcdt_v2_loss_component_history": [
            {"member": int(item["member"]), "history": item["loss_component_history"]}
            for item in member_histories
        ],
        "wcdt_v2_prediction_validation_history": validation_history,
        "wcdt_v2_prediction_cv_baseline_validation": cv_baseline,
        "wcdt_v2_prediction_ensemble_validation": ensemble_validation,
        "wcdt_v2_vs_cv_summary": vs_cv_summary,
        "wcdt_v2_architecture_version": ARCHITECTURE_VERSION,
        "wcdt_v2_loss_version": LOSS_VERSION,
        "wcdt_v2_trajectory_schema_version": _trajectory_schema_version(data),
        "wcdt_v2_early_stopping_config": early_stopping,
        "wcdt_v2_prediction_best_val_score": float(payload["best_val_score"]),
    }


def _build_wcdt_v3_loader(cfg: Any, data: Any, sample_indices: np.ndarray, device: Any, *, shuffle: bool):
    torch, DataLoader, TensorDataset = _require_torch()
    from safe_rl.prediction.wcdt_v3_predictor import build_v3_numpy_batch

    if not _has_key(data, "agent_history") or data["agent_history"].shape[0] == 0 or sample_indices.size == 0:
        return None
    _require_trajectory_schema_v2(data, "WcDT v3")
    batch = build_v3_numpy_batch(
        cfg,
        data["agent_history"],
        data["agent_future"],
        data["agent_mask"],
        sample_indices,
        lane_indices=data["agent_lane_index"] if _has_key(data, "agent_lane_index") else None,
        edge_roles=data["agent_edge_role"] if _has_key(data, "agent_edge_role") else None,
        history_valid_mask=data["agent_history_valid_mask"],
        future_valid_mask=data["agent_future_valid_mask"],
        history_lane_indices=data["agent_history_lane_index"],
        history_edge_roles=data["agent_history_edge_role"],
    )
    dataset = TensorDataset(
        torch.tensor(batch["history_features"], dtype=torch.float32),
        torch.tensor(batch["baseline"], dtype=torch.float32),
        torch.tensor(batch["target"], dtype=torch.float32),
        torch.tensor(batch["mask"], dtype=torch.float32),
        torch.tensor(batch["role_ids"], dtype=torch.long),
        torch.tensor(batch["lane_ids"], dtype=torch.long),
        torch.tensor(batch["edge_role_ids"], dtype=torch.long),
        torch.tensor(batch["ego_future"], dtype=torch.float32),
        torch.tensor(batch["history_valid_mask"], dtype=torch.float32),
        torch.tensor(batch["future_valid_mask"], dtype=torch.float32),
        torch.tensor(batch["ego_future_valid_mask"], dtype=torch.float32),
        torch.tensor(batch["history_lane_ids"], dtype=torch.long),
        torch.tensor(batch["history_edge_role_ids"], dtype=torch.long),
    )
    return DataLoader(dataset, batch_size=_wcdt_v3_batch_size(cfg), shuffle=shuffle, **_loader_kwargs(cfg, device))


def _wcdt_v3_tensor_batch(items: tuple[Any, ...], device: Any, pin_memory: bool) -> dict[str, Any]:
    (
        history_features,
        baseline,
        target,
        mask,
        role_ids,
        lane_ids,
        edge_role_ids,
        ego_future,
        history_valid_mask,
        future_valid_mask,
        ego_future_valid_mask,
        history_lane_ids,
        history_edge_role_ids,
    ) = items
    return {
        "history_features": history_features.to(device, non_blocking=pin_memory),
        "baseline": baseline.to(device, non_blocking=pin_memory),
        "target": target.to(device, non_blocking=pin_memory),
        "mask": mask.to(device, non_blocking=pin_memory),
        "role_ids": role_ids.to(device, non_blocking=pin_memory),
        "lane_ids": lane_ids.to(device, non_blocking=pin_memory),
        "edge_role_ids": edge_role_ids.to(device, non_blocking=pin_memory),
        "ego_future": ego_future.to(device, non_blocking=pin_memory),
        "history_valid_mask": history_valid_mask.to(device, non_blocking=pin_memory),
        "future_valid_mask": future_valid_mask.to(device, non_blocking=pin_memory),
        "ego_future_valid_mask": ego_future_valid_mask.to(device, non_blocking=pin_memory),
        "history_lane_ids": history_lane_ids.to(device, non_blocking=pin_memory),
        "history_edge_role_ids": history_edge_role_ids.to(device, non_blocking=pin_memory),
    }


def _wcdt_v3_validation_metrics(cfg: Any, models: list[Any], loader: Any, device: Any, pin_memory: bool) -> dict[str, Any]:
    torch, _DataLoader, _TensorDataset = _require_torch()
    from safe_rl.prediction.wcdt_v3_predictor import ensemble_predict_v3, v3_loss

    ade: list[float] = []
    fde: list[float] = []
    future_min_distance_errors: list[float] = []
    future_min_distance_abs_errors: list[float] = []
    target_gap_errors: list[float] = []
    target_gap_abs_errors: list[float] = []
    target_front_gap_abs_errors: list[float] = []
    target_rear_gap_abs_errors: list[float] = []
    uncertainty_values: list[float] = []
    uncertainty_fde_values: list[float] = []
    uncertainty_min_distance_error_values: list[float] = []
    val_losses: list[float] = []
    component_losses: dict[str, list[float]] = {}
    for model in models:
        model.eval()
    with torch.no_grad():
        for items in loader:
            batch = _wcdt_v3_tensor_batch(items, device, pin_memory)
            pred, uncertainty = ensemble_predict_v3(models, batch)
            loss, components = v3_loss(
                pred,
                batch["target"],
                batch["mask"],
                batch["ego_future"],
                batch["role_ids"],
                dict(cfg.prediction.get("wcdt_v2_loss_weights", {})),
                future_valid_mask=batch["future_valid_mask"],
                ego_future_valid_mask=batch["ego_future_valid_mask"],
            )
            val_losses.append(float(loss.detach().cpu()))
            for name, value in components.items():
                component_losses.setdefault(name, []).append(float(value.detach().cpu()))
            pred_np = pred.detach().cpu().numpy()
            target_np = batch["target"].detach().cpu().numpy()
            mask_np = batch["mask"].detach().cpu().numpy()
            ego_np = batch["ego_future"].detach().cpu().numpy()
            uncertainty_np = uncertainty.detach().cpu().numpy()
            role_ids_np = batch["role_ids"].detach().cpu().numpy()
            future_valid_np = batch["future_valid_mask"].detach().cpu().numpy()
            ego_future_valid_np = batch["ego_future_valid_mask"].detach().cpu().numpy()
            for row in range(pred_np.shape[0]):
                errors = _masked_trajectory_errors(
                    pred_np[row],
                    target_np[row],
                    mask_np[row],
                    future_valid_np[row],
                    ego_future_valid_np[row],
                )
                if errors is None:
                    continue
                row_ade, row_fde = errors
                ade.append(row_ade)
                fde.append(row_fde)
                uncertainty_values.append(float(uncertainty_np[row]))
                uncertainty_fde_values.append(row_fde)
                pred_min = _future_min_distance(
                    ego_np[row], pred_np[row], mask_np[row], future_valid_np[row], ego_future_valid_np[row]
                )
                actual_min = _future_min_distance(
                    ego_np[row], target_np[row], mask_np[row], future_valid_np[row], ego_future_valid_np[row]
                )
                min_distance_error = float(pred_min - actual_min)
                future_min_distance_errors.append(min_distance_error)
                future_min_distance_abs_errors.append(abs(min_distance_error))
                uncertainty_min_distance_error_values.append(abs(min_distance_error))
                role_gap_errors = _target_role_gap_abs_errors(
                    ego_np[row],
                    pred_np[row],
                    target_np[row],
                    mask_np[row],
                    role_ids_np[row],
                    future_valid_np[row],
                    ego_future_valid_np[row],
                )
                if role_gap_errors["target_lane_front_gap_abs_error"] is not None:
                    target_front_gap_abs_errors.append(float(role_gap_errors["target_lane_front_gap_abs_error"]))
                if role_gap_errors["target_lane_rear_gap_abs_error"] is not None:
                    target_rear_gap_abs_errors.append(float(role_gap_errors["target_lane_rear_gap_abs_error"]))
                pred_gap = _target_lane_gap(
                    ego_np[row], pred_np[row], mask_np[row], cfg, future_valid_np[row], ego_future_valid_np[row]
                )
                actual_gap = _target_lane_gap(
                    ego_np[row], target_np[row], mask_np[row], cfg, future_valid_np[row], ego_future_valid_np[row]
                )
                if pred_gap < 1.0e6 and actual_gap < 1.0e6:
                    target_gap_errors.append(float(pred_gap - actual_gap))
                    target_gap_abs_errors.append(abs(float(pred_gap - actual_gap)))
    return {
        "sample_count": int(len(ade)),
        "loss": float(np.mean(val_losses)) if val_losses else 0.0,
        "ade": _summary(ade),
        "fde": _summary(fde),
        "future_min_distance_error": _summary(future_min_distance_errors),
        "future_min_distance_abs_error": _summary(future_min_distance_abs_errors),
        "target_lane_gap_error": _summary(target_gap_errors),
        "target_lane_gap_abs_error": _summary(target_gap_abs_errors),
        "target_lane_front_gap_abs_error": _summary(target_front_gap_abs_errors),
        "target_lane_rear_gap_abs_error": _summary(target_rear_gap_abs_errors),
        "uncertainty": _summary(uncertainty_values),
        "uncertainty_fde_correlation": _correlation(uncertainty_values, uncertainty_fde_values),
        "uncertainty_future_min_distance_abs_error_correlation": _correlation(
            uncertainty_values, uncertainty_min_distance_error_values
        ),
        "loss_components": {name: _summary(values) for name, values in component_losses.items()},
    }


def _wcdt_v3_cv_baseline_metrics(cfg: Any, loader: Any) -> dict[str, Any]:
    ade: list[float] = []
    fde: list[float] = []
    future_min_distance_abs_errors: list[float] = []
    target_gap_abs_errors: list[float] = []
    target_front_gap_abs_errors: list[float] = []
    target_rear_gap_abs_errors: list[float] = []
    for items in loader:
        (
            _history_features,
            baseline,
            target,
            mask,
            role_ids,
            _lane_ids,
            _edge_role_ids,
            ego_future,
            _history_valid_mask,
            future_valid_mask,
            ego_future_valid_mask,
            _history_lane_ids,
            _history_edge_role_ids,
        ) = items
        pred_np = baseline.numpy()
        target_np = target.numpy()
        mask_np = mask.numpy()
        ego_np = ego_future.numpy()
        role_ids_np = role_ids.numpy()
        future_valid_np = future_valid_mask.numpy()
        ego_future_valid_np = ego_future_valid_mask.numpy()
        for row in range(pred_np.shape[0]):
            errors = _masked_trajectory_errors(
                pred_np[row], target_np[row], mask_np[row], future_valid_np[row], ego_future_valid_np[row]
            )
            if errors is None:
                continue
            row_ade, row_fde = errors
            ade.append(row_ade)
            fde.append(row_fde)
            pred_min = _future_min_distance(
                ego_np[row], pred_np[row], mask_np[row], future_valid_np[row], ego_future_valid_np[row]
            )
            actual_min = _future_min_distance(
                ego_np[row], target_np[row], mask_np[row], future_valid_np[row], ego_future_valid_np[row]
            )
            future_min_distance_abs_errors.append(abs(float(pred_min - actual_min)))
            role_gap_errors = _target_role_gap_abs_errors(
                ego_np[row],
                pred_np[row],
                target_np[row],
                mask_np[row],
                role_ids_np[row],
                future_valid_np[row],
                ego_future_valid_np[row],
            )
            if role_gap_errors["target_lane_front_gap_abs_error"] is not None:
                target_front_gap_abs_errors.append(float(role_gap_errors["target_lane_front_gap_abs_error"]))
            if role_gap_errors["target_lane_rear_gap_abs_error"] is not None:
                target_rear_gap_abs_errors.append(float(role_gap_errors["target_lane_rear_gap_abs_error"]))
            pred_gap = _target_lane_gap(
                ego_np[row], pred_np[row], mask_np[row], cfg, future_valid_np[row], ego_future_valid_np[row]
            )
            actual_gap = _target_lane_gap(
                ego_np[row], target_np[row], mask_np[row], cfg, future_valid_np[row], ego_future_valid_np[row]
            )
            if pred_gap < 1.0e6 and actual_gap < 1.0e6:
                target_gap_abs_errors.append(abs(float(pred_gap - actual_gap)))
    return {
        "ade": _summary(ade),
        "fde": _summary(fde),
        "future_min_distance_abs_error": _summary(future_min_distance_abs_errors),
        "target_lane_gap_abs_error": _summary(target_gap_abs_errors),
        "target_lane_front_gap_abs_error": _summary(target_front_gap_abs_errors),
        "target_lane_rear_gap_abs_error": _summary(target_rear_gap_abs_errors),
    }


def _train_wcdt_v3_predictor(
    cfg: Any,
    data: np.lib.npyio.NpzFile,
    stage_dir: Path,
    tb: TensorboardLogger,
    device: Any,
) -> dict[str, Any]:
    sample_count = int(data["agent_history"].shape[0]) if _has_key(data, "agent_history") else 0
    train_indices, val_indices = _split_indices(
        sample_count, float(cfg.prediction.get("validation_split", 0.15)), int(cfg.run.seed)
    )
    train_loader = _build_wcdt_v3_loader(cfg, data, train_indices, device, shuffle=True)
    val_loader = _build_wcdt_v3_loader(cfg, data, val_indices, device, shuffle=False) if val_indices.size else None
    if train_loader is None:
        return {"wcdt_v3_prediction_skipped": True, "wcdt_v3_prediction_skip_reason": "no trajectory samples"}
    torch, _DataLoader, _TensorDataset = _require_torch()
    from safe_rl.prediction.wcdt_v3_predictor import (
        ARCHITECTURE_VERSION,
        LOSS_VERSION,
        WcDTV3TemporalInteractionPredictor,
        _predict_model,
        ensemble_predict_v3,
        v3_loss,
    )

    ensemble_size = int(cfg.prediction.get("wcdt_v3_ensemble_size", 3))
    epochs = int(cfg.prediction.get("wcdt_v3_epochs", cfg.prediction.epochs))
    loss_weights = dict(cfg.prediction.get("wcdt_v2_loss_weights", {}))
    early_stopping = _wcdt_v3_early_stopping_config(cfg, has_validation=val_loader is not None)
    model_kwargs = {
        "history_steps": int(cfg.scenario.history_steps),
        "horizon_steps": int(cfg.prediction.get("wcdt_v3_horizon_steps", cfg.scenario.forecast_horizon_steps)),
        "hidden_dim": int(cfg.prediction.get("wcdt_v3_hidden_dim", 128)),
        "temporal_layers": int(cfg.prediction.get("wcdt_v3_temporal_layers", 2)),
        "actor_attention_layers": int(cfg.prediction.get("wcdt_v3_actor_attention_layers", 2)),
        "num_heads": int(cfg.prediction.get("wcdt_v3_num_heads", 4)),
        "dropout": float(cfg.prediction.get("wcdt_v3_dropout", 0.1)),
    }
    model_states: list[dict[str, Any]] = []
    member_histories: list[dict[str, Any]] = []
    validation_history: list[dict[str, Any]] = []
    cv_baseline = _wcdt_v3_cv_baseline_metrics(cfg, val_loader) if val_loader is not None else None
    pin_memory = bool(getattr(train_loader, "pin_memory", False))
    stage_log(
        "stage2",
        f"WcDT v3 train_samples={train_indices.shape[0]}, val_samples={val_indices.shape[0]}, "
        f"ensemble={ensemble_size}, epochs={epochs}, batch_size={_wcdt_v3_batch_size(cfg)}",
    )
    for member_idx in range(ensemble_size):
        torch.manual_seed(int(cfg.run.seed) + 2000 + member_idx)
        model = WcDTV3TemporalInteractionPredictor(**model_kwargs).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=float(cfg.prediction.get("wcdt_v3_learning_rate", cfg.prediction.learning_rate))
        )
        losses: list[float] = []
        loss_component_history: list[dict[str, float]] = []
        best_state: dict[str, Any] | None = None
        best_score: float | None = None
        best_epoch = 0
        stale_epochs = 0
        stopped_early = False
        early_stopping_reason: str | None = None
        member_validation: list[dict[str, Any]] = []
        for epoch in progress_iter(range(epochs), desc=f"Stage2 WcDT v3 member {member_idx + 1}/{ensemble_size}"):
            epoch_losses: list[float] = []
            epoch_components: dict[str, list[float]] = {}
            model.train()
            for items in train_loader:
                batch = _wcdt_v3_tensor_batch(items, device, pin_memory)
                pred = _predict_model(model, batch)
                loss, components = v3_loss(
                    pred,
                    batch["target"],
                    batch["mask"],
                    batch["ego_future"],
                    batch["role_ids"],
                    loss_weights,
                    future_valid_mask=batch["future_valid_mask"],
                    ego_future_valid_mask=batch["ego_future_valid_mask"],
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_losses.append(float(loss.detach().cpu()))
                for name, value in components.items():
                    epoch_components.setdefault(name, []).append(float(value.detach().cpu()))
            epoch_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
            losses.append(epoch_loss)
            component_summary = {
                name: float(np.mean(values)) if values else 0.0 for name, values in epoch_components.items()
            }
            loss_component_history.append(component_summary)
            val_score = epoch_loss
            validation_metrics: dict[str, Any] | None = None
            if val_loader is not None:
                validation_metrics = _wcdt_v3_validation_metrics(cfg, [model], val_loader, device, pin_memory)
                validation_metrics["epoch"] = int(epoch + 1)
                validation_metrics["member"] = int(member_idx)
                val_score = _prediction_val_score(validation_metrics, cfg)
                validation_metrics["val_score"] = float(val_score)
                member_validation.append(validation_metrics)
                tb.scalar(f"stage2/wcdt_v3_member_{member_idx}/val_score", val_score, epoch)
            tb.scalar(f"stage2/wcdt_v3_member_{member_idx}/loss", epoch_loss, epoch)
            for name, value in component_summary.items():
                tb.scalar(f"stage2/wcdt_v3_member_{member_idx}/loss_{name}", value, epoch)
            improved, stale_epochs, should_stop = _wcdt_v2_early_stopping_step(
                best_score=best_score,
                score=val_score,
                epoch=epoch + 1,
                stale_epochs=stale_epochs,
                config=early_stopping,
            )
            if improved:
                best_score = float(val_score)
                best_epoch = int(epoch + 1)
                best_state = _cpu_state_dict(model)
            if validation_metrics is not None:
                stage_log(
                    "stage2",
                    f"wcdt_v3 member={member_idx + 1}/{ensemble_size} epoch={epoch + 1}/{epochs} "
                    f"loss={epoch_loss:.6f} val_score={val_score:.6f} "
                    f"ade={float(validation_metrics['ade'].get('mean', 0.0)):.3f} "
                    f"fde={float(validation_metrics['fde'].get('mean', 0.0)):.3f} "
                    f"minD={float(validation_metrics['future_min_distance_abs_error'].get('mean', 0.0)):.3f} "
                    f"front_gap={float(validation_metrics['target_lane_front_gap_abs_error'].get('mean', 0.0)):.3f} "
                    f"rear_gap={float(validation_metrics['target_lane_rear_gap_abs_error'].get('mean', 0.0)):.3f}",
                )
            else:
                stage_log("stage2", f"wcdt_v3 member={member_idx + 1}/{ensemble_size} epoch={epoch + 1}/{epochs} loss={epoch_loss:.6f}")
            if should_stop:
                stopped_early = True
                early_stopping_reason = (
                    f"no val_score improvement >= {early_stopping['min_delta']} "
                    f"for {early_stopping['patience']} epochs"
                )
                break
        if best_state is None:
            best_state = _cpu_state_dict(model)
        model_states.append(best_state)
        member_histories.append(
            {
                "member": int(member_idx),
                "loss_history": losses,
                "loss_component_history": loss_component_history,
                "validation_history": member_validation,
                "trained_epochs": int(len(losses)),
                "best_epoch": int(best_epoch),
                "best_val_score": float(best_score if best_score is not None else (losses[-1] if losses else 0.0)),
                "stopped_early": bool(stopped_early),
                "early_stopping_reason": early_stopping_reason,
            }
        )
    ensemble_models = []
    for state in model_states:
        model = WcDTV3TemporalInteractionPredictor(**model_kwargs).to(device)
        model.load_state_dict(state)
        model.eval()
        ensemble_models.append(model)
    ensemble_validation = (
        _wcdt_v3_validation_metrics(cfg, ensemble_models, val_loader, device, pin_memory)
        if val_loader is not None
        else {"sample_count": 0}
    )
    if val_loader is not None:
        ensemble_validation["val_score"] = _prediction_val_score(ensemble_validation, cfg)
        validation_history.append(ensemble_validation)
    vs_cv_summary = _wcdt_v2_vs_cv_summary(cv_baseline, ensemble_validation)
    checkpoint = stage_dir / "wcdt_v3_predictor.pt"
    best_checkpoint = stage_dir / "wcdt_v3_predictor_best.pt"
    payload = {
        "model_state_dicts": model_states,
        "member_histories": member_histories,
        "validation_history": validation_history,
        "ensemble_validation": ensemble_validation,
        "cv_baseline_validation": cv_baseline,
        "wcdt_v3_vs_cv_summary": vs_cv_summary,
        "architecture_version": ARCHITECTURE_VERSION,
        "loss_version": LOSS_VERSION,
        "trajectory_schema_version": _trajectory_schema_version(data),
        **model_kwargs,
        "ensemble_size": int(ensemble_size),
        "loss_weights": loss_weights,
        "early_stopping_config": early_stopping,
        "best_metric": "ensemble_val_score",
        "best_val_score": float(ensemble_validation.get("val_score", 0.0)),
        "train_sample_count": int(train_indices.shape[0]),
        "validation_sample_count": int(val_indices.shape[0]),
    }
    torch.save(payload, best_checkpoint)
    torch.save(payload, checkpoint)
    return {
        "wcdt_v3_prediction_checkpoint": str(checkpoint),
        "wcdt_v3_prediction_best_checkpoint": str(best_checkpoint),
        "wcdt_v3_member_histories": member_histories,
        "wcdt_v3_prediction_validation_history": validation_history,
        "wcdt_v3_prediction_cv_baseline_validation": cv_baseline,
        "wcdt_v3_prediction_ensemble_validation": ensemble_validation,
        "wcdt_v3_vs_cv_summary": vs_cv_summary,
        "wcdt_v3_architecture_version": ARCHITECTURE_VERSION,
        "wcdt_v3_loss_version": LOSS_VERSION,
        "wcdt_v3_trajectory_schema_version": _trajectory_schema_version(data),
        "wcdt_v3_early_stopping_config": early_stopping,
        "wcdt_v3_prediction_best_val_score": float(payload["best_val_score"]),
    }


def run(cfg) -> Path:
    torch, _DataLoader, _TensorDataset = _require_torch()
    device = _resolve_device(cfg, torch)
    _configure_torch_backend(cfg, torch, device)
    stage_dir = prepare_run_dir(cfg, "stage2")
    input_path = _stage1_path(cfg)
    input_stage4_path = _stage4_path(cfg)
    stage_log("stage2", f"run_id={cfg.run.run_id}")
    stage_log("stage2", f"input_stage1={input_path}")
    if input_stage4_path is not None:
        stage_log("stage2", f"input_stage4={input_stage4_path}")
    stage_log("stage2", f"output_dir={stage_dir}")
    if device.type == "cuda":
        stage_log("stage2", f"device={device} ({torch.cuda.get_device_name(device)})")
    else:
        stage_log("stage2", f"device={device}")
    tb = TensorboardLogger(stage_dir / "tensorboard", enabled=bool(cfg.run.get("tensorboard", True)))
    initial_prediction_report_path = stage_dir / "stage2_initial_prediction_report.json"
    data = np.load(input_path, allow_pickle=False)
    stage4_data = np.load(input_stage4_path, allow_pickle=False) if input_stage4_path is not None else None
    risk_data = _merge_risk_buffers(data, stage4_data)
    executed_count = int(data["executed_actions"].shape[0]) if "executed_actions" in data else int(data["actions"].shape[0])
    stage_log("stage2", f"executed_transition_count={executed_count}")
    stage_log("stage2", f"risk_transition_count={int(data['actions'].shape[0])}")
    if stage4_data is not None:
        stage4_executed = (
            int(stage4_data["executed_actions"].shape[0])
            if "executed_actions" in stage4_data
            else int(stage4_data["actions"].shape[0])
        )
        stage_log("stage2", f"stage4_executed_transition_count={stage4_executed}")
        stage_log("stage2", f"stage4_risk_transition_count={int(stage4_data['actions'].shape[0])}")
        stage_log("stage2", f"risk_transition_count={int(risk_data['actions'].shape[0])}")
    if "agent_history" in data:
        stage_log("stage2", f"trajectory_samples={int(data['agent_history'].shape[0])}")
    report = {
        "stage": "stage2",
        "run_id": cfg.run.run_id,
        "input_stage1": str(input_path),
        "input_stage4": str(input_stage4_path) if input_stage4_path is not None else None,
        "transition_count": executed_count,
        "stage1_risk_transition_count": int(data["actions"].shape[0]),
        "stage4_transition_count": (
            int(stage4_data["executed_actions"].shape[0])
            if stage4_data is not None and "executed_actions" in stage4_data
            else int(stage4_data["actions"].shape[0])
            if stage4_data is not None
            else 0
        ),
        "risk_transition_count": int(risk_data["actions"].shape[0]),
        "tensorboard": str(stage_dir / "tensorboard"),
        "device": str(device),
        "prediction_train_enabled": bool(cfg.prediction.get("train_enabled", True)),
        "wcdt_v1_train_enabled": bool(cfg.prediction.get("wcdt_v1_train_enabled", False)),
        "wcdt_v2_train_enabled": bool(cfg.prediction.get("wcdt_v2_train_enabled", True)),
        "wcdt_v3_train_enabled": bool(cfg.prediction.get("wcdt_v3_train_enabled", False)),
        "wcdt_v1_batch_size": _wcdt_v1_batch_size(cfg),
        "wcdt_v2_batch_size": _wcdt_v2_batch_size(cfg),
        "wcdt_v3_batch_size": _wcdt_v3_batch_size(cfg),
    }
    report.update(_train_risk_module(cfg, risk_data, stage_dir, tb, device))
    if bool(cfg.prediction.get("train_enabled", True)):
        if bool(cfg.prediction.get("wcdt_v1_train_enabled", False)):
            report.update(_train_wcdt_predictor(cfg, data, stage_dir, tb, device))
        if bool(cfg.prediction.get("wcdt_v2_train_enabled", True)):
            report.update(_train_wcdt_v2_predictor(cfg, data, stage_dir, tb, device))
        if bool(cfg.prediction.get("wcdt_v3_train_enabled", False)):
            report.update(_train_wcdt_v3_predictor(cfg, data, stage_dir, tb, device))
        if report.get("prediction_checkpoint") or report.get("wcdt_v2_prediction_checkpoint") or report.get("wcdt_v3_prediction_checkpoint"):
            initial_prediction_report = {
                "stage": "stage2_initial_prediction",
                "run_id": cfg.run.run_id,
                "input_stage1": str(input_path),
                "prediction_checkpoint": report.get("prediction_checkpoint"),
                "prediction_best_checkpoint": report.get("prediction_best_checkpoint"),
                "prediction_loss_history": report.get("prediction_loss_history", []),
                "prediction_val_loss_history": report.get("prediction_val_loss_history", []),
                "prediction_validation_history": report.get("prediction_validation_history", []),
                "prediction_loss_summary": _prediction_loss_summary(report.get("prediction_loss_history", [])),
                "prediction_best_epoch": report.get("prediction_best_epoch"),
                "prediction_best_val_score": report.get("prediction_best_val_score"),
                "wcdt_v2_prediction_checkpoint": report.get("wcdt_v2_prediction_checkpoint"),
                "wcdt_v2_prediction_best_checkpoint": report.get("wcdt_v2_prediction_best_checkpoint"),
                "wcdt_v2_prediction_validation_history": report.get("wcdt_v2_prediction_validation_history", []),
                "wcdt_v2_prediction_cv_baseline_validation": report.get("wcdt_v2_prediction_cv_baseline_validation"),
                "wcdt_v2_prediction_ensemble_validation": report.get("wcdt_v2_prediction_ensemble_validation"),
                "wcdt_v2_member_histories": report.get("wcdt_v2_member_histories", []),
                "wcdt_v2_loss_component_history": report.get("wcdt_v2_loss_component_history", []),
                "wcdt_v2_vs_cv_summary": report.get("wcdt_v2_vs_cv_summary"),
                "wcdt_v2_architecture_version": report.get("wcdt_v2_architecture_version"),
                "wcdt_v2_loss_version": report.get("wcdt_v2_loss_version"),
                "wcdt_v2_early_stopping_config": report.get("wcdt_v2_early_stopping_config"),
                "wcdt_v2_prediction_best_val_score": report.get("wcdt_v2_prediction_best_val_score"),
                "wcdt_v3_prediction_checkpoint": report.get("wcdt_v3_prediction_checkpoint"),
                "wcdt_v3_prediction_best_checkpoint": report.get("wcdt_v3_prediction_best_checkpoint"),
                "wcdt_v3_member_histories": report.get("wcdt_v3_member_histories", []),
                "wcdt_v3_prediction_validation_history": report.get("wcdt_v3_prediction_validation_history", []),
                "wcdt_v3_prediction_cv_baseline_validation": report.get("wcdt_v3_prediction_cv_baseline_validation"),
                "wcdt_v3_prediction_ensemble_validation": report.get("wcdt_v3_prediction_ensemble_validation"),
                "wcdt_v3_vs_cv_summary": report.get("wcdt_v3_vs_cv_summary"),
                "wcdt_v3_architecture_version": report.get("wcdt_v3_architecture_version"),
                "wcdt_v3_loss_version": report.get("wcdt_v3_loss_version"),
                "wcdt_v3_early_stopping_config": report.get("wcdt_v3_early_stopping_config"),
                "wcdt_v3_prediction_best_val_score": report.get("wcdt_v3_prediction_best_val_score"),
                "device": str(device),
            }
            write_report(initial_prediction_report_path, initial_prediction_report)
            report["initial_prediction_report"] = str(initial_prediction_report_path)
    elif initial_prediction_report_path.exists():
        report["initial_prediction_report"] = str(initial_prediction_report_path)
    if stage4_data is None:
        risk_checkpoint = Path(str(report["risk_checkpoint"]))
        initial_risk_checkpoint = stage_dir / "risk_module_initial.pt"
        shutil.copy2(risk_checkpoint, initial_risk_checkpoint)
        report["risk_initial_checkpoint"] = str(initial_risk_checkpoint)
        initial_training_report_path = stage_dir / "stage2_initial_training_report.json"
        write_report(initial_training_report_path, report)
        report["initial_training_report"] = str(initial_training_report_path)
    write_report(stage_dir / "stage2_training_report.json", report)
    tb.close()
    stage_log("stage2", f"report={stage_dir / 'stage2_training_report.json'}")
    return stage_dir


def main() -> None:
    args = parse_config_arg("Stage2 WcDT-style prediction + risk module training")
    cfg = load_stage_config(args)
    run(cfg)


if __name__ == "__main__":
    main()
