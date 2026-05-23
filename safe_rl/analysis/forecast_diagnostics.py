from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from safe_rl.pipeline.common import run_root, stage_file, write_report
from safe_rl.prediction.forecast_feature_augmentor import ForecastFeatureAugmentor
from safe_rl.prediction.sumo_wcdt_adapter import SumoWcDTAdapter
from safe_rl.prediction.wcdt_v2_predictor import build_v2_numpy_batch, ensemble_predict, load_v2_ensemble, tensorize_batch
from safe_rl.risk.merge_local import merge_target_lane
from safe_rl.sim.metrics import INF_TTC
from safe_rl.sim.types import VehicleState


LANE_CENTERS = {0: -8.0, 1: -4.8, 2: -1.6}


def _summary(values: np.ndarray | list[float]) -> dict[str, float | int]:
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


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else Path.cwd() / path


def _infer_lane_index(y: float) -> int:
    return min(LANE_CENTERS, key=lambda lane: abs(float(y) - LANE_CENTERS[lane]))


def _vector_to_state(vehicle_id: str, vector: np.ndarray) -> VehicleState:
    x, y, heading, speed, accel = [float(item) for item in vector[:5]]
    lane_index = _infer_lane_index(y)
    edge_id = "ramp_in" if y > 0.5 and x < 224.0 else ("main_in" if x < 224.0 else "main_out")
    return VehicleState(
        vehicle_id=vehicle_id,
        x=x,
        y=y,
        heading=heading,
        speed=speed,
        lane_index=lane_index,
        lane_id=f"{edge_id}_{lane_index}",
        lane_pos=x,
        edge_id=edge_id,
        accel=accel,
    )


def _latest_states(history: np.ndarray, mask: np.ndarray) -> list[VehicleState]:
    states: list[VehicleState] = []
    for agent_idx in range(history.shape[0]):
        if float(mask[agent_idx]) <= 0.0:
            continue
        vehicle_id = "ego" if agent_idx == 0 else f"agent_{agent_idx}"
        states.append(_vector_to_state(vehicle_id, history[agent_idx, -1]))
    return states


def _cv_feature_matrix(cfg: Any, history: np.ndarray, mask: np.ndarray, indices: np.ndarray) -> np.ndarray:
    augmentor = ForecastFeatureAugmentor(cfg)
    rows = []
    for sample_idx in indices:
        states = _latest_states(history[sample_idx], mask[sample_idx])
        ego = next((state for state in states if state.vehicle_id == "ego"), None)
        rows.append(augmentor.extract({"ego": ego, "vehicles": states, "config": cfg}))
    return np.asarray(rows, dtype=np.float32)


def _constant_velocity_future(last: np.ndarray, horizon: int, dt: float) -> np.ndarray:
    output = np.zeros((horizon, 5), dtype=np.float32)
    x = float(last[0])
    y = float(last[1])
    heading = float(last[2])
    speed = max(0.0, float(last[3]))
    vx = speed * np.cos(heading)
    vy = speed * np.sin(heading)
    for step_idx in range(horizon):
        x += vx * dt
        y += vy * dt
        output[step_idx] = [x, y, heading, speed, 0.0]
    return output


def _cv_prediction_diagnostics(
    cfg: Any,
    history: np.ndarray,
    future: np.ndarray,
    mask: np.ndarray,
    indices: np.ndarray,
) -> dict[str, Any]:
    horizon = int(min(future.shape[2], cfg.forecast_features.get("horizon_steps", cfg.scenario.forecast_horizon_steps)))
    dt = float(cfg.scenario.step_length)
    ade: list[float] = []
    fde: list[float] = []
    min_distance_errors: list[float] = []
    min_distance_abs_errors: list[float] = []
    target_gap_errors: list[float] = []
    target_gap_abs_errors: list[float] = []
    for sample_idx in indices:
        actual_future = future[sample_idx, :, :horizon]
        pred_future = np.zeros_like(actual_future)
        for agent_idx in range(history.shape[1]):
            if float(mask[sample_idx, agent_idx]) <= 0.0:
                continue
            pred_future[agent_idx] = _constant_velocity_future(history[sample_idx, agent_idx, -1], horizon, dt)
        valid_agents = mask[sample_idx] > 0.0
        if np.sum(valid_agents) <= 1:
            continue
        other_valid = valid_agents.copy()
        other_valid[0] = False
        if not np.any(other_valid):
            continue
        diff = pred_future[other_valid, :, :2] - actual_future[other_valid, :, :2]
        per_step = np.linalg.norm(diff, axis=-1)
        ade.append(float(np.mean(per_step)))
        fde.append(float(np.mean(per_step[:, -1])))
        other_mask = mask[sample_idx].copy()
        other_mask[0] = 0.0
        pred_min = _future_min_distance(actual_future[0], pred_future, other_mask)
        actual_min = _future_min_distance(actual_future[0], actual_future, other_mask)
        min_distance_errors.append(float(pred_min - actual_min))
        min_distance_abs_errors.append(abs(float(pred_min - actual_min)))
        pred_gap = _target_lane_gap(actual_future[0], pred_future, other_mask, cfg)
        actual_gap = _target_lane_gap(actual_future[0], actual_future, other_mask, cfg)
        if pred_gap < INF_TTC and actual_gap < INF_TTC:
            target_gap_errors.append(float(pred_gap - actual_gap))
            target_gap_abs_errors.append(abs(float(pred_gap - actual_gap)))
    return {
        "sample_count": int(len(ade)),
        "ade": _summary(ade),
        "fde": _summary(fde),
        "future_min_distance_error": _summary(min_distance_errors),
        "future_min_distance_abs_error": _summary(min_distance_abs_errors),
        "target_lane_gap_error": _summary(target_gap_errors),
        "target_lane_gap_abs_error": _summary(target_gap_abs_errors),
    }


def _require_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Forecast diagnostics require torch. Activate the SAFE_RL environment.") from exc
    return torch


def _resolve_device(cfg: Any, torch: Any):
    requested = str(cfg.get("training", {}).get("device", "auto")).strip().lower()
    if requested in ("auto", ""):
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if requested == "gpu":
        requested = "cuda"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def _build_wcdt_inputs(
    cfg: Any,
    history: np.ndarray,
    future: np.ndarray,
    mask: np.ndarray,
    indices: np.ndarray,
    torch: Any,
    device: Any,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    max_pred = int(cfg.prediction.max_pred_num)
    max_other = int(cfg.prediction.max_other_num)
    hist_steps = int(cfg.scenario.history_steps)
    horizon = int(min(future.shape[2], cfg.forecast_features.get("horizon_steps", cfg.scenario.forecast_horizon_steps)))
    batch = indices.shape[0]
    pred_indices = list(range(1, min(history.shape[1], max_pred + 1)))
    other_indices = [0] + list(range(max_pred + 1, min(history.shape[1], max_pred + max_other)))

    predicted_his = np.zeros((batch, max_pred, hist_steps, 5), dtype=np.float32)
    predicted_future = np.zeros((batch, max_pred, 80, 5), dtype=np.float32)
    predicted_mask = np.zeros((batch, max_pred), dtype=np.float32)
    other_his = np.zeros((batch, max_other, hist_steps, 5), dtype=np.float32)
    other_mask = np.zeros((batch, max_other), dtype=np.float32)
    ego_future = np.zeros((batch, horizon, 5), dtype=np.float32)

    for row, sample_idx in enumerate(indices):
        ego_future[row] = future[sample_idx, 0, :horizon]
        for pred_row, agent_idx in enumerate(pred_indices[:max_pred]):
            predicted_his[row, pred_row] = history[sample_idx, agent_idx]
            predicted_future[row, pred_row, : future.shape[2]] = future[sample_idx, agent_idx]
            if future.shape[2] < 80:
                predicted_future[row, pred_row, future.shape[2] :] = future[sample_idx, agent_idx, -1]
            predicted_mask[row, pred_row] = mask[sample_idx, agent_idx]
        for other_row, agent_idx in enumerate(other_indices[:max_other]):
            other_his[row, other_row] = history[sample_idx, agent_idx]
            other_mask[row, other_row] = mask[sample_idx, agent_idx]

    predicted_feature = np.zeros((batch, max_pred, 7), dtype=np.float32)
    other_feature = np.zeros((batch, max_other, 7), dtype=np.float32)
    predicted_feature[..., 0] = 1.8
    predicted_feature[..., 1] = 4.8
    predicted_feature[..., 3] = 1.0
    other_feature[..., 0] = 1.8
    other_feature[..., 1] = 4.8
    other_feature[..., 3] = 1.0

    lane_batch = np.repeat(SumoWcDTAdapter(cfg).lane_list[None, ...], batch, axis=0)
    data = {
        "predicted_feature": torch.tensor(predicted_feature, dtype=torch.float32, device=device),
        "other_his_pos": torch.tensor(other_his[:, :, -1, :2], dtype=torch.float32, device=device),
        "other_his_traj_delt": torch.tensor(other_his[:, :, 1:] - other_his[:, :, :-1], dtype=torch.float32, device=device),
        "other_feature": torch.tensor(other_feature, dtype=torch.float32, device=device),
        "other_traj_mask": torch.tensor(other_mask, dtype=torch.float32, device=device),
        "predicted_his_pos": torch.tensor(predicted_his[:, :, -1, :2], dtype=torch.float32, device=device),
        "predicted_his_traj_delt": torch.tensor(predicted_his[:, :, 1:] - predicted_his[:, :, :-1], dtype=torch.float32, device=device),
        "predicted_his_traj": torch.tensor(predicted_his, dtype=torch.float32, device=device),
        "predicted_future_traj": torch.tensor(predicted_future, dtype=torch.float32, device=device),
        "predicted_traj_mask": torch.tensor(predicted_mask, dtype=torch.float32, device=device),
        "traffic_light": torch.zeros((batch, int(cfg.prediction.max_traffic_light), hist_steps), dtype=torch.float32, device=device),
        "traffic_light_pos": torch.zeros((batch, int(cfg.prediction.max_traffic_light), 2), dtype=torch.float32, device=device),
        "lane_list": torch.tensor(lane_batch, dtype=torch.float32, device=device),
    }
    return data, predicted_future[:, :, :horizon], predicted_mask, ego_future


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


def _future_min_distance(ego_future: np.ndarray, other_future: np.ndarray, other_mask: np.ndarray) -> float:
    min_distance = INF_TTC
    for agent_idx in range(other_future.shape[0]):
        if float(other_mask[agent_idx]) <= 0.0:
            continue
        distances = np.linalg.norm(other_future[agent_idx, :, :2] - ego_future[:, :2], axis=-1) - 3.0
        min_distance = min(min_distance, float(np.min(np.maximum(0.0, distances))))
    return float(min_distance)


def _target_lane_gap(ego_future: np.ndarray, other_future: np.ndarray, other_mask: np.ndarray, cfg: Any) -> float:
    target_y = LANE_CENTERS.get(merge_target_lane(cfg), -1.6)
    min_gap = INF_TTC
    for step_idx in range(ego_future.shape[0]):
        ego_x = float(ego_future[step_idx, 0])
        for agent_idx in range(other_future.shape[0]):
            if float(other_mask[agent_idx]) <= 0.0:
                continue
            x = float(other_future[agent_idx, step_idx, 0])
            y = float(other_future[agent_idx, step_idx, 1])
            if abs(y - target_y) > 2.0:
                continue
            min_gap = min(min_gap, max(0.0, abs(x - ego_x) - 4.8))
    return float(min_gap)


def _forecast_features_from_prediction(
    ego_state: VehicleState,
    trajectories: np.ndarray,
    uncertainty: float,
    cfg: Any,
) -> np.ndarray:
    min_distance = 50.0
    min_ttc = INF_TTC
    max_drac = 0.0
    nearest_dx = 0.0
    nearest_dy = 0.0
    top_risks: list[float] = []
    dt = float(cfg.scenario.step_length)
    horizon = trajectories.shape[1]
    ego_x = float(ego_state.x)
    ego_y = float(ego_state.y)
    ego_speed = float(ego_state.speed)
    ego_heading = float(ego_state.heading)
    ego_future = np.zeros((horizon, 2), dtype=np.float32)
    for step_idx in range(horizon):
        ego_x += ego_speed * np.cos(ego_heading) * dt
        ego_y += ego_speed * np.sin(ego_heading) * dt
        ego_future[step_idx] = [ego_x, ego_y]
    for traj in trajectories:
        previous_distance = INF_TTC
        agent_min = 50.0
        for step_idx, step in enumerate(traj):
            dx = float(step[0] - ego_future[step_idx, 0])
            dy = float(step[1] - ego_future[step_idx, 1])
            distance = max(0.0, float(np.hypot(dx, dy)) - 3.0)
            if distance < min_distance:
                min_distance = distance
                nearest_dx = dx
                nearest_dy = dy
            agent_min = min(agent_min, distance)
            if previous_distance < INF_TTC:
                closing = max(0.0, (previous_distance - distance) / max(dt, 1.0e-6))
                if closing > 1.0e-6:
                    min_ttc = min(min_ttc, distance / closing)
                    max_drac = max(max_drac, (closing * closing) / (2.0 * max(distance, 1.0e-6)))
            previous_distance = distance
        top_risks.append(1.0 / (1.0 + agent_min))
    top = np.sort(np.asarray(top_risks, dtype=np.float32))[::-1]
    top = np.pad(top[:3], (0, max(0, 3 - len(top))), constant_values=0.0)
    return np.asarray(
        [
            min_distance,
            min_ttc,
            max_drac,
            float(min_distance < float(cfg.risk_module.collision_distance_threshold)),
            float(uncertainty),
            min_distance,
            nearest_dx,
            nearest_dy,
            float(top[0]),
            float(top[1]),
            float(top[2]),
        ],
        dtype=np.float32,
    )


def _load_wcdt_model(cfg: Any, checkpoint: Path, torch: Any, device: Any):
    from net_works import BackBone
    from utils import MathUtil

    betas = MathUtil.generate_linear_schedule(50, 1e-4, 0.008)
    model = BackBone(betas).to(device)
    payload = torch.load(checkpoint, map_location=device)
    state = payload["model_state_dict"] if isinstance(payload, dict) and "model_state_dict" in payload else payload
    model.load_state_dict(state, strict=False)
    model.eval()
    loss_history = payload.get("loss_history") if isinstance(payload, dict) else None
    return model, loss_history


def _wcdt_diagnostics(
    cfg: Any,
    checkpoint: Path,
    history: np.ndarray,
    future: np.ndarray,
    mask: np.ndarray,
    indices: np.ndarray,
    batch_size: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    torch = _require_torch()
    device = _resolve_device(cfg, torch)
    model, loss_history = _load_wcdt_model(cfg, checkpoint, torch, device)
    augmentor = ForecastFeatureAugmentor(cfg)
    feature_rows: list[np.ndarray] = []
    ade: list[float] = []
    fde: list[float] = []
    min_distance_errors: list[float] = []
    min_distance_abs_errors: list[float] = []
    target_gap_errors: list[float] = []
    target_gap_abs_errors: list[float] = []
    uncertainty_values: list[float] = []
    confidence_values: list[float] = []
    confidence_fde_values: list[float] = []

    with torch.no_grad():
        for start in range(0, indices.shape[0], batch_size):
            batch_indices = indices[start : start + batch_size]
            data, actual_future, pred_mask, ego_future = _build_wcdt_inputs(
                cfg, history, future, mask, batch_indices, torch, device
            )
            output = model.predict(data, horizon_steps=int(actual_future.shape[2]))
            traj = output["future_trajectories"].detach().cpu().numpy()
            confidence = output.get("mode_confidence")
            confidence_np = confidence.detach().cpu().numpy() if confidence is not None else None
            selected = _select_best_mode(traj, confidence_np)
            uncertainty = output.get("uncertainty")
            uncertainty_np = uncertainty.detach().cpu().numpy() if uncertainty is not None else np.zeros(pred_mask.shape)

            for row in range(selected.shape[0]):
                valid_agents = pred_mask[row] > 0.0
                if not np.any(valid_agents):
                    continue
                diff = selected[row, valid_agents, :, :2] - actual_future[row, valid_agents, :, :2]
                per_step = np.linalg.norm(diff, axis=-1)
                row_ade = float(np.mean(per_step))
                row_fde = float(np.mean(per_step[:, -1]))
                ade.append(row_ade)
                fde.append(row_fde)
                pred_min_distance = _future_min_distance(ego_future[row], selected[row], pred_mask[row])
                actual_min_distance = _future_min_distance(ego_future[row], actual_future[row], pred_mask[row])
                min_distance_errors.append(float(pred_min_distance - actual_min_distance))
                min_distance_abs_errors.append(abs(float(pred_min_distance - actual_min_distance)))
                pred_gap = _target_lane_gap(ego_future[row], selected[row], pred_mask[row], cfg)
                actual_gap = _target_lane_gap(ego_future[row], actual_future[row], pred_mask[row], cfg)
                if pred_gap < INF_TTC and actual_gap < INF_TTC:
                    target_gap_errors.append(float(pred_gap - actual_gap))
                    target_gap_abs_errors.append(abs(float(pred_gap - actual_gap)))
                states = _latest_states(history[batch_indices[row]], mask[batch_indices[row]])
                ego = next((state for state in states if state.vehicle_id == "ego"), None)
                if ego is not None:
                    sample_uncertainty = float(np.mean(uncertainty_np[row][valid_agents]))
                    uncertainty_values.append(sample_uncertainty)
                    if confidence_np is not None:
                        confidence_values.append(float(np.mean(np.max(confidence_np[row][valid_agents], axis=-1))))
                        confidence_fde_values.append(row_fde)
                    features = _forecast_features_from_prediction(ego, selected[row, valid_agents], sample_uncertainty, cfg)
                    if bool(cfg.forecast_features.normalize):
                        features = augmentor._normalize(features)
                    feature_rows.append(features)

    checkpoint_loss_summary = None
    if loss_history:
        loss_history = [float(item) for item in loss_history]
        checkpoint_loss_summary = {
            "epochs": len(loss_history),
            "first": loss_history[0],
            "last": loss_history[-1],
            "min": float(min(loss_history)),
            "source": "checkpoint",
        }
    report = {
        "device": str(device),
        "checkpoint": str(checkpoint),
        "checkpoint_loss_summary": checkpoint_loss_summary,
        "checkpoint_loss_history": loss_history or [],
        "sample_count": int(len(feature_rows)),
        "ade": _summary(ade),
        "fde": _summary(fde),
        "future_min_distance_error": _summary(min_distance_errors),
        "future_min_distance_abs_error": _summary(min_distance_abs_errors),
        "target_lane_gap_error": _summary(target_gap_errors),
        "target_lane_gap_abs_error": _summary(target_gap_abs_errors),
        "uncertainty": _summary(uncertainty_values),
        "confidence": _summary(confidence_values),
        "confidence_fde_correlation": _correlation(confidence_values, confidence_fde_values),
    }
    return np.asarray(feature_rows, dtype=np.float32), report


def _wcdt_v2_diagnostics(
    cfg: Any,
    checkpoint: Path,
    history: np.ndarray,
    future: np.ndarray,
    mask: np.ndarray,
    indices: np.ndarray,
    batch_size: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    torch = _require_torch()
    device = _resolve_device(cfg, torch)
    models, payload, device = load_v2_ensemble(cfg, checkpoint, device)
    augmentor = ForecastFeatureAugmentor(cfg)
    feature_rows: list[np.ndarray] = []
    ade: list[float] = []
    fde: list[float] = []
    min_distance_errors: list[float] = []
    min_distance_abs_errors: list[float] = []
    target_gap_errors: list[float] = []
    target_gap_abs_errors: list[float] = []
    uncertainty_values: list[float] = []
    uncertainty_fde_values: list[float] = []

    for start in range(0, indices.shape[0], batch_size):
        batch_indices = indices[start : start + batch_size]
        numpy_batch = build_v2_numpy_batch(cfg, history, future, mask, batch_indices)
        tensor_batch = tensorize_batch(numpy_batch, torch, device)
        pred, uncertainty = ensemble_predict(models, tensor_batch)
        selected = pred.detach().cpu().numpy()
        actual_future = numpy_batch["target"]
        pred_mask = numpy_batch["mask"]
        ego_future = numpy_batch["ego_future"]
        uncertainty_np = uncertainty.detach().cpu().numpy()
        for row in range(selected.shape[0]):
            valid_agents = pred_mask[row] > 0.0
            if not np.any(valid_agents):
                continue
            diff = selected[row, valid_agents, :, :2] - actual_future[row, valid_agents, :, :2]
            per_step = np.linalg.norm(diff, axis=-1)
            row_ade = float(np.mean(per_step))
            row_fde = float(np.mean(per_step[:, -1]))
            ade.append(row_ade)
            fde.append(row_fde)
            sample_uncertainty = float(uncertainty_np[row])
            uncertainty_values.append(sample_uncertainty)
            uncertainty_fde_values.append(row_fde)
            pred_min_distance = _future_min_distance(ego_future[row], selected[row], pred_mask[row])
            actual_min_distance = _future_min_distance(ego_future[row], actual_future[row], pred_mask[row])
            min_distance_errors.append(float(pred_min_distance - actual_min_distance))
            min_distance_abs_errors.append(abs(float(pred_min_distance - actual_min_distance)))
            pred_gap = _target_lane_gap(ego_future[row], selected[row], pred_mask[row], cfg)
            actual_gap = _target_lane_gap(ego_future[row], actual_future[row], pred_mask[row], cfg)
            if pred_gap < INF_TTC and actual_gap < INF_TTC:
                target_gap_errors.append(float(pred_gap - actual_gap))
                target_gap_abs_errors.append(abs(float(pred_gap - actual_gap)))
            states = _latest_states(history[batch_indices[row]], mask[batch_indices[row]])
            ego = next((state for state in states if state.vehicle_id == "ego"), None)
            if ego is not None:
                features = _forecast_features_from_prediction(ego, selected[row, valid_agents], sample_uncertainty, cfg)
                if bool(cfg.forecast_features.normalize):
                    features = augmentor._normalize(features)
                feature_rows.append(features)

    report = {
        "device": str(device),
        "checkpoint": str(checkpoint),
        "ensemble_size": int(payload.get("ensemble_size", len(models))) if isinstance(payload, dict) else len(models),
        "sample_count": int(len(feature_rows)),
        "ade": _summary(ade),
        "fde": _summary(fde),
        "future_min_distance_error": _summary(min_distance_errors),
        "future_min_distance_abs_error": _summary(min_distance_abs_errors),
        "target_lane_gap_error": _summary(target_gap_errors),
        "target_lane_gap_abs_error": _summary(target_gap_abs_errors),
        "uncertainty": _summary(uncertainty_values),
        "uncertainty_fde_correlation": _correlation(uncertainty_values, uncertainty_fde_values),
        "cv_baseline_validation": payload.get("cv_baseline_validation") if isinstance(payload, dict) else None,
        "ensemble_validation": payload.get("ensemble_validation") if isinstance(payload, dict) else None,
    }
    return np.asarray(feature_rows, dtype=np.float32), report


def _correlation(a_values: list[float], b_values: list[float]) -> float:
    a = np.asarray(a_values, dtype=np.float32)
    b = np.asarray(b_values, dtype=np.float32)
    mask = np.isfinite(a) & np.isfinite(b)
    a = a[mask]
    b = b[mask]
    if a.size < 2 or float(np.std(a)) <= 1.0e-8 or float(np.std(b)) <= 1.0e-8:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _forecast_conclusion(report: dict[str, Any]) -> dict[str, Any]:
    cv = report.get("cv_prediction", {})
    wcdt = report.get("wcdt_prediction", {})
    wcdt_v2 = report.get("wcdt_v2_prediction", {})
    behavior = report.get("forecast_behavior", {})
    cv_ade = float(cv.get("ade", {}).get("mean", 0.0))
    cv_fde = float(cv.get("fde", {}).get("mean", 0.0))
    cv_min_distance_error = float(cv.get("future_min_distance_abs_error", {}).get("mean", 1.0e6))
    wcdt_ade = float(wcdt.get("ade", {}).get("mean", 1.0e6))
    wcdt_fde = float(wcdt.get("fde", {}).get("mean", 1.0e6))
    quality_pass = bool(
        wcdt.get("available", False)
        and wcdt_ade <= max(cv_ade * 1.10, cv_ade + 1.0)
        and wcdt_fde <= max(cv_fde * 1.10, cv_fde + 1.0)
    )
    uncertainty_std = float(wcdt.get("uncertainty", {}).get("std", 0.0))
    confidence_corr = float(wcdt.get("confidence_fde_correlation", 0.0))
    uncertainty_pass = bool(wcdt.get("available", False) and uncertainty_std >= 0.02 and abs(confidence_corr) >= 0.10)
    wcdt_v2_fde = float(wcdt_v2.get("fde", {}).get("mean", 1.0e6))
    wcdt_v2_min_distance_error = float(wcdt_v2.get("future_min_distance_abs_error", {}).get("mean", 1.0e6))
    wcdt_v2_quality_pass = bool(
        wcdt_v2.get("available", False)
        and wcdt_v2_fde <= cv_fde
        and wcdt_v2_min_distance_error <= cv_min_distance_error
    )
    wcdt_v2_uncertainty_std = float(wcdt_v2.get("uncertainty", {}).get("std", 0.0))
    wcdt_v2_uncertainty_corr = float(wcdt_v2.get("uncertainty_fde_correlation", 0.0))
    wcdt_v2_uncertainty_pass = bool(
        wcdt_v2.get("available", False)
        and wcdt_v2_uncertainty_std > 0.02
        and wcdt_v2_uncertainty_corr > 0.0
    )
    return {
        "cv_vs_wcdt_action_agreement": float(behavior.get("step_action_agreement_rate", 0.0)),
        "wcdt_prediction_quality_pass": quality_pass,
        "wcdt_uncertainty_quality_pass": uncertainty_pass,
        "wcdt_recommended_for_stage5": bool(quality_pass and uncertainty_pass),
        "wcdt_v2_prediction_quality_pass": wcdt_v2_quality_pass,
        "wcdt_v2_uncertainty_quality_pass": wcdt_v2_uncertainty_pass,
        "wcdt_v2_recommended_for_stage5": bool(wcdt_v2_quality_pass and wcdt_v2_uncertainty_pass),
        "decision_basis": {
            "cv_ade_mean": cv_ade,
            "cv_fde_mean": cv_fde,
            "cv_future_min_distance_abs_error_mean": cv_min_distance_error,
            "wcdt_ade_mean": wcdt_ade,
            "wcdt_fde_mean": wcdt_fde,
            "wcdt_uncertainty_std": uncertainty_std,
            "wcdt_confidence_fde_correlation": confidence_corr,
            "wcdt_v2_fde_mean": wcdt_v2_fde,
            "wcdt_v2_future_min_distance_abs_error_mean": wcdt_v2_min_distance_error,
            "wcdt_v2_uncertainty_std": wcdt_v2_uncertainty_std,
            "wcdt_v2_uncertainty_fde_correlation": wcdt_v2_uncertainty_corr,
        },
    }


def _feature_distribution_report(cv_features: np.ndarray, wcdt_features: np.ndarray) -> dict[str, Any]:
    names = ForecastFeatureAugmentor.FEATURE_NAMES
    report: dict[str, Any] = {}
    count = min(cv_features.shape[0], wcdt_features.shape[0])
    cv_features = cv_features[:count]
    wcdt_features = wcdt_features[:count]
    for idx, name in enumerate(names):
        report[name] = {
            "cv": _summary(cv_features[:, idx]),
            "wcdt": _summary(wcdt_features[:, idx]),
            "wcdt_minus_cv": _summary(wcdt_features[:, idx] - cv_features[:, idx]),
        }
    return report


def _low_min_distance_replays(base_run_id: str, stage5_report: dict[str, Any], count: int) -> list[dict[str, Any]]:
    group = stage5_report.get("groups", {}).get("ppo_cv_features", {})
    episodes = sorted(group.get("episodes", []), key=lambda item: float(item.get("min_distance", INF_TTC)))
    rows = []
    for item in episodes[:count]:
        seed = int(item["seed"])
        replay_path = Path("safe_rl_output") / "runs" / base_run_id / "stage5" / "replay" / f"ppo_cv_features_seed_{seed}.json"
        shield_replay_path = Path("safe_rl_output") / "runs" / base_run_id / "stage5" / "replay" / f"cv_prediction_shield_seed_{seed}.json"
        rows.append(
            {
                "seed": seed,
                "min_distance": float(item.get("min_distance", INF_TTC)),
                "ttc_p1": float(item.get("ttc_p1", INF_TTC)),
                "drac_p99": float(item.get("drac_p99", 0.0)),
                "episode_reward": float(item.get("episode_reward", 0.0)),
                "replay": str(replay_path),
                "command": f"python -m safe_rl.tools.replay_episode --replay {replay_path} --gui --delay-ms 200",
                "shield_replay": str(shield_replay_path),
                "shield_command": f"python -m safe_rl.tools.replay_episode --replay {shield_replay_path} --gui --delay-ms 200",
            }
        )
    return rows


def _write_replay_commands(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Low-min-distance ppo_cv_features replay commands",
        "# Run one command at a time in PowerShell.",
        "",
    ]
    for row in rows:
        lines.append(f"# seed={row['seed']} min_distance={row['min_distance']:.3f} ttc_p1={row['ttc_p1']:.3f}")
        lines.append(row["command"])
        lines.append(f"# Compare with CV shield for the same seed")
        lines.append(row["shield_command"])
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _action_histogram(actions: list[int]) -> dict[str, int]:
    return {str(index): int(sum(1 for action in actions if int(action) == index)) for index in range(9)}


def _load_replay_actions(path: Path) -> list[int] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    return [int(action) for action in payload.get("actions", [])]


def _forecast_behavior_diagnostics(base_run_id: str, stage5_report: dict[str, Any]) -> dict[str, Any]:
    groups = stage5_report.get("groups", {})
    cv = groups.get("ppo_cv_features", {})
    wcdt = groups.get("ppo_wcdt_features", {})
    if not cv or not wcdt:
        return {"available": False, "reason": "missing ppo_cv_features or ppo_wcdt_features group"}
    replay_dir = Path("safe_rl_output") / "runs" / base_run_id / "stage5" / "replay"
    wcdt_by_seed = {int(item["seed"]): item for item in wcdt.get("episodes", [])}
    rows = []
    cv_actions_all: list[int] = []
    wcdt_actions_all: list[int] = []
    compared_steps = 0
    matching_steps = 0
    missing_replays = 0
    for cv_episode in cv.get("episodes", []):
        seed = int(cv_episode["seed"])
        if seed not in wcdt_by_seed:
            continue
        cv_actions = _load_replay_actions(replay_dir / f"ppo_cv_features_seed_{seed}.json")
        wcdt_actions = _load_replay_actions(replay_dir / f"ppo_wcdt_features_seed_{seed}.json")
        if cv_actions is None or wcdt_actions is None:
            missing_replays += 1
            continue
        cv_actions_all.extend(cv_actions)
        wcdt_actions_all.extend(wcdt_actions)
        limit = min(len(cv_actions), len(wcdt_actions))
        compared_steps += limit
        step_matches = sum(1 for idx in range(limit) if cv_actions[idx] == wcdt_actions[idx])
        matching_steps += step_matches
        first_diff = next((idx for idx in range(limit) if cv_actions[idx] != wcdt_actions[idx]), -1)
        rows.append(
            {
                "seed": seed,
                "cv_action_count": len(cv_actions),
                "wcdt_action_count": len(wcdt_actions),
                "exact_action_match": bool(len(cv_actions) == len(wcdt_actions) and step_matches == limit),
                "step_action_agreement_rate": float(step_matches / limit) if limit else 0.0,
                "first_diff_step": int(first_diff),
            }
        )
    exact_rates = [float(row["exact_action_match"]) for row in rows]
    return {
        "available": bool(rows),
        "compared_episode_count": int(len(rows)),
        "missing_replay_count": int(missing_replays),
        "exact_episode_action_match_rate": float(np.mean(exact_rates)) if exact_rates else 0.0,
        "step_action_agreement_rate": float(matching_steps / compared_steps) if compared_steps else 0.0,
        "cv_action_histogram": _action_histogram(cv_actions_all),
        "wcdt_action_histogram": _action_histogram(wcdt_actions_all),
        "episodes": rows,
        "action_sensitive_to_forecast_source": bool(matching_steps < compared_steps or np.mean(exact_rates) < 1.0)
        if rows
        else False,
    }


def run_forecast_diagnostics(
    cfg: Any,
    max_samples: int = 512,
    batch_size: int = 32,
    low_seed_count: int = 5,
) -> Path:
    base_run = run_root(cfg)
    stage1_path = stage_file(cfg, "stage1", str(cfg.stage1.output_name))
    checkpoint = stage_file(cfg, "stage2", "wcdt_predictor.pt")
    wcdt_v2_checkpoint = stage_file(cfg, "stage2", "wcdt_v2_predictor.pt")
    stage5_path = stage_file(cfg, "stage5", "formal_paired_eval_report.json")
    output_dir = base_run / "stage5" / "diagnostics"
    output_dir.mkdir(parents=True, exist_ok=True)
    data = np.load(stage1_path, allow_pickle=False)
    history = data["agent_history"]
    future = data["agent_future"]
    mask = data["agent_mask"]
    if history.shape[0] == 0:
        raise ValueError(f"no trajectory samples in {stage1_path}")
    sample_count = min(int(max_samples), int(history.shape[0]))
    rng = np.random.default_rng(int(cfg.run.seed))
    indices = np.sort(rng.choice(history.shape[0], size=sample_count, replace=False))
    cv_features = _cv_feature_matrix(cfg, history, mask, indices)
    cv_prediction = _cv_prediction_diagnostics(cfg, history, future, mask, indices)
    wcdt_features = np.zeros((0, ForecastFeatureAugmentor.feature_dim(cfg)), dtype=np.float32)
    wcdt_report: dict[str, Any] = {"available": False, "checkpoint": str(checkpoint)}
    wcdt_v2_features = np.zeros((0, ForecastFeatureAugmentor.feature_dim(cfg)), dtype=np.float32)
    wcdt_v2_report: dict[str, Any] = {"available": False, "checkpoint": str(wcdt_v2_checkpoint)}
    if checkpoint.exists():
        wcdt_features, wcdt_report = _wcdt_diagnostics(cfg, checkpoint, history, future, mask, indices, batch_size)
        wcdt_report["available"] = True
        initial_report_path = base_run / "stage2" / "stage2_initial_prediction_report.json"
        if not initial_report_path.exists() and wcdt_report.get("checkpoint_loss_history"):
            write_report(
                initial_report_path,
                {
                    "stage": "stage2_initial_prediction",
                    "run_id": str(cfg.run.run_id),
                    "input_stage1": str(stage1_path),
                    "prediction_checkpoint": str(checkpoint),
                    "prediction_loss_history": wcdt_report["checkpoint_loss_history"],
                    "prediction_loss_summary": wcdt_report.get("checkpoint_loss_summary"),
                    "recovered_from_checkpoint": True,
                },
            )
        if initial_report_path.exists():
            wcdt_report["initial_prediction_report"] = str(initial_report_path)
    if wcdt_v2_checkpoint.exists():
        wcdt_v2_features, wcdt_v2_report = _wcdt_v2_diagnostics(
            cfg, wcdt_v2_checkpoint, history, future, mask, indices, batch_size
        )
        wcdt_v2_report["available"] = True
    report: dict[str, Any] = {
        "run_id": str(cfg.run.run_id),
        "stage1_buffer": str(stage1_path),
        "sample_count": int(sample_count),
        "feature_names": list(ForecastFeatureAugmentor.FEATURE_NAMES),
        "cv_feature_summary": {
            name: _summary(cv_features[:, idx])
            for idx, name in enumerate(ForecastFeatureAugmentor.FEATURE_NAMES)
        },
        "cv_prediction": cv_prediction,
        "wcdt_prediction": wcdt_report,
        "wcdt_v2_prediction": wcdt_v2_report,
    }
    if wcdt_features.shape[0] > 0:
        report["cv_vs_wcdt_feature_distribution"] = _feature_distribution_report(cv_features, wcdt_features)
    if wcdt_v2_features.shape[0] > 0:
        report["cv_vs_wcdt_v2_feature_distribution"] = _feature_distribution_report(cv_features, wcdt_v2_features)
    if stage5_path.exists():
        with stage5_path.open("r", encoding="utf-8") as file:
            stage5_report = json.load(file)
        low_rows = _low_min_distance_replays(str(cfg.run.run_id), stage5_report, int(low_seed_count))
        report["low_min_distance_ppo_cv_features"] = low_rows
        report["forecast_behavior"] = _forecast_behavior_diagnostics(str(cfg.run.run_id), stage5_report)
        _write_replay_commands(output_dir / "replay_low_min_distance_ppo_cv_features.ps1", low_rows)
    report["forecast_conclusion"] = _forecast_conclusion(report)
    output_path = output_dir / "forecast_diagnostics.json"
    write_report(output_path, report)
    return output_path
