from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from safe_rl.pipeline.common import make_env, run_root, stage_file, write_report
from safe_rl.prediction.forecast_feature_augmentor import (
    ForecastFeatureAugmentor,
    forecast_target_lane_gap_from_trajectories,
)
from safe_rl.prediction.sumo_wcdt_adapter import SumoWcDTAdapter
from safe_rl.prediction.wcdt_v2_predictor import build_v2_numpy_batch, ensemble_predict, load_v2_ensemble, tensorize_batch
from safe_rl.prediction.wcdt_v3_predictor import (
    build_v3_numpy_batch,
    ensemble_predict_v3,
    load_v3_ensemble,
    tensorize_v3_batch,
)
from safe_rl.rl.ppo import load_ppo
from safe_rl.risk.merge_local import route_aware_constant_velocity_rollout
from safe_rl.sim.metrics import INF_TTC
from safe_rl.sim.scenario_semantics import (
    EDGE_ROLE_AUXILIARY,
    EDGE_ROLE_MAINLINE,
    EDGE_ROLE_RAMP,
    EDGE_ROLE_TARGET,
    auxiliary_edges,
    infer_lane_index,
    infer_route_position,
    mainline_edges,
    ramp_edges,
    taper_edge,
)
from safe_rl.sim.types import VehicleState
from safe_rl.utils.config import clone_with_overrides


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


def _vector_to_state(
    vehicle_id: str,
    vector: np.ndarray,
    cfg: Any,
    lane_index: int | None = None,
    edge_role_id: int | None = None,
) -> VehicleState:
    x, y, heading, speed, accel = [float(item) for item in vector[:5]]
    lane_index = infer_lane_index(cfg, y) if lane_index is None or int(lane_index) < 0 else int(lane_index)
    candidate_edges = None
    if edge_role_id == EDGE_ROLE_RAMP:
        candidate_edges = ramp_edges(cfg)
    elif edge_role_id == EDGE_ROLE_AUXILIARY:
        candidate_edges = auxiliary_edges(cfg)
    elif edge_role_id in (EDGE_ROLE_MAINLINE, EDGE_ROLE_TARGET):
        candidate_edges = mainline_edges(cfg)
    edge_id, lane_pos = infer_route_position(cfg, x, y, lane_index, edge_ids=candidate_edges)
    if edge_id is None:
        edge_id = taper_edge(cfg)
        lane_pos = max(0.0, x)
    return VehicleState(
        vehicle_id=vehicle_id,
        x=x,
        y=y,
        heading=heading,
        speed=speed,
        lane_index=lane_index,
        lane_id=f"{edge_id}_{lane_index}",
        lane_pos=lane_pos,
        edge_id=edge_id,
        accel=accel,
    )


def _latest_states(
    cfg: Any,
    history: np.ndarray,
    mask: np.ndarray,
    lane_indices: np.ndarray | None = None,
    edge_roles: np.ndarray | None = None,
) -> list[VehicleState]:
    states: list[VehicleState] = []
    for agent_idx in range(history.shape[0]):
        if float(mask[agent_idx]) <= 0.0:
            continue
        vehicle_id = "ego" if agent_idx == 0 else f"agent_{agent_idx}"
        states.append(
            _vector_to_state(
                vehicle_id,
                history[agent_idx, -1],
                cfg,
                None if lane_indices is None else int(lane_indices[agent_idx]),
                None if edge_roles is None else int(edge_roles[agent_idx]),
            )
        )
    return states


def _cv_feature_matrix(
    cfg: Any,
    history: np.ndarray,
    mask: np.ndarray,
    indices: np.ndarray,
    lane_indices: np.ndarray | None = None,
    edge_roles: np.ndarray | None = None,
) -> np.ndarray:
    augmentor = ForecastFeatureAugmentor(cfg)
    rows = []
    for sample_idx in indices:
        states = _latest_states(
            cfg,
            history[sample_idx],
            mask[sample_idx],
            None if lane_indices is None else lane_indices[sample_idx],
            None if edge_roles is None else edge_roles[sample_idx],
        )
        ego = next((state for state in states if state.vehicle_id == "ego"), None)
        rows.append(augmentor.extract({"ego": ego, "vehicles": states, "config": cfg}))
    return np.asarray(rows, dtype=np.float32)


def _constant_velocity_future(
    last: np.ndarray,
    horizon: int,
    dt: float,
    cfg: Any,
    lane_index: int | None = None,
    edge_role_id: int | None = None,
) -> np.ndarray:
    state = _vector_to_state("_cv", last, cfg, lane_index, edge_role_id)
    rollout = route_aware_constant_velocity_rollout(state, horizon, dt, cfg)[0]
    return np.asarray([item.as_vector() for item in rollout], dtype=np.float32)


def _target_role_gap_abs_errors(
    ego_future: np.ndarray,
    pred_future: np.ndarray,
    actual_future: np.ndarray,
    mask: np.ndarray,
    role_ids: np.ndarray,
) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    ego_x = np.asarray(ego_future, dtype=np.float32)[:, 0]
    for name, role_id in (("target_lane_front_gap_abs_error", 0), ("target_lane_rear_gap_abs_error", 1)):
        role_indices = np.where((np.asarray(mask) > 0.0) & (np.asarray(role_ids) == role_id))[0]
        if role_indices.size == 0:
            result[name] = None
            continue
        pred_gap = np.asarray(pred_future, dtype=np.float32)[role_indices, :, 0] - ego_x[None, :]
        actual_gap = np.asarray(actual_future, dtype=np.float32)[role_indices, :, 0] - ego_x[None, :]
        result[name] = float(np.mean(np.abs(pred_gap - actual_gap)))
    return result


def _cv_prediction_diagnostics(
    cfg: Any,
    history: np.ndarray,
    future: np.ndarray,
    mask: np.ndarray,
    indices: np.ndarray,
    lane_indices: np.ndarray | None = None,
    edge_roles: np.ndarray | None = None,
) -> dict[str, Any]:
    horizon = int(min(future.shape[2], cfg.forecast_features.get("horizon_steps", cfg.scenario.forecast_horizon_steps)))
    dt = float(cfg.scenario.step_length)
    ade: list[float] = []
    fde: list[float] = []
    min_distance_errors: list[float] = []
    min_distance_abs_errors: list[float] = []
    target_gap_errors: list[float] = []
    target_gap_abs_errors: list[float] = []
    target_front_gap_abs_errors: list[float] = []
    target_rear_gap_abs_errors: list[float] = []
    for sample_idx in indices:
        actual_future = future[sample_idx, :, :horizon]
        pred_future = np.zeros_like(actual_future)
        for agent_idx in range(history.shape[1]):
            if float(mask[sample_idx, agent_idx]) <= 0.0:
                continue
            pred_future[agent_idx] = _constant_velocity_future(
                history[sample_idx, agent_idx, -1],
                horizon,
                dt,
                cfg,
                None if lane_indices is None else int(lane_indices[sample_idx, agent_idx]),
                None if edge_roles is None else int(edge_roles[sample_idx, agent_idx]),
            )
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
    selected_batch = build_v2_numpy_batch(
        cfg,
        history,
        future,
        mask,
        indices,
        lane_indices=lane_indices,
        edge_roles=edge_roles,
    )
    for row in range(selected_batch["baseline"].shape[0]):
        role_gap_errors = _target_role_gap_abs_errors(
            selected_batch["ego_future"][row],
            selected_batch["baseline"][row],
            selected_batch["target"][row],
            selected_batch["mask"][row],
            selected_batch["role_ids"][row],
        )
        if role_gap_errors["target_lane_front_gap_abs_error"] is not None:
            target_front_gap_abs_errors.append(float(role_gap_errors["target_lane_front_gap_abs_error"]))
        if role_gap_errors["target_lane_rear_gap_abs_error"] is not None:
            target_rear_gap_abs_errors.append(float(role_gap_errors["target_lane_rear_gap_abs_error"]))
    return {
        "sample_count": int(len(ade)),
        "ade": _summary(ade),
        "fde": _summary(fde),
        "future_min_distance_error": _summary(min_distance_errors),
        "future_min_distance_abs_error": _summary(min_distance_abs_errors),
        "target_lane_gap_error": _summary(target_gap_errors),
        "target_lane_gap_abs_error": _summary(target_gap_abs_errors),
        "target_lane_front_gap_abs_error": _summary(target_front_gap_abs_errors),
        "target_lane_rear_gap_abs_error": _summary(target_rear_gap_abs_errors),
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
    return forecast_target_lane_gap_from_trajectories(
        ego_future,
        other_future[np.asarray(other_mask) > 0.0],
        cfg,
        default_gap=INF_TTC,
    )


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
    ego_rollout = route_aware_constant_velocity_rollout(ego_state, horizon, dt, cfg)[0]
    ego_future = np.asarray([[state.x, state.y] for state in ego_rollout], dtype=np.float32)
    target_lane_gap = forecast_target_lane_gap_from_trajectories(ego_future, trajectories, cfg)
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
            target_lane_gap,
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
    lane_indices: np.ndarray | None = None,
    edge_roles: np.ndarray | None = None,
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
                sample_idx = batch_indices[row]
                states = _latest_states(
                    cfg,
                    history[sample_idx],
                    mask[sample_idx],
                    None if lane_indices is None else lane_indices[sample_idx],
                    None if edge_roles is None else edge_roles[sample_idx],
                )
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


def _residual_ensemble_diagnostics(
    cfg: Any,
    checkpoint: Path,
    history: np.ndarray,
    future: np.ndarray,
    mask: np.ndarray,
    indices: np.ndarray,
    batch_size: int,
    lane_indices: np.ndarray | None = None,
    edge_roles: np.ndarray | None = None,
    *,
    build_batch: Any,
    tensorize: Any,
    ensemble_fn: Any,
    load_ensemble: Any,
    comparison_summary_key: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    torch = _require_torch()
    device = _resolve_device(cfg, torch)
    models, payload, device = load_ensemble(cfg, checkpoint, device)
    augmentor = ForecastFeatureAugmentor(cfg)
    feature_rows: list[np.ndarray] = []
    ade: list[float] = []
    fde: list[float] = []
    min_distance_errors: list[float] = []
    min_distance_abs_errors: list[float] = []
    target_gap_errors: list[float] = []
    target_gap_abs_errors: list[float] = []
    target_front_gap_abs_errors: list[float] = []
    target_rear_gap_abs_errors: list[float] = []
    uncertainty_values: list[float] = []
    uncertainty_fde_values: list[float] = []
    uncertainty_min_distance_error_values: list[float] = []

    for start in range(0, indices.shape[0], batch_size):
        batch_indices = indices[start : start + batch_size]
        numpy_batch = build_batch(
            cfg,
            history,
            future,
            mask,
            batch_indices,
            lane_indices=lane_indices,
            edge_roles=edge_roles,
        )
        tensor_batch = tensorize(numpy_batch, torch, device)
        pred, uncertainty = ensemble_fn(models, tensor_batch)
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
            min_distance_abs_error = abs(float(pred_min_distance - actual_min_distance))
            min_distance_abs_errors.append(min_distance_abs_error)
            uncertainty_min_distance_error_values.append(min_distance_abs_error)
            role_gap_errors = _target_role_gap_abs_errors(
                ego_future[row],
                selected[row],
                actual_future[row],
                pred_mask[row],
                numpy_batch["role_ids"][row],
            )
            if role_gap_errors["target_lane_front_gap_abs_error"] is not None:
                target_front_gap_abs_errors.append(float(role_gap_errors["target_lane_front_gap_abs_error"]))
            if role_gap_errors["target_lane_rear_gap_abs_error"] is not None:
                target_rear_gap_abs_errors.append(float(role_gap_errors["target_lane_rear_gap_abs_error"]))
            pred_gap = _target_lane_gap(ego_future[row], selected[row], pred_mask[row], cfg)
            actual_gap = _target_lane_gap(ego_future[row], actual_future[row], pred_mask[row], cfg)
            if pred_gap < INF_TTC and actual_gap < INF_TTC:
                target_gap_errors.append(float(pred_gap - actual_gap))
                target_gap_abs_errors.append(abs(float(pred_gap - actual_gap)))
            sample_idx = batch_indices[row]
            states = _latest_states(
                cfg,
                history[sample_idx],
                mask[sample_idx],
                None if lane_indices is None else lane_indices[sample_idx],
                None if edge_roles is None else edge_roles[sample_idx],
            )
            ego = next((state for state in states if state.vehicle_id == "ego"), None)
            if ego is not None:
                features = _forecast_features_from_prediction(ego, selected[row, valid_agents], sample_uncertainty, cfg)
                if bool(cfg.forecast_features.normalize):
                    features = augmentor._normalize(features)
                feature_rows.append(features)

    architecture_version = payload.get("architecture_version") if isinstance(payload, dict) else None
    loss_version = payload.get("loss_version") if isinstance(payload, dict) else None
    member_histories = payload.get("member_histories", []) if isinstance(payload, dict) else []
    report = {
        "device": str(device),
        "checkpoint": str(checkpoint),
        "ensemble_size": int(payload.get("ensemble_size", len(models))) if isinstance(payload, dict) else len(models),
        "architecture_version": architecture_version,
        "loss_version": loss_version,
        "legacy_checkpoint_metadata": not bool(architecture_version and loss_version),
        "early_stopped_member_count": int(
            sum(bool(item.get("stopped_early", False)) for item in member_histories if isinstance(item, dict))
        ),
        "sample_count": int(len(feature_rows)),
        "ade": _summary(ade),
        "fde": _summary(fde),
        "future_min_distance_error": _summary(min_distance_errors),
        "future_min_distance_abs_error": _summary(min_distance_abs_errors),
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
        "cv_baseline_validation": payload.get("cv_baseline_validation") if isinstance(payload, dict) else None,
        "ensemble_validation": payload.get("ensemble_validation") if isinstance(payload, dict) else None,
        comparison_summary_key: payload.get(comparison_summary_key) if isinstance(payload, dict) else None,
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
    lane_indices: np.ndarray | None = None,
    edge_roles: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    return _residual_ensemble_diagnostics(
        cfg,
        checkpoint,
        history,
        future,
        mask,
        indices,
        batch_size,
        lane_indices,
        edge_roles,
        build_batch=build_v2_numpy_batch,
        tensorize=tensorize_batch,
        ensemble_fn=ensemble_predict,
        load_ensemble=load_v2_ensemble,
        comparison_summary_key="wcdt_v2_vs_cv_summary",
    )


def _wcdt_v3_diagnostics(
    cfg: Any,
    checkpoint: Path,
    history: np.ndarray,
    future: np.ndarray,
    mask: np.ndarray,
    indices: np.ndarray,
    batch_size: int,
    lane_indices: np.ndarray | None = None,
    edge_roles: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    return _residual_ensemble_diagnostics(
        cfg,
        checkpoint,
        history,
        future,
        mask,
        indices,
        batch_size,
        lane_indices,
        edge_roles,
        build_batch=build_v3_numpy_batch,
        tensorize=tensorize_v3_batch,
        ensemble_fn=ensemble_predict_v3,
        load_ensemble=load_v3_ensemble,
        comparison_summary_key="wcdt_v3_vs_cv_summary",
    )


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
    wcdt_v3 = report.get("wcdt_v3_prediction", {})
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
    cv_v2_baseline = wcdt_v2.get("cv_baseline_validation")
    if not isinstance(cv_v2_baseline, dict):
        cv_v2_baseline = {}

    def _cv_role_gap_mean(name: str) -> float:
        diagnostics_summary = cv.get(name, {})
        if isinstance(diagnostics_summary, dict) and "mean" in diagnostics_summary:
            return float(diagnostics_summary["mean"])
        checkpoint_summary = cv_v2_baseline.get(name, {})
        return float(checkpoint_summary.get("mean", 1.0e6)) if isinstance(checkpoint_summary, dict) else 1.0e6

    cv_front_gap_error = _cv_role_gap_mean("target_lane_front_gap_abs_error")
    cv_rear_gap_error = _cv_role_gap_mean("target_lane_rear_gap_abs_error")
    wcdt_v2_front_gap_error = float(wcdt_v2.get("target_lane_front_gap_abs_error", {}).get("mean", 1.0e6))
    wcdt_v2_rear_gap_error = float(wcdt_v2.get("target_lane_rear_gap_abs_error", {}).get("mean", 1.0e6))
    wcdt_v2_quality_pass = bool(
        wcdt_v2.get("available", False)
        and wcdt_v2_fde <= cv_fde
        and wcdt_v2_min_distance_error <= cv_min_distance_error
        and wcdt_v2_front_gap_error <= cv_front_gap_error
        and wcdt_v2_rear_gap_error <= cv_rear_gap_error
    )
    wcdt_v2_uncertainty_std = float(wcdt_v2.get("uncertainty", {}).get("std", 0.0))
    wcdt_v2_uncertainty_corr = float(wcdt_v2.get("uncertainty_fde_correlation", 0.0))
    wcdt_v2_uncertainty_min_distance_corr = float(
        wcdt_v2.get("uncertainty_future_min_distance_abs_error_correlation", 0.0)
    )
    wcdt_v2_uncertainty_pass = bool(
        wcdt_v2.get("available", False)
        and wcdt_v2_uncertainty_std > 0.02
        and (wcdt_v2_uncertainty_corr > 0.0 or wcdt_v2_uncertainty_min_distance_corr > 0.0)
    )
    sensitivity = report.get("policy_feature_sensitivity", {})
    wcdt_v2_sensitivity = sensitivity.get("groups", {}).get("ppo_wcdt_v2_features", {})
    wcdt_v2_action_sensitive = bool(wcdt_v2_sensitivity.get("action_sensitive_to_forecast_features", False))
    forecast_policy_underutilized = bool(
        wcdt_v2_quality_pass
        and wcdt_v2_uncertainty_pass
        and wcdt_v2_sensitivity.get("available", False)
        and not wcdt_v2_action_sensitive
    )
    wcdt_v3_fde = float(wcdt_v3.get("fde", {}).get("mean", 1.0e6))
    wcdt_v3_min_distance_error = float(wcdt_v3.get("future_min_distance_abs_error", {}).get("mean", 1.0e6))
    wcdt_v3_front_gap_error = float(wcdt_v3.get("target_lane_front_gap_abs_error", {}).get("mean", 1.0e6))
    wcdt_v3_rear_gap_error = float(wcdt_v3.get("target_lane_rear_gap_abs_error", {}).get("mean", 1.0e6))
    wcdt_v3_uncertainty_std = float(wcdt_v3.get("uncertainty", {}).get("std", 0.0))
    wcdt_v3_uncertainty_fde_corr = float(wcdt_v3.get("uncertainty_fde_correlation", 0.0))
    wcdt_v3_uncertainty_min_distance_corr = float(
        wcdt_v3.get("uncertainty_future_min_distance_abs_error_correlation", 0.0)
    )
    wcdt_v3_prediction_pass = bool(
        wcdt_v3.get("available", False)
        and wcdt_v3_fde <= wcdt_v2_fde
        and wcdt_v3_min_distance_error < wcdt_v2_min_distance_error
        and wcdt_v3_front_gap_error < wcdt_v2_front_gap_error
        and wcdt_v3_rear_gap_error < wcdt_v2_rear_gap_error
    )
    wcdt_v3_uncertainty_pass = bool(
        wcdt_v3.get("available", False)
        and wcdt_v3_uncertainty_std > 0.02
        and (wcdt_v3_uncertainty_fde_corr > 0.0 or wcdt_v3_uncertainty_min_distance_corr > 0.0)
    )
    return {
        "cv_vs_wcdt_action_agreement": float(behavior.get("step_action_agreement_rate", 0.0)),
        "wcdt_prediction_quality_pass": quality_pass,
        "wcdt_uncertainty_quality_pass": uncertainty_pass,
        "wcdt_recommended_for_stage5": bool(quality_pass and uncertainty_pass),
        "wcdt_v2_prediction_quality_pass": wcdt_v2_quality_pass,
        "wcdt_v2_uncertainty_quality_pass": wcdt_v2_uncertainty_pass,
        "wcdt_v2_recommended_for_stage5": bool(wcdt_v2_quality_pass and wcdt_v2_uncertainty_pass),
        "wcdt_v2_policy_feature_sensitive": wcdt_v2_action_sensitive,
        "wcdt_v3_prediction_quality_pass": wcdt_v3_prediction_pass,
        "wcdt_v3_uncertainty_quality_pass": wcdt_v3_uncertainty_pass,
        "wcdt_v3_candidate_for_promotion": bool(wcdt_v3_prediction_pass and wcdt_v3_uncertainty_pass),
        "forecast_policy_underutilized": forecast_policy_underutilized,
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
            "cv_target_lane_front_gap_abs_error_mean": cv_front_gap_error,
            "cv_target_lane_rear_gap_abs_error_mean": cv_rear_gap_error,
            "wcdt_v2_target_lane_front_gap_abs_error_mean": wcdt_v2_front_gap_error,
            "wcdt_v2_target_lane_rear_gap_abs_error_mean": wcdt_v2_rear_gap_error,
            "wcdt_v2_uncertainty_std": wcdt_v2_uncertainty_std,
            "wcdt_v2_uncertainty_fde_correlation": wcdt_v2_uncertainty_corr,
            "wcdt_v2_uncertainty_future_min_distance_abs_error_correlation": wcdt_v2_uncertainty_min_distance_corr,
            "wcdt_v2_original_vs_zeroed_action_agreement_rate": wcdt_v2_sensitivity.get(
                "original_vs_zeroed_action_agreement_rate"
            ),
            "wcdt_v2_original_vs_shuffled_action_agreement_rate": wcdt_v2_sensitivity.get(
                "original_vs_shuffled_action_agreement_rate"
            ),
            "wcdt_v3_fde_mean": wcdt_v3_fde,
            "wcdt_v3_future_min_distance_abs_error_mean": wcdt_v3_min_distance_error,
            "wcdt_v3_target_lane_front_gap_abs_error_mean": wcdt_v3_front_gap_error,
            "wcdt_v3_target_lane_rear_gap_abs_error_mean": wcdt_v3_rear_gap_error,
            "wcdt_v3_uncertainty_std": wcdt_v3_uncertainty_std,
            "wcdt_v3_uncertainty_fde_correlation": wcdt_v3_uncertainty_fde_corr,
            "wcdt_v3_uncertainty_future_min_distance_abs_error_correlation": wcdt_v3_uncertainty_min_distance_corr,
        },
    }


def _feature_distribution_report(
    cv_features: np.ndarray,
    wcdt_features: np.ndarray,
    *,
    left_label: str = "cv",
    right_label: str = "wcdt",
) -> dict[str, Any]:
    names = ForecastFeatureAugmentor.FEATURE_NAMES
    report: dict[str, Any] = {}
    count = min(cv_features.shape[0], wcdt_features.shape[0])
    cv_features = cv_features[:count]
    wcdt_features = wcdt_features[:count]
    for idx, name in enumerate(names):
        report[name] = {
            left_label: _summary(cv_features[:, idx]),
            right_label: _summary(wcdt_features[:, idx]),
            f"{right_label}_minus_{left_label}": _summary(wcdt_features[:, idx] - cv_features[:, idx]),
        }
    return report


def _feature_source_summary(features_by_source: dict[str, np.ndarray]) -> dict[str, Any]:
    names = ForecastFeatureAugmentor.FEATURE_NAMES
    sources: dict[str, Any] = {}
    highlights: dict[str, Any] = {}
    equal_rates: dict[str, float] = {}
    warnings: list[str] = []
    min_idx = names.index("forecast_min_distance")
    gap_idx = names.index("forecast_merge_gap")
    uncertainty_idx = names.index("forecast_uncertainty")
    for source, features in features_by_source.items():
        features = np.asarray(features, dtype=np.float32)
        if features.ndim != 2 or features.shape[0] == 0:
            sources[source] = {"available": False, "sample_count": 0}
            highlights[source] = {"available": False}
            equal_rates[source] = 0.0
            continue
        sources[source] = {
            "available": True,
            "sample_count": int(features.shape[0]),
            "features": {name: _summary(features[:, idx]) for idx, name in enumerate(names)},
        }
        highlights[source] = {
            "available": True,
            "forecast_min_distance": _summary(features[:, min_idx]),
            "forecast_merge_gap": _summary(features[:, gap_idx]),
            "forecast_uncertainty": _summary(features[:, uncertainty_idx]),
        }
        equal_rate = float(np.mean(np.isclose(features[:, min_idx], features[:, gap_idx], atol=1.0e-6)))
        equal_rates[source] = equal_rate
        if source in {"wcdt", "wcdt_v2", "wcdt_v3"} and equal_rate > 0.95:
            warnings.append(
                f"{source}: forecast_merge_gap equals forecast_min_distance for {equal_rate:.2%} of samples"
            )
    pairwise_abs_difference: dict[str, Any] = {}
    available_sources = [
        source
        for source, features in features_by_source.items()
        if np.asarray(features).ndim == 2 and np.asarray(features).shape[0] > 0
    ]
    for left_idx, left in enumerate(available_sources):
        for right in available_sources[left_idx + 1 :]:
            left_features = np.asarray(features_by_source[left], dtype=np.float32)
            right_features = np.asarray(features_by_source[right], dtype=np.float32)
            count = min(left_features.shape[0], right_features.shape[0])
            if count <= 0:
                continue
            diff = np.abs(left_features[:count] - right_features[:count])
            pairwise_abs_difference[f"{left}_vs_{right}"] = {
                "sample_count": int(count),
                "features": {name: _summary(diff[:, idx]) for idx, name in enumerate(names)},
                "highlight": {
                    "forecast_min_distance": _summary(diff[:, min_idx]),
                    "forecast_merge_gap": _summary(diff[:, gap_idx]),
                    "forecast_uncertainty": _summary(diff[:, uncertainty_idx]),
                },
            }
    return {
        "feature_names": list(names),
        "sources": sources,
        "pairwise_abs_difference": pairwise_abs_difference,
        "highlight": highlights,
        "forecast_merge_gap_equals_min_distance_rate": equal_rates,
        "runtime_diagnostics_feature_semantics_consistent": True,
        "warnings": warnings,
    }


def _low_min_distance_replays(
    base_run_id: str,
    stage5_report: dict[str, Any],
    count: int,
    *,
    group_name: str = "ppo_cv_features",
    compare_group_name: str | None = "cv_prediction_shield",
) -> list[dict[str, Any]]:
    group = stage5_report.get("groups", {}).get(group_name, {})
    episodes = sorted(group.get("episodes", []), key=lambda item: float(item.get("min_distance", INF_TTC)))
    rows = []
    for item in episodes[:count]:
        seed = int(item["seed"])
        replay_path = Path("safe_rl_output") / "runs" / base_run_id / "stage5" / "replay" / f"{group_name}_seed_{seed}.json"
        row = {
            "seed": seed,
            "group": group_name,
            "min_distance": float(item.get("min_distance", INF_TTC)),
            "ttc_p1": float(item.get("ttc_p1", INF_TTC)),
            "drac_p99": float(item.get("drac_p99", 0.0)),
            "episode_reward": float(item.get("episode_reward", 0.0)),
            "replay": str(replay_path),
            "command": f"python -m safe_rl.tools.replay_episode --replay {replay_path} --gui --delay-ms 200",
        }
        if compare_group_name:
            compare_path = (
                Path("safe_rl_output") / "runs" / base_run_id / "stage5" / "replay" / f"{compare_group_name}_seed_{seed}.json"
            )
            row["compare_group"] = compare_group_name
            row["compare_replay"] = str(compare_path)
            row["compare_command"] = f"python -m safe_rl.tools.replay_episode --replay {compare_path} --gui --delay-ms 200"
        rows.append(row)
    return rows


def _write_replay_commands(path: Path, rows: list[dict[str, Any]], *, title: str) -> None:
    lines = [
        f"# {title}",
        "# Run one command at a time in PowerShell.",
        "",
    ]
    for row in rows:
        lines.append(
            f"# group={row.get('group', '')} seed={row['seed']} "
            f"min_distance={row['min_distance']:.3f} ttc_p1={row['ttc_p1']:.3f}"
        )
        lines.append(row["command"])
        if row.get("compare_command"):
            lines.append(f"# Compare with {row.get('compare_group')} for the same seed")
            lines.append(row["compare_command"])
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _action_histogram(actions: list[int]) -> dict[str, int]:
    return {str(index): int(sum(1 for action in actions if int(action) == index)) for index in range(9)}


def _first_diff_step_summary(left_actions: list[int], right_actions: list[int]) -> dict[str, Any]:
    limit = min(len(left_actions), len(right_actions))
    if limit <= 0:
        return _summary([])
    first_diff = next((idx for idx in range(limit) if int(left_actions[idx]) != int(right_actions[idx])), -1)
    return _summary([] if first_diff < 0 else [float(first_diff)])


def _action_agreement(left_actions: list[int], right_actions: list[int]) -> float:
    limit = min(len(left_actions), len(right_actions))
    if limit <= 0:
        return 0.0
    return float(sum(1 for idx in range(limit) if int(left_actions[idx]) == int(right_actions[idx])) / limit)


def _policy_feature_sensitivity_from_actions(
    original_actions: list[int],
    zeroed_actions: list[int],
    shuffled_actions: list[int],
) -> dict[str, Any]:
    zeroed_agreement = _action_agreement(original_actions, zeroed_actions)
    shuffled_agreement = _action_agreement(original_actions, shuffled_actions)
    return {
        "available": bool(original_actions),
        "state_count": int(len(original_actions)),
        "original_vs_zeroed_action_agreement_rate": zeroed_agreement,
        "original_vs_shuffled_action_agreement_rate": shuffled_agreement,
        "original_action_histogram": _action_histogram(original_actions),
        "zeroed_action_histogram": _action_histogram(zeroed_actions),
        "shuffled_action_histogram": _action_histogram(shuffled_actions),
        "first_diff_zeroed_step_summary": _first_diff_step_summary(original_actions, zeroed_actions),
        "first_diff_shuffled_step_summary": _first_diff_step_summary(original_actions, shuffled_actions),
        "action_sensitive_to_forecast_features": bool(zeroed_agreement < 0.98 or shuffled_agreement < 0.98),
    }


def _mutate_forecast_observation(obs: np.ndarray, feature_dim: int, mode: str) -> np.ndarray:
    output = np.asarray(obs, dtype=np.float32).copy()
    if feature_dim <= 0 or output.shape[-1] <= feature_dim:
        return output
    start = output.shape[-1] - feature_dim
    if mode == "zeroed":
        output[..., start:] = 0.0
    elif mode == "shuffled":
        output[..., start:] = np.roll(output[..., start:], shift=1, axis=-1)
    return output


def _forecast_policy_specs(base_run_id: str) -> dict[str, dict[str, str]]:
    base = Path("safe_rl_output") / "runs"
    return {
        "ppo_cv_features": {
            "source": "constant_velocity",
            "model_path": str(base / f"{base_run_id}_forecast_cv" / "stage3" / "ppo_model.zip"),
            "checkpoint": "",
        },
        "ppo_wcdt_features": {
            "source": "wcdt",
            "model_path": str(base / f"{base_run_id}_forecast_wcdt" / "stage3" / "ppo_model.zip"),
            "checkpoint": str(base / base_run_id / "stage2" / "wcdt_predictor.pt"),
        },
        "ppo_wcdt_v2_features": {
            "source": "wcdt_v2",
            "model_path": str(base / f"{base_run_id}_forecast_wcdt_v2" / "stage3" / "ppo_model.zip"),
            "checkpoint": str(base / base_run_id / "stage2" / "wcdt_v2_predictor.pt"),
        },
        "ppo_wcdt_v3_features": {
            "source": "wcdt_v3",
            "model_path": str(base / f"{base_run_id}_forecast_wcdt_v3" / "stage3" / "ppo_model.zip"),
            "checkpoint": str(base / base_run_id / "stage2" / "wcdt_v3_predictor.pt"),
        },
    }


def _policy_feature_sensitivity(
    cfg: Any,
    base_run_id: str,
    stage5_report: dict[str, Any],
    *,
    seed_count: int,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    feature_dim = ForecastFeatureAugmentor.feature_dim(cfg)
    device = str(cfg.get("training", {}).get("device", "auto") or "auto")
    if device.lower() == "gpu":
        device = "cuda"
    for group_name, spec in _forecast_policy_specs(base_run_id).items():
        model_path = _resolve(spec["model_path"])
        checkpoint = _resolve(spec["checkpoint"]) if spec.get("checkpoint") else None
        if not model_path.exists():
            output[group_name] = {"available": False, "reason": f"missing model checkpoint: {model_path}"}
            continue
        if checkpoint is not None and not checkpoint.exists():
            output[group_name] = {"available": False, "reason": f"missing forecast checkpoint: {checkpoint}"}
            continue
        forecast_overrides = {
            "enabled": True,
            "source": spec["source"],
            "allow_heuristic_fallback": False,
        }
        if checkpoint is not None:
            forecast_overrides["checkpoint"] = str(checkpoint)
        group_cfg = clone_with_overrides(
            cfg,
            {
                "forecast_features": forecast_overrides,
                "rl": {"use_wcdt_forecast_features": True},
                "shield": {"enabled": False},
            },
        )
        model = load_ppo(model_path, device=device)
        seeds = [
            int(item["seed"])
            for item in stage5_report.get("groups", {}).get(group_name, {}).get("episodes", [])
            if "seed" in item
        ]
        if not seeds:
            seeds = list(range(1, int(seed_count) + 1))
        seeds = seeds[: max(1, int(seed_count))]
        original_actions: list[int] = []
        zeroed_actions: list[int] = []
        shuffled_actions: list[int] = []
        env = make_env(group_cfg, seed=seeds[0], shield_enabled=False)
        try:
            model_shape = tuple(getattr(model.observation_space, "shape", ()) or ())
            env_shape = tuple(getattr(env.observation_space, "shape", ()) or ())
            if model_shape != env_shape:
                output[group_name] = {
                    "available": False,
                    "reason": f"PPO model observation shape {model_shape} does not match env observation shape {env_shape}",
                }
                continue
        finally:
            env.close()
        for seed in seeds:
            env = make_env(group_cfg, seed=seed, shield_enabled=False)
            try:
                obs, _info = env.reset(seed=seed)
                terminated = truncated = False
                while not (terminated or truncated):
                    action, _state = model.predict(obs, deterministic=True)
                    zeroed_action, _state = model.predict(
                        _mutate_forecast_observation(obs, feature_dim, "zeroed"), deterministic=True
                    )
                    shuffled_action, _state = model.predict(
                        _mutate_forecast_observation(obs, feature_dim, "shuffled"), deterministic=True
                    )
                    action_int = int(np.asarray(action).reshape(-1)[0])
                    original_actions.append(action_int)
                    zeroed_actions.append(int(np.asarray(zeroed_action).reshape(-1)[0]))
                    shuffled_actions.append(int(np.asarray(shuffled_action).reshape(-1)[0]))
                    obs, _reward, terminated, truncated, _info = env.step(action_int)
            finally:
                env.close()
        output[group_name] = {
            "forecast_source": spec["source"],
            "model_path": str(model_path),
            "seed_count": int(len(seeds)),
            **_policy_feature_sensitivity_from_actions(original_actions, zeroed_actions, shuffled_actions),
        }
    available = {name: item for name, item in output.items() if item.get("available")}
    return {
        "available": bool(available),
        "groups": output,
        "forecast_policy_underutilized": bool(
            output.get("ppo_wcdt_v2_features", {}).get("available")
            and not output.get("ppo_wcdt_v2_features", {}).get("action_sensitive_to_forecast_features", False)
        ),
        "reason": None if available else "no forecast PPO policy sensitivity groups were available",
    }


def _load_replay_actions(path: Path) -> list[int] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    actions = payload.get("executed_actions")
    if actions is None:
        actions = payload.get("actions", [])
    return [int(action) for action in actions]


def _behavior_pair_diagnostics(
    base_run_id: str,
    stage5_report: dict[str, Any],
    left_name: str,
    right_name: str,
) -> dict[str, Any]:
    groups = stage5_report.get("groups", {})
    left_group = groups.get(left_name, {})
    right_group = groups.get(right_name, {})
    if not left_group or not right_group:
        return {"available": False, "reason": f"missing {left_name} or {right_name} group"}
    replay_dir = Path("safe_rl_output") / "runs" / base_run_id / "stage5" / "replay"
    right_by_seed = {int(item["seed"]): item for item in right_group.get("episodes", [])}
    rows = []
    left_actions_all: list[int] = []
    right_actions_all: list[int] = []
    compared_steps = 0
    matching_steps = 0
    missing_replays = 0
    first_diff_steps: list[int] = []
    for left_episode in left_group.get("episodes", []):
        seed = int(left_episode["seed"])
        if seed not in right_by_seed:
            continue
        left_actions = _load_replay_actions(replay_dir / f"{left_name}_seed_{seed}.json")
        right_actions = _load_replay_actions(replay_dir / f"{right_name}_seed_{seed}.json")
        if left_actions is None or right_actions is None:
            missing_replays += 1
            continue
        left_actions_all.extend(left_actions)
        right_actions_all.extend(right_actions)
        limit = min(len(left_actions), len(right_actions))
        compared_steps += limit
        step_matches = sum(1 for idx in range(limit) if left_actions[idx] == right_actions[idx])
        matching_steps += step_matches
        first_diff = next((idx for idx in range(limit) if left_actions[idx] != right_actions[idx]), -1)
        if first_diff >= 0:
            first_diff_steps.append(float(first_diff))
        rows.append(
            {
                "seed": seed,
                "left_action_count": len(left_actions),
                "right_action_count": len(right_actions),
                "exact_action_match": bool(len(left_actions) == len(right_actions) and step_matches == limit),
                "step_action_agreement_rate": float(step_matches / limit) if limit else 0.0,
                "first_diff_step": int(first_diff),
            }
        )
    exact_rates = [float(row["exact_action_match"]) for row in rows]
    return {
        "available": bool(rows),
        "left_group": left_name,
        "right_group": right_name,
        "compared_episode_count": int(len(rows)),
        "missing_replay_count": int(missing_replays),
        "exact_episode_action_match_rate": float(np.mean(exact_rates)) if exact_rates else 0.0,
        "step_action_agreement_rate": float(matching_steps / compared_steps) if compared_steps else 0.0,
        "left_action_histogram": _action_histogram(left_actions_all),
        "right_action_histogram": _action_histogram(right_actions_all),
        "first_diff_step_summary": _summary(first_diff_steps),
        "episodes": rows,
        "action_sensitive_to_forecast_source": bool(matching_steps < compared_steps or np.mean(exact_rates) < 1.0)
        if rows
        else False,
    }


def _forecast_behavior_diagnostics(base_run_id: str, stage5_report: dict[str, Any]) -> dict[str, Any]:
    pairs = [
        ("ppo_cv_features", "ppo_wcdt_features"),
        ("ppo_cv_features", "ppo_wcdt_v2_features"),
        ("ppo_wcdt_v2_features", "wcdt_v2_prediction_shield"),
        ("ppo_cv_features", "ppo_wcdt_v3_features"),
        ("ppo_wcdt_v2_features", "ppo_wcdt_v3_features"),
        ("ppo_wcdt_v3_features", "wcdt_v3_prediction_shield"),
    ]
    comparisons = {
        f"{left}_vs_{right}": _behavior_pair_diagnostics(base_run_id, stage5_report, left, right)
        for left, right in pairs
    }
    available = {name: item for name, item in comparisons.items() if item.get("available")}
    primary_key = (
        "ppo_cv_features_vs_ppo_wcdt_v3_features"
        if comparisons.get("ppo_cv_features_vs_ppo_wcdt_v3_features", {}).get("available")
        else (
            "ppo_cv_features_vs_ppo_wcdt_v2_features"
            if comparisons.get("ppo_cv_features_vs_ppo_wcdt_v2_features", {}).get("available")
            else "ppo_cv_features_vs_ppo_wcdt_features"
        )
    )
    primary = comparisons.get(primary_key, {})
    return {
        "available": bool(available),
        "primary_comparison": primary_key if primary.get("available") else None,
        "comparisons": comparisons,
        "step_action_agreement_rate": primary.get("step_action_agreement_rate", 0.0),
        "action_sensitive_to_forecast_source": bool(primary.get("action_sensitive_to_forecast_source", False)),
        "reason": None if available else "no supported forecast behavior comparison groups were available",
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
    wcdt_v3_checkpoint = stage_file(cfg, "stage2", "wcdt_v3_predictor.pt")
    stage5_path = stage_file(cfg, "stage5", "formal_paired_eval_report.json")
    output_dir = base_run / "stage5" / "diagnostics"
    output_dir.mkdir(parents=True, exist_ok=True)
    data = np.load(stage1_path, allow_pickle=False)
    history = data["agent_history"]
    future = data["agent_future"]
    mask = data["agent_mask"]
    lane_indices = data["agent_lane_index"] if "agent_lane_index" in data else None
    edge_roles = data["agent_edge_role"] if "agent_edge_role" in data else None
    if history.shape[0] == 0:
        raise ValueError(f"no trajectory samples in {stage1_path}")
    sample_count = min(int(max_samples), int(history.shape[0]))
    rng = np.random.default_rng(int(cfg.run.seed))
    indices = np.sort(rng.choice(history.shape[0], size=sample_count, replace=False))
    cv_features = _cv_feature_matrix(
        cfg,
        history,
        mask,
        indices,
        lane_indices=lane_indices,
        edge_roles=edge_roles,
    )
    cv_prediction = _cv_prediction_diagnostics(
        cfg,
        history,
        future,
        mask,
        indices,
        lane_indices=lane_indices,
        edge_roles=edge_roles,
    )
    wcdt_features = np.zeros((0, ForecastFeatureAugmentor.feature_dim(cfg)), dtype=np.float32)
    wcdt_report: dict[str, Any] = {"available": False, "checkpoint": str(checkpoint)}
    wcdt_v2_features = np.zeros((0, ForecastFeatureAugmentor.feature_dim(cfg)), dtype=np.float32)
    wcdt_v2_report: dict[str, Any] = {"available": False, "checkpoint": str(wcdt_v2_checkpoint)}
    wcdt_v3_features = np.zeros((0, ForecastFeatureAugmentor.feature_dim(cfg)), dtype=np.float32)
    wcdt_v3_report: dict[str, Any] = {"available": False, "checkpoint": str(wcdt_v3_checkpoint)}
    if checkpoint.exists():
        wcdt_features, wcdt_report = _wcdt_diagnostics(
            cfg,
            checkpoint,
            history,
            future,
            mask,
            indices,
            batch_size,
            lane_indices=lane_indices,
            edge_roles=edge_roles,
        )
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
            cfg,
            wcdt_v2_checkpoint,
            history,
            future,
            mask,
            indices,
            batch_size,
            lane_indices=lane_indices,
            edge_roles=edge_roles,
        )
        wcdt_v2_report["available"] = True
    if wcdt_v3_checkpoint.exists():
        wcdt_v3_features, wcdt_v3_report = _wcdt_v3_diagnostics(
            cfg,
            wcdt_v3_checkpoint,
            history,
            future,
            mask,
            indices,
            batch_size,
            lane_indices=lane_indices,
            edge_roles=edge_roles,
        )
        wcdt_v3_report["available"] = True
    feature_summary = _feature_source_summary(
        {
            "constant_velocity": cv_features,
            "wcdt": wcdt_features,
            "wcdt_v2": wcdt_v2_features,
            "wcdt_v3": wcdt_v3_features,
        }
    )
    report: dict[str, Any] = {
        "run_id": str(cfg.run.run_id),
        "stage1_buffer": str(stage1_path),
        "sample_count": int(sample_count),
        "feature_names": list(ForecastFeatureAugmentor.FEATURE_NAMES),
        "forecast_feature_summary": feature_summary,
        "runtime_diagnostics_feature_semantics_consistent": bool(
            feature_summary.get("runtime_diagnostics_feature_semantics_consistent", False)
        ),
        "forecast_merge_gap_equals_min_distance_rate": feature_summary.get(
            "forecast_merge_gap_equals_min_distance_rate", {}
        ),
        "cv_feature_summary": {
            name: _summary(cv_features[:, idx])
            for idx, name in enumerate(ForecastFeatureAugmentor.FEATURE_NAMES)
        },
        "cv_prediction": cv_prediction,
        "wcdt_prediction": wcdt_report,
        "wcdt_v2_prediction": wcdt_v2_report,
        "wcdt_v3_prediction": wcdt_v3_report,
    }
    if wcdt_features.shape[0] > 0:
        report["cv_vs_wcdt_feature_distribution"] = _feature_distribution_report(cv_features, wcdt_features)
    if wcdt_v2_features.shape[0] > 0:
        report["cv_vs_wcdt_v2_feature_distribution"] = _feature_distribution_report(cv_features, wcdt_v2_features)
    if wcdt_v3_features.shape[0] > 0:
        report["cv_vs_wcdt_v3_feature_distribution"] = _feature_distribution_report(
            cv_features,
            wcdt_v3_features,
            right_label="wcdt_v3",
        )
    if wcdt_v2_features.shape[0] > 0 and wcdt_v3_features.shape[0] > 0:
        report["wcdt_v2_vs_wcdt_v3_feature_distribution"] = _feature_distribution_report(
            wcdt_v2_features,
            wcdt_v3_features,
            left_label="wcdt_v2",
            right_label="wcdt_v3",
        )
    if stage5_path.exists():
        with stage5_path.open("r", encoding="utf-8") as file:
            stage5_report = json.load(file)
        low_rows = _low_min_distance_replays(
            str(cfg.run.run_id),
            stage5_report,
            int(low_seed_count),
            group_name="ppo_cv_features",
            compare_group_name="cv_prediction_shield",
        )
        low_v2_rows = _low_min_distance_replays(
            str(cfg.run.run_id),
            stage5_report,
            int(low_seed_count),
            group_name="ppo_wcdt_v2_features",
            compare_group_name="wcdt_v2_prediction_shield",
        )
        low_v3_rows = _low_min_distance_replays(
            str(cfg.run.run_id),
            stage5_report,
            int(low_seed_count),
            group_name="ppo_wcdt_v3_features",
            compare_group_name="wcdt_v3_prediction_shield",
        )
        report["low_min_distance_ppo_cv_features"] = low_rows
        report["low_min_distance_ppo_wcdt_v2_features"] = low_v2_rows
        report["low_min_distance_ppo_wcdt_v3_features"] = low_v3_rows
        report["forecast_behavior"] = _forecast_behavior_diagnostics(str(cfg.run.run_id), stage5_report)
        report["policy_feature_sensitivity"] = _policy_feature_sensitivity(
            cfg,
            str(cfg.run.run_id),
            stage5_report,
            seed_count=int(low_seed_count),
        )
        _write_replay_commands(
            output_dir / "replay_low_min_distance_ppo_cv_features.ps1",
            low_rows,
            title="Low-min-distance ppo_cv_features replay commands",
        )
        _write_replay_commands(
            output_dir / "replay_low_min_distance_ppo_wcdt_v2_features.ps1",
            low_v2_rows,
            title="Low-min-distance ppo_wcdt_v2_features replay commands",
        )
        _write_replay_commands(
            output_dir / "replay_low_min_distance_ppo_wcdt_v3_features.ps1",
            low_v3_rows,
            title="Low-min-distance ppo_wcdt_v3_features replay commands",
        )
    else:
        report["policy_feature_sensitivity"] = {
            "available": False,
            "reason": f"missing Stage5 report: {stage5_path}",
        }
    report["forecast_conclusion"] = _forecast_conclusion(report)
    output_path = output_dir / "forecast_diagnostics.json"
    write_report(output_path, report)
    return output_path
