from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from safe_rl.risk.merge_local import merge_target_lane, merge_x
from safe_rl.sim.history_buffer import HistoryBuffer
from safe_rl.sim.types import VehicleState


LANE_CENTERS = {0: -8.0, 1: -4.8, 2: -1.6}
ROLE_COUNT = 5
INPUT_DIM = 14


def _require_torch():
    try:
        import torch
        from torch import nn
    except ImportError as exc:  # pragma: no cover
        raise ImportError("WcDT v2 requires torch. Activate the SAFE_RL training environment.") from exc
    return torch, nn


def _resolve_device(config: Any, torch: Any):
    requested = str(config.get("training", {}).get("device", "auto")).strip().lower()
    if requested in ("auto", ""):
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "gpu":
        requested = "cuda"
    return torch.device(requested)


try:
    _BASE_TORCH, _BASE_NN = _require_torch()
    _TORCH_MODULE_BASE = _BASE_NN.Module
except ImportError:  # pragma: no cover
    _TORCH_MODULE_BASE = object


def infer_lane_index(y: float) -> int:
    return min(LANE_CENTERS, key=lambda lane: abs(float(y) - LANE_CENTERS[lane]))


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


def _role_for_agent(cfg: Any, ego_latest: np.ndarray, agent_latest: np.ndarray) -> int:
    target_lane = merge_target_lane(cfg)
    target_center = LANE_CENTERS.get(target_lane, -1.6)
    dx = float(agent_latest[0] - ego_latest[0])
    y = float(agent_latest[1])
    lane = infer_lane_index(y)
    is_target_lane = abs(y - target_center) <= 2.0 and lane == target_lane
    is_ramp_local = y > 0.5 and float(agent_latest[0]) < merge_x(cfg) + 20.0
    if is_target_lane and dx >= 0.0:
        return 0
    if is_target_lane and dx < 0.0:
        return 1
    if is_ramp_local:
        return 2
    if abs(dx) < 30.0:
        return 3
    return 4


def ordered_merge_local_indices(cfg: Any, sample_history: np.ndarray, sample_mask: np.ndarray) -> list[int]:
    if sample_history.shape[0] <= 1:
        return []
    ego = sample_history[0, -1]
    target_lane = merge_target_lane(cfg)
    target_center = LANE_CENTERS.get(target_lane, -1.6)

    def _priority(agent_idx: int) -> tuple[float, float, float, int]:
        latest = sample_history[agent_idx, -1]
        role = _role_for_agent(cfg, ego, latest)
        dx = float(latest[0] - ego[0])
        target_lat = abs(float(latest[1]) - target_center)
        distance = abs(dx) + 0.5 * abs(float(latest[1]) - float(ego[1]))
        return (float(role), distance, target_lat, int(agent_idx))

    valid = [idx for idx in range(1, sample_history.shape[0]) if float(sample_mask[idx]) > 0.0]
    return sorted(valid, key=_priority)


def _actor_features(cfg: Any, ego_latest: np.ndarray, agent_latest: np.ndarray, role: int) -> np.ndarray:
    ego_speed = max(0.0, float(ego_latest[3]))
    agent_speed = max(0.0, float(agent_latest[3]))
    heading = float(agent_latest[2])
    ego_heading = float(ego_latest[2])
    role_onehot = np.zeros((ROLE_COUNT,), dtype=np.float32)
    role_onehot[min(max(role, 0), ROLE_COUNT - 1)] = 1.0
    dx = float(agent_latest[0] - ego_latest[0])
    dy = float(agent_latest[1] - ego_latest[1])
    merge_distance = float(merge_x(cfg) - ego_latest[0])
    return np.asarray(
        [
            dx / 100.0,
            dy / 20.0,
            agent_speed / 40.0,
            ego_speed / 40.0,
            float(agent_latest[4]) / 5.0,
            np.sin(heading),
            np.cos(heading),
            np.sin(ego_heading),
            merge_distance / 100.0,
            *role_onehot.tolist(),
        ][:INPUT_DIM],
        dtype=np.float32,
    )


def build_v2_numpy_batch(
    cfg: Any,
    history: np.ndarray,
    future: np.ndarray,
    mask: np.ndarray,
    indices: np.ndarray,
) -> dict[str, np.ndarray]:
    sample_indices = np.asarray(indices, dtype=np.int64)
    horizon = int(min(future.shape[2], cfg.prediction.get("wcdt_v2_horizon_steps", cfg.scenario.forecast_horizon_steps)))
    max_agents = int(cfg.prediction.get("wcdt_v2_max_agents", min(cfg.prediction.max_pred_num, history.shape[1] - 1)))
    max_agents = max(1, max_agents)
    dt = float(cfg.scenario.step_length)
    batch = sample_indices.shape[0]
    features = np.zeros((batch, max_agents, INPUT_DIM), dtype=np.float32)
    baseline = np.zeros((batch, max_agents, horizon, 5), dtype=np.float32)
    target = np.zeros((batch, max_agents, horizon, 5), dtype=np.float32)
    actor_mask = np.zeros((batch, max_agents), dtype=np.float32)
    role_ids = np.full((batch, max_agents), ROLE_COUNT - 1, dtype=np.int64)
    ego_future = np.zeros((batch, horizon, 5), dtype=np.float32)
    selected_indices = np.full((batch, max_agents), -1, dtype=np.int64)

    for row, sample_idx in enumerate(sample_indices):
        ego_latest = history[sample_idx, 0, -1]
        ego_future[row] = future[sample_idx, 0, :horizon]
        ordered = ordered_merge_local_indices(cfg, history[sample_idx], mask[sample_idx])
        for actor_row, agent_idx in enumerate(ordered[:max_agents]):
            latest = history[sample_idx, agent_idx, -1]
            role = _role_for_agent(cfg, ego_latest, latest)
            features[row, actor_row] = _actor_features(cfg, ego_latest, latest, role)
            baseline[row, actor_row] = _constant_velocity_future(latest, horizon, dt)
            target[row, actor_row] = future[sample_idx, agent_idx, :horizon]
            actor_mask[row, actor_row] = mask[sample_idx, agent_idx]
            role_ids[row, actor_row] = role
            selected_indices[row, actor_row] = int(agent_idx)
    return {
        "features": features,
        "baseline": baseline,
        "target": target,
        "mask": actor_mask,
        "role_ids": role_ids,
        "ego_future": ego_future,
        "selected_indices": selected_indices,
    }


def build_v2_runtime_batch(cfg: Any, history: HistoryBuffer, ego_id: str) -> dict[str, np.ndarray]:
    agent_history, agent_mask = history.to_tensor_arrays(ego_id)
    horizon = int(cfg.forecast_features.get("horizon_steps", cfg.scenario.forecast_horizon_steps))
    future = np.zeros((1, agent_history.shape[0], horizon, 5), dtype=np.float32)
    full_history = agent_history[None, ...]
    full_mask = agent_mask[None, ...]
    batch = build_v2_numpy_batch(cfg, full_history, future, full_mask, np.asarray([0], dtype=np.int64))
    return {key: value[0:1] if key != "selected_indices" else value[0] for key, value in batch.items()}


class WcDTV2ResidualPredictor(_TORCH_MODULE_BASE):
    def __init__(self, input_dim: int = INPUT_DIM, horizon_steps: int = 30, hidden_dim: int = 128):
        torch, nn = _require_torch()
        super().__init__()
        self.horizon_steps = int(horizon_steps)
        self.net = nn.Sequential(
            nn.Linear(int(input_dim), int(hidden_dim)),
            nn.ReLU(inplace=True),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.ReLU(inplace=True),
            nn.Linear(int(hidden_dim), self.horizon_steps * 2),
        )

    def forward(self, features, baseline):
        residual = self.net(features).view(features.shape[0], features.shape[1], self.horizon_steps, 2)
        if baseline.shape[2] != self.horizon_steps:
            baseline = baseline[:, :, : self.horizon_steps]
        output = baseline.clone()
        output[..., :2] = output[..., :2] + residual
        return output


def tensorize_batch(batch: dict[str, np.ndarray], torch: Any, device: Any) -> dict[str, Any]:
    return {
        "features": torch.tensor(batch["features"], dtype=torch.float32, device=device),
        "baseline": torch.tensor(batch["baseline"], dtype=torch.float32, device=device),
        "target": torch.tensor(batch["target"], dtype=torch.float32, device=device),
        "mask": torch.tensor(batch["mask"], dtype=torch.float32, device=device),
        "role_ids": torch.tensor(batch["role_ids"], dtype=torch.long, device=device),
        "ego_future": torch.tensor(batch["ego_future"], dtype=torch.float32, device=device),
    }


def masked_mean(values, mask):
    if values.ndim > mask.ndim:
        mask = mask.view(mask.shape[0], mask.shape[1], *([1] * (values.ndim - 2)))
    denom = mask.sum() * (values.numel() / max(mask.numel(), 1))
    if float(denom.detach().cpu()) <= 1.0e-8:
        return values.sum() * 0.0
    return (values * mask).sum() / denom


def v2_loss(pred, target, mask, ego_future, role_ids, weights: dict[str, float] | None = None):
    torch, _nn = _require_torch()
    weights = weights or {}
    distance = torch.linalg.norm(pred[..., :2] - target[..., :2], dim=-1)
    ade = masked_mean(distance, mask)
    fde = masked_mean(distance[:, :, -1], mask)
    ego_xy = ego_future[:, None, :, :2]
    pred_gap = torch.clamp(torch.linalg.norm(pred[..., :2] - ego_xy, dim=-1) - 3.0, min=0.0)
    target_gap = torch.clamp(torch.linalg.norm(target[..., :2] - ego_xy, dim=-1) - 3.0, min=0.0)
    min_dist = masked_mean(torch.abs(pred_gap - target_gap), mask)
    target_lane_mask = ((role_ids == 0) | (role_ids == 1)).float()
    gap_error = masked_mean(torch.abs(pred_gap - target_gap).mean(dim=-1), mask * target_lane_mask)
    ordering_penalties = []
    for row in range(pred.shape[0]):
        front = torch.where(role_ids[row] == 0)[0]
        rear = torch.where(role_ids[row] == 1)[0]
        if front.numel() and rear.numel():
            ordering_penalties.append(torch.relu(pred[row, rear[0], :, 0] - pred[row, front[0], :, 0] + 4.8).mean())
    ordering = torch.stack(ordering_penalties).mean() if ordering_penalties else pred.sum() * 0.0
    return (
        float(weights.get("ade", 1.0)) * ade
        + float(weights.get("fde", 0.5)) * fde
        + float(weights.get("target_lane_gap", 0.25)) * gap_error
        + float(weights.get("future_min_distance", 0.25)) * min_dist
        + float(weights.get("ordering", 0.1)) * ordering
    )


def load_v2_ensemble(config: Any, checkpoint: str | Path, device: Any | None = None):
    torch, _nn = _require_torch()
    device = device or _resolve_device(config, torch)
    payload = torch.load(checkpoint, map_location=device)
    horizon = int(payload.get("horizon_steps", config.forecast_features.get("horizon_steps", config.scenario.forecast_horizon_steps)))
    hidden_dim = int(payload.get("hidden_dim", config.prediction.get("wcdt_v2_hidden_dim", 128)))
    states = payload.get("model_state_dicts")
    if states is None and "model_state_dict" in payload:
        states = [payload["model_state_dict"]]
    if not states:
        raise ValueError(f"checkpoint has no WcDT v2 state dicts: {checkpoint}")
    models = []
    for state in states:
        model = WcDTV2ResidualPredictor(INPUT_DIM, horizon, hidden_dim).to(device)
        model.load_state_dict(state)
        model.eval()
        models.append(model)
    return models, payload, device


def ensemble_predict(models: list[Any], tensor_batch: dict[str, Any]):
    torch, _nn = _require_torch()
    predictions = []
    with torch.no_grad():
        for model in models:
            predictions.append(model(tensor_batch["features"], tensor_batch["baseline"]))
    stacked = torch.stack(predictions, dim=0)
    mean = stacked.mean(dim=0)
    uncertainty = torch.linalg.norm(stacked[..., :2].std(dim=0), dim=-1).mean(dim=(-1, -2))
    return mean, uncertainty


class WcDTV2Predictor:
    """Runtime wrapper for the merge-centric WcDT v2 residual ensemble."""

    def __init__(self, config: Any, checkpoint: str | Path):
        torch, _nn = _require_torch()
        self.config = config
        self.checkpoint_path = str(Path(checkpoint).resolve())
        self.device = _resolve_device(config, torch)
        self.models, self.payload, self.device = load_v2_ensemble(config, checkpoint, self.device)
        self._torch = torch

    def predict(self, context: dict[str, Any]) -> dict[str, Any]:
        ego = context.get("ego")
        history = context.get("history")
        if ego is None or history is None:
            raise ValueError("WcDTV2Predictor requires ego and history in the risk context.")
        batch = build_v2_runtime_batch(self.config, history, str(ego.vehicle_id))
        tensor_batch = tensorize_batch(batch, self._torch, self.device)
        mean, uncertainty = ensemble_predict(self.models, tensor_batch)
        return {
            "future_trajectories": mean.detach().cpu().numpy()[0],
            "uncertainty": float(uncertainty.detach().cpu().numpy()[0]),
            "mode_confidence": None,
            "selected_indices": batch["selected_indices"].tolist(),
            "checkpoint": self.checkpoint_path,
        }
