from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from safe_rl.risk.merge_local import route_aware_constant_velocity_rollout
from safe_rl.prediction.merge_safety_loss import (
    LOSS_VERSION,
    ROLE_TARGET_FRONT,
    ROLE_TARGET_REAR,
    merge_safety_loss,
)
from safe_rl.sim.scenario_semantics import (
    EDGE_ROLE_AUXILIARY,
    EDGE_ROLE_RAMP,
    EDGE_ROLE_TARGET,
    distance_to_taper_for_position,
    edge_role,
    infer_lane_index,
    infer_route_position,
    is_ramp_side_y,
    target_lane_center_at_x,
)
from safe_rl.sim.history_buffer import HistoryBuffer
from safe_rl.sim.types import VehicleState


ROLE_COUNT = 6
INPUT_DIM = 14
ARCHITECTURE_VERSION = "wcdt_v2_residual_mlp_v1"


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


def _constant_velocity_future(
    last: np.ndarray,
    horizon: int,
    dt: float,
    cfg: Any | None = None,
    lane_index: int | None = None,
) -> np.ndarray:
    if cfg is not None:
        lane = infer_lane_index(cfg, float(last[1])) if lane_index is None or int(lane_index) < 0 else int(lane_index)
        edge_id, lane_pos = infer_route_position(cfg, float(last[0]), float(last[1]), lane)
        if edge_id is not None:
            state = VehicleState(
                vehicle_id="_wcdt_v2",
                x=float(last[0]),
                y=float(last[1]),
                heading=float(last[2]),
                speed=max(0.0, float(last[3])),
                lane_index=lane,
                lane_id=f"{edge_id}_{lane}",
                lane_pos=float(lane_pos),
                edge_id=str(edge_id),
                accel=float(last[4]),
            )
            rollout = route_aware_constant_velocity_rollout(state, horizon, dt, cfg)[0]
            return np.asarray([item.as_vector() for item in rollout], dtype=np.float32)
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


def _role_for_agent(
    cfg: Any,
    ego_latest: np.ndarray,
    agent_latest: np.ndarray,
    lane_index: int | None = None,
    edge_role_id: int | None = None,
) -> int:
    dx = float(agent_latest[0] - ego_latest[0])
    y = float(agent_latest[1])
    target_center = target_lane_center_at_x(cfg, float(agent_latest[0]))
    lane = infer_lane_index(cfg, y) if lane_index is None or int(lane_index) < 0 else int(lane_index)
    is_target_lane = (
        int(edge_role_id) == EDGE_ROLE_TARGET
        if edge_role_id is not None and int(edge_role_id) > 0
        else abs(y - target_center) <= 2.0
    )
    if is_target_lane and dx >= 0.0:
        return 0
    if is_target_lane and dx < 0.0:
        return 1
    if edge_role_id == EDGE_ROLE_AUXILIARY:
        return 2
    if edge_role_id == EDGE_ROLE_RAMP or (
        (edge_role_id is None or int(edge_role_id) <= 0)
        and is_ramp_side_y(cfg, y)
        and distance_to_taper_for_position(cfg, float(agent_latest[0]), y, lane) > 0.0
    ):
        return 3
    if abs(dx) < 30.0:
        return 4
    return 5


def ordered_merge_local_indices(
    cfg: Any,
    sample_history: np.ndarray,
    sample_mask: np.ndarray,
    lane_indices: np.ndarray | None = None,
    edge_roles: np.ndarray | None = None,
) -> list[int]:
    if sample_history.shape[0] <= 1:
        return []
    ego = sample_history[0, -1]

    def _priority(agent_idx: int) -> tuple[float, float, float, int]:
        latest = sample_history[agent_idx, -1]
        role = _role_for_agent(
            cfg,
            ego,
            latest,
            None if lane_indices is None else int(lane_indices[agent_idx]),
            None if edge_roles is None else int(edge_roles[agent_idx]),
        )
        dx = float(latest[0] - ego[0])
        target_center = target_lane_center_at_x(cfg, float(latest[0]))
        target_lat = abs(float(latest[1]) - target_center)
        distance = abs(dx) + 0.5 * abs(float(latest[1]) - float(ego[1]))
        return (float(role), distance, target_lat, int(agent_idx))

    valid = [idx for idx in range(1, sample_history.shape[0]) if float(sample_mask[idx]) > 0.0]
    return sorted(valid, key=_priority)


def _actor_features(
    cfg: Any,
    ego_latest: np.ndarray,
    agent_latest: np.ndarray,
    role: int,
    ego_lane_index: int | None = None,
) -> np.ndarray:
    ego_speed = max(0.0, float(ego_latest[3]))
    agent_speed = max(0.0, float(agent_latest[3]))
    heading = float(agent_latest[2])
    ego_heading = float(ego_latest[2])
    role_onehot = np.zeros((ROLE_COUNT,), dtype=np.float32)
    role_onehot[min(max(role, 0), ROLE_COUNT - 1)] = 1.0
    dx = float(agent_latest[0] - ego_latest[0])
    dy = float(agent_latest[1] - ego_latest[1])
    merge_distance = float(
        distance_to_taper_for_position(
            cfg,
            float(ego_latest[0]),
            float(ego_latest[1]),
            ego_lane_index,
        )
    )
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
    lane_indices: np.ndarray | None = None,
    edge_roles: np.ndarray | None = None,
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
        sample_lanes = None if lane_indices is None else lane_indices[sample_idx]
        sample_roles = None if edge_roles is None else edge_roles[sample_idx]
        ordered = ordered_merge_local_indices(cfg, history[sample_idx], mask[sample_idx], sample_lanes, sample_roles)
        for actor_row, agent_idx in enumerate(ordered[:max_agents]):
            latest = history[sample_idx, agent_idx, -1]
            role = _role_for_agent(
                cfg,
                ego_latest,
                latest,
                None if sample_lanes is None else int(sample_lanes[agent_idx]),
                None if sample_roles is None else int(sample_roles[agent_idx]),
            )
            features[row, actor_row] = _actor_features(
                cfg,
                ego_latest,
                latest,
                role,
                None if sample_lanes is None else int(sample_lanes[0]),
            )
            baseline[row, actor_row] = _constant_velocity_future(
                latest,
                horizon,
                dt,
                cfg,
                None if sample_lanes is None else int(sample_lanes[agent_idx]),
            )
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
    latest = history.latest()
    ids = history.agent_ids(ego_id)
    lane_indices = np.full((1, agent_history.shape[0]), -1, dtype=np.int64)
    edge_roles = np.zeros((1, agent_history.shape[0]), dtype=np.int64)
    for agent_idx, vehicle_id in enumerate(ids):
        state = latest.get(vehicle_id)
        if state is None:
            continue
        lane_indices[0, agent_idx] = int(state.lane_index)
        edge_roles[0, agent_idx] = int(edge_role(cfg, state.edge_id, state.lane_index))
    batch = build_v2_numpy_batch(
        cfg,
        full_history,
        future,
        full_mask,
        np.asarray([0], dtype=np.int64),
        lane_indices=lane_indices,
        edge_roles=edge_roles,
    )
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


def v2_loss(pred, target, mask, ego_future, role_ids, weights: dict[str, float] | None = None):
    return merge_safety_loss(pred, target, mask, ego_future, role_ids, weights)


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
    uncertainty = torch.linalg.norm(stacked[..., :2].std(dim=0, unbiased=False), dim=-1).mean(dim=(-1, -2))
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
