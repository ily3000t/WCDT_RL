from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from safe_rl.prediction.merge_safety_loss import LOSS_VERSION, merge_safety_loss
from safe_rl.sim.metrics import SAFETY_METRIC_VERSION
from safe_rl.prediction.wcdt_v2_predictor import (
    ROLE_COUNT,
    _constant_velocity_future,
    _resolve_device,
    _role_for_agent,
    ordered_merge_local_indices,
)
from safe_rl.sim.history_buffer import HistoryBuffer
from safe_rl.sim.scenario_semantics import distance_to_taper_for_position, edge_role


HISTORY_INPUT_DIM = 10
LANE_EMBEDDING_COUNT = 17
EDGE_ROLE_EMBEDDING_COUNT = 5
ARCHITECTURE_VERSION = "wcdt_v3_temporal_actor_transformer_v2"
TRAJECTORY_SCHEMA_VERSION = 3


def _require_torch():
    try:
        import torch
        from torch import nn
    except ImportError as exc:  # pragma: no cover
        raise ImportError("WcDT v3 requires torch. Activate the SAFE_RL training environment.") from exc
    return torch, nn


try:
    _BASE_TORCH, _BASE_NN = _require_torch()
    _TORCH_MODULE_BASE = _BASE_NN.Module
except ImportError:  # pragma: no cover
    _TORCH_MODULE_BASE = object


def _lane_embedding_id(lane_index: int | None) -> int:
    if lane_index is None:
        return 0
    return int(np.clip(int(lane_index) + 1, 0, LANE_EMBEDDING_COUNT - 1))


def _edge_role_embedding_id(edge_role_id: int | None) -> int:
    return int(np.clip(int(edge_role_id or 0), 0, EDGE_ROLE_EMBEDDING_COUNT - 1))


def _history_actor_features(
    cfg: Any,
    ego_history: np.ndarray,
    actor_history: np.ndarray,
    ego_lane_indices: np.ndarray | None,
) -> np.ndarray:
    history_steps = min(ego_history.shape[0], actor_history.shape[0])
    output = np.zeros((history_steps, HISTORY_INPUT_DIM), dtype=np.float32)
    for step_idx in range(history_steps):
        ego = ego_history[step_idx]
        actor = actor_history[step_idx]
        ego_lane_index = None if ego_lane_indices is None else int(ego_lane_indices[step_idx])
        relative_heading = float(actor[2] - ego[2])
        output[step_idx] = np.asarray(
            [
                float(actor[0] - ego[0]) / 100.0,
                float(actor[1] - ego[1]) / 20.0,
                max(0.0, float(actor[3])) / 40.0,
                max(0.0, float(ego[3])) / 40.0,
                float(actor[3] - ego[3]) / 20.0,
                float(actor[4]) / 5.0,
                float(ego[4]) / 5.0,
                np.sin(relative_heading),
                np.cos(relative_heading),
                float(distance_to_taper_for_position(cfg, float(ego[0]), float(ego[1]), ego_lane_index)) / 100.0,
            ],
            dtype=np.float32,
        )
    return output


def build_v3_numpy_batch(
    cfg: Any,
    history: np.ndarray,
    future: np.ndarray,
    mask: np.ndarray,
    indices: np.ndarray,
    lane_indices: np.ndarray | None = None,
    edge_roles: np.ndarray | None = None,
    history_valid_mask: np.ndarray | None = None,
    future_valid_mask: np.ndarray | None = None,
    history_lane_indices: np.ndarray | None = None,
    history_edge_roles: np.ndarray | None = None,
    agent_length: np.ndarray | None = None,
    agent_width: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    sample_indices = np.asarray(indices, dtype=np.int64)
    history_steps = int(history.shape[2])
    horizon = int(min(future.shape[2], cfg.prediction.get("wcdt_v3_horizon_steps", cfg.scenario.forecast_horizon_steps)))
    max_agents = int(cfg.prediction.get("wcdt_v3_max_agents", min(cfg.prediction.max_pred_num, history.shape[1] - 1)))
    max_agents = max(1, max_agents)
    dt = float(cfg.scenario.step_length)
    batch = sample_indices.shape[0]
    history_features = np.zeros((batch, max_agents, history_steps, HISTORY_INPUT_DIM), dtype=np.float32)
    baseline = np.zeros((batch, max_agents, horizon, 5), dtype=np.float32)
    target = np.zeros((batch, max_agents, horizon, 5), dtype=np.float32)
    actor_mask = np.zeros((batch, max_agents), dtype=np.float32)
    role_ids = np.full((batch, max_agents), ROLE_COUNT - 1, dtype=np.int64)
    lane_ids = np.zeros((batch, max_agents), dtype=np.int64)
    edge_role_ids = np.zeros((batch, max_agents), dtype=np.int64)
    ego_future = np.zeros((batch, horizon, 5), dtype=np.float32)
    selected_history_valid_mask = np.ones((batch, max_agents, history_steps), dtype=np.float32)
    selected_future_valid_mask = np.zeros((batch, max_agents, horizon), dtype=np.float32)
    ego_future_valid_mask = np.ones((batch, horizon), dtype=np.float32)
    selected_history_lane_ids = np.zeros((batch, max_agents, history_steps), dtype=np.int64)
    selected_history_edge_role_ids = np.zeros((batch, max_agents, history_steps), dtype=np.int64)
    selected_indices = np.full((batch, max_agents), -1, dtype=np.int64)
    selected_length = np.full((batch, max_agents), 4.8, dtype=np.float32)
    selected_width = np.full((batch, max_agents), 1.8, dtype=np.float32)
    ego_length = np.full((batch,), 4.8, dtype=np.float32)
    ego_width = np.full((batch,), 1.8, dtype=np.float32)

    for row, sample_idx in enumerate(sample_indices):
        ego_history = history[sample_idx, 0]
        ego_latest = ego_history[-1]
        if agent_length is not None:
            ego_length[row] = float(agent_length[sample_idx, 0])
        if agent_width is not None:
            ego_width[row] = float(agent_width[sample_idx, 0])
        ego_future[row] = future[sample_idx, 0, :horizon]
        if future_valid_mask is not None:
            ego_future_valid_mask[row] = future_valid_mask[sample_idx, 0, :horizon]
        sample_lanes = None if lane_indices is None else lane_indices[sample_idx]
        sample_roles = None if edge_roles is None else edge_roles[sample_idx]
        sample_history_lanes = None if history_lane_indices is None else history_lane_indices[sample_idx]
        sample_history_roles = None if history_edge_roles is None else history_edge_roles[sample_idx]
        ego_history_lanes = None if sample_history_lanes is None else sample_history_lanes[0]
        ordered = ordered_merge_local_indices(cfg, history[sample_idx], mask[sample_idx], sample_lanes, sample_roles)
        for actor_row, agent_idx in enumerate(ordered[:max_agents]):
            latest = history[sample_idx, agent_idx, -1]
            lane = None if sample_lanes is None else int(sample_lanes[agent_idx])
            actor_edge_role = None if sample_roles is None else int(sample_roles[agent_idx])
            role = _role_for_agent(cfg, ego_latest, latest, lane, actor_edge_role)
            history_features[row, actor_row] = _history_actor_features(
                cfg,
                ego_history,
                history[sample_idx, agent_idx],
                ego_history_lanes,
            )
            baseline[row, actor_row] = _constant_velocity_future(latest, horizon, dt, cfg, lane)
            target[row, actor_row] = future[sample_idx, agent_idx, :horizon]
            actor_mask[row, actor_row] = mask[sample_idx, agent_idx]
            selected_history_valid_mask[row, actor_row] = (
                mask[sample_idx, agent_idx]
                if history_valid_mask is None
                else history_valid_mask[sample_idx, agent_idx, :history_steps]
            )
            selected_future_valid_mask[row, actor_row] = (
                mask[sample_idx, agent_idx]
                if future_valid_mask is None
                else future_valid_mask[sample_idx, agent_idx, :horizon]
            )
            role_ids[row, actor_row] = role
            lane_ids[row, actor_row] = _lane_embedding_id(lane)
            edge_role_ids[row, actor_row] = _edge_role_embedding_id(actor_edge_role)
            if sample_history_lanes is not None:
                selected_history_lane_ids[row, actor_row] = np.asarray(
                    [_lane_embedding_id(value) for value in sample_history_lanes[agent_idx, :history_steps]],
                    dtype=np.int64,
                )
            if sample_history_roles is not None:
                selected_history_edge_role_ids[row, actor_row] = np.asarray(
                    [_edge_role_embedding_id(value) for value in sample_history_roles[agent_idx, :history_steps]],
                    dtype=np.int64,
                )
            selected_indices[row, actor_row] = int(agent_idx)
            if agent_length is not None:
                selected_length[row, actor_row] = float(agent_length[sample_idx, agent_idx])
            if agent_width is not None:
                selected_width[row, actor_row] = float(agent_width[sample_idx, agent_idx])
    return {
        "history_features": history_features,
        "baseline": baseline,
        "target": target,
        "mask": actor_mask,
        "role_ids": role_ids,
        "lane_ids": lane_ids,
        "edge_role_ids": edge_role_ids,
        "ego_future": ego_future,
        "history_valid_mask": selected_history_valid_mask,
        "future_valid_mask": selected_future_valid_mask,
        "ego_future_valid_mask": ego_future_valid_mask,
        "history_lane_ids": selected_history_lane_ids,
        "history_edge_role_ids": selected_history_edge_role_ids,
        "selected_indices": selected_indices,
        "agent_length": selected_length,
        "agent_width": selected_width,
        "ego_length": ego_length,
        "ego_width": ego_width,
    }


def build_v3_runtime_batch(cfg: Any, history: HistoryBuffer, ego_id: str) -> dict[str, np.ndarray]:
    runtime_history = history.to_tensor_arrays_with_metadata(ego_id, cfg)
    agent_history = runtime_history["history"]
    agent_mask = runtime_history["mask"]
    horizon = int(cfg.forecast_features.get("horizon_steps", cfg.scenario.forecast_horizon_steps))
    future = np.zeros((1, agent_history.shape[0], horizon, 5), dtype=np.float32)
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
    batch = build_v3_numpy_batch(
        cfg,
        agent_history[None, ...],
        future,
        agent_mask[None, ...],
        np.asarray([0], dtype=np.int64),
        lane_indices=lane_indices,
        edge_roles=edge_roles,
        history_valid_mask=runtime_history["history_valid_mask"][None, ...],
        history_lane_indices=runtime_history["history_lane_index"][None, ...],
        history_edge_roles=runtime_history["history_edge_role"][None, ...],
    )
    return {key: value[0:1] if key != "selected_indices" else value[0] for key, value in batch.items()}


class WcDTV3TemporalInteractionPredictor(_TORCH_MODULE_BASE):
    def __init__(
        self,
        history_steps: int,
        horizon_steps: int = 30,
        hidden_dim: int = 128,
        temporal_layers: int = 2,
        actor_attention_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        torch, nn = _require_torch()
        super().__init__()
        self.history_steps = int(history_steps)
        self.horizon_steps = int(horizon_steps)
        self.history_projection = nn.Linear(HISTORY_INPUT_DIM, int(hidden_dim))
        self.position_embedding = nn.Parameter(torch.zeros(1, 1, self.history_steps, int(hidden_dim)))
        temporal_layer = nn.TransformerEncoderLayer(
            d_model=int(hidden_dim),
            nhead=int(num_heads),
            dim_feedforward=int(hidden_dim) * 2,
            dropout=float(dropout),
            batch_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(temporal_layer, num_layers=int(temporal_layers))
        self.role_embedding = nn.Embedding(ROLE_COUNT, 16)
        self.lane_embedding = nn.Embedding(LANE_EMBEDDING_COUNT, 8)
        self.edge_role_embedding = nn.Embedding(EDGE_ROLE_EMBEDDING_COUNT, 8)
        self.temporal_route_projection = nn.Linear(16, int(hidden_dim))
        self.static_projection = nn.Linear(32, int(hidden_dim))
        actor_layer = nn.TransformerEncoderLayer(
            d_model=int(hidden_dim),
            nhead=int(num_heads),
            dim_feedforward=int(hidden_dim) * 2,
            dropout=float(dropout),
            batch_first=True,
        )
        self.actor_encoder = nn.TransformerEncoder(actor_layer, num_layers=int(actor_attention_layers))
        self.decoder = nn.Sequential(
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.ReLU(inplace=True),
            nn.Linear(int(hidden_dim), self.horizon_steps * 2),
        )

    def forward(
        self,
        history_features,
        history_valid_mask,
        history_lane_ids,
        history_edge_role_ids,
        role_ids,
        lane_ids,
        edge_role_ids,
        actor_mask,
        baseline,
    ):
        batch, actors, history_steps, _dim = history_features.shape
        temporal_route = self.temporal_route_projection(
            _BASE_TORCH.cat(
                [
                    self.lane_embedding(history_lane_ids),
                    self.edge_role_embedding(history_edge_role_ids),
                ],
                dim=-1,
            )
        )
        temporal = self.history_projection(history_features) + temporal_route + self.position_embedding[:, :, :history_steps]
        temporal = temporal.reshape(batch * actors, history_steps, -1)
        temporal_padding_mask = (history_valid_mask <= 0.0).reshape(batch * actors, history_steps)
        safe_temporal_padding_mask = temporal_padding_mask.clone()
        all_history_padding = safe_temporal_padding_mask.all(dim=1)
        safe_temporal_padding_mask[all_history_padding, 0] = False
        temporal = self.temporal_encoder(temporal, src_key_padding_mask=safe_temporal_padding_mask)
        valid_indices = (~temporal_padding_mask).long() * _BASE_TORCH.arange(
            history_steps,
            device=history_features.device,
        ).view(1, history_steps)
        last_valid = valid_indices.amax(dim=1)
        temporal = temporal.gather(
            dim=1,
            index=last_valid[:, None, None].expand(-1, 1, temporal.shape[-1]),
        ).squeeze(1)
        temporal = temporal.reshape(batch, actors, -1)
        static = self.static_projection(
            _BASE_TORCH.cat(
                [
                    self.role_embedding(role_ids),
                    self.lane_embedding(lane_ids),
                    self.edge_role_embedding(edge_role_ids),
                ],
                dim=-1,
            )
        )
        actor_tokens = temporal + static
        padding_mask = actor_mask <= 0.0
        safe_padding_mask = padding_mask.clone()
        all_padding = safe_padding_mask.all(dim=1)
        if bool(all_padding.any()):
            safe_padding_mask[all_padding, 0] = False
        contextual = self.actor_encoder(actor_tokens, src_key_padding_mask=safe_padding_mask)
        residual = self.decoder(contextual).view(batch, actors, self.horizon_steps, 2)
        residual = residual * actor_mask[:, :, None, None]
        output = baseline[:, :, : self.horizon_steps].clone()
        output[..., :2] = output[..., :2] + residual
        return output * actor_mask[:, :, None, None]


def tensorize_v3_batch(batch: dict[str, np.ndarray], torch: Any, device: Any) -> dict[str, Any]:
    return {
        "history_features": torch.tensor(batch["history_features"], dtype=torch.float32, device=device),
        "baseline": torch.tensor(batch["baseline"], dtype=torch.float32, device=device),
        "target": torch.tensor(batch["target"], dtype=torch.float32, device=device),
        "mask": torch.tensor(batch["mask"], dtype=torch.float32, device=device),
        "role_ids": torch.tensor(batch["role_ids"], dtype=torch.long, device=device),
        "lane_ids": torch.tensor(batch["lane_ids"], dtype=torch.long, device=device),
        "edge_role_ids": torch.tensor(batch["edge_role_ids"], dtype=torch.long, device=device),
        "ego_future": torch.tensor(batch["ego_future"], dtype=torch.float32, device=device),
        "history_valid_mask": torch.tensor(batch["history_valid_mask"], dtype=torch.float32, device=device),
        "future_valid_mask": torch.tensor(batch["future_valid_mask"], dtype=torch.float32, device=device),
        "ego_future_valid_mask": torch.tensor(batch["ego_future_valid_mask"], dtype=torch.float32, device=device),
        "history_lane_ids": torch.tensor(batch["history_lane_ids"], dtype=torch.long, device=device),
        "history_edge_role_ids": torch.tensor(batch["history_edge_role_ids"], dtype=torch.long, device=device),
        "agent_length": torch.tensor(batch["agent_length"], dtype=torch.float32, device=device),
        "agent_width": torch.tensor(batch["agent_width"], dtype=torch.float32, device=device),
        "ego_length": torch.tensor(batch["ego_length"], dtype=torch.float32, device=device),
        "ego_width": torch.tensor(batch["ego_width"], dtype=torch.float32, device=device),
    }


def v3_loss(
    pred,
    target,
    mask,
    ego_future,
    role_ids,
    weights: dict[str, float] | None = None,
    future_valid_mask=None,
    ego_future_valid_mask=None,
    agent_length=None,
    agent_width=None,
    ego_length=None,
    ego_width=None,
):
    return merge_safety_loss(
        pred,
        target,
        mask,
        ego_future,
        role_ids,
        weights,
        future_valid_mask=future_valid_mask,
        ego_future_valid_mask=ego_future_valid_mask,
        agent_length=agent_length,
        agent_width=agent_width,
        ego_length=ego_length,
        ego_width=ego_width,
    )


def _predict_model(model: Any, batch: dict[str, Any]):
    return model(
        batch["history_features"],
        batch["history_valid_mask"],
        batch["history_lane_ids"],
        batch["history_edge_role_ids"],
        batch["role_ids"],
        batch["lane_ids"],
        batch["edge_role_ids"],
        batch["mask"],
        batch["baseline"],
    )


def ensemble_predict_v3(models: list[Any], tensor_batch: dict[str, Any]):
    torch, _nn = _require_torch()
    with torch.no_grad():
        stacked = torch.stack([_predict_model(model, tensor_batch) for model in models], dim=0)
    mean = stacked.mean(dim=0)
    uncertainty = torch.linalg.norm(stacked[..., :2].std(dim=0, unbiased=False), dim=-1).mean(dim=(-1, -2))
    return mean, uncertainty


def load_v3_ensemble(config: Any, checkpoint: str | Path, device: Any | None = None):
    torch, _nn = _require_torch()
    device = device or _resolve_device(config, torch)
    payload = torch.load(checkpoint, map_location=device)
    architecture = payload.get("architecture_version")
    if architecture != ARCHITECTURE_VERSION:
        raise ValueError(f"unsupported WcDT v3 architecture_version={architecture!r}; expected {ARCHITECTURE_VERSION!r}")
    loss_version = payload.get("loss_version")
    if loss_version != LOSS_VERSION:
        raise ValueError(f"unsupported WcDT v3 loss_version={loss_version!r}; expected {LOSS_VERSION!r}")
    schema_version = int(payload.get("trajectory_schema_version", -1))
    if schema_version != TRAJECTORY_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported WcDT v3 trajectory_schema_version={schema_version!r}; "
            f"expected {TRAJECTORY_SCHEMA_VERSION!r}"
        )
    metric_version = str(payload.get("safety_metric_version", ""))
    if metric_version != SAFETY_METRIC_VERSION:
        raise ValueError(
            f"unsupported WcDT v3 safety_metric_version={metric_version!r}; expected {SAFETY_METRIC_VERSION!r}"
        )
    model_kwargs = {
        "history_steps": int(payload.get("history_steps", config.scenario.history_steps)),
        "horizon_steps": int(payload.get("horizon_steps", config.scenario.forecast_horizon_steps)),
        "hidden_dim": int(payload.get("hidden_dim", config.prediction.get("wcdt_v3_hidden_dim", 128))),
        "temporal_layers": int(payload.get("temporal_layers", config.prediction.get("wcdt_v3_temporal_layers", 2))),
        "actor_attention_layers": int(
            payload.get("actor_attention_layers", config.prediction.get("wcdt_v3_actor_attention_layers", 2))
        ),
        "num_heads": int(payload.get("num_heads", config.prediction.get("wcdt_v3_num_heads", 4))),
        "dropout": float(payload.get("dropout", config.prediction.get("wcdt_v3_dropout", 0.1))),
    }
    states = payload.get("model_state_dicts")
    if not states:
        raise ValueError(f"checkpoint has no WcDT v3 state dicts: {checkpoint}")
    models = []
    for state in states:
        model = WcDTV3TemporalInteractionPredictor(**model_kwargs).to(device)
        model.load_state_dict(state)
        model.eval()
        models.append(model)
    return models, payload, device


class WcDTV3Predictor:
    """Runtime wrapper for the temporal interaction WcDT v3 residual ensemble."""

    def __init__(self, config: Any, checkpoint: str | Path):
        torch, _nn = _require_torch()
        self.config = config
        self.checkpoint_path = str(Path(checkpoint).resolve())
        self.device = _resolve_device(config, torch)
        self.models, self.payload, self.device = load_v3_ensemble(config, checkpoint, self.device)
        self._torch = torch

    def predict(self, context: dict[str, Any]) -> dict[str, Any]:
        ego = context.get("ego")
        history = context.get("history")
        if ego is None or history is None:
            raise ValueError("WcDTV3Predictor requires ego and history in the risk context.")
        batch = build_v3_runtime_batch(self.config, history, str(ego.vehicle_id))
        tensor_batch = tensorize_v3_batch(batch, self._torch, self.device)
        mean, uncertainty = ensemble_predict_v3(self.models, tensor_batch)
        return {
            "future_trajectories": mean.detach().cpu().numpy()[0],
            "uncertainty": float(uncertainty.detach().cpu().numpy()[0]),
            "mode_confidence": None,
            "selected_indices": batch["selected_indices"].tolist(),
            "checkpoint": self.checkpoint_path,
        }
