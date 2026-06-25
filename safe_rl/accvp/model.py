from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from safe_rl.accvp.schema import COUNTERFACTUAL_SCHEMA_VERSION
from safe_rl.prediction.wcdt_v3_predictor import (
    ARCHITECTURE_VERSION as WCDT_V3_ARCHITECTURE_VERSION,
    WcDTV3TemporalInteractionPredictor,
)


ACCVP_ARCHITECTURE_VERSION = "accvp_v1_conditional_response_transformer"
EVENT_NAMES = (
    "proxy_collision",
    "safety_violation",
    "taper_miss",
    "merge_before_taper",
)


def _require_torch():
    try:
        import torch
        from torch import nn
        from torch.nn import functional as functional
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError("ACCVP training/inference requires torch.") from exc
    return torch, nn, functional


try:  # Keep config/schema-only imports usable without torch.
    _torch_for_base, _nn_for_base, _functional_for_base = _require_torch()
    _ModuleBase = _nn_for_base.Module
except ImportError:  # pragma: no cover
    _ModuleBase = object


class ACCVPPredictor(_ModuleBase):
    """WcDT-v3 warm-startable scene encoder plus candidate-conditioned heads."""

    def __init__(
        self,
        *,
        history_steps: int,
        response_horizon_steps: int,
        candidate_plan_horizon_steps: int,
        hidden_dim: int = 128,
        temporal_layers: int = 2,
        actor_attention_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        torch, nn, _functional = _require_torch()
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.response_horizon_steps = int(response_horizon_steps)
        self.candidate_plan_horizon_steps = int(candidate_plan_horizon_steps)
        # The legacy decoder exists only to make WcDT-v3 state loading simple;
        # ACCVP never calls it for inference.
        self.scene = WcDTV3TemporalInteractionPredictor(
            history_steps=int(history_steps),
            horizon_steps=int(response_horizon_steps),
            hidden_dim=self.hidden_dim,
            temporal_layers=int(temporal_layers),
            actor_attention_layers=int(actor_attention_layers),
            num_heads=int(num_heads),
            dropout=float(dropout),
        )
        self.action_embedding = nn.Embedding(9, 16)
        self.plan_encoder = nn.GRU(5, self.hidden_dim, batch_first=True)
        self.candidate_projection = nn.Sequential(
            nn.Linear(self.hidden_dim + 16, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.relation_bias = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.response_decoder = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.response_horizon_steps * 5),
        )
        self.event_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, len(EVENT_NAMES)),
        )
        self.geometry_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, 5),
        )
        self.scene_encode_calls = 0

    def encode_scene(
        self,
        history_features,
        history_valid_mask,
        history_lane_ids,
        history_edge_role_ids,
        role_ids,
        lane_ids,
        edge_role_ids,
        actor_mask,
    ):
        """Encode root scene once; callers may fan it out to all candidates."""

        torch, _nn, _functional = _require_torch()
        self.scene_encode_calls += 1
        batch, actors, history_steps, _ = history_features.shape
        temporal_route = self.scene.temporal_route_projection(
            torch.cat(
                [
                    self.scene.lane_embedding(history_lane_ids),
                    self.scene.edge_role_embedding(history_edge_role_ids),
                ],
                dim=-1,
            )
        )
        temporal = self.scene.history_projection(history_features)
        temporal = temporal + temporal_route + self.scene.position_embedding[:, :, :history_steps]
        temporal = temporal.reshape(batch * actors, history_steps, -1)
        temporal_padding = (history_valid_mask <= 0.0).reshape(batch * actors, history_steps)
        safe_padding = temporal_padding.clone()
        all_padding = safe_padding.all(dim=1)
        if bool(all_padding.any()):
            safe_padding[all_padding, 0] = False
        temporal = self.scene.temporal_encoder(temporal, src_key_padding_mask=safe_padding)
        valid_indices = (~temporal_padding).long() * torch.arange(
            history_steps, device=history_features.device
        ).view(1, history_steps)
        last_valid = valid_indices.amax(dim=1)
        temporal = temporal.gather(1, last_valid[:, None, None].expand(-1, 1, temporal.shape[-1])).squeeze(1)
        temporal = temporal.reshape(batch, actors, -1)
        static = self.scene.static_projection(
            torch.cat(
                [
                    self.scene.role_embedding(role_ids),
                    self.scene.lane_embedding(lane_ids),
                    self.scene.edge_role_embedding(edge_role_ids),
                ],
                dim=-1,
            )
        )
        actor_tokens = temporal + static
        padding = actor_mask <= 0.0
        safe_actor_padding = padding.clone()
        all_actor_padding = safe_actor_padding.all(dim=1)
        if bool(all_actor_padding.any()):
            safe_actor_padding[all_actor_padding, 0] = False
        return self.scene.actor_encoder(actor_tokens, src_key_padding_mask=safe_actor_padding)

    def forward_from_scene(self, scene_tokens, actor_mask, candidate_plan, candidate_action_ids):
        """Score a batch of candidate plans using previously encoded root scenes."""

        torch, _nn, _functional = _require_torch()
        _plan_sequence, plan_hidden = self.plan_encoder(candidate_plan)
        plan_hidden = plan_hidden[-1]
        candidate = self.candidate_projection(
            torch.cat([plan_hidden, self.action_embedding(candidate_action_ids)], dim=-1)
        )
        candidate_tokens = candidate[:, None, :].expand(-1, scene_tokens.shape[1], -1)
        relation = self.relation_bias(torch.cat([scene_tokens, candidate_tokens], dim=-1))
        conditioned = scene_tokens + relation
        response = self.response_decoder(conditioned).view(
            conditioned.shape[0], conditioned.shape[1], self.response_horizon_steps, 5
        )
        response = response * actor_mask[:, :, None, None]
        weights = actor_mask / actor_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled = (conditioned * weights[:, :, None]).sum(dim=1)
        return {
            "actor_response": response,
            "event_logits": self.event_head(pooled),
            "geometry": self.geometry_head(pooled),
        }

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
        candidate_plan,
        candidate_action_ids,
    ):
        scene = self.encode_scene(
            history_features,
            history_valid_mask,
            history_lane_ids,
            history_edge_role_ids,
            role_ids,
            lane_ids,
            edge_role_ids,
            actor_mask,
        )
        return self.forward_from_scene(scene, actor_mask, candidate_plan, candidate_action_ids)


def model_kwargs_from_config(config: Any) -> dict[str, Any]:
    prediction = config.prediction
    return {
        "history_steps": int(config.scenario.history_steps),
        "response_horizon_steps": int(config.accvp.response_horizon_steps),
        "candidate_plan_horizon_steps": int(config.accvp.candidate_plan_horizon_steps),
        "hidden_dim": int(prediction.get("wcdt_v3_hidden_dim", 128)),
        "temporal_layers": int(prediction.get("wcdt_v3_temporal_layers", 2)),
        "actor_attention_layers": int(prediction.get("wcdt_v3_actor_attention_layers", 2)),
        "num_heads": int(prediction.get("wcdt_v3_num_heads", 4)),
        "dropout": float(prediction.get("wcdt_v3_dropout", 0.1)),
    }


def warm_start_scene_encoder(model: ACCVPPredictor, v3_state_dict: dict[str, Any]) -> dict[str, list[str]]:
    """Load WcDT-v3 temporal/actor encoder weights, leaving ACCVP heads random."""

    source = {key: value for key, value in v3_state_dict.items() if not key.startswith("decoder.")}
    result = model.scene.load_state_dict(source, strict=False)
    return {"missing": list(result.missing_keys), "unexpected": list(result.unexpected_keys)}


def set_scene_encoder_trainable(model: ACCVPPredictor, trainable: bool) -> None:
    for parameter in model.scene.parameters():
        parameter.requires_grad = bool(trainable)


def accvp_loss(output: dict[str, Any], batch: dict[str, Any], weights: dict[str, float] | None = None):
    """Masked v1 loss; viability is explicitly masked for censored examples."""

    torch, _nn, functional = _require_torch()
    weights = weights or {}
    actor_mask = batch["actor_response_mask"][:, :, :, None]
    response_error = functional.smooth_l1_loss(output["actor_response"], batch["actor_response"], reduction="none")
    trajectory = (response_error * actor_mask).sum() / actor_mask.sum().clamp_min(1.0)
    event_logits = output["event_logits"]
    event_targets = batch["event_targets"]
    event_mask = batch["event_mask"]
    positive_weights = weights.get("event_positive_weights")
    pos_weight = None
    if positive_weights is not None:
        pos_weight = torch.as_tensor(positive_weights, dtype=event_logits.dtype, device=event_logits.device)
    event_loss = functional.binary_cross_entropy_with_logits(
        event_logits,
        event_targets,
        reduction="none",
        pos_weight=pos_weight,
    )
    event_loss = (event_loss * event_mask).sum() / event_mask.sum().clamp_min(1.0)
    geometry_error = functional.smooth_l1_loss(output["geometry"], batch["geometry_targets"], reduction="none")
    geometry_mask = batch.get("geometry_mask", torch.ones_like(geometry_error))
    quantile_error = torch.maximum(
        0.10 * (batch["geometry_targets"][:, 0] - output["geometry"][:, 0]),
        (0.10 - 1.0) * (batch["geometry_targets"][:, 0] - output["geometry"][:, 0]),
    ) + torch.maximum(
        0.90 * (batch["geometry_targets"][:, 1] - output["geometry"][:, 1]),
        (0.90 - 1.0) * (batch["geometry_targets"][:, 1] - output["geometry"][:, 1]),
    )
    geometry_loss = (
        quantile_error.sum()
        + (geometry_error[:, 2:] * geometry_mask[:, 2:]).sum()
    ) / (2.0 * geometry_error.shape[0] + geometry_mask[:, 2:].sum()).clamp_min(1.0)
    response_xy = output["actor_response"][..., :2]
    response_delta = response_xy[:, :, 1:] - response_xy[:, :, :-1]
    smooth_mask = actor_mask[:, :, 1:] * actor_mask[:, :, :-1]
    smoothness = (response_delta.abs().sum(dim=-1) * smooth_mask[..., 0]).sum() / smooth_mask.sum().clamp_min(1.0)
    ego_x = batch["candidate_plan"][:, None, : output["actor_response"].shape[2], 0]
    target_sign = torch.sign(batch["actor_response"][..., 0] - ego_x)
    predicted_relative = output["actor_response"][..., 0] - ego_x
    ordering = (torch.relu(-target_sign * predicted_relative) * actor_mask[..., 0]).sum() / actor_mask.sum().clamp_min(1.0)
    total = (
        float(weights.get("trajectory", 1.0)) * trajectory
        + float(weights.get("events", 1.0)) * event_loss
        + float(weights.get("geometry", 0.25)) * geometry_loss
        + float(weights.get("ordering", 0.10)) * ordering
        + float(weights.get("smoothness", 0.01)) * smoothness
    )
    return total, {
        "trajectory": trajectory.detach(),
        "events": event_loss.detach(),
        "geometry": geometry_loss.detach(),
        "ordering": ordering.detach(),
        "smoothness": smoothness.detach(),
    }


def checkpoint_metadata(config: Any, *, warm_start: dict[str, Any]) -> dict[str, Any]:
    return {
        "architecture_version": ACCVP_ARCHITECTURE_VERSION,
        "counterfactual_schema_version": COUNTERFACTUAL_SCHEMA_VERSION,
        "wcdt_v3_architecture_version": WCDT_V3_ARCHITECTURE_VERSION,
        "model_kwargs": model_kwargs_from_config(config),
        "warm_start": warm_start,
    }
