"""ACCVP network, loss and tightly coupled checkpoint metadata contract."""

from __future__ import annotations

from typing import Any

from safe_rl.accvp.contracts.schema import COUNTERFACTUAL_SCHEMA_VERSION
from safe_rl.accvp.modeling.omitted_actor_summary import (
    ACCVPOmittedActorSummaryAdapter,
    omitted_actor_summary_config,
)
from safe_rl.prediction.wcdt_v3_predictor import (
    ARCHITECTURE_VERSION as WCDT_V3_ARCHITECTURE_VERSION,
    WcDTV3TemporalInteractionPredictor,
)


ACCVP_ARCHITECTURE_VERSION = "accvp_v1_conditional_response_transformer"
ACCVP_HYBRID_ARCHITECTURE_VERSION = (
    "accvp_v2_hybrid_omitted_actor_summary_adapter"
)
ACCVP_LOSS_VERSION = "accvp_loss_v2"
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
        omitted_actor_summary_feature_dim: int = 0,
        omitted_actor_summary_group_count: int = 0,
    ):
        torch, nn, _functional = _require_torch()
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.response_horizon_steps = int(response_horizon_steps)
        self.candidate_plan_horizon_steps = int(candidate_plan_horizon_steps)
        self.omitted_actor_summary_feature_dim = int(
            omitted_actor_summary_feature_dim
        )
        self.omitted_actor_summary_group_count = int(
            omitted_actor_summary_group_count
        )
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
        self.omitted_actor_summary_adapter = None
        if self.omitted_actor_summary_feature_dim > 0:
            if self.omitted_actor_summary_group_count <= 0:
                raise ValueError(
                    "hybrid omitted-actor model requires a positive group count"
                )
            self.omitted_actor_summary_adapter = (
                ACCVPOmittedActorSummaryAdapter(
                    feature_dim=self.omitted_actor_summary_feature_dim,
                    group_count=self.omitted_actor_summary_group_count,
                    hidden_dim=self.hidden_dim,
                )
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
        omitted_actor_summary_features=None,
        omitted_actor_summary_group_mask=None,
        omitted_actor_summary_mask=None,
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
        scene_mask = actor_mask
        if self.omitted_actor_summary_adapter is not None:
            if (
                omitted_actor_summary_features is None
                or omitted_actor_summary_group_mask is None
                or omitted_actor_summary_mask is None
            ):
                raise ValueError(
                    "hybrid ACCVP scene encoding requires omitted-actor "
                    "summary features and masks"
                )
            summary_token = self.omitted_actor_summary_adapter(
                omitted_actor_summary_features,
                omitted_actor_summary_group_mask,
                omitted_actor_summary_mask,
            )
            actor_tokens = torch.cat([actor_tokens, summary_token], dim=1)
            scene_mask = torch.cat(
                [actor_mask, omitted_actor_summary_mask], dim=1
            )
        padding = scene_mask <= 0.0
        safe_actor_padding = padding.clone()
        all_actor_padding = safe_actor_padding.all(dim=1)
        if bool(all_actor_padding.any()):
            safe_actor_padding[all_actor_padding, 0] = False
        return self.scene.actor_encoder(actor_tokens, src_key_padding_mask=safe_actor_padding)

    def forward_from_scene(
        self,
        scene_tokens,
        actor_mask,
        candidate_plan,
        candidate_action_ids,
        omitted_actor_summary_mask=None,
    ):
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
        physical_actor_count = int(actor_mask.shape[1])
        if int(conditioned.shape[1]) == physical_actor_count + 1:
            if omitted_actor_summary_mask is None:
                raise ValueError(
                    "hybrid ACCVP candidate scoring requires the summary mask"
                )
            scene_mask = torch.cat(
                [actor_mask, omitted_actor_summary_mask], dim=1
            )
        elif int(conditioned.shape[1]) == physical_actor_count:
            scene_mask = actor_mask
        else:
            raise ValueError(
                "ACCVP scene token count does not match physical actor rows"
            )
        physical_conditioned = conditioned[:, :physical_actor_count]
        response = self.response_decoder(physical_conditioned).view(
            physical_conditioned.shape[0],
            physical_actor_count,
            self.response_horizon_steps,
            5,
        )
        response = response * actor_mask[:, :, None, None]
        weights = scene_mask / scene_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
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
        omitted_actor_summary_features=None,
        omitted_actor_summary_group_mask=None,
        omitted_actor_summary_mask=None,
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
            omitted_actor_summary_features,
            omitted_actor_summary_group_mask,
            omitted_actor_summary_mask,
        )
        return self.forward_from_scene(
            scene,
            actor_mask,
            candidate_plan,
            candidate_action_ids,
            omitted_actor_summary_mask,
        )


def model_kwargs_from_config(config: Any) -> dict[str, Any]:
    prediction = config.prediction
    result = {
        "history_steps": int(config.scenario.history_steps),
        "response_horizon_steps": int(config.accvp.response_horizon_steps),
        "candidate_plan_horizon_steps": int(config.accvp.candidate_plan_horizon_steps),
        "hidden_dim": int(prediction.get("wcdt_v3_hidden_dim", 128)),
        "temporal_layers": int(prediction.get("wcdt_v3_temporal_layers", 2)),
        "actor_attention_layers": int(prediction.get("wcdt_v3_actor_attention_layers", 2)),
        "num_heads": int(prediction.get("wcdt_v3_num_heads", 4)),
        "dropout": float(prediction.get("wcdt_v3_dropout", 0.1)),
    }
    summary = omitted_actor_summary_config(config)
    if bool(summary["enabled"]):
        result.update(
            {
                "omitted_actor_summary_feature_dim": int(
                    summary["feature_dim"]
                ),
                "omitted_actor_summary_group_count": int(
                    summary["group_count"]
                ),
            }
        )
    return result


def architecture_version_from_config(config: Any) -> str:
    return (
        ACCVP_HYBRID_ARCHITECTURE_VERSION
        if bool(omitted_actor_summary_config(config)["enabled"])
        else ACCVP_ARCHITECTURE_VERSION
    )


def warm_start_scene_encoder(model: ACCVPPredictor, v3_state_dict: dict[str, Any]) -> dict[str, list[str]]:
    """Load WcDT-v3 temporal/actor encoder weights, leaving ACCVP heads random."""

    source = {key: value for key, value in v3_state_dict.items() if not key.startswith("decoder.")}
    result = model.scene.load_state_dict(source, strict=False)
    return {"missing": list(result.missing_keys), "unexpected": list(result.unexpected_keys)}


def set_scene_encoder_trainable(model: ACCVPPredictor, trainable: bool) -> None:
    for parameter in model.scene.parameters():
        parameter.requires_grad = bool(trainable)


def accvp_loss(output: dict[str, Any], batch: dict[str, Any], weights: dict[str, Any] | None = None):
    """Masked v2 loss with per-scalar normalization and residual smoothness."""

    torch, _nn, functional = _require_torch()
    weights = weights or {}
    response = output["actor_response"]
    response_target = batch["actor_response"]
    response_mask = batch["actor_response_mask"]
    if response.shape != response_target.shape or response.ndim != 4 or response.shape[-1] != 5:
        raise ValueError(
            "actor_response prediction/target must have identical [batch,actor,time,5] shapes"
        )
    if response_mask.shape != response.shape[:-1]:
        raise ValueError("actor_response_mask must match actor_response [batch,actor,time]")
    if output["event_logits"].shape != batch["event_targets"].shape or output["event_logits"].shape[-1] != 4:
        raise ValueError("event logits/targets must have identical [batch,4] shapes")
    if batch["event_mask"].shape != output["event_logits"].shape:
        raise ValueError("event_mask must match event logits")
    if output["geometry"].shape != batch["geometry_targets"].shape or output["geometry"].shape[-1] != 5:
        raise ValueError("geometry prediction/target must have identical [batch,5] shapes")
    candidate_plan = batch["candidate_plan"]
    if candidate_plan.ndim != 3 or candidate_plan.shape[0] != response.shape[0] or candidate_plan.shape[-1] != 5:
        raise ValueError("candidate_plan must have shape [batch,time,5]")
    if candidate_plan.shape[1] < response.shape[2]:
        raise ValueError("candidate_plan horizon must cover the response horizon")
    sample_weight = batch.get("sample_weight")
    if sample_weight is None:
        sample_weight = torch.ones((response.shape[0],), dtype=response.dtype, device=response.device)
    sample_weight = sample_weight.reshape(-1).to(dtype=response.dtype, device=response.device)
    if sample_weight.shape != (response.shape[0],):
        raise ValueError("sample_weight must have shape [batch]")
    if not bool(torch.isfinite(sample_weight).all()) or bool((sample_weight <= 0.0).any()):
        raise ValueError("sample_weight must contain finite positive values")
    response_sample_weight = sample_weight[:, None, None, None]
    actor_mask = batch["actor_response_mask"][:, :, :, None]
    response_delta = response - response_target
    # Heading is circular; comparing raw radians creates an artificial jump at
    # +/-pi. Remaining features retain their physical units unless an explicit
    # train-only scale vector is supplied in the loss configuration.
    response_delta = torch.stack(
        [
            response_delta[..., 0],
            response_delta[..., 1],
            torch.atan2(torch.sin(response_delta[..., 2]), torch.cos(response_delta[..., 2])),
            response_delta[..., 3],
            response_delta[..., 4],
        ],
        dim=-1,
    )
    feature_scales = torch.as_tensor(
        weights.get("response_feature_scales", [1.0, 1.0, 1.0, 1.0, 1.0]),
        dtype=response_delta.dtype,
        device=response_delta.device,
    )
    if feature_scales.shape != (5,) or bool((feature_scales <= 0.0).any()):
        raise ValueError("response_feature_scales must contain five positive values")
    normalized_delta = response_delta / feature_scales
    response_error = functional.smooth_l1_loss(
        normalized_delta,
        torch.zeros_like(normalized_delta),
        reduction="none",
    )
    response_feature_mask = actor_mask.expand_as(response_error) * response_sample_weight
    trajectory_numerator = (response_error * response_feature_mask).sum()
    trajectory_denominator = response_feature_mask.sum()
    trajectory = trajectory_numerator / trajectory_denominator.clamp_min(
        torch.finfo(trajectory_denominator.dtype).eps
    )
    event_logits = output["event_logits"]
    event_targets = batch["event_targets"]
    event_mask = batch["event_mask"]
    positive_weights = weights.get("event_positive_weights")
    pos_weight = None
    if positive_weights is not None:
        pos_weight = torch.as_tensor(positive_weights, dtype=event_logits.dtype, device=event_logits.device)
        if pos_weight.shape != (4,):
            raise ValueError("event_positive_weights must contain four values")
    event_loss_weights = torch.as_tensor(
        weights.get("event_loss_weights", [1.0, 1.0, 1.0, 1.0]),
        dtype=event_logits.dtype,
        device=event_logits.device,
    )
    if event_loss_weights.shape != (4,) or bool((event_loss_weights < 0.0).any()):
        raise ValueError("event_loss_weights must contain four non-negative values")
    event_loss = functional.binary_cross_entropy_with_logits(
        event_logits,
        event_targets,
        reduction="none",
        pos_weight=pos_weight,
    )
    weighted_event_mask = event_mask * event_loss_weights[None, :] * sample_weight[:, None]
    event_numerator = (event_loss * weighted_event_mask).sum()
    event_denominator = weighted_event_mask.sum()
    event_loss = event_numerator / event_denominator.clamp_min(
        torch.finfo(event_denominator.dtype).eps
    )
    geometry_error = functional.smooth_l1_loss(output["geometry"], batch["geometry_targets"], reduction="none")
    geometry_mask = batch.get("geometry_mask", torch.ones_like(geometry_error))
    if geometry_mask.shape != geometry_error.shape:
        raise ValueError("geometry_mask must match geometry prediction/target [batch,5]")
    quantile_error = torch.maximum(
        0.10 * (batch["geometry_targets"][:, 0] - output["geometry"][:, 0]),
        (0.10 - 1.0) * (batch["geometry_targets"][:, 0] - output["geometry"][:, 0]),
    ) + torch.maximum(
        0.90 * (batch["geometry_targets"][:, 1] - output["geometry"][:, 1]),
        (0.90 - 1.0) * (batch["geometry_targets"][:, 1] - output["geometry"][:, 1]),
    )
    geometry_numerator = (
        (quantile_error * sample_weight).sum()
        + (geometry_error[:, 2:] * geometry_mask[:, 2:] * sample_weight[:, None]).sum()
    )
    geometry_denominator = (
        2.0 * sample_weight.sum()
        + (geometry_mask[:, 2:] * sample_weight[:, None]).sum()
    )
    geometry_loss = geometry_numerator / geometry_denominator.clamp_min(
        torch.finfo(geometry_denominator.dtype).eps
    )
    response_residual_xy = response[..., :2] - response_target[..., :2]
    residual_second_difference = (
        response_residual_xy[:, :, 2:]
        - 2.0 * response_residual_xy[:, :, 1:-1]
        + response_residual_xy[:, :, :-2]
    )
    smooth_mask = actor_mask[:, :, 2:] * actor_mask[:, :, 1:-1] * actor_mask[:, :, :-2]
    smooth_feature_mask = (
        smooth_mask.expand_as(residual_second_difference) * response_sample_weight
    )
    smoothness_numerator = (
        residual_second_difference.abs() * smooth_feature_mask
    ).sum()
    smoothness_denominator = smooth_feature_mask.sum()
    smoothness = smoothness_numerator / smoothness_denominator.clamp_min(
        torch.finfo(smoothness_denominator.dtype).eps
    )
    ego_x = batch["candidate_plan"][:, None, : output["actor_response"].shape[2], 0]
    target_sign = torch.sign(batch["actor_response"][..., 0] - ego_x)
    predicted_relative = output["actor_response"][..., 0] - ego_x
    ordering_weight = actor_mask[..., 0] * sample_weight[:, None, None]
    ordering_numerator = (torch.relu(-target_sign * predicted_relative) * ordering_weight).sum()
    ordering_denominator = ordering_weight.sum()
    ordering = ordering_numerator / ordering_denominator.clamp_min(
        torch.finfo(ordering_denominator.dtype).eps
    )
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
        "statistics": {
            "trajectory": {
                "numerator": trajectory_numerator.detach(),
                "denominator": trajectory_denominator.detach(),
            },
            "events": {
                "numerator": event_numerator.detach(),
                "denominator": event_denominator.detach(),
            },
            "geometry": {
                "numerator": geometry_numerator.detach(),
                "denominator": geometry_denominator.detach(),
            },
            "ordering": {
                "numerator": ordering_numerator.detach(),
                "denominator": ordering_denominator.detach(),
            },
            "smoothness": {
                "numerator": smoothness_numerator.detach(),
                "denominator": smoothness_denominator.detach(),
            },
        },
    }


def checkpoint_metadata(config: Any, *, warm_start: dict[str, Any]) -> dict[str, Any]:
    return {
        "architecture_version": architecture_version_from_config(config),
        "loss_version": ACCVP_LOSS_VERSION,
        "counterfactual_schema_version": COUNTERFACTUAL_SCHEMA_VERSION,
        "wcdt_v3_architecture_version": WCDT_V3_ARCHITECTURE_VERSION,
        "model_kwargs": model_kwargs_from_config(config),
        "omitted_actor_summary": omitted_actor_summary_config(config),
        "warm_start": warm_start,
    }
