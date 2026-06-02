from __future__ import annotations

from typing import Any


ROLE_TARGET_FRONT = 0
ROLE_TARGET_REAR = 1
LOSS_VERSION = "merge_safety_v3_masked"


def _require_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Merge-safety prediction loss requires torch.") from exc
    return torch


def masked_mean(values, mask):
    expanded = mask
    while expanded.ndim < values.ndim:
        expanded = expanded.unsqueeze(-1)
    expanded = expanded.expand_as(values).to(dtype=values.dtype)
    return (values * expanded).sum() / expanded.sum().clamp_min(1.0)


def _future_min_distance_error(pred_gap, target_gap, valid_mask):
    torch = _require_torch()
    valid = valid_mask > 0.0
    has_valid = valid.any(dim=(1, 2))
    large = torch.full_like(pred_gap, 1.0e6)
    pred_min = torch.where(valid, pred_gap, large).amin(dim=(1, 2))
    target_min = torch.where(valid, target_gap, large).amin(dim=(1, 2))
    error = torch.where(has_valid, torch.abs(pred_min - target_min), torch.zeros_like(pred_min))
    return masked_mean(error, has_valid.float())


def _role_gap_error(pred, target, ego_xy, valid_mask, role_ids, role_id: int):
    torch = _require_torch()
    role_mask = valid_mask * (role_ids == int(role_id)).float()[:, :, None]
    pred_dx = pred[..., 0] - ego_xy[..., 0]
    target_dx = target[..., 0] - ego_xy[..., 0]
    return masked_mean(torch.abs(pred_dx - target_dx), role_mask)


def _smoothness_error(pred, target, valid_mask):
    torch = _require_torch()
    if pred.shape[2] < 3:
        return pred.sum() * 0.0
    pred_delta2 = pred[:, :, 2:, :2] - 2.0 * pred[:, :, 1:-1, :2] + pred[:, :, :-2, :2]
    target_delta2 = target[:, :, 2:, :2] - 2.0 * target[:, :, 1:-1, :2] + target[:, :, :-2, :2]
    triplet_mask = valid_mask[:, :, 2:] * valid_mask[:, :, 1:-1] * valid_mask[:, :, :-2]
    return masked_mean(torch.linalg.norm(pred_delta2 - target_delta2, dim=-1), triplet_mask)


def _last_valid_error(distance, valid_mask):
    torch = _require_torch()
    horizon = int(distance.shape[2])
    indices = torch.arange(horizon, device=distance.device).view(1, 1, horizon)
    last_indices = torch.where(valid_mask > 0.0, indices, torch.full_like(indices, -1)).amax(dim=2)
    has_valid = last_indices >= 0
    gathered = distance.gather(dim=2, index=last_indices.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    return masked_mean(gathered, has_valid.float())


def _ordering_error(pred, valid_mask, role_ids):
    torch = _require_torch()
    front_mask = valid_mask * (role_ids == ROLE_TARGET_FRONT).float()[:, :, None]
    rear_mask = valid_mask * (role_ids == ROLE_TARGET_REAR).float()[:, :, None]
    front_count = front_mask.sum(dim=1)
    rear_count = rear_mask.sum(dim=1)
    front_x = (pred[..., 0] * front_mask).sum(dim=1) / front_count.clamp_min(1.0)
    rear_x = (pred[..., 0] * rear_mask).sum(dim=1) / rear_count.clamp_min(1.0)
    pair_mask = ((front_count > 0.0) & (rear_count > 0.0)).float()
    return masked_mean(torch.relu(rear_x - front_x + 4.8), pair_mask)


def merge_safety_loss(
    pred,
    target,
    mask,
    ego_future,
    role_ids,
    weights: dict[str, float] | None = None,
    future_valid_mask=None,
    ego_future_valid_mask=None,
) -> tuple[Any, dict[str, Any]]:
    torch = _require_torch()
    weights = weights or {}
    if future_valid_mask is None:
        future_valid_mask = mask[:, :, None].expand(pred.shape[0], pred.shape[1], pred.shape[2])
    if ego_future_valid_mask is None:
        ego_future_valid_mask = torch.ones(
            (pred.shape[0], pred.shape[2]),
            dtype=future_valid_mask.dtype,
            device=future_valid_mask.device,
        )
    valid_mask = future_valid_mask * mask[:, :, None] * ego_future_valid_mask[:, None, :]
    distance = torch.linalg.norm(pred[..., :2] - target[..., :2], dim=-1)
    ade = masked_mean(distance, valid_mask)
    fde = _last_valid_error(distance, valid_mask)
    ego_xy = ego_future[:, None, :, :2]
    pred_gap = torch.clamp(torch.linalg.norm(pred[..., :2] - ego_xy, dim=-1) - 3.0, min=0.0)
    target_gap = torch.clamp(torch.linalg.norm(target[..., :2] - ego_xy, dim=-1) - 3.0, min=0.0)
    min_dist = _future_min_distance_error(pred_gap, target_gap, valid_mask)
    front_gap = _role_gap_error(pred, target, ego_xy, valid_mask, role_ids, ROLE_TARGET_FRONT)
    rear_gap = _role_gap_error(pred, target, ego_xy, valid_mask, role_ids, ROLE_TARGET_REAR)
    ordering = _ordering_error(pred, valid_mask, role_ids)
    smoothness = _smoothness_error(pred, target, valid_mask)
    components = {
        "ade": ade,
        "fde": fde,
        "future_min_distance": min_dist,
        "target_lane_front_gap": front_gap,
        "target_lane_rear_gap": rear_gap,
        "ordering": ordering,
        "smoothness": smoothness,
    }
    total = (
        float(weights.get("ade", 1.0)) * ade
        + float(weights.get("fde", 0.5)) * fde
        + float(weights.get("future_min_distance", 0.75)) * min_dist
        + float(weights.get("target_lane_front_gap", 0.40)) * front_gap
        + float(weights.get("target_lane_rear_gap", 0.40)) * rear_gap
        + float(weights.get("ordering", 0.1)) * ordering
        + float(weights.get("smoothness", 0.05)) * smoothness
    )
    return total, components
