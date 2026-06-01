from __future__ import annotations

from typing import Any


ROLE_TARGET_FRONT = 0
ROLE_TARGET_REAR = 1
LOSS_VERSION = "merge_safety_v2"


def _require_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Merge-safety prediction loss requires torch.") from exc
    return torch


def masked_mean(values, mask):
    if values.ndim > mask.ndim:
        mask = mask.view(mask.shape[0], mask.shape[1], *([1] * (values.ndim - 2)))
    denom = mask.sum() * (values.numel() / max(mask.numel(), 1))
    if float(denom.detach().cpu()) <= 1.0e-8:
        return values.sum() * 0.0
    return (values * mask).sum() / denom


def _future_min_distance_error(pred_gap, target_gap, mask):
    torch = _require_torch()
    valid = mask > 0.0
    has_valid = valid.any(dim=1)
    if not torch.any(has_valid):
        return pred_gap.sum() * 0.0
    expanded_valid = valid[:, :, None].expand_as(pred_gap)
    large = torch.full_like(pred_gap, 1.0e6)
    pred_min = torch.where(expanded_valid, pred_gap, large).amin(dim=(1, 2))
    target_min = torch.where(expanded_valid, target_gap, large).amin(dim=(1, 2))
    return torch.abs(pred_min[has_valid] - target_min[has_valid]).mean()


def _role_gap_error(pred, target, ego_xy, mask, role_ids, role_id: int):
    torch = _require_torch()
    role_mask = mask * (role_ids == int(role_id)).float()
    pred_dx = pred[..., 0] - ego_xy[..., 0]
    target_dx = target[..., 0] - ego_xy[..., 0]
    return masked_mean(torch.abs(pred_dx - target_dx), role_mask)


def _smoothness_error(pred, target, mask):
    torch = _require_torch()
    if pred.shape[2] < 3:
        return pred.sum() * 0.0
    pred_delta2 = pred[:, :, 2:, :2] - 2.0 * pred[:, :, 1:-1, :2] + pred[:, :, :-2, :2]
    target_delta2 = target[:, :, 2:, :2] - 2.0 * target[:, :, 1:-1, :2] + target[:, :, :-2, :2]
    return masked_mean(torch.linalg.norm(pred_delta2 - target_delta2, dim=-1), mask)


def merge_safety_loss(
    pred,
    target,
    mask,
    ego_future,
    role_ids,
    weights: dict[str, float] | None = None,
) -> tuple[Any, dict[str, Any]]:
    torch = _require_torch()
    weights = weights or {}
    distance = torch.linalg.norm(pred[..., :2] - target[..., :2], dim=-1)
    ade = masked_mean(distance, mask)
    fde = masked_mean(distance[:, :, -1], mask)
    ego_xy = ego_future[:, None, :, :2]
    pred_gap = torch.clamp(torch.linalg.norm(pred[..., :2] - ego_xy, dim=-1) - 3.0, min=0.0)
    target_gap = torch.clamp(torch.linalg.norm(target[..., :2] - ego_xy, dim=-1) - 3.0, min=0.0)
    min_dist = _future_min_distance_error(pred_gap, target_gap, mask)
    front_gap = _role_gap_error(pred, target, ego_xy, mask, role_ids, ROLE_TARGET_FRONT)
    rear_gap = _role_gap_error(pred, target, ego_xy, mask, role_ids, ROLE_TARGET_REAR)
    ordering_penalties = []
    for row in range(pred.shape[0]):
        front = torch.where((role_ids[row] == ROLE_TARGET_FRONT) & (mask[row] > 0.0))[0]
        rear = torch.where((role_ids[row] == ROLE_TARGET_REAR) & (mask[row] > 0.0))[0]
        if front.numel() and rear.numel():
            ordering_penalties.append(torch.relu(pred[row, rear[0], :, 0] - pred[row, front[0], :, 0] + 4.8).mean())
    ordering = torch.stack(ordering_penalties).mean() if ordering_penalties else pred.sum() * 0.0
    smoothness = _smoothness_error(pred, target, mask)
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
