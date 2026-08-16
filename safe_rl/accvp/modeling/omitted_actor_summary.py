"""Deterministic omitted-actor aggregation and the ACCVP-only token adapter.

The physical WcDT rows remain unchanged.  This module builds a bounded side
channel from vehicles that did not receive a physical row and projects it to
one scene token owned by :class:`ACCVPPredictor`.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import numpy as np

from safe_rl.accvp.contracts.schema import stable_hash


OMITTED_ACTOR_SUMMARY_CONTRACT_VERSION = "accvp_omitted_actor_summary_v1"
OMITTED_ACTOR_SUMMARY_GROUPS = (
    "conflict_surface",
    "target_lane_front",
    "target_lane_rear",
    "auxiliary_front",
    "auxiliary_rear",
    "ramp_front",
    "ramp_rear",
    "other",
)
OMITTED_ACTOR_SUMMARY_FEATURES = (
    "log1p_actor_count",
    "actor_count_fraction",
    "minimum_ttc",
    "minimum_ttc_valid",
    "minimum_surface_gap",
    "minimum_effective_gap",
    "earliest_conflict_time",
    "earliest_conflict_time_valid",
    "mean_relative_speed",
    "max_absolute_relative_speed",
    "maximum_closing_speed",
    "overflow_indicator",
)
OMITTED_ACTOR_SUMMARY_TENSOR_FIELDS = (
    "omitted_actor_summary_features",
    "omitted_actor_summary_group_mask",
    "omitted_actor_summary_mask",
)
PHYSICAL_ACTOR_MANDATORY_CLASSES = (
    "target_front",
    "target_rear",
    "candidate_conflict_eligible",
    "nearest_candidate_conflict",
    "lowest_conflict_ttc",
    "critical_auxiliary_or_ramp_local",
)
PHYSICAL_ACTOR_PRIORITY_ORDER = (
    "relevance_class",
    "role",
    "earliest_conflict_time_s",
    "ttc",
    "effective_gap",
    "current_surface_gap",
    "minimum_swept_obb_gap",
    "vehicle_id",
)

_TIME_SCALE_S = 20.0
_SURFACE_GAP_SCALE_M = 200.0
_EFFECTIVE_GAP_SCALE_M = 100.0
_RELATIVE_SPEED_SCALE_MPS = 40.0
_COUNT_SCALE = math.log1p(32.0)


def _value(item: Any, name: str, default: Any) -> Any:
    if isinstance(item, Mapping):
        return item.get(name, default)
    return getattr(item, name, default)


def _finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _finite_time(value: Any) -> float | None:
    result = _finite(value)
    # Selector metadata uses a large finite sentinel for "no TTC/conflict".
    return result if result is not None and 0.0 <= result < 1.0e5 else None


def omitted_actor_summary_config(config: Any) -> dict[str, Any]:
    configured = dict(config.accvp.get("omitted_actor_summary", {}) or {})
    enabled = bool(configured.get("enabled", False))
    version = str(
        configured.get(
            "contract_version",
            OMITTED_ACTOR_SUMMARY_CONTRACT_VERSION,
        )
    )
    if enabled and version != OMITTED_ACTOR_SUMMARY_CONTRACT_VERSION:
        raise ValueError(
            "unsupported accvp.omitted_actor_summary.contract_version: "
            f"{version!r}"
        )
    physical_capacity = int(
        configured.get("physical_actor_capacity", config.accvp.actor_count)
        if enabled
        else config.accvp.actor_count
    )
    if enabled and physical_capacity != int(config.accvp.actor_count):
        raise ValueError(
            "omitted-actor summary physical capacity must equal accvp.actor_count"
        )
    return {
        "enabled": enabled,
        "contract_version": version,
        "physical_actor_capacity": physical_capacity,
        "group_names": list(OMITTED_ACTOR_SUMMARY_GROUPS),
        "feature_names": list(OMITTED_ACTOR_SUMMARY_FEATURES),
        "group_count": len(OMITTED_ACTOR_SUMMARY_GROUPS),
        "feature_dim": len(OMITTED_ACTOR_SUMMARY_FEATURES),
        "summary_token_count": 1,
        "response_rows_include_summary": False,
        "mandatory_physical_actor_classes": list(
            PHYSICAL_ACTOR_MANDATORY_CLASSES
        ),
        "physical_actor_priority_order": list(PHYSICAL_ACTOR_PRIORITY_ORDER),
        "critical_overflow_policy": "fail_closed",
        "risk_uses_full_vehicle_set": True,
    }


def _metadata_map(selection: Any) -> dict[str, Any]:
    raw = _value(selection, "actor_metadata", {}) or {}
    return {str(vehicle_id): item for vehicle_id, item in dict(raw).items()}


def _selected_ids(selection: Any) -> set[str]:
    return {
        str(value)
        for value in (_value(selection, "selected_actor_ids", ()) or ())
        if str(value)
    }


def _semantic_group(item: Any) -> str:
    earliest = _finite_time(_value(item, "earliest_conflict_time_s", None))
    if (
        bool(_value(item, "candidate_conflict_eligible", False))
        or bool(_value(item, "nearest_candidate_conflict", False))
        or earliest is not None
    ):
        return "conflict_surface"
    role = str(_value(item, "role", "other"))
    signed_gap = _finite(_value(item, "signed_longitudinal_gap", None))
    if role == "target_front":
        direction = "front"
    elif role == "target_rear":
        direction = "rear"
    elif signed_gap is None:
        return "other"
    else:
        direction = "front" if signed_gap >= 0.0 else "rear"
    if role in {"target_front", "target_rear", "target_lane_other"}:
        return f"target_lane_{direction}"
    if role == "auxiliary_local":
        return f"auxiliary_{direction}"
    if role == "ramp_local":
        return f"ramp_{direction}"
    return "other"


def _state_speed(state: Any) -> float | None:
    return _finite(_value(state, "speed", None))


@dataclass(frozen=True)
class OmittedActorSummary:
    features: np.ndarray
    group_mask: np.ndarray
    summary_mask: np.ndarray
    omitted_actor_ids: tuple[str, ...]
    group_counts: tuple[int, ...]
    tensor_hash: str

    def tensors(self, *, leading_batch: bool = True) -> dict[str, np.ndarray]:
        values = {
            "omitted_actor_summary_features": self.features.astype(
                np.float32, copy=False
            ),
            "omitted_actor_summary_group_mask": self.group_mask.astype(
                np.float32, copy=False
            ),
            "omitted_actor_summary_mask": self.summary_mask.astype(
                np.float32, copy=False
            ),
        }
        if leading_batch:
            return {name: value[None, ...] for name, value in values.items()}
        return values

    def metadata(self) -> dict[str, Any]:
        return {
            "contract_version": OMITTED_ACTOR_SUMMARY_CONTRACT_VERSION,
            "group_names": list(OMITTED_ACTOR_SUMMARY_GROUPS),
            "feature_names": list(OMITTED_ACTOR_SUMMARY_FEATURES),
            "omitted_actor_ids": list(self.omitted_actor_ids),
            "omitted_actor_count": len(self.omitted_actor_ids),
            "group_counts": {
                name: int(count)
                for name, count in zip(
                    OMITTED_ACTOR_SUMMARY_GROUPS, self.group_counts
                )
            },
            "tensor_hash": self.tensor_hash,
        }


def build_omitted_actor_summary(
    selection: Any,
    *,
    actor_capacity: int,
    latest_states: Mapping[str, Any] | None = None,
    ego_state: Any | None = None,
) -> OmittedActorSummary:
    """Aggregate every unselected vehicle into fixed semantic groups.

    Vehicle IDs affect deterministic ordering and the audit record, but never
    enter the model tensor.  Relative speed is read from the exact latest
    history frame used by both collection and runtime.
    """

    metadata = _metadata_map(selection)
    selected = _selected_ids(selection)
    omitted_ids = tuple(sorted(set(metadata) - selected))
    grouped: dict[str, list[tuple[str, Any]]] = {
        name: [] for name in OMITTED_ACTOR_SUMMARY_GROUPS
    }
    for vehicle_id in omitted_ids:
        item = metadata[vehicle_id]
        grouped[_semantic_group(item)].append((vehicle_id, item))

    state_map = {str(key): value for key, value in dict(latest_states or {}).items()}
    ego_speed = _state_speed(ego_state)
    if omitted_ids and ego_speed is None:
        raise ValueError(
            "omitted-actor summary requires the ego speed from the latest frame"
        )
    missing_actor_speeds = [
        vehicle_id
        for vehicle_id in omitted_ids
        if _state_speed(state_map.get(vehicle_id)) is None
    ]
    if missing_actor_speeds:
        raise ValueError(
            "omitted-actor summary is missing latest actor speeds: "
            f"{missing_actor_speeds[:10]}"
        )
    features = np.zeros(
        (len(OMITTED_ACTOR_SUMMARY_GROUPS), len(OMITTED_ACTOR_SUMMARY_FEATURES)),
        dtype=np.float32,
    )
    group_mask = np.zeros((len(OMITTED_ACTOR_SUMMARY_GROUPS),), dtype=np.float32)
    counts: list[int] = []
    overflow = float(bool(omitted_ids))
    capacity = max(1, int(actor_capacity))
    for group_index, group_name in enumerate(OMITTED_ACTOR_SUMMARY_GROUPS):
        members = grouped[group_name]
        count = len(members)
        counts.append(count)
        if not members:
            continue
        group_mask[group_index] = 1.0
        ttc_values = [
            value
            for _vehicle_id, item in members
            if (value := _finite_time(_value(item, "ttc", None))) is not None
        ]
        surface_gaps = [
            value
            for _vehicle_id, item in members
            if (
                value := _finite(_value(item, "current_surface_gap", None))
            )
            is not None
        ]
        effective_gaps = [
            value
            for _vehicle_id, item in members
            if (value := _finite(_value(item, "effective_gap", None)))
            is not None
        ]
        conflict_times = [
            value
            for _vehicle_id, item in members
            if (
                value := _finite(
                    _value(item, "earliest_conflict_time_s", None)
                )
            )
            is not None and value < 1.0e5
        ]
        relative_speeds: list[float] = []
        for vehicle_id, _item in members:
            actor_speed = _state_speed(state_map.get(vehicle_id))
            if actor_speed is not None and ego_speed is not None:
                relative_speeds.append(float(actor_speed - ego_speed))
        closing_speeds = [
            max(0.0, value)
            for _vehicle_id, item in members
            if (value := _finite(_value(item, "closing_speed", None)))
            is not None
        ]
        min_ttc = min(ttc_values) if ttc_values else _TIME_SCALE_S
        min_surface = (
            min(surface_gaps) if surface_gaps else _SURFACE_GAP_SCALE_M
        )
        min_effective = (
            min(effective_gaps) if effective_gaps else _EFFECTIVE_GAP_SCALE_M
        )
        earliest = min(conflict_times) if conflict_times else _TIME_SCALE_S
        mean_relative = float(np.mean(relative_speeds)) if relative_speeds else 0.0
        max_absolute_relative = (
            max(abs(value) for value in relative_speeds)
            if relative_speeds
            else 0.0
        )
        maximum_closing = max(closing_speeds) if closing_speeds else 0.0
        features[group_index] = np.asarray(
            [
                min(1.0, math.log1p(count) / _COUNT_SCALE),
                min(1.0, float(count) / float(capacity)),
                np.clip(min_ttc / _TIME_SCALE_S, 0.0, 1.0),
                float(bool(ttc_values)),
                np.clip(min_surface / _SURFACE_GAP_SCALE_M, 0.0, 1.0),
                np.clip(
                    min_effective / _EFFECTIVE_GAP_SCALE_M,
                    -1.0,
                    1.0,
                ),
                np.clip(earliest / _TIME_SCALE_S, 0.0, 1.0),
                float(bool(conflict_times)),
                np.clip(
                    mean_relative / _RELATIVE_SPEED_SCALE_MPS,
                    -1.0,
                    1.0,
                ),
                np.clip(
                    max_absolute_relative / _RELATIVE_SPEED_SCALE_MPS,
                    0.0,
                    1.0,
                ),
                np.clip(
                    maximum_closing / _RELATIVE_SPEED_SCALE_MPS,
                    0.0,
                    1.0,
                ),
                overflow,
            ],
            dtype=np.float32,
        )
    summary_mask = np.asarray([overflow], dtype=np.float32)
    tensor_hash = stable_hash(
        {
            "contract_version": OMITTED_ACTOR_SUMMARY_CONTRACT_VERSION,
            "features": features.tolist(),
            "group_mask": group_mask.tolist(),
            "summary_mask": summary_mask.tolist(),
        }
    )
    return OmittedActorSummary(
        features=features,
        group_mask=group_mask,
        summary_mask=summary_mask,
        omitted_actor_ids=omitted_ids,
        group_counts=tuple(counts),
        tensor_hash=tensor_hash,
    )


def validate_omitted_actor_summary_tensors(
    tensors: Mapping[str, Any],
    *,
    expected_hash: str | None = None,
) -> None:
    missing = [
        name for name in OMITTED_ACTOR_SUMMARY_TENSOR_FIELDS if name not in tensors
    ]
    if missing:
        raise ValueError(f"omitted-actor summary tensors are missing: {missing}")
    features = np.asarray(
        tensors["omitted_actor_summary_features"], dtype=np.float32
    )
    group_mask = np.asarray(
        tensors["omitted_actor_summary_group_mask"], dtype=np.float32
    )
    summary_mask = np.asarray(
        tensors["omitted_actor_summary_mask"], dtype=np.float32
    )
    if features.shape != (
        len(OMITTED_ACTOR_SUMMARY_GROUPS),
        len(OMITTED_ACTOR_SUMMARY_FEATURES),
    ):
        raise ValueError("omitted-actor summary feature shape mismatch")
    if group_mask.shape != (len(OMITTED_ACTOR_SUMMARY_GROUPS),):
        raise ValueError("omitted-actor summary group-mask shape mismatch")
    if summary_mask.shape != (1,):
        raise ValueError("omitted-actor summary mask shape mismatch")
    if not (
        np.isfinite(features).all()
        and np.isfinite(group_mask).all()
        and np.isfinite(summary_mask).all()
    ):
        raise ValueError("omitted-actor summary tensors must be finite")
    if np.any((group_mask != 0.0) & (group_mask != 1.0)) or np.any(
        (summary_mask != 0.0) & (summary_mask != 1.0)
    ):
        raise ValueError("omitted-actor summary masks must be binary")
    if bool(summary_mask[0]) != bool(np.any(group_mask > 0.0)):
        raise ValueError(
            "omitted-actor summary mask disagrees with its group mask"
        )
    if np.any(features[group_mask <= 0.0] != 0.0):
        raise ValueError(
            "masked omitted-actor summary groups must contain zero features"
        )
    computed = stable_hash(
        {
            "contract_version": OMITTED_ACTOR_SUMMARY_CONTRACT_VERSION,
            "features": features.tolist(),
            "group_mask": group_mask.tolist(),
            "summary_mask": summary_mask.tolist(),
        }
    )
    if expected_hash is not None and str(expected_hash) != computed:
        raise ValueError("omitted-actor summary tensor hash mismatch")


def _require_torch():
    try:
        import torch
        from torch import nn
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError("ACCVP omitted-actor adapter requires torch") from exc
    return torch, nn


try:
    _torch_base, _nn_base = _require_torch()
    _ModuleBase = _nn_base.Module
except ImportError:  # pragma: no cover
    _ModuleBase = object


class ACCVPOmittedActorSummaryAdapter(_ModuleBase):
    """Project fixed group statistics to one ACCVP-owned scene token."""

    def __init__(self, *, feature_dim: int, group_count: int, hidden_dim: int):
        _torch, nn = _require_torch()
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.group_count = int(group_count)
        self.hidden_dim = int(hidden_dim)
        if self.feature_dim <= 0 or self.group_count <= 0:
            raise ValueError("omitted-actor adapter dimensions must be positive")
        self.feature_projection = nn.Sequential(
            nn.Linear(self.feature_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
        )
        self.group_embedding = nn.Embedding(self.group_count, self.hidden_dim)
        self.group_score = nn.Linear(self.hidden_dim, 1)
        self.output_projection = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
        )

    def forward(self, features, group_mask, summary_mask):
        torch, _nn = _require_torch()
        if features.ndim != 3 or features.shape[1:] != (
            self.group_count,
            self.feature_dim,
        ):
            raise ValueError(
                "omitted-actor adapter features must be [batch,group,feature]"
            )
        if group_mask.shape != features.shape[:2]:
            raise ValueError("omitted-actor adapter group mask shape mismatch")
        if summary_mask.shape != (features.shape[0], 1):
            raise ValueError("omitted-actor adapter summary mask shape mismatch")
        if bool(
            (
                (summary_mask > 0.0).reshape(-1)
                != (group_mask > 0.0).any(dim=1)
            ).any()
        ):
            raise ValueError(
                "omitted-actor adapter summary mask disagrees with group mask"
            )
        group_ids = torch.arange(
            self.group_count, device=features.device, dtype=torch.long
        )
        tokens = self.feature_projection(features)
        tokens = tokens + self.group_embedding(group_ids)[None, :, :]
        valid = group_mask > 0.0
        safe_valid = valid.clone()
        empty = ~safe_valid.any(dim=1)
        if bool(empty.any()):
            safe_valid[empty, 0] = True
        scores = self.group_score(torch.tanh(tokens)).squeeze(-1)
        scores = scores.masked_fill(~safe_valid, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=1) * valid.to(scores.dtype)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled = (tokens * weights[:, :, None]).sum(dim=1)
        token = self.output_projection(pooled)
        return token[:, None, :] * summary_mask[:, :, None]
