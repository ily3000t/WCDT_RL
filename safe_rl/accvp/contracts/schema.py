"""Schema-v3 row validation, fingerprints and canonical serialization."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping


# Schema v3 binds every response row to the selected WcDT actor row and makes
# observation-equivalence fingerprints and conditional entry-time labels part
# of the immutable contract. Earlier schemas remain diagnostic history only.
COUNTERFACTUAL_SCHEMA_VERSION = 3
COUNTERFACTUAL_SHARD_MANIFEST_VERSION = 3
COUNTERFACTUAL_DATASET_MANIFEST_VERSION = 3
ACTOR_ROW_MAPPING_VERSION = "selected_indices_v2"
ROOT_OBSERVATION_FINGERPRINT_VERSION = "model_input_fingerprint_v3"
HYBRID_ROOT_OBSERVATION_FINGERPRINT_VERSION = (
    "model_input_fingerprint_v4_hybrid_actor_summary"
)
SCENARIO_EPISODE_KEY_VERSION = "scenario_route_traffic_seed_v1"
ENTRY_TIME_LABEL_VERSION = "conditional_entry_time_v1"
VIABILITY_STATUSES = frozenset({"observed_success", "observed_failure", "censored"})
BRANCH_REQUIRED_FIELDS = frozenset(
    {
        "counterfactual_schema_version",
        "root_id",
        "branch_id",
        "action_id",
        "snapshot_sha256",
        "candidate_plan_profile",
        "accvp_activation_distance_m",
        "data_contract_hash",
        "risk_model_fingerprint",
        "secondary_safety_pass",
        "event_observed",
        "censor_time",
        "censor_reason",
        "viability_observation_status",
        "actor_row_ids",
        "actor_row_source_indices",
        "actor_row_mapping_hash",
        "entry_time_observed",
        "entry_time_censor_time_s",
        "entry_time_censor_reason",
        "entry_time_label_version",
        "branch_status",
    }
)

ROOT_MODEL_INPUT_FIELDS = (
    "history_features",
    "history_valid_mask",
    "history_lane_ids",
    "history_edge_role_ids",
    "role_ids",
    "lane_ids",
    "edge_role_ids",
    "mask",
)

OPTIONAL_HYBRID_ROOT_MODEL_INPUT_FIELDS = (
    "omitted_actor_summary_features",
    "omitted_actor_summary_group_mask",
    "omitted_actor_summary_mask",
)

_ROOT_MODEL_INPUT_NDIMS = {
    "history_features": 3,
    "history_valid_mask": 2,
    "history_lane_ids": 2,
    "history_edge_role_ids": 2,
    "role_ids": 1,
    "lane_ids": 1,
    "edge_role_ids": 1,
    "mask": 1,
    "omitted_actor_summary_features": 2,
    "omitted_actor_summary_group_mask": 1,
    "omitted_actor_summary_mask": 1,
}


def _json_default(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"unsupported value in canonical JSON: {type(value)!r}")


def _json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if hasattr(value, "item") and not hasattr(value, "tolist"):
        return _json_safe(value.item())
    if hasattr(value, "tolist"):
        return _json_safe(value.tolist())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def canonical_json(value: Mapping[str, Any] | dict[str, Any]) -> str:
    return json.dumps(
        _json_safe(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=_json_default,
        allow_nan=False,
    )


def stable_hash(value: Mapping[str, Any] | dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def scenario_episode_key(
    *,
    scenario_route_hash: str,
    traffic_profile: str,
    episode_seed: int,
) -> str:
    """Return the policy-independent identity of one traffic realization.

    Root-policy provenance is deliberately excluded: different policies run
    against the same scenario/profile/seed remain one dependent experimental
    unit for dataset splitting and ensemble cluster bootstrap.
    """

    route_hash = "" if scenario_route_hash is None else str(scenario_route_hash).strip()
    profile = "" if traffic_profile is None else str(traffic_profile).strip()
    if not route_hash:
        raise ValueError("scenario_episode_key requires a non-empty scenario_route_hash")
    if not profile:
        raise ValueError("scenario_episode_key requires a non-empty traffic_profile")
    if isinstance(episode_seed, bool):
        raise ValueError("scenario_episode_key requires an integer episode_seed")
    try:
        seed = int(episode_seed)
    except (TypeError, ValueError) as exc:
        raise ValueError("scenario_episode_key requires an integer episode_seed") from exc
    if isinstance(episode_seed, float) and not episode_seed.is_integer():
        raise ValueError("scenario_episode_key requires an integer episode_seed")
    digest = stable_hash(
        {
            "version": SCENARIO_EPISODE_KEY_VERSION,
            "scenario_route_hash": route_hash,
            "traffic_profile": profile,
            "episode_seed": seed,
        }
    )
    return f"{SCENARIO_EPISODE_KEY_VERSION}:{digest}"


def actor_row_mapping_hash(
    actor_row_ids: list[str] | tuple[str, ...],
    selected_indices: list[int] | tuple[int, ...] | Any,
    actor_mask: list[float] | tuple[float, ...] | Any,
) -> str:
    """Hash the row-to-vehicle mapping independently of root provenance."""

    ids = [str(value) for value in actor_row_ids]
    indices = [int(value) for value in list(selected_indices)]
    mask = [float(value) for value in list(actor_mask)]
    if not (len(ids) == len(indices) == len(mask)):
        raise ValueError(
            "actor row mapping lengths differ: "
            f"ids={len(ids)}, selected_indices={len(indices)}, mask={len(mask)}"
        )
    valid_ids = [vehicle_id for vehicle_id, valid in zip(ids, mask) if valid > 0.0]
    valid_indices = [index for index, valid in zip(indices, mask) if valid > 0.0]
    if any(not value for value in valid_ids):
        raise ValueError("valid actor rows must have non-empty vehicle IDs")
    if len(valid_ids) != len(set(valid_ids)):
        raise ValueError("actor row mapping contains duplicate valid vehicle IDs")
    if len(valid_indices) != len(set(valid_indices)):
        raise ValueError("actor row mapping contains duplicate valid source indices")
    for vehicle_id, index, valid in zip(ids, indices, mask):
        if valid > 0.0 and index <= 0:
            raise ValueError("valid actor rows must reference a non-ego source index")
        if valid <= 0.0 and vehicle_id:
            raise ValueError("padded actor rows must use an empty vehicle ID")
        if valid <= 0.0 and index > 0:
            raise ValueError("padded actor rows must not reference a live source index")
    digest = stable_hash(
        {
            "version": ACTOR_ROW_MAPPING_VERSION,
            "actor_row_ids": ids,
            "selected_indices": indices,
            "actor_mask": [1 if value > 0.0 else 0 for value in mask],
        }
    )
    return f"{ACTOR_ROW_MAPPING_VERSION}:{digest}"


def root_observation_fingerprint(
    *,
    actor_row_ids: list[str] | tuple[str, ...] | None = None,
    root_ego: Mapping[str, Any],
    data_contract_hash: str,
    tensors: Mapping[str, Any],
    fingerprint_version: str = ROOT_OBSERVATION_FINGERPRINT_VERSION,
) -> str:
    """Hash exactly the immutable root inputs consumed by the ACCVP model.

    Root tensors are accepted either with or without their leading singleton
    batch dimension.  The ego plan seed is included because candidate plans
    are generated from ``x/y/heading/speed`` outside the scene encoder.
    Provenance such as actor identity, policy, seed, run ID and collection
    source is excluded. Actor identity remains bound separately by
    ``actor_row_mapping_hash``; it is deliberately not part of this hash
    because the predictor never consumes vehicle IDs.

    ``actor_row_ids`` is retained as a keyword-only compatibility argument for
    schema-v2 callers, but it has no effect on the model-input fingerprint.
    """

    missing = [name for name in ROOT_MODEL_INPUT_FIELDS if name not in tensors]
    if missing:
        raise ValueError(f"root observation fingerprint is missing model inputs: {missing}")
    present_hybrid = [
        name for name in OPTIONAL_HYBRID_ROOT_MODEL_INPUT_FIELDS if name in tensors
    ]
    if present_hybrid and len(present_hybrid) != len(
        OPTIONAL_HYBRID_ROOT_MODEL_INPUT_FIELDS
    ):
        missing_hybrid = [
            name
            for name in OPTIONAL_HYBRID_ROOT_MODEL_INPUT_FIELDS
            if name not in tensors
        ]
        raise ValueError(
            "root observation fingerprint has an incomplete hybrid actor "
            f"summary: missing={missing_hybrid}"
        )
    model_input_fields = (
        ROOT_MODEL_INPUT_FIELDS
        + OPTIONAL_HYBRID_ROOT_MODEL_INPUT_FIELDS
        if present_hybrid
        else ROOT_MODEL_INPUT_FIELDS
    )
    canonical_tensors: dict[str, Any] = {}
    for name in model_input_fields:
        value = tensors[name]
        ndim = int(getattr(value, "ndim", -1))
        expected_ndim = _ROOT_MODEL_INPUT_NDIMS[name]
        if ndim == expected_ndim + 1:
            if int(value.shape[0]) != 1:
                raise ValueError(f"root model input {name!r} has a non-singleton batch dimension")
            value = value[0]
            ndim -= 1
        if ndim != expected_ndim:
            raise ValueError(
                f"root model input {name!r} has ndim={ndim}; expected {expected_ndim}"
            )
        canonical_tensors[name] = value
    ego_plan_seed = {
        name: float(root_ego[name])
        for name in ("x", "y", "heading", "speed")
    }
    return stable_hash(
        {
            "version": str(fingerprint_version),
            "data_contract_hash": str(data_contract_hash),
            "ego_candidate_plan_seed": ego_plan_seed,
            "model_inputs": canonical_tensors,
        }
    )


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def jsonl_sha256(path: str | Path) -> str:
    """Return the byte hash of a manifest file with a clear missing-file error."""

    value = Path(path)
    if not value.exists():
        raise FileNotFoundError(f"required manifest does not exist: {value}")
    return file_sha256(value)


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def write_json_atomic(path: str | Path, value: Mapping[str, Any] | dict[str, Any]) -> Path:
    """Write a canonical JSON artifact without exposing partial files."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(canonical_json(value))
    temporary.replace(output)
    return output


def validate_branch_row(row: Mapping[str, Any]) -> None:
    missing = sorted(BRANCH_REQUIRED_FIELDS.difference(row))
    if missing:
        raise ValueError(f"ACCVP branch row missing required fields: {missing}")
    if int(row["counterfactual_schema_version"]) != COUNTERFACTUAL_SCHEMA_VERSION:
        raise ValueError(
            "unsupported counterfactual schema version "
            f"{row['counterfactual_schema_version']!r}; expected {COUNTERFACTUAL_SCHEMA_VERSION}"
        )
    status = str(row["viability_observation_status"])
    if status not in VIABILITY_STATUSES:
        raise ValueError(f"invalid viability_observation_status={status!r}")
    observed = bool(row["event_observed"])
    if observed != (status != "censored"):
        raise ValueError("event_observed must match viability_observation_status")
    if str(row["branch_status"]) != "completed":
        raise ValueError("only completed ACCVP branches are valid training rows")
    mapping_hash = str(row["actor_row_mapping_hash"])
    if not mapping_hash.startswith(f"{ACTOR_ROW_MAPPING_VERSION}:"):
        raise ValueError("invalid ACCVP actor_row_mapping_hash version")
    if str(row["entry_time_label_version"]) != ENTRY_TIME_LABEL_VERSION:
        raise ValueError("invalid ACCVP entry_time_label_version")
    actor_row_ids = [str(value) for value in row["actor_row_ids"]]
    source_indices = [int(value) for value in row["actor_row_source_indices"]]
    if len(actor_row_ids) != len(source_indices) or not actor_row_ids:
        raise ValueError("ACCVP branch actor row IDs and source indices must have equal non-zero length")
    valid_ids = [value for value in actor_row_ids if value]
    valid_indices = [index for value, index in zip(actor_row_ids, source_indices) if value]
    if len(valid_ids) != len(set(valid_ids)):
        raise ValueError("ACCVP branch actor row IDs must be unique")
    if len(valid_indices) != len(set(valid_indices)) or any(index <= 0 for index in valid_indices):
        raise ValueError("ACCVP branch valid actor rows require unique positive source indices")
    if any(index > 0 for value, index in zip(actor_row_ids, source_indices) if not value):
        raise ValueError("ACCVP branch padded actor rows must not reference live source indices")
    entry_observed = bool(row["entry_time_observed"])
    if entry_observed != (row.get("target_lane_entry_time_s") is not None):
        raise ValueError("entry_time_observed must match target_lane_entry_time_s presence")
    if entry_observed != (status == "observed_success"):
        raise ValueError("entry-time observation must match observed-success viability status")
    censor_time = float(row["entry_time_censor_time_s"])
    if not math.isfinite(censor_time) or censor_time < 0.0:
        raise ValueError("entry_time_censor_time_s must be finite and non-negative")
    reason = str(row["entry_time_censor_reason"])
    expected_reason = "" if entry_observed else ("taper_miss" if status == "observed_failure" else "horizon_elapsed")
    if reason != expected_reason:
        raise ValueError(
            f"entry_time_censor_reason={reason!r} does not match viability status {status!r}"
        )
