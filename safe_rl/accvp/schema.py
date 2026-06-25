from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping


# Schema v2 makes the ACV-Shield activation protocol part of the immutable
# data contract.  Schema-v1 artifacts remain diagnostic history and cannot be
# mixed into a formal v2 training dataset.
COUNTERFACTUAL_SCHEMA_VERSION = 2
COUNTERFACTUAL_SHARD_MANIFEST_VERSION = 2
COUNTERFACTUAL_DATASET_MANIFEST_VERSION = 2
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
        "branch_status",
    }
)


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
