from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping


COUNTERFACTUAL_SCHEMA_VERSION = 1
VIABILITY_STATUSES = frozenset({"observed_success", "observed_failure", "censored"})
BRANCH_REQUIRED_FIELDS = frozenset(
    {
        "counterfactual_schema_version",
        "root_id",
        "branch_id",
        "action_id",
        "snapshot_sha256",
        "candidate_plan_profile",
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
