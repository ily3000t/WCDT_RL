from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any, Mapping

import yaml

from safe_rl.accvp.contracts.artifacts import resolve_v2_bundle
from safe_rl.accvp.contracts.runtime_contract import (
    candidate_table_semantic_contract,
    closed_loop_execution_contract,
    formal_runtime_contract_sha256,
)
from safe_rl.accvp.serving.observation import RiskGatedACCVPCandidateTableAugmentor
from safe_rl.evaluation_protocol import file_sha256, stable_hash
from safe_rl.utils.config import REPO_ROOT


REPLICATE_MANIFEST_KIND = "ppo_optimizer_replicate_manifest_v1"
REPLICATE_MANIFEST_SCHEMA_VERSION = 1
MIN_FORMAL_OPTIMIZER_REPLICATES = 5

_REWARD_TOKEN = {
    "opportunity_window_v2": "reward_v2_mainline_001",
    "opportunity_window_v3_persistence": "reward_v3_persistence_001",
    "opportunity_window_v3_1_risk_gated_persistence": "reward_v3_1_persistence_001",
}


def plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    return value


def optimizer_seed(config: Any) -> int:
    value = int(config.get("rl", {}).get("optimizer_seed", config.get("run", {}).get("seed", 0)))
    if value < 0:
        raise ValueError("rl.optimizer_seed must be non-negative")
    return value


def _configured_file(path: Any) -> Path | None:
    if not path:
        return None
    value = Path(str(path))
    return value.resolve() if value.is_absolute() else (REPO_ROOT / value).resolve()


def validate_reward_semantics(config: Any) -> dict[str, Any]:
    rl = config.get("rl", {}) or {}
    profile = str(rl.get("reward_profile", "default"))
    merge = plain(rl.get("merge_timing_reward", {}) or {})
    reward_version = str(merge.get("reward_version", "")) if profile == "merge_timing_forecast" else ""
    declared = str(rl.get("training_semantics_version", "legacy_unspecified"))
    tokens = [token for token in declared.split("+") if token]
    expected_reward_token = _REWARD_TOKEN.get(reward_version)
    if profile == "merge_timing_forecast" and expected_reward_token is None:
        raise ValueError(f"unsupported merge_timing_reward.reward_version={reward_version!r}")
    reward_tokens = [token for token in tokens if re.fullmatch(r"reward_v[^+]+", token)]
    if expected_reward_token and expected_reward_token not in tokens:
        raise ValueError(
            "training_semantics_version disagrees with executed reward_version: "
            f"expected token {expected_reward_token!r} for {reward_version!r}"
        )
    if expected_reward_token and any(token != expected_reward_token for token in reward_tokens):
        raise ValueError(
            "training_semantics_version contains a conflicting reward token: "
            f"declared={reward_tokens!r} expected={expected_reward_token!r}"
        )

    commitment = plain(rl.get("policy_lateral_commitment", {}) or {})
    commitment_enabled = bool(commitment.get("enabled", False))
    commitment_profile = str(commitment.get("profile", "left_intent_1s_risk_gated_v2"))
    commitment_token = f"policy_lateral_commitment_{commitment_profile}"
    if commitment_enabled and commitment_token not in tokens:
        raise ValueError(
            "enabled policy commitment is missing from training_semantics_version: "
            f"{commitment_token!r}"
        )
    if not commitment_enabled and any(token.startswith("policy_lateral_commitment_") for token in tokens):
        raise ValueError("training_semantics_version declares policy commitment while it is disabled")

    observation = plain(config.get("accvp", {}).get("observation", {}) or {})
    accvp_enabled = bool(observation.get("enabled", False))
    feature_version = str(
        observation.get(
            "feature_version",
            RiskGatedACCVPCandidateTableAugmentor.FEATURE_VERSION,
        )
    )
    if accvp_enabled and feature_version not in tokens:
        raise ValueError(
            "ACCVP observation feature version is missing from training_semantics_version: "
            f"{feature_version!r}"
        )
    if not accvp_enabled and feature_version in tokens:
        raise ValueError("training_semantics_version declares ACCVP features while they are disabled")

    shield_guided = plain(rl.get("shield_guided_reward", {}) or {})
    reward_risk = _configured_file(shield_guided.get("risk_checkpoint"))
    if reward_risk is not None and reward_risk.is_file():
        shield_guided["risk_checkpoint_sha256"] = file_sha256(reward_risk)
    payload = {
        "reward_profile": profile,
        "reward_version": reward_version,
        "merge_timing_reward": merge if profile == "merge_timing_forecast" else None,
        "policy_lateral_commitment": commitment,
        "shield_guided_reward": shield_guided,
        "training_semantics_version": declared,
        "accvp_observation_enabled": accvp_enabled,
        "accvp_observation_feature_version": feature_version if accvp_enabled else "",
    }
    return {"payload": payload, "sha256": stable_hash(payload)}


def observation_contract(config: Any, *, require_artifacts: bool = False) -> dict[str, Any]:
    forecast = plain(config.get("forecast_features", {}) or {})
    rl = config.get("rl", {}) or {}
    accvp = plain(config.get("accvp", {}) or {})
    accvp_enabled = bool(accvp.get("observation", {}).get("enabled", False))
    payload: dict[str, Any] = {
        "forecast_features_enabled": bool(forecast.get("enabled", False)),
        "forecast_source": str(forecast.get("source", "")),
        "forecast_checkpoint": str(forecast.get("checkpoint", "")),
        "use_wcdt_forecast_features": bool(rl.get("use_wcdt_forecast_features", False)),
        "accvp_observation_enabled": accvp_enabled,
        "accvp_observation": plain(accvp.get("observation", {}) or {}),
        "vehicle_state_ordering_version": str(
            config.get("scenario", {}).get("vehicle_state_ordering_version", "")
        ),
    }
    if accvp_enabled:
        execution = closed_loop_execution_contract(
            plain(accvp.get("observation", {}) or {})
        )
        payload["closed_loop_execution_contract"] = execution
        payload["closed_loop_execution_contract_sha256"] = stable_hash(execution)
    forecast_checkpoint = _configured_file(forecast.get("checkpoint"))
    if forecast_checkpoint is not None:
        if require_artifacts and not forecast_checkpoint.is_file():
            raise FileNotFoundError(forecast_checkpoint)
        if forecast_checkpoint.is_file():
            payload["forecast_checkpoint_sha256"] = file_sha256(forecast_checkpoint)
    artifact_fingerprint = ""
    artifact_manifest = str(accvp.get("artifact_manifest", "")) if accvp_enabled else ""
    if artifact_manifest:
        path = Path(artifact_manifest)
        if require_artifacts or path.exists():
            if not path.is_file():
                raise FileNotFoundError(path)
            bundle, _resolved = resolve_v2_bundle(path)
            artifact_fingerprint = str(bundle.get("artifact_fingerprint", ""))
            payload["accvp_artifact_manifest"] = str(path.resolve())
            payload["accvp_artifact_manifest_sha256"] = file_sha256(path)
            payload["accvp_artifact_fingerprint"] = artifact_fingerprint
            payload["accvp_artifact_variant"] = str(bundle.get("artifact_variant", ""))
            deployment_contract = dict(bundle.get("formal_runtime_contract", {}) or {})
            deployment_contract_sha = formal_runtime_contract_sha256(
                deployment_contract
            )
            if deployment_contract_sha != str(
                bundle.get("formal_runtime_contract_sha256", "")
            ):
                raise ValueError("ACCVP bundle deployment runtime contract hash mismatch")
            semantic_contract = candidate_table_semantic_contract(deployment_contract)
            payload["candidate_table_semantic_contract"] = semantic_contract
            payload["candidate_table_semantic_contract_sha256"] = stable_hash(
                semantic_contract
            )
            payload["deployment_runtime_contract_sha256"] = deployment_contract_sha
            # Backward-compatible alias retained for existing bundle consumers.
            payload["formal_runtime_contract_sha256"] = deployment_contract_sha
    elif accvp_enabled and require_artifacts:
        raise ValueError("ACCVP replicate requires accvp.artifact_manifest")
    return {
        "payload": payload,
        "sha256": stable_hash(payload),
        "accvp_artifact_fingerprint": artifact_fingerprint,
    }


def write_yaml_atomic(path: str | Path, payload: Mapping[str, Any]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(plain(payload), handle, sort_keys=False, allow_unicode=True)
    temporary.replace(output)
    return output


def write_json_new(path: str | Path, payload: Mapping[str, Any]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(plain(payload), handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
    temporary.replace(output)
    return output


def apply_variant(config: Mapping[str, Any], variant: Mapping[str, Any]) -> dict[str, Any]:
    resolved = copy.deepcopy(plain(config))
    rl = resolved.setdefault("rl", {})
    rl.setdefault("merge_timing_reward", {})["reward_version"] = str(variant["reward_version"])
    rl.setdefault("policy_lateral_commitment", {})["enabled"] = bool(
        variant.get("policy_lateral_commitment", False)
    )
    rl["training_semantics_version"] = str(variant["training_semantics_version"])
    expected_candidate = bool(variant.get("candidate_table", False))
    actual_candidate = bool(resolved.get("accvp", {}).get("observation", {}).get("enabled", False))
    if actual_candidate != expected_candidate:
        raise ValueError(
            "method matrix candidate_table flag disagrees with template: "
            f"matrix={expected_candidate} template={actual_candidate}"
        )
    expected_source = variant.get("forecast_source")
    actual_forecast = bool(resolved.get("forecast_features", {}).get("enabled", False))
    if bool(expected_source) != actual_forecast:
        raise ValueError(
            "method matrix forecast_source disagrees with template: "
            f"matrix={expected_source!r} enabled={actual_forecast}"
        )
    if expected_source and str(resolved["forecast_features"].get("source", "")) != str(expected_source):
        raise ValueError("method matrix forecast_source disagrees with template source")
    return resolved
