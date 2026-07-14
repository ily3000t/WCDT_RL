"""Artifact naming, lifecycle validation and bundle resolution contracts."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Mapping

from safe_rl.accvp.contracts.runtime_contract import validate_manifest_runtime_contract
from safe_rl.accvp.contracts.schema import file_sha256, read_json, stable_hash


ACCVP_ARTIFACT_KIND = "accvp_artifact_bundle"
ACCVP_BUNDLE_SCHEMA_VERSION = 2
ACCVP_ARTIFACT_GENERATION = "vnext_schema3"
VNEXT_LITE_DECISION_WEIGHTING_VERSION = (
    "fingerprint_raw_action_total_weight_one_v2"
)
ACCVP_ARTIFACT_PREFIX = "accvp_vnext_schema3"

LIFECYCLE_SHADOW = "shadow"
LIFECYCLE_SEALED_CANDIDATE = "sealed_candidate"
LIFECYCLE_HOLDOUT_NO_GO = "holdout_evaluated_no_go"
LIFECYCLE_HOLDOUT_GO = "holdout_evaluated_go"
LIFECYCLE_REVOKED = "revoked"
FINAL_HOLDOUT_LIFECYCLE_AUTHORIZATION = "validated_by_final_holdout_v1"
VALID_LIFECYCLE_STATES = {
    LIFECYCLE_SHADOW,
    LIFECYCLE_SEALED_CANDIDATE,
    LIFECYCLE_HOLDOUT_NO_GO,
    LIFECYCLE_HOLDOUT_GO,
    LIFECYCLE_REVOKED,
}

V2_FILE_KEYS = ("predictor", "calibration", "operating_point", "training_history")
V2_TOP_LEVEL_HASH_KEYS = {
    "predictor": "predictor_sha256",
    "calibration": "calibration_sha256",
    "operating_point": "operating_point_sha256",
    "training_history": "training_history_sha256",
}


def artifact_filename(kind: str) -> str:
    names = {
        "predictor": f"{ACCVP_ARTIFACT_PREFIX}_predictor.pt",
        "calibration": f"{ACCVP_ARTIFACT_PREFIX}_calibration.json",
        "operating_point": f"{ACCVP_ARTIFACT_PREFIX}_operating_point.json",
        "lite_operating_point": f"{ACCVP_ARTIFACT_PREFIX}_lite_operating_point.json",
        "training_history": f"{ACCVP_ARTIFACT_PREFIX}_training_history.json",
        "candidate_manifest": f"{ACCVP_ARTIFACT_PREFIX}_candidate_manifest.json",
        "lite_candidate_manifest": f"{ACCVP_ARTIFACT_PREFIX}_lite_candidate_manifest.json",
        "shadow_manifest": f"{ACCVP_ARTIFACT_PREFIX}_shadow_manifest.json",
        "validated_manifest": f"{ACCVP_ARTIFACT_PREFIX}_validated_manifest.json",
        "final_test_diagnostics": f"{ACCVP_ARTIFACT_PREFIX}_final_test_diagnostics.json",
        "training_manifest": f"{ACCVP_ARTIFACT_PREFIX}_training_manifest.json",
        "tuning_failure": f"{ACCVP_ARTIFACT_PREFIX}_tuning_failure_diagnostics.json",
        "lite_tuning_summary": f"{ACCVP_ARTIFACT_PREFIX}_lite_tuning_summary.json",
    }
    try:
        return names[str(kind)]
    except KeyError as exc:
        raise ValueError(f"unknown ACCVP artifact filename kind={kind!r}") from exc


def bundle_file_entry(path: str | Path, *, manifest_dir: str | Path) -> dict[str, str]:
    value = Path(path).resolve()
    base = Path(manifest_dir).resolve()
    if not value.is_file():
        raise FileNotFoundError(value)
    try:
        recorded = os.path.relpath(value, base)
    except ValueError as exc:
        raise ValueError("ACCVP bundle files must be on the manifest filesystem") from exc
    return {"path": recorded, "sha256": file_sha256(value)}


def _resolve_entry(manifest_path: Path, entry: Mapping[str, Any], key: str) -> Path:
    recorded = str(entry.get("path", "")).strip()
    expected_hash = str(entry.get("sha256", "")).strip()
    if not recorded or not expected_hash:
        raise ValueError(f"ACCVP bundle file entry {key!r} requires path and sha256")
    path = Path(recorded)
    if not path.is_absolute():
        path = manifest_path.parent / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    actual_hash = file_sha256(path)
    if actual_hash != expected_hash:
        raise ValueError(f"ACCVP bundle file hash mismatch for {key}")
    return path


def _validate_manifest_fingerprint(manifest: Mapping[str, Any]) -> None:
    expected = str(manifest.get("artifact_fingerprint", "")).strip()
    if not expected:
        raise ValueError("ACCVP bundle-v2 manifest requires artifact_fingerprint")
    payload = dict(manifest)
    payload.pop("artifact_fingerprint", None)
    if stable_hash(payload) != expected:
        raise ValueError("ACCVP bundle-v2 manifest fingerprint mismatch")


def _validate_lifecycle_fields(manifest: Mapping[str, Any], lifecycle: str) -> None:
    deployable = bool(manifest.get("deployable_artifact", False))
    holdout_state = str(manifest.get("holdout_state", ""))
    holdout_decision = str(manifest.get("holdout_decision", ""))
    if lifecycle == LIFECYCLE_SEALED_CANDIDATE:
        if deployable or holdout_state != "sealed":
            raise ValueError("sealed_candidate bundle must be non-deployable with holdout_state='sealed'")
    elif lifecycle == LIFECYCLE_HOLDOUT_GO:
        if not deployable or holdout_state != "evaluated" or holdout_decision != "go":
            raise ValueError("holdout_evaluated_go bundle has inconsistent holdout fields")
    elif lifecycle == LIFECYCLE_HOLDOUT_NO_GO:
        if deployable or holdout_state != "evaluated" or holdout_decision != "no_go":
            raise ValueError("holdout_evaluated_no_go bundle has inconsistent holdout fields")
    elif lifecycle in {LIFECYCLE_SHADOW, LIFECYCLE_REVOKED} and deployable:
        raise ValueError(f"{lifecycle} bundle must not be deployable")
    if lifecycle in {LIFECYCLE_HOLDOUT_GO, LIFECYCLE_HOLDOUT_NO_GO}:
        if (
            str(manifest.get("lifecycle_authorization", ""))
            != FINAL_HOLDOUT_LIFECYCLE_AUTHORIZATION
        ):
            raise ValueError(
                "holdout-evaluated bundle requires "
                f"lifecycle_authorization={FINAL_HOLDOUT_LIFECYCLE_AUTHORIZATION!r}"
            )
        required_text = (
            "source_frozen_manifest",
            "final_test_diagnostics_path",
        )
        missing_text = [
            key for key in required_text if not str(manifest.get(key, "")).strip()
        ]
        if missing_text:
            raise ValueError(
                "holdout-evaluated bundle is missing final-holdout provenance: "
                f"{missing_text}"
            )
        required_hashes = (
            "source_frozen_manifest_sha256",
            "final_test_diagnostics_sha256",
            "holdout_claim_fingerprint",
        )
        invalid_hashes = [
            key
            for key in required_hashes
            if re.fullmatch(r"[0-9a-f]{64}", str(manifest.get(key, ""))) is None
        ]
        if invalid_hashes:
            raise ValueError(
                "holdout-evaluated bundle has invalid final-holdout hashes: "
                f"{invalid_hashes}"
            )
        promotion = manifest.get("promotion_evidence", {})
        if not isinstance(promotion, Mapping):
            raise ValueError("holdout-evaluated bundle promotion_evidence must be an object")
        for evidence_key in ("runtime_benchmark", "stage5_replicated_report"):
            evidence = promotion.get(evidence_key)
            if not isinstance(evidence, Mapping):
                raise ValueError(
                    "holdout-evaluated bundle is missing promotion evidence "
                    f"{evidence_key!r}"
                )
            if not str(evidence.get("path", "")).strip():
                raise ValueError(
                    f"holdout promotion evidence {evidence_key!r} requires path"
                )
            for hash_key in ("sha256", "report_fingerprint"):
                if re.fullmatch(r"[0-9a-f]{64}", str(evidence.get(hash_key, ""))) is None:
                    raise ValueError(
                        f"holdout promotion evidence {evidence_key!r} requires valid {hash_key}"
                    )


def _validate_artifact_variant(
    manifest: Mapping[str, Any],
    resolved: Mapping[str, Path],
) -> None:
    variant = str(manifest.get("artifact_variant", ""))
    if variant not in {"full_candidate_gate_v1", "viability_lite_task_v1"}:
        raise ValueError(f"unsupported ACCVP bundle artifact_variant={variant!r}")
    operating_path = resolved.get("operating_point")
    if operating_path is None:
        if variant == "viability_lite_task_v1":
            raise ValueError("VNext lite bundle requires a bound operating point")
        return
    operating_point = read_json(operating_path)
    if str(operating_point.get("split", "")) != "operating_point":
        raise ValueError("ACCVP bundle operating point must declare split='operating_point'")
    selected = dict(operating_point.get("selected", {}) or {})
    if variant == "full_candidate_gate_v1":
        required = {
            "proxy_collision_upper_bound",
            "safety_violation_upper_bound",
            "merge_viability_lower_bound",
        }
    else:
        required = {
            "min_p_merge_before_taper",
            "min_improvement_over_raw",
            "max_target_entry_time_s",
            "max_ensemble_disagreement",
        }
        if (
            str(manifest.get("operating_point_schema", ""))
            != "accvp_viability_lite_operating_point_v1"
        ):
            raise ValueError("VNext lite bundle operating_point_schema mismatch")
        manifest_weighting = dict(manifest.get("decision_weighting", {}) or {})
        operating_weighting = dict(
            operating_point.get("decision_weighting", {}) or {}
        )
        if (
            str(manifest_weighting.get("decision_weighting_version", ""))
            != VNEXT_LITE_DECISION_WEIGHTING_VERSION
            or manifest_weighting != operating_weighting
        ):
            raise ValueError("VNext lite bundle decision-weighting provenance mismatch")
        source_reference = dict(
            manifest.get("source_candidate_manifest_reference", {}) or {}
        )
        if (
            str(source_reference.get("reference_kind", "")) != "digest_only_v1"
            or str(source_reference.get("sha256", ""))
            != str(manifest.get("source_candidate_manifest_sha256", ""))
            or str(source_reference.get("artifact_fingerprint", ""))
            != str(manifest.get("source_candidate_fingerprint", ""))
            or str(source_reference.get("artifact_variant", ""))
            != "full_candidate_gate_v1"
        ):
            raise ValueError("VNext lite bundle source-candidate reference mismatch")
    missing = sorted(required.difference(selected))
    if missing:
        raise ValueError(
            f"ACCVP {variant} operating point is missing thresholds: {missing}"
        )


def resolve_v2_bundle(manifest_path: str | Path) -> tuple[dict[str, Any], dict[str, Path]]:
    path = Path(manifest_path).resolve()
    manifest = read_json(path)
    if str(manifest.get("artifact_kind", "")) != ACCVP_ARTIFACT_KIND:
        raise ValueError("not an ACCVP bundle-v2 manifest")
    if int(manifest.get("bundle_schema_version", -1)) != ACCVP_BUNDLE_SCHEMA_VERSION:
        raise ValueError("unsupported ACCVP bundle schema version")
    if str(manifest.get("artifact_generation", "")) != ACCVP_ARTIFACT_GENERATION:
        raise ValueError("ACCVP bundle artifact generation mismatch")
    lifecycle = str(manifest.get("lifecycle_state", ""))
    if lifecycle not in VALID_LIFECYCLE_STATES:
        raise ValueError(f"invalid ACCVP bundle lifecycle_state={lifecycle!r}")
    _validate_manifest_fingerprint(manifest)
    _validate_lifecycle_fields(manifest, lifecycle)
    validate_manifest_runtime_contract(manifest)
    files = dict(manifest.get("files", {}) or {})
    required = {"predictor", "calibration", "training_history"}
    if lifecycle in {
        LIFECYCLE_SEALED_CANDIDATE,
        LIFECYCLE_HOLDOUT_NO_GO,
        LIFECYCLE_HOLDOUT_GO,
    }:
        required.add("operating_point")
    if str(manifest.get("artifact_variant", "")) == "viability_lite_task_v1":
        required.add("operating_point")
    missing = sorted(key for key in required if key not in files)
    if missing:
        raise ValueError(f"ACCVP bundle is missing required files: {missing}")
    resolved = {
        key: _resolve_entry(path, dict(entry or {}), key)
        for key, entry in files.items()
        if key in V2_FILE_KEYS
    }
    for file_key in required:
        top_level_key = V2_TOP_LEVEL_HASH_KEYS[file_key]
        expected = str(manifest.get(top_level_key, "")).strip()
        if not expected:
            raise ValueError(f"ACCVP bundle-v2 manifest requires {top_level_key}")
        if file_sha256(resolved[file_key]) != expected:
            raise ValueError(f"ACCVP bundle top-level hash mismatch for {file_key}")
    _validate_artifact_variant(manifest, resolved)
    return manifest, resolved


def apply_v2_bundle_paths(config: Any) -> tuple[dict[str, Any] | None, dict[str, Path]]:
    manifest_path = config.accvp.get("artifact_manifest")
    if not manifest_path:
        return None, {}
    manifest_path = Path(str(manifest_path)).resolve()
    config.accvp["artifact_manifest"] = str(manifest_path)
    payload = read_json(manifest_path)
    configured_generation = str(config.accvp.get("artifact_generation") or "").strip()
    if str(payload.get("artifact_kind", "")) != ACCVP_ARTIFACT_KIND:
        if (
            "bundle_schema_version" in payload
            or str(payload.get("artifact_generation", "")) == ACCVP_ARTIFACT_GENERATION
        ):
            raise ValueError("ACCVP bundle-v2 manifest cannot downgrade to a legacy artifact kind")
        if configured_generation:
            raise ValueError(
                "generation-aware ACCVP config cannot load a legacy artifact manifest: "
                f"configured={configured_generation!r}"
            )
        return payload, {}
    manifest, resolved = resolve_v2_bundle(manifest_path)
    manifest_generation = str(manifest.get("artifact_generation", ""))
    if configured_generation and configured_generation != manifest_generation:
        raise ValueError(
            "ACCVP bundle/config artifact_generation mismatch: "
            f"config={configured_generation!r}, manifest={manifest_generation!r}"
        )
    config.accvp["artifact_generation"] = manifest_generation
    mapping = {
        "predictor": "checkpoint",
        "calibration": "calibration_bundle",
        "operating_point": "operating_point",
    }
    for source, destination in mapping.items():
        path = resolved.get(source)
        if path is not None:
            configured = config.accvp.get(destination)
            if configured and Path(str(configured)).resolve() != path:
                raise ValueError(
                    f"ACCVP bundle/config path mismatch for accvp.{destination}; "
                    "VNext bundle is the source of truth"
                )
            config.accvp[destination] = str(path)
    return manifest, resolved


def validate_lifecycle_for_mode(manifest: Mapping[str, Any], mode: str) -> None:
    lifecycle = str(manifest.get("lifecycle_state", ""))
    if lifecycle not in VALID_LIFECYCLE_STATES:
        raise ValueError(f"invalid ACCVP bundle lifecycle_state={lifecycle!r}")
    _validate_lifecycle_fields(manifest, lifecycle)
    capabilities = {str(value) for value in list(manifest.get("capabilities", []) or [])}
    runtime_mode = str(mode).strip().lower()
    artifact_variant = str(manifest.get("artifact_variant", ""))
    if lifecycle in {LIFECYCLE_REVOKED, LIFECYCLE_HOLDOUT_NO_GO}:
        raise ValueError(f"ACCVP bundle lifecycle {lifecycle!r} is not runnable")
    if runtime_mode in {"viability_branch", "viability_lite"}:
        if lifecycle != LIFECYCLE_HOLDOUT_GO:
            raise ValueError(f"ACCVP {runtime_mode} requires holdout_evaluated_go bundle")
        if not bool(manifest.get("deployable_artifact", False)):
            raise ValueError(f"ACCVP {runtime_mode} requires deployable_artifact=true")
        expected_variant = (
            "viability_lite_task_v1"
            if runtime_mode == "viability_lite"
            else "full_candidate_gate_v1"
        )
        if artifact_variant != expected_variant:
            raise ValueError(
                f"ACCVP {runtime_mode} requires artifact_variant={expected_variant!r}"
            )
        if runtime_mode == "viability_lite":
            if str(manifest.get("deployable_claim", "")) != "task_viability_only":
                raise ValueError(
                    "ACCVP viability_lite requires deployable_claim='task_viability_only'"
                )
            if bool(manifest.get("accvp_safety_head_hard_gate", True)):
                raise ValueError("ACCVP viability_lite must not use its safety head as authority")
    elif runtime_mode in {"shadow", "viability_lite_shadow"}:
        if lifecycle not in {LIFECYCLE_SHADOW, LIFECYCLE_SEALED_CANDIDATE, LIFECYCLE_HOLDOUT_GO}:
            raise ValueError(f"ACCVP shadow runtime does not allow lifecycle={lifecycle!r}")
        if runtime_mode == "viability_lite_shadow" and artifact_variant != "viability_lite_task_v1":
            raise ValueError(
                "ACCVP viability_lite_shadow requires artifact_variant='viability_lite_task_v1'"
            )
        if runtime_mode == "viability_lite_shadow":
            if str(manifest.get("deployable_claim", "")) != "task_viability_only":
                raise ValueError(
                    "ACCVP viability_lite_shadow requires deployable_claim='task_viability_only'"
                )
            if bool(manifest.get("accvp_safety_head_hard_gate", True)):
                raise ValueError(
                    "ACCVP viability_lite_shadow must not use its safety head as authority"
                )
    elif runtime_mode != "off":
        raise ValueError(f"unsupported ACCVP runtime mode={runtime_mode!r}")
    if "candidate_table_observation" not in capabilities:
        raise ValueError("ACCVP bundle lacks candidate_table_observation capability")
