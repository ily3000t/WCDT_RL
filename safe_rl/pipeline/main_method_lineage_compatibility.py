from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from safe_rl.accvp.contracts.schema import file_sha256, read_json, stable_hash
from safe_rl.evaluation_protocol import EvidenceProtocolError
from safe_rl.pipeline.main_method_ppo_suite import METHOD_MANIFEST_KIND
from safe_rl.utils.config import REPO_ROOT


COMPATIBILITY_KIND = "main_method_stage5_parent_lineage_compatibility_v1"
COMPATIBILITY_VERSION = "audited_main_method_protocol_migration_v1"
ALLOWED_MISMATCH_FIELDS = ("protocol_id", "seed_ledger_sha256")


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (REPO_ROOT / value).resolve()


def _validate_fingerprint(payload: Mapping[str, Any], *, field: str, name: str) -> str:
    declared = str(payload.get(field, ""))
    content = {key: value for key, value in payload.items() if key != field}
    if not declared or stable_hash(content) != declared:
        raise EvidenceProtocolError(f"{name} fingerprint mismatch")
    return declared


def _manifest_record(
    manifest_path: Path,
    *,
    method_id: str,
    optimizer_seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = read_json(manifest_path)
    _validate_fingerprint(
        manifest,
        field="manifest_fingerprint",
        name="main-method PPO manifest",
    )
    if (
        manifest.get("artifact_kind") != METHOD_MANIFEST_KIND
        or manifest.get("status") != "complete"
        or str(manifest.get("method_id", "")) != method_id
    ):
        raise EvidenceProtocolError("invalid main-method PPO manifest binding")
    rows = [
        dict(row)
        for row in manifest.get("records", []) or []
        if int(row.get("optimizer_seed", row.get("training_seed", -1)))
        == int(optimizer_seed)
    ]
    if len(rows) != 1:
        raise EvidenceProtocolError("main-method PPO optimizer seed is ambiguous")
    return manifest, rows[0]


def _parent_lineage(report_path: Path) -> dict[str, Any]:
    report = read_json(report_path)
    lineage = dict(report.get("evidence_lineage", {}) or {})
    _validate_fingerprint(
        lineage,
        field="lineage_fingerprint",
        name="Stage3 evidence lineage",
    )
    return lineage


def build_compatibility(
    *,
    group_name: str,
    method_id: str,
    optimizer_seed: int,
    method_manifest: str | Path,
    row: Mapping[str, Any],
    target_lineage: Mapping[str, Any],
) -> dict[str, Any]:
    manifest_path = _resolve(method_manifest)
    manifest, frozen = _manifest_record(
        manifest_path,
        method_id=method_id,
        optimizer_seed=optimizer_seed,
    )
    if stable_hash(frozen) != stable_hash(dict(row)):
        raise EvidenceProtocolError("Stage5 row differs from the frozen main-method manifest")
    checkpoint = _resolve(str(frozen.get("checkpoint", "")))
    report_path = _resolve(str(frozen.get("stage3_report", "")))
    for path, digest, name in (
        (checkpoint, frozen.get("checkpoint_sha256"), "checkpoint"),
        (report_path, frozen.get("stage3_report_sha256"), "Stage3 report"),
    ):
        if not path.is_file() or file_sha256(path) != str(digest):
            raise EvidenceProtocolError(f"main-method {name} binding mismatch")
    parent = _parent_lineage(report_path)
    target = {
        key: str(target_lineage.get(key, ""))
        for key in ALLOWED_MISMATCH_FIELDS
    }
    if not bool(target_lineage.get("protocol_enabled", False)) or not bool(
        target_lineage.get("protocol_strict", False)
    ):
        raise EvidenceProtocolError("target main-method Stage5 lineage must be strict")
    mismatches = [
        key for key in ALLOWED_MISMATCH_FIELDS if str(parent.get(key, "")) != target[key]
    ]
    if not mismatches:
        raise EvidenceProtocolError("parent-lineage compatibility supplied without a mismatch")
    payload: dict[str, Any] = {
        "artifact_kind": COMPATIBILITY_KIND,
        "compatibility_version": COMPATIBILITY_VERSION,
        "method_id": method_id,
        "group_name": str(group_name),
        "optimizer_seed": int(optimizer_seed),
        "allowed_mismatch_fields": mismatches,
        "method_manifest": {
            "path": str(manifest_path),
            "sha256": file_sha256(manifest_path),
            "manifest_fingerprint": str(manifest["manifest_fingerprint"]),
            "record_fingerprint": stable_hash(frozen),
        },
        "checkpoint": {"path": str(checkpoint), "sha256": file_sha256(checkpoint)},
        "stage3_report": {"path": str(report_path), "sha256": file_sha256(report_path)},
        "parent_lineage": {
            "lineage_fingerprint": str(parent.get("lineage_fingerprint", "")),
            **{key: str(parent.get(key, "")) for key in ALLOWED_MISMATCH_FIELDS},
        },
        "target_lineage": target,
    }
    payload["compatibility_fingerprint"] = stable_hash(payload)
    return payload


def validate_compatibility(
    compatibility: Mapping[str, Any],
    *,
    group_name: str,
    model_path: Path,
    parent_lineage: Mapping[str, Any],
    stage5_lineage: Mapping[str, Any],
) -> dict[str, Any]:
    record = dict(compatibility)
    fingerprint = _validate_fingerprint(
        record,
        field="compatibility_fingerprint",
        name="main-method Stage5 parent compatibility",
    )
    if (
        record.get("artifact_kind") != COMPATIBILITY_KIND
        or record.get("compatibility_version") != COMPATIBILITY_VERSION
        or str(record.get("group_name", "")) != str(group_name)
    ):
        raise EvidenceProtocolError("unsupported main-method Stage5 compatibility")
    method_id = str(record.get("method_id", ""))
    seed = int(record.get("optimizer_seed", -1))
    manifest_binding = dict(record.get("method_manifest", {}) or {})
    manifest_path = _resolve(str(manifest_binding.get("path", "")))
    if (
        not manifest_path.is_file()
        or file_sha256(manifest_path) != str(manifest_binding.get("sha256", ""))
    ):
        raise EvidenceProtocolError("main-method compatibility manifest binding changed")
    manifest, row = _manifest_record(
        manifest_path,
        method_id=method_id,
        optimizer_seed=seed,
    )
    if (
        str(manifest.get("manifest_fingerprint", ""))
        != str(manifest_binding.get("manifest_fingerprint", ""))
        or stable_hash(row) != str(manifest_binding.get("record_fingerprint", ""))
    ):
        raise EvidenceProtocolError("main-method compatibility manifest record changed")
    expected_model = _resolve(str(row.get("checkpoint", "")))
    checkpoint_binding = dict(record.get("checkpoint", {}) or {})
    if (
        model_path.resolve() != expected_model
        or _resolve(str(checkpoint_binding.get("path", ""))) != expected_model
        or not expected_model.is_file()
        or file_sha256(expected_model) != str(row.get("checkpoint_sha256", ""))
        or str(checkpoint_binding.get("sha256", ""))
        != str(row.get("checkpoint_sha256", ""))
    ):
        raise EvidenceProtocolError("main-method compatibility checkpoint changed")
    expected_report = _resolve(str(row.get("stage3_report", "")))
    report_binding = dict(record.get("stage3_report", {}) or {})
    if (
        expected_report != model_path.parent / "stage3_training_report.json"
        or _resolve(str(report_binding.get("path", ""))) != expected_report
        or not expected_report.is_file()
        or file_sha256(expected_report) != str(row.get("stage3_report_sha256", ""))
        or str(report_binding.get("sha256", ""))
        != str(row.get("stage3_report_sha256", ""))
    ):
        raise EvidenceProtocolError("main-method compatibility Stage3 report changed")
    actual_parent = _parent_lineage(expected_report)
    if dict(actual_parent) != dict(parent_lineage):
        raise EvidenceProtocolError("main-method compatibility parent lineage changed")
    expected_parent = {
        "lineage_fingerprint": str(parent_lineage.get("lineage_fingerprint", "")),
        **{key: str(parent_lineage.get(key, "")) for key in ALLOWED_MISMATCH_FIELDS},
    }
    expected_target = {
        key: str(stage5_lineage.get(key, "")) for key in ALLOWED_MISMATCH_FIELDS
    }
    mismatches = [
        key
        for key in ALLOWED_MISMATCH_FIELDS
        if str(parent_lineage.get(key, "")) != expected_target[key]
    ]
    if (
        dict(record.get("parent_lineage", {}) or {}) != expected_parent
        or dict(record.get("target_lineage", {}) or {}) != expected_target
        or list(record.get("allowed_mismatch_fields", []) or []) != mismatches
    ):
        raise EvidenceProtocolError("main-method compatibility lineage endpoints changed")
    return {
        "applied": True,
        "compatibility_version": COMPATIBILITY_VERSION,
        "compatibility_fingerprint": fingerprint,
        "method_id": method_id,
        "parent_protocol_id": expected_parent["protocol_id"],
        "target_protocol_id": expected_target["protocol_id"],
        "allowed_mismatch_fields": mismatches,
    }
