from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from safe_rl.accvp.contracts.schema import file_sha256, read_json
from safe_rl.evaluation_protocol import EvidenceProtocolError, stable_hash
from safe_rl.pipeline.audit_ppo_replicate_lineage import (
    LINEAGE_AUDIT_IMPLEMENTATION_VERSION,
    LINEAGE_AUDIT_KIND,
)
from safe_rl.ppo_replicates import REPLICATE_MANIFEST_KIND
from safe_rl.utils.config import REPO_ROOT


PARENT_LINEAGE_COMPATIBILITY_KIND = "stage5_parent_lineage_compatibility_v1"
PARENT_LINEAGE_COMPATIBILITY_VERSION = "audited_wcdt_protocol_migration_v1"
LEGACY_WCDT_PROTOCOL_ID = "accvp-vnext-correctness-v1"
ALLOWED_PARENT_MISMATCH_FIELDS = ("protocol_id", "seed_ledger_sha256")


def _resolve(path: str | Path, *, relative_to: Path | None = None) -> Path:
    value = Path(path)
    if value.is_absolute():
        return value.resolve()
    return ((relative_to or REPO_ROOT) / value).resolve()


def _validate_fingerprint(payload: Mapping[str, Any], *, field: str, name: str) -> str:
    declared = str(payload.get(field, ""))
    content = {key: value for key, value in payload.items() if key != field}
    if not declared or stable_hash(content) != declared:
        raise EvidenceProtocolError(f"{name} fingerprint mismatch")
    return declared


def validate_baseline_lineage_audit(
    audit_path: str | Path,
    *,
    baseline_manifest: str | Path,
    required_seeds: list[int],
) -> dict[str, Any]:
    """Validate and bind the independent legacy-baseline reuse audit."""

    audit_source = _resolve(audit_path)
    manifest_source = _resolve(baseline_manifest)
    if not audit_source.is_file():
        raise FileNotFoundError(audit_source)
    if not manifest_source.is_file():
        raise FileNotFoundError(manifest_source)
    audit = read_json(audit_source)
    fingerprint = _validate_fingerprint(
        audit,
        field="audit_fingerprint",
        name="baseline lineage audit",
    )
    expected_seeds = sorted(int(seed) for seed in required_seeds)
    checks = {
        "artifact_kind": str(audit.get("artifact_kind", "")) == LINEAGE_AUDIT_KIND,
        "implementation_version": str(audit.get("audit_implementation_version", ""))
        == LINEAGE_AUDIT_IMPLEMENTATION_VERSION,
        "status": str(audit.get("status", "")) == "reusable",
        "method_id": str(audit.get("method_id", "")) == "wcdt_reward_v2",
        "manifest_path": _resolve(str(audit.get("manifest", ""))) == manifest_source,
        "manifest_sha256": str(audit.get("manifest_sha256", ""))
        == file_sha256(manifest_source),
        "required_seeds": [int(seed) for seed in audit.get("required_seeds", [])]
        == expected_seeds,
        "valid_seeds": [int(seed) for seed in audit.get("valid_seeds", [])]
        == expected_seeds,
        "missing_seeds": not list(audit.get("missing_seeds", []) or []),
        "invalid_records": not list(audit.get("invalid_records", []) or []),
        "global_reasons": not list(audit.get("global_reasons", []) or []),
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise EvidenceProtocolError(
            f"baseline lineage audit cannot authorize Stage5 migration: failed={failed}"
        )
    return {
        "path": str(audit_source),
        "sha256": file_sha256(audit_source),
        "audit_fingerprint": fingerprint,
        "audit_implementation_version": LINEAGE_AUDIT_IMPLEMENTATION_VERSION,
        "baseline_manifest": str(manifest_source),
        "baseline_manifest_sha256": file_sha256(manifest_source),
        "required_seeds": expected_seeds,
    }


def _lineage_from_stage3_report(report_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    report = read_json(report_path)
    parent = dict(report.get("evidence_lineage", {}) or {})
    _validate_fingerprint(
        parent,
        field="lineage_fingerprint",
        name="Stage3 evidence lineage",
    )
    return report, parent


def build_wcdt_parent_lineage_compatibility(
    *,
    group_name: str,
    optimizer_seed: int,
    row: Mapping[str, Any],
    audit_binding: Mapping[str, Any],
    target_protocol: Mapping[str, Any],
) -> dict[str, Any]:
    """Create a hash-bound, per-checkpoint proof for the legacy WcDT baseline."""

    if str(row.get("method_id", "")) != "wcdt_reward_v2":
        raise EvidenceProtocolError("parent-lineage migration is restricted to wcdt_reward_v2")
    seed = int(optimizer_seed)
    if int(row.get("optimizer_seed", row.get("training_seed", -1))) != seed:
        raise EvidenceProtocolError("baseline optimizer seed does not match the Stage5 group")
    if seed not in [int(value) for value in audit_binding.get("required_seeds", [])]:
        raise EvidenceProtocolError("baseline optimizer seed is not authorized by the lineage audit")

    checkpoint = _resolve(str(row.get("checkpoint", "")))
    report_path = _resolve(str(row.get("stage3_report", "")))
    for path, expected, name in (
        (checkpoint, str(row.get("checkpoint_sha256", "")), "checkpoint"),
        (report_path, str(row.get("stage3_report_sha256", "")), "Stage3 report"),
    ):
        if not path.is_file() or file_sha256(path) != expected:
            raise EvidenceProtocolError(f"legacy baseline {name} binding mismatch")
    _report, parent = _lineage_from_stage3_report(report_path)
    if not bool(parent.get("protocol_enabled", False)) or not bool(
        parent.get("protocol_strict", False)
    ):
        raise EvidenceProtocolError("legacy WcDT parent lineage was not strict and enabled")
    if str(parent.get("protocol_id", "")) != LEGACY_WCDT_PROTOCOL_ID:
        raise EvidenceProtocolError(
            "parent-lineage migration is restricted to the registered WcDT v1 protocol"
        )
    if not bool(target_protocol.get("enabled", False)) or not bool(
        target_protocol.get("strict", False)
    ):
        raise EvidenceProtocolError("target Stage5 protocol must be strict and enabled")
    target = {
        "protocol_id": str(target_protocol.get("protocol_id", "")),
        "seed_ledger_sha256": str(target_protocol.get("seed_ledger_sha256", "")),
    }
    if not all(target.values()):
        raise EvidenceProtocolError("target Stage5 protocol binding is incomplete")
    mismatches = sorted(
        key
        for key in ALLOWED_PARENT_MISMATCH_FIELDS
        if parent.get(key) != target[key]
    )
    if mismatches != sorted(ALLOWED_PARENT_MISMATCH_FIELDS):
        raise EvidenceProtocolError(
            "legacy WcDT migration must bridge exactly protocol_id and seed_ledger_sha256"
        )

    payload = {
        "artifact_kind": PARENT_LINEAGE_COMPATIBILITY_KIND,
        "compatibility_version": PARENT_LINEAGE_COMPATIBILITY_VERSION,
        "method_id": "wcdt_reward_v2",
        "group_name": str(group_name),
        "optimizer_seed": seed,
        "allowed_mismatch_fields": list(ALLOWED_PARENT_MISMATCH_FIELDS),
        "lineage_audit": dict(audit_binding),
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": file_sha256(checkpoint),
        },
        "stage3_report": {
            "path": str(report_path),
            "sha256": file_sha256(report_path),
        },
        "parent_lineage": {
            "lineage_fingerprint": str(parent.get("lineage_fingerprint", "")),
            "protocol_id": str(parent.get("protocol_id", "")),
            "seed_ledger_sha256": str(parent.get("seed_ledger_sha256", "")),
        },
        "target_lineage": target,
    }
    payload["compatibility_fingerprint"] = stable_hash(payload)
    return payload


def validate_wcdt_parent_lineage_compatibility(
    compatibility: Mapping[str, Any],
    *,
    group_name: str,
    model_path: Path,
    parent_lineage: Mapping[str, Any],
    stage5_lineage: Mapping[str, Any],
) -> dict[str, Any]:
    """Revalidate a migration proof at execution time before any episode runs."""

    record = dict(compatibility)
    fingerprint = _validate_fingerprint(
        record,
        field="compatibility_fingerprint",
        name="Stage5 parent-lineage compatibility",
    )
    if str(record.get("artifact_kind", "")) != PARENT_LINEAGE_COMPATIBILITY_KIND:
        raise EvidenceProtocolError("unsupported Stage5 parent-lineage compatibility artifact")
    if str(record.get("compatibility_version", "")) != PARENT_LINEAGE_COMPATIBILITY_VERSION:
        raise EvidenceProtocolError("unsupported Stage5 parent-lineage compatibility version")
    if str(record.get("method_id", "")) != "wcdt_reward_v2":
        raise EvidenceProtocolError("Stage5 parent-lineage migration is not a WcDT baseline")
    if str(record.get("group_name", "")) != str(group_name):
        raise EvidenceProtocolError("Stage5 parent-lineage migration group mismatch")
    if list(record.get("allowed_mismatch_fields", []) or []) != list(
        ALLOWED_PARENT_MISMATCH_FIELDS
    ):
        raise EvidenceProtocolError("Stage5 parent-lineage migration scope changed")

    audit_record = dict(record.get("lineage_audit", {}) or {})
    audit_binding = validate_baseline_lineage_audit(
        str(audit_record.get("path", "")),
        baseline_manifest=str(audit_record.get("baseline_manifest", "")),
        required_seeds=[int(seed) for seed in audit_record.get("required_seeds", [])],
    )
    if audit_binding != audit_record:
        raise EvidenceProtocolError("Stage5 lineage-audit binding changed after generation")

    manifest = read_json(Path(audit_binding["baseline_manifest"]))
    if str(manifest.get("artifact_kind", "")) != REPLICATE_MANIFEST_KIND or str(
        manifest.get("method_id", "")
    ) != "wcdt_reward_v2":
        raise EvidenceProtocolError("Stage5 migration baseline manifest is invalid")
    seed = int(record.get("optimizer_seed", -1))
    rows = [
        dict(row)
        for row in manifest.get("records", []) or []
        if int(row.get("optimizer_seed", row.get("training_seed", -1))) == seed
    ]
    if len(rows) != 1:
        raise EvidenceProtocolError("Stage5 migration baseline manifest seed is ambiguous")
    row = rows[0]
    expected_model = _resolve(str(row.get("checkpoint", "")))
    model_binding = dict(record.get("checkpoint", {}) or {})
    if (
        model_path.resolve() != expected_model
        or _resolve(str(model_binding.get("path", ""))) != expected_model
        or str(model_binding.get("sha256", "")) != str(row.get("checkpoint_sha256", ""))
        or not expected_model.is_file()
        or file_sha256(expected_model) != str(row.get("checkpoint_sha256", ""))
    ):
        raise EvidenceProtocolError("Stage5 migration checkpoint binding mismatch")

    expected_report = _resolve(str(row.get("stage3_report", "")))
    report_binding = dict(record.get("stage3_report", {}) or {})
    if (
        expected_report != model_path.parent / "stage3_training_report.json"
        or _resolve(str(report_binding.get("path", ""))) != expected_report
        or str(report_binding.get("sha256", "")) != str(row.get("stage3_report_sha256", ""))
        or not expected_report.is_file()
        or file_sha256(expected_report) != str(row.get("stage3_report_sha256", ""))
    ):
        raise EvidenceProtocolError("Stage5 migration Stage3-report binding mismatch")

    _report, actual_parent = _lineage_from_stage3_report(expected_report)
    if dict(actual_parent) != dict(parent_lineage):
        raise EvidenceProtocolError("Stage5 migration parent lineage changed")
    parent_binding = dict(record.get("parent_lineage", {}) or {})
    expected_parent_binding = {
        "lineage_fingerprint": str(parent_lineage.get("lineage_fingerprint", "")),
        "protocol_id": str(parent_lineage.get("protocol_id", "")),
        "seed_ledger_sha256": str(parent_lineage.get("seed_ledger_sha256", "")),
    }
    target_binding = dict(record.get("target_lineage", {}) or {})
    expected_target_binding = {
        "protocol_id": str(stage5_lineage.get("protocol_id", "")),
        "seed_ledger_sha256": str(stage5_lineage.get("seed_ledger_sha256", "")),
    }
    if parent_binding != expected_parent_binding or target_binding != expected_target_binding:
        raise EvidenceProtocolError("Stage5 migration lineage endpoints changed")
    mismatches = sorted(
        key
        for key in ALLOWED_PARENT_MISMATCH_FIELDS
        if parent_lineage.get(key) != stage5_lineage.get(key)
    )
    if mismatches != sorted(ALLOWED_PARENT_MISMATCH_FIELDS):
        raise EvidenceProtocolError("Stage5 migration mismatch scope is not exact")
    return {
        "applied": True,
        "compatibility_version": PARENT_LINEAGE_COMPATIBILITY_VERSION,
        "compatibility_fingerprint": fingerprint,
        "lineage_audit_path": audit_binding["path"],
        "lineage_audit_sha256": audit_binding["sha256"],
        "parent_protocol_id": expected_parent_binding["protocol_id"],
        "target_protocol_id": expected_target_binding["protocol_id"],
        "allowed_mismatch_fields": list(ALLOWED_PARENT_MISMATCH_FIELDS),
    }
