from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from safe_rl.accvp.contracts.schema import (
    file_sha256,
    read_json,
    write_json_atomic,
)
from safe_rl.evaluation_protocol import stable_hash
from safe_rl.ppo_replicates import (
    REPLICATE_MANIFEST_KIND,
    observation_contract,
    validate_reward_semantics,
)
from safe_rl.utils.config import REPO_ROOT, load_config


LINEAGE_AUDIT_KIND = "ppo_replicate_lineage_audit_v1"
LINEAGE_AUDIT_IMPLEMENTATION_VERSION = (
    "effective_observation_contract_from_frozen_stage3_v2"
)
INACTIVE_ACCVP_COMPATIBILITY_VERSION = "inactive_accvp_defaults_ignored_v1"


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (REPO_ROOT / value).resolve()


def _effective_observation_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return only observation semantics that can affect the policy input.

    Historical WcDT-only runs recorded the complete inherited ACCVP observation
    block even though ACCVP was disabled.  Adding a new default inside that
    inactive block changed the raw hash without changing the 63D policy input.
    Active ACCVP fields remain strict and are never removed here.
    """

    effective = dict(payload)
    if not bool(effective.get("accvp_observation_enabled", False)):
        effective.pop("accvp_observation", None)
        for key in tuple(effective):
            if key.startswith("accvp_artifact_") or key == "formal_runtime_contract_sha256":
                effective.pop(key, None)
    return effective


def _frozen_observation_payload(report: Mapping[str, Any]) -> dict[str, Any] | None:
    value = report.get("observation_contract")
    if not isinstance(value, Mapping):
        return None
    if isinstance(value.get("payload"), Mapping):
        return dict(value["payload"])
    return dict(value)


def _observation_compatibility(
    row: Mapping[str, Any],
    *,
    expected_contract: Mapping[str, Any],
) -> tuple[bool, dict[str, Any] | None, str | None]:
    declared = str(row.get("observation_contract_hash", ""))
    expected = str(expected_contract.get("sha256", ""))
    if declared == expected:
        return True, None, None

    report_value = str(row.get("stage3_report", ""))
    if not report_value:
        return False, None, "observation contract differs from method config"
    report_path = _resolve(report_value)
    if not report_path.is_file():
        return False, None, "observation contract differs from method config"
    try:
        report = read_json(report_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False, None, "stage3 report cannot prove the observation contract"
    frozen_hash = str(report.get("observation_contract_hash", ""))
    frozen_payload = _frozen_observation_payload(report)
    if frozen_hash != declared or frozen_payload is None:
        return False, None, "stage3 report does not prove the declared observation contract"
    if stable_hash(frozen_payload) != declared:
        return False, None, "stage3 observation payload SHA-256 mismatch"

    expected_payload = expected_contract.get("payload")
    if not isinstance(expected_payload, Mapping):
        return False, None, "method config observation contract lacks a payload"
    frozen_effective = _effective_observation_payload(frozen_payload)
    expected_effective = _effective_observation_payload(expected_payload)
    if frozen_effective != expected_effective:
        return False, None, "effective observation contract differs from method config"
    migration = {
        "optimizer_seed": int(row.get("optimizer_seed", row.get("training_seed", -1))),
        "compatibility_version": INACTIVE_ACCVP_COMPATIBILITY_VERSION,
        "declared_observation_contract_hash": declared,
        "current_observation_contract_hash": expected,
        "effective_observation_contract_hash": stable_hash(frozen_effective),
    }
    return True, migration, None


def write_audit_report(path: str | Path, report: Mapping[str, Any]) -> tuple[Path, str]:
    """Write an audit idempotently while preserving failed predecessors."""

    output = _resolve(path)
    core = dict(report)
    if output.exists():
        existing = read_json(output)
        declared_fingerprint = str(existing.get("audit_fingerprint", ""))
        fingerprint_payload = {
            key: value for key, value in existing.items() if key != "audit_fingerprint"
        }
        fingerprint_valid = bool(declared_fingerprint) and (
            stable_hash(fingerprint_payload) == declared_fingerprint
        )
        existing_core = {
            key: value
            for key, value in existing.items()
            if key not in {"audit_fingerprint", "prior_failed_audit"}
        }
        if existing_core == core and fingerprint_valid:
            return output, "reuse_identical"
        if (
            str(existing.get("status", "")) == "reusable"
            and str(existing.get("audit_implementation_version", ""))
            == LINEAGE_AUDIT_IMPLEMENTATION_VERSION
            and fingerprint_valid
        ):
            raise FileExistsError(
                "passing PPO lineage audit is immutable and differs from the requested audit: "
                f"{output}"
            )
        archive = (
            output.parent
            / "failed_attempts"
            / f"{output.stem}.{file_sha256(output)[:16]}{output.suffix}"
        )
        if archive.exists():
            if read_json(archive) != existing:
                raise FileExistsError(f"PPO lineage audit archive collision: {archive}")
        else:
            write_json_atomic(archive, existing)
        core["prior_failed_audit"] = {
            "archived_source_report": str(archive),
            "report_sha256": file_sha256(archive),
            "status": str(existing.get("status", "")),
            "audit_implementation_version": str(
                existing.get("audit_implementation_version", "legacy")
            ),
        }
        action = "archive_failed_and_replace"
    else:
        action = "write_new"
    core["audit_fingerprint"] = stable_hash(core)
    write_json_atomic(output, core)
    return output, action


def audit_manifest(
    manifest_path: str | Path | None,
    *,
    required_seeds: list[int],
    method_config: str | Path | None = None,
) -> dict[str, Any]:
    required = sorted(set(int(seed) for seed in required_seeds))
    if len(required) != len(required_seeds):
        raise ValueError("required optimizer seeds must be unique")
    expected_reward = None
    expected_observation_contract: dict[str, Any] | None = None
    if method_config is not None:
        cfg = load_config(_resolve(method_config))
        expected_reward = validate_reward_semantics(cfg)["sha256"]
        expected_observation_contract = observation_contract(cfg, require_artifacts=False)
    if manifest_path is None:
        return {
            "artifact_kind": LINEAGE_AUDIT_KIND,
            "schema_version": 1,
            "audit_implementation_version": LINEAGE_AUDIT_IMPLEMENTATION_VERSION,
            "status": "retrain_required",
            "reason": "a method config alone cannot prove independent checkpoint lineage",
            "required_seeds": required,
            "valid_seeds": [],
            "missing_seeds": required,
            "invalid_records": [],
        }
    source = _resolve(manifest_path)
    manifest = read_json(source)
    if str(manifest.get("artifact_kind", "")) != REPLICATE_MANIFEST_KIND:
        raise ValueError("unsupported PPO replicate manifest")
    rows = list(manifest.get("records", []) or [])
    by_seed: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        seed = int(row.get("optimizer_seed", row.get("training_seed", -1)))
        by_seed.setdefault(seed, []).append(dict(row))
    valid: list[int] = []
    invalid: list[dict[str, Any]] = []
    valid_rows: list[dict[str, Any]] = []
    compatibility_migrations: list[dict[str, Any]] = []
    for seed in required:
        candidates = by_seed.get(seed, [])
        reasons: list[str] = []
        if len(candidates) != 1:
            reasons.append(f"expected one record, found {len(candidates)}")
            row: dict[str, Any] = {}
        else:
            row = candidates[0]
            if int(row.get("training_seed", -1)) != seed:
                reasons.append("training_seed and optimizer_seed differ")
            for path_field, hash_field in (
                ("checkpoint", "checkpoint_sha256"),
                ("resolved_config", "resolved_config_sha256"),
                ("stage3_report", "stage3_report_sha256"),
            ):
                configured = str(row.get(path_field, ""))
                digest = str(row.get(hash_field, ""))
                if not configured:
                    reasons.append(f"missing {path_field}")
                    continue
                path = _resolve(configured)
                if not path.is_file():
                    reasons.append(f"missing file {path_field}")
                elif file_sha256(path) != digest:
                    reasons.append(f"{path_field} SHA-256 mismatch")
            if expected_reward and str(row.get("reward_semantics_hash", "")) != expected_reward:
                reasons.append("reward semantics differ from method config")
            if expected_observation_contract is not None:
                compatible, migration, reason = _observation_compatibility(
                    row,
                    expected_contract=expected_observation_contract,
                )
                if not compatible:
                    reasons.append(str(reason))
                elif migration is not None:
                    compatibility_migrations.append(migration)
        if reasons:
            invalid.append({"optimizer_seed": seed, "reasons": reasons})
        else:
            valid.append(seed)
            valid_rows.append(row)
    checkpoint_hashes = [str(row.get("checkpoint_sha256", "")) for row in valid_rows]
    duplicate_checkpoint_hashes = sorted(
        digest for digest in set(checkpoint_hashes) if checkpoint_hashes.count(digest) > 1
    )
    budgets = {str(row.get("training_budget", {})) for row in valid_rows}
    reward_hashes = {str(row.get("reward_semantics_hash", "")) for row in valid_rows}
    observation_hashes = {str(row.get("observation_contract_hash", "")) for row in valid_rows}
    global_reasons: list[str] = []
    if duplicate_checkpoint_hashes:
        global_reasons.append("optimizer replicates reuse checkpoint bytes")
    if len(budgets) > 1:
        global_reasons.append("training budgets differ")
    if len(reward_hashes) > 1:
        global_reasons.append("reward semantics differ across replicates")
    if len(observation_hashes) > 1:
        global_reasons.append("observation contracts differ across replicates")
    missing = sorted(set(required).difference(valid))
    if global_reasons or not valid:
        status = "retrain_required"
    elif missing:
        status = "partially_reusable"
    else:
        status = "reusable"
    return {
        "artifact_kind": LINEAGE_AUDIT_KIND,
        "schema_version": 1,
        "audit_implementation_version": LINEAGE_AUDIT_IMPLEMENTATION_VERSION,
        "status": status,
        "manifest": str(source),
        "manifest_sha256": file_sha256(source),
        "method_id": str(manifest.get("method_id", "")),
        "required_seeds": required,
        "valid_seeds": valid,
        "missing_seeds": missing,
        "invalid_records": invalid,
        "global_reasons": global_reasons,
        "duplicate_checkpoint_hashes": duplicate_checkpoint_hashes,
        "compatibility_migrations": compatibility_migrations,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit whether PPO checkpoint replicates are formally reusable")
    parser.add_argument("--replicate-manifest")
    parser.add_argument("--method-config")
    parser.add_argument("--required-seeds", nargs="+", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = audit_manifest(
        args.replicate_manifest,
        required_seeds=args.required_seeds,
        method_config=args.method_config,
    )
    output, action = write_audit_report(args.output, report)
    print(
        f"ppo_replicate_lineage_audit={output} "
        f"status={report['status']} action={action}"
    )


if __name__ == "__main__":
    main()
