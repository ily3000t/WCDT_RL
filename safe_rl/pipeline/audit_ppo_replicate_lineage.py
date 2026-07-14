from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from safe_rl.accvp.contracts.schema import file_sha256, read_json
from safe_rl.ppo_replicates import (
    REPLICATE_MANIFEST_KIND,
    observation_contract,
    validate_reward_semantics,
    write_json_new,
)
from safe_rl.utils.config import REPO_ROOT, load_config


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (REPO_ROOT / value).resolve()


def audit_manifest(
    manifest_path: str | Path | None,
    *,
    required_seeds: list[int],
    method_config: str | Path | None = None,
) -> dict[str, Any]:
    required = sorted(set(int(seed) for seed in required_seeds))
    if len(required) != len(required_seeds):
        raise ValueError("required optimizer seeds must be unique")
    expected_reward = expected_observation = None
    if method_config is not None:
        cfg = load_config(_resolve(method_config))
        expected_reward = validate_reward_semantics(cfg)["sha256"]
        expected_observation = observation_contract(cfg, require_artifacts=False)["sha256"]
    if manifest_path is None:
        return {
            "artifact_kind": "ppo_replicate_lineage_audit_v1",
            "schema_version": 1,
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
            if expected_observation and str(row.get("observation_contract_hash", "")) != expected_observation:
                reasons.append("observation contract differs from method config")
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
        "artifact_kind": "ppo_replicate_lineage_audit_v1",
        "schema_version": 1,
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
    output = write_json_new(_resolve(args.output), report)
    print(f"ppo_replicate_lineage_audit={output} status={report['status']}")


if __name__ == "__main__":
    main()
