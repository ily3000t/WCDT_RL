from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import yaml

from safe_rl.accvp.contracts.schema import file_sha256
from safe_rl.evaluation_protocol import stable_hash
from safe_rl.ppo_replicates import (
    MIN_FORMAL_OPTIMIZER_REPLICATES,
    REPLICATE_MANIFEST_KIND,
    REPLICATE_MANIFEST_SCHEMA_VERSION,
    plain,
)
from safe_rl.utils.config import REPO_ROOT


FACTORIAL_MANIFEST_KIND = "ppo_factorial_manifest_v1"
FACTORIAL_MANIFEST_SCHEMA_VERSION = 1
FACTORIAL_PLAN_KIND = "ppo_factorial_plan_v1"
FACTORIAL_PLAN_SCHEMA_VERSION = 1

BASELINE_ROLE = "action_independent_forecast_baseline"
FINAL_METHOD_ROLE = "final_complete_method_candidate"
EXPECTED_CANDIDATE_METHOD_ROLES: dict[str, str] = {
    "candidate_table_reward_v2": "candidate_table_single_factor_ablation",
    "candidate_table_reward_v2_commitment": "commitment_factor_at_reward_v2",
    "candidate_table_reward_v3_1": "persistence_factor_without_commitment",
    "candidate_table_reward_v3_1_commitment": FINAL_METHOD_ROLE,
}
EXPECTED_FINAL_METHOD_ID = "candidate_table_reward_v3_1_commitment"
_MANIFEST_STATUSES = {"planned", "prepared", "complete"}


def resolve_path(path: str | Path, *, relative_to: str | Path | None = None) -> Path:
    value = Path(path)
    if value.is_absolute():
        return value.resolve()
    base = Path(relative_to).resolve() if relative_to is not None else REPO_ROOT
    return (base / value).resolve()


def resolve_manifest_path(manifest_path: str | Path, referenced_path: str | Path) -> Path:
    """Resolve a manifest reference relative to the manifest that contains it."""

    source = Path(manifest_path).resolve()
    return resolve_path(referenced_path, relative_to=source.parent)


def load_yaml_mapping(path: str | Path) -> dict[str, Any]:
    source = resolve_path(path)
    with source.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, Mapping):
        raise ValueError(f"expected a mapping in {source}")
    return plain(payload)


def read_json_mapping(path: str | Path) -> dict[str, Any]:
    source = Path(path).resolve()
    with source.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"expected a JSON object in {source}")
    return plain(payload)


def atomic_write_json(path: str | Path, payload: Mapping[str, Any], *, replace: bool) -> Path:
    output = Path(path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not replace:
        raise FileExistsError(output)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(plain(payload), handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
    temporary.replace(output)
    return output


def fingerprint_payload(payload: Mapping[str, Any], field: str) -> str:
    return stable_hash({key: value for key, value in plain(payload).items() if key != field})


def load_factorial_contract(
    workflow_path: str | Path,
    matrix_path: str | Path,
) -> dict[str, Any]:
    workflow_source = resolve_path(workflow_path)
    matrix_source = resolve_path(matrix_path)
    workflow = load_yaml_mapping(workflow_source)
    matrix = load_yaml_mapping(matrix_source)
    if workflow.get("artifact_kind") != "accvp_vnext_workflow_contract_v1":
        raise ValueError("unsupported ACCVP VNext workflow contract")
    if matrix.get("artifact_kind") != "accvp_vnext_ppo_ablation_matrix_v1":
        raise ValueError("unsupported PPO ablation matrix")
    workflow_protocol = str(workflow.get("protocol_id", ""))
    matrix_protocol = str(matrix.get("protocol_id", ""))
    if not workflow_protocol or workflow_protocol != matrix_protocol:
        raise ValueError("workflow and PPO ablation matrix protocol_id disagree")

    all_roles = {str(key): str(value) for key, value in (workflow.get("method_roles", {}) or {}).items()}
    candidate_roles = {
        method_id: role
        for method_id, role in all_roles.items()
        if role != BASELINE_ROLE
    }
    if candidate_roles != EXPECTED_CANDIDATE_METHOD_ROLES:
        raise ValueError(
            "workflow must declare exactly the four frozen Candidate factorial methods: "
            f"declared={candidate_roles!r} expected={EXPECTED_CANDIDATE_METHOD_ROLES!r}"
        )
    final_method_id = str(workflow.get("final_method_id", ""))
    if final_method_id != EXPECTED_FINAL_METHOD_ID:
        raise ValueError(
            "workflow final_method_id must be the Reward-v3.1 + commitment method: "
            f"expected={EXPECTED_FINAL_METHOD_ID!r} actual={final_method_id!r}"
        )
    if candidate_roles.get(final_method_id) != FINAL_METHOD_ROLE:
        raise ValueError("workflow final method does not carry the final complete-method role")

    variants = matrix.get("variants", {}) or {}
    if not isinstance(variants, Mapping):
        raise ValueError("PPO ablation matrix variants must be a mapping")
    missing = sorted(set(candidate_roles) - set(variants))
    if missing:
        raise ValueError(f"PPO ablation matrix is missing Candidate methods: {missing}")
    for method_id in candidate_roles:
        variant = variants[method_id]
        if not isinstance(variant, Mapping) or not bool(variant.get("candidate_table", False)):
            raise ValueError(f"factorial method {method_id!r} must enable Candidate Table")

    minimum = int(matrix.get("minimum_optimizer_replicates", MIN_FORMAL_OPTIMIZER_REPLICATES))
    if minimum < MIN_FORMAL_OPTIMIZER_REPLICATES:
        raise ValueError(
            f"factorial protocol requires at least {MIN_FORMAL_OPTIMIZER_REPLICATES} optimizer replicates"
        )
    return {
        "protocol_id": workflow_protocol,
        "method_roles": candidate_roles,
        "method_ids": list(EXPECTED_CANDIDATE_METHOD_ROLES),
        "final_method_id": final_method_id,
        "minimum_optimizer_replicates": minimum,
        "optimizer_seed_role": str(matrix.get("optimizer_seed_role", "ppo_optimizer_replicates")),
        "workflow": workflow,
        "workflow_path": str(workflow_source),
        "workflow_sha256": file_sha256(workflow_source),
        "matrix": matrix,
        "matrix_path": str(matrix_source),
        "matrix_sha256": file_sha256(matrix_source),
    }


def _normalise_hash(value: Any, *, field: str) -> str:
    result = str(value or "")
    if len(result) != 64:
        raise ValueError(f"{field} must be a SHA-256 hex digest")
    try:
        int(result, 16)
    except ValueError as exc:
        raise ValueError(f"{field} must be a SHA-256 hex digest") from exc
    return result.lower()


def _verify_file(path: Path, expected_sha256: Any, *, field: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    expected = _normalise_hash(expected_sha256, field=field)
    actual = file_sha256(path)
    if actual != expected:
        raise ValueError(f"{field} disagrees with file contents: {path}")
    return actual


def validate_replicate_manifest(
    manifest: Mapping[str, Any],
    *,
    method_id: str,
    expected_seeds: list[int],
    verify_files: bool,
) -> dict[str, Any]:
    payload = plain(manifest)
    if payload.get("artifact_kind") != REPLICATE_MANIFEST_KIND:
        raise ValueError(f"{method_id}: unsupported optimizer-replicate manifest")
    if int(payload.get("schema_version", -1)) != REPLICATE_MANIFEST_SCHEMA_VERSION:
        raise ValueError(f"{method_id}: unsupported optimizer-replicate manifest schema")
    if str(payload.get("method_id", "")) != method_id:
        raise ValueError(f"{method_id}: child manifest method_id mismatch")
    status = str(payload.get("status", ""))
    if status not in _MANIFEST_STATUSES:
        raise ValueError(f"{method_id}: invalid child manifest status={status!r}")
    minimum = int(payload.get("minimum_optimizer_replicates", 0))
    if minimum < MIN_FORMAL_OPTIMIZER_REPLICATES:
        raise ValueError(f"{method_id}: child manifest weakens the formal replicate minimum")
    seeds = [int(seed) for seed in payload.get("optimizer_seeds", [])]
    required = [int(seed) for seed in expected_seeds]
    if seeds != required or len(seeds) != len(set(seeds)) or len(seeds) < minimum:
        raise ValueError(f"{method_id}: optimizer seed schedule disagrees with factorial plan")
    records = payload.get("records", []) or []
    if len(records) != len(required):
        raise ValueError(f"{method_id}: expected {len(required)} replicate records")

    budgets: set[str] = set()
    observation_hashes: set[str] = set()
    reward_hashes: set[str] = set()
    checkpoint_hashes: list[str] = []
    checkpoint_paths: set[str] = set()
    run_ids: set[str] = set()
    actual_seeds: list[int] = []
    for record in records:
        if str(record.get("method_id", "")) != method_id:
            raise ValueError(f"{method_id}: record method_id mismatch")
        optimizer_seed = int(record.get("optimizer_seed", -1))
        if int(record.get("training_seed", -1)) != optimizer_seed:
            raise ValueError(f"{method_id}: training_seed and optimizer_seed disagree")
        actual_seeds.append(optimizer_seed)
        run_id = str(record.get("run_id", ""))
        if not run_id or run_id in run_ids:
            raise ValueError(f"{method_id}: replicate run_id values must be unique and non-empty")
        run_ids.add(run_id)
        budget = record.get("training_budget", {}) or {}
        required_budget_fields = {"total_timesteps", "n_steps", "batch_size", "ppo_num_envs"}
        if not isinstance(budget, Mapping) or not required_budget_fields.issubset(budget):
            raise ValueError(f"{method_id}: replicate training budget is incomplete")
        if any(int(budget[field]) <= 0 for field in required_budget_fields):
            raise ValueError(f"{method_id}: replicate training budget values must be positive")
        budgets.add(stable_hash(budget))
        observation_hash = _normalise_hash(
            record.get("observation_contract_hash"), field="observation_contract_hash"
        )
        reward_hash = _normalise_hash(
            record.get("reward_semantics_hash"), field="reward_semantics_hash"
        )
        if stable_hash(record.get("observation_contract", {}) or {}) != observation_hash:
            raise ValueError(f"{method_id}: observation contract hash disagrees with its payload")
        if stable_hash(record.get("reward_semantics", {}) or {}) != reward_hash:
            raise ValueError(f"{method_id}: reward semantics hash disagrees with its payload")
        observation_hashes.add(observation_hash)
        reward_hashes.add(reward_hash)

        config_value = str(record.get("resolved_config", ""))
        config_sha = str(record.get("resolved_config_sha256", ""))
        if verify_files and config_sha:
            _verify_file(resolve_path(config_value), config_sha, field="resolved_config_sha256")

        checkpoint_value = str(record.get("checkpoint", ""))
        checkpoint_sha = str(record.get("checkpoint_sha256", ""))
        report_value = str(record.get("stage3_report", ""))
        report_sha = str(record.get("stage3_report_sha256", ""))
        if status == "complete":
            if not all((config_sha, checkpoint_value, checkpoint_sha, report_value, report_sha)):
                raise ValueError(f"{method_id}: complete replicate record has incomplete lineage")
            checkpoint_hash = _normalise_hash(checkpoint_sha, field="checkpoint_sha256")
            checkpoint_path = str(resolve_path(checkpoint_value))
            if checkpoint_path in checkpoint_paths:
                raise ValueError(f"{method_id}: checkpoint paths must be unique")
            checkpoint_paths.add(checkpoint_path)
            checkpoint_hashes.append(checkpoint_hash)
            if verify_files:
                _verify_file(Path(checkpoint_path), checkpoint_hash, field="checkpoint_sha256")
                _verify_file(resolve_path(report_value), report_sha, field="stage3_report_sha256")
                report = read_json_mapping(resolve_path(report_value))
                if int(report.get("optimizer_seed", -1)) != optimizer_seed:
                    raise ValueError(f"{method_id}: Stage3 report optimizer seed mismatch")
                for field, expected in (
                    ("observation_contract_hash", observation_hash),
                    ("reward_semantics_hash", reward_hash),
                ):
                    if str(report.get(field, "")) != expected:
                        raise ValueError(f"{method_id}: Stage3 report {field} mismatch")
            actual_timesteps = int(record.get("actual_total_timesteps", 0))
            if actual_timesteps < int(budget["total_timesteps"]):
                raise ValueError(f"{method_id}: PPO replicate did not reach its frozen training budget")
    if actual_seeds != required:
        raise ValueError(f"{method_id}: replicate record order/seed schedule disagrees with plan")
    if len(budgets) != 1 or len(observation_hashes) != 1 or len(reward_hashes) != 1:
        raise ValueError(f"{method_id}: replicate contracts or training budgets are inconsistent")
    if status == "complete" and len(set(checkpoint_hashes)) != len(checkpoint_hashes):
        raise ValueError(f"{method_id}: checkpoint content hashes must be unique")

    declared_fingerprint = str(payload.get("manifest_fingerprint", ""))
    if declared_fingerprint:
        expected_fingerprint = fingerprint_payload(payload, "manifest_fingerprint")
        if declared_fingerprint != expected_fingerprint:
            raise ValueError(f"{method_id}: child manifest fingerprint mismatch")
    return {
        "status": status,
        "optimizer_seeds": seeds,
        "training_budget_hash": next(iter(budgets)),
        "observation_contract_hash": next(iter(observation_hashes)),
        "reward_semantics_hash": next(iter(reward_hashes)),
        "checkpoint_sha256s": checkpoint_hashes,
        "manifest_fingerprint": declared_fingerprint,
    }


def build_factorial_manifest(
    *,
    protocol_id: str,
    final_method_id: str,
    method_roles: Mapping[str, str],
    optimizer_seeds: list[int],
    plan_path: str | Path,
    child_manifests: Mapping[str, str | Path],
    status: str,
    verify_files: bool = True,
) -> dict[str, Any]:
    if status not in _MANIFEST_STATUSES:
        raise ValueError(f"invalid factorial manifest status={status!r}")
    roles = {str(key): str(value) for key, value in method_roles.items()}
    if roles != EXPECTED_CANDIDATE_METHOD_ROLES:
        raise ValueError("factorial manifest method roles disagree with frozen workflow")
    if final_method_id != EXPECTED_FINAL_METHOD_ID or roles.get(final_method_id) != FINAL_METHOD_ROLE:
        raise ValueError("factorial manifest final method is not Reward-v3.1 + commitment")
    if set(child_manifests) != set(roles):
        raise ValueError("factorial manifest must bind one child manifest per Candidate method")

    methods: dict[str, Any] = {}
    summaries: dict[str, Any] = {}
    for method_id in EXPECTED_CANDIDATE_METHOD_ROLES:
        source = Path(child_manifests[method_id]).resolve()
        child = read_json_mapping(source)
        summary = validate_replicate_manifest(
            child,
            method_id=method_id,
            expected_seeds=optimizer_seeds,
            verify_files=verify_files,
        )
        summaries[method_id] = summary
        methods[method_id] = {
            "role": roles[method_id],
            "status": summary["status"],
            "replicate_manifest": str(source),
            "replicate_manifest_sha256": file_sha256(source),
            "manifest_fingerprint": summary["manifest_fingerprint"],
            "optimizer_seeds": summary["optimizer_seeds"],
            "checkpoint_sha256s": summary["checkpoint_sha256s"],
            "training_budget_hash": summary["training_budget_hash"],
            "observation_contract_hash": summary["observation_contract_hash"],
            "reward_semantics_hash": summary["reward_semantics_hash"],
        }

    budget_hashes = {item["training_budget_hash"] for item in summaries.values()}
    observation_hashes = {item["observation_contract_hash"] for item in summaries.values()}
    if len(budget_hashes) != 1:
        raise ValueError("Candidate factorial methods do not share one frozen PPO training budget")
    if len(observation_hashes) != 1:
        raise ValueError("Candidate factorial methods do not share one observation contract")
    all_checkpoints = [
        checkpoint
        for item in summaries.values()
        for checkpoint in item["checkpoint_sha256s"]
    ]
    if status == "complete":
        if any(item["status"] != "complete" for item in summaries.values()):
            raise ValueError("complete factorial manifest requires four complete child manifests")
        expected_count = len(EXPECTED_CANDIDATE_METHOD_ROLES) * len(optimizer_seeds)
        if len(all_checkpoints) != expected_count or len(set(all_checkpoints)) != expected_count:
            raise ValueError("factorial checkpoint hashes must be complete and globally unique")
    elif status == "prepared" and any(item["status"] not in {"prepared", "complete"} for item in summaries.values()):
        raise ValueError("prepared factorial manifest requires prepared/complete child manifests")

    plan_source = Path(plan_path).resolve()
    if not plan_source.is_file():
        raise FileNotFoundError(plan_source)
    payload = {
        "artifact_kind": FACTORIAL_MANIFEST_KIND,
        "schema_version": FACTORIAL_MANIFEST_SCHEMA_VERSION,
        "status": status,
        "protocol_id": str(protocol_id),
        "final_method_id": final_method_id,
        "method_roles": roles,
        "optimizer_seeds": [int(seed) for seed in optimizer_seeds],
        "factorial_plan": str(plan_source),
        "factorial_plan_sha256": file_sha256(plan_source),
        "methods": methods,
        "training_budget_hash": next(iter(budget_hashes)),
        "observation_contract_hash": next(iter(observation_hashes)),
        "checkpoint_sha256s": all_checkpoints,
    }
    payload["manifest_fingerprint"] = fingerprint_payload(payload, "manifest_fingerprint")
    return payload


def validate_factorial_manifest(
    manifest_path: str | Path,
    *,
    require_complete: bool = True,
    verify_files: bool = True,
) -> dict[str, Any]:
    source = Path(manifest_path).resolve()
    payload = read_json_mapping(source)
    if payload.get("artifact_kind") != FACTORIAL_MANIFEST_KIND:
        raise ValueError("unsupported PPO factorial manifest")
    if int(payload.get("schema_version", -1)) != FACTORIAL_MANIFEST_SCHEMA_VERSION:
        raise ValueError("unsupported PPO factorial manifest schema")
    status = str(payload.get("status", ""))
    if status not in _MANIFEST_STATUSES:
        raise ValueError(f"invalid PPO factorial manifest status={status!r}")
    if require_complete and status != "complete":
        raise ValueError("formal PPO factorial manifest is not complete")
    roles = {str(key): str(value) for key, value in (payload.get("method_roles", {}) or {}).items()}
    final_method_id = str(payload.get("final_method_id", ""))
    seeds = [int(seed) for seed in payload.get("optimizer_seeds", [])]
    if len(seeds) < MIN_FORMAL_OPTIMIZER_REPLICATES or len(seeds) != len(set(seeds)):
        raise ValueError("PPO factorial manifest requires at least five unique optimizer seeds")
    if roles != EXPECTED_CANDIDATE_METHOD_ROLES:
        raise ValueError("PPO factorial manifest has an unexpected method set or role mapping")
    if final_method_id != EXPECTED_FINAL_METHOD_ID or roles.get(final_method_id) != FINAL_METHOD_ROLE:
        raise ValueError("PPO factorial manifest final method is not Reward-v3.1 + commitment")

    methods = payload.get("methods", {}) or {}
    if set(methods) != set(roles):
        raise ValueError("PPO factorial manifest methods are incomplete")
    child_paths: dict[str, Path] = {}
    for method_id, entry in methods.items():
        if str(entry.get("role", "")) != roles[method_id]:
            raise ValueError(f"{method_id}: factorial method role mismatch")
        child_path = resolve_manifest_path(source, str(entry.get("replicate_manifest", "")))
        if verify_files:
            _verify_file(
                child_path,
                entry.get("replicate_manifest_sha256"),
                field="replicate_manifest_sha256",
            )
        child_paths[method_id] = child_path
    plan_path = resolve_manifest_path(source, str(payload.get("factorial_plan", "")))
    if verify_files:
        _verify_file(plan_path, payload.get("factorial_plan_sha256"), field="factorial_plan_sha256")
    plan = read_json_mapping(plan_path)
    if plan.get("artifact_kind") != FACTORIAL_PLAN_KIND or int(plan.get("schema_version", -1)) != FACTORIAL_PLAN_SCHEMA_VERSION:
        raise ValueError("PPO factorial manifest references an unsupported factorial plan")
    if str(plan.get("plan_fingerprint", "")) != fingerprint_payload(plan, "plan_fingerprint"):
        raise ValueError("referenced PPO factorial plan fingerprint mismatch")
    for field, expected in (
        ("protocol_id", str(payload.get("protocol_id", ""))),
        ("final_method_id", final_method_id),
        ("method_roles", roles),
        ("optimizer_seeds", seeds),
    ):
        if plan.get(field) != expected:
            raise ValueError(f"PPO factorial manifest {field} disagrees with its frozen plan")

    rebuilt = build_factorial_manifest(
        protocol_id=str(payload.get("protocol_id", "")),
        final_method_id=final_method_id,
        method_roles=roles,
        optimizer_seeds=seeds,
        plan_path=plan_path,
        child_manifests=child_paths,
        status=status,
        verify_files=verify_files,
    )
    normalised_methods = plain(methods)
    for method_id, child_path in child_paths.items():
        normalised_methods[method_id]["replicate_manifest"] = str(child_path)
    if normalised_methods != rebuilt.get("methods"):
        raise ValueError("PPO factorial manifest methods disagree with child manifests")
    for field in (
        "training_budget_hash",
        "observation_contract_hash",
        "checkpoint_sha256s",
        "factorial_plan_sha256",
    ):
        if payload.get(field) != rebuilt.get(field):
            raise ValueError(f"PPO factorial manifest {field} disagrees with child manifests")
    declared = str(payload.get("manifest_fingerprint", ""))
    if declared != fingerprint_payload(payload, "manifest_fingerprint"):
        raise ValueError("PPO factorial manifest fingerprint mismatch")
    return payload
