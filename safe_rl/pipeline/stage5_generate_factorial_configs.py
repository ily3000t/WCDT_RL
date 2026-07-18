from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping

import yaml

from safe_rl.accvp.contracts.schema import file_sha256, read_json, stable_hash
from safe_rl.evaluation_protocol import protocol_snapshot
from safe_rl.pipeline.audit_ppo_replicate_lineage import audit_manifest
from safe_rl.pipeline.stage5_generate_replicated_configs import (
    _group,
    _load_complete_manifest,
    _row_map,
)
from safe_rl.pipeline.stage5_replicated_aggregate import REQUEST_ARTIFACT_KIND
from safe_rl.ppo_factorial import (
    EXPECTED_FINAL_METHOD_ID,
    FACTORIAL_MANIFEST_KIND,
    validate_factorial_manifest,
)
from safe_rl.ppo_replicates import plain, write_json_new, write_yaml_atomic
from safe_rl.utils.config import REPO_ROOT, load_config


FACTORIAL_REQUEST_KIND = "stage5_factorial_request_v1"
FACTORIAL_REQUEST_SCHEMA_VERSION = 1
FACTORIAL_RUNTIME_KIND = "accvp_runtime_benchmark_factorial_v1"
SINGLE_RUNTIME_KIND = "accvp_runtime_benchmark_replicates_v1"
DEFAULT_SECONDARY_SIMULATOR_SEED_COUNT = 100
DEFAULT_PRIMARY_SIMULATOR_SEED_COUNT = 300
FINAL_COMPARISON_ID = (
    "wcdt_reward_v2__vs__candidate_table_reward_v3_1_commitment"
)

# These defaults are also checked against workflow.factorial.comparisons when
# that preregistration block is present.  Keeping the frozen fallback here makes
# the standalone generator usable without weakening the formal workflow check.
DEFAULT_COMPARISONS: tuple[dict[str, str], ...] = (
    {
        "comparison_id": "wcdt_reward_v2__vs__candidate_table_reward_v2",
        "left_method_id": "wcdt_reward_v2",
        "right_method_id": "candidate_table_reward_v2",
        "family": "candidate_table_attribution",
        "role": "candidate_table_single_factor",
    },
    {
        "comparison_id": (
            "candidate_table_reward_v2__vs__candidate_table_reward_v2_commitment"
        ),
        "left_method_id": "candidate_table_reward_v2",
        "right_method_id": "candidate_table_reward_v2_commitment",
        "family": "candidate_factorial_attribution",
        "role": "commitment_at_reward_v2",
    },
    {
        "comparison_id": "candidate_table_reward_v2__vs__candidate_table_reward_v3_1",
        "left_method_id": "candidate_table_reward_v2",
        "right_method_id": "candidate_table_reward_v3_1",
        "family": "candidate_factorial_attribution",
        "role": "persistence_without_commitment",
    },
    {
        "comparison_id": (
            "candidate_table_reward_v2_commitment__vs__"
            "candidate_table_reward_v3_1_commitment"
        ),
        "left_method_id": "candidate_table_reward_v2_commitment",
        "right_method_id": "candidate_table_reward_v3_1_commitment",
        "family": "candidate_factorial_attribution",
        "role": "persistence_with_commitment",
    },
    {
        "comparison_id": (
            "candidate_table_reward_v3_1__vs__"
            "candidate_table_reward_v3_1_commitment"
        ),
        "left_method_id": "candidate_table_reward_v3_1",
        "right_method_id": "candidate_table_reward_v3_1_commitment",
        "family": "candidate_factorial_attribution",
        "role": "commitment_at_reward_v3_1",
    },
    {
        "comparison_id": FINAL_COMPARISON_ID,
        "left_method_id": "wcdt_reward_v2",
        "right_method_id": EXPECTED_FINAL_METHOD_ID,
        "family": "final_method_confirmatory",
        "role": "primary_final",
    },
)


def _resolve(path: str | Path, *, relative_to: Path | None = None) -> Path:
    value = Path(path)
    if value.is_absolute():
        return value.resolve()
    return ((relative_to or REPO_ROOT) / value).resolve()


def _normalise_sha256(value: Any, *, field: str) -> str:
    digest = str(value or "").strip().lower()
    if len(digest) != 64:
        raise ValueError(f"{field} must be a 64-character SHA-256 digest")
    try:
        int(digest, 16)
    except ValueError as exc:
        raise ValueError(f"{field} must be a hexadecimal SHA-256 digest") from exc
    return digest


def _verified_json_reference(
    source: str | Path,
    expected_sha256: Any,
    *,
    relative_to: Path,
    field: str,
) -> tuple[Path, dict[str, Any]]:
    path = _resolve(source, relative_to=relative_to)
    if not path.is_file():
        raise FileNotFoundError(path)
    expected = _normalise_sha256(expected_sha256, field=field)
    if file_sha256(path) != expected:
        raise ValueError(f"{field} mismatch: {path}")
    return path, read_json(path)


def _normalise_comparison(row: Mapping[str, Any]) -> dict[str, str]:
    result = {
        key: str(row.get(key, "")).strip()
        for key in (
            "comparison_id",
            "left_method_id",
            "right_method_id",
            "family",
            "role",
        )
    }
    if not all(result.values()):
        raise ValueError(f"factorial comparison declaration is incomplete: {dict(row)!r}")
    return result


def _comparison_specs(workflow_config: str | Path | None) -> list[dict[str, str]]:
    frozen = [_normalise_comparison(item) for item in DEFAULT_COMPARISONS]
    if workflow_config is None:
        return frozen
    source = _resolve(workflow_config)
    with source.open("r", encoding="utf-8") as handle:
        workflow = yaml.safe_load(handle) or {}
    if str(workflow.get("artifact_kind", "")) != "accvp_vnext_workflow_contract_v1":
        raise ValueError("unsupported ACCVP VNext workflow contract")
    block = workflow.get("factorial", {}) or {}
    declared = block.get("comparisons") if isinstance(block, Mapping) else None
    if declared is None:
        return frozen
    if not isinstance(declared, list):
        raise ValueError("workflow.factorial.comparisons must be a list")
    parsed = [_normalise_comparison(item) for item in declared]
    expected_by_id = {item["comparison_id"]: item for item in frozen}
    actual_by_id = {item["comparison_id"]: item for item in parsed}
    if len(actual_by_id) != len(parsed):
        raise ValueError("workflow factorial comparisons contain duplicate comparison_id values")
    if actual_by_id != expected_by_id:
        raise ValueError(
            "workflow factorial comparisons disagree with the frozen six-comparison design"
        )
    return parsed


def _stage5_evaluation_budget(
    workflow_config: str | Path | None,
    *,
    available_seed_count: int,
) -> dict[str, Any]:
    if available_seed_count <= 0:
        raise ValueError("Stage5 confirmatory seed cohort is empty")
    if workflow_config is None:
        return {
            "secondary_simulator_seed_count": int(available_seed_count),
            "primary_simulator_seed_count": int(available_seed_count),
            "nested_seed_prefix": True,
            "episode_cache_enabled": False,
            "source": "standalone_full_cohort_default",
        }
    source = _resolve(workflow_config)
    with source.open("r", encoding="utf-8") as handle:
        workflow = yaml.safe_load(handle) or {}
    factorial = workflow.get("factorial", {}) or {}
    raw = factorial.get("stage5_evaluation", {}) if isinstance(factorial, Mapping) else {}
    if not isinstance(raw, Mapping):
        raise ValueError("workflow.factorial.stage5_evaluation must be a mapping")
    secondary = int(
        raw.get(
            "secondary_simulator_seed_count",
            DEFAULT_SECONDARY_SIMULATOR_SEED_COUNT,
        )
    )
    primary = int(
        raw.get(
            "primary_simulator_seed_count",
            DEFAULT_PRIMARY_SIMULATOR_SEED_COUNT,
        )
    )
    nested = bool(raw.get("nested_seed_prefix", True))
    cache_enabled = bool(raw.get("episode_cache_enabled", True))
    if secondary <= 0 or primary <= 0:
        raise ValueError("Stage5 simulator-seed budgets must be positive")
    if secondary > primary:
        raise ValueError("Stage5 secondary simulator-seed budget exceeds primary budget")
    if primary > available_seed_count:
        raise ValueError(
            "Stage5 primary simulator-seed budget exceeds the registered confirmatory cohort: "
            f"required={primary} available={available_seed_count}"
        )
    if not nested:
        raise ValueError("formal Stage5 requires a nested common seed prefix")
    if not cache_enabled:
        raise ValueError("formal Stage5 workflow requires episode-cache reuse")
    return {
        "secondary_simulator_seed_count": secondary,
        "primary_simulator_seed_count": primary,
        "nested_seed_prefix": True,
        "episode_cache_enabled": True,
        "source": str(source),
    }


def _runtime_seed_reports(
    runtime_path: Path,
    runtime: Mapping[str, Any],
    *,
    expected_manifest_sha256: str,
) -> dict[int, dict[str, str]]:
    if str(runtime.get("artifact_kind", "")) != SINGLE_RUNTIME_KIND:
        raise ValueError(f"unsupported per-method runtime report: {runtime_path}")
    if not bool((runtime.get("gate", {}) or {}).get("pass", False)):
        raise ValueError(f"per-method runtime gate failed: {runtime_path}")
    recorded_manifest = _normalise_sha256(
        runtime.get("replicate_manifest_sha256"),
        field="runtime replicate_manifest_sha256",
    )
    if recorded_manifest != expected_manifest_sha256:
        raise ValueError(
            "runtime report is not bound to the expected optimizer-replicate manifest: "
            f"{runtime_path}"
        )
    reports: dict[int, dict[str, str]] = {}
    for raw in runtime.get("replicates", []) or []:
        seed = int(raw.get("optimizer_seed", -1))
        if seed in reports:
            raise ValueError(f"duplicate optimizer seed in runtime report: {seed}")
        report, _ = _verified_json_reference(
            raw.get("report", ""),
            raw.get("report_sha256"),
            relative_to=runtime_path.parent,
            field=f"runtime replicate report sha256 for seed {seed}",
        )
        reports[seed] = {
            "path": str(report),
            "sha256": file_sha256(report),
        }
    if not reports:
        raise ValueError(f"runtime report has no optimizer replicates: {runtime_path}")
    return reports


def _runtime_coverage(
    runtime_report: str | Path,
    *,
    factorial_path: Path,
    factorial: Mapping[str, Any],
) -> tuple[Path, dict[str, dict[str, Any]]]:
    source = _resolve(runtime_report)
    payload = read_json(source)
    methods = factorial.get("methods", {}) or {}
    if not isinstance(methods, Mapping):
        raise ValueError("PPO factorial methods must be a mapping")

    if str(payload.get("artifact_kind", "")) == SINGLE_RUNTIME_KIND:
        manifest_sha = _normalise_sha256(
            payload.get("replicate_manifest_sha256"),
            field="runtime replicate_manifest_sha256",
        )
        matched = [
            str(method_id)
            for method_id, entry in methods.items()
            if str(entry.get("replicate_manifest_sha256", "")).lower() == manifest_sha
        ]
        if len(matched) != 1:
            raise ValueError("single runtime report does not bind exactly one factorial method")
        method_id = matched[0]
        return source, {
            method_id: {
                "runtime_report": str(source),
                "runtime_report_sha256": file_sha256(source),
                "replicate_manifest_sha256": manifest_sha,
                "reports": _runtime_seed_reports(
                    source,
                    payload,
                    expected_manifest_sha256=manifest_sha,
                ),
            }
        }

    if str(payload.get("artifact_kind", "")) != FACTORIAL_RUNTIME_KIND:
        raise ValueError("Stage5 factorial generation requires a factorial runtime report")
    if int(payload.get("schema_version", -1)) != 1 or payload.get("status") != "complete":
        raise ValueError("factorial runtime report is not complete")
    if not bool((payload.get("gate", {}) or {}).get("pass", False)):
        raise ValueError("factorial runtime gate did not pass")
    if str(payload.get("final_method_id", "")) != str(factorial.get("final_method_id", "")):
        raise ValueError("factorial runtime final_method_id mismatch")
    declared_factorial, _ = _verified_json_reference(
        payload.get("factorial_manifest", ""),
        payload.get("factorial_manifest_sha256"),
        relative_to=source.parent,
        field="runtime factorial_manifest_sha256",
    )
    if declared_factorial != factorial_path:
        raise ValueError("factorial runtime report binds a different PPO factorial manifest")

    runtime_methods = payload.get("methods", {}) or {}
    if not isinstance(runtime_methods, Mapping):
        raise ValueError("factorial runtime methods must be a mapping")
    coverage: dict[str, dict[str, Any]] = {}
    for method_id, factorial_entry in methods.items():
        entry = runtime_methods.get(method_id)
        if not isinstance(entry, Mapping):
            raise ValueError(f"factorial runtime report is missing method {method_id!r}")
        expected_manifest_sha = _normalise_sha256(
            factorial_entry.get("replicate_manifest_sha256"),
            field=f"{method_id} replicate_manifest_sha256",
        )
        declared_manifest_sha = _normalise_sha256(
            entry.get("replicate_manifest_sha256"),
            field=f"{method_id} runtime replicate_manifest_sha256",
        )
        if declared_manifest_sha != expected_manifest_sha:
            raise ValueError(f"factorial runtime method {method_id!r} child-manifest mismatch")
        runtime_path, runtime = _verified_json_reference(
            entry.get("runtime_report", ""),
            entry.get("runtime_report_sha256"),
            relative_to=source.parent,
            field=f"{method_id} runtime_report_sha256",
        )
        if not bool((entry.get("gate", {}) or {}).get("pass", False)):
            raise ValueError(f"factorial runtime method gate failed: {method_id}")
        coverage[str(method_id)] = {
            "runtime_report": str(runtime_path),
            "runtime_report_sha256": file_sha256(runtime_path),
            "replicate_manifest_sha256": expected_manifest_sha,
            "reports": _runtime_seed_reports(
                runtime_path,
                runtime,
                expected_manifest_sha256=expected_manifest_sha,
            ),
        }
    return source, coverage


def _candidate_binding(row: Mapping[str, Any]) -> dict[str, str]:
    observation = dict(row.get("observation_contract", {}) or {})
    binding = {
        "path": str(observation.get("accvp_artifact_manifest", "")),
        "sha256": str(observation.get("accvp_artifact_manifest_sha256", "")),
        "artifact_fingerprint": str(observation.get("accvp_artifact_fingerprint", "")),
        "artifact_variant": str(observation.get("accvp_artifact_variant", "")),
        "formal_runtime_contract_sha256": str(
            observation.get("formal_runtime_contract_sha256", "")
        ),
    }
    if not all(binding.values()):
        raise ValueError("Candidate replicate record lacks complete ACCVP bundle binding")
    return binding


def _read_resolved_config(row: Mapping[str, Any]) -> dict[str, Any]:
    path = _resolve(str(row.get("resolved_config", "")))
    if not path.is_file():
        raise FileNotFoundError(path)
    expected = str(row.get("resolved_config_sha256", ""))
    if expected and file_sha256(path) != expected:
        raise ValueError(f"resolved PPO config hash mismatch: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, Mapping):
        raise ValueError(f"resolved PPO config must be a mapping: {path}")
    return plain(payload)


def _risk_checkpoint_binding(config: Mapping[str, Any]) -> dict[str, str]:
    accvp = dict(config.get("accvp", {}) or {})
    rl = dict(config.get("rl", {}) or {})
    reward = dict(rl.get("shield_guided_reward", {}) or {})
    stage5 = dict(config.get("stage5", {}) or {})
    candidates = [
        str(stage5.get("risk_checkpoint", "") or "").strip(),
        str(accvp.get("risk_checkpoint", "") or "").strip(),
        str(reward.get("risk_checkpoint", "") or "").strip(),
    ]
    paths = [_resolve(source) for source in candidates if source]
    if not paths:
        raise ValueError("formal Stage5 method config is missing a frozen Risk checkpoint")
    records = {
        (str(path), file_sha256(path))
        for path in paths
        if path.is_file()
    }
    if len(records) != len(set(paths)):
        missing = sorted(str(path) for path in paths if not path.is_file())
        raise FileNotFoundError(
            "formal Stage5 Risk checkpoint does not exist: " + ", ".join(missing)
        )
    hashes = {digest for _path, digest in records}
    if len(hashes) != 1:
        raise ValueError("one method config binds multiple different Risk checkpoints")
    canonical_path = sorted(path for path, _digest in records)[0]
    return {
        "path": canonical_path,
        "sha256": hashes.pop(),
    }


def _episode_cache_binding(
    *,
    output: Path,
    group: Mapping[str, Any],
    row: Mapping[str, Any],
    risk_binding: Mapping[str, str],
) -> dict[str, Any]:
    method_id = str(row.get("method_id", "")).strip()
    optimizer_seed = int(row.get("optimizer_seed", row.get("training_seed", -1)))
    if not method_id or optimizer_seed < 0:
        raise ValueError("Stage5 cache binding requires method_id and optimizer_seed")
    group_contract = {
        key: plain(value)
        for key, value in group.items()
        if key not in {"name", "comparative", "evaluation_cache"}
    }
    identity = {
        "method_id": method_id,
        "optimizer_seed": optimizer_seed,
        "checkpoint_sha256": _normalise_sha256(
            row.get("checkpoint_sha256"),
            field=f"{method_id} checkpoint_sha256",
        ),
        "resolved_config": str(_resolve(str(row.get("resolved_config", "")))),
        "resolved_config_sha256": _normalise_sha256(
            row.get("resolved_config_sha256"),
            field=f"{method_id} resolved_config_sha256",
        ),
        "reward_semantics_hash": _normalise_sha256(
            row.get("reward_semantics_hash"),
            field=f"{method_id} reward_semantics_hash",
        ),
        "observation_contract_hash": _normalise_sha256(
            row.get("observation_contract_hash"),
            field=f"{method_id} observation_contract_hash",
        ),
        "risk_checkpoint_sha256": _normalise_sha256(
            risk_binding.get("sha256"),
            field="Stage5 Risk checkpoint sha256",
        ),
        "group_execution_contract_sha256": stable_hash(group_contract),
        "shield_enabled": bool(group.get("shield", False)),
        "policy_type": str(group.get("policy_type", "sb3_ppo")),
    }
    cache_dir = (
        output.parent
        / "episode_cache"
        / method_id
        / f"optimizer_seed_{optimizer_seed}"
    ).resolve()
    return {
        "artifact_kind": "stage5_episode_cache_binding_v1",
        "schema_version": 1,
        "cache_dir": str(cache_dir),
        "execution_fingerprint": stable_hash(identity),
        "identity": identity,
    }


def _write_yaml_idempotent(path: Path, payload: Mapping[str, Any]) -> Path:
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            existing = yaml.safe_load(handle) or {}
        if plain(existing) != plain(payload):
            raise FileExistsError(f"refusing to replace a different frozen Stage5 config: {path}")
        return path
    return write_yaml_atomic(path, payload)


def _write_json_idempotent(path: Path, payload: Mapping[str, Any]) -> Path:
    if path.exists():
        if read_json(path) != plain(payload):
            raise FileExistsError(f"refusing to replace a different frozen Stage5 request: {path}")
        return path
    return write_json_new(path, payload)


def generate(
    *,
    baseline_manifest: str | Path,
    factorial_manifest: str | Path,
    protocol: str | Path,
    seed_role: str,
    runtime_factorial_report: str | Path,
    output_dir: str | Path,
    workflow_config: str | Path | None = None,
) -> Path:
    factorial_path = _resolve(factorial_manifest)
    factorial = validate_factorial_manifest(
        factorial_path,
        require_complete=True,
        verify_files=True,
    )
    if str(factorial.get("artifact_kind", "")) != FACTORIAL_MANIFEST_KIND:
        raise ValueError("unsupported PPO factorial manifest")
    if str(factorial.get("final_method_id", "")) != EXPECTED_FINAL_METHOD_ID:
        raise ValueError("Stage5 factorial final method must be Reward-v3.1 + commitment")
    comparisons = _comparison_specs(workflow_config)

    baseline_path, baseline = _load_complete_manifest(baseline_manifest)
    if str(baseline.get("method_id", "")) != "wcdt_reward_v2":
        raise ValueError("formal Stage5 factorial baseline must be wcdt_reward_v2")
    manifests: dict[str, tuple[Path, dict[str, Any]]] = {
        "wcdt_reward_v2": (baseline_path, baseline)
    }
    factorial_methods = factorial.get("methods", {}) or {}
    for method_id, entry in factorial_methods.items():
        child_path = _resolve(
            str(entry.get("replicate_manifest", "")),
            relative_to=factorial_path.parent,
        )
        expected = _normalise_sha256(
            entry.get("replicate_manifest_sha256"),
            field=f"{method_id} replicate_manifest_sha256",
        )
        if file_sha256(child_path) != expected:
            raise ValueError(f"factorial child manifest hash mismatch: {method_id}")
        manifests[str(method_id)] = _load_complete_manifest(child_path)

    row_maps = {method_id: _row_map(payload) for method_id, (_path, payload) in manifests.items()}
    seed_sets = {method_id: set(rows) for method_id, rows in row_maps.items()}
    expected_seeds = set(int(seed) for seed in factorial.get("optimizer_seeds", []) or [])
    if len(expected_seeds) < 5:
        raise ValueError("Stage5 factorial requires at least five optimizer seeds")
    if any(seeds != expected_seeds for seeds in seed_sets.values()):
        raise ValueError("baseline and factorial methods must share one exact optimizer-seed set")
    seeds = sorted(expected_seeds)
    for path, _payload in manifests.values():
        if audit_manifest(path, required_seeds=seeds)["status"] != "reusable":
            raise ValueError(f"replicate manifest is not reusable: {path}")
    config_maps = {
        method_id: {
            seed: _read_resolved_config(rows[seed])
            for seed in seeds
        }
        for method_id, rows in row_maps.items()
    }
    risk_bindings = {
        (
            binding["path"],
            binding["sha256"],
        )
        for configs in config_maps.values()
        for binding in (
            _risk_checkpoint_binding(config)
            for config in configs.values()
        )
    }
    risk_hashes = {digest for _path, digest in risk_bindings}
    if len(risk_hashes) != 1:
        raise ValueError("formal Stage5 methods do not bind one frozen Risk checkpoint")
    risk_paths = sorted(path for path, _digest in risk_bindings)
    risk_binding = {
        "path": risk_paths[0],
        "sha256": risk_hashes.pop(),
    }

    # All Candidate arms are observation-identical and must bind the same frozen
    # ACCVP bundle.  Validate the explicit binding, not only its summary hash.
    frozen_binding: dict[str, str] | None = None
    for method_id, rows in row_maps.items():
        if method_id == "wcdt_reward_v2":
            continue
        for seed in seeds:
            binding = _candidate_binding(rows[seed])
            if frozen_binding is not None and binding != frozen_binding:
                raise ValueError("Candidate factorial methods do not bind one frozen ACCVP bundle")
            frozen_binding = binding
    if frozen_binding is None:
        raise ValueError("PPO factorial manifest contains no Candidate methods")

    runtime_path, runtime_coverage = _runtime_coverage(
        runtime_factorial_report,
        factorial_path=factorial_path,
        factorial=factorial,
    )
    required_runtime_methods = {
        method_id
        for comparison in comparisons
        for method_id in (
            comparison["left_method_id"],
            comparison["right_method_id"],
        )
        if method_id != "wcdt_reward_v2"
    }
    missing_runtime = sorted(required_runtime_methods - set(runtime_coverage))
    if missing_runtime:
        raise ValueError(
            "factorial Stage5 requires checkpoint-bound runtime reports for every Candidate arm; "
            f"missing={missing_runtime}"
        )
    for method_id in required_runtime_methods:
        if set(runtime_coverage[method_id]["reports"]) != expected_seeds:
            raise ValueError(f"runtime coverage seed mismatch for {method_id}")

    protocol_path = _resolve(protocol)
    protocol_cfg = load_config(protocol_path)
    snapshot = protocol_snapshot(protocol_cfg)
    role = str(seed_role)
    cohort = snapshot["cohort_roles"].get(role, role)
    if cohort not in snapshot["cohorts"]:
        raise ValueError(f"unknown simulator seed role/cohort={seed_role!r}")
    simulator_seeds = [int(seed) for seed in snapshot["cohorts"][cohort]]
    if not simulator_seeds:
        raise ValueError("Stage5 simulator seed cohort is empty")
    if role == cohort:
        matching = [key for key, value in snapshot["cohort_roles"].items() if value == cohort]
        role = "stage5_confirmatory" if "stage5_confirmatory" in matching else matching[0]

    output = _resolve(output_dir)
    evaluation_budget = _stage5_evaluation_budget(
        workflow_config,
        available_seed_count=len(simulator_seeds),
    )
    comparison_requests: list[dict[str, Any]] = []
    for comparison in comparisons:
        comparison_id = comparison["comparison_id"]
        left_method = comparison["left_method_id"]
        right_method = comparison["right_method_id"]
        simulator_seed_count = (
            int(evaluation_budget["primary_simulator_seed_count"])
            if comparison["role"] == "primary_final"
            else int(evaluation_budget["secondary_simulator_seed_count"])
        )
        comparison_simulator_seeds = simulator_seeds[:simulator_seed_count]
        left_path, _ = manifests[left_method]
        right_path, _ = manifests[right_method]
        comparison_dir = output / "comparisons" / comparison_id
        request_rows: list[dict[str, Any]] = []
        for seed in seeds:
            left = row_maps[left_method][seed]
            right = row_maps[right_method][seed]
            left_cfg = config_maps[left_method][seed]
            right_cfg = config_maps[right_method][seed]
            resolved = plain(right_cfg)
            resolved.setdefault("run", {})
            run_id = f"stage5_{comparison_id}_seed_{seed}"
            resolved["run"]["run_id"] = run_id
            resolved["evaluation_protocol"] = plain(protocol_cfg.evaluation_protocol)
            resolved["evaluation_protocol"]["stage5_role"] = role
            left_name = f"{left_method}_seed_{seed}"
            right_name = f"{right_method}_seed_{seed}"
            runtime_reports: dict[str, str] = {}
            runtime_report_hashes: dict[str, str] = {}
            for group_name, method_id in ((left_name, left_method), (right_name, right_method)):
                if method_id == "wcdt_reward_v2":
                    continue
                item = runtime_coverage[method_id]["reports"][seed]
                runtime_reports[group_name] = str(item["path"])
                runtime_report_hashes[group_name] = str(item["sha256"])
            resolved["experiment"] = {
                "purpose": "formal_candidate_factorial_attribution",
                "comparison_id": comparison_id,
                "comparison_family": comparison["family"],
                "comparison_role": comparison["role"],
                "optimizer_seed": seed,
                "deployable_claim": comparison["role"] == "primary_final",
            }
            left_group = _group(name=left_name, row=left, config=left_cfg)
            right_group = _group(name=right_name, row=right, config=right_cfg)
            if bool(evaluation_budget["episode_cache_enabled"]):
                left_group["evaluation_cache"] = _episode_cache_binding(
                    output=output,
                    group=left_group,
                    row=left,
                    risk_binding=risk_binding,
                )
                right_group["evaluation_cache"] = _episode_cache_binding(
                    output=output,
                    group=right_group,
                    row=right,
                    risk_binding=risk_binding,
                )
            stage5: dict[str, Any] = {
                "paired_eval": True,
                "same_seed": True,
                "compare_shield_off_on": False,
                "replay_enabled": True,
                "episodes_per_group": len(comparison_simulator_seeds),
                "seeds": comparison_simulator_seeds,
                "risk_checkpoint": str(risk_binding["path"]),
                "require_accvp_observation_runtime_gate": True,
                "accvp_observation_preflight_reports": runtime_reports,
                "accvp_observation_preflight_report_sha256s": runtime_report_hashes,
                "statistics": {
                    "confidence": 0.95,
                    "bootstrap_replicates": 10000,
                    "bootstrap_seed": 42001,
                },
                "acceptance": {
                    "paired_policy_non_regression_v1": {
                        "max_actual_replacement_rate": 0.05,
                        "reward_tolerance": 1.0e-6,
                    }
                },
                "pairs": [
                    {
                        "name": comparison_id,
                        "left": left_name,
                        "right": right_name,
                        "acceptance_profile": "paired_policy_non_regression_v1",
                    }
                ],
                "groups": [
                    left_group,
                    right_group,
                ],
            }
            resolved["stage5"] = stage5
            config_path = comparison_dir / f"stage5_seed_{seed}.yaml"
            _write_yaml_idempotent(config_path, resolved)
            run_root = _resolve(resolved["run"]["output_root"]) / run_id
            report_path = run_root / "stage5" / "formal_paired_eval_report.json"
            request_rows.append(
                {
                    "training_seed": seed,
                    "stage5_config": str(config_path),
                    "stage5_config_sha256": file_sha256(config_path),
                    "stage5_report": str(report_path),
                    "left_group": left_name,
                    "right_group": right_name,
                    "left_method_id": left_method,
                    "right_method_id": right_method,
                    "left_checkpoint_sha256": str(left["checkpoint_sha256"]),
                    "right_checkpoint_sha256": str(right["checkpoint_sha256"]),
                    "runtime_preflight_reports": runtime_reports,
                    "runtime_preflight_report_sha256s": runtime_report_hashes,
                }
            )

        right_binding = _candidate_binding(row_maps[right_method][seeds[0]])
        reward_semantics_match = all(
            str(row_maps[left_method][seed].get("reward_semantics_hash", ""))
            == str(row_maps[right_method][seed].get("reward_semantics_hash", ""))
            for seed in seeds
        )
        request = {
            "artifact_kind": REQUEST_ARTIFACT_KIND,
            "comparison_id": comparison_id,
            "comparison_family": comparison["family"],
            "comparison_role": comparison["role"],
            "simulator_seed_count": len(comparison_simulator_seeds),
            "simulator_seeds": comparison_simulator_seeds,
            "simulator_seed_sha256": stable_hash(
                {"episode_seeds": comparison_simulator_seeds}
            ),
            "left_method_id": left_method,
            "right_method_id": right_method,
            "formal_aggregation": True,
            "minimum_training_seed_count": 5,
            "require_strict_lineage": True,
            "require_episode_cache": bool(
                evaluation_budget["episode_cache_enabled"]
            ),
            "candidate_manifest": {
                key: right_binding[key]
                for key in ("path", "sha256", "artifact_fingerprint", "artifact_variant")
            },
            "candidate_side": "right",
            "source_acceptance_key": comparison_id,
            "formal_runtime_contract_sha256": right_binding[
                "formal_runtime_contract_sha256"
            ],
            "statistics": {
                # Episode rewards only have a common scale when the two policies
                # share the exact frozen reward semantics.  Cross-reward arms
                # are compared on task/safety outcomes only.
                "continuous_metrics": ["episode_reward"] if reward_semantics_match else [],
                "binary_metrics": [
                    "proxy_collision",
                    "safety_violation",
                    "taper_miss",
                    "merge_success",
                ],
                "confidence": 0.95,
                "bootstrap_replicates": 10000,
                "bootstrap_seed": 42001,
                "require_distinct_checkpoints": True,
            },
            "baseline_replicate_manifest": str(left_path),
            "candidate_replicate_manifest": str(right_path),
            "runtime_factorial_report": str(runtime_path),
            "runtime_method_reports": {
                method_id: {
                    key: value
                    for key, value in runtime_coverage[method_id].items()
                    if key != "reports"
                }
                for method_id in {left_method, right_method}
                if method_id != "wcdt_reward_v2"
            },
            "replicates": request_rows,
        }
        request_path = _write_json_idempotent(
            comparison_dir / "replicated_request.json",
            request,
        )
        comparison_requests.append(
            {
                **comparison,
                "simulator_seed_count": len(comparison_simulator_seeds),
                "simulator_seeds": comparison_simulator_seeds,
                "simulator_seed_sha256": stable_hash(
                    {"episode_seeds": comparison_simulator_seeds}
                ),
                "generated_dir": str(comparison_dir),
                "replicated_request": str(request_path),
                "replicated_request_sha256": file_sha256(request_path),
                "aggregate_report": str(comparison_dir / "replicated_report.json"),
            }
        )

    final_rows = [
        row for row in comparison_requests if row["comparison_id"] == FINAL_COMPARISON_ID
    ]
    if len(final_rows) != 1 or final_rows[0]["right_method_id"] != EXPECTED_FINAL_METHOD_ID:
        raise ValueError("factorial request does not contain the frozen primary final comparison")
    request = {
        "artifact_kind": FACTORIAL_REQUEST_KIND,
        "schema_version": FACTORIAL_REQUEST_SCHEMA_VERSION,
        "status": "prepared",
        "protocol_id": str(factorial.get("protocol_id", "")),
        "factorial_manifest": str(factorial_path),
        "factorial_manifest_sha256": file_sha256(factorial_path),
        "factorial_manifest_fingerprint": str(factorial.get("manifest_fingerprint", "")),
        "baseline_replicate_manifest": str(baseline_path),
        "baseline_replicate_manifest_sha256": file_sha256(baseline_path),
        "runtime_factorial_report": str(runtime_path),
        "runtime_factorial_report_sha256": file_sha256(runtime_path),
        "final_method_id": EXPECTED_FINAL_METHOD_ID,
        "final_comparison_id": FINAL_COMPARISON_ID,
        "minimum_training_seed_count": 5,
        "optimizer_seeds": seeds,
        "simulator_seed_role": role,
        "simulator_seeds": simulator_seeds,
        "evaluation_budget": {
            **evaluation_budget,
            "registered_simulator_seed_count": len(simulator_seeds),
            "naive_episode_count": sum(
                2 * len(seeds) * int(row["simulator_seed_count"])
                for row in comparison_requests
            ),
            "unique_episode_count": (
                len(manifests)
                * len(seeds)
                * int(evaluation_budget["secondary_simulator_seed_count"])
                + 2
                * len(seeds)
                * (
                    int(evaluation_budget["primary_simulator_seed_count"])
                    - int(evaluation_budget["secondary_simulator_seed_count"])
                )
            ),
        },
        "risk_checkpoint": str(risk_binding["path"]),
        "risk_checkpoint_sha256": str(risk_binding["sha256"]),
        "comparisons": comparison_requests,
        "final_comparison_report": final_rows[0]["aggregate_report"],
    }
    return _write_json_idempotent(output / "factorial_request.json", request)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the frozen six-comparison Stage5 factorial request"
    )
    parser.add_argument("--baseline-manifest", required=True)
    parser.add_argument("--factorial-manifest", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--seed-role", required=True)
    parser.add_argument("--runtime-factorial-report", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workflow-config")
    args = parser.parse_args()
    path = generate(
        baseline_manifest=args.baseline_manifest,
        factorial_manifest=args.factorial_manifest,
        protocol=args.protocol,
        seed_role=args.seed_role,
        runtime_factorial_report=args.runtime_factorial_report,
        output_dir=args.output_dir,
        workflow_config=args.workflow_config,
    )
    print(f"stage5_factorial_request={path}")


if __name__ == "__main__":
    main()
