from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from safe_rl.accvp.contracts.schema import file_sha256, read_json
from safe_rl.accvp.evaluation.pilot import (
    PILOT_VALIDATION_IMPLEMENTATION_VERSION,
)
from safe_rl.accvp.evaluation.selector_capacity_v4 import (
    validate_selector4_capacity_report,
)
from safe_rl.evaluation_protocol import stable_hash
from safe_rl.pipeline.accvp_runtime_benchmark_factorial import (
    FACTORIAL_RUNTIME_REPORT_KIND,
    validate_factorial_runtime_report,
)
from safe_rl.pipeline.accvp_runtime_benchmark import (
    RUNTIME_IMPLEMENTATION_VERSION,
)
from safe_rl.pipeline.accvp_pilot_latency_smoke import (
    SMOKE_ARTIFACT_KIND,
    SMOKE_IMPLEMENTATION_VERSION,
)
from safe_rl.pipeline.audit_ppo_replicate_lineage import (
    LINEAGE_AUDIT_IMPLEMENTATION_VERSION,
)
from safe_rl.pipeline.stage5_factorial_aggregate import FACTORIAL_REPORT_KIND
from safe_rl.pipeline.stage5_generate_factorial_configs import (
    FACTORIAL_REQUEST_KIND,
    FINAL_COMPARISON_ID,
)
from safe_rl.ppo_factorial import (
    EXPECTED_CANDIDATE_METHOD_ROLES,
    EXPECTED_FINAL_METHOD_ID,
    validate_factorial_manifest,
    validate_replicate_manifest,
)
from safe_rl.utils.config import REPO_ROOT
from safe_rl.utils.io import write_json


PILOT_CONFIG = "safe_rl/config/active/accvp_vnext/pilot.yaml"
ORACLE_CONFIG = "safe_rl/config/active/accvp_vnext/oracle_regression.yaml"
FORMAL_CONFIG = "safe_rl/config/active/accvp_vnext/formal.yaml"
TRAIN_CONFIG = "safe_rl/config/active/accvp_vnext/train.yaml"
PPO_CONFIG = "safe_rl/config/active/accvp_vnext/ppo_candidate_table_full.yaml"
BASELINE_PPO_CONFIG = "safe_rl/config/baselines/wcdt/ppo_wcdt_v3_reward_v2.yaml"
MATRIX_CONFIG = "safe_rl/config/active/accvp_vnext/ppo_ablation_matrix.yaml"
PROTOCOL_CONFIG = "safe_rl/config/examples/vnext/evaluation_protocol_vnext.example.yaml"
WORKFLOW_CONFIG = (
    "safe_rl/config/active/accvp_vnext_selector4/workflow.yaml"
)
OPTIMIZER_SEEDS = [1001, 1002, 1003, 1004, 1005]
# The first 30 development seeds produced only 608 activation-window
# decisions on the frozen rule-policy workload.  Sixty remain within the
# preregistered development cohort and provide margin above the >=1000 gate.
RUNTIME_SEEDS = list(range(50001, 50061))
DEFAULT_BASELINE_MANIFEST = (
    "safe_rl_output/runs/wcdt_vnext_replicates/"
    "ppo_replicate_manifest.json"
)


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (REPO_ROOT / value).resolve()


def _complete_shards(root: Path) -> list[Path]:
    result: list[Path] = []
    for manifest_path in sorted(root.glob("*/manifests/dataset_manifest.json")):
        try:
            manifest = read_json(manifest_path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if (
            int(manifest.get("complete_roots", -1))
            == int(manifest.get("collected_roots", -2))
            and int(manifest.get("failed_branches", 1)) == 0
            and int(manifest.get("counterfactual_schema_version", -1)) == 3
        ):
            result.append(manifest_path.parents[1])
    return result


def _artifact_ok(
    path: Path,
    *,
    state_field: str | None = None,
    artifact_kind: str | None = None,
    status: str | None = None,
    gate_pass: bool = False,
) -> bool:
    if not path.is_file():
        return False
    try:
        payload = read_json(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    if state_field is not None and str(payload.get(state_field, "")).lower() not in {
        "pass",
        "go",
        "complete",
    }:
        return False
    if artifact_kind is not None and str(payload.get("artifact_kind", "")) != artifact_kind:
        return False
    if status is not None and str(payload.get("status", "")) != status:
        return False
    if gate_pass and not bool(dict(payload.get("gate", {}) or {}).get("pass", False)):
        return False
    return True


def _selector4_capacity_audit_ok(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        validate_selector4_capacity_report(read_json(path))
    except (KeyError, TypeError, ValueError):
        return False
    return True


def _oracle_report_ok(path: Path) -> bool:
    """Accept only the scoped oracle artifact required by pilot/training gates."""

    if not _artifact_ok(path, state_field="oracle_state"):
        return False
    try:
        payload = read_json(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return (
        bool(payload.get("go_for_training", False))
        and str(payload.get("root_policy", "")) == "merge_timing"
        and str(payload.get("cohort_role", "")) == "oracle_regression"
        and bool(payload.get("oracle_only", False))
        and bool(payload.get("exclude_from_model_splits", False))
        and [int(value) for value in payload.get("required_seeds", [])] == [2, 5]
    )


def _pilot_validation_ok(path: Path) -> bool:
    """Reject pre-fix selector-v4 reports that bypassed strict gates."""

    if not _artifact_ok(
        path,
        artifact_kind="accvp_pilot_validation_v2",
        state_field="pilot_state",
    ):
        return False
    try:
        payload = read_json(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    conditions = dict(payload.get("conditions", {}) or {})
    required_conditions = {
        "rejected_root_count_zero",
        "critical_actor_overflow_zero",
        "task_actor_coverage_complete",
        "risk_safety_actor_coverage_complete",
        "actor_mapping_mismatch_zero",
        "root_observation_fingerprint_mismatch_zero",
        "protected_actor_coverage_complete",
        "branch_success_rate",
        "oracle_regression",
    }
    return bool(
        str(payload.get("validation_implementation_version", ""))
        == PILOT_VALIDATION_IMPLEMENTATION_VERSION
        and bool(payload.get("strict_selector_contract", False))
        and required_conditions.issubset(conditions)
        and all(bool(conditions[name]) for name in required_conditions)
        and int(payload.get("critical_actor_overflow_count", -1)) == 0
        and int(payload.get("rejected_root_count", -1)) == 0
        and int(payload.get("coverage_incomplete_count", -1)) == 0
        and int(payload.get("actor_mapping_mismatch_count", -1)) == 0
        and int(
            payload.get(
                "root_observation_fingerprint_mismatch_count", -1
            )
        )
        == 0
        and float(payload.get("protected_actor_coverage_rate", -1.0)) == 1.0
        and float(payload.get("branch_success_rate", 0.0)) >= 0.99
    )


def _pilot_latency_smoke_ok(path: Path) -> bool:
    if not _artifact_ok(
        path,
        artifact_kind=SMOKE_ARTIFACT_KIND,
        state_field="smoke_state",
    ):
        return False
    try:
        payload = read_json(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    conditions = dict(payload.get("conditions", {}) or {})
    return bool(
        str(payload.get("implementation_version", ""))
        == SMOKE_IMPLEMENTATION_VERSION
        and str(payload.get("evidence_role", ""))
        == "diagnostic_only_pre_formal_feasibility"
        and not bool(payload.get("formal_runtime_evidence", True))
        and not bool(payload.get("hard_realtime_claim", True))
        and conditions
        and all(bool(value) for value in conditions.values())
    )


def _factorial_manifest_ok(path: Path, *, protocol_id: str) -> bool:
    if not path.is_file():
        return False
    try:
        payload = validate_factorial_manifest(
            path,
            require_complete=True,
            verify_files=True,
        )
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return (
        str(payload.get("protocol_id", "")) == str(protocol_id)
        and str(payload.get("final_method_id", "")) == EXPECTED_FINAL_METHOD_ID
        and set(dict(payload.get("methods", {}) or {}))
        == set(EXPECTED_CANDIDATE_METHOD_ROLES)
    )


def _baseline_manifest_ok(
    path: Path,
    *,
    optimizer_seeds: list[int] = OPTIMIZER_SEEDS,
) -> bool:
    if not path.is_file():
        return False
    try:
        payload = read_json(path)
        summary = validate_replicate_manifest(
            payload,
            method_id="wcdt_reward_v2",
            expected_seeds=optimizer_seeds,
            verify_files=True,
        )
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return str(payload.get("status", "")) == "complete" and summary["status"] == "complete"


def _baseline_lineage_audit_ok(
    path: Path,
    *,
    optimizer_seeds: list[int],
) -> bool:
    if not path.is_file():
        return False
    try:
        payload = read_json(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    declared_fingerprint = str(payload.get("audit_fingerprint", ""))
    fingerprint_payload = {
        key: value for key, value in payload.items() if key != "audit_fingerprint"
    }
    return bool(
        str(payload.get("artifact_kind", ""))
        == "ppo_replicate_lineage_audit_v1"
        and str(payload.get("audit_implementation_version", ""))
        == LINEAGE_AUDIT_IMPLEMENTATION_VERSION
        and str(payload.get("status", "")) == "reusable"
        and [int(value) for value in payload.get("required_seeds", [])]
        == sorted(optimizer_seeds)
        and [int(value) for value in payload.get("valid_seeds", [])]
        == sorted(optimizer_seeds)
        and not list(payload.get("missing_seeds", []) or [])
        and not list(payload.get("invalid_records", []) or [])
        and not list(payload.get("global_reasons", []) or [])
        and bool(declared_fingerprint)
        and stable_hash(fingerprint_payload) == declared_fingerprint
    )


def _factorial_runtime_ok(
    path: Path,
    *,
    factorial_manifest: Path,
    runtime_seeds: list[int] = RUNTIME_SEEDS,
) -> bool:
    if not path.is_file() or not factorial_manifest.is_file():
        return False
    try:
        payload = validate_factorial_runtime_report(
            path,
            factorial_manifest=factorial_manifest,
            seeds=runtime_seeds,
            backend="vectorized",
            device="auto",
        )
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return (
        str(payload.get("artifact_kind", "")) == FACTORIAL_RUNTIME_REPORT_KIND
        and str(payload.get("status", "")) == "complete"
        and bool(
            dict(payload.get("gate", {}) or {})
            .get("checks", {})
            .get("complete_four_method_set", False)
        )
        and bool(
            dict(payload.get("gate", {}) or {})
            .get("checks", {})
            .get("final_method_binding", False)
        )
    )


def _failed_gate_summary(payload: dict[str, Any]) -> str:
    gate = dict(payload.get("gate", {}) or {})
    failed = sorted(
        str(name)
        for name, value in dict(gate.get("checks", {}) or {}).items()
        if not bool(value)
    )
    metrics = dict(payload.get("metrics", {}) or {})
    details: list[str] = []
    for name in (
        "accvp_table_latency_p95",
        "accvp_table_latency_p99",
        "accvp_table_latency_max",
        "accvp_table_timeout_rate_activation_window",
        "accvp_table_valid_rate_activation_window",
        "accvp_table_critical_actor_overflow_count",
    ):
        if name in metrics:
            details.append(f"{name}={metrics[name]}")
    return (
        f"failed_checks={failed}"
        + (f" metrics=({', '.join(details)})" if details else "")
    )


def _scorer_runtime_failure_reason(
    path: Path,
    *,
    expected_seed_count: int,
) -> str | None:
    """Block an identical complete failed runtime request from being rerun.

    A report from an older implementation remains eligible for the benchmark's
    audited archive-and-rerun path.  A shorter prefix remains eligible only for
    the preregistered sample-size extension path.
    """

    if not path.is_file():
        return None
    try:
        payload = read_json(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    if bool(dict(payload.get("gate", {}) or {}).get("pass", False)):
        return None
    if str(payload.get("runtime_implementation_version", "")) != (
        RUNTIME_IMPLEMENTATION_VERSION
    ):
        return None
    workload = dict(payload.get("workload", {}) or {})
    if int(workload.get("requested_episode_seed_count", -1)) < int(
        expected_seed_count
    ):
        return None
    if str(workload.get("requested_episode_seed_sha256", "")) != str(
        workload.get("observed_episode_seed_sha256", "")
    ):
        return None
    return (
        "the complete scorer runtime request already produced an immutable "
        "failed report under the current implementation; change/fix the "
        "implementation and bump its version before a clean rerun. "
        + _failed_gate_summary(payload)
    )


def _factorial_runtime_failure_reason(
    path: Path,
    *,
    factorial_manifest: Path,
    runtime_seeds: list[int],
) -> str | None:
    if not path.is_file() or not factorial_manifest.is_file():
        return None
    try:
        payload = validate_factorial_runtime_report(
            path,
            factorial_manifest=factorial_manifest,
            seeds=runtime_seeds,
            backend="vectorized",
            device="auto",
        )
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    if bool(dict(payload.get("gate", {}) or {}).get("pass", False)):
        return None
    return (
        "the complete factorial policy-runtime request already produced an "
        "immutable failed report under the current implementation; inspect "
        "the per-method/per-replicate gates before changing code and bumping "
        "the runtime implementation version. "
        + _failed_gate_summary(payload)
    )


def _stage5_factorial_request_ok(
    path: Path,
    *,
    factorial_manifest: Path,
    runtime_report: Path,
    baseline_manifest: Path,
) -> bool:
    if not path.is_file():
        return False
    try:
        payload = read_json(path)
        if (
            str(payload.get("artifact_kind", "")) != FACTORIAL_REQUEST_KIND
            or str(payload.get("status", "")) != "prepared"
            or str(payload.get("final_method_id", "")) != EXPECTED_FINAL_METHOD_ID
            or str(payload.get("final_comparison_id", "")) != FINAL_COMPARISON_ID
            or str(payload.get("factorial_manifest_sha256", ""))
            != file_sha256(factorial_manifest)
            or str(payload.get("runtime_factorial_report_sha256", ""))
            != file_sha256(runtime_report)
            or str(payload.get("baseline_replicate_manifest_sha256", ""))
            != file_sha256(baseline_manifest)
        ):
            return False
        comparisons = list(payload.get("comparisons", []) or [])
        if len(comparisons) != 6:
            return False
        for row in comparisons:
            request_path = _resolve(row.get("replicated_request", ""))
            if (
                not request_path.is_file()
                or file_sha256(request_path)
                != str(row.get("replicated_request_sha256", ""))
            ):
                return False
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False
    return True


def _stage5_factorial_report_ok(
    path: Path,
    *,
    request_path: Path,
    final_child_report: Path,
) -> bool:
    if not path.is_file() or not request_path.is_file():
        return False
    try:
        payload = read_json(path)
        fingerprint = str(payload.get("report_fingerprint", ""))
        content = dict(payload)
        content.pop("report_fingerprint", None)
        gate = dict(payload.get("gate", {}) or {})
        final = dict(payload.get("final_comparison_report", {}) or {})
        if (
            str(payload.get("artifact_kind", "")) != FACTORIAL_REPORT_KIND
            or str(payload.get("status", "")) != "complete"
            or str(payload.get("final_method_id", "")) != EXPECTED_FINAL_METHOD_ID
            or str(payload.get("final_comparison_id", "")) != FINAL_COMPARISON_ID
            or str(payload.get("factorial_request_sha256", ""))
            != file_sha256(request_path)
            or not bool(gate.get("pass", False))
            or not all(bool(value) for value in gate.values())
            or len(list(payload.get("comparisons", []) or [])) != 6
            or stable_hash(content) != fingerprint
            or _resolve(final.get("path", "")) != final_child_report.resolve()
            or not final_child_report.is_file()
            or str(final.get("sha256", "")) != file_sha256(final_child_report)
        ):
            return False
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False
    return True


def _module_command(module: str, *args: Any) -> list[str]:
    return [sys.executable, "-m", module, *(str(value) for value in args)]


def _load_workflow_contract(path: str | Path) -> tuple[Path, dict[str, Any]]:
    source = _resolve(path)
    with source.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if str(payload.get("artifact_kind", "")) != "accvp_vnext_workflow_contract_v1":
        raise ValueError(f"unsupported ACCVP VNext workflow contract: {source}")
    if int(payload.get("schema_version", 0)) != 1:
        raise ValueError("unsupported ACCVP VNext workflow schema version")
    phase_order = list(payload.get("phase_order", []) or [])
    if not phase_order or len(phase_order) != len(set(phase_order)):
        raise ValueError("workflow phase_order must be a non-empty unique list")
    return source, payload


def _workflow_path_value(
    workflow: dict[str, Any],
    key: str,
    default: str | Path,
) -> str:
    return str(dict(workflow.get("paths", {}) or {}).get(key, default))


def _workflow_seed_values(
    workflow: dict[str, Any],
    key: str,
    default: list[int],
) -> list[int]:
    configured = dict(workflow.get("seeds", {}) or {}).get(key)
    if configured is None:
        return list(default)
    if isinstance(configured, dict) and {"start", "count"}.issubset(
        configured
    ):
        start = int(configured["start"])
        count = int(configured["count"])
        values = list(range(start, start + count))
    else:
        values = [int(value) for value in list(configured)]
    if not values or len(values) != len(set(values)):
        raise ValueError(f"workflow seed schedule {key!r} must be non-empty and unique")
    return values


def workflow_status(
    *,
    baseline_manifest: str | Path = DEFAULT_BASELINE_MANIFEST,
    workflow_config: str | Path = WORKFLOW_CONFIG,
) -> dict[str, Any]:
    workflow_path, workflow = _load_workflow_contract(workflow_config)
    pilot_config = _workflow_path_value(workflow, "pilot_config", PILOT_CONFIG)
    oracle_config = _workflow_path_value(workflow, "oracle_config", ORACLE_CONFIG)
    formal_config = _workflow_path_value(workflow, "formal_config", FORMAL_CONFIG)
    train_config = _workflow_path_value(workflow, "train_config", TRAIN_CONFIG)
    ppo_config = _workflow_path_value(workflow, "ppo_config", PPO_CONFIG)
    baseline_ppo_config = _workflow_path_value(
        workflow, "baseline_ppo_config", BASELINE_PPO_CONFIG
    )
    matrix_config = _workflow_path_value(workflow, "matrix_config", MATRIX_CONFIG)
    protocol_config = _workflow_path_value(
        workflow, "protocol_config", PROTOCOL_CONFIG
    )
    optimizer_seeds = _workflow_seed_values(
        workflow, "optimizer_replicates", OPTIMIZER_SEEDS
    )
    runtime_seeds = _workflow_seed_values(
        workflow, "runtime", RUNTIME_SEEDS
    )
    pilot_shard_root = _resolve(
        _workflow_path_value(
            workflow,
            "pilot_shard_root",
            "safe_rl_output/runs/accvp_vnext_pilot/stage1_counterfactual/"
            "accvp_vnext_schema3_pilot/shards",
        )
    )
    oracle_shard_root = _resolve(
        _workflow_path_value(
            workflow,
            "oracle_shard_root",
            "safe_rl_output/runs/accvp_vnext_oracle_regression/"
            "stage1_counterfactual/accvp_vnext_schema3_oracle_regression/shards",
        )
    )
    formal_shard_root = _resolve(
        _workflow_path_value(
            workflow,
            "formal_shard_root",
            "safe_rl_output/runs/accvp_vnext_formal/stage1_counterfactual/"
            "accvp_vnext_schema3_formal/shards",
        )
    )
    pilot_shards = _complete_shards(pilot_shard_root)
    oracle_shards = _complete_shards(oracle_shard_root)
    formal_shards = _complete_shards(formal_shard_root)
    pilot_dataset = _resolve(
        _workflow_path_value(
            workflow, "pilot_dataset", "safe_rl_output/runs/accvp_vnext_pilot_dataset"
        )
    )
    oracle_dataset = _resolve(
        _workflow_path_value(
            workflow,
            "oracle_dataset",
            "safe_rl_output/runs/accvp_vnext_oracle_regression_dataset",
        )
    )
    formal_dataset = _resolve(
        _workflow_path_value(
            workflow, "formal_dataset", "safe_rl_output/runs/accvp_vnext_formal_dataset"
        )
    )
    oracle_report = _resolve(
        _workflow_path_value(
            workflow,
            "oracle_report",
            "safe_rl_output/runs/accvp_vnext_oracle_regression/oracle_report.json",
        )
    )
    pilot_report = _resolve(
        _workflow_path_value(
            workflow,
            "pilot_report",
            "safe_rl_output/runs/accvp_vnext_pilot/pilot_report.json",
        )
    )
    pilot_latency_smoke_report = _resolve(
        _workflow_path_value(
            workflow,
            "pilot_latency_smoke_report",
            "safe_rl_output/runs/accvp_vnext_selector4_pilot_latency_smoke/feasibility_report.json",
        )
    )
    formal_validation_report = _resolve(
        _workflow_path_value(
            workflow,
            "formal_validation_report",
            "safe_rl_output/runs/accvp_vnext_formal/formal_validation.json",
        )
    )
    predictor_manifest = _resolve(
        _workflow_path_value(
            workflow,
            "predictor_manifest",
            "safe_rl_output/runs/accvp_vnext_train/accvp/"
            "accvp_vnext_schema3_candidate_manifest.json",
        )
    )
    scorer_report = _resolve(
        _workflow_path_value(
            workflow,
            "scorer_report",
            "safe_rl_output/runs/accvp_vnext_runtime/scorer_preflight.json",
        )
    )
    factorial_manifest = _resolve(
        _workflow_path_value(
            workflow,
            "factorial_manifest",
            "safe_rl_output/runs/accvp_vnext_factorial/ppo_factorial_manifest.json",
        )
    )
    baseline_default = _workflow_path_value(
        workflow, "baseline_manifest", DEFAULT_BASELINE_MANIFEST
    )
    baseline_path = _resolve(
        baseline_default
        if str(baseline_manifest) == DEFAULT_BASELINE_MANIFEST
        else baseline_manifest
    )
    baseline_lineage_audit = _resolve(
        _workflow_path_value(
            workflow,
            "baseline_lineage_audit",
            "safe_rl_output/runs/accvp_vnext_baseline_audit/"
            "wcdt_baseline_report.json",
        )
    )
    runtime_factorial = _resolve(
        _workflow_path_value(
            workflow,
            "runtime_factorial_report",
            "safe_rl_output/runs/accvp_vnext_runtime/factorial_runtime_report.json",
        )
    )
    final_runtime_replicates = (
        runtime_factorial.parent
        / "methods"
        / EXPECTED_FINAL_METHOD_ID
        / "replicated_runtime_report.json"
    )
    stage5_request = _resolve(
        _workflow_path_value(
            workflow,
            "stage5_request",
            "safe_rl_output/runs/accvp_vnext_stage5/generated/factorial_request.json",
        )
    )
    stage5_report = _resolve(
        _workflow_path_value(
            workflow,
            "stage5_report",
            "safe_rl_output/runs/accvp_vnext_stage5/factorial_report.json",
        )
    )
    final_stage5_report = (
        stage5_request.parent
        / "comparisons"
        / FINAL_COMPARISON_ID
        / "replicated_report.json"
    )
    holdout_report = _resolve(
        _workflow_path_value(
            workflow,
            "holdout_report",
            "safe_rl_output/runs/accvp_vnext_final_holdout/"
            "accvp_vnext_schema3_final_test_diagnostics.json",
        )
    )
    selector_audit_report = _resolve(
        _workflow_path_value(
            workflow,
            "selector_audit_report",
            "safe_rl_output/runs/accvp_vnext_selector3_audit/"
            "selector_contract_audit.json",
        )
    )

    phases: list[dict[str, Any]] = []

    def add(name: str, complete: bool, command: list[str] | None, artifact: str | Path) -> None:
        phases.append(
            {
                "name": name,
                "complete": bool(complete),
                "artifact": str(artifact),
                "command": command,
            }
        )

    declared_phase_names = {
        str(value) for value in list(workflow["phase_order"])
    }
    if "selector_contract_audit" in declared_phase_names:
        selector_audit_config = _workflow_path_value(
            workflow,
            "selector_audit_config",
            "safe_rl/config/active/accvp_vnext_selector3/selector_audit.yaml",
        )
        selector_source_dataset = _workflow_path_value(
            workflow,
            "selector_source_dataset",
            "safe_rl_output/runs/accvp_vnext_formal_dataset",
        )
        selector_source_factorial = _workflow_path_value(
            workflow,
            "selector_source_factorial_manifest",
            "safe_rl_output/runs/accvp_vnext_factorial/"
            "ppo_factorial_manifest.json",
        )
        if str(workflow.get("protocol_id", "")) in {
            "accvp-vnext-correctness-v3-selector4",
            "accvp-vnext-correctness-v4-selector4-hybrid",
        }:
            selector_cache = _workflow_path_value(
                workflow,
                "selector_replay_cache",
                "safe_rl_output/runs/accvp_vnext_selector4_audit/"
                "replay_cache",
            )
            selector_historical_overflow = _workflow_path_value(
                workflow,
                "selector_historical_overflow_report",
                "safe_rl_output/runs/accvp_vnext_selector3_runtime/"
                "diagnostics/reward_v2_commitment_seed1005_"
                "lane_aware_capacity_sweep.json",
            )
            selector_workers = int(
                workflow.get("selector_audit", {}).get("workers", 2)
            )
            add(
                "selector_contract_audit",
                _selector4_capacity_audit_ok(selector_audit_report),
                _module_command(
                    "safe_rl.pipeline.accvp_selector4_capacity_audit",
                    "--config",
                    selector_audit_config,
                    "--dataset",
                    selector_source_dataset,
                    "--factorial-manifest",
                    selector_source_factorial,
                    "--historical-overflow-report",
                    selector_historical_overflow,
                    "--cache-root",
                    selector_cache,
                    "--workers",
                    selector_workers,
                    "--output",
                    selector_audit_report,
                ),
                selector_audit_report,
            )
        else:
            selector_optimizer_seeds = _workflow_seed_values(
                workflow, "selector_optimizer_replicates", [1002, 1004]
            )
            selector_simulator_seeds = _workflow_seed_values(
                workflow, "selector_diagnostic", [50021, 50027]
            )
            add(
                "selector_contract_audit",
                _artifact_ok(
                    selector_audit_report,
                    artifact_kind="accvp_selector_contract_audit_v1",
                    state_field="audit_state",
                ),
                _module_command(
                    "safe_rl.pipeline.accvp_selector_contract_audit",
                    "--config",
                    selector_audit_config,
                    "--dataset",
                    selector_source_dataset,
                    "--factorial-manifest",
                    selector_source_factorial,
                    "--optimizer-seeds",
                    *selector_optimizer_seeds,
                    "--simulator-seeds",
                    *selector_simulator_seeds,
                    "--output",
                    selector_audit_report,
                ),
                selector_audit_report,
            )

    add(
        "pilot_collection",
        len(pilot_shards) >= 10,
        _module_command(
            "safe_rl.pipeline.stage1_collect_accvp_jobs",
            "--config",
            pilot_config,
        ),
        pilot_shard_root,
    )
    merge_pilot = _module_command(
        "safe_rl.pipeline.stage1_merge_counterfactual",
        "--config",
        pilot_config,
        *[value for shard in pilot_shards for value in ("--shard", shard)],
        "--output",
        pilot_dataset,
    )
    add(
        "pilot_merge",
        (pilot_dataset / "manifests" / "dataset_manifest.json").is_file(),
        merge_pilot if pilot_shards else None,
        pilot_dataset,
    )
    add(
        "oracle_collection",
        bool(oracle_shards),
        _module_command(
            "safe_rl.pipeline.stage1_collect_accvp_jobs",
            "--config",
            oracle_config,
        ),
        oracle_shard_root,
    )
    add(
        "oracle_merge",
        (oracle_dataset / "manifests" / "dataset_manifest.json").is_file(),
        (
            _module_command(
                "safe_rl.pipeline.stage1_merge_counterfactual",
                "--config",
                oracle_config,
                *[value for shard in oracle_shards for value in ("--shard", shard)],
                "--output",
                oracle_dataset,
            )
            if oracle_shards
            else None
        ),
        oracle_dataset,
    )
    add(
        "oracle_regression",
        _oracle_report_ok(oracle_report),
        _module_command(
            "safe_rl.pipeline.accvp_oracle_smoke",
            "--dataset",
            oracle_dataset,
            "--output",
            oracle_report,
            "--seeds",
            2,
            5,
            "--root-policy",
            "merge_timing",
            "--cohort-role",
            "oracle_regression",
        ),
        oracle_report,
    )
    add(
        "pilot_validation",
        _pilot_validation_ok(pilot_report),
        _module_command(
            "safe_rl.pipeline.stage1_validate_accvp_pilot",
            "--config",
            pilot_config,
            "--dataset",
            pilot_dataset,
            "--oracle-report",
            oracle_report,
            "--output",
            pilot_report,
        ),
        pilot_report,
    )
    if "pilot_latency_feasibility_smoke" in declared_phase_names:
        pilot_latency_train_config = _workflow_path_value(
            workflow,
            "pilot_latency_smoke_train_config",
            "safe_rl/config/active/accvp_vnext_selector4/pilot_latency_smoke_train.yaml",
        )
        pilot_latency_runtime_config = _workflow_path_value(
            workflow,
            "pilot_latency_smoke_runtime_config",
            "safe_rl/config/active/accvp_vnext_selector4/pilot_latency_smoke_runtime.yaml",
        )
        pilot_latency_seeds = _workflow_seed_values(
            workflow,
            "pilot_latency_smoke_development",
            [66001, 66002, 66003, 66004, 66005],
        )
        add(
            "pilot_latency_feasibility_smoke",
            _pilot_latency_smoke_ok(pilot_latency_smoke_report),
            _module_command(
                "safe_rl.pipeline.accvp_pilot_latency_smoke",
                "--train-config",
                pilot_latency_train_config,
                "--runtime-config",
                pilot_latency_runtime_config,
                "--seeds",
                *pilot_latency_seeds,
                "--output",
                pilot_latency_smoke_report,
            ),
            pilot_latency_smoke_report,
        )
    add(
        "formal_collection",
        len(formal_shards) >= 50,
        _module_command(
            "safe_rl.pipeline.stage1_collect_accvp_jobs",
            "--config",
            formal_config,
        ),
        formal_shard_root,
    )
    add(
        "formal_merge",
        (formal_dataset / "manifests" / "dataset_manifest.json").is_file(),
        (
            _module_command(
                "safe_rl.pipeline.stage1_merge_counterfactual",
                "--config",
                formal_config,
                *[value for shard in formal_shards for value in ("--shard", shard)],
                "--output",
                formal_dataset,
            )
            if formal_shards
            else None
        ),
        formal_dataset,
    )
    if "formal_validation" in declared_phase_names:
        add(
            "formal_validation",
            _artifact_ok(
                formal_validation_report,
                artifact_kind="accvp_selector3_formal_validation_v1",
                state_field="formal_state",
            ),
            _module_command(
                "safe_rl.pipeline.stage1_validate_accvp_formal",
                "--config",
                formal_config,
                "--dataset",
                formal_dataset,
                "--output",
                formal_validation_report,
            ),
            formal_validation_report,
        )
    add(
        "accvp_training",
        predictor_manifest.is_file(),
        _module_command(
            "safe_rl.pipeline.stage2_train_accvp",
            "--config",
            train_config,
        ),
        predictor_manifest,
    )
    add(
        "scorer_runtime_preflight",
        _artifact_ok(
            scorer_report,
            artifact_kind="accvp_runtime_benchmark_v1",
            gate_pass=True,
        ),
        _module_command(
            "safe_rl.pipeline.accvp_runtime_benchmark",
            "--config",
            ppo_config,
            "--policy-type",
            "rule_gap_acceptance",
            "--seeds",
            *runtime_seeds,
            "--backend",
            "vectorized",
            "--extend-failed-report",
            "--output",
            scorer_report,
        ),
        scorer_report,
    )
    add(
        "candidate_ppo_replicates",
        _factorial_manifest_ok(
            factorial_manifest,
            protocol_id=str(workflow.get("protocol_id", "")),
        ),
        _module_command(
            "safe_rl.pipeline.stage3_train_ppo_factorial",
            "--config",
            ppo_config,
            "--matrix",
            matrix_config,
            "--workflow-config",
            workflow_path,
            "--optimizer-seeds",
            *optimizer_seeds,
            "--output-root",
            factorial_manifest.parent,
        ),
        factorial_manifest,
    )
    add(
        "baseline_ppo_replicates",
        _baseline_manifest_ok(
            baseline_path,
            optimizer_seeds=optimizer_seeds,
        ),
        _module_command(
            "safe_rl.pipeline.stage3_train_ppo_replicates",
            "--config",
            baseline_ppo_config,
            "--matrix",
            matrix_config,
            "--method-id",
            "wcdt_reward_v2",
            "--optimizer-seeds",
            *optimizer_seeds,
            "--run-id-prefix",
            "ppo_wcdt_vnext",
            "--output-root",
            baseline_path.parent,
        ),
        baseline_path,
    )
    if "baseline_lineage_audit" in declared_phase_names:
        add(
            "baseline_lineage_audit",
            _baseline_lineage_audit_ok(
                baseline_lineage_audit,
                optimizer_seeds=optimizer_seeds,
            ),
            _module_command(
                "safe_rl.pipeline.audit_ppo_replicate_lineage",
                "--replicate-manifest",
                baseline_path,
                "--method-config",
                baseline_ppo_config,
                "--required-seeds",
                *optimizer_seeds,
                "--output",
                baseline_lineage_audit,
            ),
            baseline_lineage_audit,
        )
    add(
        "policy_runtime_replicates",
        _factorial_runtime_ok(
            runtime_factorial,
            factorial_manifest=factorial_manifest,
            runtime_seeds=runtime_seeds,
        ),
        _module_command(
            "safe_rl.pipeline.accvp_runtime_benchmark_factorial",
            "--factorial-manifest",
            factorial_manifest,
            "--seeds",
            *runtime_seeds,
            "--backend",
            "vectorized",
            "--output",
            runtime_factorial,
        ),
        runtime_factorial,
    )
    add(
        "stage5_generate",
        _stage5_factorial_request_ok(
            stage5_request,
            factorial_manifest=factorial_manifest,
            runtime_report=runtime_factorial,
            baseline_manifest=baseline_path,
        ),
        _module_command(
            "safe_rl.pipeline.stage5_generate_factorial_configs",
            "--baseline-manifest",
            baseline_path,
            "--factorial-manifest",
            factorial_manifest,
            "--protocol",
            protocol_config,
            "--seed-role",
            "natural_confirmatory",
            "--runtime-factorial-report",
            runtime_factorial,
            "--output-dir",
            stage5_request.parent,
            "--workflow-config",
            workflow_path,
        ),
        stage5_request,
    )
    add(
        "stage5_replicates_and_aggregate",
        _stage5_factorial_report_ok(
            stage5_report,
            request_path=stage5_request,
            final_child_report=final_stage5_report,
        ),
        _module_command(
            "safe_rl.pipeline.stage5_run_factorial",
            "--request",
            stage5_request,
            "--output",
            stage5_report,
        ),
        stage5_report,
    )
    add(
        "one_shot_final_holdout",
        holdout_report.is_file(),
        _module_command(
            "safe_rl.pipeline.accvp_final_holdout_eval",
            "--config",
            train_config,
            "--artifact-manifest",
            predictor_manifest,
            "--runtime-benchmark",
            final_runtime_replicates,
            "--stage5-replicated-report",
            final_stage5_report,
            "--output-dir",
            holdout_report.parent,
            "--mode",
            "full",
        ),
        holdout_report,
    )
    declared_order = [str(value) for value in workflow["phase_order"]]
    by_name = {str(phase["name"]): phase for phase in phases}
    if set(declared_order) != set(by_name):
        raise ValueError(
            "workflow contract phase_order disagrees with implemented phases: "
            f"missing={sorted(set(by_name).difference(declared_order))} "
            f"unknown={sorted(set(declared_order).difference(by_name))}"
        )
    phases = [by_name[name] for name in declared_order]
    first_incomplete = next((phase for phase in phases if not phase["complete"]), None)
    blocked_phases = dict(workflow.get("automation", {}).get("blocked_phases", {}) or {})
    blocked_reason = (
        None
        if first_incomplete is None
        else blocked_phases.get(str(first_incomplete["name"]))
    )
    if first_incomplete is not None and blocked_reason is None:
        if str(first_incomplete["name"]) == "scorer_runtime_preflight":
            blocked_reason = _scorer_runtime_failure_reason(
                scorer_report,
                expected_seed_count=len(runtime_seeds),
            )
        elif str(first_incomplete["name"]) == "policy_runtime_replicates":
            blocked_reason = _factorial_runtime_failure_reason(
                runtime_factorial,
                factorial_manifest=factorial_manifest,
                runtime_seeds=runtime_seeds,
            )
    return {
        "artifact_kind": "accvp_vnext_pipeline_status_v1",
        "schema_version": 1,
        "complete": first_incomplete is None,
        "next_phase": None if first_incomplete is None else first_incomplete["name"],
        "next_command": (
            None
            if first_incomplete is None or blocked_reason is not None
            else first_incomplete["command"]
        ),
        "blocked": blocked_reason is not None,
        "blocked_reason": blocked_reason,
        "workflow_contract": {
            "path": str(workflow_path),
            "sha256": file_sha256(workflow_path),
            "protocol_id": str(workflow.get("protocol_id", "")),
            "final_method_id": str(workflow.get("final_method_id", "")),
        },
        "phases": phases,
    }


def _execute_current_phase(
    status: dict[str, Any],
    *,
    allow_final_holdout: bool,
) -> None:
    if bool(status.get("blocked", False)):
        raise RuntimeError(
            f"phase {status.get('next_phase')!r} is blocked by the workflow contract: "
            f"{status.get('blocked_reason')}"
        )
    if status["next_phase"] == "one_shot_final_holdout" and not allow_final_holdout:
        raise RuntimeError(
            "final holdout is sealed and cannot be opened without --allow-final-holdout"
        )
    command = status.get("next_command")
    if not command:
        raise RuntimeError(
            f"phase {status['next_phase']!r} requires explicit manual input"
        )
    print(f"[accvp_vnext_pipeline] executing phase={status['next_phase']}")
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fail-closed ACCVP VNext workflow coordinator; executes at most one gated phase per invocation"
    )
    parser.add_argument("--baseline-manifest", default="safe_rl_output/runs/wcdt_vnext_replicates/ppo_replicate_manifest.json")
    parser.add_argument("--workflow-config", default=WORKFLOW_CONFIG)
    parser.add_argument("--status-output")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute-next", action="store_true")
    mode.add_argument(
        "--run-until",
        help="Continuously execute gated phases through and including this phase.",
    )
    parser.add_argument("--allow-final-holdout", action="store_true")
    args = parser.parse_args()
    status = workflow_status(
        baseline_manifest=args.baseline_manifest,
        workflow_config=args.workflow_config,
    )
    if args.status_output:
        write_json(_resolve(args.status_output), status)
    print(json.dumps(status, indent=2, ensure_ascii=False))
    if not args.execute_next and not args.run_until:
        return
    if status["complete"]:
        return
    if args.execute_next:
        _execute_current_phase(status, allow_final_holdout=args.allow_final_holdout)
        return

    _workflow_path, workflow = _load_workflow_contract(args.workflow_config)
    phase_order = [str(value) for value in workflow["phase_order"]]
    target = str(args.run_until)
    if target not in phase_order:
        raise ValueError(
            f"unknown --run-until phase={target!r}; available={phase_order}"
        )
    target_index = phase_order.index(target)
    while True:
        status = workflow_status(
            baseline_manifest=args.baseline_manifest,
            workflow_config=args.workflow_config,
        )
        if status["complete"]:
            return
        next_phase = str(status["next_phase"])
        next_index = phase_order.index(next_phase)
        if next_index > target_index:
            print(f"[accvp_vnext_pipeline] reached target phase={target}")
            return
        previous_phase = next_phase
        _execute_current_phase(status, allow_final_holdout=args.allow_final_holdout)
        updated = workflow_status(
            baseline_manifest=args.baseline_manifest,
            workflow_config=args.workflow_config,
        )
        if str(updated.get("next_phase")) == previous_phase:
            detail = (
                f": {updated.get('blocked_reason')}"
                if updated.get("blocked_reason")
                else ""
            )
            raise RuntimeError(
                f"phase {previous_phase!r} command returned but its artifact "
                f"gate remains closed{detail}"
            )
        if previous_phase == target:
            print(f"[accvp_vnext_pipeline] reached target phase={target}")
            return


if __name__ == "__main__":
    main()
