from __future__ import annotations

import json
from pathlib import Path

from safe_rl.accvp.contracts.schema import stable_hash
from safe_rl.pipeline.run_accvp_vnext_pipeline import (
    WORKFLOW_CONFIG,
    _load_workflow_contract,
    _oracle_report_ok,
    _pilot_latency_smoke_ok,
    _pilot_validation_ok,
    _scorer_preflight_ok,
    _scorer_runtime_failure_reason,
)
from safe_rl.pipeline.accvp_runtime_benchmark import (
    METHOD_EFFECT_ADMISSION_PROFILE,
    RUNTIME_IMPLEMENTATION_VERSION,
)
from safe_rl.accvp.contracts.runtime_contract import (
    SIMULATION_BLOCKING_EXACT_CONTRACT,
)
from safe_rl.accvp.evaluation.pilot import PILOT_VALIDATION_IMPLEMENTATION_VERSION
from safe_rl.pipeline.accvp_pilot_latency_smoke import (
    SMOKE_ARTIFACT_KIND,
    SMOKE_IMPLEMENTATION_VERSION,
)
from safe_rl.ppo_factorial import (
    EXPECTED_CANDIDATE_METHOD_ROLES,
    EXPECTED_FINAL_METHOD_ID,
)


def _write_oracle(path: Path, **overrides: object) -> None:
    payload: dict[str, object] = {
        "oracle_state": "go",
        "go_for_training": True,
        "root_policy": "merge_timing",
        "cohort_role": "oracle_regression",
        "oracle_only": True,
        "exclude_from_model_splits": True,
        "required_seeds": [2, 5],
    }
    payload.update(overrides)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_oracle_pipeline_gate_requires_merge_timing_scope(tmp_path: Path) -> None:
    report = tmp_path / "oracle_report.json"
    _write_oracle(report)
    assert _oracle_report_ok(report)

    _write_oracle(report, root_policy=None)
    assert not _oracle_report_ok(report)


def test_oracle_pipeline_gate_requires_split_exclusion_and_fixed_cohort(tmp_path: Path) -> None:
    report = tmp_path / "oracle_report.json"
    _write_oracle(report, exclude_from_model_splits=False)
    assert not _oracle_report_ok(report)


def test_pilot_gate_requires_every_strict_selector_condition(tmp_path: Path) -> None:
    report = tmp_path / "pilot_report.json"
    conditions = {
        "rejected_root_count_zero": True,
        "critical_actor_overflow_zero": True,
        "task_actor_coverage_complete": True,
        "risk_safety_actor_coverage_complete": True,
        "actor_mapping_mismatch_zero": True,
        "root_observation_fingerprint_mismatch_zero": True,
        "protected_actor_coverage_complete": True,
        "branch_success_rate": True,
        "oracle_regression": True,
    }
    payload = {
        "artifact_kind": "accvp_pilot_validation_v2",
        "pilot_state": "pass",
        "validation_implementation_version": PILOT_VALIDATION_IMPLEMENTATION_VERSION,
        "strict_selector_contract": True,
        "conditions": conditions,
        "critical_actor_overflow_count": 0,
        "rejected_root_count": 0,
        "coverage_incomplete_count": 0,
        "actor_mapping_mismatch_count": 0,
        "root_observation_fingerprint_mismatch_count": 0,
        "protected_actor_coverage_rate": 1.0,
        "branch_success_rate": 0.99,
    }
    report.write_text(json.dumps(payload), encoding="utf-8")
    assert _pilot_validation_ok(report)

    payload["conditions"] = {**conditions, "actor_mapping_mismatch_zero": False}
    report.write_text(json.dumps(payload), encoding="utf-8")
    assert not _pilot_validation_ok(report)


def test_pilot_latency_smoke_is_diagnostic_and_all_conditions_must_pass(
    tmp_path: Path,
) -> None:
    report = tmp_path / "smoke.json"
    payload = {
        "artifact_kind": SMOKE_ARTIFACT_KIND,
        "implementation_version": SMOKE_IMPLEMENTATION_VERSION,
        "smoke_state": "pass",
        "evidence_role": "diagnostic_only_pre_formal_feasibility",
        "formal_runtime_evidence": False,
        "hard_realtime_claim": False,
        "conditions": {"latency": True, "overflow_zero": True},
    }
    report.write_text(json.dumps(payload), encoding="utf-8")
    assert _pilot_latency_smoke_ok(report)

    payload["formal_runtime_evidence"] = True
    report.write_text(json.dumps(payload), encoding="utf-8")
    assert not _pilot_latency_smoke_ok(report)

    _write_oracle(report, required_seeds=[2])
    assert not _oracle_report_ok(report)


def test_vnext_workflow_unblocks_and_preregisters_factorial_pipeline() -> None:
    _path, workflow = _load_workflow_contract(WORKFLOW_CONFIG)
    assert not dict(workflow["automation"].get("blocked_phases", {}) or {})
    assert workflow["final_method_id"] == EXPECTED_FINAL_METHOD_ID
    assert {
        method_id: workflow["method_roles"][method_id]
        for method_id in EXPECTED_CANDIDATE_METHOD_ROLES
    } == EXPECTED_CANDIDATE_METHOD_ROLES
    factorial = workflow["factorial"]
    assert factorial["runtime_scope"] == "all_candidate_methods"
    assert factorial["candidate_method_ids"] == list(EXPECTED_CANDIDATE_METHOD_ROLES)
    comparisons = list(factorial["comparisons"])
    assert len(comparisons) == 6
    final = [
        row
        for row in comparisons
        if row["comparison_id"] == factorial["final_comparison_id"]
    ]
    assert len(final) == 1
    assert final[0]["right_method_id"] == EXPECTED_FINAL_METHOD_ID


def test_complete_same_implementation_runtime_failure_blocks_identical_retry(
    tmp_path: Path,
) -> None:
    report = tmp_path / "runtime.json"
    report.write_text(
        json.dumps(
            {
                "runtime_implementation_version": RUNTIME_IMPLEMENTATION_VERSION,
                "workload": {
                    "requested_episode_seed_count": 60,
                    "requested_episode_seed_sha256": "same",
                    "observed_episode_seed_sha256": "same",
                },
                "metrics": {"accvp_table_latency_p95": 0.301},
                "gate": {
                    "pass": False,
                    "checks": {"latency_p95_within_0_30s": False},
                },
                "method_effect_admission_gate": {
                    "pass": False,
                    "checks": {"model_error_count_zero": False},
                },
            }
        ),
        encoding="utf-8",
    )
    reason = _scorer_runtime_failure_reason(report, expected_seed_count=60)
    assert reason is not None
    assert "model_error_count_zero" in reason


def test_deployment_gate_failure_does_not_block_blocking_exact_admission(
    tmp_path: Path,
) -> None:
    seeds = list(range(56001, 56061))
    seed_hash = stable_hash({"episode_seeds": seeds})
    payload = {
        "artifact_kind": "accvp_runtime_benchmark_v1",
        "status": "complete",
        "runtime_implementation_version": RUNTIME_IMPLEMENTATION_VERSION,
        "benchmark_scope": "scorer_preflight",
        "policy_type": "rule_gap_acceptance",
        "backend": "vectorized",
        "conclusion_scope": "deployment_runtime_only",
        "hard_realtime_claim": False,
        "policy_method_effect_execution_contract": {
            "execution_contract": SIMULATION_BLOCKING_EXACT_CONTRACT,
        },
        "workload": {
            "requested_episode_seed_count": len(seeds),
            "requested_episode_seed_sha256": seed_hash,
            "observed_episode_seed_sha256": seed_hash,
        },
        "gate": {
            "pass": False,
            "checks": {"latency_max_within_0_50s": False},
        },
        "method_effect_admission_gate": {
            "profile": METHOD_EFFECT_ADMISSION_PROFILE,
            "pass": True,
            "checks": {"model_error_count_zero": True},
        },
    }
    payload["report_fingerprint"] = stable_hash(payload)
    report = tmp_path / "runtime.json"
    report.write_text(json.dumps(payload), encoding="utf-8")

    assert _scorer_preflight_ok(report, runtime_seeds=seeds)
    assert _scorer_runtime_failure_reason(
        report,
        expected_seed_count=len(seeds),
    ) is None


def test_old_or_incomplete_runtime_failure_remains_eligible_for_audited_retry(
    tmp_path: Path,
) -> None:
    report = tmp_path / "runtime.json"
    payload = {
        "runtime_implementation_version": "older_implementation",
        "workload": {
            "requested_episode_seed_count": 60,
            "requested_episode_seed_sha256": "same",
            "observed_episode_seed_sha256": "same",
        },
        "gate": {"pass": False, "checks": {"latency": False}},
    }
    report.write_text(json.dumps(payload), encoding="utf-8")
    assert _scorer_runtime_failure_reason(report, expected_seed_count=60) is None

    payload["runtime_implementation_version"] = RUNTIME_IMPLEMENTATION_VERSION
    payload["workload"]["requested_episode_seed_count"] = 30
    report.write_text(json.dumps(payload), encoding="utf-8")
    assert _scorer_runtime_failure_reason(report, expected_seed_count=60) is None
