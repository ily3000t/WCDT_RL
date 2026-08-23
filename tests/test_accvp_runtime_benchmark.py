from __future__ import annotations

from pathlib import Path

import pytest

from safe_rl.accvp.contracts.schema import stable_hash
from safe_rl.pipeline.accvp_runtime_benchmark import (
    _method_effect_admission_gate,
    _software_hardware,
    _validate_failed_report_extension,
    run,
)
from safe_rl.accvp.contracts.runtime_contract import (
    SIMULATION_BLOCKING_EXACT_CONTRACT,
)


def _admission_metrics() -> dict:
    return {
        "accvp_table_unique_episode_seed_count": 60,
        "accvp_table_missing_episode_seed_count": 0,
        "accvp_table_seed_schedule_match": True,
        "accvp_table_activation_window_decision_count": 1204,
        "accvp_table_model_error_count": 0,
        "accvp_table_invalid_bundle_count": 0,
        "accvp_table_invalid_output_count": 0,
        "accvp_table_runtime_context_error_count": 0,
        "accvp_table_critical_actor_overflow_count": 0,
        "accvp_table_task_actor_overflow_count": 0,
        "accvp_table_risk_safety_actor_coverage_incomplete_count": 0,
        "accvp_table_unexpected_value_error_count": 0,
        "accvp_table_warmup_error_count": 0,
        "accvp_table_warmup_ready_rate": 1.0,
        # These deployment-only misses must remain visible but non-blocking for
        # a simulation-blocking-exact method-effect run.
        "accvp_table_hard_fail_closed_count": 1,
        "accvp_table_latency_max": 0.6218245,
        "accvp_table_timeout_rate_activation_window": 0.0017,
    }


def test_method_effect_admission_separates_soft_runtime_outliers_from_integrity() -> None:
    gate = _method_effect_admission_gate(
        _admission_metrics(),
        runtime_contract_check={"pass": True},
        requested_execution_contract={
            "execution_contract": SIMULATION_BLOCKING_EXACT_CONTRACT,
        },
    )
    assert gate["pass"] is True
    assert "latency_max_within_0_50s" in gate["excluded_deployment_only_checks"]

    metrics = _admission_metrics()
    metrics["accvp_table_model_error_count"] = 1
    failed = _method_effect_admission_gate(
        metrics,
        runtime_contract_check={"pass": True},
        requested_execution_contract={
            "execution_contract": SIMULATION_BLOCKING_EXACT_CONTRACT,
        },
    )
    assert failed["pass"] is False
    assert failed["checks"]["model_error_count_zero"] is False


def test_runtime_benchmark_rejects_nonformal_seed_schedule_before_loading_artifacts(tmp_path: Path):
    with pytest.raises(ValueError, match="30 distinct"):
        run(
            config_path=tmp_path / "missing.yaml",
            policy_model=tmp_path / "missing.zip",
            seeds=list(range(29)),
            output=tmp_path / "report.json",
        )
    with pytest.raises(ValueError, match="30 distinct"):
        run(
            config_path=tmp_path / "missing.yaml",
            policy_model=tmp_path / "missing.zip",
            seeds=[*range(29), 28],
            output=tmp_path / "report.json",
        )


def test_runtime_benchmark_rejects_unknown_backend_and_records_environment(tmp_path: Path):
    with pytest.raises(ValueError, match="reference or vectorized"):
        run(
            config_path=tmp_path / "missing.yaml",
            policy_model=tmp_path / "missing.zip",
            seeds=list(range(30)),
            output=tmp_path / "report.json",
            backend="unknown",
        )
    environment = _software_hardware()
    assert environment["python_version"]
    assert environment["numpy_version"]
    assert environment["cpu_count"] is None or environment["cpu_count"] > 0


def test_runtime_benchmark_separates_scorer_preflight_from_policy_runtime(tmp_path: Path):
    with pytest.raises(ValueError, match="requires --policy-model"):
        run(
            config_path=tmp_path / "missing.yaml",
            policy_model=None,
            policy_type="sb3_ppo",
            seeds=list(range(30)),
            output=tmp_path / "report.json",
        )
    with pytest.raises(ValueError, match="must not receive a policy model"):
        run(
            config_path=tmp_path / "missing.yaml",
            policy_model=tmp_path / "unused.zip",
            policy_type="rule_gap_acceptance",
            seeds=list(range(30)),
            output=tmp_path / "report.json",
        )
    with pytest.raises(ValueError, match="30 distinct"):
        run(
            config_path=tmp_path / "missing.yaml",
            policy_model=None,
            policy_type="rule_gap_acceptance",
            seeds=list(range(29)),
            output=tmp_path / "report.json",
        )


def _failed_extension_fixture(seeds: list[int], identity: dict) -> dict:
    reports = [
        {"seed": seed, "episode_reward": float(seed)}
        for seed in seeds
    ]
    seed_hash = stable_hash({"episode_seeds": seeds})
    return {
        "artifact_kind": "accvp_runtime_benchmark_v1",
        "gate": {"pass": False},
        **identity,
        "workload": {
            "requested_episode_seed_sha256": seed_hash,
            "observed_episode_seed_sha256": seed_hash,
        },
        "episodes": reports,
    }


def test_runtime_benchmark_extension_reuses_only_an_exact_failed_prefix():
    identity = {
        "benchmark_scope": "scorer_preflight",
        "policy_type": "rule_gap_acceptance",
        "backend": "vectorized",
        "config_file_sha256": "config",
        "artifact_lineage": {"bundle": "same"},
        "software_hardware": {"platform": "same", "process_id": 1},
    }
    existing_seeds = list(range(50001, 50031))
    requested = list(range(50001, 50061))
    payload = _failed_extension_fixture(existing_seeds, identity)

    reports = _validate_failed_report_extension(
        payload,
        requested_seeds=requested,
        expected_identity=identity,
    )
    assert [report["seed"] for report in reports] == existing_seeds

    passed = {**payload, "gate": {"pass": True}}
    with pytest.raises(ValueError, match="passing.*immutable"):
        _validate_failed_report_extension(
            passed,
            requested_seeds=requested,
            expected_identity=identity,
        )

    changed = {**payload, "config_file_sha256": "changed"}
    with pytest.raises(ValueError, match="lineage mismatch"):
        _validate_failed_report_extension(
            changed,
            requested_seeds=requested,
            expected_identity=identity,
        )

    new_process = {
        **payload,
        "software_hardware": {"platform": "same", "process_id": 999},
    }
    reports = _validate_failed_report_extension(
        new_process,
        requested_seeds=requested,
        expected_identity={
            **identity,
            "software_hardware": {"platform": "same", "process_id": 2},
        },
    )
    assert len(reports) == 30

    changed_host = {
        **payload,
        "software_hardware": {"platform": "different", "process_id": 1},
    }
    with pytest.raises(ValueError, match="software_hardware"):
        _validate_failed_report_extension(
            changed_host,
            requested_seeds=requested,
            expected_identity=identity,
        )

    with pytest.raises(ValueError, match="exact prefix"):
        _validate_failed_report_extension(
            payload,
            requested_seeds=[*range(60001, 60031), *range(50031, 50061)],
            expected_identity=identity,
        )
