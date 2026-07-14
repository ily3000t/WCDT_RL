from __future__ import annotations

from pathlib import Path

import pytest

from safe_rl.accvp.contracts.schema import stable_hash
from safe_rl.pipeline.accvp_runtime_benchmark import (
    _software_hardware,
    _validate_failed_report_extension,
    run,
)


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
