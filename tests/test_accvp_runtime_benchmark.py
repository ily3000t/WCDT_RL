from __future__ import annotations

from pathlib import Path

import pytest

from safe_rl.pipeline.accvp_runtime_benchmark import _software_hardware, run


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
