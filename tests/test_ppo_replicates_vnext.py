from __future__ import annotations

import json
from pathlib import Path

import pytest

from safe_rl.accvp.schema import file_sha256
from safe_rl.evaluation_protocol import normalise_seed_cohorts
from safe_rl.pipeline.accvp_runtime_benchmark_replicates import aggregate_runtime_reports
from safe_rl.pipeline.audit_ppo_replicate_lineage import audit_manifest
from safe_rl.pipeline.stage3_train_ppo_replicates import build_replicate_plan
from safe_rl.ppo_replicates import optimizer_seed, validate_reward_semantics
from safe_rl.rl.ppo import _build_ppo_worker_env, _ppo_optimizer_seed
from safe_rl.utils.config import clone_with_overrides, load_config


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = ROOT / "safe_rl/config/active/accvp_vnext/ppo_candidate_table_full.yaml"
BASELINE = ROOT / "safe_rl/config/baselines/wcdt/ppo_wcdt_v3_reward_v2.yaml"
MATRIX = ROOT / "safe_rl/config/active/accvp_vnext/ppo_ablation_matrix.yaml"


def test_vnext_reward_semantics_are_explicit_and_optimizer_seed_is_independent():
    cfg = load_config(CANDIDATE)
    semantics = validate_reward_semantics(cfg)
    assert semantics["payload"]["reward_version"] == "opportunity_window_v2"
    assert semantics["payload"]["policy_lateral_commitment"]["enabled"] is False
    assert optimizer_seed(cfg) == 1001
    assert int(cfg.run.seed) == 20001
    assert _ppo_optimizer_seed(cfg) == 1001


def test_ppo_worker_keeps_simulator_seed_independent_from_optimizer(monkeypatch):
    cfg = load_config(CANDIDATE)
    captured = {}

    def fake_make_env(config, *, seed, **kwargs):
        captured["seed"] = seed
        return object()

    monkeypatch.setattr("safe_rl.pipeline.common.make_env", fake_make_env)
    _build_ppo_worker_env(cfg, rank=0, num_envs=1)
    assert captured["seed"] == 20001
    assert captured["seed"] != _ppo_optimizer_seed(cfg)


def test_reward_semantics_mismatch_fails_before_training():
    cfg = clone_with_overrides(
        load_config(CANDIDATE),
        {"rl": {"training_semantics_version": "reward_v3_1_persistence_001"}},
    )
    with pytest.raises(ValueError, match="disagrees with executed reward_version"):
        validate_reward_semantics(cfg)


def test_replicate_plan_changes_only_optimizer_seed(tmp_path: Path):
    manifest, configs = build_replicate_plan(
        config_path=BASELINE,
        matrix_path=MATRIX,
        method_id="wcdt_reward_v2",
        optimizer_seeds=[1001, 1002, 1003, 1004, 1005],
        output_root=tmp_path,
        require_artifacts=False,
    )
    assert manifest["simulator_training_start_seed"] == 20001
    assert [record["optimizer_seed"] for record in manifest["records"]] == [
        1001,
        1002,
        1003,
        1004,
        1005,
    ]
    assert {config["run"]["seed"] for _path, config in configs} == {20001}
    assert len({config["run"]["run_id"] for _path, config in configs}) == 5


def test_seed_ledger_values_alias_is_not_silently_empty():
    cohorts = normalise_seed_cohorts(
        {"cohorts": {"optimizer": {"values": [1001, 1002, 1003]}}}
    )
    assert cohorts["optimizer"] == [1001, 1002, 1003]
    with pytest.raises(ValueError, match="either 'seeds'.*'values'"):
        normalise_seed_cohorts(
            {"cohorts": {"bad": {"seeds": [1], "values": [2]}}}
        )


def _runtime_report(valid: float, timeout: float, stale: float, latency: float) -> dict:
    return {
        "gate": {"pass": True},
        "metrics": {
            "accvp_table_valid_rate_activation_window": valid,
            "accvp_table_timeout_rate_activation_window": timeout,
            "accvp_table_last_valid_fallback_rate_activation_window": stale,
            "accvp_table_max_consecutive_timeout_count": 1,
            "accvp_table_latency_p95": latency,
            "accvp_table_latency_p99": latency + 0.05,
            "accvp_table_latency_max": latency + 0.10,
            "accvp_table_latency_per_stage": {"risk_secondary": {"p95": 0.10}},
        },
    }


def test_runtime_replicate_gate_uses_worst_member_not_average():
    reports = [_runtime_report(0.999, 0.001, 0.001, 0.20) for _ in range(4)]
    reports.append(_runtime_report(0.994, 0.006, 0.006, 0.31))
    aggregate = aggregate_runtime_reports(reports)
    assert aggregate["worst_case"]["min_fresh_valid_rate_activation_window"] == 0.994
    assert aggregate["worst_case"]["max_timeout_rate_activation_window"] == 0.006
    assert aggregate["pass"] is False


def test_lineage_audit_can_reuse_valid_subset_and_reports_missing_seeds(tmp_path: Path):
    records = []
    for seed in (1001, 1002, 1003):
        checkpoint = tmp_path / f"checkpoint_{seed}.zip"
        config = tmp_path / f"config_{seed}.yaml"
        report = tmp_path / f"report_{seed}.json"
        checkpoint.write_bytes(f"checkpoint-{seed}".encode())
        config.write_text(f"optimizer_seed: {seed}\n", encoding="utf-8")
        report.write_text("{}\n", encoding="utf-8")
        records.append(
            {
                "method_id": "baseline",
                "training_seed": seed,
                "optimizer_seed": seed,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": file_sha256(checkpoint),
                "resolved_config": str(config),
                "resolved_config_sha256": file_sha256(config),
                "stage3_report": str(report),
                "stage3_report_sha256": file_sha256(report),
                "training_budget": {"total_timesteps": 100000},
                "reward_semantics_hash": "a" * 64,
                "observation_contract_hash": "b" * 64,
            }
        )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "artifact_kind": "ppo_optimizer_replicate_manifest_v1",
                "schema_version": 1,
                "status": "complete",
                "method_id": "baseline",
                "records": records,
            }
        ),
        encoding="utf-8",
    )
    audit = audit_manifest(
        manifest,
        required_seeds=[1001, 1002, 1003, 1004, 1005],
    )
    assert audit["status"] == "partially_reusable"
    assert audit["valid_seeds"] == [1001, 1002, 1003]
    assert audit["missing_seeds"] == [1004, 1005]
