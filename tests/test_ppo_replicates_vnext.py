from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from safe_rl.accvp.contracts.schema import file_sha256
from safe_rl.evaluation_protocol import (
    EvidenceProtocolError,
    normalise_seed_cohorts,
    stable_hash,
)
from safe_rl.pipeline.accvp_runtime_benchmark_replicates import aggregate_runtime_reports
from safe_rl.pipeline.audit_ppo_replicate_lineage import (
    INACTIVE_ACCVP_COMPATIBILITY_VERSION,
    LINEAGE_AUDIT_IMPLEMENTATION_VERSION,
    audit_manifest,
    write_audit_report,
)
from safe_rl.pipeline.stage3_train_ppo_replicates import build_replicate_plan
from safe_rl.ppo_replicates import (
    observation_contract,
    optimizer_seed,
    validate_reward_semantics,
)
from safe_rl.rl.ppo import (
    _EpisodeSeedTraceCallback,
    _build_ppo_worker_env,
    _ppo_optimizer_seed,
    _restore_simulator_seed_after_model_setup,
    _validate_stage3_seed_preflight,
)
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


def test_sb3_model_setup_restores_simulator_seed_after_optimizer_seed_propagation():
    cfg = load_config(CANDIDATE)

    class FakeVecEnv:
        def __init__(self):
            self.requests = []

        def seed(self, seed):
            self.requests.append(int(seed))
            return [int(seed)]

    class FakeModel:
        def __init__(self):
            self.env = FakeVecEnv()

        def get_env(self):
            return self.env

    model = FakeModel()
    applied = _restore_simulator_seed_after_model_setup(model, cfg)

    assert _ppo_optimizer_seed(cfg) == 1001
    assert applied == [20001]
    assert model.env.requests == [20001]


def test_stage3_seed_preflight_audits_simulator_and_optimizer_cohorts_separately():
    cfg = load_config(CANDIDATE)
    preflight = _validate_stage3_seed_preflight(cfg)

    assert preflight["training_start_seed"] == 20001
    assert preflight["optimizer_seed"] == 1001
    assert set(preflight["seed_audit"]["cohort_counts"]) == {
        "stage3_training_start",
        "stage3_selection",
        "ppo_optimizer_replicates",
    }

    invalid = clone_with_overrides(cfg, {"rl": {"optimizer_seed": 999}})
    with pytest.raises(EvidenceProtocolError, match="optimizer-replicate cohort"):
        _validate_stage3_seed_preflight(invalid)


def test_episode_seed_guard_fails_on_the_first_observed_out_of_cohort_step():
    class FakeBaseCallback:
        def __init__(self):
            self.locals = {}
            self.num_timesteps = 1

    callback = _EpisodeSeedTraceCallback(
        FakeBaseCallback,
        allowed_episode_seeds=[20001, 20002],
    ).callback
    callback.locals = {
        "dones": [False],
        "infos": [{"episode_seed": 1001, "episode_index": 0}],
    }

    with pytest.raises(EvidenceProtocolError, match="outside.*stage3_training cohort"):
        callback._on_step()


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


def _lineage_manifest_with_frozen_observation(
    tmp_path: Path,
    frozen_payload: dict,
) -> Path:
    checkpoint = tmp_path / "checkpoint_1001.zip"
    config = tmp_path / "config_1001.yaml"
    report = tmp_path / "report_1001.json"
    checkpoint.write_bytes(b"checkpoint-1001")
    config.write_text("optimizer_seed: 1001\n", encoding="utf-8")
    frozen_hash = stable_hash(frozen_payload)
    report.write_text(
        json.dumps(
            {
                "observation_contract": frozen_payload,
                "observation_contract_hash": frozen_hash,
                "observation_dim": 63,
                "observation_shape": [63],
            }
        ),
        encoding="utf-8",
    )
    reward_hash = validate_reward_semantics(load_config(BASELINE))["sha256"]
    manifest = tmp_path / "lineage_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "artifact_kind": "ppo_optimizer_replicate_manifest_v1",
                "schema_version": 1,
                "status": "complete",
                "method_id": "wcdt_reward_v2",
                "records": [
                    {
                        "method_id": "wcdt_reward_v2",
                        "training_seed": 1001,
                        "optimizer_seed": 1001,
                        "checkpoint": str(checkpoint),
                        "checkpoint_sha256": file_sha256(checkpoint),
                        "resolved_config": str(config),
                        "resolved_config_sha256": file_sha256(config),
                        "stage3_report": str(report),
                        "stage3_report_sha256": file_sha256(report),
                        "training_budget": {"total_timesteps": 100000},
                        "reward_semantics_hash": reward_hash,
                        "observation_contract_hash": frozen_hash,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return manifest


def test_lineage_audit_ignores_only_inactive_accvp_default_drift(tmp_path: Path):
    current = observation_contract(load_config(BASELINE), require_artifacts=False)
    frozen_payload = copy.deepcopy(current["payload"])
    assert frozen_payload["accvp_observation_enabled"] is False
    frozen_payload["accvp_observation"].pop("critical_actor_overflow_sample_limit")
    assert stable_hash(frozen_payload) != current["sha256"]
    manifest = _lineage_manifest_with_frozen_observation(tmp_path, frozen_payload)

    audit = audit_manifest(
        manifest,
        required_seeds=[1001],
        method_config=BASELINE,
    )

    assert audit["status"] == "reusable"
    assert audit["audit_implementation_version"] == LINEAGE_AUDIT_IMPLEMENTATION_VERSION
    assert audit["invalid_records"] == []
    assert audit["compatibility_migrations"] == [
        {
            "optimizer_seed": 1001,
            "compatibility_version": INACTIVE_ACCVP_COMPATIBILITY_VERSION,
            "declared_observation_contract_hash": stable_hash(frozen_payload),
            "current_observation_contract_hash": current["sha256"],
            "effective_observation_contract_hash": stable_hash(
                {
                    key: value
                    for key, value in frozen_payload.items()
                    if key != "accvp_observation"
                }
            ),
        }
    ]


def test_lineage_audit_rejects_active_observation_drift(tmp_path: Path):
    current = observation_contract(load_config(BASELINE), require_artifacts=False)
    frozen_payload = copy.deepcopy(current["payload"])
    frozen_payload["forecast_source"] = "changed_active_predictor"
    manifest = _lineage_manifest_with_frozen_observation(tmp_path, frozen_payload)

    audit = audit_manifest(
        manifest,
        required_seeds=[1001],
        method_config=BASELINE,
    )

    assert audit["status"] == "retrain_required"
    assert audit["invalid_records"] == [
        {
            "optimizer_seed": 1001,
            "reasons": ["effective observation contract differs from method config"],
        }
    ]


def test_lineage_audit_archives_failed_report_and_is_idempotent(tmp_path: Path):
    output = tmp_path / "baseline_audit.json"
    output.write_text(
        json.dumps(
            {
                "artifact_kind": "ppo_replicate_lineage_audit_v1",
                "schema_version": 1,
                "status": "retrain_required",
            }
        ),
        encoding="utf-8",
    )
    report = {
        "artifact_kind": "ppo_replicate_lineage_audit_v1",
        "schema_version": 1,
        "audit_implementation_version": LINEAGE_AUDIT_IMPLEMENTATION_VERSION,
        "status": "reusable",
    }

    written, action = write_audit_report(output, report)
    assert written == output.resolve()
    assert action == "archive_failed_and_replace"
    payload = json.loads(output.read_text(encoding="utf-8"))
    archive = Path(payload["prior_failed_audit"]["archived_source_report"])
    assert archive.is_file()
    assert json.loads(archive.read_text(encoding="utf-8"))["status"] == "retrain_required"

    _written_again, second_action = write_audit_report(output, report)
    assert second_action == "reuse_identical"


def test_lineage_audit_replaces_unfingerprinted_legacy_reusable_report(tmp_path: Path):
    output = tmp_path / "legacy_passing_audit.json"
    output.write_text(
        json.dumps(
            {
                "artifact_kind": "ppo_replicate_lineage_audit_v1",
                "schema_version": 1,
                "status": "reusable",
            }
        ),
        encoding="utf-8",
    )
    report = {
        "artifact_kind": "ppo_replicate_lineage_audit_v1",
        "schema_version": 1,
        "audit_implementation_version": LINEAGE_AUDIT_IMPLEMENTATION_VERSION,
        "status": "reusable",
    }

    _written, action = write_audit_report(output, report)

    assert action == "archive_failed_and_replace"
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["audit_fingerprint"] == stable_hash(
        {key: value for key, value in payload.items() if key != "audit_fingerprint"}
    )
