from __future__ import annotations

import json
from pathlib import Path

import pytest

from safe_rl.main_method_protocol import (
    EXPECTED_METHOD_ROLES,
    FINAL_METHOD_ID,
    FORECAST_ATTRIBUTION_METHODS,
    load_protocol,
)
from safe_rl.pipeline import accvp_runtime_benchmark_replicates
from safe_rl.pipeline.main_method_ppo_suite import _equivalence_authorization
from safe_rl.pipeline.main_method_runtime import _generic_metrics, _runtime_seeds
from safe_rl.pipeline.main_method_stage5 import aggregate as aggregate_main_method_stage5
from safe_rl.accvp.contracts.schema import stable_hash
from safe_rl.utils.performance import PerformanceTracker


PROTOCOL = Path(
    "safe_rl/config/config_fix/accvp_main_method_table_v1/protocol.yaml"
)


def test_main_method_protocol_freezes_methods_rewards_and_executor_budget() -> None:
    protocol = load_protocol(PROTOCOL, verify_artifacts=False)

    assert protocol["protocol_id"] == "accvp-main-method-table-v1"
    assert {
        method_id: method["role"]
        for method_id, method in protocol["methods"].items()
    } == EXPECTED_METHOD_ROLES
    assert protocol["optimizer_seeds"] == [1000, 1001, 1002, 1003, 1004]
    assert all(
        protocol["_executor_contracts"][method_id]["rollout_size"] == 1024
        for method_id in protocol["_executor_contracts"]
    )
    assert all(
        protocol["_executor_contracts"][method_id]["ppo_num_envs"] == 4
        for method_id in protocol["_executor_contracts"]
    )
    assert len(
        {
            protocol["_configs"][method_id]["reward"]["sha256"]
            for method_id in FORECAST_ATTRIBUTION_METHODS
        }
    ) == 1
    assert protocol["_configs"][FINAL_METHOD_ID]["config"].rl[
        "policy_lateral_commitment"
    ]["enabled"] is True


def test_main_method_runtime_uses_frozen_runtime_cohort_not_legacy_seed_1005() -> None:
    protocol = load_protocol(PROTOCOL, verify_artifacts=False)
    seeds = _runtime_seeds(protocol)

    assert seeds == list(range(56001, 56061))
    legacy = protocol["deployment_runtime"]["legacy_accvp_report"]
    assert legacy["optimizer_seeds"] == [1001, 1002, 1003, 1004, 1005]
    assert legacy["reusable_for_current_cohort"] is False


def test_cross_executor_reuse_requires_method_specific_exact_equivalence(
    tmp_path: Path,
) -> None:
    source = {
        "ppo_num_envs": 1,
        "n_steps": 1024,
        "checkpoint_selection_workers": 1,
    }
    target = {
        "ppo_num_envs": 4,
        "n_steps": 256,
        "checkpoint_selection_workers": 4,
    }
    assert (
        _equivalence_authorization(
            method_id="wcdt_reward_v2",
            source_contract=source,
            target_contract=target,
            report_path=None,
            report=None,
        )
        is None
    )

    report_path = tmp_path / "equivalence.json"
    report_path.write_text("{}", encoding="utf-8")
    report = {
        "report_fingerprint": "fingerprint",
        "methods": {
            "wcdt_reward_v2": {
                "comparisons": [
                    {**source, "equivalent": True},
                    {**target, "equivalent": True},
                ]
            }
        },
    }
    authorization = _equivalence_authorization(
        method_id="wcdt_reward_v2",
        source_contract=source,
        target_contract=target,
        report_path=report_path,
        report=report,
    )
    assert authorization is not None
    assert authorization["kind"] == "exact_state_and_selection_equivalence"


def test_runtime_operation_samples_are_opt_in_and_aggregated(monkeypatch) -> None:
    values = iter([10.0, 10.2, 10.5, 11.0])
    monkeypatch.setattr("safe_rl.utils.performance.time.perf_counter", lambda: next(values))
    tracker = PerformanceTracker(record_operation_samples=True)
    with tracker.measure("forecast_inference_time"):
        pass
    report = tracker.summary(steps=2)

    assert report["operation_samples_s"]["forecast_inference_time"] == pytest.approx([0.3])
    metrics = _generic_metrics(
        [
            {
                "steps": 2,
                "performance": {
                    "wall_time": 0.5,
                    "steps_per_second": 4.0,
                    "operation_samples_s": {
                        "forecast_inference_time": [0.1, 0.2]
                    },
                },
            }
        ]
    )
    assert metrics["operation_latency_s"]["forecast_inference_time"]["count"] == 2
    assert metrics["operation_latency_s"]["forecast_inference_time"]["p99"] > 0.19


def test_accvp_runtime_accepts_audited_main_method_manifest_kind() -> None:
    assert (
        "main_method_ppo_method_manifest_v1"
        in accvp_runtime_benchmark_replicates.SUPPORTED_REPLICATE_MANIFEST_KINDS
    )


def test_main_method_stage5_aggregate_promotes_crossed_pair_statistics(
    tmp_path: Path,
) -> None:
    rows = []
    for optimizer_seed in range(1000, 1005):
        left_name = f"left_seed_{optimizer_seed}"
        right_name = f"right_seed_{optimizer_seed}"
        episodes_left = []
        episodes_right = []
        for simulator_seed in (80001, 80002):
            common = {
                "seed": simulator_seed,
                "min_distance": 2.0,
                "ttc_p1": 3.0,
                "drac_p99": 1.0,
                "completion_time": 10.0,
                "ego_speed_mean": 20.0,
                "hard_brake_rate": 0.0,
                "proxy_collision": False,
                "safety_violation": False,
                "geometric_overlap": False,
                "taper_miss": False,
                "merge_success": True,
                "timely_merge_success": True,
            }
            episodes_left.append({**common, "episode_reward": 1.0})
            episodes_right.append({**common, "episode_reward": 2.0})
        report = {
            "configured_pairs": [
                {
                    "name": "left__vs__right",
                    "left": left_name,
                    "right": right_name,
                }
            ],
            "acceptance": {
                "left__vs__right": {"available": True, "regression": False}
            },
            "groups": {
                left_name: {
                    "comparative": {
                        "method": "left",
                        "training_seed": optimizer_seed,
                        "checkpoint_sha256": f"{optimizer_seed - 999:064x}",
                    },
                    "episodes": episodes_left,
                    "metrics": {"average_reward": 1.0},
                },
                right_name: {
                    "comparative": {
                        "method": "right",
                        "training_seed": optimizer_seed,
                        "checkpoint_sha256": f"{optimizer_seed - 899:064x}",
                    },
                    "episodes": episodes_right,
                    "metrics": {"average_reward": 2.0},
                },
            },
        }
        report_path = tmp_path / f"stage5_{optimizer_seed}.json"
        report_path.write_text(json.dumps(report), encoding="utf-8")
        rows.append(
            {
                "scope": "secondary",
                "optimizer_seed": optimizer_seed,
                "stage5_report": str(report_path),
                "method_ids": ["left", "right"],
                "simulator_seed_count": 2,
            }
        )
    request = {
        "artifact_kind": "main_method_stage5_request_v1",
        "schema_version": 1,
        "status": "prepared",
        "protocol_id": "accvp-main-method-table-v1",
        "mode": "method_effect",
        "conclusion_scope": "method_effect",
        "optimizer_seeds": list(range(1000, 1005)),
        "statistics": {
            "confidence": 0.95,
            "bootstrap_replicates": 50,
            "bootstrap_seed": 42001,
        },
        "rows": rows,
    }
    request["request_fingerprint"] = stable_hash(request)
    request_path = tmp_path / "method_effect_request.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")

    output = aggregate_main_method_stage5(request_path)
    aggregate = json.loads(output.read_text(encoding="utf-8"))

    comparison = aggregate["comparisons"]["left__vs__right"]
    assert comparison["statistics"]["training_seed_count"] == 5
    assert comparison["statistics"]["simulator_seed_count"] == 2
    assert comparison["statistics"]["continuous"]["episode_reward"][
        "mean_delta"
    ] == pytest.approx(1.0)
    assert comparison["any_source_regression"] is False
