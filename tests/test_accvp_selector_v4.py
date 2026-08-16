from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from safe_rl.accvp.contracts.protocol import (
    ACCVP_SELECTOR4_DATA_CONTRACT_VERSION,
)
from safe_rl.accvp.contracts.runtime_contract import (
    FORMAL_RUNTIME_FEATURE_VERSION,
)
from safe_rl.accvp.contracts.schema import file_sha256, stable_hash
from safe_rl.accvp.evaluation.selector_capacity_v4 import (
    SELECTOR4_PROTOCOL_ID,
    build_selector4_capacity_report,
    selection_audit_row,
)
from safe_rl.pipeline.run_accvp_vnext_pipeline import (
    _load_workflow_contract,
    _workflow_seed_values,
)
from safe_rl.ppo_factorial import EXPECTED_CANDIDATE_METHOD_ROLES
from safe_rl.pipeline.accvp_selector4_capacity_audit import (
    _recorded_overflow_rows,
)
from safe_rl.prediction import actor_selector
from safe_rl.prediction.actor_selector import (
    ACTOR_SELECTION_VERSION_V3,
    ACTOR_SELECTION_VERSION_V4,
    actor_selection_config_hash,
    select_merge_relevant_actors,
)
from safe_rl.prediction.candidate_conflict import (
    ActorConflictEvidence,
    candidate_union_conflict_oracle_reference,
)
from safe_rl.prediction.wcdt_v3_predictor import build_v3_runtime_batch
from safe_rl.sim.history_buffer import HistoryBuffer
from safe_rl.sim.types import VehicleState
from safe_rl.utils.config import clone_with_overrides, load_config


def _vehicle(
    vehicle_id: str,
    lane_pos: float,
    *,
    lane_index: int,
    edge_id: str = "main_aux",
) -> VehicleState:
    return VehicleState(
        vehicle_id=vehicle_id,
        x=300.0 + lane_pos,
        y=53.8 + 3.2 * lane_index,
        heading=0.0,
        speed=20.0,
        accel=0.0,
        lane_index=lane_index,
        lane_id=f"{edge_id}_{lane_index}",
        lane_pos=lane_pos,
        edge_id=edge_id,
        route_position_valid=True,
    )


def _config(version: str):
    return clone_with_overrides(
        load_config(),
        {"accvp": {"actor_relevance": {"version": version}}},
    )


def _no_conflict(vehicle_id: str) -> ActorConflictEvidence:
    return ActorConflictEvidence(
        vehicle_id=vehicle_id,
        candidate_conflict_eligible=False,
        conflict_candidate_ids=(),
        conflict_hypothesis_ids=(),
        conflict_surface_ids=(),
        earliest_conflict_time_s=1_000_000.0,
        earliest_overlap_time_s=1_000_000.0,
        minimum_swept_obb_gap=1_000_000.0,
        nearest_candidate_conflict=False,
    )


def test_selector_v4_is_lane_aware_without_mutating_v3(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ego = _vehicle("ego", 100.0, lane_index=0)
    actor = _vehicle("mainline_other", 110.0, lane_index=2)
    monkeypatch.setattr(
        actor_selector,
        "candidate_union_conflict_oracle",
        lambda *_args, **_kwargs: (
            {actor.vehicle_id: _no_conflict(actor.vehicle_id)},
            (3, 4, 5, 6, 7, 8),
        ),
    )

    v3 = select_merge_relevant_actors(
        _config(ACTOR_SELECTION_VERSION_V3),
        ego,
        [ego, actor],
        max_actors=12,
        selector_scope="accvp",
    )
    v4 = select_merge_relevant_actors(
        _config(ACTOR_SELECTION_VERSION_V4),
        ego,
        [ego, actor],
        max_actors=12,
        selector_scope="accvp",
    )

    assert v3.actor_metadata[actor.vehicle_id].role == "auxiliary_local"
    assert v3.actor_metadata[actor.vehicle_id].critical
    assert v4.actor_metadata[actor.vehicle_id].role == "other"
    assert v4.actor_metadata[actor.vehicle_id].contextual
    assert not v4.actor_metadata[actor.vehicle_id].critical


def test_selector_v4_keeps_candidate_conflict_on_non_auxiliary_lane(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ego = _vehicle("ego", 100.0, lane_index=0)
    actor = _vehicle("future_conflict", 110.0, lane_index=3)
    evidence = _no_conflict(actor.vehicle_id)
    evidence = ActorConflictEvidence(
        **{
            **evidence.__dict__,
            "candidate_conflict_eligible": True,
            "conflict_candidate_ids": (6,),
            "nearest_candidate_conflict": True,
            "earliest_conflict_time_s": 1.0,
        }
    )
    monkeypatch.setattr(
        actor_selector,
        "candidate_union_conflict_oracle",
        lambda *_args, **_kwargs: ({actor.vehicle_id: evidence}, (6,)),
    )

    selection = select_merge_relevant_actors(
        _config(ACTOR_SELECTION_VERSION_V4),
        ego,
        [ego, actor],
        max_actors=12,
        selector_scope="accvp",
    )

    assert selection.actor_metadata[actor.vehicle_id].role == "other"
    assert selection.actor_metadata[actor.vehicle_id].critical
    assert selection.actor_metadata[actor.vehicle_id].candidate_conflict_eligible


def _audit_row(critical_count: int) -> dict:
    ids = [f"actor_{index}" for index in range(critical_count)]
    return {
        "state_id": "state",
        "provenance": {
            "scope": "fixture",
            "method_id": "candidate_table_reward_v3_1_commitment",
            "optimizer_seed": 1001,
            "traffic_profile": "dense",
        },
        "selector_latency_s": 0.001,
        "critical_count": critical_count,
        "contextual_count": 0,
        "critical_actor_ids": ids,
        "target_front_rear_ids": ids[:2],
        "candidate_conflict_ids": ids,
        "nearest_conflict_ids": ids[:1],
        "lowest_conflict_ttc_ids": ids[:1],
        "protected_actor_ids": ids,
        "critical_actors": [{"vehicle_id": value} for value in ids],
    }


def test_selector_v4_capacity_requires_two_actor_headroom() -> None:
    report = build_selector4_capacity_report(
        [_audit_row(10)],
        source_coverage={"fixture": True},
        selector_config={"version": ACTOR_SELECTION_VERSION_V4},
        source_lineage={},
    )

    assert report["capacity_reports"]["8"]["critical_overflow_count"] == 1
    assert not report["capacity_reports"]["10"]["gate_pass"]
    assert report["capacity_reports"]["10"]["capacity_headroom"] == 0
    assert report["capacity_reports"]["12"]["gate_pass"]
    assert report["selected_capacity"] == 12
    assert report["scorer_runtime_gate"]["capacity_fallback_after_failure_allowed"] is False


def test_selector_v4_keeps_capacity_ten_diagnostic_when_it_also_passes() -> None:
    report = build_selector4_capacity_report(
        [_audit_row(7)],
        source_coverage={"fixture": True},
        selector_config={"version": ACTOR_SELECTION_VERSION_V4},
        source_lineage={},
    )

    assert not report["capacity_reports"]["8"]["gate_pass"]
    assert report["capacity_reports"]["10"]["gate_pass"]
    assert report["capacity_reports"]["12"]["gate_pass"]
    assert report["selected_capacity"] == 12
    assert report["diagnostic_capacity_results_only"] == [8, 10]
    assert report["selector_latency_interpretation"]["gate_eligible"] is False


def test_recorded_overflow_telemetry_preserves_exact_historical_state(
    tmp_path: Path,
) -> None:
    runtime_path = tmp_path / "runtime.json"
    runtime = {
        "report_fingerprint": "runtime-fixture",
        "episodes": [
            {
                "episode_seed": 55027,
                "curriculum_profile": "boundary",
                "accvp_table_critical_actor_overflow_examples": [
                    {
                        "episode_seed": 55027,
                        "optimizer_seed": 1005,
                        "decision_index": 69,
                        "critical_actors": [
                            {
                                "vehicle_id": "front",
                                "role": "target_front",
                                "candidate_conflict_eligible": False,
                                "nearest_candidate_conflict": False,
                                "trigger_reasons": "",
                            },
                            {
                                "vehicle_id": "conflict",
                                "role": "other",
                                "candidate_conflict_eligible": True,
                                "nearest_candidate_conflict": True,
                                "trigger_reasons": (
                                    "candidate_union_conflict lowest_ttc"
                                ),
                            },
                        ],
                    }
                ],
            }
        ],
    }
    runtime_path.write_text(json.dumps(runtime), encoding="utf-8")
    diagnostic_path = tmp_path / "diagnostic.json"
    diagnostic = {
        "artifact_kind": "accvp_selector_overflow_shadow_diagnostic_v1",
        "diagnostic_only": True,
        "examples_complete": True,
        "reconstruction_complete": True,
        "protected_coverage_complete": True,
        "frozen_selector_version": ACTOR_SELECTION_VERSION_V3,
        "source_runtime_report": str(runtime_path),
        "source_runtime_report_sha256": file_sha256(runtime_path),
        "source_runtime_report_fingerprint": "runtime-fixture",
        "reported_overflow_count": 1,
        "rows": [
            {
                "episode_seed": 55027,
                "optimizer_seed": 1005,
                "decision_index": 69,
                "taper_distance_m": 35.0,
                "lane_aware_critical_ids": ["front", "conflict"],
                "protected_actor_ids": ["front", "conflict"],
            }
        ],
    }
    diagnostic["diagnostic_fingerprint"] = stable_hash(diagnostic)
    diagnostic_path.write_text(json.dumps(diagnostic), encoding="utf-8")

    rows, lineage = _recorded_overflow_rows(diagnostic_path)

    assert len(rows) == 1
    assert rows[0]["critical_actor_ids"] == ["front", "conflict"]
    assert rows[0]["candidate_conflict_ids"] == ["conflict"]
    assert rows[0]["lowest_conflict_ttc_ids"] == ["conflict"]
    assert rows[0]["contextual_count_observed"] is False
    assert lineage["recorded_overflow_state_count"] == 1


def test_selector_v4_report_locks_capacity_12(tmp_path: Path) -> None:
    report = build_selector4_capacity_report(
        [_audit_row(10)],
        source_coverage={"fixture": True},
        selector_config={"version": ACTOR_SELECTION_VERSION_V4},
        source_lineage={},
    )
    report_path = tmp_path / "selector4.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    overlay = tmp_path / "locked.yaml"
    overlay.write_text(
        yaml.safe_dump(
            {
                "extends": (
                    "safe_rl/config/active/accvp_vnext_selector4/"
                    "selector_audit.yaml"
                ),
                "accvp": {
                    "selector_contract": {
                        "require_capacity_lock": True,
                        "audit_report": str(report_path),
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    cfg = load_config(overlay)
    assert cfg.accvp.actor_count == 12
    assert cfg.accvp.data_contract_version == ACCVP_SELECTOR4_DATA_CONTRACT_VERSION


def test_selector4_workflow_uses_new_runtime_seeds_and_sources() -> None:
    _path, workflow = _load_workflow_contract(
        "safe_rl/config/active/accvp_vnext_selector4/workflow.yaml"
    )
    assert workflow["phase_order"][0] == "selector_contract_audit"
    assert _workflow_seed_values(workflow, "runtime", []) == list(
        range(56001, 56061)
    )
    assert "selector3_formal_dataset" in str(
        workflow["paths"]["selector_source_dataset"]
    )
    assert workflow["freeze_before_stage5"]["post_runtime_capacity_fallback_allowed"] is False


def test_selector4_factorial_keeps_the_frozen_ppo_feature_contract() -> None:
    matrix_path = Path(
        "safe_rl/config/active/accvp_vnext_selector4/"
        "ppo_ablation_matrix.yaml"
    )
    matrix = yaml.safe_load(matrix_path.read_text(encoding="utf-8"))
    variants = dict(matrix["variants"])
    for method_id in EXPECTED_CANDIDATE_METHOD_ROLES:
        declared = str(variants[method_id]["training_semantics_version"])
        assert FORMAL_RUNTIME_FEATURE_VERSION in declared.split("+")
        assert "risk_gated_candidate_table_v4_bounded_stale" not in declared


def _assert_semantically_equal(actual, expected) -> None:
    if isinstance(expected, float):
        assert float(actual) == pytest.approx(expected, abs=1.0e-9)
        return
    if isinstance(expected, dict):
        assert set(actual) == set(expected)
        for key in expected:
            _assert_semantically_equal(actual[key], expected[key])
        return
    if isinstance(expected, (list, tuple)):
        assert type(actual) is type(expected)
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected):
            _assert_semantically_equal(actual_item, expected_item)
        return
    assert actual == expected


@pytest.mark.parametrize(
    ("ego", "vehicles"),
    [
        (
            _vehicle("ego", 100.0, lane_index=0),
            [
                _vehicle("front", 114.0, lane_index=1),
                _vehicle("rear", 87.0, lane_index=1),
                _vehicle("adjacent", 104.0, lane_index=2),
            ],
        ),
        (
            _vehicle("ego", 106.0, lane_index=1),
            [
                _vehicle("front", 121.0, lane_index=1),
                _vehicle("rear", 92.0, lane_index=1),
                _vehicle("changed_lane", 108.0, lane_index=0),
            ],
        ),
        (
            _vehicle("ego", 112.0, lane_index=0),
            [
                _vehicle("moved_front", 150.0, lane_index=1),
                _vehicle("moved_rear", 109.0, lane_index=1),
                _vehicle(
                    "other_edge",
                    116.0,
                    lane_index=2,
                    edge_id="unrelated_edge",
                ),
            ],
        ),
    ],
)
def test_selector_v4_optimized_geometry_matches_scalar_reference_across_state_changes(
    monkeypatch: pytest.MonkeyPatch,
    ego: VehicleState,
    vehicles: list[VehicleState],
) -> None:
    cfg = _config(ACTOR_SELECTION_VERSION_V4)
    current = [ego, *vehicles]

    optimized = select_merge_relevant_actors(
        cfg,
        ego,
        current,
        max_actors=12,
        selector_scope="accvp",
    )
    monkeypatch.setattr(
        actor_selector,
        "candidate_union_conflict_oracle",
        candidate_union_conflict_oracle_reference,
    )
    reference = select_merge_relevant_actors(
        cfg,
        ego,
        current,
        max_actors=12,
        selector_scope="accvp",
    )

    _assert_semantically_equal(optimized.to_dict(), reference.to_dict())


def test_selector_v4_optimized_geometry_preserves_runtime_model_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _config(ACTOR_SELECTION_VERSION_V4)
    ego = _vehicle("ego", 106.0, lane_index=1)
    current = [
        ego,
        _vehicle("front", 121.0, lane_index=1),
        _vehicle("rear", 92.0, lane_index=1),
        _vehicle("changed_lane", 108.0, lane_index=0),
        _vehicle("adjacent", 114.0, lane_index=2),
    ]
    history = HistoryBuffer(
        history_steps=int(cfg.scenario.history_steps),
        max_agents=13,
    )
    history.append(current)

    optimized = build_v3_runtime_batch(
        cfg,
        history,
        "ego",
        max_actors_override=12,
        selector_scope="accvp",
    )
    monkeypatch.setattr(
        actor_selector,
        "candidate_union_conflict_oracle",
        candidate_union_conflict_oracle_reference,
    )
    reference = build_v3_runtime_batch(
        cfg,
        history,
        "ego",
        max_actors_override=12,
        selector_scope="accvp",
    )

    assert optimized["actor_row_ids"] == reference["actor_row_ids"]
    assert optimized["runtime_agent_ids"] == reference["runtime_agent_ids"]
    _assert_semantically_equal(
        optimized["actor_selection"].to_dict(),
        reference["actor_selection"].to_dict(),
    )
    for name in (
        "history_features",
        "history_valid_mask",
        "history_lane_ids",
        "history_edge_role_ids",
        "role_ids",
        "lane_ids",
        "edge_role_ids",
        "mask",
        "selected_indices",
    ):
        np.testing.assert_array_equal(optimized[name], reference[name])


def test_sumo_scale_is_explicit_and_validated() -> None:
    from safe_rl.sim.sumo_highway_merge_env import SumoHighwayMergeEnv

    cfg = clone_with_overrides(
        load_config(), {"scenario": {"sumo_scale": 1.5}}
    )
    env = object.__new__(SumoHighwayMergeEnv)
    env.config = cfg
    env.seed_value = 1
    env.step_length = 0.1
    args = env._sumo_load_args()
    assert args[args.index("--scale") + 1] == "1.5"

    env.config = clone_with_overrides(
        cfg, {"scenario": {"sumo_scale": 0.0}}
    )
    with pytest.raises(ValueError, match="sumo_scale"):
        env._sumo_load_args()
