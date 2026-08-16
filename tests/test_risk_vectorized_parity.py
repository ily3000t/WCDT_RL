from __future__ import annotations

from dataclasses import fields

import numpy as np
import pytest

import safe_rl.risk.merge_local as merge_local_module
from safe_rl.risk.merge_local import (
    CandidateRiskSample,
    evaluate_candidate_actions,
    evaluate_candidate_actions_reference,
)
from safe_rl.risk.risk_module import RiskModuleWrapper
from safe_rl.shield.safety_shield import SafetyShield
from safe_rl.sim.action_space import ACTIONS
from safe_rl.sim.metrics import (
    INF_TTC,
    batch_pairwise_obb_gap_overlap,
    batch_pairwise_obb_metrics,
    bbox_gap,
    drac,
    geometric_overlap,
    relative_ttc,
)
from safe_rl.sim.types import VehicleState
from safe_rl.utils.config import load_config
from safe_rl.utils.performance import PerformanceTracker


def _state(
    vehicle_id: str,
    *,
    x: float,
    y: float,
    heading: float = 0.0,
    speed: float = 15.0,
    lane: int = 0,
    edge: str = "main_aux",
    lane_pos: float | None = None,
) -> VehicleState:
    return VehicleState(
        vehicle_id=vehicle_id,
        x=float(x),
        y=float(y),
        heading=float(heading),
        speed=float(speed),
        lane_index=int(lane),
        lane_id=f"{edge}_{lane}",
        lane_pos=float(x if lane_pos is None else lane_pos),
        edge_id=edge,
        length=4.8,
        width=1.8,
        accel=0.0,
    )


def _advanced(state: VehicleState, step: int, dt: float = 0.1) -> VehicleState:
    distance = float(state.speed) * float(step) * dt
    return VehicleState(
        **{
            **state.to_dict(),
            "x": float(state.x + distance * np.cos(state.heading)),
            "y": float(state.y + distance * np.sin(state.heading)),
            "lane_pos": float(state.lane_pos + distance),
        }
    )


def _assert_scalar(value: float, expected: float) -> None:
    if expected >= INF_TTC:
        assert value == expected
    else:
        assert value == pytest.approx(expected, abs=1.0e-6)


def test_pairwise_batch_geometry_matches_scalar_reference_at_boundaries():
    ego_initial = [
        _state("ego_a", x=100.0, y=0.0, speed=18.0),
        _state("ego_b", x=100.0, y=0.0, heading=0.12, speed=20.0),
    ]
    other_initial = [
        _state("overlap", x=100.0, y=0.0, speed=18.0),
        _state("front", x=114.0, y=0.1, speed=10.0),
        _state("crossing", x=105.0, y=5.0, heading=-np.pi / 2.0, speed=8.0),
        _state("touching", x=104.8, y=0.0, speed=18.0),
    ]
    ego_rollouts = [[_advanced(state, step) for step in range(4)] for state in ego_initial]
    other_rollouts = [[_advanced(state, step) for step in range(4)] for state in other_initial]
    batch = batch_pairwise_obb_metrics(ego_rollouts, other_rollouts)
    assert batch.gap.shape == batch.ttc.shape == batch.drac.shape == batch.overlap.shape == (2, 4, 4)
    for action_idx, ego_rollout in enumerate(ego_rollouts):
        for step_idx, ego in enumerate(ego_rollout):
            for actor_idx, other_rollout in enumerate(other_rollouts):
                other = other_rollout[step_idx]
                _assert_scalar(float(batch.gap[action_idx, step_idx, actor_idx]), bbox_gap(ego, other))
                _assert_scalar(float(batch.ttc[action_idx, step_idx, actor_idx]), relative_ttc(ego, other))
                _assert_scalar(float(batch.drac[action_idx, step_idx, actor_idx]), drac(ego, other))
                assert bool(batch.overlap[action_idx, step_idx, actor_idx]) is geometric_overlap(ego, other)


def test_pairwise_batch_geometry_matches_seeded_scalar_corpus():
    rng = np.random.default_rng(20260712)

    def random_state(vehicle_id: str) -> VehicleState:
        state = _state(
            vehicle_id,
            x=float(rng.uniform(-20.0, 40.0)),
            y=float(rng.uniform(-6.0, 6.0)),
            heading=float(rng.uniform(-np.pi, np.pi)),
            speed=float(rng.uniform(0.0, 30.0)),
        )
        return VehicleState(
            **{
                **state.to_dict(),
                "length": float(rng.uniform(3.5, 12.0)),
                "width": float(rng.uniform(1.6, 2.8)),
            }
        )

    ego_rollouts = [[_advanced(random_state(f"ego_{a}"), step) for step in range(5)] for a in range(5)]
    other_rollouts = [[_advanced(random_state(f"other_{n}"), step) for step in range(5)] for n in range(6)]
    batch = batch_pairwise_obb_metrics(ego_rollouts, other_rollouts)
    for action_idx, ego_rollout in enumerate(ego_rollouts):
        for step_idx, ego in enumerate(ego_rollout):
            for actor_idx, other_rollout in enumerate(other_rollouts):
                other = other_rollout[step_idx]
                _assert_scalar(float(batch.gap[action_idx, step_idx, actor_idx]), bbox_gap(ego, other))
                _assert_scalar(float(batch.ttc[action_idx, step_idx, actor_idx]), relative_ttc(ego, other))
                _assert_scalar(float(batch.drac[action_idx, step_idx, actor_idx]), drac(ego, other))
                assert bool(batch.overlap[action_idx, step_idx, actor_idx]) is geometric_overlap(ego, other)


def test_pairwise_batch_geometry_handles_empty_actor_set():
    ego_rollouts = [[_advanced(_state("ego", x=1.0, y=2.0), step) for step in range(3)]]
    batch = batch_pairwise_obb_metrics(ego_rollouts, [])
    assert batch.gap.shape == (1, 3, 0)
    assert batch.ttc.shape == (1, 3, 0)
    assert batch.drac.shape == (1, 3, 0)
    assert batch.overlap.shape == (1, 3, 0)


def test_gap_overlap_batch_matches_scalar_geometry_at_tight_tolerance():
    rng = np.random.default_rng(20260816)
    ego_rollouts = []
    other_rollouts = []
    for action_idx in range(4):
        state = _state(
            f"ego_{action_idx}",
            x=float(rng.uniform(80.0, 120.0)),
            y=float(rng.uniform(-2.0, 2.0)),
            heading=float(rng.uniform(-0.2, 0.2)),
            speed=float(rng.uniform(5.0, 25.0)),
        )
        ego_rollouts.append([_advanced(state, step) for step in range(6)])
    for actor_idx in range(7):
        state = _state(
            f"actor_{actor_idx}",
            x=float(rng.uniform(70.0, 140.0)),
            y=float(rng.uniform(-5.0, 5.0)),
            heading=float(rng.uniform(-0.3, 0.3)),
            speed=float(rng.uniform(3.0, 28.0)),
        )
        other_rollouts.append([_advanced(state, step) for step in range(6)])

    gap, overlap = batch_pairwise_obb_gap_overlap(
        ego_rollouts,
        other_rollouts,
    )

    for action_idx, ego_rollout in enumerate(ego_rollouts):
        for step_idx, ego in enumerate(ego_rollout):
            for actor_idx, actor_rollout in enumerate(other_rollouts):
                actor = actor_rollout[step_idx]
                assert float(gap[action_idx, step_idx, actor_idx]) == (
                    pytest.approx(bbox_gap(ego, actor), abs=1.0e-12)
                )
                assert bool(overlap[action_idx, step_idx, actor_idx]) is (
                    geometric_overlap(ego, actor)
                )


def test_gap_overlap_array_expansion_matches_explicit_vehicle_boxes():
    ego_rollouts = [
        [_advanced(_state("ego", x=100.0, y=0.0), step) for step in range(4)]
    ]
    other_rollouts = [
        [_advanced(_state("other", x=112.0, y=0.2), step) for step in range(4)]
    ]
    length_expansion = np.asarray([[0.0, 0.4, 1.2, 2.4]], dtype=np.float64)
    width_expansion = np.asarray([[0.0, 0.1, 0.2, 0.3]], dtype=np.float64)
    explicit = [
        [
            VehicleState(
                **{
                    **state.to_dict(),
                    "length": float(state.length + length_expansion[0, step]),
                    "width": float(state.width + width_expansion[0, step]),
                }
            )
            for step, state in enumerate(other_rollouts[0])
        ]
    ]

    expanded_gap, expanded_overlap = batch_pairwise_obb_gap_overlap(
        ego_rollouts,
        other_rollouts,
        other_length_expansion=length_expansion,
        other_width_expansion=width_expansion,
    )
    explicit_gap, explicit_overlap = batch_pairwise_obb_gap_overlap(
        ego_rollouts,
        explicit,
    )

    np.testing.assert_allclose(
        expanded_gap,
        explicit_gap,
        rtol=0.0,
        atol=1.0e-12,
    )
    np.testing.assert_array_equal(expanded_overlap, explicit_overlap)


def test_gap_overlap_array_expansion_rejects_shape_mismatch():
    ego_rollouts = [[_state("ego", x=100.0, y=0.0)]]
    other_rollouts = [[_state("other", x=112.0, y=0.0)]]

    with pytest.raises(ValueError, match="length expansion"):
        batch_pairwise_obb_gap_overlap(
            ego_rollouts,
            other_rollouts,
            other_length_expansion=np.zeros((2, 1), dtype=np.float64),
        )
    with pytest.raises(ValueError, match="width expansion"):
        batch_pairwise_obb_gap_overlap(
            ego_rollouts,
            other_rollouts,
            other_width_expansion=np.zeros((1, 2), dtype=np.float64),
        )


def _candidate_context():
    cfg = load_config()
    cfg.risk_module["collision_horizon_steps"] = 8
    ego = _state("ego", x=500.0, y=53.8, speed=10.0, lane=0, lane_pos=190.0)
    front = _state("front", x=515.0, y=57.0, speed=12.0, lane=1, edge="main_in", lane_pos=205.0)
    rear = _state("rear", x=485.0, y=57.0, speed=19.0, lane=1, edge="main_in", lane_pos=175.0)
    return cfg, {
        "ego": ego,
        "vehicles": [ego, front, rear],
        "lane_count": 4,
        "config": cfg,
    }


def _assert_candidate_parity(batch: CandidateRiskSample, reference: CandidateRiskSample) -> None:
    assert batch.action == reference.action
    assert batch.candidate_legal is reference.candidate_legal
    assert batch.boundary_sample is reference.boundary_sample
    assert batch.ego_on_auxiliary is reference.ego_on_auxiliary
    assert np.array_equal(batch.risk_types, reference.risk_types)
    assert np.allclose(batch.features, reference.features, rtol=0.0, atol=1.0e-6)
    for name in (
        "overall_risk",
        "lane_oob",
        "traffic_risk",
        "continuous_risk_target",
        "distance_to_taper",
    ):
        _assert_scalar(float(getattr(batch, name)), float(getattr(reference, name)))
    for item in fields(batch.local_stats):
        actual = getattr(batch.local_stats, item.name)
        expected = getattr(reference.local_stats, item.name)
        if isinstance(expected, float):
            _assert_scalar(float(actual), expected)
        else:
            assert actual == expected


def test_vectorized_candidate_samples_match_scalar_reference_and_preserve_order_cache():
    _cfg, batch_context = _candidate_context()
    _cfg, reference_context = _candidate_context()
    requested = [ACTIONS[8], *ACTIONS, ACTIONS[8]]
    batch = evaluate_candidate_actions(requested, batch_context)
    reference = evaluate_candidate_actions_reference(requested, reference_context)
    assert [sample.action for sample in batch] == [8, *range(9), 8]
    assert batch[0] is batch[-1]
    for actual, expected in zip(batch, reference):
        _assert_candidate_parity(actual, expected)
    repeated = evaluate_candidate_actions(requested, batch_context)
    assert all(actual is cached for actual, cached in zip(batch, repeated))


def test_vectorized_candidate_path_does_not_reenter_scalar_step_metrics(monkeypatch):
    _cfg, context = _candidate_context()
    calls = 0
    original = merge_local_module.batch_pairwise_obb_metrics

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(merge_local_module, "batch_pairwise_obb_metrics", counted)
    monkeypatch.setattr(
        merge_local_module,
        "compute_step_metrics",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("vectorized path re-entered scalar step metrics")
        ),
    )
    samples = evaluate_candidate_actions(list(ACTIONS), context)
    assert len(samples) == len(ACTIONS)
    assert calls == 1


class _DeterministicRisk:
    def predict_many(self, actions, context):
        return [self.predict(action, context) for action in actions]

    @staticmethod
    def predict(action, context):
        value = float(action.index) / 10.0
        return type(
            "Prediction",
            (),
            {
                "risk_score": value,
                "risk_uncertainty": value / 2.0,
            },
        )()


def test_safety_shield_batch_api_matches_sequential_and_preserves_illegal_scores():
    cfg = load_config()
    cfg.shield["risk_threshold"] = 0.5
    cfg.shield["uncertainty_threshold"] = 0.3
    ego = _state("ego", x=100.0, y=0.0, lane=0)
    context = {"ego": ego, "vehicles": [ego], "lane_count": 1, "config": cfg}
    sequential_shield = SafetyShield(cfg, _DeterministicRisk())
    batch_shield = SafetyShield(cfg, _DeterministicRisk())
    actions = [ACTIONS[0], ACTIONS[4], ACTIONS[8]]
    sequential = [sequential_shield.evaluate_candidate(action, context) for action in actions]
    batch = batch_shield.evaluate_candidates(actions, context)
    assert batch == sequential
    assert batch[0]["candidate_legal"] is False
    assert batch[0]["veto_reason"] == "candidate_illegal"
    assert batch[0]["risk_score"] == pytest.approx(0.0)


def test_risk_batch_exposes_granular_latency_and_preserves_prediction_cache():
    cfg, context = _candidate_context()
    tracker = PerformanceTracker()
    context["performance_tracker"] = tracker
    wrapper = RiskModuleWrapper(cfg)
    first = wrapper.predict_many(list(ACTIONS), context)
    second = wrapper.predict_many(list(ACTIONS), context)
    assert all(actual is cached for actual, cached in zip(first, second))
    detail = context["_risk_last_latency"]
    assert {
        "candidate_features",
        "other_rollout",
        "ego_rollout",
        "pairwise_geometry",
        "merge_local_reduction",
        "network_forward",
        "result_pack",
    }.issubset(detail)
    summary = tracker.summary()
    assert summary["risk_candidate_features_time"] >= 0.0
    assert summary["risk_network_forward_time"] >= 0.0
    assert summary["operation_counts"]["risk_forwards"] == 1
