from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from safe_rl.accvp.serving.observation import (
    RISK_GATED_ACCVP_OBSERVATION_BOUNDED_STALE_VERSION,
    RiskGatedACCVPCandidateTableAugmentor,
    validate_accvp_observation_config,
)
from safe_rl.accvp.serving.predictor import (
    ACCVPCriticalActorOverflow,
    validate_runtime_actor_rows,
)
from safe_rl.accvp.training.reproducibility import configure_deterministic_training
from safe_rl.pipeline.accvp_observation_preflight import _gate
from safe_rl.risk import merge_local
from safe_rl.risk.risk_aggregator import aggregate_episode_reports
from safe_rl.risk.risk_module import RiskModuleWrapper
from safe_rl.sim.action_space import ACTIONS
from safe_rl.sim.sumo_highway_merge_env import SumoHighwayMergeEnv
from safe_rl.sim.types import VehicleState
from safe_rl.utils.config import clone_with_overrides, load_config


class _AlwaysSafeShield:
    def evaluate_candidate(self, action, context):
        return {
            "candidate_legal": True,
            "safety_pass": True,
            "risk_score": 0.1,
            "risk_uncertainty": 0.0,
            "veto_reason": "",
        }


class _FakeTorchProcessState:
    __version__ = "fake-torch"

    def __init__(self):
        self._deterministic = False
        self._threads = 8
        self.backends = SimpleNamespace(
            cudnn=SimpleNamespace(
                deterministic=False,
                benchmark=True,
                allow_tf32=True,
                version=lambda: None,
            ),
            cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True)),
        )
        self.cuda = SimpleNamespace(is_available=lambda: False)
        self.version = SimpleNamespace(cuda=None)

    def use_deterministic_algorithms(self, enabled):
        self._deterministic = bool(enabled)

    def are_deterministic_algorithms_enabled(self):
        return self._deterministic

    def get_num_threads(self):
        return self._threads

    def set_num_threads(self, value):
        self._threads = int(value)

    @staticmethod
    def get_num_interop_threads():
        return 2


class _FlakyPredictor:
    def __init__(self, failures: set[int] | None = None):
        self.calls = 0
        self.failures = set(failures or set())
        self.action_counts: list[int] = []

    def score_candidates(self, context, actions, *, timeout_s=None):
        self.calls += 1
        self.action_counts.append(len(actions))
        if self.calls in self.failures:
            raise TimeoutError("injected timeout")
        return [
            {
                "action_id": int(action.index),
                "p_merge_before_taper": 0.8,
                "target_lane_entry_time_s": 2.0,
                "ensemble_disagreement": 0.05,
            }
            for action in actions
        ]


def _config(*, warmup: bool = False):
    return clone_with_overrides(
        load_config(),
        {
            "forecast_features": {"enabled": False},
            "rl": {"use_wcdt_forecast_features": False},
            "accvp": {
                "enabled": False,
                "mode": "off",
                "checkpoint": "unused.pt",
                "risk_checkpoint": "unused-risk.pt",
                "activation_distance": 240.0,
                "observation": {
                    "enabled": True,
                    "feature_version": RISK_GATED_ACCVP_OBSERVATION_BOUNDED_STALE_VERSION,
                    "invalid_table_strategy": "bounded_last_valid_v2",
                    "activation_distance": 240.0,
                    "timeout_s": 0.5,
                    "fail_closed_defaults": True,
                    "include_risk_secondary": True,
                    "warmup_enabled": warmup,
                    "warmup_max_attempts": 3,
                    "last_valid_max_decisions": 1,
                    "last_valid_ttl_s": 0.5,
                    "allow_with_forecast_features": False,
                },
            },
        },
    )


def _context(
    decision: int,
    *,
    merge_distance: float = 50.0,
    front_id: str = "front",
    front_gap: float = 20.0,
):
    return {
        "episode_seed": 7,
        "episode_step": decision * 5,
        "decision_index": decision,
        "ego": SimpleNamespace(
            edge_id="aux",
            lane_id="aux_0",
            lane_index=0,
            speed=20.0,
            vehicle_id="ego",
        ),
        "merge_local": SimpleNamespace(
            ego_on_auxiliary=True,
            merge_distance=merge_distance,
            target_front_vehicle_id=front_id,
            target_rear_vehicle_id="rear",
            target_front_gap=front_gap,
            target_rear_gap=18.0,
        ),
        "candidate_legal_by_action": {int(action.index): True for action in ACTIONS},
    }


def _table(vector: np.ndarray) -> np.ndarray:
    return vector[: 9 * 11].reshape((9, 11))


def test_runtime_actor_rows_allow_masked_padding_but_reject_critical_overflow():
    runtime = {
        "actor_row_ids": ["front", "rear", "", ""],
        "mask": np.asarray([[1.0, 1.0, 0.0, 0.0]], dtype=np.float32),
        "actor_selection": SimpleNamespace(critical_overflow=False),
    }
    assert validate_runtime_actor_rows(runtime, 4) == 2

    runtime["actor_selection"] = SimpleNamespace(critical_overflow=True)
    with pytest.raises(ACCVPCriticalActorOverflow, match="critical actor coverage"):
        validate_runtime_actor_rows(runtime, 4)


def test_v3_feature_contract_is_107d_and_validates():
    cfg = _config()
    validate_accvp_observation_config(cfg)
    assert RiskGatedACCVPCandidateTableAugmentor.feature_dim(cfg) == 107
    names = RiskGatedACCVPCandidateTableAugmentor.feature_names(cfg)
    assert len(names) == 107
    assert names[-4:] == list(RiskGatedACCVPCandidateTableAugmentor.FRESHNESS_FEATURE_NAMES)


def test_v3_environment_contract_derives_total_159d_without_hardcoded_overlay_dim():
    cfg = _config()
    env = SumoHighwayMergeEnv(
        cfg,
        accvp_observation_augmentor=SimpleNamespace(),
    )
    try:
        assert env._base_obs_dim == 52
        assert env._accvp_observation_dim == 107
        assert env.observation_space.shape == (159,)
    finally:
        env.close()


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"invalid_table_strategy": "fail_closed_v1"}, "requires bounded_last_valid_v2"),
        ({"fail_closed_defaults": False}, "fail_closed_defaults=true"),
        ({"last_valid_max_decisions": 2}, "at most one stale decision"),
        ({"last_valid_ttl_s": 0.501}, "must not exceed 0.5 seconds"),
        ({"last_valid_max_gap_delta_m": 8.001}, "must be in"),
    ],
)
def test_v3_config_rejects_weaker_bounded_stale_contract(override, message):
    cfg = clone_with_overrides(
        _config(),
        {"accvp": {"observation": override}},
    )
    with pytest.raises(ValueError, match=message):
        validate_accvp_observation_config(cfg)


def test_bounded_stale_reuses_one_decision_with_conservative_safety_then_expires():
    predictor = _FlakyPredictor(failures={2, 3})
    augmentor = RiskGatedACCVPCandidateTableAugmentor(
        _config(), predictor=predictor, shield=_AlwaysSafeShield()
    )

    fresh = augmentor.extract(_context(0))
    stale = augmentor.extract(_context(1, merge_distance=49.0, front_gap=21.0))
    expired = augmentor.extract(_context(2, merge_distance=48.0, front_gap=22.0))

    assert fresh.shape == stale.shape == expired.shape == (107,)
    assert fresh[-4:].tolist() == [0.0, 0.0, 0.0, 0.0]
    assert stale[-4] == 1.0
    assert stale[-3] == 0.0
    assert stale[-2] == 1.0
    assert stale[-1] > 0.0
    assert np.all(_table(stale)[:, 0] == 0.0)
    assert np.all(_table(stale)[:, 2] == 0.0)
    assert np.all(_table(stale)[:, 3] == 1.0)
    assert np.all(_table(stale)[:, 6] == 1.0)
    assert np.allclose(_table(stale)[:, 4], _table(fresh)[:, 4])
    assert expired[-4:].tolist() == [0.0, 1.0, 1.0, 0.0]
    assert np.all(_table(expired)[:, 4] == 0.0)

    summary = augmentor.summary()
    assert summary["accvp_table_bounded_stale_reuse_count"] == 1
    assert summary["accvp_table_stale_expired_count"] == 1
    assert summary["accvp_table_max_consecutive_timeout_count"] == 2


def test_bounded_stale_rejects_changed_critical_actor_immediately():
    predictor = _FlakyPredictor(failures={2})
    augmentor = RiskGatedACCVPCandidateTableAugmentor(
        _config(), predictor=predictor, shield=_AlwaysSafeShield()
    )
    augmentor.extract(_context(0))
    changed = _context(1, front_id="new-front")
    changed["merge_local"] = vars(changed["merge_local"])
    changed["ego"] = vars(changed["ego"])
    rejected = augmentor.extract(changed)
    assert rejected[-4:].tolist() == [0.0, 1.0, 1.0, 1.0]
    summary = augmentor.summary()
    assert summary["accvp_table_bounded_stale_reuse_count"] == 0
    assert summary["accvp_table_stale_context_rejected_count"] == 1
    assert summary["accvp_table_stale_rejection_reasons"]["target_front_changed"] == 1


def test_bounded_stale_rejects_replayed_same_decision_context():
    predictor = _FlakyPredictor(failures={2})
    augmentor = RiskGatedACCVPCandidateTableAugmentor(
        _config(), predictor=predictor, shield=_AlwaysSafeShield()
    )
    augmentor.extract(_context(0))
    replayed = _context(0)
    replayed["episode_step"] = 1
    rejected = augmentor.extract(replayed)
    assert rejected[-4] == 0.0
    assert rejected[-3] == 1.0
    assert augmentor.summary()["accvp_table_stale_rejection_reasons"]["decision_order"] == 1


def test_string_sumo_lane_identity_is_preserved_in_stale_context_contract():
    predictor = _FlakyPredictor(failures={2})
    augmentor = RiskGatedACCVPCandidateTableAugmentor(
        _config(), predictor=predictor, shield=_AlwaysSafeShield()
    )
    fresh = augmentor.extract(_context(0))
    stale = augmentor.extract(_context(1))
    assert fresh.shape == stale.shape == (107,)
    assert stale[-4] == 1.0


def test_non_finite_runtime_output_is_rejected_before_reaching_observation():
    class _NaNPredictor(_FlakyPredictor):
        def score_candidates(self, context, actions, *, timeout_s=None):
            rows = super().score_candidates(context, actions, timeout_s=timeout_s)
            if self.calls == 2:
                rows[0]["p_merge_before_taper"] = float("nan")
            return rows

    augmentor = RiskGatedACCVPCandidateTableAugmentor(
        _config(), predictor=_NaNPredictor(), shield=_AlwaysSafeShield()
    )
    augmentor.extract(_context(0))
    stale = augmentor.extract(_context(1))
    assert np.isfinite(stale).all()
    assert stale[-4] == 1.0
    assert augmentor.summary()["accvp_table_invalid_output_count"] == 1


def test_warmup_retries_after_failure_and_covers_all_legal_actions():
    predictor = _FlakyPredictor(failures={1})
    augmentor = RiskGatedACCVPCandidateTableAugmentor(
        _config(warmup=True), predictor=predictor, shield=_AlwaysSafeShield()
    )

    augmentor.extract(_context(0))
    assert augmentor.summary()["accvp_observation_warmup_state"] == "failed"
    augmentor.extract(_context(1))
    summary = augmentor.summary()
    assert summary["accvp_observation_warmup_state"] == "ready"
    assert summary["accvp_observation_warmup_attempts"] == 2
    assert summary["accvp_table_warmup_error_count"] == 1
    assert summary["accvp_table_warmup_success_count"] == 1
    assert 9 in predictor.action_counts


def test_inactive_v3_table_is_neutral_not_a_hard_runtime_failure():
    predictor = _FlakyPredictor()
    augmentor = RiskGatedACCVPCandidateTableAugmentor(
        _config(), predictor=predictor, shield=_AlwaysSafeShield()
    )
    context = _context(0, merge_distance=300.0)
    table = augmentor.extract(context)
    assert table[-4:].tolist() == [0.0, 0.0, 0.0, 0.0]
    assert augmentor.summary()["accvp_table_hard_fail_closed_count"] == 0


def test_warmup_state_persists_across_episode_stats_reset_without_rewarming():
    predictor = _FlakyPredictor()
    augmentor = RiskGatedACCVPCandidateTableAugmentor(
        _config(warmup=True), predictor=predictor, shield=_AlwaysSafeShield()
    )
    augmentor.extract(_context(0))
    calls_after_warmup = predictor.calls
    augmentor.reset_episode_state()
    summary = augmentor.summary()
    assert summary["accvp_observation_warmup_state"] == "ready"
    assert summary["accvp_observation_warmup_ready"] is True
    assert summary["accvp_table_warmup_count"] == 0
    augmentor.extract(_context(0))
    assert predictor.calls == calls_after_warmup + 1


def test_risk_secondary_batches_only_legal_actions_and_fails_illegal_rows_closed():
    class _RiskModel:
        def __init__(self):
            self.action_batches: list[list[int]] = []

        def predict_many(self, actions, context):
            self.action_batches.append([int(action.index) for action in actions])
            return [
                SimpleNamespace(risk_score=0.1, risk_uncertainty=0.0)
                for _action in actions
            ]

    risk_model = _RiskModel()
    shield = SimpleNamespace(ranker=SimpleNamespace(risk_model=risk_model))
    context = _context(0)
    legal_ids = {4, 7}
    context["candidate_legal_by_action"] = {
        int(action.index): int(action.index) in legal_ids for action in ACTIONS
    }
    augmentor = RiskGatedACCVPCandidateTableAugmentor(
        _config(), predictor=_FlakyPredictor(), shield=shield
    )
    table = _table(augmentor.extract(context))
    assert risk_model.action_batches == [[4, 7]]
    assert np.all(table[[4, 7], 2] == 1.0)
    assert np.all(table[[4, 7], 3] == pytest.approx(0.1))
    illegal_ids = [index for index in range(9) if index not in legal_ids]
    assert np.all(table[illegal_ids, 2] == 0.0)
    assert np.all(table[illegal_ids, 3] == 1.0)


def _strict_metrics():
    return {
        "accvp_table_unique_episode_seed_count": 30,
        "accvp_table_missing_episode_seed_count": 0,
        "accvp_table_seed_schedule_match": True,
        "accvp_table_activation_window_decision_count": 1000,
        "accvp_table_valid_rate_activation_window": 0.995,
        "accvp_table_timeout_rate_activation_window": 0.005,
        "accvp_table_last_valid_fallback_rate_activation_window": 0.005,
        "accvp_table_hard_fail_closed_count": 0,
        "accvp_table_max_consecutive_timeout_count": 1,
        "accvp_table_model_error_count": 0,
        "accvp_table_invalid_bundle_count": 0,
        "accvp_table_invalid_output_count": 0,
        "accvp_table_runtime_context_error_count": 0,
        "accvp_table_critical_actor_overflow_count": 0,
        "accvp_table_unexpected_value_error_count": 0,
        "accvp_table_runtime_error_reasons": {},
        "accvp_table_warmup_error_count": 0,
        "accvp_table_warmup_ready_rate": 1.0,
        "accvp_table_latency_p95": 0.30,
        "accvp_table_latency_p99": 0.40,
        "accvp_table_latency_max": 0.50,
        "accvp_table_latency_per_stage": {"risk_secondary": {"p95": 0.15}},
    }


def test_strict_runtime_gate_enforces_tail_latency_and_failure_rates():
    passing = _gate(_strict_metrics())
    assert passing["profile"] == "bounded_stale_runtime_v3_strict"
    assert passing["pass"] is True
    failing_metrics = _strict_metrics()
    failing_metrics["accvp_table_latency_p99"] = 0.401
    failing_metrics["accvp_table_max_consecutive_timeout_count"] = 2
    failing = _gate(failing_metrics)
    assert failing["pass"] is False
    assert failing["checks"]["latency_p99_within_0_40s"] is False
    assert failing["checks"]["max_consecutive_timeouts"] is False

    context_failure = _strict_metrics()
    context_failure["accvp_table_critical_actor_overflow_count"] = 1
    assert _gate(context_failure)["pass"] is False


def test_strict_runtime_gate_cannot_pass_a_mismatched_runtime_contract():
    gate = _gate(
        _strict_metrics(),
        require_vnext=True,
        runtime_contract_check={"pass": False},
    )
    assert gate["pass"] is False
    assert gate["checks"]["formal_runtime_contract_match"] is False


def test_vnext_preflight_never_falls_back_to_legacy_gate_when_metrics_are_missing():
    incomplete = _strict_metrics()
    incomplete.pop("accvp_table_invalid_output_count")
    gate = _gate(incomplete, require_vnext=True)
    assert gate["profile"] == "bounded_stale_runtime_v3_strict"
    assert gate["pass"] is False
    assert gate["checks"]["required_metric_fields_present"] is False


def test_strict_runtime_gate_requires_independent_complete_seed_schedule():
    too_few = _strict_metrics()
    too_few["accvp_table_unique_episode_seed_count"] = 29
    assert _gate(too_few, require_vnext=True)["pass"] is False

    missing = _strict_metrics()
    missing["accvp_table_missing_episode_seed_count"] = 1
    assert _gate(missing, require_vnext=True)["pass"] is False

    mismatched = _strict_metrics()
    mismatched["accvp_table_seed_schedule_match"] = False
    assert _gate(mismatched, require_vnext=True)["pass"] is False


def test_runtime_aggregation_exposes_p99_activation_rates_and_warmup_readiness():
    reports = [
        {
            "seed": seed,
            "accvp_observation_warmup_enabled": True,
            "accvp_observation_warmup_ready": True,
        }
        for seed in range(29)
    ] + [
        {
            "seed": 29,
            "accvp_table_decision_count": 1000,
            "accvp_table_activation_window_decision_count": 1000,
            "accvp_table_valid_decision_count": 995,
            "accvp_table_activation_window_valid_decision_count": 995,
            "accvp_table_timeout_count": 5,
            "accvp_table_last_valid_fallback_count": 5,
            "accvp_table_bounded_stale_reuse_count": 5,
            "accvp_table_max_consecutive_timeout_count": 1,
            "accvp_table_latency_total_s": [0.1] * 1000,
            "accvp_table_latency_stage_s": {"risk_secondary": [0.05] * 1000},
            "accvp_observation_warmup_enabled": True,
            "accvp_observation_warmup_ready": True,
        }
    ]
    metrics = aggregate_episode_reports(reports)
    assert metrics["accvp_table_timeout_rate_activation_window"] == 0.005
    assert metrics["accvp_table_last_valid_fallback_rate_activation_window"] == 0.005
    assert metrics["accvp_table_latency_p99"] == 0.1
    assert metrics["accvp_table_warmup_ready_rate"] == 1.0
    assert metrics["accvp_table_unique_episode_seed_count"] == 30
    assert metrics["accvp_table_missing_episode_seed_count"] == 0
    assert metrics["accvp_table_runtime_gate_pass"] is True


def test_ego_rollout_config_fingerprint_is_shared_across_candidate_batch(monkeypatch):
    hash_calls = 0

    def _hash(config):
        nonlocal hash_calls
        hash_calls += 1
        return "fixed"

    monkeypatch.setattr(merge_local, "_ego_rollout_config_hash", _hash)
    monkeypatch.setattr(merge_local, "rollout_ego", lambda *args, **kwargs: (["state"], False))
    context = {"ego": object(), "config": object()}
    merge_local.get_cached_ego_rollout(context, ACTIONS[0], horizon_steps=8, dt=0.1)
    merge_local.get_cached_ego_rollout(context, ACTIONS[1], horizon_steps=8, dt=0.1)
    assert hash_calls == 1


def test_candidate_action_reference_prepares_all_rollouts_before_metric_evaluation(monkeypatch):
    order: list[str] = []
    prepared = SimpleNamespace(ego=object(), samples={}, ego_rollouts={}, horizon_steps=8, dt=0.1)
    monkeypatch.setattr(merge_local, "prepare_candidate_rollout_context", lambda context: prepared)

    def _rollout(context, action, **kwargs):
        order.append(f"rollout-{action.index}")
        return ["state"], False

    def _evaluate(action, context):
        order.append(f"evaluate-{action.index}")
        return int(action.index)

    monkeypatch.setattr(merge_local, "get_cached_ego_rollout", _rollout)
    monkeypatch.setattr(merge_local, "evaluate_candidate_action_risk", _evaluate)
    actions = [ACTIONS[0], ACTIONS[1], ACTIONS[2]]
    assert merge_local.evaluate_candidate_actions_reference(actions, {}) == [0, 1, 2]
    assert order == [
        "rollout-0",
        "rollout-1",
        "rollout-2",
        "evaluate-0",
        "evaluate-1",
        "evaluate-2",
    ]


def test_risk_predict_many_is_numerically_equivalent_to_sequential_legal_predictions():
    cfg = load_config()
    ego = VehicleState("ego", 500.0, 53.8, 0.0, 10.0, 0, "main_aux_0", 190.0, "main_aux")
    vehicle = VehicleState("other", 515.0, 57.0, 0.0, 12.0, 1, "main_1", 205.0, "main_in")
    actions = [ACTIONS[4], ACTIONS[7]]
    batch_context = {"ego": ego, "vehicles": [ego, vehicle], "lane_count": 4, "config": cfg}
    sequential_context = {"ego": ego, "vehicles": [ego, vehicle], "lane_count": 4, "config": cfg}
    batch = RiskModuleWrapper(cfg).predict_many(actions, batch_context)
    sequential_wrapper = RiskModuleWrapper(cfg)
    sequential = [sequential_wrapper.predict(action, sequential_context) for action in actions]
    for batch_prediction, sequential_prediction in zip(batch, sequential):
        assert batch_prediction.risk_score == pytest.approx(
            sequential_prediction.risk_score, abs=1.0e-12
        )
        assert batch_prediction.risk_uncertainty == pytest.approx(
            sequential_prediction.risk_uncertainty, abs=1.0e-12
        )
        assert np.array_equal(
            batch_prediction.risk_type_logits,
            sequential_prediction.risk_type_logits,
        )
        assert np.array_equal(
            batch_prediction.explicit_features,
            sequential_prediction.explicit_features,
        )


def test_deterministic_training_restores_process_baseline_for_shadow(monkeypatch):
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    torch = _FakeTorchProcessState()
    formal = configure_deterministic_training(torch, enabled=True, torch_threads=1)
    assert formal["deterministic_algorithms"] is True
    assert torch.get_num_threads() == 1
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False
    assert torch.backends.cudnn.allow_tf32 is False
    assert torch.backends.cuda.matmul.allow_tf32 is False
    assert formal["cuda_initialized_before_configuration"] is False
    assert formal["cudnn_allow_tf32"] is False
    assert formal["cuda_matmul_allow_tf32"] is False
    assert formal["process_baseline"]["cudnn_allow_tf32"] is True

    shadow = configure_deterministic_training(torch, enabled=False)
    assert shadow["deterministic_algorithms"] is False
    assert torch.get_num_threads() == 8
    assert torch.backends.cudnn.deterministic is False
    assert torch.backends.cudnn.benchmark is True
    assert torch.backends.cudnn.allow_tf32 is True
    assert torch.backends.cuda.matmul.allow_tf32 is True
    assert shadow["cublas_workspace_config"] == ""


def test_formal_determinism_rejects_late_cuda_initialization(monkeypatch):
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    torch = _FakeTorchProcessState()
    torch.cuda = SimpleNamespace(
        is_available=lambda: True,
        is_initialized=lambda: True,
    )

    with __import__("pytest").raises(RuntimeError, match="already been initialized"):
        configure_deterministic_training(torch, enabled=True, torch_threads=1)
