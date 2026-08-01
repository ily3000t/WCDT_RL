"""Candidate Table observation construction and bounded-stale state."""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import time
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any

import numpy as np

from safe_rl.accvp.contracts.protocol import effective_activation_distance
from safe_rl.accvp.serving.predictor import (
    ACCVPCriticalActorOverflow,
    ACCVPRuntimeContextError,
    ACCVPRuntimePredictor,
)
from safe_rl.accvp.contracts.schema import file_sha256
from safe_rl.risk.merge_local import is_candidate_legal
from safe_rl.risk.risk_module import RiskModuleWrapper
from safe_rl.shield.safety_shield import SafetyShield
from safe_rl.sim.action_space import ACTIONS, CandidateAction
from safe_rl.utils.config import clone_with_overrides


RISK_GATED_ACCVP_OBSERVATION_VERSION = "risk_gated_candidate_table_v1"
RISK_GATED_ACCVP_OBSERVATION_PERSISTENCE_VERSION = "risk_gated_candidate_table_v2_persistence"
RISK_GATED_ACCVP_OBSERVATION_BOUNDED_STALE_VERSION = "risk_gated_candidate_table_v3_bounded_stale"


class ACCVPInvalidRuntimeOutput(ValueError):
    pass


@dataclass(frozen=True)
class ACCVPTableContextSnapshot:
    episode_seed: int
    decision_index: int
    episode_step: int
    simulation_time_s: float
    ego_edge_id: str
    ego_lane_id: str
    ego_speed: float
    ego_on_auxiliary: bool
    merge_distance: float
    target_front_vehicle_id: str
    target_rear_vehicle_id: str
    target_front_gap: float
    target_rear_gap: float
    legality_mask: tuple[bool, ...]


@dataclass(frozen=True)
class ACCVPCachedCandidateTable:
    value: np.ndarray
    context: ACCVPTableContextSnapshot


@dataclass(frozen=True)
class ACCVPTableFreshness:
    stale_fallback_active: bool = False
    hard_default_active: bool = False
    age_norm: float = 0.0
    context_delta_norm: float = 0.0


@dataclass
class ACCVPObservationStats:
    total_decisions: int = 0
    activation_window_decisions: int = 0
    valid_decisions: int = 0
    activation_window_valid_decisions: int = 0
    timeout_count: int = 0
    model_error_count: int = 0
    invalid_bundle_count: int = 0
    invalid_output_count: int = 0
    runtime_context_error_count: int = 0
    critical_actor_overflow_count: int = 0
    risk_safety_actor_coverage_incomplete_count: int = 0
    unexpected_value_error_count: int = 0
    fail_closed_count: int = 0
    hard_fail_closed_count: int = 0
    last_valid_fallback_count: int = 0
    bounded_stale_reuse_count: int = 0
    stale_expired_count: int = 0
    stale_context_rejected_count: int = 0
    no_prior_fallback_count: int = 0
    invalid_table_dropout_count: int = 0
    warmup_count: int = 0
    warmup_success_count: int = 0
    warmup_error_count: int = 0
    consecutive_timeout_count: int = 0
    max_consecutive_timeout_count: int = 0
    stale_rejection_reasons: dict[str, int] = field(default_factory=dict)
    runtime_error_reasons: dict[str, int] = field(default_factory=dict)
    critical_actor_overflow_histogram: dict[str, int] = field(default_factory=dict)
    critical_actor_overflow_examples: list[dict[str, Any]] = field(default_factory=list)
    critical_actor_overflow_sample_limit: int = 20
    stale_age_s: list[float] = field(default_factory=list)
    stale_context_delta_norm: list[float] = field(default_factory=list)
    warmup_latency_s: list[float] = field(default_factory=list)
    latency_total_s: list[float] = field(default_factory=list)
    latency_context_legality_s: list[float] = field(default_factory=list)
    latency_accvp_prepare_candidates_s: list[float] = field(default_factory=list)
    latency_accvp_score_s: list[float] = field(default_factory=list)
    latency_risk_secondary_s: list[float] = field(default_factory=list)
    latency_risk_context_s: list[float] = field(default_factory=list)
    latency_risk_predict_many_s: list[float] = field(default_factory=list)
    latency_risk_candidate_features_s: list[float] = field(default_factory=list)
    latency_risk_other_rollout_s: list[float] = field(default_factory=list)
    latency_risk_ego_rollout_s: list[float] = field(default_factory=list)
    latency_risk_pairwise_geometry_s: list[float] = field(default_factory=list)
    latency_risk_merge_local_reduction_s: list[float] = field(default_factory=list)
    latency_risk_network_forward_s: list[float] = field(default_factory=list)
    latency_risk_result_pack_s: list[float] = field(default_factory=list)
    latency_table_pack_s: list[float] = field(default_factory=list)

    @staticmethod
    def _latency_summary(values: list[float]) -> dict[str, float | None]:
        if not values:
            return {"p50": None, "p95": None, "p99": None, "max": None}
        arr = np.asarray(values, dtype=np.float64)
        return {
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
            "max": float(np.max(arr)),
        }

    def to_dict(self) -> dict[str, Any]:
        total_latency = self._latency_summary(self.latency_total_s)
        warmup_latency = self._latency_summary(self.warmup_latency_s)
        per_stage = {
            "context_legality": self._latency_summary(self.latency_context_legality_s),
            "accvp_prepare_candidates": self._latency_summary(self.latency_accvp_prepare_candidates_s),
            "accvp_score": self._latency_summary(self.latency_accvp_score_s),
            "risk_secondary": self._latency_summary(self.latency_risk_secondary_s),
            "risk_context": self._latency_summary(self.latency_risk_context_s),
            "risk_predict_many": self._latency_summary(self.latency_risk_predict_many_s),
            "risk_candidate_features": self._latency_summary(self.latency_risk_candidate_features_s),
            "risk_other_rollout": self._latency_summary(self.latency_risk_other_rollout_s),
            "risk_ego_rollout": self._latency_summary(self.latency_risk_ego_rollout_s),
            "risk_pairwise_geometry": self._latency_summary(self.latency_risk_pairwise_geometry_s),
            "risk_merge_local_reduction": self._latency_summary(self.latency_risk_merge_local_reduction_s),
            "risk_network_forward": self._latency_summary(self.latency_risk_network_forward_s),
            "risk_result_pack": self._latency_summary(self.latency_risk_result_pack_s),
            "table_pack": self._latency_summary(self.latency_table_pack_s),
        }
        return {
            "accvp_table_decision_count": int(self.total_decisions),
            "accvp_table_activation_window_decision_count": int(self.activation_window_decisions),
            "accvp_table_valid_decision_count": int(self.valid_decisions),
            "accvp_table_activation_window_valid_decision_count": int(self.activation_window_valid_decisions),
            "accvp_table_valid_rate_all_decisions": (
                float(self.valid_decisions / self.total_decisions) if self.total_decisions else 0.0
            ),
            "accvp_table_valid_rate_activation_window": (
                float(self.activation_window_valid_decisions / self.activation_window_decisions)
                if self.activation_window_decisions
                else 0.0
            ),
            "accvp_table_timeout_count": int(self.timeout_count),
            "accvp_table_model_error_count": int(self.model_error_count),
            "accvp_table_invalid_bundle_count": int(self.invalid_bundle_count),
            "accvp_table_invalid_output_count": int(self.invalid_output_count),
            "accvp_table_runtime_context_error_count": int(self.runtime_context_error_count),
            "accvp_table_critical_actor_overflow_count": int(self.critical_actor_overflow_count),
            "accvp_table_task_actor_overflow_count": int(
                self.critical_actor_overflow_count
            ),
            "accvp_table_risk_safety_actor_coverage_incomplete_count": int(
                self.risk_safety_actor_coverage_incomplete_count
            ),
            "accvp_table_critical_actor_overflow_histogram": dict(
                self.critical_actor_overflow_histogram
            ),
            "accvp_table_critical_actor_overflow_examples": list(
                self.critical_actor_overflow_examples[
                    : max(0, int(self.critical_actor_overflow_sample_limit))
                ]
            ),
            "accvp_table_critical_actor_overflow_sample_limit": int(
                self.critical_actor_overflow_sample_limit
            ),
            "accvp_table_unexpected_value_error_count": int(self.unexpected_value_error_count),
            "accvp_table_runtime_error_reasons": dict(self.runtime_error_reasons),
            "accvp_table_fail_closed_count": int(self.fail_closed_count),
            "accvp_table_hard_fail_closed_count": int(self.hard_fail_closed_count),
            "accvp_table_last_valid_fallback_count": int(self.last_valid_fallback_count),
            "accvp_table_bounded_stale_reuse_count": int(self.bounded_stale_reuse_count),
            "accvp_table_stale_expired_count": int(self.stale_expired_count),
            "accvp_table_stale_context_rejected_count": int(self.stale_context_rejected_count),
            "accvp_table_stale_rejection_reasons": dict(self.stale_rejection_reasons),
            "accvp_table_stale_age_s": list(self.stale_age_s),
            "accvp_table_stale_context_delta_norm": list(self.stale_context_delta_norm),
            "accvp_table_no_prior_fallback_count": int(self.no_prior_fallback_count),
            "accvp_table_invalid_dropout_count": int(self.invalid_table_dropout_count),
            "accvp_table_warmup_count": int(self.warmup_count),
            "accvp_table_warmup_success_count": int(self.warmup_success_count),
            "accvp_table_warmup_error_count": int(self.warmup_error_count),
            "accvp_table_consecutive_timeout_count": int(self.consecutive_timeout_count),
            "accvp_table_max_consecutive_timeout_count": int(self.max_consecutive_timeout_count),
            "accvp_table_timeout_rate": (
                float(self.timeout_count / self.total_decisions) if self.total_decisions else 0.0
            ),
            "accvp_table_timeout_rate_activation_window": (
                float(self.timeout_count / self.activation_window_decisions)
                if self.activation_window_decisions
                else 0.0
            ),
            "accvp_table_fail_closed_rate": (
                float(self.fail_closed_count / self.total_decisions) if self.total_decisions else 0.0
            ),
            "accvp_table_hard_fail_closed_rate": (
                float(self.hard_fail_closed_count / self.total_decisions) if self.total_decisions else 0.0
            ),
            "accvp_table_last_valid_fallback_rate": (
                float(self.last_valid_fallback_count / self.total_decisions) if self.total_decisions else 0.0
            ),
            "accvp_table_last_valid_fallback_rate_activation_window": (
                float(self.last_valid_fallback_count / self.activation_window_decisions)
                if self.activation_window_decisions
                else 0.0
            ),
            "accvp_table_invalid_dropout_rate": (
                float(self.invalid_table_dropout_count / self.total_decisions) if self.total_decisions else 0.0
            ),
            "accvp_table_latency_count": int(len(self.latency_total_s)),
            "accvp_table_latency_p50": total_latency["p50"],
            "accvp_table_latency_p95": total_latency["p95"],
            "accvp_table_latency_p99": total_latency["p99"],
            "accvp_table_latency_max": total_latency["max"],
            "accvp_table_warmup_latency_p50": warmup_latency["p50"],
            "accvp_table_warmup_latency_p95": warmup_latency["p95"],
            "accvp_table_warmup_latency_p99": warmup_latency["p99"],
            "accvp_table_warmup_latency_max": warmup_latency["max"],
            "accvp_table_latency_per_stage": per_stage,
            "accvp_table_runtime_gate_pass": bool(
                self.activation_window_decisions >= 1000
                and (
                    float(self.activation_window_valid_decisions / self.activation_window_decisions)
                    if self.activation_window_decisions
                    else 0.0
                )
                >= 0.995
                and (
                    float(self.timeout_count / self.activation_window_decisions)
                    if self.activation_window_decisions
                    else 1.0
                )
                <= 0.005
                and (
                    float(self.last_valid_fallback_count / self.activation_window_decisions)
                    if self.activation_window_decisions
                    else 1.0
                )
                <= 0.005
                and int(self.hard_fail_closed_count) == 0
                and int(self.max_consecutive_timeout_count) <= 1
                and int(self.model_error_count) == 0
                and int(self.invalid_bundle_count) == 0
                and int(self.invalid_output_count) == 0
                and int(self.runtime_context_error_count) == 0
                and int(self.critical_actor_overflow_count) == 0
                and int(
                    self.risk_safety_actor_coverage_incomplete_count
                )
                == 0
                and int(self.unexpected_value_error_count) == 0
                and int(self.warmup_error_count) == 0
                and (total_latency["p95"] is not None and float(total_latency["p95"]) <= 0.30)
                and (total_latency["p99"] is not None and float(total_latency["p99"]) <= 0.40)
                and (total_latency["max"] is not None and float(total_latency["max"]) <= 0.50)
                and (
                    per_stage["risk_secondary"]["p95"] is not None
                    and float(per_stage["risk_secondary"]["p95"]) <= 0.15
                )
            ),
            "accvp_table_latency_total_s": list(self.latency_total_s),
            "accvp_table_warmup_latency_s": list(self.warmup_latency_s),
            "accvp_table_latency_stage_s": {
                "context_legality": list(self.latency_context_legality_s),
                "accvp_prepare_candidates": list(self.latency_accvp_prepare_candidates_s),
                "accvp_score": list(self.latency_accvp_score_s),
                "risk_secondary": list(self.latency_risk_secondary_s),
                "risk_context": list(self.latency_risk_context_s),
                "risk_predict_many": list(self.latency_risk_predict_many_s),
                "risk_candidate_features": list(self.latency_risk_candidate_features_s),
                "risk_other_rollout": list(self.latency_risk_other_rollout_s),
                "risk_ego_rollout": list(self.latency_risk_ego_rollout_s),
                "risk_pairwise_geometry": list(self.latency_risk_pairwise_geometry_s),
                "risk_merge_local_reduction": list(self.latency_risk_merge_local_reduction_s),
                "risk_network_forward": list(self.latency_risk_network_forward_s),
                "risk_result_pack": list(self.latency_risk_result_pack_s),
                "table_pack": list(self.latency_table_pack_s),
            },
        }


class RiskGatedACCVPCandidateTableAugmentor:
    """Build PPO observation features from ACCVP task viability and Risk secondary checks.

    The table is intentionally not a pure ACCVP output.  It combines ACCVP's
    action-conditioned merge-before-taper estimate with Risk Module secondary
    safety signals and explicit legality/action-identity masks.
    """

    FEATURE_NAMES = (
        "table_valid",
        "candidate_legal",
        "risk_secondary_pass",
        "secondary_risk_score",
        "p_merge_before_taper",
        "target_entry_time_norm",
        "ensemble_disagreement",
        "is_left_action",
        "is_keep_action",
        "lateral_cmd_norm",
        "accel_cmd_norm",
    )
    POLICY_STATE_FEATURE_NAMES = (
        "policy_commitment_active",
        "policy_commitment_remaining_norm",
        "last_raw_lateral_cmd_norm",
        "last_raw_accel_cmd_norm",
    )
    FRESHNESS_FEATURE_NAMES = (
        "stale_fallback_active",
        "hard_default_active",
        "table_age_norm",
        "context_delta_norm",
    )
    ACTION_COUNT = len(ACTIONS)
    FEATURE_VERSION = RISK_GATED_ACCVP_OBSERVATION_VERSION
    PERSISTENCE_FEATURE_VERSION = RISK_GATED_ACCVP_OBSERVATION_PERSISTENCE_VERSION
    BOUNDED_STALE_FEATURE_VERSION = RISK_GATED_ACCVP_OBSERVATION_BOUNDED_STALE_VERSION

    def __init__(
        self,
        config: Any,
        *,
        predictor: Any | None = None,
        shield: Any | None = None,
        risk_checkpoint: str | Path | None = None,
    ) -> None:
        self.config = config
        validate_accvp_observation_config(config)
        obs_cfg = dict(config.accvp.get("observation", {}) or {})
        feature_version = str(obs_cfg.get("feature_version", self.FEATURE_VERSION))
        if feature_version not in {
            self.FEATURE_VERSION,
            self.PERSISTENCE_FEATURE_VERSION,
            self.BOUNDED_STALE_FEATURE_VERSION,
        }:
            raise ValueError(
                f"unsupported ACCVP observation feature_version={feature_version!r}; "
                f"expected {self.FEATURE_VERSION!r}, {self.PERSISTENCE_FEATURE_VERSION!r}, "
                f"or {self.BOUNDED_STALE_FEATURE_VERSION!r}"
            )
        self.feature_version = feature_version
        self.timeout_s = float(obs_cfg.get("timeout_s", config.accvp.get("max_decision_latency_s", 0.5)))
        self.use_inference_worker = bool(obs_cfg.get("use_inference_worker", False))
        self.profile_latency = bool(obs_cfg.get("profile_latency", True))
        self.fail_closed_defaults = bool(obs_cfg.get("fail_closed_defaults", True))
        default_invalid_strategy = (
            "bounded_last_valid_v2"
            if feature_version == self.BOUNDED_STALE_FEATURE_VERSION
            else "fail_closed_v1"
        )
        self.invalid_table_strategy = str(obs_cfg.get("invalid_table_strategy", default_invalid_strategy))
        if self.invalid_table_strategy not in {
            "fail_closed_v1",
            "last_valid_with_invalid_mask_v1",
            "bounded_last_valid_v2",
        }:
            raise ValueError(
                "accvp.observation.invalid_table_strategy must be fail_closed_v1, "
                "last_valid_with_invalid_mask_v1, or bounded_last_valid_v2"
            )
        if (
            self.feature_version == self.BOUNDED_STALE_FEATURE_VERSION
            and self.invalid_table_strategy == "last_valid_with_invalid_mask_v1"
        ):
            raise ValueError("bounded-stale observation cannot use unbounded last-valid fallback")
        self.invalid_table_dropout_rate = float(obs_cfg.get("invalid_table_dropout_rate", 0.0) or 0.0)
        if not 0.0 <= self.invalid_table_dropout_rate <= 1.0:
            raise ValueError("accvp.observation.invalid_table_dropout_rate must be in [0, 1]")
        self.warmup_enabled = bool(obs_cfg.get("warmup_enabled", False))
        self.warmup_max_attempts = max(1, int(obs_cfg.get("warmup_max_attempts", 3)))
        self.include_risk_secondary = bool(obs_cfg.get("include_risk_secondary", True))
        self.secondary_safety_profile = str(obs_cfg.get("secondary_safety_profile", "risk_model_rollout_v1"))
        self.risk_horizon_steps = int(obs_cfg.get("risk_horizon_steps", 0) or 0)
        self.activation_distance = float(obs_cfg.get("activation_distance", effective_activation_distance(config)))
        self.viability_horizon_s = float(config.accvp.get("viability_horizon_s", 8.0))
        self.last_valid_max_decisions = max(0, int(obs_cfg.get("last_valid_max_decisions", 1)))
        self.last_valid_ttl_s = max(0.0, float(obs_cfg.get("last_valid_ttl_s", 0.5)))
        self.last_valid_max_merge_distance_delta_m = max(
            0.0, float(obs_cfg.get("last_valid_max_merge_distance_delta_m", 15.0))
        )
        self.last_valid_max_ego_speed_delta_mps = max(
            0.0, float(obs_cfg.get("last_valid_max_ego_speed_delta_mps", 3.0))
        )
        self.last_valid_max_gap_delta_m = max(
            0.0, float(obs_cfg.get("last_valid_max_gap_delta_m", 8.0))
        )
        self.critical_actor_overflow_sample_limit = max(
            0,
            int(obs_cfg.get("critical_actor_overflow_sample_limit", 20)),
        )
        self.stats = ACCVPObservationStats(
            critical_actor_overflow_sample_limit=self.critical_actor_overflow_sample_limit
        )
        self._cache_key: tuple[Any, ...] | None = None
        self._cache_value: np.ndarray | None = None
        self._last_valid_value: np.ndarray | None = None
        self._last_valid_snapshot: ACCVPCachedCandidateTable | None = None
        self._warmup_state = "disabled" if not self.warmup_enabled else "not_started"
        self._warmup_attempts = 0
        self._risk_observation_config: Any | None = None
        self._predictor = predictor
        self._shield = shield
        if self._predictor is None:
            checkpoint = config.accvp.get("checkpoint")
            if not checkpoint:
                raise FileNotFoundError("accvp.observation.enabled requires accvp.checkpoint")
            self._predictor = ACCVPRuntimePredictor(
                config,
                checkpoint,
                use_inference_worker=self.use_inference_worker,
            )
            if config.accvp.get("artifact_manifest"):
                self._predictor.validate_artifact_bundle(config.accvp.get("operating_point"))
        if self.include_risk_secondary and self._shield is None:
            checkpoint = risk_checkpoint or config.accvp.get("risk_checkpoint")
            if not checkpoint:
                raise FileNotFoundError("Risk-gated ACCVP observation requires accvp.risk_checkpoint")
            risk_model = RiskModuleWrapper(config, checkpoint=str(checkpoint))
            self._shield = SafetyShield(config, risk_model)
        if self.secondary_safety_profile == "short_horizon_risk_v1" and self.risk_horizon_steps > 0:
            current_steps = int(self.config.risk_module.get("collision_horizon_steps", 30))
            if self.risk_horizon_steps != current_steps:
                self._risk_observation_config = clone_with_overrides(
                    self.config,
                    {"risk_module": {"collision_horizon_steps": int(self.risk_horizon_steps)}},
                )

    @classmethod
    def feature_dim(cls, config: Any | None = None) -> int:
        table_dim = cls.ACTION_COUNT * len(cls.FEATURE_NAMES)
        if config is not None:
            obs_cfg = dict(config.accvp.get("observation", {}) or {})
            feature_version = str(obs_cfg.get("feature_version", cls.FEATURE_VERSION))
            if feature_version == cls.PERSISTENCE_FEATURE_VERSION:
                return table_dim + len(cls.POLICY_STATE_FEATURE_NAMES)
            if feature_version == cls.BOUNDED_STALE_FEATURE_VERSION:
                return table_dim + len(cls.POLICY_STATE_FEATURE_NAMES) + len(cls.FRESHNESS_FEATURE_NAMES)
        return table_dim

    @classmethod
    def feature_names(cls, config: Any | None = None) -> list[str]:
        names: list[str] = []
        for action in ACTIONS:
            for feature in cls.FEATURE_NAMES:
                names.append(f"action_{action.index}_{feature}")
        if config is not None:
            obs_cfg = dict(config.accvp.get("observation", {}) or {})
            feature_version = str(obs_cfg.get("feature_version", cls.FEATURE_VERSION))
            if feature_version in {cls.PERSISTENCE_FEATURE_VERSION, cls.BOUNDED_STALE_FEATURE_VERSION}:
                names.extend(cls.POLICY_STATE_FEATURE_NAMES)
            if feature_version == cls.BOUNDED_STALE_FEATURE_VERSION:
                names.extend(cls.FRESHNESS_FEATURE_NAMES)
        return names

    @staticmethod
    def enabled(config: Any) -> bool:
        return bool(config.accvp.get("observation", {}).get("enabled", False))

    def reset_episode_state(self) -> None:
        self.stats = ACCVPObservationStats(
            critical_actor_overflow_sample_limit=self.critical_actor_overflow_sample_limit
        )
        self._cache_key = None
        self._cache_value = None
        self._last_valid_value = None
        self._last_valid_snapshot = None

    def close(self) -> None:
        close = getattr(self._predictor, "close", None)
        if callable(close):
            close()

    def summary(self) -> dict[str, Any]:
        feature_names = self.feature_names(self.config)
        feature_contract = {
            "feature_version": self.feature_version,
            "feature_dim": int(self.feature_dim(self.config)),
            "feature_names": feature_names,
        }
        feature_contract_hash = hashlib.sha256(
            json.dumps(feature_contract, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        payload = self.stats.to_dict()
        payload.update(
            {
                "accvp_observation_enabled": True,
                "accvp_observation_feature_version": self.feature_version,
                "accvp_observation_dim": int(self.feature_dim(self.config)),
                "accvp_observation_feature_names": feature_names,
                "accvp_observation_feature_contract_hash": feature_contract_hash,
                "accvp_observation_activation_distance_m": float(self.activation_distance),
                "accvp_observation_use_inference_worker": bool(self.use_inference_worker),
                "accvp_observation_profile_latency": bool(self.profile_latency),
                "accvp_observation_invalid_table_strategy": str(self.invalid_table_strategy),
                "accvp_observation_invalid_table_dropout_rate": float(self.invalid_table_dropout_rate),
                "accvp_observation_warmup_enabled": bool(self.warmup_enabled),
                "accvp_observation_warmup_state": str(self._warmup_state),
                "accvp_observation_warmup_ready": bool(
                    not self.warmup_enabled or self._warmup_state == "ready"
                ),
                "accvp_observation_warmup_attempts": int(self._warmup_attempts),
                "accvp_observation_last_valid_max_decisions": int(self.last_valid_max_decisions),
                "accvp_observation_last_valid_ttl_s": float(self.last_valid_ttl_s),
                "accvp_observation_secondary_safety_profile": str(self.secondary_safety_profile),
                "accvp_observation_risk_horizon_steps": int(self.risk_horizon_steps),
                "accvp_observation_checkpoint": str(self.config.accvp.get("checkpoint", "")),
                "accvp_observation_artifact_manifest": str(self.config.accvp.get("artifact_manifest", "")),
                "accvp_observation_risk_checkpoint": str(self.config.accvp.get("risk_checkpoint", "")),
            }
        )
        payload["accvp_table_runtime_gate_profile"] = "bounded_stale_runtime_v3_strict"
        payload["accvp_table_runtime_gate_pass"] = bool(
            payload.get("accvp_table_runtime_gate_pass", False)
            and (not self.warmup_enabled or self._warmup_state == "ready")
        )
        for key, path in (
            ("accvp_artifact_manifest_sha256", self.config.accvp.get("artifact_manifest")),
            ("risk_checkpoint_sha256", self.config.accvp.get("risk_checkpoint")),
        ):
            try:
                payload[key] = file_sha256(path) if path else ""
            except Exception:
                payload[key] = ""
        return payload

    def extract(self, context: dict[str, Any]) -> np.ndarray:
        key = (
            int(context.get("episode_seed", -1)),
            int(context.get("episode_step", -1)),
            int(context.get("decision_index", -1)),
        )
        if self._cache_key == key and self._cache_value is not None:
            return self._cache_value.copy()
        self.warmup(context)
        table = self._extract_uncached(context)
        self._cache_key = key
        self._cache_value = table.astype(np.float32, copy=True)
        return self._cache_value.copy()

    def warmup(self, context: dict[str, Any] | None = None) -> None:
        if not self.warmup_enabled or self._warmup_state in {"ready", "running", "exhausted"}:
            return
        if context is None:
            return
        if self._warmup_attempts >= self.warmup_max_attempts:
            self._warmup_state = "exhausted"
            return
        started = time.perf_counter()
        record_attempt = True
        self._warmup_state = "running"
        self._warmup_attempts += 1
        self.stats.warmup_count += 1
        try:
            warmup_context = dict(context)
            for key in (
                "_candidate_rollout_context",
                "_ego_rollout_cache",
                "_ego_rollout_config_hash_cache",
                "_risk_prediction_cache",
            ):
                warmup_context.pop(key, None)
            legality = self._legality_map(warmup_context)
            legal_actions = [action for action in ACTIONS if bool(legality.get(int(action.index), False))]
            if not legal_actions:
                self._warmup_state = "waiting_context"
                self._warmup_attempts -= 1
                self.stats.warmup_count -= 1
                record_attempt = False
                return
            if (
                not self.use_inference_worker
                and hasattr(self._predictor, "prepare_candidates")
                and hasattr(self._predictor, "score_prepared")
            ):
                prepared = self._predictor.prepare_candidates(warmup_context, legal_actions)
                warmup_scores = self._predictor.score_prepared(prepared)
            else:
                scorer = self._predictor.score_candidates
                if "timeout_s" in inspect.signature(scorer).parameters:
                    warmup_scores = scorer(warmup_context, legal_actions, timeout_s=self.timeout_s)
                else:
                    warmup_scores = scorer(warmup_context, legal_actions)
            self._validate_scores(warmup_scores, legal_actions)
            self._secondary_safety_many(ACTIONS, warmup_context, legality)
            self._warmup_state = "ready"
            self.stats.warmup_success_count += 1
        except Exception:
            self._warmup_state = (
                "exhausted" if self._warmup_attempts >= self.warmup_max_attempts else "failed"
            )
            self.stats.warmup_error_count += 1
        finally:
            if self.profile_latency and record_attempt:
                self.stats.warmup_latency_s.append(float(time.perf_counter() - started))

    def _extract_uncached(self, context: dict[str, Any]) -> np.ndarray:
        self.stats.total_decisions += 1
        active = self._activation_window(context)
        self.stats.activation_window_decisions += int(active)
        if not active:
            self.stats.consecutive_timeout_count = 0
            # Outside the activation window the all-invalid table is the
            # expected neutral representation, not a runtime hard failure.
            return self._fail_closed_rows(context, freshness=ACCVPTableFreshness())
        started = time.perf_counter()
        stage_latencies = {
            "context_legality": 0.0,
            "accvp_prepare_candidates": 0.0,
            "accvp_score": 0.0,
            "risk_secondary": 0.0,
            "risk_context": 0.0,
            "risk_predict_many": 0.0,
            "risk_candidate_features": 0.0,
            "risk_other_rollout": 0.0,
            "risk_ego_rollout": 0.0,
            "risk_pairwise_geometry": 0.0,
            "risk_merge_local_reduction": 0.0,
            "risk_network_forward": 0.0,
            "risk_result_pack": 0.0,
            "table_pack": 0.0,
        }
        try:
            legality_started = time.perf_counter()
            legality = self._legality_map(context)
            legal_actions = [action for action in ACTIONS if bool(legality.get(int(action.index), False))]
            base = self._fail_closed_rows_from_legality(legality, context)
            stage_latencies["context_legality"] = time.perf_counter() - legality_started
            if self._invalid_table_dropout(context):
                self.stats.invalid_table_dropout_count += 1
                self._record_latency(time.perf_counter() - started, stage_latencies)
                return self._invalid_table_rows(context, base)
            if not legal_actions:
                if self.invalid_table_strategy == "bounded_last_valid_v2":
                    reason = "no_prior" if self._last_valid_snapshot is None else "no_legal_actions"
                    self._record_stale_rejection(reason)
                    self._record_hard_fail_closed(no_prior=self._last_valid_snapshot is None)
                    base = self._fail_closed_rows_from_legality(
                        legality,
                        context,
                        freshness=ACCVPTableFreshness(
                            hard_default_active=True,
                            context_delta_norm=1.0,
                        ),
                    )
                else:
                    self._record_hard_fail_closed()
                self._record_latency(time.perf_counter() - started, stage_latencies)
                return base
            prepare_started = time.perf_counter()
            prepared = None
            if (
                not self.use_inference_worker
                and hasattr(self._predictor, "prepare_candidates")
                and hasattr(self._predictor, "score_prepared")
            ):
                prepared = self._predictor.prepare_candidates(context, legal_actions)
                stage_latencies["accvp_prepare_candidates"] = time.perf_counter() - prepare_started
                score_started = time.perf_counter()
                scores = self._predictor.score_prepared(prepared)
                stage_latencies["accvp_score"] = time.perf_counter() - score_started
            else:
                scorer = self._predictor.score_candidates
                stage_latencies["accvp_prepare_candidates"] = time.perf_counter() - prepare_started
                score_started = time.perf_counter()
                if "timeout_s" in inspect.signature(scorer).parameters:
                    scores = scorer(context, legal_actions, timeout_s=self.timeout_s)
                else:
                    scores = scorer(context, legal_actions)
                stage_latencies["accvp_score"] = time.perf_counter() - score_started
            scores = self._validate_scores(scores, legal_actions)
            risk_started = time.perf_counter()
            try:
                secondary = self._secondary_safety_many(
                    ACTIONS,
                    context,
                    legality,
                    stage_latencies=stage_latencies,
                )
            finally:
                stage_latencies["risk_secondary"] = time.perf_counter() - risk_started
            pack_started = time.perf_counter()
            rows = self._rows_from_scores(scores, legality, secondary, context)
            stage_latencies["table_pack"] = time.perf_counter() - pack_started
            elapsed = time.perf_counter() - started
            self._record_latency(elapsed, stage_latencies)
            if elapsed > self.timeout_s:
                self._record_timeout()
                return self._invalid_table_rows(context, base)
            self.stats.valid_decisions += 1
            self.stats.activation_window_valid_decisions += 1
            self.stats.consecutive_timeout_count = 0
            self._last_valid_value = rows.astype(np.float32, copy=True)
            self._last_valid_snapshot = ACCVPCachedCandidateTable(
                value=rows.astype(np.float32, copy=True),
                context=self._context_snapshot(context, legality),
            )
            return rows
        except TimeoutError:
            self._record_timeout()
        except ACCVPInvalidRuntimeOutput:
            self.stats.consecutive_timeout_count = 0
            self.stats.invalid_output_count += 1
            self._record_runtime_error("invalid_runtime_output")
        except ACCVPCriticalActorOverflow as exc:
            self.stats.consecutive_timeout_count = 0
            self.stats.runtime_context_error_count += 1
            self.stats.critical_actor_overflow_count += 1
            self._record_runtime_error(exc.reason_code)
            self._record_critical_actor_overflow(exc.details)
        except ACCVPRuntimeContextError as exc:
            self.stats.consecutive_timeout_count = 0
            self.stats.runtime_context_error_count += 1
            if (
                exc.reason_code
                == "risk_safety_actor_coverage_incomplete"
            ):
                self.stats.risk_safety_actor_coverage_incomplete_count += 1
            self._record_runtime_error(exc.reason_code)
        except ValueError:
            self.stats.consecutive_timeout_count = 0
            self.stats.unexpected_value_error_count += 1
            self.stats.model_error_count += 1
            self._record_runtime_error("unexpected_value_error")
        except Exception as exc:
            self.stats.consecutive_timeout_count = 0
            self.stats.model_error_count += 1
            self._record_runtime_error(f"model_error:{type(exc).__name__}")
        self._record_latency(time.perf_counter() - started, stage_latencies)
        if self.fail_closed_defaults:
            return self._invalid_table_rows(context, self._fail_closed_rows(context))
        raise

    def _record_timeout(self) -> None:
        self.stats.timeout_count += 1
        self.stats.consecutive_timeout_count += 1
        self.stats.max_consecutive_timeout_count = max(
            self.stats.max_consecutive_timeout_count,
            self.stats.consecutive_timeout_count,
        )

    def _record_runtime_error(self, reason: str) -> None:
        key = str(reason or "unknown_runtime_error")
        self.stats.runtime_error_reasons[key] = int(
            self.stats.runtime_error_reasons.get(key, 0)
        ) + 1

    def _record_critical_actor_overflow(self, details: dict[str, Any]) -> None:
        payload = dict(details or {})
        key = (
            f"capacity={int(payload.get('capacity', -1))},"
            f"critical_count={int(payload.get('critical_count', -1))}"
        )
        self.stats.critical_actor_overflow_histogram[key] = int(
            self.stats.critical_actor_overflow_histogram.get(key, 0)
        ) + 1
        if (
            len(self.stats.critical_actor_overflow_examples)
            < self.stats.critical_actor_overflow_sample_limit
        ):
            self.stats.critical_actor_overflow_examples.append(payload)

    def _validate_scores(
        self,
        scores: Any,
        legal_actions: list[CandidateAction],
    ) -> list[dict[str, Any]]:
        if not isinstance(scores, (list, tuple)):
            raise ACCVPInvalidRuntimeOutput("ACCVP scores must be a list or tuple")
        rows = list(scores)
        expected = {int(action.index) for action in legal_actions}
        seen: set[int] = set()
        validated: list[dict[str, Any]] = []
        for raw in rows:
            if not isinstance(raw, dict):
                raise ACCVPInvalidRuntimeOutput("ACCVP score row must be a mapping")
            action_id = int(raw.get("action_id", -1))
            if action_id not in expected or action_id in seen:
                raise ACCVPInvalidRuntimeOutput("ACCVP score action ids are missing, duplicated, or unexpected")
            for key in (
                "p_merge_before_taper",
                "target_lane_entry_time_s",
                "ensemble_disagreement",
            ):
                if key in raw and not math.isfinite(float(raw[key])):
                    raise ACCVPInvalidRuntimeOutput(f"ACCVP score {key} must be finite")
            seen.add(action_id)
            validated.append(dict(raw))
        # The VNext bounded-stale contract requires an atomic, complete table:
        # reusing a partially fresh table would mix timestamps across actions.
        # Legacy v1/v2 observations intentionally allowed a sparse score set
        # and represented missing rows with the existing fail-closed defaults.
        if self.feature_version == self.BOUNDED_STALE_FEATURE_VERSION and seen != expected:
            raise ACCVPInvalidRuntimeOutput("ACCVP scores do not cover every legal action")
        return validated

    def _record_hard_fail_closed(self, *, no_prior: bool = True) -> None:
        self.stats.fail_closed_count += 1
        self.stats.hard_fail_closed_count += 1
        self.stats.no_prior_fallback_count += int(no_prior)

    def _invalid_table_rows(self, context: dict[str, Any], hard_default: np.ndarray) -> np.ndarray:
        if self.invalid_table_strategy == "bounded_last_valid_v2":
            eligible, freshness, reason = self._bounded_stale_freshness(context)
            if eligible:
                self.stats.fail_closed_count += 1
                self.stats.last_valid_fallback_count += 1
                self.stats.bounded_stale_reuse_count += 1
                self.stats.stale_age_s.append(float(freshness.age_norm * self.last_valid_ttl_s))
                self.stats.stale_context_delta_norm.append(float(freshness.context_delta_norm))
                return self._bounded_last_valid_rows(context, freshness)
            self._record_stale_rejection(reason)
            self._record_hard_fail_closed(no_prior=self._last_valid_snapshot is None)
            if self.feature_version == self.BOUNDED_STALE_FEATURE_VERSION:
                return self._fail_closed_rows(context, freshness=freshness)
            return hard_default
        if self.invalid_table_strategy == "last_valid_with_invalid_mask_v1" and self._last_valid_value is not None:
            self.stats.fail_closed_count += 1
            self.stats.last_valid_fallback_count += 1
            return self._last_valid_rows_with_invalid_mask(context)
        self._record_hard_fail_closed()
        return hard_default

    def _record_stale_rejection(self, reason: str) -> None:
        key = str(reason or "unknown")
        self.stats.stale_rejection_reasons[key] = int(self.stats.stale_rejection_reasons.get(key, 0)) + 1
        if key in {"decision_ttl", "time_ttl"}:
            self.stats.stale_expired_count += 1
        elif key != "no_prior":
            self.stats.stale_context_rejected_count += 1

    @staticmethod
    def _normalised_delta(current: float, previous: float, limit: float) -> float:
        if not math.isfinite(current) or not math.isfinite(previous):
            return 0.0 if current == previous else float("inf")
        if limit <= 0.0:
            return 0.0 if current == previous else float("inf")
        return abs(float(current) - float(previous)) / float(limit)

    @staticmethod
    def _context_value(value: Any, name: str, default: Any) -> Any:
        if isinstance(value, dict):
            return value.get(name, default)
        return getattr(value, name, default)

    def _bounded_stale_freshness(
        self,
        context: dict[str, Any],
    ) -> tuple[bool, ACCVPTableFreshness, str]:
        cached = self._last_valid_snapshot
        if cached is None:
            return False, ACCVPTableFreshness(hard_default_active=True), "no_prior"
        current = self._context_snapshot(context, self._legality_map(context))
        previous = cached.context
        decision_age = int(current.decision_index - previous.decision_index)
        time_age_s = float(current.simulation_time_s - previous.simulation_time_s)
        age_norm = (
            float(
                np.clip(
                    time_age_s / max(self.last_valid_ttl_s, 1.0e-9),
                    0.0,
                    1.0,
                )
            )
            if math.isfinite(time_age_s)
            else 1.0
        )
        if current.episode_seed < 0 or previous.episode_seed < 0:
            return False, ACCVPTableFreshness(hard_default_active=True, age_norm=age_norm), "missing_episode_identity"
        if current.decision_index < 0 or previous.decision_index < 0:
            return False, ACCVPTableFreshness(hard_default_active=True, age_norm=age_norm), "missing_decision_identity"
        if current.episode_seed != previous.episode_seed:
            return False, ACCVPTableFreshness(hard_default_active=True, age_norm=1.0), "episode_mismatch"
        # A cached table may be consumed only by a later control decision.  A
        # zero age indicates a malformed/replayed context, not a stale step.
        if decision_age <= 0:
            return False, ACCVPTableFreshness(hard_default_active=True, age_norm=age_norm), "decision_order"
        if decision_age > self.last_valid_max_decisions:
            return False, ACCVPTableFreshness(hard_default_active=True, age_norm=1.0), "decision_ttl"
        if not math.isfinite(time_age_s):
            return False, ACCVPTableFreshness(hard_default_active=True, age_norm=1.0), "invalid_time"
        if time_age_s < -1.0e-9 or time_age_s > self.last_valid_ttl_s + 1.0e-9:
            return False, ACCVPTableFreshness(hard_default_active=True, age_norm=1.0), "time_ttl"

        categorical_checks = (
            (current.ego_edge_id == previous.ego_edge_id, "ego_edge_changed"),
            (current.ego_lane_id == previous.ego_lane_id, "ego_lane_changed"),
            (current.ego_on_auxiliary == previous.ego_on_auxiliary, "auxiliary_state_changed"),
            (current.target_front_vehicle_id == previous.target_front_vehicle_id, "target_front_changed"),
            (current.target_rear_vehicle_id == previous.target_rear_vehicle_id, "target_rear_changed"),
            (current.legality_mask == previous.legality_mask, "legality_changed"),
        )
        for passed, reason in categorical_checks:
            if not passed:
                return False, ACCVPTableFreshness(hard_default_active=True, age_norm=age_norm, context_delta_norm=1.0), reason

        context_delta = max(
            self._normalised_delta(
                current.merge_distance,
                previous.merge_distance,
                self.last_valid_max_merge_distance_delta_m,
            ),
            self._normalised_delta(
                current.ego_speed,
                previous.ego_speed,
                self.last_valid_max_ego_speed_delta_mps,
            ),
            self._normalised_delta(
                current.target_front_gap,
                previous.target_front_gap,
                self.last_valid_max_gap_delta_m,
            ),
            self._normalised_delta(
                current.target_rear_gap,
                previous.target_rear_gap,
                self.last_valid_max_gap_delta_m,
            ),
        )
        context_delta_norm = float(np.clip(context_delta, 0.0, 1.0))
        freshness = ACCVPTableFreshness(
            stale_fallback_active=True,
            age_norm=age_norm,
            context_delta_norm=context_delta_norm,
        )
        if context_delta > 1.0:
            return False, ACCVPTableFreshness(
                hard_default_active=True,
                age_norm=age_norm,
                context_delta_norm=1.0,
            ), "context_delta"
        return True, freshness, "eligible"

    def _context_snapshot(
        self,
        context: dict[str, Any],
        legality: dict[int, bool],
    ) -> ACCVPTableContextSnapshot:
        ego = context.get("ego")
        local = context.get("merge_local")
        episode_step = int(context.get("episode_step", -1))
        configured_time = context.get("simulation_time_s", context.get("sim_time_s"))
        if configured_time is None:
            step_length = float(self.config.scenario.get("step_length", 0.1))
            configured_time = episode_step * step_length
        ego_lane_identity = self._context_value(ego, "lane_id", None)
        if ego_lane_identity in {None, ""}:
            ego_lane_identity = self._context_value(ego, "lane_index", -1)
        return ACCVPTableContextSnapshot(
            episode_seed=int(context.get("episode_seed", -1)),
            decision_index=int(context.get("decision_index", -1)),
            episode_step=episode_step,
            simulation_time_s=float(configured_time),
            ego_edge_id=str(self._context_value(ego, "edge_id", "")),
            # SUMO lane IDs are strings (for example ``edge_0``).  Preserve
            # that identity instead of coercing it to an integer lane index.
            ego_lane_id=str(ego_lane_identity),
            ego_speed=float(self._context_value(ego, "speed", 0.0)),
            ego_on_auxiliary=bool(self._context_value(local, "ego_on_auxiliary", False)),
            merge_distance=float(self._context_value(local, "merge_distance", float("inf"))),
            target_front_vehicle_id=str(self._context_value(local, "target_front_vehicle_id", "")),
            target_rear_vehicle_id=str(self._context_value(local, "target_rear_vehicle_id", "")),
            target_front_gap=float(self._context_value(local, "target_front_gap", float("inf"))),
            target_rear_gap=float(self._context_value(local, "target_rear_gap", float("inf"))),
            legality_mask=tuple(bool(legality.get(int(action.index), False)) for action in ACTIONS),
        )

    def _invalid_table_dropout(self, context: dict[str, Any]) -> bool:
        if self.invalid_table_dropout_rate <= 0.0:
            return False
        seed = int(context.get("episode_seed", 0))
        decision = int(context.get("decision_index", 0))
        step = int(context.get("episode_step", 0))
        value = ((seed * 1_000_003 + decision * 9_176 + step * 131) % 1_000_000) / 1_000_000.0
        return bool(value < self.invalid_table_dropout_rate)

    def _last_valid_rows_with_invalid_mask(self, context: dict[str, Any]) -> np.ndarray:
        assert self._last_valid_value is not None
        rows = np.asarray(self._last_valid_value, dtype=np.float32).copy()
        table_dim = self.ACTION_COUNT * len(self.FEATURE_NAMES)
        table = rows[:table_dim].reshape((self.ACTION_COUNT, len(self.FEATURE_NAMES)))
        legality = self._legality_map(context)
        for action in ACTIONS:
            index = int(action.index)
            table[index, 0] = 0.0
            table[index, 1] = float(bool(legality.get(index, False)))
            table[index, 7] = float(action.lateral_cmd > 0)
            table[index, 8] = float(action.lateral_cmd == 0)
            table[index, 9] = float(action.lateral_cmd)
            table[index, 10] = float(action.accel_cmd)
        return self._append_policy_state(table.reshape(-1).astype(np.float32), context)

    def _bounded_last_valid_rows(
        self,
        context: dict[str, Any],
        freshness: ACCVPTableFreshness,
    ) -> np.ndarray:
        cached = self._last_valid_snapshot
        if cached is None:  # pragma: no cover - guarded by _bounded_stale_freshness
            return self._fail_closed_rows(context)
        rows = np.asarray(cached.value, dtype=np.float32).copy()
        table_dim = self.ACTION_COUNT * len(self.FEATURE_NAMES)
        table = rows[:table_dim].reshape((self.ACTION_COUNT, len(self.FEATURE_NAMES)))
        legality = self._legality_map(context)
        for action in ACTIONS:
            index = int(action.index)
            table[index, 0] = 0.0
            table[index, 1] = float(bool(legality.get(index, False)))
            # A stale table never carries forward a safety approval or a stale
            # uncertainty claim. Only bounded task-viability values survive.
            table[index, 2] = 0.0
            table[index, 3] = 1.0
            table[index, 6] = 1.0
            table[index, 7] = float(action.lateral_cmd > 0)
            table[index, 8] = float(action.lateral_cmd == 0)
            table[index, 9] = float(action.lateral_cmd)
            table[index, 10] = float(action.accel_cmd)
        return self._append_policy_state(
            table.reshape(-1).astype(np.float32),
            context,
            freshness=freshness,
        )

    def _append_policy_state(
        self,
        rows: np.ndarray,
        context: dict[str, Any],
        *,
        freshness: ACCVPTableFreshness | None = None,
    ) -> np.ndarray:
        if self.feature_version not in {
            self.PERSISTENCE_FEATURE_VERSION,
            self.BOUNDED_STALE_FEATURE_VERSION,
        }:
            return rows
        duration_s = float(context.get("policy_commitment_duration_s", 1.0) or 1.0)
        remaining_s = float(context.get("policy_commitment_remaining_s", 0.0) or 0.0)
        last_raw_action_id = int(context.get("last_raw_action", -1))
        if 0 <= last_raw_action_id < len(ACTIONS):
            last_raw = ACTIONS[last_raw_action_id]
            last_lateral = float(last_raw.lateral_cmd)
            last_accel = float(last_raw.accel_cmd)
        else:
            last_lateral = 0.0
            last_accel = 0.0
        state = np.asarray(
            [
                float(bool(context.get("policy_commitment_active", False))),
                float(np.clip(remaining_s / max(1.0e-6, duration_s), 0.0, 1.0)),
                float(last_lateral),
                float(last_accel),
            ],
            dtype=np.float32,
        )
        payload = np.concatenate([rows.astype(np.float32, copy=False), state], axis=0)
        if self.feature_version != self.BOUNDED_STALE_FEATURE_VERSION:
            return payload
        current = freshness or ACCVPTableFreshness()
        freshness_state = np.asarray(
            [
                float(current.stale_fallback_active),
                float(current.hard_default_active),
                float(np.clip(current.age_norm, 0.0, 1.0)),
                float(np.clip(current.context_delta_norm, 0.0, 1.0)),
            ],
            dtype=np.float32,
        )
        return np.concatenate([payload, freshness_state], axis=0)

    def _record_latency(self, total_s: float, stage_latencies: dict[str, float]) -> None:
        if not self.profile_latency:
            return
        self.stats.latency_total_s.append(float(total_s))
        self.stats.latency_context_legality_s.append(float(stage_latencies.get("context_legality", 0.0)))
        self.stats.latency_accvp_prepare_candidates_s.append(float(stage_latencies.get("accvp_prepare_candidates", 0.0)))
        self.stats.latency_accvp_score_s.append(float(stage_latencies.get("accvp_score", 0.0)))
        self.stats.latency_risk_secondary_s.append(float(stage_latencies.get("risk_secondary", 0.0)))
        self.stats.latency_risk_context_s.append(float(stage_latencies.get("risk_context", 0.0)))
        self.stats.latency_risk_predict_many_s.append(float(stage_latencies.get("risk_predict_many", 0.0)))
        self.stats.latency_risk_candidate_features_s.append(float(stage_latencies.get("risk_candidate_features", 0.0)))
        self.stats.latency_risk_other_rollout_s.append(float(stage_latencies.get("risk_other_rollout", 0.0)))
        self.stats.latency_risk_ego_rollout_s.append(float(stage_latencies.get("risk_ego_rollout", 0.0)))
        self.stats.latency_risk_pairwise_geometry_s.append(float(stage_latencies.get("risk_pairwise_geometry", 0.0)))
        self.stats.latency_risk_merge_local_reduction_s.append(float(stage_latencies.get("risk_merge_local_reduction", 0.0)))
        self.stats.latency_risk_network_forward_s.append(float(stage_latencies.get("risk_network_forward", 0.0)))
        self.stats.latency_risk_result_pack_s.append(float(stage_latencies.get("risk_result_pack", 0.0)))
        self.stats.latency_table_pack_s.append(float(stage_latencies.get("table_pack", 0.0)))

    def _rows_from_scores(
        self,
        scores: list[dict[str, Any]],
        legality: dict[int, bool],
        secondary_by_action: dict[int, dict[str, Any]],
        context: dict[str, Any],
    ) -> np.ndarray:
        by_action = {int(row.get("action_id", -1)): row for row in scores}
        rows: list[float] = []
        for action in ACTIONS:
            legal = bool(legality.get(int(action.index), False))
            score = by_action.get(int(action.index))
            if score is None:
                rows.extend(self._fail_closed_row(action, candidate_legal=legal))
                continue
            secondary = secondary_by_action.get(
                int(action.index),
                {"safety_pass": False, "risk_score": 1.0},
            )
            rows.extend(
                [
                    1.0,
                    float(legal),
                    float(secondary["safety_pass"]),
                    float(np.clip(secondary.get("risk_score", 0.0), 0.0, 1.0)),
                    float(np.clip(score.get("p_merge_before_taper", 0.0), 0.0, 1.0)),
                    float(
                        np.clip(
                            float(score.get("target_lane_entry_time_s", self.viability_horizon_s))
                            / max(1.0e-6, self.viability_horizon_s),
                            0.0,
                            1.0,
                        )
                    ),
                    float(max(0.0, score.get("ensemble_disagreement", 1.0))),
                    float(action.lateral_cmd > 0),
                    float(action.lateral_cmd == 0),
                    float(action.lateral_cmd),
                    float(action.accel_cmd),
                ]
            )
        return self._append_policy_state(np.asarray(rows, dtype=np.float32), context)

    def _fail_closed_rows(
        self,
        context: dict[str, Any],
        *,
        freshness: ACCVPTableFreshness | None = None,
    ) -> np.ndarray:
        return self._fail_closed_rows_from_legality(
            self._legality_map(context),
            context,
            freshness=freshness,
        )

    def _fail_closed_rows_from_legality(
        self,
        legality: dict[int, bool],
        context: dict[str, Any] | None = None,
        *,
        freshness: ACCVPTableFreshness | None = None,
    ) -> np.ndarray:
        rows: list[float] = []
        for action in ACTIONS:
            rows.extend(self._fail_closed_row(action, candidate_legal=bool(legality.get(int(action.index), False))))
        if freshness is None and self.feature_version == self.BOUNDED_STALE_FEATURE_VERSION:
            freshness = ACCVPTableFreshness(hard_default_active=True)
        return self._append_policy_state(
            np.asarray(rows, dtype=np.float32),
            context or {},
            freshness=freshness,
        )

    @staticmethod
    def _fail_closed_row(action: CandidateAction, *, candidate_legal: bool) -> list[float]:
        return [
            0.0,
            float(candidate_legal),
            0.0,
            1.0,
            0.0,
            1.0,
            1.0,
            float(action.lateral_cmd > 0),
            float(action.lateral_cmd == 0),
            float(action.lateral_cmd),
            float(action.accel_cmd),
        ]

    def _secondary_safety(self, action: CandidateAction, context: dict[str, Any]) -> dict[str, Any]:
        if not self.include_risk_secondary or self._shield is None:
            return {"safety_pass": True, "risk_score": 0.0}
        return dict(self._shield.evaluate_candidate(action, context))

    def _secondary_safety_many(
        self,
        actions: tuple[CandidateAction, ...] | list[CandidateAction],
        context: dict[str, Any],
        legality: dict[int, bool],
        *,
        stage_latencies: dict[str, float] | None = None,
    ) -> dict[int, dict[str, Any]]:
        if not self.include_risk_secondary or self._shield is None:
            result: dict[int, dict[str, Any]] = {}
            for action in actions:
                candidate_legal = bool(legality.get(int(action.index), False))
                result[int(action.index)] = {
                    "candidate_legal": candidate_legal,
                    "safety_pass": candidate_legal,
                    "risk_score": 0.0 if candidate_legal else 1.0,
                    "risk_uncertainty": 0.0 if candidate_legal else 1.0,
                    "veto_reason": "" if candidate_legal else "candidate_illegal",
                }
            return result
        risk_model = getattr(getattr(self._shield, "ranker", None), "risk_model", None)
        if risk_model is None or not hasattr(risk_model, "predict_many"):
            return {int(action.index): self._secondary_safety(action, context) for action in actions}
        context_started = time.perf_counter()
        risk_context = self._risk_context_for_observation(context)
        if stage_latencies is not None:
            stage_latencies["risk_context"] = time.perf_counter() - context_started
        predict_started = time.perf_counter()
        legal_actions = [
            action for action in actions if bool(legality.get(int(action.index), False))
        ]
        if hasattr(self._shield, "evaluate_candidates"):
            checks = self._shield.evaluate_candidates(
                legal_actions,
                risk_context,
                candidate_legality=legality,
            )
        else:
            predictions = risk_model.predict_many(legal_actions, risk_context)
            risk_threshold = float(self.config.shield.risk_threshold)
            uncertainty_threshold = float(self.config.shield.uncertainty_threshold)
            checks = []
            for prediction in predictions:
                risk_score = float(prediction.risk_score)
                risk_uncertainty = float(prediction.risk_uncertainty)
                if risk_score >= risk_threshold:
                    veto_reason = "risk_score"
                elif risk_uncertainty >= uncertainty_threshold:
                    veto_reason = "risk_uncertainty"
                else:
                    veto_reason = ""
                checks.append(
                    {
                        "candidate_legal": True,
                        "risk_score": risk_score,
                        "risk_uncertainty": risk_uncertainty,
                        "safety_pass": not veto_reason,
                        "veto_reason": veto_reason,
                    }
                )
        if stage_latencies is not None:
            stage_latencies["risk_predict_many"] = time.perf_counter() - predict_started
            detailed = dict(risk_context.get("_risk_last_latency", {}) or {})
            stage_latencies["risk_candidate_features"] = float(
                detailed.get("candidate_features", 0.0)
            )
            stage_latencies["risk_other_rollout"] = float(
                detailed.get("other_rollout", 0.0)
            )
            stage_latencies["risk_ego_rollout"] = float(
                detailed.get("ego_rollout", 0.0)
            )
            stage_latencies["risk_pairwise_geometry"] = float(
                detailed.get("pairwise_geometry", 0.0)
            )
            stage_latencies["risk_merge_local_reduction"] = float(
                detailed.get("merge_local_reduction", 0.0)
            )
            stage_latencies["risk_network_forward"] = float(
                detailed.get("network_forward", 0.0)
            )
            stage_latencies["risk_result_pack"] = float(
                detailed.get("result_pack", 0.0)
            )
        if len(checks) != len(legal_actions):
            raise ACCVPInvalidRuntimeOutput(
                "Risk-secondary prediction count does not match the legal action batch"
            )
        result: dict[int, dict[str, Any]] = {}
        check_by_action = {
            int(action.index): dict(check)
            for action, check in zip(legal_actions, checks)
        }
        for action in actions:
            candidate_legal = bool(legality.get(int(action.index), False))
            if not candidate_legal:
                result[int(action.index)] = {
                    "candidate_legal": False,
                    "risk_score": 1.0,
                    "risk_uncertainty": 1.0,
                    "safety_pass": False,
                    "veto_reason": "candidate_illegal",
                }
                continue
            check = check_by_action[int(action.index)]
            risk_score = float(check.get("risk_score", 1.0))
            risk_uncertainty = float(check.get("risk_uncertainty", 1.0))
            if not math.isfinite(risk_score) or not math.isfinite(risk_uncertainty):
                raise ACCVPInvalidRuntimeOutput("Risk-secondary prediction must be finite")
            veto_reason = str(check.get("veto_reason", ""))
            result[int(action.index)] = {
                "candidate_legal": candidate_legal,
                "risk_score": risk_score,
                "risk_uncertainty": risk_uncertainty,
                "safety_pass": bool(check.get("safety_pass", not veto_reason)),
                "veto_reason": veto_reason,
            }
        return result

    def _risk_context_for_observation(self, context: dict[str, Any]) -> dict[str, Any]:
        if self.secondary_safety_profile != "short_horizon_risk_v1" or self.risk_horizon_steps <= 0:
            return context
        current_steps = int(self.config.risk_module.get("collision_horizon_steps", 30))
        if self.risk_horizon_steps == current_steps:
            return context
        risk_context = dict(context)
        risk_context["config"] = (
            self._risk_observation_config
            if self._risk_observation_config is not None
            else self.config
        )
        # Avoid accidentally reusing rollout samples computed under the full
        # deployment horizon.  This profile is observation-only and is reported
        # separately from the Safety Shield's strict secondary gate.
        risk_context.pop("_candidate_rollout_context", None)
        risk_context.pop("_ego_rollout_cache", None)
        risk_context.pop("_ego_rollout_config_hash_cache", None)
        risk_context.pop("_risk_prediction_cache", None)
        risk_context.pop("_risk_candidate_latency", None)
        risk_context.pop("_risk_last_latency", None)
        return risk_context

    def _activation_window(self, context: dict[str, Any]) -> bool:
        local = context.get("merge_local")
        return bool(
            local is not None
            and bool(self._context_value(local, "ego_on_auxiliary", False))
            and 0.0
            < float(self._context_value(local, "merge_distance", float("inf")))
            <= self.activation_distance
        )

    @staticmethod
    def _candidate_legal(action: CandidateAction, context: dict[str, Any]) -> bool:
        configured = context.get("candidate_legal_by_action")
        if configured is not None:
            return bool(dict(configured).get(int(action.index), False))
        return bool(is_candidate_legal(action, context, missing_ego_is_legal=False))

    def _legality_map(self, context: dict[str, Any]) -> dict[int, bool]:
        configured = context.get("candidate_legal_by_action")
        if configured is not None:
            configured_map = dict(configured)
            return {int(action.index): bool(configured_map.get(int(action.index), False)) for action in ACTIONS}
        return {int(action.index): bool(is_candidate_legal(action, context, missing_ego_is_legal=False)) for action in ACTIONS}


def validate_accvp_observation_config(config: Any) -> None:
    if not RiskGatedACCVPCandidateTableAugmentor.enabled(config):
        return
    obs_cfg = dict(config.accvp.get("observation", {}) or {})
    if str(obs_cfg.get("feature_version", RISK_GATED_ACCVP_OBSERVATION_VERSION)) not in {
        RISK_GATED_ACCVP_OBSERVATION_VERSION,
        RISK_GATED_ACCVP_OBSERVATION_PERSISTENCE_VERSION,
        RISK_GATED_ACCVP_OBSERVATION_BOUNDED_STALE_VERSION,
    }:
        raise ValueError("unsupported ACCVP observation feature_version")
    feature_version = str(obs_cfg.get("feature_version", RISK_GATED_ACCVP_OBSERVATION_VERSION))
    default_strategy = (
        "bounded_last_valid_v2"
        if feature_version == RISK_GATED_ACCVP_OBSERVATION_BOUNDED_STALE_VERSION
        else "fail_closed_v1"
    )
    invalid_strategy = str(obs_cfg.get("invalid_table_strategy", default_strategy))
    if invalid_strategy not in {
        "fail_closed_v1",
        "last_valid_with_invalid_mask_v1",
        "bounded_last_valid_v2",
    }:
        raise ValueError("unsupported ACCVP observation invalid_table_strategy")
    if (
        feature_version == RISK_GATED_ACCVP_OBSERVATION_BOUNDED_STALE_VERSION
        and invalid_strategy != "bounded_last_valid_v2"
    ):
        raise ValueError("bounded-stale observation requires bounded_last_valid_v2")
    if (
        invalid_strategy == "bounded_last_valid_v2"
        and feature_version != RISK_GATED_ACCVP_OBSERVATION_BOUNDED_STALE_VERSION
    ):
        raise ValueError("bounded_last_valid_v2 requires the bounded-stale feature contract")
    last_valid_ttl_s = float(obs_cfg.get("last_valid_ttl_s", 0.5))
    if last_valid_ttl_s < 0.0:
        raise ValueError("accvp.observation.last_valid_ttl_s must be non-negative")
    last_valid_max_decisions = int(obs_cfg.get("last_valid_max_decisions", 1))
    if last_valid_max_decisions < 0:
        raise ValueError("accvp.observation.last_valid_max_decisions must be non-negative")
    if feature_version == RISK_GATED_ACCVP_OBSERVATION_BOUNDED_STALE_VERSION:
        if not bool(obs_cfg.get("fail_closed_defaults", True)):
            raise ValueError("bounded-stale observation requires fail_closed_defaults=true")
        if last_valid_max_decisions > 1:
            raise ValueError("bounded-stale observation permits at most one stale decision")
        if last_valid_ttl_s > 0.5:
            raise ValueError("bounded-stale observation TTL must not exceed 0.5 seconds")
        for key, upper_bound in (
            ("last_valid_max_merge_distance_delta_m", 15.0),
            ("last_valid_max_ego_speed_delta_mps", 3.0),
            ("last_valid_max_gap_delta_m", 8.0),
        ):
            value = float(obs_cfg.get(key, upper_bound))
            if value < 0.0 or value > upper_bound:
                raise ValueError(
                    f"accvp.observation.{key} must be in [0, {upper_bound}] for bounded stale"
                )
    dropout_rate = float(obs_cfg.get("invalid_table_dropout_rate", 0.0) or 0.0)
    if not 0.0 <= dropout_rate <= 1.0:
        raise ValueError("unsupported ACCVP observation invalid_table_dropout_rate")
    forecast_enabled = bool(config.forecast_features.enabled or config.rl.use_wcdt_forecast_features)
    if forecast_enabled and not bool(obs_cfg.get("allow_with_forecast_features", False)):
        raise ValueError(
            "Risk-gated ACCVP candidate table observation is mutually exclusive with legacy forecast_features by default"
        )
