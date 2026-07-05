from __future__ import annotations

import time
import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from safe_rl.accvp.protocol import effective_activation_distance
from safe_rl.accvp.runtime import ACCVPRuntimePredictor
from safe_rl.accvp.schema import file_sha256
from safe_rl.risk.merge_local import is_candidate_legal
from safe_rl.risk.risk_module import RiskModuleWrapper
from safe_rl.shield.safety_shield import SafetyShield
from safe_rl.sim.action_space import ACTIONS, CandidateAction


RISK_GATED_ACCVP_OBSERVATION_VERSION = "risk_gated_candidate_table_v1"


@dataclass
class ACCVPObservationStats:
    total_decisions: int = 0
    activation_window_decisions: int = 0
    valid_decisions: int = 0
    activation_window_valid_decisions: int = 0
    timeout_count: int = 0
    model_error_count: int = 0
    invalid_bundle_count: int = 0
    fail_closed_count: int = 0

    def to_dict(self) -> dict[str, Any]:
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
            "accvp_table_fail_closed_count": int(self.fail_closed_count),
            "accvp_table_timeout_rate": (
                float(self.timeout_count / self.total_decisions) if self.total_decisions else 0.0
            ),
            "accvp_table_fail_closed_rate": (
                float(self.fail_closed_count / self.total_decisions) if self.total_decisions else 0.0
            ),
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
    ACTION_COUNT = len(ACTIONS)
    FEATURE_VERSION = RISK_GATED_ACCVP_OBSERVATION_VERSION

    def __init__(
        self,
        config: Any,
        *,
        predictor: Any | None = None,
        shield: Any | None = None,
        risk_checkpoint: str | Path | None = None,
    ) -> None:
        self.config = config
        obs_cfg = dict(config.accvp.get("observation", {}) or {})
        feature_version = str(obs_cfg.get("feature_version", self.FEATURE_VERSION))
        if feature_version != self.FEATURE_VERSION:
            raise ValueError(
                f"unsupported ACCVP observation feature_version={feature_version!r}; "
                f"expected {self.FEATURE_VERSION!r}"
            )
        self.timeout_s = float(obs_cfg.get("timeout_s", config.accvp.get("max_decision_latency_s", 0.5)))
        self.fail_closed_defaults = bool(obs_cfg.get("fail_closed_defaults", True))
        self.include_risk_secondary = bool(obs_cfg.get("include_risk_secondary", True))
        self.activation_distance = float(obs_cfg.get("activation_distance", effective_activation_distance(config)))
        self.viability_horizon_s = float(config.accvp.get("viability_horizon_s", 8.0))
        self.stats = ACCVPObservationStats()
        self._cache_key: tuple[Any, ...] | None = None
        self._cache_value: np.ndarray | None = None
        self._predictor = predictor
        self._shield = shield
        if self._predictor is None:
            checkpoint = config.accvp.get("checkpoint")
            if not checkpoint:
                raise FileNotFoundError("accvp.observation.enabled requires accvp.checkpoint")
            self._predictor = ACCVPRuntimePredictor(config, checkpoint)
            if config.accvp.get("artifact_manifest"):
                self._predictor.validate_artifact_bundle(config.accvp.get("operating_point"))
        if self.include_risk_secondary and self._shield is None:
            checkpoint = risk_checkpoint or config.accvp.get("risk_checkpoint")
            if not checkpoint:
                raise FileNotFoundError("Risk-gated ACCVP observation requires accvp.risk_checkpoint")
            risk_model = RiskModuleWrapper(config, checkpoint=str(checkpoint))
            self._shield = SafetyShield(config, risk_model)

    @classmethod
    def feature_dim(cls, _config: Any | None = None) -> int:
        return cls.ACTION_COUNT * len(cls.FEATURE_NAMES)

    @classmethod
    def feature_names(cls) -> list[str]:
        names: list[str] = []
        for action in ACTIONS:
            for feature in cls.FEATURE_NAMES:
                names.append(f"action_{action.index}_{feature}")
        return names

    @staticmethod
    def enabled(config: Any) -> bool:
        return bool(config.accvp.get("observation", {}).get("enabled", False))

    def reset_episode_state(self) -> None:
        self.stats = ACCVPObservationStats()
        self._cache_key = None
        self._cache_value = None

    def close(self) -> None:
        close = getattr(self._predictor, "close", None)
        if callable(close):
            close()

    def summary(self) -> dict[str, Any]:
        payload = self.stats.to_dict()
        payload.update(
            {
                "accvp_observation_enabled": True,
                "accvp_observation_feature_version": self.FEATURE_VERSION,
                "accvp_observation_dim": int(self.feature_dim(self.config)),
                "accvp_observation_feature_names": self.feature_names(),
                "accvp_observation_activation_distance_m": float(self.activation_distance),
                "accvp_observation_checkpoint": str(self.config.accvp.get("checkpoint", "")),
                "accvp_observation_artifact_manifest": str(self.config.accvp.get("artifact_manifest", "")),
                "accvp_observation_risk_checkpoint": str(self.config.accvp.get("risk_checkpoint", "")),
            }
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
        table = self._extract_uncached(context)
        self._cache_key = key
        self._cache_value = table.astype(np.float32, copy=True)
        return self._cache_value.copy()

    def _extract_uncached(self, context: dict[str, Any]) -> np.ndarray:
        self.stats.total_decisions += 1
        active = self._activation_window(context)
        self.stats.activation_window_decisions += int(active)
        base = self._fail_closed_rows(context)
        if not active:
            return base
        started = time.perf_counter()
        try:
            legal_actions = [action for action in ACTIONS if self._candidate_legal(action, context)]
            if not legal_actions:
                self.stats.fail_closed_count += 1
                return base
            scorer = self._predictor.score_candidates
            if "timeout_s" in inspect.signature(scorer).parameters:
                scores = scorer(context, legal_actions, timeout_s=self.timeout_s)
            else:
                scores = scorer(context, legal_actions)
            rows = self._rows_from_scores(scores, context)
            elapsed = time.perf_counter() - started
            if elapsed > self.timeout_s:
                self.stats.timeout_count += 1
                self.stats.fail_closed_count += 1
                return base
            self.stats.valid_decisions += 1
            self.stats.activation_window_valid_decisions += 1
            return rows
        except TimeoutError:
            self.stats.timeout_count += 1
        except ValueError:
            self.stats.invalid_bundle_count += 1
        except Exception:
            self.stats.model_error_count += 1
        self.stats.fail_closed_count += 1
        if self.fail_closed_defaults:
            return base
        raise

    def _rows_from_scores(self, scores: list[dict[str, Any]], context: dict[str, Any]) -> np.ndarray:
        by_action = {int(row.get("action_id", -1)): row for row in scores}
        rows: list[float] = []
        for action in ACTIONS:
            legal = self._candidate_legal(action, context)
            score = by_action.get(int(action.index))
            if score is None:
                rows.extend(self._fail_closed_row(action, candidate_legal=legal))
                continue
            secondary = self._secondary_safety(action, context)
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
        return np.asarray(rows, dtype=np.float32)

    def _fail_closed_rows(self, context: dict[str, Any]) -> np.ndarray:
        rows: list[float] = []
        for action in ACTIONS:
            rows.extend(self._fail_closed_row(action, candidate_legal=self._candidate_legal(action, context)))
        return np.asarray(rows, dtype=np.float32)

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

    def _activation_window(self, context: dict[str, Any]) -> bool:
        local = context.get("merge_local")
        return bool(
            local is not None
            and bool(getattr(local, "ego_on_auxiliary", False))
            and 0.0 < float(getattr(local, "merge_distance", float("inf"))) <= self.activation_distance
        )

    @staticmethod
    def _candidate_legal(action: CandidateAction, context: dict[str, Any]) -> bool:
        configured = context.get("candidate_legal_by_action")
        if configured is not None:
            return bool(dict(configured).get(int(action.index), False))
        return bool(is_candidate_legal(action, context, missing_ego_is_legal=False))


def validate_accvp_observation_config(config: Any) -> None:
    if not RiskGatedACCVPCandidateTableAugmentor.enabled(config):
        return
    obs_cfg = dict(config.accvp.get("observation", {}) or {})
    if str(obs_cfg.get("feature_version", RISK_GATED_ACCVP_OBSERVATION_VERSION)) != RISK_GATED_ACCVP_OBSERVATION_VERSION:
        raise ValueError("unsupported ACCVP observation feature_version")
    forecast_enabled = bool(config.forecast_features.enabled or config.rl.use_wcdt_forecast_features)
    if forecast_enabled and not bool(obs_cfg.get("allow_with_forecast_features", False)):
        raise ValueError(
            "Risk-gated ACCVP candidate table observation is mutually exclusive with legacy forecast_features by default"
        )
