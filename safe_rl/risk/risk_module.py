from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from safe_rl.risk.merge_local import evaluate_candidate_action_risk, evaluate_candidate_actions
from safe_rl.sim.action_space import CandidateAction
from safe_rl.sim.metrics import SAFETY_METRIC_VERSION

try:
    import torch
    from torch import nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover
    torch = None
    nn = None
    F = None


@dataclass
class RiskPrediction:
    risk_score: float
    risk_type_logits: np.ndarray
    risk_uncertainty: float
    explicit_features: np.ndarray


if torch is not None:

    class RiskModule(nn.Module):
        def __init__(
            self,
            explicit_dim: int = 8,
            latent_dim: int = 256,
            action_embedding_dim: int = 4,
            hidden_dim: int = 128,
            risk_type_count: int = 6,
        ):
            super().__init__()
            self.action_embedding = nn.Embedding(9, action_embedding_dim)
            input_dim = explicit_dim + latent_dim + action_embedding_dim + 1
            self.backbone = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
            )
            self.risk_head = nn.Linear(hidden_dim, 1)
            self.type_head = nn.Linear(hidden_dim, int(risk_type_count))
            self.uncertainty_head = nn.Linear(hidden_dim, 1)

        def forward(self, explicit_features, action_index, latent=None, uncertainty=None):
            batch = explicit_features.shape[0]
            if latent is None:
                latent = torch.zeros((batch, 256), dtype=explicit_features.dtype, device=explicit_features.device)
            if uncertainty is None:
                uncertainty = torch.zeros((batch, 1), dtype=explicit_features.dtype, device=explicit_features.device)
            if uncertainty.ndim == 1:
                uncertainty = uncertainty.unsqueeze(-1)
            action_emb = self.action_embedding(action_index.to(torch.long))
            x = torch.cat([explicit_features, latent, action_emb, uncertainty], dim=-1)
            hidden = self.backbone(x)
            return {
                "risk_score": torch.sigmoid(self.risk_head(hidden)).squeeze(-1),
                "risk_type_logits": self.type_head(hidden),
                "risk_uncertainty": torch.sigmoid(self.uncertainty_head(hidden)).squeeze(-1),
            }

else:

    class RiskModule:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError("RiskModule requires torch. Activate the SAFE_RL training environment.")


class HeuristicRiskEstimator:
    """Deterministic fallback used before a learned risk checkpoint exists."""

    def __init__(self, config: Any):
        self.config = config

    def predict(self, action: CandidateAction, context: dict[str, Any]) -> RiskPrediction:
        candidate = evaluate_candidate_action_risk(action, context)
        return self.predict_candidate(candidate)

    def predict_candidate(self, candidate: Any) -> RiskPrediction:
        features = candidate.features
        weights = np.asarray([0.30, 0.25, 0.20, 0.50, 0.15, 0.80, 0.05, 0.25], dtype=np.float32)
        score = float(np.clip(np.dot(features, weights), 0.0, 1.0))
        uncertainty = float(0.1 + 0.2 * features[-1])
        type_logits = np.zeros((int(self.config.risk_module.get("risk_type_count", 6)),), dtype=np.float32)
        legacy_type_logits = features[[3, 7, 1, 2, 4]].astype(np.float32)
        type_logits[: min(type_logits.shape[0], legacy_type_logits.shape[0])] = legacy_type_logits[: type_logits.shape[0]]
        if type_logits.shape[0] > 5:
            type_logits[5] = float(candidate.risk_types[5])
        return RiskPrediction(
            risk_score=score,
            risk_type_logits=type_logits,
            risk_uncertainty=uncertainty,
            explicit_features=features,
        )


class RiskModuleWrapper:
    def __init__(self, config: Any, checkpoint: str | None = None):
        self.config = config
        self.estimator: HeuristicRiskEstimator | None = HeuristicRiskEstimator(config)
        self.model = None
        self.temperature = 1.0
        self.apply_temperature = False
        if checkpoint:
            self.load(checkpoint)

    def load(self, checkpoint: str | Path) -> None:
        if torch is None:
            raise ImportError("Loading learned risk checkpoints requires torch.")
        payload = torch.load(checkpoint, map_location="cpu")
        if isinstance(payload, dict):
            metric_version = str(payload.get("safety_metric_version", ""))
            expected = str(self.config.risk_module.get("safety_metric_version", SAFETY_METRIC_VERSION))
            if metric_version != expected:
                raise ValueError(
                    f"unsupported Risk Module safety_metric_version={metric_version!r}; expected {expected!r}"
                )
            checkpoint_ordering = str(payload.get("vehicle_state_ordering_version", ""))
            configured_ordering = str(
                self.config.scenario.get(
                    "vehicle_state_ordering_version",
                    "unspecified_legacy",
                )
            )
            if checkpoint_ordering != configured_ordering:
                raise ValueError(
                    "Risk Module vehicle_state_ordering_version mismatch: "
                    f"checkpoint={checkpoint_ordering!r}, runtime={configured_ordering!r}"
                )
        state = payload["model_state_dict"] if isinstance(payload, dict) and "model_state_dict" in payload else payload
        model = RiskModule(
            explicit_dim=int(self.config.risk_module.explicit_feature_dim),
            latent_dim=int(self.config.risk_module.latent_dim),
            action_embedding_dim=int(self.config.risk_module.action_embedding_dim),
            hidden_dim=int(self.config.risk_module.hidden_dim),
            risk_type_count=int(state.get("type_head.weight", torch.empty((6, 0))).shape[0]),
        )
        model.load_state_dict(state)
        model.eval()
        self.model = model
        calibration_cfg = self.config.risk_module.get("calibration", {})
        if not isinstance(calibration_cfg, dict):
            calibration_cfg = {}
        self.temperature = float(payload.get("temperature", 1.0)) if isinstance(payload, dict) else 1.0
        self.apply_temperature = bool(
            (payload.get("apply_temperature", False) if isinstance(payload, dict) else False)
            or calibration_cfg.get("use_for_runtime", False)
        )

    def save(self, checkpoint: str | Path) -> None:
        if torch is None or self.model is None:
            raise RuntimeError("No learned risk model is available to save.")
        Path(checkpoint).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "safety_metric_version": str(
                    self.config.risk_module.get("safety_metric_version", SAFETY_METRIC_VERSION)
                ),
                "vehicle_state_ordering_version": str(
                    self.config.scenario.get(
                        "vehicle_state_ordering_version",
                        "unspecified_legacy",
                    )
                ),
            },
            checkpoint,
        )

    def predict(self, action: CandidateAction, context: dict[str, Any]) -> RiskPrediction:
        return self.predict_many([action], context)[0]

    def predict_many(
        self,
        actions: list[CandidateAction] | tuple[CandidateAction, ...],
        context: dict[str, Any],
    ) -> list[RiskPrediction]:
        actions = list(actions)
        if not actions:
            return []
        cache = context.setdefault("_risk_prediction_cache", {})
        missing = [action for action in actions if (id(self), int(action.index)) not in cache]
        if missing:
            tracker = context.get("performance_tracker")
            started = None
            if tracker is not None:
                import time

                started = time.perf_counter()
            candidates = evaluate_candidate_actions(missing, context)
            if self.model is None:
                predictions = [self.estimator.predict_candidate(candidate) for candidate in candidates]  # type: ignore[union-attr]
            else:
                features = np.stack([candidate.features for candidate in candidates], axis=0).astype(np.float32)
                action_indices = np.asarray([action.index for action in missing], dtype=np.int64)
                with torch.no_grad():
                    output = self.model(
                        torch.as_tensor(features, dtype=torch.float32),
                        torch.as_tensor(action_indices, dtype=torch.long),
                    )
                risk_scores = output["risk_score"].detach().cpu().numpy().astype(np.float64)
                if self.apply_temperature and self.temperature > 1.0e-6:
                    clipped = np.clip(risk_scores, 1.0e-6, 1.0 - 1.0e-6)
                    logits = np.log(clipped / (1.0 - clipped))
                    risk_scores = 1.0 / (1.0 + np.exp(-logits / self.temperature))
                type_logits = output["risk_type_logits"].detach().cpu().numpy()
                uncertainties = output["risk_uncertainty"].detach().cpu().numpy()
                predictions = [
                    RiskPrediction(
                        risk_score=float(risk_scores[index]),
                        risk_type_logits=type_logits[index],
                        risk_uncertainty=float(uncertainties[index]),
                        explicit_features=features[index],
                    )
                    for index in range(len(missing))
                ]
            for action, prediction in zip(missing, predictions):
                cache[(id(self), int(action.index))] = prediction
            if tracker is not None and started is not None:
                import time

                tracker.add_time("risk_forward_time", time.perf_counter() - started)
                tracker.increment("risk_forwards")
                tracker.increment("risk_candidates", len(missing))
        return [cache[(id(self), int(action.index))] for action in actions]


def _weighted_mean(values: Any, sample_weight: Any | None = None) -> Any:
    if sample_weight is None:
        return values.mean()
    sample_weight = sample_weight.float()
    if values.ndim > sample_weight.ndim:
        sample_weight = sample_weight.view(sample_weight.shape[0], *([1] * (values.ndim - 1)))
    weighted = values * sample_weight
    denom = sample_weight.sum() * (values.numel() / max(values.shape[0], 1))
    if float(denom.detach().cpu()) <= 1.0e-8:
        return values.sum() * 0.0
    return weighted.sum() / denom


def risk_loss(output: dict[str, Any], labels: dict[str, Any], weights: dict[str, float]) -> Any:
    if torch is None or F is None:
        raise ImportError("risk_loss requires torch.")
    sample_weight = labels.get("sample_weight")
    target = labels["risk_score"].float()
    risk = _weighted_mean(
        F.binary_cross_entropy(output["risk_score"], target, reduction="none"),
        sample_weight,
    )
    type_logits = output["risk_type_logits"]
    type_targets = labels["risk_types"].float()
    type_dim = min(int(type_logits.shape[-1]), int(type_targets.shape[-1]))
    type_loss = _weighted_mean(
        F.binary_cross_entropy_with_logits(type_logits[..., :type_dim], type_targets[..., :type_dim], reduction="none"),
        sample_weight,
    )
    calib = _weighted_mean(
        torch.square(output["risk_uncertainty"] - torch.abs(output["risk_score"].detach() - target)),
        sample_weight,
    )
    return weights.get("risk", 1.0) * (risk + type_loss) + weights.get("calibration", 0.1) * calib
