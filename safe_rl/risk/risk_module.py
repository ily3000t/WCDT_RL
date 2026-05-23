from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from safe_rl.risk.risk_feature_extractor import extract_candidate_features
from safe_rl.sim.action_space import CandidateAction

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
            self.type_head = nn.Linear(hidden_dim, 5)
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
        features = extract_candidate_features(action, context)
        weights = np.asarray([0.30, 0.25, 0.20, 0.50, 0.15, 0.80, 0.05, 0.25], dtype=np.float32)
        score = float(np.clip(np.dot(features, weights), 0.0, 1.0))
        uncertainty = float(0.1 + 0.2 * features[-1])
        return RiskPrediction(
            risk_score=score,
            risk_type_logits=features[[3, 7, 1, 2, 4]].astype(np.float32),
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
        model = RiskModule(
            explicit_dim=int(self.config.risk_module.explicit_feature_dim),
            latent_dim=int(self.config.risk_module.latent_dim),
            action_embedding_dim=int(self.config.risk_module.action_embedding_dim),
            hidden_dim=int(self.config.risk_module.hidden_dim),
        )
        state = payload["model_state_dict"] if isinstance(payload, dict) and "model_state_dict" in payload else payload
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
        torch.save({"model_state_dict": self.model.state_dict()}, checkpoint)

    def predict(self, action: CandidateAction, context: dict[str, Any]) -> RiskPrediction:
        if self.model is None:
            return self.estimator.predict(action, context)  # type: ignore[union-attr]
        features = extract_candidate_features(action, context)
        with torch.no_grad():
            explicit = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
            action_index = torch.tensor([action.index], dtype=torch.long)
            output = self.model(explicit, action_index)
        risk_score = float(output["risk_score"].cpu().numpy()[0])
        if self.apply_temperature and self.temperature > 1.0e-6:
            clipped = float(np.clip(risk_score, 1.0e-6, 1.0 - 1.0e-6))
            logit = np.log(clipped / (1.0 - clipped))
            risk_score = float(1.0 / (1.0 + np.exp(-logit / self.temperature)))
        return RiskPrediction(
            risk_score=risk_score,
            risk_type_logits=output["risk_type_logits"].cpu().numpy()[0],
            risk_uncertainty=float(output["risk_uncertainty"].cpu().numpy()[0]),
            explicit_features=features,
        )


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
    type_loss = _weighted_mean(
        F.binary_cross_entropy_with_logits(output["risk_type_logits"], labels["risk_types"].float(), reduction="none"),
        sample_weight,
    )
    calib = _weighted_mean(
        torch.square(output["risk_uncertainty"] - torch.abs(output["risk_score"].detach() - target)),
        sample_weight,
    )
    return weights.get("risk", 1.0) * (risk + type_loss) + weights.get("calibration", 0.1) * calib
