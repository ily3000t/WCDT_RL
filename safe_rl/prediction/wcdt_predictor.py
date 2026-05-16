from __future__ import annotations

from pathlib import Path
from typing import Any

from safe_rl.prediction.sumo_wcdt_adapter import SumoWcDTAdapter


def _require_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise ImportError("WcDTPredictor requires torch. Activate the SAFE_RL training environment.") from exc
    return torch


def _resolve_device(config: Any, torch: Any):
    requested = str(config.get("training", {}).get("device", "auto")).strip().lower()
    if requested in ("auto", ""):
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "gpu":
        requested = "cuda"
    return torch.device(requested)


class WcDTPredictor:
    """Runtime wrapper for the Stage2 WcDT-style predictor checkpoint."""

    def __init__(self, config: Any, checkpoint: str | Path):
        torch = _require_torch()
        from net_works import BackBone
        from utils import MathUtil

        self.config = config
        self.checkpoint_path = str(Path(checkpoint).resolve())
        self.device = _resolve_device(config, torch)
        self.adapter = SumoWcDTAdapter(config)
        betas = MathUtil.generate_linear_schedule(50, 1e-4, 0.008)
        self.model = BackBone(betas).to(self.device)
        payload = torch.load(self.checkpoint_path, map_location=self.device)
        state = payload["model_state_dict"] if isinstance(payload, dict) and "model_state_dict" in payload else payload
        self.model.load_state_dict(state, strict=False)
        self.model.eval()
        self._torch = torch

    def predict(self, context: dict[str, Any]) -> dict[str, Any]:
        ego = context.get("ego")
        history = context.get("history")
        if ego is None or history is None:
            raise ValueError("WcDTPredictor requires ego and history in the risk context.")
        data = self.adapter.to_wcdt_input(history, str(ego.vehicle_id))
        tensor_data = {
            key: value.to(self.device)
            for key, value in data.items()
            if hasattr(value, "to")
        }
        with self._torch.no_grad():
            return self.model.predict(
                tensor_data,
                horizon_steps=int(self.config.forecast_features.horizon_steps),
            )
