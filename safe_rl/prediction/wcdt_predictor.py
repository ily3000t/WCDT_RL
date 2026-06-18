from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from safe_rl.prediction.actor_selector import ACTOR_SELECTION_VERSION, actor_selection_config_hash
from safe_rl.prediction.sumo_wcdt_adapter import SumoWcDTAdapter
from safe_rl.sim.metrics import SAFETY_METRIC_VERSION
from safe_rl.utils.stage1_dataset import STAGE1_BUFFER_SCHEMA_VERSION


def _require_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise ImportError("WcDTPredictor requires torch. Activate the SAFE_RL training environment.") from exc
    return torch


def _resolve_device(config: Any, torch: Any):
    training = config.get("training", {})
    requested = str(training.get("forecast_runtime_device", training.get("device", "auto"))).strip().lower()
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
        self.payload = payload if isinstance(payload, dict) else {}
        self.legacy_checkpoint_metadata = not self._has_formal_metadata(self.payload)
        self._validate_metadata(self.payload)
        state = payload["model_state_dict"] if isinstance(payload, dict) and "model_state_dict" in payload else payload
        self.model.load_state_dict(state, strict=False)
        self.model.eval()
        self._torch = torch

    @staticmethod
    def _has_formal_metadata(payload: dict[str, Any]) -> bool:
        required = {
            "safety_metric_version",
            "actor_selection_version",
            "actor_selection_config_hash",
            "trajectory_schema_version",
            "stage1_buffer_schema_version",
            "max_actor_count",
        }
        return bool(payload) and required.issubset(payload)

    def _validate_metadata(self, payload: dict[str, Any]) -> None:
        if not payload or not self._has_formal_metadata(payload):
            return
        expected_hash = actor_selection_config_hash(self.config)
        checks = {
            "safety_metric_version": (payload.get("safety_metric_version"), SAFETY_METRIC_VERSION),
            "actor_selection_version": (payload.get("actor_selection_version"), ACTOR_SELECTION_VERSION),
            "actor_selection_config_hash": (payload.get("actor_selection_config_hash"), expected_hash),
            "stage1_buffer_schema_version": (
                int(payload.get("stage1_buffer_schema_version", 0)),
                STAGE1_BUFFER_SCHEMA_VERSION,
            ),
            "max_actor_count": (
                int(payload.get("max_actor_count", -1)),
                int(self.config.prediction.max_pred_num),
            ),
        }
        mismatches = {
            key: {"found": found, "expected": expected}
            for key, (found, expected) in checks.items()
            if found != expected
        }
        if int(payload.get("trajectory_schema_version", 0)) < 4:
            mismatches["trajectory_schema_version"] = {
                "found": payload.get("trajectory_schema_version"),
                "expected": ">=4",
            }
        if mismatches:
            raise ValueError(
                f"WcDT v1 checkpoint metadata is incompatible with the current route/selector schema: "
                f"{mismatches}. Re-train the legacy WcDT v1 predictor for this run."
            )

    @staticmethod
    def _to_numpy(value: Any) -> np.ndarray:
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        return np.asarray(value)

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
            prediction = dict(self.model.predict(
                tensor_data,
                horizon_steps=int(self.config.forecast_features.horizon_steps),
            ))
        selected_ids = list(data.get("selected_vehicle_ids", data.get("predicted_ids", [])))
        prediction["selected_vehicle_ids"] = selected_ids
        prediction["forecast_source"] = "wcdt"
        prediction["checkpoint"] = self.checkpoint_path
        prediction["legacy_checkpoint_metadata"] = bool(self.legacy_checkpoint_metadata)
        uncertainty = prediction.get("uncertainty")
        if uncertainty is not None:
            values = self._to_numpy(uncertainty).astype(np.float32)
            if values.ndim >= 2:
                values = values[0]
            values = values.reshape(-1)
            prediction["actor_uncertainty"] = values[: len(selected_ids)].tolist()
            prediction["uncertainty"] = float(np.nanmax(values)) if values.size else 0.0
        return prediction
