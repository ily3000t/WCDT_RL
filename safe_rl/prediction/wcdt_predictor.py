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
        if int(self.model.traj_decoder.multimodal) != 10:
            raise ValueError(
                "WcDT v1 adapted baseline requires the upstream 10-mode TrajDecoder; "
                f"found {self.model.traj_decoder.multimodal}."
            )
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
            "actor_row_alignment",
            "trajectory_selector_order_version",
            "num_modes",
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
                int(
                    self.config.prediction.get(
                        "wcdt_v1_max_agents",
                        self.config.prediction.max_pred_num,
                    )
                ),
            ),
            "actor_row_alignment": (
                payload.get("actor_row_alignment"),
                "selector_v2_vehicle_id_verified",
            ),
            "trajectory_selector_order_version": (
                payload.get("trajectory_selector_order_version"),
                "selector_v2_vehicle_id_order_v1",
            ),
            "num_modes": (int(payload.get("num_modes", -1)), 10),
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
        prediction["num_modes"] = int(self.model.traj_decoder.multimodal)
        mode_confidence = prediction.get("mode_confidence")
        if mode_confidence is not None:
            confidence = self._to_numpy(mode_confidence).astype(np.float32)
            if confidence.ndim == 3:
                confidence = confidence[0]
            if confidence.ndim == 2 and confidence.shape[1] > 0:
                mode_probabilities = np.mean(confidence[: len(selected_ids)], axis=0)
                mode_probabilities = mode_probabilities / max(float(np.sum(mode_probabilities)), 1.0e-8)
                prediction["mode_probabilities"] = mode_probabilities.tolist()
                entropy = -float(np.sum(mode_probabilities * np.log(np.maximum(mode_probabilities, 1.0e-8))))
                entropy /= max(float(np.log(max(2, mode_probabilities.size))), 1.0)
                trajectories = self._to_numpy(prediction["future_trajectories"]).astype(np.float32)
                if trajectories.ndim == 5:
                    trajectories = trajectories[0]
                dispersion = 0.0
                if trajectories.ndim == 4 and trajectories.shape[1] > 1:
                    centered = trajectories[: len(selected_ids)] - np.mean(
                        trajectories[: len(selected_ids)], axis=1, keepdims=True
                    )
                    dispersion = float(np.mean(np.linalg.norm(centered[..., :2], axis=-1)))
                prediction["mode_entropy"] = float(entropy)
                prediction["mode_trajectory_dispersion"] = float(dispersion)
                prediction["uncertainty"] = float(
                    np.clip(0.5 * entropy + 0.5 * (dispersion / 5.0), 0.0, 1.0)
                )
        uncertainty = prediction.get("uncertainty")
        if uncertainty is not None:
            values = self._to_numpy(uncertainty).astype(np.float32)
            if values.ndim >= 2:
                values = values[0]
            values = values.reshape(-1)
            prediction["actor_uncertainty"] = values[: len(selected_ids)].tolist()
            prediction["uncertainty"] = float(np.nanmax(values)) if values.size else 0.0
        return prediction
