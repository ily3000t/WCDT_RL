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
            "mode_aggregation_version",
            "joint_world_count",
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
            "mode_aggregation_version": (
                payload.get("mode_aggregation_version"),
                str(
                    self.config.prediction.get("wcdt_v1_mode_aggregation", {}).get(
                        "version", "per_actor_joint_world_v1"
                    )
                ),
            ),
            "joint_world_count": (
                int(payload.get("joint_world_count", -1)),
                int(
                    self.config.prediction.get("wcdt_v1_mode_aggregation", {}).get(
                        "joint_world_count", 32
                    )
                ),
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
        prediction["num_modes"] = int(self.model.traj_decoder.multimodal)
        mode_confidence = prediction.get("mode_confidence")
        if mode_confidence is not None:
            confidence = self._to_numpy(mode_confidence).astype(np.float32)
            if confidence.ndim == 3:
                confidence = confidence[0]
            if confidence.ndim == 2 and confidence.shape[1] > 0:
                actor_probabilities = np.clip(confidence[: len(selected_ids)], 0.0, None)
                normalizer = np.sum(actor_probabilities, axis=1, keepdims=True)
                invalid = normalizer[:, 0] <= 1.0e-8
                if np.any(invalid):
                    actor_probabilities[invalid] = 1.0
                    normalizer = np.sum(actor_probabilities, axis=1, keepdims=True)
                actor_probabilities = actor_probabilities / np.maximum(normalizer, 1.0e-8)
                trajectories = self._to_numpy(prediction["future_trajectories"]).astype(np.float32)
                if trajectories.ndim == 5:
                    trajectories = trajectories[0]
                dispersion = np.zeros((actor_probabilities.shape[0],), dtype=np.float32)
                if trajectories.ndim == 4 and trajectories.shape[1] > 1:
                    trajectories = trajectories[: len(selected_ids), : actor_probabilities.shape[1]]
                    mean_trajectory = np.sum(
                        trajectories * actor_probabilities[:, :, None, None], axis=1, keepdims=True
                    )
                    displacement = np.linalg.norm(trajectories[..., :2] - mean_trajectory[..., :2], axis=-1)
                    dispersion = np.sum(
                        np.mean(displacement, axis=-1) * actor_probabilities,
                        axis=1,
                    ).astype(np.float32)
                entropy = -np.sum(
                    actor_probabilities * np.log(np.maximum(actor_probabilities, 1.0e-8)), axis=1
                ) / max(float(np.log(max(2, actor_probabilities.shape[1]))), 1.0)
                actor_uncertainty = np.clip(0.5 * entropy + 0.5 * (dispersion / 5.0), 0.0, 1.0)
                prediction["actor_mode_probabilities"] = actor_probabilities.tolist()
                prediction["actor_mode_entropy"] = entropy.astype(np.float32).tolist()
                prediction["actor_mode_trajectory_dispersion"] = dispersion.tolist()
                prediction["actor_uncertainty"] = actor_uncertainty.astype(np.float32).tolist()
                prediction["uncertainty"] = float(np.max(actor_uncertainty)) if actor_uncertainty.size else 0.0
        elif prediction.get("uncertainty") is not None:
            values = self._to_numpy(prediction["uncertainty"]).astype(np.float32)
            if values.ndim >= 2:
                values = values[0]
            values = values.reshape(-1)
            prediction["actor_uncertainty"] = values[: len(selected_ids)].tolist()
            prediction["uncertainty"] = float(np.nanmax(values)) if values.size else 0.0
        return prediction
