from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from safe_rl.accvp.calibration import CalibrationBundle
from safe_rl.accvp.candidate_plan import build_commitment_plan, profile_from_config
from safe_rl.accvp.controller import ACCVPController
from safe_rl.accvp.model import ACCVP_ARCHITECTURE_VERSION, ACCVPPredictor, model_kwargs_from_config
from safe_rl.accvp.protocol import counterfactual_data_contract, data_contract_hash, effective_activation_distance
from safe_rl.accvp.schema import COUNTERFACTUAL_SCHEMA_VERSION, file_sha256, read_json
from safe_rl.prediction.wcdt_v3_predictor import build_v3_runtime_batch
from safe_rl.sim.action_space import CandidateAction


def _require_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError("ACCVP runtime requires torch.") from exc
    return torch


class ACCVPRuntimePredictor:
    def __init__(self, config: Any, checkpoint: str | Path, *, use_inference_worker: bool = True):
        torch = _require_torch()
        self.config = config
        self.torch = torch
        self.checkpoint_path = Path(checkpoint).resolve()
        payload = torch.load(self.checkpoint_path, map_location="cpu")
        metadata = dict(payload.get("metadata", {}))
        if metadata.get("architecture_version") != ACCVP_ARCHITECTURE_VERSION:
            raise ValueError("ACCVP checkpoint architecture_version mismatch")
        if int(metadata.get("counterfactual_schema_version", -1)) != COUNTERFACTUAL_SCHEMA_VERSION:
            raise ValueError("ACCVP checkpoint counterfactual schema mismatch")
        expected = model_kwargs_from_config(config)
        checkpoint_kwargs = dict(metadata.get("model_kwargs", {}))
        for key in ("history_steps", "response_horizon_steps", "candidate_plan_horizon_steps"):
            if int(checkpoint_kwargs.get(key, -1)) != int(expected[key]):
                raise ValueError(f"ACCVP checkpoint {key} is incompatible with runtime config")
        states = payload.get("model_state_dicts")
        if not states:
            raise ValueError("ACCVP checkpoint has no ensemble state dicts")
        if len(states) != int(config.accvp.ensemble_size):
            raise ValueError("ACCVP checkpoint ensemble size does not match config")
        self.models = []
        for state in states:
            model = ACCVPPredictor(**checkpoint_kwargs)
            model.load_state_dict(state)
            model.eval()
            self.models.append(model)
        self.calibration = None
        calibration_payload = payload.get("calibration")
        bundle_path = config.accvp.get("calibration_bundle")
        if bundle_path:
            with Path(bundle_path).open("r", encoding="utf-8") as handle:
                calibration_payload = json.load(handle)
        if calibration_payload:
            self.calibration = CalibrationBundle.from_dict(calibration_payload)
        self._inference_worker = None
        if use_inference_worker and bool(config.accvp.get("inference_worker", {}).get("enabled", True)):
            from safe_rl.accvp.inference_worker import PersistentACCVPInferenceWorker

            self._inference_worker = PersistentACCVPInferenceWorker(config, self.checkpoint_path)
            worker_cfg = config.accvp.get("inference_worker", {})
            if bool(worker_cfg.get("prewarm_on_init", True)):
                self._inference_worker.start(float(worker_cfg.get("startup_timeout_s", 15.0)))

    def validate_artifact_bundle(self, operating_point: str | Path | None) -> None:
        manifest_path = self.config.accvp.get("artifact_manifest")
        if not manifest_path:
            raise FileNotFoundError("enabled ACCVP runtime requires accvp.artifact_manifest")
        manifest = read_json(manifest_path)
        if str(manifest.get("artifact_kind", "")) != "accvp_v1_artifact_bundle":
            raise ValueError("invalid ACCVP artifact manifest kind")
        expected = {
            "predictor_sha256": file_sha256(self.checkpoint_path),
        }
        calibration_path = self.config.accvp.get("calibration_bundle")
        if not calibration_path:
            raise FileNotFoundError("enabled ACCVP runtime requires accvp.calibration_bundle")
        expected["calibration_sha256"] = file_sha256(calibration_path)
        if operating_point:
            expected["operating_point_sha256"] = file_sha256(operating_point)
        for key, value in expected.items():
            if str(manifest.get(key, "")) != value:
                raise ValueError(f"ACCVP artifact bundle mismatch for {key}")
        risk_checkpoint = self.config.accvp.get("risk_checkpoint")
        if not risk_checkpoint:
            raise FileNotFoundError("enabled ACCVP runtime requires accvp.risk_checkpoint")
        fingerprint = f"risk_checkpoint:{file_sha256(risk_checkpoint)}"
        if str(manifest.get("risk_model_fingerprint", "")) != fingerprint:
            raise ValueError("ACCVP artifact Risk Module fingerprint mismatch")
        if int(manifest.get("counterfactual_schema_version", -1)) != COUNTERFACTUAL_SCHEMA_VERSION:
            raise ValueError("ACCVP artifact counterfactual schema mismatch")
        expected_activation = effective_activation_distance(self.config)
        if abs(float(manifest.get("accvp_activation_distance_m", -1.0)) - expected_activation) > 1.0e-9:
            raise ValueError("ACCVP artifact activation window mismatch")
        expected_contract = counterfactual_data_contract(self.config, fingerprint)
        if str(manifest.get("data_contract_hash", "")) != data_contract_hash(expected_contract):
            raise ValueError("ACCVP artifact counterfactual data-contract mismatch")

    def prepare_candidates(self, context: dict[str, Any], legal_actions: list[CandidateAction]) -> dict[str, Any]:
        if not legal_actions:
            raise ValueError("cannot prepare ACCVP inference without legal actions")
        ego = context.get("ego")
        history = context.get("history")
        if ego is None or history is None:
            raise ValueError("ACCVP runtime requires ego and history")
        profile_from_config(self.config)
        runtime = build_v3_runtime_batch(self.config, history, str(ego.vehicle_id))
        selected = [str(value) for value in runtime.get("runtime_agent_ids", [])[1:]]
        selection = runtime.get("actor_selection")
        if len(selected) < int(self.config.accvp.actor_count) or bool(
            getattr(selection, "critical_overflow", False)
        ):
            raise ValueError("ACCVP runtime actor coverage is incomplete")
        candidate_plans = np.stack(
            [
                build_commitment_plan(
                    ego,
                    action,
                    step_length=float(self.config.scenario.step_length),
                    horizon_steps=int(self.config.accvp.candidate_plan_horizon_steps),
                ).states
                for action in legal_actions
            ],
            axis=0,
        )
        return {
            "root_inputs": {
                "history_features": np.asarray(runtime["history_features"], dtype=np.float32),
                "history_valid_mask": np.asarray(runtime["history_valid_mask"], dtype=np.float32),
                "history_lane_ids": np.asarray(runtime["history_lane_ids"], dtype=np.int64),
                "history_edge_role_ids": np.asarray(runtime["history_edge_role_ids"], dtype=np.int64),
                "role_ids": np.asarray(runtime["role_ids"], dtype=np.int64),
                "lane_ids": np.asarray(runtime["lane_ids"], dtype=np.int64),
                "edge_role_ids": np.asarray(runtime["edge_role_ids"], dtype=np.int64),
                "actor_mask": np.asarray(runtime["mask"], dtype=np.float32),
            },
            "candidate_plans": candidate_plans.astype(np.float32),
            "action_ids": np.asarray([action.index for action in legal_actions], dtype=np.int64),
        }

    def score_prepared(self, prepared: dict[str, Any]) -> list[dict[str, Any]]:
        torch = self.torch
        root_inputs = {
            name: torch.as_tensor(
                np.asarray(value),
                dtype=torch.long if name in {"history_lane_ids", "history_edge_role_ids", "role_ids", "lane_ids", "edge_role_ids"} else torch.float32,
            )
            for name, value in dict(prepared["root_inputs"]).items()
        }
        action_ids = torch.as_tensor(np.asarray(prepared["action_ids"]), dtype=torch.long)
        plans = torch.as_tensor(np.asarray(prepared["candidate_plans"]), dtype=torch.float32)
        candidate_count = int(action_ids.shape[0])
        event_members = []
        geometry_members = []
        with torch.no_grad():
            for model in self.models:
                scene = model.encode_scene(**root_inputs)
                expanded_scene = scene.expand(candidate_count, -1, -1)
                expanded_mask = root_inputs["actor_mask"].expand(candidate_count, -1)
                output = model.forward_from_scene(expanded_scene, expanded_mask, plans, action_ids)
                event_members.append(torch.sigmoid(output["event_logits"]).cpu().numpy())
                geometry_members.append(output["geometry"].cpu().numpy())
        events = np.stack(event_members, axis=0)
        geometry = np.stack(geometry_members, axis=0)
        # Conservative ensemble aggregation: risk heads use max, viability uses min.
        result: list[dict[str, Any]] = []
        for index, action_id in enumerate(np.asarray(prepared["action_ids"], dtype=np.int64)):
            q = np.median(geometry[:, index], axis=0)
            result.append(
                {
                    "action_id": int(action_id),
                    "p_proxy_collision": float(events[:, index, 0].max()),
                    "p_safety_violation": float(events[:, index, 1].max()),
                    "p_taper_miss": float(events[:, index, 2].max()),
                    "p_merge_before_taper": float(events[:, index, 3].min()),
                    "q10_min_distance": float(max(0.0, q[0])),
                    "q90_drac": float(max(0.0, q[1])),
                    "target_front_gap": float(max(0.0, q[2])),
                    "target_rear_gap": float(max(0.0, q[3])),
                    "target_lane_entry_time_s": float(max(0.0, q[4])),
                    "ensemble_disagreement": float(events[:, index].std(axis=0).mean()),
                }
            )
        return result

    def score_candidates(
        self,
        context: dict[str, Any],
        legal_actions: list[CandidateAction],
        *,
        timeout_s: float | None = None,
    ) -> list[dict[str, Any]]:
        if not legal_actions:
            return []
        prepared = self.prepare_candidates(context, legal_actions)
        if self._inference_worker is not None:
            return self._inference_worker.score(
                prepared,
                float(self.config.accvp.max_decision_latency_s) if timeout_s is None else float(timeout_s),
            )
        return self.score_prepared(prepared)

    def close(self) -> None:
        if self._inference_worker is not None:
            self._inference_worker.close()
            self._inference_worker = None


def build_accvp_controller(config: Any) -> ACCVPController | None:
    if not bool(config.accvp.get("enabled", False)) or str(config.accvp.get("mode", "off")) == "off":
        return None
    checkpoint = config.accvp.get("checkpoint")
    if not checkpoint:
        raise FileNotFoundError("accvp.enabled requires accvp.checkpoint")
    predictor = ACCVPRuntimePredictor(config, checkpoint)
    operating_point = config.accvp.get("operating_point")
    if operating_point:
        with Path(operating_point).open("r", encoding="utf-8") as handle:
            selected = dict(json.load(handle).get("selected", {}))
        required = {
            "proxy_collision_upper_bound",
            "safety_violation_upper_bound",
            "merge_viability_lower_bound",
        }
        if required.difference(selected):
            raise ValueError("ACCVP operating-point bundle is missing selected gate thresholds")
        config.accvp["proxy_collision_upper_bound"] = float(selected["proxy_collision_upper_bound"])
        config.accvp["safety_violation_upper_bound"] = float(selected["safety_violation_upper_bound"])
        config.accvp["merge_viability_lower_bound"] = float(selected["merge_viability_lower_bound"])
    elif str(config.accvp.get("mode", "off")) == "viability_branch":
        raise FileNotFoundError("accvp viability_branch requires accvp.operating_point from the held-out tuning split")
    predictor.validate_artifact_bundle(operating_point)
    return ACCVPController(config, predictor, predictor.calibration)
