from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from safe_rl.accvp.calibration import CalibrationBundle
from safe_rl.accvp.candidate_plan import build_commitment_plan, profile_from_config
from safe_rl.accvp.controller import ACCVPController
from safe_rl.accvp.model import ACCVP_ARCHITECTURE_VERSION, ACCVPPredictor, model_kwargs_from_config
from safe_rl.prediction.wcdt_v3_predictor import build_v3_runtime_batch
from safe_rl.sim.action_space import CandidateAction


def _require_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError("ACCVP runtime requires torch.") from exc
    return torch


class ACCVPRuntimePredictor:
    def __init__(self, config: Any, checkpoint: str | Path):
        torch = _require_torch()
        self.config = config
        self.torch = torch
        self.checkpoint_path = Path(checkpoint).resolve()
        payload = torch.load(self.checkpoint_path, map_location="cpu")
        metadata = dict(payload.get("metadata", {}))
        if metadata.get("architecture_version") != ACCVP_ARCHITECTURE_VERSION:
            raise ValueError("ACCVP checkpoint architecture_version mismatch")
        if int(metadata.get("counterfactual_schema_version", -1)) != 1:
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

    def score_candidates(self, context: dict[str, Any], legal_actions: list[CandidateAction]) -> list[dict[str, Any]]:
        if not legal_actions:
            return []
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
        torch = self.torch
        def tensor(name: str, dtype: Any):
            array = np.asarray(runtime[name])
            return torch.as_tensor(array, dtype=dtype)
        root_inputs = {
            "history_features": tensor("history_features", torch.float32),
            "history_valid_mask": tensor("history_valid_mask", torch.float32),
            "history_lane_ids": tensor("history_lane_ids", torch.long),
            "history_edge_role_ids": tensor("history_edge_role_ids", torch.long),
            "role_ids": tensor("role_ids", torch.long),
            "lane_ids": tensor("lane_ids", torch.long),
            "edge_role_ids": tensor("edge_role_ids", torch.long),
            "actor_mask": tensor("mask", torch.float32),
        }
        action_ids = torch.as_tensor([action.index for action in legal_actions], dtype=torch.long)
        plans = torch.as_tensor(candidate_plans, dtype=torch.float32)
        event_members = []
        geometry_members = []
        with torch.no_grad():
            for model in self.models:
                scene = model.encode_scene(**root_inputs)
                expanded_scene = scene.expand(len(legal_actions), -1, -1)
                expanded_mask = root_inputs["actor_mask"].expand(len(legal_actions), -1)
                output = model.forward_from_scene(expanded_scene, expanded_mask, plans, action_ids)
                event_members.append(torch.sigmoid(output["event_logits"]).cpu().numpy())
                geometry_members.append(output["geometry"].cpu().numpy())
        events = np.stack(event_members, axis=0)
        geometry = np.stack(geometry_members, axis=0)
        # Conservative ensemble aggregation: risk heads use max, viability uses min.
        result: list[dict[str, Any]] = []
        for index, action in enumerate(legal_actions):
            q = np.median(geometry[:, index], axis=0)
            result.append(
                {
                    "action_id": int(action.index),
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
    return ACCVPController(config, predictor, predictor.calibration)
