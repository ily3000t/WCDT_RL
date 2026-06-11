from __future__ import annotations

import numpy as np

from safe_rl.prediction.actor_selector import select_merge_relevant_actors
from safe_rl.prediction.forecast_rollout_bundle import build_forecast_rollout_bundle
from safe_rl.sim.history_buffer import HistoryBuffer
from safe_rl.sim.types import VehicleState
from safe_rl.utils.config import load_config


def _vehicle(
    vehicle_id: str,
    lane_pos: float,
    *,
    lane_index: int,
    speed: float = 20.0,
    edge_id: str = "main_aux",
) -> VehicleState:
    return VehicleState(
        vehicle_id=vehicle_id,
        x=300.0 + lane_pos,
        y=53.8 + 3.2 * lane_index,
        heading=0.0,
        speed=speed,
        lane_index=lane_index,
        lane_id=f"{edge_id}_{lane_index}",
        lane_pos=lane_pos,
        edge_id=edge_id,
    )


def test_history_buffer_explicit_actor_order_preserves_legacy_agent_ids():
    cfg = load_config()
    history = HistoryBuffer(history_steps=2, max_agents=4)
    ego = _vehicle("ego", 100.0, lane_index=0)
    alpha = _vehicle("alpha", 110.0, lane_index=1)
    beta = _vehicle("beta", 90.0, lane_index=1)
    gamma = _vehicle("gamma", 80.0, lane_index=2)
    delta = _vehicle("delta", 70.0, lane_index=2)
    history.append([ego, alpha, beta, gamma, delta])

    assert history.agent_ids("ego") == ["ego", "alpha", "beta", "delta"]
    assert history.all_agent_ids("ego") == ["ego", "alpha", "beta", "delta", "gamma"]
    arrays = history.build_tensor_for_ids("ego", ["gamma", "beta"], cfg)
    assert arrays["agent_ids"].tolist() == ["ego", "gamma", "beta"]
    assert arrays["history"][1, -1, 0] == gamma.x
    assert arrays["history"][2, -1, 0] == beta.x


def test_far_non_closing_rear_is_not_relevant_but_fast_closing_rear_is():
    cfg = load_config()
    ego = _vehicle("ego", 120.0, lane_index=0, speed=20.0)
    front = _vehicle("front", 132.0, lane_index=1, speed=20.0)
    rear = _vehicle("rear", 51.0, lane_index=1, speed=20.0)
    selection = select_merge_relevant_actors(cfg, ego, [ego, front, rear], 1)
    assert "front" in selection.relevant_actor_ids
    assert "rear" not in selection.relevant_actor_ids
    assert selection.selected_actor_ids == ("front",)

    closing_rear = _vehicle("rear", 51.0, lane_index=1, speed=45.0)
    closing = select_merge_relevant_actors(cfg, ego, [ego, front, closing_rear], 2)
    assert "rear" in closing.relevant_actor_ids
    assert "effective_gap" in closing.actor_metadata["rear"].relevance_reasons


def test_selector_reports_relevant_overflow():
    cfg = load_config()
    ego = _vehicle("ego", 100.0, lane_index=0)
    actors = [
        _vehicle(f"actor_{index}", 105.0 + index, lane_index=1 + index % 2)
        for index in range(6)
    ]
    selection = select_merge_relevant_actors(cfg, ego, [ego, *actors], 5)
    assert selection.relevant_count == 6
    assert selection.overflow
    assert len(selection.selected_actor_ids) == 5
    assert len(selection.dropped_relevant_ids) == 1


def test_forecast_bundle_uses_cv_fallback_for_non_selected_target_rear():
    cfg = load_config()
    cfg.forecast_features["source"] = "wcdt_v3"
    ego = _vehicle("ego", 120.0, lane_index=0, speed=20.0)
    front = _vehicle("front", 132.0, lane_index=1, speed=20.0)
    rear = _vehicle("rear", 51.0, lane_index=1, speed=20.0)
    history = HistoryBuffer(history_steps=cfg.scenario.history_steps, max_agents=6)
    history.append([ego, front, rear])
    horizon = int(cfg.forecast_features.horizon_steps)
    front_trajectory = np.zeros((1, horizon, 5), dtype=np.float32)
    front_trajectory[0, :, 0] = np.linspace(front.x, front.x + front.speed * horizon * cfg.scenario.step_length, horizon)
    front_trajectory[0, :, 1] = front.y
    front_trajectory[0, :, 3] = front.speed

    class _FrontOnlyPredictor:
        checkpoint_path = "front_only.pt"

        def predict(self, _context):
            return {
                "future_trajectories": front_trajectory,
                "selected_vehicle_ids": ["front"],
                "uncertainty": 0.10,
                "forecast_source": "wcdt_v3",
            }

    context = {
        "ego": ego,
        "vehicles": [ego, front, rear],
        "history": history,
        "config": cfg,
    }
    bundle = build_forecast_rollout_bundle(cfg, context, _FrontOnlyPredictor())
    assert bundle.wcdt_selected_vehicle_ids == ["front"]
    assert "rear" in bundle.cv_fallback_vehicle_ids
    assert bundle.actor_sources["rear"] == "constant_velocity"
    assert bundle.cv_fallback_uncertainty > 0.0
    assert bundle.combined_uncertainty >= bundle.cv_fallback_uncertainty
    assert bundle.wcdt_required_actor_coverage_complete
    assert bundle.forecast_safety_actor_coverage_complete
