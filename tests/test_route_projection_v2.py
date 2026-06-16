from __future__ import annotations

import numpy as np
import pytest

from safe_rl.prediction.trajectory_postprocess import trajectory_to_states
from safe_rl.sim.scenario_semantics import project_route_position
from safe_rl.sim.sumo_highway_merge_env import SumoHighwayMergeEnv
from safe_rl.sim.types import StepMetrics, VehicleState
from safe_rl.utils.config import load_config


def _state(
    *,
    x: float,
    y: float,
    edge_id: str,
    lane_index: int,
    lane_pos: float,
) -> VehicleState:
    return VehicleState(
        vehicle_id="actor",
        x=x,
        y=y,
        heading=0.0,
        speed=20.0,
        lane_index=lane_index,
        lane_id=f"{edge_id}_{lane_index}",
        lane_pos=lane_pos,
        edge_id=edge_id,
    )


def test_route_projection_tracks_lane_change_and_connected_edge():
    cfg = load_config()
    auxiliary = _state(
        x=400.0,
        y=53.8,
        edge_id="main_aux",
        lane_index=0,
        lane_pos=98.5,
    )
    changed = project_route_position(cfg, 404.0, 56.9, auxiliary)
    assert changed.valid
    assert changed.edge_id == "main_aux"
    assert changed.lane_index == 1

    before_out = _state(
        x=515.0,
        y=57.0,
        edge_id="main_aux",
        lane_index=1,
        lane_pos=213.5,
    )
    downstream = project_route_position(cfg, 525.0, 57.0, before_out)
    assert downstream.valid
    assert downstream.edge_id == "main_out"
    assert downstream.lane_index == 0


def test_route_projection_marks_far_point_invalid():
    cfg = load_config()
    reference = _state(
        x=400.0,
        y=53.8,
        edge_id="main_aux",
        lane_index=0,
        lane_pos=98.5,
    )
    projection = project_route_position(cfg, 402.0, 100.0, reference)
    assert not projection.valid
    assert projection.failure_reason == "projection_distance_exceeded"


def test_trajectory_postprocess_uses_motion_heading_and_route_metadata():
    cfg = load_config()
    reference = _state(
        x=400.0,
        y=53.8,
        edge_id="main_aux",
        lane_index=0,
        lane_pos=98.5,
    )
    states = trajectory_to_states(
        np.asarray([[402.0, 54.0], [404.0, 56.9]], dtype=np.float32),
        reference=reference,
        dt=0.1,
        vehicle_id="actor",
        config=cfg,
    )
    assert states[0].route_position_valid
    assert states[1].route_position_valid
    assert states[1].lane_index == 1
    assert states[1].heading == pytest.approx(np.arctan2(2.9, 2.0), abs=1.0e-5)


def test_reward_components_sum_to_total_without_changing_reward():
    cfg = load_config()
    cfg.rl["reward_profile"] = "default"
    env = object.__new__(SumoHighwayMergeEnv)
    env.config = cfg
    env._last_reward_debug = {}
    env._reward_debug_records = []
    env._reward_component_records = []
    ego = _state(
        x=10.0,
        y=53.8,
        edge_id="main_aux",
        lane_index=0,
        lane_pos=10.0,
    )
    metrics = StepMetrics(
        min_distance=10.0,
        min_ttc=10.0,
        max_drac=0.0,
        collision=False,
        near_miss=False,
        low_ttc=False,
        high_drac=False,
        merge_gap=20.0,
        lane_oob=True,
    )
    reward = env._reward(8.0, ego, metrics, "")
    components = env._reward_component_records[-1]
    component_sum = sum(
        components[name]
        for name in (
            "progress_reward",
            "speed_reward",
            "terminal_reward",
            "lane_oob_penalty",
            "safety_penalty",
            "safety_forecast_shaping",
            "shield_guided_shaping",
            "merge_timing_shaping",
        )
    )
    assert reward == pytest.approx(component_sum)
    assert components["total_episode_reward"] == pytest.approx(reward)
