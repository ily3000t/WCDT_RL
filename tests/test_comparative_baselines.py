from __future__ import annotations

from pathlib import Path

import numpy as np

from safe_rl.baselines.api import RuleControlContext
from safe_rl.baselines.rule_gap_acceptance import RuleGapAcceptancePolicy
from safe_rl.analysis.comparative_report import _policy_rows, _shield_deltas
from safe_rl.pipeline.stage1_risk_probe import _encode_trajectory_vehicle_ids
from safe_rl.prediction.actor_selector import ActorSelectionResult
from safe_rl.prediction.forecast_rollout_bundle import (
    ActorModeForecast,
    ForecastActorRollout,
    ForecastRolloutBundle,
    _prediction_modes,
)
from safe_rl.risk.merge_local import merge_local_stats
from safe_rl.sim.action_space import ACTIONS
from safe_rl.sim.types import VehicleState
from safe_rl.utils.config import load_config
from safe_rl.utils.stage1_dataset import merge_stage1_shards, open_stage1_dataset, write_stage1_dataset


def _state(vehicle_id: str, x: float, lane_index: int, speed: float = 20.0) -> VehicleState:
    return VehicleState(
        vehicle_id=vehicle_id,
        x=x,
        y=53.8 + 3.2 * lane_index,
        heading=0.0,
        speed=speed,
        lane_index=lane_index,
        lane_id=f"main_aux_{lane_index}",
        lane_pos=x - 300.0,
        edge_id="main_aux",
    )


def test_trajectory_vehicle_id_encoding_preserves_padding_and_row_identity():
    table, indices = _encode_trajectory_vehicle_ids(
        np.asarray([["ego", "front", ""], ["ego", "rear", "front"]])
    )
    assert table.tolist() == ["ego", "front", "rear"]
    assert indices.tolist() == [[0, 1, -1], [0, 2, 1]]


def test_stage1_shard_merge_remaps_worker_local_vehicle_id_tables(tmp_path):
    shards = []
    for index, (table, rows) in enumerate(
        [(["ego", "front"], [[0, 1]]), (["rear", "ego"], [[1, 0]])]
    ):
        path = tmp_path / f"shard_{index}"
        write_stage1_dataset(
            path,
            {
                "trajectory_episode_id": np.asarray([index], dtype=np.int64),
                "trajectory_window_end_step": np.asarray([0], dtype=np.int64),
                "trajectory_agent_vehicle_id_index": np.asarray(rows, dtype=np.int32),
                "trajectory_selector_selected_count": np.asarray([1], dtype=np.int64),
                "trajectory_vehicle_id_table": np.asarray(table),
            },
            metadata={"trajectory_schema_version": 4},
        )
        shards.append(path)
    merged = tmp_path / "merged"
    merge_stage1_shards(
        shards,
        merged,
        transition_keys=set(),
        candidate_keys=set(),
        trajectory_keys={
            "trajectory_episode_id",
            "trajectory_window_end_step",
            "trajectory_agent_vehicle_id_index",
            "trajectory_selector_selected_count",
        },
        metadata={"trajectory_schema_version": 4},
    )
    with open_stage1_dataset(merged) as data:
        assert data["trajectory_vehicle_id_table"].tolist() == ["ego", "front", "rear"]
        assert data["trajectory_agent_vehicle_id_index"].tolist() == [[0, 1], [0, 2]]


def test_wcdt_multimodal_bundle_keeps_mode_axis_without_coordinate_average():
    trajectories = np.zeros((1, 2, 3, 4, 3), dtype=np.float32)
    trajectories[0, :, 0, :, 0] = 10.0
    trajectories[0, :, 1, :, 0] = 30.0
    modes = _prediction_modes({"future_trajectories": trajectories})
    assert modes is not None
    assert modes.shape == (3, 2, 4, 3)
    assert float(modes[0, 0, 0, 0]) == 10.0
    assert float(modes[1, 0, 0, 0]) == 30.0


def test_joint_worlds_are_deterministic_and_do_not_use_a_global_mode_index():
    first = _state("front", 410.0, 1)
    second = _state("front", 460.0, 1)
    selection = ActorSelectionResult(
        selected_actor_ids=("front",),
        relevant_actor_ids=("front",),
        dropped_relevant_ids=(),
        relevant_count=1,
        overflow=False,
        actor_metadata={},
        version="test",
        config_hash="test",
    )
    bundle = ForecastRolloutBundle(
        actors=[ForecastActorRollout("front", "wcdt", [first], 0.2, current_state=first)],
        selection_result=selection,
        wcdt_uncertainty=0.2,
        cv_fallback_uncertainty=0.0,
        combined_uncertainty=0.2,
        wcdt_selected_vehicle_ids=["front"],
        cv_fallback_vehicle_ids=[],
        safety_required_vehicle_ids=["front"],
        wcdt_required_actor_coverage_complete=True,
        forecast_safety_actor_coverage_complete=True,
        critical_wcdt_coverage_complete=True,
        combined_critical_coverage_complete=True,
        actor_selector_overflow=False,
        cv_fallback_overflow=False,
        cv_fallback_dropped_vehicle_ids=[],
        actor_mode_forecasts=[
            ActorModeForecast(
                "front", "wcdt", [[first], [second]], np.asarray([0.5, 0.5]), 0.2, first
            )
        ],
        joint_world_count=32,
        joint_world_seed=(1, 2, 3),
    )
    worlds = bundle.joint_world_actor_sets()
    assert len(worlds) == 32
    assert {world[0].trajectory[0].x for world in worlds} == {410.0, 460.0}
    assert [world[0].trajectory[0].x for world in worlds] == [
        world[0].trajectory[0].x for world in bundle.joint_world_actor_sets()
    ]


def test_comparative_summary_uses_training_seeds_not_scenario_episode_count():
    groups = {
        "v3_101": {
            "comparative": {"method": "wcdt_v3", "training_seed": 101, "evaluation_variant": "policy"},
            "episodes": [{"seed": 1, "episode_reward": 1.0}, {"seed": 2, "episode_reward": 3.0}],
        },
        "v3_202": {
            "comparative": {"method": "wcdt_v3", "training_seed": 202, "evaluation_variant": "policy"},
            "episodes": [{"seed": 1, "episode_reward": 3.0}, {"seed": 2, "episode_reward": 5.0}],
        },
        "v3_shield_101": {
            "comparative": {"method": "wcdt_v3", "training_seed": 101, "evaluation_variant": "shield"},
            "episodes": [{"seed": 1, "episode_reward": 2.0}, {"seed": 2, "episode_reward": 4.0}],
        },
    }
    by_seed, headline = _policy_rows(groups, "policy")
    assert len(by_seed) == 2
    assert headline[0]["training_trial_count"] == 2
    assert headline[0]["episode_reward_mean"] == 3.0
    deltas = _shield_deltas(groups)
    assert len(deltas) == 1
    assert deltas[0]["episode_reward_delta"] == 1.0


def test_rule_gap_acceptance_merges_only_when_current_target_gap_is_safe():
    cfg = load_config()
    ego = _state("ego", 400.0, 0)
    front = _state("front", 430.0, 1)
    rear = _state("rear", 370.0, 1, speed=18.0)
    vehicles = [ego, front, rear]
    def context_for() -> RuleControlContext:
        local = merge_local_stats(ego, vehicles, cfg)
        front_closing = max(0.0, ego.speed - front.speed)
        rear_closing = max(0.0, rear.speed - ego.speed)
        return RuleControlContext(
            ego=ego,
            current_lane_front=None,
            target_front=front,
            target_rear=rear,
            current_lane_front_gap=float("inf"),
            target_front_gap=float(local.target_front_gap),
            target_rear_gap=float(local.target_rear_gap),
            target_front_closing_speed=front_closing,
            target_rear_closing_speed=rear_closing,
            target_front_ttc=float("inf") if front_closing == 0.0 else local.target_front_gap / front_closing,
            target_rear_ttc=float("inf") if rear_closing == 0.0 else local.target_rear_gap / rear_closing,
            lane_speed_limit=30.0,
            distance_to_taper=float(local.merge_distance),
            ego_on_auxiliary=bool(local.ego_on_auxiliary),
            merge_lateral_cmd=1,
            legal_action_indices=frozenset(action.index for action in ACTIONS),
        )
    context = context_for()
    decision = RuleGapAcceptancePolicy(cfg).act(context)
    assert decision.reason == "safe_gap_merge"

    front.x = 405.0
    unsafe = RuleGapAcceptancePolicy(cfg).act(context_for())
    assert unsafe.reason != "safe_gap_merge"


def test_rule_gap_acceptance_never_returns_an_illegal_merge_action():
    cfg = load_config()
    ego = _state("ego", 400.0, 0)
    front = _state("front", 430.0, 1)
    rear = _state("rear", 370.0, 1, speed=18.0)
    local = merge_local_stats(ego, [ego, front, rear], cfg)
    safe_context = RuleControlContext(
        ego=ego,
        current_lane_front=None,
        target_front=front,
        target_rear=rear,
        current_lane_front_gap=float("inf"),
        target_front_gap=float(local.target_front_gap),
        target_rear_gap=float(local.target_rear_gap),
        target_front_closing_speed=0.0,
        target_rear_closing_speed=0.0,
        target_front_ttc=float("inf"),
        target_rear_ttc=float("inf"),
        lane_speed_limit=30.0,
        distance_to_taper=float(local.merge_distance),
        ego_on_auxiliary=True,
        merge_lateral_cmd=1,
        legal_action_indices=frozenset(action.index for action in ACTIONS if action.lateral_cmd == 0),
    )
    decision = RuleGapAcceptancePolicy(cfg).act(safe_context)
    assert next(action for action in ACTIONS if action.index == decision.action).lateral_cmd == 0


def test_rule_controller_has_no_prediction_or_risk_dependencies():
    source = Path(RuleGapAcceptancePolicy.__module__.replace(".", "/") + ".py")
    text = (Path.cwd() / source).read_text(encoding="utf-8")
    assert "safe_rl.risk" not in text
    assert "safe_rl.prediction" not in text
    assert "safe_rl.shield" not in text
