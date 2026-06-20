from __future__ import annotations

import numpy as np

from safe_rl.baselines.rule_gap_acceptance import RuleGapAcceptancePolicy
from safe_rl.pipeline.stage1_risk_probe import _encode_trajectory_vehicle_ids
from safe_rl.prediction.forecast_rollout_bundle import _prediction_modes
from safe_rl.risk.merge_local import merge_local_stats
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


def test_rule_gap_acceptance_merges_only_when_current_target_gap_is_safe():
    cfg = load_config()
    ego = _state("ego", 400.0, 0)
    front = _state("front", 430.0, 1)
    rear = _state("rear", 370.0, 1, speed=18.0)
    vehicles = [ego, front, rear]
    context = {
        "ego": ego,
        "vehicles": vehicles,
        "merge_local": merge_local_stats(ego, vehicles, cfg),
        "target_front": front,
        "target_rear": rear,
        "lane_count": 4,
        "config": cfg,
        "lane_speed_limit": 30.0,
    }
    decision = RuleGapAcceptancePolicy(cfg).act(context)
    assert decision.reason == "safe_gap_merge"

    front.x = 405.0
    unsafe_context = dict(context)
    unsafe_context["merge_local"] = merge_local_stats(ego, vehicles, cfg)
    unsafe = RuleGapAcceptancePolicy(cfg).act(unsafe_context)
    assert unsafe.reason != "safe_gap_merge"
