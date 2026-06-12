from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml

from safe_rl.pipeline.run_full_pipeline import build_generated_configs
from safe_rl.pipeline.stage1_risk_probe import _aggregate_worker_performance, _concatenate_shards
from safe_rl.rl.ppo import _worker_model_memory_estimate
from safe_rl.risk.merge_local import candidate_action_risk_samples
from safe_rl.sim.sumo_highway_merge_env import SumoHighwayMergeEnv
from safe_rl.sim.types import VehicleState
from safe_rl.utils.config import load_config
from safe_rl.utils.sumo_installation import resolve_sumo_installation


def _state(
    vehicle_id: str,
    *,
    x: float,
    lane: int,
    speed: float = 20.0,
    edge: str = "main_aux",
) -> VehicleState:
    return VehicleState(
        vehicle_id=vehicle_id,
        x=x,
        y=float(lane * 3.2),
        heading=0.0,
        speed=speed,
        lane_index=lane,
        lane_id=f"{edge}_{lane}",
        lane_pos=max(0.0, x),
        edge_id=edge,
        length=4.8,
        width=1.8,
        accel=0.0,
    )


def test_performance_profile_enables_parallelism_without_changing_defaults(tmp_path):
    defaults = build_generated_configs("default_run", tmp_path / "default")
    default_main = yaml.safe_load(defaults["main"].read_text(encoding="utf-8"))
    assert default_main.get("stage1", {}).get("workers", 1) == 1
    assert default_main.get("training", {}).get("ppo_num_envs", 1) == 1
    assert default_main["shield"]["forecast_task_shadow_enabled"] is False

    configs = build_generated_configs(
        "performance_run",
        tmp_path / "performance",
        pipeline_profile="performance",
    )
    main = yaml.safe_load(configs["main"].read_text(encoding="utf-8"))
    forecast = yaml.safe_load(configs["forecast_wcdt_v3_ppo"].read_text(encoding="utf-8"))
    stage5 = yaml.safe_load(configs["stage5_multi_groups"].read_text(encoding="utf-8"))
    assert main["stage1"]["workers"] == 6
    assert main["training"]["ppo_num_envs"] == 4
    assert main["rl"]["n_steps"] == 256
    assert forecast["shield"]["forecast_task_shadow_enabled"] is False
    assert stage5["shield"]["forecast_task_shadow_enabled"] is True


def test_candidate_rollout_reuses_surrounding_vehicle_rollouts(monkeypatch):
    cfg = load_config()
    ego = _state("ego", x=100.0, lane=0)
    vehicles = [ego, _state("front", x=125.0, lane=1), _state("rear", x=80.0, lane=1, speed=24.0)]
    context = {"ego": ego, "vehicles": vehicles, "lane_count": 4, "config": cfg}
    import safe_rl.risk.merge_local as merge_local

    original = merge_local.route_aware_constant_velocity_rollout
    calls = {"count": 0}

    def counted(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(merge_local, "route_aware_constant_velocity_rollout", counted)
    samples = candidate_action_risk_samples(context)
    assert len(samples) == 9
    assert calls["count"] == 2
    candidate_action_risk_samples(context)
    assert calls["count"] == 2


def test_decision_context_cache_is_invalidated_explicitly():
    cfg = load_config()
    env = SumoHighwayMergeEnv(cfg, seed=1)
    ego = _state("ego", x=100.0, lane=0)
    env.history.append([ego, _state("front", x=120.0, lane=1)])
    env._lane_count = lambda _edge: 4  # type: ignore[method-assign]
    first = env.get_risk_context()
    assert env.get_risk_context() is first
    env._invalidate_decision_cache()
    assert env.get_risk_context() is not first


def test_subscription_state_collection_uses_batch_results_without_getter_fallback():
    cfg = load_config()
    env = SumoHighwayMergeEnv(cfg, seed=1)

    class Constants:
        VAR_POSITION = 1
        VAR_ANGLE = 2
        VAR_LANE_ID = 3
        VAR_LANE_INDEX = 4
        VAR_SPEED = 5
        VAR_ACCELERATION = 6
        VAR_LANEPOSITION = 7
        VAR_ROAD_ID = 8
        VAR_LENGTH = 9
        VAR_WIDTH = 10

    result = {
        1: (10.0, 2.0),
        2: 90.0,
        3: "main_aux_0",
        4: 0,
        5: 20.0,
        6: 0.5,
        7: 10.0,
        8: "main_aux",
        9: 4.8,
        10: 1.8,
    }

    class VehicleAPI:
        def getIDList(self):
            return ["ego"]

        def subscribe(self, _vehicle_id, _variables):
            return None

        def getAllSubscriptionResults(self):
            return {"ego": result}

        def getPosition(self, _vehicle_id):
            raise AssertionError("getter fallback should not run")

    env._traci_module = SimpleNamespace(constants=Constants)
    env._traci = SimpleNamespace(vehicle=VehicleAPI())
    states = env._collect_states()
    assert len(states) == 1
    assert states[0].vehicle_id == "ego"
    assert states[0].speed == 20.0
    assert env._subscription_fallback_count == 0


def test_stage1_shard_merge_is_stable_and_rebuilds_candidate_ids(tmp_path):
    shard_a = tmp_path / "a.npz"
    shard_b = tmp_path / "b.npz"
    metadata = {
        "trajectory_schema_version": np.asarray(4, dtype=np.int64),
        "safety_metric_version": np.asarray("oriented_box_v1"),
    }
    np.savez(
        shard_a,
        transition_episode_id=np.asarray([2]),
        transition_episode_step=np.asarray([5]),
        executed_actions=np.asarray([4]),
        episode_id=np.asarray([2, 2]),
        candidate_episode_step=np.asarray([5, 5]),
        actions=np.asarray([1, 0]),
        candidate_transition_id=np.asarray([0, 0]),
        **metadata,
    )
    np.savez(
        shard_b,
        transition_episode_id=np.asarray([1]),
        transition_episode_step=np.asarray([5]),
        executed_actions=np.asarray([3]),
        episode_id=np.asarray([1, 1]),
        candidate_episode_step=np.asarray([5, 5]),
        actions=np.asarray([1, 0]),
        candidate_transition_id=np.asarray([0, 0]),
        **metadata,
    )
    output = tmp_path / "merged.npz"
    merged = _concatenate_shards([shard_a, shard_b], output)
    assert merged["transition_episode_id"].tolist() == [1, 2]
    assert merged["episode_id"].tolist() == [1, 1, 2, 2]
    assert merged["actions"].tolist() == [0, 1, 0, 1]
    assert merged["candidate_transition_id"].tolist() == [0, 0, 1, 1]


def test_parallel_stage1_performance_is_aggregated_at_top_level():
    reports = [
        {
            "performance": {
                "candidate_rollout_time": 1.25,
                "risk_forward_time": 0.5,
                "operation_counts": {"risk_forwards": 4},
            }
        },
        {
            "performance": {
                "candidate_rollout_time": 0.75,
                "risk_forward_time": 0.25,
                "operation_counts": {"risk_forwards": 3},
            }
        },
    ]
    result = _aggregate_worker_performance(
        reports,
        {"wall_time": 5.0},
        episodes=10,
        transition_count=100,
    )
    assert result["candidate_rollout_time"] == 2.0
    assert result["risk_forward_time"] == 0.75
    assert result["operation_counts"]["risk_forwards"] == 7
    assert result["steps_per_second"] == 20.0
    assert result["worker_count"] == 2


def test_ppo_worker_memory_estimate_counts_each_process(tmp_path):
    checkpoint = tmp_path / "predictor.pt"
    checkpoint.write_bytes(b"x" * 128)
    config = {
        "forecast_features": {"checkpoint": str(checkpoint)},
        "rl": {"shield_guided_reward": {"risk_checkpoint": None}},
    }
    estimate = _worker_model_memory_estimate(config, 4)
    assert estimate["per_worker_checkpoint_bytes"] == 128
    assert estimate["all_workers_checkpoint_bytes"] == 512


def test_persistent_reload_uses_current_seed_and_resets_subscriptions():
    cfg = load_config()
    env = SumoHighwayMergeEnv(cfg, seed=7)
    calls = []
    env._traci = SimpleNamespace(load=lambda args: calls.append(list(args)))
    env._subscribed_vehicle_ids.add("ego")
    env.seed_value = 11
    env._reload_sumo()
    assert calls
    seed_index = calls[0].index("--seed")
    assert calls[0][seed_index + 1] == "11"
    assert not env._subscribed_vehicle_ids
    assert env._sumo_reload_count == 1


def test_sumo_resolver_uses_explicit_installation_without_project_path_fallback(tmp_path, monkeypatch):
    home = tmp_path / "sumo"
    bin_dir = home / "bin"
    tools_dir = home / "tools"
    bin_dir.mkdir(parents=True)
    tools_dir.mkdir()
    sumo = bin_dir / "sumo.exe"
    netconvert = bin_dir / "netconvert.exe"
    sumo.write_bytes(b"")
    netconvert.write_bytes(b"")
    import safe_rl.utils.sumo_installation as installation

    monkeypatch.setattr(installation, "_version", lambda path: f"{path.stem} 1.22.0")
    resolved = resolve_sumo_installation({"sumo_binary": str(sumo)})
    assert Path(resolved.sumo_binary) == sumo.resolve()
    assert Path(resolved.netconvert_binary) == netconvert.resolve()
    assert Path(resolved.tools_directory) == tools_dir.resolve()


def test_sumo_resolver_prefers_sumo_home_over_path(tmp_path, monkeypatch):
    home = tmp_path / "sumo_home"
    path_home = tmp_path / "path_home"
    for candidate in (home, path_home):
        (candidate / "bin").mkdir(parents=True)
        (candidate / "tools").mkdir()
        for name in ("sumo.exe", "netconvert.exe"):
            (candidate / "bin" / name).write_bytes(b"")
    import safe_rl.utils.sumo_installation as installation

    monkeypatch.setenv("SUMO_HOME", str(home))
    monkeypatch.setattr(
        installation.shutil,
        "which",
        lambda name: str(path_home / "bin" / name) if name in {"sumo", "sumo.exe"} else None,
    )
    monkeypatch.setattr(installation, "_version", lambda path: f"{path.stem} 1.22.0")
    resolved = resolve_sumo_installation({"sumo_binary": "sumo"})
    assert Path(resolved.sumo_binary) == (home / "bin" / "sumo.exe").resolve()
