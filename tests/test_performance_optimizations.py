from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml

from safe_rl.pipeline.run_full_pipeline import (
    _artifact_sha256,
    _new_pipeline_state,
    _validate_completed_outputs,
    build_generated_configs,
)
from safe_rl.pipeline.stage1_risk_probe import (
    _aggregate_worker_performance,
    _concatenate_shards,
    run as run_stage1,
)
from safe_rl.prediction.forecast_rollout_bundle import build_forecast_rollout_bundle
from safe_rl.rl.ppo import _worker_model_memory_estimate
from safe_rl.risk.merge_local import (
    candidate_action_risk_samples,
    prepare_candidate_rollout_context,
)
from safe_rl.sim.sumo_highway_merge_env import (
    SumoHighwayMergeEnv,
    scheduled_episode_seed,
)
from safe_rl.sim.types import VehicleState
from safe_rl.utils.config import load_config
from safe_rl.utils.stage1_dataset import (
    open_stage1_dataset,
    validate_stage1_dataset,
    write_stage1_dataset,
)
from safe_rl.utils.sumo_installation import (
    SumoInstallation,
    configure_sumo_python,
    resolve_sumo_installation,
)


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


def test_episode_seed_schedule_is_unique_across_parallel_envs():
    base_seed = 17
    for num_envs in (1, 2, 4):
        seeds = {
            scheduled_episode_seed(base_seed, rank, episode, num_envs)
            for episode in range(10)
            for rank in range(num_envs)
        }
        assert len(seeds) == 10 * num_envs
        assert min(seeds) == base_seed
        assert max(seeds) == base_seed + 10 * num_envs - 1


def test_explicit_reset_seed_does_not_apply_parallel_schedule(monkeypatch):
    cfg = load_config()
    env = SumoHighwayMergeEnv(
        cfg,
        seed=10,
        worker_rank=3,
        num_envs=4,
        advance_episode_seed=False,
    )
    monkeypatch.setattr(env, "_close_sumo", lambda: None)
    monkeypatch.setattr(env, "_start_sumo", lambda: None)
    monkeypatch.setattr(env, "_simulation_step", lambda: None)
    monkeypatch.setattr(env, "_apply_curriculum_perturbation", lambda: None)
    monkeypatch.setattr(env, "_configure_ego_control", lambda: None)
    monkeypatch.setattr(env, "_collect_states", lambda: [])
    monkeypatch.setattr(env, "_build_observation", lambda: np.zeros(env.observation_space.shape))
    observation, info = env.reset(seed=1234)
    assert observation.shape == env.observation_space.shape
    assert env.seed_value == 1234
    assert info["episode_seed"] == 1234


def test_risk_cv_rollout_and_forecast_bundle_do_not_share_surrounding_trajectories():
    cfg = load_config()
    cfg.forecast_features["source"] = "wcdt_v3"
    ego = _state("ego", x=100.0, lane=0)
    front = _state("front", x=125.0, lane=1)
    context = {"ego": ego, "vehicles": [ego, front], "lane_count": 4, "config": cfg}
    risk_context = prepare_candidate_rollout_context(context)

    horizon = int(cfg.forecast_features.horizon_steps)
    predicted = np.zeros((1, horizon, 2), dtype=np.float32)
    predicted[0, :, 0] = np.linspace(500.0, 520.0, horizon)
    predicted[0, :, 1] = front.y

    class Predictor:
        checkpoint_path = "synthetic-wcdt-v3"

        def predict(self, _context):
            return {
                "future_trajectories": predicted,
                "selected_vehicle_ids": ["front"],
                "forecast_source": "wcdt_v3",
                "uncertainty": 0.2,
            }

    bundle = build_forecast_rollout_bundle(cfg, context, Predictor())
    forecast_actor = bundle.actor_by_id("front")
    assert forecast_actor is not None
    assert forecast_actor.source == "wcdt_v3"
    assert not np.isclose(
        risk_context.risk_surrounding_cv_rollouts[0][0].x,
        forecast_actor.trajectory[0].x,
    )
    assert risk_context.risk_surrounding_cv_rollouts is not bundle.rollout_lists()


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
    output = tmp_path / "merged"
    merged = _concatenate_shards([shard_a, shard_b], output)
    assert merged["transition_episode_id"].tolist() == [1, 2]
    assert merged["episode_id"].tolist() == [1, 1, 2, 2]
    assert merged["actions"].tolist() == [0, 1, 0, 1]
    assert merged["candidate_transition_id"].tolist() == [0, 0, 1, 1]
    assert (output / "manifest.json").exists()
    merged.close()


def test_stage1_shard_merge_does_not_use_full_array_concatenate(tmp_path, monkeypatch):
    shard_paths = []
    for episode_id in (2, 1):
        shard = tmp_path / f"shard_{episode_id}.npz"
        np.savez(
            shard,
            transition_episode_id=np.asarray([episode_id]),
            transition_episode_step=np.asarray([5]),
            executed_actions=np.asarray([episode_id]),
            episode_id=np.asarray([episode_id, episode_id]),
            candidate_episode_step=np.asarray([5, 5]),
            actions=np.asarray([1, 0]),
            candidate_transition_id=np.asarray([0, 0]),
            trajectory_schema_version=np.asarray(4, dtype=np.int64),
            safety_metric_version=np.asarray("oriented_box_v1"),
        )
        shard_paths.append(shard)

    def reject_concatenate(*_args, **_kwargs):
        raise AssertionError("streaming shard merge must not concatenate complete arrays")

    monkeypatch.setattr(np, "concatenate", reject_concatenate)
    output = tmp_path / "streamed"
    merged = _concatenate_shards(shard_paths, output)
    assert merged["transition_episode_id"].tolist() == [1, 2]
    merged.close()


def test_manifest_dataset_and_legacy_npz_expose_same_array_semantics(tmp_path):
    legacy = tmp_path / "legacy.npz"
    arrays = {
        "transition_episode_id": np.asarray([1, 2], dtype=np.int64),
        "observations": np.arange(6, dtype=np.float32).reshape(2, 3),
    }
    np.savez(legacy, **arrays)
    shard = tmp_path / "shard.npz"
    np.savez(
        shard,
        transition_episode_id=arrays["transition_episode_id"],
        transition_episode_step=np.asarray([1, 1]),
        executed_actions=np.asarray([0, 1]),
        observations=arrays["observations"],
    )
    output = tmp_path / "manifest"
    merged = _concatenate_shards([shard], output)
    old = open_stage1_dataset(legacy)
    try:
        assert merged.legacy_npz_format is False
        assert old.legacy_npz_format is True
        np.testing.assert_array_equal(merged["transition_episode_id"], old["transition_episode_id"])
        np.testing.assert_array_equal(merged["observations"], old["observations"])
    finally:
        merged.close()
        old.close()


def test_stage1_dataset_array_damage_is_detected_by_resume_validation(tmp_path):
    output = tmp_path / "risk_probe_buffer"
    write_stage1_dataset(
        output,
        {
            "transition_episode_id": np.asarray([1, 2], dtype=np.int64),
            "observations": np.arange(6, dtype=np.float32).reshape(2, 3),
        },
        metadata={
            "trajectory_schema_version": 4,
            "episode_seed_schedule": "incrementing_v1",
        },
    )
    validate_stage1_dataset(output)
    invocation = {
        "stage1_episodes": 2,
        "stage4_episodes": None,
        "stage5_episodes": None,
        "ppo_timesteps": None,
        "forecast_ppo_timesteps": None,
        "forecast_ppo_profile": "default",
        "forecast_sources": ["constant_velocity"],
    }
    state = _new_pipeline_state("test_run", invocation)
    task = state["tasks"]["stage1"]
    task["status"] = "completed"
    task["required_outputs"] = [str(output)]
    task["output_hashes"] = {str(output): _artifact_sha256(output)}

    observations = np.load(output / "arrays" / "observations.npy", mmap_mode="r+")
    observations[0, 0] = 999.0
    observations.flush()
    mmap_handle = getattr(observations, "_mmap", None)
    if mmap_handle is not None:
        mmap_handle.close()

    with np.testing.assert_raises_regex(ValueError, "Stage1 array hash changed"):
        _validate_completed_outputs(state)


def test_stage1_rejects_unsupported_output_format(monkeypatch):
    cfg = load_config()
    cfg.stage1["output_format"] = "legacy_npz"
    monkeypatch.setattr(
        "safe_rl.pipeline.stage1_risk_probe._run_parallel",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("collection must not start")
        ),
    )
    with np.testing.assert_raises_regex(ValueError, "stage1.output_format"):
        run_stage1(cfg)


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
    (tools_dir / "traci").mkdir()
    (tools_dir / "traci" / "__init__.py").write_text("", encoding="utf-8")
    (tools_dir / "traci" / "constants.py").write_text("TRACI_VERSION = 21\n", encoding="utf-8")
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
        (candidate / "tools" / "traci").mkdir()
        (candidate / "tools" / "traci" / "__init__.py").write_text("", encoding="utf-8")
        (candidate / "tools" / "traci" / "constants.py").write_text(
            "TRACI_VERSION = 21\n",
            encoding="utf-8",
        )
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


def test_sumo_resolver_does_not_take_netconvert_from_another_installation(
    tmp_path,
    monkeypatch,
):
    selected = tmp_path / "selected"
    other = tmp_path / "other"
    (selected / "bin").mkdir(parents=True)
    (selected / "tools" / "traci").mkdir(parents=True)
    (selected / "tools" / "traci" / "__init__.py").write_text("", encoding="utf-8")
    (selected / "bin" / "sumo.exe").write_bytes(b"sumo")
    (other / "bin").mkdir(parents=True)
    (other / "bin" / "netconvert.exe").write_bytes(b"netconvert")
    monkeypatch.setattr(
        "safe_rl.utils.sumo_installation.shutil.which",
        lambda name: str(other / "bin" / "netconvert.exe") if "netconvert" in name else None,
    )
    with np.testing.assert_raises(FileNotFoundError):
        resolve_sumo_installation({"sumo_binary": str(selected / "bin" / "sumo.exe")})


def test_sumo_resolver_rejects_missing_explicit_absolute_path(tmp_path, monkeypatch):
    fallback = tmp_path / "fallback"
    (fallback / "bin").mkdir(parents=True)
    (fallback / "tools" / "traci").mkdir(parents=True)
    (fallback / "bin" / "sumo.exe").write_bytes(b"sumo")
    (fallback / "bin" / "netconvert.exe").write_bytes(b"netconvert")
    (fallback / "tools" / "traci" / "__init__.py").write_text("", encoding="utf-8")
    monkeypatch.setenv("SUMO_HOME", str(fallback))
    missing = tmp_path / "missing" / "sumo.exe"
    with np.testing.assert_raises_regex(FileNotFoundError, "Configured sumo binary"):
        resolve_sumo_installation({"sumo_binary": str(missing.resolve())})


def test_configure_sumo_python_rejects_preloaded_traci_from_other_installation(
    tmp_path,
    monkeypatch,
):
    tools = tmp_path / "sumo" / "tools"
    expected = tools / "traci" / "__init__.py"
    expected.parent.mkdir(parents=True)
    expected.write_text("", encoding="utf-8")
    installation = SumoInstallation(
        sumo_binary=str(tmp_path / "sumo" / "bin" / "sumo.exe"),
        sumo_gui_binary="",
        netconvert_binary=str(tmp_path / "sumo" / "bin" / "netconvert.exe"),
        tools_directory=str(tools),
        sumo_home=str(tmp_path / "sumo"),
        sumo_version="SUMO 1.22.0",
        netconvert_version="netconvert 1.22.0",
        sumo_binary_sha256="a",
        netconvert_binary_sha256="b",
        traci_module_path=str(expected),
        traci_version="protocol_21",
    )
    wrong = SimpleNamespace(__file__=str(tmp_path / "other" / "traci" / "__init__.py"))
    monkeypatch.setitem(__import__("sys").modules, "traci", wrong)
    with np.testing.assert_raises(RuntimeError):
        configure_sumo_python(installation)
