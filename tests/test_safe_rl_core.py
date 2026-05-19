from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from safe_rl.prediction.forecast_feature_augmentor import ForecastFeatureAugmentor
from safe_rl.pipeline.run_full_pipeline import build_generated_configs, resolve_forecast_sources
from safe_rl.pipeline.stage5_paired_eval import _build_acceptance, _build_paired_delta, _select_eval_seeds
from safe_rl.risk.merge_local import candidate_action_risk_samples, target_lane_neighbors
from safe_rl.risk.risk_feature_extractor import extract_candidate_features
from safe_rl.risk.risk_aggregator import aggregate_episode_reports
from safe_rl.risk.risk_module import RiskPrediction
from safe_rl.risk.stage1_sampling import configured_sampling_probs, sampling_summary, select_stage1_action
from safe_rl.rl.evaluation import validate_model_env_observation_shape
from safe_rl.shield.safety_shield import SafetyShield
from safe_rl.sim.action_space import ACTIONS, decode_action
from safe_rl.sim.metrics import compute_step_metrics
from safe_rl.sim.scenario_validation import validate_scenario_geometry
from safe_rl.sim.sumo_highway_merge_env import SumoHighwayMergeEnv
from safe_rl.sim.types import StepMetrics, VehicleState
from safe_rl.utils.config import load_config


def test_action_space_has_nine_actions():
    assert len(ACTIONS) == 9
    assert decode_action(4).name == "keep_hold"


def test_metrics_detect_near_miss():
    ego = VehicleState("ego", 0.0, 0.0, 0.0, 10.0, 0, "lane", 0.0, "ramp_in")
    other = VehicleState("other", 4.0, 0.0, 0.0, 0.0, 0, "lane", 0.0, "main_in")
    metrics = compute_step_metrics(ego, [ego, other], collision=False)
    assert metrics.min_distance < 1.0
    assert metrics.near_miss


def test_scenario_validation_passes():
    cfg = load_config()
    report = validate_scenario_geometry(cfg.scenario.sumocfg)
    assert report["passed"], report["errors"]
    ego = next(item for item in report["seed_positions"] if item["vehicle_id"] == "ego")
    assert ego["first_edge"] == "ramp_in"


def test_ramp_connection_targets_adjacent_main_lane():
    con_file = Path("scenarios/highway_merge/highway_merge.con.xml")
    root = ET.parse(con_file).getroot()
    ramp_connection = next(
        connection
        for connection in root.findall("connection")
        if connection.attrib.get("from") == "ramp_in" and connection.attrib.get("to") == "main_out"
    )
    assert ramp_connection.attrib["toLane"] == "2"


def test_merge_junction_uses_zipper_right_of_way():
    node_file = Path("scenarios/highway_merge/highway_merge.nod.xml")
    root = ET.parse(node_file).getroot()
    merge_node = next(node for node in root.findall("node") if node.attrib.get("id") == "merge")
    assert merge_node.attrib["type"] == "zipper"


def test_route_file_uses_harder_traffic_distribution():
    route_file = Path("scenarios/highway_merge/highway_merge.rou.xml")
    root = ET.parse(route_file).getroot()
    vtypes = {item.attrib["id"]: item.attrib for item in root.findall("vType")}
    assert float(vtypes["car_main"]["sigma"]) == pytest.approx(0.48)
    assert float(vtypes["car_ramp"]["sigma"]) == pytest.approx(0.50)

    flows = {item.attrib["id"]: item.attrib for item in root.findall("flow")}
    assert int(flows["main_flow_left"]["vehsPerHour"]) == 1350
    assert int(flows["main_flow_mid"]["vehsPerHour"]) == 1150
    assert int(flows["main_flow_right"]["vehsPerHour"]) == 900
    assert int(flows["ramp_flow"]["vehsPerHour"]) == 650
    assert flows["main_flow_left"]["departLane"] == "2"

    vehicles = {item.attrib["id"]: item.attrib for item in root.findall("vehicle")}
    assert vehicles["ego"]["route"] == "route_ramp"
    target_lane_seeds = [
        vehicle
        for vehicle in vehicles.values()
        if vehicle["route"] == "route_main" and vehicle["departLane"] == "2"
    ]
    assert len(target_lane_seeds) >= 3


def test_stage1_mixed_sampler_configures_three_sources():
    cfg = load_config()
    probs = configured_sampling_probs(cfg)
    assert cfg.stage1.action_sampling == "mixed"
    assert probs["random"] == pytest.approx(0.10)
    assert probs["merge_heuristic"] == pytest.approx(0.60)
    assert probs["risk_seek"] == pytest.approx(0.30)
    summary = sampling_summary(["random", "merge_heuristic", "merge_heuristic", "risk_seek"])
    assert summary["counts"]["merge_heuristic"] == 2
    assert summary["proportions"]["risk_seek"] == pytest.approx(0.25)


def test_target_lane_front_rear_gap_uses_lane_2_only():
    cfg = load_config()
    ego = VehicleState("ego", 200.0, 0.0, 0.0, 20.0, 0, "ramp_0", 100.0, "ramp_in")
    front = VehicleState("front", 215.0, 0.0, 0.0, 18.0, 2, "main_2", 215.0, "main_in")
    rear = VehicleState("rear", 190.0, 0.0, 0.0, 22.0, 2, "main_2", 190.0, "main_in")
    other_lane = VehicleState("other", 202.0, 0.0, 0.0, 18.0, 1, "main_1", 202.0, "main_in")
    gaps = target_lane_neighbors(ego, [ego, front, rear, other_lane], cfg)
    assert gaps["front_gap"] == pytest.approx(10.2)
    assert gaps["rear_gap"] == pytest.approx(5.2)
    assert gaps["front_rel_speed"] == pytest.approx(-2.0)
    assert gaps["rear_rel_speed"] == pytest.approx(2.0)


def test_candidate_action_buffer_generates_nine_samples_per_state():
    cfg = load_config()
    ego = VehicleState("ego", 205.0, 0.0, 0.0, 22.0, 0, "ramp_0", 120.0, "ramp_in")
    vehicle = VehicleState("main", 212.0, 0.0, 0.0, 18.0, 2, "main_2", 212.0, "main_in")
    context = {"ego": ego, "vehicles": [ego, vehicle], "lane_count": 1, "config": cfg}
    samples = candidate_action_risk_samples(context)
    assert len(samples) == 9
    assert sorted(sample.action for sample in samples) == list(range(9))
    assert all(sample.features.shape == (cfg.risk_module.explicit_feature_dim,) for sample in samples)


def test_extract_candidate_features_reflects_candidate_action():
    cfg = load_config()
    ego = VehicleState("ego", 205.0, 0.0, 0.0, 22.0, 0, "ramp_0", 120.0, "ramp_in")
    vehicle = VehicleState("main", 212.0, 0.0, 0.0, 18.0, 2, "main_2", 212.0, "main_in")
    context = {"ego": ego, "vehicles": [ego, vehicle], "lane_count": 1, "config": cfg}
    keep = extract_candidate_features(decode_action(4), context)
    lateral_oob = extract_candidate_features(decode_action(0), context)
    assert lateral_oob[5] == 1.0
    assert keep[5] == 0.0
    assert not np.allclose(keep, lateral_oob)


def test_constant_velocity_forecast_runs_without_checkpoint():
    cfg = load_config()
    cfg.forecast_features["enabled"] = True
    cfg.forecast_features["source"] = "constant_velocity"
    cfg.forecast_features["checkpoint"] = None
    ego = VehicleState("ego", 205.0, 0.0, 0.0, 22.0, 0, "ramp_0", 120.0, "ramp_in")
    vehicle = VehicleState("main", 212.0, 0.0, 0.0, 18.0, 2, "main_2", 212.0, "main_in")
    features = ForecastFeatureAugmentor(cfg).extract({"ego": ego, "vehicles": [ego, vehicle], "config": cfg})
    assert features.shape == (ForecastFeatureAugmentor.feature_dim(cfg),)
    assert np.all(np.isfinite(features))


class _StaticRiskModel:
    def __init__(self, scores: dict[int, float], uncertainty: float = 0.1):
        self.scores = scores
        self.uncertainty = uncertainty

    def predict(self, action, _context):
        return RiskPrediction(
            risk_score=float(self.scores.get(action.index, 0.95)),
            risk_type_logits=np.zeros((5,), dtype=np.float32),
            risk_uncertainty=self.uncertainty,
            explicit_features=np.zeros((8,), dtype=np.float32),
        )


def _shield_cfg():
    cfg = load_config()
    cfg.shield["enabled"] = True
    cfg.shield["risk_threshold"] = 0.65
    cfg.shield["uncertainty_threshold"] = 0.40
    cfg.shield["activation_risk_threshold"] = 0.90
    cfg.shield["replacement_margin"] = 0.15
    cfg.shield["allow_fallback"] = False
    return cfg


def _shield_context():
    return {
        "current_metrics": StepMetrics(
            min_distance=5.0,
            min_ttc=5.0,
            max_drac=0.0,
            collision=False,
            near_miss=False,
            low_ttc=False,
            high_drac=False,
            merge_gap=50.0,
        )
    }


def test_shield_keeps_raw_action_below_activation_threshold():
    cfg = _shield_cfg()
    shield = SafetyShield(cfg, _StaticRiskModel({4: 0.50}))
    raw = decode_action(4)
    final, record = shield.select_action(raw, _shield_context())
    assert final.index == raw.index
    assert record["replacement_reason"] == "raw_safe"
    assert not record["fallback"]


def test_shield_does_not_fallback_without_clear_safe_replacement():
    cfg = _shield_cfg()
    shield = SafetyShield(cfg, _StaticRiskModel({4: 0.95, 5: 0.83, 3: 0.82}))
    raw = decode_action(4)
    final, record = shield.select_action(raw, _shield_context())
    assert final.index == raw.index
    assert record["replacement_reason"] == "fallback_disabled"
    assert not record["fallback"]


def test_shield_replaces_only_when_candidate_improves_by_margin():
    cfg = _shield_cfg()
    shield = SafetyShield(cfg, _StaticRiskModel({4: 0.95, 5: 0.40}))
    raw = decode_action(4)
    final, record = shield.select_action(raw, _shield_context())
    assert final.index == 5
    assert record["replacement_reason"] == "replacement"
    assert record["risk_before"] - record["risk_after"] >= cfg.shield.replacement_margin


def test_stage5_rejects_insufficient_seeds():
    cfg = load_config()
    cfg.stage5["episodes_per_group"] = 3
    cfg.stage5["seeds"] = [1, 2]
    with pytest.raises(ValueError, match="requires at least 3 seeds"):
        _select_eval_seeds(cfg)


def test_stage5_rejects_ppo_observation_shape_mismatch():
    model = SimpleNamespace(observation_space=SimpleNamespace(shape=(52,)))
    env = SimpleNamespace(observation_space=SimpleNamespace(shape=(63,)))
    with pytest.raises(ValueError, match="does not match"):
        validate_model_env_observation_shape(model, env, "ppo_model.zip")


def test_stage5_metrics_distinguish_shield_calls_from_replacements():
    metrics = aggregate_episode_reports(
        [
            {
                "collision": False,
                "near_miss": False,
                "min_distance": 5.0,
                "ttc_p1": 2.0,
                "drac_p99": 1.0,
                "intervention_count": 3,
                "shield_call_count": 3,
                "actual_replacement_count": 0,
                "fallback_count": 0,
            },
            {
                "collision": False,
                "near_miss": False,
                "min_distance": 4.0,
                "ttc_p1": 1.5,
                "drac_p99": 1.2,
                "intervention_count": 4,
                "shield_call_count": 4,
                "actual_replacement_count": 2,
                "fallback_count": 0,
            },
        ]
    )
    assert metrics["shield_call_rate"] == 1.0
    assert metrics["actual_replacement_rate"] == 0.5
    assert metrics["mean_shield_calls"] == pytest.approx(3.5)
    assert metrics["mean_actual_replacements"] == pytest.approx(1.0)


def test_forecast_source_parser_rejects_conflicting_legacy_and_multi_args():
    assert resolve_forecast_sources("constant_velocity,wcdt") == ["constant_velocity", "wcdt"]
    assert resolve_forecast_sources(forecast_source="wcdt") == ["wcdt"]
    with pytest.raises(ValueError, match="either"):
        resolve_forecast_sources("wcdt", forecast_source="constant_velocity")


def test_full_pipeline_generated_configs_use_forecast_model_and_checkpoint(tmp_path):
    configs = build_generated_configs(
        "safe_rl_test_run",
        tmp_path,
        stage1_episodes=2,
        ppo_timesteps=128,
    )
    assert "forecast_cv_ppo" in configs
    assert "forecast_wcdt_ppo" in configs
    stage5 = yaml.safe_load(configs["stage5_multi_groups"].read_text(encoding="utf-8"))
    groups = {item["name"]: item for item in stage5["stage5"]["groups"]}
    assert stage5["stage5"]["episodes_per_group"] == 20
    assert len(stage5["stage5"]["seeds"]) == 20
    assert groups["ppo"]["model_path"] == "safe_rl_output/runs/safe_rl_test_run/stage3/ppo_model.zip"
    assert groups["ppo_shield"]["model_path"] == "safe_rl_output/runs/safe_rl_test_run/stage3/ppo_model.zip"
    assert groups["ppo_cv_features"]["model_path"] == (
        "safe_rl_output/runs/safe_rl_test_run_forecast_cv/stage3/ppo_model.zip"
    )
    assert groups["ppo_cv_features"]["forecast_source"] == "constant_velocity"
    assert "forecast_checkpoint" not in groups["ppo_cv_features"]
    assert groups["cv_prediction_shield"]["model_path"] == (
        "safe_rl_output/runs/safe_rl_test_run_forecast_cv/stage3/ppo_model.zip"
    )
    assert groups["ppo_wcdt_features"]["model_path"] == (
        "safe_rl_output/runs/safe_rl_test_run_forecast_wcdt/stage3/ppo_model.zip"
    )
    assert groups["ppo_wcdt_features"]["forecast_checkpoint"] == (
        "safe_rl_output/runs/safe_rl_test_run/stage2/wcdt_predictor.pt"
    )
    assert groups["wcdt_prediction_shield"]["forecast_checkpoint"] == (
        "safe_rl_output/runs/safe_rl_test_run/stage2/wcdt_predictor.pt"
    )

    stage2_stage4 = yaml.safe_load(configs["stage2_with_stage4"].read_text(encoding="utf-8"))
    assert stage2_stage4["prediction"]["train_enabled"] is False

    forecast_cv = yaml.safe_load(configs["forecast_cv_ppo"].read_text(encoding="utf-8"))
    assert forecast_cv["run"]["run_id"] == "safe_rl_test_run_forecast_cv"
    assert forecast_cv["forecast_features"]["source"] == "constant_velocity"
    assert forecast_cv["forecast_features"]["checkpoint"] is None
    assert forecast_cv["forecast_features"]["allow_heuristic_fallback"] is False
    assert forecast_cv["rl"]["total_timesteps"] == 128

    forecast_wcdt = yaml.safe_load(configs["forecast_wcdt_ppo"].read_text(encoding="utf-8"))
    assert forecast_wcdt["run"]["run_id"] == "safe_rl_test_run_forecast_wcdt"
    assert forecast_wcdt["forecast_features"]["source"] == "wcdt"
    assert forecast_wcdt["forecast_features"]["checkpoint"] == (
        "safe_rl_output/runs/safe_rl_test_run/stage2/wcdt_predictor.pt"
    )


def test_full_pipeline_generated_configs_support_single_forecast_source(tmp_path):
    cv_configs = build_generated_configs(
        "safe_rl_test_run",
        tmp_path / "cv",
        forecast_sources=["constant_velocity"],
    )
    cv_stage5 = yaml.safe_load(cv_configs["stage5_multi_groups"].read_text(encoding="utf-8"))
    cv_groups = {item["name"]: item for item in cv_stage5["stage5"]["groups"]}
    assert "ppo_cv_features" in cv_groups
    assert "ppo_wcdt_features" not in cv_groups
    assert "forecast_cv_ppo" in cv_configs
    assert "forecast_wcdt_ppo" not in cv_configs

    wcdt_configs = build_generated_configs(
        "safe_rl_test_run",
        tmp_path / "wcdt",
        stage1_episodes=2,
        ppo_timesteps=128,
        forecast_sources=["wcdt"],
    )
    wcdt_stage5 = yaml.safe_load(wcdt_configs["stage5_multi_groups"].read_text(encoding="utf-8"))
    wcdt_groups = {item["name"]: item for item in wcdt_stage5["stage5"]["groups"]}
    assert "ppo_cv_features" not in wcdt_groups
    assert "ppo_wcdt_features" in wcdt_groups
    assert "forecast_cv_ppo" not in wcdt_configs
    assert "forecast_wcdt_ppo" in wcdt_configs
    assert wcdt_groups["ppo_wcdt_features"]["forecast_checkpoint"] == (
        "safe_rl_output/runs/safe_rl_test_run/stage2/wcdt_predictor.pt"
    )


def _fake_group(seed_rewards: list[tuple[int, float]], reward: float, near_miss: float = 0.0, min_distance: float = 5.0):
    return {
        "episodes": [
            {
                "seed": seed,
                "episode_reward": episode_reward,
                "min_distance": min_distance,
                "ttc_p1": 2.0,
                "drac_p99": 1.0,
                "intervention_count": 0,
                "actual_replacement_count": 0,
                "fallback_count": 0,
            }
            for seed, episode_reward in seed_rewards
        ],
        "metrics": {
            "average_reward": reward,
            "near_miss_rate": near_miss,
            "min_distance_p1": min_distance,
            "fallback_rate": 0.0,
        },
    }


def test_stage5_dynamic_paired_delta_and_acceptance_for_optional_forecast_groups():
    reports = {
        "ppo": _fake_group([(1, 100.0)], 100.0),
        "ppo_shield": _fake_group([(1, 101.0)], 101.0),
        "ppo_cv_features": _fake_group([(1, 99.0)], 99.0),
        "cv_prediction_shield": _fake_group([(1, 100.0)], 100.0),
        "ppo_wcdt_features": _fake_group([(1, 98.0)], 98.0),
        "wcdt_prediction_shield": _fake_group([(1, 99.0)], 99.0),
    }
    paired = _build_paired_delta(reports)
    assert set(paired) >= {
        "ppo_vs_ppo_shield",
        "ppo_cv_features_vs_cv_prediction_shield",
        "ppo_wcdt_features_vs_wcdt_prediction_shield",
        "ppo_vs_ppo_cv_features",
        "ppo_cv_features_vs_ppo_wcdt_features",
    }
    acceptance = _build_acceptance(reports)
    assert acceptance["ppo_shield"]["available"]
    assert acceptance["cv_prediction_shield"]["available"]
    assert acceptance["wcdt_prediction_shield"]["available"]
    assert acceptance["forecast_cv_vs_baseline"]["available"]
    assert acceptance["forecast_wcdt_vs_cv"]["available"]

    single = {
        "ppo": reports["ppo"],
        "ppo_shield": reports["ppo_shield"],
        "ppo_wcdt_features": reports["ppo_wcdt_features"],
        "wcdt_prediction_shield": reports["wcdt_prediction_shield"],
    }
    single_acceptance = _build_acceptance(single)
    assert "forecast_wcdt_vs_cv" not in single_acceptance
    assert single_acceptance["wcdt_prediction_shield"]["available"]


def test_sumo_start_retries_after_transient_traci_failure(monkeypatch):
    cfg = load_config()
    cfg.scenario["sumo_start_retries"] = 2
    cfg.scenario["sumo_start_retry_delay"] = 0.0
    env = SumoHighwayMergeEnv(cfg, seed=1)

    class _FakeTraci:
        def __init__(self):
            self.calls = 0

        def start(self, _cmd, label, numRetries):
            self.calls += 1
            if self.calls == 1:
                raise OSError("transient port collision")

        def getConnection(self, _label):
            return SimpleNamespace(close=lambda wait=True: None)

        def close(self, wait=True):
            return None

    fake = _FakeTraci()
    monkeypatch.setattr(env, "_import_traci", lambda: fake)
    env._start_sumo()
    assert fake.calls == 2
    assert env._traci is not None
