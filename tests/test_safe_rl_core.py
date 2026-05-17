from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from safe_rl.pipeline.run_full_pipeline import build_generated_configs
from safe_rl.pipeline.stage5_paired_eval import _select_eval_seeds
from safe_rl.risk.risk_aggregator import aggregate_episode_reports
from safe_rl.risk.risk_module import RiskPrediction
from safe_rl.rl.evaluation import validate_model_env_observation_shape
from safe_rl.shield.safety_shield import SafetyShield
from safe_rl.sim.action_space import ACTIONS, decode_action
from safe_rl.sim.metrics import compute_step_metrics
from safe_rl.sim.scenario_validation import validate_scenario_geometry
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


def test_full_pipeline_generated_configs_use_forecast_model_and_checkpoint(tmp_path):
    configs = build_generated_configs(
        "safe_rl_test_run",
        tmp_path,
        stage1_episodes=2,
        ppo_timesteps=128,
    )
    stage5 = yaml.safe_load(configs["stage5_four_groups"].read_text(encoding="utf-8"))
    groups = {item["name"]: item for item in stage5["stage5"]["groups"]}
    assert stage5["stage5"]["episodes_per_group"] == 20
    assert len(stage5["stage5"]["seeds"]) == 20
    assert groups["ppo"]["model_path"] == "safe_rl_output/runs/safe_rl_test_run/stage3/ppo_model.zip"
    assert groups["ppo_wcdt_features"]["model_path"] == (
        "safe_rl_output/runs/safe_rl_test_run_forecast/stage3/ppo_model.zip"
    )
    assert groups["ppo_wcdt_features"]["forecast_checkpoint"] == (
        "safe_rl_output/runs/safe_rl_test_run/stage2/wcdt_predictor.pt"
    )

    forecast = yaml.safe_load(configs["forecast_ppo"].read_text(encoding="utf-8"))
    assert forecast["forecast_features"]["checkpoint"] == "safe_rl_output/runs/safe_rl_test_run/stage2/wcdt_predictor.pt"
    assert forecast["forecast_features"]["allow_heuristic_fallback"] is False
    assert forecast["rl"]["total_timesteps"] == 128
