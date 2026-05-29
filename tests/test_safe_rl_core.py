from __future__ import annotations

import xml.etree.ElementTree as ET
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from safe_rl.prediction.forecast_feature_augmentor import (
    ForecastFeatureAugmentor,
    forecast_target_lane_gap_from_trajectories,
)
from safe_rl.analysis.forecast_diagnostics import (
    _feature_source_summary,
    _forecast_behavior_diagnostics,
    _forecast_conclusion,
    _forecast_features_from_prediction,
    _policy_feature_sensitivity_from_actions,
)
from safe_rl.pipeline.run_full_pipeline import build_generated_configs, resolve_forecast_sources
from safe_rl.pipeline.common import write_report
from safe_rl.pipeline.stage2_train_prediction_risk import (
    _binary_calibration_summary,
    _configured_sample_weight,
    _ordered_prediction_indices,
    _risk_ranking_summary,
    _risk_training_arrays,
    _split_indices,
    _split_risk_indices,
    _temperature_scaled_probabilities,
    _temperature_scaling_diagnostics,
)
from safe_rl.pipeline.stage5_paired_eval import _build_acceptance, _build_paired_delta, _group_overrides, _select_eval_seeds
from safe_rl.pipeline.stage5_confirmatory_eval import (
    build_confirmatory_payload,
    build_confirmatory_summary,
    validate_confirmatory_inputs,
)
from safe_rl.pipeline.stage5_shield_sweep import (
    AGGRESSIVE_VARIANTS,
    DEFAULT_VARIANTS,
    _calibration_effect_summary,
    _shield_score_diagnostics,
    _threshold_sensitivity_summary,
    _variant_report,
    build_sweep_groups,
    sweep_variants,
)
from safe_rl.prediction.sumo_wcdt_adapter import SumoWcDTAdapter
from safe_rl.prediction.wcdt_v2_predictor import (
    INPUT_DIM as WCDT_V2_INPUT_DIM,
    build_v2_numpy_batch,
    ordered_merge_local_indices,
)
from safe_rl.risk.candidate_risk_ranker import CandidateRiskRanker
from safe_rl.risk.merge_local import candidate_action_risk_samples, is_candidate_legal, target_lane_neighbors
from safe_rl.risk.risk_feature_extractor import extract_candidate_features
from safe_rl.risk.risk_aggregator import aggregate_episode_reports
from safe_rl.risk.risk_module import RiskPrediction, risk_loss
from safe_rl.risk.stage1_sampling import configured_sampling_probs, sampling_summary, select_stage1_action
from safe_rl.rl.evaluation import validate_model_env_observation_shape
from safe_rl.rl.ppo import _safety_score
from safe_rl.shield.safety_shield import SafetyShield
from safe_rl.sim.action_space import ACTIONS, decode_action
from safe_rl.sim.history_buffer import HistoryBuffer
from safe_rl.sim.metrics import compute_step_metrics
from safe_rl.sim.scenario_validation import validate_scenario_geometry
from safe_rl.sim.sumo_highway_merge_env import SumoHighwayMergeEnv
from safe_rl.sim.types import StepMetrics, VehicleState
from safe_rl.utils.config import load_config
from safe_rl.utils.io import write_json


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
    assert not samples[0].candidate_legal
    assert samples[0].lane_oob == 1.0
    assert samples[4].candidate_legal
    assert samples[4].lane_oob == 0.0


def test_lane_oob_is_split_from_overall_traffic_risk():
    cfg = load_config()
    ego = VehicleState("ego", 100.0, 0.0, 0.0, 12.0, 0, "ramp_0", 30.0, "ramp_in")
    context = {"ego": ego, "vehicles": [ego], "lane_count": 1, "config": cfg}
    sample = next(item for item in candidate_action_risk_samples(context) if item.action == 0)
    assert not sample.candidate_legal
    assert sample.lane_oob == 1.0
    assert sample.traffic_risk == 0.0
    assert sample.overall_risk == 0.0


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


def test_wcdt_forecast_merge_gap_uses_target_lane_gap_not_min_distance():
    cfg = load_config()
    ego = VehicleState("ego", 0.0, -1.6, 0.0, 0.0, 2, "main_2", 0.0, "main_in")
    trajectories = np.zeros((3, 4, 5), dtype=np.float32)
    trajectories[0, :, :2] = np.asarray([20.0, -1.6], dtype=np.float32)
    trajectories[1, :, :2] = np.asarray([-12.0, -1.6], dtype=np.float32)
    trajectories[2, :, :2] = np.asarray([2.0, -8.0], dtype=np.float32)
    features = ForecastFeatureAugmentor(cfg)._from_prediction(
        ego,
        [],
        {"future_trajectories": trajectories, "uncertainty": 0.25},
    )
    assert features[5] == pytest.approx(7.2, abs=1.0e-5)
    assert features[5] != pytest.approx(features[0])

    wider_gap = trajectories.copy()
    wider_gap[1, :, 0] = -30.0
    wider_features = ForecastFeatureAugmentor(cfg)._from_prediction(
        ego,
        [],
        {"future_trajectories": wider_gap, "uncertainty": 0.25},
    )
    assert wider_features[5] > features[5]


def test_forecast_diagnostics_prediction_features_match_runtime_gap_semantics():
    cfg = load_config()
    ego = VehicleState("ego", 0.0, -1.6, 0.0, 0.0, 2, "main_2", 0.0, "main_in")
    trajectories = np.zeros((3, 4, 5), dtype=np.float32)
    trajectories[0, :, :2] = np.asarray([20.0, -1.6], dtype=np.float32)
    trajectories[1, :, :2] = np.asarray([-12.0, -1.6], dtype=np.float32)
    trajectories[2, :, :2] = np.asarray([2.0, -8.0], dtype=np.float32)
    runtime = ForecastFeatureAugmentor(cfg)._from_prediction(
        ego,
        [],
        {"future_trajectories": trajectories, "uncertainty": 0.25},
    )
    diagnostics = _forecast_features_from_prediction(ego, trajectories, 0.25, cfg)
    assert diagnostics[5] == pytest.approx(runtime[5])
    assert diagnostics[5] != pytest.approx(diagnostics[0])
    assert forecast_target_lane_gap_from_trajectories(np.zeros((4, 2), dtype=np.float32), trajectories, cfg) == pytest.approx(
        diagnostics[5]
    )


def test_forecast_feature_summary_reports_gap_min_distance_equal_rate():
    features = np.asarray(
        [
            [1.0, 2.0, 0.0, 0.0, 0.1, 1.0, 0.0, 0.0, 0.2, 0.1, 0.0],
            [2.0, 3.0, 0.0, 0.0, 0.2, 5.0, 0.0, 0.0, 0.3, 0.1, 0.0],
        ],
        dtype=np.float32,
    )
    summary = _feature_source_summary({"wcdt_v2": features})
    assert summary["runtime_diagnostics_feature_semantics_consistent"]
    assert summary["sources"]["wcdt_v2"]["features"]["forecast_merge_gap"]["count"] == 2
    assert summary["highlight"]["wcdt_v2"]["forecast_uncertainty"]["mean"] == pytest.approx(0.15)
    assert summary["forecast_merge_gap_equals_min_distance_rate"]["wcdt_v2"] == pytest.approx(0.5)


def test_policy_feature_sensitivity_detects_zero_and_shuffle_action_changes():
    sensitivity = _policy_feature_sensitivity_from_actions(
        original_actions=[4, 4, 5, 5],
        zeroed_actions=[4, 3, 5, 5],
        shuffled_actions=[4, 4, 4, 5],
    )
    assert sensitivity["available"]
    assert sensitivity["original_vs_zeroed_action_agreement_rate"] == pytest.approx(0.75)
    assert sensitivity["original_vs_shuffled_action_agreement_rate"] == pytest.approx(0.75)
    assert sensitivity["first_diff_zeroed_step_summary"]["min"] == 1
    assert sensitivity["first_diff_shuffled_step_summary"]["min"] == 2
    assert sensitivity["action_sensitive_to_forecast_features"]


def test_json_writers_convert_non_finite_numbers_to_null(tmp_path):
    report_path = tmp_path / "report.json"
    write_report(
        report_path,
        {
            "nan": float("nan"),
            "inf": float("inf"),
            "nested": {"np_nan": np.float32(np.nan), "ok": 1.0},
        },
    )
    text = report_path.read_text(encoding="utf-8")
    assert "NaN" not in text
    assert "Infinity" not in text
    parsed = json.loads(text)
    assert parsed["nan"] is None
    assert parsed["inf"] is None
    assert parsed["nested"]["np_nan"] is None
    assert parsed["nested"]["ok"] == pytest.approx(1.0)

    io_path = tmp_path / "io.json"
    write_json(io_path, {"bad": np.float64(np.inf)})
    assert json.loads(io_path.read_text(encoding="utf-8"))["bad"] is None


def test_risk_calibration_summary_reports_ece_brier_and_nll():
    pred = np.asarray([0.05, 0.20, 0.80, 0.95], dtype=np.float32)
    target = np.asarray([0.0, 0.0, 1.0, 1.0], dtype=np.float32)
    weight = np.ones((4,), dtype=np.float32)
    legal = np.ones((4,), dtype=np.float32)
    summary = _binary_calibration_summary(pred, target, weight, legal, bin_count=2)
    assert summary["sample_count"] == 4
    assert summary["brier"] < 0.05
    assert summary["nll"] < 0.25
    assert len(summary["reliability_bins"]) == 2
    assert summary["ece"] >= 0.0


def test_temperature_scaling_diagnostics_can_improve_nll_without_changing_rank():
    cfg = load_config()
    pred = np.asarray([0.01, 0.20, 0.80, 0.99], dtype=np.float32)
    target = np.asarray([1.0, 1.0, 0.0, 0.0], dtype=np.float32)
    weight = np.ones((4,), dtype=np.float32)
    legal = np.ones((4,), dtype=np.float32)
    cfg.risk_module["calibration"]["temperature_grid"] = [1.0, 2.0, 5.0]
    report = _temperature_scaling_diagnostics(pred, target, weight, legal, cfg)
    assert report["available"]
    scaled = _temperature_scaled_probabilities(pred, report["temperature"])
    assert list(np.argsort(pred)) == list(np.argsort(scaled))
    assert report["calibrated_summary"]["nll"] <= _binary_calibration_summary(
        pred, target, weight, legal
    )["nll"]


def test_forecast_conclusion_rejects_wcdt_with_worse_fde_and_flat_uncertainty():
    report = {
        "cv_prediction": {"ade": {"mean": 2.0}, "fde": {"mean": 4.0}},
        "wcdt_prediction": {
            "available": True,
            "ade": {"mean": 6.0},
            "fde": {"mean": 13.0},
            "uncertainty": {"std": 0.0},
            "confidence_fde_correlation": 0.0,
        },
        "forecast_behavior": {"step_action_agreement_rate": 0.2},
    }
    conclusion = _forecast_conclusion(report)
    assert conclusion["cv_vs_wcdt_action_agreement"] == pytest.approx(0.2)
    assert not conclusion["wcdt_prediction_quality_pass"]
    assert not conclusion["wcdt_uncertainty_quality_pass"]
    assert not conclusion["wcdt_recommended_for_stage5"]
    assert not conclusion["wcdt_v2_recommended_for_stage5"]


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


def _shield_context_with_ramp_ego():
    context = _shield_context()
    context.update(
        {
            "ego": VehicleState("ego", 100.0, 0.0, 0.0, 12.0, 0, "ramp_0", 30.0, "ramp_in"),
            "vehicles": [],
            "lane_count": 1,
            "config": load_config(),
        }
    )
    return context


def test_shield_guided_reward_penalizes_action_shield_would_replace():
    cfg = load_config()
    cfg.rl["reward_profile"] = "shield_guided_forecast"
    cfg.rl["shield_guided_reward"]["raw_risk_threshold"] = 0.85
    cfg.rl["shield_guided_reward"]["risk_margin_threshold"] = 0.15
    env = SumoHighwayMergeEnv(cfg, seed=1, reward_risk_model=_StaticRiskModel({4: 0.95, 5: 0.40}))
    penalty, debug = env._shield_guided_reward_adjustment(decode_action(4), _shield_context())
    assert penalty < 0.0
    assert debug["raw_action_risk"] == pytest.approx(0.95)
    assert debug["best_candidate_risk"] == pytest.approx(0.40)
    assert debug["risk_margin"] == pytest.approx(0.55)
    assert debug["would_replace"]
    assert debug["shield_guided_reward_penalty"] == pytest.approx(penalty)


def test_shield_guided_reward_does_not_penalize_safe_raw_action():
    cfg = load_config()
    cfg.rl["reward_profile"] = "shield_guided_forecast"
    env = SumoHighwayMergeEnv(cfg, seed=1, reward_risk_model=_StaticRiskModel({4: 0.50, 5: 0.45}))
    penalty, debug = env._shield_guided_reward_adjustment(decode_action(4), _shield_context())
    assert penalty == pytest.approx(0.0)
    assert debug["raw_action_risk"] == pytest.approx(0.50)
    assert not debug["would_replace"]


def test_shield_keeps_raw_action_below_activation_threshold():
    cfg = _shield_cfg()
    shield = SafetyShield(cfg, _StaticRiskModel({4: 0.50}))
    raw = decode_action(4)
    final, record = shield.select_action(raw, _shield_context())
    assert final.index == raw.index
    assert record["replacement_reason"] == "raw_safe"
    assert not record["fallback"]
    assert record["raw_candidate_legal"]
    assert record["legal_candidate_count"] == 9


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
    assert record["best_candidate_action"] == 5
    assert record["best_candidate_risk"] == pytest.approx(0.40)
    assert record["best_candidate_risk_delta"] == pytest.approx(0.55)


def test_ranker_filters_illegal_candidates_on_ramp():
    cfg = _shield_cfg()
    context = _shield_context_with_ramp_ego()
    context["config"] = cfg
    ranker = CandidateRiskRanker(cfg, _StaticRiskModel({index: 0.1 for index in range(9)}))
    ranked = ranker.rank(decode_action(4), context)
    assert {action.index for action, _prediction, _score in ranked} == {3, 4, 5}
    assert all(is_candidate_legal(action, context) for action, _prediction, _score in ranked)


def test_stage2_infers_legacy_lane_oob_and_weights_from_features():
    cfg = load_config()
    data = {
        "risk_features": np.asarray(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        ),
        "actions": np.asarray([0, 4], dtype=np.int64),
        "overall_risk": np.asarray([1.0, 1.0], dtype=np.float32),
        "risk_types": np.asarray(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
    }
    arrays = _risk_training_arrays(data)
    weights = _configured_sample_weight(cfg, arrays)
    assert arrays["lane_oob_risk"].tolist() == [1.0, 0.0]
    assert arrays["candidate_legal"].tolist() == [0.0, 1.0]
    assert arrays["traffic_risk"].tolist() == [0.0, 1.0]
    assert weights[0] == pytest.approx(0.0)
    assert weights[1] == pytest.approx(cfg.risk_module.positive_traffic_risk_weight)


def test_stage2_ranking_summary_infers_legacy_nine_row_groups():
    actions = np.asarray(list(range(9)) * 2, dtype=np.int64)
    risk_features = np.zeros((18, 8), dtype=np.float32)
    risk_features[:, 5] = np.where(np.isin(actions, [3, 4, 5]), 0.0, 1.0)
    labels = np.zeros((18, 5), dtype=np.float32)
    labels[actions == 3, 1] = 1.0
    labels[actions == 4, 1] = 1.0
    data = {
        "risk_features": risk_features,
        "actions": actions,
        "overall_risk": np.max(labels, axis=1),
        "risk_types": labels,
        "executed_actions": np.asarray([4, 4], dtype=np.int64),
    }
    arrays = _risk_training_arrays(data)
    train_idx, val_idx = _split_risk_indices(arrays, 0.5, seed=1)
    assert train_idx.shape[0] % 9 == 0
    assert val_idx.shape[0] % 9 == 0

    predictions = np.ones((18,), dtype=np.float32)
    predictions[actions == 5] = 0.1
    summary = _risk_ranking_summary(arrays, np.arange(18), predictions)
    assert summary["available"]
    assert summary["evaluated_group_count"] == 2
    assert summary["skipped_incomplete_group_count"] == 0
    assert summary["top1_match_rate"] == pytest.approx(1.0)
    assert summary["model_best_action_histogram"]["5"] == 2


def test_stage2_ranking_summary_skips_incomplete_candidate_groups():
    data = {
        "risk_features": np.zeros((3, 8), dtype=np.float32),
        "actions": np.asarray([0, 1, 2], dtype=np.int64),
        "overall_risk": np.zeros((3,), dtype=np.float32),
        "risk_types": np.zeros((3, 5), dtype=np.float32),
    }
    arrays = _risk_training_arrays(data)
    summary = _risk_ranking_summary(arrays, np.arange(3), np.zeros((3,), dtype=np.float32))
    assert not summary["available"]
    assert summary["skipped_incomplete_group_count"] == 1


def test_stage2_prediction_split_has_disjoint_validation_samples():
    train_idx, val_idx = _split_indices(10, 0.2, seed=1)
    assert len(train_idx) == 8
    assert len(val_idx) == 2
    assert set(train_idx).isdisjoint(set(val_idx))
    assert set(train_idx).union(set(val_idx)) == set(range(10))


def test_stage2_wcdt_prediction_order_prioritizes_merge_local_agents():
    cfg = load_config()
    history = np.zeros((6, cfg.scenario.history_steps, 5), dtype=np.float32)
    mask = np.ones((6,), dtype=np.float32)
    history[:, :, 3] = 20.0
    history[0, :, 0] = 200.0
    history[0, :, 1] = 0.0
    history[1, :, 0] = 212.0
    history[1, :, 1] = -1.6
    history[2, :, 0] = 190.0
    history[2, :, 1] = -1.6
    history[3, :, 0] = 205.0
    history[3, :, 1] = 2.0
    history[4, :, 0] = 201.0
    history[4, :, 1] = -8.0
    history[5, :, 0] = 260.0
    history[5, :, 1] = -4.8
    ordered = _ordered_prediction_indices(cfg, history, mask)
    assert ordered[:3] == [1, 2, 3]


def test_runtime_wcdt_adapter_prioritizes_target_lane_front_rear_and_ramp():
    cfg = load_config()
    history = HistoryBuffer(cfg.scenario.history_steps, max_agents=6)
    states = [
        VehicleState("ego", 200.0, 2.0, 0.0, 20.0, 0, "ramp_0", 100.0, "ramp_in"),
        VehicleState("target_front", 214.0, -1.6, 0.0, 20.0, 2, "main_2", 214.0, "main_in"),
        VehicleState("target_rear", 190.0, -1.6, 0.0, 20.0, 2, "main_2", 190.0, "main_in"),
        VehicleState("ramp_front", 208.0, 2.0, 0.0, 18.0, 0, "ramp_0", 108.0, "ramp_in"),
        VehicleState("other_lane", 202.0, -8.0, 0.0, 20.0, 0, "main_0", 202.0, "main_in"),
    ]
    for _ in range(cfg.scenario.history_steps):
        history.append(states)
    ordered = SumoWcDTAdapter(cfg)._ordered_agent_ids(history, "ego")
    assert ordered[:3] == ["target_front", "target_rear", "ramp_front"]


def test_wcdt_v2_actor_selection_prioritizes_merge_local_agents():
    cfg = load_config()
    history = np.zeros((6, cfg.scenario.history_steps, 5), dtype=np.float32)
    mask = np.ones((6,), dtype=np.float32)
    history[:, :, 3] = 20.0
    history[0, :, 0] = 200.0
    history[0, :, 1] = 2.0
    history[1, :, 0] = 214.0
    history[1, :, 1] = -1.6
    history[2, :, 0] = 190.0
    history[2, :, 1] = -1.6
    history[3, :, 0] = 208.0
    history[3, :, 1] = 2.0
    history[4, :, 0] = 202.0
    history[4, :, 1] = -8.0
    history[5, :, 0] = 260.0
    history[5, :, 1] = -4.8
    ordered = ordered_merge_local_indices(cfg, history, mask)
    assert ordered[:3] == [1, 2, 3]


def test_wcdt_v2_batch_has_fixed_shape_and_cv_baseline():
    cfg = load_config()
    cfg.prediction["wcdt_v2_max_agents"] = 3
    history = np.zeros((2, 5, cfg.scenario.history_steps, 5), dtype=np.float32)
    future = np.zeros((2, 5, cfg.scenario.forecast_horizon_steps, 5), dtype=np.float32)
    mask = np.ones((2, 5), dtype=np.float32)
    history[..., 3] = 10.0
    history[:, 0, :, 0] = 200.0
    history[:, 0, :, 1] = 2.0
    history[:, 1, :, 0] = 212.0
    history[:, 1, :, 1] = -1.6
    history[:, 2, :, 0] = 190.0
    history[:, 2, :, 1] = -1.6
    history[:, 3, :, 0] = 208.0
    history[:, 3, :, 1] = 2.0
    history[:, 4, :, 0] = 240.0
    history[:, 4, :, 1] = -8.0
    batch = build_v2_numpy_batch(cfg, history, future, mask, np.asarray([0, 1], dtype=np.int64))
    assert batch["features"].shape == (2, 3, WCDT_V2_INPUT_DIM)
    assert batch["baseline"].shape == (2, 3, cfg.scenario.forecast_horizon_steps, 5)
    assert batch["target"].shape == (2, 3, cfg.scenario.forecast_horizon_steps, 5)
    assert batch["mask"].shape == (2, 3)
    assert np.all(np.isfinite(batch["features"]))
    assert batch["selected_indices"][0].tolist() == [1, 2, 3]
    assert batch["baseline"][0, 0, 0, 0] > history[0, 1, -1, 0]


def test_risk_loss_ignores_zero_weight_samples():
    torch = pytest.importorskip("torch")
    output = {
        "risk_score": torch.tensor([0.99, 0.90], dtype=torch.float32),
        "risk_type_logits": torch.zeros((2, 5), dtype=torch.float32),
        "risk_uncertainty": torch.zeros((2,), dtype=torch.float32),
    }
    labels_a = {
        "risk_score": torch.tensor([0.0, 1.0], dtype=torch.float32),
        "risk_types": torch.zeros((2, 5), dtype=torch.float32),
        "sample_weight": torch.tensor([0.0, 1.0], dtype=torch.float32),
    }
    labels_b = {
        "risk_score": torch.tensor([1.0, 1.0], dtype=torch.float32),
        "risk_types": torch.tensor(
            [
                [1.0, 1.0, 1.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        ),
        "sample_weight": torch.tensor([0.0, 1.0], dtype=torch.float32),
    }
    assert risk_loss(output, labels_a, {"risk": 1.0, "calibration": 0.1}).item() == pytest.approx(
        risk_loss(output, labels_b, {"risk": 1.0, "calibration": 0.1}).item()
    )


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
                "drac_p99_raw": 1.0,
                "drac_p99_capped": 1.0,
                "proxy_collision": False,
                "safety_violation": False,
                "steps": 100,
                "completion_time": 10.0,
                "ego_speed_mean": 20.0,
                "ego_speed_p10": 15.0,
                "hard_brake_rate": 0.0,
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
                "drac_p99": 1.0e6,
                "drac_p99_raw": 1.0e6,
                "drac_p99_capped": 20.0,
                "proxy_collision": True,
                "safety_violation": True,
                "steps": 120,
                "completion_time": 12.0,
                "ego_speed_mean": 18.0,
                "ego_speed_p10": 12.0,
                "hard_brake_rate": 0.25,
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
    assert metrics["proxy_collision_rate"] == pytest.approx(0.5)
    assert metrics["safety_violation_rate"] == pytest.approx(0.5)
    assert metrics["proxy_collision_count"] == 1
    assert metrics["safety_violation_count"] == 1
    assert metrics["min_distance_le_collision_threshold_count"] == 1
    assert metrics["drac_p99_raw"] > 900000.0
    assert metrics["drac_p99_capped"] == pytest.approx(19.81)
    assert metrics["steps_mean"] == pytest.approx(110.0)
    assert metrics["steps_p95"] == pytest.approx(119.0)
    assert metrics["completion_time_mean"] == pytest.approx(11.0)
    assert metrics["completion_time_p95"] == pytest.approx(11.9)
    assert metrics["ego_speed_mean"] == pytest.approx(19.0)
    assert metrics["ego_speed_p10"] == pytest.approx(12.3)
    assert metrics["hard_brake_rate"] == pytest.approx(0.125)


def test_episode_report_includes_efficiency_and_comfort_metrics():
    cfg = load_config()
    env = SumoHighwayMergeEnv(cfg, seed=1)
    env._episode_step = 10
    env._ego_speeds = [10.0, 20.0, 30.0]
    env._episode_metrics = [
        StepMetrics(5.0, 2.0, 1.0, False, False, False, False, 20.0, hard_brake=False),
        StepMetrics(4.0, 1.0, 2.0, False, False, False, False, 18.0, hard_brake=True),
    ]
    report = env.episode_report()
    assert report["completion_time"] == pytest.approx(10 * float(cfg.scenario.step_length))
    assert report["ego_speed_mean"] == pytest.approx(20.0)
    assert report["ego_speed_p10"] == pytest.approx(12.0)
    assert report["hard_brake_count"] == 1
    assert report["hard_brake_rate"] == pytest.approx(0.5)
    assert report["drac_p99_raw"] == pytest.approx(report["drac_p99"])
    assert report["drac_p99_capped"] == pytest.approx(report["drac_p99"])
    assert not report["proxy_collision"]
    assert not report["safety_violation"]
    assert report["proxy_collision_count"] == 0
    assert report["safety_violation_count"] == 0
    assert report["min_distance_le_collision_threshold_count"] == 0


def test_episode_report_defaults_efficiency_metrics_without_ego_samples():
    cfg = load_config()
    env = SumoHighwayMergeEnv(cfg, seed=1)
    report = env.episode_report()
    assert report["completion_time"] == 0.0
    assert report["ego_speed_mean"] == 0.0
    assert report["ego_speed_p10"] == 0.0
    assert report["hard_brake_count"] == 0
    assert report["hard_brake_rate"] == 0.0
    assert report["proxy_collision"] is False
    assert report["safety_violation"] is False
    assert report["proxy_collision_count"] == 0
    assert report["safety_violation_count"] == 0


def test_safety_forecast_reward_profile_penalizes_tail_risk():
    cfg = load_config()
    cfg.rl["reward_profile"] = "safety_forecast"
    env = SumoHighwayMergeEnv(cfg, seed=1)
    ego = VehicleState("ego", float(cfg.scenario.merge_x), 0.0, 0.0, 20.0, 0, "ramp_0", 120.0, "ramp_in")
    front = VehicleState("front", ego.x + 5.0, 0.0, 0.0, 15.0, 2, "main_2", ego.x + 5.0, "main_in")
    rear = VehicleState("rear", ego.x - 4.0, 0.0, 0.0, 22.0, 2, "main_2", ego.x - 4.0, "main_in")
    env.history.append([ego, front, rear])
    risky = StepMetrics(1.0, 0.5, 10.0, False, False, True, True, 5.0)
    safe = StepMetrics(10.0, 5.0, 0.0, False, False, False, False, 20.0)
    assert env._safety_forecast_reward_adjustment(ego, risky) < env._safety_forecast_reward_adjustment(ego, safe)


def test_stage3_safety_score_penalizes_proxy_collision_and_capped_drac():
    score = _safety_score(
        {
            "average_reward": 100.0,
            "min_distance_p1": 2.0,
            "ttc_p1": 1.0,
            "drac_p99_capped": 20.0,
            "proxy_collision_rate": 1.0,
            "safety_violation_rate": 1.0,
        }
    )
    assert score == pytest.approx(-14.0)


def test_stage5_group_shield_overrides_update_shield_config():
    group = SimpleNamespace(
        forecast_features=False,
        shield=True,
        get=lambda key, default=None: {
            "shield_overrides": {
                "activation_risk_threshold": 0.85,
                "replacement_margin": 0.10,
            },
            "risk_module_overrides": {"calibration": {"use_for_runtime": True}},
        }.get(key, default),
    )
    overrides = _group_overrides(group)
    assert overrides["shield"]["enabled"] is True
    assert overrides["shield"]["activation_risk_threshold"] == pytest.approx(0.85)
    assert overrides["shield"]["replacement_margin"] == pytest.approx(0.10)
    assert overrides["risk_module"]["calibration"]["use_for_runtime"] is True


def test_stage5_shield_sweep_generates_default_threshold_variants():
    groups = build_sweep_groups("safe_rl_test_run")
    shield_groups = [group for group in groups if group["name"].startswith("ppo_shield_")]
    assert len(DEFAULT_VARIANTS) == 4
    assert len(shield_groups) == 4
    assert {group["name"] for group in shield_groups} == {
        "ppo_shield_a090_m015",
        "ppo_shield_a085_m015",
        "ppo_shield_a085_m010",
        "ppo_shield_a080_m010",
    }
    assert all(group["shield_overrides"]["allow_fallback"] is False for group in shield_groups)


def test_stage5_shield_sweep_aggressive_variants_are_opt_in():
    assert len(sweep_variants()) == len(DEFAULT_VARIANTS)
    variants = sweep_variants(include_aggressive=True)
    assert len(variants) == len(DEFAULT_VARIANTS) + len(AGGRESSIVE_VARIANTS)
    groups = build_sweep_groups("safe_rl_test_run", variants)
    names = {group["name"] for group in groups if group["name"].startswith("ppo_shield_")}
    assert "ppo_shield_a060_m005" in names
    assert "ppo_shield_a075_m010" in names


def test_stage5_shield_sweep_can_generate_calibrated_variants():
    groups = build_sweep_groups("safe_rl_test_run", include_calibrated=True)
    calibrated = [group for group in groups if group["name"].startswith("ppo_shield_cal_")]
    assert len(calibrated) == len(DEFAULT_VARIANTS)
    assert calibrated[0]["risk_module_overrides"]["calibration"]["use_for_runtime"] is True


def test_stage5_shield_sweep_score_diagnostics_summarize_records():
    report = {
        "shield_overrides": {"activation_risk_threshold": 0.90},
        "episodes": [
            {
                "shield_score_records": [
                    {
                        "replacement_reason": "raw_safe",
                        "raw_risk_score": 0.4,
                        "best_candidate_risk_score": 0.3,
                        "replacement_risk_delta": 0.0,
                        "best_candidate_risk_delta": 0.1,
                    },
                    {
                        "replacement_reason": "replacement",
                        "raw_risk_score": 0.95,
                        "best_candidate_risk_score": 0.5,
                        "replacement_risk_delta": 0.45,
                        "best_candidate_risk_delta": 0.45,
                    },
                ]
            }
        ],
    }
    diagnostics = _shield_score_diagnostics(report)
    assert diagnostics["record_count"] == 2
    assert diagnostics["raw_risk_score"]["count"] == 2
    assert diagnostics["reason_ratios"]["replacement"] == pytest.approx(0.5)
    assert diagnostics["raw_risk_activation_margin"]["max"] == pytest.approx(0.05)


def test_stage5_shield_sweep_variant_report_includes_efficiency_metrics():
    base = _fake_group([(1, 100.0)], 100.0, completion_time=10.0, ego_speed=20.0, hard_brake_rate=0.0)
    candidate = _fake_group(
        [(1, 101.0)],
        101.0,
        completion_time=9.5,
        ego_speed=21.0,
        hard_brake_rate=0.1,
        replacements=1.0,
    )
    variant = _variant_report(base, candidate)
    assert variant["metrics"]["merge_success_rate"] == pytest.approx(1.0)
    assert variant["metrics"]["completion_time_mean"] == pytest.approx(9.5)
    assert variant["metrics"]["ego_speed_mean"] == pytest.approx(21.0)
    assert variant["metrics"]["hard_brake_rate"] == pytest.approx(0.1)
    assert variant["metrics"]["proxy_collision_count"] == 0
    assert variant["metrics"]["safety_violation_count"] == 0
    assert variant["delta"]["mean_completion_time_delta"] == pytest.approx(-0.5)


def test_forecast_source_parser_rejects_conflicting_legacy_and_multi_args():
    assert resolve_forecast_sources("constant_velocity,wcdt,wcdt_v2") == ["constant_velocity", "wcdt", "wcdt_v2"]
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
    assert "forecast_wcdt_v2_ppo" not in configs
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

    v2_configs = build_generated_configs(
        "safe_rl_test_run",
        tmp_path / "wcdt_v2",
        forecast_sources=["wcdt_v2"],
    )
    v2_stage5 = yaml.safe_load(v2_configs["stage5_multi_groups"].read_text(encoding="utf-8"))
    v2_groups = {item["name"]: item for item in v2_stage5["stage5"]["groups"]}
    assert "ppo_wcdt_v2_features" in v2_groups
    assert "wcdt_v2_prediction_shield" in v2_groups
    assert "forecast_wcdt_v2_ppo" in v2_configs
    assert v2_groups["ppo_wcdt_v2_features"]["model_path"] == (
        "safe_rl_output/runs/safe_rl_test_run_forecast_wcdt_v2/stage3/ppo_model.zip"
    )
    assert v2_groups["ppo_wcdt_v2_features"]["forecast_checkpoint"] == (
        "safe_rl_output/runs/safe_rl_test_run/stage2/wcdt_v2_predictor.pt"
    )


def test_full_pipeline_forecast_ppo_overrides_are_forecast_only(tmp_path):
    configs = build_generated_configs(
        "safe_rl_test_run",
        tmp_path,
        ppo_timesteps=20000,
        forecast_ppo_timesteps=100000,
        forecast_ppo_profile="safety",
        forecast_sources=["constant_velocity"],
    )
    main = yaml.safe_load(configs["main"].read_text(encoding="utf-8"))
    forecast_cv = yaml.safe_load(configs["forecast_cv_ppo"].read_text(encoding="utf-8"))
    assert main["rl"]["total_timesteps"] == 20000
    assert "reward_profile" not in main["rl"]
    assert forecast_cv["rl"]["total_timesteps"] == 100000
    assert forecast_cv["rl"]["reward_profile"] == "safety_forecast"


def test_full_pipeline_shield_guided_profile_binds_base_risk_module_for_forecast_only(tmp_path):
    configs = build_generated_configs(
        "safe_rl_test_run",
        tmp_path,
        ppo_timesteps=20000,
        forecast_ppo_timesteps=100000,
        forecast_ppo_profile="shield_guided",
        forecast_sources=["wcdt_v2"],
    )
    main = yaml.safe_load(configs["main"].read_text(encoding="utf-8"))
    forecast = yaml.safe_load(configs["forecast_wcdt_v2_ppo"].read_text(encoding="utf-8"))
    assert "reward_profile" not in main["rl"]
    assert "shield_guided_reward" not in main["rl"]
    assert forecast["rl"]["total_timesteps"] == 100000
    assert forecast["rl"]["reward_profile"] == "shield_guided_forecast"
    assert forecast["rl"]["shield_guided_reward"]["risk_checkpoint"] == (
        "safe_rl_output/runs/safe_rl_test_run/stage2/risk_module.pt"
    )


def _fake_group(
    seed_rewards: list[tuple[int, float]],
    reward: float,
    near_miss: float = 0.0,
    min_distance: float = 5.0,
    drac: float = 1.0,
    drac_capped: float | None = None,
    proxy_collision: float = 0.0,
    safety_violation: float = 0.0,
    success: float = 1.0,
    replacements: float = 0.0,
    completion_time: float = 10.0,
    ego_speed: float = 20.0,
    hard_brake_rate: float = 0.0,
):
    return {
        "episodes": [
            {
                "seed": seed,
                "episode_reward": episode_reward,
                "min_distance": min_distance,
                "ttc_p1": 2.0,
                "drac_p99": drac,
                "drac_p99_raw": drac,
                "drac_p99_capped": drac if drac_capped is None else drac_capped,
                "proxy_collision": bool(proxy_collision),
                "safety_violation": bool(safety_violation),
                "proxy_collision_count": int(bool(proxy_collision)),
                "safety_violation_count": int(bool(safety_violation)),
                "min_distance_le_collision_threshold_count": int(bool(proxy_collision)),
                "completion_time": completion_time,
                "ego_speed_mean": ego_speed,
                "hard_brake_rate": hard_brake_rate,
                "intervention_count": 0,
                "actual_replacement_count": int(replacements),
                "fallback_count": 0,
            }
            for seed, episode_reward in seed_rewards
        ],
        "metrics": {
            "average_reward": reward,
            "near_miss_rate": near_miss,
            "min_distance_p1": min_distance,
            "fallback_rate": 0.0,
            "drac_p99": drac,
            "drac_p99_raw": drac,
            "drac_p99_capped": drac if drac_capped is None else drac_capped,
            "proxy_collision_rate": proxy_collision,
            "safety_violation_rate": safety_violation,
            "proxy_collision_count": int(bool(proxy_collision)),
            "safety_violation_count": int(bool(safety_violation)),
            "min_distance_le_collision_threshold_count": int(bool(proxy_collision)),
            "merge_success_rate": success,
            "completion_time_mean": completion_time,
            "completion_time_p95": completion_time,
            "ego_speed_mean": ego_speed,
            "ego_speed_p10": ego_speed,
            "hard_brake_rate": hard_brake_rate,
            "mean_actual_replacements": replacements,
            "actual_replacement_rate": float(replacements > 0.0),
        },
    }


def test_stage5_dynamic_paired_delta_and_acceptance_for_optional_forecast_groups():
    reports = {
        "ppo": _fake_group([(1, 100.0)], 100.0),
        "ppo_shield": _fake_group([(1, 101.0)], 101.0, completion_time=9.0, ego_speed=21.0, hard_brake_rate=0.1),
        "ppo_cv_features": _fake_group([(1, 99.0)], 99.0),
        "cv_prediction_shield": _fake_group([(1, 100.0)], 100.0),
        "ppo_wcdt_features": _fake_group([(1, 98.0)], 98.0),
        "wcdt_prediction_shield": _fake_group([(1, 99.0)], 99.0),
        "ppo_wcdt_v2_features": _fake_group([(1, 100.0)], 100.0),
        "wcdt_v2_prediction_shield": _fake_group([(1, 101.0)], 101.0),
    }
    paired = _build_paired_delta(reports)
    assert set(paired) >= {
        "ppo_vs_ppo_shield",
        "ppo_cv_features_vs_cv_prediction_shield",
        "ppo_wcdt_features_vs_wcdt_prediction_shield",
        "ppo_wcdt_v2_features_vs_wcdt_v2_prediction_shield",
        "ppo_vs_ppo_cv_features",
        "ppo_cv_features_vs_ppo_wcdt_features",
        "ppo_cv_features_vs_ppo_wcdt_v2_features",
    }
    assert paired["ppo_vs_ppo_shield"]["mean_completion_time_delta"] == pytest.approx(-1.0)
    assert paired["ppo_vs_ppo_shield"]["mean_ego_speed_delta"] == pytest.approx(1.0)
    assert paired["ppo_vs_ppo_shield"]["mean_hard_brake_rate_delta"] == pytest.approx(0.1)
    assert paired["ppo_vs_ppo_shield"]["proxy_collision_count_delta"] == 0
    assert paired["ppo_vs_ppo_shield"]["safety_violation_count_delta"] == 0
    acceptance = _build_acceptance(reports)
    assert acceptance["ppo_shield"]["available"]
    assert acceptance["cv_prediction_shield"]["available"]
    assert acceptance["wcdt_prediction_shield"]["available"]
    assert acceptance["wcdt_v2_prediction_shield"]["available"]
    assert acceptance["forecast_cv_vs_baseline"]["available"]
    assert acceptance["forecast_wcdt_vs_cv"]["available"]
    assert acceptance["forecast_wcdt_v2_vs_cv"]["available"]

    single = {
        "ppo": reports["ppo"],
        "ppo_shield": reports["ppo_shield"],
        "ppo_wcdt_features": reports["ppo_wcdt_features"],
        "wcdt_prediction_shield": reports["wcdt_prediction_shield"],
    }
    single_acceptance = _build_acceptance(single)
    assert "forecast_wcdt_vs_cv" not in single_acceptance
    assert single_acceptance["wcdt_prediction_shield"]["available"]


def test_forecast_behavior_diagnostics_supports_cv_vs_wcdt_v2(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    replay_dir = tmp_path / "safe_rl_output" / "runs" / "safe_rl_behavior_test" / "stage5" / "replay"
    replay_dir.mkdir(parents=True)
    (replay_dir / "ppo_cv_features_seed_1.json").write_text(json.dumps({"actions": [4, 4, 5]}), encoding="utf-8")
    (replay_dir / "ppo_wcdt_v2_features_seed_1.json").write_text(json.dumps({"actions": [4, 5, 5]}), encoding="utf-8")
    report = {
        "groups": {
            "ppo_cv_features": {"episodes": [{"seed": 1}]},
            "ppo_wcdt_v2_features": {"episodes": [{"seed": 1}]},
        }
    }
    diagnostics = _forecast_behavior_diagnostics("safe_rl_behavior_test", report)
    assert diagnostics["available"]
    assert diagnostics["primary_comparison"] == "ppo_cv_features_vs_ppo_wcdt_v2_features"
    comparison = diagnostics["comparisons"]["ppo_cv_features_vs_ppo_wcdt_v2_features"]
    assert comparison["available"]
    assert comparison["step_action_agreement_rate"] == pytest.approx(2 / 3)
    assert comparison["first_diff_step_summary"]["min"] == 1
    assert comparison["left_action_histogram"]["4"] == 2
    assert comparison["right_action_histogram"]["5"] == 2


def test_confirmatory_payload_generates_fifty_seed_six_group_config():
    payload = build_confirmatory_payload("safe_rl_test_run")
    groups = {item["name"]: item for item in payload["stage5"]["groups"]}
    assert payload["stage5"]["episodes_per_group"] == 50
    assert payload["stage5"]["seeds"] == list(range(1, 51))
    assert set(groups) == {
        "ppo",
        "ppo_shield",
        "ppo_cv_features",
        "cv_prediction_shield",
        "ppo_wcdt_v2_features",
        "wcdt_v2_prediction_shield",
    }
    assert groups["ppo_wcdt_v2_features"]["forecast_checkpoint"] == (
        "safe_rl_output/runs/safe_rl_test_run/stage2/wcdt_v2_predictor.pt"
    )


def test_confirmatory_input_validation_reports_missing_checkpoints():
    payload = build_confirmatory_payload("safe_rl_missing_confirmatory_run", episodes=5)
    with pytest.raises(FileNotFoundError, match="Stage5 confirmatory eval requires existing"):
        validate_confirmatory_inputs(payload)


def test_confirmatory_summary_marks_wcdt_v2_shield_not_needed():
    reports = {
        "ppo": _fake_group([(1, 100.0)], 100.0, min_distance=2.0),
        "ppo_shield": _fake_group([(1, 101.0)], 101.0, min_distance=2.1, replacements=1.0),
        "ppo_cv_features": _fake_group([(1, 105.0)], 105.0, min_distance=3.0, drac=8.0),
        "ppo_wcdt_v2_features": _fake_group([(1, 110.0)], 110.0, min_distance=5.0, drac=4.0),
        "wcdt_v2_prediction_shield": _fake_group([(1, 110.0)], 110.0, min_distance=5.0, drac=4.0),
    }
    paired = _build_paired_delta(reports)
    acceptance = _build_acceptance(reports)
    summary = build_confirmatory_summary(reports, paired, acceptance)
    assert summary["ppo_shield_mainline"]["pass"]
    assert summary["wcdt_v2_forecast_mainline"]["pass"]
    assert summary["final_result_summary"]["trusted_mainline"] == ["ppo", "ppo_shield"]
    assert summary["final_result_summary"]["best_safety_combo"] == "wcdt_v2_prediction_shield"
    assert summary["model_role_explanations"]["wcdt_v2_prediction_shield"]["shield_enabled"] is True
    assert summary["reporting_recommendation"][0]["comparison"] == "ppo_vs_ppo_shield"
    assert summary["wcdt_v2_shield"]["shield_not_needed_on_wcdt_v2_policy"]
    assert summary["forecast_policy_utilization_summary"]["available"] is False
    assert summary["overall_pass"]


def test_confirmatory_summary_marks_wcdt_v2_shield_low_frequency_backstop():
    reports = {
        "ppo": _fake_group([(1, 100.0)], 100.0, min_distance=2.0),
        "ppo_shield": _fake_group([(1, 101.0)], 101.0, min_distance=2.1, replacements=1.0),
        "ppo_cv_features": _fake_group([(1, 105.0)], 105.0, min_distance=3.0, drac=8.0),
        "ppo_wcdt_v2_features": _fake_group([(1, 110.0)], 110.0, min_distance=5.0, drac=4.0),
        "wcdt_v2_prediction_shield": _fake_group([(1, 110.5)], 110.5, min_distance=5.2, drac=3.8, replacements=0.1),
    }
    paired = _build_paired_delta(reports)
    acceptance = _build_acceptance(reports)
    summary = build_confirmatory_summary(reports, paired, acceptance)
    assert not summary["wcdt_v2_shield"]["shield_not_needed_on_wcdt_v2_policy"]
    assert summary["wcdt_v2_shield"]["low_frequency_safety_backstop"]
    assert summary["wcdt_v2_shield"]["shield_status"] == "low_frequency_safety_backstop"
    assert summary["overall_pass"]


def test_confirmatory_summary_uses_forecast_policy_utilization_diagnostics():
    reports = {
        "ppo": _fake_group([(1, 100.0)], 100.0, min_distance=2.0),
        "ppo_shield": _fake_group([(1, 101.0)], 101.0, min_distance=2.1, replacements=1.0),
        "ppo_cv_features": _fake_group([(1, 105.0)], 105.0, min_distance=3.0, drac=8.0),
        "ppo_wcdt_v2_features": _fake_group([(1, 110.0)], 110.0, min_distance=5.0, drac=4.0),
        "wcdt_v2_prediction_shield": _fake_group([(1, 110.0)], 110.0, min_distance=5.0, drac=4.0),
    }
    diagnostics = {
        "path": "safe_rl_output/runs/test/stage5/diagnostics/forecast_diagnostics.json",
        "forecast_conclusion": {
            "wcdt_v2_prediction_quality_pass": True,
            "wcdt_v2_uncertainty_quality_pass": True,
            "wcdt_v2_recommended_for_stage5": True,
        },
        "policy_feature_sensitivity": {
            "groups": {
                "ppo_wcdt_v2_features": {
                    "available": True,
                    "action_sensitive_to_forecast_features": False,
                    "original_vs_zeroed_action_agreement_rate": 1.0,
                    "original_vs_shuffled_action_agreement_rate": 1.0,
                }
            }
        },
    }
    summary = build_confirmatory_summary(reports, _build_paired_delta(reports), _build_acceptance(reports), diagnostics)
    utilization = summary["forecast_policy_utilization_summary"]
    assert utilization["available"]
    assert utilization["wcdt_v2_predictor_quality_pass"]
    assert utilization["wcdt_v2_ppo_better_than_cv"]
    assert utilization["forecast_policy_underutilized"]


def test_shield_sweep_summarizes_calibration_effect_and_threshold_sensitivity():
    variants = {
        "ppo_shield_a090_m015": {
            "metrics": {
                "average_reward": 100.0,
                "min_distance_p1": 2.0,
                "ttc_p1": 1.0,
                "drac_p99": 5.0,
                "actual_replacement_rate": 0.2,
                "mean_actual_replacements": 1.0,
                "fallback_rate": 0.0,
                "near_miss_rate": 0.0,
                "collision_rate": 0.0,
            },
            "acceptance": {"shield_regression": False},
            "delta": {"mean_min_distance_delta": 0.1, "mean_drac_delta": -0.1, "mean_reward_delta": 0.0},
            "improved_tail": True,
        },
        "ppo_shield_cal_a090_m015": {
            "metrics": {
                "average_reward": 100.5,
                "min_distance_p1": 2.2,
                "ttc_p1": 1.1,
                "drac_p99": 4.8,
                "actual_replacement_rate": 0.3,
                "mean_actual_replacements": 1.5,
                "fallback_rate": 0.0,
                "near_miss_rate": 0.0,
                "collision_rate": 0.0,
            },
            "acceptance": {"shield_regression": False},
            "delta": {"mean_min_distance_delta": 0.2, "mean_drac_delta": -0.2, "mean_reward_delta": 0.5},
            "improved_tail": True,
        },
        "ppo_shield_a085_m015": {
            "metrics": {"actual_replacement_rate": 0.2, "mean_actual_replacements": 1.0},
            "acceptance": {"shield_regression": False},
            "delta": {},
            "improved_tail": False,
        },
        "ppo_shield_cal_a085_m015": {
            "metrics": {"actual_replacement_rate": 0.4, "mean_actual_replacements": 2.0},
            "acceptance": {"shield_regression": False},
            "delta": {},
            "improved_tail": False,
        },
    }
    calibration = _calibration_effect_summary(variants, include_calibrated=True)
    assert calibration["available"]
    assert calibration["paired_variant_count"] == 2
    assert calibration["replacement_behavior_changed_count"] == 2
    assert calibration["pairs"]["ppo_shield_a090_m015"]["mean_replacements_changed"]

    sensitivity = _threshold_sensitivity_summary(variants)
    assert sensitivity["available"]
    assert sensitivity["families"]["ppo_shield_raw"]["threshold_sensitive"] is False
    assert sensitivity["families"]["ppo_shield_calibrated"]["threshold_sensitive"] is True
    assert sensitivity["risk_score_saturation_suspected"]
    assert sensitivity["calibration_helpful_for_shield"]


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
