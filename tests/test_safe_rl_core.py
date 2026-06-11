from __future__ import annotations

import xml.etree.ElementTree as ET
import hashlib
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
    _constant_velocity_future as diagnostics_constant_velocity_future,
    _feature_source_summary,
    _forecast_behavior_diagnostics,
    _forecast_conclusion,
    _forecast_features_from_prediction,
    _policy_feature_sensitivity_from_actions,
    _vector_to_state,
)
from safe_rl.pipeline.run_full_pipeline import (
    _managed_run_dirs,
    _new_pipeline_state,
    _load_pipeline_state,
    _pipeline_profile_config_sha256,
    _predictor_training_flags,
    _prepare_new_run_dir,
    _remove_managed_run_dirs,
    _reset_unfinished_tasks,
    _resume_invocation,
    _run_pipeline_task,
    _validate_completed_outputs,
    _validate_resume_state,
    _validate_run_id,
    build_generated_configs,
    resolve_forecast_sources,
)
from safe_rl.pipeline.common import write_report
from safe_rl.pipeline.stage2_train_prediction_risk import (
    _binary_calibration_summary,
    _configured_sample_weight,
    _ordered_prediction_indices,
    _risk_ranking_summary,
    _risk_training_arrays,
    _require_trajectory_schema_v2,
    _split_indices,
    _split_risk_indices,
    _temperature_scaled_probabilities,
    _temperature_scaling_diagnostics,
    _target_lane_gap as stage2_target_lane_gap,
    _wcdt_v1_batch_size,
    _wcdt_v2_batch_size,
    _wcdt_v2_early_stopping_config,
    _wcdt_v2_early_stopping_step,
    _wcdt_v3_batch_size,
    _wcdt_v3_early_stopping_config,
)
from safe_rl.pipeline.stage5_paired_eval import _build_acceptance, _build_paired_delta, _group_overrides, _select_eval_seeds
from safe_rl.pipeline.stage5_confirmatory_eval import (
    _wcdt_v3_candidate_summary,
    build_confirmatory_payload,
    build_confirmatory_summary,
    validate_confirmatory_inputs,
)
from safe_rl.pipeline.stage5_failure_audit import build_failure_audit, write_replay_commands
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
from safe_rl.prediction.merge_safety_loss import LOSS_VERSION as MERGE_SAFETY_LOSS_VERSION
from safe_rl.prediction.wcdt_v2_predictor import (
    INPUT_DIM as WCDT_V2_INPUT_DIM,
    WcDTV2ResidualPredictor,
    build_v2_numpy_batch,
    load_v2_ensemble,
    ordered_merge_local_indices,
    v2_loss,
)
from safe_rl.prediction.wcdt_v3_predictor import (
    HISTORY_INPUT_DIM as WCDT_V3_HISTORY_INPUT_DIM,
    WcDTV3TemporalInteractionPredictor,
    _predict_model as predict_v3_model,
    build_v3_numpy_batch,
    load_v3_ensemble,
    tensorize_v3_batch,
    v3_loss,
)
from safe_rl.risk.candidate_risk_ranker import CandidateRiskRanker
from safe_rl.risk.merge_local import (
    candidate_action_risk_samples,
    continuous_risk_target,
    is_candidate_legal,
    rollout_ego,
    route_aware_constant_velocity_rollout,
    target_lane_neighbors,
)
from safe_rl.risk.risk_feature_extractor import extract_candidate_features
from safe_rl.risk.risk_aggregator import aggregate_episode_reports
from safe_rl.risk.risk_module import RiskModuleWrapper, RiskPrediction, risk_loss
from safe_rl.risk.stage1_sampling import _merge_heuristic_action, configured_sampling_probs, sampling_summary, select_stage1_action
from safe_rl.rl.evaluation import _step_safety_record, validate_model_env_observation_shape
from safe_rl.rl.ppo import _checkpoint_selection_score, _checkpoint_selection_weights, _safety_score, _training_device
from safe_rl.pipeline.stage3_train_ppo import _prediction_loss_summary, _prediction_loss_summary_from_checkpoint
from safe_rl.shield.forecast_task_scorer import ForecastAwareTaskScorer
from safe_rl.shield.safety_shield import SafetyShield
from safe_rl.sim.action_space import ACTIONS, decode_action
from safe_rl.sim.history_buffer import HistoryBuffer
from safe_rl.sim.metrics import INF_TTC, bbox_gap, compute_step_metrics, drac, geometric_overlap, relative_ttc
from safe_rl.sim.scenario_validation import validate_scenario_geometry
from safe_rl.sim.scenario_semantics import (
    EDGE_ROLE_AUXILIARY,
    EDGE_ROLE_MAINLINE,
    EDGE_ROLE_RAMP,
    EDGE_ROLE_TARGET,
    advance_route_state,
    edge_length,
    is_taper_miss,
    lane_center,
    lane_center_at_x,
    target_lane_center_at_x,
    target_lane_index,
)
from safe_rl.sim.scenario_snapshot import snapshot_scenario
from safe_rl.sim.sumo_highway_merge_env import SumoHighwayMergeEnv
from safe_rl.sim.types import StepMetrics, VehicleState
from safe_rl.utils.config import load_config
from safe_rl.utils.io import write_json


def test_action_space_has_nine_actions():
    assert len(ACTIONS) == 9
    assert decode_action(4).name == "keep_hold"
    assert decode_action(0).name == "right_decelerate"
    assert decode_action(6).name == "left_decelerate"


def test_metrics_detect_near_miss():
    ego = VehicleState("ego", 0.0, 0.0, 0.0, 10.0, 0, "lane", 0.0, "ramp_in")
    other = VehicleState("other", 4.0, 0.0, 0.0, 0.0, 0, "lane", 0.0, "main_in")
    metrics = compute_step_metrics(ego, [ego, other], collision=False)
    assert metrics.min_distance < 1.0
    assert metrics.near_miss


def test_oriented_box_gap_does_not_flag_parallel_adjacent_lane_vehicle():
    ego = VehicleState("ego", 10.0, 0.0, 0.0, 10.0, 0, "lane0", 0.0, "main_aux")
    other = VehicleState("other", 10.0, 3.2, 0.0, 10.0, 1, "lane1", 0.0, "main_aux")
    assert bbox_gap(ego, other) == pytest.approx(1.4, abs=1.0e-6)
    assert not geometric_overlap(ego, other)
    assert relative_ttc(ego, other) == INF_TTC
    assert drac(ego, other) == 0.0
    metrics = compute_step_metrics(ego, [ego, other], collision=False)
    assert not metrics.near_miss
    assert not metrics.geometric_overlap


def test_oriented_box_gap_matches_same_lane_longitudinal_clearance():
    ego = VehicleState("ego", 10.0, 0.0, 0.0, 10.0, 0, "lane", 0.0, "main_aux")
    other = VehicleState("other", 20.0, 0.0, 0.0, 10.0, 0, "lane", 0.0, "main_aux")
    assert bbox_gap(ego, other) == pytest.approx(5.2, abs=1.0e-6)


def test_oriented_box_overlap_and_crossing_path_ttc():
    ego = VehicleState("ego", 2.4, 0.0, 0.0, 10.0, 0, "lane", 0.0, "main_aux")
    overlap = VehicleState("overlap", 2.4, 0.0, 0.0, 0.0, 0, "lane", 0.0, "main_aux")
    assert geometric_overlap(ego, overlap)
    assert bbox_gap(ego, overlap) == 0.0

    crossing = VehicleState("crossing", 10.0, 7.6, -0.5 * np.pi, 10.0, 0, "lane", 0.0, "main_aux")
    ttc = relative_ttc(ego, crossing)
    assert 0.0 < ttc < 2.0
    assert drac(ego, crossing) > 0.0


def test_metrics_merge_gap_supports_auxiliary_corridor_and_target_lane_filter():
    ego = VehicleState("ego", 100.0, 53.8, 0.0, 10.0, 0, "main_aux_0", 100.0, "main_aux")
    target = VehicleState("target", 112.0, 57.0, 0.0, 10.0, 1, "main_aux_1", 112.0, "main_aux")
    other_lane = VehicleState("other", 102.0, 60.2, 0.0, 10.0, 2, "main_aux_2", 102.0, "main_aux")
    metrics = compute_step_metrics(
        ego,
        [ego, target, other_lane],
        collision=False,
        merge_ego_edges=["ramp_in", "main_aux"],
        merge_target_edges=["main_in", "main_aux", "main_out"],
        merge_target_lane=1,
    )
    assert metrics.merge_gap == pytest.approx(12.0)


def test_scenario_validation_passes():
    cfg = load_config()
    report = validate_scenario_geometry(cfg.scenario.sumocfg)
    assert report["passed"], report["errors"]
    ego = next(item for item in report["seed_positions"] if item["vehicle_id"] == "ego")
    assert ego["first_edge"] == "ramp_in"
    assert report["edge_lane_counts"] == {"main_aux": 4, "main_in": 3, "main_out": 3, "ramp_in": 1}
    assert report["routes"]["route_ramp"] == ["ramp_in", "main_aux", "main_out"]
    assert {"from": "ramp_in", "to": "main_aux", "from_lane": 0, "to_lane": 0} in report["connections"]
    assert not report["warnings"]
    assert all(item["lateral_shift"] <= 0.5 for item in report["through_lane_lateral_shift"])
    assert report["merge_side_consistency"]["ramp_connects_to_auxiliary_lane"]
    assert report["target_seed_lane_consistency"]["target_seeds_consistent"]
    assert report["auxiliary_drop_lane"]["drops_before_main_out"]
    assert report["ramp_entry_angle"] <= 10.0


def test_ramp_connection_targets_adjacent_main_lane():
    con_file = Path("scenarios/highway_merge/highway_merge.con.xml")
    root = ET.parse(con_file).getroot()
    ramp_connection = next(
        connection
        for connection in root.findall("connection")
        if connection.attrib.get("from") == "ramp_in" and connection.attrib.get("to") == "main_aux"
    )
    assert ramp_connection.attrib["toLane"] == "0"


def test_auxiliary_lane_drops_before_main_out():
    con_file = Path("scenarios/highway_merge/highway_merge.con.xml")
    root = ET.parse(con_file).getroot()
    outgoing = [
        connection
        for connection in root.findall("connection")
        if connection.attrib.get("from") == "main_aux" and connection.attrib.get("to") == "main_out"
    ]
    assert {connection.attrib["fromLane"] for connection in outgoing} == {"1", "2", "3"}


def test_scenario_semantics_use_generated_net_lane_geometry():
    cfg = load_config()
    assert edge_length(cfg, "main_aux", 0) == pytest.approx(214.50)
    assert lane_center(cfg, 0, "main_aux", 100.0) == pytest.approx(53.8)
    assert lane_center(cfg, 1, "main_aux", 100.0) == pytest.approx(57.0)
    assert lane_center(cfg, 0, "main_out", 100.0) == pytest.approx(57.0)
    assert target_lane_center_at_x(cfg, 400.0) == pytest.approx(57.0)


def test_target_lane_mapping_tracks_same_physical_lane_across_edges():
    cfg = load_config()
    assert target_lane_index(cfg, "main_in") == 0
    assert target_lane_index(cfg, "main_aux") == 1
    assert target_lane_index(cfg, "main_out") == 0


def test_route_aware_rollout_maps_through_lanes_across_edges():
    cfg = load_config()
    before_aux = VehicleState("main", 297.0, 57.0, 0.0, 20.0, 0, "main_in_0", edge_length(cfg, "main_in", 0) - 1.0, "main_in")
    on_aux, missed = advance_route_state(cfg, before_aux, 2.0)
    assert not missed
    assert on_aux.edge_id == "main_aux"
    assert on_aux.lane_index == 1

    before_out = VehicleState("main", 515.0, 57.0, 0.0, 20.0, 1, "main_aux_1", edge_length(cfg, "main_aux", 1) - 1.0, "main_aux")
    on_out, missed = advance_route_state(cfg, before_out, 2.0)
    assert not missed
    assert on_out.edge_id == "main_out"
    assert on_out.lane_index == 0


def test_forecast_gap_uses_aux_corridor_target_lane_not_auxiliary_lane():
    cfg = load_config()
    ego_rollout = np.asarray([[390.0, 53.8], [400.0, 53.8]], dtype=np.float32)
    trajectories = np.asarray(
        [
            [[405.0, 57.0], [415.0, 57.0]],
            [[392.0, 53.8], [402.0, 53.8]],
        ],
        dtype=np.float32,
    )
    assert forecast_target_lane_gap_from_trajectories(ego_rollout, trajectories, cfg) == pytest.approx(10.2)


def test_stage2_target_lane_gap_uses_aux_corridor_target_lane_not_auxiliary_lane():
    cfg = load_config()
    ego_rollout = np.asarray([[390.0, 53.8], [400.0, 53.8]], dtype=np.float32)
    trajectories = np.asarray(
        [
            [[405.0, 57.0], [415.0, 57.0]],
            [[392.0, 53.8], [402.0, 53.8]],
        ],
        dtype=np.float32,
    )
    assert stage2_target_lane_gap(ego_rollout, trajectories, np.ones((2,), dtype=np.float32), cfg) == pytest.approx(10.2)


def test_route_aware_rollout_enters_auxiliary_lane_and_detects_taper_miss():
    cfg = load_config()
    ramp = VehicleState("ego", 290.0, 53.8, 0.0, 20.0, 0, "ramp_in_0", edge_length(cfg, "ramp_in", 0) - 5.0, "ramp_in")
    on_aux, missed = advance_route_state(cfg, ramp, 10.0)
    assert on_aux.edge_id == "main_aux"
    assert on_aux.lane_index == 0
    assert not missed

    near_taper = VehicleState("ego", 514.0, 53.8, 0.0, 20.0, 0, "main_aux_0", edge_length(cfg, "main_aux", 0) - 2.0, "main_aux")
    rollout, missed = route_aware_constant_velocity_rollout(near_taper, 3, 0.1, cfg)
    assert missed
    assert rollout[-1].edge_id == "main_aux"
    assert rollout[-1].lane_index == 0


def test_route_aware_rollout_projects_cross_edge_position_from_net_geometry():
    cfg = load_config()
    ramp = VehicleState("ego", 290.0, 53.8, 0.0, 20.0, 0, "ramp_in_0", edge_length(cfg, "ramp_in", 0) - 5.0, "ramp_in")
    on_aux, missed = advance_route_state(cfg, ramp, 10.0)
    assert not missed
    assert on_aux.edge_id == "main_aux"
    assert on_aux.lane_index == 0
    assert on_aux.lane_pos == pytest.approx(5.0, abs=0.05)
    assert on_aux.x == pytest.approx(306.50, abs=0.10)
    assert on_aux.y == pytest.approx(53.8)


def test_candidate_lane_change_rollout_is_continuous():
    cfg = load_config()
    ego = VehicleState("ego", 400.0, 53.8, 0.0, 10.0, 0, "main_aux_0", 100.0, "main_aux")
    merge_action = next(action for action in ACTIONS if action.name == "left_hold")
    rollout, missed = rollout_ego(ego, merge_action, 12, 0.1, cfg)
    assert not missed
    assert rollout[0].y > ego.y
    assert rollout[0].y < 57.0
    assert rollout[-1].lane_index == 1
    assert rollout[-1].y == pytest.approx(57.0, abs=0.05)


def test_taper_miss_is_separate_from_lane_oob():
    cfg = load_config()
    ego = VehicleState("ego", 514.0, 53.8, 0.0, 12.0, 0, "main_aux_0", edge_length(cfg, "main_aux", 0) - 2.0, "main_aux")
    assert is_taper_miss(cfg, ego)
    context = {"ego": ego, "vehicles": [ego], "config": cfg, "lane_count": 4}
    keep = next(action for action in ACTIONS if action.name == "keep_hold")
    assert is_candidate_legal(keep, context)


def test_taper_miss_triggers_in_warning_zone_before_auxiliary_lane_end():
    cfg = load_config()
    ego = VehicleState("ego", 500.0, 53.8, 0.0, 25.0, 0, "main_aux_0", edge_length(cfg, "main_aux", 0) - 20.0, "main_aux")
    assert is_taper_miss(cfg, ego)


def test_continuous_risk_target_increases_for_boundary_and_extreme_states():
    cfg = load_config()
    stats = SimpleNamespace(
        target_lane_gap=20.0,
        merge_distance=80.0,
        ego_on_auxiliary=False,
        merge_zone_risk=False,
        taper_miss=False,
    )
    safe = StepMetrics(20.0, 5.0, 0.0, False, False, False, False, 20.0)
    boundary = StepMetrics(4.0, 1.5, 2.0, False, False, False, False, 9.0)
    extreme = StepMetrics(0.2, 0.2, 12.0, False, True, True, True, 3.0)
    assert continuous_risk_target(safe, stats) < continuous_risk_target(boundary, stats)
    assert continuous_risk_target(boundary, stats) < continuous_risk_target(extreme, stats)


def test_route_file_uses_harder_traffic_distribution():
    route_file = Path("scenarios/highway_merge/highway_merge.rou.xml")
    root = ET.parse(route_file).getroot()
    vtypes = {item.attrib["id"]: item.attrib for item in root.findall("vType")}
    assert float(vtypes["car_main"]["sigma"]) == pytest.approx(0.48)
    assert float(vtypes["car_ramp"]["sigma"]) == pytest.approx(0.50)

    flows = {item.attrib["id"]: item.attrib for item in root.findall("flow")}
    assert int(flows["main_flow_left"]["vehsPerHour"]) == 900
    assert int(flows["main_flow_mid"]["vehsPerHour"]) == 1150
    assert int(flows["main_flow_right"]["vehsPerHour"]) == 1350
    assert int(flows["ramp_flow"]["vehsPerHour"]) == 650
    assert flows["main_flow_right"]["departLane"] == "0"

    vehicles = {item.attrib["id"]: item.attrib for item in root.findall("vehicle")}
    assert vehicles["ego"]["route"] == "route_ramp"
    target_lane_seeds = [
        vehicle
        for vehicle in vehicles.values()
        if vehicle["route"] == "route_main" and vehicle["departLane"] == "0"
    ]
    assert len(target_lane_seeds) >= 3


def test_scenario_snapshot_writes_hash_manifest(tmp_path):
    cfg = load_config()
    manifest = snapshot_scenario(cfg, tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    names = {item["name"] for item in payload["files"]}
    assert "highway_merge.net.xml" in names
    assert "highway_merge.rou.xml" in names
    assert all(len(item["sha256"]) == 64 for item in payload["files"])


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


def test_stage1_merge_heuristic_uses_configured_right_onramp_direction():
    cfg = load_config()
    ego = VehicleState("ego", 400.0, 53.8, 0.0, 20.0, 0, "main_aux_0", 100.0, "main_aux")
    action = decode_action(_merge_heuristic_action(cfg, {"ego": ego, "vehicles": [ego]}))
    assert action.lateral_cmd == 1
    assert action.name == "left_accelerate"


def test_target_lane_front_rear_gap_uses_edge_specific_target_lane():
    cfg = load_config()
    ego = VehicleState("ego", 200.0, 20.0, 0.0, 20.0, 0, "ramp_0", 100.0, "ramp_in")
    front = VehicleState("front", 215.0, 57.0, 0.0, 18.0, 0, "main_0", 215.0, "main_in")
    rear = VehicleState("rear", 190.0, 57.0, 0.0, 22.0, 0, "main_0", 190.0, "main_in")
    other_lane = VehicleState("other", 202.0, 60.2, 0.0, 18.0, 1, "main_1", 202.0, "main_in")
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


def test_heuristic_risk_fallback_matches_configured_type_count_and_taper_miss():
    cfg = load_config()
    ego = VehicleState("ego", 500.0, 53.8, 0.0, 10.0, 0, "main_aux_0", 190.0, "main_aux")
    context = {"ego": ego, "vehicles": [ego], "lane_count": 4, "config": cfg}
    prediction = RiskModuleWrapper(cfg).predict(decode_action(4), context)
    assert prediction.risk_type_logits.shape == (cfg.risk_module.risk_type_count,)
    assert prediction.risk_type_logits[5] == pytest.approx(1.0)


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
    ego = VehicleState("ego", 0.0, 57.0, 0.0, 0.0, 0, "main_0", 0.0, "main_in")
    trajectories = np.zeros((3, 4, 5), dtype=np.float32)
    trajectories[0, :, :2] = np.asarray([20.0, 57.0], dtype=np.float32)
    trajectories[1, :, :2] = np.asarray([-12.0, 57.0], dtype=np.float32)
    trajectories[2, :, :2] = np.asarray([2.0, 63.4], dtype=np.float32)
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
    ego = VehicleState("ego", 0.0, 57.0, 0.0, 0.0, 0, "main_0", 0.0, "main_in")
    trajectories = np.zeros((3, 4, 5), dtype=np.float32)
    trajectories[0, :, :2] = np.asarray([20.0, 57.0], dtype=np.float32)
    trajectories[1, :, :2] = np.asarray([-12.0, 57.0], dtype=np.float32)
    trajectories[2, :, :2] = np.asarray([2.0, 63.4], dtype=np.float32)
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


def test_forecast_diagnostics_vector_to_state_uses_generated_auxiliary_geometry():
    cfg = load_config()
    state = _vector_to_state(
        "aux",
        np.asarray([310.48, 53.8, 0.0, 10.0, 0.0], dtype=np.float32),
        cfg,
        lane_index=0,
        edge_role_id=EDGE_ROLE_AUXILIARY,
    )
    assert state.edge_id == "main_aux"
    assert state.lane_pos == pytest.approx(8.98, abs=0.10)


def test_forecast_diagnostics_cv_rollout_projects_ramp_vehicle_to_auxiliary_lane():
    cfg = load_config()
    rollout = diagnostics_constant_velocity_future(
        np.asarray([298.0, 53.8, 0.0, 20.0, 0.0], dtype=np.float32),
        1,
        0.5,
        cfg,
        lane_index=0,
        edge_role_id=EDGE_ROLE_RAMP,
    )
    assert rollout[0, 0] > 301.0
    assert rollout[0, 1] == pytest.approx(53.8)


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


def test_forecast_conclusion_requires_wcdt_v2_front_rear_gap_quality():
    report = {
        "cv_prediction": {
            "ade": {"mean": 2.0},
            "fde": {"mean": 4.0},
            "future_min_distance_abs_error": {"mean": 2.0},
            "target_lane_front_gap_abs_error": {"mean": 2.0},
            "target_lane_rear_gap_abs_error": {"mean": 2.0},
        },
        "wcdt_v2_prediction": {
            "available": True,
            "fde": {"mean": 3.0},
            "future_min_distance_abs_error": {"mean": 1.0},
            "target_lane_front_gap_abs_error": {"mean": 1.0},
            "target_lane_rear_gap_abs_error": {"mean": 1.0},
            "uncertainty": {"std": 0.10},
            "uncertainty_fde_correlation": 0.0,
            "uncertainty_future_min_distance_abs_error_correlation": 0.20,
        },
    }
    conclusion = _forecast_conclusion(report)
    assert conclusion["wcdt_v2_prediction_quality_pass"]
    assert conclusion["wcdt_v2_uncertainty_quality_pass"]
    assert conclusion["wcdt_v2_recommended_for_stage5"]

    report["wcdt_v2_prediction"]["target_lane_rear_gap_abs_error"]["mean"] = 3.0
    conclusion = _forecast_conclusion(report)
    assert not conclusion["wcdt_v2_prediction_quality_pass"]
    assert not conclusion["wcdt_v2_recommended_for_stage5"]


def test_forecast_conclusion_marks_v3_candidate_only_when_it_beats_v2():
    report = {
        "cv_prediction": {
            "ade": {"mean": 5.0},
            "fde": {"mean": 6.0},
            "future_min_distance_abs_error": {"mean": 3.0},
            "target_lane_front_gap_abs_error": {"mean": 3.0},
            "target_lane_rear_gap_abs_error": {"mean": 3.0},
        },
        "wcdt_v2_prediction": {
            "available": True,
            "fde": {"mean": 4.0},
            "future_min_distance_abs_error": {"mean": 2.0},
            "target_lane_front_gap_abs_error": {"mean": 2.0},
            "target_lane_rear_gap_abs_error": {"mean": 2.0},
            "uncertainty": {"std": 0.1},
            "uncertainty_fde_correlation": 0.2,
        },
        "wcdt_v3_prediction": {
            "available": True,
            "fde": {"mean": 3.5},
            "future_min_distance_abs_error": {"mean": 1.5},
            "target_lane_front_gap_abs_error": {"mean": 1.5},
            "target_lane_rear_gap_abs_error": {"mean": 1.5},
            "uncertainty": {"std": 0.1},
            "uncertainty_fde_correlation": 0.2,
        },
    }
    conclusion = _forecast_conclusion(report)
    assert conclusion["wcdt_v3_prediction_quality_pass"]
    assert conclusion["wcdt_v3_uncertainty_quality_pass"]
    assert conclusion["wcdt_v3_candidate_for_promotion"]


def test_forecast_conclusion_allows_v3_when_both_models_lack_rear_gap_samples():
    report = {
        "cv_prediction": {
            "ade": {"mean": 5.0},
            "fde": {"mean": 6.0},
            "future_min_distance_abs_error": {"mean": 3.0},
        },
        "wcdt_v2_prediction": {
            "available": True,
            "fde": {"mean": 4.0},
            "future_min_distance_abs_error": {"mean": 2.0},
            "target_lane_front_gap_abs_error": {"mean": 2.0},
            "target_lane_rear_gap_abs_error": {"count": 0},
        },
        "wcdt_v3_prediction": {
            "available": True,
            "fde": {"mean": 4.0},
            "future_min_distance_abs_error": {"mean": 2.0},
            "target_lane_front_gap_abs_error": {"mean": 2.0},
            "target_lane_rear_gap_abs_error": {"count": 0},
            "uncertainty": {"std": 0.1},
            "uncertainty_fde_correlation": 0.2,
        },
    }
    conclusion = _forecast_conclusion(report)
    assert conclusion["wcdt_v3_prediction_quality_pass"]
    assert conclusion["wcdt_v3_candidate_for_promotion"]


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


def _shield_context(min_distance: float = 5.0, min_ttc: float = 5.0):
    return {
        "current_metrics": StepMetrics(
            min_distance=min_distance,
            min_ttc=min_ttc,
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


def _merge_timing_context(cfg=None, *, front_gap=20.0, rear_gap=20.0, distance_to_taper=30.0):
    cfg = cfg or load_config()
    ego = VehicleState("ego", 490.0, 53.8, 0.0, 20.0, 0, "main_aux_0", 190.0, "main_aux")
    return {
        "ego": ego,
        "vehicles": [ego],
        "lane_count": 4,
        "config": cfg,
        "merge_local": SimpleNamespace(
            target_front_gap=front_gap,
            target_rear_gap=rear_gap,
            merge_distance=distance_to_taper,
        ),
    }


def test_merge_timing_reward_penalty_scales_with_deadline_urgency():
    cfg = load_config()
    cfg.rl["reward_profile"] = "merge_timing_forecast"
    cfg.rl["merge_timing_reward"]["deadline_distance"] = 120.0
    cfg.rl["merge_timing_reward"]["consecutive_missed_grace"] = 0
    env = SumoHighwayMergeEnv(cfg, seed=1, reward_risk_model=_StaticRiskModel({4: 0.10}))
    far, far_debug = env._merge_timing_reward_adjustment(
        _merge_timing_context(cfg, distance_to_taper=110.0)["ego"],
        "",
        decode_action(5),
        _merge_timing_context(cfg, distance_to_taper=110.0),
    )
    near, near_debug = env._merge_timing_reward_adjustment(
        _merge_timing_context(cfg, distance_to_taper=30.0)["ego"],
        "",
        decode_action(5),
        _merge_timing_context(cfg, distance_to_taper=30.0),
    )
    assert far_debug["task_missed_merge"]
    assert near_debug["task_deadline_urgency"] > far_debug["task_deadline_urgency"]
    assert near < far


def test_merge_timing_reward_respects_missed_opportunity_grace():
    cfg = load_config()
    cfg.rl["merge_timing_reward"]["consecutive_missed_grace"] = 2
    env = SumoHighwayMergeEnv(cfg, seed=1, reward_risk_model=_StaticRiskModel({4: 0.10}))
    context = _merge_timing_context(cfg, distance_to_taper=30.0)
    env._record_merge_opportunity(context, decode_action(5))
    penalty, debug = env._merge_timing_reward_adjustment(context["ego"], "", decode_action(5), context)
    assert debug["task_consecutive_missed_count"] == 1
    assert debug["merge_timing_missed_penalty"] == pytest.approx(0.0)
    assert penalty < 0.0  # deadline penalty still applies near taper.


def test_merge_timing_reward_does_not_bonus_unsafe_gap():
    cfg = load_config()
    env = SumoHighwayMergeEnv(cfg, seed=1, reward_risk_model=_StaticRiskModel({4: 0.10}))
    context = _merge_timing_context(cfg, front_gap=3.0, rear_gap=3.0, distance_to_taper=80.0)
    penalty, debug = env._merge_timing_reward_adjustment(context["ego"], "", decode_action(5), context)
    assert not debug["task_merge_opportunity"]
    assert debug["merge_timing_early_safe_merge_bonus"] == pytest.approx(0.0)
    assert debug["merge_timing_missed_penalty"] == pytest.approx(0.0)


def test_merge_timing_reward_penalizes_taper_miss_terminal():
    cfg = load_config()
    env = SumoHighwayMergeEnv(cfg, seed=1, reward_risk_model=_StaticRiskModel({4: 0.10}))
    penalty, debug = env._merge_timing_reward_adjustment(None, "taper_miss", decode_action(4), None)
    assert penalty <= -35.0
    assert debug["merge_timing_taper_miss_penalty"] == pytest.approx(-35.0)


def test_forecast_aware_task_scorer_recommends_merge_near_deadline_with_safe_gap():
    cfg = load_config()
    context = _merge_timing_context(cfg, front_gap=30.0, rear_gap=30.0, distance_to_taper=20.0)
    scorer = ForecastAwareTaskScorer(cfg)
    result = scorer.score(context, decode_action(5), merge_cmd=1, deadline_distance=120.0, urgency=0.8)
    assert result["forecast_aware_available"]
    assert str(result["forecast_aware_best_action_name"]).startswith("left_")
    assert result["forecast_aware_would_merge"]


def test_forecast_aware_task_scorer_blocks_merge_when_gap_is_unsafe():
    cfg = load_config()
    context = _merge_timing_context(cfg, front_gap=2.0, rear_gap=2.0, distance_to_taper=20.0)
    scorer = ForecastAwareTaskScorer(cfg)
    result = scorer.score(context, decode_action(5), merge_cmd=1, deadline_distance=120.0, urgency=0.8)
    assert result["forecast_aware_available"]
    assert not result["forecast_aware_would_merge"]
    assert float(result["forecast_aware_safety_risk"]) > cfg.shield.task_backstop_safety_risk_threshold


def test_task_backstop_requires_consecutive_shadow_and_counts_separately():
    cfg = load_config()
    cfg.shield["enabled"] = True
    cfg.shield["task_backstop_enabled"] = True
    cfg.shield["task_backstop_consecutive_steps"] = 2
    env = SumoHighwayMergeEnv(cfg, seed=1, shield=SafetyShield(cfg, _StaticRiskModel({4: 0.1, 5: 0.1, 7: 0.1})))
    context = _merge_timing_context(cfg, front_gap=30.0, rear_gap=30.0, distance_to_taper=20.0)
    raw = decode_action(5)
    env._record_merge_opportunity(context, raw)
    first = env._maybe_task_backstop(raw, raw, context, {"raw_action": raw.index, "final_action": raw.index})
    assert first is None
    env._record_merge_opportunity(context, raw)
    second = env._maybe_task_backstop(raw, raw, context, {"raw_action": raw.index, "final_action": raw.index})
    assert second is not None
    assert second["replacement_reason"] == "task_backstop"
    assert str(second["final_action_name"]).startswith("left_")
    env._task_replacements.append(second)
    env._interventions.append({"raw_action": raw.index, "final_action": raw.index, "replacement_reason": "raw_safe"})
    report = env.episode_report()
    assert report["task_replacement_count"] == 1
    assert report["actual_replacement_count"] == 0


def test_step_safety_record_includes_forecast_task_trace_fields():
    record = _step_safety_record(
        step_index=3,
        raw_action=5,
        final_action=5,
        reward=0.0,
        terminated=False,
        truncated=False,
        collision_threshold=0.25,
        shield_enabled=True,
        info={
            "step": 3,
            "raw_action_name": "keep_accelerate",
            "final_action_name": "keep_accelerate",
            "ego_edge": "main_aux",
            "ego_lane": 0,
            "forecast_aware_best_action_name": "left_hold",
            "forecast_aware_best_task_risk": 0.2,
            "forecast_aware_would_merge": True,
            "task_replacement": False,
        },
    )
    assert record["ego_edge"] == "main_aux"
    assert record["ego_lane"] == 0
    assert record["forecast_aware_best_action_name"] == "left_hold"
    assert record["forecast_aware_would_merge"]


def test_shield_keeps_raw_action_below_activation_threshold():
    cfg = _shield_cfg()
    shield = SafetyShield(cfg, _StaticRiskModel({4: 0.50}))
    raw = decode_action(4)
    final, record = shield.select_action(raw, _shield_context())
    assert final.index == raw.index
    assert record["replacement_reason"] == "raw_safe"
    assert not record["fallback"]
    assert not record["emergency_fallback"]
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
    assert not record["emergency_trigger"]


def test_shield_emergency_fallback_on_extreme_physical_risk():
    cfg = _shield_cfg()
    shield = SafetyShield(cfg, _StaticRiskModel({4: 0.10, 3: 0.20, 5: 0.20}))
    raw = decode_action(4)
    final, record = shield.select_action(raw, _shield_context(min_distance=0.8, min_ttc=5.0))
    assert final.name == "keep_decelerate"
    assert record["replacement_reason"] == "emergency_fallback"
    assert record["emergency_fallback"]
    assert record["emergency_trigger"]
    assert record["emergency_reason"] == "min_distance"
    assert not record["fallback"]


def test_shield_emergency_fallback_on_saturated_risk_watch_zone():
    cfg = _shield_cfg()
    shield = SafetyShield(cfg, _StaticRiskModel({index: 1.0 for index in range(9)}))
    raw = decode_action(4)
    final, record = shield.select_action(raw, _shield_context(min_distance=1.5, min_ttc=5.0))
    assert final.name == "keep_decelerate"
    assert record["replacement_reason"] == "emergency_fallback"
    assert record["emergency_reason"] == "saturated_risk_watch_zone"
    assert record["best_candidate_risk"] == pytest.approx(1.0)


def test_shield_single_saturated_risk_step_does_not_trigger_consecutive_emergency():
    cfg = _shield_cfg()
    cfg.shield["emergency_saturated_consecutive_enabled"] = True
    shield = SafetyShield(cfg, _StaticRiskModel({index: 1.0 for index in range(9)}))
    raw = decode_action(4)
    final, record = shield.select_action(raw, _shield_context(min_distance=5.0, min_ttc=5.0))
    assert final.index == raw.index
    assert record["replacement_reason"] == "fallback_disabled"
    assert record["emergency_saturated_count"] == 1
    assert record["emergency_saturated_required"] == 2
    assert not record["emergency_fallback"]


def test_shield_consecutive_saturated_emergency_is_disabled_by_default():
    cfg = _shield_cfg()
    assert not cfg.shield.emergency_saturated_consecutive_enabled
    shield = SafetyShield(cfg, _StaticRiskModel({index: 1.0 for index in range(9)}))
    raw = decode_action(4)
    shield.select_action(raw, _shield_context(min_distance=5.0, min_ttc=5.0))
    final, record = shield.select_action(raw, _shield_context(min_distance=5.0, min_ttc=5.0))
    assert final.index == raw.index
    assert record["replacement_reason"] == "fallback_disabled"
    assert record["emergency_saturated_count"] == 0


def test_shield_consecutive_saturated_risk_triggers_emergency():
    cfg = _shield_cfg()
    cfg.shield["emergency_saturated_consecutive_enabled"] = True
    shield = SafetyShield(cfg, _StaticRiskModel({index: 1.0 for index in range(9)}))
    raw = decode_action(4)
    _first, first_record = shield.select_action(raw, _shield_context(min_distance=5.0, min_ttc=5.0))
    final, record = shield.select_action(raw, _shield_context(min_distance=5.0, min_ttc=5.0))
    assert first_record["replacement_reason"] == "fallback_disabled"
    assert final.name == "keep_decelerate"
    assert record["replacement_reason"] == "emergency_fallback"
    assert record["emergency_reason"] == "saturated_risk_consecutive"
    assert record["emergency_saturated_count"] == 2
    assert record["emergency_saturated_required"] == 2


def test_shield_saturated_counter_resets_when_risk_drops():
    cfg = _shield_cfg()
    cfg.shield["emergency_saturated_consecutive_enabled"] = True
    risk_model = _StaticRiskModel({index: 1.0 for index in range(9)})
    shield = SafetyShield(cfg, risk_model)
    raw = decode_action(4)
    _first, first_record = shield.select_action(raw, _shield_context(min_distance=5.0, min_ttc=5.0))
    risk_model.scores = {4: 0.95, 5: 0.83}
    final, record = shield.select_action(raw, _shield_context(min_distance=5.0, min_ttc=5.0))
    assert first_record["emergency_saturated_count"] == 1
    assert final.index == raw.index
    assert record["replacement_reason"] == "fallback_disabled"
    assert record["emergency_saturated_count"] == 0


def test_shield_saturated_counter_resets_after_replacement():
    cfg = _shield_cfg()
    cfg.shield["emergency_saturated_consecutive_enabled"] = True
    risk_model = _StaticRiskModel({index: 1.0 for index in range(9)})
    shield = SafetyShield(cfg, risk_model)
    raw = decode_action(4)
    _first, first_record = shield.select_action(raw, _shield_context(min_distance=5.0, min_ttc=5.0))
    risk_model.scores = {4: 0.95, 5: 0.40}
    final, record = shield.select_action(raw, _shield_context(min_distance=5.0, min_ttc=5.0))
    assert first_record["emergency_saturated_count"] == 1
    assert final.index == 5
    assert record["replacement_reason"] == "replacement"
    assert record["emergency_saturated_count"] == 0


def test_shield_reset_episode_state_clears_saturated_counter():
    cfg = _shield_cfg()
    cfg.shield["emergency_saturated_consecutive_enabled"] = True
    shield = SafetyShield(cfg, _StaticRiskModel({index: 1.0 for index in range(9)}))
    shield.select_action(decode_action(4), _shield_context(min_distance=5.0, min_ttc=5.0))
    assert shield._emergency_saturated_count == 1
    shield.reset_episode_state()
    assert shield._emergency_saturated_count == 0


def test_shield_emergency_action_must_be_legal():
    cfg = _shield_cfg()
    cfg.shield["emergency_actions"] = ["left_decelerate", "keep_hold"]
    context = _shield_context_with_ramp_ego()
    context["config"] = cfg
    context["current_metrics"] = _shield_context(min_distance=0.8)["current_metrics"]
    shield = SafetyShield(cfg, _StaticRiskModel({index: 1.0 for index in range(9)}))
    final, record = shield.select_action(decode_action(5), context)
    assert final.name == "keep_hold"
    assert record["replacement_reason"] == "emergency_fallback"
    assert record["final_candidate_legal"]


def test_shield_records_emergency_unavailable_when_no_legal_backstop():
    cfg = _shield_cfg()
    context = _shield_context_with_ramp_ego()
    context["config"] = cfg
    context["lane_count"] = 0
    context["current_metrics"] = _shield_context(min_distance=0.8)["current_metrics"]
    shield = SafetyShield(cfg, _StaticRiskModel({index: 1.0 for index in range(9)}))
    raw = decode_action(5)
    final, record = shield.select_action(raw, context)
    assert final.index == raw.index
    assert record["replacement_reason"] == "emergency_unavailable"
    assert record["emergency_trigger"]
    assert not record["emergency_fallback"]


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
    history[0, :, 1] = 20.0
    history[1, :, 0] = 212.0
    history[1, :, 1] = 57.0
    history[2, :, 0] = 190.0
    history[2, :, 1] = 57.0
    history[3, :, 0] = 205.0
    history[3, :, 1] = 20.0
    history[4, :, 0] = 201.0
    history[4, :, 1] = 60.2
    history[5, :, 0] = 260.0
    history[5, :, 1] = 63.4
    ordered = _ordered_prediction_indices(cfg, history, mask)
    assert ordered[:3] == [1, 2, 3]


def test_stage2_legacy_wcdt_order_uses_serialized_edge_roles_for_auxiliary_lane():
    cfg = load_config()
    history = np.zeros((4, cfg.scenario.history_steps, 5), dtype=np.float32)
    mask = np.ones((4,), dtype=np.float32)
    history[:, :, 3] = 20.0
    history[0, :, :2] = [400.0, 53.8]
    history[1, :, :2] = [414.0, 57.0]
    history[2, :, :2] = [402.0, 53.8]
    history[3, :, :2] = [401.0, 60.2]
    lane_indices = np.asarray([0, 1, 0, 2], dtype=np.int64)
    edge_roles = np.asarray([EDGE_ROLE_AUXILIARY, EDGE_ROLE_TARGET, EDGE_ROLE_AUXILIARY, EDGE_ROLE_MAINLINE])
    ordered = _ordered_prediction_indices(cfg, history, mask, lane_indices, edge_roles)
    assert ordered == [1, 2, 3]


def test_runtime_wcdt_adapter_prioritizes_target_lane_front_rear_and_ramp():
    cfg = load_config()
    history = HistoryBuffer(cfg.scenario.history_steps, max_agents=6)
    states = [
        VehicleState("ego", 200.0, 20.0, 0.0, 20.0, 0, "ramp_0", 100.0, "ramp_in"),
        VehicleState("target_front", 214.0, 57.0, 0.0, 20.0, 0, "main_0", 214.0, "main_in"),
        VehicleState("target_rear", 190.0, 57.0, 0.0, 20.0, 0, "main_0", 190.0, "main_in"),
        VehicleState("ramp_front", 208.0, 20.0, 0.0, 18.0, 0, "ramp_0", 108.0, "ramp_in"),
        VehicleState("other_lane", 202.0, 60.2, 0.0, 20.0, 1, "main_1", 202.0, "main_in"),
    ]
    for _ in range(cfg.scenario.history_steps):
        history.append(states)
    ordered = SumoWcDTAdapter(cfg)._ordered_agent_ids(history, "ego")
    assert ordered[:3] == ["target_front", "target_rear", "ramp_front"]


def test_runtime_wcdt_adapter_keeps_auxiliary_neighbor_local_past_merge_x_fallback():
    cfg = load_config()
    history = HistoryBuffer(cfg.scenario.history_steps, max_agents=4)
    states = [
        VehicleState("ego", 400.0, 53.8, 0.0, 20.0, 0, "main_aux_0", 90.0, "main_aux"),
        VehicleState("aux_front", 600.0, 53.8, 0.0, 18.0, 0, "main_aux_0", 110.0, "main_aux"),
        VehicleState("other_lane", 402.0, 60.2, 0.0, 20.0, 2, "main_aux_2", 92.0, "main_aux"),
    ]
    for _ in range(cfg.scenario.history_steps):
        history.append(states)
    ordered = SumoWcDTAdapter(cfg)._ordered_agent_ids(history, "ego")
    assert ordered[0] == "aux_front"


def test_history_buffer_marks_imputed_history_steps_invalid():
    cfg = load_config()
    history = HistoryBuffer(history_steps=3, max_agents=2)
    ego = VehicleState("ego", 100.0, 53.8, 0.0, 10.0, 0, "main_aux_0", 10.0, "main_aux")
    history.append([])
    history.append([ego])
    history.append([ego])
    arrays = history.to_tensor_arrays_with_metadata("ego", cfg)
    assert arrays["mask"].tolist() == [1.0, 0.0]
    assert arrays["history_valid_mask"][0].tolist() == [0.0, 1.0, 1.0]
    assert arrays["history_lane_index"][0].tolist() == [-1, 0, 0]
    assert arrays["history_edge_role"][0, 0] == 0


def test_trajectory_window_future_missing_state_is_zero_and_masked():
    cfg = load_config()
    env = SumoHighwayMergeEnv.__new__(SumoHighwayMergeEnv)
    env.history_steps = 2
    env.top_k = 1
    env.ego_id = "ego"
    env.config = cfg
    ego = VehicleState("ego", 100.0, 53.8, 0.0, 10.0, 0, "main_aux_0", 10.0, "main_aux")
    actor = VehicleState("actor", 110.0, 57.0, 0.0, 10.0, 1, "main_aux_1", 20.0, "main_aux")
    horizon = int(cfg.scenario.forecast_horizon_steps)
    frames = [
        {"ego": ego, "actor": actor},
        {"ego": ego, "actor": actor},
        {"ego": ego, "actor": actor},
    ]
    frames.extend({"ego": ego} for _ in range(horizon))
    env._trajectory_frames = frames
    (
        _history,
        future,
        mask,
        _lane_index,
        _edge_role,
        _history_valid_mask,
        future_valid_mask,
        _history_lane_index,
        _history_edge_role,
        _future_lane_index,
        _future_edge_role,
    ) = env.trajectory_window_samples()
    assert mask[0, 1] == 1.0
    assert future_valid_mask[0, 1, 0] == 1.0
    assert future_valid_mask[0, 1, 1] == 0.0
    assert np.count_nonzero(future[0, 1, 1:]) == 0


def test_wcdt_v2_actor_selection_prioritizes_merge_local_agents():
    cfg = load_config()
    history = np.zeros((6, cfg.scenario.history_steps, 5), dtype=np.float32)
    mask = np.ones((6,), dtype=np.float32)
    history[:, :, 3] = 20.0
    history[0, :, 0] = 200.0
    history[0, :, 1] = 20.0
    history[1, :, 0] = 214.0
    history[1, :, 1] = 57.0
    history[2, :, 0] = 190.0
    history[2, :, 1] = 57.0
    history[3, :, 0] = 208.0
    history[3, :, 1] = 20.0
    history[4, :, 0] = 202.0
    history[4, :, 1] = 60.2
    history[5, :, 0] = 260.0
    history[5, :, 1] = 63.4
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
    history[:, 0, :, 1] = 20.0
    history[:, 1, :, 0] = 212.0
    history[:, 1, :, 1] = 57.0
    history[:, 2, :, 0] = 190.0
    history[:, 2, :, 1] = 57.0
    history[:, 3, :, 0] = 208.0
    history[:, 3, :, 1] = 20.0
    history[:, 4, :, 0] = 240.0
    history[:, 4, :, 1] = 60.2
    batch = build_v2_numpy_batch(cfg, history, future, mask, np.asarray([0, 1], dtype=np.int64))
    assert batch["features"].shape == (2, 3, WCDT_V2_INPUT_DIM)
    assert batch["baseline"].shape == (2, 3, cfg.scenario.forecast_horizon_steps, 5)
    assert batch["target"].shape == (2, 3, cfg.scenario.forecast_horizon_steps, 5)
    assert batch["mask"].shape == (2, 3)
    assert np.all(np.isfinite(batch["features"]))
    assert batch["selected_indices"][0].tolist() == [1, 2, 3]
    assert batch["baseline"][0, 0, 0, 0] > history[0, 1, -1, 0]


def test_wcdt_v2_loss_uses_future_minimum_distance_not_mean_gap_error():
    torch = pytest.importorskip("torch")
    target = torch.tensor([[[[5.0, 0.0], [5.0, 0.0], [5.0, 0.0]]]], dtype=torch.float32)
    pred = torch.tensor([[[[8.0, 0.0], [5.0, 0.0], [5.0, 0.0]]]], dtype=torch.float32)
    mask = torch.ones((1, 1), dtype=torch.float32)
    ego_future = torch.zeros((1, 3, 2), dtype=torch.float32)
    role_ids = torch.zeros((1, 1), dtype=torch.long)
    _total, components = v2_loss(pred, target, mask, ego_future, role_ids)
    assert components["ade"].item() > 0.0
    assert components["future_min_distance"].item() == pytest.approx(0.0)


def test_wcdt_v2_loss_role_gap_and_smoothness_components_are_safe():
    torch = pytest.importorskip("torch")
    ego_future = torch.zeros((1, 4, 2), dtype=torch.float32)
    target = torch.tensor(
        [[[[5.0, 0.0], [6.0, 0.0], [7.0, 0.0], [8.0, 0.0]]]],
        dtype=torch.float32,
    )
    smooth = target.clone()
    jitter = target.clone()
    jitter[0, 0, 2, 0] += 4.0
    mask = torch.ones((1, 1), dtype=torch.float32)
    front_role = torch.zeros((1, 1), dtype=torch.long)
    _total, smooth_components = v2_loss(smooth, target, mask, ego_future, front_role)
    _total, jitter_components = v2_loss(jitter, target, mask, ego_future, front_role)
    assert smooth_components["target_lane_rear_gap"].item() == pytest.approx(0.0)
    assert jitter_components["target_lane_front_gap"].item() > 0.0
    assert jitter_components["smoothness"].item() > smooth_components["smoothness"].item()


def test_wcdt_v2_loss_ignores_invalid_future_tail_and_uses_last_valid_fde():
    torch = pytest.importorskip("torch")
    target = torch.tensor([[[[5.0, 0.0], [6.0, 0.0], [0.0, 0.0]]]], dtype=torch.float32)
    pred = target.clone()
    pred[0, 0, 2, 0] = 1000.0
    mask = torch.ones((1, 1), dtype=torch.float32)
    future_valid_mask = torch.tensor([[[1.0, 1.0, 0.0]]], dtype=torch.float32)
    ego_valid_mask = torch.ones((1, 3), dtype=torch.float32)
    ego_future = torch.zeros((1, 3, 2), dtype=torch.float32)
    role_ids = torch.zeros((1, 1), dtype=torch.long)
    _total, components = v2_loss(
        pred,
        target,
        mask,
        ego_future,
        role_ids,
        future_valid_mask=future_valid_mask,
        ego_future_valid_mask=ego_valid_mask,
    )
    assert components["ade"].item() == pytest.approx(0.0)
    assert components["fde"].item() == pytest.approx(0.0)
    assert components["future_min_distance"].item() == pytest.approx(0.0)


def _wcdt_v3_test_batch():
    cfg = load_config()
    cfg.prediction["wcdt_v3_max_agents"] = 3
    history = np.zeros((1, 5, cfg.scenario.history_steps, 5), dtype=np.float32)
    future = np.zeros((1, 5, cfg.scenario.forecast_horizon_steps, 5), dtype=np.float32)
    mask = np.ones((1, 5), dtype=np.float32)
    history[..., 3] = 10.0
    history[0, 0, :, 0] = np.linspace(190.0, 200.0, cfg.scenario.history_steps)
    history[0, 0, :, 1] = 20.0
    history[0, 1, :, 0] = np.linspace(202.0, 212.0, cfg.scenario.history_steps)
    history[0, 1, :, 1] = 57.0
    history[0, 2, :, 0] = np.linspace(180.0, 190.0, cfg.scenario.history_steps)
    history[0, 2, :, 1] = 57.0
    history[0, 3, :, 0] = np.linspace(198.0, 208.0, cfg.scenario.history_steps)
    history[0, 3, :, 1] = 20.0
    history[0, 4, :, 0] = np.linspace(230.0, 240.0, cfg.scenario.history_steps)
    history[0, 4, :, 1] = 60.2
    batch = build_v3_numpy_batch(cfg, history, future, mask, np.asarray([0], dtype=np.int64))
    return cfg, history, future, mask, batch


def test_wcdt_v3_batch_uses_full_history_with_fixed_shape():
    cfg, _history, _future, _mask, batch = _wcdt_v3_test_batch()
    assert batch["history_features"].shape == (1, 3, cfg.scenario.history_steps, WCDT_V3_HISTORY_INPUT_DIM)
    assert batch["baseline"].shape == (1, 3, cfg.scenario.forecast_horizon_steps, 5)
    assert batch["target"].shape == (1, 3, cfg.scenario.forecast_horizon_steps, 5)
    assert np.all(np.isfinite(batch["history_features"]))
    assert batch["selected_indices"][0].tolist() == [1, 2, 3]


def test_wcdt_v3_batch_preserves_timestep_masks_and_route_metadata():
    cfg, history, future, mask, _batch = _wcdt_v3_test_batch()
    history_valid = np.ones(history.shape[:3], dtype=np.float32)
    future_valid = np.ones(future.shape[:3], dtype=np.float32)
    history_lane_index = np.zeros(history.shape[:3], dtype=np.int64)
    history_edge_role = np.zeros(history.shape[:3], dtype=np.int64)
    history_valid[0, 1, 0] = 0.0
    history_lane_index[0, 1, 1:] = 1
    history_edge_role[0, 1, 1:] = EDGE_ROLE_TARGET
    batch = build_v3_numpy_batch(
        cfg,
        history,
        future,
        mask,
        np.asarray([0], dtype=np.int64),
        history_valid_mask=history_valid,
        future_valid_mask=future_valid,
        history_lane_indices=history_lane_index,
        history_edge_roles=history_edge_role,
    )
    assert batch["history_valid_mask"][0, 0, 0] == 0.0
    assert batch["history_lane_ids"][0, 0, 1] == 2  # embedding id reserves 0 for unknown/padding
    assert batch["history_edge_role_ids"][0, 0, 1] == EDGE_ROLE_TARGET


def test_wcdt_v3_output_changes_when_history_route_metadata_changes():
    torch = pytest.importorskip("torch")
    cfg, history, future, mask, _batch = _wcdt_v3_test_batch()
    history_lane_index = np.zeros(history.shape[:3], dtype=np.int64)
    changed_lane_index = history_lane_index.copy()
    changed_lane_index[0, 1, :-1] = 1
    kwargs = {
        "history_valid_mask": np.ones(history.shape[:3], dtype=np.float32),
        "future_valid_mask": np.ones(future.shape[:3], dtype=np.float32),
        "history_edge_roles": np.zeros(history.shape[:3], dtype=np.int64),
    }
    original = build_v3_numpy_batch(
        cfg,
        history,
        future,
        mask,
        np.asarray([0], dtype=np.int64),
        history_lane_indices=history_lane_index,
        **kwargs,
    )
    changed = build_v3_numpy_batch(
        cfg,
        history,
        future,
        mask,
        np.asarray([0], dtype=np.int64),
        history_lane_indices=changed_lane_index,
        **kwargs,
    )
    model = WcDTV3TemporalInteractionPredictor(
        history_steps=int(cfg.scenario.history_steps),
        horizon_steps=int(cfg.scenario.forecast_horizon_steps),
        hidden_dim=32,
        temporal_layers=1,
        actor_attention_layers=1,
        num_heads=4,
        dropout=0.0,
    ).eval()
    with torch.no_grad():
        original_pred = predict_v3_model(model, tensorize_v3_batch(original, torch, torch.device("cpu")))
        changed_pred = predict_v3_model(model, tensorize_v3_batch(changed, torch, torch.device("cpu")))
    assert not torch.allclose(original_pred, changed_pred)


def test_wcdt_v3_changes_output_when_history_changes_but_last_frame_is_fixed():
    torch = pytest.importorskip("torch")
    cfg, history, future, mask, batch = _wcdt_v3_test_batch()
    changed_history = history.copy()
    changed_history[0, 1, 1:-1, 0] += 8.0
    changed = build_v3_numpy_batch(cfg, changed_history, future, mask, np.asarray([0], dtype=np.int64))
    model = WcDTV3TemporalInteractionPredictor(
        history_steps=int(cfg.scenario.history_steps),
        horizon_steps=int(cfg.scenario.forecast_horizon_steps),
        hidden_dim=32,
        temporal_layers=1,
        actor_attention_layers=1,
        num_heads=4,
        dropout=0.0,
    ).eval()
    original_tensor = tensorize_v3_batch(batch, torch, torch.device("cpu"))
    changed_tensor = tensorize_v3_batch(changed, torch, torch.device("cpu"))
    with torch.no_grad():
        original = predict_v3_model(model, original_tensor)
        modified = predict_v3_model(model, changed_tensor)
    assert not torch.allclose(original, modified)


def test_wcdt_v3_actor_attention_and_padding_are_stable():
    torch = pytest.importorskip("torch")
    cfg, _history, _future, _mask, batch = _wcdt_v3_test_batch()
    model = WcDTV3TemporalInteractionPredictor(
        history_steps=int(cfg.scenario.history_steps),
        horizon_steps=int(cfg.scenario.forecast_horizon_steps),
        hidden_dim=32,
        temporal_layers=1,
        actor_attention_layers=1,
        num_heads=4,
        dropout=0.0,
    ).eval()
    original_tensor = tensorize_v3_batch(batch, torch, torch.device("cpu"))
    modified_tensor = {key: value.clone() for key, value in original_tensor.items()}
    modified_tensor["history_features"][:, 1, :, 0] += 7.0
    with torch.no_grad():
        original = predict_v3_model(model, original_tensor)
        modified = predict_v3_model(model, modified_tensor)
    assert not torch.allclose(original[:, 0], modified[:, 0])

    empty_tensor = {key: value.clone() for key, value in original_tensor.items()}
    empty_tensor["mask"].zero_()
    with torch.no_grad():
        empty = predict_v3_model(model, empty_tensor)
    assert torch.all(torch.isfinite(empty))
    assert torch.count_nonzero(empty) == 0


def test_wcdt_v2_and_v3_share_merge_safety_loss():
    torch = pytest.importorskip("torch")
    target = torch.tensor([[[[5.0, 0.0], [6.0, 0.0], [7.0, 0.0]]]], dtype=torch.float32)
    pred = target.clone()
    pred[0, 0, 1, 0] += 1.5
    mask = torch.ones((1, 1), dtype=torch.float32)
    ego_future = torch.zeros((1, 3, 2), dtype=torch.float32)
    role_ids = torch.zeros((1, 1), dtype=torch.long)
    v2_total, v2_components = v2_loss(pred, target, mask, ego_future, role_ids)
    v3_total, v3_components = v3_loss(pred, target, mask, ego_future, role_ids)
    assert torch.allclose(v2_total, v3_total)
    assert set(v2_components) == set(v3_components)


def test_wcdt_v3_rejects_wrong_architecture_checkpoint(tmp_path):
    torch = pytest.importorskip("torch")
    cfg = load_config()
    checkpoint = tmp_path / "wrong_wcdt_v3.pt"
    torch.save({"architecture_version": "wrong_architecture", "model_state_dicts": []}, checkpoint)
    with pytest.raises(ValueError, match="architecture_version"):
        load_v3_ensemble(cfg, checkpoint, torch.device("cpu"))


def test_wcdt_v3_rejects_checkpoint_without_required_metadata(tmp_path):
    torch = pytest.importorskip("torch")
    cfg = load_config()
    checkpoint = tmp_path / "missing_metadata_wcdt_v3.pt"
    torch.save({"model_state_dicts": []}, checkpoint)
    with pytest.raises(ValueError, match="architecture_version"):
        load_v3_ensemble(cfg, checkpoint, torch.device("cpu"))


def test_wcdt_v2_early_stopping_and_validation_unavailable_behavior():
    cfg = load_config()
    disabled = _wcdt_v2_early_stopping_config(cfg, has_validation=False)
    assert not disabled["enabled"]
    assert disabled["disabled_reason"] == "validation_unavailable"

    enabled = _wcdt_v2_early_stopping_config(cfg, has_validation=True)
    stale = 0
    best = 1.0
    for epoch in range(1, 11):
        improved, stale, should_stop = _wcdt_v2_early_stopping_step(
            best_score=best,
            score=1.0,
            epoch=epoch,
            stale_epochs=stale,
            config=enabled,
        )
        assert not improved
    assert should_stop
    improved, stale, should_stop = _wcdt_v2_early_stopping_step(
        best_score=best,
        score=0.5,
        epoch=11,
        stale_epochs=stale,
        config=enabled,
    )
    assert improved
    assert stale == 0
    assert not should_stop


def test_wcdt_v3_early_stopping_disables_without_validation():
    cfg = load_config()
    disabled = _wcdt_v3_early_stopping_config(cfg, has_validation=False)
    assert not disabled["enabled"]
    assert disabled["disabled_reason"] == "validation_unavailable"
    enabled = _wcdt_v3_early_stopping_config(cfg, has_validation=True)
    assert enabled["enabled"]
    assert enabled["patience"] == 10


def test_wcdt_v2_old_checkpoint_without_metadata_is_rejected(tmp_path):
    torch = pytest.importorskip("torch")
    cfg = load_config()
    model = WcDTV2ResidualPredictor(WCDT_V2_INPUT_DIM, horizon_steps=3, hidden_dim=8)
    checkpoint = tmp_path / "legacy_wcdt_v2.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "horizon_steps": 3,
            "hidden_dim": 8,
        },
        checkpoint,
    )
    with pytest.raises(ValueError, match="architecture_version"):
        load_v2_ensemble(cfg, checkpoint, torch.device("cpu"))


def test_stage2_v2_v3_training_rejects_legacy_unmasked_trajectory_buffer(tmp_path):
    path = tmp_path / "legacy_buffer.npz"
    np.savez_compressed(path, agent_history=np.zeros((1, 1, 1, 5), dtype=np.float32))
    with np.load(path) as data:
        with pytest.raises(ValueError, match="trajectory_schema_version"):
            _require_trajectory_schema_v2(data, "WcDT v3")


def test_stage3_predictor_summary_reads_v3_member_histories(tmp_path):
    torch = pytest.importorskip("torch")
    checkpoint = tmp_path / "wcdt_v3_predictor.pt"
    torch.save(
        {
            "architecture_version": "wcdt_v3_temporal_actor_transformer_v2",
            "loss_version": MERGE_SAFETY_LOSS_VERSION,
            "trajectory_schema_version": 2,
            "ensemble_size": 1,
            "member_histories": [
                {
                    "member": 0,
                    "loss_history": [2.0, 1.0],
                    "trained_epochs": 2,
                    "best_epoch": 2,
                    "best_val_score": 1.0,
                    "stopped_early": False,
                }
            ],
        },
        checkpoint,
    )
    summary = _prediction_loss_summary_from_checkpoint(str(checkpoint))
    assert summary["architecture_version"] == "wcdt_v3_temporal_actor_transformer_v2"
    assert summary["loss_version"] == MERGE_SAFETY_LOSS_VERSION
    assert summary["trajectory_schema_version"] == 2
    assert summary["members"][0]["trained_epochs"] == 2


def test_stage3_predictor_summary_does_not_misattribute_v1_loss_to_v2_v3(tmp_path):
    torch = pytest.importorskip("torch")
    stage2_dir = tmp_path / "stage2"
    stage2_dir.mkdir()
    (stage2_dir / "stage2_training_report.json").write_text(
        json.dumps({"prediction_loss_history": [9.0, 8.0, 7.0]}),
        encoding="utf-8",
    )
    checkpoint = stage2_dir / "wcdt_v3_predictor.pt"
    torch.save(
        {
            "architecture_version": "wcdt_v3_temporal_actor_transformer_v2",
            "loss_version": MERGE_SAFETY_LOSS_VERSION,
            "trajectory_schema_version": 2,
            "ensemble_size": 1,
            "member_histories": [
                {
                    "member": 0,
                    "loss_history": [2.0, 1.0],
                    "trained_epochs": 2,
                    "best_epoch": 2,
                    "best_val_score": 1.0,
                    "stopped_early": False,
                }
            ],
        },
        checkpoint,
    )
    v3_summary = _prediction_loss_summary(str(checkpoint), "wcdt_v3")
    assert v3_summary["source"] == "checkpoint_member_histories"
    assert v3_summary["architecture_version"] == "wcdt_v3_temporal_actor_transformer_v2"

    legacy_checkpoint = stage2_dir / "wcdt_predictor.pt"
    torch.save({"loss_history": [3.0, 2.0]}, legacy_checkpoint)
    legacy_summary = _prediction_loss_summary(str(legacy_checkpoint), "wcdt")
    assert legacy_summary == {"epochs": 3, "first": 9.0, "last": 7.0, "min": 7.0}
    assert _prediction_loss_summary(None, "constant_velocity") is None


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
                "emergency_fallback_count": 0,
                "task_merge_opportunity_count": 2,
                "task_would_merge_count": 1,
                "task_missed_merge_count": 1,
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
                "emergency_fallback_count": 1,
                "task_merge_opportunity_count": 3,
                "task_would_merge_count": 2,
                "task_missed_merge_count": 2,
            },
        ]
    )
    assert metrics["shield_call_rate"] == 1.0
    assert metrics["actual_replacement_rate"] == 0.5
    assert metrics["mean_shield_calls"] == pytest.approx(3.5)
    assert metrics["mean_actual_replacements"] == pytest.approx(1.0)
    assert metrics["fallback_rate"] == pytest.approx(0.0)
    assert metrics["emergency_fallback_rate"] == pytest.approx(0.5)
    assert metrics["mean_emergency_fallbacks"] == pytest.approx(0.5)
    assert metrics["emergency_fallback_count"] == 1
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
    assert metrics["task_merge_opportunity_count"] == 5
    assert metrics["task_would_merge_count"] == 3
    assert metrics["task_would_merge_rate"] == pytest.approx(0.6)
    assert metrics["task_missed_merge_count"] == 3
    assert metrics["task_missed_merge_rate"] == pytest.approx(0.6)
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
    env._interventions = [
        {
            "raw_action": 4,
            "final_action": 3,
            "replacement_reason": "emergency_fallback",
            "risk_before": 1.0,
            "risk_after": 1.0,
            "best_candidate_risk": 1.0,
            "replacement_risk_delta": 0.0,
            "best_candidate_risk_delta": 0.0,
            "raw_candidate_legal": True,
            "final_candidate_legal": True,
            "fallback": False,
            "emergency_fallback": True,
            "emergency_trigger": True,
            "emergency_reason": "min_distance",
        }
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
    assert report["actual_replacement_count"] == 1
    assert report["fallback_count"] == 0
    assert report["emergency_fallback_count"] == 1
    assert report["emergency_fallback_rate"] == 1.0
    assert report["shield_score_records"][0]["emergency_reason"] == "min_distance"
    assert report["shield_score_records"][0]["emergency_saturated_count"] == 0
    assert report["shield_score_records"][0]["emergency_saturated_required"] == 0


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
    assert report["emergency_fallback_count"] == 0


def test_env_reset_resets_shield_episode_state(monkeypatch):
    class DummyShield:
        def __init__(self):
            self.calls = 0

        def reset_episode_state(self):
            self.calls += 1

    cfg = load_config()
    shield = DummyShield()
    env = SumoHighwayMergeEnv(cfg, seed=1, shield=shield)
    ego = VehicleState("ego", 0.0, 0.0, 0.0, 12.0, 0, "ramp_0", 10.0, "ramp_in")
    monkeypatch.setattr(env, "_close_sumo", lambda: None)
    monkeypatch.setattr(env, "_start_sumo", lambda: None)
    monkeypatch.setattr(env, "_simulation_step", lambda: None)
    monkeypatch.setattr(env, "_collect_states", lambda: [ego])
    env.reset(seed=1)
    assert shield.calls == 1


def test_env_configures_ego_lane_change_mode_for_policy_control():
    class DummyVehicleApi:
        def __init__(self):
            self.calls = []

        def getIDList(self):
            return ["ego"]

        def setLaneChangeMode(self, vehicle_id, mode):
            self.calls.append((vehicle_id, mode))

    cfg = load_config()
    env = SumoHighwayMergeEnv(cfg, seed=1)
    vehicle_api = DummyVehicleApi()
    env._traci = SimpleNamespace(vehicle=vehicle_api)
    env._configure_ego_control()
    assert vehicle_api.calls == [("ego", 512)]


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
    metrics = {
        "average_reward": 100.0,
        "min_distance_p1": 2.0,
        "ttc_p1": 1.0,
        "drac_p99_capped": 20.0,
        "proxy_collision_rate": 1.0,
        "safety_violation_rate": 1.0,
    }
    score = _safety_score(metrics)
    assert score == pytest.approx(-14.0)

    cfg = load_config()
    cfg.stage3["checkpoint_selection_profile"] = "safety"
    assert _checkpoint_selection_score(metrics, cfg) == pytest.approx(score)


def test_stage3_safety_efficiency_checkpoint_selection_adds_efficiency_terms():
    cfg = load_config()
    cfg.stage3["checkpoint_selection_profile"] = "safety_efficiency"
    metrics = {
        "average_reward": 100.0,
        "min_distance_p1": 2.0,
        "ttc_p1": 1.0,
        "drac_p99_capped": 20.0,
        "proxy_collision_rate": 1.0,
        "safety_violation_rate": 1.0,
        "completion_time_mean": 10.0,
        "ego_speed_mean": 20.0,
    }
    weights = _checkpoint_selection_weights(cfg)
    assert weights["completion_time_mean"] == pytest.approx(-2.0)
    assert weights["ego_speed_mean"] == pytest.approx(0.5)
    assert _checkpoint_selection_score(metrics, cfg) == pytest.approx(-24.0)


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
        emergency_fallbacks=1.0,
    )
    variant = _variant_report(base, candidate)
    assert variant["metrics"]["merge_success_rate"] == pytest.approx(1.0)
    assert variant["metrics"]["completion_time_mean"] == pytest.approx(9.5)
    assert variant["metrics"]["ego_speed_mean"] == pytest.approx(21.0)
    assert variant["metrics"]["hard_brake_rate"] == pytest.approx(0.1)
    assert variant["metrics"]["proxy_collision_count"] == 0
    assert variant["metrics"]["safety_violation_count"] == 0
    assert variant["metrics"]["emergency_fallback_count"] == 1
    assert variant["metrics"]["emergency_fallback_rate"] == pytest.approx(1.0)
    assert variant["delta"]["mean_completion_time_delta"] == pytest.approx(-0.5)
    assert variant["delta"]["emergency_fallback_count_delta"] == 1


def test_forecast_source_parser_rejects_conflicting_legacy_and_multi_args():
    assert resolve_forecast_sources() == ["constant_velocity", "wcdt_v3"]
    assert resolve_forecast_sources("constant_velocity,wcdt,wcdt_v2") == ["constant_velocity", "wcdt", "wcdt_v2"]
    assert resolve_forecast_sources("constant_velocity,wcdt_v2,wcdt_v3") == [
        "constant_velocity",
        "wcdt_v2",
        "wcdt_v3",
    ]
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
    assert "forecast_wcdt_ppo" not in configs
    assert "forecast_wcdt_v2_ppo" not in configs
    assert "forecast_wcdt_v3_ppo" in configs
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
    assert "ppo_wcdt_features" not in groups
    assert groups["ppo_wcdt_v3_features"]["model_path"] == (
        "safe_rl_output/runs/safe_rl_test_run_forecast_wcdt_v3/stage3/ppo_model.zip"
    )
    assert groups["ppo_wcdt_v3_features"]["forecast_checkpoint"] == (
        "safe_rl_output/runs/safe_rl_test_run/stage2/wcdt_v3_predictor.pt"
    )
    assert groups["wcdt_v3_prediction_shield"]["forecast_checkpoint"] == (
        "safe_rl_output/runs/safe_rl_test_run/stage2/wcdt_v3_predictor.pt"
    )
    main = yaml.safe_load(configs["main"].read_text(encoding="utf-8"))
    assert main["prediction"] == {
        "train_enabled": True,
        "wcdt_v1_train_enabled": False,
        "wcdt_v2_train_enabled": False,
        "wcdt_v3_train_enabled": True,
    }

    stage2_stage4 = yaml.safe_load(configs["stage2_with_stage4"].read_text(encoding="utf-8"))
    assert stage2_stage4["prediction"]["train_enabled"] is False

    forecast_cv = yaml.safe_load(configs["forecast_cv_ppo"].read_text(encoding="utf-8"))
    assert forecast_cv["run"]["run_id"] == "safe_rl_test_run_forecast_cv"
    assert forecast_cv["forecast_features"]["source"] == "constant_velocity"
    assert forecast_cv["forecast_features"]["checkpoint"] is None
    assert forecast_cv["forecast_features"]["allow_heuristic_fallback"] is False
    assert forecast_cv["rl"]["total_timesteps"] == 128

    forecast_wcdt_v3 = yaml.safe_load(configs["forecast_wcdt_v3_ppo"].read_text(encoding="utf-8"))
    assert forecast_wcdt_v3["run"]["run_id"] == "safe_rl_test_run_forecast_wcdt_v3"
    assert forecast_wcdt_v3["forecast_features"]["source"] == "wcdt_v3"
    assert forecast_wcdt_v3["forecast_features"]["checkpoint"] == (
        "safe_rl_output/runs/safe_rl_test_run/stage2/wcdt_v3_predictor.pt"
    )


def test_full_pipeline_generated_configs_support_single_forecast_source(tmp_path):
    cv_configs = build_generated_configs(
        "safe_rl_test_run",
        tmp_path / "cv",
        forecast_sources=["constant_velocity"],
    )
    cv_stage5 = yaml.safe_load(cv_configs["stage5_multi_groups"].read_text(encoding="utf-8"))
    cv_main = yaml.safe_load(cv_configs["main"].read_text(encoding="utf-8"))
    cv_groups = {item["name"]: item for item in cv_stage5["stage5"]["groups"]}
    assert "ppo_cv_features" in cv_groups
    assert "ppo_wcdt_features" not in cv_groups
    assert "forecast_cv_ppo" in cv_configs
    assert "forecast_wcdt_ppo" not in cv_configs
    assert cv_main["prediction"] == {
        "train_enabled": False,
        "wcdt_v1_train_enabled": False,
        "wcdt_v2_train_enabled": False,
        "wcdt_v3_train_enabled": False,
    }

    wcdt_configs = build_generated_configs(
        "safe_rl_test_run",
        tmp_path / "wcdt",
        stage1_episodes=2,
        ppo_timesteps=128,
        forecast_sources=["wcdt"],
    )
    wcdt_stage5 = yaml.safe_load(wcdt_configs["stage5_multi_groups"].read_text(encoding="utf-8"))
    wcdt_main = yaml.safe_load(wcdt_configs["main"].read_text(encoding="utf-8"))
    wcdt_groups = {item["name"]: item for item in wcdt_stage5["stage5"]["groups"]}
    assert "ppo_cv_features" not in wcdt_groups
    assert "ppo_wcdt_features" in wcdt_groups
    assert "forecast_cv_ppo" not in wcdt_configs
    assert "forecast_wcdt_ppo" in wcdt_configs
    assert wcdt_groups["ppo_wcdt_features"]["forecast_checkpoint"] == (
        "safe_rl_output/runs/safe_rl_test_run/stage2/wcdt_predictor.pt"
    )
    assert wcdt_main["prediction"] == {
        "train_enabled": True,
        "wcdt_v1_train_enabled": True,
        "wcdt_v2_train_enabled": False,
        "wcdt_v3_train_enabled": False,
    }

    v2_configs = build_generated_configs(
        "safe_rl_test_run",
        tmp_path / "wcdt_v2",
        forecast_sources=["wcdt_v2"],
    )
    v2_stage5 = yaml.safe_load(v2_configs["stage5_multi_groups"].read_text(encoding="utf-8"))
    v2_main = yaml.safe_load(v2_configs["main"].read_text(encoding="utf-8"))
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
    assert v2_main["prediction"] == {
        "train_enabled": True,
        "wcdt_v1_train_enabled": False,
        "wcdt_v2_train_enabled": True,
        "wcdt_v3_train_enabled": False,
    }


def test_full_pipeline_generated_configs_enable_both_predictors_for_v1_v2_comparison(tmp_path):
    configs = build_generated_configs(
        "safe_rl_test_run",
        tmp_path,
        forecast_sources=["constant_velocity", "wcdt", "wcdt_v2"],
    )
    main = yaml.safe_load(configs["main"].read_text(encoding="utf-8"))
    assert main["prediction"] == {
        "train_enabled": True,
        "wcdt_v1_train_enabled": True,
        "wcdt_v2_train_enabled": True,
        "wcdt_v3_train_enabled": False,
    }


def test_full_pipeline_generated_configs_enable_explicit_v3_ablation(tmp_path):
    configs = build_generated_configs(
        "safe_rl_test_run",
        tmp_path,
        forecast_sources=["constant_velocity", "wcdt_v2", "wcdt_v3"],
    )
    main = yaml.safe_load(configs["main"].read_text(encoding="utf-8"))
    assert main["prediction"] == {
        "train_enabled": True,
        "wcdt_v1_train_enabled": False,
        "wcdt_v2_train_enabled": True,
        "wcdt_v3_train_enabled": True,
    }
    assert "forecast_wcdt_v3_ppo" in configs
    stage5 = yaml.safe_load(configs["stage5_multi_groups"].read_text(encoding="utf-8"))
    groups = {item["name"]: item for item in stage5["stage5"]["groups"]}
    assert len(groups) == 8
    assert groups["ppo_wcdt_v3_features"]["model_path"].endswith("_forecast_wcdt_v3/stage3/ppo_model.zip")
    assert groups["ppo_wcdt_v3_features"]["forecast_checkpoint"].endswith("/stage2/wcdt_v3_predictor.pt")
    assert groups["wcdt_v3_prediction_shield"]["forecast_source"] == "wcdt_v3"


def test_wcdt_predictors_use_independent_batch_sizes():
    cfg = load_config()
    assert _wcdt_v1_batch_size(cfg) == 16
    assert _wcdt_v2_batch_size(cfg) == 32
    assert _wcdt_v3_batch_size(cfg) == 16
    del cfg.prediction["wcdt_v1_batch_size"]
    del cfg.prediction["wcdt_v2_batch_size"]
    del cfg.prediction["wcdt_v3_batch_size"]
    assert _wcdt_v1_batch_size(cfg) == int(cfg.prediction.batch_size)
    assert _wcdt_v2_batch_size(cfg) == int(cfg.prediction.batch_size)
    assert _wcdt_v3_batch_size(cfg) == int(cfg.prediction.batch_size)


def test_runner_run_id_validation_and_managed_cleanup(tmp_path):
    assert _validate_run_id("safe_rl.test-001") == "safe_rl.test-001"
    with pytest.raises(ValueError):
        _validate_run_id("../escape")
    paths = _managed_run_dirs(tmp_path, "safe_rl_test")
    assert {path.name for path in paths} == {
        "safe_rl_test",
        "safe_rl_test_forecast_cv",
        "safe_rl_test_forecast_wcdt",
        "safe_rl_test_forecast_wcdt_v2",
        "safe_rl_test_forecast_wcdt_v3",
    }
    for path in paths:
        path.mkdir()
        (path / "sentinel.txt").write_text("managed", encoding="utf-8")
    outside = tmp_path.parent / "outside-sentinel"
    outside.mkdir(exist_ok=True)
    _remove_managed_run_dirs(tmp_path, "safe_rl_test")
    assert not any(path.exists() for path in paths)
    assert outside.exists()


def test_runner_new_refuses_existing_and_overwrite_recreates_managed_dirs(tmp_path):
    existing = tmp_path / "safe_rl_test_forecast_wcdt_v2"
    existing.mkdir()
    with pytest.raises(FileExistsError):
        _prepare_new_run_dir(tmp_path, "safe_rl_test", "new")
    run_dir = _prepare_new_run_dir(tmp_path, "safe_rl_test", "overwrite")
    assert run_dir.exists()
    assert not existing.exists()


def test_runner_loads_schema_v1_state_with_disabled_v3_task(tmp_path):
    state_path = tmp_path / "pipeline_state.json"
    state_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": "safe_rl_old_state",
                "tasks": {},
            }
        ),
        encoding="utf-8",
    )
    state = _load_pipeline_state(state_path)
    assert state["schema_version"] == 4
    assert state["normalized_invocation"]["pipeline_profile"] == "default"
    assert state["pipeline_profile"] == "default"
    assert state["pipeline_profile_config_sha256"] is None
    assert not state["tasks"]["stage3_forecast_wcdt_v3"]["enabled"]
    assert state["tasks"]["stage3_forecast_wcdt_v3"]["status"] == "completed"


def test_runner_rejects_old_non_default_profile_state_without_profile_hash(tmp_path):
    state_path = tmp_path / "pipeline_state.json"
    state_path.write_text(
        json.dumps(
            {
                "schema_version": 3,
                "run_id": "safe_rl_old_smoke",
                "normalized_invocation": {"pipeline_profile": "smoke"},
                "forecast_sources": ["constant_velocity"],
                "tasks": {},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="pipeline profile hash"):
        _load_pipeline_state(state_path)


def test_runner_resume_validates_pipeline_profile_hash():
    cfg = load_config()
    invocation = {
        "stage1_episodes": None,
        "stage4_episodes": None,
        "stage5_episodes": None,
        "ppo_timesteps": None,
        "forecast_ppo_timesteps": None,
        "forecast_ppo_profile": "default",
        "forecast_sources": ["constant_velocity"],
        "pipeline_profile": "smoke",
    }
    state = _new_pipeline_state("safe_rl_test", invocation)
    assert state["pipeline_profile_config_sha256"] == _pipeline_profile_config_sha256("smoke")
    state["default_config_sha256"] = hashlib.sha256(Path("safe_rl/config/default_safe_rl.yaml").read_bytes()).hexdigest()
    _validate_resume_state(state, cfg)
    state["pipeline_profile_config_sha256"] = "changed"
    with pytest.raises(ValueError, match="pipeline profile config changed"):
        _validate_resume_state(state, cfg)


def test_runner_pipeline_task_records_outputs_and_skips_completed(tmp_path):
    invocation = {
        "stage1_episodes": None,
        "stage4_episodes": None,
        "stage5_episodes": None,
        "ppo_timesteps": None,
        "forecast_ppo_timesteps": None,
        "forecast_ppo_profile": "default",
        "forecast_sources": ["constant_velocity"],
    }
    state = _new_pipeline_state("safe_rl_test", invocation)
    state_path = tmp_path / "pipeline_state.json"
    artifact = tmp_path / "artifact.txt"
    calls = []

    def write_artifact():
        calls.append("called")
        artifact.write_text("done", encoding="utf-8")

    assert _run_pipeline_task(state_path, state, "stage1", [artifact], write_artifact) is True
    assert state["tasks"]["stage1"]["status"] == "completed"
    assert state_path.exists()
    assert _run_pipeline_task(state_path, state, "stage1", [artifact], write_artifact) is False
    assert calls == ["called"]


def test_runner_resume_reuses_saved_invocation_and_rejects_conflicts():
    invocation = {
        "stage1_episodes": 1000,
        "stage4_episodes": None,
        "stage5_episodes": None,
        "ppo_timesteps": 100000,
        "forecast_ppo_timesteps": 100000,
        "forecast_ppo_profile": "shield_guided",
        "forecast_sources": ["constant_velocity", "wcdt_v2"],
    }
    state = _new_pipeline_state("safe_rl_test", invocation)
    resumed = _resume_invocation(
        state,
        stage1_episodes=None,
        stage4_episodes=None,
        stage5_episodes=None,
        ppo_timesteps=None,
        forecast_ppo_timesteps=None,
        forecast_ppo_profile=None,
        forecast_sources=None,
        forecast_source=None,
    )
    assert resumed == {**invocation, "pipeline_profile": "default"}
    with pytest.raises(ValueError, match="ppo_timesteps"):
        _resume_invocation(
            state,
            stage1_episodes=None,
            stage4_episodes=None,
            stage5_episodes=None,
            ppo_timesteps=20000,
            forecast_ppo_timesteps=None,
            forecast_ppo_profile=None,
            forecast_sources=None,
            forecast_source=None,
        )


def test_runner_resume_resets_first_unfinished_task_and_downstream():
    invocation = {
        "stage1_episodes": 2,
        "stage4_episodes": 1,
        "stage5_episodes": 1,
        "ppo_timesteps": 128,
        "forecast_ppo_timesteps": 128,
        "forecast_ppo_profile": "shield_guided",
        "forecast_sources": ["constant_velocity", "wcdt_v2"],
    }
    state = _new_pipeline_state("safe_rl_test", invocation)
    for name in ("network_snapshot", "stage1", "stage2_initial"):
        state["tasks"][name]["status"] = "completed"
    state["tasks"]["stage3_baseline"]["status"] = "running"
    state["tasks"]["stage4"]["status"] = "completed"
    _reset_unfinished_tasks(state)
    assert state["tasks"]["stage2_initial"]["status"] == "completed"
    assert state["tasks"]["stage3_baseline"]["status"] == "pending"
    assert state["tasks"]["stage4"]["status"] == "pending"
    assert state["tasks"]["stage3_forecast_wcdt"]["enabled"] is False


def test_runner_resume_validates_completed_artifact_hash(tmp_path):
    artifact = tmp_path / "artifact.txt"
    artifact.write_text("before", encoding="utf-8")
    invocation = {
        "stage1_episodes": None,
        "stage4_episodes": None,
        "stage5_episodes": None,
        "ppo_timesteps": None,
        "forecast_ppo_timesteps": None,
        "forecast_ppo_profile": "default",
        "forecast_sources": ["constant_velocity"],
    }
    state = _new_pipeline_state("safe_rl_test", invocation)
    task = state["tasks"]["stage1"]
    task["status"] = "completed"
    task["required_outputs"] = [str(artifact)]
    import hashlib

    task["output_hashes"] = {str(artifact): hashlib.sha256(b"before").hexdigest()}
    _validate_completed_outputs(state)
    artifact.write_text("after", encoding="utf-8")
    with pytest.raises(ValueError, match="hash changed"):
        _validate_completed_outputs(state)


def test_full_pipeline_generated_configs_support_smoke_episode_overrides(tmp_path):
    configs = build_generated_configs(
        "safe_rl_smoke",
        tmp_path,
        stage1_episodes=2,
        stage4_episodes=2,
        stage5_episodes=2,
    )
    main = yaml.safe_load(configs["main"].read_text(encoding="utf-8"))
    stage5 = yaml.safe_load(configs["stage5_multi_groups"].read_text(encoding="utf-8"))
    assert main["stage1"]["episodes"] == 2
    assert main["stage4"]["episodes"] == 2
    assert stage5["stage5"]["episodes_per_group"] == 2
    assert stage5["stage5"]["seeds"] == [1, 2]


def test_full_pipeline_smoke_profile_is_lightweight_and_cli_overrides_profile(tmp_path):
    configs = build_generated_configs(
        "safe_rl_smoke_profile",
        tmp_path,
        pipeline_profile="smoke",
        ppo_timesteps=16,
        forecast_sources=["constant_velocity", "wcdt_v2", "wcdt_v3"],
    )
    main = yaml.safe_load(configs["main"].read_text(encoding="utf-8"))
    stage5 = yaml.safe_load(configs["stage5_multi_groups"].read_text(encoding="utf-8"))
    assert main["scenario"]["episode_seconds"] == pytest.approx(6.0)
    assert main["stage1"]["episodes"] == 2
    assert main["stage1"]["audit_gate"]["enabled"] is False
    assert main["rl"]["total_timesteps"] == 16
    assert main["rl"]["n_steps"] == 8
    assert main["rl"]["batch_size"] == 8
    assert main["stage3"]["eval_enabled"] is False
    assert main["prediction"]["wcdt_v2_epochs"] == 1
    assert main["prediction"]["wcdt_v2_ensemble_size"] == 1
    assert main["prediction"]["wcdt_v3_epochs"] == 1
    assert main["prediction"]["wcdt_v3_ensemble_size"] == 1
    assert main["risk_module"]["epochs"] == 1
    assert stage5["stage5"]["episodes_per_group"] == 1
    assert stage5["stage5"]["seeds"] == [1]


def test_ppo_training_device_defaults_to_cpu_and_legacy_fallback_remains_available():
    cfg = load_config()
    assert _training_device(cfg) == "cpu"
    del cfg.training["ppo_device"]
    cfg.training["device"] = "gpu"
    assert _training_device(cfg) == "cuda"


def test_legacy_wcdt_v1_forecast_config_still_loads():
    cfg = load_config("safe_rl/config/advanced/ppo_wcdt_v1_features_legacy.yaml")
    assert cfg.forecast_features.enabled is True
    assert cfg.forecast_features.source == "wcdt"
    assert cfg.forecast_features.allow_heuristic_fallback is False


def test_stage5_six_group_cv_wcdt_v2_example_uses_current_mainline():
    path = Path("safe_rl/config/advanced/stage5_six_groups_cv_wcdt_v2.example.yaml")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    groups = {item["name"]: item for item in payload["stage5"]["groups"]}
    assert set(groups) == {
        "ppo",
        "ppo_shield",
        "ppo_cv_features",
        "cv_prediction_shield",
        "ppo_wcdt_v2_features",
        "wcdt_v2_prediction_shield",
    }
    assert groups["ppo_wcdt_v2_features"]["model_path"].endswith("_forecast_wcdt_v2/stage3/ppo_model.zip")
    assert groups["ppo_wcdt_v2_features"]["forecast_checkpoint"].endswith("/stage2/wcdt_v2_predictor.pt")


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


def test_full_pipeline_merge_timing_profile_binds_base_risk_module_for_forecast_only(tmp_path):
    configs = build_generated_configs(
        "safe_rl_test_run",
        tmp_path,
        ppo_timesteps=20000,
        forecast_ppo_timesteps=100000,
        forecast_ppo_profile="merge_timing",
        forecast_sources=["wcdt_v3"],
    )
    main = yaml.safe_load(configs["main"].read_text(encoding="utf-8"))
    forecast = yaml.safe_load(configs["forecast_wcdt_v3_ppo"].read_text(encoding="utf-8"))
    assert "reward_profile" not in main["rl"]
    assert forecast["rl"]["total_timesteps"] == 100000
    assert forecast["rl"]["reward_profile"] == "merge_timing_forecast"
    assert forecast["rl"]["shield_guided_reward"]["risk_checkpoint"] == (
        "safe_rl_output/runs/safe_rl_test_run/stage2/risk_module.pt"
    )


def test_forecast_branch_runner_builds_merge_timing_stage5_groups():
    from safe_rl.pipeline.train_forecast_branches import _forecast_training_payload, _stage5_payload

    payload = _forecast_training_payload(
        base_run_id="safe_rl_base",
        source="wcdt_v3",
        suffix="merge_timing",
        profile="merge_timing",
        timesteps=100000,
    )
    assert payload["run"]["run_id"] == "safe_rl_base_forecast_wcdt_v3_merge_timing"
    assert payload["rl"]["reward_profile"] == "merge_timing_forecast"
    assert payload["rl"]["shield_guided_reward"]["risk_checkpoint"] == (
        "safe_rl_output/runs/safe_rl_base/stage2/risk_module.pt"
    )

    stage5 = _stage5_payload("safe_rl_base", ["constant_velocity", "wcdt_v3"], "merge_timing")
    groups = {item["name"]: item for item in stage5["stage5"]["groups"]}
    assert "ppo_wcdt_v3_features" in groups
    assert "wcdt_v3_prediction_shield" in groups
    assert "ppo_wcdt_v3_merge_timing_features" in groups
    assert "wcdt_v3_merge_timing_prediction_shield" in groups
    assert groups["ppo_wcdt_v3_merge_timing_features"]["model_path"].endswith(
        "safe_rl_base_forecast_wcdt_v3_merge_timing/stage3/ppo_model.zip"
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
    emergency_fallbacks: float = 0.0,
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
                "emergency_fallback_count": int(emergency_fallbacks),
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
            "emergency_fallback_rate": float(emergency_fallbacks > 0.0),
            "mean_emergency_fallbacks": emergency_fallbacks,
            "emergency_fallback_count": int(emergency_fallbacks),
        },
    }


def test_wcdt_v3_promotion_gate_rejects_smoke_episode_count_and_slow_policy():
    reports = {
        "ppo_wcdt_v2_features": _fake_group([(1, 100.0)], 100.0, completion_time=10.0),
        "wcdt_v2_prediction_shield": _fake_group([(1, 101.0)], 101.0, replacements=0.1),
        "ppo_wcdt_v3_features": _fake_group([(1, 101.0)], 101.0, completion_time=12.0),
        "wcdt_v3_prediction_shield": _fake_group([(1, 102.0)], 102.0, replacements=0.3),
    }
    for report in reports.values():
        report["metrics"]["episodes"] = len(report["episodes"])
    candidate = _wcdt_v3_candidate_summary(reports)
    assert not candidate["stage5_candidate_pass"]
    assert not candidate["checks"]["formal_episode_count"]
    assert not candidate["checks"]["completion_time_not_degraded_vs_v2"]
    assert not candidate["checks"]["shield_replacements_not_worse_than_v2"]


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
        "ppo_wcdt_v3_features": _fake_group([(1, 101.0)], 101.0),
        "wcdt_v3_prediction_shield": _fake_group([(1, 102.0)], 102.0),
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
        "ppo_cv_features_vs_ppo_wcdt_v3_features",
        "ppo_wcdt_v2_features_vs_ppo_wcdt_v3_features",
    }
    assert paired["ppo_vs_ppo_shield"]["mean_completion_time_delta"] == pytest.approx(-1.0)
    assert paired["ppo_vs_ppo_shield"]["mean_ego_speed_delta"] == pytest.approx(1.0)
    assert paired["ppo_vs_ppo_shield"]["mean_hard_brake_rate_delta"] == pytest.approx(0.1)
    assert paired["ppo_vs_ppo_shield"]["proxy_collision_count_delta"] == 0
    assert paired["ppo_vs_ppo_shield"]["safety_violation_count_delta"] == 0
    assert paired["ppo_vs_ppo_shield"]["emergency_fallback_count_delta"] == 0
    acceptance = _build_acceptance(reports)
    assert acceptance["ppo_shield"]["available"]
    assert acceptance["cv_prediction_shield"]["available"]
    assert acceptance["wcdt_prediction_shield"]["available"]
    assert acceptance["wcdt_v2_prediction_shield"]["available"]
    assert acceptance["wcdt_v3_prediction_shield"]["available"]
    assert acceptance["forecast_cv_vs_baseline"]["available"]
    assert acceptance["forecast_wcdt_vs_cv"]["available"]
    assert acceptance["forecast_wcdt_v2_vs_cv"]["available"]
    assert acceptance["forecast_wcdt_v3_vs_cv"]["available"]

    single = {
        "ppo": reports["ppo"],
        "ppo_shield": reports["ppo_shield"],
        "ppo_wcdt_features": reports["ppo_wcdt_features"],
        "wcdt_prediction_shield": reports["wcdt_prediction_shield"],
    }
    single_acceptance = _build_acceptance(single)
    assert "forecast_wcdt_vs_cv" not in single_acceptance
    assert single_acceptance["wcdt_prediction_shield"]["available"]


def test_stage5_failure_audit_identifies_min_distance_zero_and_writes_replay_commands(tmp_path, monkeypatch):
    cfg = load_config()
    cfg.run["output_root"] = str(tmp_path / "runs")
    monkeypatch.setattr("safe_rl.pipeline.stage5_failure_audit.load_config", lambda: cfg)
    run_dir = tmp_path / "runs" / "safe_rl_audit" / "stage5"
    replay_dir = run_dir / "replay"
    replay_dir.mkdir(parents=True)
    report = {
        "groups": {
            "cv_prediction_shield": {
                "episodes": [
                    {
                        "seed": 7,
                        "episode_reward": 12.0,
                        "merge_success": True,
                        "proxy_collision": True,
                        "safety_violation": True,
                        "min_distance": 0.0,
                        "ttc_p1": 0.0,
                        "drac_p99_raw": 1_000_000.0,
                        "actual_replacement_count": 2,
                        "emergency_fallback_count": 1,
                        "replacement_reason_counts": {"emergency_fallback": 1},
                        "shield_score_records": [
                            {
                                "raw_action": 4,
                                "final_action": 1,
                                "raw_risk": 0.99,
                                "best_candidate_risk": 0.99,
                                "replacement_reason": "emergency_fallback",
                                "emergency_fallback": True,
                                "emergency_reason": "min_distance",
                            }
                        ],
                    },
                    {
                        "seed": 8,
                        "episode_reward": -5.0,
                        "merge_success": False,
                        "done_reason": "taper_miss",
                        "taper_miss": True,
                        "proxy_collision": False,
                        "safety_violation": False,
                        "min_distance": 3.0,
                        "ttc_p1": 2.0,
                        "drac_p99_raw": 2.0,
                        "missed_safe_merge_opportunity_count": 4,
                        "missed_safe_merge_opportunity_rate": 1.0,
                    },
                ]
            }
        }
    }
    (run_dir / "formal_paired_eval_report.json").write_text(json.dumps(report), encoding="utf-8")
    (replay_dir / "cv_prediction_shield_seed_7.json").write_text(json.dumps({"actions": [4, 4]}), encoding="utf-8")

    audit = build_failure_audit("safe_rl_audit", groups=["cv_prediction_shield"], eval_stage="stage5")
    assert audit["failure_counts"] == {"cv_prediction_shield": 1}
    assert audit["safety_failure_counts"] == {"cv_prediction_shield": 1}
    assert audit["task_failure_counts"] == {"cv_prediction_shield": 1}
    failure = audit["failures"]["cv_prediction_shield"][0]
    assert failure["seed"] == 7
    assert failure["first_failure_step"] == "unavailable"
    assert "missing_step_trace" in failure["failure_classification"]
    assert "late_emergency" in failure["failure_classification"]
    assert failure["shield_records_near_failure"][0]["raw_risk"] == 0.99

    commands = tmp_path / "commands.ps1"
    write_replay_commands(commands, audit)
    content = commands.read_text(encoding="utf-8")
    assert "cv_prediction_shield_seed_7.json" in content
    assert "--gui --delay-ms 200" in content

    audit_with_tasks = build_failure_audit(
        "safe_rl_audit",
        groups=["cv_prediction_shield"],
        eval_stage="stage5",
        include_task_failures=True,
    )
    assert audit_with_tasks["failure_counts"] == {"cv_prediction_shield": 2}
    task_failure = [item for item in audit_with_tasks["failures"]["cv_prediction_shield"] if item["seed"] == 8][0]
    assert "taper_miss" in task_failure["failure_classification"]
    assert "merge_task_failure" in task_failure["failure_classification"]


def test_step_safety_record_marks_proxy_collision_for_failure_audit():
    record = _step_safety_record(
        step_index=4,
        raw_action=5,
        final_action=3,
        reward=-2.5,
        terminated=False,
        truncated=False,
        info={
            "step": 20,
            "done_reason": "",
            "min_distance": 0.1,
            "min_ttc": 0.2,
            "max_drac": 123.0,
            "near_miss": True,
            "target_front_gap": 1.2,
            "target_rear_gap": 0.8,
            "target_lane_gap": 2.0,
            "distance_to_taper": 15.0,
        },
        collision_threshold=0.25,
        shield_enabled=True,
    )
    assert record["step"] == 20
    assert record["control_step"] == 4
    assert record["shield_record_index"] == 4
    assert record["raw_action"] == 5
    assert record["final_action"] == 3
    assert record["proxy_collision"]
    assert record["safety_violation"]
    assert record["drac_raw"] == pytest.approx(123.0)


def test_forecast_behavior_diagnostics_supports_cv_vs_wcdt_v2(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    replay_dir = tmp_path / "safe_rl_output" / "runs" / "safe_rl_behavior_test" / "stage5" / "replay"
    replay_dir.mkdir(parents=True)
    (replay_dir / "ppo_cv_features_seed_1.json").write_text(json.dumps({"actions": [4, 4, 5]}), encoding="utf-8")
    (replay_dir / "ppo_wcdt_v2_features_seed_1.json").write_text(
        json.dumps({"actions": [4, 4, 5], "executed_actions": [4, 5, 5]}),
        encoding="utf-8",
    )
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


def test_confirmatory_payload_generates_fifty_seed_cv_wcdt_v3_config():
    payload = build_confirmatory_payload("safe_rl_test_run")
    groups = {item["name"]: item for item in payload["stage5"]["groups"]}
    assert payload["stage5"]["episodes_per_group"] == 50
    assert payload["stage5"]["seeds"] == list(range(1, 51))
    assert set(groups) == {
        "ppo",
        "ppo_shield",
        "ppo_cv_features",
        "cv_prediction_shield",
        "ppo_wcdt_v3_features",
        "wcdt_v3_prediction_shield",
    }
    assert groups["ppo_wcdt_v3_features"]["forecast_checkpoint"] == (
        "safe_rl_output/runs/safe_rl_test_run/stage2/wcdt_v3_predictor.pt"
    )


def test_confirmatory_payload_supports_optional_v3_ablation_groups():
    payload = build_confirmatory_payload(
        "safe_rl_test_run",
        episodes=5,
        forecast_sources=["constant_velocity", "wcdt_v2", "wcdt_v3"],
    )
    groups = {item["name"]: item for item in payload["stage5"]["groups"]}
    assert len(groups) == 8
    assert groups["ppo_wcdt_v3_features"]["forecast_checkpoint"] == (
        "safe_rl_output/runs/safe_rl_test_run/stage2/wcdt_v3_predictor.pt"
    )
    assert groups["wcdt_v3_prediction_shield"]["forecast_source"] == "wcdt_v3"


def test_confirmatory_input_validation_reports_missing_checkpoints():
    payload = build_confirmatory_payload("safe_rl_missing_confirmatory_run", episodes=5)
    with pytest.raises(FileNotFoundError, match="Stage5 confirmatory eval requires existing"):
        validate_confirmatory_inputs(payload)


def test_confirmatory_summary_uses_wcdt_v3_as_main_prediction_branch():
    reports = {
        "ppo": _fake_group([(1, 100.0)], 100.0, min_distance=2.0),
        "ppo_shield": _fake_group([(1, 101.0)], 101.0, min_distance=2.1, replacements=1.0),
        "ppo_cv_features": _fake_group([(1, 105.0)], 105.0, min_distance=3.0, drac=8.0),
        "ppo_wcdt_v3_features": _fake_group([(1, 110.0)], 110.0, min_distance=5.0, drac=4.0),
        "wcdt_v3_prediction_shield": _fake_group([(1, 110.0)], 110.0, min_distance=5.0, drac=4.0),
    }
    for report in reports.values():
        report["metrics"]["episodes"] = 50
    paired = _build_paired_delta(reports)
    acceptance = _build_acceptance(reports)
    summary = build_confirmatory_summary(reports, paired, acceptance)
    assert summary["ppo_shield_mainline"]["pass"]
    assert summary["wcdt_v3_forecast_mainline"]["pass"]
    assert summary["wcdt_v3_candidate"]["available"]
    assert summary["wcdt_v3_candidate"]["reference_branch"] == "constant_velocity"
    assert summary["wcdt_v3_candidate"]["stage5_candidate_pass"]
    assert summary["final_result_summary"]["trusted_mainline"] == ["ppo", "ppo_shield"]
    assert summary["final_result_summary"]["recommended_prediction_branch"] == "ppo_wcdt_v3_features"
    assert summary["final_result_summary"]["best_safety_combo"] == "wcdt_v3_prediction_shield"
    assert summary["model_role_explanations"]["wcdt_v3_prediction_shield"]["shield_enabled"] is True
    assert summary["reporting_recommendation"][0]["comparison"] == "ppo_vs_ppo_shield"
    assert summary["forecast_policy_utilization_summary"]["available"] is False
    assert summary["overall_pass"]


def test_confirmatory_summary_marks_wcdt_v2_shield_low_frequency_backstop():
    reports = {
        "ppo": _fake_group([(1, 100.0)], 100.0, min_distance=2.0),
        "ppo_shield": _fake_group([(1, 101.0)], 101.0, min_distance=2.1, replacements=1.0),
        "ppo_cv_features": _fake_group([(1, 105.0)], 105.0, min_distance=3.0, drac=8.0),
        "ppo_wcdt_v2_features": _fake_group([(1, 110.0)], 110.0, min_distance=5.0, drac=4.0),
        "wcdt_v2_prediction_shield": _fake_group(
            [(1, 110.5)],
            110.5,
            min_distance=5.2,
            drac=3.8,
            replacements=0.1,
            emergency_fallbacks=0.1,
        ),
    }
    paired = _build_paired_delta(reports)
    acceptance = _build_acceptance(reports)
    summary = build_confirmatory_summary(reports, paired, acceptance)
    assert not summary["wcdt_v2_shield"]["shield_not_needed_on_wcdt_v2_policy"]
    assert summary["wcdt_v2_shield"]["low_frequency_safety_backstop"]
    assert summary["wcdt_v2_shield"]["shield_status"] == "low_frequency_safety_backstop"
    assert summary["wcdt_v2_shield"]["mean_emergency_fallbacks"] == pytest.approx(0.1)
    assert not summary["overall_pass"]


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
    assert utilization["main_forecast_branch"] == "wcdt_v2"
    assert utilization["predictor_quality_pass"]
    assert utilization["ppo_better_than_cv"]
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
