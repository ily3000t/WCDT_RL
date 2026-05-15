from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from safe_rl.risk.risk_module import HeuristicRiskEstimator
from safe_rl.shield.safety_shield import SafetyShield
from safe_rl.sim.action_space import ACTIONS, decode_action
from safe_rl.sim.metrics import compute_step_metrics
from safe_rl.sim.scenario_validation import validate_scenario_geometry
from safe_rl.sim.types import VehicleState
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


def test_shield_replaces_lane_oob_action():
    cfg = load_config()
    cfg.shield["enabled"] = True
    cfg.shield["risk_threshold"] = 0.65
    cfg.shield["uncertainty_threshold"] = 0.40
    ego = VehicleState("ego", 0.0, 0.0, 0.0, 10.0, 0, "ramp_in_0", 0.0, "ramp_in")
    context = {"ego": ego, "vehicles": [ego], "lane_count": 1, "config": cfg}
    shield = SafetyShield(cfg)
    raw = decode_action(0)
    final, record = shield.select_action(raw, context)
    assert final.index != raw.index
    assert record["replacement_reason"] in {"replacement", "fallback"}
