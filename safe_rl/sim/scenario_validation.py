from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from safe_rl.utils.io import json_ready


def _resolve(path: str | Path, base: Path | None = None) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    if base is not None:
        candidate = base / path
        if candidate.exists():
            return candidate
    return path.resolve()


def _parse_cfg(cfg_file: Path) -> tuple[Path, Path]:
    root = ET.parse(cfg_file).getroot()
    input_node = root.find("input")
    if input_node is None:
        raise ValueError(f"{cfg_file} has no <input> block")
    net_value = input_node.find("net-file").attrib["value"]
    route_value = input_node.find("route-files").attrib["value"].split(",")[0].strip()
    return _resolve(net_value, cfg_file.parent), _resolve(route_value, cfg_file.parent)


def _shape_points(value: str) -> list[tuple[float, float]]:
    points = []
    for item in str(value).split():
        x_text, y_text = item.split(",", 1)
        points.append((float(x_text), float(y_text)))
    return points


def _net_lanes(net_file: Path) -> dict[str, dict[int, dict[str, Any]]]:
    root = ET.parse(net_file).getroot()
    edges: dict[str, dict[int, dict[str, float]]] = {}
    for edge in root.findall("edge"):
        edge_id = edge.attrib.get("id", "")
        if edge.attrib.get("function") == "internal":
            continue
        lanes: dict[int, dict[str, float]] = {}
        for lane in edge.findall("lane"):
            index = int(lane.attrib.get("index", "0"))
            lanes[index] = {
                "length": float(lane.attrib.get("length", "0")),
                "speed": float(lane.attrib.get("speed", "0")),
                "points": _shape_points(lane.attrib.get("shape", "")),
            }
        if lanes:
            edges[edge_id] = lanes
    return edges


def _net_connections(net_file: Path) -> list[dict[str, Any]]:
    root = ET.parse(net_file).getroot()
    connections: list[dict[str, Any]] = []
    for connection in root.findall("connection"):
        from_edge = connection.attrib.get("from", "")
        to_edge = connection.attrib.get("to", "")
        if from_edge.startswith(":") or to_edge.startswith(":"):
            continue
        connections.append(
            {
                "from": from_edge,
                "to": to_edge,
                "from_lane": int(connection.attrib.get("fromLane", "0")),
                "to_lane": int(connection.attrib.get("toLane", "0")),
            }
        )
    return connections


def _routes(route_file: Path) -> tuple[dict[str, list[str]], dict[str, dict[str, float]]]:
    root = ET.parse(route_file).getroot()
    routes = {
        route.attrib["id"]: route.attrib.get("edges", "").split()
        for route in root.findall("route")
    }
    vtypes = {
        vtype.attrib["id"]: {
            "length": float(vtype.attrib.get("length", "4.8")),
            "min_gap": float(vtype.attrib.get("minGap", "2.5")),
        }
        for vtype in root.findall("vType")
    }
    return routes, vtypes


def validate_scenario_geometry(cfg_file: str | Path) -> dict[str, Any]:
    cfg_file = Path(cfg_file).resolve()
    net_file, route_file = _parse_cfg(cfg_file)
    edges = _net_lanes(net_file)
    connections = _net_connections(net_file)
    routes, vtypes = _routes(route_file)
    route_root = ET.parse(route_file).getroot()

    errors: list[str] = []
    warnings: list[str] = []
    seed_positions: list[dict[str, Any]] = []
    edge_lane_counts = {edge_id: len(lanes) for edge_id, lanes in edges.items()}
    through_lane_lateral_shift: list[dict[str, Any]] = []
    merge_side_consistency: dict[str, Any] = {}
    target_seed_lane_consistency: dict[str, Any] = {}
    auxiliary_drop_lane: dict[str, Any] = {}
    ramp_entry_angle = 0.0

    if "main_aux" in edges:
        expected_lane_counts = {"main_in": 3, "main_aux": 4, "main_out": 3, "ramp_in": 1}
        for edge_id, expected in expected_lane_counts.items():
            actual = edge_lane_counts.get(edge_id)
            if actual != expected:
                errors.append(f"edge {edge_id} expected {expected} lanes, got {actual}")
        route_ramp = routes.get("route_ramp", [])
        if route_ramp != ["ramp_in", "main_aux", "main_out"]:
            errors.append(f"route_ramp expected ramp_in main_aux main_out, got {' '.join(route_ramp)}")
        route_main = routes.get("route_main", [])
        if route_main != ["main_in", "main_aux", "main_out"]:
            errors.append(f"route_main expected main_in main_aux main_out, got {' '.join(route_main)}")
        route_aux = routes.get("route_aux", [])
        if route_aux != ["main_aux", "main_out"]:
            errors.append(f"route_aux expected main_aux main_out, got {' '.join(route_aux)}")

        connection_set = {
            (item["from"], item["from_lane"], item["to"], item["to_lane"])
            for item in connections
        }
        expected_ramp_connection = ("ramp_in", 0, "main_aux", 0)
        if expected_ramp_connection not in connection_set:
            errors.append("right-side ramp_in lane 0 must connect to main_aux auxiliary lane 0")
        expected_main_connections = {
            ("main_in", 0, "main_aux", 1),
            ("main_in", 1, "main_aux", 2),
            ("main_in", 2, "main_aux", 3),
        }
        actual_main_connections = {
            item
            for item in connection_set
            if item[0] == "main_in" and item[2] == "main_aux"
        }
        if actual_main_connections != expected_main_connections:
            errors.append("main_in lanes 0/1/2 must align with main_aux through lanes 1/2/3")
        expected_aux_connections = {
            ("main_aux", 1, "main_out", 0),
            ("main_aux", 2, "main_out", 1),
            ("main_aux", 3, "main_out", 2),
        }
        actual_aux_connections = {
            item
            for item in connection_set
            if item[0] == "main_aux" and item[2] == "main_out"
        }
        if actual_aux_connections != expected_aux_connections:
            errors.append(
                "main_aux through lanes 1/2/3 must align with main_out lanes 0/1/2 and auxiliary lane 0 must terminate"
            )
        auxiliary_drop_lane = {
            "lane": 0,
            "drops_before_main_out": ("main_aux", 0, "main_out", 0) not in connection_set,
        }
        if not auxiliary_drop_lane["drops_before_main_out"]:
            errors.append("main_aux auxiliary lane 0 must not connect directly to main_out")

        through_pairs = [
            ("main_in", 0, "main_aux", 1),
            ("main_in", 1, "main_aux", 2),
            ("main_in", 2, "main_aux", 3),
            ("main_aux", 1, "main_out", 0),
            ("main_aux", 2, "main_out", 1),
            ("main_aux", 3, "main_out", 2),
        ]
        for from_edge, from_lane, to_edge, to_lane in through_pairs:
            from_points = edges.get(from_edge, {}).get(from_lane, {}).get("points", [])
            to_points = edges.get(to_edge, {}).get(to_lane, {}).get("points", [])
            if not from_points or not to_points:
                continue
            shift = abs(float(to_points[0][1]) - float(from_points[-1][1]))
            through_lane_lateral_shift.append(
                {
                    "from": f"{from_edge}_{from_lane}",
                    "to": f"{to_edge}_{to_lane}",
                    "lateral_shift": float(shift),
                }
            )
            if shift > 0.5:
                errors.append(f"through lane {from_edge}_{from_lane} -> {to_edge}_{to_lane} shifts laterally by {shift:.2f}m")

        ramp_points = edges.get("ramp_in", {}).get(0, {}).get("points", [])
        if ramp_points:
            mainline_start_y = min(
                float(lane["points"][0][1])
                for lane in edges.get("main_in", {}).values()
                if lane.get("points")
            )
            merge_side_consistency = {
                "declared_side": "right",
                "ramp_start_y": float(ramp_points[0][1]),
                "mainline_rightmost_y": float(mainline_start_y),
                "ramp_connects_to_auxiliary_lane": expected_ramp_connection in connection_set,
            }
            if float(ramp_points[0][1]) >= mainline_start_y:
                errors.append("right-side ramp must start below the mainline")
        if len(ramp_points) >= 2:
            dx = float(ramp_points[-1][0] - ramp_points[-2][0])
            dy = float(ramp_points[-1][1] - ramp_points[-2][1])
            ramp_entry_angle = abs(math.degrees(math.atan2(dy, dx)))
            if ramp_entry_angle > 10.0:
                warnings.append(f"ramp entry angle {ramp_entry_angle:.2f}deg exceeds 10deg")

    for vehicle in route_root.findall("vehicle"):
        vehicle_id = vehicle.attrib.get("id", "")
        route_id = vehicle.attrib.get("route", "")
        type_id = vehicle.attrib.get("type", "")
        if route_id not in routes:
            errors.append(f"vehicle {vehicle_id} references unknown route {route_id}")
            continue
        first_edge = routes[route_id][0]
        if first_edge not in edges:
            errors.append(f"vehicle {vehicle_id} first edge {first_edge} is not in net")
            continue

        depart_lane_text = vehicle.attrib.get("departLane", "0")
        if depart_lane_text in ("random", "free", "best", "allowed"):
            warnings.append(f"vehicle {vehicle_id} uses dynamic departLane={depart_lane_text}")
            continue
        depart_lane = int(float(depart_lane_text))
        lanes = edges[first_edge]
        if depart_lane not in lanes:
            errors.append(f"vehicle {vehicle_id} departLane {depart_lane} outside edge {first_edge}")
            continue

        depart_pos_text = vehicle.attrib.get("departPos", "0")
        if depart_pos_text in ("random", "free", "random_free", "base", "last"):
            warnings.append(f"vehicle {vehicle_id} uses dynamic departPos={depart_pos_text}")
            continue
        depart_pos = float(depart_pos_text)
        edge_length = lanes[depart_lane]["length"]
        vehicle_shape = vtypes.get(type_id, {"length": 4.8, "min_gap": 2.5})
        if depart_pos < 0 or depart_pos + vehicle_shape["length"] > edge_length:
            errors.append(
                f"vehicle {vehicle_id} departPos {depart_pos} does not fit on {first_edge} length {edge_length}"
            )

        seed_positions.append(
            {
                "vehicle_id": vehicle_id,
                "type_id": type_id,
                "route_id": route_id,
                "first_edge": first_edge,
                "depart_time": float(vehicle.attrib.get("depart", "0")),
                "depart_lane": depart_lane,
                "depart_pos": depart_pos,
                "edge_length": edge_length,
                "vehicle_length": vehicle_shape["length"],
                "min_gap": vehicle_shape["min_gap"],
            }
        )

    target_seed_ids = {"target_lane_front_seed", "target_lane_gap_seed", "target_lane_rear_seed"}
    target_seed_rows = [item for item in seed_positions if item["vehicle_id"] in target_seed_ids]
    auxiliary_seed_rows = [item for item in seed_positions if item["vehicle_id"].startswith("auxiliary_")]
    target_seed_lane_consistency = {
        "expected_target_first_edge": "main_in",
        "expected_target_lane": 0,
        "target_seed_ids": [item["vehicle_id"] for item in target_seed_rows],
        "target_seeds_consistent": bool(target_seed_rows)
        and all(item["first_edge"] == "main_in" and item["depart_lane"] == 0 for item in target_seed_rows),
        "expected_auxiliary_first_edge": "main_aux",
        "expected_auxiliary_lane": 0,
        "auxiliary_seed_ids": [item["vehicle_id"] for item in auxiliary_seed_rows],
        "auxiliary_seeds_consistent": bool(auxiliary_seed_rows)
        and all(item["first_edge"] == "main_aux" and item["depart_lane"] == 0 for item in auxiliary_seed_rows),
    }
    if not target_seed_lane_consistency["target_seeds_consistent"]:
        errors.append("target lane seeds must depart from main_in lane 0")
    if not target_seed_lane_consistency["auxiliary_seeds_consistent"]:
        errors.append("auxiliary seeds must depart from main_aux lane 0")

    return {
        "scenario_cfg": str(cfg_file),
        "net_file": str(net_file),
        "route_file": str(route_file),
        "passed": not errors,
        "errors": errors,
        "warnings": warnings,
        "edge_lane_counts": edge_lane_counts,
        "connections": connections,
        "routes": routes,
        "seed_positions": seed_positions,
        "through_lane_lateral_shift": through_lane_lateral_shift,
        "merge_side_consistency": merge_side_consistency,
        "target_seed_lane_consistency": target_seed_lane_consistency,
        "auxiliary_drop_lane": auxiliary_drop_lane,
        "ramp_entry_angle": float(ramp_entry_angle),
    }


def write_validation_report(cfg_file: str | Path, output_file: str | Path) -> dict[str, Any]:
    report = validate_scenario_geometry(cfg_file)
    with Path(output_file).open("w", encoding="utf-8") as file:
        json.dump(json_ready(report), file, ensure_ascii=False, indent=2, allow_nan=False)
    return report
