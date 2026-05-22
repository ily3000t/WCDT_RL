from __future__ import annotations

import json
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


def _net_lanes(net_file: Path) -> dict[str, dict[int, dict[str, float]]]:
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
            }
        if lanes:
            edges[edge_id] = lanes
    return edges


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
    routes, vtypes = _routes(route_file)
    route_root = ET.parse(route_file).getroot()

    errors: list[str] = []
    warnings: list[str] = []
    seed_positions: list[dict[str, Any]] = []

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

    return {
        "scenario_cfg": str(cfg_file),
        "net_file": str(net_file),
        "route_file": str(route_file),
        "passed": not errors,
        "errors": errors,
        "warnings": warnings,
        "seed_positions": seed_positions,
    }


def write_validation_report(cfg_file: str | Path, output_file: str | Path) -> dict[str, Any]:
    report = validate_scenario_geometry(cfg_file)
    with Path(output_file).open("w", encoding="utf-8") as file:
        json.dump(json_ready(report), file, ensure_ascii=False, indent=2, allow_nan=False)
    return report
