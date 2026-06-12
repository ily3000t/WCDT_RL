#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import subprocess
import sys
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from safe_rl.sim.scenario_validation import validate_scenario_geometry
from safe_rl.utils.sumo_installation import resolve_sumo_installation


def main():
    parser = argparse.ArgumentParser(description="Build the highway merge SUMO network.")
    parser.add_argument("--netconvert", default=None, help="Absolute netconvert executable path.")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    net_file = root / "highway_merge.net.xml"
    cfg_file = root / "highway_merge.sumocfg"
    node_file = root / "highway_merge.nod.xml"
    edge_file = root / "highway_merge.edg.xml"
    con_file = root / "highway_merge.con.xml"
    validation_report_file = root / "scenario_geometry_check.json"

    netconvert = str(args.netconvert or resolve_sumo_installation({"sumo_binary": "sumo"}).netconvert_binary)
    cmd = [
        netconvert,
        "--node-files", str(node_file),
        "--edge-files", str(edge_file),
        "--connection-files", str(con_file),
        "--output-file", str(net_file),
    ]
    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr, file=sys.stderr)
        raise SystemExit(result.returncode)
    print(f"Generated: {net_file}")

    report = validate_scenario_geometry(cfg_file)
    with validation_report_file.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Geometry check report: {validation_report_file}")
    if not bool(report.get("passed", False)):
        errors = list(report.get("errors", []) or [])
        preview = "; ".join(str(item) for item in errors[:3])
        print(f"Geometry validation failed: {preview}", file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
