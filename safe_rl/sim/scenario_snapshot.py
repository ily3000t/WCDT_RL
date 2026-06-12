from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import Any

from safe_rl.utils.io import write_json


SCENARIO_SUFFIXES = (".nod.xml", ".edg.xml", ".con.xml", ".rou.xml", ".net.xml", ".sumocfg")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot_scenario(cfg: Any, run_dir: str | Path) -> Path:
    source = Path(cfg.scenario.root)
    if not source.is_absolute():
        source = Path.cwd() / source
    target = Path(run_dir) / "scenario_snapshot"
    target.mkdir(parents=True, exist_ok=True)
    files = []
    for path in sorted(source.iterdir()):
        if not path.is_file() or not path.name.endswith(SCENARIO_SUFFIXES):
            continue
        copied = target / path.name
        shutil.copy2(path, copied)
        files.append({"name": path.name, "sha256": _sha256(copied)})
    manifest = target / "manifest.json"
    write_json(
        manifest,
        {
            "scenario_name": str(cfg.scenario.get("name", "highway_merge")),
            "source": str(source.resolve()),
            "sumo_installation": {
                "sumo_binary": str(cfg.scenario.get("sumo_binary", "")),
                "sumo_gui_binary": str(cfg.scenario.get("sumo_gui_binary", "")),
                "netconvert_binary": str(cfg.scenario.get("netconvert_binary", "")),
                "sumo_home": str(cfg.scenario.get("sumo_home", "")),
                "sumo_version": str(cfg.scenario.get("sumo_version", "")),
            },
            "files": files,
        },
    )
    return manifest
