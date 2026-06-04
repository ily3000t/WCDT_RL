from __future__ import annotations

from pathlib import Path
from typing import Any
import hashlib

from safe_rl.utils.io import write_json


def _scenario_snapshot_metadata(path: Path, run_id: str) -> dict[str, str | None]:
    run_dir = path
    while run_dir.name != str(run_id) and run_dir.parent != run_dir:
        run_dir = run_dir.parent
    manifest = run_dir / "scenario_snapshot" / "manifest.json"
    manifest_hash = hashlib.sha256(manifest.read_bytes()).hexdigest() if manifest.exists() else None
    return {
        "scenario_snapshot": str(manifest.parent) if manifest.exists() else None,
        "scenario_snapshot_manifest": str(manifest) if manifest.exists() else None,
        "scenario_snapshot_manifest_sha256": manifest_hash,
    }


def write_replay_file(
    path: str | Path,
    *,
    run_id: str,
    stage: str,
    episode: int,
    seed: int,
    actions: list[int],
    executed_actions: list[int] | None = None,
    shield_enabled: bool = False,
    risk_checkpoint: str | None = None,
    model_path: str | None = None,
    group_name: str | None = None,
    safety_metric_version: str | None = None,
    notes: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    write_json(
        path,
        {
            "run_id": run_id,
            "stage": stage,
            "episode": int(episode),
            "seed": int(seed),
            "actions": [int(action) for action in actions],
            "executed_actions": (
                [int(action) for action in executed_actions]
                if executed_actions is not None
                else None
            ),
            "shield_enabled": bool(shield_enabled),
            "risk_checkpoint": risk_checkpoint,
            "model_path": model_path,
            "group_name": group_name,
            "safety_metric_version": safety_metric_version,
            "notes": notes or {},
            **_scenario_snapshot_metadata(path, run_id),
        },
    )
