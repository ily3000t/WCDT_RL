from __future__ import annotations

from pathlib import Path
from typing import Any

from safe_rl.utils.io import write_json


def write_replay_file(
    path: str | Path,
    *,
    run_id: str,
    stage: str,
    episode: int,
    seed: int,
    actions: list[int],
    shield_enabled: bool = False,
    risk_checkpoint: str | None = None,
    model_path: str | None = None,
    group_name: str | None = None,
    notes: dict[str, Any] | None = None,
) -> None:
    write_json(
        path,
        {
            "run_id": run_id,
            "stage": stage,
            "episode": int(episode),
            "seed": int(seed),
            "actions": [int(action) for action in actions],
            "shield_enabled": bool(shield_enabled),
            "risk_checkpoint": risk_checkpoint,
            "model_path": model_path,
            "group_name": group_name,
            "notes": notes or {},
        },
    )
