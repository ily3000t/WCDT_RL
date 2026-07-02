from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from safe_rl.accvp.selection import LEFT_ACTION_IDS
from safe_rl.accvp.schema import write_json_atomic


def _as_int(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _seed_from_path(path: Path) -> int:
    match = re.search(r"_seed_(\d+)\.json$", path.name)
    return int(match.group(1)) if match else -1


def _records_from_replay(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    notes = dict(payload.get("notes", {}) or {})
    episode_report = dict(notes.get("episode_report", {}) or {})
    records = list(notes.get("accvp_records", episode_report.get("accvp_records", [])) or [])
    meta = {
        "path": str(path),
        "seed": _as_int(payload.get("seed", _seed_from_path(path))),
        "group_name": str(payload.get("group_name", "")),
        "accvp_mode": str(notes.get("accvp_mode", episode_report.get("accvp_mode", ""))),
    }
    return meta, records


def _recommended_left(record: dict[str, Any]) -> bool:
    action = record.get("accvp_shadow_recommended_action")
    return action is not None and _as_int(action) in LEFT_ACTION_IDS


def _would_trigger(record: dict[str, Any]) -> bool:
    recommended = record.get("accvp_shadow_recommended_action")
    safety_action = record.get("safety_shield_action")
    selection_reason = str(record.get("accvp_selection_reason", ""))
    reason_ok = selection_reason in {"", "raw_task_infeasible_lite_viable_left"}
    return bool(
        record.get("candidate_set_available", False)
        and _recommended_left(record)
        and recommended is not None
        and _as_int(recommended) != _as_int(safety_action)
        and reason_ok
        and not record.get("accvp_bypass_reason")
        and not record.get("accvp_skip_reason")
    )


def _actual_replacement(record: dict[str, Any]) -> bool:
    reason = str(record.get("accvp_replacement_reason", ""))
    return bool(record.get("accvp_replacement", False) and reason != "lateral_commitment")


def _record_reason(record: dict[str, Any]) -> str:
    if record.get("accvp_bypass_reason"):
        return f"bypass:{record.get('accvp_bypass_reason')}"
    if record.get("accvp_skip_reason"):
        return f"skip:{record.get('accvp_skip_reason')}"
    if str(record.get("accvp_selection_reason", "")) == "raw_action_illegal_or_missing":
        return "raw_illegal_or_missing"
    if _actual_replacement(record):
        return "actual_replacement"
    if _would_trigger(record):
        return "would_trigger"
    if record.get("candidate_set_available") and record.get("raw_feasible"):
        return "raw_feasible"
    if str(record.get("accvp_selection_reason", "")):
        return f"selector:{record.get('accvp_selection_reason')}"
    return "no_candidate_or_noop"


def audit_online_triggers(replay_dirs: list[str | Path], *, group_contains: str | None = None) -> dict[str, Any]:
    files: list[Path] = []
    for replay_dir in replay_dirs:
        root = Path(replay_dir)
        files.extend(sorted(root.rglob("*.json") if root.is_dir() else [root]))
    episode_reports = []
    reason_counts: Counter[str] = Counter()
    targeted_seed_candidates: set[int] = set()
    actual_replacement_seeds: set[int] = set()
    would_trigger_records: list[dict[str, Any]] = []
    actual_records: list[dict[str, Any]] = []
    for path in files:
        meta, records = _records_from_replay(path)
        if group_contains and group_contains not in meta["group_name"]:
            continue
        episode_reasons: Counter[str] = Counter()
        for record in records:
            reason = _record_reason(record)
            reason_counts[reason] += 1
            episode_reasons[reason] += 1
            if _would_trigger(record):
                targeted_seed_candidates.add(meta["seed"])
                would_trigger_records.append(
                    {
                        "seed": meta["seed"],
                        "group_name": meta["group_name"],
                        "step": _as_int(record.get("step")),
                        "decision_index": _as_int(record.get("decision_index")),
                        "raw_action": _as_int(record.get("raw_action")),
                        "safety_shield_action": _as_int(record.get("safety_shield_action")),
                        "recommended_action": _as_int(record.get("accvp_shadow_recommended_action")),
                        "p_merge_improvement": float(record.get("accvp_lite_p_merge_improvement", 0.0)),
                    }
                )
            if _actual_replacement(record):
                actual_replacement_seeds.add(meta["seed"])
                actual_records.append(
                    {
                        "seed": meta["seed"],
                        "group_name": meta["group_name"],
                        "step": _as_int(record.get("step")),
                        "decision_index": _as_int(record.get("decision_index")),
                        "raw_action": _as_int(record.get("raw_action")),
                        "safety_shield_action": _as_int(record.get("safety_shield_action")),
                        "selected_action": _as_int(record.get("accvp_selected_action")),
                        "reason": str(record.get("accvp_replacement_reason", "")),
                    }
                )
        episode_reports.append(
            {
                "seed": meta["seed"],
                "group_name": meta["group_name"],
                "path": meta["path"],
                "record_count": len(records),
                "reason_counts": dict(sorted(episode_reasons.items())),
                "would_trigger_count": int(episode_reasons.get("would_trigger", 0)),
                "actual_replacement_count": int(episode_reasons.get("actual_replacement", 0)),
            }
        )
    targeted_seeds = sorted(seed for seed in targeted_seed_candidates if seed >= 0)
    return {
        "artifact_kind": "accvp_online_trigger_audit_v1",
        "group_filter": group_contains,
        "episode_count": len(episode_reports),
        "record_count": int(sum(item["record_count"] for item in episode_reports)),
        "reason_counts": dict(sorted(reason_counts.items())),
        "would_trigger_count": int(len(would_trigger_records)),
        "actual_replacement_count": int(len(actual_records)),
        "would_trigger_seed_count": int(len(targeted_seeds)),
        "actual_replacement_seed_count": int(len(actual_replacement_seeds)),
        "online_targeted_seeds": targeted_seeds,
        "actual_replacement_seeds": sorted(seed for seed in actual_replacement_seeds if seed >= 0),
        "would_trigger_records": would_trigger_records,
        "actual_replacement_records": actual_records,
        "episodes": episode_reports,
    }


def write_online_trigger_audit(*, output_dir: str | Path, report: dict[str, Any]) -> dict[str, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    return {
        "report": write_json_atomic(output / "accvp_online_trigger_audit.json", report),
        "online_targeted_seeds": write_json_atomic(
            output / "online_targeted_seeds.json",
            {"seeds": report.get("online_targeted_seeds", [])},
        ),
    }
