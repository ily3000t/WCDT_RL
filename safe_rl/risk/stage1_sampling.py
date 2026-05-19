from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from safe_rl.risk.merge_local import merge_local_stats, merge_zone_distance
from safe_rl.sim.action_space import ACTIONS


RANDOM = "random"
MERGE_HEURISTIC = "merge_heuristic"
RISK_SEEK = "risk_seek"


def _keep_action(accel_cmd: int) -> int:
    for action in ACTIONS:
        if action.lateral_cmd == 0 and action.accel_cmd == accel_cmd:
            return int(action.index)
    return 4


KEEP_DECELERATE = _keep_action(-1)
KEEP_HOLD = _keep_action(0)
KEEP_ACCELERATE = _keep_action(1)


def configured_sampling_probs(cfg: Any) -> dict[str, float]:
    stage_cfg = cfg.stage1
    raw = stage_cfg.get("sampling_probs", {})
    probs = {
        RANDOM: float(raw.get(RANDOM, 1.0)),
        MERGE_HEURISTIC: float(raw.get(MERGE_HEURISTIC, 0.0)),
        RISK_SEEK: float(raw.get(RISK_SEEK, 0.0)),
    }
    total = sum(max(0.0, value) for value in probs.values())
    if total <= 0.0:
        return {RANDOM: 1.0, MERGE_HEURISTIC: 0.0, RISK_SEEK: 0.0}
    return {key: max(0.0, value) / total for key, value in probs.items()}


def choose_sampling_mode(cfg: Any, rng: np.random.Generator) -> str:
    mode = str(cfg.stage1.get("action_sampling", "random")).lower()
    if mode != "mixed":
        return RANDOM
    probs = configured_sampling_probs(cfg)
    return str(rng.choice(list(probs.keys()), p=list(probs.values())))


def select_stage1_action(
    cfg: Any,
    rng: np.random.Generator,
    context: dict[str, Any],
) -> tuple[int, str]:
    mode = choose_sampling_mode(cfg, rng)
    if mode == MERGE_HEURISTIC:
        return _merge_heuristic_action(cfg, context), mode
    if mode == RISK_SEEK:
        return _risk_seek_action(cfg, rng, context), mode
    return int(rng.integers(0, len(ACTIONS))), RANDOM


def _merge_heuristic_action(cfg: Any, context: dict[str, Any]) -> int:
    ego = context.get("ego")
    if ego is None:
        return KEEP_HOLD
    stats = merge_local_stats(ego, list(context.get("vehicles") or []), cfg)
    if stats.ego_on_ramp and stats.merge_distance > merge_zone_distance(cfg):
        return KEEP_ACCELERATE
    if not stats.in_merge_zone:
        return KEEP_HOLD
    if stats.target_front_gap < 10.0:
        return KEEP_DECELERATE
    if stats.target_rear_gap < 8.0:
        return KEEP_ACCELERATE
    if stats.target_lane_gap < 14.0:
        return KEEP_HOLD
    return KEEP_ACCELERATE


def _risk_seek_action(cfg: Any, rng: np.random.Generator, context: dict[str, Any]) -> int:
    ego = context.get("ego")
    if ego is None:
        return int(rng.integers(0, len(ACTIONS)))
    stats = merge_local_stats(ego, list(context.get("vehicles") or []), cfg)
    if stats.in_merge_zone and stats.target_lane_gap < 18.0:
        return int(rng.choice([KEEP_ACCELERATE, KEEP_ACCELERATE, KEEP_HOLD]))
    if stats.ego_on_ramp and stats.merge_distance <= merge_zone_distance(cfg) * 1.5:
        return int(rng.choice([KEEP_ACCELERATE, KEEP_HOLD, KEEP_DECELERATE]))
    return int(rng.integers(0, len(ACTIONS)))


def sampling_summary(modes: list[str]) -> dict[str, Any]:
    counts = Counter(modes)
    total = max(1, len(modes))
    return {
        "count": len(modes),
        "counts": {mode: int(counts.get(mode, 0)) for mode in (RANDOM, MERGE_HEURISTIC, RISK_SEEK)},
        "proportions": {
            mode: float(counts.get(mode, 0) / total)
            for mode in (RANDOM, MERGE_HEURISTIC, RISK_SEEK)
        },
    }
