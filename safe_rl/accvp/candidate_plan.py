from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from safe_rl.sim.action_space import CandidateAction


ACCVP_COMMITMENT_PROFILE = "accvp_commitment_v1"


@dataclass(frozen=True)
class CandidatePlan:
    """Nominal ego plan used as the action-conditioned ACCVP model input."""

    action_index: int
    profile: str
    states: np.ndarray


def profile_from_config(config: Any) -> str:
    profile = str(config.accvp.get("candidate_plan_profile", ACCVP_COMMITMENT_PROFILE))
    if profile != ACCVP_COMMITMENT_PROFILE:
        raise ValueError(f"unsupported ACCVP candidate plan profile={profile!r}")
    scenario_profile = str(config.scenario.get("accvp_candidate_plan_profile", ACCVP_COMMITMENT_PROFILE))
    if scenario_profile != profile:
        raise ValueError(
            "scenario.accvp_candidate_plan_profile and accvp.candidate_plan_profile must match; "
            f"got {scenario_profile!r} and {profile!r}"
        )
    return profile


def build_commitment_plan(
    ego: Any,
    action: CandidateAction,
    *,
    step_length: float,
    horizon_steps: int,
    lane_width: float = 3.2,
) -> CandidatePlan:
    """Build the versioned ACCVP nominal commitment input.

    The plan is not a replacement for SUMO rollout dynamics.  It encodes the
    0--0.5 s command, 0.5--1.0 s lateral commitment and the fixed-speed
    continuation assumed by the counterfactual worker.
    """

    horizon_steps = max(1, int(horizon_steps))
    dt = max(float(step_length), 1.0e-6)
    states = np.zeros((horizon_steps, 5), dtype=np.float32)
    x = float(ego.x)
    y = float(ego.y)
    heading = float(ego.heading)
    speed = max(0.0, float(ego.speed))
    command_accel = float(action.accel_cmd) * 1.5
    target_speed = max(0.0, speed + command_accel * 0.5)
    lateral_offset = float(action.lateral_cmd) * float(lane_width)
    for step in range(horizon_steps):
        elapsed = step * dt
        if elapsed < 0.5:
            speed = max(0.0, speed + command_accel * dt)
            accel = command_accel
        else:
            speed = target_speed
            accel = 0.0
        x += speed * dt
        if action.lateral_cmd:
            progress = min(1.0, max(0.0, (elapsed + dt) / 1.0))
            y = float(ego.y) + lateral_offset * progress
        states[step] = np.asarray([x, y, heading, speed, accel], dtype=np.float32)
    return CandidatePlan(int(action.index), ACCVP_COMMITMENT_PROFILE, states)


def apply_branch_command(env: Any, action: CandidateAction, elapsed_s: float) -> bool:
    """Apply one ACCVP branch command without changing legacy env semantics."""

    ego = env._get_ego()
    if ego is None:
        return True
    dt = float(env.step_length)
    if elapsed_s < 0.5:
        target_speed = max(0.0, float(ego.speed) + float(action.accel_cmd) * 1.5 * dt)
    else:
        root_speed = float(getattr(env, "_accvp_branch_target_speed", ego.speed))
        target_speed = root_speed
    env._traci.vehicle.setSpeed(env.ego_id, target_speed)
    if action.lateral_cmd == 0 or elapsed_s >= 1.0:
        return False
    target_lane = int(ego.lane_index) + int(action.lateral_cmd)
    lane_count = env._lane_count(ego.edge_id)
    if target_lane < 0 or target_lane >= lane_count:
        return True
    remaining = max(dt, 1.0 - float(elapsed_s))
    env._traci.vehicle.changeLane(env.ego_id, target_lane, remaining)
    return False
