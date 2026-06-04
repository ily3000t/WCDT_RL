from __future__ import annotations

from typing import Any

import numpy as np

from safe_rl.sim.metrics import bbox_gap, drac, relative_ttc
from safe_rl.sim.types import VehicleState


def modal_to_numpy(prediction: dict[str, Any], mode: int = 0) -> np.ndarray:
    trajectories = prediction.get("future_trajectories")
    if trajectories is None:
        return np.zeros((0, 0, 5), dtype=np.float32)
    if hasattr(trajectories, "detach"):
        trajectories = trajectories.detach().cpu().numpy()
    trajectories = np.asarray(trajectories)
    if trajectories.ndim == 5:
        trajectories = trajectories[0]
    if trajectories.ndim == 4:
        trajectories = trajectories[:, mode]
    return trajectories.astype(np.float32)


def trajectory_to_states(
    trajectory: np.ndarray,
    *,
    reference: VehicleState | None = None,
    dt: float = 0.1,
    vehicle_id: str = "pred",
) -> list[VehicleState]:
    """Convert predicted front-bumper positions into states with derived motion."""

    trajectory = np.asarray(trajectory, dtype=np.float32)
    if trajectory.ndim != 2 or trajectory.shape[0] == 0:
        return []
    previous_x = float(reference.x) if reference is not None else float(trajectory[0, 0])
    previous_y = float(reference.y) if reference is not None else float(trajectory[0, 1])
    previous_heading = float(reference.heading) if reference is not None else 0.0
    states: list[VehicleState] = []
    for step in trajectory:
        x = float(step[0])
        y = float(step[1])
        dx = x - previous_x
        dy = y - previous_y
        distance = float(np.hypot(dx, dy))
        heading = float(np.arctan2(dy, dx)) if distance > 1.0e-6 else previous_heading
        speed = distance / max(float(dt), 1.0e-6)
        states.append(
            VehicleState(
                vehicle_id=vehicle_id,
                x=x,
                y=y,
                heading=heading,
                speed=speed,
                lane_index=int(reference.lane_index) if reference is not None else 0,
                lane_id=str(reference.lane_id) if reference is not None else "",
                lane_pos=float(reference.lane_pos) if reference is not None else 0.0,
                edge_id=str(reference.edge_id) if reference is not None else "",
                length=float(reference.length) if reference is not None else 4.8,
                width=float(reference.width) if reference is not None else 1.8,
            )
        )
        previous_x = x
        previous_y = y
        previous_heading = heading
    return states


def trajectory_risk_summary(
    ego: VehicleState,
    predicted_trajectories: np.ndarray,
    uncertainty: float = 0.0,
) -> np.ndarray:
    if predicted_trajectories.size == 0:
        return np.zeros((8,), dtype=np.float32)
    min_distance = 50.0
    min_ttc = 10.0
    max_drac = 0.0
    collision = 0.0
    for agent_traj in predicted_trajectories:
        for step in agent_traj:
            other = VehicleState(
                vehicle_id="pred",
                x=float(step[0]),
                y=float(step[1]),
                heading=float(step[2]) if len(step) > 2 else 0.0,
                speed=float(np.hypot(step[3], step[4])) if len(step) > 4 else 0.0,
                lane_index=0,
                lane_id="",
                lane_pos=0.0,
                edge_id="",
            )
            gap = bbox_gap(ego, other)
            min_distance = min(min_distance, gap)
            min_ttc = min(min_ttc, relative_ttc(ego, other))
            max_drac = max(max_drac, drac(ego, other))
            collision = max(collision, float(gap <= 0.25))
    return np.asarray(
        [
            min_distance,
            min_ttc,
            max_drac,
            collision,
            float(uncertainty),
            0.0,
            0.0,
            0.0,
        ],
        dtype=np.float32,
    )
