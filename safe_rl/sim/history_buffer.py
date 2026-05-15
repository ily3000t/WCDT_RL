from __future__ import annotations

from collections import deque
from typing import Iterable

import numpy as np

from safe_rl.sim.types import VehicleState


class HistoryBuffer:
    """Fixed-length vehicle-state history for SUMO observations."""

    def __init__(self, history_steps: int, max_agents: int):
        self.history_steps = int(history_steps)
        self.max_agents = int(max_agents)
        self._frames: deque[dict[str, VehicleState]] = deque(maxlen=self.history_steps)

    def append(self, states: Iterable[VehicleState]) -> None:
        self._frames.append({state.vehicle_id: state for state in states})

    def clear(self) -> None:
        self._frames.clear()

    def __len__(self) -> int:
        return len(self._frames)

    @property
    def ready(self) -> bool:
        return len(self._frames) >= self.history_steps

    def latest(self) -> dict[str, VehicleState]:
        if not self._frames:
            return {}
        return self._frames[-1]

    def agent_ids(self, ego_id: str) -> list[str]:
        latest = self.latest()
        ids = [ego_id] if ego_id in latest else []
        ids.extend(vehicle_id for vehicle_id in sorted(latest) if vehicle_id != ego_id)
        return ids[: self.max_agents]

    def to_tensor_arrays(self, ego_id: str) -> tuple[np.ndarray, np.ndarray]:
        """Return padded arrays shaped [agents, history, 5] and [agents]."""

        agent_ids = self.agent_ids(ego_id)
        agent_count = len(agent_ids)
        history = np.zeros((self.max_agents, self.history_steps, 5), dtype=np.float32)
        mask = np.zeros((self.max_agents,), dtype=np.float32)
        padded_frames = list(self._frames)
        if len(padded_frames) < self.history_steps:
            padding = [padded_frames[0] if padded_frames else {}] * (self.history_steps - len(padded_frames))
            padded_frames = padding + padded_frames
        for agent_idx, vehicle_id in enumerate(agent_ids):
            mask[agent_idx] = 1.0
            last_state = None
            for step_idx, frame in enumerate(padded_frames[-self.history_steps :]):
                state = frame.get(vehicle_id) or last_state
                if state is None:
                    continue
                history[agent_idx, step_idx] = np.asarray(state.as_vector(), dtype=np.float32)
                last_state = state
        if agent_count == 0:
            return history, mask
        return history, mask

    def trajectory_samples(
        self,
        ego_id: str,
        future_frames: list[dict[str, VehicleState]],
        horizon_steps: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        history, mask = self.to_tensor_arrays(ego_id)
        agent_ids = self.agent_ids(ego_id)
        future = np.zeros((self.max_agents, horizon_steps, 5), dtype=np.float32)
        for agent_idx, vehicle_id in enumerate(agent_ids):
            last_state = None
            for step_idx, frame in enumerate(future_frames[:horizon_steps]):
                state = frame.get(vehicle_id) or last_state
                if state is None:
                    continue
                future[agent_idx, step_idx] = np.asarray(state.as_vector(), dtype=np.float32)
                last_state = state
        return history, future, mask
