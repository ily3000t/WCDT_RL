from __future__ import annotations

from collections import deque
from typing import Any, Iterable

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

        arrays = self.to_tensor_arrays_with_metadata(ego_id)
        return arrays["history"], arrays["mask"]

    def to_tensor_arrays_with_metadata(self, ego_id: str, cfg: Any | None = None) -> dict[str, np.ndarray]:
        """Return runtime history plus timestep validity and route metadata."""

        agent_ids = self.agent_ids(ego_id)
        history = np.zeros((self.max_agents, self.history_steps, 5), dtype=np.float32)
        mask = np.zeros((self.max_agents,), dtype=np.float32)
        history_valid_mask = np.zeros((self.max_agents, self.history_steps), dtype=np.float32)
        history_lane_index = np.full((self.max_agents, self.history_steps), -1, dtype=np.int64)
        history_edge_role = np.zeros((self.max_agents, self.history_steps), dtype=np.int64)
        padded_frames = list(self._frames)
        if len(padded_frames) < self.history_steps:
            padding = [{}] * (self.history_steps - len(padded_frames))
            padded_frames = padding + padded_frames
        for agent_idx, vehicle_id in enumerate(agent_ids):
            mask[agent_idx] = 1.0
            last_state = None
            for step_idx, frame in enumerate(padded_frames[-self.history_steps :]):
                observed_state = frame.get(vehicle_id)
                state = observed_state or last_state
                if state is None:
                    continue
                history[agent_idx, step_idx] = np.asarray(state.as_vector(), dtype=np.float32)
                if observed_state is not None:
                    history_valid_mask[agent_idx, step_idx] = 1.0
                    history_lane_index[agent_idx, step_idx] = int(observed_state.lane_index)
                    if cfg is not None:
                        from safe_rl.sim.scenario_semantics import edge_role

                        history_edge_role[agent_idx, step_idx] = int(
                            edge_role(cfg, observed_state.edge_id, observed_state.lane_index)
                        )
                last_state = state
        return {
            "history": history,
            "mask": mask,
            "history_valid_mask": history_valid_mask,
            "history_lane_index": history_lane_index,
            "history_edge_role": history_edge_role,
        }

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
            for step_idx, frame in enumerate(future_frames[:horizon_steps]):
                state = frame.get(vehicle_id)
                if state is None:
                    continue
                future[agent_idx, step_idx] = np.asarray(state.as_vector(), dtype=np.float32)
        return history, future, mask
