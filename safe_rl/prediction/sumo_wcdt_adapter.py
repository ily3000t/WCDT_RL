from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np

from safe_rl.sim.scenario_semantics import (
    auxiliary_lane_index,
    distance_to_taper,
    is_auxiliary_edge,
    is_ramp_edge,
    is_target_lane,
    lane_center,
    target_lane_index,
)
from safe_rl.sim.history_buffer import HistoryBuffer
from safe_rl.sim.types import VehicleState


class SumoWcDTAdapter:
    """Convert SUMO history buffers into the tensor dict expected by WcDT."""

    def __init__(self, config: Any):
        self.config = config
        self.his_step = int(config.scenario.history_steps)
        self.max_pred_num = int(config.prediction.max_pred_num)
        self.max_other_num = int(config.prediction.max_other_num)
        self.max_traffic_light = int(config.prediction.max_traffic_light)
        self.max_lane_num = int(config.prediction.max_lane_num)
        self.max_point_num = int(config.prediction.max_point_num)
        self.lane_list = self._load_lane_points(Path(config.scenario.net_file))

    def to_wcdt_input(self, history: HistoryBuffer, ego_id: str) -> dict[str, Any]:
        torch = _require_torch()
        agent_history, agent_mask = history.to_tensor_arrays(ego_id)
        latest = history.latest()
        ordered_ids = self._ordered_agent_ids(history, ego_id)
        predicted_ids = [vehicle_id for vehicle_id in ordered_ids if vehicle_id != ego_id]
        predicted_ids = predicted_ids[: self.max_pred_num]
        other_ids = [ego_id] + [
            vehicle_id for vehicle_id in ordered_ids if vehicle_id not in predicted_ids and vehicle_id != ego_id
        ]
        other_ids = other_ids[: self.max_other_num]

        id_to_index = {vehicle_id: idx for idx, vehicle_id in enumerate(history.agent_ids(ego_id))}
        predicted = np.zeros((self.max_pred_num, self.his_step, 5), dtype=np.float32)
        predicted_feature = np.zeros((self.max_pred_num, 7), dtype=np.float32)
        predicted_mask = np.zeros((self.max_pred_num,), dtype=np.float32)
        for row, vehicle_id in enumerate(predicted_ids):
            predicted[row] = agent_history[id_to_index[vehicle_id]]
            predicted_feature[row] = self._vehicle_feature(latest.get(vehicle_id))
            predicted_mask[row] = 1.0

        other = np.zeros((self.max_other_num, self.his_step, 5), dtype=np.float32)
        other_feature = np.zeros((self.max_other_num, 7), dtype=np.float32)
        other_mask = np.zeros((self.max_other_num,), dtype=np.float32)
        for row, vehicle_id in enumerate(other_ids):
            other[row] = agent_history[id_to_index[vehicle_id]]
            other_feature[row] = self._vehicle_feature(latest.get(vehicle_id))
            other_mask[row] = 1.0

        predicted_future = np.zeros((self.max_pred_num, 80, 5), dtype=np.float32)
        predicted_his_traj_delt = predicted[:, 1:] - predicted[:, :-1]
        other_his_traj_delt = other[:, 1:] - other[:, :-1]
        data = {
            "other_his_traj": torch.tensor(other).unsqueeze(0),
            "other_feature": torch.tensor(other_feature).unsqueeze(0),
            "other_traj_mask": torch.tensor(other_mask).unsqueeze(0),
            "other_his_traj_delt": torch.tensor(other_his_traj_delt).unsqueeze(0),
            "other_his_pos": torch.tensor(other[:, -1, :2]).unsqueeze(0),
            "predicted_future_traj": torch.tensor(predicted_future).unsqueeze(0),
            "predicted_his_traj": torch.tensor(predicted).unsqueeze(0),
            "predicted_traj_mask": torch.tensor(predicted_mask).unsqueeze(0),
            "predicted_feature": torch.tensor(predicted_feature).unsqueeze(0),
            "predicted_his_traj_delt": torch.tensor(predicted_his_traj_delt).unsqueeze(0),
            "predicted_his_pos": torch.tensor(predicted[:, -1, :2]).unsqueeze(0),
            "traffic_light": torch.zeros((1, self.max_traffic_light, self.his_step), dtype=torch.float32),
            "traffic_light_pos": torch.zeros((1, self.max_traffic_light, 2), dtype=torch.float32),
            "traffic_mask": torch.zeros((1, self.max_traffic_light), dtype=torch.float32),
            "lane_list": torch.tensor(self.lane_list).unsqueeze(0),
            "predicted_ids": predicted_ids,
        }
        return data

    def _ordered_agent_ids(self, history: HistoryBuffer, ego_id: str) -> list[str]:
        latest = history.latest()
        ego = latest.get(ego_id)
        ids = [vehicle_id for vehicle_id in history.agent_ids(ego_id) if vehicle_id != ego_id]
        if ego is None:
            return ids
        def _priority(vehicle_id: str) -> tuple[float, float, float, str]:
            state = latest.get(vehicle_id)
            if state is None:
                return (9.0, 1.0e6, 1.0e6, vehicle_id)
            dx = float(state.x - ego.x)
            target_lane = target_lane_index(self.config, state.edge_id)
            state_is_target_lane = is_target_lane(self.config, state.edge_id, state.lane_index)
            is_ramp_local = (
                (
                    is_ramp_edge(self.config, state.edge_id)
                    or (
                        is_auxiliary_edge(self.config, state.edge_id)
                        and int(state.lane_index) == auxiliary_lane_index(self.config, state.edge_id)
                    )
                )
                and abs(float(state.lane_pos - ego.lane_pos)) < 80.0
            )
            if state_is_target_lane and dx >= 0.0:
                group = 0
            elif state_is_target_lane and dx < 0.0:
                group = 1
            elif state_is_target_lane:
                group = 2
            elif is_ramp_local and distance_to_taper(self.config, state) > 0.0:
                group = 3
            else:
                group = 4
            target_center = lane_center(self.config, target_lane, state.edge_id, state.lane_pos)
            return (float(group), abs(dx), abs(float(state.y) - target_center), vehicle_id)

        return sorted(ids, key=_priority)

    def _vehicle_feature(self, state: VehicleState | None) -> np.ndarray:
        if state is None:
            return np.zeros((7,), dtype=np.float32)
        type_onehot = [0.0, 1.0, 0.0, 0.0, 0.0]
        return np.asarray([state.width, state.length, *type_onehot], dtype=np.float32)

    def _load_lane_points(self, net_file: Path) -> np.ndarray:
        lanes: list[np.ndarray] = []
        root = ET.parse(net_file).getroot()
        for edge in root.findall("edge"):
            if edge.attrib.get("function") == "internal":
                continue
            for lane in edge.findall("lane"):
                shape = lane.attrib.get("shape", "")
                points = []
                for pair in shape.split():
                    x_text, y_text = pair.split(",")
                    points.append([float(x_text), float(y_text)])
                if not points:
                    continue
                lanes.append(_resample_polyline(np.asarray(points, dtype=np.float32), self.max_point_num))
                if len(lanes) >= self.max_lane_num:
                    break
            if len(lanes) >= self.max_lane_num:
                break
        while len(lanes) < self.max_lane_num:
            lanes.append(np.zeros((self.max_point_num, 2), dtype=np.float32))
        return np.stack(lanes, axis=0).astype(np.float32)


def _resample_polyline(points: np.ndarray, count: int) -> np.ndarray:
    if len(points) == 1:
        return np.repeat(points, count, axis=0)
    distances = [0.0]
    for idx in range(1, len(points)):
        dx = points[idx, 0] - points[idx - 1, 0]
        dy = points[idx, 1] - points[idx - 1, 1]
        distances.append(distances[-1] + math.hypot(float(dx), float(dy)))
    total = distances[-1]
    if total <= 1.0e-6:
        return np.repeat(points[:1], count, axis=0)
    sample_at = np.linspace(0.0, total, count)
    xs = np.interp(sample_at, distances, points[:, 0])
    ys = np.interp(sample_at, distances, points[:, 1])
    return np.stack([xs, ys], axis=-1).astype(np.float32)


def _require_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise ImportError("WcDT adapter requires torch. Activate the SAFE_RL training environment.") from exc
    return torch
