from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from safe_rl.accvp.schema import COUNTERFACTUAL_SCHEMA_VERSION, canonical_json, file_sha256, stable_hash
from safe_rl.prediction.wcdt_v3_predictor import build_v3_runtime_batch
from safe_rl.sim.action_space import decode_action
from safe_rl.sim.history_buffer import HistoryBuffer
from safe_rl.sim.types import VehicleState


@dataclass
class RootContext:
    """External state required to reconstruct a branch after SUMO loadState()."""

    metadata: dict[str, Any]
    tensors: dict[str, np.ndarray]

    @property
    def root_id(self) -> str:
        return str(self.metadata["root_id"])


def _serialise_history(env: Any) -> list[list[dict[str, Any]]]:
    return [[state.to_dict() for state in frame.values()] for frame in list(env.history._frames)]


def _current_states_from_getters(env: Any) -> list[VehicleState]:
    """Read the exact current TraCI state, never a previous subscription result."""

    enabled = bool(env.config.scenario.get("traci_subscriptions_enabled", True))
    env.config.scenario["traci_subscriptions_enabled"] = False
    try:
        return env._collect_states()
    finally:
        env.config.scenario["traci_subscriptions_enabled"] = enabled


def synchronise_root_state(env: Any) -> None:
    """Make root history's latest frame identical to the state that will be snapshotted."""

    states = _current_states_from_getters(env)
    if not states:
        raise RuntimeError("cannot synchronise ACCVP root: SUMO has no vehicles")
    if not env.history._frames:
        env.history.append(states)
    else:
        env.history._frames[-1] = {state.vehicle_id: state for state in states}
    env._invalidate_decision_cache()


def _tensor_fields(runtime_batch: dict[str, Any]) -> dict[str, np.ndarray]:
    fields = (
        "history_features",
        "baseline",
        "mask",
        "role_ids",
        "lane_ids",
        "edge_role_ids",
        "history_valid_mask",
        "history_lane_ids",
        "history_edge_role_ids",
        "agent_length",
        "agent_width",
        "ego_length",
        "ego_width",
    )
    return {name: np.asarray(runtime_batch[name]) for name in fields}


def capture_root_context(
    env: Any,
    *,
    root_id: str | None = None,
    root_policy: str,
    root_filter: str,
    raw_action_id: int,
    raw_action_legal: bool,
    traffic_profile: str,
    deadline_bin: str,
    snapshot_path: str | Path,
) -> RootContext:
    """Capture root metadata on the collector connection; never calls loadState()."""

    synchronise_root_state(env)
    context = env.get_risk_context()
    ego = context.get("ego")
    if ego is None:
        raise RuntimeError("cannot capture ACCVP root without an ego state")
    runtime = build_v3_runtime_batch(env.config, env.history, env.ego_id)
    selection = runtime["actor_selection"].to_dict()
    selected_actor_ids = [str(value) for value in runtime.get("runtime_agent_ids", [])[1:]]
    critical_actor_ids = [str(value) for value in selection.get("critical_actor_ids", [])]
    selected_actor_coverage_complete = len(selected_actor_ids) >= int(env.config.accvp.actor_count)
    safety_actor_coverage_complete = set(critical_actor_ids).issubset(set(selected_actor_ids)) and not bool(
        selection.get("critical_overflow", False)
    )
    snapshot = Path(snapshot_path)
    root_id = root_id or f"seed{int(env.seed_value)}_decision{int(env._decision_index)}_{uuid.uuid4().hex[:12]}"
    scenario_hash = stable_hash(dict(env.config.scenario))
    metadata = {
        "counterfactual_schema_version": COUNTERFACTUAL_SCHEMA_VERSION,
        "root_id": root_id,
        "root_episode_id": f"{root_policy}:{int(env.seed_value)}",
        "episode_seed": int(env.seed_value),
        "decision_index": int(env._decision_index),
        "sim_step": int(env._episode_step),
        # root_source is retained for schema-v1 consumers; it is exactly the
        # root policy, never a mixed policy/filter encoding.
        "root_source": str(root_policy),
        "root_policy": str(root_policy),
        "root_filter": str(root_filter),
        "raw_action_id": int(raw_action_id),
        "raw_action_name": str(decode_action(raw_action_id).name),
        "raw_action_legal": bool(raw_action_legal),
        "traffic_profile": str(traffic_profile),
        "deadline_bin": str(deadline_bin),
        "snapshot_path": str(snapshot.resolve()),
        "snapshot_sha256": file_sha256(snapshot),
        "scenario_config_hash": scenario_hash,
        "sumo_version": str(env.config.scenario.get("sumo_version", "unknown")),
        "action_execution_profile": str(env.config.scenario.get("action_execution_profile", "current_v1")),
        "candidate_plan_profile": str(env.config.accvp.candidate_plan_profile),
        "step_length": float(env.config.scenario.step_length),
        "candidate_plan_horizon_steps": int(env.config.accvp.candidate_plan_horizon_steps),
        "ego_id": str(env.ego_id),
        "history_frames": _serialise_history(env),
        "selected_actor_ids": selected_actor_ids,
        "selected_actor_count": len(selected_actor_ids),
        "selected_actor_capacity": int(env.config.accvp.actor_count),
        "selected_actor_coverage_complete": bool(selected_actor_coverage_complete),
        "safety_actor_coverage_complete": bool(safety_actor_coverage_complete),
        "selector": selection,
        "root_ego": ego.to_dict(),
        "root_context_hash": "",
    }
    metadata["root_context_hash"] = stable_hash({key: value for key, value in metadata.items() if key != "snapshot_path"})
    return RootContext(metadata=metadata, tensors=_tensor_fields(runtime))


def restore_root_context(env: Any, root: RootContext) -> None:
    """Rebuild collector-only Python state after an independent worker loadState()."""

    env.history.clear()
    for frame in root.metadata["history_frames"]:
        env.history.append([VehicleState(**state) for state in frame])
    env._decision_index = int(root.metadata["decision_index"])
    env._episode_step = int(root.metadata["sim_step"])
    env._lane_count_cache.clear()
    env._invalidate_decision_cache()
    env._reset_subscription_state()
    env._configure_ego_control()
    loaded = _current_states_from_getters(env)
    current_ego = next((state for state in loaded if state.vehicle_id == env.ego_id), None)
    expected = root.metadata["root_ego"]
    if (
        current_ego is None
        # SUMO XML state serialisation rounds lane geometry; values are
        # reproducible per snapshot but can differ from the live connection by
        # a few millimetres after reload.
        or abs(float(current_ego.x) - float(expected["x"])) > 1.0e-2
        or abs(float(current_ego.y) - float(expected["y"])) > 1.0e-2
        or str(current_ego.edge_id) != str(expected["edge_id"])
        or int(current_ego.lane_index) != int(expected["lane_index"])
    ):
        actual = None if current_ego is None else current_ego.to_dict()
        raise RuntimeError(f"SUMO loadState root ego state does not match captured root context: expected={expected}, actual={actual}")
    latest = env.history.latest()
    for actor_id in root.metadata.get("selected_actor_ids", []):
        if str(actor_id) not in latest:
            raise RuntimeError(f"SUMO loadState lost selected ACCVP actor {actor_id!r}")
    env._refresh_vehicle_subscriptions()


def write_root_context(root: RootContext, output_dir: str | Path) -> tuple[Path, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metadata_path = output / f"{root.root_id}.json"
    tensor_path = output / f"{root.root_id}.npz"
    temp_metadata = metadata_path.with_suffix(".json.tmp")
    temp_tensor = tensor_path.with_suffix(".npz.tmp")
    with temp_metadata.open("w", encoding="utf-8") as handle:
        handle.write(canonical_json(root.metadata))
    with temp_tensor.open("wb") as handle:
        np.savez_compressed(handle, **root.tensors)
    temp_metadata.replace(metadata_path)
    temp_tensor.replace(tensor_path)
    return metadata_path, tensor_path


def load_root_context(metadata_path: str | Path, tensor_path: str | Path | None = None) -> RootContext:
    metadata_file = Path(metadata_path)
    with metadata_file.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    arrays = Path(tensor_path) if tensor_path is not None else metadata_file.with_suffix(".npz")
    with np.load(arrays, allow_pickle=False) as values:
        tensors = {key: np.asarray(values[key]) for key in values.files}
    return RootContext(metadata=metadata, tensors=tensors)
