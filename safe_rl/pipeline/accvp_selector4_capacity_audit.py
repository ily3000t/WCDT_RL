"""Replay frozen development policies and freeze Selector-v4 capacity.

This command is deliberately selector-only: it does not create new
counterfactual branches or labels. Historical Selector-v3 checkpoints are
used only to reproduce their visited development states; Selector-v4 runs in
shadow on the full current vehicle set.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import time
from typing import Any, Iterable

from safe_rl.accvp.contracts.protocol import effective_activation_distance
from safe_rl.accvp.contracts.schema import (
    file_sha256,
    read_json,
    stable_hash,
    write_json_atomic,
)
from safe_rl.accvp.evaluation.selector_capacity_v4 import (
    SELECTOR4_AUDIT_IMPLEMENTATION,
    SELECTOR4_PROTOCOL_ID,
    build_selector4_capacity_report,
    selection_audit_row,
)
from safe_rl.evaluation_protocol import seeds_for_role
from safe_rl.pipeline.common import make_env
from safe_rl.ppo_factorial import (
    EXPECTED_CANDIDATE_METHOD_ROLES,
    EXPECTED_FINAL_METHOD_ID,
    read_json_mapping,
    resolve_manifest_path,
    validate_factorial_manifest,
)
from safe_rl.prediction.actor_selector import (
    ACTOR_SELECTION_VERSION_V3,
    ACTOR_SELECTION_VERSION_V4,
    actor_relevance_config,
    actor_selection_config_hash,
    reclassify_v3_telemetry_for_selector_v4,
    select_merge_relevant_actors,
)
from safe_rl.rl.ppo import load_ppo
from safe_rl.sim.action_space import ACTIONS
from safe_rl.sim.scenario_semantics import distance_to_taper
from safe_rl.sim.types import VehicleState
from safe_rl.utils.config import REPO_ROOT, clone_with_overrides, load_config


DEVELOPMENT_SEEDS = tuple(range(40001, 40051))
HISTORICAL_FAILURE_SEEDS = (50021, 50027, 55027)
STRESS_SEEDS = {
    "dense": tuple(range(65001, 65006)),
    "aggressive": tuple(range(65006, 65011)),
    "late_taper": tuple(range(65011, 65016)),
}
REPLAY_CACHE_KIND = "accvp_selector4_replay_episode_v1"


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (REPO_ROOT / value).resolve()


def _jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _selector4_config(config: Any) -> Any:
    return clone_with_overrides(
        config,
        {
            "accvp": {
                "actor_relevance": {
                    "version": ACTOR_SELECTION_VERSION_V4,
                    "candidate_conflict_horizon_s": 3.0,
                    "candidate_conflict_surface_gap": 30.0,
                    "actor_longitudinal_accel_bound": 2.0,
                    "local_actor_distance": 45.0,
                    "critical_taper_distance": 120.0,
                }
            }
        },
    )


def _select_row(
    config: Any,
    vehicles: list[VehicleState],
    *,
    state_id: str,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    ego_id = str(config.scenario.get("ego_id", "ego"))
    ego = next(
        (vehicle for vehicle in vehicles if vehicle.vehicle_id == ego_id),
        None,
    )
    if ego is None:
        raise ValueError(f"Selector-v4 audit state has no ego: {state_id}")
    started = time.perf_counter()
    selection = select_merge_relevant_actors(
        config,
        ego,
        vehicles,
        max_actors=12,
        selector_scope="accvp",
    )
    elapsed = time.perf_counter() - started
    return selection_audit_row(
        selection,
        state_id=state_id,
        provenance=provenance,
        selector_latency_s=elapsed,
    )


def _formal_rows(
    audit_config: Any,
    dataset_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    roots_path = dataset_dir / "manifests" / "roots.jsonl"
    if not roots_path.is_file():
        raise FileNotFoundError(roots_path)
    rows: list[dict[str, Any]] = []
    selector3_config = clone_with_overrides(
        audit_config,
        {
            "accvp": {
                "actor_relevance": {
                    "version": ACTOR_SELECTION_VERSION_V3,
                }
            }
        },
    )
    expected_selector3_hash = actor_selection_config_hash(
        selector3_config, selector_scope="accvp"
    )
    complete_roots = [
        row for row in _jsonl(roots_path) if bool(row.get("complete", False))
    ]
    for index, root in enumerate(complete_roots, start=1):
        metadata_path = _resolve(str(root["metadata_path"]))
        metadata = read_json(metadata_path)
        frames = list(metadata.get("history_frames", []) or [])
        if not frames:
            raise ValueError(
                f"formal selector audit root has no full history: {root['root_id']}"
            )
        selected_count = int(metadata.get("selected_actor_count", 0))
        latest = list(frames[-1] or [])
        if len(latest) <= selected_count + 1:
            raise ValueError(
                "formal selector audit cannot prove capacity from selected-only rows: "
                f"{root['root_id']}"
            )
        vehicles = [VehicleState(**vehicle) for vehicle in latest]
        ego = next(
            (
                vehicle
                for vehicle in vehicles
                if vehicle.vehicle_id == str(audit_config.scenario.ego_id)
            ),
            None,
        )
        if ego is None:
            raise ValueError(
                f"formal selector audit root has no ego: {root['root_id']}"
            )
        selector_payload = dict(metadata.get("selector", {}) or {})
        if str(selector_payload.get("config_hash", "")) != (
            expected_selector3_hash
        ):
            raise ValueError(
                "formal Selector-v3 telemetry hash mismatch: "
                f"root={root['root_id']} expected={expected_selector3_hash} "
                f"actual={selector_payload.get('config_hash', '')}"
            )
        expected_actor_ids = {
            vehicle.vehicle_id
            for vehicle in vehicles
            if vehicle.vehicle_id != str(audit_config.scenario.ego_id)
        }
        telemetry_actor_ids = set(
            dict(selector_payload.get("actor_metadata", {}) or {})
        )
        if telemetry_actor_ids != expected_actor_ids:
            raise ValueError(
                "formal Selector-v3 telemetry does not cover the full current "
                f"actor set: root={root['root_id']} "
                f"missing={sorted(expected_actor_ids - telemetry_actor_ids)} "
                f"extra={sorted(telemetry_actor_ids - expected_actor_ids)}"
            )
        taper_distance_m = float(
            metadata.get(
                "taper_distance_m",
                distance_to_taper(audit_config, ego),
            )
        )
        started = time.perf_counter()
        selection = reclassify_v3_telemetry_for_selector_v4(
            audit_config,
            selector_payload,
            ego_taper_distance=taper_distance_m,
            max_actors=12,
        )
        elapsed = time.perf_counter() - started
        rows.append(
            selection_audit_row(
                selection,
                state_id=str(root["root_id"]),
                provenance={
                    "scope": "formal_root_history",
                    "episode_seed": int(root.get("episode_seed", -1)),
                    "decision_index": int(metadata.get("decision_index", -1)),
                    "traffic_profile": str(
                        root.get(
                            "traffic_profile",
                            metadata.get("traffic_profile", "unknown"),
                        )
                    ),
                    "root_policy": str(root.get("root_policy", "unknown")),
                    "taper_distance_m": taper_distance_m,
                },
                selector_latency_s=elapsed,
                latency_semantics="v3_telemetry_reclassification",
            )
        )
        if index % 500 == 0 or index == len(complete_roots):
            print(
                "[selector4_audit] formal "
                f"states={index}/{len(complete_roots)}",
                flush=True,
            )
    return rows, {
        "dataset": str(dataset_dir),
        "roots_manifest_sha256": file_sha256(roots_path),
        "complete_root_count": len(complete_roots),
    }


def _factorial_records(
    manifest_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    factorial = validate_factorial_manifest(
        manifest_path,
        require_complete=True,
        verify_files=True,
    )
    records: list[dict[str, Any]] = []
    for method_id in EXPECTED_CANDIDATE_METHOD_ROLES:
        method_entry = dict(factorial["methods"][method_id])
        child_path = resolve_manifest_path(
            manifest_path,
            str(method_entry["replicate_manifest"]),
        )
        child = read_json_mapping(child_path)
        child_records = [dict(row) for row in child.get("records", [])]
        seeds = sorted(int(row["optimizer_seed"]) for row in child_records)
        if seeds != [1001, 1002, 1003, 1004, 1005]:
            raise ValueError(
                f"Selector-v4 audit requires five {method_id} replicas"
            )
        records.extend(child_records)
    return records, {
        "factorial_manifest": str(manifest_path),
        "factorial_manifest_sha256": file_sha256(manifest_path),
        "factorial_manifest_fingerprint": str(
            factorial.get("manifest_fingerprint", "")
        ),
        "usage": "diagnostic_state_source_only",
    }


def _recorded_overflow_rows(
    diagnostic_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load exact historical states that isolated seed replay cannot prove."""

    diagnostic = read_json(diagnostic_path)
    fingerprint = str(diagnostic.get("diagnostic_fingerprint", ""))
    expected = stable_hash(
        {
            key: value
            for key, value in diagnostic.items()
            if key != "diagnostic_fingerprint"
        }
    )
    if not fingerprint or fingerprint != expected:
        raise ValueError("historical overflow diagnostic fingerprint mismatch")
    if str(diagnostic.get("artifact_kind", "")) != (
        "accvp_selector_overflow_shadow_diagnostic_v1"
    ):
        raise ValueError("unexpected historical overflow diagnostic kind")
    required_flags = (
        "diagnostic_only",
        "examples_complete",
        "reconstruction_complete",
        "protected_coverage_complete",
    )
    if not all(bool(diagnostic.get(key, False)) for key in required_flags):
        raise ValueError("historical overflow diagnostic is incomplete")
    if str(diagnostic.get("frozen_selector_version", "")) != (
        ACTOR_SELECTION_VERSION_V3
    ):
        raise ValueError("historical overflow diagnostic selector mismatch")

    runtime_path = _resolve(str(diagnostic["source_runtime_report"]))
    if file_sha256(runtime_path) != str(
        diagnostic.get("source_runtime_report_sha256", "")
    ):
        raise ValueError("historical runtime report SHA-256 mismatch")
    runtime = read_json(runtime_path)
    if str(runtime.get("report_fingerprint", "")) != str(
        diagnostic.get("source_runtime_report_fingerprint", "")
    ):
        raise ValueError("historical runtime report fingerprint mismatch")

    examples: dict[tuple[int, int, int], tuple[dict[str, Any], str]] = {}
    for episode in list(runtime.get("episodes", []) or []):
        episode_seed = int(episode.get("episode_seed", episode.get("seed", -1)))
        traffic_profile = str(episode.get("curriculum_profile", "unknown"))
        for value in list(
            episode.get("accvp_table_critical_actor_overflow_examples", []) or []
        ):
            example = dict(value)
            key = (
                int(example.get("episode_seed", episode_seed)),
                int(example.get("optimizer_seed", -1)),
                int(example.get("decision_index", -1)),
            )
            if key in examples:
                raise ValueError(f"duplicate historical overflow state: {key}")
            examples[key] = (example, traffic_profile)

    rows: list[dict[str, Any]] = []
    diagnostic_keys: set[tuple[int, int, int]] = set()
    for diagnostic_row in list(diagnostic.get("rows", []) or []):
        key = (
            int(diagnostic_row.get("episode_seed", -1)),
            int(diagnostic_row.get("optimizer_seed", -1)),
            int(diagnostic_row.get("decision_index", -1)),
        )
        diagnostic_keys.add(key)
        if key not in examples:
            raise ValueError(
                f"historical overflow diagnostic row has no source example: {key}"
            )
        example, traffic_profile = examples[key]
        actor_map = {
            str(actor.get("vehicle_id", "")): dict(actor)
            for actor in list(example.get("critical_actors", []) or [])
        }
        critical_ids = [
            str(value)
            for value in diagnostic_row.get("lane_aware_critical_ids", []) or []
        ]
        if not critical_ids or any(
            vehicle_id not in actor_map for vehicle_id in critical_ids
        ):
            raise ValueError(
                f"historical overflow actor reconstruction failed: {key}"
            )
        retained = [actor_map[vehicle_id] for vehicle_id in critical_ids]

        def _reasons(actor: dict[str, Any]) -> set[str]:
            raw = actor.get("trigger_reasons", []) or []
            values = raw.split() if isinstance(raw, str) else list(raw)
            return {str(value) for value in values}

        target_ids = [
            str(actor["vehicle_id"])
            for actor in retained
            if str(actor.get("role", "")) in {"target_front", "target_rear"}
        ]
        conflict_ids = [
            str(actor["vehicle_id"])
            for actor in retained
            if bool(actor.get("candidate_conflict_eligible", False))
        ]
        nearest_ids = [
            str(actor["vehicle_id"])
            for actor in retained
            if bool(actor.get("nearest_candidate_conflict", False))
        ]
        lowest_ids = [
            str(actor["vehicle_id"])
            for actor in retained
            if "lowest_ttc" in _reasons(actor)
        ]
        protected_ids = [
            str(value)
            for value in diagnostic_row.get("protected_actor_ids", []) or []
        ]
        if not set(protected_ids).issubset(set(critical_ids)):
            raise ValueError(
                f"historical overflow lost a protected actor: {key}"
            )
        rows.append(
            {
                "state_id": (
                    "recorded_runtime_overflow:"
                    f"optimizer_{key[1]}:seed_{key[0]}:decision_{key[2]}"
                ),
                "provenance": {
                    "scope": "recorded_historical_overflow_telemetry",
                    "method_id": "candidate_table_reward_v2_commitment",
                    "optimizer_seed": key[1],
                    "episode_seed": key[0],
                    "decision_index": key[2],
                    "traffic_profile": traffic_profile,
                    "stress_profile": "",
                    "taper_distance_m": float(
                        diagnostic_row.get("taper_distance_m", float("inf"))
                    ),
                    "seed_role": "selector_development_only",
                },
                "selector_latency_s": 0.0,
                "latency_semantics": "recorded_runtime_telemetry",
                "critical_count": len(critical_ids),
                "contextual_count": 0,
                "contextual_count_observed": False,
                "critical_actor_ids": critical_ids,
                "target_front_rear_ids": target_ids,
                "candidate_conflict_ids": conflict_ids,
                "nearest_conflict_ids": nearest_ids,
                "lowest_conflict_ttc_ids": lowest_ids,
                "protected_actor_ids": protected_ids,
                "critical_actors": retained,
            }
        )
    reported_count = int(diagnostic.get("reported_overflow_count", -1))
    if len(rows) != reported_count or set(examples) != diagnostic_keys:
        raise ValueError("historical overflow telemetry coverage mismatch")
    return rows, {
        "diagnostic_report": str(diagnostic_path),
        "diagnostic_report_sha256": file_sha256(diagnostic_path),
        "diagnostic_fingerprint": fingerprint,
        "runtime_report": str(runtime_path),
        "runtime_report_sha256": file_sha256(runtime_path),
        "runtime_report_fingerprint": str(runtime.get("report_fingerprint", "")),
        "recorded_overflow_state_count": len(rows),
        "reported_overflow_count": reported_count,
        "episode_seeds": sorted(
            {int(row["provenance"]["episode_seed"]) for row in rows}
        ),
        "optimizer_seeds": sorted(
            {int(row["provenance"]["optimizer_seed"]) for row in rows}
        ),
        "usage": "exact_historical_overflow_state_telemetry",
    }


def _stress_override(config: Any, profile: str) -> Any:
    if profile == "aggressive":
        return clone_with_overrides(
            config,
            {
                "stage1": {
                    "curriculum": {
                        "enabled": True,
                        "profiles": {
                            "aggressive_kinematic": {
                                "probability": 1.0,
                                "position_jitter": 20.0,
                                "speed_jitter": 6.0,
                            }
                        },
                    }
                }
            },
        )
    return config


def _cache_path(
    cache_root: Path,
    *,
    scope: str,
    method_id: str,
    optimizer_seed: int,
    simulator_seed: int,
    stress_profile: str,
) -> Path:
    profile = stress_profile or "none"
    return (
        cache_root
        / scope
        / method_id
        / f"optimizer_{optimizer_seed}"
        / profile
        / f"seed_{simulator_seed}.json"
    )


def _episode_identity(
    record: dict[str, Any],
    *,
    scope: str,
    simulator_seed: int,
    stress_profile: str,
    selector_hash: str,
) -> dict[str, Any]:
    return {
        "method_id": str(record["method_id"]),
        "optimizer_seed": int(record["optimizer_seed"]),
        "simulator_seed": int(simulator_seed),
        "scope": str(scope),
        "stress_profile": str(stress_profile),
        "checkpoint_sha256": str(record["checkpoint_sha256"]),
        "resolved_config_sha256": str(record["resolved_config_sha256"]),
        "selector_config_hash": str(selector_hash),
    }


def _read_cached_episode(path: Path, identity: dict[str, Any]) -> dict[str, Any]:
    payload = read_json(path)
    if str(payload.get("artifact_kind", "")) != REPLAY_CACHE_KIND:
        raise ValueError(f"unexpected selector replay cache: {path}")
    if dict(payload.get("identity", {})) != identity:
        raise ValueError(f"selector replay cache identity mismatch: {path}")
    fingerprint = str(payload.get("report_fingerprint", ""))
    expected = stable_hash(
        {
            key: value
            for key, value in payload.items()
            if key != "report_fingerprint"
        }
    )
    if not fingerprint or fingerprint != expected:
        raise ValueError(f"selector replay cache fingerprint mismatch: {path}")
    return payload


def _run_replay_episode(
    record: dict[str, Any],
    simulator_seed: int,
    *,
    scope: str,
    stress_profile: str,
    cache_root: Path,
    cfg: Any | None = None,
    selector_cfg: Any | None = None,
    env: Any | None = None,
    model: Any | None = None,
) -> dict[str, Any]:
    config_path = _resolve(str(record["resolved_config"]))
    checkpoint_path = _resolve(str(record["checkpoint"]))
    if file_sha256(config_path) != str(record["resolved_config_sha256"]):
        raise ValueError("selector replay resolved-config SHA-256 mismatch")
    if file_sha256(checkpoint_path) != str(record["checkpoint_sha256"]):
        raise ValueError("selector replay checkpoint SHA-256 mismatch")
    cfg = (
        _stress_override(load_config(config_path), stress_profile)
        if cfg is None
        else cfg
    )
    selector_cfg = _selector4_config(cfg) if selector_cfg is None else selector_cfg
    selector_hash = actor_selection_config_hash(
        selector_cfg,
        selector_scope="accvp",
    )
    identity = _episode_identity(
        record,
        scope=scope,
        simulator_seed=simulator_seed,
        stress_profile=stress_profile,
        selector_hash=selector_hash,
    )
    output = _cache_path(
        cache_root,
        scope=scope,
        method_id=str(record["method_id"]),
        optimizer_seed=int(record["optimizer_seed"]),
        simulator_seed=simulator_seed,
        stress_profile=stress_profile,
    )
    if output.is_file():
        return _read_cached_episode(output, identity)

    owns_env = env is None
    if env is None:
        env = make_env(cfg, seed=int(simulator_seed), shield_enabled=False)
        if stress_profile == "dense":
            # Mutate only after artifact validation: this is a diagnostic
            # SUMO workload control, not a claim that the old ACCVP bundle
            # was trained under the denser scenario contract.
            env.config.scenario["sumo_scale"] = 1.5
    if model is None:
        model = load_ppo(checkpoint_path, device="cpu")
    activation_distance = effective_activation_distance(cfg)
    neutral_action = next(
        action.index
        for action in ACTIONS
        if action.name == "keep_hold"
    )
    rows: list[dict[str, Any]] = []
    try:
        observation, _info = env.reset(seed=int(simulator_seed))
        terminated = truncated = False
        decision_index = 0
        while not (terminated or truncated):
            latest = list(env.history.latest().values())
            ego = next(
                (
                    vehicle
                    for vehicle in latest
                    if vehicle.vehicle_id == str(cfg.scenario.ego_id)
                ),
                None,
            )
            taper_distance = (
                float(distance_to_taper(cfg, ego))
                if ego is not None
                else float("inf")
            )
            context = env.get_risk_context()
            if latest and 0.0 < taper_distance <= activation_distance:
                rows.append(
                    _select_row(
                        selector_cfg,
                        [
                            VehicleState(**vehicle.to_dict())
                            for vehicle in latest
                        ],
                        state_id=(
                            f"{record['method_id']}:optimizer_"
                            f"{record['optimizer_seed']}:simulator_"
                            f"{simulator_seed}:{stress_profile or 'development'}:"
                            f"decision_{decision_index}"
                        ),
                        provenance={
                            "scope": str(scope),
                            "method_id": str(record["method_id"]),
                            "optimizer_seed": int(record["optimizer_seed"]),
                            "episode_seed": int(simulator_seed),
                            "decision_index": int(decision_index),
                            "traffic_profile": str(
                                context.get("curriculum_profile", "disabled")
                            ),
                            "stress_profile": str(stress_profile),
                            "taper_distance_m": taper_distance,
                            "seed_role": "selector_development_only",
                        },
                    )
                )
            action, _state = model.predict(observation, deterministic=True)
            if stress_profile == "late_taper" and taper_distance > 55.0:
                action = neutral_action
            observation, _reward, terminated, truncated, _info = env.step(
                int(action)
            )
            decision_index += 1
    finally:
        if owns_env:
            env.close()
    payload = {
        "artifact_kind": REPLAY_CACHE_KIND,
        "implementation_version": SELECTOR4_AUDIT_IMPLEMENTATION,
        "identity": identity,
        "activation_distance_m": float(activation_distance),
        "decision_state_count": len(rows),
        "rows": rows,
    }
    payload["report_fingerprint"] = stable_hash(payload)
    write_json_atomic(output, payload)
    return payload


def _task(payload: dict[str, Any]) -> dict[str, Any]:
    record = dict(payload["record"])
    scope = str(payload["scope"])
    stress_profile = str(payload.get("stress_profile", ""))
    cache_root = Path(str(payload["cache_root"]))
    config_path = _resolve(str(record["resolved_config"]))
    checkpoint_path = _resolve(str(record["checkpoint"]))
    cfg = _stress_override(load_config(config_path), stress_profile)
    selector_cfg = _selector4_config(cfg)
    selector_hash = actor_selection_config_hash(
        selector_cfg, selector_scope="accvp"
    )
    cached: list[dict[str, Any]] = []
    all_cached = True
    for simulator_seed in list(payload["simulator_seeds"]):
        identity = _episode_identity(
            record,
            scope=scope,
            simulator_seed=int(simulator_seed),
            stress_profile=stress_profile,
            selector_hash=selector_hash,
        )
        output = _cache_path(
            cache_root,
            scope=scope,
            method_id=str(record["method_id"]),
            optimizer_seed=int(record["optimizer_seed"]),
            simulator_seed=int(simulator_seed),
            stress_profile=stress_profile,
        )
        if not output.is_file():
            all_cached = False
            break
        cached.append(_read_cached_episode(output, identity))
    if all_cached:
        return {"episodes": cached}

    env = make_env(
        cfg,
        seed=int(list(payload["simulator_seeds"])[0]),
        shield_enabled=False,
    )
    if stress_profile == "dense":
        env.config.scenario["sumo_scale"] = 1.5
    model = load_ppo(checkpoint_path, device="cpu")
    episodes: list[dict[str, Any]] = []
    try:
        for simulator_seed in list(payload["simulator_seeds"]):
            episodes.append(
                _run_replay_episode(
                    record,
                    int(simulator_seed),
                    scope=scope,
                    stress_profile=stress_profile,
                    cache_root=cache_root,
                    cfg=cfg,
                    selector_cfg=selector_cfg,
                    env=env,
                    model=model,
                )
            )
    finally:
        env.close()
    return {"episodes": episodes}


def _replay_tasks(
    records: list[dict[str, Any]],
    cache_root: Path,
) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for record in records:
        cfg = load_config(_resolve(str(record["resolved_config"])))
        actual_development = tuple(
            int(seed) for seed in seeds_for_role(cfg, "stage3_selection")
        )
        if actual_development != DEVELOPMENT_SEEDS:
            raise ValueError(
                "historical Stage3 selection cohort is not the frozen "
                f"40001-40050 range: {actual_development}"
            )
        tasks.append(
            {
                "record": record,
                "simulator_seeds": list(DEVELOPMENT_SEEDS),
                "scope": "selected_checkpoint_development",
                "stress_profile": "",
                "cache_root": str(cache_root),
            }
        )
        tasks.append(
            {
                "record": record,
                "simulator_seeds": list(HISTORICAL_FAILURE_SEEDS),
                "scope": "historical_overflow_development",
                "stress_profile": "",
                "cache_root": str(cache_root),
            }
        )
        if str(record["method_id"]) == EXPECTED_FINAL_METHOD_ID:
            for profile, seeds in STRESS_SEEDS.items():
                tasks.append(
                    {
                        "record": record,
                        "simulator_seeds": list(seeds),
                        "scope": "selector_stress_development",
                        "stress_profile": profile,
                        "cache_root": str(cache_root),
                    }
                )
    return tasks


def _execute_tasks(
    tasks: list[dict[str, Any]],
    *,
    workers: int,
) -> list[dict[str, Any]]:
    completed: list[dict[str, Any]] = []
    if workers <= 1:
        for index, task in enumerate(tasks, start=1):
            completed.extend(_task(task)["episodes"])
            print(
                f"[selector4_audit] replay groups={index}/{len(tasks)}",
                flush=True,
            )
        return completed
    with ProcessPoolExecutor(max_workers=int(workers)) as executor:
        futures = {executor.submit(_task, task): task for task in tasks}
        for index, future in enumerate(as_completed(futures), start=1):
            completed.extend(future.result()["episodes"])
            print(
                f"[selector4_audit] replay groups={index}/{len(tasks)}",
                flush=True,
            )
    return completed


def _source_coverage(
    formal_count: int,
    episodes: list[dict[str, Any]],
    recorded_overflow_rows: list[dict[str, Any]],
    recorded_overflow_lineage: dict[str, Any],
) -> dict[str, bool]:
    identities = [dict(item["identity"]) for item in episodes]
    development = [
        row
        for row in identities
        if row["scope"] == "selected_checkpoint_development"
    ]
    historical = [
        row
        for row in identities
        if row["scope"] == "historical_overflow_development"
    ]
    stress = [
        row
        for row in identities
        if row["scope"] == "selector_stress_development"
    ]
    methods = set(EXPECTED_CANDIDATE_METHOD_ROLES)
    optimizers = {1001, 1002, 1003, 1004, 1005}
    development_keys = {
        (
            str(row["method_id"]),
            int(row["optimizer_seed"]),
            int(row["simulator_seed"]),
        )
        for row in development
    }
    historical_keys = {
        (
            str(row["method_id"]),
            int(row["optimizer_seed"]),
            int(row["simulator_seed"]),
        )
        for row in historical
    }
    stress_keys = {
        (
            int(row["optimizer_seed"]),
            str(row["stress_profile"]),
            int(row["simulator_seed"]),
        )
        for row in stress
    }
    return {
        "formal_5000_full_state_roots": formal_count >= 5000,
        "four_candidate_methods_present": {
            str(row["method_id"]) for row in development
        }
        == methods,
        "five_optimizer_replicates_per_method": all(
            {
                int(row["optimizer_seed"])
                for row in development
                if str(row["method_id"]) == method
            }
            == optimizers
            for method in methods
        ),
        "all_selected_checkpoint_development_episodes": len(
            development_keys
        )
        == 4 * 5 * 50,
        "historical_failure_seeds_development_only": len(historical_keys)
        == 4 * 5 * 3,
        "dense_aggressive_late_taper_stress_complete": len(stress_keys)
        == 5 * 3 * 5,
        "every_replay_episode_has_activation_states": all(
            int(item.get("decision_state_count", 0)) > 0 for item in episodes
        ),
        "recorded_historical_overflow_telemetry_complete": bool(
            recorded_overflow_rows
            and len(recorded_overflow_rows)
            == int(recorded_overflow_lineage.get("reported_overflow_count", -1))
            and max(
                int(row.get("critical_count", 0))
                for row in recorded_overflow_rows
            )
            >= 10
        ),
    }


def run(
    *,
    config_path: str | Path,
    dataset_dir: str | Path,
    factorial_manifest: str | Path,
    historical_overflow_report: str | Path,
    output_path: str | Path,
    cache_root: str | Path,
    workers: int = 2,
) -> Path:
    output = _resolve(output_path)
    if output.exists():
        raise FileExistsError(output)
    audit_cfg = load_config(_resolve(config_path))
    if str(audit_cfg.evaluation_protocol.protocol_id) != SELECTOR4_PROTOCOL_ID:
        raise ValueError("Selector-v4 audit config protocol mismatch")
    if (
        actor_relevance_config(audit_cfg, selector_scope="accvp")["version"]
        != ACTOR_SELECTION_VERSION_V4
    ):
        raise ValueError("Selector-v4 audit config uses the wrong selector")
    formal, formal_lineage = _formal_rows(
        audit_cfg,
        _resolve(dataset_dir),
    )
    records, factorial_lineage = _factorial_records(
        _resolve(factorial_manifest)
    )
    recorded_overflow, recorded_overflow_lineage = _recorded_overflow_rows(
        _resolve(historical_overflow_report)
    )
    tasks = _replay_tasks(records, _resolve(cache_root))
    episodes = _execute_tasks(tasks, workers=max(1, int(workers)))
    replay_rows = [
        dict(row)
        for episode in episodes
        for row in list(episode.get("rows", []) or [])
    ]
    coverage = _source_coverage(
        len(formal),
        episodes,
        recorded_overflow,
        recorded_overflow_lineage,
    )
    report = build_selector4_capacity_report(
        [*formal, *replay_rows, *recorded_overflow],
        source_coverage=coverage,
        selector_config={
            "version": ACTOR_SELECTION_VERSION_V4,
            "resolved": actor_relevance_config(
                audit_cfg,
                selector_scope="accvp",
            ),
            "config_hash": actor_selection_config_hash(
                audit_cfg,
                selector_scope="accvp",
            ),
        },
        source_lineage={
            "formal": formal_lineage,
            "factorial": factorial_lineage,
            "recorded_historical_overflow": recorded_overflow_lineage,
            "replay_cache_root": str(_resolve(cache_root)),
            "replay_episode_count": len(episodes),
            "replay_state_count": len(replay_rows),
            "historical_failure_seeds": list(HISTORICAL_FAILURE_SEEDS),
            "historical_failure_seed_role": "selector_development_only",
            "stress_seeds": {
                key: list(value) for key, value in STRESS_SEEDS.items()
            },
            "stress_semantics": {
                "dense": "SUMO --scale 1.5",
                "aggressive": (
                    "forced seed-vehicle position jitter 20m and speed "
                    "jitter 6m/s; not a driver-model aggressiveness claim"
                ),
                "late_taper": (
                    "keep_hold until taper distance <=55m, then frozen policy"
                ),
            },
        },
    )
    write_json_atomic(output, report)
    print(
        "[selector4_audit] "
        f"state={report['audit_state']} "
        f"selected_capacity={report['selected_capacity']} "
        f"states={report['overall_distribution']['state_count']} "
        f"report={output}",
        flush=True,
    )
    if str(report["audit_state"]) != "pass":
        raise RuntimeError(
            "Selector-v4 capacity audit is blocked; do not collect pilot data"
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Audit lane-aware Selector-v4 on 5,000 formal roots, all four "
            "Candidate methods x five replicas x frozen development seeds, "
            "historical overflow seeds, and three development stress profiles."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--factorial-manifest", required=True)
    parser.add_argument("--historical-overflow-report", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    run(
        config_path=args.config,
        dataset_dir=args.dataset,
        factorial_manifest=args.factorial_manifest,
        historical_overflow_report=args.historical_overflow_report,
        output_path=args.output,
        cache_root=args.cache_root,
        workers=int(args.workers),
    )


if __name__ == "__main__":
    main()
