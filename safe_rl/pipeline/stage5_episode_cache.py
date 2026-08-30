from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from safe_rl.accvp.contracts.schema import file_sha256, stable_hash
from safe_rl.risk.risk_aggregator import aggregate_episode_reports
from safe_rl.utils.io import json_ready


CACHE_BINDING_KIND = "stage5_episode_cache_binding_v1"
CACHE_IDENTITY_KIND = "stage5_episode_cache_identity_v1"
CACHE_EPISODE_KIND = "stage5_episode_cache_record_v1"
CACHE_SCHEMA_VERSION = 1


def _resolve(path: str | Path) -> Path:
    return Path(path).resolve()


def _normalise_sha256(value: Any, *, field: str) -> str:
    digest = str(value or "").strip().lower()
    if len(digest) != 64:
        raise ValueError(f"{field} must be a 64-character SHA-256 digest")
    try:
        int(digest, 16)
    except ValueError as exc:
        raise ValueError(f"{field} must be a hexadecimal SHA-256 digest") from exc
    return digest


def _validate_binding(
    binding: Mapping[str, Any],
    *,
    model_path: str | Path | None,
    risk_checkpoint: str | Path | None,
) -> tuple[Path, dict[str, Any], str]:
    if str(binding.get("artifact_kind", "")) != CACHE_BINDING_KIND:
        raise ValueError("unsupported Stage5 episode-cache binding")
    if int(binding.get("schema_version", -1)) != CACHE_SCHEMA_VERSION:
        raise ValueError("unsupported Stage5 episode-cache binding schema")
    identity = dict(binding.get("identity", {}) or {})
    if not identity:
        raise ValueError("Stage5 episode-cache binding is missing identity")
    fingerprint = _normalise_sha256(
        binding.get("execution_fingerprint"),
        field="Stage5 episode-cache execution_fingerprint",
    )
    if stable_hash(identity) != fingerprint:
        raise ValueError("Stage5 episode-cache binding fingerprint mismatch")
    if not str(binding.get("cache_dir", "")).strip():
        raise ValueError("Stage5 episode-cache binding is missing cache_dir")
    cache_dir = _resolve(str(binding["cache_dir"]))

    expected_checkpoint = _normalise_sha256(
        identity.get("checkpoint_sha256"),
        field="Stage5 episode-cache checkpoint_sha256",
    )
    if model_path is None or not Path(model_path).is_file():
        raise FileNotFoundError(model_path or "")
    if file_sha256(model_path) != expected_checkpoint:
        raise ValueError("Stage5 episode-cache checkpoint hash mismatch")

    shield_enabled = bool(identity.get("shield_enabled", False))
    if shield_enabled:
        expected_risk = _normalise_sha256(
            identity.get("risk_checkpoint_sha256"),
            field="Stage5 episode-cache risk_checkpoint_sha256",
        )
        if risk_checkpoint is None or not Path(risk_checkpoint).is_file():
            raise FileNotFoundError(risk_checkpoint or "")
        if file_sha256(risk_checkpoint) != expected_risk:
            raise ValueError("Stage5 episode-cache Risk checkpoint hash mismatch")
    elif str(identity.get("risk_checkpoint_sha256", "")):
        raise ValueError(
            "unshielded Stage5 episode-cache identity must not bind a Risk checkpoint"
        )
    elif risk_checkpoint is not None:
        raise ValueError("unshielded Stage5 episode cache received a Risk checkpoint")
    return cache_dir, identity, fingerprint


def _write_new_or_validate(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ready = json_ready(dict(payload))
    if path.is_file():
        with path.open("r", encoding="utf-8") as handle:
            existing = json.load(handle)
        if stable_hash(existing) != stable_hash(ready):
            raise ValueError(f"refusing to replace a different Stage5 cache artifact: {path}")
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(
                ready,
                handle,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
                sort_keys=True,
            )
        try:
            os.link(temporary, path)
        except FileExistsError:
            with path.open("r", encoding="utf-8") as handle:
                existing = json.load(handle)
            if stable_hash(existing) != stable_hash(ready):
                raise ValueError(
                    f"concurrent Stage5 cache output differs for immutable artifact: {path}"
                )
    finally:
        temporary.unlink(missing_ok=True)


def _identity_artifact(
    cache_dir: Path,
    *,
    identity: Mapping[str, Any],
    fingerprint: str,
) -> Path:
    path = cache_dir / "identity.json"
    _write_new_or_validate(
        path,
        {
            "artifact_kind": CACHE_IDENTITY_KIND,
            "schema_version": CACHE_SCHEMA_VERSION,
            "execution_fingerprint": fingerprint,
            "identity": dict(identity),
        },
    )
    return path


def _episode_path(cache_dir: Path, seed: int) -> Path:
    return cache_dir / "episodes" / f"seed_{int(seed)}.json"


def _read_episode(path: Path, *, seed: int, fingerprint: str) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Stage5 episode-cache record must be an object: {path}")
    if str(payload.get("artifact_kind", "")) != CACHE_EPISODE_KIND:
        raise ValueError(f"unsupported Stage5 episode-cache record: {path}")
    if int(payload.get("schema_version", -1)) != CACHE_SCHEMA_VERSION:
        raise ValueError(f"unsupported Stage5 episode-cache record schema: {path}")
    if str(payload.get("execution_fingerprint", "")) != fingerprint:
        raise ValueError(f"Stage5 episode-cache record fingerprint mismatch: {path}")
    if int(payload.get("seed", -1)) != int(seed):
        raise ValueError(f"Stage5 episode-cache record seed mismatch: {path}")
    report = payload.get("episode")
    if not isinstance(report, dict) or int(report.get("seed", -1)) != int(seed):
        raise ValueError(f"Stage5 episode-cache payload seed mismatch: {path}")
    return payload


def _write_episode(
    cache_dir: Path,
    *,
    seed: int,
    fingerprint: str,
    report: Mapping[str, Any],
    partial: Mapping[str, Any],
) -> Path:
    path = _episode_path(cache_dir, seed)
    _write_new_or_validate(
        path,
        {
            "artifact_kind": CACHE_EPISODE_KIND,
            "schema_version": CACHE_SCHEMA_VERSION,
            "execution_fingerprint": fingerprint,
            "seed": int(seed),
            "model_observation_shape": list(
                partial.get("model_observation_shape", []) or []
            ),
            "env_observation_shape": list(
                partial.get("env_observation_shape", []) or []
            ),
            "policy_type": str(partial.get("policy_type", "sb3_ppo")),
            "episode": dict(report),
        },
    )
    return path


def _group_report(
    *,
    cache_dir: Path,
    identity_path: Path,
    fingerprint: str,
    seeds: list[int],
    task_quality: Mapping[str, Any],
    hit_count: int,
    executed_count: int,
) -> dict[str, Any]:
    reports: list[dict[str, Any]] = []
    records: dict[str, dict[str, str]] = {}
    model_shape: list[int] | None = None
    env_shape: list[int] | None = None
    policy_type: str | None = None
    for seed in seeds:
        path = _episode_path(cache_dir, seed)
        if not path.is_file():
            raise ValueError(f"Stage5 episode cache is incomplete after evaluation: {path}")
        payload = _read_episode(path, seed=seed, fingerprint=fingerprint)
        current_model_shape = list(payload.get("model_observation_shape", []) or [])
        current_env_shape = list(payload.get("env_observation_shape", []) or [])
        current_policy_type = str(payload.get("policy_type", "sb3_ppo"))
        if model_shape is not None and current_model_shape != model_shape:
            raise ValueError("Stage5 cached model observation shape changed across seeds")
        if env_shape is not None and current_env_shape != env_shape:
            raise ValueError("Stage5 cached environment observation shape changed across seeds")
        if policy_type is not None and current_policy_type != policy_type:
            raise ValueError("Stage5 cached policy type changed across seeds")
        model_shape = current_model_shape
        env_shape = current_env_shape
        policy_type = current_policy_type
        reports.append(dict(payload["episode"]))
        records[str(seed)] = {
            "path": str(path),
            "sha256": file_sha256(path),
        }

    metrics = aggregate_episode_reports(reports, task_quality=dict(task_quality))
    rewards = [float(report.get("episode_reward", 0.0)) for report in reports]
    metrics["average_reward"] = float(np.mean(rewards)) if rewards else 0.0
    metrics["merge_success_rate"] = (
        float(np.mean([float(report.get("merge_success", False)) for report in reports]))
        if reports
        else 0.0
    )
    metrics["terminal_success_rate"] = metrics["merge_success_rate"]
    record_fingerprint = stable_hash(
        {
            "execution_fingerprint": fingerprint,
            "episode_records": records,
            "requested_seeds": seeds,
        }
    )
    return {
        "episodes": reports,
        "metrics": metrics,
        "model_observation_shape": model_shape or [],
        "env_observation_shape": env_shape or [],
        "policy_type": policy_type or "sb3_ppo",
        "episode_cache": {
            "artifact_kind": CACHE_IDENTITY_KIND,
            "schema_version": CACHE_SCHEMA_VERSION,
            "execution_fingerprint": fingerprint,
            "identity_artifact": str(identity_path),
            "identity_artifact_sha256": file_sha256(identity_path),
            "episode_record_fingerprint": record_fingerprint,
            "episode_records": records,
            "replay_dir": str(cache_dir / "replay"),
            "requested_seed_count": len(seeds),
            "requested_seed_sha256": stable_hash({"episode_seeds": seeds}),
            "cache_hit_count": int(hit_count),
            "executed_episode_count": int(executed_count),
        },
    }


def evaluate_policy_cached(
    *,
    evaluator: Callable[..., dict[str, Any]],
    cfg: Any,
    model_path: str | Path | None,
    seeds: list[int],
    shield_enabled: bool,
    risk_checkpoint: str | Path | None,
    group_name: str,
    policy_type: str,
    binding: Mapping[str, Any],
) -> dict[str, Any]:
    cache_dir, identity, fingerprint = _validate_binding(
        binding,
        model_path=model_path,
        risk_checkpoint=risk_checkpoint,
    )
    identity_path = _identity_artifact(
        cache_dir,
        identity=identity,
        fingerprint=fingerprint,
    )
    requested = [int(seed) for seed in seeds]
    if len(requested) != len(set(requested)):
        raise ValueError("Stage5 episode-cache request contains duplicate seeds")
    missing = [
        seed
        for seed in requested
        if not _episode_path(cache_dir, seed).is_file()
    ]
    hit_count = len(requested) - len(missing)
    if missing:
        partial = evaluator(
            cfg,
            model_path,
            seeds=missing,
            shield_enabled=shield_enabled,
            risk_checkpoint=str(risk_checkpoint) if risk_checkpoint is not None else None,
            replay_dir=cache_dir / "replay",
            group_name=group_name,
            tensorboard=None,
            tensorboard_step_offset=0,
            policy_type=policy_type,
        )
        observed = [int(row.get("seed", -1)) for row in partial.get("episodes", []) or []]
        if observed != missing:
            raise ValueError(
                "Stage5 evaluator returned a different seed schedule while extending cache: "
                f"requested={missing} observed={observed}"
            )
        for report in list(partial.get("episodes", []) or []):
            seed = int(report["seed"])
            _write_episode(
                cache_dir,
                seed=seed,
                fingerprint=fingerprint,
                report=report,
                partial=partial,
            )
    return _group_report(
        cache_dir=cache_dir,
        identity_path=identity_path,
        fingerprint=fingerprint,
        seeds=requested,
        task_quality=dict(cfg.stage5.get("task_quality", {}) or {}),
        hit_count=hit_count,
        executed_count=len(missing),
    )
