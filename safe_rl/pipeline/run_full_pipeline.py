from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import yaml

from safe_rl.pipeline import (
    forecast_diagnostics,
    stage1_risk_probe,
    stage2_train_prediction_risk,
    stage3_train_ppo,
    stage4_collect_failures,
    stage5_paired_eval,
)
from safe_rl.utils.config import DEFAULT_CONFIG_PATH, REPO_ROOT, load_config
from safe_rl.utils.progress import stage_log
from safe_rl.sim.scenario_snapshot import SCENARIO_SOURCE_SUFFIXES, snapshot_scenario
from safe_rl.sim.metrics import SAFETY_METRIC_VERSION
from safe_rl.utils.sumo_installation import (
    SumoInstallation,
    configure_sumo_python,
    resolve_sumo_installation,
    sumo_subprocess_environment,
)
from safe_rl.utils.stage1_dataset import (
    STAGE1_FORMAT_VERSION,
    stage1_dataset_manifest_hash,
    validate_stage1_dataset,
)


VALID_FORECAST_SOURCES = ("constant_velocity", "wcdt", "wcdt_v2", "wcdt_v3")
DEFAULT_FORECAST_SOURCES = ("constant_velocity", "wcdt_v3")
VALID_FORECAST_PPO_PROFILES = ("default", "safety", "shield_guided", "merge_timing")
VALID_RUN_MODES = ("new", "resume", "overwrite")
VALID_PIPELINE_PROFILES = ("default", "smoke", "performance")
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")
PIPELINE_STATE_SCHEMA_VERSION = 5
PIPELINE_TASK_ORDER = (
    "network_snapshot",
    "stage1",
    "stage2_initial",
    "stage3_baseline",
    "stage4",
    "stage2_with_stage4",
    "stage3_forecast_cv",
    "stage3_forecast_wcdt",
    "stage3_forecast_wcdt_v2",
    "stage3_forecast_wcdt_v3",
    "stage5",
    "diagnostics",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_sha256(path: str | Path) -> str:
    candidate = Path(path)
    if candidate.is_file():
        return _sha256(candidate)
    if candidate.is_dir() and (candidate / "manifest.json").exists():
        return stage1_dataset_manifest_hash(candidate)
    if candidate.is_dir():
        digest = hashlib.sha256()
        for child in sorted(item for item in candidate.rglob("*") if item.is_file()):
            digest.update(child.relative_to(candidate).as_posix().encode("utf-8"))
            digest.update(_sha256(child).encode("ascii"))
        return digest.hexdigest()
    raise FileNotFoundError(candidate)


def _atomic_write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2, allow_nan=False)
    temporary.replace(path)
    return path


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(dict(merged[key]), value)
        else:
            merged[key] = value
    return merged


def _pipeline_profile_overrides(profile: str) -> dict[str, Any]:
    profile = str(profile or "default").strip().lower()
    if profile not in VALID_PIPELINE_PROFILES:
        raise ValueError(f"pipeline profile must be one of {VALID_PIPELINE_PROFILES}; got {profile!r}")
    if profile == "default":
        return {}
    path = _pipeline_profile_config_path(profile)
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def _pipeline_profile_config_path(profile: str) -> Path | None:
    profile = str(profile or "default").strip().lower()
    if profile == "default":
        return None
    if profile == "smoke":
        return REPO_ROOT / "safe_rl" / "config" / "advanced" / "pipeline_smoke_fast.yaml"
    if profile == "performance":
        return REPO_ROOT / "safe_rl" / "config" / "advanced" / "pipeline_performance.yaml"
    raise ValueError(f"pipeline profile must be one of {VALID_PIPELINE_PROFILES}; got {profile!r}")


def _pipeline_profile_config_sha256(profile: str) -> str | None:
    path = _pipeline_profile_config_path(profile)
    return _sha256(path) if path is not None else None


def _validate_run_id(run_id: str) -> str:
    run_id = str(run_id).strip()
    if not run_id or not RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError("run_id may contain only letters, digits, '.', '_' and '-'")
    if run_id in {".", ".."}:
        raise ValueError("run_id must identify a managed run directory")
    return run_id


def _output_root(cfg: Any) -> Path:
    root = Path(cfg.run.output_root)
    if not root.is_absolute():
        root = REPO_ROOT / root
    return root.resolve()


def _managed_run_dirs(output_root: str | Path, run_id: str) -> list[Path]:
    output_root = Path(output_root).resolve()
    run_id = _validate_run_id(run_id)
    names = [run_id, *[f"{run_id}_forecast_{_source_suffix(source)}" for source in VALID_FORECAST_SOURCES]]
    managed: list[Path] = []
    for name in names:
        candidate = (output_root / name).resolve()
        if candidate.parent != output_root:
            raise ValueError(f"refusing unmanaged run directory: {candidate}")
        managed.append(candidate)
    return managed


def _existing_managed_run_dirs(output_root: str | Path, run_id: str) -> list[Path]:
    return [path for path in _managed_run_dirs(output_root, run_id) if path.exists()]


def _remove_managed_run_dirs(output_root: str | Path, run_id: str) -> None:
    for path in _managed_run_dirs(output_root, run_id):
        if path.exists():
            shutil.rmtree(path)


def _prepare_new_run_dir(output_root: str | Path, run_id: str, run_mode: str) -> Path:
    output_root = Path(output_root).resolve()
    existing = _existing_managed_run_dirs(output_root, run_id)
    if run_mode == "new" and existing:
        raise FileExistsError(f"run directories already exist; use --run-mode resume or overwrite: {existing}")
    if run_mode == "overwrite":
        _remove_managed_run_dirs(output_root, run_id)
    run_dir = output_root / _validate_run_id(run_id)
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _scenario_source_sha256(cfg: Any) -> str:
    source = Path(cfg.scenario.root).resolve()
    digest = hashlib.sha256()
    for path in sorted(source.iterdir(), key=lambda item: item.name):
        if not path.is_file() or not path.name.endswith(SCENARIO_SOURCE_SUFFIXES):
            continue
        digest.update(path.name.encode("utf-8"))
        digest.update(_sha256(path).encode("ascii"))
    return digest.hexdigest()


def _predictor_training_flags(sources: list[str] | tuple[str, ...]) -> dict[str, bool]:
    source_set = set(sources)
    train_v1 = "wcdt" in source_set
    train_v2 = "wcdt_v2" in source_set
    train_v3 = "wcdt_v3" in source_set
    return {
        "train_enabled": bool(train_v1 or train_v2 or train_v3),
        "wcdt_v1_train_enabled": bool(train_v1),
        "wcdt_v2_train_enabled": bool(train_v2),
        "wcdt_v3_train_enabled": bool(train_v3),
    }


def _task_enabled(task_name: str, sources: list[str] | tuple[str, ...]) -> bool:
    if task_name == "stage3_forecast_cv":
        return "constant_velocity" in sources
    if task_name == "stage3_forecast_wcdt":
        return "wcdt" in sources
    if task_name == "stage3_forecast_wcdt_v2":
        return "wcdt_v2" in sources
    if task_name == "stage3_forecast_wcdt_v3":
        return "wcdt_v3" in sources
    return True


def _normalize_invocation(
    *,
    stage1_episodes: int | None,
    stage4_episodes: int | None,
    stage5_episodes: int | None,
    ppo_timesteps: int | None,
    forecast_ppo_timesteps: int | None,
    forecast_ppo_profile: str | None,
    forecast_sources: list[str],
    pipeline_profile: str,
    stage1_workers: int | None = None,
    ppo_num_envs: int | None = None,
) -> dict[str, Any]:
    profile = str(forecast_ppo_profile or "default").strip().lower()
    if profile not in VALID_FORECAST_PPO_PROFILES:
        raise ValueError(f"forecast PPO profile must be one of {VALID_FORECAST_PPO_PROFILES}; got {profile!r}")
    normalized_pipeline_profile = str(pipeline_profile or "default").strip().lower()
    _pipeline_profile_overrides(normalized_pipeline_profile)
    return {
        "stage1_episodes": int(stage1_episodes) if stage1_episodes is not None else None,
        "stage4_episodes": int(stage4_episodes) if stage4_episodes is not None else None,
        "stage5_episodes": int(stage5_episodes) if stage5_episodes is not None else None,
        "ppo_timesteps": int(ppo_timesteps) if ppo_timesteps is not None else None,
        "forecast_ppo_timesteps": int(forecast_ppo_timesteps) if forecast_ppo_timesteps is not None else None,
        "forecast_ppo_profile": profile,
        "forecast_sources": list(forecast_sources),
        "pipeline_profile": normalized_pipeline_profile,
        "stage1_workers": int(stage1_workers) if stage1_workers is not None else None,
        "ppo_num_envs": int(ppo_num_envs) if ppo_num_envs is not None else None,
    }


def _new_pipeline_state(
    run_id: str,
    invocation: dict[str, Any],
    sumo_installation: SumoInstallation | None = None,
    cfg: Any | None = None,
) -> dict[str, Any]:
    invocation = dict(invocation)
    invocation.setdefault("pipeline_profile", "default")
    sources = list(invocation["forecast_sources"])
    pipeline_profile = str(invocation.get("pipeline_profile", "default"))
    episode_seed_schedule = str(
        (cfg or {}).get("run", {}).get("episode_seed_schedule", "incrementing_v1")
    )
    vehicle_state_ordering_version = str(
        (cfg or {}).get("scenario", {}).get(
            "vehicle_state_ordering_version",
            "lexicographic_id_v1",
        )
    )
    stage1_storage_format = str(
        (cfg or {}).get("stage1", {}).get("output_format", "manifest_npy_v1")
    )
    tasks = {}
    for task_name in PIPELINE_TASK_ORDER:
        enabled = _task_enabled(task_name, sources)
        tasks[task_name] = {
            "enabled": enabled,
            "status": "pending" if enabled else "completed",
            "started_at": None,
            "completed_at": None,
            "required_outputs": [],
            "output_hashes": {},
        }
    return {
        "schema_version": PIPELINE_STATE_SCHEMA_VERSION,
        "run_id": run_id,
        "normalized_invocation": invocation,
        "forecast_sources": sources,
        "pipeline_profile": pipeline_profile,
        "pipeline_profile_config_sha256": _pipeline_profile_config_sha256(pipeline_profile),
        "default_config_sha256": _sha256(DEFAULT_CONFIG_PATH),
        "safety_metric_version": SAFETY_METRIC_VERSION,
        "episode_seed_schedule": episode_seed_schedule,
        "vehicle_state_ordering_version": vehicle_state_ordering_version,
        "stage1_storage_format": stage1_storage_format,
        "scenario_snapshot_sha256": None,
        "scenario_source_sha256": None,
        "sumo_installation": sumo_installation.to_dict() if sumo_installation is not None else None,
        "tasks": tasks,
    }


def _load_pipeline_state(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"resume requires pipeline state: {path}")
    with path.open("r", encoding="utf-8") as file:
        state = json.load(file)
    schema_version = int(state.get("schema_version", -1))
    if schema_version == 1:
        state.setdefault("tasks", {}).setdefault(
            "stage3_forecast_wcdt_v3",
            {
                "enabled": False,
                "status": "completed",
                "started_at": None,
                "completed_at": None,
                "required_outputs": [],
                "output_hashes": {},
            },
        )
        state["schema_version"] = 2
        schema_version = 2
    if schema_version == 2:
        state.setdefault("normalized_invocation", {}).setdefault("pipeline_profile", "default")
        state["schema_version"] = 3
        schema_version = 3
    if schema_version == 3:
        profile = str(state.setdefault("normalized_invocation", {}).setdefault("pipeline_profile", "default"))
        state["normalized_invocation"].setdefault("stage1_workers", None)
        state["normalized_invocation"].setdefault("ppo_num_envs", None)
        state.setdefault("pipeline_profile", profile)
        state.setdefault("sumo_installation", None)
        if "pipeline_profile_config_sha256" not in state:
            if profile != "default":
                raise ValueError(
                    "pipeline state is missing pipeline profile hash for a non-default profile; "
                    "use a new run id or --run-mode overwrite"
                )
            state["pipeline_profile_config_sha256"] = None
        state.setdefault("episode_seed_schedule", "fixed_legacy")
        state.setdefault("vehicle_state_ordering_version", "unspecified_legacy")
        state.setdefault("stage1_storage_format", "legacy_npz")
        state.setdefault("resume_diagnostics", {})
        state["schema_version"] = PIPELINE_STATE_SCHEMA_VERSION
        schema_version = PIPELINE_STATE_SCHEMA_VERSION
    if schema_version == 4:
        state.setdefault("sumo_installation", None)
        state.setdefault("normalized_invocation", {}).setdefault("stage1_workers", None)
        state.setdefault("normalized_invocation", {}).setdefault("ppo_num_envs", None)
        state.setdefault("episode_seed_schedule", "fixed_legacy")
        state.setdefault("vehicle_state_ordering_version", "unspecified_legacy")
        state.setdefault("stage1_storage_format", "legacy_npz")
        state.setdefault("resume_diagnostics", {})
        state["schema_version"] = PIPELINE_STATE_SCHEMA_VERSION
        schema_version = PIPELINE_STATE_SCHEMA_VERSION
    if schema_version == 5:
        state.setdefault("resume_diagnostics", {})
    elif schema_version != PIPELINE_STATE_SCHEMA_VERSION:
        raise ValueError(f"unsupported pipeline state schema: {state.get('schema_version')}")
    return state


def _resume_invocation(
    state: dict[str, Any],
    *,
    stage1_episodes: int | None,
    stage4_episodes: int | None,
    stage5_episodes: int | None,
    ppo_timesteps: int | None,
    forecast_ppo_timesteps: int | None,
    forecast_ppo_profile: str | None,
    forecast_sources: str | list[str] | tuple[str, ...] | None,
    forecast_source: str | None,
    pipeline_profile: str | None = None,
    stage1_workers: int | None = None,
    ppo_num_envs: int | None = None,
) -> dict[str, Any]:
    saved = dict(state["normalized_invocation"])
    explicit_sources = forecast_sources is not None or forecast_source is not None
    sources = (
        resolve_forecast_sources(forecast_sources=forecast_sources, forecast_source=forecast_source)
        if explicit_sources
        else list(saved["forecast_sources"])
    )
    requested = {
        "stage1_episodes": stage1_episodes,
        "stage4_episodes": stage4_episodes,
        "stage5_episodes": stage5_episodes,
        "ppo_timesteps": ppo_timesteps,
        "forecast_ppo_timesteps": forecast_ppo_timesteps,
        "forecast_ppo_profile": forecast_ppo_profile,
        "forecast_sources": sources if explicit_sources else None,
        "pipeline_profile": pipeline_profile,
        "stage1_workers": stage1_workers,
        "ppo_num_envs": ppo_num_envs,
    }
    for key, value in requested.items():
        if value is not None and value != saved.get(key):
            raise ValueError(f"resume argument mismatch for {key}: saved={saved.get(key)!r}, requested={value!r}")
    return saved


def _task_output_paths(run_dir: Path, cfg: Any, sources: list[str], task_name: str) -> list[Path]:
    stage3_model = str(cfg.stage3.model_name)
    paths: dict[str, list[Path]] = {
        "network_snapshot": [run_dir / "scenario_snapshot" / "manifest.json"],
        "stage1": [run_dir / "stage1" / str(cfg.stage1.output_name), run_dir / "stage1" / "stage1_report.json"],
        "stage2_initial": [
            run_dir / "stage2" / "risk_module_initial.pt",
            run_dir / "stage2" / "stage2_initial_training_report.json",
        ],
        "stage3_baseline": [
            run_dir / "stage3" / stage3_model,
            run_dir / "stage3" / "stage3_training_report.json",
            run_dir / "stage3" / "stage3_checkpoint_selection_report.json",
        ],
        "stage4": [run_dir / "stage4" / "on_policy_failure_buffer.npz", run_dir / "stage4" / "stage4_report.json"],
        "stage2_with_stage4": [run_dir / "stage2" / "risk_module.pt", run_dir / "stage2" / "stage2_training_report.json"],
        "stage5": [
            run_dir / "stage5" / "formal_paired_eval_report.json",
            run_dir / "stage5" / "shield_off_metrics.json",
            run_dir / "stage5" / "shield_on_metrics.json",
        ],
        "diagnostics": [run_dir / "stage5" / "diagnostics" / "forecast_diagnostics.json"],
    }
    if "wcdt" in sources:
        paths["stage2_initial"].extend(
            [run_dir / "stage2" / "wcdt_predictor.pt", run_dir / "stage2" / "wcdt_predictor_best.pt"]
        )
    if "wcdt_v2" in sources:
        paths["stage2_initial"].extend(
            [run_dir / "stage2" / "wcdt_v2_predictor.pt", run_dir / "stage2" / "wcdt_v2_predictor_best.pt"]
        )
    if "wcdt_v3" in sources:
        paths["stage2_initial"].extend(
            [run_dir / "stage2" / "wcdt_v3_predictor.pt", run_dir / "stage2" / "wcdt_v3_predictor_best.pt"]
        )
    for source in VALID_FORECAST_SOURCES:
        forecast_run_dir = run_dir.parent / _forecast_run_id(run_dir.name, source)
        paths[f"stage3_forecast_{_source_suffix(source)}"] = [
            forecast_run_dir / "stage3" / stage3_model,
            forecast_run_dir / "stage3" / "stage3_training_report.json",
            forecast_run_dir / "stage3" / "stage3_checkpoint_selection_report.json",
        ]
    return paths[task_name]


def _validate_completed_outputs(state: dict[str, Any]) -> None:
    for task_name in PIPELINE_TASK_ORDER:
        task = state["tasks"][task_name]
        if not bool(task.get("enabled", True)) or task.get("status") != "completed":
            continue
        for value in task.get("required_outputs", []):
            path = Path(value)
            if not path.exists():
                raise FileNotFoundError(f"completed task {task_name} is missing output: {path}")
            manifest_path = path / "manifest.json" if path.is_dir() else None
            if manifest_path is not None and manifest_path.exists():
                try:
                    with manifest_path.open("r", encoding="utf-8") as file:
                        manifest = json.load(file)
                except (OSError, json.JSONDecodeError) as exc:
                    raise ValueError(f"invalid completed artifact manifest: {manifest_path}") from exc
                if str(manifest.get("format_version", "")) == STAGE1_FORMAT_VERSION:
                    validate_stage1_dataset(path, verify_hashes=True)
            expected = task.get("output_hashes", {}).get(str(path))
            actual = _artifact_sha256(path)
            if expected != actual:
                raise ValueError(f"completed task {task_name} output hash changed: {path}")


def _sumo_major_minor(installation: dict[str, Any] | None) -> str | None:
    if not installation:
        return None
    version = str(installation.get("sumo_version", ""))
    for token in version.replace(",", " ").split():
        stripped = token.strip("vV")
        if stripped and stripped[0].isdigit():
            return ".".join(stripped.split(".")[:2])
    return version or None


def _validate_resume_state(
    state: dict[str, Any],
    cfg: Any,
    sumo_installation: SumoInstallation | None = None,
) -> None:
    if state.get("safety_metric_version") != SAFETY_METRIC_VERSION:
        raise ValueError(
            "safety metric version changed since the run started; use a new run id or --run-mode overwrite"
        )
    if state.get("default_config_sha256") != _sha256(DEFAULT_CONFIG_PATH):
        raise ValueError("default config changed since the run started; use a new run id or --run-mode overwrite")
    pipeline_profile = str(
        state.get("pipeline_profile")
        or state.get("normalized_invocation", {}).get("pipeline_profile", "default")
    )
    expected_profile_hash = _pipeline_profile_config_sha256(pipeline_profile)
    if state.get("pipeline_profile_config_sha256") != expected_profile_hash:
        raise ValueError("pipeline profile config changed since the run started; use a new run id or --run-mode overwrite")
    snapshot_hash = state.get("scenario_snapshot_sha256")
    source_hash = state.get("scenario_source_sha256")
    if snapshot_hash:
        manifest = _output_root(cfg) / str(state["run_id"]) / "scenario_snapshot" / "manifest.json"
        if not manifest.exists() or _sha256(manifest) != snapshot_hash:
            raise ValueError("scenario snapshot changed since the run started")
    if source_hash and _scenario_source_sha256(cfg) != source_hash:
        raise ValueError("scenario source changed since the run started; use a new run id or --run-mode overwrite")
    saved_sumo = state.get("sumo_installation")
    if saved_sumo and sumo_installation is not None:
        if _sumo_major_minor(saved_sumo) != sumo_installation.major_minor_version:
            raise ValueError("SUMO major/minor version changed since the run started; use a new run id or overwrite")
        strict_fields = (
            "sumo_home",
            "sumo_binary",
            "netconvert_binary",
            "traci_module_path",
        )
        for field in strict_fields:
            if str(saved_sumo.get(field, "")) != str(getattr(sumo_installation, field)):
                raise ValueError(
                    f"SUMO installation field changed ({field}); use a new run id or overwrite"
                )
        hash_changes = {
            field: {
                "saved": str(saved_sumo.get(field, "")),
                "current": str(getattr(sumo_installation, field)),
            }
            for field in ("sumo_binary_sha256", "netconvert_binary_sha256")
            if str(saved_sumo.get(field, "")) != str(getattr(sumo_installation, field))
        }
        state.setdefault("resume_diagnostics", {})["binary_hash_changed"] = bool(hash_changes)
        if hash_changes:
            state["resume_diagnostics"]["binary_hash_changes"] = hash_changes
    semantic_values = {
        "episode_seed_schedule": str(
            cfg.get("run", {}).get("episode_seed_schedule", "fixed_legacy")
        ),
        "vehicle_state_ordering_version": str(
            cfg.get("scenario", {}).get(
                "vehicle_state_ordering_version",
                "unspecified_legacy",
            )
        ),
        "stage1_storage_format": str(
            cfg.get("stage1", {}).get("output_format", "legacy_npz")
        ),
    }
    for field, current in semantic_values.items():
        if str(state.get(field, "")) != current:
            raise ValueError(
                f"{field} changed since the run started; use a new run id or overwrite"
            )
    _validate_completed_outputs(state)


def _reset_unfinished_tasks(state: dict[str, Any]) -> None:
    reset = False
    for task_name in PIPELINE_TASK_ORDER:
        task = state["tasks"][task_name]
        if not bool(task.get("enabled", True)):
            continue
        if task.get("status") != "completed":
            reset = True
        if reset:
            task.update(
                {
                    "status": "pending",
                    "started_at": None,
                    "completed_at": None,
                    "required_outputs": [],
                    "output_hashes": {},
                }
            )


def _run_pipeline_task(
    state_path: Path,
    state: dict[str, Any],
    task_name: str,
    required_outputs: list[Path],
    action: Callable[[], None],
) -> bool:
    task = state["tasks"][task_name]
    if not bool(task.get("enabled", True)):
        return False
    if task.get("status") == "completed":
        stage_log("full", f"resume skip completed task={task_name}")
        return False
    task.update(
        {
            "status": "running",
            "started_at": _utc_now(),
            "completed_at": None,
            "required_outputs": [str(path.resolve()) for path in required_outputs],
            "output_hashes": {},
        }
    )
    _atomic_write_json(state_path, state)
    action()
    missing = [path for path in required_outputs if not path.exists()]
    if missing:
        raise FileNotFoundError(f"task {task_name} did not produce required outputs: {missing}")
    task["status"] = "completed"
    task["completed_at"] = _utc_now()
    task["output_hashes"] = {
        str(path.resolve()): _artifact_sha256(path)
        for path in required_outputs
    }
    _atomic_write_json(state_path, state)
    return True


def _relative_run_path(run_id: str, stage: str, name: str) -> str:
    return (Path("safe_rl_output") / "runs" / run_id / stage / name).as_posix()


def _write_yaml(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(payload, file, sort_keys=False, allow_unicode=True)
    return path


def _source_suffix(source: str) -> str:
    if source == "constant_velocity":
        return "cv"
    if source == "wcdt_v2":
        return "wcdt_v2"
    if source == "wcdt_v3":
        return "wcdt_v3"
    return "wcdt"


def _forecast_run_id(run_id: str, source: str) -> str:
    return f"{run_id}_forecast_{_source_suffix(source)}"


def resolve_forecast_sources(
    forecast_sources: str | list[str] | tuple[str, ...] | None = None,
    forecast_source: str | None = None,
) -> list[str]:
    if forecast_sources is not None and forecast_source is not None:
        raise ValueError("Use either --forecast-source or --forecast-sources, not both.")
    raw = forecast_source if forecast_source is not None else forecast_sources
    if raw is None:
        values = list(DEFAULT_FORECAST_SOURCES)
    elif isinstance(raw, str):
        values = [item.strip().lower() for item in raw.split(",") if item.strip()]
    else:
        values = [str(item).strip().lower() for item in raw if str(item).strip()]
    if not values:
        raise ValueError("forecast sources cannot be empty")
    invalid = [item for item in values if item not in VALID_FORECAST_SOURCES]
    if invalid:
        raise ValueError(f"forecast sources must be one of {VALID_FORECAST_SOURCES}; invalid={invalid}")
    deduped: list[str] = []
    for item in values:
        if item not in deduped:
            deduped.append(item)
    return deduped


def _forecast_group_name(source: str) -> str:
    if source == "wcdt":
        return "ppo_wcdt_features"
    if source == "wcdt_v2":
        return "ppo_wcdt_v2_features"
    if source == "wcdt_v3":
        return "ppo_wcdt_v3_features"
    return "ppo_cv_features"


def _forecast_shield_group_name(source: str) -> str:
    if source == "wcdt":
        return "wcdt_prediction_shield"
    if source == "wcdt_v2":
        return "wcdt_v2_prediction_shield"
    if source == "wcdt_v3":
        return "wcdt_v3_prediction_shield"
    return "cv_prediction_shield"


def _forecast_checkpoint_name(source: str) -> str | None:
    if source == "wcdt":
        return "wcdt_predictor.pt"
    if source == "wcdt_v2":
        return "wcdt_v2_predictor.pt"
    if source == "wcdt_v3":
        return "wcdt_v3_predictor.pt"
    return None


def _forecast_payload(
    run_id: str,
    source: str,
    ppo_timesteps: int | None,
    forecast_ppo_profile: str = "default",
) -> dict[str, Any]:
    forecast_run_id = _forecast_run_id(run_id, source)
    payload: dict[str, Any] = {
        "run": {"run_id": forecast_run_id},
        "forecast_features": {
            "enabled": True,
            "use_for_ppo_observation": True,
            "source": source,
            "checkpoint": (
                _relative_run_path(run_id, "stage2", _forecast_checkpoint_name(source))
                if _forecast_checkpoint_name(source)
                else None
            ),
            "allow_heuristic_fallback": False,
        },
        "rl": {"use_wcdt_forecast_features": True},
        "shield": {
            "forecast_task_shadow_enabled": False,
            "task_backstop_enabled": False,
        },
    }
    profile = str(forecast_ppo_profile or "default").lower()
    if profile not in VALID_FORECAST_PPO_PROFILES:
        raise ValueError(f"forecast PPO profile must be one of {VALID_FORECAST_PPO_PROFILES}; got {profile!r}")
    if profile == "safety":
        payload["rl"]["reward_profile"] = "safety_forecast"
    elif profile in {"shield_guided", "merge_timing"}:
        payload["rl"]["reward_profile"] = (
            "merge_timing_forecast" if profile == "merge_timing" else "shield_guided_forecast"
        )
        payload["rl"]["shield_guided_reward"] = {
            "risk_checkpoint": _relative_run_path(run_id, "stage2", "risk_module.pt"),
        }
    if ppo_timesteps is not None:
        payload["rl"]["total_timesteps"] = int(ppo_timesteps)
    return payload


def _forecast_stage5_groups(run_id: str, source: str) -> list[dict[str, Any]]:
    forecast_run_id = _forecast_run_id(run_id, source)
    base = {
        "name": _forecast_group_name(source),
        "forecast_features": True,
        "shield": False,
        "model_path": _relative_run_path(forecast_run_id, "stage3", "ppo_model.zip"),
        "forecast_source": source,
    }
    shield = {
        "name": _forecast_shield_group_name(source),
        "forecast_features": True,
        "shield": True,
        "model_path": _relative_run_path(forecast_run_id, "stage3", "ppo_model.zip"),
        "forecast_source": source,
    }
    checkpoint_name = _forecast_checkpoint_name(source)
    if checkpoint_name:
        checkpoint = _relative_run_path(run_id, "stage2", checkpoint_name)
        base["forecast_checkpoint"] = checkpoint
        shield["forecast_checkpoint"] = checkpoint
    return [base, shield]


def build_generated_configs(
    run_id: str,
    generated_dir: str | Path,
    stage1_episodes: int | None = None,
    stage4_episodes: int | None = None,
    stage5_episodes: int | None = None,
    ppo_timesteps: int | None = None,
    forecast_ppo_timesteps: int | None = None,
    forecast_ppo_profile: str = "default",
    forecast_sources: str | list[str] | tuple[str, ...] | None = None,
    forecast_source: str | None = None,
    pipeline_profile: str = "default",
    stage1_workers: int | None = None,
    ppo_num_envs: int | None = None,
    sumo_installation: SumoInstallation | None = None,
) -> dict[str, Path]:
    generated_dir = Path(generated_dir)
    sources = resolve_forecast_sources(forecast_sources=forecast_sources, forecast_source=forecast_source)
    profile_payload = _pipeline_profile_overrides(pipeline_profile)
    if ppo_num_envs is not None:
        profile_payload = _deep_merge(profile_payload, {"training": {"ppo_num_envs": int(ppo_num_envs)}})
    if sumo_installation is not None:
        profile_payload = _deep_merge(
            profile_payload,
            {
                "scenario": {
                    "sumo_binary": sumo_installation.sumo_binary,
                    "sumo_gui_binary": sumo_installation.sumo_gui_binary,
                    "netconvert_binary": sumo_installation.netconvert_binary,
                    "sumo_tools_directory": sumo_installation.tools_directory,
                    "sumo_home": sumo_installation.sumo_home,
                    "sumo_version": sumo_installation.sumo_version,
                    "netconvert_version": sumo_installation.netconvert_version,
                    "sumo_installation_fingerprint": sumo_installation.to_dict(),
                }
            },
        )

    main_payload: dict[str, Any] = _deep_merge(profile_payload, {
        "run": {"run_id": run_id},
        "prediction": _predictor_training_flags(sources),
        "shield": {
            "forecast_task_shadow_enabled": False,
            "task_backstop_enabled": False,
        },
    })
    if stage1_episodes is not None:
        main_payload = _deep_merge(main_payload, {"stage1": {"episodes": int(stage1_episodes)}})
    if stage4_episodes is not None:
        main_payload = _deep_merge(main_payload, {"stage4": {"episodes": int(stage4_episodes)}})
    if ppo_timesteps is not None:
        main_payload = _deep_merge(main_payload, {"rl": {"total_timesteps": int(ppo_timesteps)}})
    if stage1_workers is not None:
        main_payload = _deep_merge(main_payload, {"stage1": {"workers": int(stage1_workers)}})

    stage2_stage4_payload: dict[str, Any] = _deep_merge(profile_payload, {
        "run": {"run_id": run_id},
        "stage2": {"input_stage4": "auto"},
        "prediction": {"train_enabled": False},
    })

    groups: list[dict[str, Any]] = [
        {
            "name": "ppo",
            "forecast_features": False,
            "shield": False,
            "model_path": _relative_run_path(run_id, "stage3", "ppo_model.zip"),
        },
        {
            "name": "ppo_shield",
            "forecast_features": False,
            "shield": True,
            "model_path": _relative_run_path(run_id, "stage3", "ppo_model.zip"),
        },
    ]
    for source in sources:
        groups.extend(_forecast_stage5_groups(run_id, source))

    requested_stage5_episodes = (
        int(stage5_episodes)
        if stage5_episodes is not None
        else int(profile_payload.get("stage5", {}).get("episodes_per_group", 20))
    )
    stage5_payload: dict[str, Any] = _deep_merge(profile_payload, {
        "run": {"run_id": run_id},
        "shield": {"forecast_task_shadow_enabled": True},
        "stage5": {
            "episodes_per_group": requested_stage5_episodes,
            "seeds": list(range(1, requested_stage5_episodes + 1)),
            "groups": groups,
        },
    })

    configs = {
        "main": _write_yaml(generated_dir / "main_overrides.yaml", main_payload),
        "stage2_with_stage4": _write_yaml(generated_dir / "stage2_with_stage4.yaml", stage2_stage4_payload),
        "stage5_multi_groups": _write_yaml(generated_dir / "stage5_multi_groups.yaml", stage5_payload),
    }
    configs["stage5_four_groups"] = configs["stage5_multi_groups"]
    for source in sources:
        key = f"forecast_{_source_suffix(source)}_ppo"
        configs[key] = _write_yaml(
            generated_dir / f"{key}.yaml",
            _deep_merge(
                profile_payload,
                _forecast_payload(
                    run_id,
                    source,
                    forecast_ppo_timesteps if forecast_ppo_timesteps is not None else ppo_timesteps,
                    forecast_ppo_profile=forecast_ppo_profile,
                ),
            ),
        )
    if sources:
        configs["forecast_ppo"] = configs[f"forecast_{_source_suffix(sources[0])}_ppo"]
    return configs


def _load_stage_cfg(config_path: Path, run_id: str):
    cfg = load_config(config_path)
    cfg.run["run_id"] = run_id
    return cfg


def _run_subprocess(command: list[str], label: str, env: dict[str, str] | None = None) -> None:
    stage_log("full", f"{label}: {' '.join(command)}")
    subprocess.run(command, cwd=REPO_ROOT, check=True, env=env)


def _sumo_smoke_check(config_path: Path, run_id: str) -> None:
    cfg = _load_stage_cfg(config_path, run_id)
    _run_subprocess(
        [
            str(cfg.scenario.get("sumo_binary", "sumo")),
            "-c",
            str(cfg.scenario.sumocfg),
            "--end",
            "1",
            "--no-step-log",
            "true",
            "--duration-log.disable",
            "true",
            "--seed",
            str(cfg.run.seed),
        ],
        "SUMO smoke check",
    )


def _print_stage5_summary(run_id: str) -> None:
    report_path = REPO_ROOT / "safe_rl_output" / "runs" / run_id / "stage5" / "formal_paired_eval_report.json"
    if not report_path.exists():
        stage_log("full", f"Stage5 report not found: {report_path}")
        return
    with report_path.open("r", encoding="utf-8") as file:
        report = json.load(file)
    stage_log("full", "Stage5 acceptance:")
    print(json.dumps(report.get("acceptance", {}), ensure_ascii=False, indent=2))
    metrics = {name: item.get("metrics", {}) for name, item in report.get("groups", {}).items()}
    stage_log("full", "Stage5 metrics:")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def run_full_pipeline(
    run_id: str,
    run_mode: str = "new",
    stage1_episodes: int | None = None,
    stage4_episodes: int | None = None,
    stage5_episodes: int | None = None,
    ppo_timesteps: int | None = None,
    forecast_ppo_timesteps: int | None = None,
    forecast_ppo_profile: str | None = None,
    forecast_sources: str | list[str] | tuple[str, ...] | None = None,
    forecast_source: str | None = None,
    pipeline_profile: str | None = None,
    stage1_workers: int | None = None,
    ppo_num_envs: int | None = None,
) -> Path:
    run_id = _validate_run_id(run_id)
    run_mode = str(run_mode).strip().lower()
    if run_mode not in VALID_RUN_MODES:
        raise ValueError(f"run mode must be one of {VALID_RUN_MODES}; got {run_mode!r}")
    bootstrap_cfg = load_config()
    bootstrap_cfg.run["run_id"] = run_id
    sumo_installation = resolve_sumo_installation(bootstrap_cfg.scenario)
    configure_sumo_python(sumo_installation)
    sumo_env = sumo_subprocess_environment(sumo_installation)
    output_root = _output_root(bootstrap_cfg)
    run_dir = output_root / run_id
    state_path = run_dir / "pipeline_state.json"
    if run_mode == "resume":
        state = _load_pipeline_state(state_path)
        if state.get("run_id") != run_id:
            raise ValueError(f"pipeline state run_id mismatch: {state.get('run_id')!r}")
        invocation = _resume_invocation(
            state,
            stage1_episodes=stage1_episodes,
            stage4_episodes=stage4_episodes,
            stage5_episodes=stage5_episodes,
            ppo_timesteps=ppo_timesteps,
            forecast_ppo_timesteps=forecast_ppo_timesteps,
            forecast_ppo_profile=forecast_ppo_profile,
            forecast_sources=forecast_sources,
            forecast_source=forecast_source,
            pipeline_profile=pipeline_profile,
            stage1_workers=stage1_workers,
            ppo_num_envs=ppo_num_envs,
        )
        _validate_resume_state(state, bootstrap_cfg, sumo_installation)
        _reset_unfinished_tasks(state)
        _atomic_write_json(state_path, state)
    else:
        sources = resolve_forecast_sources(forecast_sources=forecast_sources, forecast_source=forecast_source)
        invocation = _normalize_invocation(
            stage1_episodes=stage1_episodes,
            stage4_episodes=stage4_episodes,
            stage5_episodes=stage5_episodes,
            ppo_timesteps=ppo_timesteps,
            forecast_ppo_timesteps=forecast_ppo_timesteps,
            forecast_ppo_profile=forecast_ppo_profile,
            forecast_sources=sources,
            pipeline_profile=str(pipeline_profile or "default"),
            stage1_workers=stage1_workers,
            ppo_num_envs=ppo_num_envs,
        )
        run_dir = _prepare_new_run_dir(output_root, run_id, run_mode)
        state = _new_pipeline_state(run_id, invocation, sumo_installation, bootstrap_cfg)
        _atomic_write_json(state_path, state)
    sources = list(invocation["forecast_sources"])
    generated_dir = run_dir / "generated_configs"
    configs = build_generated_configs(
        run_id,
        generated_dir,
        stage1_episodes=invocation["stage1_episodes"],
        stage4_episodes=invocation["stage4_episodes"],
        stage5_episodes=invocation["stage5_episodes"],
        ppo_timesteps=invocation["ppo_timesteps"],
        forecast_ppo_timesteps=invocation["forecast_ppo_timesteps"],
        forecast_ppo_profile=invocation["forecast_ppo_profile"],
        forecast_sources=sources,
        pipeline_profile=invocation["pipeline_profile"],
        stage1_workers=invocation.get("stage1_workers"),
        ppo_num_envs=invocation.get("ppo_num_envs"),
        sumo_installation=sumo_installation,
    )
    main_cfg = _load_stage_cfg(configs["main"], run_id)

    stage_log("full", f"run_id={run_id}")
    stage_log("full", f"run_mode={run_mode}")
    stage_log("full", f"forecast_sources={sources}")
    stage_log("full", f"forecast_ppo_profile={invocation['forecast_ppo_profile']}")
    stage_log("full", f"pipeline_profile={invocation['pipeline_profile']}")
    stage_log("full", f"sumo={sumo_installation.sumo_binary} ({sumo_installation.sumo_version})")
    if invocation["forecast_ppo_timesteps"] is not None:
        stage_log("full", f"forecast_ppo_timesteps={invocation['forecast_ppo_timesteps']}")
    for source in sources:
        stage_log("full", f"forecast_run_id[{source}]={_forecast_run_id(run_id, source)}")
    stage_log("full", f"generated_configs={generated_dir}")

    def _build_network_snapshot() -> None:
        _run_subprocess(
            [
                sys.executable,
                str(REPO_ROOT / "scenarios" / "highway_merge" / "build_network.py"),
                "--netconvert",
                sumo_installation.netconvert_binary,
            ],
            "build network",
            env=sumo_env,
        )
        snapshot_manifest = snapshot_scenario(main_cfg, run_dir)
        state["scenario_snapshot_sha256"] = _sha256(snapshot_manifest)
        state["scenario_source_sha256"] = _scenario_source_sha256(main_cfg)
        stage_log("full", f"scenario_snapshot={snapshot_manifest}")

    _run_pipeline_task(
        state_path,
        state,
        "network_snapshot",
        _task_output_paths(run_dir, main_cfg, sources, "network_snapshot"),
        _build_network_snapshot,
    )
    _sumo_smoke_check(configs["main"], run_id)

    tasks: list[tuple[str, str, Callable[[], None]]] = [
        ("stage1", "Stage1 risk probe", lambda: stage1_risk_probe.run(_load_stage_cfg(configs["main"], run_id))),
        (
            "stage2_initial",
            "Stage2 initial prediction + risk",
            lambda: stage2_train_prediction_risk.run(_load_stage_cfg(configs["main"], run_id)),
        ),
        (
            "stage3_baseline",
            "Stage3 baseline PPO",
            lambda: stage3_train_ppo.run(_load_stage_cfg(configs["main"], run_id)),
        ),
        (
            "stage4",
            "Stage4 shadow collection",
            lambda: stage4_collect_failures.run(_load_stage_cfg(configs["main"], run_id)),
        ),
        (
            "stage2_with_stage4",
            "Stage2 risk retraining with Stage4 buffer",
            lambda: stage2_train_prediction_risk.run(_load_stage_cfg(configs["stage2_with_stage4"], run_id)),
        ),
    ]
    for task_name, label, action in tasks:
        stage_log("full", label)
        _run_pipeline_task(state_path, state, task_name, _task_output_paths(run_dir, main_cfg, sources, task_name), action)
    for source in sources:
        forecast_run_id = _forecast_run_id(run_id, source)
        stage_log("full", f"Stage3 forecast PPO ({source})")
        task_name = f"stage3_forecast_{_source_suffix(source)}"
        _run_pipeline_task(
            state_path,
            state,
            task_name,
            _task_output_paths(run_dir, main_cfg, sources, task_name),
            lambda source=source, forecast_run_id=forecast_run_id: stage3_train_ppo.run(
                _load_stage_cfg(configs[f"forecast_{_source_suffix(source)}_ppo"], forecast_run_id)
            ),
        )
    stage_log("full", "Stage5 multi-group paired evaluation")
    _run_pipeline_task(
        state_path,
        state,
        "stage5",
        _task_output_paths(run_dir, main_cfg, sources, "stage5"),
        lambda: stage5_paired_eval.run(_load_stage_cfg(configs["stage5_multi_groups"], run_id)),
    )
    stage_log("full", "Forecast diagnostics")
    _run_pipeline_task(
        state_path,
        state,
        "diagnostics",
        _task_output_paths(run_dir, main_cfg, sources, "diagnostics"),
        lambda: forecast_diagnostics.run_forecast_diagnostics(_load_stage_cfg(configs["main"], run_id)),
    )
    _print_stage5_summary(run_id)
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full SAFE_RL Stage1-Stage5 pipeline.")
    parser.add_argument(
        "--run-id",
        required=True,
        help="Baseline run id. Forecast run ids use '<run-id>_forecast_cv|wcdt|wcdt_v2|wcdt_v3'.",
    )
    parser.add_argument(
        "--run-mode",
        choices=list(VALID_RUN_MODES),
        default="new",
        help="Run directory handling: new refuses existing runs, resume continues verified state, overwrite recreates runs.",
    )
    parser.add_argument("--stage1-episodes", type=int, default=None, help="Optional override for Stage1 episodes.")
    parser.add_argument("--stage1-workers", type=int, default=None, help="Optional Stage1 SUMO worker count.")
    parser.add_argument("--stage4-episodes", type=int, default=None, help="Optional override for Stage4 episodes.")
    parser.add_argument("--stage5-episodes", type=int, default=None, help="Optional override for Stage5 episodes per group.")
    parser.add_argument("--ppo-timesteps", type=int, default=None, help="Optional override for baseline and forecast PPO timesteps.")
    parser.add_argument("--ppo-num-envs", type=int, default=None, help="Optional PPO vector environment count.")
    parser.add_argument(
        "--forecast-ppo-timesteps",
        type=int,
        default=None,
        help="Optional override for forecast PPO timesteps only.",
    )
    parser.add_argument(
        "--forecast-ppo-profile",
        choices=list(VALID_FORECAST_PPO_PROFILES),
        default=None,
        help=(
            "Forecast PPO reward profile. 'safety' writes rl.reward_profile=safety_forecast; "
            "'shield_guided' writes rl.reward_profile=shield_guided_forecast; "
            "'merge_timing' writes rl.reward_profile=merge_timing_forecast. "
            "The guided profiles bind the base risk module."
        ),
    )
    parser.add_argument(
        "--forecast-source",
        choices=list(VALID_FORECAST_SOURCES),
        default=None,
        help="Legacy single forecast feature source for the forecast PPO branch.",
    )
    parser.add_argument(
        "--forecast-sources",
        default=None,
        help="Comma-separated forecast sources. Default: constant_velocity,wcdt_v3. Use wcdt/wcdt_v2 explicitly for legacy or ablation branches.",
    )
    parser.add_argument(
        "--pipeline-profile",
        choices=list(VALID_PIPELINE_PROFILES),
        default=None,
        help="Pipeline profile. Use 'smoke' for validation or 'performance' for explicit parallel execution.",
    )
    args = parser.parse_args()
    if args.forecast_source and args.forecast_sources:
        parser.error("Use either --forecast-source or --forecast-sources, not both.")
    run_full_pipeline(
        args.run_id,
        run_mode=args.run_mode,
        stage1_episodes=args.stage1_episodes,
        stage1_workers=args.stage1_workers,
        stage4_episodes=args.stage4_episodes,
        stage5_episodes=args.stage5_episodes,
        ppo_timesteps=args.ppo_timesteps,
        ppo_num_envs=args.ppo_num_envs,
        forecast_ppo_timesteps=args.forecast_ppo_timesteps,
        forecast_ppo_profile=args.forecast_ppo_profile,
        forecast_sources=args.forecast_sources,
        forecast_source=args.forecast_source,
        pipeline_profile=args.pipeline_profile,
    )


if __name__ == "__main__":
    main()
