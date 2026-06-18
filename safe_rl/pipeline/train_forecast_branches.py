from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from safe_rl.pipeline import stage3_train_ppo
from safe_rl.pipeline.run_full_pipeline import (
    _forecast_checkpoint_name,
    _forecast_group_name,
    _forecast_run_id,
    _forecast_shield_group_name,
    _relative_run_path,
    _source_suffix,
    resolve_forecast_sources,
)
from safe_rl.pipeline.common import _resolve_repo_path
from safe_rl.utils.config import load_config
from safe_rl.utils.progress import stage_log


VALID_FORECAST_BRANCH_PROFILES = ("default", "safety", "shield_guided", "merge_timing")


def _write_yaml(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(payload, file, sort_keys=False, allow_unicode=True)
    return path


def _merge_timing_run_id(base_run_id: str, source: str, suffix: str) -> str:
    return f"{base_run_id}_forecast_{_source_suffix(source)}_{suffix}"


def _forecast_checkpoint(base_run_id: str, source: str) -> str | None:
    checkpoint_name = _forecast_checkpoint_name(source)
    if not checkpoint_name:
        return None
    return _relative_run_path(base_run_id, "stage2", checkpoint_name)


def _forecast_training_payload(
    *,
    base_run_id: str,
    source: str,
    suffix: str,
    profile: str,
    timesteps: int | None,
) -> dict[str, Any]:
    run_id = _merge_timing_run_id(base_run_id, source, suffix)
    payload: dict[str, Any] = {
        "run": {"run_id": run_id},
        "forecast_features": {
            "enabled": True,
            "use_for_ppo_observation": True,
            "source": source,
            "checkpoint": _forecast_checkpoint(base_run_id, source),
            "allow_heuristic_fallback": False,
        },
        "rl": {"use_wcdt_forecast_features": True},
        "shield": {
            "forecast_aware_candidate_ranking_mode": "off",
            "forecast_task_shadow_enabled": False,
            "task_backstop_enabled": False,
        },
    }
    if profile == "safety":
        payload["rl"]["reward_profile"] = "safety_forecast"
    elif profile in {"shield_guided", "merge_timing"}:
        payload["rl"]["reward_profile"] = (
            "merge_timing_forecast" if profile == "merge_timing" else "shield_guided_forecast"
        )
        payload["rl"]["shield_guided_reward"] = {
            "risk_checkpoint": _relative_run_path(base_run_id, "stage2", "risk_module.pt"),
        }
    if timesteps is not None:
        payload["rl"]["total_timesteps"] = int(timesteps)
    return payload


def _forecast_group(base_run_id: str, source: str, *, shield: bool) -> dict[str, Any]:
    group = {
        "name": _forecast_shield_group_name(source) if shield else _forecast_group_name(source),
        "forecast_features": True,
        "shield": bool(shield),
        "model_path": _relative_run_path(_forecast_run_id(base_run_id, source), "stage3", "ppo_model.zip"),
        "forecast_source": source,
    }
    checkpoint = _forecast_checkpoint(base_run_id, source)
    if checkpoint:
        group["forecast_checkpoint"] = checkpoint
    if shield:
        group["shield_overrides"] = {
            "forecast_aware_candidate_ranking_mode": "off",
            "forecast_task_shadow_enabled": False,
            "task_backstop_enabled": False,
        }
    return group


def _merge_timing_group(base_run_id: str, source: str, suffix: str, *, shield: bool) -> dict[str, Any]:
    run_id = _merge_timing_run_id(base_run_id, source, suffix)
    if source == "constant_velocity":
        name = "cv_merge_timing_prediction_shield" if shield else "ppo_cv_merge_timing_features"
    else:
        name = f"{source}_merge_timing_prediction_shield" if shield else f"ppo_{source}_merge_timing_features"
    group = {
        "name": name,
        "forecast_features": True,
        "shield": bool(shield),
        "model_path": _relative_run_path(run_id, "stage3", "ppo_model.zip"),
        "forecast_source": source,
    }
    checkpoint = _forecast_checkpoint(base_run_id, source)
    if checkpoint:
        group["forecast_checkpoint"] = checkpoint
    if shield:
        group["shield_overrides"] = {
            "forecast_aware_candidate_ranking_mode": "off",
            "forecast_task_shadow_enabled": False,
            "task_backstop_enabled": False,
        }
    return group


def _merge_timing_forecast_aware_groups(base_run_id: str, source: str, suffix: str) -> list[dict[str, Any]]:
    if source != "wcdt_v3":
        return []
    groups: list[dict[str, Any]] = []
    for mode_name, overrides in (
        (
            "shadow",
            {
                "forecast_aware_candidate_ranking_mode": "shadow",
                "forecast_task_shadow_enabled": True,
                "task_backstop_enabled": False,
            },
        ),
        (
            "task_backstop",
            {
                "forecast_aware_candidate_ranking_mode": "task_backstop",
                "forecast_task_shadow_enabled": True,
                "task_backstop_enabled": True,
            },
        ),
        (
            "full_ranking",
            {
                "forecast_aware_candidate_ranking_mode": "full_ranking",
                "forecast_task_shadow_enabled": True,
                "task_backstop_enabled": False,
            },
        ),
    ):
        group = _merge_timing_group(base_run_id, source, suffix, shield=True)
        group["name"] = f"wcdt_v3_merge_timing_prediction_shield_{mode_name}"
        group["shield_overrides"] = overrides
        groups.append(group)
    return groups


def _stage5_payload(base_run_id: str, sources: list[str], suffix: str) -> dict[str, Any]:
    groups: list[dict[str, Any]] = [
        {
            "name": "ppo",
            "forecast_features": False,
            "shield": False,
            "model_path": _relative_run_path(base_run_id, "stage3", "ppo_model.zip"),
        },
        {
            "name": "ppo_shield",
            "forecast_features": False,
            "shield": True,
            "model_path": _relative_run_path(base_run_id, "stage3", "ppo_model.zip"),
        },
    ]
    for source in sources:
        groups.append(_forecast_group(base_run_id, source, shield=False))
        groups.append(_forecast_group(base_run_id, source, shield=True))
        groups.append(_merge_timing_group(base_run_id, source, suffix, shield=False))
        groups.append(_merge_timing_group(base_run_id, source, suffix, shield=True))
        groups.extend(_merge_timing_forecast_aware_groups(base_run_id, source, suffix))
    return {
        "run": {"run_id": base_run_id},
        "shield": {
            "forecast_aware_candidate_ranking_mode": "off",
            "forecast_task_shadow_enabled": False,
            "task_backstop_enabled": False,
        },
        "stage5": {"groups": groups},
    }


def _assert_base_artifacts(base_run_id: str, sources: list[str]) -> None:
    required = [
        _resolve_repo_path(_relative_run_path(base_run_id, "stage2", "risk_module.pt")),
        _resolve_repo_path(_relative_run_path(base_run_id, "stage3", "ppo_model.zip")),
    ]
    for source in sources:
        checkpoint = _forecast_checkpoint(base_run_id, source)
        if checkpoint:
            required.append(_resolve_repo_path(checkpoint))
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Required base artifacts are missing:\n" + "\n".join(missing))


def run(
    *,
    base_run_id: str,
    suffix: str = "merge_timing",
    forecast_sources: str | list[str] | tuple[str, ...] = "constant_velocity,wcdt_v3",
    forecast_ppo_profile: str = "merge_timing",
    forecast_ppo_timesteps: int | None = None,
) -> dict[str, Path]:
    profile = str(forecast_ppo_profile or "default").strip().lower()
    if profile not in VALID_FORECAST_BRANCH_PROFILES:
        raise ValueError(f"forecast PPO profile must be one of {VALID_FORECAST_BRANCH_PROFILES}; got {profile!r}")
    sources = resolve_forecast_sources(forecast_sources=forecast_sources)
    _assert_base_artifacts(base_run_id, sources)
    cfg = load_config()
    base_run_dir = Path(cfg.run.output_root) / base_run_id
    generated_dir = base_run_dir / "generated_configs"
    written: dict[str, Path] = {}
    stage_log("forecast_branches", f"base_run_id={base_run_id}")
    stage_log("forecast_branches", f"sources={sources}")
    stage_log("forecast_branches", f"profile={profile}")
    for source in sources:
        payload = _forecast_training_payload(
            base_run_id=base_run_id,
            source=source,
            suffix=suffix,
            profile=profile,
            timesteps=forecast_ppo_timesteps,
        )
        config_path = _write_yaml(generated_dir / f"forecast_{_source_suffix(source)}_{suffix}_ppo.yaml", payload)
        written[f"forecast_{_source_suffix(source)}_{suffix}_ppo"] = config_path
        forecast_run_id = _merge_timing_run_id(base_run_id, source, suffix)
        stage_log("forecast_branches", f"train source={source} run_id={forecast_run_id} config={config_path}")
        stage3_train_ppo.run(load_config(config_path))
    stage5_path = _write_yaml(generated_dir / f"stage5_{suffix}_groups.yaml", _stage5_payload(base_run_id, sources, suffix))
    written[f"stage5_{suffix}_groups"] = stage5_path
    stage_log("forecast_branches", f"stage5_config={stage5_path}")
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Train forecast PPO branches from an existing base run.")
    parser.add_argument("--base-run-id", required=True)
    parser.add_argument("--suffix", default="merge_timing")
    parser.add_argument("--forecast-sources", default="constant_velocity,wcdt_v3")
    parser.add_argument("--forecast-ppo-profile", choices=list(VALID_FORECAST_BRANCH_PROFILES), default="merge_timing")
    parser.add_argument("--forecast-ppo-timesteps", type=int, default=None)
    args = parser.parse_args()
    run(
        base_run_id=str(args.base_run_id),
        suffix=str(args.suffix),
        forecast_sources=str(args.forecast_sources),
        forecast_ppo_profile=str(args.forecast_ppo_profile),
        forecast_ppo_timesteps=args.forecast_ppo_timesteps,
    )


if __name__ == "__main__":
    main()
