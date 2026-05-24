from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from safe_rl.pipeline.common import run_root, write_report
from safe_rl.pipeline.run_full_pipeline import _forecast_stage5_groups, _relative_run_path
from safe_rl.pipeline.stage5_paired_eval import (
    _build_acceptance,
    _build_paired_delta,
    _forecast_acceptance,
    _group_model_path,
    _group_overrides,
    _risk_path,
    _shield_acceptance,
)
from safe_rl.rl.evaluation import evaluate_ppo
from safe_rl.utils.config import REPO_ROOT, _to_config_dict, clone_with_overrides, load_config
from safe_rl.utils.progress import TensorboardLogger, stage_log


DEFAULT_FORECAST_SOURCES = ("constant_velocity", "wcdt_v2")

MODEL_ROLE_EXPLANATIONS: dict[str, dict[str, str | bool]] = {
    "ppo": {
        "role": "baseline_policy",
        "observation": "52D base observation",
        "forecast_source": "none",
        "shield_enabled": False,
        "meaning": "Baseline PPO policy without forecast features or Shield.",
    },
    "ppo_shield": {
        "role": "trusted_shield_mainline",
        "observation": "52D base observation",
        "forecast_source": "none",
        "shield_enabled": True,
        "meaning": "Baseline PPO policy with Risk Module Shield action replacement enabled.",
    },
    "ppo_cv_features": {
        "role": "forecast_control",
        "observation": "63D forecast-augmented observation",
        "forecast_source": "constant_velocity",
        "shield_enabled": False,
        "meaning": "Forecast-feature PPO trained with constant-velocity forecast features.",
    },
    "cv_prediction_shield": {
        "role": "forecast_control_with_shield",
        "observation": "63D forecast-augmented observation",
        "forecast_source": "constant_velocity",
        "shield_enabled": True,
        "meaning": "Constant-velocity forecast-feature PPO with Shield enabled.",
    },
    "ppo_wcdt_features": {
        "role": "diagnostic_only",
        "observation": "63D forecast-augmented observation",
        "forecast_source": "wcdt",
        "shield_enabled": False,
        "meaning": "Legacy WcDT v1 forecast-feature PPO kept for ablation/diagnostics only.",
    },
    "wcdt_prediction_shield": {
        "role": "diagnostic_only",
        "observation": "63D forecast-augmented observation",
        "forecast_source": "wcdt",
        "shield_enabled": True,
        "meaning": "Legacy WcDT v1 forecast-feature PPO with Shield, kept for diagnostics only.",
    },
    "ppo_wcdt_v2_features": {
        "role": "recommended_prediction_branch",
        "observation": "63D forecast-augmented observation",
        "forecast_source": "wcdt_v2",
        "shield_enabled": False,
        "meaning": "Recommended WcDT v2 forecast-feature PPO branch.",
    },
    "wcdt_v2_prediction_shield": {
        "role": "best_safety_combo",
        "observation": "63D forecast-augmented observation",
        "forecast_source": "wcdt_v2",
        "shield_enabled": True,
        "meaning": "WcDT v2 forecast-feature PPO with Shield enabled.",
    },
}

REPORTING_RECOMMENDATION: list[dict[str, str]] = [
    {
        "comparison": "ppo_vs_ppo_shield",
        "purpose": "Report the trusted Shield mainline against the baseline PPO.",
    },
    {
        "comparison": "ppo_cv_features_vs_ppo_wcdt_v2_features",
        "purpose": "Report WcDT v2 forecast features against the constant-velocity forecast control.",
    },
    {
        "comparison": "ppo_wcdt_v2_features_vs_wcdt_v2_prediction_shield",
        "purpose": "Report whether Shield further improves the WcDT v2 forecast policy.",
    },
    {
        "comparison": "legacy_wcdt_v1",
        "purpose": "Keep old WcDT v1 groups in ablation/diagnostic tables only.",
    },
]

FINAL_RESULT_SUMMARY: dict[str, str | list[str]] = {
    "trusted_mainline": ["ppo", "ppo_shield"],
    "forecast_control": "ppo_cv_features",
    "recommended_prediction_branch": "ppo_wcdt_v2_features",
    "best_safety_combo": "wcdt_v2_prediction_shield",
    "diagnostic_only": ["ppo_wcdt_features", "wcdt_prediction_shield"],
}


def _resolve_repo_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def _base_groups(run_id: str) -> list[dict[str, Any]]:
    return [
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


def build_confirmatory_payload(
    run_id: str,
    episodes: int = 50,
    forecast_sources: tuple[str, ...] | list[str] = DEFAULT_FORECAST_SOURCES,
) -> dict[str, Any]:
    groups = _base_groups(run_id)
    for source in forecast_sources:
        groups.extend(_forecast_stage5_groups(run_id, str(source)))
    return {
        "run": {"run_id": run_id},
        "stage5": {
            "episodes_per_group": int(episodes),
            "seeds": list(range(1, int(episodes) + 1)),
            "groups": groups,
        },
    }


def _required_paths(payload: dict[str, Any]) -> list[Path]:
    paths: list[Path] = []
    for group in payload.get("stage5", {}).get("groups", []):
        model_path = group.get("model_path")
        if model_path:
            paths.append(_resolve_repo_path(model_path))
        checkpoint = group.get("forecast_checkpoint")
        if checkpoint:
            paths.append(_resolve_repo_path(checkpoint))
    paths.append(_resolve_repo_path(_relative_run_path(str(payload["run"]["run_id"]), "stage2", "risk_module.pt")))
    return paths


def validate_confirmatory_inputs(payload: dict[str, Any]) -> None:
    missing = [str(path) for path in _required_paths(payload) if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Stage5 confirmatory eval requires existing model/checkpoint files; missing: "
            + ", ".join(missing)
        )


def _metric(report: dict | None, key: str, default: float = 0.0) -> float:
    if not report:
        return default
    return float(report.get("metrics", {}).get(key, default))


def _mainline_shield_summary(group_reports: dict[str, dict]) -> dict[str, Any]:
    baseline = group_reports.get("ppo")
    shielded = group_reports.get("ppo_shield")
    acceptance = _shield_acceptance(baseline, shielded)
    checks = dict(acceptance.get("checks", {}))
    checks["fallback_rate_zero"] = _metric(shielded, "fallback_rate", 1.0) == 0.0
    checks["actual_replacement_count_positive"] = _metric(shielded, "mean_actual_replacements", 0.0) > 0.0
    return {
        "available": bool(baseline and shielded),
        "checks": checks,
        "pass": bool(baseline and shielded and all(checks.values())),
        "acceptance": acceptance,
    }


def _wcdt_v2_forecast_summary(group_reports: dict[str, dict]) -> dict[str, Any]:
    cv = group_reports.get("ppo_cv_features")
    v2 = group_reports.get("ppo_wcdt_v2_features")
    checks = {
        "reward_not_degraded_vs_cv": _metric(v2, "average_reward") >= _metric(cv, "average_reward") - 5.0,
        "min_distance_p1_not_worse_than_cv": _metric(v2, "min_distance_p1") >= _metric(cv, "min_distance_p1"),
        "drac_p99_not_worse_than_cv": _metric(v2, "drac_p99", 1.0e9) <= _metric(cv, "drac_p99", -1.0),
        "merge_success_complete": _metric(v2, "merge_success_rate") == 1.0,
    }
    return {
        "available": bool(cv and v2),
        "checks": checks,
        "pass": bool(cv and v2 and all(checks.values())),
        "acceptance": _forecast_acceptance(cv, v2),
    }


def _wcdt_v2_shield_summary(group_reports: dict[str, dict]) -> dict[str, Any]:
    base = group_reports.get("ppo_wcdt_v2_features")
    shielded = group_reports.get("wcdt_v2_prediction_shield")
    acceptance = _shield_acceptance(base, shielded)
    mean_replacements = _metric(shielded, "mean_actual_replacements", 0.0)
    not_regressed = not acceptance.get("shield_regression", False)
    shield_not_needed = bool(
        base
        and shielded
        and mean_replacements == 0.0
        and not_regressed
    )
    low_frequency_backstop = bool(base and shielded and 0.0 < mean_replacements <= 0.25 and not_regressed)
    if not base or not shielded:
        shield_status = "unavailable"
    elif acceptance.get("shield_regression", False):
        shield_status = "regression"
    elif shield_not_needed:
        shield_status = "shield_not_needed_on_wcdt_v2_policy"
    elif low_frequency_backstop:
        shield_status = "low_frequency_safety_backstop"
    else:
        shield_status = "active_safety_backstop"
    return {
        "available": bool(base and shielded),
        "shield_not_needed_on_wcdt_v2_policy": shield_not_needed,
        "low_frequency_safety_backstop": low_frequency_backstop,
        "mean_actual_replacements": mean_replacements,
        "shield_status": shield_status,
        "acceptance": acceptance,
    }


def build_confirmatory_summary(group_reports: dict[str, dict], paired_delta: dict, acceptance: dict) -> dict[str, Any]:
    mainline = _mainline_shield_summary(group_reports)
    wcdt_v2 = _wcdt_v2_forecast_summary(group_reports)
    wcdt_v2_shield = _wcdt_v2_shield_summary(group_reports)
    return {
        "final_result_summary": FINAL_RESULT_SUMMARY,
        "model_role_explanations": MODEL_ROLE_EXPLANATIONS,
        "reporting_recommendation": REPORTING_RECOMMENDATION,
        "main_result_groups": {
            "trusted_baseline": ["ppo", "ppo_shield"],
            "forecast_control": "ppo_cv_features",
            "recommended_prediction_branch": "ppo_wcdt_v2_features",
            "diagnostic_only": ["ppo_wcdt_features", "wcdt_prediction_shield"],
        },
        "ppo_shield_mainline": mainline,
        "wcdt_v2_forecast_mainline": wcdt_v2,
        "wcdt_v2_shield": wcdt_v2_shield,
        "overall_pass": bool(mainline.get("pass") and wcdt_v2.get("pass")),
        "paired_delta": paired_delta,
        "acceptance": acceptance,
    }


def run(cfg, episodes: int = 50) -> Path:
    run_id = str(cfg.run.run_id)
    payload = build_confirmatory_payload(run_id, episodes=episodes)
    validate_confirmatory_inputs(payload)
    stage_dir = run_root(cfg) / "stage5_confirmatory"
    stage_dir.mkdir(parents=True, exist_ok=True)
    generated_dir = stage_dir / "generated_configs"
    generated_dir.mkdir(parents=True, exist_ok=True)
    config_path = generated_dir / "stage5_confirmatory.yaml"
    with config_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(payload, file, sort_keys=False, allow_unicode=True)

    risk_checkpoint = str(_risk_path(cfg))
    replay_dir = stage_dir / "replay"
    tb = TensorboardLogger(stage_dir / "tensorboard", enabled=bool(cfg.run.get("tensorboard", True)))
    seeds = [int(seed) for seed in payload["stage5"]["seeds"]]
    group_reports: dict[str, dict] = {}
    stage_log("stage5_confirmatory", f"run_id={run_id}")
    stage_log("stage5_confirmatory", f"episodes={episodes} seeds={seeds}")
    for group_idx, group_dict in enumerate(payload["stage5"]["groups"]):
        group = _to_config_dict(group_dict)
        group_cfg = clone_with_overrides(cfg, _group_overrides(group))
        model_path = _group_model_path(group, Path(_relative_run_path(run_id, "stage3", "ppo_model.zip")))
        stage_log(
            "stage5_confirmatory",
            f"group={group.name} forecast={bool(group.forecast_features)} shield={bool(group.shield)} model={model_path}",
        )
        report = evaluate_ppo(
            group_cfg,
            model_path,
            seeds=seeds,
            shield_enabled=bool(group.shield),
            risk_checkpoint=risk_checkpoint if bool(group.shield) else None,
            replay_dir=replay_dir if bool(cfg.stage5.get("replay_enabled", True)) and bool(cfg.run.get("replay", True)) else None,
            group_name=str(group.name),
            tensorboard=tb,
            tensorboard_step_offset=group_idx * max(1, len(seeds)),
        )
        report["forecast_source"] = str(group.get("forecast_source", "")) if bool(group.forecast_features) else ""
        report["forecast_checkpoint"] = str(group.get("forecast_checkpoint", "")) if bool(group.forecast_features) else ""
        group_reports[str(group.name)] = report
    tb.close()

    paired_delta = _build_paired_delta(group_reports)
    acceptance = _build_acceptance(group_reports)
    summary = build_confirmatory_summary(group_reports, paired_delta, acceptance)
    report = {
        "stage": "stage5_confirmatory",
        "config": str(config_path),
        "seeds": seeds,
        "groups": group_reports,
        "paired_delta": paired_delta,
        "acceptance": acceptance,
        "confirmatory_summary": summary,
    }
    write_report(stage_dir / "formal_paired_eval_report.json", report)
    write_report(stage_dir / "confirmatory_summary.json", summary)
    stage_log("stage5_confirmatory", f"report={stage_dir / 'formal_paired_eval_report.json'}")
    return stage_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage5 confirmatory paired evaluation for final SAFE_RL results")
    parser.add_argument("--config", default=None, help="Optional YAML config overlay.")
    parser.add_argument("--run-id", required=True, help="Existing base run id to evaluate.")
    parser.add_argument("--episodes", type=int, default=50, help="Number of paired seeds to evaluate. Default: 50.")
    args = parser.parse_args()
    cfg = load_config(args.config)
    cfg.run["run_id"] = args.run_id
    run(cfg, episodes=int(args.episodes))


if __name__ == "__main__":
    main()
