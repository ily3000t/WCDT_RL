from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from safe_rl.pipeline.common import run_root, write_report
from safe_rl.pipeline.stage5_paired_eval import (
    _group_model_path,
    _group_overrides,
    _paired_delta,
    _risk_path,
    _select_eval_seeds,
    _shield_acceptance,
)
from safe_rl.rl.evaluation import evaluate_ppo
from safe_rl.utils.config import REPO_ROOT, clone_with_overrides, load_config, _to_config_dict
from safe_rl.utils.progress import TensorboardLogger, stage_log


DEFAULT_VARIANTS = (
    {"activation_risk_threshold": 0.90, "replacement_margin": 0.15},
    {"activation_risk_threshold": 0.85, "replacement_margin": 0.15},
    {"activation_risk_threshold": 0.85, "replacement_margin": 0.10},
    {"activation_risk_threshold": 0.80, "replacement_margin": 0.10},
)

AGGRESSIVE_VARIANTS = (
    {"activation_risk_threshold": 0.75, "replacement_margin": 0.10},
    {"activation_risk_threshold": 0.70, "replacement_margin": 0.10},
    {"activation_risk_threshold": 0.70, "replacement_margin": 0.05},
    {"activation_risk_threshold": 0.60, "replacement_margin": 0.05},
)


def sweep_variants(include_aggressive: bool = False) -> tuple[dict[str, float], ...]:
    if not include_aggressive:
        return DEFAULT_VARIANTS
    seen: set[tuple[float, float]] = set()
    variants: list[dict[str, float]] = []
    for variant in (*DEFAULT_VARIANTS, *AGGRESSIVE_VARIANTS):
        key = (float(variant["activation_risk_threshold"]), float(variant["replacement_margin"]))
        if key in seen:
            continue
        seen.add(key)
        variants.append(dict(variant))
    return tuple(variants)


def _variant_name(prefix: str, variant: dict[str, float], calibrated: bool = False) -> str:
    activation = int(round(float(variant["activation_risk_threshold"]) * 100))
    margin = int(round(float(variant["replacement_margin"]) * 100))
    calibration_suffix = "_cal" if calibrated else ""
    return f"{prefix}{calibration_suffix}_a{activation:03d}_m{margin:03d}"


def _run_path(run_id: str, stage: str, name: str) -> str:
    return (Path("safe_rl_output") / "runs" / run_id / stage / name).as_posix()


def _forecast_run_id(run_id: str, source: str) -> str:
    if source == "constant_velocity":
        suffix = "cv"
    elif source == "wcdt_v2":
        suffix = "wcdt_v2"
    else:
        suffix = "wcdt"
    return f"{run_id}_forecast_{suffix}"


def _forecast_model_exists(run_id: str, source: str) -> bool:
    path = REPO_ROOT / _run_path(_forecast_run_id(run_id, source), "stage3", "ppo_model.zip")
    return path.exists()


def _calibrated_group(group: dict[str, Any]) -> dict[str, Any]:
    calibrated = dict(group)
    calibrated["name"] = str(group["name"]).replace("_a", "_cal_a", 1)
    calibrated["risk_module_overrides"] = {"calibration": {"use_for_runtime": True}}
    return calibrated


def build_sweep_groups(
    run_id: str,
    variants: tuple[dict[str, float], ...] = DEFAULT_VARIANTS,
    include_calibrated: bool = False,
) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = [
        {
            "name": "ppo",
            "forecast_features": False,
            "shield": False,
            "model_path": _run_path(run_id, "stage3", "ppo_model.zip"),
        }
    ]
    for variant in variants:
        raw_group = {
            "name": _variant_name("ppo_shield", variant),
            "forecast_features": False,
            "shield": True,
            "model_path": _run_path(run_id, "stage3", "ppo_model.zip"),
            "shield_overrides": {**variant, "allow_fallback": False},
        }
        groups.append(raw_group)
        if include_calibrated:
            groups.append(_calibrated_group(raw_group))

    for source, base_name, shield_prefix in (
        ("constant_velocity", "ppo_cv_features", "cv_prediction_shield"),
        ("wcdt", "ppo_wcdt_features", "wcdt_prediction_shield"),
        ("wcdt_v2", "ppo_wcdt_v2_features", "wcdt_v2_prediction_shield"),
    ):
        if not _forecast_model_exists(run_id, source):
            continue
        forecast_run = _forecast_run_id(run_id, source)
        base = {
            "name": base_name,
            "forecast_features": True,
            "shield": False,
            "model_path": _run_path(forecast_run, "stage3", "ppo_model.zip"),
            "forecast_source": source,
        }
        if source in ("wcdt", "wcdt_v2"):
            checkpoint_name = "wcdt_v2_predictor.pt" if source == "wcdt_v2" else "wcdt_predictor.pt"
            base["forecast_checkpoint"] = _run_path(run_id, "stage2", checkpoint_name)
        groups.append(base)
        for variant in variants:
            shield_group = {
                "name": _variant_name(shield_prefix, variant),
                "forecast_features": True,
                "shield": True,
                "model_path": _run_path(forecast_run, "stage3", "ppo_model.zip"),
                "forecast_source": source,
                "shield_overrides": {**variant, "allow_fallback": False},
            }
            if source in ("wcdt", "wcdt_v2"):
                checkpoint_name = "wcdt_v2_predictor.pt" if source == "wcdt_v2" else "wcdt_predictor.pt"
                shield_group["forecast_checkpoint"] = _run_path(run_id, "stage2", checkpoint_name)
            groups.append(shield_group)
            if include_calibrated:
                groups.append(_calibrated_group(shield_group))
    return groups


def _sweep_payload(run_id: str, groups: list[dict[str, Any]], seeds: list[int]) -> dict[str, Any]:
    return {
        "run": {"run_id": run_id},
        "stage5": {
            "episodes_per_group": len(seeds),
            "seeds": seeds,
            "groups": groups,
        },
    }


def _base_group_for(name: str) -> str | None:
    if name.startswith("ppo_shield"):
        return "ppo"
    if name.startswith("cv_prediction_shield"):
        return "ppo_cv_features"
    if name.startswith("wcdt_prediction_shield"):
        return "ppo_wcdt_features"
    if name.startswith("wcdt_v2_prediction_shield"):
        return "ppo_wcdt_v2_features"
    return None


def _summary(values: list[float]) -> dict[str, float | int]:
    arr = np.asarray(values, dtype=np.float32)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"count": 0}
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "p01": float(np.percentile(arr, 1)),
        "p05": float(np.percentile(arr, 5)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _reason_ratios(counts: dict[str, int], total: int) -> dict[str, float]:
    if total <= 0:
        return {}
    return {key: float(value / total) for key, value in sorted(counts.items())}


def _shield_score_diagnostics(report: dict[str, Any]) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for episode in report.get("episodes", []):
        records.extend(episode.get("shield_score_records", []) or [])
    reason_counts: dict[str, int] = {}
    for record in records:
        reason = str(record.get("replacement_reason", ""))
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
    raw_scores = [float(record.get("raw_risk_score", 0.0)) for record in records]
    best_scores = [float(record.get("best_candidate_risk_score", 0.0)) for record in records]
    replacement_delta = [float(record.get("replacement_risk_delta", 0.0)) for record in records]
    best_delta = [float(record.get("best_candidate_risk_delta", 0.0)) for record in records]
    activation = float((report.get("shield_overrides") or {}).get("activation_risk_threshold", 0.90))
    return {
        "record_count": int(len(records)),
        "raw_risk_score": _summary(raw_scores),
        "best_candidate_risk_score": _summary(best_scores),
        "replacement_risk_delta": _summary(replacement_delta),
        "best_candidate_risk_delta": _summary(best_delta),
        "raw_risk_activation_margin": _summary([score - activation for score in raw_scores]),
        "reason_counts": reason_counts,
        "reason_ratios": _reason_ratios(reason_counts, len(records)),
    }


def _variant_report(base: dict, candidate: dict) -> dict[str, Any]:
    metrics = candidate.get("metrics", {})
    base_metrics = base.get("metrics", {})
    delta = _paired_delta(base, candidate)
    acceptance = _shield_acceptance(base, candidate)
    return {
        "metrics": {
            "average_reward": metrics.get("average_reward", 0.0),
            "min_distance_p1": metrics.get("min_distance_p1", 0.0),
            "ttc_p1": metrics.get("ttc_p1", 0.0),
            "drac_p99": metrics.get("drac_p99", 0.0),
            "actual_replacement_rate": metrics.get("actual_replacement_rate", 0.0),
            "mean_actual_replacements": metrics.get("mean_actual_replacements", 0.0),
            "fallback_rate": metrics.get("fallback_rate", 0.0),
            "near_miss_rate": metrics.get("near_miss_rate", 0.0),
            "collision_rate": metrics.get("collision_rate", 0.0),
        },
        "delta": delta,
        "acceptance": acceptance,
        "shield_score_diagnostics": _shield_score_diagnostics(candidate),
        "improved_tail": bool(
            float(metrics.get("min_distance_p1", 0.0)) > float(base_metrics.get("min_distance_p1", 0.0))
            or float(metrics.get("drac_p99", 0.0)) < float(base_metrics.get("drac_p99", 0.0))
        ),
    }


def _recommend_variant(variants: dict[str, dict[str, Any]]) -> str | None:
    best_name = None
    best_score = None
    for name, item in variants.items():
        acceptance = item.get("acceptance", {})
        metrics = item.get("metrics", {})
        delta = item.get("delta") or {}
        if acceptance.get("shield_regression", False):
            continue
        if float(metrics.get("mean_actual_replacements", 0.0)) <= 0.0:
            continue
        if not item.get("improved_tail", False):
            continue
        score = (
            float(delta.get("mean_min_distance_delta", 0.0))
            - 0.05 * float(delta.get("mean_drac_delta", 0.0))
            + 0.01 * float(delta.get("mean_reward_delta", 0.0))
        )
        if best_score is None or score > best_score:
            best_name = name
            best_score = score
    return best_name


def _is_calibrated_variant(name: str) -> bool:
    return "_cal_a" in name


def _raw_variant_name(name: str) -> str:
    return name.replace("_cal_a", "_a")


def _variant_family(name: str) -> tuple[str, bool] | None:
    marker = "_cal_a" if _is_calibrated_variant(name) else "_a"
    if marker not in name:
        return None
    return name.split(marker, 1)[0], _is_calibrated_variant(name)


def _metric_delta(left: dict[str, Any], right: dict[str, Any]) -> dict[str, float]:
    left_metrics = left.get("metrics", {})
    right_metrics = right.get("metrics", {})
    keys = (
        "average_reward",
        "min_distance_p1",
        "ttc_p1",
        "drac_p99",
        "actual_replacement_rate",
        "mean_actual_replacements",
        "fallback_rate",
        "near_miss_rate",
        "collision_rate",
    )
    return {
        f"{key}_delta": float(right_metrics.get(key, 0.0)) - float(left_metrics.get(key, 0.0))
        for key in keys
    }


def _calibration_effect_summary(variants: dict[str, dict[str, Any]], include_calibrated: bool) -> dict[str, Any]:
    raw_variants = {name: item for name, item in variants.items() if not _is_calibrated_variant(name)}
    calibrated_variants = {name: item for name, item in variants.items() if _is_calibrated_variant(name)}
    pairs: dict[str, dict[str, Any]] = {}
    for calibrated_name, calibrated_item in calibrated_variants.items():
        raw_name = _raw_variant_name(calibrated_name)
        raw_item = raw_variants.get(raw_name)
        if raw_item is None:
            continue
        delta = _metric_delta(raw_item, calibrated_item)
        pairs[raw_name] = {
            "raw_variant": raw_name,
            "calibrated_variant": calibrated_name,
            "metric_delta_calibrated_minus_raw": delta,
            "replacement_rate_changed": abs(delta["actual_replacement_rate_delta"]) > 1.0e-6,
            "mean_replacements_changed": abs(delta["mean_actual_replacements_delta"]) > 1.0e-6,
            "calibrated_regression": bool(
                calibrated_item.get("acceptance", {}).get("shield_regression", False)
            ),
        }
    raw_recommended = _recommend_variant(raw_variants)
    calibrated_recommended = _recommend_variant(calibrated_variants)
    normalized_calibrated = _raw_variant_name(calibrated_recommended) if calibrated_recommended else None
    recommendation_changed = (
        bool(raw_recommended or normalized_calibrated) and raw_recommended != normalized_calibrated
    )
    return {
        "available": bool(include_calibrated and pairs),
        "include_calibrated": bool(include_calibrated),
        "raw_variant_count": len(raw_variants),
        "calibrated_variant_count": len(calibrated_variants),
        "paired_variant_count": len(pairs),
        "raw_recommended_variant": raw_recommended,
        "calibrated_recommended_variant": calibrated_recommended,
        "recommended_variant_changed": recommendation_changed,
        "calibrated_regression_count": sum(1 for item in pairs.values() if item["calibrated_regression"]),
        "replacement_behavior_changed_count": sum(
            1
            for item in pairs.values()
            if item["replacement_rate_changed"] or item["mean_replacements_changed"]
        ),
        "pairs": pairs,
    }


def _threshold_sensitivity_summary(variants: dict[str, dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, bool], list[tuple[str, dict[str, Any]]]] = {}
    for name, item in variants.items():
        family = _variant_family(name)
        if family is None:
            continue
        grouped.setdefault(family, []).append((name, item))

    families: dict[str, dict[str, Any]] = {}
    raw_sensitive = False
    calibrated_sensitive = False
    for (prefix, calibrated), rows in sorted(grouped.items()):
        if len(rows) < 2:
            continue
        replacement_rates = [
            float(item.get("metrics", {}).get("actual_replacement_rate", 0.0)) for _, item in rows
        ]
        mean_replacements = [
            float(item.get("metrics", {}).get("mean_actual_replacements", 0.0)) for _, item in rows
        ]
        rate_range = max(replacement_rates) - min(replacement_rates)
        mean_range = max(mean_replacements) - min(mean_replacements)
        threshold_sensitive = bool(rate_range > 0.05 or mean_range > 0.25)
        key = f"{prefix}{'_calibrated' if calibrated else '_raw'}"
        families[key] = {
            "group_count": len(rows),
            "variant_names": [name for name, _ in rows],
            "replacement_rate_min": min(replacement_rates),
            "replacement_rate_max": max(replacement_rates),
            "replacement_rate_range": rate_range,
            "mean_replacements_min": min(mean_replacements),
            "mean_replacements_max": max(mean_replacements),
            "mean_replacements_range": mean_range,
            "threshold_sensitive": threshold_sensitive,
        }
        if calibrated:
            calibrated_sensitive = calibrated_sensitive or threshold_sensitive
        else:
            raw_sensitive = raw_sensitive or threshold_sensitive

    raw_families = [item for key, item in families.items() if key.endswith("_raw")]
    return {
        "available": bool(families),
        "families": families,
        "risk_score_saturation_suspected": bool(raw_families and not raw_sensitive),
        "calibration_helpful_for_shield": bool(calibrated_sensitive and not raw_sensitive),
        "sensitivity_rule": "sensitive if replacement_rate_range > 0.05 or mean_replacements_range > 0.25",
    }


def run(cfg, include_aggressive: bool = False, include_calibrated: bool = False) -> Path:
    stage_dir = run_root(cfg) / "stage5_sweep"
    stage_dir.mkdir(parents=True, exist_ok=True)
    generated_dir = stage_dir / "generated_configs"
    generated_dir.mkdir(parents=True, exist_ok=True)
    seeds = _select_eval_seeds(cfg)
    variants_to_run = sweep_variants(include_aggressive=include_aggressive)
    groups = build_sweep_groups(str(cfg.run.run_id), variants_to_run, include_calibrated=include_calibrated)
    payload = _sweep_payload(str(cfg.run.run_id), groups, seeds)
    config_path = generated_dir / "stage5_shield_sweep.yaml"
    with config_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(payload, file, sort_keys=False, allow_unicode=True)

    risk_checkpoint = str(_risk_path(cfg))
    tb = TensorboardLogger(stage_dir / "tensorboard", enabled=bool(cfg.run.get("tensorboard", True)))
    replay_dir = stage_dir / "replay"
    group_reports: dict[str, dict] = {}
    stage_log("stage5_sweep", f"run_id={cfg.run.run_id}")
    stage_log("stage5_sweep", f"groups={len(groups)} seeds={seeds}")
    for group_idx, group_dict in enumerate(groups):
        group = _to_config_dict(yaml.safe_load(yaml.safe_dump(group_dict)))
        group_cfg = clone_with_overrides(cfg, _group_overrides(group))
        model_path = _group_model_path(group, Path(_run_path(str(cfg.run.run_id), "stage3", "ppo_model.zip")))
        stage_log("stage5_sweep", f"group={group['name']} model={model_path}")
        report = evaluate_ppo(
            group_cfg,
            model_path,
            seeds=seeds,
            shield_enabled=bool(group["shield"]),
            risk_checkpoint=risk_checkpoint if bool(group["shield"]) else None,
            replay_dir=replay_dir if bool(cfg.stage5.get("replay_enabled", True)) and bool(cfg.run.get("replay", True)) else None,
            group_name=str(group["name"]),
            tensorboard=tb,
            tensorboard_step_offset=group_idx * max(1, len(seeds)),
        )
        report["shield_overrides"] = dict(group.get("shield_overrides", {}) or {})
        report["risk_module_overrides"] = dict(group.get("risk_module_overrides", {}) or {})
        report["forecast_source"] = str(group.get("forecast_source", ""))
        group_reports[str(group["name"])] = report
    tb.close()

    variants: dict[str, dict[str, Any]] = {}
    for name, report in group_reports.items():
        base_name = _base_group_for(name)
        if base_name and base_name in group_reports:
            variants[name] = _variant_report(group_reports[base_name], report)
    recommended = _recommend_variant(variants)
    calibration_effect = _calibration_effect_summary(variants, include_calibrated=include_calibrated)
    threshold_sensitivity = _threshold_sensitivity_summary(variants)
    final_report = {
        "stage": "stage5_sweep",
        "run_id": cfg.run.run_id,
        "config": str(config_path),
        "risk_checkpoint": risk_checkpoint,
        "seeds": seeds,
        "include_aggressive": bool(include_aggressive),
        "include_calibrated": bool(include_calibrated),
        "sweep_variants": list(variants_to_run),
        "groups": group_reports,
        "variants": variants,
        "calibration_effect_summary": calibration_effect,
        "threshold_sensitivity_summary": threshold_sensitivity,
        "recommended_variant": recommended,
        "recommendation_reason": (
            "selected non-regressive variant with actual replacements and improved min_distance_p1 or drac_p99"
            if recommended
            else "no sweep variant improved tail safety without regression; keep default 0.90/0.15"
        ),
    }
    report_path = stage_dir / "shield_sweep_report.json"
    write_report(report_path, final_report)
    stage_log("stage5_sweep", f"report={report_path}")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage5 Shield threshold sweep")
    parser.add_argument("--config", default=None, help="Optional YAML config overlay.")
    parser.add_argument("--run-id", required=True, help="Existing run id to evaluate.")
    parser.add_argument(
        "--include-aggressive",
        action="store_true",
        help="Also scan lower diagnostic thresholds. These are not part of the default recommendation set.",
    )
    parser.add_argument(
        "--include-calibrated",
        action="store_true",
        help="Also evaluate groups with risk_module.calibration.use_for_runtime=true.",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    cfg.run["run_id"] = args.run_id
    run(cfg, include_aggressive=bool(args.include_aggressive), include_calibrated=bool(args.include_calibrated))


if __name__ == "__main__":
    main()
