from __future__ import annotations

import re
from pathlib import Path

from safe_rl.pipeline.common import latest_stage_file, load_stage_config, parse_config_arg, write_report
from safe_rl.rl.evaluation import evaluate_policy
from safe_rl.sim.metrics import SAFETY_METRIC_VERSION
from safe_rl.utils.config import clone_with_overrides, prepare_run_dir
from safe_rl.utils.progress import TensorboardLogger, stage_log


def _default_model_path(cfg) -> Path:
    configured = cfg.stage5.get("default_model_path")
    if configured:
        return Path(configured)
    return latest_stage_file(cfg, "stage3", str(cfg.stage3.model_name))


def _risk_path(cfg) -> Path:
    configured = cfg.stage5.get("risk_checkpoint")
    if configured:
        return Path(configured)
    return latest_stage_file(cfg, "stage2", "risk_module.pt")


def _select_eval_seeds(cfg) -> list[int]:
    requested = int(cfg.stage5.episodes_per_group)
    seeds = [int(seed) for seed in cfg.stage5.seeds]
    if len(seeds) < requested:
        raise ValueError(
            f"stage5.episodes_per_group={requested} requires at least {requested} seeds, "
            f"but stage5.seeds has {len(seeds)}"
        )
    return seeds[:requested]


def _group_overrides(group) -> dict:
    forecast_overrides = {"enabled": bool(group.forecast_features)}
    if group.get("forecast_source"):
        forecast_overrides["source"] = str(group.forecast_source)
    if group.get("forecast_checkpoint"):
        forecast_overrides["checkpoint"] = str(group.forecast_checkpoint)
    shield_overrides = {"enabled": bool(group.shield)}
    requested_shield_overrides = group.get("shield_overrides")
    if requested_shield_overrides:
        shield_overrides.update(dict(requested_shield_overrides))
    overrides = {
        "forecast_features": forecast_overrides,
        "rl": {"use_wcdt_forecast_features": bool(group.forecast_features)},
        "shield": shield_overrides,
    }
    requested_risk_overrides = group.get("risk_module_overrides")
    if requested_risk_overrides:
        overrides["risk_module"] = dict(requested_risk_overrides)
    return overrides


def _group_model_path(group, default_model: Path) -> Path | None:
    if str(group.get("policy_type", "sb3_ppo")) == "rule_gap_acceptance":
        return None
    model_path = group.get("model_path")
    if bool(group.forecast_features) and not model_path:
        raise ValueError(
            f"stage5 group '{group.name}' enables forecast_features, so it must set "
            "model_path to a PPO checkpoint trained with forecast observations."
        )
    return Path(model_path or default_model)


def _paired_delta(a_report: dict | None, b_report: dict | None) -> dict | None:
    if not a_report or not b_report or "episodes" not in a_report or "episodes" not in b_report:
        return None
    right = {int(item["seed"]): item for item in b_report["episodes"]}
    rows = []
    for left in a_report["episodes"]:
        seed = int(left["seed"])
        if seed not in right:
            continue
        item = right[seed]
        rows.append(
            {
                "seed": seed,
                "reward_delta": float(item["episode_reward"] - left["episode_reward"]),
                "min_distance_delta": float(item["min_distance"] - left["min_distance"]),
                "ttc_delta": float(item["ttc_p1"] - left["ttc_p1"]),
                "drac_delta": float(item["drac_p99"] - left["drac_p99"]),
                "drac_raw_delta": float(
                    item.get("drac_p99_raw", item.get("drac_p99", 0.0))
                    - left.get("drac_p99_raw", left.get("drac_p99", 0.0))
                ),
                "drac_capped_delta": float(
                    item.get("drac_p99_capped", min(float(item.get("drac_p99", 0.0)), 20.0))
                    - left.get("drac_p99_capped", min(float(left.get("drac_p99", 0.0)), 20.0))
                ),
                "proxy_collision_delta": int(
                    int(bool(item.get("proxy_collision", False))) - int(bool(left.get("proxy_collision", False)))
                ),
                "safety_violation_delta": int(
                    int(bool(item.get("safety_violation", False))) - int(bool(left.get("safety_violation", False)))
                ),
                "geometric_overlap_delta": int(
                    int(bool(item.get("geometric_overlap", False))) - int(bool(left.get("geometric_overlap", False)))
                ),
                "proxy_collision_count_delta": int(
                    int(item.get("proxy_collision_count", int(bool(item.get("proxy_collision", False)))))
                    - int(left.get("proxy_collision_count", int(bool(left.get("proxy_collision", False)))))
                ),
                "safety_violation_count_delta": int(
                    int(item.get("safety_violation_count", int(bool(item.get("safety_violation", False)))))
                    - int(left.get("safety_violation_count", int(bool(left.get("safety_violation", False)))))
                ),
                "taper_miss_delta": int(
                    int(bool(item.get("taper_miss", False))) - int(bool(left.get("taper_miss", False)))
                ),
                "min_distance_le_collision_threshold_count_delta": int(
                    int(
                        item.get(
                            "min_distance_le_collision_threshold_count",
                            item.get("proxy_collision_count", int(bool(item.get("proxy_collision", False)))),
                        )
                    )
                    - int(
                        left.get(
                            "min_distance_le_collision_threshold_count",
                            left.get("proxy_collision_count", int(bool(left.get("proxy_collision", False)))),
                        )
                    )
                ),
                "completion_time_delta": float(item.get("completion_time", 0.0) - left.get("completion_time", 0.0)),
                "ego_speed_mean_delta": float(item.get("ego_speed_mean", 0.0) - left.get("ego_speed_mean", 0.0)),
                "hard_brake_rate_delta": float(item.get("hard_brake_rate", 0.0) - left.get("hard_brake_rate", 0.0)),
                "intervention_delta": int(item["intervention_count"] - left["intervention_count"]),
                "actual_replacement_delta": int(
                    item.get("actual_replacement_count", 0) - left.get("actual_replacement_count", 0)
                ),
                "task_replacement_delta": int(
                    item.get("task_replacement_count", 0) - left.get("task_replacement_count", 0)
                ),
                "forecast_ranking_replacement_delta": int(
                    item.get("forecast_ranking_replacement_count", 0)
                    - left.get("forecast_ranking_replacement_count", 0)
                ),
                "fallback_delta": int(item["fallback_count"] - left["fallback_count"]),
                "emergency_fallback_delta": int(
                    item.get("emergency_fallback_count", 0) - left.get("emergency_fallback_count", 0)
                ),
                "missed_safe_merge_opportunity_delta": int(
                    item.get("missed_safe_merge_opportunity_count", 0)
                    - left.get("missed_safe_merge_opportunity_count", 0)
                ),
                "deadline_missed_safe_merge_delta": int(
                    item.get("deadline_missed_safe_merge_count", 0)
                    - left.get("deadline_missed_safe_merge_count", 0)
                ),
                "no_merge_request_before_taper_delta": int(
                    item.get("no_merge_request_before_taper_count", 0)
                    - left.get("no_merge_request_before_taper_count", 0)
                ),
            }
        )
    if not rows:
        return None
    return {
        "episodes": rows,
        "mean_reward_delta": sum(row["reward_delta"] for row in rows) / len(rows),
        "mean_min_distance_delta": sum(row["min_distance_delta"] for row in rows) / len(rows),
        "mean_ttc_delta": sum(row["ttc_delta"] for row in rows) / len(rows),
        "mean_drac_delta": sum(row["drac_delta"] for row in rows) / len(rows),
        "mean_drac_raw_delta": sum(row["drac_raw_delta"] for row in rows) / len(rows),
        "mean_drac_capped_delta": sum(row["drac_capped_delta"] for row in rows) / len(rows),
        "mean_proxy_collision_delta": sum(row["proxy_collision_delta"] for row in rows) / len(rows),
        "mean_safety_violation_delta": sum(row["safety_violation_delta"] for row in rows) / len(rows),
        "mean_geometric_overlap_delta": sum(row["geometric_overlap_delta"] for row in rows) / len(rows),
        "mean_missed_safe_merge_opportunity_delta": (
            sum(row["missed_safe_merge_opportunity_delta"] for row in rows) / len(rows)
        ),
        "proxy_collision_count_delta": sum(row["proxy_collision_count_delta"] for row in rows),
        "safety_violation_count_delta": sum(row["safety_violation_count_delta"] for row in rows),
        "taper_miss_count_delta": sum(row["taper_miss_delta"] for row in rows),
        "mean_taper_miss_delta": sum(row["taper_miss_delta"] for row in rows) / len(rows),
        "min_distance_le_collision_threshold_count_delta": sum(
            row["min_distance_le_collision_threshold_count_delta"] for row in rows
        ),
        "mean_completion_time_delta": sum(row["completion_time_delta"] for row in rows) / len(rows),
        "mean_ego_speed_delta": sum(row["ego_speed_mean_delta"] for row in rows) / len(rows),
        "mean_hard_brake_rate_delta": sum(row["hard_brake_rate_delta"] for row in rows) / len(rows),
        "mean_intervention_delta": sum(row["intervention_delta"] for row in rows) / len(rows),
        "mean_actual_replacement_delta": sum(row["actual_replacement_delta"] for row in rows) / len(rows),
        "mean_task_replacement_delta": sum(row["task_replacement_delta"] for row in rows) / len(rows),
        "task_replacement_count_delta": sum(row["task_replacement_delta"] for row in rows),
        "mean_forecast_ranking_replacement_delta": (
            sum(row["forecast_ranking_replacement_delta"] for row in rows) / len(rows)
        ),
        "forecast_ranking_replacement_count_delta": sum(
            row["forecast_ranking_replacement_delta"] for row in rows
        ),
        "mean_fallback_delta": sum(row["fallback_delta"] for row in rows) / len(rows),
        "mean_emergency_fallback_delta": sum(row["emergency_fallback_delta"] for row in rows) / len(rows),
        "emergency_fallback_count_delta": sum(row["emergency_fallback_delta"] for row in rows),
        "deadline_missed_safe_merge_count_delta": sum(row["deadline_missed_safe_merge_delta"] for row in rows),
        "mean_deadline_missed_safe_merge_delta": (
            sum(row["deadline_missed_safe_merge_delta"] for row in rows) / len(rows)
        ),
        "no_merge_request_before_taper_count_delta": sum(row["no_merge_request_before_taper_delta"] for row in rows),
    }


def _shield_acceptance(baseline: dict | None, shielded: dict | None) -> dict:
    if not baseline or not shielded or "metrics" not in baseline or "metrics" not in shielded:
        return {"available": False, "shield_regression": False}
    base = baseline["metrics"]
    shield = shielded["metrics"]
    checks = {
        "reward_not_degraded": float(shield["average_reward"]) >= float(base["average_reward"]) - 5.0,
        "near_miss_not_worse": float(shield["near_miss_rate"]) <= float(base["near_miss_rate"]),
        "min_distance_not_degraded": float(shield["min_distance_p1"]) >= float(base["min_distance_p1"]) - 1.0,
        "safety_violation_not_worse": float(shield.get("safety_violation_rate", 0.0))
        <= float(base.get("safety_violation_rate", 0.0)),
        "proxy_collision_zero": float(shield.get("proxy_collision_rate", 0.0)) == 0.0,
        "fallback_rate_low": float(shield["fallback_rate"]) < 0.10,
        "fallback_rate_zero": float(shield["fallback_rate"]) == 0.0,
    }
    return {
        "available": True,
        "checks": checks,
        "shield_regression": not all(checks.values()),
    }


def _forecast_acceptance(reference: dict | None, candidate: dict | None) -> dict:
    if not reference or not candidate or "metrics" not in reference or "metrics" not in candidate:
        return {"available": False, "forecast_regression": False}
    ref = reference["metrics"]
    item = candidate["metrics"]
    checks = {
        "reward_not_degraded": float(item["average_reward"]) >= float(ref["average_reward"]) - 5.0,
        "near_miss_not_worse": float(item["near_miss_rate"]) <= float(ref["near_miss_rate"]),
        "min_distance_not_degraded": float(item["min_distance_p1"]) >= float(ref["min_distance_p1"]) - 1.0,
        "safety_violation_not_worse": float(item.get("safety_violation_rate", 0.0))
        <= float(ref.get("safety_violation_rate", 0.0)),
        "proxy_collision_zero": float(item.get("proxy_collision_rate", 0.0)) == 0.0,
    }
    return {
        "available": True,
        "checks": checks,
        "forecast_regression": not all(checks.values()),
    }


def _forecast_baseline_group(group_reports: dict) -> str | None:
    for name in group_reports:
        if name in ("ppo", "ppo_shield", "full_prediction_shield") or str(name).endswith("shield"):
            continue
        report = group_reports.get(name) or {}
        if report.get("forecast_source") or int(report.get("env_observation_shape", [0])[0]) > 52:
            return str(name)
    for name in group_reports:
        if name not in ("ppo", "ppo_shield", "full_prediction_shield") and not str(name).endswith("shield"):
            return str(name)
    return None


def _add_delta(target: dict, key: str, a_report: dict | None, b_report: dict | None) -> None:
    delta = _paired_delta(a_report, b_report)
    if delta is not None:
        target[key] = delta


def _build_paired_delta(group_reports: dict) -> dict:
    paired_delta: dict = {}
    _add_delta(paired_delta, "ppo_vs_ppo_shield", group_reports.get("ppo"), group_reports.get("ppo_shield"))
    for base_name, shield_name in (
        ("ppo_cv_features", "cv_prediction_shield"),
        ("ppo_wcdt_features", "wcdt_prediction_shield"),
        ("ppo_wcdt_v2_features", "wcdt_v2_prediction_shield"),
        ("ppo_wcdt_v3_features", "wcdt_v3_prediction_shield"),
    ):
        _add_delta(
            paired_delta,
            f"{base_name}_vs_{shield_name}",
            group_reports.get(base_name),
            group_reports.get(shield_name),
        )
    _add_delta(paired_delta, "ppo_vs_ppo_cv_features", group_reports.get("ppo"), group_reports.get("ppo_cv_features"))
    _add_delta(
        paired_delta,
        "ppo_cv_features_vs_ppo_wcdt_features",
        group_reports.get("ppo_cv_features"),
        group_reports.get("ppo_wcdt_features"),
    )
    _add_delta(
        paired_delta,
        "ppo_cv_features_vs_ppo_wcdt_v2_features",
        group_reports.get("ppo_cv_features"),
        group_reports.get("ppo_wcdt_v2_features"),
    )
    _add_delta(
        paired_delta,
        "ppo_wcdt_features_vs_ppo_wcdt_v2_features",
        group_reports.get("ppo_wcdt_features"),
        group_reports.get("ppo_wcdt_v2_features"),
    )
    _add_delta(
        paired_delta,
        "ppo_cv_features_vs_ppo_wcdt_v3_features",
        group_reports.get("ppo_cv_features"),
        group_reports.get("ppo_wcdt_v3_features"),
    )
    _add_delta(
        paired_delta,
        "ppo_wcdt_v2_features_vs_ppo_wcdt_v3_features",
        group_reports.get("ppo_wcdt_v2_features"),
        group_reports.get("ppo_wcdt_v3_features"),
    )
    legacy_forecast = _forecast_baseline_group(group_reports)
    if "full_prediction_shield" in group_reports and legacy_forecast:
        _add_delta(
            paired_delta,
            f"{legacy_forecast}_vs_full_prediction_shield",
            group_reports.get(legacy_forecast),
            group_reports.get("full_prediction_shield"),
        )
    return paired_delta


def _build_acceptance(group_reports: dict) -> dict:
    acceptance: dict = {}
    if "ppo_shield" in group_reports:
        acceptance["ppo_shield"] = _shield_acceptance(group_reports.get("ppo"), group_reports.get("ppo_shield"))
    if "cv_prediction_shield" in group_reports:
        acceptance["cv_prediction_shield"] = _shield_acceptance(
            group_reports.get("ppo_cv_features"),
            group_reports.get("cv_prediction_shield"),
        )
    if "wcdt_prediction_shield" in group_reports:
        acceptance["wcdt_prediction_shield"] = _shield_acceptance(
            group_reports.get("ppo_wcdt_features"),
            group_reports.get("wcdt_prediction_shield"),
        )
    if "wcdt_v2_prediction_shield" in group_reports:
        acceptance["wcdt_v2_prediction_shield"] = _shield_acceptance(
            group_reports.get("ppo_wcdt_v2_features"),
            group_reports.get("wcdt_v2_prediction_shield"),
        )
    if "wcdt_v3_prediction_shield" in group_reports:
        acceptance["wcdt_v3_prediction_shield"] = _shield_acceptance(
            group_reports.get("ppo_wcdt_v3_features"),
            group_reports.get("wcdt_v3_prediction_shield"),
        )
    if "ppo_cv_features" in group_reports:
        acceptance["forecast_cv_vs_baseline"] = _forecast_acceptance(
            group_reports.get("ppo"),
            group_reports.get("ppo_cv_features"),
        )
    if "ppo_wcdt_features" in group_reports and "ppo_cv_features" in group_reports:
        acceptance["forecast_wcdt_vs_cv"] = _forecast_acceptance(
            group_reports.get("ppo_cv_features"),
            group_reports.get("ppo_wcdt_features"),
        )
    if "ppo_wcdt_v2_features" in group_reports and "ppo_cv_features" in group_reports:
        acceptance["forecast_wcdt_v2_vs_cv"] = _forecast_acceptance(
            group_reports.get("ppo_cv_features"),
            group_reports.get("ppo_wcdt_v2_features"),
        )
    if "ppo_wcdt_v3_features" in group_reports and "ppo_cv_features" in group_reports:
        acceptance["forecast_wcdt_v3_vs_cv"] = _forecast_acceptance(
            group_reports.get("ppo_cv_features"),
            group_reports.get("ppo_wcdt_v3_features"),
        )
    legacy_forecast = _forecast_baseline_group(group_reports)
    if "full_prediction_shield" in group_reports and legacy_forecast:
        acceptance["full_prediction_shield"] = _shield_acceptance(
            group_reports.get(legacy_forecast),
            group_reports.get("full_prediction_shield"),
        )
    return acceptance


def _comparison_tables(group_reports: dict) -> dict[str, dict]:
    policy: dict[str, dict] = {}
    shield: dict[str, dict] = {}
    high_impact: dict[str, dict] = {}
    for name, report in group_reports.items():
        lowered = str(name).lower()
        metrics = dict(report.get("metrics", {}) or {})
        if "task_backstop" in lowered or "full_ranking" in lowered:
            high_impact[name] = metrics
        elif bool(report.get("shield_enabled", False)):
            shield[name] = metrics
        else:
            policy[name] = metrics
    return {
        "policy_comparison": policy,
        "shield_ablation": shield,
        "high_impact_controller_ablation": high_impact,
    }


def _training_seed_summary(group_reports: dict) -> dict[str, dict]:
    grouped: dict[str, list[dict]] = {}
    for name, report in group_reports.items():
        base = re.sub(r"_seed_\d+$", "", str(name))
        grouped.setdefault(base, []).append(dict(report.get("metrics", {}) or {}))
    result: dict[str, dict] = {}
    for name, metrics_rows in grouped.items():
        numeric_keys = {
            key
            for row in metrics_rows
            for key, value in row.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
        result[name] = {
            "training_seed_count": len(metrics_rows),
            "mean": {
                key: sum(float(row.get(key, 0.0)) for row in metrics_rows) / len(metrics_rows)
                for key in sorted(numeric_keys)
            },
            "std": {
                key: (
                    sum(
                        (float(row.get(key, 0.0))
                        - sum(float(item.get(key, 0.0)) for item in metrics_rows) / len(metrics_rows)) ** 2
                        for row in metrics_rows
                    )
                    / len(metrics_rows)
                ) ** 0.5
                for key in sorted(numeric_keys)
            },
        }
    return result


def run(cfg) -> Path:
    stage_dir = prepare_run_dir(cfg, "stage5")
    seeds = _select_eval_seeds(cfg)
    risk_checkpoint = str(_risk_path(cfg))
    default_model = _default_model_path(cfg)
    stage_log("stage5", f"run_id={cfg.run.run_id}")
    stage_log("stage5", f"seeds={seeds}")
    stage_log("stage5", f"default_model={default_model}")
    stage_log("stage5", f"risk_checkpoint={risk_checkpoint}")
    stage_log("stage5", f"output_dir={stage_dir}")
    tb = TensorboardLogger(stage_dir / "tensorboard", enabled=bool(cfg.run.get("tensorboard", True)))
    replay_dir = stage_dir / "replay"
    group_reports = {}
    for group_idx, group in enumerate(cfg.stage5.groups):
        group_cfg = clone_with_overrides(cfg, _group_overrides(group))
        policy_type = str(group.get("policy_type", "sb3_ppo"))
        if policy_type == "rule_gap_acceptance" and (
            bool(group.shield) or bool(group.forecast_features)
        ):
            raise ValueError(
                "rule_gap_acceptance is an unshielded current-state baseline; "
                "do not enable Shield or forecast features in this comparison group."
            )
        model_path = _group_model_path(group, default_model)
        stage_log(
            "stage5",
            f"group={group.name} policy_type={policy_type} forecast={bool(group.forecast_features)} shield={bool(group.shield)} model={model_path}",
        )
        group_reports[group.name] = evaluate_policy(
            group_cfg,
            model_path,
            seeds=seeds,
            shield_enabled=bool(group.shield),
            risk_checkpoint=risk_checkpoint if bool(group.shield) else None,
            replay_dir=replay_dir if bool(cfg.stage5.get("replay_enabled", True)) and bool(cfg.run.get("replay", True)) else None,
            group_name=str(group.name),
            tensorboard=tb,
            tensorboard_step_offset=group_idx * max(1, len(seeds)),
            policy_type=policy_type,
        )
        if bool(group.forecast_features):
            group_reports[group.name]["forecast_source"] = str(
                group.get("forecast_source", group_cfg.forecast_features.get("source", ""))
            )
            group_reports[group.name]["forecast_checkpoint"] = str(group_cfg.forecast_features.get("checkpoint", ""))
        else:
            group_reports[group.name]["forecast_source"] = None
            group_reports[group.name]["forecast_checkpoint"] = ""
        group_reports[group.name]["shield_enabled"] = bool(group.shield)
        group_reports[group.name]["shield_overrides"] = dict(group.get("shield_overrides", {}) or {})
        group_reports[group.name]["risk_module_overrides"] = dict(group.get("risk_module_overrides", {}) or {})
        group_reports[group.name]["policy_type"] = policy_type

    shield_off = {name: report for name, report in group_reports.items() if not bool(report.get("shield_enabled", False))}
    shield_on = {name: report for name, report in group_reports.items() if bool(report.get("shield_enabled", False))}
    forecast_baseline = _forecast_baseline_group(group_reports)
    paired_delta = _build_paired_delta(group_reports)
    acceptance = _build_acceptance(group_reports)
    write_report(stage_dir / "shield_off_metrics.json", shield_off)
    write_report(stage_dir / "shield_on_metrics.json", shield_on)
    report = {
        "stage": "stage5",
        "paired_eval": bool(cfg.stage5.paired_eval),
        "safety_metric_version": str(
            cfg.risk_module.get("safety_metric_version", SAFETY_METRIC_VERSION)
        ),
        "seeds": seeds,
        "groups": group_reports,
        "forecast_baseline_group": forecast_baseline,
        "paired_delta": paired_delta,
        "acceptance": acceptance,
        "comparison_tables": _comparison_tables(group_reports),
        "training_seed_summary": _training_seed_summary(group_reports),
    }
    write_report(stage_dir / "formal_paired_eval_report.json", report)
    tb.close()
    stage_log("stage5", f"shield_off_metrics={stage_dir / 'shield_off_metrics.json'}")
    stage_log("stage5", f"shield_on_metrics={stage_dir / 'shield_on_metrics.json'}")
    stage_log("stage5", f"report={stage_dir / 'formal_paired_eval_report.json'}")
    return stage_dir


def main() -> None:
    args = parse_config_arg("Stage5 paired shield evaluation")
    cfg = load_stage_config(args)
    run(cfg)


if __name__ == "__main__":
    main()
