from __future__ import annotations

from pathlib import Path

from safe_rl.pipeline.common import latest_stage_file, load_stage_config, parse_config_arg, write_report
from safe_rl.rl.evaluation import evaluate_ppo
from safe_rl.utils.config import clone_with_overrides, prepare_run_dir
from safe_rl.utils.progress import TensorboardLogger, stage_log


def _default_model_path(cfg) -> Path:
    return latest_stage_file(cfg, "stage3", str(cfg.stage3.model_name))


def _risk_path(cfg) -> Path:
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
    if group.get("forecast_checkpoint"):
        forecast_overrides["checkpoint"] = str(group.forecast_checkpoint)
    return {
        "forecast_features": forecast_overrides,
        "rl": {"use_wcdt_forecast_features": bool(group.forecast_features)},
        "shield": {"enabled": bool(group.shield)},
    }


def _group_model_path(group, default_model: Path) -> Path:
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
                "intervention_delta": int(item["intervention_count"] - left["intervention_count"]),
                "fallback_delta": int(item["fallback_count"] - left["fallback_count"]),
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
        "mean_intervention_delta": sum(row["intervention_delta"] for row in rows) / len(rows),
        "mean_fallback_delta": sum(row["fallback_delta"] for row in rows) / len(rows),
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
        "fallback_rate_low": float(shield["fallback_rate"]) < 0.10,
    }
    return {
        "available": True,
        "checks": checks,
        "shield_regression": not all(checks.values()),
    }


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
        model_path = _group_model_path(group, default_model)
        stage_log(
            "stage5",
            f"group={group.name} forecast={bool(group.forecast_features)} shield={bool(group.shield)} model={model_path}",
        )
        group_reports[group.name] = evaluate_ppo(
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

    shield_off = {name: report for name, report in group_reports.items() if not name.endswith("shield")}
    shield_on = {name: report for name, report in group_reports.items() if name.endswith("shield") or name == "full_prediction_shield"}
    paired_delta = {
        "ppo_vs_ppo_shield": _paired_delta(group_reports.get("ppo"), group_reports.get("ppo_shield")),
        "ppo_wcdt_features_vs_full_prediction_shield": _paired_delta(
            group_reports.get("ppo_wcdt_features"),
            group_reports.get("full_prediction_shield"),
        ),
    }
    acceptance = {
        "ppo_shield": _shield_acceptance(group_reports.get("ppo"), group_reports.get("ppo_shield")),
        "full_prediction_shield": _shield_acceptance(
            group_reports.get("ppo_wcdt_features"),
            group_reports.get("full_prediction_shield"),
        ),
    }
    write_report(stage_dir / "shield_off_metrics.json", shield_off)
    write_report(stage_dir / "shield_on_metrics.json", shield_on)
    report = {
        "stage": "stage5",
        "paired_eval": bool(cfg.stage5.paired_eval),
        "seeds": seeds,
        "groups": group_reports,
        "paired_delta": paired_delta,
        "acceptance": acceptance,
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
