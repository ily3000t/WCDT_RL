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


def run(cfg) -> Path:
    stage_dir = prepare_run_dir(cfg, "stage5")
    seeds = [int(seed) for seed in cfg.stage5.seeds[: int(cfg.stage5.episodes_per_group)]]
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
        group_cfg = clone_with_overrides(
            cfg,
            {
                "forecast_features": {"enabled": bool(group.forecast_features)},
                "rl": {"use_wcdt_forecast_features": bool(group.forecast_features)},
                "shield": {"enabled": bool(group.shield)},
            },
        )
        model_path = Path(group.get("model_path") or default_model)
        stage_log(
            "stage5",
            f"group={group.name} forecast={bool(group.forecast_features)} shield={bool(group.shield)} model={model_path}",
        )
        try:
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
        except Exception as exc:
            stage_log("stage5", f"group={group.name} skipped: {exc}")
            group_reports[group.name] = {
                "skipped": True,
                "reason": str(exc),
                "model_path": str(model_path),
                "forecast_features": bool(group.forecast_features),
                "shield": bool(group.shield),
            }

    shield_off = {name: report for name, report in group_reports.items() if not name.endswith("shield")}
    shield_on = {name: report for name, report in group_reports.items() if name.endswith("shield") or name == "full_prediction_shield"}
    write_report(stage_dir / "shield_off_metrics.json", shield_off)
    write_report(stage_dir / "shield_on_metrics.json", shield_on)
    report = {
        "stage": "stage5",
        "paired_eval": bool(cfg.stage5.paired_eval),
        "seeds": seeds,
        "groups": group_reports,
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
