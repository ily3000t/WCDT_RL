from __future__ import annotations

import json
from pathlib import Path

from safe_rl.prediction.forecast_feature_augmentor import ForecastFeatureAugmentor
from safe_rl.prediction.actor_selector import actor_selection_config_hash
from safe_rl.prediction.forecast_rollout_bundle import FORECAST_ROLLOUT_BUNDLE_VERSION
from safe_rl.pipeline.common import load_stage_config, make_env, parse_config_arg, write_report
from safe_rl.rl.ppo import train_ppo
from safe_rl.sim.metrics import SAFETY_METRIC_VERSION
from safe_rl.utils.config import prepare_run_dir
from safe_rl.utils.progress import stage_log


def _prediction_loss_summary(checkpoint: str | None, forecast_source: str | None = None) -> dict | None:
    if not checkpoint:
        return None
    source = str(forecast_source or "").strip().lower()
    if source == "constant_velocity":
        return None
    if source in {"wcdt_v2", "wcdt_v3"}:
        return _prediction_loss_summary_from_checkpoint(checkpoint)
    report_path = Path(checkpoint).parent / "stage2_training_report.json"
    if not report_path.exists():
        return _prediction_loss_summary_from_checkpoint(checkpoint)
    with report_path.open("r", encoding="utf-8") as file:
        report = json.load(file)
    history = report.get("prediction_loss_history")
    if not history:
        initial_path = report.get("initial_prediction_report")
        if initial_path and Path(initial_path).exists():
            with Path(initial_path).open("r", encoding="utf-8") as file:
                history = json.load(file).get("prediction_loss_history")
    if not history:
        return _prediction_loss_summary_from_checkpoint(checkpoint)
    if not history:
        return report.get("prediction_skip_reason") and {"prediction_skip_reason": report["prediction_skip_reason"]}
    return {
        "epochs": len(history),
        "first": float(history[0]),
        "last": float(history[-1]),
        "min": float(min(history)),
    }


def _prediction_loss_summary_from_checkpoint(checkpoint: str) -> dict | None:
    try:
        import torch
        payload = torch.load(checkpoint, map_location="cpu")
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    history = payload.get("loss_history")
    member_histories = payload.get("member_histories", [])
    if member_histories:
        return {
            "source": "checkpoint_member_histories",
            "ensemble_size": int(payload.get("ensemble_size", len(member_histories))),
            "architecture_version": payload.get("architecture_version"),
            "loss_version": payload.get("loss_version"),
            "trajectory_schema_version": payload.get("trajectory_schema_version"),
            "actor_selection_version": payload.get("actor_selection_version"),
            "actor_selection_config_hash": payload.get("actor_selection_config_hash"),
            "max_actor_count": payload.get("max_actor_count"),
            "members": [
                {
                    "member": int(item.get("member", index)),
                    "trained_epochs": int(item.get("trained_epochs", len(item.get("loss_history", [])))),
                    "best_epoch": int(item.get("best_epoch", 0)),
                    "best_val_score": float(item.get("best_val_score", 0.0)),
                    "stopped_early": bool(item.get("stopped_early", False)),
                }
                for index, item in enumerate(member_histories)
            ],
        }
    if not history:
        return None
    return {
        "epochs": len(history),
        "first": float(history[0]),
        "last": float(history[-1]),
        "min": float(min(history)),
        "source": "checkpoint",
    }


def run(cfg):
    cfg.shield["forecast_task_shadow_enabled"] = False
    cfg.shield["task_backstop_enabled"] = False
    stage_dir = prepare_run_dir(cfg, "stage3")
    stage_log("stage3", f"run_id={cfg.run.run_id}")
    stage_log("stage3", f"SUMO config={cfg.scenario.sumocfg}")
    stage_log("stage3", f"total_timesteps={cfg.rl.total_timesteps}")
    stage_log("stage3", f"forecast_features={bool(cfg.forecast_features.enabled or cfg.rl.use_wcdt_forecast_features)}")
    stage_log(
        "stage3",
        f"device={cfg.get('training', {}).get('ppo_device', cfg.get('training', {}).get('device', 'auto'))}",
    )
    stage_log("stage3", f"model_output={stage_dir / str(cfg.stage3.model_name)}")
    env = make_env(
        cfg,
        seed=int(cfg.run.seed),
        shield_enabled=False,
        worker_rank=0,
        num_envs=max(1, int(cfg.get("training", {}).get("ppo_num_envs", 1))),
        advance_episode_seed=True,
    )
    try:
        observation_shape = list(env.observation_space.shape)
        prediction_checkpoint = None
        if env.forecast_augmentor is not None and env.forecast_augmentor.predictor is not None:
            prediction_checkpoint = getattr(env.forecast_augmentor.predictor, "checkpoint_path", None)
            prediction_checkpoint = str(prediction_checkpoint) if prediction_checkpoint is not None else None
        report = train_ppo(
            cfg,
            env,
            stage_dir / str(cfg.stage3.model_name),
            tensorboard_dir=stage_dir / "tensorboard",
        )
    finally:
        env.close()
    report["stage"] = "stage3"
    report["forecast_features_enabled"] = bool(cfg.forecast_features.enabled or cfg.rl.use_wcdt_forecast_features)
    report["forecast_source"] = str(cfg.forecast_features.get("source", ""))
    report["prediction_checkpoint"] = prediction_checkpoint
    report["prediction_loss_summary"] = _prediction_loss_summary(prediction_checkpoint, report["forecast_source"])
    report["predictor_summary"] = report["prediction_loss_summary"]
    report["forecast_feature_summary"] = {
        "feature_dim": ForecastFeatureAugmentor.feature_dim(cfg),
        "feature_names": list(ForecastFeatureAugmentor.FEATURE_NAMES),
        "source": str(cfg.forecast_features.get("source", "")),
    } if report["forecast_features_enabled"] else None
    shield_guided_cfg = dict(cfg.rl.get("shield_guided_reward", {}) or {})
    reward_profile = str(cfg.rl.get("reward_profile", "default"))
    report["reward_risk_checkpoint"] = (
        str(shield_guided_cfg.get("risk_checkpoint", ""))
        if reward_profile in {"shield_guided_forecast", "merge_timing_forecast"}
        else ""
    )
    report["shield_guided_reward_config"] = (
        shield_guided_cfg if reward_profile in {"shield_guided_forecast", "merge_timing_forecast"} else None
    )
    report["merge_timing_reward_config"] = (
        dict(cfg.rl.get("merge_timing_reward", {}) or {}) if reward_profile == "merge_timing_forecast" else None
    )
    report["observation_dim"] = int(observation_shape[0]) if observation_shape else 0
    report["observation_shape"] = observation_shape
    report["safety_metric_version"] = str(
        cfg.risk_module.get("safety_metric_version", SAFETY_METRIC_VERSION)
    )
    report["forecast_rollout_bundle_version"] = FORECAST_ROLLOUT_BUNDLE_VERSION
    report["forecast_rollout_bundle_config_hash"] = actor_selection_config_hash(cfg)
    report["sumo_installation"] = {
        "binary": str(cfg.scenario.get("sumo_binary", "")),
        "version": str(cfg.scenario.get("sumo_version", "")),
        "home": str(cfg.scenario.get("sumo_home", "")),
    }
    write_report(stage_dir / "stage3_training_report.json", report)
    stage_log("stage3", f"tensorboard={stage_dir / 'tensorboard'}")
    stage_log("stage3", f"report={stage_dir / 'stage3_training_report.json'}")
    return stage_dir / str(cfg.stage3.model_name)


def main() -> None:
    args = parse_config_arg("Stage3 PPO training")
    cfg = load_stage_config(args)
    run(cfg)


if __name__ == "__main__":
    main()
