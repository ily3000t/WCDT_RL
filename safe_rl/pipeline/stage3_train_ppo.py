from __future__ import annotations

from safe_rl.pipeline.common import load_stage_config, make_env, parse_config_arg, write_report
from safe_rl.rl.ppo import train_ppo
from safe_rl.utils.config import prepare_run_dir
from safe_rl.utils.progress import stage_log


def run(cfg):
    stage_dir = prepare_run_dir(cfg, "stage3")
    stage_log("stage3", f"run_id={cfg.run.run_id}")
    stage_log("stage3", f"SUMO config={cfg.scenario.sumocfg}")
    stage_log("stage3", f"total_timesteps={cfg.rl.total_timesteps}")
    stage_log("stage3", f"forecast_features={bool(cfg.forecast_features.enabled or cfg.rl.use_wcdt_forecast_features)}")
    stage_log("stage3", f"device={cfg.get('training', {}).get('device', 'auto')}")
    stage_log("stage3", f"model_output={stage_dir / str(cfg.stage3.model_name)}")
    env = make_env(cfg, seed=int(cfg.run.seed), shield_enabled=False)
    try:
        observation_shape = list(env.observation_space.shape)
        prediction_checkpoint = None
        if env.forecast_augmentor is not None and env.forecast_augmentor.predictor is not None:
            prediction_checkpoint = getattr(env.forecast_augmentor.predictor, "checkpoint_path", None)
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
    report["observation_dim"] = int(observation_shape[0]) if observation_shape else 0
    report["observation_shape"] = observation_shape
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
