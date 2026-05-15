from __future__ import annotations

from pathlib import Path
from typing import Any

from safe_rl.sim.sumo_highway_merge_env import SumoHighwayMergeEnv


def _require_sb3():
    try:
        from stable_baselines3 import PPO
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Stage3 requires stable-baselines3. Activate the SAFE_RL environment.") from exc
    return PPO


def train_ppo(
    config: Any,
    env: SumoHighwayMergeEnv,
    output_path: str | Path,
    tensorboard_dir: str | Path | None = None,
) -> dict:
    PPO = _require_sb3()
    try:
        from stable_baselines3.common.monitor import Monitor
        env = Monitor(env)
    except Exception:
        pass
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=float(config.rl.learning_rate),
        n_steps=int(config.rl.n_steps),
        batch_size=int(config.rl.batch_size),
        gamma=float(config.rl.gamma),
        gae_lambda=float(config.rl.gae_lambda),
        ent_coef=float(config.rl.ent_coef),
        vf_coef=float(config.rl.vf_coef),
        tensorboard_log=str(tensorboard_dir) if tensorboard_dir else None,
        verbose=1,
    )
    model.learn(
        total_timesteps=int(config.rl.total_timesteps),
        tb_log_name=str(config.stage3.get("tensorboard_log_name", "ppo")),
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(output_path))
    return {
        "model_path": str(output_path),
        "total_timesteps": int(config.rl.total_timesteps),
        "tensorboard": str(tensorboard_dir) if tensorboard_dir else None,
    }


def load_ppo(path: str | Path):
    PPO = _require_sb3()
    return PPO.load(str(path))
