from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import numpy as np

from safe_rl.sim.sumo_highway_merge_env import SumoHighwayMergeEnv
from safe_rl.utils.io import write_json


def _require_sb3():
    try:
        from stable_baselines3 import PPO
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Stage3 requires stable-baselines3. Activate the SAFE_RL environment.") from exc
    return PPO


def _training_device(config: Any) -> str:
    requested = str(config.get("training", {}).get("device", "auto")).strip().lower()
    return "cuda" if requested == "gpu" else requested or "auto"


def _safety_score(metrics: dict[str, Any]) -> float:
    return float(
        float(metrics.get("average_reward", 0.0))
        + 2.0 * float(metrics.get("min_distance_p1", 0.0))
        + 2.0 * float(metrics.get("ttc_p1", 0.0))
        - 2.0 * float(metrics.get("drac_p99_capped", metrics.get("drac_p99", 0.0)))
        - 50.0 * float(metrics.get("proxy_collision_rate", 0.0))
        - 30.0 * float(metrics.get("safety_violation_rate", 0.0))
    )


def _checkpoint_paths(output_path: Path) -> tuple[Path, Path, Path]:
    return (
        output_path.with_name(f"{output_path.stem}_final{output_path.suffix}"),
        output_path.with_name(f"{output_path.stem}_best_safety{output_path.suffix}"),
        output_path.parent / "stage3_checkpoint_selection_report.json",
    )


def _evaluate_model_for_safety(model: Any, config: Any, seeds: list[int]) -> dict[str, Any]:
    from safe_rl.pipeline.common import make_env
    from safe_rl.risk.risk_aggregator import aggregate_episode_reports

    reports: list[dict[str, Any]] = []
    rewards: list[float] = []
    for seed in seeds:
        env = make_env(config, seed=int(seed), shield_enabled=False)
        total_reward = 0.0
        try:
            obs, _info = env.reset(seed=int(seed))
            terminated = truncated = False
            while not (terminated or truncated):
                action, _state = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _info = env.step(int(action))
                total_reward += float(reward)
            report = env.episode_report()
            report["episode_reward"] = total_reward
            report["merge_success"] = _info.get("done_reason") == "merge_success"
            reports.append(report)
            rewards.append(total_reward)
        finally:
            env.close()
    metrics = aggregate_episode_reports(reports)
    metrics["average_reward"] = float(np.mean(rewards)) if rewards else 0.0
    metrics["merge_success_rate"] = float(np.mean([float(item.get("merge_success", False)) for item in reports])) if reports else 0.0
    return metrics


class _SafetyEvalCallback:
    def __init__(
        self,
        base_callback: Any,
        config: Any,
        output_path: Path,
    ) -> None:
        class SafetyEvalCallback(base_callback):
            def __init__(self, cfg: Any, model_output: Path) -> None:
                super().__init__()
                self.cfg = cfg
                self.model_output = model_output
                self.eval_freq = int(cfg.stage3.get("eval_freq", 10000))
                self.eval_seeds = [int(seed) for seed in cfg.stage3.get("eval_seeds", [])]
                self.checkpoint_dir = model_output.parent / "checkpoints"
                self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
                _final_path, self.best_path, _selection_report_path = _checkpoint_paths(model_output)
                self.records: list[dict[str, Any]] = []
                self.best_score: float | None = None
                self.best_record: dict[str, Any] | None = None
                self._last_eval_step = 0

            def _on_step(self) -> bool:
                if self.eval_freq <= 0 or not self.eval_seeds:
                    return True
                if int(self.num_timesteps) - self._last_eval_step >= self.eval_freq:
                    self._last_eval_step = int(self.num_timesteps)
                    self._run_eval("periodic")
                return True

            def _on_training_end(self) -> None:
                if self.eval_seeds:
                    self._run_eval("final")

            def _run_eval(self, kind: str) -> None:
                timesteps = int(self.num_timesteps)
                checkpoint_path = self.checkpoint_dir / f"{self.model_output.stem}_step_{timesteps:08d}.zip"
                self.model.save(str(checkpoint_path))
                metrics = _evaluate_model_for_safety(self.model, self.cfg, self.eval_seeds)
                score = _safety_score(metrics)
                selected_best = self.best_score is None or score > self.best_score
                record = {
                    "kind": kind,
                    "timesteps": timesteps,
                    "checkpoint_path": str(checkpoint_path),
                    "metrics": metrics,
                    "safety_score": score,
                    "selected_best": bool(selected_best),
                }
                if selected_best:
                    self.best_score = score
                    self.best_record = record
                    self.model.save(str(self.best_path))
                self.records.append(record)

        self.callback = SafetyEvalCallback(config, output_path)


def train_ppo(
    config: Any,
    env: SumoHighwayMergeEnv,
    output_path: str | Path,
    tensorboard_dir: str | Path | None = None,
) -> dict:
    PPO = _require_sb3()
    from stable_baselines3.common.callbacks import BaseCallback
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
        device=_training_device(config),
        verbose=1,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_path, best_path, selection_report_path = _checkpoint_paths(output_path)
    safety_callback = None
    callback = None
    if bool(config.stage3.get("eval_enabled", False)):
        safety_callback = _SafetyEvalCallback(BaseCallback, config, output_path).callback
        callback = safety_callback
    model.learn(
        total_timesteps=int(config.rl.total_timesteps),
        tb_log_name=str(config.stage3.get("tensorboard_log_name", "ppo")),
        callback=callback,
    )
    model.save(str(final_path))
    checkpoint_selection: dict[str, Any]
    if safety_callback is not None and getattr(safety_callback, "best_record", None) is not None and best_path.exists():
        shutil.copyfile(best_path, output_path)
        checkpoint_selection = {
            "enabled": True,
            "selected_model_path": str(output_path),
            "final_model_path": str(final_path),
            "best_safety_model_path": str(best_path),
            "best_record": safety_callback.best_record,
            "records": safety_callback.records,
            "selection_metric": (
                "average_reward + 2*min_distance_p1 + 2*ttc_p1 - 2*drac_p99_capped "
                "- 50*proxy_collision_rate - 30*safety_violation_rate"
            ),
        }
    else:
        shutil.copyfile(final_path, output_path)
        checkpoint_selection = {
            "enabled": bool(config.stage3.get("eval_enabled", False)),
            "selected_model_path": str(output_path),
            "final_model_path": str(final_path),
            "best_safety_model_path": str(best_path) if best_path.exists() else None,
            "best_record": None,
            "records": getattr(safety_callback, "records", []) if safety_callback is not None else [],
            "selection_metric": None,
        }
    write_json(selection_report_path, checkpoint_selection)
    report = {
        "model_path": str(output_path),
        "final_model_path": str(final_path),
        "best_safety_model_path": str(best_path) if best_path.exists() else None,
        "checkpoint_selection_report": str(selection_report_path),
        "checkpoint_selection": checkpoint_selection,
        "total_timesteps": int(config.rl.total_timesteps),
        "reward_profile": str(config.rl.get("reward_profile", "default")),
        "tensorboard": str(tensorboard_dir) if tensorboard_dir else None,
        "device": str(model.device),
    }
    return report


def load_ppo(path: str | Path, device: str = "auto"):
    PPO = _require_sb3()
    return PPO.load(str(path), device=device)
