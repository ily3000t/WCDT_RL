from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from safe_rl.pipeline.common import latest_stage_file, load_stage_config, parse_config_arg, write_report
from safe_rl.utils.config import prepare_run_dir
from safe_rl.utils.progress import TensorboardLogger, progress_iter, stage_log


def _require_torch():
    try:
        import torch
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Stage2 requires torch. Activate the SAFE_RL training environment.") from exc
    return torch, DataLoader, TensorDataset


def _stage1_path(cfg) -> Path:
    if cfg.stage2.input_stage1:
        return Path(cfg.stage2.input_stage1)
    return latest_stage_file(cfg, "stage1", str(cfg.stage1.output_name))


def _train_risk_module(cfg: Any, data: np.lib.npyio.NpzFile, stage_dir: Path, tb: TensorboardLogger) -> dict:
    torch, DataLoader, TensorDataset = _require_torch()
    from safe_rl.risk.risk_module import RiskModule, risk_loss

    x = torch.tensor(data["risk_features"], dtype=torch.float32)
    actions = torch.tensor(data["actions"], dtype=torch.long)
    y = torch.tensor(data["overall_risk"], dtype=torch.float32)
    risk_types = torch.tensor(data["risk_types"], dtype=torch.float32)
    dataset = TensorDataset(x, actions, y, risk_types)
    loader = DataLoader(dataset, batch_size=int(cfg.risk_module.batch_size), shuffle=True)
    model = RiskModule(
        explicit_dim=int(cfg.risk_module.explicit_feature_dim),
        latent_dim=int(cfg.risk_module.latent_dim),
        action_embedding_dim=int(cfg.risk_module.action_embedding_dim),
        hidden_dim=int(cfg.risk_module.hidden_dim),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.risk_module.learning_rate))
    weights = dict(cfg.risk_module.loss_weights)
    history: list[float] = []
    stage_log("stage2", f"risk module samples={len(dataset)}, batch_size={cfg.risk_module.batch_size}")
    for epoch in progress_iter(range(int(cfg.risk_module.epochs)), desc="Stage2 risk epochs"):
        losses = []
        for batch_x, batch_actions, batch_y, batch_types in loader:
            output = model(batch_x, batch_actions)
            loss = risk_loss(
                output,
                {"risk_score": batch_y, "risk_types": batch_types},
                {"risk": weights.get("risk", 1.0), "calibration": weights.get("calibration", 0.1)},
            )
            if bool(cfg.risk_module.ranking_loss_enabled) and batch_y.numel() > 1:
                scores = output["risk_score"]
                pos = batch_y.view(-1, 1)
                label_diff = pos - pos.t()
                score_diff = scores.view(-1, 1) - scores.view(1, -1)
                mask = label_diff > 0
                if torch.any(mask):
                    rank_loss = torch.relu(0.05 - score_diff[mask]).mean()
                    loss = loss + weights.get("ranking", 0.5) * rank_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        epoch_loss = float(np.mean(losses)) if losses else 0.0
        history.append(epoch_loss)
        tb.scalar("stage2/risk_loss", epoch_loss, epoch)
        stage_log("stage2", f"risk epoch={epoch + 1}/{cfg.risk_module.epochs} loss={epoch_loss:.6f}")
    checkpoint = stage_dir / "risk_module.pt"
    torch.save({"model_state_dict": model.state_dict(), "loss_history": history}, checkpoint)
    return {"risk_checkpoint": str(checkpoint), "risk_loss_history": history}


def _build_wcdt_batch(cfg: Any, data: np.lib.npyio.NpzFile):
    torch, DataLoader, TensorDataset = _require_torch()
    history = data["agent_history"]
    future = data["agent_future"]
    mask = data["agent_mask"]
    if history.shape[0] == 0 or history.ndim != 4:
        return None
    max_pred = int(cfg.prediction.max_pred_num)
    max_other = int(cfg.prediction.max_other_num)
    hist_steps = int(cfg.scenario.history_steps)
    horizon = future.shape[2]
    padded_future = np.zeros((future.shape[0], max_pred, 80, 5), dtype=np.float32)
    predicted_his = np.zeros((future.shape[0], max_pred, hist_steps, 5), dtype=np.float32)
    other_his = np.zeros((future.shape[0], max_other, hist_steps, 5), dtype=np.float32)
    predicted_mask = np.zeros((future.shape[0], max_pred), dtype=np.float32)
    other_mask = np.zeros((future.shape[0], max_other), dtype=np.float32)

    for sample_idx in range(history.shape[0]):
        pred_agents = list(range(1, min(history.shape[1], max_pred + 1)))
        other_agents = [0] + list(range(max_pred + 1, min(history.shape[1], max_pred + max_other)))
        for row, agent_idx in enumerate(pred_agents[:max_pred]):
            predicted_his[sample_idx, row] = history[sample_idx, agent_idx]
            padded_future[sample_idx, row, :horizon] = future[sample_idx, agent_idx]
            if horizon < 80:
                padded_future[sample_idx, row, horizon:] = future[sample_idx, agent_idx, -1]
            predicted_mask[sample_idx, row] = mask[sample_idx, agent_idx]
        for row, agent_idx in enumerate(other_agents[:max_other]):
            other_his[sample_idx, row] = history[sample_idx, agent_idx]
            other_mask[sample_idx, row] = mask[sample_idx, agent_idx]

    predicted_feature = np.zeros((history.shape[0], max_pred, 7), dtype=np.float32)
    other_feature = np.zeros((history.shape[0], max_other, 7), dtype=np.float32)
    predicted_feature[..., 0] = 1.8
    predicted_feature[..., 1] = 4.8
    predicted_feature[..., 3] = 1.0
    other_feature[..., 0] = 1.8
    other_feature[..., 1] = 4.8
    other_feature[..., 3] = 1.0

    from safe_rl.prediction.sumo_wcdt_adapter import SumoWcDTAdapter

    lane_list = SumoWcDTAdapter(cfg).lane_list
    lane_batch = np.repeat(lane_list[None, ...], history.shape[0], axis=0)
    dataset = TensorDataset(
        torch.tensor(predicted_his, dtype=torch.float32),
        torch.tensor(padded_future, dtype=torch.float32),
        torch.tensor(predicted_mask, dtype=torch.float32),
        torch.tensor(predicted_feature, dtype=torch.float32),
        torch.tensor(other_his, dtype=torch.float32),
        torch.tensor(other_feature, dtype=torch.float32),
        torch.tensor(other_mask, dtype=torch.float32),
        torch.tensor(lane_batch, dtype=torch.float32),
    )
    return DataLoader(dataset, batch_size=int(cfg.prediction.batch_size), shuffle=True)


def _train_wcdt_predictor(cfg: Any, data: np.lib.npyio.NpzFile, stage_dir: Path, tb: TensorboardLogger) -> dict:
    loader = _build_wcdt_batch(cfg, data)
    if loader is None:
        return {"prediction_skipped": True, "prediction_skip_reason": "no trajectory samples in Stage1 buffer"}
    torch, _DataLoader, _TensorDataset = _require_torch()
    from net_works import BackBone
    from utils import MathUtil

    betas = MathUtil.generate_linear_schedule(50, 1e-4, 0.008)
    model = BackBone(betas)
    if cfg.prediction.checkpoint:
        state = torch.load(cfg.prediction.checkpoint, map_location="cpu")
        model.load_state_dict(state.get("model_state_dict", state), strict=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.prediction.learning_rate))
    loss_history: list[float] = []
    stage_log("stage2", f"WcDT predictor batches={len(loader)}, batch_size={cfg.prediction.batch_size}")
    for epoch in progress_iter(range(int(cfg.prediction.epochs)), desc="Stage2 prediction epochs"):
        losses = []
        for pred_his, pred_future, pred_mask, pred_feat, other_his, other_feat, other_mask, lane_list in loader:
            data_dict = {
                "predicted_feature": pred_feat,
                "other_his_pos": other_his[:, :, -1, :2],
                "other_his_traj_delt": other_his[:, :, 1:] - other_his[:, :, :-1],
                "other_feature": other_feat,
                "other_traj_mask": other_mask,
                "predicted_his_pos": pred_his[:, :, -1, :2],
                "predicted_his_traj_delt": pred_his[:, :, 1:] - pred_his[:, :, :-1],
                "predicted_his_traj": pred_his,
                "predicted_future_traj": pred_future,
                "predicted_traj_mask": pred_mask,
                "traffic_light": torch.zeros((pred_his.shape[0], int(cfg.prediction.max_traffic_light), int(cfg.scenario.history_steps))),
                "traffic_light_pos": torch.zeros((pred_his.shape[0], int(cfg.prediction.max_traffic_light), 2)),
                "lane_list": lane_list,
            }
            diffusion_loss, traj_loss, confidence_loss, _min_loss_traj = model(data_dict)
            loss = diffusion_loss.mean() + traj_loss.mean() + confidence_loss.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        epoch_loss = float(np.mean(losses)) if losses else 0.0
        loss_history.append(epoch_loss)
        tb.scalar("stage2/prediction_loss", epoch_loss, epoch)
        stage_log("stage2", f"prediction epoch={epoch + 1}/{cfg.prediction.epochs} loss={epoch_loss:.6f}")
    checkpoint = stage_dir / "wcdt_predictor.pt"
    torch.save({"model_state_dict": model.state_dict(), "loss_history": loss_history}, checkpoint)
    return {"prediction_checkpoint": str(checkpoint), "prediction_loss_history": loss_history}


def run(cfg) -> Path:
    stage_dir = prepare_run_dir(cfg, "stage2")
    input_path = _stage1_path(cfg)
    stage_log("stage2", f"run_id={cfg.run.run_id}")
    stage_log("stage2", f"input_stage1={input_path}")
    stage_log("stage2", f"output_dir={stage_dir}")
    tb = TensorboardLogger(stage_dir / "tensorboard", enabled=bool(cfg.run.get("tensorboard", True)))
    data = np.load(input_path, allow_pickle=False)
    stage_log("stage2", f"transition_count={int(data['actions'].shape[0])}")
    if "agent_history" in data:
        stage_log("stage2", f"trajectory_samples={int(data['agent_history'].shape[0])}")
    report = {
        "stage": "stage2",
        "run_id": cfg.run.run_id,
        "input_stage1": str(input_path),
        "transition_count": int(data["actions"].shape[0]),
        "tensorboard": str(stage_dir / "tensorboard"),
    }
    report.update(_train_risk_module(cfg, data, stage_dir, tb))
    if bool(cfg.prediction.train_enabled):
        report.update(_train_wcdt_predictor(cfg, data, stage_dir, tb))
    write_report(stage_dir / "stage2_training_report.json", report)
    tb.close()
    stage_log("stage2", f"report={stage_dir / 'stage2_training_report.json'}")
    return stage_dir


def main() -> None:
    args = parse_config_arg("Stage2 WcDT-style prediction + risk module training")
    cfg = load_stage_config(args)
    run(cfg)


if __name__ == "__main__":
    main()
