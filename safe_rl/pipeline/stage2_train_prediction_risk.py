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


def _resolve_device(cfg: Any, torch: Any):
    training_cfg = cfg.get("training", {})
    requested = str(training_cfg.get("device", "auto")).strip().lower()
    if requested in ("auto", ""):
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "gpu":
        requested = "cuda"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("training.device requests CUDA, but torch.cuda.is_available() is false.")
    if requested == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        raise RuntimeError("training.device requests MPS, but PyTorch MPS is unavailable.")
    return torch.device(requested)


def _configure_torch_backend(cfg: Any, torch: Any, device: Any) -> None:
    training_cfg = cfg.get("training", {})
    if device.type == "cuda":
        allow_tf32 = bool(training_cfg.get("cuda_tf32", True))
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32
        if allow_tf32 and hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")


def _loader_kwargs(cfg: Any, device: Any) -> dict[str, Any]:
    training_cfg = cfg.get("training", {})
    num_workers = int(training_cfg.get("num_workers", 0))
    pin_memory = bool(training_cfg.get("pin_memory", True)) and device.type == "cuda"
    kwargs: dict[str, Any] = {"num_workers": num_workers, "pin_memory": pin_memory}
    if num_workers > 0:
        kwargs["persistent_workers"] = bool(training_cfg.get("persistent_workers", True))
    return kwargs


def _to_device(batch, device: Any, non_blocking: bool):
    return tuple(tensor.to(device, non_blocking=non_blocking) for tensor in batch)


def _cpu_state_dict(model: Any) -> dict[str, Any]:
    return {key: value.detach().cpu() for key, value in model.state_dict().items()}


def _stage1_path(cfg) -> Path:
    if cfg.stage2.input_stage1:
        return Path(cfg.stage2.input_stage1)
    return latest_stage_file(cfg, "stage1", str(cfg.stage1.output_name))


def _stage4_path(cfg) -> Path | None:
    input_stage4 = cfg.stage2.get("input_stage4")
    if not input_stage4:
        return None
    if str(input_stage4).lower() == "auto":
        return latest_stage_file(cfg, "stage4", "on_policy_failure_buffer.npz")
    return Path(input_stage4)


def _merge_risk_buffers(stage1_data: Any, stage4_data: Any | None) -> dict[str, np.ndarray] | Any:
    if stage4_data is None:
        return stage1_data
    risk_keys = ("risk_features", "actions", "overall_risk", "risk_types")
    merged = {key: np.concatenate([stage1_data[key], stage4_data[key]], axis=0) for key in risk_keys}
    return merged


def _train_risk_module(
    cfg: Any,
    data: np.lib.npyio.NpzFile,
    stage_dir: Path,
    tb: TensorboardLogger,
    device: Any,
) -> dict:
    torch, DataLoader, TensorDataset = _require_torch()
    from safe_rl.risk.risk_module import RiskModule, risk_loss

    x = torch.tensor(data["risk_features"], dtype=torch.float32)
    actions = torch.tensor(data["actions"], dtype=torch.long)
    y = torch.tensor(data["overall_risk"], dtype=torch.float32)
    risk_types = torch.tensor(data["risk_types"], dtype=torch.float32)
    dataset = TensorDataset(x, actions, y, risk_types)
    loader_kwargs = _loader_kwargs(cfg, device)
    loader = DataLoader(dataset, batch_size=int(cfg.risk_module.batch_size), shuffle=True, **loader_kwargs)
    model = RiskModule(
        explicit_dim=int(cfg.risk_module.explicit_feature_dim),
        latent_dim=int(cfg.risk_module.latent_dim),
        action_embedding_dim=int(cfg.risk_module.action_embedding_dim),
        hidden_dim=int(cfg.risk_module.hidden_dim),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.risk_module.learning_rate))
    weights = dict(cfg.risk_module.loss_weights)
    history: list[float] = []
    stage_log(
        "stage2",
        f"risk module samples={len(dataset)}, batch_size={cfg.risk_module.batch_size}, "
        f"pin_memory={loader_kwargs['pin_memory']}",
    )
    for epoch in progress_iter(range(int(cfg.risk_module.epochs)), desc="Stage2 risk epochs"):
        losses = []
        for batch in loader:
            batch_x, batch_actions, batch_y, batch_types = _to_device(
                batch,
                device,
                non_blocking=bool(loader_kwargs["pin_memory"]),
            )
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
    torch.save({"model_state_dict": _cpu_state_dict(model), "loss_history": history}, checkpoint)
    return {"risk_checkpoint": str(checkpoint), "risk_loss_history": history}


def _build_wcdt_batch(cfg: Any, data: np.lib.npyio.NpzFile, device: Any | None = None):
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
    loader_kwargs = _loader_kwargs(cfg, device or torch.device("cpu"))
    return DataLoader(dataset, batch_size=int(cfg.prediction.batch_size), shuffle=True, **loader_kwargs)


def _train_wcdt_predictor(
    cfg: Any,
    data: np.lib.npyio.NpzFile,
    stage_dir: Path,
    tb: TensorboardLogger,
    device: Any,
) -> dict:
    loader = _build_wcdt_batch(cfg, data, device=device)
    if loader is None:
        return {"prediction_skipped": True, "prediction_skip_reason": "no trajectory samples in Stage1 buffer"}
    torch, _DataLoader, _TensorDataset = _require_torch()
    from net_works import BackBone
    from utils import MathUtil

    betas = MathUtil.generate_linear_schedule(50, 1e-4, 0.008)
    model = BackBone(betas).to(device)
    if cfg.prediction.checkpoint:
        state = torch.load(cfg.prediction.checkpoint, map_location=device)
        model.load_state_dict(state.get("model_state_dict", state), strict=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.prediction.learning_rate))
    loss_history: list[float] = []
    pin_memory = bool(getattr(loader, "pin_memory", False))
    stage_log(
        "stage2",
        f"WcDT predictor batches={len(loader)}, batch_size={cfg.prediction.batch_size}, "
        f"pin_memory={pin_memory}",
    )
    for epoch in progress_iter(range(int(cfg.prediction.epochs)), desc="Stage2 prediction epochs"):
        losses = []
        for batch in loader:
            pred_his, pred_future, pred_mask, pred_feat, other_his, other_feat, other_mask, lane_list = _to_device(
                batch,
                device,
                non_blocking=pin_memory,
            )
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
                "traffic_light": torch.zeros(
                    (pred_his.shape[0], int(cfg.prediction.max_traffic_light), int(cfg.scenario.history_steps)),
                    device=device,
                ),
                "traffic_light_pos": torch.zeros(
                    (pred_his.shape[0], int(cfg.prediction.max_traffic_light), 2),
                    device=device,
                ),
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
    torch.save({"model_state_dict": _cpu_state_dict(model), "loss_history": loss_history}, checkpoint)
    return {"prediction_checkpoint": str(checkpoint), "prediction_loss_history": loss_history}


def run(cfg) -> Path:
    torch, _DataLoader, _TensorDataset = _require_torch()
    device = _resolve_device(cfg, torch)
    _configure_torch_backend(cfg, torch, device)
    stage_dir = prepare_run_dir(cfg, "stage2")
    input_path = _stage1_path(cfg)
    input_stage4_path = _stage4_path(cfg)
    stage_log("stage2", f"run_id={cfg.run.run_id}")
    stage_log("stage2", f"input_stage1={input_path}")
    if input_stage4_path is not None:
        stage_log("stage2", f"input_stage4={input_stage4_path}")
    stage_log("stage2", f"output_dir={stage_dir}")
    if device.type == "cuda":
        stage_log("stage2", f"device={device} ({torch.cuda.get_device_name(device)})")
    else:
        stage_log("stage2", f"device={device}")
    tb = TensorboardLogger(stage_dir / "tensorboard", enabled=bool(cfg.run.get("tensorboard", True)))
    data = np.load(input_path, allow_pickle=False)
    stage4_data = np.load(input_stage4_path, allow_pickle=False) if input_stage4_path is not None else None
    risk_data = _merge_risk_buffers(data, stage4_data)
    executed_count = int(data["executed_actions"].shape[0]) if "executed_actions" in data else int(data["actions"].shape[0])
    stage_log("stage2", f"executed_transition_count={executed_count}")
    stage_log("stage2", f"risk_transition_count={int(data['actions'].shape[0])}")
    if stage4_data is not None:
        stage4_executed = (
            int(stage4_data["executed_actions"].shape[0])
            if "executed_actions" in stage4_data
            else int(stage4_data["actions"].shape[0])
        )
        stage_log("stage2", f"stage4_executed_transition_count={stage4_executed}")
        stage_log("stage2", f"stage4_risk_transition_count={int(stage4_data['actions'].shape[0])}")
        stage_log("stage2", f"risk_transition_count={int(risk_data['actions'].shape[0])}")
    if "agent_history" in data:
        stage_log("stage2", f"trajectory_samples={int(data['agent_history'].shape[0])}")
    report = {
        "stage": "stage2",
        "run_id": cfg.run.run_id,
        "input_stage1": str(input_path),
        "input_stage4": str(input_stage4_path) if input_stage4_path is not None else None,
        "transition_count": executed_count,
        "stage1_risk_transition_count": int(data["actions"].shape[0]),
        "stage4_transition_count": (
            int(stage4_data["executed_actions"].shape[0])
            if stage4_data is not None and "executed_actions" in stage4_data
            else int(stage4_data["actions"].shape[0])
            if stage4_data is not None
            else 0
        ),
        "risk_transition_count": int(risk_data["actions"].shape[0]),
        "tensorboard": str(stage_dir / "tensorboard"),
        "device": str(device),
    }
    report.update(_train_risk_module(cfg, risk_data, stage_dir, tb, device))
    if bool(cfg.prediction.train_enabled):
        report.update(_train_wcdt_predictor(cfg, data, stage_dir, tb, device))
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
