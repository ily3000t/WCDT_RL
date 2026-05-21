from __future__ import annotations

from collections import Counter
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


def _prediction_loss_summary(history: list[float] | None) -> dict | None:
    if not history:
        return None
    return {
        "epochs": len(history),
        "first": float(history[0]),
        "last": float(history[-1]),
        "min": float(min(history)),
    }


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
    risk_keys = (
        "risk_features",
        "actions",
        "overall_risk",
        "risk_types",
        "lane_oob_risk",
        "candidate_legal",
        "traffic_risk",
        "risk_sample_weight",
        "candidate_transition_id",
        "candidate_raw_action",
    )
    stage1 = _risk_training_arrays(stage1_data)
    stage4 = _risk_training_arrays(stage4_data)
    if stage1["candidate_transition_id"].size and stage4["candidate_transition_id"].size:
        offset = int(np.max(stage1["candidate_transition_id"])) + 1
        stage4["candidate_transition_id"] = stage4["candidate_transition_id"] + offset
    merged = {key: np.concatenate([stage1[key], stage4[key]], axis=0) for key in risk_keys}
    return merged


def _has_key(data: Any, key: str) -> bool:
    return key in data.files if hasattr(data, "files") else key in data


def _risk_training_arrays(data: Any) -> dict[str, np.ndarray]:
    risk_features = np.asarray(data["risk_features"], dtype=np.float32)
    actions = np.asarray(data["actions"], dtype=np.int64)
    risk_types = np.asarray(data["risk_types"], dtype=np.float32)
    if _has_key(data, "traffic_risk"):
        traffic_risk = np.asarray(data["traffic_risk"], dtype=np.float32)
    elif risk_types.ndim == 2 and risk_types.shape[1] > 0:
        traffic_risk = np.max(risk_types, axis=1).astype(np.float32)
    else:
        traffic_risk = np.asarray(data["overall_risk"], dtype=np.float32)
    if _has_key(data, "lane_oob_risk"):
        lane_oob = np.asarray(data["lane_oob_risk"], dtype=np.float32)
    elif risk_features.ndim == 2 and risk_features.shape[1] > 5:
        lane_oob = (risk_features[:, 5] > 0.5).astype(np.float32)
    else:
        lane_oob = np.zeros_like(traffic_risk, dtype=np.float32)
    if _has_key(data, "candidate_legal"):
        candidate_legal = (np.asarray(data["candidate_legal"], dtype=np.float32) > 0.5).astype(np.float32)
    else:
        candidate_legal = (lane_oob <= 0.5).astype(np.float32)
    if _has_key(data, "risk_sample_weight"):
        sample_weight = np.asarray(data["risk_sample_weight"], dtype=np.float32)
    else:
        sample_weight = candidate_legal.astype(np.float32)
    if _has_key(data, "candidate_transition_id"):
        transition_id = np.asarray(data["candidate_transition_id"], dtype=np.int64)
        transition_id_source = "explicit"
    else:
        transition_id = (np.arange(actions.shape[0], dtype=np.int64) // 9).astype(np.int64)
        transition_id_source = "inferred_by_9_rows"
    if _has_key(data, "candidate_raw_action"):
        raw_actions = np.asarray(data["candidate_raw_action"], dtype=np.int64)
    elif _has_key(data, "executed_actions"):
        executed = np.asarray(data["executed_actions"], dtype=np.int64)
        raw_actions = np.full_like(actions, -1, dtype=np.int64)
        valid = (transition_id >= 0) & (transition_id < executed.shape[0])
        raw_actions[valid] = executed[transition_id[valid]]
    else:
        raw_actions = np.full_like(actions, -1, dtype=np.int64)
    return {
        "risk_features": risk_features,
        "actions": actions,
        "overall_risk": traffic_risk.astype(np.float32),
        "risk_types": risk_types,
        "lane_oob_risk": lane_oob.astype(np.float32),
        "candidate_legal": candidate_legal.astype(np.float32),
        "traffic_risk": traffic_risk.astype(np.float32),
        "risk_sample_weight": sample_weight.astype(np.float32),
        "candidate_transition_id": transition_id.astype(np.int64),
        "candidate_raw_action": raw_actions.astype(np.int64),
        "candidate_transition_id_source": np.asarray([transition_id_source]),
    }


def _configured_sample_weight(cfg: Any, arrays: dict[str, np.ndarray]) -> np.ndarray:
    legal = arrays["candidate_legal"] > 0.5
    if bool(cfg.risk_module.get("legal_candidates_only_for_training", True)):
        weights = np.where(
            legal,
            np.maximum(arrays["risk_sample_weight"], 1.0),
            float(cfg.risk_module.get("illegal_candidate_sample_weight", 0.0)),
        ).astype(np.float32)
    else:
        weights = np.ones_like(arrays["traffic_risk"], dtype=np.float32)
    weights = weights * np.where(
        arrays["traffic_risk"] > 0.5,
        float(cfg.risk_module.get("positive_traffic_risk_weight", 1.0)),
        1.0,
    ).astype(np.float32)
    if arrays["risk_types"].ndim == 2 and arrays["risk_types"].shape[1] > 4:
        weights = weights * np.where(
            arrays["risk_types"][:, 4] > 0.5,
            float(cfg.risk_module.get("merge_conflict_weight", 1.0)),
            1.0,
        ).astype(np.float32)
    return weights.astype(np.float32)


def _split_indices(count: int, val_split: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(count, dtype=np.int64)
    if count <= 1 or val_split <= 0.0:
        return indices, np.asarray([], dtype=np.int64)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    val_count = int(round(count * val_split))
    val_count = min(max(val_count, 1), count - 1)
    return indices[val_count:], indices[:val_count]


def _split_risk_indices(arrays: dict[str, np.ndarray], val_split: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    transition_ids = np.asarray(arrays.get("candidate_transition_id", []), dtype=np.int64)
    if transition_ids.shape[0] != arrays["traffic_risk"].shape[0] or transition_ids.size == 0:
        return _split_indices(int(arrays["traffic_risk"].shape[0]), val_split, seed)
    unique_ids = np.unique(transition_ids)
    if unique_ids.shape[0] <= 1 or val_split <= 0.0:
        return np.arange(transition_ids.shape[0], dtype=np.int64), np.asarray([], dtype=np.int64)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_ids)
    val_count = int(round(unique_ids.shape[0] * val_split))
    val_count = min(max(val_count, 1), unique_ids.shape[0] - 1)
    val_ids = set(int(item) for item in unique_ids[:val_count])
    val_mask = np.asarray([int(item) in val_ids for item in transition_ids], dtype=bool)
    return np.where(~val_mask)[0].astype(np.int64), np.where(val_mask)[0].astype(np.int64)


def _risk_data_summary(arrays: dict[str, np.ndarray], sample_weight: np.ndarray) -> dict:
    actions = arrays["actions"]
    legal = arrays["candidate_legal"] > 0.5
    risk = arrays["traffic_risk"]
    lane_oob = arrays["lane_oob_risk"]
    weighted = sample_weight > 0.0
    return {
        "sample_count": int(risk.shape[0]),
        "traffic_risk_rate": float(np.mean(risk)) if risk.size else 0.0,
        "lane_oob_risk_rate": float(np.mean(lane_oob)) if lane_oob.size else 0.0,
        "illegal_candidate_rate": float(np.mean(~legal)) if legal.size else 0.0,
        "legal_candidate_risk_rate": float(np.mean(risk[legal])) if np.any(legal) else 0.0,
        "weighted_sample_rate": float(np.mean(weighted)) if weighted.size else 0.0,
        "weighted_positive_risk_rate": float(np.mean(risk[weighted])) if np.any(weighted) else 0.0,
        "traffic_risk_by_action": {
            str(index): float(np.mean(risk[actions == index])) if np.any(actions == index) else 0.0
            for index in range(9)
        },
        "legal_candidate_action_risk_rate": {
            str(index): (
                float(np.mean(risk[(actions == index) & legal])) if np.any((actions == index) & legal) else 0.0
            )
            for index in range(9)
        },
    }


def _risk_ranking_summary(arrays: dict[str, np.ndarray], indices: np.ndarray, predictions: np.ndarray) -> dict:
    if indices.size == 0 or predictions.size == 0:
        return {"available": False, "reason": "no validation samples"}
    actions = arrays["actions"][indices]
    risk = arrays["traffic_risk"][indices]
    legal = arrays["candidate_legal"][indices] > 0.5
    transition_ids = arrays["candidate_transition_id"][indices]
    raw_actions = arrays["candidate_raw_action"][indices]
    grouped: dict[int, list[int]] = {}
    for local_idx, transition_id in enumerate(transition_ids.tolist()):
        grouped.setdefault(int(transition_id), []).append(local_idx)

    top1_matches = []
    top3_matches = []
    raw_ranks = []
    oracle_best_risks = []
    model_best_label_risks = []
    model_minus_oracle = []
    oracle_hist: Counter[str] = Counter()
    model_hist: Counter[str] = Counter()
    skipped_incomplete = 0
    skipped_no_legal = 0
    skipped_raw_missing = 0

    for positions in grouped.values():
        group_actions = actions[positions]
        if set(int(item) for item in group_actions.tolist()) != set(range(9)):
            skipped_incomplete += 1
            continue
        legal_positions = [pos for pos in positions if bool(legal[pos])]
        if not legal_positions:
            skipped_no_legal += 1
            continue
        pred_order = sorted(legal_positions, key=lambda pos: (float(predictions[pos]), int(actions[pos])))
        label_order = sorted(legal_positions, key=lambda pos: (float(risk[pos]), int(actions[pos])))
        oracle_pos = label_order[0]
        model_pos = pred_order[0]
        oracle_best = float(risk[oracle_pos])
        model_label = float(risk[model_pos])
        top3 = pred_order[: min(3, len(pred_order))]
        top1_matches.append(float(model_label <= oracle_best + 1.0e-6))
        top3_matches.append(float(any(float(risk[pos]) <= oracle_best + 1.0e-6 for pos in top3)))
        oracle_best_risks.append(oracle_best)
        model_best_label_risks.append(model_label)
        model_minus_oracle.append(model_label - oracle_best)
        oracle_hist[str(int(actions[oracle_pos]))] += 1
        model_hist[str(int(actions[model_pos]))] += 1

        raw_action = int(raw_actions[positions[0]])
        rank = next((rank_idx + 1 for rank_idx, pos in enumerate(pred_order) if int(actions[pos]) == raw_action), None)
        if rank is None:
            skipped_raw_missing += 1
        else:
            raw_ranks.append(float(rank))

    evaluated = len(top1_matches)
    return {
        "available": evaluated > 0,
        "transition_group_count": int(len(grouped)),
        "evaluated_group_count": int(evaluated),
        "skipped_incomplete_group_count": int(skipped_incomplete),
        "skipped_no_legal_group_count": int(skipped_no_legal),
        "skipped_raw_missing_group_count": int(skipped_raw_missing),
        "top1_match_rate": float(np.mean(top1_matches)) if top1_matches else 0.0,
        "top3_match_rate": float(np.mean(top3_matches)) if top3_matches else 0.0,
        "raw_action_rank_mean": float(np.mean(raw_ranks)) if raw_ranks else 0.0,
        "oracle_best_action_histogram": dict(oracle_hist),
        "model_best_action_histogram": dict(model_hist),
        "mean_oracle_best_risk": float(np.mean(oracle_best_risks)) if oracle_best_risks else 0.0,
        "mean_model_best_label_risk": float(np.mean(model_best_label_risks)) if model_best_label_risks else 0.0,
        "mean_model_best_minus_oracle_risk": float(np.mean(model_minus_oracle)) if model_minus_oracle else 0.0,
    }


def _risk_validation_summary(pred: np.ndarray, target: np.ndarray, sample_weight: np.ndarray, legal: np.ndarray) -> dict:
    if pred.size == 0:
        return {"sample_count": 0}
    active = sample_weight > 0.0
    legal_mask = legal > 0.5
    legal_active = active & legal_mask
    pred_label = pred >= 0.5
    target_label = target >= 0.5
    weighted_abs = np.abs(pred - target)
    return {
        "sample_count": int(pred.shape[0]),
        "active_sample_count": int(np.sum(active)),
        "positive_risk_rate": float(np.mean(target[active])) if np.any(active) else 0.0,
        "predicted_positive_rate": float(np.mean(pred_label[active])) if np.any(active) else 0.0,
        "accuracy": float(np.mean(pred_label[active] == target_label[active])) if np.any(active) else 0.0,
        "legal_candidate_accuracy": (
            float(np.mean(pred_label[legal_active] == target_label[legal_active])) if np.any(legal_active) else 0.0
        ),
        "mean_abs_calibration_error": float(np.mean(weighted_abs[active])) if np.any(active) else 0.0,
    }


def _train_risk_module(
    cfg: Any,
    data: np.lib.npyio.NpzFile,
    stage_dir: Path,
    tb: TensorboardLogger,
    device: Any,
) -> dict:
    torch, DataLoader, TensorDataset = _require_torch()
    from safe_rl.risk.risk_module import RiskModule, risk_loss

    arrays = _risk_training_arrays(data)
    sample_weight = _configured_sample_weight(cfg, arrays)
    train_indices, val_indices = _split_risk_indices(
        arrays,
        float(cfg.risk_module.get("validation_split", 0.0)),
        int(cfg.run.seed),
    )

    def _dataset(indices: np.ndarray):
        return TensorDataset(
            torch.tensor(arrays["risk_features"][indices], dtype=torch.float32),
            torch.tensor(arrays["actions"][indices], dtype=torch.long),
            torch.tensor(arrays["traffic_risk"][indices], dtype=torch.float32),
            torch.tensor(arrays["risk_types"][indices], dtype=torch.float32),
            torch.tensor(sample_weight[indices], dtype=torch.float32),
            torch.tensor(arrays["candidate_legal"][indices], dtype=torch.float32),
        )

    train_dataset = _dataset(train_indices)
    val_dataset = _dataset(val_indices) if val_indices.size else None
    loader_kwargs = _loader_kwargs(cfg, device)
    loader = DataLoader(train_dataset, batch_size=int(cfg.risk_module.batch_size), shuffle=True, **loader_kwargs)
    val_loader = (
        DataLoader(val_dataset, batch_size=int(cfg.risk_module.batch_size), shuffle=False, **loader_kwargs)
        if val_dataset is not None
        else None
    )
    model = RiskModule(
        explicit_dim=int(cfg.risk_module.explicit_feature_dim),
        latent_dim=int(cfg.risk_module.latent_dim),
        action_embedding_dim=int(cfg.risk_module.action_embedding_dim),
        hidden_dim=int(cfg.risk_module.hidden_dim),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.risk_module.learning_rate))
    weights = dict(cfg.risk_module.loss_weights)
    history: list[float] = []
    val_history: list[float] = []
    stage_log(
        "stage2",
        f"risk module samples={len(train_dataset)}, val_samples={len(val_dataset) if val_dataset is not None else 0}, "
        f"batch_size={cfg.risk_module.batch_size}, "
        f"pin_memory={loader_kwargs['pin_memory']}",
    )
    for epoch in progress_iter(range(int(cfg.risk_module.epochs)), desc="Stage2 risk epochs"):
        losses = []
        model.train()
        for batch in loader:
            batch_x, batch_actions, batch_y, batch_types, batch_weights, _batch_legal = _to_device(
                batch,
                device,
                non_blocking=bool(loader_kwargs["pin_memory"]),
            )
            output = model(batch_x, batch_actions)
            loss = risk_loss(
                output,
                {"risk_score": batch_y, "risk_types": batch_types, "sample_weight": batch_weights},
                {"risk": weights.get("risk", 1.0), "calibration": weights.get("calibration", 0.1)},
            )
            if bool(cfg.risk_module.ranking_loss_enabled) and batch_y.numel() > 1:
                active = batch_weights > 0.0
                scores = output["risk_score"][active]
                active_y = batch_y[active]
                pos = active_y.view(-1, 1)
                label_diff = pos - pos.t()
                score_diff = scores.view(-1, 1) - scores.view(1, -1)
                mask = label_diff > 0
                if scores.numel() > 1 and torch.any(mask):
                    rank_loss = torch.relu(0.05 - score_diff[mask]).mean()
                    loss = loss + weights.get("ranking", 0.5) * rank_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        epoch_loss = float(np.mean(losses)) if losses else 0.0
        history.append(epoch_loss)
        val_loss = 0.0
        if val_loader is not None:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for batch in val_loader:
                    batch_x, batch_actions, batch_y, batch_types, batch_weights, _batch_legal = _to_device(
                        batch,
                        device,
                        non_blocking=bool(loader_kwargs["pin_memory"]),
                    )
                    output = model(batch_x, batch_actions)
                    loss = risk_loss(
                        output,
                        {"risk_score": batch_y, "risk_types": batch_types, "sample_weight": batch_weights},
                        {"risk": weights.get("risk", 1.0), "calibration": weights.get("calibration", 0.1)},
                    )
                    val_losses.append(float(loss.detach().cpu()))
            val_loss = float(np.mean(val_losses)) if val_losses else 0.0
            val_history.append(val_loss)
        tb.scalar("stage2/risk_loss", epoch_loss, epoch)
        if val_loader is not None:
            tb.scalar("stage2/risk_val_loss", val_loss, epoch)
            stage_log(
                "stage2",
                f"risk epoch={epoch + 1}/{cfg.risk_module.epochs} loss={epoch_loss:.6f} val_loss={val_loss:.6f}",
            )
        else:
            stage_log("stage2", f"risk epoch={epoch + 1}/{cfg.risk_module.epochs} loss={epoch_loss:.6f}")
    checkpoint = stage_dir / "risk_module.pt"
    validation_summary = {"sample_count": 0}
    ranking_summary = {"available": False, "reason": "no validation samples"}
    if val_indices.size:
        model.eval()
        val_x = torch.tensor(arrays["risk_features"][val_indices], dtype=torch.float32, device=device)
        val_actions = torch.tensor(arrays["actions"][val_indices], dtype=torch.long, device=device)
        with torch.no_grad():
            val_pred = model(val_x, val_actions)["risk_score"].detach().cpu().numpy()
        validation_summary = _risk_validation_summary(
            val_pred,
            arrays["traffic_risk"][val_indices],
            sample_weight[val_indices],
            arrays["candidate_legal"][val_indices],
        )
        ranking_summary = _risk_ranking_summary(arrays, val_indices, val_pred)
    training_summary = {
        "data": _risk_data_summary(arrays, sample_weight),
        "train_sample_count": int(train_indices.shape[0]),
        "validation_sample_count": int(val_indices.shape[0]),
        "validation": validation_summary,
        "ranking": ranking_summary,
        "config": {
            "legal_candidates_only_for_training": bool(cfg.risk_module.get("legal_candidates_only_for_training", True)),
            "illegal_candidate_sample_weight": float(cfg.risk_module.get("illegal_candidate_sample_weight", 0.0)),
            "positive_traffic_risk_weight": float(cfg.risk_module.get("positive_traffic_risk_weight", 1.0)),
            "merge_conflict_weight": float(cfg.risk_module.get("merge_conflict_weight", 1.0)),
            "validation_split": float(cfg.risk_module.get("validation_split", 0.0)),
        },
    }
    torch.save(
        {
            "model_state_dict": _cpu_state_dict(model),
            "loss_history": history,
            "val_loss_history": val_history,
            "training_summary": training_summary,
        },
        checkpoint,
    )
    return {
        "risk_checkpoint": str(checkpoint),
        "risk_loss_history": history,
        "risk_val_loss_history": val_history,
        "risk_training_summary": training_summary,
        "risk_ranking_summary": ranking_summary,
    }


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
    initial_prediction_report_path = stage_dir / "stage2_initial_prediction_report.json"
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
        if report.get("prediction_checkpoint"):
            initial_prediction_report = {
                "stage": "stage2_initial_prediction",
                "run_id": cfg.run.run_id,
                "input_stage1": str(input_path),
                "prediction_checkpoint": report.get("prediction_checkpoint"),
                "prediction_loss_history": report.get("prediction_loss_history", []),
                "prediction_loss_summary": _prediction_loss_summary(report.get("prediction_loss_history", [])),
                "device": str(device),
            }
            write_report(initial_prediction_report_path, initial_prediction_report)
            report["initial_prediction_report"] = str(initial_prediction_report_path)
    elif initial_prediction_report_path.exists():
        report["initial_prediction_report"] = str(initial_prediction_report_path)
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
