from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from safe_rl.accvp.calibration import CalibrationBundle, OneSidedBinnedCalibrator
from safe_rl.accvp.dataset import ACCVPBranchDataset, build_split_manifest, collate_numpy
from safe_rl.accvp.model import (
    ACCVPPredictor,
    accvp_loss,
    checkpoint_metadata,
    model_kwargs_from_config,
    set_scene_encoder_trainable,
    warm_start_scene_encoder,
)
from safe_rl.accvp.schema import file_sha256, read_json, stable_hash, write_json_atomic
from safe_rl.accvp.oracle import validate_oracle_for_training
from safe_rl.utils.config import prepare_run_dir


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError("ACCVP training requires torch.") from exc
    return torch


def _tensor_batch(batch: dict[str, np.ndarray], torch: Any) -> dict[str, Any]:
    integer = {"history_lane_ids", "history_edge_role_ids", "role_ids", "lane_ids", "edge_role_ids", "candidate_action_ids"}
    return {
        key: torch.as_tensor(value, dtype=torch.long if key in integer else torch.float32)
        for key, value in batch.items()
    }


def _model_output(model: Any, batch: dict[str, Any]) -> dict[str, Any]:
    return model(
        batch["history_features"],
        batch["history_valid_mask"],
        batch["history_lane_ids"],
        batch["history_edge_role_ids"],
        batch["role_ids"],
        batch["lane_ids"],
        batch["edge_role_ids"],
        batch["actor_mask"],
        batch["candidate_plan"],
        batch["candidate_action_ids"],
    )


def _batches(dataset: ACCVPBranchDataset, indices: list[int], batch_size: int):
    for start in range(0, len(indices), max(1, batch_size)):
        yield collate_numpy(dataset[index] for index in indices[start : start + max(1, batch_size)])


def _root_bootstrap_indices(dataset: ACCVPBranchDataset, rng: np.random.Generator) -> list[int]:
    by_root: dict[str, list[int]] = {}
    for index, row in enumerate(dataset.rows):
        by_root.setdefault(str(row["root_id"]), []).append(index)
    roots = list(by_root)
    sampled = rng.choice(roots, size=len(roots), replace=True)
    indices = [index for root in sampled for index in by_root[str(root)]]
    rng.shuffle(indices)
    return indices


def _evaluate_loss(model: Any, dataset: ACCVPBranchDataset, torch: Any, weights: dict[str, float]) -> float:
    if not len(dataset):
        return float("inf")
    model.eval()
    losses = []
    with torch.no_grad():
        for batch_np in _batches(dataset, list(range(len(dataset))), 64):
            batch = _tensor_batch(batch_np, torch)
            loss, _parts = accvp_loss(_model_output(model, batch), batch, weights)
            losses.append(float(loss.item()))
    return float(np.mean(losses)) if losses else float("inf")


def _event_positive_weights(dataset: ACCVPBranchDataset) -> list[float]:
    positives = np.zeros((4,), dtype=np.float64)
    counts = np.zeros((4,), dtype=np.float64)
    for index in range(len(dataset)):
        item = dataset[index]
        mask = item["event_mask"] > 0.0
        positives[mask] += item["event_targets"][mask]
        counts[mask] += 1.0
    negatives = np.maximum(counts - positives, 0.0)
    return np.clip(negatives / np.maximum(positives, 1.0), 1.0, 50.0).tolist()


def _calibrate(models: list[Any], dataset: ACCVPBranchDataset, torch: Any, calibration_config: Any) -> CalibrationBundle:
    if not len(dataset):
        raise ValueError("ACCVP calibration split is empty")
    proxy_scores: list[float] = []
    proxy_labels: list[float] = []
    safety_scores: list[float] = []
    safety_labels: list[float] = []
    viability_scores: list[float] = []
    viability_labels: list[float] = []
    for model in models:
        model.eval()
    with torch.no_grad():
        for batch_np in _batches(dataset, list(range(len(dataset))), 64):
            batch = _tensor_batch(batch_np, torch)
            events = []
            for model in models:
                events.append(torch.sigmoid(_model_output(model, batch)["event_logits"]).cpu().numpy())
            stacked = np.stack(events, axis=0)
            proxy_scores.extend(stacked[:, :, 0].max(axis=0).tolist())
            safety_scores.extend(stacked[:, :, 1].max(axis=0).tolist())
            viability_prediction = stacked[:, :, 3].min(axis=0)
            proxy_labels.extend(batch_np["event_targets"][:, 0].tolist())
            safety_labels.extend(batch_np["event_targets"][:, 1].tolist())
            observed = batch_np["event_mask"][:, 3] > 0.0
            viability_labels.extend(batch_np["event_targets"][observed, 3].tolist())
            viability_scores.extend(viability_prediction[observed].tolist())
    fit_kwargs = {
        "bins": int(calibration_config.get("bins", 20)),
        "nominal_alpha": float(calibration_config.get("nominal_alpha", 0.05)),
        "bonferroni_family_size": int(calibration_config.get("bonferroni_signal_count", 3)),
    }
    return CalibrationBundle(
        proxy_collision=OneSidedBinnedCalibrator.fit(proxy_scores, proxy_labels, **fit_kwargs),
        safety_violation=OneSidedBinnedCalibrator.fit(safety_scores, safety_labels, **fit_kwargs),
        merge_viability=OneSidedBinnedCalibrator.fit(viability_scores, viability_labels, **fit_kwargs),
        provenance={
            "split": "calibration",
            "candidate_level_only": True,
            "proxy_count": len(proxy_labels),
            "safety_count": len(safety_labels),
            "eligible_viability_count": len(viability_labels),
            **fit_kwargs,
        },
    )


def train_accvp(config: Any, dataset_dir: str | Path) -> Path:
    torch = _torch()
    dataset_dir = Path(dataset_dir)
    oracle_report = validate_oracle_for_training(config, dataset_dir)
    split_path = dataset_dir / "manifests" / "split_manifest.jsonl"
    if not split_path.exists():
        build_split_manifest(dataset_dir, seed=int(config.run.seed), require_all_splits=True)
    training = config.accvp.training
    train_set = ACCVPBranchDataset(dataset_dir, "train")
    validation_set = ACCVPBranchDataset(dataset_dir, "validation")
    calibration_set = ACCVPBranchDataset(dataset_dir, "calibration")
    operating_set = ACCVPBranchDataset(dataset_dir, "operating_point")
    test_set = ACCVPBranchDataset(dataset_dir, "test")
    required_splits = {
        "train": train_set,
        "validation": validation_set,
        "calibration": calibration_set,
        "operating_point": operating_set,
        "test": test_set,
    }
    empty = [name for name, split in required_splits.items() if not len(split)]
    if empty:
        raise ValueError(f"ACCVP formal training requires non-empty grouped splits; empty={empty}")
    loss_weights = dict(training.loss_weights)
    loss_weights["event_positive_weights"] = _event_positive_weights(train_set)
    output_dir = prepare_run_dir(config, "accvp")
    warm = config.accvp.warm_start
    warm_source = Path(str(warm.checkpoint)) if warm.get("checkpoint") else None
    source_payload = None
    if bool(warm.enabled):
        if warm_source is None or not warm_source.exists():
            raise FileNotFoundError("accvp.warm_start.enabled requires an existing WcDT v3 checkpoint")
        source_payload = torch.load(warm_source, map_location="cpu")
        if not source_payload.get("model_state_dicts"):
            raise ValueError("WcDT v3 warm-start checkpoint has no model_state_dicts")
    models: list[Any] = []
    warm_records: list[dict[str, Any]] = []
    best_losses: list[float] = []
    for member in range(int(config.accvp.ensemble_size)):
        rng = np.random.default_rng(int(config.run.seed) + int(training.ensemble_seed_offset) * member)
        model = ACCVPPredictor(**model_kwargs_from_config(config))
        warm_record: dict[str, Any] = {"enabled": bool(warm.enabled), "member": member}
        if source_payload is not None:
            states = source_payload["model_state_dicts"]
            warm_record.update(warm_start_scene_encoder(model, states[member % len(states)]))
            warm_record["source_checkpoint"] = str(warm_source.resolve())
            warm_record["source_sha256"] = file_sha256(warm_source)
        encoder_parameters = list(model.scene.parameters())
        head_parameters = [parameter for name, parameter in model.named_parameters() if not name.startswith("scene.")]
        optimizer = torch.optim.AdamW(
            [
                {"params": encoder_parameters, "lr": float(training.learning_rate) * float(warm.encoder_lr_multiplier)},
                {"params": head_parameters, "lr": float(training.learning_rate)},
            ],
            weight_decay=float(training.weight_decay),
        )
        best_state = None
        best_loss = float("inf")
        for epoch in range(int(training.epochs)):
            set_scene_encoder_trainable(model, epoch >= int(warm.freeze_encoder_epochs))
            model.train()
            indices = _root_bootstrap_indices(train_set, rng)
            for batch_np in _batches(train_set, indices, int(training.batch_size)):
                batch = _tensor_batch(batch_np, torch)
                optimizer.zero_grad(set_to_none=True)
                loss, _parts = accvp_loss(_model_output(model, batch), batch, loss_weights)
                loss.backward()
                optimizer.step()
            validation_loss = _evaluate_loss(model, validation_set, torch, loss_weights)
            if validation_loss < best_loss:
                best_loss = validation_loss
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if best_state is not None:
            model.load_state_dict(best_state)
        models.append(model)
        best_losses.append(best_loss)
        warm_record["freeze_encoder_epochs"] = int(warm.freeze_encoder_epochs)
        warm_records.append(warm_record)
    calibration = _calibrate(models, calibration_set, torch, config.accvp.calibration)
    from safe_rl.accvp.tuning import tune_operating_point
    from safe_rl.accvp.diagnostics import final_test_diagnostics

    operating_point = tune_operating_point(models, operating_set, calibration, torch, config.accvp.tuning)
    final_test = final_test_diagnostics(models, test_set, calibration, operating_point, torch)
    metadata = checkpoint_metadata(
        config,
        warm_start={"members": warm_records, "config_hash": stable_hash(dict(config))},
    )
    payload = {
        "metadata": metadata,
        "model_state_dicts": [model.state_dict() for model in models],
        "calibration": calibration.to_dict(),
        "best_validation_losses": best_losses,
    }
    checkpoint = output_dir / "accvp_v1_predictor.pt"
    torch.save(payload, checkpoint)
    calibration_path = output_dir / "accvp_v1_calibration.json"
    with calibration_path.open("w", encoding="utf-8") as handle:
        json.dump(calibration.to_dict(), handle, indent=2, sort_keys=True)
    operating_point_path = output_dir / "accvp_v1_operating_point.json"
    with operating_point_path.open("w", encoding="utf-8") as handle:
        json.dump(operating_point, handle, indent=2, sort_keys=True)
    final_test_path = output_dir / "accvp_v1_final_test_diagnostics.json"
    with final_test_path.open("w", encoding="utf-8") as handle:
        json.dump(final_test, handle, indent=2, sort_keys=True)
    dataset_manifest_path = dataset_dir / "manifests" / "dataset_manifest.json"
    split_manifest_path = dataset_dir / "manifests" / "split_manifest.jsonl"
    dataset_manifest = read_json(dataset_manifest_path)
    artifact_manifest = {
        "artifact_kind": "accvp_v1_artifact_bundle",
        "architecture_version": metadata["architecture_version"],
        "counterfactual_schema_version": int(metadata["counterfactual_schema_version"]),
        "predictor_sha256": file_sha256(checkpoint),
        "calibration_sha256": file_sha256(calibration_path),
        "operating_point_sha256": file_sha256(operating_point_path),
        "dataset_manifest_sha256": file_sha256(dataset_manifest_path),
        "split_manifest_sha256": file_sha256(split_manifest_path),
        "dataset_fingerprint": str(dataset_manifest.get("dataset_fingerprint", "")),
        "risk_model_fingerprint": str(dataset_manifest.get("risk_model_fingerprint", "")),
        "config_hash": stable_hash(dict(config)),
        "oracle_report": oracle_report,
    }
    artifact_manifest["artifact_fingerprint"] = stable_hash(artifact_manifest)
    artifact_manifest_path = write_json_atomic(output_dir / "accvp_v1_artifact_manifest.json", artifact_manifest)
    with (output_dir / "training_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "dataset_dir": str(dataset_dir.resolve()),
                "oracle_report": oracle_report,
                "checkpoint": str(checkpoint.resolve()),
                "calibration": str(calibration_path.resolve()),
                "operating_point": str(operating_point_path.resolve()),
                "final_test_diagnostics": str(final_test_path.resolve()),
                "artifact_manifest": str(artifact_manifest_path.resolve()),
                "best_validation_losses": best_losses,
                "event_positive_weights": loss_weights["event_positive_weights"],
                "checkpoint_metadata": metadata,
            },
            handle,
            indent=2,
            sort_keys=True,
        )
    return checkpoint
