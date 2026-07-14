from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from safe_rl.accvp.artifacts import (
    ACCVP_ARTIFACT_GENERATION,
    ACCVP_ARTIFACT_KIND,
    ACCVP_ARTIFACT_PREFIX,
    ACCVP_BUNDLE_SCHEMA_VERSION,
    LIFECYCLE_SEALED_CANDIDATE,
    LIFECYCLE_SHADOW,
    artifact_filename,
    bundle_file_entry,
)
from safe_rl.accvp.availability import OperatingPointAvailabilityError
from safe_rl.accvp.calibration import CalibrationBundle, OneSidedBinnedCalibrator
from safe_rl.accvp.dataset import ACCVPBranchDataset, SPLIT_ALGORITHM_VERSION, build_split_manifest, collate_numpy
from safe_rl.accvp.model import (
    ACCVPPredictor,
    accvp_loss,
    checkpoint_metadata,
    model_kwargs_from_config,
    set_scene_encoder_trainable,
    warm_start_scene_encoder,
)
from safe_rl.accvp.schema import (
    COUNTERFACTUAL_SCHEMA_VERSION,
    SCENARIO_EPISODE_KEY_VERSION,
    file_sha256,
    read_json,
    stable_hash,
    write_json_atomic,
)
from safe_rl.accvp.oracle import validate_oracle_for_training
from safe_rl.accvp.protocol import ACCVP_DATA_CONTRACT_VERSION
from safe_rl.accvp.reproducibility import configure_deterministic_training
from safe_rl.accvp.runtime_contract import (
    formal_runtime_contract_from_config,
    formal_runtime_contract_sha256,
)
from safe_rl.evaluation_protocol import protocol_snapshot
from safe_rl.utils.config import prepare_run_dir
from safe_rl.utils.progress import stage_log


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError("ACCVP training requires torch.") from exc
    return torch


MIN_FORMAL_ENSEMBLE_SIZE = 3
COMPONENT_BOOTSTRAP_VERSION = "fixed_member_split_component_fingerprint_action_group_v2"
OPTIMIZATION_BATCHING_VERSION = "complete_fingerprint_action_group_batches_v1"
TRAINING_HISTORY_SCHEMA_VERSION = 2
LOSS_REDUCTION_VERSION = "nested_exact_weighted_numerator_denominator_sum_v2"
LOSS_COMPONENTS = ("trajectory", "events", "geometry", "ordering", "smoothness")


def _configured_artifact_generation(config: Any) -> str:
    value = config.accvp.get("artifact_generation")
    return "" if value is None else str(value).strip()


def validate_ensemble_configuration(ensemble_size: int, *, mode: str) -> int:
    size = int(ensemble_size)
    if size < 1:
        raise ValueError("ACCVP ensemble_size must be positive")
    if str(mode) == "deployable" and size < MIN_FORMAL_ENSEMBLE_SIZE:
        raise ValueError(
            f"formal ACCVP candidate training requires at least {MIN_FORMAL_ENSEMBLE_SIZE} ensemble members"
        )
    return size


def _state_dict_sha256(state_dict: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        value = state_dict[name].detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


def _resolve_accvp_training_device(config: Any, torch: Any) -> tuple[str, Any]:
    """Resolve the batch-training device without changing runtime inference."""

    training = config.get("training", {})
    requested = str(
        training.get("stage2_device", training.get("device", "auto"))
    ).strip().lower()
    requested = requested or "auto"
    canonical = "cuda" if requested == "gpu" else requested
    if canonical == "auto":
        if bool(torch.cuda.is_available()):
            return requested, torch.device("cuda:0")
        mps = getattr(getattr(torch, "backends", None), "mps", None)
        if mps is not None and bool(mps.is_available()):
            return requested, torch.device("mps")
        return requested, torch.device("cpu")
    if canonical.startswith("cuda") and not bool(torch.cuda.is_available()):
        raise RuntimeError(
            "training.stage2_device requests CUDA for ACCVP training, but "
            "torch.cuda.is_available() is false"
        )
    if canonical == "mps":
        mps = getattr(getattr(torch, "backends", None), "mps", None)
        if mps is None or not bool(mps.is_available()):
            raise RuntimeError(
                "training.stage2_device requests MPS for ACCVP training, but "
                "PyTorch MPS is unavailable"
            )
    return requested, torch.device(canonical)


def _tensor_batch(
    batch: dict[str, np.ndarray],
    torch: Any,
    *,
    device: Any | None = None,
) -> dict[str, Any]:
    integer = {"history_lane_ids", "history_edge_role_ids", "role_ids", "lane_ids", "edge_role_ids", "candidate_action_ids"}
    return {
        key: torch.as_tensor(
            value,
            dtype=torch.long if key in integer else torch.float32,
            device=device,
        )
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


def _fingerprint_action_group_batches(
    dataset: ACCVPBranchDataset,
    groups: list[tuple[int, ...]],
    group_batch_size: int,
):
    """Batch complete fingerprint-action groups as indivisible SGD units."""

    size = max(1, int(group_batch_size))
    if not groups:
        raise ValueError("ACCVP grouped training requires at least one fingerprint-action group")
    for start in range(0, len(groups), size):
        batch_groups = groups[start : start + size]
        items: list[dict[str, np.ndarray]] = []
        for group in batch_groups:
            if not group:
                raise ValueError("ACCVP grouped training contains an empty fingerprint-action group")
            group_items = [dataset[int(index)] for index in group]
            group_weight = sum(
                float(np.asarray(item.get("sample_weight", 1.0), dtype=np.float64))
                for item in group_items
            )
            if not np.isfinite(group_weight) or not np.isclose(
                group_weight,
                1.0,
                rtol=1.0e-6,
                atol=1.0e-7,
            ):
                raise ValueError(
                    "ACCVP fingerprint-action training group must have total sample weight one; "
                    f"indices={list(group)} weight={group_weight}"
                )
            items.extend(group_items)
        yield collate_numpy(items)


def build_component_bootstrap_plan(
    dataset: ACCVPBranchDataset,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Draw one fixed cluster-bootstrap replicate for an ensemble member."""

    if not dataset.component_metadata_complete:
        raise ValueError("ACCVP component bootstrap requires split_component_id for every training root")
    by_component = {
        str(component_id): tuple(int(index) for index in indices)
        for component_id, indices in dataset.branch_indices_by_component.items()
        if indices
    }
    expected_components = {str(value) for value in dataset.split_component_by_root.values() if value}
    missing = sorted(expected_components.difference(by_component))
    if missing:
        raise ValueError(f"ACCVP training components contain no completed branch rows: {missing[:10]}")
    components = tuple(sorted(by_component))
    if not components:
        raise ValueError("ACCVP component bootstrap requires at least one training component")
    component_by_index: dict[int, str] = {}
    for component_id, indices in by_component.items():
        for index in indices:
            if index in component_by_index:
                raise ValueError(f"ACCVP branch row belongs to multiple split components: {index}")
            component_by_index[index] = component_id
    raw_groups = dict(getattr(dataset, "branch_indices_by_fingerprint_action", {}) or {})
    if not raw_groups:
        raise ValueError(
            "ACCVP component bootstrap requires complete fingerprint-action group metadata"
        )
    groups_by_component: dict[str, list[tuple[int, ...]]] = {
        component_id: [] for component_id in components
    }
    grouped_indices: set[int] = set()
    for _key, raw_indices in sorted(raw_groups.items(), key=lambda item: str(item[0])):
        group = tuple(int(index) for index in raw_indices)
        if not group or len(group) != len(set(group)):
            raise ValueError("ACCVP fingerprint-action groups must contain unique branch rows")
        group_components = {component_by_index.get(index, "") for index in group}
        if "" in group_components or len(group_components) != 1:
            raise ValueError(
                "ACCVP fingerprint-action group must map to exactly one split component"
            )
        overlap = grouped_indices.intersection(group)
        if overlap:
            raise ValueError(
                f"ACCVP branch rows belong to multiple fingerprint-action groups: {sorted(overlap)}"
            )
        grouped_indices.update(group)
        groups_by_component[next(iter(group_components))].append(group)
    expected_indices = set(component_by_index)
    if grouped_indices != expected_indices:
        raise ValueError(
            "ACCVP fingerprint-action group metadata does not cover every training row"
        )
    for component_id in groups_by_component:
        groups_by_component[component_id].sort()
    sampled = tuple(
        str(value)
        for value in rng.choice(np.asarray(components, dtype=object), size=len(components), replace=True).tolist()
    )
    multiplicities = Counter(sampled)
    fixed_groups = tuple(
        group
        for component_id in sampled
        for group in groups_by_component[component_id]
    )
    fixed_indices = tuple(index for group in fixed_groups for index in group)
    if not fixed_indices:
        raise ValueError("ACCVP component bootstrap produced no branch rows")
    multiplicity_payload = {key: int(value) for key, value in sorted(multiplicities.items())}
    return {
        "version": COMPONENT_BOOTSTRAP_VERSION,
        "population_component_count": len(components),
        "population_components_hash": stable_hash({"components": list(components)}),
        "sampled_components": sampled,
        "sampled_component_sequence_hash": stable_hash({"components": list(sampled)}),
        "component_multiplicities": multiplicity_payload,
        "component_multiset_hash": stable_hash({"multiplicities": multiplicity_payload}),
        "unique_sampled_component_count": len(multiplicities),
        "within_component_weighting": OPTIMIZATION_BATCHING_VERSION,
        "fixed_group_count": len(fixed_groups),
        "fixed_groups": fixed_groups,
        "fixed_group_hash": stable_hash(
            {"groups": [list(group) for group in fixed_groups]}
        ),
        "fixed_row_count": len(fixed_indices),
        "fixed_indices": fixed_indices,
        "fixed_index_hash": stable_hash({"indices": list(fixed_indices)}),
    }


def shuffled_component_bootstrap_indices(
    plan: dict[str, Any],
    rng: np.random.Generator,
) -> list[int]:
    """Shuffle a member's fixed bootstrap rows without redrawing clusters."""

    if str(plan.get("version", "")) != COMPONENT_BOOTSTRAP_VERSION:
        raise ValueError("unsupported ACCVP component bootstrap plan version")
    indices = [int(value) for value in plan.get("fixed_indices", ())]
    if not indices:
        raise ValueError("ACCVP component bootstrap plan contains no rows")
    rng.shuffle(indices)
    return indices


def shuffled_component_bootstrap_groups(
    plan: dict[str, Any],
    rng: np.random.Generator,
) -> list[tuple[int, ...]]:
    """Shuffle complete fingerprint-action units without splitting replicates."""

    if str(plan.get("version", "")) != COMPONENT_BOOTSTRAP_VERSION:
        raise ValueError("unsupported ACCVP component bootstrap plan version")
    groups = [
        tuple(int(index) for index in group)
        for group in plan.get("fixed_groups", ())
    ]
    if not groups or any(not group for group in groups):
        raise ValueError("ACCVP component bootstrap plan contains no complete groups")
    order = np.arange(len(groups), dtype=np.int64)
    rng.shuffle(order)
    return [groups[int(index)] for index in order.tolist()]


def _loss_component_weight(weights: dict[str, Any], component: str) -> float:
    defaults = {
        "trajectory": 1.0,
        "events": 1.0,
        "geometry": 0.25,
        "ordering": 0.10,
        "smoothness": 0.01,
    }
    return float(weights.get(component, defaults[component]))


def _new_loss_accumulator() -> dict[str, Any]:
    return {
        "batch_count": 0,
        "row_count": 0,
        "sample_weight_sum": 0.0,
        "optimization_batch_total_sum": 0.0,
        "components": {
            component: {"numerator": 0.0, "denominator": 0.0}
            for component in LOSS_COMPONENTS
        },
    }


def _accumulate_loss_statistics(
    accumulator: dict[str, Any],
    *,
    total: Any,
    parts: dict[str, Any],
    batch_np: dict[str, np.ndarray],
) -> None:
    statistics = dict(parts.get("statistics", {}) or {})
    missing = [component for component in LOSS_COMPONENTS if component not in statistics]
    if missing:
        raise ValueError(f"ACCVP loss statistics missing components: {missing}")
    row_count = int(np.asarray(batch_np["candidate_action_ids"]).shape[0])
    sample_weights = np.asarray(
        batch_np.get("sample_weight", np.ones((row_count,), dtype=np.float32)),
        dtype=np.float64,
    ).reshape(-1)
    if sample_weights.shape != (row_count,) or not np.isfinite(sample_weights).all():
        raise ValueError("ACCVP batch sample_weight must be finite with shape [batch]")
    accumulator["batch_count"] += 1
    accumulator["row_count"] += row_count
    accumulator["sample_weight_sum"] += float(sample_weights.sum())
    accumulator["optimization_batch_total_sum"] += float(total.detach().item())
    for component in LOSS_COMPONENTS:
        record = dict(statistics[component])
        numerator = float(record["numerator"].detach().item())
        denominator = float(record["denominator"].detach().item())
        if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator < 0.0:
            raise ValueError(f"invalid ACCVP loss reduction statistics for {component}")
        if denominator == 0.0 and abs(numerator) > np.finfo(np.float64).eps:
            raise ValueError(
                f"ACCVP loss reduction for {component} has a non-zero numerator "
                "with no valid weighted scalars"
            )
        accumulator["components"][component]["numerator"] += numerator
        accumulator["components"][component]["denominator"] += denominator


def _finalise_loss_statistics(
    accumulator: dict[str, Any],
    weights: dict[str, Any],
) -> dict[str, Any]:
    if int(accumulator["batch_count"]) <= 0:
        raise ValueError("cannot finalise empty ACCVP loss statistics")
    components: dict[str, dict[str, float]] = {}
    total = 0.0
    for component in LOSS_COMPONENTS:
        numerator = float(accumulator["components"][component]["numerator"])
        denominator = float(accumulator["components"][component]["denominator"])
        if denominator > 0.0:
            value = numerator / denominator
        elif abs(numerator) <= np.finfo(np.float64).eps:
            value = 0.0
        else:
            raise ValueError(
                f"ACCVP loss reduction for {component} has a non-zero numerator "
                "with no valid weighted scalars"
            )
        weight = _loss_component_weight(weights, component)
        weighted_value = weight * value
        components[component] = {
            "numerator": numerator,
            "denominator": denominator,
            "value": value,
            "weight": weight,
            "weighted_value": weighted_value,
        }
        total += weighted_value
    batch_count = int(accumulator["batch_count"])
    return {
        "reduction_version": LOSS_REDUCTION_VERSION,
        "total": float(total),
        "optimization_batch_total_mean": float(
            accumulator["optimization_batch_total_sum"] / batch_count
        ),
        "batch_count": batch_count,
        "row_count": int(accumulator["row_count"]),
        "sample_weight_sum": float(accumulator["sample_weight_sum"]),
        "components": components,
    }


def _evaluate_loss(
    model: Any,
    dataset: ACCVPBranchDataset,
    torch: Any,
    weights: dict[str, Any],
    *,
    device: Any | None = None,
) -> dict[str, Any]:
    if not len(dataset):
        raise ValueError("cannot evaluate ACCVP loss on an empty dataset")
    model.eval()
    accumulator = _new_loss_accumulator()
    with torch.no_grad():
        for batch_np in _batches(dataset, list(range(len(dataset))), 64):
            batch = _tensor_batch(batch_np, torch, device=device)
            loss, parts = accvp_loss(_model_output(model, batch), batch, weights)
            _accumulate_loss_statistics(
                accumulator,
                total=loss,
                parts=parts,
                batch_np=batch_np,
            )
    return _finalise_loss_statistics(accumulator, weights)


def _event_positive_weights(dataset: ACCVPBranchDataset) -> list[float]:
    positives = np.zeros((4,), dtype=np.float64)
    counts = np.zeros((4,), dtype=np.float64)
    for index in range(len(dataset)):
        item = dataset[index]
        sample_weight = float(np.asarray(item.get("sample_weight", 1.0), dtype=np.float64))
        if not np.isfinite(sample_weight) or sample_weight <= 0.0:
            raise ValueError(f"invalid ACCVP sample_weight in train item {index}")
        mask = item["event_mask"] > 0.0
        positives[mask] += item["event_targets"][mask] * sample_weight
        counts[mask] += sample_weight
    negatives = np.maximum(counts - positives, 0.0)
    return np.clip(
        negatives / np.maximum(positives, np.finfo(np.float64).eps),
        1.0,
        50.0,
    ).tolist()


def _train_response_feature_scales(
    dataset: ACCVPBranchDataset,
    minimums: list[float] | tuple[float, ...],
) -> list[float]:
    """Compute response normalization from train-only valid scalar targets."""

    floors = np.asarray(minimums, dtype=np.float64)
    if floors.shape != (5,) or not np.isfinite(floors).all() or np.any(floors <= 0.0):
        raise ValueError("response_feature_scale_minimums must contain five positive finite values")
    count = 0.0
    total = np.zeros((5,), dtype=np.float64)
    total_sq = np.zeros((5,), dtype=np.float64)
    heading_sin = 0.0
    heading_cos = 0.0
    for index in range(len(dataset)):
        item = dataset[index]
        sample_weight = float(np.asarray(item.get("sample_weight", 1.0), dtype=np.float64))
        if not np.isfinite(sample_weight) or sample_weight <= 0.0:
            raise ValueError(f"invalid ACCVP sample_weight in train item {index}")
        values = np.asarray(item["actor_response"], dtype=np.float64)
        valid = np.asarray(item["actor_response_mask"], dtype=np.float64) > 0.0
        selected = values[valid]
        if not selected.size:
            continue
        if not np.isfinite(selected).all():
            raise ValueError(f"non-finite ACCVP actor response target in train item {index}")
        count += float(selected.shape[0]) * sample_weight
        total += selected.sum(axis=0) * sample_weight
        total_sq += np.square(selected).sum(axis=0) * sample_weight
        heading_sin += float(np.sin(selected[:, 2]).sum()) * sample_weight
        heading_cos += float(np.cos(selected[:, 2]).sum()) * sample_weight
    if count <= 0:
        raise ValueError("cannot derive response feature scales without valid train response targets")
    variance = np.maximum(total_sq / count - np.square(total / count), 0.0)
    scales = np.sqrt(variance)
    resultant = min(1.0, max(1.0e-12, float(np.hypot(heading_sin, heading_cos) / count)))
    scales[2] = np.sqrt(max(0.0, -2.0 * np.log(resultant)))
    scales = np.maximum(scales, floors)
    return scales.tolist()


def _duplicate_weighting_provenance(
    splits: dict[str, ACCVPBranchDataset],
) -> dict[str, Any]:
    versions = {
        str(getattr(dataset, "duplicate_weighting_version", ""))
        for dataset in splits.values()
    }
    if len(versions) != 1 or not next(iter(versions), ""):
        raise ValueError("ACCVP splits do not share one duplicate-weighting contract")
    rows: dict[str, dict[str, Any]] = {}
    for name, dataset in splits.items():
        sample_weights = dict(getattr(dataset, "sample_weight_by_index", {}) or {})
        rows[str(name)] = {
            "raw_row_count": int(len(dataset)),
            "fingerprint_action_group_count": int(
                getattr(dataset, "duplicate_group_count", len(dataset))
            ),
            "duplicate_row_count": int(getattr(dataset, "duplicate_row_count", 0)),
            "sample_weight_sum": float(sum(float(value) for value in sample_weights.values())),
        }
    return {
        "version": next(iter(versions)),
        "weighting_unit": "model_input_fingerprint_x_action",
        "cluster_sampling_unit": "split_component",
        "statistical_independence_claim": False,
        "group_total_weight": 1.0,
        "splits": rows,
    }


def _clear_generation_artifacts(output_dir: Path) -> None:
    for kind in (
        "predictor",
        "calibration",
        "operating_point",
        "training_history",
        "candidate_manifest",
        "shadow_manifest",
        "validated_manifest",
        "final_test_diagnostics",
        "training_manifest",
        "tuning_failure",
    ):
        name = artifact_filename(kind)
        path = output_dir / name
        if path.exists():
            path.unlink()
        temporary = path.with_suffix(path.suffix + ".tmp")
        if temporary.exists():
            temporary.unlink()


def _write_tuning_failure_diagnostics(output_dir: Path, error: OperatingPointAvailabilityError) -> Path:
    diagnostics = dict(error.diagnostics)
    diagnostics["deployable_artifact"] = False
    diagnostics["error"] = str(error)
    diagnostics["artifact_generation"] = ACCVP_ARTIFACT_GENERATION
    return write_json_atomic(output_dir / artifact_filename("tuning_failure"), diagnostics)


def _calibrate(models: list[Any], dataset: ACCVPBranchDataset, torch: Any, calibration_config: Any) -> CalibrationBundle:
    if not len(dataset):
        raise ValueError("ACCVP calibration split is empty")
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for model in models:
        model.eval()
    with torch.no_grad():
        all_indices = list(range(len(dataset)))
        for start in range(0, len(all_indices), 64):
            batch_indices = all_indices[start : start + 64]
            batch_np = collate_numpy(dataset[index] for index in batch_indices)
            batch = _tensor_batch(batch_np, torch)
            events = []
            for model in models:
                events.append(torch.sigmoid(_model_output(model, batch)["event_logits"]).cpu().numpy())
            stacked = np.stack(events, axis=0)
            for local_index, dataset_index in enumerate(batch_indices):
                branch_row = dataset.rows[dataset_index]
                root_id = str(branch_row["root_id"])
                fingerprint = str(dataset.observation_fingerprint_by_root.get(root_id, ""))
                if not fingerprint:
                    raise ValueError(
                        f"calibration row {dataset_index} is missing model-input fingerprint"
                    )
                key = (fingerprint, int(branch_row["action_id"]))
                grouped.setdefault(key, []).append(
                    {
                        "proxy_score": float(stacked[:, local_index, 0].max()),
                        "safety_score": float(stacked[:, local_index, 1].max()),
                        "viability_score": float(stacked[:, local_index, 3].min()),
                        "proxy_label": float(batch_np["event_targets"][local_index, 0]),
                        "safety_label": float(batch_np["event_targets"][local_index, 1]),
                        "viability_label": float(batch_np["event_targets"][local_index, 3]),
                        "viability_eligible": bool(batch_np["event_mask"][local_index, 3] > 0.0),
                        "split_component_id": str(
                            dataset.split_component_by_root.get(root_id, "")
                        ),
                    }
                )
    score_tolerance = float(calibration_config.get("duplicate_score_tolerance", 1.0e-6))
    if score_tolerance < 0.0:
        raise ValueError("calibration duplicate_score_tolerance must be non-negative")
    proxy_scores: list[float] = []
    proxy_labels: list[float] = []
    safety_scores: list[float] = []
    safety_labels: list[float] = []
    viability_scores: list[float] = []
    viability_labels: list[float] = []
    proxy_clusters: list[str] = []
    safety_clusters: list[str] = []
    viability_clusters: list[str] = []
    max_score_delta = 0.0
    raw_viability_count = 0
    for key in sorted(grouped):
        rows = grouped[key]
        for score_name in ("proxy_score", "safety_score", "viability_score"):
            values = [float(row[score_name]) for row in rows]
            delta = max(values) - min(values)
            max_score_delta = max(max_score_delta, delta)
            if delta > score_tolerance:
                raise ValueError(
                    "duplicate fingerprint-action calibration scores diverged beyond tolerance: "
                    f"key={key} score={score_name} delta={delta} tolerance={score_tolerance}"
                )
        # Risk scores use the conservative group maximum. Viability is a lower
        # bound, so its conservative group representative is the minimum.
        proxy_scores.append(max(float(row["proxy_score"]) for row in rows))
        safety_scores.append(max(float(row["safety_score"]) for row in rows))
        proxy_labels.append(float(np.mean([row["proxy_label"] for row in rows])))
        safety_labels.append(float(np.mean([row["safety_label"] for row in rows])))
        components = {str(row["split_component_id"]) for row in rows}
        if "" in components or len(components) != 1:
            raise ValueError(
                "fingerprint-action calibration group must map to exactly one split component"
            )
        component_id = next(iter(components))
        proxy_clusters.append(component_id)
        safety_clusters.append(component_id)
        eligible = [row for row in rows if bool(row["viability_eligible"])]
        raw_viability_count += len(eligible)
        if eligible:
            viability_scores.append(min(float(row["viability_score"]) for row in eligible))
            viability_labels.append(
                float(np.mean([row["viability_label"] for row in eligible]))
            )
            viability_clusters.append(component_id)
    fit_kwargs = {
        "bins": int(calibration_config.get("bins", 20)),
        "nominal_alpha": float(calibration_config.get("nominal_alpha", 0.05)),
        "bonferroni_family_size": int(calibration_config.get("bonferroni_signal_count", 3)),
    }
    proxy_calibrator = OneSidedBinnedCalibrator.fit_clustered_bounded_means(
        proxy_scores, proxy_labels, proxy_clusters, **fit_kwargs
    )
    safety_calibrator = OneSidedBinnedCalibrator.fit_clustered_bounded_means(
        safety_scores, safety_labels, safety_clusters, **fit_kwargs
    )
    viability_calibrator = OneSidedBinnedCalibrator.fit_clustered_bounded_means(
        viability_scores, viability_labels, viability_clusters, **fit_kwargs
    )
    return CalibrationBundle(
        proxy_collision=proxy_calibrator,
        safety_violation=safety_calibrator,
        merge_viability=viability_calibrator,
        provenance={
            "split": "calibration",
            "candidate_level_only": True,
            "fingerprint_action_weighting_unit": "model_input_fingerprint_x_action_group",
            "calibration_statistical_unit": "split_component_x_score_bin",
            "calibration_estimand": (
                "equal_weight_split_component_mean_within_score_bin_v1"
            ),
            "confidence_method": "one_sided_hoeffding_component_mean_v1",
            "proxy_count": len(proxy_labels),
            "safety_count": len(safety_labels),
            "eligible_viability_count": len(viability_labels),
            "duplicate_weighting_version": str(
                getattr(dataset, "duplicate_weighting_version", "")
            ),
            "raw_candidate_count": int(len(dataset)),
            "effective_fingerprint_action_group_count": len(grouped),
            "raw_eligible_viability_count": int(raw_viability_count),
            "effective_eligible_viability_group_count": len(viability_labels),
            "effective_split_component_count": {
                "proxy_collision": len(set(proxy_clusters)),
                "safety_violation": len(set(safety_clusters)),
                "merge_viability": len(set(viability_clusters)),
            },
            "effective_component_count_by_score_bin": {
                "proxy_collision": proxy_calibrator.bin_effective_counts.tolist(),
                "safety_violation": safety_calibrator.bin_effective_counts.tolist(),
                "merge_viability": viability_calibrator.bin_effective_counts.tolist(),
            },
            "duplicate_row_count": int(getattr(dataset, "duplicate_row_count", 0)),
            "group_label_aggregation": "mean_stochastic_outcome_v1",
            "within_component_bin_aggregation": "mean_of_fingerprint_action_groups_v1",
            "group_risk_score_aggregation": "maximum",
            "group_viability_score_aggregation": "minimum",
            "duplicate_score_tolerance": score_tolerance,
            "max_within_group_score_delta": float(max_score_delta),
            "duplicate_weighting_applied_to_component_bound": True,
            "wilson_independence_assumption_used": False,
            **fit_kwargs,
        },
    )


def _write_predictor_and_calibration(
    *,
    output_dir: Path,
    config: Any,
    dataset_dir: Path,
    models: list[Any],
    calibration: CalibrationBundle,
    warm_records: list[dict[str, Any]],
    best_losses: list[float],
    loss_weights: dict[str, Any],
    oracle_report: dict[str, Any],
    training_history: dict[str, Any],
    reproducibility_profile: dict[str, Any],
    duplicate_weighting: dict[str, Any],
    formal_runtime_contract: dict[str, Any],
    torch: Any,
    mode: str,
    operating_point: dict[str, Any] | None = None,
) -> Path:
    history_payload = dict(training_history)
    history_payload["history_fingerprint"] = stable_hash(history_payload)
    history_path = write_json_atomic(
        output_dir / artifact_filename("training_history"),
        history_payload,
    )
    history_sha256 = file_sha256(history_path)
    metadata = checkpoint_metadata(
        config,
        warm_start={"members": warm_records, "config_hash": stable_hash(dict(config))},
    )
    scale_payload = {
        "source": str(loss_weights.get("response_feature_scales_source", "configured")),
        "values": [float(value) for value in loss_weights["response_feature_scales"]],
        "duplicate_weighting": {
            "version": duplicate_weighting["version"],
            "weighting_unit": duplicate_weighting["weighting_unit"],
            "cluster_sampling_unit": duplicate_weighting["cluster_sampling_unit"],
            "train": duplicate_weighting["splits"]["train"],
        },
    }
    metadata["response_feature_normalization"] = {
        **scale_payload,
        "sha256": stable_hash(scale_payload),
    }
    metadata["training_history"] = {
        "artifact_generation": ACCVP_ARTIFACT_GENERATION,
        "filename": history_path.name,
        "sha256": history_sha256,
        "history_fingerprint": history_payload["history_fingerprint"],
        "schema_version": TRAINING_HISTORY_SCHEMA_VERSION,
    }
    metadata["reproducibility"] = dict(reproducibility_profile)
    metadata["duplicate_weighting"] = dict(duplicate_weighting)
    metadata["artifact_generation"] = ACCVP_ARTIFACT_GENERATION
    metadata["configured_artifact_generation"] = _configured_artifact_generation(config)
    payload = {
        "metadata": metadata,
        "model_state_dicts": [model.state_dict() for model in models],
        "calibration": calibration.to_dict(),
        "best_validation_losses": best_losses,
    }
    checkpoint = output_dir / artifact_filename("predictor")
    torch.save(payload, checkpoint)
    calibration_path = write_json_atomic(
        output_dir / artifact_filename("calibration"),
        calibration.to_dict(),
    )
    operating_point_path = None
    if operating_point is not None:
        operating_point_path = write_json_atomic(
            output_dir / artifact_filename("operating_point"),
            operating_point,
        )
    dataset_manifest_path = dataset_dir / "manifests" / "dataset_manifest.json"
    split_manifest_path = dataset_dir / "manifests" / "split_manifest.jsonl"
    split_provenance_path = dataset_dir / "manifests" / "split_provenance.json"
    dataset_manifest = read_json(dataset_manifest_path)
    formal_candidate = mode == "deployable"
    evidence = protocol_snapshot(config, base_dir=Path.cwd())
    formal_runtime_contract_hash = formal_runtime_contract_sha256(
        formal_runtime_contract
    )
    files = {
        "predictor": bundle_file_entry(checkpoint, manifest_dir=output_dir),
        "calibration": bundle_file_entry(calibration_path, manifest_dir=output_dir),
        "training_history": bundle_file_entry(history_path, manifest_dir=output_dir),
    }
    if operating_point_path is not None:
        files["operating_point"] = bundle_file_entry(
            operating_point_path,
            manifest_dir=output_dir,
        )
    artifact_manifest = {
        "artifact_kind": ACCVP_ARTIFACT_KIND,
        "bundle_schema_version": ACCVP_BUNDLE_SCHEMA_VERSION,
        "artifact_generation": ACCVP_ARTIFACT_GENERATION,
        "configured_artifact_generation": _configured_artifact_generation(config),
        "artifact_prefix": ACCVP_ARTIFACT_PREFIX,
        "artifact_variant": "full_candidate_gate_v1",
        "lifecycle_state": (
            LIFECYCLE_SEALED_CANDIDATE if formal_candidate else LIFECYCLE_SHADOW
        ),
        "capabilities": ["candidate_table_observation", "task_viability"],
        "deployable_artifact": False,
        "deployable_claim": False,
        "deployment_scope": "experimental_candidate_table_policy_runtime_v1",
        "runtime_timeout_contract": str(
            config.accvp.get("observation", {}).get(
                "timeout_contract", "soft_realtime_post_return_v1"
            )
        ),
        "hard_realtime_claim": False,
        "safety_certified": False,
        "formal_runtime_contract": formal_runtime_contract,
        "formal_runtime_contract_sha256": formal_runtime_contract_hash,
        "holdout_state": "sealed" if formal_candidate else "not_applicable",
        "threshold_selection_split": "operating_point" if formal_candidate else None,
        "test_used_for_threshold_selection": False,
        "architecture_version": metadata["architecture_version"],
        "counterfactual_schema_version": int(metadata["counterfactual_schema_version"]),
        "accvp_activation_distance_m": float(dataset_manifest.get("accvp_activation_distance_m", -1.0)),
        "data_contract_hash": str(dataset_manifest.get("data_contract_hash", "")),
        "predictor_sha256": file_sha256(checkpoint),
        "calibration_sha256": file_sha256(calibration_path),
        "dataset_manifest_sha256": file_sha256(dataset_manifest_path),
        "split_manifest_sha256": file_sha256(split_manifest_path),
        "split_provenance_sha256": file_sha256(split_provenance_path),
        "dataset_fingerprint": str(dataset_manifest.get("dataset_fingerprint", "")),
        "risk_model_fingerprint": str(dataset_manifest.get("risk_model_fingerprint", "")),
        "config_hash": stable_hash(dict(config)),
        "oracle_report": oracle_report,
        "response_feature_normalization": metadata["response_feature_normalization"],
        "training_history_sha256": history_sha256,
        "training_history_fingerprint": history_payload["history_fingerprint"],
        "reproducibility": dict(reproducibility_profile),
        "duplicate_weighting": dict(duplicate_weighting),
        "files": files,
        "evidence_protocol_id": str(evidence.get("protocol_id", "")),
        "seed_ledger_sha256": evidence.get("seed_ledger_sha256"),
    }
    if operating_point_path is not None:
        artifact_manifest["operating_point_sha256"] = file_sha256(operating_point_path)
    artifact_manifest["artifact_fingerprint"] = stable_hash(artifact_manifest)
    manifest_name = artifact_filename(
        "candidate_manifest" if formal_candidate else "shadow_manifest"
    )
    artifact_manifest_path = write_json_atomic(output_dir / manifest_name, artifact_manifest)
    write_json_atomic(
        output_dir / artifact_filename("training_manifest"),
        {
            "artifact_kind": "accvp_training_manifest_v2",
            "artifact_generation": ACCVP_ARTIFACT_GENERATION,
            "configured_artifact_generation": _configured_artifact_generation(config),
            "dataset_dir": str(dataset_dir.resolve()),
            "oracle_report": oracle_report,
            "checkpoint": str(checkpoint.resolve()),
            "calibration": str(calibration_path.resolve()),
            "operating_point": "" if operating_point_path is None else str(operating_point_path.resolve()),
            "training_history": str(history_path.resolve()),
            "training_history_sha256": history_sha256,
            "training_history_fingerprint": history_payload["history_fingerprint"],
            "final_test_diagnostics": "",
            "artifact_manifest": str(artifact_manifest_path.resolve()),
            "artifact_manifest_sha256": file_sha256(artifact_manifest_path),
            "deployable_artifact": False,
            "deployable_claim": False,
            "holdout_state": "sealed" if formal_candidate else "not_applicable",
            "lifecycle_state": artifact_manifest["lifecycle_state"],
            "threshold_selection_split": "operating_point" if formal_candidate else None,
            "test_used_for_threshold_selection": False,
            "evidence_protocol_id": str(evidence.get("protocol_id", "")),
            "seed_ledger_sha256": evidence.get("seed_ledger_sha256"),
            "mode": mode,
            "best_validation_losses": best_losses,
            "event_positive_weights": loss_weights["event_positive_weights"],
            "duplicate_weighting": dict(duplicate_weighting),
            "reproducibility": dict(reproducibility_profile),
            "formal_runtime_contract": formal_runtime_contract,
            "formal_runtime_contract_sha256": formal_runtime_contract_hash,
            "checkpoint_metadata": metadata,
        },
    )
    return checkpoint


def train_accvp(config: Any, dataset_dir: str | Path, *, mode: str = "deployable") -> Path:
    training_started = perf_counter()
    mode = str(mode).strip().lower()
    if mode not in {"shadow", "deployable"}:
        raise ValueError("ACCVP training mode must be 'shadow' or 'deployable'")
    configured_artifact_generation = _configured_artifact_generation(config)
    if (
        mode == "deployable"
        and configured_artifact_generation
        and configured_artifact_generation != ACCVP_ARTIFACT_GENERATION
    ):
        raise ValueError(
            "formal ACCVP artifact_generation mismatch: "
            f"configured={configured_artifact_generation!r} "
            f"required={ACCVP_ARTIFACT_GENERATION!r}"
        )
    training = config.accvp.training
    ensemble_size = validate_ensemble_configuration(int(config.accvp.ensemble_size), mode=mode)
    dataset_dir = Path(dataset_dir)
    oracle_report = validate_oracle_for_training(config, dataset_dir)
    dataset_manifest = read_json(dataset_dir / "manifests" / "dataset_manifest.json")
    if int(dataset_manifest.get("counterfactual_schema_version", -1)) != COUNTERFACTUAL_SCHEMA_VERSION:
        raise ValueError(
            "formal ACCVP training requires current counterfactual schema "
            f"{COUNTERFACTUAL_SCHEMA_VERSION}"
        )
    configured_contract_version = str(
        config.accvp.get("data_contract_version", ACCVP_DATA_CONTRACT_VERSION)
    )
    if configured_contract_version != ACCVP_DATA_CONTRACT_VERSION:
        raise ValueError(
            "formal ACCVP training has an unsupported data contract: "
            f"configured={configured_contract_version!r} "
            f"supported={ACCVP_DATA_CONTRACT_VERSION!r}"
        )
    if (
        str(dict(dataset_manifest.get("data_contract", {})).get("protocol_version", ""))
        != configured_contract_version
    ):
        raise ValueError(
            "formal ACCVP training requires dataset/config data-contract agreement: "
            f"configured={configured_contract_version!r}"
        )
    split_path = dataset_dir / "manifests" / "split_manifest.jsonl"
    if not split_path.exists():
        oracle_exclusion = dict(oracle_report["training_exclusion_audit"])
        build_split_manifest(
            dataset_dir,
            seed=int(config.run.seed),
            require_all_splits=True,
            excluded_episode_seeds=oracle_exclusion["required_seeds"],
            excluded_cohort_roles=[oracle_exclusion["cohort_role"]],
        )
    split_provenance = read_json(dataset_dir / "manifests" / "split_provenance.json")
    if str(split_provenance.get("split_algorithm_version", "")) != SPLIT_ALGORITHM_VERSION:
        raise ValueError(
            "formal ACCVP training requires episode/observation/scenario connected-component splitting"
        )
    if str(split_provenance.get("scenario_episode_key_version", "")) != SCENARIO_EPISODE_KEY_VERSION:
        raise ValueError("formal ACCVP training requires the current scenario-episode key contract")
    expected_scenario_route_hash = str(
        dataset_manifest.get("scenario_route_hash")
        or dict(dataset_manifest.get("data_contract", {})).get("scenario_route_hash", "")
    )
    if not expected_scenario_route_hash or str(split_provenance.get("scenario_route_hash", "")) != expected_scenario_route_hash:
        raise ValueError("formal ACCVP training requires split provenance bound to the dataset scenario route")
    if int(split_provenance.get("cross_split_episode_overlap_count", -1)) != 0:
        raise ValueError("formal ACCVP training blocked by cross-split episode overlap")
    if int(split_provenance.get("cross_split_observation_fingerprint_overlap_count", -1)) != 0:
        raise ValueError("formal ACCVP training blocked by cross-split observation-fingerprint overlap")
    if int(split_provenance.get("cross_split_scenario_episode_overlap_count", -1)) != 0:
        raise ValueError("formal ACCVP training blocked by cross-split scenario-episode overlap")
    if int(split_provenance.get("missing_observation_fingerprint_count", -1)) != 0:
        raise ValueError("formal ACCVP training requires a model-input fingerprint for every root")
    if int(split_provenance.get("missing_scenario_episode_key_count", -1)) != 0:
        raise ValueError("formal ACCVP training requires a scenario-episode key for every root")
    train_set = ACCVPBranchDataset(dataset_dir, "train")
    validation_set = ACCVPBranchDataset(dataset_dir, "validation")
    calibration_set = ACCVPBranchDataset(dataset_dir, "calibration")
    operating_set = ACCVPBranchDataset(dataset_dir, "operating_point")
    required_splits = {
        "train": train_set,
        "validation": validation_set,
        "calibration": calibration_set,
        "operating_point": operating_set,
    }
    empty = [name for name, split in required_splits.items() if not len(split)]
    if empty:
        raise ValueError(f"ACCVP formal training requires non-empty grouped splits; empty={empty}")
    incomplete_components = [name for name, split in required_splits.items() if not split.component_metadata_complete]
    if incomplete_components:
        raise ValueError(
            "ACCVP formal training requires split_component_id for every root; "
            f"incomplete={incomplete_components}"
        )
    duplicate_weighting = _duplicate_weighting_provenance(required_splits)
    stage_log(
        "accvp_training",
        "TRAIN_START "
        f"mode={mode} dataset={dataset_dir.resolve()} ensemble_size={ensemble_size} "
        f"epochs={int(training.epochs)} batch_size={int(training.batch_size)} "
        + " ".join(
            f"{name}_rows={len(split)}" for name, split in required_splits.items()
        ),
    )
    # Keep every fail-closed provenance check ahead of Torch import and global
    # deterministic-state mutation.  Besides producing the most relevant
    # validation error, this lets callers reject bad inputs in a long-lived
    # process whose CUDA context may already have been initialized.  A valid
    # formal run should still be launched in a fresh process so the CUBLAS
    # workspace contract can be established before CUDA initialization.
    formal_runtime_contract = formal_runtime_contract_from_config(
        config,
        declared=True,
        base_dir=Path.cwd(),
    )
    torch = _torch()
    requested_training_device, training_device = _resolve_accvp_training_device(
        config,
        torch,
    )
    deterministic_requested = bool(training.get("deterministic", False))
    deterministic_enabled = bool(mode == "deployable" or deterministic_requested)
    configured_threads = training.get("deterministic_torch_threads")
    if configured_threads is None and mode == "deployable":
        configured_threads = 1
    reproducibility_profile = configure_deterministic_training(
        torch,
        enabled=deterministic_enabled,
        torch_threads=None if configured_threads is None else int(configured_threads),
        cuda_in_scope=getattr(training_device, "type", str(training_device)) == "cuda",
    )
    reproducibility_profile.update(
        {
            "formal_mode_requires_determinism": mode == "deployable",
            "configured_deterministic": deterministic_requested,
            "effective_deterministic": deterministic_enabled,
            "training_device_requested": requested_training_device,
            "training_device_effective": str(training_device),
            "post_training_inference_device": "cpu",
        }
    )
    if mode == "deployable" and not bool(
        reproducibility_profile.get("deterministic_algorithms", False)
    ):
        raise RuntimeError("formal ACCVP training requires deterministic Torch algorithms")
    stage_log(
        "accvp_training",
        "TRAIN_DEVICE "
        f"requested={requested_training_device} effective={training_device} "
        "calibration=cpu operating_point=cpu runtime=cpu",
    )
    loss_weights = dict(training.loss_weights)
    configured_scales = loss_weights.get("response_feature_scales", "auto_train")
    auto_scales = configured_scales is None or (
        isinstance(configured_scales, str) and configured_scales in {"auto", "auto_train"}
    )
    if auto_scales:
        loss_weights["response_feature_scales"] = _train_response_feature_scales(
            train_set,
            list(training.get("response_feature_scale_minimums", [1.0, 1.0, 0.1, 1.0, 0.5])),
        )
        loss_weights["response_feature_scales_source"] = "train_split"
    else:
        loss_weights["response_feature_scales"] = [float(value) for value in configured_scales]
        loss_weights["response_feature_scales_source"] = "configured"
    loss_weights["event_positive_weights"] = _event_positive_weights(train_set)
    output_dir = prepare_run_dir(config, "accvp")
    _clear_generation_artifacts(output_dir)
    warm = config.accvp.warm_start
    warm_source = Path(str(warm.checkpoint)) if warm.get("checkpoint") else None
    source_payload = None
    if bool(warm.enabled):
        if warm_source is None or not warm_source.exists():
            raise FileNotFoundError("accvp.warm_start.enabled requires an existing WcDT v3 checkpoint")
        source_payload = torch.load(
            warm_source,
            map_location="cpu",
            weights_only=True,
        )
        if not source_payload.get("model_state_dicts"):
            raise ValueError("WcDT v3 warm-start checkpoint has no model_state_dicts")
        stage_log(
            "accvp_training",
            f"WARM_START_LOADED checkpoint={warm_source.resolve()} members={len(source_payload['model_state_dicts'])}",
        )
    models: list[Any] = []
    warm_records: list[dict[str, Any]] = []
    best_losses: list[float] = []
    member_histories: list[dict[str, Any]] = []
    for member in range(ensemble_size):
        member_started = perf_counter()
        member_seed = int(config.run.seed) + int(training.ensemble_seed_offset) * member
        rng = np.random.default_rng(member_seed)
        torch.manual_seed(member_seed)
        if getattr(training_device, "type", str(training_device)) == "cuda":
            torch.cuda.manual_seed_all(member_seed)
        model = ACCVPPredictor(**model_kwargs_from_config(config))
        warm_record: dict[str, Any] = {
            "enabled": bool(warm.enabled),
            "member": member,
            "member_seed": member_seed,
        }
        bootstrap_plan = build_component_bootstrap_plan(train_set, rng)
        stage_log(
            "accvp_training",
            "MEMBER_START "
            f"index={member + 1}/{ensemble_size} member_seed={member_seed} "
            f"bootstrap_components={bootstrap_plan['population_component_count']} "
            f"sampled_unique_components={bootstrap_plan['unique_sampled_component_count']}",
        )
        warm_record.update(
            {
                "bootstrap_sampling_version": str(bootstrap_plan["version"]),
                "bootstrap_population_component_count": int(bootstrap_plan["population_component_count"]),
                "bootstrap_population_components_hash": str(bootstrap_plan["population_components_hash"]),
                "bootstrap_sampled_component_sequence_hash": str(
                    bootstrap_plan["sampled_component_sequence_hash"]
                ),
                "bootstrap_component_multiset_hash": str(bootstrap_plan["component_multiset_hash"]),
                "bootstrap_component_multiplicities": dict(bootstrap_plan["component_multiplicities"]),
                "bootstrap_unique_sampled_component_count": int(
                    bootstrap_plan["unique_sampled_component_count"]
                ),
                "bootstrap_within_component_weighting": str(
                    bootstrap_plan["within_component_weighting"]
                ),
                "bootstrap_fixed_group_count": int(
                    bootstrap_plan["fixed_group_count"]
                ),
                "bootstrap_fixed_group_hash": str(
                    bootstrap_plan["fixed_group_hash"]
                ),
                "bootstrap_fixed_row_count": int(bootstrap_plan["fixed_row_count"]),
                "bootstrap_fixed_index_hash": str(bootstrap_plan["fixed_index_hash"]),
            }
        )
        if source_payload is not None:
            states = source_payload["model_state_dicts"]
            warm_record.update(warm_start_scene_encoder(model, states[member % len(states)]))
            warm_record["source_checkpoint"] = str(warm_source.resolve())
            warm_record["source_sha256"] = file_sha256(warm_source)
        model = model.to(training_device)
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
        best_epoch = -1
        bootstrap_epoch_order_hashes: list[str] = []
        epoch_history: list[dict[str, Any]] = []
        for epoch in range(int(training.epochs)):
            scene_encoder_trainable = epoch >= int(warm.freeze_encoder_epochs)
            set_scene_encoder_trainable(model, scene_encoder_trainable)
            model.train()
            groups = shuffled_component_bootstrap_groups(bootstrap_plan, rng)
            order_hash = stable_hash(
                {
                    "epoch": epoch,
                    "batching_version": OPTIMIZATION_BATCHING_VERSION,
                    "groups": [list(group) for group in groups],
                }
            )
            bootstrap_epoch_order_hashes.append(order_hash)
            train_accumulator = _new_loss_accumulator()
            for batch_np in _fingerprint_action_group_batches(
                train_set,
                groups,
                int(training.batch_size),
            ):
                batch = _tensor_batch(batch_np, torch, device=training_device)
                optimizer.zero_grad(set_to_none=True)
                loss, parts = accvp_loss(_model_output(model, batch), batch, loss_weights)
                _accumulate_loss_statistics(
                    train_accumulator,
                    total=loss,
                    parts=parts,
                    batch_np=batch_np,
                )
                loss.backward()
                optimizer.step()
            train_statistics = _finalise_loss_statistics(train_accumulator, loss_weights)
            validation_statistics = _evaluate_loss(
                model,
                validation_set,
                torch,
                loss_weights,
                device=training_device,
            )
            validation_loss = float(validation_statistics["total"])
            epoch_record = {
                "epoch": int(epoch),
                "scene_encoder_trainable": bool(scene_encoder_trainable),
                "learning_rates": {
                    "scene_encoder": float(optimizer.param_groups[0]["lr"]),
                    "heads": float(optimizer.param_groups[1]["lr"]),
                },
                "train_order_sha256": order_hash,
                "train_batching_version": OPTIMIZATION_BATCHING_VERSION,
                "train_fingerprint_action_group_count": len(groups),
                "validation_order_sha256": stable_hash(
                    {"indices": list(range(len(validation_set)))}
                ),
                "selected_best": False,
                "train": train_statistics,
                "validation": validation_statistics,
            }
            epoch_history.append(epoch_record)
            selected_best = validation_loss < best_loss
            if validation_loss < best_loss:
                best_loss = validation_loss
                best_epoch = int(epoch)
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stage_log(
                "accvp_training",
                "EPOCH_END "
                f"member={member + 1}/{ensemble_size} epoch={epoch + 1}/{int(training.epochs)} "
                f"train_total={float(train_statistics['total']):.8f} "
                f"validation_total={validation_loss:.8f} selected_best={selected_best}",
            )
        if best_state is not None:
            model.load_state_dict(best_state)
        if best_epoch < 0:
            raise RuntimeError("ACCVP training did not select a best epoch")
        epoch_history[best_epoch]["selected_best"] = True
        # Calibration, operating-point selection, checkpoint serialization and
        # deployment runtime intentionally remain on CPU. This keeps the
        # existing runtime latency contract independent of the training device.
        model = model.to(torch.device("cpu"))
        del optimizer
        if getattr(training_device, "type", str(training_device)) == "cuda":
            torch.cuda.empty_cache()
        models.append(model)
        best_losses.append(best_loss)
        warm_record["freeze_encoder_epochs"] = int(warm.freeze_encoder_epochs)
        warm_record["bootstrap_epoch_order_hashes"] = bootstrap_epoch_order_hashes
        # Compatibility alias for older diagnostics. These are order hashes of
        # one fixed member-level cluster replicate, not fresh epoch bootstraps.
        warm_record["bootstrap_index_hashes"] = bootstrap_epoch_order_hashes
        warm_record["state_dict_sha256"] = _state_dict_sha256(model.state_dict())
        warm_record["best_epoch"] = int(best_epoch)
        warm_record["best_validation_loss"] = float(best_loss)
        warm_record["epoch_history_sha256"] = stable_hash(epoch_history)
        warm_records.append(warm_record)
        member_histories.append(
            {
                "member": int(member),
                "member_seed": int(member_seed),
                "training_device": str(training_device),
                "post_training_device": "cpu",
                "bootstrap": {
                    "version": str(bootstrap_plan["version"]),
                    "population_component_count": int(
                        bootstrap_plan["population_component_count"]
                    ),
                    "population_components_hash": str(
                        bootstrap_plan["population_components_hash"]
                    ),
                    "sampled_component_sequence_hash": str(
                        bootstrap_plan["sampled_component_sequence_hash"]
                    ),
                    "component_multiset_hash": str(
                        bootstrap_plan["component_multiset_hash"]
                    ),
                    "component_multiplicities": dict(
                        bootstrap_plan["component_multiplicities"]
                    ),
                    "optimization_batching_version": OPTIMIZATION_BATCHING_VERSION,
                    "fixed_group_count": int(bootstrap_plan["fixed_group_count"]),
                    "fixed_group_hash": str(bootstrap_plan["fixed_group_hash"]),
                    "fixed_row_count": int(bootstrap_plan["fixed_row_count"]),
                    "fixed_index_hash": str(bootstrap_plan["fixed_index_hash"]),
                },
                "best_epoch": int(best_epoch),
                "best_validation_loss": float(best_loss),
                "final_state_dict_sha256": str(warm_record["state_dict_sha256"]),
                "epoch_history_sha256": str(warm_record["epoch_history_sha256"]),
                "epochs": epoch_history,
            }
        )
        stage_log(
            "accvp_training",
            "MEMBER_END "
            f"index={member + 1}/{ensemble_size} best_epoch={best_epoch + 1} "
            f"best_validation_loss={best_loss:.8f} elapsed_s={perf_counter() - member_started:.3f}",
        )
    member_hashes = [str(record["state_dict_sha256"]) for record in warm_records]
    if mode == "deployable" and len(set(member_hashes)) != len(member_hashes):
        raise ValueError("formal ACCVP candidate ensemble contains duplicate trained member state dicts")
    stage_log("accvp_training", "CALIBRATION_START")
    calibration = _calibrate(models, calibration_set, torch, config.accvp.calibration)
    stage_log("accvp_training", "CALIBRATION_END")
    training_history: dict[str, Any] = {
        "artifact_kind": "accvp_training_history_v2",
        "schema_version": TRAINING_HISTORY_SCHEMA_VERSION,
        "artifact_generation": ACCVP_ARTIFACT_GENERATION,
        "configured_artifact_generation": configured_artifact_generation,
        "mode": mode,
        "loss_reduction_version": LOSS_REDUCTION_VERSION,
        "loss_component_weights": {
            component: _loss_component_weight(loss_weights, component)
            for component in LOSS_COMPONENTS
        },
        "event_positive_weights": [
            float(value) for value in loss_weights["event_positive_weights"]
        ],
        "response_feature_normalization": {
            "source": str(
                loss_weights.get("response_feature_scales_source", "configured")
            ),
            "values": [float(value) for value in loss_weights["response_feature_scales"]],
            "duplicate_weighting_version": duplicate_weighting["version"],
            "fingerprint_action_group_count": int(
                duplicate_weighting["splits"]["train"][
                    "fingerprint_action_group_count"
                ]
            ),
            "duplicate_row_count": int(
                duplicate_weighting["splits"]["train"]["duplicate_row_count"]
            ),
        },
        "duplicate_weighting": duplicate_weighting,
        "optimization_batching": {
            "version": OPTIMIZATION_BATCHING_VERSION,
            "batch_size_unit": "model_input_fingerprint_x_action_group",
            "groups_are_indivisible": True,
            "group_total_sample_weight": 1.0,
        },
        "calibration_independence_provenance": dict(calibration.provenance),
        "reproducibility": reproducibility_profile,
        "training_config_sha256": stable_hash(dict(training)),
        "members": member_histories,
    }
    if mode == "shadow":
        return _write_predictor_and_calibration(
            output_dir=output_dir,
            config=config,
            dataset_dir=dataset_dir,
            models=models,
            calibration=calibration,
            warm_records=warm_records,
            best_losses=best_losses,
            loss_weights=loss_weights,
            oracle_report=oracle_report,
            training_history=training_history,
            reproducibility_profile=reproducibility_profile,
            duplicate_weighting=duplicate_weighting,
            formal_runtime_contract=formal_runtime_contract,
            torch=torch,
            mode="shadow",
        )
    from safe_rl.accvp.tuning import tune_operating_point

    try:
        stage_log("accvp_training", "OPERATING_POINT_START split=operating_point")
        operating_point = tune_operating_point(models, operating_set, calibration, torch, config.accvp.tuning)
    except OperatingPointAvailabilityError as exc:
        _clear_generation_artifacts(output_dir)
        diagnostic_path = _write_tuning_failure_diagnostics(output_dir, exc)
        stage_log(
            "accvp_training",
            f"OPERATING_POINT_FAILED diagnostics={diagnostic_path} error={str(exc)!r}",
        )
        raise
    stage_log(
        "accvp_training",
        "OPERATING_POINT_END "
        f"conditional_availability={operating_point['selected']['model_conditional_availability']:.6f} "
        f"unconditional_coverage={operating_point['selected']['unconditional_candidate_set_availability']:.6f} "
        f"risk_eligible={operating_point['risk_eligible_decision_count']}/"
        f"{operating_point['effective_decision_count']}",
    )
    operating_point = dict(operating_point)
    operating_point_duplicate_provenance = {
        "split": "operating_point",
        "decision_weighting_version": str(
            operating_point.get("decision_weighting_version", "")
        ),
        "raw_candidate_row_count": int(
            operating_point.get("raw_candidate_row_count", -1)
        ),
        "effective_candidate_row_count": int(
            operating_point.get("effective_candidate_row_count", -1)
        ),
        "effective_decision_count": int(
            operating_point.get("effective_decision_count", -1)
        ),
        "selection_weighting_unit": "model_input_fingerprint_x_raw_action_decision",
        "statistical_independence_claim": False,
        "duplicate_weighting_applied_to_threshold_selection": True,
    }
    if (
        not operating_point_duplicate_provenance["decision_weighting_version"]
        or operating_point_duplicate_provenance["raw_candidate_row_count"] < 0
        or operating_point_duplicate_provenance["effective_candidate_row_count"] < 0
        or operating_point_duplicate_provenance["effective_decision_count"] < 0
    ):
        raise ValueError("operating point is missing duplicate-decision weighting provenance")
    operating_point["duplicate_weighting_provenance"] = operating_point_duplicate_provenance
    training_history["operating_point_independence_provenance"] = (
        operating_point_duplicate_provenance
    )
    checkpoint = _write_predictor_and_calibration(
        output_dir=output_dir,
        config=config,
        dataset_dir=dataset_dir,
        models=models,
        calibration=calibration,
        warm_records=warm_records,
        best_losses=best_losses,
        loss_weights=loss_weights,
        oracle_report=oracle_report,
        training_history=training_history,
        reproducibility_profile=reproducibility_profile,
        duplicate_weighting=duplicate_weighting,
        formal_runtime_contract=formal_runtime_contract,
        torch=torch,
        mode="deployable",
        operating_point=operating_point,
    )
    stage_log(
        "accvp_training",
        f"TRAIN_END checkpoint={checkpoint} elapsed_s={perf_counter() - training_started:.3f}",
    )
    return checkpoint
