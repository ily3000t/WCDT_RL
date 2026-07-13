from __future__ import annotations

import hashlib
import json
import math
import re
from statistics import NormalDist
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


STATISTICS_SCHEMA_VERSION = 1
REPLICATED_STATISTICS_SCHEMA_VERSION = 1
CROSSED_BOOTSTRAP_METHOD = "crossed_training_simulator_paired_percentile_bootstrap"


DEFAULT_CONTINUOUS_METRICS = (
    "episode_reward",
    "min_distance",
    "ttc_p1",
    "drac_p99",
    "completion_time",
    "ego_speed_mean",
    "hard_brake_rate",
)

DEFAULT_BINARY_METRICS = (
    "proxy_collision",
    "safety_violation",
    "geometric_overlap",
    "taper_miss",
    "merge_success",
    "timely_merge_success",
)


def _finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _episode_seed(row: Mapping[str, Any]) -> int | None:
    value = row.get("seed", row.get("episode_seed"))
    return None if value is None else int(value)


def _stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _normalise_sha256(value: Any, *, field: str) -> str:
    digest = str(value or "").strip().lower()
    if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError(f"{field} must be an explicit 64-character SHA-256 digest")
    return digest


def _binary_value(value: Any, *, field: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and int(value) in {0, 1}:
        return bool(value)
    if isinstance(value, (float, np.floating)) and float(value) in {0.0, 1.0}:
        return bool(value)
    raise ValueError(f"{field} must be a boolean or numeric 0/1 value")


def paired_episode_rows(
    left_report: Mapping[str, Any],
    right_report: Mapping[str, Any],
) -> list[dict[str, Any]]:
    def _index(report: Mapping[str, Any], side: str) -> dict[int, Mapping[str, Any]]:
        indexed: dict[int, Mapping[str, Any]] = {}
        for row in report.get("episodes", []):
            seed = _episode_seed(row)
            if seed is None:
                continue
            if seed in indexed:
                raise ValueError(f"{side} report contains duplicate episode seed {seed}")
            indexed[seed] = row
        return indexed

    left_index = _index(left_report, "left")
    right = _index(right_report, "right")
    if set(left_index) != set(right):
        raise ValueError(
            "paired reports must contain identical episode seeds: "
            f"left_only={sorted(set(left_index) - set(right))[:20]} "
            f"right_only={sorted(set(right) - set(left_index))[:20]}"
        )
    training_seed = right_report.get("training_seed", right_report.get("comparative", {}).get("training_seed"))
    rows: list[dict[str, Any]] = []
    for seed in sorted(left_index):
        left = left_index[seed]
        rows.append(
            {
                "seed": seed,
                "training_seed": None if training_seed is None else int(training_seed),
                "left": dict(left),
                "right": dict(right[seed]),
            }
        )
    return rows


def percentile_interval(
    samples: Sequence[float],
    *,
    confidence: float = 0.95,
) -> tuple[float, float]:
    if not samples:
        return float("nan"), float("nan")
    if not 0.0 < float(confidence) < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    alpha = 1.0 - float(confidence)
    values = np.asarray(samples, dtype=np.float64)
    return float(np.quantile(values, alpha / 2.0)), float(np.quantile(values, 1.0 - alpha / 2.0))


def paired_bootstrap_mean_ci(
    deltas: Sequence[float],
    *,
    confidence: float = 0.95,
    replicates: int = 10_000,
    seed: int = 0,
) -> dict[str, Any]:
    if int(replicates) <= 0:
        raise ValueError("bootstrap replicates must be positive")
    if not 0.0 < float(confidence) < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    values = np.asarray([float(value) for value in deltas if np.isfinite(float(value))], dtype=np.float64)
    if values.size == 0:
        return {"count": 0, "mean_delta": None, "ci_low": None, "ci_high": None}
    rng = np.random.default_rng(int(seed))
    bootstrap = np.empty((max(1, int(replicates)),), dtype=np.float64)
    for index in range(bootstrap.size):
        sample = rng.choice(values, size=values.size, replace=True)
        bootstrap[index] = float(np.mean(sample))
    observed = float(np.mean(values))
    normal = NormalDist()
    proportion = float(np.mean(bootstrap < observed))
    epsilon = 0.5 / float(bootstrap.size)
    bias = normal.inv_cdf(min(1.0 - epsilon, max(epsilon, proportion)))
    acceleration = 0.0
    if values.size > 2:
        jackknife = np.asarray(
            [float(np.mean(np.delete(values, index))) for index in range(values.size)],
            dtype=np.float64,
        )
        centre = float(np.mean(jackknife))
        residual = centre - jackknife
        denominator = 6.0 * float(np.sum(residual**2)) ** 1.5
        if denominator > 0.0:
            acceleration = float(np.sum(residual**3) / denominator)
    alpha = 1.0 - float(confidence)

    def _adjust(probability: float) -> float:
        z = normal.inv_cdf(probability)
        denominator = 1.0 - acceleration * (bias + z)
        if abs(denominator) < 1.0e-12:
            return probability
        return min(1.0, max(0.0, normal.cdf(bias + (bias + z) / denominator)))

    low_probability = _adjust(alpha / 2.0)
    high_probability = _adjust(1.0 - alpha / 2.0)
    low = float(np.quantile(bootstrap, min(low_probability, high_probability)))
    high = float(np.quantile(bootstrap, max(low_probability, high_probability)))
    return {
        "count": int(values.size),
        "mean_delta": observed,
        "median_delta": float(np.median(values)),
        "ci_low": low,
        "ci_high": high,
        "confidence": float(confidence),
        "bootstrap_replicates": int(bootstrap.size),
        "bootstrap_seed": int(seed),
        "method": "paired_bca_bootstrap",
        "bca_bias_correction": float(bias),
        "bca_acceleration": float(acceleration),
    }


def hierarchical_paired_bootstrap_mean_ci(
    rows: Sequence[Mapping[str, Any]],
    *,
    delta_key: str = "delta",
    cluster_key: str = "training_seed",
    confidence: float = 0.95,
    replicates: int = 10_000,
    seed: int = 0,
) -> dict[str, Any]:
    if int(replicates) <= 0:
        raise ValueError("bootstrap replicates must be positive")
    if not 0.0 < float(confidence) < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    grouped: dict[str, list[float]] = {}
    for row in rows:
        value = _finite(row.get(delta_key))
        if value is None:
            continue
        cluster = str(row.get(cluster_key, "single"))
        grouped.setdefault(cluster, []).append(value)
    if not grouped:
        return {"count": 0, "cluster_count": 0, "mean_delta": None, "ci_low": None, "ci_high": None}
    clusters = sorted(grouped)
    observed = [value for cluster in clusters for value in grouped[cluster]]
    rng = np.random.default_rng(int(seed))
    bootstrap = np.empty((max(1, int(replicates)),), dtype=np.float64)
    for index in range(bootstrap.size):
        sampled_clusters = rng.choice(clusters, size=len(clusters), replace=True)
        cluster_means = []
        for cluster in sampled_clusters:
            values = np.asarray(grouped[str(cluster)], dtype=np.float64)
            cluster_means.append(float(np.mean(rng.choice(values, size=values.size, replace=True))))
        bootstrap[index] = float(np.mean(cluster_means))
    low, high = percentile_interval(bootstrap.tolist(), confidence=confidence)
    return {
        "count": len(observed),
        "cluster_count": len(clusters),
        "mean_delta": float(np.mean([np.mean(grouped[cluster]) for cluster in clusters])),
        "ci_low": low,
        "ci_high": high,
        "confidence": float(confidence),
        "bootstrap_replicates": int(bootstrap.size),
        "bootstrap_seed": int(seed),
        "method": "hierarchical_paired_percentile_bootstrap",
    }


def crossed_paired_bootstrap_mean_ci(
    delta_matrix: Sequence[Sequence[float]],
    *,
    confidence: float = 0.95,
    replicates: int = 10_000,
    seed: int = 0,
) -> dict[str, Any]:
    """Bootstrap a complete optimizer-seed by simulator-seed paired matrix.

    The two experimental axes are crossed: every optimizer-seed pair must be
    evaluated on the same simulator seeds.  Each bootstrap replicate therefore
    samples optimizer rows and simulator columns independently, then evaluates
    their Cartesian product.  This deliberately differs from the legacy
    hierarchical helper above, which resamples observations independently
    inside each cluster and cannot preserve a shared simulator-seed axis.
    """

    if int(replicates) <= 0:
        raise ValueError("bootstrap replicates must be positive")
    if not 0.0 < float(confidence) < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    values = np.asarray(delta_matrix, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] <= 0 or values.shape[1] <= 0:
        raise ValueError("crossed bootstrap requires a non-empty two-dimensional matrix")
    if not np.isfinite(values).all():
        raise ValueError("crossed bootstrap matrix must contain only finite values")
    training_seed_count, simulator_seed_count = (int(values.shape[0]), int(values.shape[1]))
    rng = np.random.default_rng(int(seed))
    bootstrap = np.empty((int(replicates),), dtype=np.float64)
    for index in range(bootstrap.size):
        sampled_training = rng.integers(0, training_seed_count, size=training_seed_count)
        sampled_simulator = rng.integers(0, simulator_seed_count, size=simulator_seed_count)
        bootstrap[index] = float(np.mean(values[np.ix_(sampled_training, sampled_simulator)]))
    low, high = percentile_interval(bootstrap.tolist(), confidence=confidence)
    return {
        "count": int(values.size),
        "training_seed_count": training_seed_count,
        "simulator_seed_count": simulator_seed_count,
        "mean_delta": float(np.mean(values)),
        "median_delta": float(np.median(values)),
        "ci_low": low,
        "ci_high": high,
        "confidence": float(confidence),
        "bootstrap_replicates": int(bootstrap.size),
        "bootstrap_seed": int(seed),
        "method": CROSSED_BOOTSTRAP_METHOD,
        "training_seed_equal_weighted": True,
        "simulator_seed_equal_weighted": True,
        "delta_matrix_sha256": _stable_hash(values.tolist()),
    }


def crossed_paired_binary_risk_difference(
    left_matrix: Sequence[Sequence[bool]],
    right_matrix: Sequence[Sequence[bool]],
    *,
    confidence: float = 0.95,
    replicates: int = 10_000,
    seed: int = 0,
) -> dict[str, Any]:
    """Estimate a replicated binary risk difference without pooled McNemar.

    Cells sharing either an optimizer seed or simulator seed are correlated, so
    pooling their discordant counts and labelling the result an exact McNemar
    test would be invalid.  Inference here is exclusively the crossed bootstrap
    interval for the right-minus-left risk difference.
    """

    left = np.asarray(left_matrix, dtype=np.bool_)
    right = np.asarray(right_matrix, dtype=np.bool_)
    if left.shape != right.shape:
        raise ValueError("replicated binary matrices must have identical shapes")
    if left.ndim != 2 or left.shape[0] <= 0 or left.shape[1] <= 0:
        raise ValueError("replicated binary risk difference requires non-empty matrices")
    deltas = right.astype(np.float64) - left.astype(np.float64)
    interval = crossed_paired_bootstrap_mean_ci(
        deltas,
        confidence=confidence,
        replicates=replicates,
        seed=seed,
    )
    return {
        "count": int(left.size),
        "training_seed_count": int(left.shape[0]),
        "simulator_seed_count": int(left.shape[1]),
        "left_events": int(np.sum(left)),
        "right_events": int(np.sum(right)),
        "left_rate": float(np.mean(left)),
        "right_rate": float(np.mean(right)),
        "risk_difference": float(interval["mean_delta"]),
        "risk_difference_ci_low": float(interval["ci_low"]),
        "risk_difference_ci_high": float(interval["ci_high"]),
        "confidence": float(confidence),
        "bootstrap_replicates": int(interval["bootstrap_replicates"]),
        "bootstrap_seed": int(seed),
        "method": CROSSED_BOOTSTRAP_METHOD,
        "delta_direction": "right_minus_left",
        "delta_matrix_sha256": str(interval["delta_matrix_sha256"]),
        "pooled_exact_mcnemar_performed": False,
        "pooled_exact_mcnemar_reason": (
            "optimizer-seed by simulator-seed cells are crossed and are not independent"
        ),
    }


def _report_training_seed(report: Mapping[str, Any]) -> int | None:
    candidates = [report.get("training_seed")]
    comparative = report.get("comparative", {})
    if isinstance(comparative, Mapping):
        candidates.append(comparative.get("training_seed"))
    present = [int(value) for value in candidates if value is not None]
    if len(set(present)) > 1:
        raise ValueError(f"report contains conflicting training seeds: {present}")
    return present[0] if present else None


def build_replicated_pair_statistics(
    replicate_pairs: Sequence[Mapping[str, Any]],
    *,
    continuous_metrics: Iterable[str] = DEFAULT_CONTINUOUS_METRICS,
    binary_metrics: Iterable[str] = DEFAULT_BINARY_METRICS,
    confidence: float = 0.95,
    replicates: int = 10_000,
    seed: int = 0,
    require_distinct_checkpoints: bool = True,
) -> dict[str, Any]:
    """Build strict statistics for paired optimizer replicas on shared seeds.

    Every item must provide ``training_seed``, explicit left/right checkpoint
    SHA-256 digests, and left/right Stage5-style reports.  The function refuses
    incomplete or unbalanced matrices rather than silently dropping cells.
    """

    if int(replicates) <= 0:
        raise ValueError("bootstrap replicates must be positive")
    if not 0.0 < float(confidence) < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    if not replicate_pairs:
        raise ValueError("replicated statistics require at least one optimizer-seed pair")

    normalised: list[dict[str, Any]] = []
    seen_training_seeds: set[int] = set()
    for position, pair in enumerate(replicate_pairs):
        if "training_seed" not in pair or pair.get("training_seed") is None:
            raise ValueError(f"replicate pair {position} requires an explicit training_seed")
        training_seed = int(pair["training_seed"])
        if training_seed in seen_training_seeds:
            raise ValueError(f"duplicate replicated training seed {training_seed}")
        seen_training_seeds.add(training_seed)
        left_hash = _normalise_sha256(
            pair.get("left_checkpoint_sha256"),
            field=f"replicate pair {training_seed} left_checkpoint_sha256",
        )
        right_hash = _normalise_sha256(
            pair.get("right_checkpoint_sha256"),
            field=f"replicate pair {training_seed} right_checkpoint_sha256",
        )
        left_report = pair.get("left_report")
        right_report = pair.get("right_report")
        if not isinstance(left_report, Mapping) or not isinstance(right_report, Mapping):
            raise ValueError(f"replicate pair {training_seed} requires left_report and right_report")
        for side, report in (("left", left_report), ("right", right_report)):
            reported_seed = _report_training_seed(report)
            if reported_seed is not None and reported_seed != training_seed:
                raise ValueError(
                    f"{side} report training seed {reported_seed} does not match explicit "
                    f"training seed {training_seed}"
                )
        rows = paired_episode_rows(left_report, right_report)
        if not rows:
            raise ValueError(f"replicate pair {training_seed} contains no paired simulator seeds")
        normalised.append(
            {
                "training_seed": training_seed,
                "left_checkpoint_sha256": left_hash,
                "right_checkpoint_sha256": right_hash,
                "rows": rows,
            }
        )

    normalised.sort(key=lambda item: int(item["training_seed"]))
    if require_distinct_checkpoints:
        for side in ("left", "right"):
            hashes = [str(item[f"{side}_checkpoint_sha256"]) for item in normalised]
            if len(set(hashes)) != len(hashes):
                raise ValueError(
                    f"replicated {side} checkpoints must be distinct across training seeds"
                )

    simulator_seeds = [int(row["seed"]) for row in normalised[0]["rows"]]
    for item in normalised[1:]:
        candidate = [int(row["seed"]) for row in item["rows"]]
        if candidate != simulator_seeds:
            raise ValueError(
                "replicated optimizer pairs must contain the same simulator seeds in a "
                "complete balanced matrix"
            )
    training_seeds = [int(item["training_seed"]) for item in normalised]
    checkpoint_records = [
        {
            "training_seed": int(item["training_seed"]),
            "left_checkpoint_sha256": str(item["left_checkpoint_sha256"]),
            "right_checkpoint_sha256": str(item["right_checkpoint_sha256"]),
        }
        for item in normalised
    ]

    continuous: dict[str, Any] = {}
    binary: dict[str, Any] = {}
    matrix_lineage: dict[str, Any] = {}
    for offset, metric in enumerate(dict.fromkeys(str(value) for value in continuous_metrics)):
        seen_any = any(
            metric in row[side]
            for item in normalised
            for row in item["rows"]
            for side in ("left", "right")
        )
        if not seen_any:
            continue
        matrix: list[list[float]] = []
        for item in normalised:
            values: list[float] = []
            for row in item["rows"]:
                left = _finite(row["left"].get(metric))
                right = _finite(row["right"].get(metric))
                if left is None or right is None:
                    raise ValueError(
                        f"continuous metric {metric!r} is incomplete at training_seed="
                        f"{item['training_seed']} simulator_seed={row['seed']}"
                    )
                values.append(right - left)
            matrix.append(values)
        result = crossed_paired_bootstrap_mean_ci(
            matrix,
            confidence=confidence,
            replicates=replicates,
            seed=int(seed) + offset,
        )
        result["training_seed_means"] = {
            str(training_seed): float(np.mean(matrix[index]))
            for index, training_seed in enumerate(training_seeds)
        }
        continuous[metric] = result
        matrix_lineage[f"continuous:{metric}"] = str(result["delta_matrix_sha256"])

    for offset, metric in enumerate(dict.fromkeys(str(value) for value in binary_metrics)):
        seen_any = any(
            metric in row[side]
            for item in normalised
            for row in item["rows"]
            for side in ("left", "right")
        )
        if not seen_any:
            continue
        left_matrix: list[list[bool]] = []
        right_matrix: list[list[bool]] = []
        for item in normalised:
            left_values: list[bool] = []
            right_values: list[bool] = []
            for row in item["rows"]:
                if metric not in row["left"] or metric not in row["right"]:
                    raise ValueError(
                        f"binary metric {metric!r} is incomplete at training_seed="
                        f"{item['training_seed']} simulator_seed={row['seed']}"
                    )
                left_values.append(
                    _binary_value(
                        row["left"][metric],
                        field=(
                            f"left {metric} at training_seed={item['training_seed']} "
                            f"simulator_seed={row['seed']}"
                        ),
                    )
                )
                right_values.append(
                    _binary_value(
                        row["right"][metric],
                        field=(
                            f"right {metric} at training_seed={item['training_seed']} "
                            f"simulator_seed={row['seed']}"
                        ),
                    )
                )
            left_matrix.append(left_values)
            right_matrix.append(right_values)
        result = crossed_paired_binary_risk_difference(
            left_matrix,
            right_matrix,
            confidence=confidence,
            replicates=replicates,
            seed=int(seed) + 10_000 + offset,
        )
        result["training_seed_risk_differences"] = {
            str(training_seed): float(
                np.mean(
                    np.asarray(right_matrix[index], dtype=np.float64)
                    - np.asarray(left_matrix[index], dtype=np.float64)
                )
            )
            for index, training_seed in enumerate(training_seeds)
        }
        binary[metric] = result
        matrix_lineage[f"binary:{metric}"] = str(result["delta_matrix_sha256"])

    design = {
        "training_seeds": training_seeds,
        "simulator_seeds": simulator_seeds,
        "checkpoint_records": checkpoint_records,
        "matrix_lineage": matrix_lineage,
        "delta_direction": "right_minus_left",
    }
    payload: dict[str, Any] = {
        "schema_version": REPLICATED_STATISTICS_SCHEMA_VERSION,
        "method": CROSSED_BOOTSTRAP_METHOD,
        "balanced_matrix": True,
        "training_seed_count": len(training_seeds),
        "training_seeds": training_seeds,
        "training_seed_sha256": _stable_hash(training_seeds),
        "simulator_seed_count": len(simulator_seeds),
        "simulator_seeds": simulator_seeds,
        "simulator_seed_sha256": _stable_hash(simulator_seeds),
        "cell_count": len(training_seeds) * len(simulator_seeds),
        "checkpoint_records": checkpoint_records,
        "checkpoint_matrix_sha256": _stable_hash(checkpoint_records),
        "paired_matrix_sha256": _stable_hash(matrix_lineage),
        "design_sha256": _stable_hash(design),
        "confidence": float(confidence),
        "bootstrap_replicates": int(replicates),
        "bootstrap_seed": int(seed),
        "delta_direction": "right_minus_left",
        "continuous": continuous,
        "binary": binary,
        "pooled_exact_mcnemar_performed": False,
    }
    payload["statistics_fingerprint"] = _stable_hash(payload)
    return payload


def exact_mcnemar_pvalue(candidate_only: int, baseline_only: int) -> float:
    discordant = int(candidate_only) + int(baseline_only)
    if discordant <= 0:
        return 1.0
    tail = min(int(candidate_only), int(baseline_only))
    probability = _binomial_cdf(tail, discordant, 0.5)
    return float(min(1.0, 2.0 * probability))


def _binomial_cdf(successes: int, trials: int, probability: float) -> float:
    if probability <= 0.0:
        return 1.0
    if probability >= 1.0:
        return 1.0 if successes >= trials else 0.0
    logs = [
        math.lgamma(trials + 1)
        - math.lgamma(value + 1)
        - math.lgamma(trials - value + 1)
        + value * math.log(probability)
        + (trials - value) * math.log1p(-probability)
        for value in range(successes + 1)
    ]
    maximum = max(logs)
    return float(min(1.0, math.exp(maximum) * sum(math.exp(item - maximum) for item in logs)))


def exact_binomial_upper_bound(successes: int, trials: int, *, confidence: float = 0.95) -> float:
    successes = int(successes)
    trials = int(trials)
    if trials <= 0:
        return float("nan")
    if successes < 0 or successes > trials:
        raise ValueError("successes must be between zero and trials")
    if not 0.0 < float(confidence) < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    if successes >= trials:
        return 1.0
    alpha = 1.0 - float(confidence)
    low = successes / trials
    high = 1.0
    for _ in range(80):
        middle = (low + high) / 2.0
        if _binomial_cdf(successes, trials, middle) > alpha:
            low = middle
        else:
            high = middle
    return float(high)


def paired_binary_statistics(
    left: Sequence[bool],
    right: Sequence[bool],
    *,
    confidence: float = 0.95,
    replicates: int = 10_000,
    seed: int = 0,
) -> dict[str, Any]:
    if len(left) != len(right):
        raise ValueError("paired binary samples must have equal length")
    baseline = np.asarray([bool(value) for value in left], dtype=np.int8)
    candidate = np.asarray([bool(value) for value in right], dtype=np.int8)
    count = int(baseline.size)
    if count == 0:
        return {"count": 0, "risk_difference": None}
    candidate_only = int(np.sum((candidate == 1) & (baseline == 0)))
    baseline_only = int(np.sum((candidate == 0) & (baseline == 1)))
    deltas = (candidate - baseline).astype(np.float64)
    ci = paired_bootstrap_mean_ci(
        deltas.tolist(),
        confidence=confidence,
        replicates=replicates,
        seed=seed,
    )
    candidate_events = int(np.sum(candidate))
    return {
        "count": count,
        "baseline_events": int(np.sum(baseline)),
        "candidate_events": candidate_events,
        "candidate_only_events": candidate_only,
        "baseline_only_events": baseline_only,
        "baseline_rate": float(np.mean(baseline)),
        "candidate_rate": float(np.mean(candidate)),
        "candidate_rate_upper_bound": exact_binomial_upper_bound(
            candidate_events, count, confidence=confidence
        ),
        "risk_difference": float(np.mean(deltas)),
        "risk_difference_ci_low": ci["ci_low"],
        "risk_difference_ci_high": ci["ci_high"],
        "mcnemar_exact_pvalue": exact_mcnemar_pvalue(candidate_only, baseline_only),
        "confidence": float(confidence),
    }


def holm_adjust(pvalues: Mapping[str, float]) -> dict[str, float]:
    ordered = sorted((str(name), float(value)) for name, value in pvalues.items())
    ordered.sort(key=lambda item: item[1])
    count = len(ordered)
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, (count - rank) * value))
        adjusted[name] = running
    return adjusted


def build_pair_statistics(
    left_report: Mapping[str, Any],
    right_report: Mapping[str, Any],
    *,
    continuous_metrics: Iterable[str] = DEFAULT_CONTINUOUS_METRICS,
    binary_metrics: Iterable[str] = DEFAULT_BINARY_METRICS,
    confidence: float = 0.95,
    replicates: int = 10_000,
    seed: int = 0,
) -> dict[str, Any]:
    if int(replicates) <= 0:
        raise ValueError("bootstrap replicates must be positive")
    if not 0.0 < float(confidence) < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    rows = paired_episode_rows(left_report, right_report)
    continuous: dict[str, Any] = {}
    binary: dict[str, Any] = {}
    for offset, metric in enumerate(continuous_metrics):
        deltas = []
        for row in rows:
            left = _finite(row["left"].get(metric))
            right = _finite(row["right"].get(metric))
            if left is not None and right is not None:
                deltas.append(right - left)
        if deltas:
            continuous[str(metric)] = paired_bootstrap_mean_ci(
                deltas,
                confidence=confidence,
                replicates=replicates,
                seed=int(seed) + offset,
            )
    pvalues: dict[str, float] = {}
    for offset, metric in enumerate(binary_metrics):
        usable = [row for row in rows if metric in row["left"] and metric in row["right"]]
        if not usable:
            continue
        result = paired_binary_statistics(
            [bool(row["left"][metric]) for row in usable],
            [bool(row["right"][metric]) for row in usable],
            confidence=confidence,
            replicates=replicates,
            seed=int(seed) + 10_000 + offset,
        )
        binary[str(metric)] = result
        pvalues[str(metric)] = float(result["mcnemar_exact_pvalue"])
    adjusted = holm_adjust(pvalues)
    for metric, value in adjusted.items():
        binary[metric]["mcnemar_holm_pvalue"] = value
    return {
        "schema_version": STATISTICS_SCHEMA_VERSION,
        "paired_seed_count": len(rows),
        "paired_seed_sha256": __import__("hashlib").sha256(
            json_seed_list(row["seed"] for row in rows).encode("utf-8")
        ).hexdigest(),
        "confidence": float(confidence),
        "bootstrap_replicates": int(replicates),
        "bootstrap_seed": int(seed),
        "delta_direction": "right_minus_left",
        "continuous": continuous,
        "binary": binary,
    }


def json_seed_list(seeds: Iterable[int]) -> str:
    return "[" + ",".join(str(int(seed)) for seed in sorted(seeds)) + "]"
