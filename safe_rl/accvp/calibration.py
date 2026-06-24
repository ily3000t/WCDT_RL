from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist
from typing import Any, Iterable

import numpy as np


def brier_score(probabilities: np.ndarray, labels: np.ndarray) -> float:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    return float(np.mean((probabilities - labels) ** 2)) if probabilities.size else float("nan")


def expected_calibration_error(probabilities: np.ndarray, labels: np.ndarray, bins: int = 10) -> float:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    if not probabilities.size:
        return float("nan")
    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    error = 0.0
    for index in range(int(bins)):
        mask = (probabilities >= edges[index]) & (
            probabilities <= edges[index + 1] if index == bins - 1 else probabilities < edges[index + 1]
        )
        if not np.any(mask):
            continue
        error += float(mask.mean()) * abs(float(probabilities[mask].mean()) - float(labels[mask].mean()))
    return float(error)


def _wilson_bounds(successes: np.ndarray, counts: np.ndarray, z: float) -> tuple[np.ndarray, np.ndarray]:
    counts = np.maximum(np.asarray(counts, dtype=np.float64), 1.0)
    probability = np.asarray(successes, dtype=np.float64) / counts
    denominator = 1.0 + z**2 / counts
    centre = (probability + z**2 / (2.0 * counts)) / denominator
    radius = z * np.sqrt((probability * (1.0 - probability) + z**2 / (4.0 * counts)) / counts) / denominator
    return np.clip(centre - radius, 0.0, 1.0), np.clip(centre + radius, 0.0, 1.0)


@dataclass
class OneSidedBinnedCalibrator:
    """Score-bin conservative estimates; not a per-state safety guarantee."""

    edges: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    nominal_alpha: float = 0.05
    effective_alpha: float = 0.05
    z_value: float = 1.6448536269514722

    @classmethod
    def fit(
        cls,
        scores: Iterable[float],
        labels: Iterable[float],
        bins: int = 20,
        nominal_alpha: float = 0.05,
        bonferroni_family_size: int = 3,
    ) -> "OneSidedBinnedCalibrator":
        values = np.clip(np.asarray(list(scores), dtype=np.float64), 0.0, 1.0)
        targets = np.asarray(list(labels), dtype=np.float64)
        if values.shape != targets.shape or not values.size:
            raise ValueError("calibration requires equally sized non-empty scores and labels")
        edges = np.linspace(0.0, 1.0, int(bins) + 1)
        indices = np.clip(np.digitize(values, edges, right=False) - 1, 0, bins - 1)
        counts = np.bincount(indices, minlength=bins)
        successes = np.bincount(indices, weights=targets, minlength=bins)
        effective_alpha = float(nominal_alpha) / max(1, int(bins) * int(bonferroni_family_size))
        z_value = NormalDist().inv_cdf(1.0 - effective_alpha)
        lower, upper = _wilson_bounds(successes, counts, z_value)
        # No observed event rate can justify a safety/viability claim for an
        # empty score bin. Preserve the conservative no-information bounds.
        lower[counts == 0] = 0.0
        upper[counts == 0] = 1.0
        return cls(
            edges=edges,
            lower=lower,
            upper=upper,
            nominal_alpha=float(nominal_alpha),
            effective_alpha=effective_alpha,
            z_value=float(z_value),
        )

    def transform_upper(self, scores: Iterable[float]) -> np.ndarray:
        values = np.clip(np.asarray(list(scores), dtype=np.float64), 0.0, 1.0)
        index = np.clip(np.digitize(values, self.edges, right=False) - 1, 0, len(self.upper) - 1)
        return self.upper[index]

    def transform_lower(self, scores: Iterable[float]) -> np.ndarray:
        values = np.clip(np.asarray(list(scores), dtype=np.float64), 0.0, 1.0)
        index = np.clip(np.digitize(values, self.edges, right=False) - 1, 0, len(self.lower) - 1)
        return self.lower[index]

    def to_dict(self) -> dict[str, Any]:
        return {
            "edges": self.edges.tolist(),
            "lower": self.lower.tolist(),
            "upper": self.upper.tolist(),
            "nominal_alpha": self.nominal_alpha,
            "effective_alpha": self.effective_alpha,
            "z_value": self.z_value,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "OneSidedBinnedCalibrator":
        return cls(
            edges=np.asarray(value["edges"], dtype=np.float64),
            lower=np.asarray(value["lower"], dtype=np.float64),
            upper=np.asarray(value["upper"], dtype=np.float64),
            nominal_alpha=float(value.get("nominal_alpha", 0.05)),
            effective_alpha=float(value.get("effective_alpha", 0.05)),
            z_value=float(value.get("z_value", 1.6448536269514722)),
        )


@dataclass
class CalibrationBundle:
    proxy_collision: OneSidedBinnedCalibrator
    safety_violation: OneSidedBinnedCalibrator
    merge_viability: OneSidedBinnedCalibrator
    provenance: dict[str, Any]

    def score(self, raw: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return {
            "pU_proxy_collision": self.proxy_collision.transform_upper(raw["p_proxy_collision"]),
            "pU_safety_violation": self.safety_violation.transform_upper(raw["p_safety_violation"]),
            "pL_merge_before_taper": self.merge_viability.transform_lower(raw["p_merge_before_taper"]),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "proxy_collision": self.proxy_collision.to_dict(),
            "safety_violation": self.safety_violation.to_dict(),
            "merge_viability": self.merge_viability.to_dict(),
            "provenance": dict(self.provenance),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "CalibrationBundle":
        return cls(
            proxy_collision=OneSidedBinnedCalibrator.from_dict(value["proxy_collision"]),
            safety_violation=OneSidedBinnedCalibrator.from_dict(value["safety_violation"]),
            merge_viability=OneSidedBinnedCalibrator.from_dict(value["merge_viability"]),
            provenance=dict(value.get("provenance", {})),
        )


def _event_bound_summary(labels: np.ndarray, bounds: np.ndarray, *, upper: bool) -> dict[str, float | bool]:
    if not len(labels):
        return {"count": 0.0, "event_rate": float("nan"), "rate_ci_lower": float("nan"), "rate_ci_upper": float("nan"), "mean_bound": float("nan"), "marginal_bound_pass": False}
    lower, upper_ci = _wilson_bounds(
        np.asarray([float(labels.sum())]),
        np.asarray([float(len(labels))]),
        NormalDist().inv_cdf(0.975),
    )
    mean_bound = float(np.mean(bounds))
    return {
        "count": float(len(labels)),
        "event_rate": float(np.mean(labels)),
        "rate_ci_lower": float(lower[0]),
        "rate_ci_upper": float(upper_ci[0]),
        "mean_bound": mean_bound,
        "marginal_bound_pass": bool(float(upper_ci[0]) <= mean_bound) if upper else bool(float(lower[0]) >= mean_bound),
    }


def selected_action_metrics(records: Iterable[dict[str, Any]], *, total_decision_count: int | None = None) -> dict[str, Any]:
    """Decision-level held-out diagnostics; never interpret binary outcomes as pointwise coverage."""

    rows = [row for row in records if bool(row.get("selected", False))]
    if not rows:
        return {
            "selected_count": 0.0,
            "candidate_set_availability": 0.0 if total_decision_count is not None else float("nan"),
        }
    proxy = np.asarray([float(row["p_proxy_collision"]) for row in rows])
    proxy_y = np.asarray([float(row["proxy_collision"]) for row in rows])
    safety = np.asarray([float(row["p_safety_violation"]) for row in rows])
    safety_y = np.asarray([float(row["safety_violation"]) for row in rows])
    viability = np.asarray([float(row["p_merge_before_taper"]) for row in rows])
    viability_y = np.asarray([float(row["merge_before_taper"]) for row in rows])
    proxy_upper = np.asarray([float(row.get("pU_proxy_collision", row["p_proxy_collision"])) for row in rows])
    safety_upper = np.asarray([float(row.get("pU_safety_violation", row["p_safety_violation"])) for row in rows])
    viability_lower = np.asarray([float(row.get("pL_merge_before_taper", row["p_merge_before_taper"])) for row in rows])
    viability_observed = np.asarray([bool(row.get("merge_observed", True)) for row in rows])
    decisions = {str(row.get("root_id", index)) for index, row in enumerate(rows)}
    available = {str(row.get("root_id", index)) for index, row in enumerate(rows) if bool(row.get("candidate_set_available", False))}
    return {
        "selected_count": float(len(rows)),
        "candidate_set_availability": float(len(available) / max(1, int(total_decision_count or len(decisions)))),
        "proxy_collision": _event_bound_summary(proxy_y, proxy_upper, upper=True),
        "safety_violation": _event_bound_summary(safety_y, safety_upper, upper=True),
        "merge_before_taper": _event_bound_summary(
            viability_y[viability_observed], viability_lower[viability_observed], upper=False
        ),
        "post_selection_safety_brier": brier_score(proxy, proxy_y),
        "post_selection_safety_ece": expected_calibration_error(proxy, proxy_y),
        "post_selection_safety_violation_brier": brier_score(safety, safety_y),
        "post_selection_safety_violation_ece": expected_calibration_error(safety, safety_y),
        "post_selection_viability_brier": brier_score(viability, viability_y),
        "post_selection_viability_ece": expected_calibration_error(viability, viability_y),
    }
