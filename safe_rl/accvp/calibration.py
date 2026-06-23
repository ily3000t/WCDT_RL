from __future__ import annotations

from dataclasses import dataclass
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


def _wilson_bounds(successes: np.ndarray, counts: np.ndarray, z: float = 1.96) -> tuple[np.ndarray, np.ndarray]:
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

    @classmethod
    def fit(cls, scores: Iterable[float], labels: Iterable[float], bins: int = 20) -> "OneSidedBinnedCalibrator":
        values = np.clip(np.asarray(list(scores), dtype=np.float64), 0.0, 1.0)
        targets = np.asarray(list(labels), dtype=np.float64)
        if values.shape != targets.shape or not values.size:
            raise ValueError("calibration requires equally sized non-empty scores and labels")
        edges = np.linspace(0.0, 1.0, int(bins) + 1)
        indices = np.clip(np.digitize(values, edges, right=False) - 1, 0, bins - 1)
        counts = np.bincount(indices, minlength=bins)
        successes = np.bincount(indices, weights=targets, minlength=bins)
        lower, upper = _wilson_bounds(successes, counts)
        # Empty bins borrow the empirical ordering of their nearest populated bin.
        global_rate = float(targets.mean())
        lower[counts == 0] = global_rate
        upper[counts == 0] = global_rate
        return cls(edges=edges, lower=lower, upper=upper)

    def transform_upper(self, scores: Iterable[float]) -> np.ndarray:
        values = np.clip(np.asarray(list(scores), dtype=np.float64), 0.0, 1.0)
        index = np.clip(np.digitize(values, self.edges, right=False) - 1, 0, len(self.upper) - 1)
        return self.upper[index]

    def transform_lower(self, scores: Iterable[float]) -> np.ndarray:
        values = np.clip(np.asarray(list(scores), dtype=np.float64), 0.0, 1.0)
        index = np.clip(np.digitize(values, self.edges, right=False) - 1, 0, len(self.lower) - 1)
        return self.lower[index]

    def to_dict(self) -> dict[str, Any]:
        return {"edges": self.edges.tolist(), "lower": self.lower.tolist(), "upper": self.upper.tolist()}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "OneSidedBinnedCalibrator":
        return cls(
            edges=np.asarray(value["edges"], dtype=np.float64),
            lower=np.asarray(value["lower"], dtype=np.float64),
            upper=np.asarray(value["upper"], dtype=np.float64),
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


def selected_action_metrics(records: Iterable[dict[str, Any]]) -> dict[str, float]:
    """Evaluate the frozen controller's chosen action, not independent branches."""

    rows = [row for row in records if bool(row.get("selected", False))]
    if not rows:
        return {"selected_count": 0.0, "candidate_set_availability": float("nan")}
    proxy = np.asarray([float(row["p_proxy_collision"]) for row in rows])
    proxy_y = np.asarray([float(row["proxy_collision"]) for row in rows])
    viability = np.asarray([float(row["p_merge_before_taper"]) for row in rows])
    viability_y = np.asarray([float(row["merge_before_taper"]) for row in rows])
    proxy_upper = np.asarray([float(row.get("pU_proxy_collision", row["p_proxy_collision"])) for row in rows])
    viability_lower = np.asarray([float(row.get("pL_merge_before_taper", row["p_merge_before_taper"])) for row in rows])
    viability_observed = np.asarray([bool(row.get("merge_observed", True)) for row in rows])
    decisions = {str(row.get("root_id", index)) for index, row in enumerate(rows)}
    available = {str(row.get("root_id", index)) for index, row in enumerate(rows) if bool(row.get("candidate_set_available", False))}
    return {
        "selected_count": float(len(rows)),
        "selected_action_safety_coverage": float(np.mean(proxy_y <= proxy_upper)) if len(rows) else float("nan"),
        "selected_action_viability_coverage": (
            float(np.mean(viability_y[viability_observed] >= viability_lower[viability_observed]))
            if np.any(viability_observed)
            else float("nan")
        ),
        "candidate_set_availability": float(len(available) / max(1, len(decisions))),
        "post_selection_safety_brier": brier_score(proxy, proxy_y),
        "post_selection_safety_ece": expected_calibration_error(proxy, proxy_y),
        "post_selection_viability_brier": brier_score(viability, viability_y),
        "post_selection_viability_ece": expected_calibration_error(viability, viability_y),
    }
