from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from safe_rl.utils.config import REPO_ROOT


EPISODE_METRICS = (
    "episode_reward",
    "merge_success",
    "collision",
    "proxy_collision",
    "safety_violation",
    "taper_miss",
    "completion_time",
    "min_distance",
    "ttc_p1",
    "drac_p99",
)


def _experiment_root(base_run_id: str, experiment_id: str) -> Path:
    return REPO_ROOT / "safe_rl_output" / "runs" / base_run_id / "comparative_eval" / experiment_id


def _mean_metrics(episodes: list[dict[str, Any]]) -> dict[str, float]:
    result: dict[str, float] = {"scenario_seed_count": float(len(episodes))}
    for metric in EPISODE_METRICS:
        values = [float(row.get(metric, 0.0)) for row in episodes]
        result[metric] = float(np.mean(values)) if values else 0.0
    return result


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted(set().union(*(row.keys() for row in rows))) if rows else ["status"]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _policy_rows(groups: dict[str, dict[str, Any]], variant: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    headline_inputs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for name, report in groups.items():
        metadata = dict(report.get("comparative", {}) or {})
        if str(metadata.get("evaluation_variant", "")) != variant:
            continue
        method = str(metadata.get("method", name))
        seed = metadata.get("training_seed")
        row = {
            "group": name,
            "method": method,
            "training_seed": seed,
            "deterministic": seed is None,
            **_mean_metrics(list(report.get("episodes", []) or [])),
        }
        rows.append(row)
        headline_inputs[method].append(row)
    headline: list[dict[str, Any]] = []
    for method, method_rows in sorted(headline_inputs.items()):
        result: dict[str, Any] = {
            "method": method,
            "training_trial_count": len(method_rows),
            "deterministic": bool(all(row["deterministic"] for row in method_rows)),
        }
        for metric in ("episode_reward", "merge_success", "collision", "proxy_collision", "safety_violation", "taper_miss", "completion_time", "min_distance", "ttc_p1", "drac_p99"):
            values = np.asarray([float(row[metric]) for row in method_rows], dtype=np.float64)
            result[f"{metric}_mean"] = float(np.mean(values))
            result[f"{metric}_std"] = 0.0 if len(values) <= 1 else float(np.std(values, ddof=1))
        headline.append(result)
    return rows, headline


def _shield_deltas(groups: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    indexed: dict[tuple[str, Any], dict[str, dict[str, Any]]] = defaultdict(dict)
    for name, report in groups.items():
        metadata = dict(report.get("comparative", {}) or {})
        variant = str(metadata.get("evaluation_variant", ""))
        if variant not in {"policy", "shield"} or metadata.get("training_seed") is None:
            continue
        indexed[(str(metadata.get("method", name)), metadata.get("training_seed"))][variant] = report
    rows: list[dict[str, Any]] = []
    for (method, seed), pair in sorted(indexed.items(), key=lambda item: (item[0][0], int(item[0][1]))):
        if "policy" not in pair or "shield" not in pair:
            continue
        off = {int(row["seed"]): row for row in pair["policy"].get("episodes", [])}
        on = {int(row["seed"]): row for row in pair["shield"].get("episodes", [])}
        common = sorted(set(off) & set(on))
        if not common:
            continue
        row: dict[str, Any] = {
            "method": method,
            "training_seed": int(seed),
            "scenario_seed_count": len(common),
        }
        for metric in EPISODE_METRICS:
            values = [float(on[value].get(metric, 0.0)) - float(off[value].get(metric, 0.0)) for value in common]
            row[f"{metric}_delta"] = float(np.mean(values))
        rows.append(row)
    return rows


def run(*, base_run_id: str, experiment_id: str, stage5_report: str | Path | None = None) -> Path:
    root = _experiment_root(base_run_id, experiment_id)
    report_path = Path(stage5_report) if stage5_report else root / "stage5" / "formal_paired_eval_report.json"
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    groups = dict(payload.get("groups", {}) or {})
    provenance_path = root / "manifests" / "input_provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8")) if provenance_path.exists() else None
    policy_by_seed, policy_headline = _policy_rows(groups, "policy")
    shield_by_seed, shield_headline = _policy_rows(groups, "shield")
    shield_ablation = _shield_deltas(groups)
    high_impact_by_seed, high_impact_headline = _policy_rows(groups, "high_impact")
    output_dir = root / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    policy_table = {"by_training_seed": policy_by_seed, "headline": policy_headline}
    shield_table = {
        "by_training_seed": shield_by_seed,
        "headline": shield_headline,
        "paired_deltas": shield_ablation,
    }
    high_impact_table = {
        "status": "not_run" if not high_impact_by_seed else "available",
        "by_training_seed": high_impact_by_seed,
        "headline": high_impact_headline,
    }
    output = {
        "base_run_id": base_run_id,
        "experiment_id": experiment_id,
        "stage5_report": str(report_path),
        "input_provenance": provenance,
        "policy_comparison": policy_table,
        "shield_ablation": shield_table,
        "high_impact_controller_ablation": high_impact_table,
    }
    (output_dir / "comparative_report.json").write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_csv(output_dir / "policy_comparison.csv", [*policy_by_seed, *policy_headline])
    _write_csv(output_dir / "shield_ablation.csv", [*shield_by_seed, *shield_ablation, *shield_headline])
    _write_csv(output_dir / "high_impact_controller_ablation.csv", [*high_impact_by_seed, *high_impact_headline])
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Create explicit WcDT comparative tables.")
    parser.add_argument("--base-run-id", required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--stage5-report")
    args = parser.parse_args()
    print(run(base_run_id=args.base_run_id, experiment_id=args.experiment_id, stage5_report=args.stage5_report))


if __name__ == "__main__":
    main()
