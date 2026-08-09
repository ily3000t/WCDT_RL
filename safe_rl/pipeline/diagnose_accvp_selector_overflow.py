from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from safe_rl.accvp.contracts.schema import (
    file_sha256,
    read_json,
    stable_hash,
    write_json_atomic,
)
from safe_rl.prediction.actor_selector import actor_relevance_config
from safe_rl.utils.config import REPO_ROOT, load_config


DIAGNOSTIC_KIND = "accvp_selector_overflow_shadow_diagnostic_v1"
DIAGNOSTIC_VERSION = "lane_aware_local_conflict_shadow_v1"
RUNTIME_REPORT_KIND = "accvp_runtime_benchmark_v1"


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (REPO_ROOT / value).resolve()


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if hasattr(value, "items"):
        return {str(key): _plain(item) for key, item in value.items()}
    return value


def validate_runtime_report(path: str | Path) -> tuple[Path, dict[str, Any]]:
    source = _resolve(path)
    report = read_json(source)
    if str(report.get("artifact_kind", "")) != RUNTIME_REPORT_KIND:
        raise ValueError(f"unsupported runtime report: {source}")
    declared = str(report.get("report_fingerprint", ""))
    expected = stable_hash(
        {key: value for key, value in report.items() if key != "report_fingerprint"}
    )
    if not declared or declared != expected:
        raise ValueError(f"runtime report fingerprint mismatch: {source}")
    return source, report


def _scenario_values(cfg: Any, key: str) -> list[str]:
    return [str(value) for value in list(cfg.scenario.get(key, []) or [])]


def _auxiliary_lane(cfg: Any, edge_id: str) -> int:
    by_edge = _plain(cfg.scenario.get("auxiliary_lane_by_edge", {}) or {})
    if edge_id in by_edge:
        return int(by_edge[edge_id])
    return int(cfg.scenario.get("auxiliary_lane", 0))


def _lane_aware_local_actor(cfg: Any, actor: Mapping[str, Any]) -> bool:
    edge_id = str(actor.get("edge_id", ""))
    lane_index = int(actor.get("lane_index", -1))
    if edge_id in _scenario_values(cfg, "ramp_edges"):
        return True
    return bool(
        edge_id in _scenario_values(cfg, "auxiliary_edges")
        and lane_index == _auxiliary_lane(cfg, edge_id)
    )


def _protected_reason(actor: Mapping[str, Any]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if str(actor.get("role", "")) in {"target_front", "target_rear"}:
        reasons.append("mandatory_target_front_rear")
    if bool(actor.get("candidate_conflict_eligible", False)):
        reasons.append("candidate_union_conflict")
    if bool(actor.get("nearest_candidate_conflict", False)):
        reasons.append("nearest_candidate_conflict")
    if "lowest_ttc" in [str(value) for value in actor.get("trigger_reasons", []) or []]:
        reasons.append("lowest_conflict_ttc")
    return bool(reasons), reasons


def _lane_aware_critical(
    cfg: Any,
    actor: Mapping[str, Any],
    example: Mapping[str, Any],
    *,
    settings: Mapping[str, Any],
) -> tuple[bool, list[str]]:
    protected, reasons = _protected_reason(actor)
    if protected:
        return True, reasons
    local = bool(
        _lane_aware_local_actor(cfg, actor)
        and float(actor.get("gap_m", float("inf")))
        <= float(settings["local_actor_distance"])
        and float(example.get("taper_distance_m", float("inf")))
        <= float(settings["critical_taper_distance"])
    )
    if local:
        return True, ["lane_aware_ramp_or_auxiliary_local"]
    return False, []


def _overflow_examples(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for episode in list(report.get("episodes", []) or []):
        for value in list(
            episode.get("accvp_table_critical_actor_overflow_examples", []) or []
        ):
            item = dict(value)
            item.setdefault("episode_seed", int(episode.get("episode_seed", -1)))
            examples.append(item)
    return examples


def diagnose_report(
    report: Mapping[str, Any],
    cfg: Any,
    *,
    capacities: Sequence[int] = (6, 8),
) -> dict[str, Any]:
    requested_capacities = sorted(set(int(value) for value in capacities))
    if not requested_capacities or requested_capacities[0] <= 0:
        raise ValueError("capacities must contain positive unique integers")
    settings = actor_relevance_config(cfg, selector_scope="accvp")
    examples = _overflow_examples(report)
    reported_overflow_count = int(
        report.get("metrics", {}).get("accvp_table_critical_actor_overflow_count", 0)
    )
    rows: list[dict[str, Any]] = []
    removed_signatures: Counter[str] = Counter()
    protected_coverage_complete = True
    reconstruction_complete = True
    for example in examples:
        actors = [dict(value) for value in list(example.get("critical_actors", []) or [])]
        original_ids = [str(actor.get("vehicle_id", "")) for actor in actors]
        reported_count = int(example.get("critical_count", -1))
        reconstruction_complete = bool(
            reconstruction_complete
            and reported_count == len(actors)
            and set(example.get("dropped_critical_ids", []) or []).issubset(original_ids)
        )
        retained: list[dict[str, Any]] = []
        removed: list[dict[str, Any]] = []
        protected_ids: list[str] = []
        for actor in actors:
            vehicle_id = str(actor.get("vehicle_id", ""))
            protected, protected_reasons = _protected_reason(actor)
            if protected:
                protected_ids.append(vehicle_id)
            keep, keep_reasons = _lane_aware_critical(
                cfg,
                actor,
                example,
                settings=settings,
            )
            record = {
                "vehicle_id": vehicle_id,
                "role": str(actor.get("role", "")),
                "edge_id": str(actor.get("edge_id", "")),
                "lane_index": int(actor.get("lane_index", -1)),
                "gap_m": float(actor.get("gap_m", float("inf"))),
                "candidate_conflict_eligible": bool(
                    actor.get("candidate_conflict_eligible", False)
                ),
                "nearest_candidate_conflict": bool(
                    actor.get("nearest_candidate_conflict", False)
                ),
                "original_trigger_reasons": [
                    str(value) for value in actor.get("trigger_reasons", []) or []
                ],
                "lane_aware_retain_reasons": keep_reasons,
                "protected_reasons": protected_reasons,
            }
            if keep:
                retained.append(record)
            else:
                removed.append(record)
                removed_signatures[
                    "|".join(
                        [
                            record["role"],
                            record["edge_id"],
                            f"lane={record['lane_index']}",
                            "conflict="
                            + str(record["candidate_conflict_eligible"]).lower(),
                        ]
                    )
                ] += 1
        retained_ids = {item["vehicle_id"] for item in retained}
        protected_coverage_complete = bool(
            protected_coverage_complete
            and set(protected_ids).issubset(retained_ids)
        )
        capacity_results = {
            str(capacity): {
                "critical_count": len(retained),
                "overflow": len(retained) > capacity,
                "overflow_by": max(0, len(retained) - capacity),
            }
            for capacity in requested_capacities
        }
        rows.append(
            {
                "episode_seed": int(example.get("episode_seed", -1)),
                "optimizer_seed": int(example.get("optimizer_seed", -1)),
                "decision_index": int(example.get("decision_index", -1)),
                "taper_distance_m": float(example.get("taper_distance_m", 0.0)),
                "original_capacity": int(example.get("capacity", -1)),
                "original_critical_count": reported_count,
                "original_dropped_critical_ids": [
                    str(value)
                    for value in example.get("dropped_critical_ids", []) or []
                ],
                "lane_aware_critical_count": len(retained),
                "lane_aware_critical_ids": [item["vehicle_id"] for item in retained],
                "lane_aware_removed": removed,
                "protected_actor_ids": protected_ids,
                "protected_coverage_complete": set(protected_ids).issubset(
                    retained_ids
                ),
                "capacity_results": capacity_results,
            }
        )
    overflow_by_capacity = {
        str(capacity): sum(
            bool(row["capacity_results"][str(capacity)]["overflow"])
            for row in rows
        )
        for capacity in requested_capacities
    }
    examples_complete = bool(len(examples) == reported_overflow_count)
    capacity8_resolved = bool(
        "8" in overflow_by_capacity and overflow_by_capacity["8"] == 0
    )
    supports_overclassification = bool(
        examples
        and examples_complete
        and reconstruction_complete
        and protected_coverage_complete
        and capacity8_resolved
        and any(row["lane_aware_removed"] for row in rows)
    )
    return {
        "diagnostic_version": DIAGNOSTIC_VERSION,
        "selector_scope": "accvp",
        "frozen_selector_version": str(settings["version"]),
        "hypothesis": (
            "restrict auxiliary-local critical promotion to configured auxiliary "
            "lane; keep ramp-local, mandatory target front/rear, candidate-union "
            "conflict, nearest candidate conflict, and lowest conflict TTC"
        ),
        "reported_overflow_count": reported_overflow_count,
        "observed_example_count": len(examples),
        "examples_complete": examples_complete,
        "reconstruction_complete": reconstruction_complete,
        "protected_coverage_complete": protected_coverage_complete,
        "overflow_example_episode_seeds": sorted(
            set(int(row["episode_seed"]) for row in rows)
        ),
        "overflow_example_optimizer_seeds": sorted(
            set(int(row["optimizer_seed"]) for row in rows)
        ),
        "original_critical_count_histogram": dict(
            sorted(Counter(str(row["original_critical_count"]) for row in rows).items())
        ),
        "lane_aware_critical_count_histogram": dict(
            sorted(Counter(str(row["lane_aware_critical_count"]) for row in rows).items())
        ),
        "overflow_example_count_by_capacity": overflow_by_capacity,
        "removed_actor_instance_count": sum(
            len(row["lane_aware_removed"]) for row in rows
        ),
        "removed_actor_signature_counts": dict(sorted(removed_signatures.items())),
        "supports_edge_wide_local_overclassification": supports_overclassification,
        "true_capacity_pressure_remains_at_capacity8": not capacity8_resolved,
        "rows": rows,
        "limitations": [
            "offline shadow over recorded overflow telemetry; SUMO was not rerun",
            "covers only recorded overflow decisions, not every non-overflow decision",
            "cannot authorize a selector/data-contract change or artifact promotion",
            "a new selector version still requires selector-only full replay audit",
        ],
        "recommended_next_step": (
            "run selector-only full replay audit for a new lane-aware selector version"
            if supports_overclassification
            else "retain the blocked gate and redesign actor capacity/representation"
        ),
    }


def run(
    *,
    runtime_report: str | Path,
    config: str | Path | None,
    capacities: Sequence[int],
    output: str | Path,
) -> Path:
    source, report = validate_runtime_report(runtime_report)
    config_path = _resolve(config or str(report.get("config", "")))
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    cfg = load_config(config_path)
    diagnostic = diagnose_report(report, cfg, capacities=capacities)
    payload = {
        "artifact_kind": DIAGNOSTIC_KIND,
        "schema_version": 1,
        "formal_evidence": False,
        "diagnostic_only": True,
        "source_runtime_report": str(source),
        "source_runtime_report_sha256": file_sha256(source),
        "source_runtime_report_fingerprint": str(report["report_fingerprint"]),
        "config": str(config_path),
        "config_file_sha256": file_sha256(config_path),
        **diagnostic,
    }
    payload["diagnostic_fingerprint"] = stable_hash(payload)
    destination = _resolve(output)
    if destination.exists():
        if read_json(destination) != payload:
            raise FileExistsError(
                f"selector diagnostic output exists with different content: {destination}"
            )
        action = "reuse_identical"
    else:
        write_json_atomic(destination, payload)
        action = "write_new"
    print(
        "[selector_overflow_diagnostic] "
        f"action={action} output={destination} "
        "supports_overclassification="
        f"{payload['supports_edge_wide_local_overclassification']} "
        "capacity8_overflows="
        f"{payload['overflow_example_count_by_capacity'].get('8')}"
    )
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only shadow diagnosis of Selector-v3 critical overflow telemetry"
        )
    )
    parser.add_argument("--runtime-report", required=True)
    parser.add_argument("--config")
    parser.add_argument("--capacities", nargs="+", type=int, default=[6, 8])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(
        runtime_report=args.runtime_report,
        config=args.config,
        capacities=args.capacities,
        output=args.output,
    )


if __name__ == "__main__":
    main()
