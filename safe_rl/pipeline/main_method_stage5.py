from __future__ import annotations

import argparse
import statistics
from pathlib import Path
from typing import Any, Mapping

import yaml

from safe_rl.accvp.contracts.schema import file_sha256, read_json, stable_hash, write_json_atomic
from safe_rl.analysis.paired_statistics import build_replicated_pair_statistics
from safe_rl.main_method_protocol import (
    FINAL_METHOD_ID,
    RULE_METHOD_ID,
    WCDT_V3_METHOD_ID,
    load_protocol,
    resolve_path,
)
from safe_rl.pipeline import stage5_paired_eval
from safe_rl.pipeline.main_method_lineage_compatibility import build_compatibility
from safe_rl.pipeline.main_method_ppo_suite import (
    METHOD_MANIFEST_KIND,
    SUITE_MANIFEST_KIND,
)
from safe_rl.ppo_replicates import plain, write_yaml_atomic
from safe_rl.utils.config import load_config


REQUEST_KIND = "main_method_stage5_request_v1"
REPORT_KIND = "main_method_stage5_aggregate_v1"
MODES = ("method_effect", "system_effect")


def _validate_fingerprint(payload: Mapping[str, Any], field: str, name: str) -> str:
    declared = str(payload.get(field, ""))
    content = {key: value for key, value in payload.items() if key != field}
    if not declared or stable_hash(content) != declared:
        raise ValueError(f"{name} fingerprint mismatch")
    return declared


def _load_suite(path: str | Path) -> tuple[Path, dict[str, Any], dict[str, dict[str, Any]]]:
    source = resolve_path(path)
    suite = read_json(source)
    _validate_fingerprint(suite, "manifest_fingerprint", "main-method PPO suite")
    if suite.get("artifact_kind") != SUITE_MANIFEST_KIND or suite.get("status") != "complete":
        raise ValueError("main-method PPO suite is not complete")
    manifests: dict[str, dict[str, Any]] = {}
    for method_id, binding in suite.get("methods", {}).items():
        if str(binding.get("policy_type", "")) == "rule_gap_acceptance":
            continue
        manifest_path = resolve_path(str(binding["manifest"]))
        if file_sha256(manifest_path) != str(binding["manifest_sha256"]):
            raise ValueError(f"{method_id}: method manifest hash changed")
        manifest = read_json(manifest_path)
        _validate_fingerprint(manifest, "manifest_fingerprint", f"{method_id} manifest")
        if (
            manifest.get("artifact_kind") != METHOD_MANIFEST_KIND
            or manifest.get("status") != "complete"
            or str(manifest.get("method_id", "")) != method_id
        ):
            raise ValueError(f"{method_id}: invalid method manifest")
        manifests[method_id] = manifest
    return source, suite, manifests


def _rows(manifest: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for raw in manifest.get("records", []) or []:
        row = dict(raw)
        seed = int(row.get("optimizer_seed", row.get("training_seed", -1)))
        if seed in result:
            raise ValueError(f"duplicate optimizer seed {seed}")
        for path_field, hash_field in (
            ("resolved_config", "resolved_config_sha256"),
            ("checkpoint", "checkpoint_sha256"),
            ("stage3_report", "stage3_report_sha256"),
        ):
            path = resolve_path(str(row[path_field]))
            if not path.is_file() or file_sha256(path) != str(row[hash_field]):
                raise ValueError(f"{manifest.get('method_id')}: {path_field} binding changed")
        result[seed] = row
    return result


def _semantic_rl(config: Mapping[str, Any]) -> dict[str, Any]:
    rl = dict(config.get("rl", {}) or {})
    return {
        "reward_profile": rl.get("reward_profile"),
        "training_semantics_version": rl.get("training_semantics_version"),
        "merge_timing_reward": plain(rl.get("merge_timing_reward", {}) or {}),
        "policy_lateral_commitment": plain(
            rl.get("policy_lateral_commitment", {}) or {}
        ),
    }


def _target_lineage(protocol: Mapping[str, Any]) -> dict[str, Any]:
    snapshot = protocol["_protocol_snapshot"]
    return {
        "protocol_id": str(snapshot["protocol_id"]),
        "protocol_enabled": bool(snapshot["enabled"]),
        "protocol_strict": bool(snapshot["strict"]),
        "seed_ledger_sha256": str(snapshot["seed_ledger_sha256"]),
    }


def _parent_matches(row: Mapping[str, Any], target: Mapping[str, Any]) -> bool:
    report = read_json(resolve_path(str(row["stage3_report"])))
    parent = dict(report.get("evidence_lineage", {}) or {})
    return all(str(parent.get(key, "")) == str(target.get(key, "")) for key in (
        "protocol_id",
        "seed_ledger_sha256",
    ))


def _group(
    *,
    method_id: str,
    row: Mapping[str, Any],
    config: Mapping[str, Any],
    shield: bool,
    mode: str,
    method_manifest_path: Path,
    target_lineage: Mapping[str, Any],
) -> dict[str, Any]:
    forecast = dict(config.get("forecast_features", {}) or {})
    enabled = bool(forecast.get("enabled", False))
    name = f"{method_id}_seed_{int(row['optimizer_seed'])}"
    group: dict[str, Any] = {
        "name": name,
        "policy_type": "sb3_ppo",
        "forecast_features": enabled,
        "forecast_source": str(forecast.get("source", "")) if enabled else "",
        "forecast_checkpoint": str(forecast.get("checkpoint", "")) if enabled else "",
        "shield": bool(shield),
        "model_path": str(row["checkpoint"]),
        "raw_policy": method_id,
        "shield_overrides": {
            "forecast_aware_candidate_ranking_mode": "off",
            "forecast_task_shadow_enabled": False,
            "task_backstop_enabled": False,
        },
        "rl_overrides": _semantic_rl(config),
        "accvp": plain(config.get("accvp", {}) or {}),
        "comparative": {
            "method": method_id,
            "training_seed": int(row["optimizer_seed"]),
            "optimizer_seed": int(row["optimizer_seed"]),
            "evaluation_variant": "policy" if mode == "method_effect" else "shield",
            "checkpoint_sha256": str(row["checkpoint_sha256"]),
            "reward_semantics_hash": str(row["reward_semantics_hash"]),
            "observation_contract_hash": str(row["observation_contract_hash"]),
        },
    }
    if not _parent_matches(row, target_lineage):
        group["comparative"]["parent_lineage_compatibility"] = build_compatibility(
            group_name=name,
            method_id=method_id,
            optimizer_seed=int(row["optimizer_seed"]),
            method_manifest=method_manifest_path,
            row=row,
            target_lineage=target_lineage,
        )
    return group


def _rule_group(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "name": RULE_METHOD_ID,
        "policy_type": "rule_gap_acceptance",
        "forecast_features": False,
        "shield": False,
        "accvp": plain(config.get("accvp", {}) or {}),
        "comparative": {
            "method": RULE_METHOD_ID,
            "training_seed": None,
            "evaluation_variant": "policy",
        },
    }


def _risk_binding(configs: list[Mapping[str, Any]]) -> dict[str, str]:
    paths = {
        str(
            resolve_path(
                str(
                    (cfg.get("rl", {}) or {})
                    .get("shield_guided_reward", {})
                    .get("risk_checkpoint", "")
                )
            )
        )
        for cfg in configs
    }
    if len(paths) != 1:
        raise ValueError("main-method configs do not bind one common Risk checkpoint")
    path = Path(paths.pop())
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path), "sha256": file_sha256(path)}


def _cache_binding(
    *,
    output: Path,
    mode: str,
    group: Mapping[str, Any],
    row: Mapping[str, Any],
    risk: Mapping[str, str],
) -> dict[str, Any]:
    shield = bool(group.get("shield", False))
    group_contract = {
        key: plain(value)
        for key, value in group.items()
        if key not in {"name", "comparative", "evaluation_cache"}
    }
    identity = {
        "method_id": str(row["method_id"]),
        "optimizer_seed": int(row["optimizer_seed"]),
        "checkpoint_sha256": str(row["checkpoint_sha256"]),
        "resolved_config": str(resolve_path(str(row["resolved_config"]))),
        "resolved_config_sha256": str(row["resolved_config_sha256"]),
        "reward_semantics_hash": str(row["reward_semantics_hash"]),
        "observation_contract_hash": str(row["observation_contract_hash"]),
        "risk_checkpoint_sha256": str(risk["sha256"]) if shield else "",
        "group_execution_contract_sha256": stable_hash(group_contract),
        "shield_enabled": shield,
        "policy_type": "sb3_ppo",
    }
    cache_dir = (
        output
        / "episode_cache"
        / mode
        / str(row["method_id"])
        / f"optimizer_seed_{int(row['optimizer_seed'])}"
    ).resolve()
    return {
        "artifact_kind": "stage5_episode_cache_binding_v1",
        "schema_version": 1,
        "cache_dir": str(cache_dir),
        "execution_fingerprint": stable_hash(identity),
        "identity": identity,
    }


def _write_yaml_idempotent(path: Path, payload: Mapping[str, Any]) -> Path:
    ready = plain(payload)
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            existing = yaml.safe_load(handle) or {}
        if plain(existing) != ready:
            raise FileExistsError(f"refusing to replace a different Stage5 config: {path}")
        return path.resolve()
    return write_yaml_atomic(path, ready).resolve()


def prepare(
    *,
    protocol_path: str | Path,
    suite_manifest: str | Path,
    mode: str,
    output_root: str | Path,
) -> Path:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    protocol = load_protocol(protocol_path, verify_artifacts=True)
    suite_path, suite, manifests = _load_suite(suite_manifest)
    if str(suite.get("protocol_id", "")) != str(protocol["protocol_id"]):
        raise ValueError("PPO suite and Stage5 protocol_id disagree")
    seeds = [int(seed) for seed in protocol["optimizer_seeds"]]
    row_maps = {method_id: _rows(manifest) for method_id, manifest in manifests.items()}
    for method_id, rows in row_maps.items():
        if sorted(rows) != seeds:
            raise ValueError(f"{method_id}: Stage5 requires the exact optimizer-seed cohort")
    config_maps = {
        method_id: {
            seed: plain(load_config(rows[seed]["resolved_config"]))
            for seed in seeds
        }
        for method_id, rows in row_maps.items()
    }
    risk = _risk_binding(
        [cfg for configs in config_maps.values() for cfg in configs.values()]
    )
    target = _target_lineage(protocol)
    output = resolve_path(output_root)
    mode_shield = bool(protocol["evaluation"][mode]["shield_enabled"])
    comparisons = [dict(item) for item in protocol.get("comparisons", [])]
    secondary_specs = [item for item in comparisons if item["role"] != "primary_final"]
    primary_specs = [item for item in comparisons if item["role"] == "primary_final"]
    if len(primary_specs) != 1:
        raise ValueError("main-method protocol requires exactly one primary comparison")
    secondary_count = int(protocol["evaluation"]["secondary_simulator_seed_count"])
    primary_count = int(protocol["evaluation"]["primary_simulator_seed_count"])
    stage5_seeds = list(protocol["_stage5_seeds"])
    request_rows: list[dict[str, Any]] = []
    for optimizer_seed in seeds:
        for scope, specs, simulator_seeds in (
            ("secondary", secondary_specs, stage5_seeds[:secondary_count]),
            ("primary", primary_specs, stage5_seeds[:primary_count]),
        ):
            method_ids = (
                sorted(manifests)
                if scope == "secondary"
                else [WCDT_V3_METHOD_ID, FINAL_METHOD_ID]
            )
            groups = []
            for method_id in method_ids:
                row = row_maps[method_id][optimizer_seed]
                cfg = config_maps[method_id][optimizer_seed]
                manifest_path = resolve_path(
                    str(suite["methods"][method_id]["manifest"])
                )
                group = _group(
                    method_id=method_id,
                    row=row,
                    config=cfg,
                    shield=mode_shield,
                    mode=mode,
                    method_manifest_path=manifest_path,
                    target_lineage=target,
                )
                group["evaluation_cache"] = _cache_binding(
                    output=output,
                    mode=mode,
                    group=group,
                    row=row,
                    risk=risk,
                )
                groups.append(group)
            if (
                scope == "secondary"
                and mode == "method_effect"
                and optimizer_seed == seeds[0]
            ):
                no_forecast_cfg = config_maps[
                    "ppo_no_forecast_reward_v2"
                ][optimizer_seed]
                groups.append(_rule_group(no_forecast_cfg))
            pair_rows = []
            names = {str(group["name"]) for group in groups}
            for spec in specs:
                left = f"{spec['left']}_seed_{optimizer_seed}"
                right = f"{spec['right']}_seed_{optimizer_seed}"
                if left in names and right in names:
                    pair_rows.append(
                        {
                            "name": str(spec["comparison_id"]),
                            "left": left,
                            "right": right,
                            "acceptance_profile": "paired_policy_non_regression_v1",
                        }
                    )
            base_method = FINAL_METHOD_ID if scope == "primary" else method_ids[-1]
            resolved = plain(config_maps[base_method][optimizer_seed])
            run_id = f"main_method_stage5_{mode}_{scope}_seed_{optimizer_seed}"
            resolved["run"]["run_id"] = run_id
            evaluation_cfg = load_config(resolve_path(str(protocol["evaluation_config"])))
            resolved["evaluation_protocol"] = plain(evaluation_cfg.evaluation_protocol)
            resolved["evaluation_protocol"]["stage5_role"] = str(
                protocol["evaluation"]["seed_role"]
            )
            resolved["experiment"] = {
                "purpose": "formal_main_method_comparison",
                "mode": mode,
                "scope": scope,
                "optimizer_seed": optimizer_seed,
                "conclusion_scope": str(protocol["evaluation"][mode]["conclusion_scope"]),
            }
            resolved["stage5"] = {
                "paired_eval": True,
                "same_seed": True,
                "compare_shield_off_on": False,
                "replay_enabled": True,
                "episodes_per_group": len(simulator_seeds),
                "seeds": simulator_seeds,
                "risk_checkpoint": str(risk["path"]),
                "execution_contract": str(protocol["evaluation"]["execution_contract"]),
                "allow_accvp_observation_without_safety_shield": mode == "method_effect",
                "require_accvp_observation_runtime_gate": False,
                "statistics": plain(protocol["evaluation"]["statistics"]),
                "acceptance": {
                    "paired_policy_non_regression_v1": {
                        "max_actual_replacement_rate": 0.05,
                        "reward_tolerance": 1.0e-6,
                    }
                },
                "pairs": pair_rows,
                "groups": groups,
            }
            config_path = (
                output
                / "configs"
                / mode
                / scope
                / f"optimizer_seed_{optimizer_seed}.yaml"
            )
            _write_yaml_idempotent(config_path, resolved)
            report_path = (
                resolve_path(resolved["run"]["output_root"])
                / run_id
                / "stage5"
                / "formal_paired_eval_report.json"
            )
            request_rows.append(
                {
                    "mode": mode,
                    "scope": scope,
                    "optimizer_seed": optimizer_seed,
                    "stage5_config": str(config_path.resolve()),
                    "stage5_config_sha256": file_sha256(config_path),
                    "stage5_report": str(report_path.resolve()),
                    "method_ids": method_ids,
                    "simulator_seed_count": len(simulator_seeds),
                }
            )
    request: dict[str, Any] = {
        "artifact_kind": REQUEST_KIND,
        "schema_version": 1,
        "status": "prepared",
        "protocol_id": str(protocol["protocol_id"]),
        "mode": mode,
        "conclusion_scope": str(protocol["evaluation"][mode]["conclusion_scope"]),
        "protocol": str(resolve_path(protocol_path)),
        "protocol_sha256": file_sha256(resolve_path(protocol_path)),
        "ppo_suite_manifest": str(suite_path),
        "ppo_suite_manifest_sha256": file_sha256(suite_path),
        "optimizer_seeds": seeds,
        "risk_checkpoint": risk,
        "statistics": plain(protocol["evaluation"]["statistics"]),
        "rows": request_rows,
    }
    request["request_fingerprint"] = stable_hash(request)
    path = output / f"{mode}_request.json"
    if path.exists():
        if read_json(path) != request:
            raise FileExistsError(f"refusing to replace a different Stage5 request: {path}")
        return path.resolve()
    write_json_atomic(path, request)
    return path.resolve()


def _receipt(report_path: Path) -> Path:
    return report_path.with_name(report_path.stem + ".main_method_receipt.json")


def execute(request_path: str | Path) -> Path:
    source = resolve_path(request_path)
    request = read_json(source)
    _validate_fingerprint(request, "request_fingerprint", "main-method Stage5 request")
    if request.get("artifact_kind") != REQUEST_KIND:
        raise ValueError("unsupported main-method Stage5 request")
    for row in request.get("rows", []) or []:
        config_path = resolve_path(str(row["stage5_config"]))
        if file_sha256(config_path) != str(row["stage5_config_sha256"]):
            raise ValueError("Stage5 config changed after request generation")
        expected = resolve_path(str(row["stage5_report"]))
        receipt_path = _receipt(expected)
        if expected.is_file():
            receipt = read_json(receipt_path) if receipt_path.is_file() else None
            if not receipt or str(receipt.get("stage5_report_sha256", "")) != file_sha256(expected):
                raise FileExistsError(
                    f"refusing to reuse Stage5 output without its matching receipt: {expected}"
                )
            continue
        produced_dir = Path(stage5_paired_eval.run(load_config(config_path))).resolve()
        produced = produced_dir / "formal_paired_eval_report.json"
        if produced != expected or not produced.is_file():
            raise RuntimeError(
                f"Stage5 output path mismatch: expected={expected} actual={produced}"
            )
        receipt = {
            "artifact_kind": "main_method_stage5_execution_receipt_v1",
            "stage5_config": str(config_path),
            "stage5_config_sha256": file_sha256(config_path),
            "stage5_report": str(produced),
            "stage5_report_sha256": file_sha256(produced),
        }
        receipt["receipt_fingerprint"] = stable_hash(receipt)
        write_json_atomic(receipt_path, receipt)
    return aggregate(source)


def _headline(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_method: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("optimizer_seed") is None:
            continue
        by_method.setdefault(str(row["method_id"]), []).append(row)
    result = []
    for method_id, items in sorted(by_method.items()):
        metrics = sorted(
            {
                key
                for item in items
                for key, value in item["metrics"].items()
                if isinstance(value, (int, float)) and not isinstance(value, bool)
            }
        )
        result.append(
            {
                "method_id": method_id,
                "optimizer_seed_count": len(items),
                "mean": {
                    key: statistics.fmean(float(item["metrics"].get(key, 0.0)) for item in items)
                    for key in metrics
                },
                "sample_std": {
                    key: (
                        statistics.stdev(
                            float(item["metrics"].get(key, 0.0)) for item in items
                        )
                        if len(items) > 1
                        else 0.0
                    )
                    for key in metrics
                },
            }
        )
    return result


def aggregate(request_path: str | Path) -> Path:
    source = resolve_path(request_path)
    request = read_json(source)
    _validate_fingerprint(request, "request_fingerprint", "main-method Stage5 request")
    rows: list[dict[str, Any]] = []
    deterministic: dict[str, Any] | None = None
    comparison_sources: dict[str, list[dict[str, Any]]] = {}
    for item in request.get("rows", []) or []:
        report_path = resolve_path(str(item["stage5_report"]))
        if not report_path.is_file():
            raise FileNotFoundError(report_path)
        report = read_json(report_path)
        groups = dict(report.get("groups", {}) or {})
        acceptance = dict(report.get("acceptance", {}) or {})
        for pair in list(report.get("configured_pairs", []) or []):
            comparison_id = str(pair.get("name", ""))
            left_name = str(pair.get("left", ""))
            right_name = str(pair.get("right", ""))
            left = dict(groups.get(left_name, {}) or {})
            right = dict(groups.get(right_name, {}) or {})
            if not comparison_id or not left or not right:
                raise ValueError(
                    f"Stage5 report lacks configured comparison groups: {report_path}"
                )
            left_comparative = dict(left.get("comparative", {}) or {})
            right_comparative = dict(right.get("comparative", {}) or {})
            comparison_sources.setdefault(comparison_id, []).append(
                {
                    "scope": str(item["scope"]),
                    "training_seed": int(item["optimizer_seed"]),
                    "left_method_id": str(left_comparative.get("method", left_name)),
                    "right_method_id": str(right_comparative.get("method", right_name)),
                    "left_checkpoint_sha256": str(
                        left_comparative.get("checkpoint_sha256", "")
                    ),
                    "right_checkpoint_sha256": str(
                        right_comparative.get("checkpoint_sha256", "")
                    ),
                    "left_report": left,
                    "right_report": right,
                    "source_acceptance": plain(acceptance.get(comparison_id, {}) or {}),
                    "source_report": str(report_path),
                    "source_report_sha256": file_sha256(report_path),
                }
            )
        for group_name, group in (report.get("groups", {}) or {}).items():
            comparative = dict(group.get("comparative", {}) or {})
            method_id = str(comparative.get("method", group_name))
            seed = comparative.get("training_seed")
            scope = str(item["scope"])
            if seed is None:
                deterministic = {
                    "method_id": method_id,
                    "optimizer_seed": None,
                    "scope": scope,
                    "simulator_seed_count": int(item["simulator_seed_count"]),
                    "metrics": plain(group.get("metrics", {}) or {}),
                }
                continue
            if scope == "secondary" and method_id in {WCDT_V3_METHOD_ID, FINAL_METHOD_ID}:
                continue
            rows.append(
                {
                    "method_id": method_id,
                    "optimizer_seed": int(seed),
                    "scope": scope,
                    "simulator_seed_count": int(item["simulator_seed_count"]),
                    "metrics": plain(group.get("metrics", {}) or {}),
                }
            )
    expected = {
        (method_id, seed)
        for method_id in request.get("rows", [])[0].get("method_ids", [])
        for seed in request["optimizer_seeds"]
    }
    observed = {(str(row["method_id"]), int(row["optimizer_seed"])) for row in rows}
    # The first request row is secondary and contains all learned methods.
    if expected != observed:
        raise ValueError(
            "main-method Stage5 aggregate lacks a complete learned-method grid: "
            f"missing={sorted(expected - observed)} extra={sorted(observed - expected)}"
        )
    statistics_cfg = dict(
        dict(request.get("statistics", {}) or {})
        or {
            "confidence": 0.95,
            "bootstrap_replicates": 10_000,
            "bootstrap_seed": 42_001,
        }
    )
    comparisons: dict[str, Any] = {}
    expected_optimizer_seeds = [int(seed) for seed in request["optimizer_seeds"]]
    for offset, (comparison_id, sources) in enumerate(sorted(comparison_sources.items())):
        ordered = sorted(sources, key=lambda value: int(value["training_seed"]))
        training_seeds = [int(value["training_seed"]) for value in ordered]
        if training_seeds != expected_optimizer_seeds:
            raise ValueError(
                f"comparison {comparison_id} lacks the complete optimizer-seed cohort"
            )
        left_methods = {str(value["left_method_id"]) for value in ordered}
        right_methods = {str(value["right_method_id"]) for value in ordered}
        scopes = {str(value["scope"]) for value in ordered}
        if len(left_methods) != 1 or len(right_methods) != 1 or len(scopes) != 1:
            raise ValueError(f"comparison {comparison_id} changes identity across replicates")
        statistics_result = build_replicated_pair_statistics(
            ordered,
            confidence=float(statistics_cfg.get("confidence", 0.95)),
            replicates=int(statistics_cfg.get("bootstrap_replicates", 10_000)),
            seed=int(statistics_cfg.get("bootstrap_seed", 42_001)) + offset,
            require_distinct_checkpoints=True,
        )
        source_acceptance = [
            {
                "training_seed": int(value["training_seed"]),
                "acceptance": value["source_acceptance"],
                "source_report": value["source_report"],
                "source_report_sha256": value["source_report_sha256"],
            }
            for value in ordered
        ]
        comparisons[comparison_id] = {
            "left_method_id": next(iter(left_methods)),
            "right_method_id": next(iter(right_methods)),
            "scope": next(iter(scopes)),
            "delta_direction": "right_minus_left",
            "statistics": statistics_result,
            "source_acceptance": source_acceptance,
            "all_source_acceptance_available": all(
                bool(value["source_acceptance"].get("available", False))
                for value in ordered
            ),
            "any_source_regression": any(
                bool(value["source_acceptance"].get("regression", False))
                for value in ordered
            ),
        }
    report: dict[str, Any] = {
        "artifact_kind": REPORT_KIND,
        "schema_version": 1,
        "status": "complete",
        "protocol_id": str(request["protocol_id"]),
        "mode": str(request["mode"]),
        "conclusion_scope": str(request["conclusion_scope"]),
        "source_request": str(source),
        "source_request_sha256": file_sha256(source),
        "by_optimizer_seed": rows,
        "deterministic_rule": deterministic,
        "headline": _headline(rows),
        "comparisons": comparisons,
        "comparison_interpretation": {
            "statistics": "crossed optimizer-seed by simulator-seed paired bootstrap",
            "delta_direction": "right_minus_left",
            "source_acceptance": (
                "per-optimizer preregistered checks are reported, not used to suppress "
                "the aggregate when a regression is observed"
            ),
        },
        "multiplicity": {
            "primary_family": str(statistics_cfg.get("primary_family", "")),
            "secondary_family_correction": str(
                statistics_cfg.get("secondary_family_correction", "holm")
            ),
            "status": "not_applied_no_explicit_valid_pvalues",
            "reason": (
                "crossed bootstrap confidence intervals do not create valid independent "
                "p-values; no p-values are invented for Holm adjustment"
            ),
        },
        "runtime_gate_applied": False,
    }
    report["report_fingerprint"] = stable_hash(report)
    output = source.with_name(f"{request['mode']}_aggregate.json")
    if output.exists():
        if read_json(output) != report:
            raise FileExistsError(f"refusing to replace a different Stage5 aggregate: {output}")
        return output.resolve()
    write_json_atomic(output, report)
    return output.resolve()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare, execute or aggregate main-method method/system-effect Stage5 reports"
    )
    sub = parser.add_subparsers(dest="action", required=True)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--protocol", required=True)
    prepare_parser.add_argument("--suite-manifest", required=True)
    prepare_parser.add_argument("--mode", choices=MODES, required=True)
    prepare_parser.add_argument("--output-root", required=True)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--request", required=True)
    aggregate_parser = sub.add_parser("aggregate")
    aggregate_parser.add_argument("--request", required=True)
    args = parser.parse_args()
    if args.action == "prepare":
        output = prepare(
            protocol_path=args.protocol,
            suite_manifest=args.suite_manifest,
            mode=args.mode,
            output_root=args.output_root,
        )
    elif args.action == "run":
        output = execute(args.request)
    else:
        output = aggregate(args.request)
    print(f"main_method_stage5_artifact={output}")


if __name__ == "__main__":
    main()
