from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from safe_rl.analysis.paired_statistics import build_pair_statistics
from safe_rl.accvp.contracts.artifacts import resolve_v2_bundle
from safe_rl.accvp.serving.observation import RiskGatedACCVPCandidateTableAugmentor
from safe_rl.evaluation_protocol import (
    EvidenceProtocolError,
    audit_seed_cohorts,
    assert_disjoint_seed_usage,
    build_stage_lineage,
    file_sha256,
    seeds_for_role,
    stable_hash,
    validate_parent_lineage,
)
from safe_rl.pipeline.common import latest_stage_file, load_stage_config, parse_config_arg, write_report
from safe_rl.pipeline.accvp_observation_preflight import _gate as _accvp_observation_runtime_gate
from safe_rl.rl.evaluation import evaluate_policy
from safe_rl.sim.metrics import SAFETY_METRIC_VERSION
from safe_rl.utils.config import clone_with_overrides, prepare_run_dir
from safe_rl.utils.progress import TensorboardLogger, stage_log


def _default_model_path(cfg) -> Path:
    configured = cfg.stage5.get("default_model_path")
    if configured:
        return Path(configured)
    return latest_stage_file(cfg, "stage3", str(cfg.stage3.model_name))


def _risk_path(cfg) -> Path:
    configured = cfg.stage5.get("risk_checkpoint")
    if configured:
        return Path(configured)
    return latest_stage_file(cfg, "stage2", "risk_module.pt")


def _accvp_bundle_lineage(cfg: Any) -> dict[str, Any]:
    """Resolve the immutable VNext bundle identity used by one Stage5 group."""

    if not RiskGatedACCVPCandidateTableAugmentor.enabled(cfg):
        return {"required": False, "enabled": False}
    source = cfg.accvp.get("artifact_manifest")
    if not source:
        raise EvidenceProtocolError(
            "ACCVP-enabled Stage5 group requires accvp.artifact_manifest"
        )
    path = Path(str(source)).resolve()
    try:
        manifest, resolved = resolve_v2_bundle(path)
    except (FileNotFoundError, ValueError) as exc:
        raise EvidenceProtocolError(f"invalid Stage5 ACCVP bundle: {exc}") from exc
    predictor = resolved.get("predictor")
    if predictor is None:
        raise EvidenceProtocolError("Stage5 ACCVP bundle is missing predictor")
    payload = {
        "required": True,
        "enabled": True,
        "manifest_path": str(path),
        "manifest_sha256": file_sha256(path),
        "artifact_fingerprint": str(manifest["artifact_fingerprint"]),
        "artifact_variant": str(manifest["artifact_variant"]),
        "artifact_generation": str(manifest["artifact_generation"]),
        "bundle_schema_version": int(manifest["bundle_schema_version"]),
        "formal_runtime_contract_sha256": str(
            manifest["formal_runtime_contract_sha256"]
        ),
        "predictor_sha256": file_sha256(predictor),
    }
    payload["binding_fingerprint"] = stable_hash(payload)
    return payload


def _validate_lineage_fingerprint(lineage: dict[str, Any], *, name: str) -> None:
    expected = str(lineage.get("lineage_fingerprint", ""))
    if not expected:
        raise EvidenceProtocolError(f"{name} is missing lineage_fingerprint")
    content = dict(lineage)
    content.pop("lineage_fingerprint", None)
    if stable_hash(content) != expected:
        raise EvidenceProtocolError(f"{name} lineage_fingerprint mismatch")


def _select_eval_seeds(cfg) -> list[int]:
    requested = int(cfg.stage5.episodes_per_group)
    seed_file = cfg.stage5.get("seed_file")
    if seed_file:
        with Path(seed_file).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        source = payload.get("seeds", payload) if isinstance(payload, dict) else payload
        seeds = [int(seed) for seed in source]
    else:
        seeds = [int(seed) for seed in cfg.stage5.seeds]
    protocol_cfg = dict(cfg.get("evaluation_protocol", {}) or {})
    stage5_role = str(protocol_cfg.get("stage5_role", "stage5_confirmatory"))
    seeds = seeds_for_role(cfg, stage5_role, fallback=seeds)
    if len(seeds) != len(set(seeds)):
        raise EvidenceProtocolError("Stage5 evaluation seeds contain duplicates")
    if requested <= 0:
        if not seeds:
            raise ValueError("stage5.episodes_per_group<=0 requires at least one configured seed")
        return seeds
    if len(seeds) < requested:
        raise ValueError(
            f"stage5.episodes_per_group={requested} requires at least {requested} seeds, "
            f"but {'stage5.seed_file' if seed_file else 'stage5.seeds'} has {len(seeds)}"
        )
    return seeds[:requested]


def _stage3_training_report(model_path: Path) -> Path:
    return model_path.parent / "stage3_training_report.json"


def _validate_stage5_model_lineage(
    *,
    model_path: Path,
    stage5_lineage: dict[str, Any],
    stage5_seeds: list[int],
) -> dict[str, Any]:
    report_path = _stage3_training_report(model_path)
    strict = bool(stage5_lineage.get("protocol_strict", False))
    if not report_path.exists():
        if strict:
            raise EvidenceProtocolError(f"strict Stage5 requires Stage3 report for model: {model_path}")
        return {"available": False, "stage3_report": str(report_path)}
    with report_path.open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    parent = dict(report.get("evidence_lineage", {}) or {})
    if strict or parent.get("lineage_fingerprint"):
        _validate_lineage_fingerprint(parent, name="Stage3 evidence lineage")
    validate_parent_lineage(parent, stage5_lineage)
    role_usage = dict(parent.get("role_usage", {}) or {})
    training = [int(seed) for seed in role_usage.get("stage3_training", {}).get("seeds", [])]
    selection = [int(seed) for seed in role_usage.get("stage3_selection", {}).get("seeds", [])]
    if not training:
        training = [
            int(item["episode_seed"])
            for item in report.get("training_episode_seed_records", [])
            if "episode_seed" in item
        ]
    if not selection:
        selection = [int(seed) for seed in report.get("checkpoint_selection_seeds", [])]
    if bool(stage5_lineage.get("protocol_enabled", False)) or bool(
        parent.get("protocol_enabled", False)
    ):
        seed_audit = assert_disjoint_seed_usage(
            stage3_training=training,
            stage3_selection=selection,
            stage5_confirmatory=stage5_seeds,
        )
        seed_audit["enforced"] = True
    else:
        seed_audit = audit_seed_cohorts(
            {
                "stage3_training": training,
                "stage3_selection": selection,
                "stage5_confirmatory": stage5_seeds,
            }
        )
        seed_audit["enforced"] = False
    return {
        "available": True,
        "stage3_report": str(report_path.resolve()),
        "stage3_lineage_fingerprint": parent.get("lineage_fingerprint"),
        "training_seed_count": len(training),
        "selection_seed_count": len(selection),
        "overlap_count": int(seed_audit.get("overlap_count", 0)),
        "seed_audit": seed_audit,
    }


def _validate_stage5_observation_contract(
    *,
    model_path: Path,
    group_cfg: Any,
    protocol_strict: bool,
) -> dict[str, Any]:
    if not RiskGatedACCVPCandidateTableAugmentor.enabled(group_cfg):
        return {"required": False, "available": True}
    report_path = _stage3_training_report(model_path)
    if not report_path.exists():
        if protocol_strict:
            raise EvidenceProtocolError(
                f"strict ACCVP Stage5 requires Stage3 observation contract: {report_path}"
            )
        return {"required": True, "available": False, "stage3_report": str(report_path)}
    with report_path.open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    expected_bundle = _accvp_bundle_lineage(group_cfg)
    parent_lineage = dict(report.get("evidence_lineage", {}) or {})
    if protocol_strict or parent_lineage.get("lineage_fingerprint"):
        _validate_lineage_fingerprint(
            parent_lineage,
            name="Stage3 ACCVP evidence lineage",
        )
    recorded_bundle = report.get("accvp_bundle_lineage")
    parent_bundle = parent_lineage.get("accvp_bundle")
    if not isinstance(recorded_bundle, dict) or not isinstance(parent_bundle, dict):
        if protocol_strict:
            raise EvidenceProtocolError(
                "strict ACCVP Stage5 requires Stage3 bundle lineage"
            )
        return {
            "required": True,
            "available": False,
            "stage3_report": str(report_path.resolve()),
            "reason": "missing_stage3_accvp_bundle_lineage",
        }
    if recorded_bundle != parent_bundle:
        raise EvidenceProtocolError(
            "Stage3 report ACCVP bundle lineage disagrees with its evidence lineage"
        )
    recorded_fingerprint = str(recorded_bundle.get("binding_fingerprint", ""))
    recorded_content = dict(recorded_bundle)
    recorded_content.pop("binding_fingerprint", None)
    if not recorded_fingerprint or stable_hash(recorded_content) != recorded_fingerprint:
        raise EvidenceProtocolError("Stage3 ACCVP bundle binding_fingerprint mismatch")
    binding_fields = (
        "manifest_sha256",
        "artifact_fingerprint",
        "artifact_variant",
        "artifact_generation",
        "bundle_schema_version",
        "formal_runtime_contract_sha256",
        "predictor_sha256",
    )
    differing = [
        key
        for key in binding_fields
        if recorded_bundle.get(key) != expected_bundle.get(key)
    ]
    if differing:
        raise EvidenceProtocolError(
            "Stage3/Stage5 ACCVP bundle lineage mismatch: "
            f"fields={differing}"
        )
    expected_names = RiskGatedACCVPCandidateTableAugmentor.feature_names(group_cfg)
    expected_hash = stable_hash(list(expected_names))
    recorded_hash = str(report.get("accvp_observation_feature_names_sha256", ""))
    if not recorded_hash:
        if protocol_strict:
            raise EvidenceProtocolError("Stage3 report is missing ACCVP feature-name hash")
        return {
            "required": True,
            "available": False,
            "stage3_report": str(report_path.resolve()),
            "expected_feature_names_sha256": expected_hash,
        }
    if recorded_hash != expected_hash:
        raise EvidenceProtocolError(
            "Stage3/Stage5 ACCVP observation feature contract mismatch: "
            f"stage3={recorded_hash} stage5={expected_hash}"
        )
    return {
        "required": True,
        "available": True,
        "stage3_report": str(report_path.resolve()),
        "feature_count": len(expected_names),
        "feature_names_sha256": expected_hash,
        "accvp_bundle": expected_bundle,
        "stage3_bundle_lineage_match": True,
    }


def _validate_frozen_runtime_preflight(
    *,
    cfg: Any,
    group_model_paths: dict[str, Path],
    protocol_strict: bool,
) -> dict[str, Any]:
    required = bool(cfg.stage5.get("require_accvp_observation_runtime_gate", False))
    source = cfg.stage5.get("accvp_observation_preflight_report")
    if not source:
        if required and protocol_strict:
            raise EvidenceProtocolError(
                "strict Stage5 runtime gate requires stage5.accvp_observation_preflight_report"
            )
        return {"required": required, "available": False, "validated_before_stage5": False}
    path = Path(str(source))
    if not path.exists():
        raise FileNotFoundError(f"ACCVP observation preflight report not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    gate = dict(report.get("gate", {}) or {})
    if not bool(gate.get("pass", False)):
        raise EvidenceProtocolError("frozen ACCVP observation preflight did not pass")
    accvp_groups: list[tuple[str, Any, Path]] = []
    for group in cfg.stage5.groups:
        group_cfg = clone_with_overrides(cfg, _group_overrides(group))
        model_path = group_model_paths.get(str(group.name))
        if model_path is not None and RiskGatedACCVPCandidateTableAugmentor.enabled(group_cfg):
            accvp_groups.append((str(group.name), group_cfg, model_path))
    model_hashes = {file_sha256(model_path) for _name, _group_cfg, model_path in accvp_groups}
    if len(model_hashes) > 1:
        raise EvidenceProtocolError("one frozen preflight cannot validate multiple ACCVP PPO models")
    if model_hashes and str(report.get("policy_model_sha256", "")) not in model_hashes:
        raise EvidenceProtocolError("frozen preflight PPO model hash does not match Stage5 model")
    feature_hashes = {
        stable_hash(
            {"feature_names": RiskGatedACCVPCandidateTableAugmentor.feature_names(group_cfg)}
        )
        for _name, group_cfg, _model_path in accvp_groups
    }
    if len(feature_hashes) > 1:
        raise EvidenceProtocolError("ACCVP Stage5 groups do not share one observation feature contract")
    if feature_hashes and str(report.get("accvp_observation_feature_names_sha256", "")) not in feature_hashes:
        raise EvidenceProtocolError("frozen preflight observation feature hash does not match Stage5")
    return {
        "required": required,
        "available": True,
        "validated_before_stage5": True,
        "report": str(path.resolve()),
        "report_sha256": file_sha256(path),
        "gate": gate,
        "policy_model_sha256": report.get("policy_model_sha256"),
        "accvp_observation_feature_contract_hash": report.get(
            "accvp_observation_feature_contract_hash"
        ),
    }


def _group_overrides(group) -> dict:
    forecast_overrides = {"enabled": bool(group.forecast_features)}
    if group.get("forecast_source"):
        forecast_overrides["source"] = str(group.forecast_source)
    if group.get("forecast_checkpoint"):
        forecast_overrides["checkpoint"] = str(group.forecast_checkpoint)
    shield_overrides = {"enabled": bool(group.shield)}
    requested_shield_overrides = group.get("shield_overrides")
    if requested_shield_overrides:
        shield_overrides.update(dict(requested_shield_overrides))
    overrides = {
        "forecast_features": forecast_overrides,
        "rl": {"use_wcdt_forecast_features": bool(group.forecast_features)},
        "shield": shield_overrides,
    }
    requested_rl_overrides = group.get("rl_overrides")
    if requested_rl_overrides:
        overrides["rl"].update(dict(requested_rl_overrides))
    requested_accvp = group.get("accvp")
    if requested_accvp:
        overrides["accvp"] = dict(requested_accvp)
    requested_risk_overrides = group.get("risk_module_overrides")
    if requested_risk_overrides:
        overrides["risk_module"] = dict(requested_risk_overrides)
    return overrides


def _group_model_path(group, default_model: Path) -> Path | None:
    if str(group.get("policy_type", "sb3_ppo")) == "rule_gap_acceptance":
        return None
    model_path = group.get("model_path")
    if bool(group.forecast_features) and not model_path:
        raise ValueError(
            f"stage5 group '{group.name}' enables forecast_features, so it must set "
            "model_path to a PPO checkpoint trained with forecast observations."
        )
    return Path(model_path or default_model)


def _paired_delta(a_report: dict | None, b_report: dict | None) -> dict | None:
    if not a_report or not b_report or "episodes" not in a_report or "episodes" not in b_report:
        return None
    right = {int(item["seed"]): item for item in b_report["episodes"]}
    rows = []
    for left in a_report["episodes"]:
        seed = int(left["seed"])
        if seed not in right:
            continue
        item = right[seed]
        rows.append(
            {
                "seed": seed,
                "reward_delta": float(item["episode_reward"] - left["episode_reward"]),
                "min_distance_delta": float(item["min_distance"] - left["min_distance"]),
                "ttc_delta": float(item["ttc_p1"] - left["ttc_p1"]),
                "drac_delta": float(item["drac_p99"] - left["drac_p99"]),
                "drac_raw_delta": float(
                    item.get("drac_p99_raw", item.get("drac_p99", 0.0))
                    - left.get("drac_p99_raw", left.get("drac_p99", 0.0))
                ),
                "drac_capped_delta": float(
                    item.get("drac_p99_capped", min(float(item.get("drac_p99", 0.0)), 20.0))
                    - left.get("drac_p99_capped", min(float(left.get("drac_p99", 0.0)), 20.0))
                ),
                "proxy_collision_delta": int(
                    int(bool(item.get("proxy_collision", False))) - int(bool(left.get("proxy_collision", False)))
                ),
                "safety_violation_delta": int(
                    int(bool(item.get("safety_violation", False))) - int(bool(left.get("safety_violation", False)))
                ),
                "geometric_overlap_delta": int(
                    int(bool(item.get("geometric_overlap", False))) - int(bool(left.get("geometric_overlap", False)))
                ),
                "proxy_collision_count_delta": int(
                    int(item.get("proxy_collision_count", int(bool(item.get("proxy_collision", False)))))
                    - int(left.get("proxy_collision_count", int(bool(left.get("proxy_collision", False)))))
                ),
                "safety_violation_count_delta": int(
                    int(item.get("safety_violation_count", int(bool(item.get("safety_violation", False)))))
                    - int(left.get("safety_violation_count", int(bool(left.get("safety_violation", False)))))
                ),
                "taper_miss_delta": int(
                    int(bool(item.get("taper_miss", False))) - int(bool(left.get("taper_miss", False)))
                ),
                "min_distance_le_collision_threshold_count_delta": int(
                    int(
                        item.get(
                            "min_distance_le_collision_threshold_count",
                            item.get("proxy_collision_count", int(bool(item.get("proxy_collision", False)))),
                        )
                    )
                    - int(
                        left.get(
                            "min_distance_le_collision_threshold_count",
                            left.get("proxy_collision_count", int(bool(left.get("proxy_collision", False)))),
                        )
                    )
                ),
                "completion_time_delta": float(item.get("completion_time", 0.0) - left.get("completion_time", 0.0)),
                "ego_speed_mean_delta": float(item.get("ego_speed_mean", 0.0) - left.get("ego_speed_mean", 0.0)),
                "hard_brake_rate_delta": float(item.get("hard_brake_rate", 0.0) - left.get("hard_brake_rate", 0.0)),
                "intervention_delta": int(item["intervention_count"] - left["intervention_count"]),
                "actual_replacement_delta": int(
                    item.get("actual_replacement_count", 0) - left.get("actual_replacement_count", 0)
                ),
                "task_replacement_delta": int(
                    item.get("task_replacement_count", 0) - left.get("task_replacement_count", 0)
                ),
                "forecast_ranking_replacement_delta": int(
                    item.get("forecast_ranking_replacement_count", 0)
                    - left.get("forecast_ranking_replacement_count", 0)
                ),
                "fallback_delta": int(item["fallback_count"] - left["fallback_count"]),
                "emergency_fallback_delta": int(
                    item.get("emergency_fallback_count", 0) - left.get("emergency_fallback_count", 0)
                ),
                "missed_safe_merge_opportunity_delta": int(
                    item.get("missed_safe_merge_opportunity_count", 0)
                    - left.get("missed_safe_merge_opportunity_count", 0)
                ),
                "deadline_missed_safe_merge_delta": int(
                    item.get("deadline_missed_safe_merge_count", 0)
                    - left.get("deadline_missed_safe_merge_count", 0)
                ),
                "no_merge_request_before_taper_delta": int(
                    item.get("no_merge_request_before_taper_count", 0)
                    - left.get("no_merge_request_before_taper_count", 0)
                ),
            }
        )
    if not rows:
        return None
    return {
        "episodes": rows,
        "mean_reward_delta": sum(row["reward_delta"] for row in rows) / len(rows),
        "mean_min_distance_delta": sum(row["min_distance_delta"] for row in rows) / len(rows),
        "mean_ttc_delta": sum(row["ttc_delta"] for row in rows) / len(rows),
        "mean_drac_delta": sum(row["drac_delta"] for row in rows) / len(rows),
        "mean_drac_raw_delta": sum(row["drac_raw_delta"] for row in rows) / len(rows),
        "mean_drac_capped_delta": sum(row["drac_capped_delta"] for row in rows) / len(rows),
        "mean_proxy_collision_delta": sum(row["proxy_collision_delta"] for row in rows) / len(rows),
        "mean_safety_violation_delta": sum(row["safety_violation_delta"] for row in rows) / len(rows),
        "mean_geometric_overlap_delta": sum(row["geometric_overlap_delta"] for row in rows) / len(rows),
        "mean_missed_safe_merge_opportunity_delta": (
            sum(row["missed_safe_merge_opportunity_delta"] for row in rows) / len(rows)
        ),
        "proxy_collision_count_delta": sum(row["proxy_collision_count_delta"] for row in rows),
        "safety_violation_count_delta": sum(row["safety_violation_count_delta"] for row in rows),
        "taper_miss_count_delta": sum(row["taper_miss_delta"] for row in rows),
        "mean_taper_miss_delta": sum(row["taper_miss_delta"] for row in rows) / len(rows),
        "min_distance_le_collision_threshold_count_delta": sum(
            row["min_distance_le_collision_threshold_count_delta"] for row in rows
        ),
        "mean_completion_time_delta": sum(row["completion_time_delta"] for row in rows) / len(rows),
        "mean_ego_speed_delta": sum(row["ego_speed_mean_delta"] for row in rows) / len(rows),
        "mean_hard_brake_rate_delta": sum(row["hard_brake_rate_delta"] for row in rows) / len(rows),
        "mean_intervention_delta": sum(row["intervention_delta"] for row in rows) / len(rows),
        "mean_actual_replacement_delta": sum(row["actual_replacement_delta"] for row in rows) / len(rows),
        "mean_task_replacement_delta": sum(row["task_replacement_delta"] for row in rows) / len(rows),
        "task_replacement_count_delta": sum(row["task_replacement_delta"] for row in rows),
        "mean_forecast_ranking_replacement_delta": (
            sum(row["forecast_ranking_replacement_delta"] for row in rows) / len(rows)
        ),
        "forecast_ranking_replacement_count_delta": sum(
            row["forecast_ranking_replacement_delta"] for row in rows
        ),
        "mean_fallback_delta": sum(row["fallback_delta"] for row in rows) / len(rows),
        "mean_emergency_fallback_delta": sum(row["emergency_fallback_delta"] for row in rows) / len(rows),
        "emergency_fallback_count_delta": sum(row["emergency_fallback_delta"] for row in rows),
        "deadline_missed_safe_merge_count_delta": sum(row["deadline_missed_safe_merge_delta"] for row in rows),
        "mean_deadline_missed_safe_merge_delta": (
            sum(row["deadline_missed_safe_merge_delta"] for row in rows) / len(rows)
        ),
        "no_merge_request_before_taper_count_delta": sum(row["no_merge_request_before_taper_delta"] for row in rows),
    }


def _shield_acceptance(baseline: dict | None, shielded: dict | None) -> dict:
    if not baseline or not shielded or "metrics" not in baseline or "metrics" not in shielded:
        return {"available": False, "shield_regression": False}
    base = baseline["metrics"]
    shield = shielded["metrics"]
    checks = {
        "reward_not_degraded": float(shield["average_reward"]) >= float(base["average_reward"]) - 5.0,
        "near_miss_not_worse": float(shield["near_miss_rate"]) <= float(base["near_miss_rate"]),
        "min_distance_not_degraded": float(shield["min_distance_p1"]) >= float(base["min_distance_p1"]) - 1.0,
        "safety_violation_not_worse": float(shield.get("safety_violation_rate", 0.0))
        <= float(base.get("safety_violation_rate", 0.0)),
        "proxy_collision_zero": float(shield.get("proxy_collision_rate", 0.0)) == 0.0,
        "fallback_rate_low": float(shield["fallback_rate"]) < 0.10,
        "fallback_rate_zero": float(shield["fallback_rate"]) == 0.0,
    }
    return {
        "available": True,
        "checks": checks,
        "shield_regression": not all(checks.values()),
    }


def _forecast_acceptance(reference: dict | None, candidate: dict | None) -> dict:
    if not reference or not candidate or "metrics" not in reference or "metrics" not in candidate:
        return {"available": False, "forecast_regression": False}
    ref = reference["metrics"]
    item = candidate["metrics"]
    checks = {
        "reward_not_degraded": float(item["average_reward"]) >= float(ref["average_reward"]) - 5.0,
        "near_miss_not_worse": float(item["near_miss_rate"]) <= float(ref["near_miss_rate"]),
        "min_distance_not_degraded": float(item["min_distance_p1"]) >= float(ref["min_distance_p1"]) - 1.0,
        "safety_violation_not_worse": float(item.get("safety_violation_rate", 0.0))
        <= float(ref.get("safety_violation_rate", 0.0)),
        "proxy_collision_zero": float(item.get("proxy_collision_rate", 0.0)) == 0.0,
    }
    return {
        "available": True,
        "checks": checks,
        "forecast_regression": not all(checks.values()),
    }


def _forecast_baseline_group(group_reports: dict) -> str | None:
    for name in group_reports:
        if name in ("ppo", "ppo_shield", "full_prediction_shield") or str(name).endswith("shield"):
            continue
        report = group_reports.get(name) or {}
        if report.get("forecast_source") or int(report.get("env_observation_shape", [0])[0]) > 52:
            return str(name)
    for name in group_reports:
        if name not in ("ppo", "ppo_shield", "full_prediction_shield") and not str(name).endswith("shield"):
            return str(name)
    return None


def _add_delta(target: dict, key: str, a_report: dict | None, b_report: dict | None) -> None:
    delta = _paired_delta(a_report, b_report)
    if delta is not None:
        target[key] = delta


def _normalised_pairs(cfg_stage5: Any) -> list[dict[str, Any]]:
    return [dict(item) for item in list(cfg_stage5.get("pairs", []) or [])]


def _build_paired_delta(group_reports: dict, pairs: list[dict[str, Any]] | None = None) -> dict:
    paired_delta: dict = {}
    for pair in pairs or []:
        _add_delta(
            paired_delta,
            str(pair["name"]),
            group_reports.get(str(pair["left"])),
            group_reports.get(str(pair["right"])),
        )
    _add_delta(paired_delta, "ppo_vs_ppo_shield", group_reports.get("ppo"), group_reports.get("ppo_shield"))
    for base_name, shield_name in (
        ("ppo_cv_features", "cv_prediction_shield"),
        ("ppo_wcdt_features", "wcdt_prediction_shield"),
        ("ppo_wcdt_v2_features", "wcdt_v2_prediction_shield"),
        ("ppo_wcdt_v3_features", "wcdt_v3_prediction_shield"),
    ):
        _add_delta(
            paired_delta,
            f"{base_name}_vs_{shield_name}",
            group_reports.get(base_name),
            group_reports.get(shield_name),
        )
    _add_delta(paired_delta, "ppo_vs_ppo_cv_features", group_reports.get("ppo"), group_reports.get("ppo_cv_features"))
    _add_delta(
        paired_delta,
        "ppo_cv_features_vs_ppo_wcdt_features",
        group_reports.get("ppo_cv_features"),
        group_reports.get("ppo_wcdt_features"),
    )
    _add_delta(
        paired_delta,
        "ppo_cv_features_vs_ppo_wcdt_v2_features",
        group_reports.get("ppo_cv_features"),
        group_reports.get("ppo_wcdt_v2_features"),
    )
    _add_delta(
        paired_delta,
        "ppo_wcdt_features_vs_ppo_wcdt_v2_features",
        group_reports.get("ppo_wcdt_features"),
        group_reports.get("ppo_wcdt_v2_features"),
    )
    _add_delta(
        paired_delta,
        "ppo_cv_features_vs_ppo_wcdt_v3_features",
        group_reports.get("ppo_cv_features"),
        group_reports.get("ppo_wcdt_v3_features"),
    )
    _add_delta(
        paired_delta,
        "ppo_wcdt_v2_features_vs_ppo_wcdt_v3_features",
        group_reports.get("ppo_wcdt_v2_features"),
        group_reports.get("ppo_wcdt_v3_features"),
    )
    legacy_forecast = _forecast_baseline_group(group_reports)
    if "full_prediction_shield" in group_reports and legacy_forecast:
        _add_delta(
            paired_delta,
            f"{legacy_forecast}_vs_full_prediction_shield",
            group_reports.get(legacy_forecast),
            group_reports.get("full_prediction_shield"),
        )
    return paired_delta


def _build_paired_statistics(
    group_reports: dict[str, dict],
    pairs: list[dict[str, Any]] | None = None,
    *,
    statistics_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = dict(statistics_config or {})
    confidence = float(config.get("confidence", 0.95))
    replicates = int(config.get("bootstrap_replicates", 10_000))
    bootstrap_seed = int(config.get("bootstrap_seed", 0))
    requested = list(pairs or [])
    if not requested:
        defaults = (
            ("ppo_vs_ppo_shield", "ppo", "ppo_shield"),
            ("ppo_cv_features_vs_ppo_wcdt_v3_features", "ppo_cv_features", "ppo_wcdt_v3_features"),
            (
                "ppo_wcdt_v3_features_vs_wcdt_v3_prediction_shield",
                "ppo_wcdt_v3_features",
                "wcdt_v3_prediction_shield",
            ),
        )
        requested = [
            {"name": name, "left": left, "right": right}
            for name, left, right in defaults
            if left in group_reports and right in group_reports
        ]
    reports: dict[str, Any] = {}
    for index, pair in enumerate(requested):
        left_name = str(pair["left"])
        right_name = str(pair["right"])
        left = group_reports.get(left_name)
        right = group_reports.get(right_name)
        if left is None or right is None:
            reports[str(pair["name"])] = {
                "available": False,
                "left": left_name,
                "right": right_name,
                "reason": "missing_group_report",
            }
            continue
        reports[str(pair["name"])] = {
            "available": True,
            "left": left_name,
            "right": right_name,
            "statistics": build_pair_statistics(
                left,
                right,
                confidence=confidence,
                replicates=replicates,
                seed=bootstrap_seed + index * 100_000,
            ),
        }
    return {
        "confidence": confidence,
        "bootstrap_replicates": replicates,
        "bootstrap_seed": bootstrap_seed,
        "pairs": reports,
    }


def _episode_records(report: dict) -> dict[int, dict]:
    return {int(item["seed"]): item for item in report.get("episodes", []) if "seed" in item}


def _action_sequence(episode: dict, key: str) -> list[int]:
    return [int(item.get(key, -999)) for item in list(episode.get("action_execution_records", []) or [])]


def _mainline_reward_v2_acceptance(
    baseline: dict | None,
    candidate: dict | None,
    *,
    profile_config: dict[str, Any] | None = None,
) -> dict:
    if not baseline or not candidate or "metrics" not in baseline or "metrics" not in candidate:
        return {"available": False, "regression": False}
    profile_config = dict(profile_config or {})
    max_replacement = float(profile_config.get("max_actual_replacement_rate", 0.05))
    base = baseline["metrics"]
    item = candidate["metrics"]
    checks = {
        "proxy_collision_not_worse": float(item.get("proxy_collision_rate", 0.0))
        <= float(base.get("proxy_collision_rate", 0.0)),
        "safety_violation_not_worse": float(item.get("safety_violation_rate", 0.0))
        <= float(base.get("safety_violation_rate", 0.0)),
        "geometric_overlap_not_worse": float(item.get("geometric_overlap_rate", 0.0))
        <= float(base.get("geometric_overlap_rate", 0.0)),
        "fallback_rate_zero": float(item.get("fallback_rate", 0.0)) == 0.0,
        "taper_miss_not_worse": float(item.get("taper_miss_rate", 0.0))
        <= float(base.get("taper_miss_rate", 0.0)),
        "timely_merge_success_not_worse": float(item.get("timely_merge_success_rate", 0.0))
        >= float(base.get("timely_merge_success_rate", 0.0)),
        "actual_replacement_rate_within_limit": float(item.get("actual_replacement_rate", 0.0))
        <= max_replacement,
    }
    return {
        "available": True,
        "profile": "mainline_reward_v2",
        "max_actual_replacement_rate": max_replacement,
        "checks": checks,
        "regression": not all(checks.values()),
    }


def _shadow_noop_acceptance(
    baseline: dict | None,
    shadow: dict | None,
    *,
    profile_config: dict[str, Any] | None = None,
) -> dict:
    if not baseline or not shadow or "episodes" not in baseline or "episodes" not in shadow:
        return {"available": False, "regression": False}
    profile_config = dict(profile_config or {})
    reward_tolerance = float(profile_config.get("reward_tolerance", 1.0e-6))
    right = _episode_records(shadow)
    mismatches: list[dict[str, Any]] = []
    compared = 0
    for left_episode in baseline.get("episodes", []):
        seed = int(left_episode.get("seed", -1))
        right_episode = right.get(seed)
        if right_episode is None:
            mismatches.append({"seed": seed, "reason": "missing_shadow_episode"})
            continue
        compared += 1
        checks = {
            "raw_actions": _action_sequence(left_episode, "raw_action")
            == _action_sequence(right_episode, "raw_action"),
            "safety_shield_actions": _action_sequence(left_episode, "safety_shield_action")
            == _action_sequence(right_episode, "safety_shield_action"),
            "final_actions": _action_sequence(left_episode, "final_action")
            == _action_sequence(right_episode, "final_action"),
            "done_reason": str(left_episode.get("done_reason", "")) == str(right_episode.get("done_reason", "")),
            "taper_miss": bool(left_episode.get("taper_miss", False))
            == bool(right_episode.get("taper_miss", False)),
            "merge_success": bool(left_episode.get("merge_success", False))
            == bool(right_episode.get("merge_success", False)),
            "proxy_collision": bool(left_episode.get("proxy_collision", False))
            == bool(right_episode.get("proxy_collision", False)),
            "safety_violation": bool(left_episode.get("safety_violation", False))
            == bool(right_episode.get("safety_violation", False)),
            "geometric_overlap": bool(left_episode.get("geometric_overlap", False))
            == bool(right_episode.get("geometric_overlap", False)),
            "episode_reward": abs(
                float(left_episode.get("episode_reward", 0.0))
                - float(right_episode.get("episode_reward", 0.0))
            )
            <= reward_tolerance,
            "no_accvp_replacement": int(right_episode.get("accvp_replacement_count", 0)) == 0,
        }
        if not all(checks.values()):
            mismatches.append({"seed": seed, "failed_checks": [name for name, ok in checks.items() if not ok]})
    checks = {
        "all_expected_seeds_compared": compared == len(baseline.get("episodes", [])),
        "raw_action_sequence_identical": not any(
            "raw_actions" in item.get("failed_checks", []) for item in mismatches
        ),
        "safety_shield_action_sequence_identical": not any(
            "safety_shield_actions" in item.get("failed_checks", []) for item in mismatches
        ),
        "final_action_sequence_identical": not any(
            "final_actions" in item.get("failed_checks", []) for item in mismatches
        ),
        "outcomes_identical": not any(
            set(item.get("failed_checks", [])).intersection(
                {"done_reason", "taper_miss", "merge_success", "proxy_collision", "safety_violation", "geometric_overlap"}
            )
            for item in mismatches
        ),
        "episode_reward_identical_within_tolerance": not any(
            "episode_reward" in item.get("failed_checks", []) for item in mismatches
        ),
        "no_accvp_replacements": not any(
            "no_accvp_replacement" in item.get("failed_checks", []) for item in mismatches
        ),
    }
    return {
        "available": True,
        "profile": "shadow_noop",
        "reward_tolerance": reward_tolerance,
        "checks": checks,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches[:20],
        "regression": bool(mismatches) or not all(checks.values()),
    }


def _configured_pair_acceptance(group_reports: dict, cfg_stage5: Any) -> dict:
    acceptance: dict = {}
    profile_configs = dict(cfg_stage5.get("acceptance", {}) or {})
    for pair in _normalised_pairs(cfg_stage5):
        name = str(pair["name"])
        left = group_reports.get(str(pair["left"]))
        right = group_reports.get(str(pair["right"]))
        profile = str(pair.get("acceptance_profile", "")).strip()
        profile_config = dict(profile_configs.get(profile, {}) or {})
        if profile == "mainline_reward_v2":
            acceptance[name] = _mainline_reward_v2_acceptance(left, right, profile_config=profile_config)
        elif profile == "shadow_noop":
            acceptance[name] = _shadow_noop_acceptance(left, right, profile_config=profile_config)
        else:
            acceptance[name] = {
                "available": False,
                "profile": profile,
                "regression": False,
                "reason": f"unknown acceptance_profile={profile}",
            }
    return acceptance


def _build_acceptance(group_reports: dict, cfg_stage5: Any | None = None) -> dict:
    acceptance: dict = {}
    if cfg_stage5 is not None:
        acceptance.update(_configured_pair_acceptance(group_reports, cfg_stage5))
    if "ppo_shield" in group_reports:
        acceptance["ppo_shield"] = _shield_acceptance(group_reports.get("ppo"), group_reports.get("ppo_shield"))
    if "cv_prediction_shield" in group_reports:
        acceptance["cv_prediction_shield"] = _shield_acceptance(
            group_reports.get("ppo_cv_features"),
            group_reports.get("cv_prediction_shield"),
        )
    if "wcdt_prediction_shield" in group_reports:
        acceptance["wcdt_prediction_shield"] = _shield_acceptance(
            group_reports.get("ppo_wcdt_features"),
            group_reports.get("wcdt_prediction_shield"),
        )
    if "wcdt_v2_prediction_shield" in group_reports:
        acceptance["wcdt_v2_prediction_shield"] = _shield_acceptance(
            group_reports.get("ppo_wcdt_v2_features"),
            group_reports.get("wcdt_v2_prediction_shield"),
        )
    if "wcdt_v3_prediction_shield" in group_reports:
        acceptance["wcdt_v3_prediction_shield"] = _shield_acceptance(
            group_reports.get("ppo_wcdt_v3_features"),
            group_reports.get("wcdt_v3_prediction_shield"),
        )
    if "ppo_cv_features" in group_reports:
        acceptance["forecast_cv_vs_baseline"] = _forecast_acceptance(
            group_reports.get("ppo"),
            group_reports.get("ppo_cv_features"),
        )
    if "ppo_wcdt_features" in group_reports and "ppo_cv_features" in group_reports:
        acceptance["forecast_wcdt_vs_cv"] = _forecast_acceptance(
            group_reports.get("ppo_cv_features"),
            group_reports.get("ppo_wcdt_features"),
        )
    if "ppo_wcdt_v2_features" in group_reports and "ppo_cv_features" in group_reports:
        acceptance["forecast_wcdt_v2_vs_cv"] = _forecast_acceptance(
            group_reports.get("ppo_cv_features"),
            group_reports.get("ppo_wcdt_v2_features"),
        )
    if "ppo_wcdt_v3_features" in group_reports and "ppo_cv_features" in group_reports:
        acceptance["forecast_wcdt_v3_vs_cv"] = _forecast_acceptance(
            group_reports.get("ppo_cv_features"),
            group_reports.get("ppo_wcdt_v3_features"),
        )
    legacy_forecast = _forecast_baseline_group(group_reports)
    if "full_prediction_shield" in group_reports and legacy_forecast:
        acceptance["full_prediction_shield"] = _shield_acceptance(
            group_reports.get(legacy_forecast),
            group_reports.get("full_prediction_shield"),
        )
    return acceptance


def _comparison_tables(group_reports: dict) -> dict[str, dict]:
    policy: dict[str, dict] = {}
    shield: dict[str, dict] = {}
    high_impact: dict[str, dict] = {}
    for name, report in group_reports.items():
        lowered = str(name).lower()
        metrics = dict(report.get("metrics", {}) or {})
        if "task_backstop" in lowered or "full_ranking" in lowered:
            high_impact[name] = metrics
        elif bool(report.get("shield_enabled", False)):
            shield[name] = metrics
        else:
            policy[name] = metrics
    return {
        "policy_comparison": policy,
        "shield_ablation": shield,
        "high_impact_controller_ablation": high_impact,
    }


def _training_seed_summary(group_reports: dict) -> dict[str, dict]:
    grouped: dict[str, list[dict]] = {}
    for name, report in group_reports.items():
        base = re.sub(r"_seed_\d+$", "", str(name))
        grouped.setdefault(base, []).append(dict(report.get("metrics", {}) or {}))
    result: dict[str, dict] = {}
    for name, metrics_rows in grouped.items():
        numeric_keys = {
            key
            for row in metrics_rows
            for key, value in row.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
        result[name] = {
            "training_seed_count": len(metrics_rows),
            "mean": {
                key: sum(float(row.get(key, 0.0)) for row in metrics_rows) / len(metrics_rows)
                for key in sorted(numeric_keys)
            },
            "std": {
                key: (
                    sum(
                        (float(row.get(key, 0.0))
                        - sum(float(item.get(key, 0.0)) for item in metrics_rows) / len(metrics_rows)) ** 2
                        for row in metrics_rows
                    )
                    / len(metrics_rows)
                ) ** 0.5
                for key in sorted(numeric_keys)
            },
        }
    return result


def run(cfg) -> Path:
    stage_dir = prepare_run_dir(cfg, "stage5")
    seeds = _select_eval_seeds(cfg)
    risk_checkpoint = str(_risk_path(cfg))
    default_model = _default_model_path(cfg)
    protocol_cfg = dict(cfg.get("evaluation_protocol", {}) or {})
    stage5_role = str(protocol_cfg.get("stage5_role", "stage5_confirmatory"))
    lineage_artifacts: dict[str, str | Path | None] = {"risk_checkpoint": risk_checkpoint}
    group_model_paths: dict[str, Path] = {}
    for group in cfg.stage5.groups:
        model = _group_model_path(group, default_model)
        if model is not None:
            group_model_paths[str(group.name)] = model
            lineage_artifacts[f"ppo_model:{group.name}"] = model
            group_cfg = clone_with_overrides(cfg, _group_overrides(group))
            if RiskGatedACCVPCandidateTableAugmentor.enabled(group_cfg):
                manifest_source = group_cfg.accvp.get("artifact_manifest")
                if not manifest_source:
                    raise EvidenceProtocolError(
                        f"ACCVP Stage5 group {group.name!r} requires artifact_manifest"
                    )
                lineage_artifacts[f"accvp_manifest:{group.name}"] = str(
                    manifest_source
                )
        if group.get("forecast_checkpoint"):
            lineage_artifacts[f"forecast_checkpoint:{group.name}"] = str(group.forecast_checkpoint)
    if cfg.stage5.get("accvp_observation_preflight_report"):
        lineage_artifacts["accvp_observation_preflight_report"] = str(
            cfg.stage5.accvp_observation_preflight_report
        )
    frozen_runtime_preflight = _validate_frozen_runtime_preflight(
        cfg=cfg,
        group_model_paths=group_model_paths,
        protocol_strict=bool(protocol_cfg.get("strict", False)),
    )
    evidence_lineage = build_stage_lineage(
        cfg,
        stage="stage5",
        role_seeds={stage5_role: seeds},
        artifact_paths=lineage_artifacts,
    )
    model_lineage = {
        name: _validate_stage5_model_lineage(
            model_path=path,
            stage5_lineage=evidence_lineage,
            stage5_seeds=seeds,
        )
        for name, path in group_model_paths.items()
    }
    stage_log("stage5", f"run_id={cfg.run.run_id}")
    stage_log("stage5", f"seeds={seeds}")
    stage_log("stage5", f"default_model={default_model}")
    stage_log("stage5", f"risk_checkpoint={risk_checkpoint}")
    stage_log("stage5", f"output_dir={stage_dir}")
    tb = TensorboardLogger(stage_dir / "tensorboard", enabled=bool(cfg.run.get("tensorboard", True)))
    replay_dir = stage_dir / "replay"
    group_reports = {}
    accvp_group_bindings: dict[str, dict[str, Any]] = {}
    for group_idx, group in enumerate(cfg.stage5.groups):
        group_cfg = clone_with_overrides(cfg, _group_overrides(group))
        if str(group_cfg.rl.get("reward_profile", "default")) in {"shield_guided_forecast", "merge_timing_forecast"}:
            group_cfg.rl.shield_guided_reward["risk_checkpoint"] = str(
                group_cfg.rl.shield_guided_reward.get("risk_checkpoint") or risk_checkpoint
            )
        policy_type = str(group.get("policy_type", "sb3_ppo"))
        if policy_type == "rule_gap_acceptance" and (
            bool(group.shield) or bool(group.forecast_features)
        ):
            raise ValueError(
                "rule_gap_acceptance is an unshielded current-state baseline; "
                "do not enable Shield or forecast features in this comparison group."
            )
        if bool(group_cfg.accvp.get("enabled", False)) and not bool(group.shield):
            raise ValueError(f"stage5 group '{group.name}' enables ACCVP but disables Safety Shield")
        model_path = group_model_paths.get(str(group.name))
        observation_contract = (
            _validate_stage5_observation_contract(
                model_path=model_path,
                group_cfg=group_cfg,
                protocol_strict=bool(evidence_lineage.get("protocol_strict", False)),
            )
            if model_path is not None
            else {"required": False, "available": True}
        )
        stage_log(
            "stage5",
            f"group={group.name} policy_type={policy_type} forecast={bool(group.forecast_features)} shield={bool(group.shield)} model={model_path}",
        )
        group_reports[group.name] = evaluate_policy(
            group_cfg,
            model_path,
            seeds=seeds,
            shield_enabled=bool(group.shield),
            risk_checkpoint=risk_checkpoint if bool(group.shield) else None,
            replay_dir=replay_dir if bool(cfg.stage5.get("replay_enabled", True)) and bool(cfg.run.get("replay", True)) else None,
            group_name=str(group.name),
            tensorboard=tb,
            tensorboard_step_offset=group_idx * max(1, len(seeds)),
            policy_type=policy_type,
        )
        group_reports[group.name]["stage3_observation_contract_validation"] = observation_contract
        bundle_lineage = observation_contract.get("accvp_bundle")
        if isinstance(bundle_lineage, dict):
            accvp_group_bindings[str(group.name)] = dict(bundle_lineage)
            group_reports[group.name]["accvp_bundle_lineage"] = dict(bundle_lineage)
        if bool(group_cfg.accvp.get("observation", {}).get("enabled", False)):
            runtime_gate = _accvp_observation_runtime_gate(dict(group_reports[group.name].get("metrics", {}) or {}))
            group_reports[group.name]["accvp_observation_preflight_pass"] = bool(runtime_gate.get("pass", False))
            group_reports[group.name]["accvp_observation_runtime_gate"] = runtime_gate
            group_reports[group.name]["accvp_table_runtime_gate_pass"] = bool(runtime_gate.get("pass", False))
            if bool(cfg.stage5.get("require_accvp_observation_runtime_gate", False)) and not bool(
                runtime_gate.get("pass", False)
            ):
                raise RuntimeError(
                    f"Stage5 group '{group.name}' failed ACCVP observation runtime gate: "
                    f"{runtime_gate.get('checks', {})}"
                )
        if bool(group.forecast_features):
            group_reports[group.name]["forecast_source"] = str(
                group.get("forecast_source", group_cfg.forecast_features.get("source", ""))
            )
            group_reports[group.name]["forecast_checkpoint"] = str(group_cfg.forecast_features.get("checkpoint", ""))
        else:
            group_reports[group.name]["forecast_source"] = None
            group_reports[group.name]["forecast_checkpoint"] = ""
        group_reports[group.name]["shield_enabled"] = bool(group.shield)
        group_reports[group.name]["shield_overrides"] = dict(group.get("shield_overrides", {}) or {})
        group_reports[group.name]["rl_overrides"] = dict(group.get("rl_overrides", {}) or {})
        group_reports[group.name]["risk_module_overrides"] = dict(group.get("risk_module_overrides", {}) or {})
        group_reports[group.name]["policy_type"] = policy_type
        group_reports[group.name]["raw_policy"] = str(group.get("raw_policy", group.name))
        group_reports[group.name]["counterfactual_predictor"] = str(
            group.get("counterfactual_predictor", "")
        )
        group_reports[group.name]["controller"] = str(group.get("controller", ""))
        comparative_metadata = dict(group.get("comparative", {}) or {})
        if comparative_metadata:
            group_reports[group.name]["comparative"] = comparative_metadata
        if str(group.name) in model_lineage:
            group_reports[group.name]["stage3_lineage_validation"] = model_lineage[str(group.name)]

    evidence_lineage.pop("lineage_fingerprint", None)
    evidence_lineage["accvp_group_bindings"] = {
        name: accvp_group_bindings[name]
        for name in sorted(accvp_group_bindings)
    }
    evidence_lineage["lineage_fingerprint"] = stable_hash(evidence_lineage)

    shield_off = {name: report for name, report in group_reports.items() if not bool(report.get("shield_enabled", False))}
    shield_on = {name: report for name, report in group_reports.items() if bool(report.get("shield_enabled", False))}
    forecast_baseline = _forecast_baseline_group(group_reports)
    configured_pairs = _normalised_pairs(cfg.stage5)
    paired_delta = _build_paired_delta(group_reports, configured_pairs)
    paired_statistics = _build_paired_statistics(
        group_reports,
        configured_pairs,
        statistics_config=dict(cfg.stage5.get("statistics", {}) or {}),
    )
    acceptance = _build_acceptance(group_reports, cfg.stage5)
    write_report(stage_dir / "shield_off_metrics.json", shield_off)
    write_report(stage_dir / "shield_on_metrics.json", shield_on)
    report = {
        "stage": "stage5",
        "paired_eval": bool(cfg.stage5.paired_eval),
        "safety_metric_version": str(
            cfg.risk_module.get("safety_metric_version", SAFETY_METRIC_VERSION)
        ),
        "seeds": seeds,
        "experiment": dict(cfg.get("experiment", {}) or {}),
        "groups": group_reports,
        "configured_pairs": configured_pairs,
        "forecast_baseline_group": forecast_baseline,
        "paired_delta": paired_delta,
        "paired_statistics": paired_statistics,
        "acceptance": acceptance,
        "comparison_tables": _comparison_tables(group_reports),
        "training_seed_summary": _training_seed_summary(group_reports),
        "evidence_lineage": evidence_lineage,
        "stage3_lineage_validation": model_lineage,
        "frozen_accvp_runtime_preflight": frozen_runtime_preflight,
    }
    write_report(stage_dir / "formal_paired_eval_report.json", report)
    tb.close()
    stage_log("stage5", f"shield_off_metrics={stage_dir / 'shield_off_metrics.json'}")
    stage_log("stage5", f"shield_on_metrics={stage_dir / 'shield_on_metrics.json'}")
    stage_log("stage5", f"report={stage_dir / 'formal_paired_eval_report.json'}")
    return stage_dir


def main() -> None:
    args = parse_config_arg("Stage5 paired shield evaluation")
    cfg = load_stage_config(args)
    run(cfg)


if __name__ == "__main__":
    main()
