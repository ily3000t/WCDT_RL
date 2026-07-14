from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import numpy as np

from safe_rl.accvp.verification.fault_injection import (
    PREDICTOR_STAGE,
    RISK_STAGE,
    SUPPORTED_FAULTS,
    SUPPORTED_STAGES,
    FaultEvent,
    FaultInjectingPredictor,
    FaultInjectingRiskModel,
    FaultSchedule,
    assert_fault_injection_allowed,
)
from safe_rl.accvp.serving.observation import (
    RISK_GATED_ACCVP_OBSERVATION_BOUNDED_STALE_VERSION,
    RiskGatedACCVPCandidateTableAugmentor,
    validate_accvp_observation_config,
)
from safe_rl.accvp.contracts.schema import write_json_atomic
from safe_rl.evaluation_protocol import stable_hash
from safe_rl.sim.action_space import ACTIONS
from safe_rl.utils.config import clone_with_overrides, load_config


FAULT_AUDIT_SCHEMA_VERSION = 1
FAULT_AUDIT_ARTIFACT_KIND = "accvp_runtime_fault_audit_v1"


class _HealthyPredictor:
    def score_candidates(self, context, actions, *, timeout_s=None):
        return [
            {
                "action_id": int(action.index),
                "p_merge_before_taper": 0.8,
                "target_lane_entry_time_s": 2.0,
                "ensemble_disagreement": 0.05,
            }
            for action in actions
        ]


class _HealthyRiskModel:
    def predict_many(self, actions, context):
        return [
            SimpleNamespace(risk_score=0.1, risk_uncertainty=0.0)
            for _action in actions
        ]


def _audit_config():
    return clone_with_overrides(
        load_config(),
        {
            "forecast_features": {"enabled": False},
            "rl": {"use_wcdt_forecast_features": False},
            "accvp": {
                "enabled": False,
                "mode": "off",
                "checkpoint": "diagnostic-wrapper-only.pt",
                "risk_checkpoint": "diagnostic-wrapper-only-risk.pt",
                "activation_distance": 240.0,
                "observation": {
                    "enabled": True,
                    "feature_version": RISK_GATED_ACCVP_OBSERVATION_BOUNDED_STALE_VERSION,
                    "invalid_table_strategy": "bounded_last_valid_v2",
                    "activation_distance": 240.0,
                    "timeout_s": 0.5,
                    "fail_closed_defaults": True,
                    "include_risk_secondary": True,
                    "secondary_safety_profile": "strict",
                    "warmup_enabled": False,
                    "profile_latency": False,
                    "invalid_table_dropout_rate": 0.0,
                    "last_valid_max_decisions": 1,
                    "last_valid_ttl_s": 0.5,
                    "last_valid_max_merge_distance_delta_m": 15.0,
                    "last_valid_max_ego_speed_delta_mps": 3.0,
                    "last_valid_max_gap_delta_m": 8.0,
                    "allow_with_forecast_features": False,
                },
            },
        },
    )


def _context(episode_seed: int, decision_index: int) -> dict[str, Any]:
    return {
        "episode_seed": int(episode_seed),
        "episode_step": int(decision_index) * 4,
        "decision_index": int(decision_index),
        "ego": SimpleNamespace(
            edge_id="aux",
            lane_id="aux_0",
            lane_index=0,
            speed=20.0,
            vehicle_id="ego",
        ),
        "merge_local": SimpleNamespace(
            ego_on_auxiliary=True,
            merge_distance=50.0 - float(decision_index),
            target_front_vehicle_id="front",
            target_rear_vehicle_id="rear",
            target_front_gap=20.0 + 0.5 * float(decision_index),
            target_rear_gap=18.0 + 0.5 * float(decision_index),
        ),
        "candidate_legal_by_action": {int(action.index): True for action in ACTIONS},
    }


def _table(vector: np.ndarray) -> np.ndarray:
    return np.asarray(vector[: 9 * 11], dtype=np.float32).reshape((9, 11))


def _observation_record(label: str, vector: np.ndarray) -> dict[str, Any]:
    values = np.asarray(vector, dtype=np.float32)
    table = _table(values)
    freshness = values[-4:]
    return {
        "label": label,
        "finite": bool(np.isfinite(values).all()),
        "all_table_valid": bool(np.all(table[:, 0] == 1.0)),
        "all_table_invalid": bool(np.all(table[:, 0] == 0.0)),
        "all_risk_pass": bool(np.all(table[:, 2] == 1.0)),
        "all_risk_closed": bool(np.all(table[:, 2] == 0.0)),
        "all_conservative_risk_score": bool(np.all(table[:, 3] == 1.0)),
        "all_zero_viability": bool(np.all(table[:, 4] == 0.0)),
        "mean_viability": float(np.mean(table[:, 4])),
        "freshness": {
            "stale_fallback_active": bool(freshness[0] == 1.0),
            "hard_default_active": bool(freshness[1] == 1.0),
            "table_age_norm": float(freshness[2]),
            "context_delta_norm": float(freshness[3]),
        },
    }


def _expected_checks(records: dict[str, dict[str, Any]]) -> dict[str, bool]:
    fresh = records["fresh"]
    stale = records["first_fault"]
    hard = records["continuous_fault"]
    recovered = records["healthy_recovery"]
    cross_episode = records["cross_episode_fault"]
    return {
        "fresh_table_is_valid": bool(
            fresh["finite"]
            and fresh["all_table_valid"]
            and fresh["all_risk_pass"]
            and not fresh["freshness"]["stale_fallback_active"]
            and not fresh["freshness"]["hard_default_active"]
        ),
        "first_fault_uses_bounded_stale": bool(
            stale["finite"]
            and stale["all_table_invalid"]
            and stale["all_risk_closed"]
            and stale["all_conservative_risk_score"]
            and stale["freshness"]["stale_fallback_active"]
            and not stale["freshness"]["hard_default_active"]
            and stale["mean_viability"] == fresh["mean_viability"]
        ),
        "continuous_fault_uses_hard_default": bool(
            hard["finite"]
            and hard["all_table_invalid"]
            and hard["all_risk_closed"]
            and hard["all_zero_viability"]
            and not hard["freshness"]["stale_fallback_active"]
            and hard["freshness"]["hard_default_active"]
        ),
        "healthy_path_recovers_immediately": bool(
            recovered["finite"]
            and recovered["all_table_valid"]
            and recovered["all_risk_pass"]
            and not recovered["freshness"]["stale_fallback_active"]
            and not recovered["freshness"]["hard_default_active"]
        ),
        "cross_episode_cache_is_not_reused": bool(
            cross_episode["finite"]
            and cross_episode["all_table_invalid"]
            and cross_episode["all_risk_closed"]
            and cross_episode["all_zero_viability"]
            and not cross_episode["freshness"]["stale_fallback_active"]
            and cross_episode["freshness"]["hard_default_active"]
        ),
    }


def _normalise_faults(values: Iterable[str] | None) -> tuple[str, ...]:
    requested = tuple(sorted(set(str(value).strip().lower() for value in (values or SUPPORTED_FAULTS))))
    unsupported = sorted(set(requested) - set(SUPPORTED_FAULTS))
    if unsupported:
        raise ValueError(f"unsupported requested fault kinds: {unsupported}")
    if not requested:
        raise ValueError("fault audit requires at least one fault kind")
    return requested


def _normalise_stages(values: Iterable[str] | None) -> tuple[str, ...]:
    requested = tuple(sorted(set(str(value).strip() for value in (values or SUPPORTED_STAGES))))
    unsupported = sorted(set(requested) - set(SUPPORTED_STAGES))
    if unsupported:
        raise ValueError(f"unsupported requested fault stages: {unsupported}")
    if not requested:
        raise ValueError("fault audit requires at least one stage")
    return requested


def _scenario_seed(stage_index: int, fault_index: int) -> int:
    return 100_000 + stage_index * 100_000 + fault_index * 100


def build_fault_audit(
    *,
    fault_kinds: Iterable[str] | None = None,
    stages: Iterable[str] | None = None,
    soft_deadline_s: float = 0.5,
    reference_hard_deadline_s: float = 0.5,
) -> dict[str, Any]:
    faults = _normalise_faults(fault_kinds)
    selected_stages = _normalise_stages(stages)
    if float(soft_deadline_s) <= 0.0:
        raise ValueError("soft_deadline_s must be positive")
    if float(reference_hard_deadline_s) <= 0.0:
        raise ValueError("reference_hard_deadline_s must be positive")

    events: list[FaultEvent] = []
    scenarios: list[dict[str, Any]] = []
    scenario_specs: list[tuple[str, str, int, int]] = []
    for stage_index, stage in enumerate(selected_stages):
        for fault_index, fault in enumerate(faults):
            seed = _scenario_seed(stage_index, fault_index)
            cross_episode_seed = seed + 50
            scenario_specs.append((stage, fault, seed, cross_episode_seed))
            events.extend(
                [
                    FaultEvent(seed, 1, stage, fault),
                    FaultEvent(seed, 2, stage, fault),
                    FaultEvent(cross_episode_seed, 0, stage, fault),
                ]
            )
    schedule = FaultSchedule(events)
    assert_fault_injection_allowed(schedule, formal_pipeline=False, random_injection=False)

    config = _audit_config()
    validate_accvp_observation_config(config)
    for stage, fault, episode_seed, cross_episode_seed in scenario_specs:
        predictor = FaultInjectingPredictor(
            _HealthyPredictor(),
            schedule,
            soft_deadline_s=float(soft_deadline_s),
        )
        risk_model = FaultInjectingRiskModel(
            _HealthyRiskModel(),
            schedule,
            soft_deadline_s=float(soft_deadline_s),
        )
        shield = SimpleNamespace(ranker=SimpleNamespace(risk_model=risk_model))
        augmentor = RiskGatedACCVPCandidateTableAugmentor(
            config,
            predictor=predictor,
            shield=shield,
        )
        vectors = {
            "fresh": augmentor.extract(_context(episode_seed, 0)),
            "first_fault": augmentor.extract(_context(episode_seed, 1)),
            "continuous_fault": augmentor.extract(_context(episode_seed, 2)),
            "healthy_recovery": augmentor.extract(_context(episode_seed, 3)),
            # Deliberately do not reset the augmentor: this directly exercises
            # the cached episode-identity check rather than relying on an
            # external environment reset to clear state.
            "cross_episode_fault": augmentor.extract(_context(cross_episode_seed, 0)),
        }
        records = {label: _observation_record(label, vector) for label, vector in vectors.items()}
        checks = _expected_checks(records)
        summary = augmentor.summary()
        trigger_records = sorted(
            predictor.trigger_records + risk_model.trigger_records,
            key=lambda row: (
                int(row["episode_seed"]),
                int(row["decision_index"]),
                str(row["stage"]),
            ),
        )
        scenario = {
            "stage": stage,
            "fault": fault,
            "episode_seed": episode_seed,
            "cross_episode_seed": cross_episode_seed,
            "triggered_event_count": len(trigger_records),
            "triggered_events": trigger_records,
            "observations": records,
            "checks": checks,
            "pass": bool(len(trigger_records) == 3 and all(checks.values())),
            "augmentor_counters": {
                "bounded_stale_reuse_count": int(
                    summary.get("accvp_table_bounded_stale_reuse_count", 0)
                ),
                "hard_fail_closed_count": int(
                    summary.get("accvp_table_hard_fail_closed_count", 0)
                ),
                "timeout_count": int(summary.get("accvp_table_timeout_count", 0)),
                "invalid_output_count": int(
                    summary.get("accvp_table_invalid_output_count", 0)
                ),
                "model_error_count": int(summary.get("accvp_table_model_error_count", 0)),
                "max_consecutive_timeout_count": int(
                    summary.get("accvp_table_max_consecutive_timeout_count", 0)
                ),
            },
        }
        scenario["scenario_fingerprint"] = stable_hash(scenario)
        scenarios.append(scenario)

    scenarios.sort(key=lambda row: (str(row["stage"]), str(row["fault"])))
    configuration_contract = {
        "feature_version": RISK_GATED_ACCVP_OBSERVATION_BOUNDED_STALE_VERSION,
        "invalid_table_strategy": "bounded_last_valid_v2",
        "last_valid_max_decisions": 1,
        "last_valid_ttl_s": 0.5,
        "all_candidate_actions_legal": True,
        "warmup_enabled": False,
        "profile_latency": False,
    }
    report: dict[str, Any] = {
        "artifact_kind": FAULT_AUDIT_ARTIFACT_KIND,
        "schema_version": FAULT_AUDIT_SCHEMA_VERSION,
        "pass": bool(all(bool(row["pass"]) for row in scenarios)),
        "scope": {
            "diagnostic_only": True,
            "actual_candidate_table_augmentor": True,
            "synthetic_context": True,
            "real_sumo_executed": False,
            "real_model_checkpoint_executed": False,
            "os_worker_process_terminated": False,
            "worker_crash_semantics": "simulated exception at the worker API boundary",
            "hard_real_time_claim": False,
        },
        "deadline_contract": {
            "soft_deadline_s": float(soft_deadline_s),
            "soft_deadline_enforcement": (
                "cooperative post-return elapsed check plus explicit TimeoutError injection"
            ),
            "soft_deadline_can_preempt_hung_delegate": False,
            "reference_hard_deadline_s": float(reference_hard_deadline_s),
            "hard_deadline_enforced": False,
            "hard_deadline_scope": "reference_only; requires an external process supervisor",
        },
        "formal_pipeline_policy": {
            "fault_injection_allowed": False,
            "random_injection_supported": False,
            "schedule_type": "explicit_episode_decision_stage_keys",
        },
        "configuration_contract": configuration_contract,
        "configuration_contract_sha256": stable_hash(configuration_contract),
        "fault_kinds": list(faults),
        "stages": list(selected_stages),
        "schedule": schedule.to_dict(),
        "schedule_fingerprint": schedule.fingerprint,
        "scenario_count": len(scenarios),
        "scenarios": scenarios,
    }
    report["report_fingerprint"] = stable_hash(report)
    return report


def run(
    output_path: str | Path,
    *,
    fault_kinds: Iterable[str] | None = None,
    stages: Iterable[str] | None = None,
    soft_deadline_s: float = 0.5,
    reference_hard_deadline_s: float = 0.5,
) -> Path:
    output = Path(output_path)
    if output.exists():
        raise FileExistsError(
            f"fault audit output already exists; refusing to overwrite immutable report: {output}"
        )
    report = build_fault_audit(
        fault_kinds=fault_kinds,
        stages=stages,
        soft_deadline_s=soft_deadline_s,
        reference_hard_deadline_s=reference_hard_deadline_s,
    )
    write_json_atomic(output, report)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit bounded-stale ACCVP behavior under deterministic API faults"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--fault", action="append", choices=sorted(SUPPORTED_FAULTS))
    parser.add_argument("--stage", action="append", choices=sorted(SUPPORTED_STAGES))
    parser.add_argument("--soft-deadline-s", type=float, default=0.5)
    parser.add_argument("--reference-hard-deadline-s", type=float, default=0.5)
    args = parser.parse_args()
    output = run(
        args.output,
        fault_kinds=args.fault,
        stages=args.stage,
        soft_deadline_s=args.soft_deadline_s,
        reference_hard_deadline_s=args.reference_hard_deadline_s,
    )
    print(f"accvp_runtime_fault_audit report={output}")


if __name__ == "__main__":
    main()
