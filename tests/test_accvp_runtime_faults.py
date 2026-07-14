from __future__ import annotations

import json

import pytest

from safe_rl.accvp.verification.fault_injection import (
    PREDICTOR_STAGE,
    RISK_STAGE,
    SUPPORTED_FAULTS,
    FaultEvent,
    FaultSchedule,
    assert_fault_injection_allowed,
)
from safe_rl.evaluation_protocol import stable_hash
from safe_rl.pipeline.accvp_runtime_fault_audit import build_fault_audit, run


def test_fault_schedule_is_explicit_keyed_and_order_deterministic():
    events = [
        FaultEvent(7, 2, RISK_STAGE, "nan"),
        FaultEvent(7, 1, PREDICTOR_STAGE, "timeout"),
    ]
    first = FaultSchedule(events)
    second = FaultSchedule(reversed(events))
    assert first.to_dict() == second.to_dict()
    assert first.fingerprint == second.fingerprint
    assert first.event_for({"episode_seed": 7, "decision_index": 1}, PREDICTOR_STAGE).fault == "timeout"
    assert first.event_for({"episode_seed": 7, "decision_index": 1}, RISK_STAGE) is None
    assert first.event_for({"episode_seed": 8, "decision_index": 1}, PREDICTOR_STAGE) is None

    with pytest.raises(ValueError, match="duplicate fault-injection schedule key"):
        FaultSchedule([events[0], events[0]])


def test_random_or_formal_fault_injection_is_prohibited():
    with pytest.raises(ValueError, match="random fault injection"):
        FaultSchedule.from_dict({"random_probability": 0.1, "events": []})
    schedule = FaultSchedule([FaultEvent(1, 1, PREDICTOR_STAGE, "exception")])
    with pytest.raises(ValueError, match="formal training/evaluation"):
        assert_fault_injection_allowed(schedule, formal_pipeline=True)
    with pytest.raises(ValueError, match="random fault injection"):
        assert_fault_injection_allowed(FaultSchedule(), formal_pipeline=False, random_injection=True)
    assert_fault_injection_allowed(FaultSchedule(), formal_pipeline=True)


def test_actual_augmentor_fault_audit_covers_every_fault_and_recovery_contract():
    report = build_fault_audit()
    assert report["pass"] is True
    assert report["scenario_count"] == 2 * len(SUPPORTED_FAULTS)
    assert set(report["fault_kinds"]) == set(SUPPORTED_FAULTS)
    assert set(report["stages"]) == {PREDICTOR_STAGE, RISK_STAGE}
    assert report["scope"]["actual_candidate_table_augmentor"] is True
    assert report["scope"]["real_sumo_executed"] is False
    assert report["scope"]["hard_real_time_claim"] is False
    assert report["deadline_contract"]["hard_deadline_enforced"] is False
    assert report["deadline_contract"]["soft_deadline_can_preempt_hung_delegate"] is False
    assert report["formal_pipeline_policy"]["random_injection_supported"] is False
    for scenario in report["scenarios"]:
        assert scenario["pass"] is True
        assert scenario["triggered_event_count"] == 3
        assert all(scenario["checks"].values())
        assert scenario["observations"]["first_fault"]["freshness"]["stale_fallback_active"] is True
        assert scenario["observations"]["first_fault"]["all_risk_closed"] is True
        assert scenario["observations"]["continuous_fault"]["freshness"]["hard_default_active"] is True
        assert scenario["observations"]["healthy_recovery"]["all_table_valid"] is True
        assert scenario["observations"]["cross_episode_fault"]["all_zero_viability"] is True
        assert scenario["augmentor_counters"]["bounded_stale_reuse_count"] == 1
    fingerprint = report.pop("report_fingerprint")
    assert fingerprint == stable_hash(report)


def test_fault_audit_is_deterministic_and_output_is_immutable(tmp_path):
    kwargs = {
        "fault_kinds": ["timeout", "nan"],
        "stages": [PREDICTOR_STAGE, RISK_STAGE],
        "soft_deadline_s": 0.25,
        "reference_hard_deadline_s": 0.5,
    }
    first = build_fault_audit(**kwargs)
    second = build_fault_audit(**kwargs)
    assert first == second
    output = tmp_path / "fault_audit.json"
    assert run(output, **kwargs) == output
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload == first
    assert payload["schedule_fingerprint"]
    assert payload["report_fingerprint"]
    with pytest.raises(FileExistsError, match="refusing to overwrite immutable report"):
        run(output, **kwargs)
