from __future__ import annotations

import json
from pathlib import Path

from safe_rl.pipeline.run_accvp_vnext_pipeline import (
    WORKFLOW_CONFIG,
    _load_workflow_contract,
    _oracle_report_ok,
)
from safe_rl.ppo_factorial import (
    EXPECTED_CANDIDATE_METHOD_ROLES,
    EXPECTED_FINAL_METHOD_ID,
)


def _write_oracle(path: Path, **overrides: object) -> None:
    payload: dict[str, object] = {
        "oracle_state": "go",
        "go_for_training": True,
        "root_policy": "merge_timing",
        "cohort_role": "oracle_regression",
        "oracle_only": True,
        "exclude_from_model_splits": True,
        "required_seeds": [2, 5],
    }
    payload.update(overrides)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_oracle_pipeline_gate_requires_merge_timing_scope(tmp_path: Path) -> None:
    report = tmp_path / "oracle_report.json"
    _write_oracle(report)
    assert _oracle_report_ok(report)

    _write_oracle(report, root_policy=None)
    assert not _oracle_report_ok(report)


def test_oracle_pipeline_gate_requires_split_exclusion_and_fixed_cohort(tmp_path: Path) -> None:
    report = tmp_path / "oracle_report.json"
    _write_oracle(report, exclude_from_model_splits=False)
    assert not _oracle_report_ok(report)

    _write_oracle(report, required_seeds=[2])
    assert not _oracle_report_ok(report)


def test_vnext_workflow_unblocks_and_preregisters_factorial_pipeline() -> None:
    _path, workflow = _load_workflow_contract(WORKFLOW_CONFIG)
    assert not dict(workflow["automation"].get("blocked_phases", {}) or {})
    assert workflow["final_method_id"] == EXPECTED_FINAL_METHOD_ID
    assert {
        method_id: workflow["method_roles"][method_id]
        for method_id in EXPECTED_CANDIDATE_METHOD_ROLES
    } == EXPECTED_CANDIDATE_METHOD_ROLES
    factorial = workflow["factorial"]
    assert factorial["runtime_scope"] == "all_candidate_methods"
    assert factorial["candidate_method_ids"] == list(EXPECTED_CANDIDATE_METHOD_ROLES)
    comparisons = list(factorial["comparisons"])
    assert len(comparisons) == 6
    final = [
        row
        for row in comparisons
        if row["comparison_id"] == factorial["final_comparison_id"]
    ]
    assert len(final) == 1
    assert final[0]["right_method_id"] == EXPECTED_FINAL_METHOD_ID
