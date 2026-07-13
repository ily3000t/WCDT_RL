from __future__ import annotations

import json
from pathlib import Path

from safe_rl.pipeline.run_accvp_vnext_pipeline import _oracle_report_ok


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
