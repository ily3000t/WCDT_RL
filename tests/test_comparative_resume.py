from __future__ import annotations

from pathlib import Path

import pytest

from safe_rl.pipeline.stage2_train_prediction_risk import (
    _reference_risk_report,
    _referenced_risk_checkpoint,
)
from safe_rl.pipeline.train_comparative_suite import (
    _forecast_settings,
    _validate_existing_wcdt_v1_checkpoint,
    _validate_input_provenance,
)
from safe_rl.prediction.actor_selector import actor_selection_config_hash
from safe_rl.utils.config import clone_with_overrides, load_config
from safe_rl.utils.stage1_dataset import STAGE1_BUFFER_SCHEMA_VERSION


def test_predictor_only_risk_reference_requires_existing_checkpoint(tmp_path: Path):
    cfg = clone_with_overrides(load_config(), {"stage2": {"risk_checkpoint_reference": None}})
    with pytest.raises(ValueError, match="requires stage2.risk_checkpoint_reference"):
        _referenced_risk_checkpoint(cfg)

    checkpoint = tmp_path / "risk_module.pt"
    checkpoint.write_bytes(b"reference")
    cfg = clone_with_overrides(
        load_config(), {"stage2": {"risk_checkpoint_reference": str(checkpoint)}}
    )
    resolved = _referenced_risk_checkpoint(cfg)
    report = _reference_risk_report(resolved)
    assert report["risk_checkpoint"] == str(checkpoint.resolve())
    assert report["risk_checkpoint_source"] == "reference"
    assert report["risk_training_skipped"] is True


def test_comparative_resume_rejects_changed_input_provenance():
    with pytest.raises(ValueError, match="input provenance mismatch"):
        _validate_input_provenance({"risk_checkpoint_sha256": "old"}, {"risk_checkpoint_sha256": "new"})


def test_existing_wcdt_v1_checkpoint_validation_uses_top_level_payload(tmp_path: Path):
    torch = pytest.importorskip("torch")
    cfg = clone_with_overrides(load_config(), {"prediction": {"wcdt_v1_max_agents": 6}})
    checkpoint = tmp_path / "wcdt_predictor.pt"
    torch.save(
        {
            "model_state_dict": {},
            "architecture_version": "wcdt_v1_adapted_multimodal_selector_v2",
            "actor_selection_config_hash": actor_selection_config_hash(cfg),
            "max_actor_count": 6,
            "stage1_buffer_schema_version": STAGE1_BUFFER_SCHEMA_VERSION,
            "trajectory_schema_version": 4,
            "best_epoch": 7,
            "best_val_score": 1.5,
        },
        checkpoint,
    )
    summary = _validate_existing_wcdt_v1_checkpoint(checkpoint, cfg)
    assert summary == {"best_epoch": 7, "best_val_score": 1.5}


def test_forecast_settings_uses_the_source_specific_checkpoint(tmp_path: Path):
    v1 = tmp_path / "wcdt_predictor.pt"
    v3 = tmp_path / "wcdt_v3_predictor.pt"

    assert _forecast_settings("ppo", v1, v3)["forecast_features"]["enabled"] is False
    assert _forecast_settings("constant_velocity", v1, v3)["forecast_features"]["source"] == "constant_velocity"
    assert _forecast_settings("wcdt", v1, v3)["forecast_features"]["checkpoint"] == str(v1)
    assert _forecast_settings("wcdt_v3", v1, v3)["forecast_features"]["checkpoint"] == str(v3)
