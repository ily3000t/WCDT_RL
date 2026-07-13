from __future__ import annotations

import pytest

from safe_rl.accvp.runtime_contract import (
    canonical_formal_runtime_contract,
    compare_formal_runtime_contracts,
    formal_runtime_contract_from_config,
    formal_runtime_contract_sha256,
)
from safe_rl.utils.config import load_config


def _observation_contract() -> dict:
    return {
        "enabled": True,
        "feature_version": "risk_gated_candidate_table_v3_bounded_stale",
        "activation_distance": 240.0,
        "include_risk_secondary": True,
        "secondary_safety_profile": "short_horizon_risk_v1",
        "risk_horizon_steps": 8,
        "invalid_table_strategy": "bounded_last_valid_v2",
        "fail_closed_defaults": True,
        "timeout_s": 0.5,
        "timeout_contract": "soft_realtime_post_return_v1",
        "full_table_hard_deadline_worker": False,
        "use_inference_worker": False,
        "profile_latency": True,
        "warmup_enabled": True,
        "warmup_max_attempts": 3,
        "invalid_table_dropout_rate": 0.0,
        "last_valid_max_decisions": 1,
        "last_valid_ttl_s": 0.5,
        "last_valid_max_merge_distance_delta_m": 15.0,
        "last_valid_max_ego_speed_delta_mps": 3.0,
        "last_valid_max_gap_delta_m": 8.0,
    }


def test_formal_runtime_contract_binds_every_safety_relevant_runtime_field():
    contract = canonical_formal_runtime_contract(
        observation=_observation_contract(),
        candidate_geometry_backend="vectorized",
        risk_checkpoint_sha256="a" * 64,
        risk_module_config_sha256="e" * 64,
    )
    assert contract["include_risk_secondary"] is True
    assert contract["risk_horizon_steps"] == 8
    assert contract["candidate_geometry_backend"] == "vectorized"
    assert len(formal_runtime_contract_sha256(contract)) == 64

    changed = dict(contract)
    changed["risk_horizon_steps"] = 1
    comparison = compare_formal_runtime_contracts(contract, changed)
    assert comparison["pass"] is False
    assert comparison["differing_fields"] == ["risk_horizon_steps"]

    changed_risk_config = dict(contract)
    changed_risk_config["risk_module_config_sha256"] = "f" * 64
    comparison = compare_formal_runtime_contracts(contract, changed_risk_config)
    assert comparison["pass"] is False
    assert comparison["differing_fields"] == ["risk_module_config_sha256"]


def test_formal_runtime_contract_rejects_disabled_risk_secondary():
    observation = _observation_contract()
    observation["include_risk_secondary"] = False
    with pytest.raises(ValueError, match="include_risk_secondary=true"):
        canonical_formal_runtime_contract(
            observation=observation,
            candidate_geometry_backend="vectorized",
            risk_checkpoint_sha256="b" * 64,
            risk_module_config_sha256="e" * 64,
        )


def test_formal_runtime_contract_rejects_unaudited_backend_or_dropout():
    with pytest.raises(ValueError, match="vectorized"):
        canonical_formal_runtime_contract(
            observation=_observation_contract(),
            candidate_geometry_backend="reference",
            risk_checkpoint_sha256="c" * 64,
            risk_module_config_sha256="e" * 64,
        )
    observation = _observation_contract()
    observation["invalid_table_dropout_rate"] = 0.01
    with pytest.raises(ValueError, match="invalid_table_dropout_rate=0"):
        canonical_formal_runtime_contract(
            observation=observation,
            candidate_geometry_backend="vectorized",
            risk_checkpoint_sha256="d" * 64,
            risk_module_config_sha256="e" * 64,
        )


def test_vnext_final_config_effective_runtime_matches_declared_contract(tmp_path):
    cfg = load_config("safe_rl/config/active/accvp_vnext/train.yaml")
    risk_checkpoint = tmp_path / "risk.pt"
    risk_checkpoint.write_bytes(b"risk-checkpoint")
    cfg.accvp.risk_checkpoint = str(risk_checkpoint)

    declared = formal_runtime_contract_from_config(cfg, declared=True)
    effective = formal_runtime_contract_from_config(cfg, declared=False)

    assert compare_formal_runtime_contracts(declared, effective)["pass"] is True
