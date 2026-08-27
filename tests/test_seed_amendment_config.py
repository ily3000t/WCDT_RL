from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import yaml

from safe_rl.accvp.contracts.schema import file_sha256
from safe_rl.evaluation_protocol import stable_hash
from safe_rl.pipeline.run_accvp_vnext_pipeline import WORKFLOW_CONFIG
from safe_rl.utils.config import REPO_ROOT


FROZEN = (
    REPO_ROOT
    / "safe_rl/config/config_fix/accvp_vnext_selector4_hybrid_v4_frozen"
)
AMENDED = (
    REPO_ROOT
    / "safe_rl/config/config_fix/"
    "accvp_vnext_selector4_hybrid_seed1000_1004_posthoc_v1"
)
AMENDED_PROTOCOL = (
    "accvp-vnext-correctness-v4-selector4-hybrid-"
    "posthoc-seed-amendment-v1"
)
REVISED_SEEDS = [1000, 1001, 1002, 1003, 1004]


def _yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    assert isinstance(payload, Mapping)
    return dict(payload)


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, Mapping)
    return dict(payload)


def test_posthoc_seed_amendment_is_isolated_from_frozen_config_tree() -> None:
    original_workflow = _yaml(
        REPO_ROOT / "safe_rl/config/active/accvp_vnext_selector4/workflow.yaml"
    )
    frozen_workflow = _yaml(FROZEN / "workflow.yaml")
    amended_workflow = _yaml(AMENDED / "workflow.yaml")
    assert original_workflow["seeds"]["optimizer_replicates"] == [
        1001,
        1002,
        1003,
        1004,
        1005,
    ]
    assert frozen_workflow == original_workflow
    assert amended_workflow["seeds"]["optimizer_replicates"] == REVISED_SEEDS
    assert amended_workflow["protocol_id"] == AMENDED_PROTOCOL
    assert amended_workflow["amendment"]["classification"] == (
        "posthoc_optimizer_seed_amendment"
    )
    assert amended_workflow["amendment"]["removed_optimizer_seed"] == 1005
    assert amended_workflow["amendment"]["replacement_optimizer_seed"] == 1000
    assert Path(WORKFLOW_CONFIG).as_posix().endswith(
        "config_fix/accvp_vnext_selector4_hybrid_seed1000_1004_posthoc_v1/"
        "workflow.yaml"
    )


def test_posthoc_bundle_keeps_upstream_configs_byte_identical() -> None:
    upstream = (
        "selector_audit.yaml",
        "pilot.yaml",
        "oracle_regression.yaml",
        "formal.yaml",
        "train.yaml",
        "pilot_latency_smoke_train.yaml",
        "pilot_latency_smoke_runtime.yaml",
    )
    for name in upstream:
        assert file_sha256(AMENDED / name) == file_sha256(FROZEN / name)
    for path in AMENDED.glob("*.yaml"):
        assert "extends" not in _yaml(path)


def test_posthoc_bundle_binds_revised_seed_ledger_and_new_outputs() -> None:
    manifest = _json(AMENDED / "amendment_manifest.json")
    fingerprint = manifest.pop("bundle_fingerprint")
    assert stable_hash(manifest) == fingerprint
    assert manifest["classification"] == "posthoc_optimizer_seed_amendment"
    assert all(
        row["sha256"] == row["frozen_sha256"]
        for row in manifest["unchanged_upstream_configs"]
    )

    ledger = _json(AMENDED / "seed_ledger.json")
    assert ledger["protocol_id"] == AMENDED_PROTOCOL
    assert ledger["cohorts"]["ppo_optimizer_replicates"]["values"] == REVISED_SEEDS
    assert ledger["amendment"]["original_values"] == [1001, 1002, 1003, 1004, 1005]

    workflow = _yaml(AMENDED / "workflow.yaml")
    config_root = (
        "safe_rl/config/config_fix/"
        "accvp_vnext_selector4_hybrid_seed1000_1004_posthoc_v1/"
    )
    for key in (
        "selector_audit_config",
        "pilot_config",
        "pilot_latency_smoke_train_config",
        "pilot_latency_smoke_runtime_config",
        "oracle_config",
        "formal_config",
        "train_config",
        "ppo_config",
        "baseline_ppo_config",
        "matrix_config",
        "protocol_config",
    ):
        assert str(workflow["paths"][key]).startswith(config_root)
    for key in (
        "factorial_manifest",
        "baseline_manifest",
        "baseline_lineage_audit",
        "runtime_factorial_report",
        "stage5_request",
        "stage5_report",
        "holdout_report",
    ):
        assert "posthoc_seed_amendment_v1" in str(workflow["paths"][key])

    for name in (
        "ppo_candidate_table_full.yaml",
        "baseline_ppo_wcdt_v3_reward_v2.yaml",
        "evaluation_protocol.yaml",
    ):
        config = _yaml(AMENDED / name)
        protocol = config["evaluation_protocol"]
        assert protocol["protocol_id"] == AMENDED_PROTOCOL
        assert protocol["seed_ledger"] == f"{config_root}seed_ledger.json"
        assert protocol["revocation_manifest"] == (
            f"{config_root}artifact_revocation_manifest.json"
        )
    assert _yaml(AMENDED / "ppo_ablation_matrix.yaml")["protocol_id"] == (
        AMENDED_PROTOCOL
    )
