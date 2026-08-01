from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from safe_rl.accvp.contracts.protocol import (
    ACCVP_SELECTOR3_DATA_CONTRACT_VERSION,
)
from safe_rl.accvp.contracts.schema import stable_hash
from safe_rl.accvp.data.dataset import build_split_manifest
from safe_rl.accvp.evaluation.selector_contract import (
    _capacity_pass,
    selector_audit_input_coverage,
)
from safe_rl.pipeline.run_accvp_vnext_pipeline import (
    _load_workflow_contract,
    _workflow_seed_values,
)
from safe_rl.prediction import actor_selector
from safe_rl.prediction.actor_selector import (
    ACTOR_SELECTION_VERSION_V2,
    ACTOR_SELECTION_VERSION_V3,
    actor_relevance_config,
    actor_selection_config_hash,
    select_merge_relevant_actors,
)
from safe_rl.prediction.candidate_conflict import (
    ActorConflictEvidence,
    candidate_union_conflict_oracle,
    candidate_union_conflict_oracle_reference,
)
from safe_rl.sim.types import VehicleState
from safe_rl.utils.config import clone_with_overrides, load_config


def _vehicle(
    vehicle_id: str,
    lane_pos: float,
    *,
    lane_index: int,
    edge_id: str = "main_aux",
    speed: float = 20.0,
) -> VehicleState:
    return VehicleState(
        vehicle_id=vehicle_id,
        x=300.0 + lane_pos,
        y=53.8 + 3.2 * lane_index,
        heading=0.0,
        speed=speed,
        accel=0.0,
        lane_index=lane_index,
        lane_id=f"{edge_id}_{lane_index}",
        lane_pos=lane_pos,
        edge_id=edge_id,
        route_position_valid=True,
    )


def _v3_config():
    return clone_with_overrides(
        load_config(),
        {
            "prediction": {
                "actor_relevance": {
                    "version": ACTOR_SELECTION_VERSION_V3,
                }
            }
        },
    )


def _evidence(
    vehicle_id: str,
    *,
    eligible: bool,
    time_s: float = 1_000_000.0,
    gap_m: float = 1_000_000.0,
    nearest: bool = False,
) -> ActorConflictEvidence:
    return ActorConflictEvidence(
        vehicle_id=vehicle_id,
        candidate_conflict_eligible=eligible,
        conflict_candidate_ids=(6,) if eligible else (),
        conflict_hypothesis_ids=("change_right",) if eligible else (),
        conflict_surface_ids=("main_aux:1",) if eligible else (),
        earliest_conflict_time_s=time_s,
        earliest_overlap_time_s=time_s,
        minimum_swept_obb_gap=gap_m,
        nearest_candidate_conflict=nearest,
    )


def test_selector_v3_keeps_non_conflict_global_gap_actor_contextual(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _v3_config()
    ego = _vehicle("ego", 100.0, lane_index=0)
    target = _vehicle("target", 112.0, lane_index=1)
    other = _vehicle(
        "other",
        105.0,
        lane_index=2,
        edge_id="unrelated_edge",
    )
    monkeypatch.setattr(
        actor_selector,
        "candidate_union_conflict_oracle",
        lambda *_args, **_kwargs: (
            {
                "target": _evidence("target", eligible=True),
                "other": _evidence("other", eligible=False),
            },
            (3, 4, 5, 6, 7, 8),
        ),
    )

    selection = select_merge_relevant_actors(
        cfg, ego, [ego, target, other], max_actors=6
    )

    assert selection.actor_metadata["target"].critical
    assert selection.actor_metadata["other"].contextual
    assert not selection.actor_metadata["other"].critical
    assert "effective_gap" in selection.actor_metadata["other"].relevance_reasons


def test_selector_v3_promotes_future_reachable_adjacent_lane_conflict() -> None:
    cfg = _v3_config()
    ego = _vehicle("ego", 100.0, lane_index=0)
    adjacent = _vehicle("adjacent", 105.0, lane_index=2)

    selection = select_merge_relevant_actors(
        cfg, ego, [ego, adjacent], max_actors=6
    )
    metadata = selection.actor_metadata["adjacent"]

    assert metadata.candidate_conflict_eligible
    assert metadata.nearest_candidate_conflict
    assert metadata.critical
    assert metadata.conflict_candidate_ids
    assert metadata.earliest_conflict_time_s < 3.1
    assert metadata.conflict_surface_ids


def test_selector_v3_vectorized_conflict_oracle_matches_scalar_reference() -> None:
    cfg = _v3_config()
    ego = _vehicle("ego", 100.0, lane_index=0, speed=21.0)
    vehicles = [
        ego,
        _vehicle("target_front", 114.0, lane_index=1, speed=19.0),
        _vehicle("target_rear", 87.0, lane_index=1, speed=24.0),
        _vehicle("adjacent", 104.0, lane_index=2, speed=20.0),
    ]
    vectorized, vectorized_candidates = candidate_union_conflict_oracle(
        cfg, ego, vehicles
    )
    reference, reference_candidates = (
        candidate_union_conflict_oracle_reference(cfg, ego, vehicles)
    )

    assert vectorized_candidates == reference_candidates
    assert set(vectorized) == set(reference)
    for vehicle_id in vectorized:
        actual = vectorized[vehicle_id]
        expected = reference[vehicle_id]
        assert (
            actual.candidate_conflict_eligible
            == expected.candidate_conflict_eligible
        )
        assert actual.conflict_candidate_ids == expected.conflict_candidate_ids
        assert (
            actual.conflict_hypothesis_ids
            == expected.conflict_hypothesis_ids
        )
        assert actual.conflict_surface_ids == expected.conflict_surface_ids
        assert actual.nearest_candidate_conflict == (
            expected.nearest_candidate_conflict
        )
        assert actual.earliest_conflict_time_s == pytest.approx(
            expected.earliest_conflict_time_s, abs=1.0e-9
        )
        assert actual.earliest_overlap_time_s == pytest.approx(
            expected.earliest_overlap_time_s, abs=1.0e-9
        )
        assert actual.minimum_swept_obb_gap == pytest.approx(
            expected.minimum_swept_obb_gap, abs=1.0e-9
        )


def test_selector_v3_orders_conflicts_before_vehicle_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _v3_config()
    ego = _vehicle("ego", 100.0, lane_index=0)
    alpha = _vehicle(
        "alpha",
        108.0,
        lane_index=2,
        edge_id="unrelated_edge",
    )
    zulu = _vehicle(
        "zulu",
        108.0,
        lane_index=2,
        edge_id="unrelated_edge",
    )
    monkeypatch.setattr(
        actor_selector,
        "candidate_union_conflict_oracle",
        lambda *_args, **_kwargs: (
            {
                "alpha": _evidence(
                    "alpha", eligible=True, time_s=2.0, gap_m=0.1
                ),
                "zulu": _evidence(
                    "zulu", eligible=True, time_s=1.0, gap_m=0.2
                ),
            },
            (6,),
        ),
    )

    selection = select_merge_relevant_actors(
        cfg, ego, [ego, alpha, zulu], max_actors=2
    )
    assert selection.selected_actor_ids == ("zulu", "alpha")


def test_selector_v2_does_not_call_candidate_conflict_or_change_hash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = load_config()
    with_extra_v3_defaults = clone_with_overrides(
        cfg,
        {
            "prediction": {
                "actor_relevance": {
                    "candidate_conflict_horizon_s": 9.0,
                    "candidate_conflict_surface_gap": 99.0,
                    "actor_longitudinal_accel_bound": 9.0,
                }
            }
        },
    )
    monkeypatch.setattr(
        actor_selector,
        "candidate_union_conflict_oracle",
        lambda *_args, **_kwargs: pytest.fail(
            "selector-v2 called the Selector-v3 conflict oracle"
        ),
    )
    ego = _vehicle("ego", 100.0, lane_index=0)
    actor = _vehicle("actor", 108.0, lane_index=1)
    selection = select_merge_relevant_actors(
        with_extra_v3_defaults,
        ego,
        [ego, actor],
        max_actors=1,
    )
    assert selection.version == ACTOR_SELECTION_VERSION_V2
    assert actor_selection_config_hash(cfg) == actor_selection_config_hash(
        with_extra_v3_defaults
    )


def test_selector_capacity_decision_is_strict_6_then_8() -> None:
    base = {
        "critical_overflow_count": 0,
        "dropped_critical_count": 0,
        "mandatory_target_not_critical_count": 0,
        "candidate_conflict_not_critical_count": 0,
        "target_front_rear_coverage_rate": 1.0,
        "candidate_conflict_coverage_rate": 1.0,
        "nearest_conflict_coverage_rate": 1.0,
    }
    assert _capacity_pass(base)
    assert not _capacity_pass({**base, "critical_overflow_count": 1})
    assert not _capacity_pass(
        {**base, "candidate_conflict_coverage_rate": 0.999}
    )
    assert not _capacity_pass(
        {**base, "mandatory_target_not_critical_count": 1}
    )


def test_selector_capacity_report_locks_model_rows_and_validates_fingerprint(
    tmp_path: Path,
) -> None:
    report = {
        "artifact_kind": "accvp_selector_contract_audit_v1",
        "protocol_id": "accvp-vnext-correctness-v2-selector3",
        "audit_state": "pass",
        "selected_capacity": 8,
    }
    report["report_fingerprint"] = stable_hash(report)
    report_path = tmp_path / "selector_report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    overlay = tmp_path / "locked.yaml"
    overlay.write_text(
        yaml.safe_dump(
            {
                "extends": (
                    "safe_rl/config/active/accvp_vnext_selector3/"
                    "selector_audit.yaml"
                ),
                "accvp": {
                    "selector_contract": {
                        "require_capacity_lock": True,
                        "audit_report": str(report_path),
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    cfg = load_config(overlay)
    assert cfg.accvp.actor_count == 8
    assert cfg.prediction.wcdt_v3_max_agents == 6
    assert (
        actor_relevance_config(cfg)["version"]
        == ACTOR_SELECTION_VERSION_V2
    )
    assert (
        actor_relevance_config(cfg, selector_scope="accvp")["version"]
        == ACTOR_SELECTION_VERSION_V3
    )
    assert (
        cfg.accvp.selector_contract.audit_report_fingerprint
        == report["report_fingerprint"]
    )

    report["selected_capacity"] = 6
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        load_config(overlay)


def test_selector3_accvp_scope_does_not_mutate_wcdt_checkpoint_contract() -> None:
    base = load_config()
    split = clone_with_overrides(
        base,
        {
            "accvp": {
                "actor_relevance": {
                    "version": ACTOR_SELECTION_VERSION_V3,
                },
                "actor_count": 8,
            }
        },
    )

    assert actor_selection_config_hash(split) == actor_selection_config_hash(
        base
    )
    assert split.prediction.wcdt_v3_max_agents == 6
    assert actor_selection_config_hash(
        split,
        selector_scope="accvp",
    ) != actor_selection_config_hash(split)


def test_selector_input_audit_requires_unselected_full_state(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset"
    manifests = dataset / "manifests"
    roots = dataset / "roots"
    manifests.mkdir(parents=True)
    roots.mkdir()
    metadata_path = roots / "root.json"
    vehicles = [
        _vehicle("ego", 100.0, lane_index=0).to_dict(),
        _vehicle("selected", 112.0, lane_index=1).to_dict(),
        _vehicle("unselected", 130.0, lane_index=2).to_dict(),
    ]
    metadata_path.write_text(
        json.dumps(
            {
                "history_frames": [vehicles],
                "selected_actor_count": 1,
            }
        ),
        encoding="utf-8",
    )
    (manifests / "roots.jsonl").write_text(
        json.dumps(
            {
                "root_id": "root",
                "complete": True,
                "metadata_path": str(metadata_path),
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = selector_audit_input_coverage(load_config(), dataset)
    assert report["input_coverage_state"] == "pass"
    assert report["unselected_actor_state_available"]
    assert not report["explicit_future_route_intent_available"]
    assert report["conservative_reachable_tube_audit_allowed"]
    assert not report["exact_route_intent_claim"]


def test_selector3_split_rejects_incomplete_task_coverage(
    tmp_path: Path,
) -> None:
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    (manifests / "dataset_manifest.json").write_text(
        json.dumps(
            {
                "counterfactual_schema_version": 3,
                "data_contract": {
                    "protocol_version": ACCVP_SELECTOR3_DATA_CONTRACT_VERSION
                },
            }
        ),
        encoding="utf-8",
    )
    (manifests / "roots.jsonl").write_text(
        json.dumps(
            {
                "root_id": "incomplete",
                "complete": True,
                "task_actor_coverage_complete": False,
                "critical_actor_overflow": False,
                "dropped_critical_actor_ids": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="incomplete task actor coverage"):
        build_split_manifest(tmp_path, seed=1)


def test_selector3_workflow_freezes_new_gates_and_runtime_seeds() -> None:
    _path, workflow = _load_workflow_contract(
        "safe_rl/config/active/accvp_vnext_selector3/workflow.yaml"
    )
    phases = list(workflow["phase_order"])
    assert phases[0] == "selector_contract_audit"
    assert phases.index("formal_validation") < phases.index("accvp_training")
    assert phases.index("baseline_ppo_replicates") < phases.index(
        "baseline_lineage_audit"
    )
    assert phases.index("baseline_lineage_audit") < phases.index(
        "policy_runtime_replicates"
    )
    assert _workflow_seed_values(workflow, "runtime", []) == list(
        range(55001, 55061)
    )
    assert _workflow_seed_values(
        workflow, "selector_diagnostic", []
    ) == [50021, 50027]
