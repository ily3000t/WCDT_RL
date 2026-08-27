from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from safe_rl.accvp.contracts.schema import root_observation_fingerprint
from safe_rl.accvp.modeling.model import (
    ACCVPPredictor,
    ACCVP_HYBRID_ARCHITECTURE_VERSION,
    architecture_version_from_config,
    model_kwargs_from_config,
)
from safe_rl.accvp.modeling.omitted_actor_summary import (
    OMITTED_ACTOR_SUMMARY_FEATURES,
    OMITTED_ACTOR_SUMMARY_GROUPS,
    build_omitted_actor_summary,
    validate_omitted_actor_summary_tensors,
)
from safe_rl.prediction import actor_selector
from safe_rl.prediction.actor_selector import ACTOR_SELECTION_VERSION_V4
from safe_rl.prediction.candidate_conflict import ActorConflictEvidence
from safe_rl.prediction.wcdt_v3_predictor import build_v3_runtime_batch
from safe_rl.sim.history_buffer import HistoryBuffer
from safe_rl.sim.types import VehicleState
from safe_rl.utils.config import clone_with_overrides, load_config


def _state(
    vehicle_id: str,
    lane_pos: float,
    *,
    speed: float,
    lane_index: int = 0,
) -> VehicleState:
    return VehicleState(
        vehicle_id=vehicle_id,
        x=300.0 + lane_pos,
        y=53.8 + 3.2 * lane_index,
        heading=0.0,
        speed=speed,
        accel=0.0,
        lane_index=lane_index,
        lane_id=f"main_aux_{lane_index}",
        lane_pos=lane_pos,
        edge_id="main_aux",
        route_position_valid=True,
    )


def _metadata(
    vehicle_id: str,
    *,
    role: str,
    signed_gap: float,
    ttc: float,
    surface_gap: float,
    effective_gap: float,
    closing_speed: float,
    conflict_time: float = 1_000_000.0,
    conflict: bool = False,
) -> dict:
    return {
        "vehicle_id": vehicle_id,
        "role": role,
        "signed_longitudinal_gap": signed_gap,
        "ttc": ttc,
        "current_surface_gap": surface_gap,
        "effective_gap": effective_gap,
        "closing_speed": closing_speed,
        "earliest_conflict_time_s": conflict_time,
        "candidate_conflict_eligible": conflict,
        "nearest_candidate_conflict": False,
    }


def test_omitted_actor_summary_is_fixed_grouped_and_order_invariant() -> None:
    selection = {
        "selected_actor_ids": ["selected"],
        "actor_metadata": {
            "selected": _metadata(
                "selected",
                role="target_front",
                signed_gap=10.0,
                ttc=2.0,
                surface_gap=8.0,
                effective_gap=4.0,
                closing_speed=4.0,
            ),
            "rear": _metadata(
                "rear",
                role="target_lane_other",
                signed_gap=-20.0,
                ttc=5.0,
                surface_gap=17.0,
                effective_gap=7.0,
                closing_speed=2.0,
            ),
            "conflict": _metadata(
                "conflict",
                role="other",
                signed_gap=30.0,
                ttc=3.0,
                surface_gap=25.0,
                effective_gap=10.0,
                closing_speed=5.0,
                conflict_time=1.5,
                conflict=True,
            ),
        },
    }
    states = {
        "ego": _state("ego", 100.0, speed=20.0),
        "selected": _state("selected", 110.0, speed=18.0),
        "rear": _state("rear", 80.0, speed=24.0),
        "conflict": _state("conflict", 130.0, speed=15.0),
    }
    summary = build_omitted_actor_summary(
        selection,
        actor_capacity=12,
        latest_states=states,
        ego_state=states["ego"],
    )
    reversed_selection = {
        **selection,
        "actor_metadata": dict(
            reversed(list(selection["actor_metadata"].items()))
        ),
    }
    reversed_summary = build_omitted_actor_summary(
        reversed_selection,
        actor_capacity=12,
        latest_states=states,
        ego_state=states["ego"],
    )

    assert summary.omitted_actor_ids == ("conflict", "rear")
    assert summary.tensor_hash == reversed_summary.tensor_hash
    np.testing.assert_array_equal(summary.features, reversed_summary.features)
    assert summary.summary_mask.tolist() == [1.0]
    conflict_group = OMITTED_ACTOR_SUMMARY_GROUPS.index("conflict_surface")
    rear_group = OMITTED_ACTOR_SUMMARY_GROUPS.index("target_lane_rear")
    conflict_time = OMITTED_ACTOR_SUMMARY_FEATURES.index(
        "earliest_conflict_time"
    )
    mean_relative = OMITTED_ACTOR_SUMMARY_FEATURES.index(
        "mean_relative_speed"
    )
    assert summary.group_mask[conflict_group] == 1.0
    assert summary.group_mask[rear_group] == 1.0
    assert summary.features[conflict_group, conflict_time] == pytest.approx(
        1.5 / 20.0
    )
    assert summary.features[rear_group, mean_relative] == pytest.approx(
        4.0 / 40.0
    )
    validate_omitted_actor_summary_tensors(
        summary.tensors(leading_batch=False),
        expected_hash=summary.tensor_hash,
    )


def test_runtime_batch_and_direct_builder_produce_identical_summary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = clone_with_overrides(
        load_config(),
        {
            "accvp": {
                "actor_count": 2,
                "actor_relevance": {"version": ACTOR_SELECTION_VERSION_V4},
                "omitted_actor_summary": {
                    "enabled": True,
                    "contract_version": "accvp_omitted_actor_summary_v1",
                    "physical_actor_capacity": 2,
                },
            }
        },
    )
    current = [
        _state("ego", 100.0, speed=20.0),
        _state("a", 108.0, speed=19.0),
        _state("b", 115.0, speed=18.0),
        _state("c", 125.0, speed=22.0),
        _state("d", 135.0, speed=17.0),
    ]

    def no_conflicts(_cfg, _ego, vehicles, **_kwargs):
        return (
            {
                vehicle.vehicle_id: ActorConflictEvidence(
                    vehicle_id=vehicle.vehicle_id,
                    candidate_conflict_eligible=False,
                    conflict_candidate_ids=(),
                    conflict_hypothesis_ids=(),
                    conflict_surface_ids=(),
                    earliest_conflict_time_s=1_000_000.0,
                    earliest_overlap_time_s=1_000_000.0,
                    minimum_swept_obb_gap=1_000_000.0,
                    nearest_candidate_conflict=False,
                )
                for vehicle in vehicles
                if vehicle.vehicle_id != "ego"
            },
            (3, 4, 5),
        )

    monkeypatch.setattr(
        actor_selector,
        "candidate_union_conflict_oracle",
        no_conflicts,
    )
    history = HistoryBuffer(
        history_steps=int(cfg.scenario.history_steps),
        max_agents=5,
    )
    history.append(current)
    runtime = build_v3_runtime_batch(
        cfg,
        history,
        "ego",
        max_actors_override=2,
        selector_scope="accvp",
    )
    latest = history.latest()
    direct = build_omitted_actor_summary(
        runtime["actor_selection"],
        actor_capacity=2,
        latest_states=latest,
        ego_state=latest["ego"],
    )

    assert len(runtime["actor_row_ids"]) == 2
    assert len(runtime["actor_selection"].actor_metadata) == 4
    assert len(direct.omitted_actor_ids) == 2
    for name, expected in direct.tensors(leading_batch=True).items():
        np.testing.assert_array_equal(runtime[name], expected)


def test_normal_wcdt_prediction_scope_ignores_accvp_only_summary_contract() -> None:
    cfg = clone_with_overrides(
        load_config(),
        {
            "accvp": {
                "actor_count": 2,
                "omitted_actor_summary": {
                    "enabled": True,
                    "contract_version": "accvp_omitted_actor_summary_v1",
                    # Deliberately invalid for ACCVP scope. A normal WcDT
                    # forecast must never parse this independent adapter.
                    "physical_actor_capacity": 3,
                },
            }
        },
    )
    history = HistoryBuffer(
        history_steps=int(cfg.scenario.history_steps),
        max_agents=3,
    )
    history.append(
        [
            _state("ego", 100.0, speed=20.0),
            _state("front", 115.0, speed=18.0),
            _state("rear", 85.0, speed=22.0),
        ]
    )
    runtime = build_v3_runtime_batch(
        cfg,
        history,
        "ego",
        max_actors_override=2,
        selector_scope="prediction",
    )
    assert "omitted_actor_summary_features" not in runtime


def test_hybrid_adapter_adds_one_scene_token_but_no_response_row() -> None:
    cfg = load_config(
        "safe_rl/config/active/accvp_vnext_selector4/train.yaml"
    )
    assert architecture_version_from_config(cfg) == (
        ACCVP_HYBRID_ARCHITECTURE_VERSION
    )
    model = ACCVPPredictor(**model_kwargs_from_config(cfg)).eval()
    actors = int(cfg.accvp.actor_count)
    history_steps = int(cfg.scenario.history_steps)
    plan_steps = int(cfg.accvp.candidate_plan_horizon_steps)
    groups = len(OMITTED_ACTOR_SUMMARY_GROUPS)
    features = len(OMITTED_ACTOR_SUMMARY_FEATURES)
    root_inputs = {
        "history_features": torch.randn(1, actors, history_steps, 10),
        "history_valid_mask": torch.ones(1, actors, history_steps),
        "history_lane_ids": torch.ones(
            1, actors, history_steps, dtype=torch.long
        ),
        "history_edge_role_ids": torch.ones(
            1, actors, history_steps, dtype=torch.long
        ),
        "role_ids": torch.ones(1, actors, dtype=torch.long),
        "lane_ids": torch.ones(1, actors, dtype=torch.long),
        "edge_role_ids": torch.ones(1, actors, dtype=torch.long),
        "actor_mask": torch.ones(1, actors),
        "omitted_actor_summary_features": torch.rand(1, groups, features),
        "omitted_actor_summary_group_mask": torch.ones(1, groups),
        "omitted_actor_summary_mask": torch.ones(1, 1),
    }
    with torch.no_grad():
        scene = model.encode_scene(**root_inputs)
        output = model.forward_from_scene(
            scene,
            root_inputs["actor_mask"],
            torch.randn(1, plan_steps, 5),
            torch.tensor([4], dtype=torch.long),
            root_inputs["omitted_actor_summary_mask"],
        )
    assert scene.shape == (1, actors + 1, model.hidden_dim)
    assert output["actor_response"].shape[1] == actors
    assert not any(
        key.startswith("omitted_actor_summary_adapter")
        for key in model.scene.state_dict()
    )


def test_zero_summary_mask_is_equivalent_to_physical_only_model() -> None:
    cfg = load_config()
    base_kwargs = model_kwargs_from_config(cfg)
    base = ACCVPPredictor(**base_kwargs).eval()
    hybrid = ACCVPPredictor(
        **base_kwargs,
        omitted_actor_summary_feature_dim=len(
            OMITTED_ACTOR_SUMMARY_FEATURES
        ),
        omitted_actor_summary_group_count=len(OMITTED_ACTOR_SUMMARY_GROUPS),
    ).eval()
    load_result = hybrid.load_state_dict(base.state_dict(), strict=False)
    assert not load_result.unexpected_keys
    assert load_result.missing_keys
    assert all(
        key.startswith("omitted_actor_summary_adapter.")
        for key in load_result.missing_keys
    )
    actors = int(cfg.accvp.actor_count)
    history_steps = int(cfg.scenario.history_steps)
    plan_steps = int(cfg.accvp.candidate_plan_horizon_steps)
    physical_inputs = {
        "history_features": torch.randn(1, actors, history_steps, 10),
        "history_valid_mask": torch.ones(1, actors, history_steps),
        "history_lane_ids": torch.ones(
            1, actors, history_steps, dtype=torch.long
        ),
        "history_edge_role_ids": torch.ones(
            1, actors, history_steps, dtype=torch.long
        ),
        "role_ids": torch.ones(1, actors, dtype=torch.long),
        "lane_ids": torch.ones(1, actors, dtype=torch.long),
        "edge_role_ids": torch.ones(1, actors, dtype=torch.long),
        "actor_mask": torch.ones(1, actors),
    }
    summary_inputs = {
        "omitted_actor_summary_features": torch.zeros(
            1,
            len(OMITTED_ACTOR_SUMMARY_GROUPS),
            len(OMITTED_ACTOR_SUMMARY_FEATURES),
        ),
        "omitted_actor_summary_group_mask": torch.zeros(
            1, len(OMITTED_ACTOR_SUMMARY_GROUPS)
        ),
        "omitted_actor_summary_mask": torch.zeros(1, 1),
    }
    plan = torch.randn(1, plan_steps, 5)
    action = torch.tensor([4], dtype=torch.long)
    with torch.no_grad():
        base_scene = base.encode_scene(**physical_inputs)
        hybrid_scene = hybrid.encode_scene(
            **physical_inputs,
            **summary_inputs,
        )
        base_output = base.forward_from_scene(
            base_scene,
            physical_inputs["actor_mask"],
            plan,
            action,
        )
        hybrid_output = hybrid.forward_from_scene(
            hybrid_scene,
            physical_inputs["actor_mask"],
            plan,
            action,
            summary_inputs["omitted_actor_summary_mask"],
        )
    torch.testing.assert_close(
        hybrid_scene[:, :actors], base_scene, atol=1.0e-6, rtol=1.0e-6
    )
    for name in ("actor_response", "event_logits", "geometry"):
        torch.testing.assert_close(
            hybrid_output[name], base_output[name], atol=1.0e-6, rtol=1.0e-6
        )


def test_hybrid_summary_is_bound_into_root_observation_fingerprint() -> None:
    base = {
        "history_features": np.zeros((2, 3, 10), dtype=np.float32),
        "history_valid_mask": np.ones((2, 3), dtype=np.float32),
        "history_lane_ids": np.zeros((2, 3), dtype=np.int64),
        "history_edge_role_ids": np.zeros((2, 3), dtype=np.int64),
        "role_ids": np.zeros((2,), dtype=np.int64),
        "lane_ids": np.zeros((2,), dtype=np.int64),
        "edge_role_ids": np.zeros((2,), dtype=np.int64),
        "mask": np.ones((2,), dtype=np.float32),
        "omitted_actor_summary_features": np.zeros(
            (
                len(OMITTED_ACTOR_SUMMARY_GROUPS),
                len(OMITTED_ACTOR_SUMMARY_FEATURES),
            ),
            dtype=np.float32,
        ),
        "omitted_actor_summary_group_mask": np.zeros(
            (len(OMITTED_ACTOR_SUMMARY_GROUPS),), dtype=np.float32
        ),
        "omitted_actor_summary_mask": np.zeros((1,), dtype=np.float32),
    }
    kwargs = {
        "root_ego": {"x": 1.0, "y": 2.0, "heading": 0.0, "speed": 3.0},
        "data_contract_hash": "hybrid-contract",
        "fingerprint_version": "model_input_fingerprint_v4_hybrid_actor_summary",
    }
    first = root_observation_fingerprint(tensors=base, **kwargs)
    changed = {name: value.copy() for name, value in base.items()}
    changed["omitted_actor_summary_features"][0, 0] = 1.0
    second = root_observation_fingerprint(tensors=changed, **kwargs)
    assert first != second

    incomplete = dict(base)
    incomplete.pop("omitted_actor_summary_mask")
    with pytest.raises(ValueError, match="incomplete hybrid actor summary"):
        root_observation_fingerprint(tensors=incomplete, **kwargs)


def test_hybrid_protocol_changes_lineage_but_not_any_seed_values() -> None:
    old = json.loads(
        Path("safe_rl/config/accvp_vnext_selector4_seed_ledger.json").read_text(
            encoding="utf-8"
        )
    )
    hybrid = json.loads(
        Path(
            "safe_rl/config/accvp_vnext_selector4_hybrid_seed_ledger.json"
        ).read_text(encoding="utf-8")
    )
    assert old["protocol_id"] != hybrid["protocol_id"]
    assert old["cohorts"] == hybrid["cohorts"]

    ppo = load_config(
        "safe_rl/config/active/accvp_vnext_selector4/"
        "ppo_candidate_table_full.yaml"
    )
    assert int(ppo.training.ppo_num_envs) * int(ppo.rl.n_steps) == 1024
    assert int(ppo.rl.total_timesteps) == 100_000
    assert int(ppo.rl.optimizer_seed) == 1001


def test_hybrid_extended_training_has_independent_hundred_epoch_lineage() -> None:
    train = load_config(
        "safe_rl/config/active/accvp_vnext_selector4/train.yaml"
    )
    ppo = load_config(
        "safe_rl/config/active/accvp_vnext_selector4/"
        "ppo_candidate_table_full.yaml"
    )

    assert train.run.run_id == "accvp_vnext_selector4_hybrid_train100"
    assert int(train.accvp.training.epochs) == 100
    assert "hybrid_train100" in str(train.accvp.artifact_manifest)
    assert ppo.accvp.checkpoint == train.accvp.checkpoint
    assert ppo.accvp.calibration_bundle == train.accvp.calibration_bundle
    assert ppo.accvp.operating_point == train.accvp.operating_point
    assert ppo.accvp.artifact_manifest == train.accvp.artifact_manifest
