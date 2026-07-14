from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from safe_rl.accvp.data.dataset import (
    build_split_manifest,
    entry_time_supervision,
    event_supervision_mask,
)
from safe_rl.stage1_counterfactual.branch_worker import _conditional_entry_time_fields
from safe_rl.accvp.data.migration import audit_legacy_dataset_migration
from safe_rl.accvp.modeling.model import ACCVP_LOSS_VERSION, accvp_loss
from safe_rl.accvp.contracts.schema import (
    ACTOR_ROW_MAPPING_VERSION,
    COUNTERFACTUAL_SCHEMA_VERSION,
    ROOT_OBSERVATION_FINGERPRINT_VERSION,
    SCENARIO_EPISODE_KEY_VERSION,
    actor_row_mapping_hash,
    root_observation_fingerprint,
)
from safe_rl.accvp.training.trainer import (
    COMPONENT_BOOTSTRAP_VERSION,
    OPTIMIZATION_BATCHING_VERSION,
    _fingerprint_action_group_batches,
    build_component_bootstrap_plan,
    shuffled_component_bootstrap_groups,
    validate_ensemble_configuration,
)
from safe_rl.prediction.wcdt_v3_predictor import selected_vehicle_ids_from_indices


def _write_roots(path: Path, rows: list[dict]) -> None:
    manifests = path / "manifests"
    manifests.mkdir(parents=True)
    with (manifests / "roots.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _write_legacy_migration_fixture(
    path: Path,
    *,
    mapping_available: bool = True,
    scalar_valid: bool = True,
) -> None:
    manifests = path / "manifests"
    roots_dir = path / "roots"
    branches_dir = path / "branches"
    manifests.mkdir(parents=True)
    roots_dir.mkdir()
    branches_dir.mkdir()
    root_id = "legacy-root"
    metadata_path = roots_dir / f"{root_id}.json"
    tensor_path = roots_dir / f"{root_id}.npz"
    metadata_path.write_text(
        json.dumps(
            {
                "counterfactual_schema_version": 2,
                "root_id": root_id,
                "ego_id": "ego",
                "selected_actor_ids": ["rear", "front"],
                "selector": {"selected_actor_ids": ["rear", "front"]},
            }
        ),
        encoding="utf-8",
    )
    root_arrays = {"mask": np.asarray([[1.0, 1.0, 0.0]], dtype=np.float32)}
    if mapping_available:
        root_arrays["selected_indices"] = np.asarray([[2, 1, -1]], dtype=np.int64)
    np.savez_compressed(tensor_path, **root_arrays)
    branch_path = branches_dir / f"{root_id}_action4.npz"
    np.savez_compressed(
        branch_path,
        actor_response=np.zeros((3, 2, 5), dtype=np.float32),
        actor_valid_mask=np.ones((3, 2), dtype=np.float32),
    )
    root = {
        "root_id": root_id,
        "complete": True,
        "expected_action_ids": [4],
        "metadata_path": str(metadata_path),
        "tensor_path": str(tensor_path),
    }
    branch = {
        "root_id": root_id,
        "branch_id": f"{root_id}_action4",
        "branch_status": "completed",
        "action_id": 4,
        "event_observed": True,
        "censor_time": 2.0,
        "viability_observation_status": "observed_success",
        "proxy_collision_within_horizon": False,
        "safety_violation_within_horizon": False,
        "taper_miss_observed": False,
        "merge_before_taper_observed": True,
        "min_obb_distance": 5.0,
        "max_drac": 0.1,
        "target_front_gap": 10.0,
        "target_rear_gap": 10.0,
        "target_lane_entry_time_s": 1.0 if scalar_valid else None,
        "selected_actor_ids": ["rear", "front"],
        "tensor_path": str(branch_path),
    }
    (manifests / "dataset_manifest.json").write_text(
        json.dumps({"counterfactual_schema_version": 2}), encoding="utf-8"
    )
    (manifests / "roots.jsonl").write_text(json.dumps(root) + "\n", encoding="utf-8")
    (manifests / "branches.jsonl").write_text(json.dumps(branch) + "\n", encoding="utf-8")


def test_selected_vehicle_ids_follow_selected_indices_not_input_order():
    assert selected_vehicle_ids_from_indices(
        ["ego", "rear", "front", "adjacent"],
        np.asarray([3, 1, 2, -1]),
        np.asarray([1.0, 1.0, 1.0, 0.0]),
    ) == ["adjacent", "rear", "front", ""]

    with pytest.raises(ValueError, match="duplicate"):
        selected_vehicle_ids_from_indices(
            ["ego", "rear", "front"],
            np.asarray([1, 1]),
            np.asarray([1.0, 1.0]),
        )


def test_actor_row_mapping_hash_binds_ids_indices_and_mask():
    first = actor_row_mapping_hash(["a", "b", ""], [2, 1, -1], [1.0, 1.0, 0.0])
    second = actor_row_mapping_hash(["b", "a", ""], [1, 2, -1], [1.0, 1.0, 0.0])
    assert ACTOR_ROW_MAPPING_VERSION in first
    assert first != second


def test_model_input_fingerprint_excludes_actor_identity_but_mapping_hash_does_not():
    tensors = {
        "history_features": np.zeros((2, 3, 10), dtype=np.float32),
        "history_valid_mask": np.ones((2, 3), dtype=np.float32),
        "history_lane_ids": np.zeros((2, 3), dtype=np.int64),
        "history_edge_role_ids": np.zeros((2, 3), dtype=np.int64),
        "role_ids": np.zeros((2,), dtype=np.int64),
        "lane_ids": np.zeros((2,), dtype=np.int64),
        "edge_role_ids": np.zeros((2,), dtype=np.int64),
        "mask": np.ones((2,), dtype=np.float32),
    }
    kwargs = {
        "root_ego": {"x": 1.0, "y": 2.0, "heading": 0.1, "speed": 4.0},
        "data_contract_hash": "contract",
        "tensors": tensors,
    }
    first = root_observation_fingerprint(actor_row_ids=["a", "b"], **kwargs)
    second = root_observation_fingerprint(actor_row_ids=["x", "y"], **kwargs)
    assert ROOT_OBSERVATION_FINGERPRINT_VERSION == "model_input_fingerprint_v3"
    assert first == second
    assert actor_row_mapping_hash(["a", "b"], [1, 2], [1.0, 1.0]) != (
        actor_row_mapping_hash(["x", "y"], [1, 2], [1.0, 1.0])
    )


def test_split_manifest_keeps_episode_and_observation_components_together(tmp_path: Path):
    rows = []
    for index in range(12):
        rows.append(
            {
                "root_id": f"root-{index}",
                "root_episode_id": f"episode-{index}",
                "episode_seed": index,
                "root_policy": "policy",
                "collection_source": "source",
                "traffic_profile": "traffic",
                "activation_bin": "activation_window",
                "root_observation_fingerprint": f"fingerprint-{index}",
                "complete": True,
            }
        )
    # Join two otherwise independent episodes by a shared model-visible state.
    rows[1]["root_observation_fingerprint"] = rows[0]["root_observation_fingerprint"]
    # Join two different observations by episode membership.
    rows[3]["root_episode_id"] = rows[2]["root_episode_id"]
    _write_roots(tmp_path, rows)

    manifest = build_split_manifest(tmp_path, seed=19, require_all_splits=True)
    split_by_root = {row["root_id"]: row["split"] for row in manifest}
    assert split_by_root["root-0"] == split_by_root["root-1"]
    assert split_by_root["root-2"] == split_by_root["root-3"]

    provenance = json.loads((tmp_path / "manifests" / "split_provenance.json").read_text(encoding="utf-8"))
    assert provenance["cross_split_episode_overlap_count"] == 0
    assert provenance["cross_split_observation_fingerprint_overlap_count"] == 0
    assert provenance["component_count"] == 10


def test_split_manifest_joins_same_scenario_seed_across_policies(tmp_path: Path):
    route_hash = "scenario-route-fixture"
    rows = []
    policies = ("mixed", "ppo", "rule", "mixed", "ppo", "rule", "mixed", "ppo", "rule")
    for index, policy in enumerate(policies):
        episode_seed = 101 if index < 3 else 101 + index
        traffic_profile = "hard" if index != 2 else "safe"
        rows.append(
            {
                "root_id": f"scenario-root-{index}",
                "root_episode_id": f"{policy}:{episode_seed}",
                "episode_seed": episode_seed,
                "root_policy": policy,
                "collection_source": policy,
                "traffic_profile": traffic_profile,
                "activation_bin": "activation_window",
                "root_observation_fingerprint": f"scenario-fingerprint-{index}",
                "scenario_route_hash": route_hash,
                "complete": True,
            }
        )
    _write_roots(tmp_path, rows)
    (tmp_path / "manifests" / "dataset_manifest.json").write_text(
        json.dumps(
            {
                "counterfactual_schema_version": COUNTERFACTUAL_SCHEMA_VERSION,
                "scenario_route_hash": route_hash,
                "data_contract": {"scenario_route_hash": route_hash},
            }
        ),
        encoding="utf-8",
    )

    manifest = build_split_manifest(tmp_path, seed=23, require_all_splits=True)
    by_root = {row["root_id"]: row for row in manifest}
    assert by_root["scenario-root-0"]["split_component_id"] == by_root["scenario-root-1"]["split_component_id"]
    assert by_root["scenario-root-0"]["split_component_id"] != by_root["scenario-root-2"]["split_component_id"]
    assert by_root["scenario-root-0"]["scenario_episode_key"] == by_root["scenario-root-1"]["scenario_episode_key"]
    assert by_root["scenario-root-0"]["scenario_episode_key_version"] == SCENARIO_EPISODE_KEY_VERSION

    provenance = json.loads((tmp_path / "manifests" / "split_provenance.json").read_text(encoding="utf-8"))
    assert provenance["cross_split_scenario_episode_overlap_count"] == 0
    assert provenance["missing_scenario_episode_key_count"] == 0
    assert provenance["scenario_episode_count"] == 8
    assert provenance["component_count"] == 8

    first_assignment = {
        row["root_id"]: (row["split_component_id"], row["split"])
        for row in manifest
    }
    (tmp_path / "manifests" / "roots.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in reversed(rows)),
        encoding="utf-8",
    )
    repeated = build_split_manifest(tmp_path, seed=23, require_all_splits=True)
    assert {
        row["root_id"]: (row["split_component_id"], row["split"])
        for row in repeated
    } == first_assignment


def test_schema3_split_rejects_missing_scenario_episode_contract(tmp_path: Path):
    rows = [
        {
            "root_id": f"root-{index}",
            "root_episode_id": f"policy:{index}",
            "episode_seed": index,
            "root_policy": "policy",
            "traffic_profile": "" if index == 0 else "hard",
            "root_observation_fingerprint": f"fingerprint-{index}",
            "complete": True,
        }
        for index in range(5)
    ]
    _write_roots(tmp_path, rows)
    (tmp_path / "manifests" / "dataset_manifest.json").write_text(
        json.dumps(
            {
                "counterfactual_schema_version": COUNTERFACTUAL_SCHEMA_VERSION,
                "data_contract": {"scenario_route_hash": "route"},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="scenario_route_hash, traffic_profile, and episode_seed"):
        build_split_manifest(tmp_path, seed=1)


def test_entry_time_regression_is_conditional_on_observed_entry():
    assert entry_time_supervision(
        target_lane_entry_time_s=2.5,
        censor_time_s=3.0,
        viability_status="observed_success",
    ) == (2.5, 1.0, True, 2.5, "")
    assert entry_time_supervision(
        target_lane_entry_time_s=None,
        censor_time_s=4.0,
        viability_status="observed_failure",
    ) == (0.0, 0.0, False, 4.0, "taper_miss")
    assert entry_time_supervision(
        target_lane_entry_time_s=None,
        censor_time_s=8.0,
        viability_status="censored",
    ) == (0.0, 0.0, False, 8.0, "horizon_elapsed")


def test_lane_entry_after_taper_miss_is_diagnostic_not_supervised():
    fields = _conditional_entry_time_fields(
        "observed_failure",
        target_lane_entry_time=4.0,
        censor_time=2.0,
    )
    assert fields == {
        "target_lane_entry_time_s": None,
        "target_lane_entry_time_raw_s": 4.0,
        "entry_time_observed": False,
        "entry_time_censor_time_s": 2.0,
        "entry_time_censor_reason": "taper_miss",
    }


def test_censored_taper_outcome_is_not_supervised_as_a_negative():
    root = {"activation_bin": "activation_window"}
    censored = event_supervision_mask({"event_observed": False}, root)
    observed = event_supervision_mask({"event_observed": True}, root)
    assert censored.tolist() == [1.0, 1.0, 0.0, 0.0]
    assert observed.tolist() == [1.0, 1.0, 1.0, 1.0]


def test_training_batches_keep_fingerprint_action_groups_atomic_across_boundaries():
    class Dataset:
        items = [
            {"sample_weight": np.asarray(0.5, dtype=np.float32), "value": np.asarray(1)},
            {"sample_weight": np.asarray(0.5, dtype=np.float32), "value": np.asarray(2)},
            {"sample_weight": np.asarray(1.0, dtype=np.float32), "value": np.asarray(3)},
        ]

        def __getitem__(self, index):
            return self.items[index]

    batches = list(
        _fingerprint_action_group_batches(
            Dataset(),
            [(0, 1), (2,)],
            group_batch_size=1,
        )
    )
    assert [batch["value"].tolist() for batch in batches] == [[1, 2], [3]]
    assert [float(batch["sample_weight"].sum()) for batch in batches] == [1.0, 1.0]


def test_component_bootstrap_shuffles_complete_fingerprint_action_groups():
    class Dataset:
        component_metadata_complete = True
        split_component_by_root = {
            "root-a": "component-a",
            "root-b": "component-b",
        }
        branch_indices_by_component = {
            "component-a": [0, 1],
            "component-b": [2],
        }
        branch_indices_by_fingerprint_action = {
            ("shared", 4): (0, 1),
            ("other", 7): (2,),
        }

    plan = build_component_bootstrap_plan(Dataset(), np.random.default_rng(7))
    groups = shuffled_component_bootstrap_groups(plan, np.random.default_rng(9))
    assert plan["version"] == COMPONENT_BOOTSTRAP_VERSION
    assert plan["within_component_weighting"] == OPTIMIZATION_BATCHING_VERSION
    assert sorted(groups) == sorted(plan["fixed_groups"])
    assert all(group in {(0, 1), (2,)} for group in groups)


def test_loss_v2_normalises_per_valid_scalar_and_smooths_residual():
    torch = pytest.importorskip("torch")
    target = torch.zeros((1, 1, 3, 5), dtype=torch.float32)
    output = {
        "actor_response": target.clone(),
        "event_logits": torch.zeros((1, 4), dtype=torch.float32),
        "geometry": torch.zeros((1, 5), dtype=torch.float32),
    }
    batch = {
        "actor_response": target,
        "actor_response_mask": torch.ones((1, 1, 3), dtype=torch.float32),
        "event_targets": torch.zeros((1, 4), dtype=torch.float32),
        "event_mask": torch.zeros((1, 4), dtype=torch.float32),
        "geometry_targets": torch.zeros((1, 5), dtype=torch.float32),
        "geometry_mask": torch.zeros((1, 5), dtype=torch.float32),
        "candidate_plan": torch.zeros((1, 3, 5), dtype=torch.float32),
    }
    _loss, parts = accvp_loss(output, batch, {"trajectory": 1.0, "events": 0.0, "geometry": 0.0})
    assert ACCVP_LOSS_VERSION == "accvp_loss_v2"
    assert float(parts["trajectory"]) == pytest.approx(0.0)
    assert float(parts["smoothness"]) == pytest.approx(0.0)

    # Correct constant-velocity motion has no residual curvature penalty.
    motion = target.clone()
    motion[..., 0] = torch.tensor([0.0, 1.0, 2.0])
    output["actor_response"] = motion.clone()
    batch["actor_response"] = motion
    _loss, parts = accvp_loss(output, batch, {"trajectory": 1.0, "events": 0.0, "geometry": 0.0})
    assert float(parts["smoothness"]) == pytest.approx(0.0)

    output["actor_response"] = motion.clone()
    output["actor_response"][..., 0] += torch.tensor([0.0, 1.0, 0.0])
    _loss, parts = accvp_loss(output, batch, {"trajectory": 1.0, "events": 0.0, "geometry": 0.0})
    assert float(parts["smoothness"]) > 0.0


def test_loss_v2_duplicate_sample_weights_preserve_group_mass():
    torch = pytest.importorskip("torch")

    def values(batch_size: int, weights: list[float]):
        output = {
            "actor_response": torch.ones((batch_size, 1, 3, 5), dtype=torch.float32),
            "event_logits": torch.full((batch_size, 4), 0.25, dtype=torch.float32),
            "geometry": torch.ones((batch_size, 5), dtype=torch.float32),
        }
        batch = {
            "actor_response": torch.zeros((batch_size, 1, 3, 5), dtype=torch.float32),
            "actor_response_mask": torch.ones((batch_size, 1, 3), dtype=torch.float32),
            "event_targets": torch.zeros((batch_size, 4), dtype=torch.float32),
            "event_mask": torch.ones((batch_size, 4), dtype=torch.float32),
            "geometry_targets": torch.zeros((batch_size, 5), dtype=torch.float32),
            "geometry_mask": torch.ones((batch_size, 5), dtype=torch.float32),
            "candidate_plan": torch.zeros((batch_size, 3, 5), dtype=torch.float32),
            "sample_weight": torch.tensor(weights, dtype=torch.float32),
        }
        return accvp_loss(output, batch)

    single_loss, single_parts = values(1, [1.0])
    repeated_loss, repeated_parts = values(2, [0.5, 0.5])
    scaled_loss, scaled_parts = values(1, [0.01])
    assert float(repeated_loss) == pytest.approx(float(single_loss))
    assert float(scaled_loss) == pytest.approx(float(single_loss))
    for name in ("trajectory", "events", "geometry", "ordering", "smoothness"):
        assert float(repeated_parts[name]) == pytest.approx(float(single_parts[name]))
        assert float(scaled_parts[name]) == pytest.approx(float(single_parts[name]))


def test_deployable_training_requires_three_member_ensemble():
    assert validate_ensemble_configuration(3, mode="deployable") == 3
    assert validate_ensemble_configuration(1, mode="shadow") == 1
    with pytest.raises(ValueError, match="at least 3"):
        validate_ensemble_configuration(1, mode="deployable")


def test_legacy_migration_audit_is_read_only_and_requires_unique_row_mapping(tmp_path: Path):
    _write_legacy_migration_fixture(tmp_path)
    roots_before = (tmp_path / "manifests" / "roots.jsonl").read_bytes()
    report = audit_legacy_dataset_migration(tmp_path)
    assert report["classification_counts"] == {
        "full_repairable": 1,
        "task_only": 0,
        "recollect_required": 0,
    }
    assert report["roots"][0]["actor_row_ids"] == ["front", "rear", ""]
    assert report["roots"][0]["legacy_to_actor_row_permutation"] == [1, 0, -1]
    assert report["schema3_derivation_allowed"] is True
    assert report["source_files_unchanged"] is True
    assert (tmp_path / "manifests" / "roots.jsonl").read_bytes() == roots_before


@pytest.mark.parametrize(
    ("mapping_available", "scalar_valid", "expected_class"),
    [(False, True, "task_only"), (True, False, "recollect_required")],
)
def test_legacy_migration_audit_never_guesses_or_promotes_bad_labels(
    tmp_path: Path,
    mapping_available: bool,
    scalar_valid: bool,
    expected_class: str,
):
    _write_legacy_migration_fixture(
        tmp_path,
        mapping_available=mapping_available,
        scalar_valid=scalar_valid,
    )
    report = audit_legacy_dataset_migration(tmp_path)
    assert report["roots"][0]["classification"] == expected_class
    assert report["schema3_derivation_allowed"] is False
