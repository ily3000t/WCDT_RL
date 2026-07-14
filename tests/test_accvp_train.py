from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from safe_rl.accvp.availability import OperatingPointAvailabilityError
from safe_rl.accvp.artifacts import (
    ACCVP_ARTIFACT_GENERATION,
    ACCVP_ARTIFACT_KIND,
    ACCVP_BUNDLE_SCHEMA_VERSION,
    LIFECYCLE_SEALED_CANDIDATE,
    LIFECYCLE_SHADOW,
    artifact_filename,
    resolve_v2_bundle,
)
from safe_rl.accvp.dataset import build_split_manifest
from safe_rl.accvp.protocol import counterfactual_data_contract, effective_activation_distance
from safe_rl.accvp.runtime_contract import FORMAL_RUNTIME_FEATURE_VERSION
from safe_rl.accvp.schema import (
    COUNTERFACTUAL_SCHEMA_VERSION,
    ENTRY_TIME_LABEL_VERSION,
    ROOT_OBSERVATION_FINGERPRINT_VERSION,
    actor_row_mapping_hash,
    file_sha256,
    root_observation_fingerprint,
    stable_hash,
)
from safe_rl.accvp.train import (
    COMPONENT_BOOTSTRAP_VERSION,
    OPTIMIZATION_BATCHING_VERSION,
    _calibrate,
    _event_positive_weights,
    _tensor_batch,
    _train_response_feature_scales,
    build_component_bootstrap_plan,
    shuffled_component_bootstrap_groups,
    shuffled_component_bootstrap_indices,
    train_accvp,
)
from safe_rl.sim.types import VehicleState
from safe_rl.utils.config import clone_with_overrides, load_config


def _attach_minimal_formal_runtime_contract(cfg, directory: Path) -> None:
    """Bind the same minimal audited runtime semantics used by VNext config."""

    risk_checkpoint = directory / "risk_module_fixture.pt"
    risk_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    risk_checkpoint.write_bytes(b"risk-module-fixture")
    cfg.accvp["risk_checkpoint"] = str(risk_checkpoint)
    cfg.accvp["formal_runtime_contract"] = {
        "candidate_geometry_backend": "vectorized",
        "enabled": True,
        "feature_version": FORMAL_RUNTIME_FEATURE_VERSION,
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


def _write_minimal_formal_dataset(dataset: Path, cfg) -> None:
    manifests = dataset / "manifests"
    roots_dir = dataset / "roots"
    branches_dir = dataset / "branches"
    manifests.mkdir(parents=True)
    roots_dir.mkdir()
    branches_dir.mkdir()
    roots = []
    branches = []
    actors = int(cfg.accvp.actor_count)
    history = int(cfg.scenario.history_steps)
    response = int(cfg.accvp.response_horizon_steps)
    actor_row_ids = [f"actor_{index}" for index in range(actors)]
    source_indices = list(range(1, actors + 1))
    actor_mask = np.ones((actors,), dtype=np.float32)
    mapping_hash = actor_row_mapping_hash(actor_row_ids, source_indices, actor_mask)
    configured_risk = cfg.accvp.get("risk_checkpoint")
    risk_model_fingerprint = (
        f"risk_checkpoint:{file_sha256(configured_risk)}"
        if configured_risk
        else "risk_checkpoint:fixture"
    )
    contract = counterfactual_data_contract(cfg, risk_model_fingerprint)
    contract_hash = stable_hash(contract)
    for seed in range(101, 106):
        root_id = f"root_{seed}"
        # Each fixture root must represent a distinct model-visible state.
        # Otherwise the fingerprint-component splitter correctly joins all
        # five episode roots into one component and refuses a five-way split.
        ego = VehicleState("ego", float(seed), 0.0, 0.0, 10.0, 0, "lane_0", float(seed), "main_aux").to_dict()
        root_npz = roots_dir / f"{root_id}.npz"
        root_json = roots_dir / f"{root_id}.json"
        np.savez_compressed(
            root_npz,
            history_features=np.zeros((1, actors, history, 10), dtype=np.float32),
            history_valid_mask=np.ones((1, actors, history), dtype=np.float32),
            history_lane_ids=np.ones((1, actors, history), dtype=np.int64),
            history_edge_role_ids=np.ones((1, actors, history), dtype=np.int64),
            role_ids=np.ones((1, actors), dtype=np.int64),
            lane_ids=np.ones((1, actors), dtype=np.int64),
            edge_role_ids=np.ones((1, actors), dtype=np.int64),
            mask=np.ones((1, actors), dtype=np.float32),
            selected_indices=np.asarray(source_indices, dtype=np.int64)[None, :],
        )
        with np.load(root_npz, allow_pickle=False) as root_values:
            observation_fingerprint = root_observation_fingerprint(
                actor_row_ids=actor_row_ids,
                root_ego=ego,
                data_contract_hash=contract_hash,
                tensors=root_values,
            )
        root_json.write_text(
            json.dumps(
                {
                    "counterfactual_schema_version": COUNTERFACTUAL_SCHEMA_VERSION,
                    "root_id": root_id,
                    "root_ego": ego,
                    "step_length": float(cfg.scenario.step_length),
                    "candidate_plan_horizon_steps": int(cfg.accvp.candidate_plan_horizon_steps),
                    "actor_row_ids": actor_row_ids,
                    "actor_row_source_indices": source_indices,
                    "actor_row_mapping_version": mapping_hash.split(":", 1)[0],
                    "actor_row_mapping_hash": mapping_hash,
                    "data_contract_hash": contract_hash,
                    "root_observation_fingerprint_version": ROOT_OBSERVATION_FINGERPRINT_VERSION,
                    "root_observation_fingerprint": observation_fingerprint,
                }
            ),
            encoding="utf-8",
        )
        roots.append(
            {
                "root_id": root_id,
                "root_episode_id": f"ppo:{seed}",
                "episode_seed": seed,
                "root_policy": "merge_timing",
                "traffic_profile": "hard" if seed % 2 else "safe",
                "deadline_bin": "deadline",
                "raw_action_id": 4,
                "raw_action_legal": True,
                "root_observation_fingerprint": observation_fingerprint,
                "root_state_fingerprint": observation_fingerprint,
                "data_contract_hash": contract_hash,
                "metadata_path": str(root_json),
                "tensor_path": str(root_npz),
                "complete": True,
            }
        )
        for action_id in (4, 7):
            branch_npz = branches_dir / f"{root_id}_action{action_id}.npz"
            np.savez_compressed(
                branch_npz,
                actor_response=np.zeros((actors, response, 5), dtype=np.float32),
                actor_valid_mask=np.ones((actors, response), dtype=np.float32),
                actor_row_ids=np.asarray(actor_row_ids, dtype=np.str_),
            )
            raw_failure = seed in {2, 5} and action_id == 4
            branches.append(
                {
                    "counterfactual_schema_version": COUNTERFACTUAL_SCHEMA_VERSION,
                    "root_id": root_id,
                    "branch_id": f"{root_id}_action{action_id}",
                    "branch_status": "completed",
                    "action_id": action_id,
                    "event_observed": True,
                    "censor_time": 1.0,
                    "censor_reason": "",
                    "proxy_collision_within_horizon": raw_failure,
                    "safety_violation_within_horizon": raw_failure,
                    "taper_miss_observed": raw_failure,
                    "merge_before_taper_observed": not raw_failure,
                    "viability_observation_status": "observed_failure" if raw_failure else "observed_success",
                    "min_obb_distance": 5.0,
                    "max_drac": 0.1,
                    "target_front_gap": 10.0,
                    "target_rear_gap": 10.0,
                    "target_lane_entry_time_s": None if raw_failure else 1.0,
                    "entry_time_observed": not raw_failure,
                    "entry_time_censor_time_s": 1.0,
                    "entry_time_censor_reason": "taper_miss" if raw_failure else "",
                    "entry_time_label_version": ENTRY_TIME_LABEL_VERSION,
                    "actor_row_ids": actor_row_ids,
                    "actor_row_source_indices": source_indices,
                    "actor_row_mapping_hash": mapping_hash,
                    "data_contract_hash": contract_hash,
                    "risk_model_fingerprint": risk_model_fingerprint,
                    "secondary_risk": {"candidate_legal": True, "secondary_safety_pass": True},
                    "secondary_safety_pass": True,
                    "tensor_path": str(branch_npz),
                }
            )
    (manifests / "roots.jsonl").write_text("".join(json.dumps(row) + "\n" for row in roots), encoding="utf-8")
    (manifests / "branches.jsonl").write_text("".join(json.dumps(row) + "\n" for row in branches), encoding="utf-8")
    manifest = {
        "artifact_kind": "counterfactual_dataset_v2",
        "counterfactual_schema_version": COUNTERFACTUAL_SCHEMA_VERSION,
        "collection_phase": "formal",
        "data_contract": contract,
        "dataset_fingerprint": "fixture-dataset",
        "data_contract_hash": contract_hash,
        "accvp_activation_distance_m": effective_activation_distance(cfg),
        "risk_model_fingerprint": risk_model_fingerprint,
    }
    (manifests / "dataset_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    oracle_dataset = dataset.parent / "oracle_dataset"
    oracle_manifests = oracle_dataset / "manifests"
    oracle_manifests.mkdir(parents=True)
    oracle_manifest = dict(manifest)
    oracle_manifest["collection_phase"] = "ad_hoc"
    oracle_manifest["dataset_fingerprint"] = "fixture-oracle-dataset"
    (oracle_manifests / "dataset_manifest.json").write_text(
        json.dumps(oracle_manifest), encoding="utf-8"
    )
    oracle_roots = [
        {
            "root_id": f"oracle-root-{seed}",
            "episode_seed": seed,
            "root_policy": "merge_timing",
            "complete": True,
            "oracle_only": True,
            "cohort_role": "oracle_regression",
            "exclude_from_model_splits": True,
        }
        for seed in (2, 5)
    ]
    (oracle_manifests / "roots.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in oracle_roots), encoding="utf-8"
    )
    (oracle_manifests / "branches.jsonl").write_text("", encoding="utf-8")
    oracle = {
        "dataset_dir": str(oracle_dataset.resolve()),
        "oracle_state": "go",
        "go_for_training": True,
        "root_policy": "merge_timing",
        "required_seeds": [2, 5],
        "cohort_role": "oracle_regression",
        "oracle_only": True,
        "exclude_from_model_splits": True,
        "dataset_provenance": {
            "formal_dataset": True,
            "dataset_manifest_sha256": file_sha256(oracle_manifests / "dataset_manifest.json"),
            "roots_manifest_sha256": file_sha256(oracle_manifests / "roots.jsonl"),
            "branches_manifest_sha256": file_sha256(oracle_manifests / "branches.jsonl"),
            "dataset_fingerprint": oracle_manifest["dataset_fingerprint"],
            "data_contract_hash": oracle_manifest["data_contract_hash"],
        },
    }
    oracle_path = oracle_manifests / "oracle_report.json"
    oracle_path.write_text(json.dumps(oracle), encoding="utf-8")
    cfg.accvp["oracle_report"] = str(oracle_path)
    cfg.accvp["oracle"] = {
        "required_seeds": [2, 5],
        "cohort_role": "oracle_regression",
        "exclude_from_model_splits": True,
    }


def test_component_bootstrap_is_fixed_per_member_and_clusters_all_rows():
    class Dataset:
        component_metadata_complete = True
        split_component_by_root = {
            "root-a": "component-a",
            "root-b": "component-b",
            "root-c": "component-c",
        }
        branch_indices_by_component = {
            "component-a": [0, 1],
            "component-b": [2, 3, 4],
            "component-c": [5],
        }
        branch_indices_by_fingerprint_action = {
            ("fingerprint-a", 4): (0, 1),
            ("fingerprint-b", 4): (2,),
            ("fingerprint-c", 7): (3, 4),
            ("fingerprint-d", 8): (5,),
        }

    first = build_component_bootstrap_plan(Dataset(), np.random.default_rng(17))
    repeated = build_component_bootstrap_plan(Dataset(), np.random.default_rng(17))
    assert first["version"] == COMPONENT_BOOTSTRAP_VERSION
    assert first["within_component_weighting"] == OPTIMIZATION_BATCHING_VERSION
    assert first["sampled_components"] == repeated["sampled_components"]
    assert first["component_multiset_hash"] == repeated["component_multiset_hash"]

    fixed_counts = Counter(first["fixed_indices"])
    for component_id, indices in Dataset.branch_indices_by_component.items():
        multiplicity = first["component_multiplicities"].get(component_id, 0)
        assert all(fixed_counts[index] == multiplicity for index in indices)

    rng = np.random.default_rng(99)
    epoch_one = shuffled_component_bootstrap_indices(first, rng)
    epoch_two = shuffled_component_bootstrap_indices(first, rng)
    assert sorted(epoch_one) == sorted(first["fixed_indices"])
    assert sorted(epoch_two) == sorted(first["fixed_indices"])
    assert first["fixed_indices"] == repeated["fixed_indices"]
    grouped_epoch = shuffled_component_bootstrap_groups(first, np.random.default_rng(99))
    assert sorted(grouped_epoch) == sorted(first["fixed_groups"])
    assert all(group in first["fixed_groups"] for group in grouped_epoch)


def test_component_bootstrap_rejects_missing_split_component_metadata():
    class Dataset:
        component_metadata_complete = False
        split_component_by_root = {"root": ""}
        branch_indices_by_component = {}

    with pytest.raises(ValueError, match="split_component_id"):
        build_component_bootstrap_plan(Dataset(), np.random.default_rng(3))


def test_duplicate_sample_weights_feed_tensor_loss_balancing_and_normalization():
    class Dataset:
        def __init__(self):
            first = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
            second = np.asarray([10.0, 10.0, 0.0, 10.0, 10.0], dtype=np.float32)
            self.items = [
                {
                    "sample_weight": np.asarray(0.25, dtype=np.float32),
                    "event_mask": np.ones((4,), dtype=np.float32),
                    "event_targets": np.ones((4,), dtype=np.float32),
                    "actor_response": first.reshape((1, 1, 5)),
                    "actor_response_mask": np.ones((1, 1), dtype=np.float32),
                },
                {
                    "sample_weight": np.asarray(0.75, dtype=np.float32),
                    "event_mask": np.ones((4,), dtype=np.float32),
                    "event_targets": np.zeros((4,), dtype=np.float32),
                    "actor_response": second.reshape((1, 1, 5)),
                    "actor_response_mask": np.ones((1, 1), dtype=np.float32),
                },
            ]

        def __len__(self):
            return len(self.items)

        def __getitem__(self, index):
            return self.items[index]

    dataset = Dataset()
    assert _event_positive_weights(dataset) == pytest.approx([3.0, 3.0, 3.0, 3.0])
    scales = _train_response_feature_scales(dataset, [0.1] * 5)
    expected = float(np.sqrt(18.75))
    assert scales == pytest.approx([expected, expected, 0.1, expected, expected])
    tensors = _tensor_batch(
        {
            "candidate_action_ids": np.asarray([4, 7], dtype=np.int64),
            "sample_weight": np.asarray([0.25, 0.75], dtype=np.float32),
        },
        torch,
    )
    assert tensors["sample_weight"].dtype == torch.float32
    assert tensors["sample_weight"].tolist() == pytest.approx([0.25, 0.75])


def test_calibration_collapses_fingerprint_action_groups_to_independent_units():
    class Dataset:
        rows = [
            {"root_id": "root-a", "action_id": 4},
            {"root_id": "root-b", "action_id": 4},
            {"root_id": "root-c", "action_id": 7},
        ]
        observation_fingerprint_by_root = {
            "root-a": "shared-fingerprint",
            "root-b": "shared-fingerprint",
            "root-c": "independent-fingerprint",
        }
        split_component_by_root = {
            "root-a": "component-shared",
            "root-b": "component-shared",
            "root-c": "component-independent",
        }
        duplicate_weighting_version = "fingerprint_action_total_weight_one_v1"
        duplicate_group_count = 2
        duplicate_row_count = 1

        def __len__(self):
            return 3

        def __getitem__(self, index):
            target = 1.0 if index == 0 else 0.0
            return {
                "history_features": np.zeros((1, 1, 10), dtype=np.float32),
                "history_valid_mask": np.ones((1, 1), dtype=np.float32),
                "history_lane_ids": np.zeros((1, 1), dtype=np.int64),
                "history_edge_role_ids": np.zeros((1, 1), dtype=np.int64),
                "role_ids": np.zeros((1,), dtype=np.int64),
                "lane_ids": np.zeros((1,), dtype=np.int64),
                "edge_role_ids": np.zeros((1,), dtype=np.int64),
                "actor_mask": np.ones((1,), dtype=np.float32),
                "candidate_plan": np.zeros((2, 5), dtype=np.float32),
                "candidate_action_ids": np.asarray(self.rows[index]["action_id"], dtype=np.int64),
                "sample_weight": np.asarray(0.5 if index < 2 else 1.0, dtype=np.float32),
                "event_targets": np.asarray([target, target, 0.0, 1.0 - target], dtype=np.float32),
                "event_mask": np.ones((4,), dtype=np.float32),
            }

    class Model:
        def eval(self):
            return self

        def __call__(self, history_features, *args):
            return {
                "event_logits": torch.zeros(
                    (history_features.shape[0], 4), dtype=torch.float32
                )
            }

    calibration = _calibrate(
        [Model()],
        Dataset(),
        torch,
        {"bins": 2, "nominal_alpha": 0.05, "bonferroni_signal_count": 3},
    )
    provenance = calibration.provenance
    assert provenance["raw_candidate_count"] == 3
    assert provenance["effective_fingerprint_action_group_count"] == 2
    assert provenance["raw_eligible_viability_count"] == 3
    assert provenance["effective_eligible_viability_group_count"] == 2
    assert provenance["duplicate_weighting_applied_to_component_bound"] is True
    assert provenance["wilson_independence_assumption_used"] is False
    assert provenance["calibration_statistical_unit"] == "split_component_x_score_bin"
    assert provenance["calibration_estimand"] == (
        "equal_weight_split_component_mean_within_score_bin_v1"
    )
    assert provenance["effective_split_component_count"]["proxy_collision"] == 2
    assert calibration.proxy_collision.method == "one_sided_hoeffding_component_mean_v1"
    assert provenance["group_label_aggregation"] == "mean_stochastic_outcome_v1"


def test_formal_training_writes_sealed_candidate_without_opening_final_test(tmp_path: Path):
    cfg = clone_with_overrides(
        load_config(),
        {
            "run": {"output_root": str(tmp_path / "output"), "run_id": "accvp_train_test", "tensorboard": False},
            "prediction": {
                "wcdt_v3_hidden_dim": 16,
                "wcdt_v3_temporal_layers": 1,
                "wcdt_v3_actor_attention_layers": 1,
                "wcdt_v3_num_heads": 4,
            },
            "accvp": {
                "artifact_generation": ACCVP_ARTIFACT_GENERATION,
                "ensemble_size": 3,
                "response_horizon_steps": 2,
                "candidate_plan_horizon_steps": 4,
                "warm_start": {"enabled": False, "freeze_encoder_epochs": 0, "encoder_lr_multiplier": 0.1},
                "training": {"epochs": 1, "batch_size": 1, "learning_rate": 0.001, "weight_decay": 0.0, "ensemble_seed_offset": 1, "loss_weights": {"trajectory": 1.0, "events": 1.0, "geometry": 0.25, "ordering": 0.1, "smoothness": 0.01}},
                "tuning": {"required_availability": 1.0, "proxy_collision_upper_bounds": [1.0], "safety_violation_upper_bounds": [1.0], "merge_viability_lower_bounds": [0.0]},
            },
        },
    )
    _attach_minimal_formal_runtime_contract(cfg, tmp_path)
    dataset = tmp_path / "dataset"
    _write_minimal_formal_dataset(dataset, cfg)
    build_split_manifest(dataset, seed=3)
    expected_output = tmp_path / "output" / "accvp_train_test" / "accvp"
    expected_output.mkdir(parents=True)
    legacy_sentinel = expected_output / "accvp_v1_predictor.pt"
    legacy_sentinel.write_bytes(b"legacy-must-not-be-cleaned")
    checkpoint = train_accvp(cfg, dataset)
    output = checkpoint.parent
    assert checkpoint.exists()
    assert checkpoint.name == artifact_filename("predictor")
    assert legacy_sentinel.read_bytes() == b"legacy-must-not-be-cleaned"
    payload = torch.load(checkpoint, map_location="cpu")
    members = payload["metadata"]["warm_start"]["members"]
    assert len(members) == 3
    assert all(member["bootstrap_sampling_version"] == COMPONENT_BOOTSTRAP_VERSION for member in members)
    assert all(member["bootstrap_component_multiset_hash"] for member in members)
    assert all(len(member["bootstrap_epoch_order_hashes"]) == 1 for member in members)
    history_path = output / artifact_filename("training_history")
    history = json.loads(history_path.read_text(encoding="utf-8"))
    assert history["artifact_generation"] == ACCVP_ARTIFACT_GENERATION
    assert history["configured_artifact_generation"] == ACCVP_ARTIFACT_GENERATION
    assert history["reproducibility"]["effective_deterministic"] is True
    assert history["reproducibility"]["deterministic_algorithms"] is True
    assert history["duplicate_weighting"]["version"] == "fingerprint_action_total_weight_one_v1"
    assert history["duplicate_weighting"]["weighting_unit"] == "model_input_fingerprint_x_action"
    assert history["duplicate_weighting"]["cluster_sampling_unit"] == "split_component"
    assert history["duplicate_weighting"]["statistical_independence_claim"] is False
    assert history["optimization_batching"] == {
        "version": OPTIMIZATION_BATCHING_VERSION,
        "batch_size_unit": "model_input_fingerprint_x_action_group",
        "groups_are_indivisible": True,
        "group_total_sample_weight": 1.0,
    }
    assert history["calibration_independence_provenance"][
        "duplicate_weighting_applied_to_component_bound"
    ] is True
    assert history["calibration_independence_provenance"][
        "calibration_statistical_unit"
    ] == "split_component_x_score_bin"
    assert history["calibration_independence_provenance"][
        "calibration_estimand"
    ] == "equal_weight_split_component_mean_within_score_bin_v1"
    assert len(history["members"]) == 3
    for member in history["members"]:
        assert member["best_epoch"] == 0
        assert len(member["epochs"]) == 1
        epoch = member["epochs"][0]
        assert epoch["selected_best"] is True
        assert epoch["train_order_sha256"]
        assert epoch["train_batching_version"] == OPTIMIZATION_BATCHING_VERSION
        assert epoch["train_fingerprint_action_group_count"] >= 1
        assert epoch["validation_order_sha256"]
        assert epoch["learning_rates"]["scene_encoder"] > 0.0
        assert epoch["learning_rates"]["heads"] > 0.0
        for split_name in ("train", "validation"):
            reduction = epoch[split_name]
            assert reduction["batch_count"] >= 1
            assert reduction["row_count"] >= 1
            assert reduction["sample_weight_sum"] > 0.0
            expected_total = 0.0
            for component in reduction["components"].values():
                assert np.isfinite(component["numerator"])
                assert component["denominator"] >= 0.0
                if component["denominator"] > 0.0:
                    assert component["value"] == pytest.approx(
                        component["numerator"] / component["denominator"]
                    )
                else:
                    assert component["numerator"] == pytest.approx(0.0)
                    assert component["value"] == pytest.approx(0.0)
                expected_total += component["weighted_value"]
            assert reduction["total"] == pytest.approx(expected_total)
    assert payload["metadata"]["training_history"]["sha256"] == file_sha256(history_path)
    assert payload["metadata"]["training_history"]["history_fingerprint"] == history["history_fingerprint"]
    assert (output / artifact_filename("operating_point")).exists()
    operating_point = json.loads(
        (output / artifact_filename("operating_point")).read_text(encoding="utf-8")
    )
    assert operating_point["decision_weighting_version"]
    assert operating_point["availability_denominator_version"] == (
        "risk_eligible_raw_or_merge_left_v1"
    )
    assert operating_point["selected"]["model_conditional_availability"] == 1.0
    assert operating_point["selected"]["unconditional_candidate_set_availability"] == 1.0
    assert operating_point["duplicate_weighting_provenance"][
        "duplicate_weighting_applied_to_threshold_selection"
    ] is True
    manifest_path = output / artifact_filename("candidate_manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["artifact_kind"] == ACCVP_ARTIFACT_KIND
    assert manifest["bundle_schema_version"] == ACCVP_BUNDLE_SCHEMA_VERSION
    assert manifest["artifact_generation"] == ACCVP_ARTIFACT_GENERATION
    assert manifest["artifact_variant"] == "full_candidate_gate_v1"
    assert manifest["configured_artifact_generation"] == ACCVP_ARTIFACT_GENERATION
    assert manifest["lifecycle_state"] == LIFECYCLE_SEALED_CANDIDATE
    assert manifest["deployable_artifact"] is False
    assert manifest["hard_realtime_claim"] is False
    assert manifest["safety_certified"] is False
    assert manifest["holdout_state"] == "sealed"
    assert manifest["threshold_selection_split"] == "operating_point"
    assert manifest["test_used_for_threshold_selection"] is False
    assert manifest["training_history_sha256"] == file_sha256(history_path)
    assert set(manifest["files"]) == {
        "predictor",
        "calibration",
        "operating_point",
        "training_history",
    }
    assert all(not Path(record["path"]).is_absolute() for record in manifest["files"].values())
    resolved_manifest, resolved_files = resolve_v2_bundle(manifest_path)
    assert resolved_manifest == manifest
    assert resolved_files["predictor"] == checkpoint.resolve()
    assert resolved_files["training_history"] == history_path.resolve()
    assert not (output / "accvp_v1_operating_point.json").exists()
    assert not (output / "accvp_v1_candidate_artifact_manifest.json").exists()
    training_manifest = json.loads(
        (output / artifact_filename("training_manifest")).read_text(encoding="utf-8")
    )
    assert training_manifest["training_history_sha256"] == file_sha256(history_path)
    assert training_manifest["artifact_manifest_sha256"] == file_sha256(manifest_path)


def test_shadow_training_writes_non_deployable_shadow_artifact(tmp_path: Path):
    cfg = clone_with_overrides(
        load_config(),
        {
            "run": {"output_root": str(tmp_path / "output"), "run_id": "accvp_shadow_train_test", "tensorboard": False},
            "prediction": {
                "wcdt_v3_hidden_dim": 16,
                "wcdt_v3_temporal_layers": 1,
                "wcdt_v3_actor_attention_layers": 1,
                "wcdt_v3_num_heads": 4,
            },
            "accvp": {
                "ensemble_size": 1,
                "response_horizon_steps": 2,
                "candidate_plan_horizon_steps": 4,
                "warm_start": {"enabled": False, "freeze_encoder_epochs": 0, "encoder_lr_multiplier": 0.1},
                "training": {"epochs": 1, "batch_size": 1, "learning_rate": 0.001, "weight_decay": 0.0, "ensemble_seed_offset": 1, "loss_weights": {"trajectory": 1.0, "events": 1.0, "geometry": 0.25, "ordering": 0.1, "smoothness": 0.01}},
                "tuning": {"required_availability": 1.0, "proxy_collision_upper_bounds": [1.0], "safety_violation_upper_bounds": [1.0], "merge_viability_lower_bounds": [0.0]},
            },
        },
    )
    _attach_minimal_formal_runtime_contract(cfg, tmp_path)
    dataset = tmp_path / "dataset"
    _write_minimal_formal_dataset(dataset, cfg)
    build_split_manifest(dataset, seed=3)
    checkpoint = train_accvp(cfg, dataset, mode="shadow")
    output = checkpoint.parent
    assert checkpoint.exists()
    assert checkpoint.name == artifact_filename("predictor")
    assert (output / artifact_filename("calibration")).exists()
    assert not (output / artifact_filename("operating_point")).exists()
    manifest = json.loads(
        (output / artifact_filename("shadow_manifest")).read_text(encoding="utf-8")
    )
    assert manifest["artifact_kind"] == ACCVP_ARTIFACT_KIND
    assert manifest["bundle_schema_version"] == ACCVP_BUNDLE_SCHEMA_VERSION
    assert manifest["lifecycle_state"] == LIFECYCLE_SHADOW
    assert manifest["deployable_artifact"] is False
    assert manifest["holdout_state"] == "not_applicable"
    assert set(manifest["files"]) == {"predictor", "calibration", "training_history"}
    history = json.loads(
        (output / artifact_filename("training_history")).read_text(encoding="utf-8")
    )
    assert history["reproducibility"]["effective_deterministic"] is False
    assert history["reproducibility"]["deterministic_algorithms"] is False
    assert not (output / "accvp_v1_calibration.json").exists()
    assert not (output / "accvp_v1_shadow_artifact_manifest.json").exists()


def test_formal_training_requires_strict_oracle_report(tmp_path: Path, monkeypatch):
    def unexpected_torch_initialization():
        raise AssertionError("Torch must not initialize before oracle validation")

    monkeypatch.setattr("safe_rl.accvp.train._torch", unexpected_torch_initialization)
    cfg = clone_with_overrides(load_config(), {"accvp": {"oracle_report": None, "ensemble_size": 3}})
    with pytest.raises(FileNotFoundError, match="oracle_report"):
        train_accvp(cfg, tmp_path / "dataset")


def test_formal_training_rejects_wrong_artifact_generation(tmp_path: Path):
    cfg = clone_with_overrides(
        load_config(),
        {"accvp": {"artifact_generation": "legacy_v1", "ensemble_size": 3}},
    )
    with pytest.raises(ValueError, match="artifact_generation mismatch"):
        train_accvp(cfg, tmp_path / "dataset")


def test_tuning_failure_writes_non_deployable_diagnostics_without_artifact(tmp_path: Path):
    cfg = clone_with_overrides(
        load_config(),
        {
            "run": {"output_root": str(tmp_path / "output"), "run_id": "accvp_train_failure_test", "tensorboard": False},
            "prediction": {
                "wcdt_v3_hidden_dim": 16,
                "wcdt_v3_temporal_layers": 1,
                "wcdt_v3_actor_attention_layers": 1,
                "wcdt_v3_num_heads": 4,
            },
            "accvp": {
                "ensemble_size": 3,
                "response_horizon_steps": 2,
                "candidate_plan_horizon_steps": 4,
                "warm_start": {"enabled": False, "freeze_encoder_epochs": 0, "encoder_lr_multiplier": 0.1},
                "training": {"epochs": 1, "batch_size": 1, "learning_rate": 0.001, "weight_decay": 0.0, "ensemble_seed_offset": 1, "loss_weights": {"trajectory": 1.0, "events": 1.0, "geometry": 0.25, "ordering": 0.1, "smoothness": 0.01}},
                "tuning": {"required_availability": 0.95, "proxy_collision_upper_bounds": [-0.1], "safety_violation_upper_bounds": [1.0], "merge_viability_lower_bounds": [0.0]},
            },
        },
    )
    _attach_minimal_formal_runtime_contract(cfg, tmp_path)
    dataset = tmp_path / "dataset"
    _write_minimal_formal_dataset(dataset, cfg)
    build_split_manifest(dataset, seed=3)
    with pytest.raises(OperatingPointAvailabilityError):
        train_accvp(cfg, dataset)
    output = tmp_path / "output" / "accvp_train_failure_test" / "accvp"
    failure = output / artifact_filename("tuning_failure")
    assert failure.exists()
    payload = json.loads(failure.read_text(encoding="utf-8"))
    assert payload["deployable_artifact"] is False
    assert payload["artifact_generation"] == ACCVP_ARTIFACT_GENERATION
    assert "model_gate_best_availability" in payload
    assert not (output / artifact_filename("candidate_manifest")).exists()
    assert not (output / artifact_filename("predictor")).exists()
    assert not (output / artifact_filename("training_history")).exists()
