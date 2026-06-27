from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from safe_rl.accvp.dataset import build_split_manifest
from safe_rl.accvp.protocol import effective_activation_distance
from safe_rl.accvp.schema import file_sha256
from safe_rl.accvp.train import train_accvp
from safe_rl.sim.types import VehicleState
from safe_rl.utils.config import clone_with_overrides, load_config


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
    ego = VehicleState("ego", 0.0, 0.0, 0.0, 10.0, 0, "lane_0", 0.0, "main_aux").to_dict()
    for seed in range(1, 6):
        root_id = f"root_{seed}"
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
        )
        root_json.write_text(
            json.dumps(
                {
                    "root_id": root_id,
                    "root_ego": ego,
                    "step_length": float(cfg.scenario.step_length),
                    "candidate_plan_horizon_steps": int(cfg.accvp.candidate_plan_horizon_steps),
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
            )
            raw_failure = seed in {2, 5} and action_id == 4
            branches.append(
                {
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
                    "target_lane_entry_time_s": 1.0,
                    "secondary_risk": {"candidate_legal": True, "secondary_safety_pass": True},
                    "secondary_safety_pass": True,
                    "tensor_path": str(branch_npz),
                }
            )
    (manifests / "roots.jsonl").write_text("".join(json.dumps(row) + "\n" for row in roots), encoding="utf-8")
    (manifests / "branches.jsonl").write_text("".join(json.dumps(row) + "\n" for row in branches), encoding="utf-8")
    manifest = {
        "artifact_kind": "counterfactual_dataset_v2",
        "collection_phase": "formal",
        "dataset_fingerprint": "fixture-dataset",
        "data_contract_hash": "fixture-contract",
        "accvp_activation_distance_m": effective_activation_distance(cfg),
        "risk_model_fingerprint": "risk_checkpoint:fixture",
    }
    (manifests / "dataset_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    oracle = {
        "dataset_dir": str(dataset.resolve()),
        "oracle_state": "go",
        "go_for_training": True,
        "root_policy": "merge_timing",
        "required_seeds": [2, 5],
        "dataset_provenance": {
            "formal_dataset": True,
            "dataset_manifest_sha256": file_sha256(manifests / "dataset_manifest.json"),
            "roots_manifest_sha256": file_sha256(manifests / "roots.jsonl"),
            "branches_manifest_sha256": file_sha256(manifests / "branches.jsonl"),
            "dataset_fingerprint": manifest["dataset_fingerprint"],
            "data_contract_hash": manifest["data_contract_hash"],
        },
    }
    oracle_path = manifests / "oracle_report.json"
    oracle_path.write_text(json.dumps(oracle), encoding="utf-8")
    cfg.accvp["oracle_report"] = str(oracle_path)


def test_formal_training_initializes_training_before_loss_weights_and_writes_final_test(tmp_path: Path):
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
                "ensemble_size": 1,
                "response_horizon_steps": 2,
                "candidate_plan_horizon_steps": 4,
                "warm_start": {"enabled": False, "freeze_encoder_epochs": 0, "encoder_lr_multiplier": 0.1},
                "training": {"epochs": 1, "batch_size": 1, "learning_rate": 0.001, "weight_decay": 0.0, "ensemble_seed_offset": 1, "loss_weights": {"trajectory": 1.0, "events": 1.0, "geometry": 0.25, "ordering": 0.1, "smoothness": 0.01}},
                "tuning": {"required_availability": 1.0, "proxy_collision_upper_bounds": [1.0], "safety_violation_upper_bounds": [1.0], "merge_viability_lower_bounds": [0.0]},
            },
        },
    )
    dataset = tmp_path / "dataset"
    _write_minimal_formal_dataset(dataset, cfg)
    build_split_manifest(dataset, seed=3)
    checkpoint = train_accvp(cfg, dataset)
    output = checkpoint.parent
    assert checkpoint.exists()
    diagnostics = json.loads((output / "accvp_v1_final_test_diagnostics.json").read_text(encoding="utf-8"))
    assert diagnostics["split"] == "test"
    assert "post_selection" in diagnostics
    assert (output / "accvp_v1_operating_point.json").exists()
    assert (output / "accvp_v1_artifact_manifest.json").exists()


def test_formal_training_requires_strict_oracle_report(tmp_path: Path):
    cfg = clone_with_overrides(load_config(), {"accvp": {"oracle_report": None}})
    with pytest.raises(FileNotFoundError, match="oracle_report"):
        train_accvp(cfg, tmp_path / "dataset")
