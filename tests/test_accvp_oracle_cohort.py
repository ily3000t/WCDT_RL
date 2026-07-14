from __future__ import annotations

import json
from pathlib import Path

import pytest

from safe_rl.accvp.data.dataset import ACCVPBranchDataset, build_split_manifest
from safe_rl.accvp.contracts.schema import read_json


def _write_roots(dataset: Path, roots: list[dict]) -> None:
    manifests = dataset / "manifests"
    manifests.mkdir(parents=True)
    (manifests / "dataset_manifest.json").write_text(
        json.dumps(
            {
                "artifact_kind": "counterfactual_dataset_v2",
                "counterfactual_schema_version": 2,
                "scenario_route_hash": "fixture-route",
            }
        ),
        encoding="utf-8",
    )
    (manifests / "roots.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in roots),
        encoding="utf-8",
    )
    (manifests / "branches.jsonl").write_text("", encoding="utf-8")


def _root(seed: int, *, oracle_only: bool = False) -> dict:
    return {
        "root_id": f"root-{seed}",
        "root_episode_id": f"merge_timing:{seed}",
        "episode_seed": seed,
        "root_policy": "merge_timing",
        "traffic_profile": "fixture",
        "activation_bin": "activation_window",
        "complete": True,
        "oracle_only": oracle_only,
        "cohort_role": "oracle_regression" if oracle_only else "data_collection",
        "exclude_from_model_splits": oracle_only,
    }


def test_split_always_excludes_explicit_oracle_only_roots(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    roots = [_root(seed) for seed in range(101, 106)] + [
        _root(2, oracle_only=True),
        _root(5, oracle_only=True),
    ]
    _write_roots(dataset, roots)

    rows = build_split_manifest(dataset, seed=7)

    assert {int(row["episode_seed"]) for row in rows} == set(range(101, 106))
    provenance = read_json(dataset / "manifests" / "split_provenance.json")
    assert provenance["oracle_exclusion_contract_version"] == "oracle_cohort_exclusion_v1"
    assert provenance["excluded_oracle_root_count"] == 2
    assert provenance["excluded_oracle_root_reason_counts"] == {"oracle_only": 2}


def test_split_can_exclude_registered_oracle_seeds_without_legacy_markers(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    roots = [_root(seed) for seed in range(101, 106)] + [_root(2), _root(5)]
    _write_roots(dataset, roots)

    rows = build_split_manifest(
        dataset,
        seed=7,
        excluded_episode_seeds=[2, 5],
        excluded_cohort_roles=["oracle_regression"],
    )

    assert {int(row["episode_seed"]) for row in rows} == set(range(101, 106))
    provenance = read_json(dataset / "manifests" / "split_provenance.json")
    assert provenance["configured_excluded_episode_seeds"] == [2, 5]
    assert provenance["excluded_oracle_root_reason_counts"] == {"episode_seed": 2}


def test_dataset_rejects_forged_split_that_reintroduces_oracle_root(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    roots = [_root(seed) for seed in range(101, 106)] + [_root(2, oracle_only=True)]
    _write_roots(dataset, roots)
    rows = build_split_manifest(dataset, seed=7)
    rows.append({"root_id": "root-2", "split": "train", "split_component_id": "forged"})
    (dataset / "manifests" / "split_manifest.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="illegally includes oracle-only roots"):
        ACCVPBranchDataset(dataset, "train")
