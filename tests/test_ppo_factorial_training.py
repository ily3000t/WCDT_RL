from __future__ import annotations

import copy
import concurrent.futures
import json
from pathlib import Path

import pytest
import yaml

from safe_rl.accvp.contracts.schema import file_sha256
from safe_rl.evaluation_protocol import stable_hash
from safe_rl.pipeline import stage3_train_ppo_factorial as factorial_stage3
from safe_rl.ppo_factorial import (
    EXPECTED_CANDIDATE_METHOD_ROLES,
    EXPECTED_FINAL_METHOD_ID,
    atomic_write_json,
    build_factorial_manifest,
    fingerprint_payload,
    load_factorial_contract,
    read_json_mapping,
    validate_factorial_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / "safe_rl/config/active/accvp_vnext/workflow.yaml"
MATRIX = ROOT / "safe_rl/config/active/accvp_vnext/ppo_ablation_matrix.yaml"
SEEDS = [1001, 1002, 1003, 1004, 1005]
OBSERVATION_PAYLOAD = {"family": "candidate"}
OBSERVATION_HASH = stable_hash(OBSERVATION_PAYLOAD)


def test_factorial_atomic_manifest_write_retries_transient_windows_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "manifest.json"
    output.write_text("{}", encoding="utf-8")
    original_replace = Path.replace
    attempts = 0

    def flaky_replace(source: Path, target: Path) -> Path:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise PermissionError("transient Windows file lock")
        return original_replace(source, target)

    monkeypatch.setattr(Path, "replace", flaky_replace)
    atomic_write_json(output, {"status": "complete"}, replace=True)
    assert attempts == 3
    assert json.loads(output.read_text(encoding="utf-8")) == {
        "status": "complete"
    }


def test_factorial_contract_freezes_four_candidate_methods_and_final_role():
    contract = load_factorial_contract(WORKFLOW, MATRIX)
    assert contract["method_roles"] == EXPECTED_CANDIDATE_METHOD_ROLES
    assert contract["method_ids"] == list(EXPECTED_CANDIDATE_METHOD_ROLES)
    assert contract["final_method_id"] == EXPECTED_FINAL_METHOD_ID
    assert "wcdt_reward_v2" not in contract["method_ids"]
    assert "no_forecast_reward_v2" not in contract["method_ids"]


def test_factorial_contract_rejects_reward_v2_as_final_method(tmp_path: Path):
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    workflow["final_method_id"] = "candidate_table_reward_v2"
    source = tmp_path / "workflow.yaml"
    source.write_text(yaml.safe_dump(workflow, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="Reward-v3.1"):
        load_factorial_contract(source, MATRIX)


def _fake_replicate_plan(
    *,
    method_id: str,
    optimizer_seeds: list[int],
    output_root: str | Path,
    run_id_prefix: str,
    **_kwargs,
):
    root = Path(output_root).resolve()
    observation_hash = OBSERVATION_HASH
    reward_hash = stable_hash({"method_id": method_id})
    records = []
    configs = []
    for seed in optimizer_seeds:
        run_id = f"{run_id_prefix}_seed_{seed}"
        config_path = root / "generated_configs" / f"{run_id}.yaml"
        payload = {
            "run": {
                "run_id": run_id,
                "seed": 20001,
                "output_root": str(root.parent.parent / "ppo_runs"),
            },
            "rl": {
                "optimizer_seed": seed,
                "total_timesteps": 100,
                "n_steps": 10,
                "batch_size": 5,
            },
            "training": {"ppo_num_envs": 1},
            "stage3": {"model_name": "ppo_model.zip"},
            "experiment": {"method_id": method_id, "optimizer_seed": seed},
        }
        records.append(
            {
                "method_id": method_id,
                "training_seed": seed,
                "optimizer_seed": seed,
                "simulator_training_start_seed": 20001,
                "run_id": run_id,
                "resolved_config": str(config_path),
                "resolved_config_sha256": "",
                "checkpoint": "",
                "checkpoint_sha256": "",
                "stage3_report": "",
                "stage3_report_sha256": "",
                "training_budget": {
                    "total_timesteps": 100,
                    "n_steps": 10,
                    "batch_size": 5,
                    "ppo_num_envs": 1,
                },
                "reward_semantics_hash": reward_hash,
                "reward_semantics": {"method_id": method_id},
                "observation_contract_hash": observation_hash,
                "observation_contract": OBSERVATION_PAYLOAD,
                "accvp_artifact_fingerprint": "artifact-vnext",
            }
        )
        configs.append((config_path, payload))
    manifest = {
        "artifact_kind": "ppo_optimizer_replicate_manifest_v1",
        "schema_version": 1,
        "status": "planned",
        "method_id": method_id,
        "minimum_optimizer_replicates": 5,
        "optimizer_seed_role": "ppo_optimizer_replicates",
        "optimizer_seeds": list(optimizer_seeds),
        "simulator_training_start_seed": 20001,
        "template_config": "template.yaml",
        "template_config_sha256": "b" * 64,
        "ablation_matrix": str(MATRIX),
        "ablation_matrix_sha256": file_sha256(MATRIX),
        "records": records,
    }
    manifest["plan_fingerprint"] = stable_hash(manifest)
    return manifest, configs


def _install_fast_training_fakes(monkeypatch, *, calls: list[tuple[str, int]]):
    monkeypatch.setattr(factorial_stage3, "build_replicate_plan", _fake_replicate_plan)

    def fake_reward(config):
        method_id = str(config["experiment"]["method_id"])
        return {"payload": {"method_id": method_id}, "sha256": stable_hash({"method_id": method_id})}

    def fake_observation(_config, *, require_artifacts):
        assert require_artifacts is True
        return {"payload": OBSERVATION_PAYLOAD, "sha256": OBSERVATION_HASH}

    def fake_train(config):
        method_id = str(config.experiment.method_id)
        seed = int(config.rl.optimizer_seed)
        calls.append((method_id, seed))
        run_dir = Path(str(config.run.output_root)) / str(config.run.run_id) / "stage3"
        run_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = run_dir / str(config.stage3.model_name)
        checkpoint.write_bytes(f"{method_id}:{seed}".encode("utf-8"))
        report = {
            "optimizer_seed": seed,
            "reward_semantics_hash": stable_hash({"method_id": method_id}),
            "observation_contract_hash": OBSERVATION_HASH,
            "actual_total_timesteps": 100,
            "checkpoint_selection_seed_sha256": stable_hash({"seed": seed}),
        }
        (run_dir / "stage3_training_report.json").write_text(
            json.dumps(report), encoding="utf-8"
        )
        return checkpoint

    monkeypatch.setattr(factorial_stage3, "validate_reward_semantics", fake_reward)
    monkeypatch.setattr(factorial_stage3, "observation_contract", fake_observation)
    monkeypatch.setattr(factorial_stage3.stage3_train_ppo, "run", fake_train)


def _template(path: Path) -> Path:
    path.write_text("run:\n  run_id: template\n", encoding="utf-8")
    return path


def test_factorial_plan_records_bounded_optimizer_parallelism(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setattr(factorial_stage3, "build_replicate_plan", _fake_replicate_plan)
    template = tmp_path / "template_parallel.yaml"
    template.write_text(
        "run:\n  run_id: template\n"
        "training:\n  max_parallel_optimizer_replicates: 2\n",
        encoding="utf-8",
    )
    plan = factorial_stage3.build_factorial_plan(
        config_path=template,
        matrix_path=MATRIX,
        workflow_path=WORKFLOW,
        optimizer_seeds=SEEDS,
        output_root=tmp_path / "factorial_parallel",
        require_artifacts=False,
    )
    assert plan["execution_parallelism"] == {
        "max_parallel_optimizer_replicates": 2,
        "start_method": "spawn",
        "manifest_writer": "parent_only",
    }


def _first_generated_run_dir(output_root: Path) -> Path:
    method_id = next(iter(EXPECTED_CANDIDATE_METHOD_ROLES))
    child = read_json_mapping(
        output_root / "methods" / method_id / "ppo_replicate_manifest.json"
    )
    resolved = yaml.safe_load(
        Path(child["records"][0]["resolved_config"]).read_text(encoding="utf-8")
    )
    return Path(resolved["run"]["output_root"]) / str(resolved["run"]["run_id"])


def test_factorial_coordinator_trains_twenty_unique_checkpoints_and_resumes(
    tmp_path: Path, monkeypatch
):
    calls: list[tuple[str, int]] = []
    _install_fast_training_fakes(monkeypatch, calls=calls)
    template = _template(tmp_path / "template.yaml")
    output_root = tmp_path / "factorial"
    output = factorial_stage3.run(
        config_path=template,
        matrix_path=MATRIX,
        workflow_path=WORKFLOW,
        optimizer_seeds=SEEDS,
        output_root=output_root,
    )
    assert len(calls) == 20
    assert set(calls) == {
        (method_id, seed)
        for method_id in EXPECTED_CANDIDATE_METHOD_ROLES
        for seed in SEEDS
    }
    manifest = validate_factorial_manifest(output)
    assert manifest["status"] == "complete"
    assert manifest["final_method_id"] == EXPECTED_FINAL_METHOD_ID
    assert len(manifest["checkpoint_sha256s"]) == 20
    assert len(set(manifest["checkpoint_sha256s"])) == 20

    calls.clear()
    resumed = factorial_stage3.run(
        config_path=template,
        matrix_path=MATRIX,
        workflow_path=WORKFLOW,
        optimizer_seeds=SEEDS,
        output_root=output_root,
        resume=True,
    )
    assert resumed == output
    assert calls == []


def test_factorial_bounded_parallel_parent_captures_all_worker_records(
    tmp_path: Path, monkeypatch
):
    calls: list[tuple[str, int]] = []
    _install_fast_training_fakes(monkeypatch, calls=calls)

    class ImmediateExecutor:
        def __init__(self, *, max_workers, mp_context):
            assert max_workers == 2
            assert mp_context.get_start_method() == "spawn"

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def submit(self, function, *args):
            future: concurrent.futures.Future = concurrent.futures.Future()
            try:
                future.set_result(function(*args))
            except Exception as exc:  # pragma: no cover - asserted by result()
                future.set_exception(exc)
            return future

    monkeypatch.setattr(
        factorial_stage3.concurrent.futures,
        "ProcessPoolExecutor",
        ImmediateExecutor,
    )
    template = tmp_path / "parallel_template.yaml"
    template.write_text(
        "run:\n  run_id: template\n"
        "training:\n  max_parallel_optimizer_replicates: 2\n",
        encoding="utf-8",
    )
    output = factorial_stage3.run(
        config_path=template,
        matrix_path=MATRIX,
        workflow_path=WORKFLOW,
        optimizer_seeds=SEEDS,
        output_root=tmp_path / "parallel_factorial",
    )
    assert len(calls) == 20
    manifest = validate_factorial_manifest(output)
    assert manifest["status"] == "complete"
    plan = read_json_mapping(manifest["factorial_plan"])
    assert plan["execution_parallelism"]["max_parallel_optimizer_replicates"] == 2


def test_factorial_resume_removes_only_an_empty_run_directory_shell(
    tmp_path: Path, monkeypatch
):
    calls: list[tuple[str, int]] = []
    _install_fast_training_fakes(monkeypatch, calls=calls)
    template = _template(tmp_path / "template.yaml")
    output_root = tmp_path / "factorial"
    factorial_stage3.run(
        config_path=template,
        matrix_path=MATRIX,
        workflow_path=WORKFLOW,
        optimizer_seeds=SEEDS,
        output_root=output_root,
        prepare_only=True,
    )
    empty_run_dir = _first_generated_run_dir(output_root)
    (empty_run_dir / "stage3").mkdir(parents=True)

    factorial_stage3.run(
        config_path=template,
        matrix_path=MATRIX,
        workflow_path=WORKFLOW,
        optimizer_seeds=SEEDS,
        output_root=output_root,
        resume=True,
    )

    assert len(calls) == 20
    assert (empty_run_dir / "stage3" / "ppo_model.zip").is_file()


def test_factorial_resume_refuses_a_nonempty_partial_run(
    tmp_path: Path, monkeypatch
):
    calls: list[tuple[str, int]] = []
    _install_fast_training_fakes(monkeypatch, calls=calls)
    template = _template(tmp_path / "template.yaml")
    output_root = tmp_path / "factorial"
    factorial_stage3.run(
        config_path=template,
        matrix_path=MATRIX,
        workflow_path=WORKFLOW,
        optimizer_seeds=SEEDS,
        output_root=output_root,
        prepare_only=True,
    )
    partial_run_dir = _first_generated_run_dir(output_root)
    stage_dir = partial_run_dir / "stage3"
    stage_dir.mkdir(parents=True)
    (stage_dir / "partial_state.bin").write_bytes(b"do not overwrite")

    with pytest.raises(FileExistsError, match="contains files or links"):
        factorial_stage3.run(
            config_path=template,
            matrix_path=MATRIX,
            workflow_path=WORKFLOW,
            optimizer_seeds=SEEDS,
            output_root=output_root,
            resume=True,
        )
    assert calls == []
    assert (stage_dir / "partial_state.bin").read_bytes() == b"do not overwrite"


def test_prepare_only_is_idempotent_with_resume_and_no_resume_refuses_reuse(
    tmp_path: Path, monkeypatch
):
    calls: list[tuple[str, int]] = []
    _install_fast_training_fakes(monkeypatch, calls=calls)
    template = _template(tmp_path / "template.yaml")
    output_root = tmp_path / "factorial"
    output = factorial_stage3.run(
        config_path=template,
        matrix_path=MATRIX,
        workflow_path=WORKFLOW,
        optimizer_seeds=SEEDS,
        output_root=output_root,
        prepare_only=True,
    )
    manifest = validate_factorial_manifest(output, require_complete=False)
    assert manifest["status"] == "prepared"
    assert calls == []
    assert len(list(output_root.glob("methods/*/generated_configs/*.yaml"))) == 20

    resumed = factorial_stage3.run(
        config_path=template,
        matrix_path=MATRIX,
        workflow_path=WORKFLOW,
        optimizer_seeds=SEEDS,
        output_root=output_root,
        prepare_only=True,
        resume=True,
    )
    assert resumed == output
    with pytest.raises(FileExistsError, match="--no-resume"):
        factorial_stage3.run(
            config_path=template,
            matrix_path=MATRIX,
            workflow_path=WORKFLOW,
            optimizer_seeds=SEEDS,
            output_root=output_root,
            prepare_only=True,
            resume=False,
        )


def test_factorial_manifest_rejects_duplicate_checkpoint_content(
    tmp_path: Path, monkeypatch
):
    calls: list[tuple[str, int]] = []
    _install_fast_training_fakes(monkeypatch, calls=calls)
    template = _template(tmp_path / "template.yaml")
    output_root = tmp_path / "factorial"
    output = factorial_stage3.run(
        config_path=template,
        matrix_path=MATRIX,
        workflow_path=WORKFLOW,
        optimizer_seeds=SEEDS,
        output_root=output_root,
    )
    total = read_json_mapping(output)
    first_method, second_method = list(EXPECTED_CANDIDATE_METHOD_ROLES)[:2]
    first_child_path = Path(total["methods"][first_method]["replicate_manifest"])
    second_child_path = Path(total["methods"][second_method]["replicate_manifest"])
    first_child = read_json_mapping(first_child_path)
    second_child = read_json_mapping(second_child_path)
    source_checkpoint = Path(first_child["records"][0]["checkpoint"])
    duplicate_checkpoint = Path(second_child["records"][0]["checkpoint"])
    duplicate_checkpoint.write_bytes(source_checkpoint.read_bytes())
    second_child["records"][0]["checkpoint_sha256"] = file_sha256(duplicate_checkpoint)
    second_child.pop("manifest_fingerprint", None)
    second_child["manifest_fingerprint"] = fingerprint_payload(
        second_child, "manifest_fingerprint"
    )
    second_child_path.write_text(
        json.dumps(second_child, indent=2, sort_keys=True), encoding="utf-8"
    )
    child_paths = {
        method_id: Path(entry["replicate_manifest"])
        for method_id, entry in total["methods"].items()
    }
    with pytest.raises(ValueError, match="globally unique"):
        build_factorial_manifest(
            protocol_id=total["protocol_id"],
            final_method_id=total["final_method_id"],
            method_roles=total["method_roles"],
            optimizer_seeds=total["optimizer_seeds"],
            plan_path=total["factorial_plan"],
            child_manifests=child_paths,
            status="complete",
        )


def test_resume_rejects_changed_frozen_template(tmp_path: Path, monkeypatch):
    calls: list[tuple[str, int]] = []
    _install_fast_training_fakes(monkeypatch, calls=calls)
    template = _template(tmp_path / "template.yaml")
    output_root = tmp_path / "factorial"
    factorial_stage3.run(
        config_path=template,
        matrix_path=MATRIX,
        workflow_path=WORKFLOW,
        optimizer_seeds=SEEDS,
        output_root=output_root,
        prepare_only=True,
    )
    template.write_text("run:\n  run_id: changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="template_config_sha256"):
        factorial_stage3.run(
            config_path=template,
            matrix_path=MATRIX,
            workflow_path=WORKFLOW,
            optimizer_seeds=SEEDS,
            output_root=output_root,
            prepare_only=True,
            resume=True,
        )


def test_factorial_validator_resolves_relative_child_manifest_paths(
    tmp_path: Path, monkeypatch
):
    calls: list[tuple[str, int]] = []
    _install_fast_training_fakes(monkeypatch, calls=calls)
    template = _template(tmp_path / "template.yaml")
    output = factorial_stage3.run(
        config_path=template,
        matrix_path=MATRIX,
        workflow_path=WORKFLOW,
        optimizer_seeds=SEEDS,
        output_root=tmp_path / "factorial",
    )
    payload = read_json_mapping(output)
    for entry in payload["methods"].values():
        entry["replicate_manifest"] = str(
            Path(entry["replicate_manifest"]).relative_to(output.parent)
        )
    payload.pop("manifest_fingerprint", None)
    payload["manifest_fingerprint"] = fingerprint_payload(
        payload, "manifest_fingerprint"
    )
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    assert validate_factorial_manifest(output)["status"] == "complete"
