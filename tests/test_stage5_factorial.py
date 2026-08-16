from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from safe_rl.accvp.contracts.schema import file_sha256, stable_hash
from safe_rl.evaluation_protocol import EvidenceProtocolError
from safe_rl.pipeline import (
    stage5_factorial_aggregate,
    stage5_episode_cache,
    stage5_generate_factorial_configs,
    stage5_paired_eval,
    stage5_run_factorial,
)
from safe_rl.pipeline.stage5_generate_factorial_configs import (
    DEFAULT_COMPARISONS,
    FACTORIAL_RUNTIME_KIND,
    FINAL_COMPARISON_ID,
)
from safe_rl.ppo_factorial import EXPECTED_FINAL_METHOD_ID
from safe_rl.ppo_replicates import REPLICATE_MANIFEST_KIND


CANDIDATE_METHODS = (
    "candidate_table_reward_v2",
    "candidate_table_reward_v2_commitment",
    "candidate_table_reward_v3_1",
    "candidate_table_reward_v3_1_commitment",
)


class _AttrDict(dict):
    __getattr__ = dict.__getitem__


def _json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def test_episode_cache_reuses_common_prefix_and_only_executes_extension(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = tmp_path / "policy.zip"
    risk = tmp_path / "risk.pt"
    model.write_bytes(b"policy")
    risk.write_bytes(b"risk")
    identity = {
        "method_id": "candidate_table_reward_v3_1_commitment",
        "optimizer_seed": 1001,
        "checkpoint_sha256": file_sha256(model),
        "resolved_config": str(tmp_path / "resolved.yaml"),
        "resolved_config_sha256": "a" * 64,
        "reward_semantics_hash": "b" * 64,
        "observation_contract_hash": "c" * 64,
        "risk_checkpoint_sha256": file_sha256(risk),
        "group_execution_contract_sha256": "d" * 64,
        "shield_enabled": True,
        "policy_type": "sb3_ppo",
    }
    binding = {
        "artifact_kind": "stage5_episode_cache_binding_v1",
        "schema_version": 1,
        "cache_dir": str(tmp_path / "cache"),
        "execution_fingerprint": stable_hash(identity),
        "identity": identity,
    }
    calls: list[list[int]] = []

    def fake_evaluator(
        _cfg,
        _model_path,
        *,
        seeds,
        **_kwargs,
    ):
        calls.append(list(seeds))
        return {
            "episodes": [
                {
                    "seed": int(seed),
                    "episode_reward": float(seed),
                    "merge_success": True,
                }
                for seed in seeds
            ],
            "model_observation_shape": [159],
            "env_observation_shape": [159],
            "policy_type": "sb3_ppo",
        }

    monkeypatch.setattr(
        stage5_episode_cache,
        "aggregate_episode_reports",
        lambda reports, task_quality=None: {"episodes": len(reports)},
    )
    cfg = SimpleNamespace(stage5=_AttrDict(task_quality={}))

    first = stage5_episode_cache.evaluate_policy_cached(
        evaluator=fake_evaluator,
        cfg=cfg,
        model_path=model,
        seeds=[80001, 80002],
        shield_enabled=True,
        risk_checkpoint=risk,
        group_name="candidate_seed_1001",
        policy_type="sb3_ppo",
        binding=binding,
    )
    repeated = stage5_episode_cache.evaluate_policy_cached(
        evaluator=fake_evaluator,
        cfg=cfg,
        model_path=model,
        seeds=[80001, 80002],
        shield_enabled=True,
        risk_checkpoint=risk,
        group_name="candidate_seed_1001",
        policy_type="sb3_ppo",
        binding=binding,
    )
    extended = stage5_episode_cache.evaluate_policy_cached(
        evaluator=fake_evaluator,
        cfg=cfg,
        model_path=model,
        seeds=[80001, 80002, 80003],
        shield_enabled=True,
        risk_checkpoint=risk,
        group_name="candidate_seed_1001",
        policy_type="sb3_ppo",
        binding=binding,
    )

    assert calls == [[80001, 80002], [80003]]
    assert first["episode_cache"]["executed_episode_count"] == 2
    assert repeated["episode_cache"]["cache_hit_count"] == 2
    assert repeated["episode_cache"]["executed_episode_count"] == 0
    assert extended["episode_cache"]["cache_hit_count"] == 2
    assert extended["episode_cache"]["executed_episode_count"] == 1
    assert [row["seed"] for row in extended["episodes"]] == [80001, 80002, 80003]


def _runtime_fixture(tmp_path: Path) -> tuple[Path, Path, dict]:
    factorial_path = _json(tmp_path / "factorial.json", {"frozen": True})
    methods = {}
    aggregate_methods = {}
    for index, method_id in enumerate(CANDIDATE_METHODS, start=1):
        manifest_sha = f"{index:064x}"
        methods[method_id] = {"replicate_manifest_sha256": manifest_sha}
        rows = []
        for seed in range(1001, 1006):
            report = _json(
                tmp_path / method_id / f"runtime_{seed}.json",
                {"gate": {"pass": True}, "optimizer_seed": seed},
            )
            rows.append(
                {
                    "optimizer_seed": seed,
                    "report": str(report),
                    "report_sha256": file_sha256(report),
                }
            )
        child = _json(
            tmp_path / method_id / "runtime_report.json",
            {
                "artifact_kind": "accvp_runtime_benchmark_replicates_v1",
                "replicate_manifest_sha256": manifest_sha,
                "replicates": rows,
                "gate": {"pass": True},
            },
        )
        aggregate_methods[method_id] = {
            "replicate_manifest_sha256": manifest_sha,
            "runtime_report": str(child),
            "runtime_report_sha256": file_sha256(child),
            "gate": {"pass": True},
        }
    aggregate = _json(
        tmp_path / "factorial_runtime.json",
        {
            "artifact_kind": FACTORIAL_RUNTIME_KIND,
            "schema_version": 1,
            "status": "complete",
            "factorial_manifest": str(factorial_path),
            "factorial_manifest_sha256": file_sha256(factorial_path),
            "final_method_id": EXPECTED_FINAL_METHOD_ID,
            "methods": aggregate_methods,
            "gate": {"pass": True},
        },
    )
    factorial = {
        "final_method_id": EXPECTED_FINAL_METHOD_ID,
        "methods": methods,
    }
    return aggregate, factorial_path, factorial


def test_runtime_coverage_requires_checkpoint_bound_report_for_each_method(tmp_path: Path) -> None:
    aggregate, factorial_path, factorial = _runtime_fixture(tmp_path)
    source, coverage = stage5_generate_factorial_configs._runtime_coverage(
        aggregate,
        factorial_path=factorial_path,
        factorial=factorial,
    )
    assert source == aggregate.resolve()
    assert set(coverage) == set(CANDIDATE_METHODS)
    assert set(coverage[EXPECTED_FINAL_METHOD_ID]["reports"]) == set(range(1001, 1006))

    payload = json.loads(aggregate.read_text(encoding="utf-8"))
    payload["methods"].pop("candidate_table_reward_v2")
    aggregate.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="missing method"):
        stage5_generate_factorial_configs._runtime_coverage(
            aggregate,
            factorial_path=factorial_path,
            factorial=factorial,
        )


def test_workflow_comparison_preregistration_cannot_drift(tmp_path: Path) -> None:
    workflow = {
        "artifact_kind": "accvp_vnext_workflow_contract_v1",
        "factorial": {"comparisons": [dict(item) for item in DEFAULT_COMPARISONS]},
    }
    source = tmp_path / "workflow.yaml"
    source.write_text(yaml.safe_dump(workflow, sort_keys=False), encoding="utf-8")
    rows = stage5_generate_factorial_configs._comparison_specs(source)
    assert len(rows) == 6
    assert rows[-1]["comparison_id"] == FINAL_COMPARISON_ID

    workflow["factorial"]["comparisons"][0]["right_method_id"] = EXPECTED_FINAL_METHOD_ID
    source.write_text(yaml.safe_dump(workflow, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="frozen six-comparison design"):
        stage5_generate_factorial_configs._comparison_specs(source)


def test_generate_writes_group_specific_runtime_bindings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seeds = list(range(1001, 1006))
    risk_checkpoint = tmp_path / "risk_module.pt"
    risk_checkpoint.write_bytes(b"frozen-risk")
    artifact_binding = {
        "accvp_artifact_manifest": str(tmp_path / "accvp_bundle.json"),
        "accvp_artifact_manifest_sha256": "a" * 64,
        "accvp_artifact_fingerprint": "b" * 64,
        "accvp_artifact_variant": "full_candidate_gate_v1",
        "formal_runtime_contract_sha256": "c" * 64,
        "deployment_runtime_contract_sha256": "c" * 64,
        "candidate_table_semantic_contract_sha256": "8" * 64,
        "closed_loop_execution_contract_sha256": "9" * 64,
    }
    manifests: dict[str, Path] = {}
    runtime_coverage: dict[str, dict] = {}
    for method_index, method_id in enumerate(("wcdt_reward_v2", *CANDIDATE_METHODS), start=1):
        config_path = tmp_path / "configs" / f"{method_id}.yaml"
        config = {
            "run": {"output_root": str(tmp_path / "runs"), "run_id": method_id},
            "forecast_features": {
                "enabled": method_id == "wcdt_reward_v2",
                "source": "wcdt_v3" if method_id == "wcdt_reward_v2" else "",
            },
            "accvp": {"candidate_table": {"enabled": method_id != "wcdt_reward_v2"}},
            "rl": {
                "reward_profile": "merge_timing",
                "training_semantics_version": method_id,
                "merge_timing_reward": {"reward_version": method_id},
                "policy_lateral_commitment": {"enabled": "commitment" in method_id},
                "shield_guided_reward": {
                    "risk_checkpoint": str(risk_checkpoint),
                },
            },
        }
        if method_id != "wcdt_reward_v2":
            config["accvp"]["risk_checkpoint"] = str(risk_checkpoint)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
        records = []
        for seed in seeds:
            record = {
                "method_id": method_id,
                "optimizer_seed": seed,
                "training_seed": seed,
                "checkpoint": str(tmp_path / method_id / f"checkpoint_{seed}.zip"),
                "checkpoint_sha256": f"{method_index * 10000 + seed:064x}",
                "resolved_config": str(config_path),
                "resolved_config_sha256": file_sha256(config_path),
                "reward_semantics_hash": f"{method_index:064x}",
                "observation_contract_hash": "d" * 64,
                "observation_contract": artifact_binding if method_id != "wcdt_reward_v2" else {},
            }
            records.append(record)
        manifest = _json(
            tmp_path / "manifests" / f"{method_id}.json",
            {
                "artifact_kind": REPLICATE_MANIFEST_KIND,
                "schema_version": 1,
                "status": "complete",
                "method_id": method_id,
                "records": records,
            },
        )
        manifests[method_id] = manifest
        if method_id != "wcdt_reward_v2":
            method_reports = {}
            for seed in seeds:
                report = _json(
                    tmp_path / "runtime" / method_id / f"runtime_{seed}.json",
                    {"gate": {"pass": True}, "optimizer_seed": seed},
                )
                method_reports[seed] = {"path": str(report), "sha256": file_sha256(report)}
            runtime_coverage[method_id] = {
                "runtime_report": str(tmp_path / "runtime" / method_id / "aggregate.json"),
                "runtime_report_sha256": "e" * 64,
                "replicate_manifest_sha256": file_sha256(manifest),
                "reports": method_reports,
            }
    factorial_path = _json(tmp_path / "factorial.json", {"placeholder": True})
    factorial = {
        "artifact_kind": "ppo_factorial_manifest_v1",
        "schema_version": 1,
        "status": "complete",
        "protocol_id": "accvp-vnext-correctness-v1",
        "final_method_id": EXPECTED_FINAL_METHOD_ID,
        "optimizer_seeds": seeds,
        "manifest_fingerprint": "f" * 64,
        "methods": {
            method_id: {
                "replicate_manifest": str(manifests[method_id]),
                "replicate_manifest_sha256": file_sha256(manifests[method_id]),
            }
            for method_id in CANDIDATE_METHODS
        },
    }
    runtime_path = _json(tmp_path / "runtime_factorial.json", {"gate": {"pass": True}})
    monkeypatch.setattr(
        stage5_generate_factorial_configs,
        "validate_factorial_manifest",
        lambda *_args, **_kwargs: factorial,
    )
    monkeypatch.setattr(
        stage5_generate_factorial_configs,
        "audit_manifest",
        lambda *_args, **_kwargs: {"status": "reusable"},
    )
    monkeypatch.setattr(
        stage5_generate_factorial_configs,
        "_runtime_coverage",
        lambda *_args, **_kwargs: (runtime_path.resolve(), runtime_coverage),
    )
    protocol_cfg = SimpleNamespace(
        evaluation_protocol={"strict": True, "stage5_role": "stage5_confirmatory"}
    )
    monkeypatch.setattr(stage5_generate_factorial_configs, "load_config", lambda _path: protocol_cfg)
    monkeypatch.setattr(
        stage5_generate_factorial_configs,
        "protocol_snapshot",
        lambda _cfg: {
            "cohort_roles": {"stage5_confirmatory": "natural_confirmatory"},
            "cohorts": {"natural_confirmatory": list(range(80001, 80301))},
        },
    )
    protocol_path = _json(tmp_path / "protocol.json", {})
    workflow = tmp_path / "workflow.yaml"
    workflow.write_text(
        yaml.safe_dump(
            {
                "artifact_kind": "accvp_vnext_workflow_contract_v1",
                "factorial": {
                    "stage5_evaluation": {
                        "secondary_simulator_seed_count": 100,
                        "primary_simulator_seed_count": 300,
                        "nested_seed_prefix": True,
                        "episode_cache_enabled": True,
                    },
                    "comparisons": [dict(item) for item in DEFAULT_COMPARISONS],
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    output = tmp_path / "generated"
    request_path = stage5_generate_factorial_configs.generate(
        baseline_manifest=manifests["wcdt_reward_v2"],
        factorial_manifest=factorial_path,
        protocol=protocol_path,
        seed_role="stage5_confirmatory",
        runtime_factorial_report=runtime_path,
        output_dir=output,
        workflow_config=workflow,
    )
    request = json.loads(request_path.read_text(encoding="utf-8"))
    assert request["final_method_id"] == EXPECTED_FINAL_METHOD_ID
    assert len(request["comparisons"]) == 6
    assert request["evaluation_budget"]["naive_episode_count"] == 8000
    assert request["evaluation_budget"]["unique_episode_count"] == 4500
    assert request["risk_checkpoint_sha256"] == file_sha256(risk_checkpoint)

    single_dir = output / "comparisons" / DEFAULT_COMPARISONS[0]["comparison_id"]
    single_cfg = yaml.safe_load((single_dir / "stage5_seed_1001.yaml").read_text(encoding="utf-8"))
    assert single_cfg["stage5"]["execution_contract"] == "simulation_blocking_exact_v1"
    assert single_cfg["stage5"]["require_accvp_observation_runtime_gate"] is False
    assert len(single_cfg["stage5"]["deployment_runtime_reports"]) == 1
    assert "accvp_observation_preflight_report" not in single_cfg["stage5"]
    assert single_cfg["stage5"]["episodes_per_group"] == 100
    assert len(single_cfg["stage5"]["seeds"]) == 100
    assert single_cfg["stage5"]["risk_checkpoint"] == str(risk_checkpoint.resolve())
    assert single_cfg["stage5"]["groups"][1]["evaluation_cache"]["artifact_kind"] == (
        "stage5_episode_cache_binding_v1"
    )

    crossed_id = DEFAULT_COMPARISONS[1]["comparison_id"]
    crossed_cfg = yaml.safe_load(
        (output / "comparisons" / crossed_id / "stage5_seed_1001.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert len(crossed_cfg["stage5"]["deployment_runtime_reports"]) == 2
    assert "accvp_observation_preflight_report" not in crossed_cfg["stage5"]
    final_cfg = yaml.safe_load(
        (
            output
            / "comparisons"
            / FINAL_COMPARISON_ID
            / "stage5_seed_1001.yaml"
        ).read_text(encoding="utf-8")
    )
    assert final_cfg["stage5"]["episodes_per_group"] == 300
    first_wcdt = single_cfg["stage5"]["groups"][0]["evaluation_cache"]
    final_wcdt = final_cfg["stage5"]["groups"][0]["evaluation_cache"]
    assert first_wcdt["cache_dir"] == final_wcdt["cache_dir"]
    assert first_wcdt["execution_fingerprint"] == final_wcdt["execution_fingerprint"]


def test_factorial_aggregate_exposes_final_child_and_does_not_invent_pvalues(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _json(tmp_path / "factorial_request.json", {"request": True})
    comparisons = []
    child_reports: dict[str, tuple[Path, dict]] = {}
    for item in DEFAULT_COMPARISONS:
        child = _json(
            tmp_path / "comparisons" / item["comparison_id"] / "replicated_report.json",
            {
                "artifact_kind": "stage5_replicated_paired_report_v1",
                "gate": {"pass": True},
                "statistics": {"statistics_fingerprint": item["comparison_id"]},
            },
        )
        comparisons.append({**item, "aggregate_report": str(child)})
        child_reports[item["comparison_id"]] = (
            child,
            json.loads(child.read_text(encoding="utf-8")),
        )
    request = {
        "protocol_id": "accvp-vnext-correctness-v1",
        "factorial_manifest": str(tmp_path / "factorial.json"),
        "factorial_manifest_sha256": "a" * 64,
        "factorial_manifest_fingerprint": "b" * 64,
        "runtime_factorial_report": str(tmp_path / "runtime.json"),
        "runtime_factorial_report_sha256": "c" * 64,
        "comparisons": comparisons,
    }
    monkeypatch.setattr(
        stage5_factorial_aggregate,
        "load_factorial_request",
        lambda _path: (source.resolve(), request),
    )
    monkeypatch.setattr(
        stage5_factorial_aggregate,
        "validate_child_report",
        lambda comparison, request_path: child_reports[comparison["comparison_id"]],
    )
    report = stage5_factorial_aggregate.aggregate(source)
    assert report["gate"]["pass"] is True
    assert report["final_method_id"] == EXPECTED_FINAL_METHOD_ID
    assert report["final_comparison_report"]["path"] == str(
        child_reports[FINAL_COMPARISON_ID][0]
    )
    assert all(
        not row["performed"]
        for row in report["multiple_comparison_correction"].values()
    )


def test_factorial_runner_resumes_completed_comparisons(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _json(tmp_path / "factorial_request.json", {"request": True})
    comparisons = []
    for item in DEFAULT_COMPARISONS:
        directory = tmp_path / "comparisons" / item["comparison_id"]
        directory.mkdir(parents=True)
        comparisons.append(
            {
                **item,
                "generated_dir": str(directory),
                "aggregate_report": str(directory / "replicated_report.json"),
            }
        )
    Path(comparisons[0]["aggregate_report"]).write_text("{}", encoding="utf-8")
    request = {"comparisons": comparisons}
    monkeypatch.setattr(
        stage5_run_factorial,
        "load_factorial_request",
        lambda _path: (source.resolve(), request),
    )
    monkeypatch.setattr(
        stage5_run_factorial,
        "validate_child_report",
        lambda *_args, **_kwargs: (tmp_path / "child.json", {"gate": {"pass": True}}),
    )
    calls = []

    def fake_run(*, generated_dir, aggregate_output, resume=False):
        calls.append((Path(generated_dir), bool(resume)))
        Path(aggregate_output).write_text("{}", encoding="utf-8")
        return []

    monkeypatch.setattr(stage5_run_factorial.stage5_run_replicates, "run", fake_run)
    final = tmp_path / "factorial_report.json"
    def fake_aggregate(_request, output):
        Path(output).write_text("{}", encoding="utf-8")
        return Path(output)

    monkeypatch.setattr(stage5_run_factorial, "aggregate_factorial", fake_aggregate)
    result = stage5_run_factorial.run(request=source, output=final, resume=True)
    assert result == final
    assert len(calls) == 5
    assert all(resume for _directory, resume in calls)


def test_factorial_acceptance_profile_is_reward_version_neutral() -> None:
    metrics = {
        "proxy_collision_rate": 0.0,
        "safety_violation_rate": 0.0,
        "geometric_overlap_rate": 0.0,
        "fallback_rate": 0.0,
        "taper_miss_rate": 0.0,
        "timely_merge_success_rate": 1.0,
        "actual_replacement_rate": 0.0,
    }
    reports = {
        "left": {"metrics": dict(metrics)},
        "right": {"metrics": dict(metrics)},
    }
    stage5 = {
        "acceptance": {
            "paired_policy_non_regression_v1": {
                "max_actual_replacement_rate": 0.05,
            }
        },
        "pairs": [
            {
                "name": "reward_factor",
                "left": "left",
                "right": "right",
                "acceptance_profile": "paired_policy_non_regression_v1",
            }
        ],
    }
    result = stage5_paired_eval._configured_pair_acceptance(reports, stage5)
    assert result["reward_factor"]["available"] is True
    assert result["reward_factor"]["regression"] is False
    assert result["reward_factor"]["profile"] == "paired_policy_non_regression_v1"


def test_stage5_per_group_preflights_bind_each_checkpoint_and_report_hash(
    tmp_path: Path,
    monkeypatch,
) -> None:
    class AttrDict(dict):
        __getattr__ = dict.__getitem__

    groups = [
        AttrDict(name=name, forecast_features=False, shield=False)
        for name in ("left", "right")
    ]
    model_paths = {}
    report_paths = {}
    report_hashes = {}
    feature_hash = stable_hash({"feature_names": ["feature"]})
    for name in ("left", "right"):
        model = tmp_path / f"{name}.zip"
        model.write_bytes(name.encode("utf-8"))
        report = tmp_path / f"{name}_runtime.json"
        report.write_text(
            json.dumps(
                {
                    "gate": {"pass": True},
                    "policy_model_sha256": file_sha256(model),
                    "accvp_observation_feature_names_sha256": feature_hash,
                    "accvp_observation_feature_contract_hash": "f" * 64,
                }
            ),
            encoding="utf-8",
        )
        model_paths[name] = model
        report_paths[name] = str(report)
        report_hashes[name] = file_sha256(report)
    cfg = SimpleNamespace(
        stage5=AttrDict(
            require_accvp_observation_runtime_gate=True,
            accvp_observation_preflight_reports=report_paths,
            accvp_observation_preflight_report_sha256s=report_hashes,
            groups=groups,
        )
    )
    monkeypatch.setattr(
        stage5_paired_eval,
        "clone_with_overrides",
        lambda config, _overrides: config,
    )
    monkeypatch.setattr(
        stage5_paired_eval.RiskGatedACCVPCandidateTableAugmentor,
        "enabled",
        lambda _config: True,
    )
    monkeypatch.setattr(
        stage5_paired_eval.RiskGatedACCVPCandidateTableAugmentor,
        "feature_names",
        lambda _config: ["feature"],
    )
    result = stage5_paired_eval._validate_frozen_runtime_preflight(
        cfg=cfg,
        group_model_paths=model_paths,
        protocol_strict=True,
    )
    assert result["mode"] == "per_group_v1"
    assert set(result["reports"]) == {"left", "right"}

    cfg.stage5["accvp_observation_preflight_report_sha256s"]["right"] = "0" * 64
    with pytest.raises(EvidenceProtocolError, match="hash changed"):
        stage5_paired_eval._validate_frozen_runtime_preflight(
            cfg=cfg,
            group_model_paths=model_paths,
            protocol_strict=True,
        )
