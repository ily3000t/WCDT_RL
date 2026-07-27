from __future__ import annotations

import json
from pathlib import Path

import pytest

from safe_rl.accvp.contracts.schema import file_sha256
from safe_rl.evaluation_protocol import stable_hash
from safe_rl.pipeline import accvp_runtime_benchmark_factorial as factorial_runtime
from safe_rl.pipeline import accvp_runtime_benchmark_replicates as replicate_runtime
from safe_rl.ppo_factorial import (
    EXPECTED_CANDIDATE_METHOD_ROLES,
    EXPECTED_FINAL_METHOD_ID,
)


SIMULATOR_SEEDS = list(range(50001, 50031))
OPTIMIZER_SEEDS = [1001, 1002, 1003, 1004, 1005]


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _factorial_fixture(tmp_path: Path) -> tuple[Path, dict]:
    factorial_source = _write_json(tmp_path / "ppo_factorial_manifest.json", {"placeholder": True})
    methods: dict[str, dict] = {}
    for method_index, (method_id, role) in enumerate(EXPECTED_CANDIDATE_METHOD_ROLES.items()):
        config = tmp_path / method_id / "resolved_1001.yaml"
        config.parent.mkdir(parents=True, exist_ok=True)
        config.write_text(f"method_id: {method_id}\n", encoding="utf-8")
        records = []
        for seed_index, seed in enumerate(OPTIMIZER_SEEDS):
            records.append(
                {
                    "method_id": method_id,
                    "optimizer_seed": seed,
                    "training_seed": seed,
                    "resolved_config": str(config),
                    "resolved_config_sha256": file_sha256(config),
                    "checkpoint": str(tmp_path / method_id / f"ppo_{seed}.zip"),
                    "checkpoint_sha256": f"{method_index * 10 + seed_index + 1:064x}",
                }
            )
        child = _write_json(
            tmp_path / method_id / "ppo_replicate_manifest.json",
            {
                "artifact_kind": "ppo_optimizer_replicate_manifest_v1",
                "schema_version": 1,
                "status": "complete",
                "method_id": method_id,
                "optimizer_seeds": OPTIMIZER_SEEDS,
                "records": records,
            },
        )
        methods[method_id] = {
            "role": role,
            "replicate_manifest": str(child),
            "replicate_manifest_sha256": file_sha256(child),
            "checkpoint_sha256s": [row["checkpoint_sha256"] for row in records],
        }
    factorial = {
        "artifact_kind": "ppo_factorial_manifest_v1",
        "schema_version": 1,
        "status": "complete",
        "manifest_fingerprint": "factorial-fingerprint",
        "final_method_id": EXPECTED_FINAL_METHOD_ID,
        "method_roles": dict(EXPECTED_CANDIDATE_METHOD_ROLES),
        "methods": methods,
    }
    return factorial_source, factorial


def _install_factorial_fakes(monkeypatch: pytest.MonkeyPatch, factorial: dict, calls: list[dict]) -> None:
    monkeypatch.setattr(
        factorial_runtime,
        "validate_factorial_manifest",
        lambda *_args, **_kwargs: factorial,
    )
    monkeypatch.setattr(
        replicate_runtime,
        "runtime_replicate_request_fingerprint",
        lambda **_kwargs: "child-request-fingerprint",
    )
    monkeypatch.setattr(
        replicate_runtime,
        "validate_runtime_replicate_report",
        lambda report, **_kwargs: report,
    )

    def fake_run(**kwargs):
        calls.append(kwargs)
        output = Path(kwargs["output"])
        if output.exists():
            if not kwargs.get("resume", False):
                raise FileExistsError(output)
            return output
        manifest = json.loads(Path(kwargs["replicate_manifest"]).read_text(encoding="utf-8"))
        payload = {
            "artifact_kind": replicate_runtime.RUNTIME_REPLICATE_REPORT_KIND,
            "schema_version": 1,
            "status": "complete",
            "method_id": manifest["method_id"],
            "hard_real_time_claim": False,
            "gate": {"checks": {"fake": True}, "pass": True},
            "report_fingerprint": f"runtime-{manifest['method_id']}",
        }
        _write_json(output, payload)
        return output

    monkeypatch.setattr(replicate_runtime, "run", fake_run)


def test_factorial_runtime_runs_all_methods_and_binds_final_method(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    factorial_source, factorial = _factorial_fixture(tmp_path)
    calls: list[dict] = []
    _install_factorial_fakes(monkeypatch, factorial, calls)
    output = tmp_path / "runtime" / "factorial_runtime_report.json"

    result = factorial_runtime.run(
        factorial_manifest=factorial_source,
        seeds=SIMULATOR_SEEDS,
        backend="vectorized",
        output=output,
    )

    assert result == output.resolve()
    assert len(calls) == 4
    assert all(call["resume"] is True for call in calls)
    assert {
        Path(call["config_template"]).read_text(encoding="utf-8").strip().split(": ")[1]
        for call in calls
    } == set(EXPECTED_CANDIDATE_METHOD_ROLES)
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["artifact_kind"] == factorial_runtime.FACTORIAL_RUNTIME_REPORT_KIND
    assert report["status"] == "complete"
    assert report["final_method_id"] == EXPECTED_FINAL_METHOD_ID
    assert set(report["methods"]) == set(EXPECTED_CANDIDATE_METHOD_ROLES)
    assert report["gate"]["pass"] is True
    for method_id in EXPECTED_CANDIDATE_METHOD_ROLES:
        expected = (
            output.parent / "methods" / method_id / "replicated_runtime_report.json"
        ).resolve()
        assert Path(report["methods"][method_id]["runtime_report"]) == expected


def test_factorial_runtime_resume_reuses_complete_total_and_rejects_changed_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    factorial_source, factorial = _factorial_fixture(tmp_path)
    calls: list[dict] = []
    _install_factorial_fakes(monkeypatch, factorial, calls)
    output = tmp_path / "runtime" / "factorial_runtime_report.json"
    factorial_runtime.run(
        factorial_manifest=factorial_source,
        seeds=SIMULATOR_SEEDS,
        backend="vectorized",
        output=output,
    )
    calls.clear()

    assert factorial_runtime.run(
        factorial_manifest=factorial_source,
        seeds=SIMULATOR_SEEDS,
        backend="vectorized",
        output=output,
    ) == output.resolve()
    assert calls == []

    with pytest.raises(ValueError, match="request fingerprint"):
        factorial_runtime.run(
            factorial_manifest=factorial_source,
            seeds=list(reversed(SIMULATOR_SEEDS)),
            backend="vectorized",
            output=output,
        )


def test_factorial_runtime_archives_failed_legacy_implementation_before_rerun(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    factorial_source, factorial = _factorial_fixture(tmp_path)
    calls: list[dict] = []
    _install_factorial_fakes(monkeypatch, factorial, calls)
    output = tmp_path / "runtime" / "factorial_runtime_report.json"
    factorial_runtime.run(
        factorial_manifest=factorial_source,
        seeds=SIMULATOR_SEEDS,
        backend="vectorized",
        output=output,
    )

    legacy = json.loads(output.read_text(encoding="utf-8"))
    legacy.pop("runtime_implementation_version")
    legacy["gate"]["pass"] = False
    legacy["report_fingerprint"] = factorial_runtime._report_fingerprint(legacy)
    _write_json(output, legacy)
    calls.clear()

    factorial_runtime.run(
        factorial_manifest=factorial_source,
        seeds=SIMULATOR_SEEDS,
        backend="vectorized",
        output=output,
    )

    assert len(calls) == 4
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["runtime_implementation_version"] == (
        factorial_runtime.RUNTIME_IMPLEMENTATION_VERSION
    )
    prior = report["prior_failed_attempt"]
    archive_dir = Path(prior["archive_dir"])
    assert prior["runtime_implementation_version"] == "legacy"
    assert (archive_dir / output.name).is_file()
    assert (archive_dir / "methods").is_dir()


def test_replicate_runtime_resume_validator_rejects_checkpoint_or_seed_drift(tmp_path: Path):
    template = tmp_path / "template.yaml"
    template.write_text("run:\n  seed: 1\n", encoding="utf-8")
    config = tmp_path / "resolved.yaml"
    config.write_text("run:\n  seed: 1001\n", encoding="utf-8")
    rows = []
    lineage = []
    expected_seed_hash = stable_hash({"episode_seeds": SIMULATOR_SEEDS})
    for index, seed in enumerate(OPTIMIZER_SEEDS):
        checkpoint_hash = f"{index + 1:064x}"
        row = {
            "optimizer_seed": seed,
            "resolved_config": str(config),
            "resolved_config_sha256": file_sha256(config),
            "checkpoint_sha256": checkpoint_hash,
        }
        rows.append(row)
        child_payload = {
            "artifact_kind": "accvp_runtime_benchmark_v1",
            "schema_version": 2,
            "runtime_implementation_version": (
                replicate_runtime.accvp_runtime_benchmark.RUNTIME_IMPLEMENTATION_VERSION
            ),
            "policy_type": "sb3_ppo",
            "backend": "vectorized",
            "policy_model_sha256": checkpoint_hash,
            "config_file_sha256": file_sha256(config),
            "workload": {
                "requested_episode_seed_count": len(SIMULATOR_SEEDS),
                "requested_episode_seed_sha256": expected_seed_hash,
                "observed_episode_seed_sha256": expected_seed_hash,
            },
            "gate": {"pass": True},
        }
        child_payload["report_fingerprint"] = stable_hash(child_payload)
        child_path = _write_json(tmp_path / "replicates" / f"runtime_seed_{seed}.json", child_payload)
        lineage.append(
            {
                "optimizer_seed": seed,
                "checkpoint_sha256": checkpoint_hash,
                "report": str(child_path),
                "report_sha256": file_sha256(child_path),
            }
        )
    manifest_path = _write_json(
        tmp_path / "ppo_replicate_manifest.json",
        {"method_id": "candidate_table_reward_v2", "records": rows},
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    request_fingerprint = replicate_runtime.runtime_replicate_request_fingerprint(
        config_template=template,
        replicate_manifest=manifest_path,
        manifest=manifest,
        seeds=SIMULATOR_SEEDS,
        backend="vectorized",
        device="auto",
    )
    aggregate = {
        "artifact_kind": replicate_runtime.RUNTIME_REPLICATE_REPORT_KIND,
        "schema_version": 1,
        "runtime_implementation_version": (
            replicate_runtime.accvp_runtime_benchmark.RUNTIME_IMPLEMENTATION_VERSION
        ),
        "status": "complete",
        "request_fingerprint": request_fingerprint,
        "replicate_manifest_sha256": file_sha256(manifest_path),
        "simulator_seeds": SIMULATOR_SEEDS,
        "checkpoint_sha256s": [row["checkpoint_sha256"] for row in rows],
        "replicates": lineage,
        "gate": {"pass": True},
    }
    aggregate["report_fingerprint"] = stable_hash(aggregate)
    replicate_runtime.validate_runtime_replicate_report(
        aggregate,
        expected_request_fingerprint=request_fingerprint,
        replicate_manifest=manifest_path,
        manifest=manifest,
        requested_seeds=SIMULATOR_SEEDS,
        backend="vectorized",
    )

    aggregate["checkpoint_sha256s"][0] = "f" * 64
    aggregate["report_fingerprint"] = replicate_runtime._report_fingerprint(aggregate)
    with pytest.raises(ValueError, match="checkpoint set"):
        replicate_runtime.validate_runtime_replicate_report(
            aggregate,
            expected_request_fingerprint=request_fingerprint,
            replicate_manifest=manifest_path,
            manifest=manifest,
            requested_seeds=SIMULATOR_SEEDS,
            backend="vectorized",
        )


def test_factorial_runtime_requires_thirty_unique_simulator_seeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    factorial_source, factorial = _factorial_fixture(tmp_path)
    _install_factorial_fakes(monkeypatch, factorial, [])
    with pytest.raises(ValueError, match="at least 30 distinct"):
        factorial_runtime.run(
            factorial_manifest=factorial_source,
            seeds=[1, 2, 3],
            backend="vectorized",
            output=tmp_path / "runtime.json",
        )
