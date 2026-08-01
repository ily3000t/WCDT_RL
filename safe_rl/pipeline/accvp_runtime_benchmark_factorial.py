from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

from safe_rl.accvp.contracts.schema import file_sha256, read_json
from safe_rl.evaluation_protocol import stable_hash
from safe_rl.pipeline import accvp_runtime_benchmark_replicates as replicate_runtime
from safe_rl.pipeline.accvp_runtime_benchmark import RUNTIME_IMPLEMENTATION_VERSION
from safe_rl.ppo_factorial import (
    EXPECTED_CANDIDATE_METHOD_ROLES,
    EXPECTED_FINAL_METHOD_ID,
    FINAL_METHOD_ROLE,
    read_json_mapping,
    resolve_manifest_path,
    resolve_path,
    validate_factorial_manifest,
)
from safe_rl.ppo_replicates import write_json_new
from safe_rl.utils.config import REPO_ROOT


FACTORIAL_RUNTIME_REPORT_KIND = "accvp_runtime_benchmark_factorial_v1"
FACTORIAL_RUNTIME_REPORT_SCHEMA_VERSION = 1


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (REPO_ROOT / value).resolve()


def _report_fingerprint(payload: dict[str, Any]) -> str:
    return stable_hash({key: value for key, value in payload.items() if key != "report_fingerprint"})


def _method_inputs(
    *,
    factorial_source: Path,
    factorial: dict[str, Any],
    method_id: str,
) -> tuple[Path, dict[str, Any], Path]:
    entry = dict(factorial["methods"][method_id])
    child_source = resolve_manifest_path(
        factorial_source,
        str(entry.get("replicate_manifest", "")),
    )
    child = read_json_mapping(child_source)
    rows = list(child.get("records", []) or [])
    if not rows:
        raise ValueError(f"{method_id}: optimizer-replicate manifest has no records")
    template = resolve_path(str(rows[0].get("resolved_config", "")))
    if not template.is_file():
        raise FileNotFoundError(template)
    return child_source, child, template


def _request_fingerprint(
    *,
    factorial_source: Path,
    factorial: dict[str, Any],
    seeds: list[int],
    backend: str,
    device: str,
) -> str:
    methods: dict[str, Any] = {}
    for method_id in EXPECTED_CANDIDATE_METHOD_ROLES:
        child_source, child, template = _method_inputs(
            factorial_source=factorial_source,
            factorial=factorial,
            method_id=method_id,
        )
        methods[method_id] = {
            "role": str(factorial["method_roles"][method_id]),
            "replicate_manifest": str(child_source),
            "replicate_manifest_sha256": file_sha256(child_source),
            "config_template": str(template),
            "config_template_sha256": file_sha256(template),
            "checkpoint_sha256s": [
                str(row.get("checkpoint_sha256", ""))
                for row in list(child.get("records", []) or [])
            ],
        }
    return stable_hash(
        {
            "artifact_kind": FACTORIAL_RUNTIME_REPORT_KIND,
            "schema_version": FACTORIAL_RUNTIME_REPORT_SCHEMA_VERSION,
            "runtime_implementation_version": RUNTIME_IMPLEMENTATION_VERSION,
            "factorial_manifest": str(factorial_source),
            "factorial_manifest_sha256": file_sha256(factorial_source),
            "factorial_manifest_fingerprint": str(factorial.get("manifest_fingerprint", "")),
            "final_method_id": str(factorial.get("final_method_id", "")),
            "method_roles": dict(factorial.get("method_roles", {}) or {}),
            "methods": methods,
            "simulator_seeds": [int(seed) for seed in seeds],
            "backend": str(backend),
            "device": str(device),
        }
    )


def _validate_factorial_runtime_report(
    report: dict[str, Any],
    *,
    output_path: Path,
    factorial_source: Path,
    factorial: dict[str, Any],
    requested_seeds: list[int],
    backend: str,
    device: str,
    expected_request_fingerprint: str,
) -> dict[str, Any]:
    if str(report.get("artifact_kind", "")) != FACTORIAL_RUNTIME_REPORT_KIND:
        raise ValueError("unsupported factorial policy-runtime report")
    if int(report.get("schema_version", -1)) != FACTORIAL_RUNTIME_REPORT_SCHEMA_VERSION:
        raise ValueError("unsupported factorial policy-runtime report schema")
    if str(report.get("status", "")) != "complete":
        raise ValueError("factorial policy-runtime report is not complete")
    if str(report.get("runtime_implementation_version", "")) != RUNTIME_IMPLEMENTATION_VERSION:
        raise ValueError("factorial policy-runtime implementation version mismatch")
    if str(report.get("request_fingerprint", "")) != expected_request_fingerprint:
        raise ValueError("factorial policy-runtime request fingerprint mismatch")
    if str(report.get("factorial_manifest_sha256", "")) != file_sha256(factorial_source):
        raise ValueError("factorial policy-runtime manifest hash mismatch")
    if str(report.get("factorial_manifest_fingerprint", "")) != str(
        factorial.get("manifest_fingerprint", "")
    ):
        raise ValueError("factorial policy-runtime manifest fingerprint mismatch")
    if str(report.get("final_method_id", "")) != EXPECTED_FINAL_METHOD_ID:
        raise ValueError("factorial policy-runtime final method binding mismatch")
    if [int(seed) for seed in report.get("simulator_seeds", [])] != requested_seeds:
        raise ValueError("factorial policy-runtime seed schedule mismatch")
    if str(report.get("backend", "")) != str(backend) or str(report.get("device", "")) != str(device):
        raise ValueError("factorial policy-runtime execution backend/device mismatch")

    methods = dict(report.get("methods", {}) or {})
    if set(methods) != set(EXPECTED_CANDIDATE_METHOD_ROLES):
        raise ValueError("factorial policy-runtime method set is incomplete")
    all_gates_pass = True
    for method_id, role in EXPECTED_CANDIDATE_METHOD_ROLES.items():
        entry = dict(methods[method_id])
        if str(entry.get("role", "")) != role:
            raise ValueError(f"{method_id}: factorial policy-runtime role mismatch")
        child_source, child, template = _method_inputs(
            factorial_source=factorial_source,
            factorial=factorial,
            method_id=method_id,
        )
        if str(entry.get("replicate_manifest_sha256", "")) != file_sha256(child_source):
            raise ValueError(f"{method_id}: factorial policy-runtime child manifest hash mismatch")
        checkpoint_hashes = [
            str(row.get("checkpoint_sha256", ""))
            for row in list(child.get("records", []) or [])
        ]
        if list(entry.get("checkpoint_sha256s", []) or []) != checkpoint_hashes:
            raise ValueError(f"{method_id}: factorial policy-runtime checkpoint set mismatch")
        runtime_path = _resolve(str(entry.get("runtime_report", "")))
        expected_runtime_path = (
            output_path.parent / "methods" / method_id / "replicated_runtime_report.json"
        ).resolve()
        if runtime_path != expected_runtime_path:
            raise ValueError(f"{method_id}: factorial policy-runtime child report path mismatch")
        if not runtime_path.is_file():
            raise FileNotFoundError(runtime_path)
        if str(entry.get("runtime_report_sha256", "")) != file_sha256(runtime_path):
            raise ValueError(f"{method_id}: factorial policy-runtime child report hash mismatch")
        child_report = read_json(runtime_path)
        child_request_fingerprint = replicate_runtime.runtime_replicate_request_fingerprint(
            config_template=template,
            replicate_manifest=child_source,
            manifest=child,
            seeds=requested_seeds,
            backend=backend,
            device=device,
        )
        replicate_runtime.validate_runtime_replicate_report(
            child_report,
            expected_request_fingerprint=child_request_fingerprint,
            replicate_manifest=child_source,
            manifest=child,
            requested_seeds=requested_seeds,
            backend=backend,
        )
        if str(entry.get("runtime_report_fingerprint", "")) != str(
            child_report.get("report_fingerprint", "")
        ):
            raise ValueError(f"{method_id}: factorial policy-runtime child fingerprint mismatch")
        if dict(entry.get("gate", {}) or {}) != dict(child_report.get("gate", {}) or {}):
            raise ValueError(f"{method_id}: factorial policy-runtime child gate mismatch")
        all_gates_pass = all_gates_pass and bool(child_report.get("gate", {}).get("pass", False))

    checks = dict(report.get("gate", {}).get("checks", {}) or {})
    expected_checks = {
        "complete_four_method_set": set(methods) == set(EXPECTED_CANDIDATE_METHOD_ROLES),
        "all_method_runtime_gates_pass": all_gates_pass,
        "final_method_binding": (
            str(report.get("final_method_id", "")) == EXPECTED_FINAL_METHOD_ID
            and EXPECTED_CANDIDATE_METHOD_ROLES[EXPECTED_FINAL_METHOD_ID] == FINAL_METHOD_ROLE
        ),
        "final_method_runtime_gate_pass": bool(
            methods[EXPECTED_FINAL_METHOD_ID].get("gate", {}).get("pass", False)
        ),
    }
    if checks != expected_checks or bool(report.get("gate", {}).get("pass", False)) != all(
        expected_checks.values()
    ):
        raise ValueError("factorial policy-runtime aggregate gate mismatch")
    declared = str(report.get("report_fingerprint", ""))
    if not declared or declared != _report_fingerprint(report):
        raise ValueError("factorial policy-runtime report fingerprint mismatch")
    return report


def validate_factorial_runtime_report(
    report_path: str | Path,
    *,
    factorial_manifest: str | Path,
    seeds: list[int],
    backend: str = "vectorized",
    device: str = "auto",
) -> dict[str, Any]:
    """Validate a completed factorial runtime artifact and all child reports."""

    output_path = _resolve(report_path)
    factorial_source = _resolve(factorial_manifest)
    factorial = validate_factorial_manifest(
        factorial_source,
        require_complete=True,
        verify_files=True,
    )
    requested_seeds = [int(seed) for seed in seeds]
    expected_request_fingerprint = _request_fingerprint(
        factorial_source=factorial_source,
        factorial=factorial,
        seeds=requested_seeds,
        backend=backend,
        device=device,
    )
    return _validate_factorial_runtime_report(
        read_json(output_path),
        output_path=output_path,
        factorial_source=factorial_source,
        factorial=factorial,
        requested_seeds=requested_seeds,
        backend=backend,
        device=device,
        expected_request_fingerprint=expected_request_fingerprint,
    )


def run(
    *,
    factorial_manifest: str | Path,
    seeds: list[int],
    backend: str,
    output: str | Path,
    device: str = "auto",
    resume: bool = True,
) -> Path:
    factorial_source = _resolve(factorial_manifest)
    factorial = validate_factorial_manifest(
        factorial_source,
        require_complete=True,
        verify_files=True,
    )
    requested_seeds = [int(seed) for seed in seeds]
    if len(requested_seeds) < 30 or len(requested_seeds) != len(set(requested_seeds)):
        raise ValueError("factorial policy runtime requires at least 30 distinct simulator seeds")
    if backend not in {"reference", "vectorized"}:
        raise ValueError("factorial policy runtime backend must be reference or vectorized")
    output_path = _resolve(output)
    request_fingerprint = _request_fingerprint(
        factorial_source=factorial_source,
        factorial=factorial,
        seeds=requested_seeds,
        backend=backend,
        device=device,
    )
    prior_failed_attempt: dict[str, Any] | None = None
    if output_path.exists():
        if not resume:
            raise FileExistsError(output_path)
        existing = read_json(output_path)
        existing_version = str(existing.get("runtime_implementation_version", ""))
        if (
            str(existing.get("artifact_kind", "")) == FACTORIAL_RUNTIME_REPORT_KIND
            and str(existing.get("status", "")) == "complete"
            and not bool(existing.get("gate", {}).get("pass", False))
            and existing_version != RUNTIME_IMPLEMENTATION_VERSION
        ):
            declared = str(existing.get("report_fingerprint", ""))
            if not declared or declared != _report_fingerprint(existing):
                raise ValueError("refusing to archive a failed runtime report with invalid fingerprint")
            archive_dir = (
                output_path.parent
                / "failed_attempts"
                / f"{existing_version or 'legacy'}_{declared[:16]}"
            ).resolve()
            if archive_dir.exists():
                raise FileExistsError(
                    f"failed runtime archive already exists; inspect it before retrying: {archive_dir}"
                )
            archive_dir.mkdir(parents=True, exist_ok=False)
            methods_dir = output_path.parent / "methods"
            if methods_dir.exists():
                methods_dir.replace(archive_dir / "methods")
            output_path.replace(archive_dir / output_path.name)
            prior_failed_attempt = {
                "runtime_implementation_version": existing_version or "legacy",
                "report_fingerprint": declared,
                "gate_pass": False,
                "archive_dir": str(archive_dir),
                "report_sha256": file_sha256(archive_dir / output_path.name),
            }
            print(
                "[accvp_runtime_factorial] action=archive_failed_implementation "
                f"from={existing_version or 'legacy'} to={RUNTIME_IMPLEMENTATION_VERSION} "
                f"archive={archive_dir}",
                flush=True,
            )
        else:
            _validate_factorial_runtime_report(
                existing,
                output_path=output_path,
                factorial_source=factorial_source,
                factorial=factorial,
                requested_seeds=requested_seeds,
                backend=backend,
                device=device,
                expected_request_fingerprint=request_fingerprint,
            )
            print(f"[accvp_runtime_factorial] status=complete action=skip report={output_path}", flush=True)
            return output_path

    methods: dict[str, Any] = {}
    for method_id, role in EXPECTED_CANDIDATE_METHOD_ROLES.items():
        child_source, child, template = _method_inputs(
            factorial_source=factorial_source,
            factorial=factorial,
            method_id=method_id,
        )
        child_output = (
            output_path.parent / "methods" / method_id / "replicated_runtime_report.json"
        ).resolve()
        print(f"[accvp_runtime_factorial] method={method_id} start", flush=True)
        replicate_runtime.run(
            config_template=template,
            replicate_manifest=child_source,
            seeds=requested_seeds,
            backend=backend,
            output=child_output,
            device=device,
            resume=resume,
        )
        child_report = read_json(child_output)
        child_request_fingerprint = replicate_runtime.runtime_replicate_request_fingerprint(
            config_template=template,
            replicate_manifest=child_source,
            manifest=child,
            seeds=requested_seeds,
            backend=backend,
            device=device,
        )
        replicate_runtime.validate_runtime_replicate_report(
            child_report,
            expected_request_fingerprint=child_request_fingerprint,
            replicate_manifest=child_source,
            manifest=child,
            requested_seeds=requested_seeds,
            backend=backend,
        )
        checkpoint_hashes = [
            str(row.get("checkpoint_sha256", ""))
            for row in list(child.get("records", []) or [])
        ]
        methods[method_id] = {
            "role": role,
            "replicate_manifest": str(child_source),
            "replicate_manifest_sha256": file_sha256(child_source),
            "checkpoint_sha256s": checkpoint_hashes,
            "runtime_report": str(child_output),
            "runtime_report_sha256": file_sha256(child_output),
            "runtime_report_fingerprint": str(child_report.get("report_fingerprint", "")),
            "gate": dict(child_report.get("gate", {}) or {}),
        }
        print(
            f"[accvp_runtime_factorial] method={method_id} "
            f"gate_pass={bool(child_report.get('gate', {}).get('pass', False))}",
            flush=True,
        )

    checks = {
        "complete_four_method_set": set(methods) == set(EXPECTED_CANDIDATE_METHOD_ROLES),
        "all_method_runtime_gates_pass": all(
            bool(entry.get("gate", {}).get("pass", False)) for entry in methods.values()
        ),
        "final_method_binding": (
            str(factorial.get("final_method_id", "")) == EXPECTED_FINAL_METHOD_ID
            and str(factorial.get("method_roles", {}).get(EXPECTED_FINAL_METHOD_ID, ""))
            == FINAL_METHOD_ROLE
        ),
        "final_method_runtime_gate_pass": bool(
            methods[EXPECTED_FINAL_METHOD_ID].get("gate", {}).get("pass", False)
        ),
    }
    overflow_histogram: Counter[str] = Counter()
    overflow_examples: list[dict[str, Any]] = []
    overflow_count = 0
    for entry in methods.values():
        child_gate = dict(entry.get("gate", {}) or {})
        overflow_count += int(child_gate.get("critical_actor_overflow_count", 0))
        overflow_histogram.update(
            dict(child_gate.get("critical_actor_overflow_histogram", {}) or {})
        )
        remaining = 20 - len(overflow_examples)
        if remaining > 0:
            overflow_examples.extend(
                dict(value)
                for value in list(
                    child_gate.get("critical_actor_overflow_examples", []) or []
                )[:remaining]
            )
    payload = {
        "artifact_kind": FACTORIAL_RUNTIME_REPORT_KIND,
        "schema_version": FACTORIAL_RUNTIME_REPORT_SCHEMA_VERSION,
        "runtime_implementation_version": RUNTIME_IMPLEMENTATION_VERSION,
        "status": "complete",
        "backend": str(backend),
        "device": str(device),
        "hard_real_time_claim": all(
            bool(read_json(entry["runtime_report"]).get("hard_real_time_claim", False))
            for entry in methods.values()
        ),
        "factorial_manifest": str(factorial_source),
        "factorial_manifest_sha256": file_sha256(factorial_source),
        "factorial_manifest_fingerprint": str(factorial.get("manifest_fingerprint", "")),
        "final_method_id": str(factorial.get("final_method_id", "")),
        "method_roles": dict(factorial.get("method_roles", {}) or {}),
        "simulator_seeds": requested_seeds,
        "methods": methods,
        "critical_actor_overflow": {
            "count": overflow_count,
            "histogram": dict(overflow_histogram),
            "examples": overflow_examples,
            "sample_limit": 20,
        },
        "gate": {"checks": checks, "pass": all(checks.values())},
        "request_fingerprint": request_fingerprint,
    }
    if prior_failed_attempt is not None:
        payload["prior_failed_attempt"] = prior_failed_attempt
    payload["report_fingerprint"] = _report_fingerprint(payload)
    write_json_new(output_path, payload)
    _validate_factorial_runtime_report(
        read_json(output_path),
        output_path=output_path,
        factorial_source=factorial_source,
        factorial=factorial,
        requested_seeds=requested_seeds,
        backend=backend,
        device=device,
        expected_request_fingerprint=request_fingerprint,
    )
    print(
        f"[accvp_runtime_factorial] status=complete gate_pass={all(checks.values())} "
        f"report={output_path}",
        flush=True,
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run and aggregate the formal policy-runtime gate for all four Candidate factorial methods"
    )
    parser.add_argument("--factorial-manifest", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--backend", choices=("reference", "vectorized"), default="vectorized")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Fail if any immutable runtime artifact already exists instead of validating and reusing it",
    )
    args = parser.parse_args()
    path = run(
        factorial_manifest=args.factorial_manifest,
        seeds=args.seeds,
        backend=args.backend,
        output=args.output,
        device=args.device,
        resume=not args.no_resume,
    )
    print(f"accvp_runtime_benchmark_factorial={path}")


if __name__ == "__main__":
    main()
