from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping

from safe_rl.accvp.contracts.schema import file_sha256, read_json
from safe_rl.analysis.paired_statistics import holm_adjust
from safe_rl.evaluation_protocol import stable_hash
from safe_rl.pipeline.common import write_report
from safe_rl.pipeline.stage5_generate_factorial_configs import (
    DEFAULT_COMPARISONS,
    FACTORIAL_REQUEST_KIND,
    FACTORIAL_REQUEST_SCHEMA_VERSION,
    FACTORIAL_RUNTIME_KIND,
    FINAL_COMPARISON_ID,
)
from safe_rl.pipeline.stage5_replicated_aggregate import (
    REPORT_ARTIFACT_KIND as CHILD_REPORT_KIND,
    aggregate_manifest,
)
from safe_rl.ppo_factorial import (
    EXPECTED_FINAL_METHOD_ID,
    validate_factorial_manifest,
)
from safe_rl.utils.config import REPO_ROOT


FACTORIAL_REPORT_KIND = "stage5_factorial_report_v1"
FACTORIAL_REPORT_SCHEMA_VERSION = 1


def _resolve(path: str | Path, *, relative_to: Path | None = None) -> Path:
    value = Path(path)
    if value.is_absolute():
        return value.resolve()
    return ((relative_to or REPO_ROOT) / value).resolve()


def _normalise_sha256(value: Any, *, field: str) -> str:
    digest = str(value or "").strip().lower()
    if len(digest) != 64:
        raise ValueError(f"{field} must be a 64-character SHA-256 digest")
    try:
        int(digest, 16)
    except ValueError as exc:
        raise ValueError(f"{field} must be a hexadecimal SHA-256 digest") from exc
    return digest


def load_factorial_request(request_path: str | Path) -> tuple[Path, dict[str, Any]]:
    source = _resolve(request_path)
    request = read_json(source)
    if str(request.get("artifact_kind", "")) != FACTORIAL_REQUEST_KIND:
        raise ValueError("unsupported Stage5 factorial request")
    if int(request.get("schema_version", -1)) != FACTORIAL_REQUEST_SCHEMA_VERSION:
        raise ValueError("unsupported Stage5 factorial request schema")
    if str(request.get("status", "")) != "prepared":
        raise ValueError("Stage5 factorial request is not prepared")
    if str(request.get("final_method_id", "")) != EXPECTED_FINAL_METHOD_ID:
        raise ValueError("Stage5 factorial request has the wrong final method")
    if str(request.get("final_comparison_id", "")) != FINAL_COMPARISON_ID:
        raise ValueError("Stage5 factorial request has the wrong final comparison")
    factorial_path = _resolve(
        request.get("factorial_manifest", ""),
        relative_to=source.parent,
    )
    expected_factorial_sha = _normalise_sha256(
        request.get("factorial_manifest_sha256"),
        field="factorial_manifest_sha256",
    )
    if not factorial_path.is_file() or file_sha256(factorial_path) != expected_factorial_sha:
        raise ValueError("Stage5 factorial request PPO-manifest binding mismatch")
    factorial = validate_factorial_manifest(
        factorial_path,
        require_complete=True,
        verify_files=True,
    )
    if str(factorial.get("final_method_id", "")) != EXPECTED_FINAL_METHOD_ID:
        raise ValueError("request-bound PPO factorial manifest has the wrong final method")
    declared_fingerprint = str(request.get("factorial_manifest_fingerprint", ""))
    if declared_fingerprint != str(factorial.get("manifest_fingerprint", "")):
        raise ValueError("Stage5 request PPO factorial fingerprint mismatch")

    baseline_path = _resolve(
        request.get("baseline_replicate_manifest", ""),
        relative_to=source.parent,
    )
    expected_baseline_sha = _normalise_sha256(
        request.get("baseline_replicate_manifest_sha256"),
        field="baseline_replicate_manifest_sha256",
    )
    if not baseline_path.is_file() or file_sha256(baseline_path) != expected_baseline_sha:
        raise ValueError("Stage5 factorial request baseline-manifest binding mismatch")

    runtime_path = _resolve(
        request.get("runtime_factorial_report", ""),
        relative_to=source.parent,
    )
    expected_runtime_sha = _normalise_sha256(
        request.get("runtime_factorial_report_sha256"),
        field="runtime_factorial_report_sha256",
    )
    if not runtime_path.is_file() or file_sha256(runtime_path) != expected_runtime_sha:
        raise ValueError("Stage5 factorial request runtime-report binding mismatch")
    runtime = read_json(runtime_path)
    if str(runtime.get("artifact_kind", "")) != FACTORIAL_RUNTIME_KIND:
        raise ValueError("Stage5 factorial request requires a factorial runtime report")
    if int(runtime.get("schema_version", -1)) != 1 or runtime.get("status") != "complete":
        raise ValueError("Stage5 factorial request runtime report is not complete")
    if str(runtime.get("final_method_id", "")) != EXPECTED_FINAL_METHOD_ID:
        raise ValueError("Stage5 factorial request runtime final-method mismatch")
    if str(runtime.get("factorial_manifest_sha256", "")) != expected_factorial_sha:
        raise ValueError("Stage5 factorial runtime/PPO manifest binding mismatch")
    if not bool((runtime.get("gate", {}) or {}).get("pass", False)):
        raise ValueError("Stage5 factorial request binds a failed runtime report")

    rows = request.get("comparisons", []) or []
    if not isinstance(rows, list) or len(rows) != 6:
        raise ValueError("formal Stage5 factorial request must contain six comparisons")
    comparison_ids = [str(row.get("comparison_id", "")) for row in rows]
    if len(set(comparison_ids)) != len(comparison_ids):
        raise ValueError("Stage5 factorial request contains duplicate comparisons")
    expected_comparisons = {
        item["comparison_id"]: {
            key: item[key]
            for key in ("left_method_id", "right_method_id", "family", "role")
        }
        for item in DEFAULT_COMPARISONS
    }
    declared_comparisons = {
        str(item.get("comparison_id", "")): {
            key: str(item.get(key, ""))
            for key in ("left_method_id", "right_method_id", "family", "role")
        }
        for item in rows
    }
    if declared_comparisons != expected_comparisons:
        raise ValueError("Stage5 factorial request comparison design is not the frozen design")
    final = [row for row in rows if str(row.get("comparison_id", "")) == FINAL_COMPARISON_ID]
    if len(final) != 1 or str(final[0].get("right_method_id", "")) != EXPECTED_FINAL_METHOD_ID:
        raise ValueError("Stage5 factorial request lacks its frozen final comparison")
    return source, request


def _explicit_primary_pvalue(report: Mapping[str, Any]) -> float | None:
    """Return only a p-value explicitly declared valid for this Holm family.

    Crossed bootstrap confidence intervals are not silently converted to
    p-values.  Current replicated Stage5 reports therefore normally return
    ``None`` here, and the total report records that Holm was not performed.
    """

    hypothesis = report.get("primary_hypothesis", {})
    if not isinstance(hypothesis, Mapping):
        return None
    if hypothesis.get("pvalue_valid_for_holm_family") is not True:
        return None
    value = hypothesis.get("pvalue")
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not 0.0 <= result <= 1.0:
        raise ValueError("explicit primary p-value must be between zero and one")
    return result


def validate_child_report(
    comparison: Mapping[str, Any],
    *,
    request_path: Path,
) -> tuple[Path, dict[str, Any]]:
    child_request_path = _resolve(
        comparison.get("replicated_request", ""),
        relative_to=request_path.parent,
    )
    expected_request_sha = _normalise_sha256(
        comparison.get("replicated_request_sha256"),
        field="replicated_request_sha256",
    )
    if not child_request_path.is_file() or file_sha256(child_request_path) != expected_request_sha:
        raise ValueError(
            f"Stage5 comparison request binding mismatch: {comparison.get('comparison_id')}"
        )
    report_path = _resolve(
        comparison.get("aggregate_report", ""),
        relative_to=request_path.parent,
    )
    if not report_path.is_file():
        raise FileNotFoundError(report_path)
    report = read_json(report_path)
    if str(report.get("artifact_kind", "")) != CHILD_REPORT_KIND:
        raise ValueError(f"unsupported child Stage5 report: {report_path}")
    if not bool((report.get("gate", {}) or {}).get("pass", False)):
        raise ValueError(f"child Stage5 gate failed: {report_path}")
    expected = aggregate_manifest(child_request_path)
    if stable_hash(report) != stable_hash(expected):
        raise ValueError(
            "existing child Stage5 aggregate does not match its frozen request/source reports: "
            f"{report_path}"
        )
    return report_path, report


def aggregate(request_path: str | Path) -> dict[str, Any]:
    source, request = load_factorial_request(request_path)
    summaries: list[dict[str, Any]] = []
    family_pvalues: dict[str, dict[str, float]] = {}
    final_binding: dict[str, Any] | None = None
    for comparison in request["comparisons"]:
        report_path, report = validate_child_report(comparison, request_path=source)
        comparison_id = str(comparison["comparison_id"])
        family = str(comparison.get("family", ""))
        pvalue = _explicit_primary_pvalue(report)
        if pvalue is not None:
            family_pvalues.setdefault(family, {})[comparison_id] = pvalue
        summary = {
            "comparison_id": comparison_id,
            "left_method_id": str(comparison.get("left_method_id", "")),
            "right_method_id": str(comparison.get("right_method_id", "")),
            "family": family,
            "role": str(comparison.get("role", "")),
            "report": str(report_path),
            "report_sha256": file_sha256(report_path),
            "gate": dict(report.get("gate", {}) or {}),
            "statistics_fingerprint": str(
                (report.get("statistics", {}) or {}).get("statistics_fingerprint", "")
            ),
            "explicit_primary_pvalue": pvalue,
        }
        summaries.append(summary)
        if comparison_id == FINAL_COMPARISON_ID:
            final_binding = summary
    if final_binding is None:
        raise ValueError("Stage5 factorial aggregate lacks the final comparison")

    corrections: dict[str, Any] = {}
    families = sorted({str(row.get("family", "")) for row in summaries})
    for family in families:
        members = [row["comparison_id"] for row in summaries if row["family"] == family]
        available = family_pvalues.get(family, {})
        if available and set(available) == set(members):
            corrections[family] = {
                "method": "holm",
                "performed": True,
                "raw_pvalues": available,
                "adjusted_pvalues": holm_adjust(available),
            }
        else:
            corrections[family] = {
                "method": "holm",
                "performed": False,
                "reason": (
                    "replicated crossed-bootstrap reports expose confidence intervals, not "
                    "valid cross-comparison p-values; no p-values were inferred"
                ),
                "comparisons": members,
                "explicit_valid_pvalues": available,
            }

    all_children_pass = all(bool(row["gate"].get("pass", False)) for row in summaries)
    payload: dict[str, Any] = {
        "artifact_kind": FACTORIAL_REPORT_KIND,
        "schema_version": FACTORIAL_REPORT_SCHEMA_VERSION,
        "status": "complete",
        "protocol_id": str(request.get("protocol_id", "")),
        "factorial_request": str(source),
        "factorial_request_sha256": file_sha256(source),
        "factorial_manifest": str(request.get("factorial_manifest", "")),
        "factorial_manifest_sha256": str(request.get("factorial_manifest_sha256", "")),
        "factorial_manifest_fingerprint": str(
            request.get("factorial_manifest_fingerprint", "")
        ),
        "runtime_factorial_report": str(request.get("runtime_factorial_report", "")),
        "runtime_factorial_report_sha256": str(
            request.get("runtime_factorial_report_sha256", "")
        ),
        "final_method_id": EXPECTED_FINAL_METHOD_ID,
        "final_comparison_id": FINAL_COMPARISON_ID,
        "comparisons": summaries,
        "multiple_comparison_correction": corrections,
        "final_comparison_report": {
            "path": final_binding["report"],
            "sha256": final_binding["report_sha256"],
            "artifact_kind": CHILD_REPORT_KIND,
            "gate": final_binding["gate"],
        },
        "gate": {
            "pass": bool(all_children_pass and final_binding["gate"].get("pass", False)),
            "all_preregistered_comparisons_complete": len(summaries) == 6,
            "all_child_artifact_gates_pass": all_children_pass,
            "final_method_bound": True,
            "final_comparison_gate_pass": bool(final_binding["gate"].get("pass", False)),
        },
    }
    payload["report_fingerprint"] = stable_hash(payload)
    return payload


def run(request_path: str | Path, output_path: str | Path) -> Path:
    output = _resolve(output_path)
    payload = aggregate(request_path)
    if output.exists():
        existing = read_json(output)
        if stable_hash(existing) != stable_hash(payload):
            raise FileExistsError(f"refusing to replace a different formal report: {output}")
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    write_report(output, payload)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate the formal Stage5 factorial")
    parser.add_argument("--request", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    path = run(args.request, args.output)
    print(f"stage5_factorial_report={path}")


if __name__ == "__main__":
    main()
