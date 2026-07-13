from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Mapping

from safe_rl.accvp.artifacts import (
    ACCVP_ARTIFACT_GENERATION,
    ACCVP_ARTIFACT_KIND,
    ACCVP_BUNDLE_SCHEMA_VERSION,
    resolve_v2_bundle,
)
from safe_rl.analysis.paired_statistics import (
    DEFAULT_BINARY_METRICS,
    DEFAULT_CONTINUOUS_METRICS,
    build_replicated_pair_statistics,
)
from safe_rl.evaluation_protocol import file_sha256, stable_hash
from safe_rl.pipeline.common import write_report


REQUEST_ARTIFACT_KIND = "stage5_replicated_aggregate_request_v1"
REPORT_ARTIFACT_KIND = "stage5_replicated_paired_report_v1"
REPORT_SCHEMA_VERSION = 2
MIN_FORMAL_TRAINING_SEED_COUNT = 5
FORMAL_CANDIDATE_VARIANTS = frozenset(
    {"full_candidate_gate_v1", "viability_lite_task_v1"}
)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must contain an object: {path}")
    return payload


def _resolve(source: Any, *, base_dir: Path) -> Path:
    path = Path(str(source))
    return path if path.is_absolute() else base_dir / path


def _explicit_sha256(value: Any, *, field: str) -> str:
    digest = str(value or "").strip().lower()
    if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError(f"{field} must be an explicit 64-character SHA-256 digest")
    return digest


def _formal_candidate_binding(
    request: Mapping[str, Any],
    *,
    request_path: Path,
) -> dict[str, Any]:
    declaration = request.get("candidate_manifest")
    if not isinstance(declaration, Mapping):
        raise ValueError(
            "formal replicated aggregation requires candidate_manifest with "
            "path, sha256, artifact_fingerprint, and artifact_variant"
        )
    recorded_path = str(declaration.get("path", "")).strip()
    if not recorded_path:
        raise ValueError("formal candidate_manifest requires path")
    candidate_path = _resolve(recorded_path, base_dir=request_path.parent).resolve()
    if not candidate_path.is_file():
        raise FileNotFoundError(candidate_path)
    expected_manifest_sha256 = _explicit_sha256(
        declaration.get("sha256"), field="candidate_manifest sha256"
    )
    actual_manifest_sha256 = file_sha256(candidate_path)
    if actual_manifest_sha256 != expected_manifest_sha256:
        raise ValueError(
            "candidate manifest SHA-256 mismatch: "
            f"request={expected_manifest_sha256} actual={actual_manifest_sha256}"
        )
    expected_fingerprint = _explicit_sha256(
        declaration.get("artifact_fingerprint"),
        field="candidate_manifest artifact_fingerprint",
    )
    expected_variant = str(declaration.get("artifact_variant", "")).strip()
    if expected_variant not in FORMAL_CANDIDATE_VARIANTS:
        raise ValueError(
            "candidate_manifest artifact_variant must be an explicit supported VNext variant"
        )
    bundle, resolved = resolve_v2_bundle(candidate_path)
    if file_sha256(candidate_path) != actual_manifest_sha256:
        raise ValueError("candidate manifest changed while formal binding was validated")
    actual_fingerprint = _explicit_sha256(
        bundle.get("artifact_fingerprint"),
        field="resolved candidate artifact_fingerprint",
    )
    if actual_fingerprint != expected_fingerprint:
        raise ValueError(
            "candidate manifest artifact_fingerprint mismatch: "
            f"request={expected_fingerprint} bundle={actual_fingerprint}"
        )
    actual_variant = str(bundle.get("artifact_variant", ""))
    if actual_variant != expected_variant:
        raise ValueError(
            "candidate manifest artifact_variant mismatch: "
            f"request={expected_variant!r} bundle={actual_variant!r}"
        )
    if str(bundle.get("artifact_kind", "")) != ACCVP_ARTIFACT_KIND:
        raise ValueError("formal candidate manifest is not an ACCVP bundle")
    if int(bundle.get("bundle_schema_version", -1)) != ACCVP_BUNDLE_SCHEMA_VERSION:
        raise ValueError("formal candidate manifest bundle schema mismatch")
    if str(bundle.get("artifact_generation", "")) != ACCVP_ARTIFACT_GENERATION:
        raise ValueError("formal candidate manifest generation mismatch")

    candidate_side = str(request.get("candidate_side", "")).strip().lower()
    if candidate_side not in {"left", "right"}:
        raise ValueError("formal candidate_side must be either 'left' or 'right'")
    acceptance_key = str(request.get("source_acceptance_key", "")).strip()
    if not acceptance_key:
        raise ValueError("formal replicated aggregation requires source_acceptance_key")
    requested_contract_sha256 = _explicit_sha256(
        request.get("formal_runtime_contract_sha256"),
        field="formal_runtime_contract_sha256",
    )
    bundle_contract_sha256 = _explicit_sha256(
        bundle.get("formal_runtime_contract_sha256"),
        field="candidate bundle formal_runtime_contract_sha256",
    )
    if bundle_contract_sha256 != requested_contract_sha256:
        raise ValueError(
            "formal runtime contract SHA-256 does not match candidate bundle: "
            f"request={requested_contract_sha256} bundle={bundle_contract_sha256}"
        )
    predictor_path = resolved.get("predictor")
    if predictor_path is None:
        raise ValueError("formal candidate bundle is missing its predictor")
    predictor_sha256 = _explicit_sha256(
        bundle.get("predictor_sha256"), field="candidate bundle predictor_sha256"
    )
    if file_sha256(predictor_path) != predictor_sha256:
        raise ValueError("candidate bundle predictor SHA-256 mismatch")
    return {
        "required": True,
        "candidate_manifest": {
            "path": str(candidate_path),
            "sha256": actual_manifest_sha256,
            "artifact_fingerprint": actual_fingerprint,
            "artifact_variant": actual_variant,
            "artifact_generation": str(bundle["artifact_generation"]),
            "bundle_schema_version": int(bundle["bundle_schema_version"]),
            "lifecycle_state": str(bundle.get("lifecycle_state", "")),
            "predictor_sha256": predictor_sha256,
            "formal_runtime_contract_sha256": bundle_contract_sha256,
        },
        "candidate_side": candidate_side,
        "source_acceptance_key": acceptance_key,
        "formal_runtime_contract_sha256": requested_contract_sha256,
    }


def _source_acceptance(
    source: Mapping[str, Any],
    *,
    source_path: Path,
    acceptance_key: str,
) -> dict[str, Any]:
    acceptance = source.get("acceptance", {})
    if not isinstance(acceptance, Mapping):
        raise ValueError(f"source Stage5 acceptance is not an object: {source_path}")
    record = acceptance.get(acceptance_key)
    if not isinstance(record, Mapping):
        raise ValueError(
            f"source Stage5 report is missing acceptance[{acceptance_key!r}]: {source_path}"
        )
    if record.get("available") is not True:
        raise ValueError(
            f"source Stage5 acceptance[{acceptance_key!r}] is not available: {source_path}"
        )
    if record.get("regression") is not False:
        raise ValueError(
            f"source Stage5 acceptance[{acceptance_key!r}] reports regression: {source_path}"
        )
    payload = dict(record)
    return {
        "key": acceptance_key,
        "available": True,
        "regression": False,
        "record_sha256": stable_hash(payload),
    }


def _candidate_group_bundle_binding(
    *,
    evidence: Mapping[str, Any],
    candidate_group: Mapping[str, Any],
    candidate_group_name: str,
    candidate_binding: Mapping[str, Any],
    source_path: Path,
) -> dict[str, Any]:
    bindings = evidence.get("accvp_group_bindings", {})
    if not isinstance(bindings, Mapping):
        raise ValueError(
            f"source Stage5 ACCVP group bindings are not an object: {source_path}"
        )
    recorded = bindings.get(candidate_group_name)
    if not isinstance(recorded, Mapping):
        raise ValueError(
            "candidate-side Stage5 group is missing ACCVP bundle lineage: "
            f"group={candidate_group_name!r} source={source_path}"
        )
    inline = candidate_group.get("accvp_bundle_lineage")
    if not isinstance(inline, Mapping) or dict(inline) != dict(recorded):
        raise ValueError(
            "candidate-side group ACCVP bundle lineage disagrees with source evidence: "
            f"group={candidate_group_name!r}"
        )
    recorded_payload = dict(recorded)
    recorded_fingerprint = _explicit_sha256(
        recorded_payload.pop("binding_fingerprint", None),
        field=f"candidate group {candidate_group_name} binding_fingerprint",
    )
    if stable_hash(recorded_payload) != recorded_fingerprint:
        raise ValueError(
            f"candidate group {candidate_group_name!r} bundle binding_fingerprint mismatch"
        )
    manifest_binding = dict(candidate_binding.get("candidate_manifest", {}) or {})
    expected = {
        "manifest_sha256": str(manifest_binding.get("sha256", "")),
        "artifact_fingerprint": str(manifest_binding.get("artifact_fingerprint", "")),
        "artifact_variant": str(manifest_binding.get("artifact_variant", "")),
        "formal_runtime_contract_sha256": str(
            candidate_binding.get("formal_runtime_contract_sha256", "")
        ),
    }
    differing = sorted(
        key for key, value in expected.items() if recorded.get(key) != value
    )
    if differing:
        raise ValueError(
            "candidate-side Stage5 group did not use the request-bound ACCVP bundle: "
            f"group={candidate_group_name!r} fields={differing}"
        )
    return {
        "group": candidate_group_name,
        **expected,
        "binding_fingerprint": recorded_fingerprint,
    }


def _group_report(source: Mapping[str, Any], group_name: str) -> Mapping[str, Any]:
    groups = source.get("groups", {})
    if not isinstance(groups, Mapping) or group_name not in groups:
        raise ValueError(f"source Stage5 report is missing group {group_name!r}")
    group = groups[group_name]
    if not isinstance(group, Mapping):
        raise ValueError(f"source Stage5 group {group_name!r} is not an object")
    return group


def _group_training_seed(group: Mapping[str, Any], *, group_name: str) -> int:
    comparative = group.get("comparative", {})
    if not isinstance(comparative, Mapping) or comparative.get("training_seed") is None:
        raise ValueError(
            f"replicated Stage5 group {group_name!r} requires comparative.training_seed"
        )
    seed = int(comparative["training_seed"])
    direct = group.get("training_seed")
    if direct is not None and int(direct) != seed:
        raise ValueError(f"Stage5 group {group_name!r} contains conflicting training seeds")
    return seed


def _episode_seeds(group: Mapping[str, Any], *, group_name: str) -> list[int]:
    seeds: list[int] = []
    for row in group.get("episodes", []):
        if not isinstance(row, Mapping):
            raise ValueError(f"Stage5 group {group_name!r} contains a non-object episode")
        value = row.get("seed", row.get("episode_seed"))
        if value is None:
            raise ValueError(f"Stage5 group {group_name!r} contains an episode without a seed")
        seeds.append(int(value))
    if len(seeds) != len(set(seeds)):
        raise ValueError(f"Stage5 group {group_name!r} contains duplicate simulator seeds")
    return sorted(seeds)


def _lineage_checkpoint_sha256(source: Mapping[str, Any], *, group_name: str) -> str:
    lineage = source.get("evidence_lineage", {})
    artifacts = lineage.get("artifacts", {}) if isinstance(lineage, Mapping) else {}
    key = f"ppo_model:{group_name}"
    record = artifacts.get(key) if isinstance(artifacts, Mapping) else None
    if not isinstance(record, Mapping):
        raise ValueError(f"source Stage5 lineage is missing checkpoint record {key!r}")
    return _explicit_sha256(record.get("sha256"), field=f"lineage {key} sha256")


def _validate_source_report(
    source: Mapping[str, Any],
    *,
    source_path: Path,
    left_group_name: str,
    right_group_name: str,
    training_seed: int,
    expected_left_sha256: str,
    expected_right_sha256: str,
    require_strict_lineage: bool,
    source_acceptance_key: str | None = None,
    candidate_binding: Mapping[str, Any] | None = None,
) -> tuple[Mapping[str, Any], Mapping[str, Any], dict[str, Any]]:
    if not bool(source.get("paired_eval", False)):
        raise ValueError(f"replicated aggregation requires paired_eval=true: {source_path}")
    left_group = _group_report(source, left_group_name)
    right_group = _group_report(source, right_group_name)
    for name, group in ((left_group_name, left_group), (right_group_name, right_group)):
        actual_seed = _group_training_seed(group, group_name=name)
        if actual_seed != int(training_seed):
            raise ValueError(
                f"Stage5 group {name!r} training seed {actual_seed} does not match "
                f"request seed {training_seed}"
            )
    left_seeds = _episode_seeds(left_group, group_name=left_group_name)
    right_seeds = _episode_seeds(right_group, group_name=right_group_name)
    if left_seeds != right_seeds:
        raise ValueError(
            f"Stage5 groups {left_group_name!r}/{right_group_name!r} do not contain "
            "identical simulator seeds"
        )
    declared_seeds = [int(value) for value in list(source.get("seeds", []) or [])]
    if declared_seeds:
        if len(declared_seeds) != len(set(declared_seeds)):
            raise ValueError(f"source Stage5 report declares duplicate simulator seeds: {source_path}")
        if sorted(declared_seeds) != left_seeds:
            raise ValueError(
                f"source Stage5 seed ledger does not match group episodes: {source_path}"
            )

    actual_left_sha256 = _lineage_checkpoint_sha256(source, group_name=left_group_name)
    actual_right_sha256 = _lineage_checkpoint_sha256(source, group_name=right_group_name)
    if actual_left_sha256 != expected_left_sha256:
        raise ValueError(
            f"left checkpoint hash mismatch for training seed {training_seed}: "
            f"request={expected_left_sha256} lineage={actual_left_sha256}"
        )
    if actual_right_sha256 != expected_right_sha256:
        raise ValueError(
            f"right checkpoint hash mismatch for training seed {training_seed}: "
            f"request={expected_right_sha256} lineage={actual_right_sha256}"
        )

    evidence = source.get("evidence_lineage", {})
    if not isinstance(evidence, Mapping):
        raise ValueError(f"source Stage5 report has invalid evidence_lineage: {source_path}")
    if require_strict_lineage and not bool(evidence.get("protocol_strict", False)):
        raise ValueError(f"replicated formal aggregation requires strict source lineage: {source_path}")
    protocol_id = str(evidence.get("protocol_id", ""))
    if require_strict_lineage and not protocol_id:
        raise ValueError(f"strict source Stage5 lineage is missing protocol_id: {source_path}")
    lineage_fingerprint = str(evidence.get("lineage_fingerprint", ""))
    if require_strict_lineage and not lineage_fingerprint:
        raise ValueError(
            f"strict source Stage5 lineage is missing lineage_fingerprint: {source_path}"
        )
    if lineage_fingerprint:
        lineage_content = dict(evidence)
        lineage_content.pop("lineage_fingerprint", None)
        if stable_hash(lineage_content) != lineage_fingerprint:
            raise ValueError(
                f"source Stage5 lineage_fingerprint mismatch: {source_path}"
            )
    acceptance = (
        None
        if source_acceptance_key is None
        else _source_acceptance(
            source,
            source_path=source_path,
            acceptance_key=source_acceptance_key,
        )
    )
    candidate_group_bundle = None
    if candidate_binding is not None:
        candidate_side = str(candidate_binding.get("candidate_side", ""))
        if candidate_side == "left":
            candidate_group_name = left_group_name
            candidate_group = left_group
        elif candidate_side == "right":
            candidate_group_name = right_group_name
            candidate_group = right_group
        else:
            raise ValueError("formal candidate binding has an invalid candidate_side")
        candidate_group_bundle = _candidate_group_bundle_binding(
            evidence=evidence,
            candidate_group=candidate_group,
            candidate_group_name=candidate_group_name,
            candidate_binding=candidate_binding,
            source_path=source_path,
        )
    return left_group, right_group, {
        "protocol_id": protocol_id,
        "protocol_strict": bool(evidence.get("protocol_strict", False)),
        "lineage_fingerprint": lineage_fingerprint,
        "safety_metric_version": str(source.get("safety_metric_version", "")),
        "simulator_seeds": left_seeds,
        "acceptance": acceptance,
        "candidate_group_bundle": candidate_group_bundle,
    }


def aggregate_manifest(manifest_path: str | Path) -> dict[str, Any]:
    manifest_path = Path(manifest_path)
    manifest = _read_json(manifest_path)
    if str(manifest.get("artifact_kind", "")) != REQUEST_ARTIFACT_KIND:
        raise ValueError(
            f"replicated aggregate request artifact_kind must be {REQUEST_ARTIFACT_KIND!r}"
        )
    comparison_id = str(manifest.get("comparison_id", "")).strip()
    if not comparison_id:
        raise ValueError("replicated aggregate request requires comparison_id")
    request_rows = list(manifest.get("replicates", []) or [])
    if not request_rows:
        raise ValueError("replicated aggregate request requires at least one replicate")
    require_strict_lineage = bool(manifest.get("require_strict_lineage", True))
    formal_aggregation = bool(manifest.get("formal_aggregation", True))
    minimum_training_seed_count = int(
        manifest.get(
            "minimum_training_seed_count",
            MIN_FORMAL_TRAINING_SEED_COUNT if formal_aggregation else 1,
        )
    )
    if minimum_training_seed_count <= 0:
        raise ValueError("minimum_training_seed_count must be positive")
    if formal_aggregation and not require_strict_lineage:
        raise ValueError("formal replicated aggregation requires strict lineage")
    if formal_aggregation and minimum_training_seed_count < MIN_FORMAL_TRAINING_SEED_COUNT:
        raise ValueError(
            "formal replicated aggregation cannot lower the five-training-seed minimum"
        )
    if len(request_rows) < minimum_training_seed_count:
        raise ValueError(
            "replicated aggregation has too few optimizer training seeds: "
            f"found={len(request_rows)} required={minimum_training_seed_count}"
        )
    candidate_binding = (
        _formal_candidate_binding(manifest, request_path=manifest_path)
        if formal_aggregation
        else {
            "required": False,
            "candidate_side": None,
            "source_acceptance_key": None,
            "formal_runtime_contract_sha256": None,
        }
    )
    candidate_side = (
        str(candidate_binding["candidate_side"])
        if formal_aggregation
        else None
    )
    source_acceptance_key = (
        str(candidate_binding["source_acceptance_key"])
        if formal_aggregation
        else None
    )

    replicate_pairs: list[dict[str, Any]] = []
    lineage_replicates: list[dict[str, Any]] = []
    source_records: dict[str, dict[str, Any]] = {}
    protocol_ids: set[str] = set()
    safety_metric_versions: set[str] = set()
    for position, request in enumerate(request_rows):
        if not isinstance(request, Mapping):
            raise ValueError(f"replicate request {position} must be an object")
        if request.get("training_seed") is None:
            raise ValueError(f"replicate request {position} requires training_seed")
        training_seed = int(request["training_seed"])
        left_group_name = str(request.get("left_group", "")).strip()
        right_group_name = str(request.get("right_group", "")).strip()
        if not left_group_name or not right_group_name:
            raise ValueError(
                f"replicate request {position} requires explicit left_group and right_group"
            )
        expected_left_sha256 = _explicit_sha256(
            request.get("left_checkpoint_sha256"),
            field=f"replicate request {position} left_checkpoint_sha256",
        )
        expected_right_sha256 = _explicit_sha256(
            request.get("right_checkpoint_sha256"),
            field=f"replicate request {position} right_checkpoint_sha256",
        )
        if not request.get("stage5_report"):
            raise ValueError(f"replicate request {position} requires stage5_report")
        source_path = _resolve(request["stage5_report"], base_dir=manifest_path.parent)
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        source = _read_json(source_path)
        left_group, right_group, source_metadata = _validate_source_report(
            source,
            source_path=source_path,
            left_group_name=left_group_name,
            right_group_name=right_group_name,
            training_seed=training_seed,
            expected_left_sha256=expected_left_sha256,
            expected_right_sha256=expected_right_sha256,
            require_strict_lineage=require_strict_lineage,
            source_acceptance_key=source_acceptance_key,
            candidate_binding=(candidate_binding if formal_aggregation else None),
        )
        if source_metadata["protocol_id"]:
            protocol_ids.add(str(source_metadata["protocol_id"]))
        if source_metadata["safety_metric_version"]:
            safety_metric_versions.add(str(source_metadata["safety_metric_version"]))
        source_sha256 = file_sha256(source_path)
        resolved_source = str(source_path.resolve())
        source_records[resolved_source] = {
            "path": resolved_source,
            "sha256": source_sha256,
            "lineage_fingerprint": str(source_metadata["lineage_fingerprint"]),
            "acceptance": source_metadata["acceptance"],
        }
        replicate_pairs.append(
            {
                "training_seed": training_seed,
                "left_checkpoint_sha256": expected_left_sha256,
                "right_checkpoint_sha256": expected_right_sha256,
                "left_report": left_group,
                "right_report": right_group,
            }
        )
        lineage_replicates.append(
            {
                "training_seed": training_seed,
                "source_report_sha256": source_sha256,
                "left_group": left_group_name,
                "right_group": right_group_name,
                "left_checkpoint_sha256": expected_left_sha256,
                "right_checkpoint_sha256": expected_right_sha256,
                "candidate_side": candidate_side,
                "candidate_group": (
                    left_group_name if candidate_side == "left" else right_group_name
                ) if formal_aggregation else None,
                "candidate_checkpoint_sha256": (
                    expected_left_sha256
                    if candidate_side == "left"
                    else expected_right_sha256
                ) if formal_aggregation else None,
                "source_acceptance": source_metadata["acceptance"],
                "candidate_group_bundle": source_metadata[
                    "candidate_group_bundle"
                ],
            }
        )

    if len(protocol_ids) > 1:
        raise ValueError(f"replicated source reports use different protocol IDs: {sorted(protocol_ids)}")
    if require_strict_lineage and len(protocol_ids) != 1:
        raise ValueError("strict replicated aggregation requires one shared non-empty protocol ID")
    if len(safety_metric_versions) > 1:
        raise ValueError(
            "replicated source reports use different safety metric versions: "
            f"{sorted(safety_metric_versions)}"
        )
    if require_strict_lineage and len(safety_metric_versions) != 1:
        raise ValueError(
            "strict replicated aggregation requires one shared non-empty safety metric version"
        )

    statistics_config = manifest.get("statistics", {}) or {}
    if not isinstance(statistics_config, Mapping):
        raise ValueError("replicated aggregate statistics configuration must be an object")
    if formal_aggregation:
        if not (
            "continuous_metrics" in statistics_config
            or "binary_metrics" in statistics_config
        ):
            raise ValueError(
                "formal replicated aggregation requires explicitly preregistered metric lists"
            )
        continuous_source = statistics_config.get("continuous_metrics", ())
        binary_source = statistics_config.get("binary_metrics", ())
    else:
        continuous_source = statistics_config.get(
            "continuous_metrics", DEFAULT_CONTINUOUS_METRICS
        )
        binary_source = statistics_config.get("binary_metrics", DEFAULT_BINARY_METRICS)
    if isinstance(continuous_source, (str, bytes)) or isinstance(
        binary_source, (str, bytes)
    ):
        raise ValueError("replicated aggregate metric lists must be arrays")
    continuous_metrics = tuple(str(value).strip() for value in continuous_source)
    binary_metrics = tuple(str(value).strip() for value in binary_source)
    if any(not value for value in continuous_metrics + binary_metrics):
        raise ValueError("replicated aggregate metric names must be non-empty")
    if formal_aggregation:
        if not continuous_metrics and not binary_metrics:
            raise ValueError(
                "formal replicated aggregation requires at least one preregistered metric"
            )
    statistics = build_replicated_pair_statistics(
        replicate_pairs,
        continuous_metrics=continuous_metrics,
        binary_metrics=binary_metrics,
        confidence=float(statistics_config.get("confidence", 0.95)),
        replicates=int(statistics_config.get("bootstrap_replicates", 10_000)),
        seed=int(statistics_config.get("bootstrap_seed", 0)),
        require_distinct_checkpoints=bool(
            statistics_config.get("require_distinct_checkpoints", True)
        ),
    )
    produced_continuous = sorted(str(value) for value in statistics["continuous"])
    produced_binary = sorted(str(value) for value in statistics["binary"])
    if formal_aggregation and not produced_continuous and not produced_binary:
        raise ValueError(
            "formal replicated aggregation produced no preregistered continuous or binary metrics"
        )

    lineage_replicates.sort(key=lambda row: int(row["training_seed"]))
    candidate_checkpoint_records = [
        {
            "training_seed": int(row["training_seed"]),
            "group": str(row["candidate_group"]),
            "checkpoint_sha256": str(row["candidate_checkpoint_sha256"]),
        }
        for row in lineage_replicates
        if formal_aggregation
    ]
    if formal_aggregation:
        candidate_binding = {
            **candidate_binding,
            "candidate_checkpoint_records": candidate_checkpoint_records,
            "candidate_checkpoint_matrix_sha256": stable_hash(
                candidate_checkpoint_records
            ),
        }
    source_report_rows = [source_records[key] for key in sorted(source_records)]
    lineage_content = {
        "request_manifest_sha256": file_sha256(manifest_path),
        "formal_aggregation": formal_aggregation,
        "minimum_training_seed_count": minimum_training_seed_count,
        "protocol_id": next(iter(protocol_ids), ""),
        "safety_metric_version": next(iter(safety_metric_versions), ""),
        "source_reports": [
            {
                "sha256": row["sha256"],
                "lineage_fingerprint": row["lineage_fingerprint"],
            }
            for row in source_report_rows
        ],
        "replicates": lineage_replicates,
        "statistics_design_sha256": statistics["design_sha256"],
        "candidate_binding": candidate_binding,
        "produced_continuous_metrics": produced_continuous,
        "produced_binary_metrics": produced_binary,
    }
    lineage = {
        "source_manifest": {
            "path": str(manifest_path.resolve()),
            "sha256": lineage_content["request_manifest_sha256"],
        },
        "protocol_id": lineage_content["protocol_id"],
        "safety_metric_version": lineage_content["safety_metric_version"],
        "source_reports": source_report_rows,
        "replicates": lineage_replicates,
        "lineage_fingerprint": stable_hash(lineage_content),
    }
    gate_checks = {
        "formal_seed_minimum_satisfied": (
            not formal_aggregation
            or len(request_rows) >= MIN_FORMAL_TRAINING_SEED_COUNT
        ),
        "strict_source_lineage_satisfied": (
            not formal_aggregation or require_strict_lineage
        ),
        "candidate_bundle_binding_satisfied": (
            not formal_aggregation or bool(candidate_binding.get("required", False))
        ),
        "candidate_side_bound": (
            not formal_aggregation
            or candidate_binding.get("candidate_side") in {"left", "right"}
        ),
        "candidate_groups_used_bound_bundle": (
            not formal_aggregation
            or all(
                isinstance(row.get("candidate_group_bundle"), Mapping)
                and row["candidate_group_bundle"].get("manifest_sha256")
                == candidate_binding["candidate_manifest"]["sha256"]
                for row in lineage_replicates
            )
        ),
        "formal_runtime_contract_bound": (
            not formal_aggregation
            or bool(candidate_binding.get("formal_runtime_contract_sha256"))
        ),
        "all_source_acceptance_passed": (
            not formal_aggregation
            or all(
                isinstance(row.get("acceptance"), Mapping)
                and row["acceptance"].get("available") is True
                and row["acceptance"].get("regression") is False
                for row in source_report_rows
            )
        ),
        "preregistered_metric_output_nonempty": (
            not formal_aggregation
            or bool(produced_continuous or produced_binary)
        ),
        "balanced_crossed_statistics": bool(statistics.get("balanced_matrix", False)),
    }
    gate = {
        "profile": (
            "formal_candidate_promotion_binding_v1"
            if formal_aggregation
            else "development_replicated_statistics_v1"
        ),
        "pass": all(bool(value) for value in gate_checks.values()),
        "checks": gate_checks,
    }
    if not gate["pass"]:
        failed = sorted(key for key, value in gate_checks.items() if not value)
        raise ValueError(f"replicated Stage5 aggregate gate failed: {failed}")
    report: dict[str, Any] = {
        "artifact_kind": REPORT_ARTIFACT_KIND,
        "schema_version": REPORT_SCHEMA_VERSION,
        "comparison_id": comparison_id,
        "formal_aggregation": formal_aggregation,
        "minimum_training_seed_count": minimum_training_seed_count,
        "statistics": statistics,
        "lineage": lineage,
        "candidate_binding": candidate_binding,
        "gate": gate,
    }
    report["report_fingerprint"] = stable_hash(report)
    return report


def run(manifest_path: str | Path, output_path: str | Path) -> Path:
    output = Path(output_path)
    if output.exists():
        raise FileExistsError(f"replicated Stage5 report already exists: {output}")
    report = aggregate_manifest(manifest_path)
    write_report(output, report)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate replicated Stage5 optimizer-seed comparisons with crossed bootstrap"
    )
    parser.add_argument("--manifest", required=True, help="Replicated aggregate request JSON")
    parser.add_argument("--output", required=True, help="Output JSON report")
    args = parser.parse_args()
    output = run(args.manifest, args.output)
    print(f"stage5_replicated_aggregate report={output}")


if __name__ == "__main__":
    main()
