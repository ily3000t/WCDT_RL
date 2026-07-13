from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


EVIDENCE_PROTOCOL_SCHEMA_VERSION = 1


class EvidenceProtocolError(ValueError):
    """Raised when an evidence run violates its frozen evaluation protocol."""


class SeedCohortOverlapError(EvidenceProtocolError):
    """Raised when two evidence cohorts contain the same simulator seed."""


class HoldoutAlreadyOpenedError(EvidenceProtocolError):
    """Raised when a sealed final holdout is opened more than once."""


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(_plain(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def stable_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _protocol_config(config: Any) -> dict[str, Any]:
    if config is None:
        return {}
    getter = getattr(config, "get", None)
    if not callable(getter):
        return {}
    raw = getter("evaluation_protocol", {}) or {}
    return dict(_plain(raw))


def _resolve(path: str | Path, *, base_dir: str | Path | None = None) -> Path:
    value = Path(path)
    if value.is_absolute() or base_dir is None:
        return value
    return Path(base_dir) / value


def _cohort_seeds(value: Any) -> list[int]:
    if isinstance(value, Mapping):
        if value.get("seeds") is not None and value.get("values") is not None:
            raise EvidenceProtocolError(
                "seed cohort must use either 'seeds' or legacy-compatible 'values', not both"
            )
        explicit = list(value.get("seeds", value.get("values", [])) or [])
        ranges = value.get("ranges", []) or []
        if not isinstance(ranges, (list, tuple)):
            raise EvidenceProtocolError("seed cohort 'ranges' must be a list")
        expanded = list(explicit)
        for item in ranges:
            if not isinstance(item, Mapping):
                raise EvidenceProtocolError("seed cohort range must be a mapping")
            start = int(item.get("start"))
            count = int(item.get("count", 0))
            if count <= 0:
                raise EvidenceProtocolError("seed cohort range count must be positive")
            expanded.extend(range(start, start + count))
        value = expanded
    if value is None:
        return []
    seeds = [int(seed) for seed in value]
    if len(seeds) != len(set(seeds)):
        duplicates = sorted(seed for seed in set(seeds) if seeds.count(seed) > 1)
        raise EvidenceProtocolError(f"seed cohort contains duplicate seeds: {duplicates}")
    return seeds


def normalise_seed_cohorts(payload: Mapping[str, Any]) -> dict[str, list[int]]:
    raw = payload.get("cohorts", payload.get("seed_cohorts", {}))
    if not isinstance(raw, Mapping):
        raise EvidenceProtocolError("seed ledger must contain a 'cohorts' mapping")
    return {str(name): _cohort_seeds(value) for name, value in raw.items()}


def audit_seed_cohorts(cohorts: Mapping[str, Iterable[int]]) -> dict[str, Any]:
    normalised = {str(name): [int(seed) for seed in seeds] for name, seeds in cohorts.items()}
    duplicate_counts = {
        name: len(seeds) - len(set(seeds))
        for name, seeds in normalised.items()
        if len(seeds) != len(set(seeds))
    }
    owners: dict[int, str] = {}
    overlaps: list[dict[str, Any]] = []
    for name in sorted(normalised):
        for seed in sorted(normalised[name]):
            previous = owners.get(seed)
            if previous is not None and previous != name:
                overlaps.append({"seed": seed, "left": previous, "right": name})
            else:
                owners[seed] = name
    return {
        "cohort_counts": {name: len(seeds) for name, seeds in sorted(normalised.items())},
        "cohort_hashes": {name: stable_hash(sorted(seeds)) for name, seeds in sorted(normalised.items())},
        "total_unique_seeds": len(owners),
        "overlap_count": len(overlaps),
        "overlaps": overlaps,
        "duplicate_count": int(sum(duplicate_counts.values())),
        "duplicate_counts": duplicate_counts,
    }


def validate_seed_cohorts(cohorts: Mapping[str, Iterable[int]]) -> dict[str, Any]:
    audit = audit_seed_cohorts(cohorts)
    if audit["duplicate_count"]:
        raise EvidenceProtocolError(
            f"seed cohorts contain duplicate seeds: {audit['duplicate_counts']}"
        )
    if audit["overlap_count"]:
        raise SeedCohortOverlapError(f"seed cohorts overlap: {audit['overlaps'][:20]}")
    return audit


def _load_revocation_manifest(path: Path, *, protocol_id: str = "") -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if int(payload.get("schema_version", -1)) != 1:
        raise EvidenceProtocolError(
            f"unsupported artifact revocation schema={payload.get('schema_version')!r}"
        )
    artifacts = payload.get("artifacts", [])
    if not isinstance(artifacts, list):
        raise EvidenceProtocolError("artifact revocation manifest 'artifacts' must be a list")
    manifest_protocol_id = str(payload.get("protocol_id", ""))
    if protocol_id and manifest_protocol_id and protocol_id != manifest_protocol_id:
        raise EvidenceProtocolError(
            "artifact revocation protocol_id does not match evaluation protocol: "
            f"manifest={manifest_protocol_id!r} protocol={protocol_id!r}"
        )
    for index, row in enumerate(artifacts):
        if not isinstance(row, Mapping):
            raise EvidenceProtocolError(f"artifact revocation row {index} must be a mapping")
        if not row.get("sha256") and not row.get("path"):
            raise EvidenceProtocolError(f"artifact revocation row {index} needs sha256 or path")
        if str(row.get("match", "exact")) not in {"exact", "path_prefix"}:
            raise EvidenceProtocolError(f"artifact revocation row {index} has invalid match mode")
        if bool(row.get("deployable", False)):
            raise EvidenceProtocolError(f"artifact revocation row {index} cannot be deployable")
    return payload


def _revocation_path_base(manifest: Mapping[str, Any], manifest_path: Path) -> Path:
    source = manifest.get("path_base")
    if not source:
        return manifest_path.parent
    value = Path(str(source))
    return value if value.is_absolute() else manifest_path.parent / value


def _is_revoked_artifact(
    path: Path,
    digest: str,
    manifest: Mapping[str, Any],
    manifest_path: Path,
) -> bool:
    candidate = path.resolve()
    base = _revocation_path_base(manifest, manifest_path).resolve()
    for row in manifest.get("artifacts", []):
        if row.get("sha256") and str(row["sha256"]) == digest:
            return True
        source = row.get("path")
        if not source:
            continue
        revoked = Path(str(source))
        revoked = revoked if revoked.is_absolute() else base / revoked
        revoked = revoked.resolve()
        if str(row.get("match", "exact")) == "exact" and candidate == revoked:
            return True
        if str(row.get("match", "exact")) == "path_prefix":
            try:
                candidate.relative_to(revoked)
            except ValueError:
                continue
            return True
    return False


def load_seed_ledger(
    config: Any,
    *,
    base_dir: str | Path | None = None,
) -> tuple[dict[str, Any] | None, Path | None, str | None]:
    protocol = _protocol_config(config)
    source = protocol.get("seed_ledger")
    if source is None:
        inline = protocol.get("seed_cohorts") or protocol.get("cohorts")
        if inline is None:
            if bool(protocol.get("strict", False)):
                raise EvidenceProtocolError("strict evaluation protocol requires a seed ledger")
            return None, None, None
        payload = {
            "schema_version": EVIDENCE_PROTOCOL_SCHEMA_VERSION,
            "protocol_id": protocol.get("protocol_id", ""),
            "cohorts": inline,
        }
        normalised = normalise_seed_cohorts(payload)
        validate_seed_cohorts(normalised)
        payload["cohorts"] = normalised
        return payload, None, stable_hash(payload)
    path = _resolve(source, base_dir=base_dir)
    if not path.exists():
        raise FileNotFoundError(f"seed ledger not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if int(payload.get("schema_version", -1)) != EVIDENCE_PROTOCOL_SCHEMA_VERSION:
        raise EvidenceProtocolError(
            f"unsupported seed ledger schema={payload.get('schema_version')}; "
            f"expected={EVIDENCE_PROTOCOL_SCHEMA_VERSION}"
        )
    normalised = normalise_seed_cohorts(payload)
    validate_seed_cohorts(normalised)
    payload["cohorts"] = normalised
    return payload, path, file_sha256(path)


def _role_mapping(protocol: Mapping[str, Any]) -> dict[str, str]:
    raw = protocol.get("cohort_roles", {}) or {}
    if not isinstance(raw, Mapping):
        raise EvidenceProtocolError("evaluation_protocol.cohort_roles must be a mapping")
    return {str(role): str(cohort) for role, cohort in raw.items()}


def protocol_snapshot(config: Any, *, base_dir: str | Path | None = None) -> dict[str, Any]:
    protocol = _protocol_config(config)
    ledger, ledger_path, ledger_hash = load_seed_ledger(config, base_dir=base_dir)
    protocol_id = str(protocol.get("protocol_id", (ledger or {}).get("protocol_id", "")))
    if ledger is not None:
        ledger_protocol_id = str(ledger.get("protocol_id", ""))
        if protocol_id and ledger_protocol_id and protocol_id != ledger_protocol_id:
            raise EvidenceProtocolError(
                f"protocol_id mismatch: config={protocol_id!r} ledger={ledger_protocol_id!r}"
            )
        protocol_id = protocol_id or ledger_protocol_id
    enabled = bool(protocol or ledger)
    if enabled and bool(protocol.get("strict", False)) and not protocol_id:
        raise EvidenceProtocolError("strict evaluation protocol requires protocol_id")
    cohorts = {} if ledger is None else normalise_seed_cohorts(ledger)
    audit = validate_seed_cohorts(cohorts) if cohorts else {
        "cohort_counts": {},
        "cohort_hashes": {},
        "total_unique_seeds": 0,
        "overlap_count": 0,
    }
    revocation_source = protocol.get("revocation_manifest")
    revocation_path = None if not revocation_source else _resolve(revocation_source, base_dir=base_dir)
    if revocation_path is not None and not revocation_path.exists():
        raise FileNotFoundError(f"artifact revocation manifest not found: {revocation_path}")
    if revocation_path is not None:
        _load_revocation_manifest(revocation_path, protocol_id=protocol_id)
    return {
        "schema_version": EVIDENCE_PROTOCOL_SCHEMA_VERSION,
        "enabled": enabled,
        "strict": bool(protocol.get("strict", False)),
        "protocol_id": protocol_id,
        "seed_ledger_path": None if ledger_path is None else str(ledger_path.resolve()),
        "seed_ledger_sha256": ledger_hash,
        "cohort_roles": _role_mapping(protocol),
        "cohorts": cohorts,
        "seed_audit": audit,
        "revocation_manifest_path": None if revocation_path is None else str(revocation_path.resolve()),
        "revocation_manifest_sha256": None if revocation_path is None else file_sha256(revocation_path),
    }


def seeds_for_role(
    config: Any,
    role: str,
    *,
    fallback: Iterable[int] | None = None,
    base_dir: str | Path | None = None,
) -> list[int]:
    snapshot = protocol_snapshot(config, base_dir=base_dir)
    cohort_name = snapshot["cohort_roles"].get(str(role))
    if cohort_name is None:
        if snapshot["strict"]:
            raise EvidenceProtocolError(f"strict evaluation protocol has no cohort mapping for role '{role}'")
        return [int(seed) for seed in (fallback or [])]
    if cohort_name not in snapshot["cohorts"]:
        raise EvidenceProtocolError(f"seed ledger is missing cohort '{cohort_name}' for role '{role}'")
    return list(snapshot["cohorts"][cohort_name])


def assert_disjoint_seed_usage(**cohorts: Iterable[int]) -> dict[str, Any]:
    return validate_seed_cohorts({name: list(values) for name, values in cohorts.items()})


def build_stage_lineage(
    config: Any,
    *,
    stage: str,
    role_seeds: Mapping[str, Iterable[int]],
    artifact_paths: Mapping[str, str | Path | None] | None = None,
    base_dir: str | Path | None = None,
) -> dict[str, Any]:
    snapshot = protocol_snapshot(config, base_dir=base_dir)
    usages: dict[str, dict[str, Any]] = {}
    local_cohorts: dict[str, list[int]] = {}
    for role, values in role_seeds.items():
        seeds = [int(seed) for seed in values]
        if len(seeds) != len(set(seeds)):
            raise EvidenceProtocolError(f"role '{role}' contains duplicate seeds")
        local_cohorts[str(role)] = seeds
        cohort_name = snapshot["cohort_roles"].get(str(role))
        if cohort_name is not None:
            expected = set(snapshot["cohorts"].get(cohort_name, []))
            unexpected = sorted(set(seeds) - expected)
            if unexpected:
                raise EvidenceProtocolError(
                    f"role '{role}' used seeds outside ledger cohort '{cohort_name}': {unexpected[:20]}"
                )
        elif snapshot["strict"]:
            raise EvidenceProtocolError(f"strict evaluation protocol has no cohort mapping for role '{role}'")
        usages[str(role)] = {
            "cohort": cohort_name,
            "count": len(seeds),
            "seed_sha256": stable_hash(sorted(seeds)),
            "seeds": seeds,
        }
    if snapshot["enabled"]:
        local_audit = validate_seed_cohorts(local_cohorts)
        local_audit["enforced"] = True
    else:
        # Historical configs intentionally reused seeds across training,
        # checkpoint selection and Stage5. Preserve their diagnostic workflows
        # while recording (but not legitimising) the overlap.
        local_audit = audit_seed_cohorts(local_cohorts)
        local_audit["enforced"] = False
    artifacts: dict[str, dict[str, Any]] = {}
    revocation: dict[str, Any] | None = None
    revocation_manifest_path: Path | None = None
    revocation_path = snapshot.get("revocation_manifest_path")
    if revocation_path:
        revocation_manifest_path = Path(revocation_path)
        revocation = _load_revocation_manifest(
            revocation_manifest_path,
            protocol_id=str(snapshot.get("protocol_id", "")),
        )
    for name, source in (artifact_paths or {}).items():
        if not source:
            continue
        path = _resolve(source, base_dir=base_dir)
        if not path.exists():
            if snapshot["strict"]:
                raise FileNotFoundError(f"lineage artifact not found for {name}: {path}")
            artifacts[str(name)] = {"path": str(path), "exists": False}
            continue
        digest = file_sha256(path)
        if (
            revocation is not None
            and revocation_manifest_path is not None
            and _is_revoked_artifact(path, digest, revocation, revocation_manifest_path)
        ):
            raise EvidenceProtocolError(f"lineage artifact '{name}' has been revoked: {path}")
        artifacts[str(name)] = {
            "path": str(path.resolve()),
            "exists": True,
            "sha256": digest,
        }
    lineage = {
        "schema_version": EVIDENCE_PROTOCOL_SCHEMA_VERSION,
        "stage": str(stage),
        "protocol_id": snapshot["protocol_id"],
        "protocol_enabled": snapshot["enabled"],
        "protocol_strict": snapshot["strict"],
        "seed_ledger_path": snapshot["seed_ledger_path"],
        "seed_ledger_sha256": snapshot["seed_ledger_sha256"],
        "revocation_manifest_path": snapshot["revocation_manifest_path"],
        "revocation_manifest_sha256": snapshot["revocation_manifest_sha256"],
        "role_usage": usages,
        "local_seed_audit": local_audit,
        "artifacts": artifacts,
    }
    lineage["lineage_fingerprint"] = stable_hash(lineage)
    return lineage


def validate_parent_lineage(parent: Mapping[str, Any], current: Mapping[str, Any]) -> None:
    if bool(current.get("protocol_strict", False)) and not parent:
        raise EvidenceProtocolError("strict Stage5 protocol requires Stage3 evidence lineage")
    if not parent:
        return
    if bool(parent.get("protocol_enabled", False)) and not bool(
        current.get("protocol_enabled", False)
    ):
        raise EvidenceProtocolError(
            "protocol-enabled Stage3 lineage cannot be evaluated with the protocol disabled"
        )
    for key in ("protocol_id", "seed_ledger_sha256"):
        left = parent.get(key)
        right = current.get(key)
        if right and left != right:
            raise EvidenceProtocolError(f"parent lineage mismatch for {key}: parent={left!r} current={right!r}")


def claim_final_holdout(
    seal_path: str | Path,
    *,
    protocol_id: str,
    artifact_manifest: str | Path,
    split_manifest: str | Path,
    metadata: Mapping[str, Any] | None = None,
    frozen_artifacts: Mapping[str, str | Path] | None = None,
    expected_sha256: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Atomically claim a sealed holdout before any test rows are loaded."""

    seal = Path(seal_path)
    seal.parent.mkdir(parents=True, exist_ok=True)
    artifact = Path(artifact_manifest)
    split = Path(split_manifest)
    frozen: dict[str, dict[str, str]] = {}
    for name, source in (frozen_artifacts or {}).items():
        path = Path(source)
        frozen[str(name)] = {
            "path": str(path.resolve()),
            "sha256": file_sha256(path),
        }
    for name, expected in (expected_sha256 or {}).items():
        actual = frozen.get(str(name), {}).get("sha256")
        if actual != str(expected):
            raise EvidenceProtocolError(
                f"frozen holdout input changed before claim: {name} expected={expected} actual={actual}"
            )
    payload = {
        "schema_version": EVIDENCE_PROTOCOL_SCHEMA_VERSION,
        "state": "opened",
        "opened_at": datetime.now(timezone.utc).isoformat(),
        "protocol_id": str(protocol_id),
        "artifact_manifest": str(artifact.resolve()),
        "artifact_manifest_sha256": file_sha256(artifact),
        "split_manifest": str(split.resolve()),
        "split_manifest_sha256": file_sha256(split),
        "frozen_artifacts": frozen,
        "metadata": dict(_plain(metadata or {})),
    }
    payload["claim_fingerprint"] = stable_hash(payload)
    try:
        descriptor = os.open(str(seal), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise HoldoutAlreadyOpenedError(f"final holdout has already been opened: {seal}") from exc
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    return payload


def finalise_holdout_claim(
    seal_path: str | Path,
    *,
    result_path: str | Path,
    decision: str,
    expected_claim_fingerprint: str | None = None,
) -> dict[str, Any]:
    seal = Path(seal_path)
    def _read_open_claim() -> dict[str, Any]:
        with seal.open("r", encoding="utf-8") as handle:
            current = json.load(handle)
        if str(current.get("state", "")) != "opened":
            raise HoldoutAlreadyOpenedError(
                f"final holdout claim is not open: state={current.get('state')!r}"
            )
        if (
            expected_claim_fingerprint is not None
            and str(current.get("claim_fingerprint", "")) != str(expected_claim_fingerprint)
        ):
            raise EvidenceProtocolError("final holdout claim fingerprint mismatch")
        return current

    _read_open_claim()
    lock = seal.with_suffix(seal.suffix + ".finalise.lock")
    try:
        descriptor = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise HoldoutAlreadyOpenedError(f"final holdout is already being finalised: {seal}") from exc
    os.close(descriptor)
    completed = False
    payload = _read_open_claim()
    result = Path(result_path)
    payload.update(
        {
            "state": "evaluated",
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
            "decision": str(decision),
            "result_path": str(result.resolve()),
            "result_sha256": file_sha256(result),
        }
    )
    payload["final_fingerprint"] = stable_hash(payload)
    temporary = seal.with_suffix(seal.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, seal)
    completed = True
    if completed:
        try:
            lock.unlink()
        except FileNotFoundError:
            pass
    return payload
