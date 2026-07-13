from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from safe_rl.evaluation_protocol import file_sha256, stable_hash


REVOCATION_SCHEMA_VERSION = 1


def build_artifact_revocation_manifest(
    artifacts: Iterable[str | Path],
    *,
    reason: str,
    protocol_id: str = "",
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    rows = []
    for source in artifacts:
        path = Path(source)
        rows.append(
            {
                "path": str(path.resolve()),
                "match": "exact",
                "exists_at_revocation": path.exists(),
                "sha256": file_sha256(path) if path.is_file() else None,
                "status": "superseded",
                "deployable": False,
                "allowed_usage": "diagnostic_only",
            }
        )
    manifest = {
        "schema_version": REVOCATION_SCHEMA_VERSION,
        "artifact_kind": "safe_rl_artifact_revocation_manifest_v1",
        "revoked_at": datetime.now(timezone.utc).isoformat(),
        "protocol_id": str(protocol_id),
        "reason": str(reason),
        "status": "superseded",
        "deployable": False,
        "allowed_usage": "diagnostic_only",
        "artifacts": rows,
        "metadata": dict(metadata or {}),
    }
    manifest["revocation_fingerprint"] = stable_hash(manifest)
    return manifest


def write_artifact_revocation_manifest(
    output_path: str | Path,
    artifacts: Iterable[str | Path],
    *,
    reason: str,
    protocol_id: str = "",
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = build_artifact_revocation_manifest(
        artifacts,
        reason=reason,
        protocol_id=protocol_id,
        metadata=metadata,
    )
    with output.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return output


def revoked_hashes(manifest: Mapping[str, Any]) -> set[str]:
    return {
        str(row["sha256"])
        for row in manifest.get("artifacts", [])
        if row.get("sha256")
    }


def assert_artifact_not_revoked(
    artifact_path: str | Path,
    revocation_manifest: str | Path,
) -> None:
    artifact = Path(artifact_path)
    with Path(revocation_manifest).open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if int(manifest.get("schema_version", -1)) != REVOCATION_SCHEMA_VERSION:
        raise ValueError("unsupported artifact revocation manifest")
    digest = file_sha256(artifact)
    path_base = Path(str(manifest.get("path_base", ".")))
    if not path_base.is_absolute():
        path_base = Path(revocation_manifest).parent / path_base
    resolved = artifact.resolve()
    path_match = False
    for row in manifest.get("artifacts", []):
        source = row.get("path")
        if not source:
            continue
        revoked = Path(str(source))
        revoked = revoked if revoked.is_absolute() else path_base / revoked
        revoked = revoked.resolve()
        if str(row.get("match", "exact")) == "path_prefix":
            try:
                resolved.relative_to(revoked)
            except ValueError:
                continue
            path_match = True
            break
        if resolved == revoked:
            path_match = True
            break
    if digest in revoked_hashes(manifest) or path_match:
        raise ValueError(
            f"artifact has been revoked: path={artifact} sha256={digest} "
            f"reason={manifest.get('reason', '')}"
        )
