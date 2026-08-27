from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import yaml

from safe_rl.accvp.contracts.schema import file_sha256
from safe_rl.evaluation_protocol import stable_hash
from safe_rl.ppo_replicates import plain
from safe_rl.utils.config import (
    DEFAULT_CONFIG_PATH,
    REPO_ROOT,
    STANDALONE_CONFIG_KEY,
    _deep_merge,
    _resolve_extended_config,
    load_config,
)


MANIFEST_KIND = "safe_rl_config_fix_bundle_v1"


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (REPO_ROOT / value).resolve()


def _portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, Mapping):
        raise ValueError(f"configuration must be a mapping: {path}")
    return dict(payload)


def standalone_payload(source: str | Path) -> dict[str, Any]:
    """Freeze defaults plus every ``extends`` layer into one YAML payload."""

    source_path = _resolve(source)
    merged = _deep_merge(
        _read_yaml(DEFAULT_CONFIG_PATH),
        _resolve_extended_config(source_path),
    )
    return {STANDALONE_CONFIG_KEY: True, **merged}


def _write_yaml_new(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            plain(payload),
            handle,
            sort_keys=False,
            allow_unicode=True,
        )
    temporary.replace(path)


def export_bundle(
    entries: Mapping[str, str | Path],
    *,
    output_dir: str | Path,
) -> Path:
    output = _resolve(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"config-fix output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for name, source in entries.items():
        if not str(name).endswith((".yaml", ".yml")):
            raise ValueError(f"snapshot entry must be YAML: {name}")
        source_path = _resolve(source)
        target = output / str(name)
        _write_yaml_new(target, standalone_payload(source_path))
        source_resolved = plain(load_config(source_path))
        target_resolved = plain(load_config(target))
        if target_resolved != source_resolved:
            raise ValueError(f"standalone snapshot changed resolved parameters: {name}")
        records.append(
            {
                "name": str(name),
                "source": _portable_path(source_path),
                "source_sha256": file_sha256(source_path),
                "snapshot": _portable_path(target),
                "snapshot_sha256": file_sha256(target),
                "resolved_config_sha256": stable_hash(source_resolved),
            }
        )
    manifest: dict[str, Any] = {
        "artifact_kind": MANIFEST_KIND,
        "schema_version": 1,
        "mode": "standalone_exact_snapshot",
        "default_config": _portable_path(DEFAULT_CONFIG_PATH),
        "default_config_sha256": file_sha256(DEFAULT_CONFIG_PATH),
        "entries": records,
    }
    manifest["manifest_fingerprint"] = stable_hash(manifest)
    manifest_path = output / "equivalence_manifest.json"
    with manifest_path.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return manifest_path


def _entry(value: str) -> tuple[str, str]:
    name, separator, source = value.partition("=")
    if not separator or not name.strip() or not source.strip():
        raise argparse.ArgumentTypeError("--entry must use NAME=SOURCE")
    return name.strip(), source.strip()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export exact standalone snapshots for a config_fix bundle"
    )
    parser.add_argument("--entry", action="append", type=_entry, required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    entries = dict(args.entry)
    if len(entries) != len(args.entry):
        raise ValueError("config-fix snapshot names must be unique")
    result = export_bundle(entries, output_dir=args.output_dir)
    print(f"config_fix_equivalence_manifest={result}")


if __name__ == "__main__":
    main()
