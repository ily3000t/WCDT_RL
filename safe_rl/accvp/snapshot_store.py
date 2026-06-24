from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from safe_rl.accvp.root_context import RootContext, write_root_context
from safe_rl.accvp.schema import canonical_json, validate_branch_row


class CounterfactualSnapshotStore:
    """Durable root/branch store with snapshot deletion only after complete roots."""

    def __init__(self, output_dir: str | Path, *, cache_dir: str | Path | None = None):
        self.output_dir = Path(output_dir)
        # SUMO state files are temporary worker cache, not dataset artifacts.
        # Keep them out of the buffer directory so completed datasets remain
        # immutable and cache cleanup never risks deleting branch records.
        # Direct callers get the same separation guarantee as the collector:
        # cache files are siblings of, never children of, durable datasets.
        self.cache_dir = (
            Path(cache_dir)
            if cache_dir is not None
            else self.output_dir.parent / ".cache" / self.output_dir.name
        )
        self.snapshots_dir = self.cache_dir / "snapshots"
        self.roots_dir = self.output_dir / "roots"
        self.branches_dir = self.output_dir / "branches"
        self.manifest_dir = self.output_dir / "manifests"
        for directory in (self.snapshots_dir, self.roots_dir, self.branches_dir, self.manifest_dir):
            directory.mkdir(parents=True, exist_ok=True)

    def snapshot_path(self, root_id: str) -> Path:
        return self.snapshots_dir / f"{root_id}.xml"

    def save_snapshot_from_root(self, env: Any, root_id: str) -> Path:
        """The only root-side SUMO operation: saveState. No loadState is permitted here."""

        final = self.snapshot_path(root_id)
        temporary = final.with_suffix(".xml.tmp")
        env._traci.simulation.saveState(str(temporary))
        temporary.replace(final)
        return final

    def write_root(self, root: RootContext, expected_action_ids: list[int]) -> tuple[Path, Path]:
        metadata_path, tensor_path = write_root_context(root, self.roots_dir)
        row = {
            "root_id": root.root_id,
            "metadata_path": str(metadata_path),
            "tensor_path": str(tensor_path),
            "snapshot_path": str(root.metadata["snapshot_path"]),
            "expected_action_ids": [int(value) for value in expected_action_ids],
            "branch_status": {str(value): "pending" for value in expected_action_ids},
            "complete": False,
        }
        self._write_root_manifest(root.root_id, row)
        return metadata_path, tensor_path

    def write_branch(self, row: dict[str, Any]) -> Path:
        validate_branch_row(row)
        path = self.branches_dir / f"{row['branch_id']}.json"
        temporary = path.with_suffix(".json.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(canonical_json(row))
        temporary.replace(path)
        manifest = self._load_root_manifest(str(row["root_id"]))
        manifest["branch_status"][str(int(row["action_id"]))] = "completed"
        self._write_root_manifest(str(row["root_id"]), manifest)
        return path

    def mark_branch_failed(self, root_id: str, action_id: int, reason: str) -> None:
        manifest = self._load_root_manifest(root_id)
        manifest["branch_status"][str(int(action_id))] = f"failed:{reason}"
        self._write_root_manifest(root_id, manifest)

    def finalise_root_if_complete(self, root_id: str) -> bool:
        manifest = self._load_root_manifest(root_id)
        expected = {str(value) for value in manifest["expected_action_ids"]}
        complete = expected == set(manifest["branch_status"]) and all(
            value == "completed" for value in manifest["branch_status"].values()
        )
        if not complete:
            return False
        manifest["complete"] = True
        self._write_root_manifest(root_id, manifest)
        snapshot = Path(manifest["snapshot_path"])
        if snapshot.exists():
            snapshot.unlink()
        return True

    def discard_incomplete_root(self, root_id: str) -> None:
        """Explicit cleanup for failed roots; never used to make a root look complete."""

        manifest = self._load_root_manifest(root_id)
        manifest["complete"] = False
        manifest["discarded"] = True
        self._write_root_manifest(root_id, manifest)

    def _manifest_path(self, root_id: str) -> Path:
        return self.manifest_dir / f"{root_id}.json"

    def _write_root_manifest(self, root_id: str, row: dict[str, Any]) -> None:
        path = self._manifest_path(root_id)
        temporary = path.with_suffix(".json.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(canonical_json(row))
        temporary.replace(path)

    def _load_root_manifest(self, root_id: str) -> dict[str, Any]:
        with self._manifest_path(root_id).open("r", encoding="utf-8") as handle:
            return json.load(handle)
