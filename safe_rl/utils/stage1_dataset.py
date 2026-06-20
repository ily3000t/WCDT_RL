from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import numpy as np


STAGE1_FORMAT_VERSION = "manifest_npy_v1"
STAGE1_BUFFER_SCHEMA_VERSION = 9


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _json_dump_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True, ensure_ascii=False)
        file.write("\n")
    os.replace(temporary, path)


def _promote_directory(temporary: Path, output: Path) -> None:
    if output.exists():
        if output.is_dir():
            shutil.rmtree(output)
        else:
            output.unlink()
    shutil.move(str(temporary), str(output))


class Stage1Dataset(Mapping[str, np.ndarray]):
    def __init__(
        self,
        path: Path,
        *,
        manifest: dict[str, Any] | None = None,
        legacy_npz: np.lib.npyio.NpzFile | None = None,
    ) -> None:
        self.path = path
        self.manifest = manifest or {}
        self.legacy_npz = legacy_npz
        self.legacy_npz_format = legacy_npz is not None
        self._cache: dict[str, np.ndarray] = {}

    @property
    def files(self) -> list[str]:
        if self.legacy_npz is not None:
            return list(self.legacy_npz.files)
        return list(self.manifest.get("arrays", {}).keys())

    def __getitem__(self, key: str) -> np.ndarray:
        if self.legacy_npz is not None:
            return self.legacy_npz[key]
        if key not in self._cache:
            entry = self.manifest["arrays"][key]
            array_path = self.path / str(entry["path"])
            self._cache[key] = np.load(array_path, mmap_mode="r", allow_pickle=False)
        return self._cache[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.files)

    def __len__(self) -> int:
        return len(self.files)

    def close(self) -> None:
        self._cache.clear()
        if self.legacy_npz is not None:
            self.legacy_npz.close()

    def __enter__(self) -> "Stage1Dataset":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()


def open_stage1_dataset(path: str | Path) -> Stage1Dataset:
    candidate = Path(path)
    if candidate.is_dir():
        manifest_path = candidate / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Stage1 dataset manifest not found: {manifest_path}")
        with manifest_path.open("r", encoding="utf-8") as file:
            manifest = json.load(file)
        if str(manifest.get("format_version")) != STAGE1_FORMAT_VERSION:
            raise ValueError(
                f"Unsupported Stage1 dataset format {manifest.get('format_version')!r}: {candidate}"
            )
        return Stage1Dataset(candidate, manifest=manifest)
    if not candidate.exists() and candidate.suffix.lower() != ".npz":
        legacy = candidate.with_suffix(".npz")
        if legacy.exists():
            candidate = legacy
    if candidate.suffix.lower() == ".npz" and candidate.exists():
        return Stage1Dataset(
            candidate,
            legacy_npz=np.load(candidate, allow_pickle=False),
        )
    raise FileNotFoundError(f"Stage1 dataset not found: {path}")


def _array_manifest_entry(root: Path, path: Path, array: np.ndarray) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "dtype": str(array.dtype),
        "shape": [int(item) for item in array.shape],
        "sha256": sha256_file(path),
    }


def _dataset_counts(arrays: Mapping[str, np.ndarray]) -> dict[str, int]:
    def count(key: str) -> int:
        value = arrays.get(key)
        return int(value.shape[0]) if value is not None and value.ndim > 0 else 0

    return {
        "transition_count": count("transition_episode_id"),
        "candidate_count": count("episode_id"),
        "trajectory_count": count("trajectory_episode_id"),
    }


def write_stage1_dataset(
    output: str | Path,
    arrays: Mapping[str, np.ndarray],
    *,
    metadata: Mapping[str, Any],
) -> Path:
    output_path = Path(output)
    temporary = output_path.with_name(f".{output_path.name}.tmp-{uuid.uuid4().hex}")
    arrays_dir = temporary / "arrays"
    arrays_dir.mkdir(parents=True, exist_ok=False)
    entries: dict[str, Any] = {}
    normalized: dict[str, np.ndarray] = {}
    try:
        for key in sorted(arrays):
            value = np.asarray(arrays[key])
            normalized[key] = value
            array_path = arrays_dir / f"{key}.npy"
            np.save(array_path, value, allow_pickle=False)
            entries[key] = _array_manifest_entry(temporary, array_path, value)
        manifest = {
            "format_version": STAGE1_FORMAT_VERSION,
            "stage1_buffer_schema_version": STAGE1_BUFFER_SCHEMA_VERSION,
            **dict(metadata),
            **_dataset_counts(normalized),
            "arrays": entries,
        }
        _json_dump_atomic(temporary / "manifest.json", manifest)
        _promote_directory(temporary, output_path)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output_path


def stage1_dataset_manifest_hash(path: str | Path) -> str:
    candidate = Path(path)
    if candidate.is_dir():
        return sha256_file(candidate / "manifest.json")
    return sha256_file(candidate)


def validate_stage1_dataset(
    path: str | Path,
    *,
    verify_hashes: bool = True,
) -> dict[str, Any]:
    candidate = Path(path)
    if not candidate.is_dir():
        raise ValueError(f"Stage1 manifest dataset must be a directory: {candidate}")
    manifest_path = candidate / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Stage1 dataset manifest not found: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as file:
        manifest = json.load(file)
    if str(manifest.get("format_version")) != STAGE1_FORMAT_VERSION:
        raise ValueError(
            f"Unsupported Stage1 dataset format {manifest.get('format_version')!r}: {candidate}"
        )
    arrays = manifest.get("arrays")
    if not isinstance(arrays, dict) or not arrays:
        raise ValueError(f"Stage1 dataset contains no array metadata: {manifest_path}")
    for key, entry in arrays.items():
        if not isinstance(entry, dict):
            raise ValueError(f"Invalid Stage1 array metadata for {key!r}")
        array_path = candidate / str(entry.get("path", ""))
        if not array_path.is_file():
            raise FileNotFoundError(f"Stage1 array is missing: {array_path}")
        array = np.load(array_path, mmap_mode="r", allow_pickle=False)
        try:
            expected_dtype = str(entry.get("dtype", ""))
            expected_shape = tuple(int(item) for item in entry.get("shape", []))
            if str(array.dtype) != expected_dtype:
                raise ValueError(
                    f"Stage1 array dtype changed for {key}: "
                    f"expected={expected_dtype}, actual={array.dtype}"
                )
            if tuple(array.shape) != expected_shape:
                raise ValueError(
                    f"Stage1 array shape changed for {key}: "
                    f"expected={expected_shape}, actual={tuple(array.shape)}"
                )
        finally:
            mmap_handle = getattr(array, "_mmap", None)
            if mmap_handle is not None:
                mmap_handle.close()
        if verify_hashes:
            expected_hash = str(entry.get("sha256", ""))
            actual_hash = sha256_file(array_path)
            if not expected_hash or actual_hash != expected_hash:
                raise ValueError(
                    f"Stage1 array hash changed for {key}: {array_path}"
                )
    return manifest


def _episode_slice(values: np.ndarray, episode_id: int) -> slice:
    left = int(np.searchsorted(values, episode_id, side="left"))
    right = int(np.searchsorted(values, episode_id, side="right"))
    return slice(left, right)


def merge_stage1_shards(
    shard_paths: list[str | Path],
    output: str | Path,
    *,
    transition_keys: set[str],
    candidate_keys: set[str],
    trajectory_keys: set[str],
    metadata: Mapping[str, Any],
) -> Path:
    if not shard_paths:
        raise ValueError("No Stage1 shards were provided")
    datasets = [open_stage1_dataset(path) for path in shard_paths]
    output_path = Path(output)
    temporary = output_path.with_name(f".{output_path.name}.tmp-{uuid.uuid4().hex}")
    arrays_dir = temporary / "arrays"
    arrays_dir.mkdir(parents=True, exist_ok=False)
    memmaps: dict[str, np.memmap] = {}
    entries: dict[str, Any] = {}
    try:
        all_keys = sorted(set().union(*(set(dataset.files) for dataset in datasets)))
        # Worker-local ID tables cannot be concatenated directly. Build one stable
        # table and remap trajectory row IDs while copying the shards.
        id_table_key = "trajectory_vehicle_id_table"
        id_index_key = "trajectory_agent_vehicle_id_index"
        global_vehicle_ids: list[str] = []
        local_id_maps: dict[int, np.ndarray] = {}
        if id_table_key in all_keys:
            global_vehicle_ids = sorted(
                {
                    str(value)
                    for dataset in datasets
                    if id_table_key in dataset
                    for value in np.asarray(dataset[id_table_key]).reshape(-1)
                    if str(value)
                }
            )
            global_lookup = {vehicle_id: index for index, vehicle_id in enumerate(global_vehicle_ids)}
            for dataset_index, dataset in enumerate(datasets):
                if id_table_key not in dataset:
                    continue
                local_values = np.asarray(dataset[id_table_key]).reshape(-1)
                local_id_maps[dataset_index] = np.asarray(
                    [global_lookup.get(str(value), -1) for value in local_values],
                    dtype=np.int32,
                )
        group_specs = (
            ("transition_episode_id", transition_keys, ("transition_episode_step",)),
            ("episode_id", candidate_keys, ("candidate_episode_step", "actions")),
            ("trajectory_episode_id", trajectory_keys, ("trajectory_window_end_step",)),
        )
        grouped_keys = set().union(*(keys for _, keys, _ in group_specs))
        scalar_keys = [
            key for key in all_keys if key not in grouped_keys and key != id_table_key
        ]

        for episode_key, keys, _sort_keys in group_specs:
            available = [dataset for dataset in datasets if episode_key in dataset]
            if not available:
                continue
            total = sum(int(dataset[episode_key].shape[0]) for dataset in available)
            for key in sorted(keys & set(all_keys)):
                source = next(
                    (
                        dataset[key]
                        for dataset in available
                        if key in dataset and dataset[key].ndim > 0
                    ),
                    None,
                )
                if source is None:
                    continue
                array_path = arrays_dir / f"{key}.npy"
                memmaps[key] = np.lib.format.open_memmap(
                    array_path,
                    mode="w+",
                    dtype=source.dtype,
                    shape=(total, *source.shape[1:]),
                )

        transition_lookup: dict[tuple[int, int], int] = {}
        offsets = {spec[0]: 0 for spec in group_specs}
        episode_ids = sorted(
            {
                int(episode)
                for dataset in datasets
                for episode_key, _keys, _sort in group_specs
                if episode_key in dataset
                for episode in np.unique(dataset[episode_key])
            }
        )
        for episode_id in episode_ids:
            for episode_key, keys, sort_keys in group_specs:
                rows: list[tuple[int, Stage1Dataset, np.ndarray]] = []
                for dataset_index, dataset in enumerate(datasets):
                    if episode_key not in dataset:
                        continue
                    episode_values = dataset[episode_key]
                    source_slice = _episode_slice(episode_values, episode_id)
                    if source_slice.start == source_slice.stop:
                        continue
                    local_count = int(source_slice.stop - source_slice.start)
                    order = np.arange(local_count, dtype=np.int64)
                    local_sort_values = [
                        np.asarray(dataset[key][source_slice])
                        for key in sort_keys
                        if key in dataset
                    ]
                    if local_sort_values:
                        order = np.lexsort(tuple(reversed(local_sort_values)))
                    rows.append((dataset_index, dataset, np.arange(source_slice.start, source_slice.stop)[order]))
                for dataset_index, dataset, indices in rows:
                    destination_start = offsets[episode_key]
                    destination_stop = destination_start + int(indices.shape[0])
                    for key in keys:
                        if key in memmaps and key in dataset:
                            values = dataset[key][indices]
                            if key == id_index_key:
                                local_map = local_id_maps.get(dataset_index)
                                if local_map is None:
                                    raise ValueError(
                                        "Trajectory ID indices require a worker-local vehicle ID table."
                                    )
                                remapped = np.full_like(values, -1, dtype=np.int32)
                                valid = np.asarray(values) >= 0
                                if np.any(valid):
                                    remapped[valid] = local_map[np.asarray(values)[valid]]
                                values = remapped
                            memmaps[key][destination_start:destination_stop] = values
                    if episode_key == "transition_episode_id":
                        steps = np.asarray(dataset["transition_episode_step"][indices], dtype=np.int64)
                        for local_index, step in enumerate(steps):
                            transition_lookup[(episode_id, int(step))] = destination_start + local_index
                    offsets[episode_key] = destination_stop

        if "candidate_transition_id" in memmaps and "episode_id" in memmaps:
            candidate_ids = memmaps["candidate_transition_id"]
            for index, (episode, step) in enumerate(
                zip(memmaps["episode_id"], memmaps["candidate_episode_step"])
            ):
                candidate_ids[index] = transition_lookup[(int(episode), int(step))]

        for key in scalar_keys:
            source = next((dataset[key] for dataset in datasets if key in dataset), None)
            if source is None:
                continue
            array_path = arrays_dir / f"{key}.npy"
            np.save(array_path, np.asarray(source), allow_pickle=False)

        if global_vehicle_ids:
            max_length = max(1, max(len(item) for item in global_vehicle_ids))
            np.save(
                arrays_dir / f"{id_table_key}.npy",
                np.asarray(global_vehicle_ids, dtype=f"<U{max_length}"),
                allow_pickle=False,
            )

        for memmap in memmaps.values():
            memmap.flush()
            mmap_handle = getattr(memmap, "_mmap", None)
            if mmap_handle is not None:
                mmap_handle.close()
        memmaps.clear()

        normalized: dict[str, np.ndarray] = {}
        for array_path in sorted(arrays_dir.glob("*.npy")):
            value = np.load(array_path, mmap_mode="r", allow_pickle=False)
            normalized[array_path.stem] = value
            entries[array_path.stem] = _array_manifest_entry(temporary, array_path, value)
        manifest = {
            "format_version": STAGE1_FORMAT_VERSION,
            "stage1_buffer_schema_version": STAGE1_BUFFER_SCHEMA_VERSION,
            **dict(metadata),
            **_dataset_counts(normalized),
            "arrays": entries,
        }
        _json_dump_atomic(temporary / "manifest.json", manifest)
        for value in normalized.values():
            mmap_handle = getattr(value, "_mmap", None)
            if mmap_handle is not None:
                mmap_handle.close()
        normalized.clear()
        _promote_directory(temporary, output_path)
    except Exception:
        memmaps.clear()
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    finally:
        for dataset in datasets:
            dataset.close()
    return output_path
