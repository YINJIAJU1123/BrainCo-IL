#!/usr/bin/env python3
"""Physically merge two or more compatible LeRobot v2.x datasets.

The source datasets are treated as read-only. Episodes are appended in source
argument order, while episode indices, global frame indices, and task indices
are rebuilt for the merged dataset. Videos are copied byte-for-byte and are
never decoded or re-encoded.

State and action vectors are preserved exactly. In particular, raw Revo3 70D
vectors remain 70D in the merged dataset; the training recipe and fast norm
stats script perform the downstream ``[..., 14:70]`` EEF removal to produce
the model's 56D representation.

Example:

    python scripts/merge_lerobot_datasets.py \
        /path/to/dataset_a /path/to/dataset_b \
        --output /path/to/merged_dataset

OUTPUT must not exist. It is first built in a temporary sibling directory and
is atomically renamed into place only after full validation succeeds.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import UTC
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
from typing import Any
import uuid

import numpy as np
import pyarrow as pa
import pyarrow.parquet as parquet


@dataclass(frozen=True)
class SourceDataset:
    root: Path
    info: dict[str, Any]
    episodes: list[dict[str, Any]]
    tasks: list[dict[str, Any]]
    task_text_by_index: dict[int, str]
    stats_by_episode: dict[int, dict[str, Any]]
    snapshot: dict[str, tuple[int, int, int, int]]


@dataclass(frozen=True)
class EpisodeMapping:
    source_dataset: str
    source_episode_index: int
    output_episode_index: int
    output_global_index_start: int
    length: int


def _read_jsonlines(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: expected a JSON object")
        rows.append(value)
    return rows


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_json(path: Path, value: Any, *, indent: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as file:
        json.dump(_json_safe(value), file, ensure_ascii=False, indent=indent, allow_nan=False)
        file.write("\n")


def _write_jsonlines(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as file:
        for row in rows:
            json.dump(
                _json_safe(row),
                file,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            )
            file.write("\n")


def _format_dataset_path(template: str, info: dict[str, Any], episode_index: int, **extra: Any) -> Path:
    chunks_size = int(info.get("chunks_size", 1000))
    return Path(
        template.format(
            episode_chunk=episode_index // chunks_size,
            episode_index=episode_index,
            **extra,
        )
    )


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _validate_paths(raw_sources: list[Path], raw_output: Path) -> tuple[list[Path], Path]:
    if len(raw_sources) < 2:
        raise ValueError("At least two SOURCE datasets are required")
    sources = [path.expanduser().resolve(strict=True) for path in raw_sources]
    if len(set(sources)) != len(sources):
        raise ValueError("Every SOURCE dataset must be distinct")
    for source in sources:
        if not source.is_dir():
            raise NotADirectoryError(source)

    output = raw_output.expanduser().resolve(strict=False)
    if output.exists():
        raise FileExistsError(f"OUTPUT already exists; refusing to overwrite it: {output}")
    for source in sources:
        if output == source or _is_relative_to(output, source) or _is_relative_to(source, output):
            raise ValueError(f"OUTPUT must be outside every SOURCE dataset tree: {source}")
    return sources, output


def _dataset_file_snapshot(root: Path) -> dict[str, tuple[int, int, int, int]]:
    snapshot: dict[str, tuple[int, int, int, int]] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            stat = path.stat()
            snapshot[str(path.relative_to(root))] = (
                stat.st_dev,
                stat.st_ino,
                stat.st_size,
                stat.st_mtime_ns,
            )
    return snapshot


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _scalar_column(table: Any, key: str, *, dtype: Any) -> np.ndarray:
    if key not in table.column_names:
        raise ValueError(f"Required Parquet column is missing: {key}")
    values = np.asarray(table.column(key).combine_chunks().to_pylist(), dtype=dtype)
    if values.ndim != 1:
        raise ValueError(f"Expected scalar Parquet column {key!r}, got shape {values.shape}")
    return values


def _load_source(root: Path, snapshot: dict[str, tuple[int, int, int, int]]) -> SourceDataset:
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(info_path)
    info = json.loads(info_path.read_text())
    if not isinstance(info, dict):
        raise ValueError(f"{info_path}: expected a JSON object")
    version = str(info.get("codebase_version", ""))
    if not version.startswith("v2."):
        raise ValueError(f"{root}: only LeRobot v2.x is supported, got {version!r}")
    chunks_size = int(info.get("chunks_size", 0))
    if chunks_size <= 0:
        raise ValueError(f"{root}: chunks_size must be positive")
    if not isinstance(info.get("data_path"), str):
        raise ValueError(f"{root}: info.json must contain a string data_path")
    features = info.get("features")
    if not isinstance(features, dict) or not features:
        raise ValueError(f"{root}: info.json features is missing or empty")

    video_keys = [key for key, feature in features.items() if feature.get("dtype") == "video"]
    if video_keys and not isinstance(info.get("video_path"), str):
        raise ValueError(f"{root}: video features require a string video_path")

    episodes = _read_jsonlines(root / "meta" / "episodes.jsonl")
    if not episodes:
        raise ValueError(f"{root}: episodes.jsonl is empty")
    episode_indices = [int(row.get("episode_index", -1)) for row in episodes]
    if episode_indices != list(range(len(episodes))):
        raise ValueError(f"{root}: episode indices must be ordered and contiguous from zero")
    if int(info.get("total_episodes", -1)) != len(episodes):
        raise ValueError(f"{root}: total_episodes does not match episodes.jsonl")

    tasks = _read_jsonlines(root / "meta" / "tasks.jsonl")
    task_text_by_index: dict[int, str] = {}
    seen_task_text: set[str] = set()
    for row in tasks:
        task_index = int(row.get("task_index", -1))
        task = row.get("task")
        if task_index < 0 or not isinstance(task, str):
            raise ValueError(f"{root}: invalid tasks.jsonl row: {row}")
        if task_index in task_text_by_index:
            raise ValueError(f"{root}: duplicate task_index {task_index}")
        if task in seen_task_text:
            raise ValueError(f"{root}: duplicate task text {task!r}")
        task_text_by_index[task_index] = task
        seen_task_text.add(task)
    if sorted(task_text_by_index) != list(range(len(tasks))):
        raise ValueError(f"{root}: task indices must be contiguous from zero")
    if int(info.get("total_tasks", -1)) != len(tasks):
        raise ValueError(f"{root}: total_tasks does not match tasks.jsonl")

    stats_rows = _read_jsonlines(root / "meta" / "episodes_stats.jsonl")
    stats_by_episode: dict[int, dict[str, Any]] = {}
    for row in stats_rows:
        episode_index = int(row.get("episode_index", -1))
        stats = row.get("stats")
        if episode_index in stats_by_episode or not isinstance(stats, dict):
            raise ValueError(f"{root}: invalid or duplicate episodes_stats row for episode {episode_index}")
        stats_by_episode[episode_index] = stats
    if sorted(stats_by_episode) != episode_indices:
        raise ValueError(f"{root}: episodes_stats.jsonl must contain exactly one ordered-independent row per episode")

    data_template = str(info["data_path"])
    video_template = info.get("video_path")
    total_frames = 0
    expected_global_index = 0
    expected_schema = None
    for episode in episodes:
        episode_index = int(episode["episode_index"])
        length = int(episode.get("length", -1))
        if length <= 0:
            raise ValueError(f"{root}: episode {episode_index} has invalid length {length}")
        parquet_path = root / _format_dataset_path(data_template, info, episode_index)
        if not parquet_path.is_file():
            raise FileNotFoundError(parquet_path)
        table = parquet.read_table(
            parquet_path,
            columns=["timestamp", "frame_index", "episode_index", "index", "task_index"],
        )
        if table.num_rows != length:
            raise ValueError(f"{root}: episode {episode_index} metadata length does not match Parquet")
        timestamps = _scalar_column(table, "timestamp", dtype=np.float64)
        frame_indices = _scalar_column(table, "frame_index", dtype=np.int64)
        episode_column = _scalar_column(table, "episode_index", dtype=np.int64)
        global_indices = _scalar_column(table, "index", dtype=np.int64)
        task_indices = _scalar_column(table, "task_index", dtype=np.int64)
        if not np.all(np.isfinite(timestamps)) or np.any(np.diff(timestamps) <= 0):
            raise ValueError(f"{root}: episode {episode_index} timestamps must be finite and strictly increasing")
        if not np.array_equal(frame_indices, np.arange(length)):
            raise ValueError(f"{root}: episode {episode_index} frame_index is not contiguous from zero")
        if not np.all(episode_column == episode_index):
            raise ValueError(f"{root}: episode {episode_index} has inconsistent episode_index values")
        if not np.array_equal(global_indices, np.arange(expected_global_index, expected_global_index + length)):
            raise ValueError(f"{root}: episode {episode_index} has invalid global index values")
        unknown_tasks = sorted(set(task_indices.tolist()) - set(task_text_by_index))
        if unknown_tasks:
            raise ValueError(f"{root}: episode {episode_index} uses unknown task indices {unknown_tasks}")
        episode_tasks = episode.get("tasks")
        if not isinstance(episode_tasks, list) or not all(isinstance(task, str) for task in episode_tasks):
            raise ValueError(f"{root}: episode {episode_index} has invalid tasks metadata")
        unknown_task_text = sorted(set(episode_tasks) - seen_task_text)
        if unknown_task_text:
            raise ValueError(f"{root}: episode {episode_index} names unknown tasks {unknown_task_text}")

        schema = parquet.read_schema(parquet_path)
        if expected_schema is None:
            expected_schema = schema
        elif not schema.equals(expected_schema):
            raise ValueError(f"{root}: Parquet schema differs at episode {episode_index}")

        episode_stats = stats_by_episode[episode_index]
        for video_key in video_keys:
            if video_key not in episode_stats:
                raise ValueError(f"{root}: episode {episode_index} stats are missing video key {video_key!r}")
            video_path = root / _format_dataset_path(str(video_template), info, episode_index, video_key=video_key)
            if not video_path.is_file():
                raise FileNotFoundError(video_path)
        expected_global_index += length
        total_frames += length

    if int(info.get("total_frames", -1)) != total_frames:
        raise ValueError(f"{root}: total_frames does not match episode data")
    expected_videos = len(episodes) * len(video_keys)
    if int(info.get("total_videos", -1)) != expected_videos:
        raise ValueError(f"{root}: total_videos does not match episode/video features")
    expected_chunks = math.ceil(len(episodes) / chunks_size)
    if int(info.get("total_chunks", -1)) != expected_chunks:
        raise ValueError(f"{root}: total_chunks is incorrect")

    return SourceDataset(
        root=root,
        info=info,
        episodes=episodes,
        tasks=tasks,
        task_text_by_index=task_text_by_index,
        stats_by_episode=stats_by_episode,
        snapshot=snapshot,
    )


def _validate_compatible(sources: list[SourceDataset]) -> None:
    reference = sources[0]
    required_equal = (
        "codebase_version",
        "robot_type",
        "fps",
        "chunks_size",
        "data_path",
        "video_path",
        "features",
    )
    for source in sources[1:]:
        for key in required_equal:
            if source.info.get(key) != reference.info.get(key):
                raise ValueError(f"Incompatible datasets: {key!r} differs between {reference.root} and {source.root}")
        reference_schema = parquet.read_schema(
            reference.root / _format_dataset_path(str(reference.info["data_path"]), reference.info, 0)
        )
        source_schema = parquet.read_schema(
            source.root / _format_dataset_path(str(source.info["data_path"]), source.info, 0)
        )
        if not source_schema.equals(reference_schema):
            raise ValueError(f"Incompatible Parquet schemas: {reference.root} and {source.root}")


def _build_tasks(
    sources: list[SourceDataset],
) -> tuple[list[dict[str, Any]], dict[Path, dict[int, int]]]:
    output_tasks: list[dict[str, Any]] = []
    output_index_by_text: dict[str, int] = {}
    remaps: dict[Path, dict[int, int]] = {}
    for source in sources:
        source_remap: dict[int, int] = {}
        for row in source.tasks:
            old_index = int(row["task_index"])
            task = str(row["task"])
            if task not in output_index_by_text:
                new_index = len(output_tasks)
                output_index_by_text[task] = new_index
                new_row = dict(row)
                new_row["task_index"] = new_index
                output_tasks.append(new_row)
            source_remap[old_index] = output_index_by_text[task]
        remaps[source.root] = source_remap
    return output_tasks, remaps


def _replace_scalar_column(table: Any, key: str, values: np.ndarray) -> Any:
    field_index = table.schema.get_field_index(key)
    if field_index < 0:
        raise ValueError(f"Required Parquet column is missing: {key}")
    field = table.schema.field(field_index)
    return table.set_column(field_index, field, pa.array(values, type=field.type))


def _remap_task_indices(values: np.ndarray, remap: dict[int, int], *, context: str) -> np.ndarray:
    result = np.empty_like(values)
    for old_index in np.unique(values):
        old = int(old_index)
        if old not in remap:
            raise ValueError(f"{context}: task_index {old} is not present in tasks.jsonl")
        result[values == old] = remap[old]
    return result


def _feature_stats(array: np.ndarray, *, axis: Any, keepdims: bool) -> dict[str, np.ndarray]:
    return {
        "min": np.min(array, axis=axis, keepdims=keepdims),
        "max": np.max(array, axis=axis, keepdims=keepdims),
        "mean": np.mean(array, axis=axis, keepdims=keepdims),
        "std": np.std(array, axis=axis, keepdims=keepdims),
        "count": np.array([len(array)]),
    }


def _numeric_episode_stats(table: Any, features: dict[str, dict[str, Any]]) -> dict[str, dict[str, np.ndarray]]:
    result: dict[str, dict[str, np.ndarray]] = {}
    for key, feature in features.items():
        if feature.get("dtype") in {"string", "image", "video"} or key not in table.column_names:
            continue
        data = np.asarray(table.column(key).combine_chunks().to_pylist())
        result[key] = _feature_stats(data, axis=0, keepdims=data.ndim == 1)
    return result


def _validate_output(
    root: Path,
    info: dict[str, Any],
    episodes: list[dict[str, Any]],
    tasks: list[dict[str, Any]],
    stats_rows: list[dict[str, Any]],
) -> None:
    written_info = json.loads((root / "meta" / "info.json").read_text())
    written_episodes = _read_jsonlines(root / "meta" / "episodes.jsonl")
    written_tasks = _read_jsonlines(root / "meta" / "tasks.jsonl")
    written_stats = _read_jsonlines(root / "meta" / "episodes_stats.jsonl")
    if (
        written_info != info
        or written_episodes != episodes
        or written_tasks != tasks
        or written_stats != _json_safe(stats_rows)
    ):
        raise ValueError("Output metadata differs from the generated metadata")
    if [int(row["episode_index"]) for row in written_episodes] != list(range(len(episodes))):
        raise ValueError("Output episode metadata is not ordered and contiguous")
    if [int(row["episode_index"]) for row in written_stats] != list(range(len(episodes))):
        raise ValueError("Output episode stats are not ordered and contiguous")
    if [int(row["task_index"]) for row in written_tasks] != list(range(len(tasks))):
        raise ValueError("Output task metadata is not ordered and contiguous")

    features = info["features"]
    video_keys = [key for key, feature in features.items() if feature.get("dtype") == "video"]
    data_template = str(info["data_path"])
    video_template = info.get("video_path")
    expected_global_index = 0
    expected_schema = None
    actual_video_count = 0
    for episode in written_episodes:
        episode_index = int(episode["episode_index"])
        length = int(episode["length"])
        path = root / _format_dataset_path(data_template, info, episode_index)
        table = parquet.read_table(path)
        if table.num_rows != length:
            raise ValueError(f"Output episode {episode_index} has wrong row count")
        frame_indices = _scalar_column(table, "frame_index", dtype=np.int64)
        episode_indices = _scalar_column(table, "episode_index", dtype=np.int64)
        global_indices = _scalar_column(table, "index", dtype=np.int64)
        task_indices = _scalar_column(table, "task_index", dtype=np.int64)
        timestamps = _scalar_column(table, "timestamp", dtype=np.float64)
        if not np.array_equal(frame_indices, np.arange(length)):
            raise ValueError(f"Output episode {episode_index} has invalid frame_index")
        if not np.all(episode_indices == episode_index):
            raise ValueError(f"Output episode {episode_index} has invalid episode_index")
        if not np.array_equal(global_indices, np.arange(expected_global_index, expected_global_index + length)):
            raise ValueError(f"Output episode {episode_index} has invalid global index")
        if not set(task_indices.tolist()).issubset(set(range(len(tasks)))):
            raise ValueError(f"Output episode {episode_index} has invalid task_index")
        if not np.all(np.isfinite(timestamps)) or np.any(np.diff(timestamps) <= 0):
            raise ValueError(f"Output episode {episode_index} has invalid timestamps")
        if expected_schema is None:
            expected_schema = table.schema
        elif not table.schema.equals(expected_schema):
            raise ValueError(f"Output episode {episode_index} has a different Parquet schema")
        for video_key in video_keys:
            video_path = root / _format_dataset_path(str(video_template), info, episode_index, video_key=video_key)
            if not video_path.is_file() or video_path.stat().st_size == 0:
                raise ValueError(f"Output video is missing or empty: {video_path}")
            actual_video_count += 1
        expected_global_index += length

    if int(info["total_episodes"]) != len(episodes):
        raise ValueError("Output total_episodes is incorrect")
    if int(info["total_frames"]) != expected_global_index:
        raise ValueError("Output total_frames is incorrect")
    if int(info["total_tasks"]) != len(tasks):
        raise ValueError("Output total_tasks is incorrect")
    if int(info["total_videos"]) != actual_video_count:
        raise ValueError("Output total_videos is incorrect")
    expected_chunks = math.ceil(len(episodes) / int(info["chunks_size"]))
    if int(info["total_chunks"]) != expected_chunks:
        raise ValueError("Output total_chunks is incorrect")


def _build_output(
    sources: list[SourceDataset],
    output: Path,
    output_tasks: list[dict[str, Any]],
    task_remaps: dict[Path, dict[int, int]],
) -> None:
    base_info = sources[0].info
    features = base_info["features"]
    video_keys = [key for key, feature in features.items() if feature.get("dtype") == "video"]
    data_template = str(base_info["data_path"])
    video_template = base_info.get("video_path")

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.tmp-{uuid.uuid4().hex}"
    committed = False
    episode_rows: list[dict[str, Any]] = []
    stats_rows: list[dict[str, Any]] = []
    mappings: list[EpisodeMapping] = []
    global_index = 0
    output_episode_index = 0
    copied_video_hashes: dict[str, str] = {}
    try:
        staging.mkdir()
        for source in sources:
            task_remap = task_remaps[source.root]
            for source_episode in source.episodes:
                source_episode_index = int(source_episode["episode_index"])
                length = int(source_episode["length"])
                source_parquet = source.root / _format_dataset_path(data_template, source.info, source_episode_index)
                output_parquet = staging / _format_dataset_path(data_template, base_info, output_episode_index)
                table = parquet.read_table(source_parquet)
                old_tasks = _scalar_column(table, "task_index", dtype=np.int64)
                new_tasks = _remap_task_indices(
                    old_tasks,
                    task_remap,
                    context=f"{source.root} episode {source_episode_index}",
                )
                table = _replace_scalar_column(
                    table,
                    "episode_index",
                    np.full(length, output_episode_index, dtype=np.int64),
                )
                table = _replace_scalar_column(
                    table,
                    "index",
                    np.arange(global_index, global_index + length, dtype=np.int64),
                )
                table = _replace_scalar_column(table, "task_index", new_tasks)
                output_parquet.parent.mkdir(parents=True, exist_ok=True)
                parquet.write_table(table, output_parquet, compression="snappy")

                episode_stats = _numeric_episode_stats(table, features)
                source_stats = source.stats_by_episode[source_episode_index]
                for video_key in video_keys:
                    source_video = source.root / _format_dataset_path(
                        str(video_template),
                        source.info,
                        source_episode_index,
                        video_key=video_key,
                    )
                    output_relative = _format_dataset_path(
                        str(video_template),
                        base_info,
                        output_episode_index,
                        video_key=video_key,
                    )
                    output_video = staging / output_relative
                    output_video.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source_video, output_video)
                    source_hash = _sha256(source_video)
                    output_hash = _sha256(output_video)
                    if output_hash != source_hash:
                        raise RuntimeError(f"Copied video checksum mismatch: {source_video}")
                    copied_video_hashes[str(output_relative)] = output_hash
                    episode_stats[video_key] = source_stats[video_key]

                new_episode = dict(source_episode)
                new_episode["episode_index"] = output_episode_index
                episode_rows.append(new_episode)
                stats_rows.append({"episode_index": output_episode_index, "stats": episode_stats})
                mappings.append(
                    EpisodeMapping(
                        source_dataset=str(source.root),
                        source_episode_index=source_episode_index,
                        output_episode_index=output_episode_index,
                        output_global_index_start=global_index,
                        length=length,
                    )
                )
                global_index += length
                output_episode_index += 1

        output_info = dict(base_info)
        output_info["total_episodes"] = len(episode_rows)
        output_info["total_frames"] = global_index
        output_info["total_tasks"] = len(output_tasks)
        output_info["total_videos"] = len(episode_rows) * len(video_keys)
        output_info["total_chunks"] = math.ceil(len(episode_rows) / int(output_info["chunks_size"]))
        output_info["splits"] = {"train": f"0:{len(episode_rows)}"}

        _write_json(staging / "meta" / "info.json", output_info, indent=4)
        _write_jsonlines(staging / "meta" / "tasks.jsonl", output_tasks)
        _write_jsonlines(staging / "meta" / "episodes.jsonl", episode_rows)
        _write_jsonlines(staging / "meta" / "episodes_stats.jsonl", stats_rows)
        provenance = {
            "schema_version": 1,
            "operation": "physical_merge_lerobot_v2_datasets",
            "created_at_utc": datetime.now(UTC).isoformat(),
            "sources": [str(source.root) for source in sources],
            "output": str(output),
            "source_unchanged": True,
            "state_and_action": {
                "operation": "preserve_source_values_and_dimensions",
                "revo3_70d_to_56d": "performed downstream by taking dimensions 14:70",
            },
            "videos": {
                "operation": "copy2_without_reencoding",
                "sha256": copied_video_hashes,
            },
            "task_index_remaps": {
                str(source.root): {str(old): new for old, new in task_remaps[source.root].items()} for source in sources
            },
            "episodes": [
                {
                    "source_dataset": mapping.source_dataset,
                    "source_episode_index": mapping.source_episode_index,
                    "output_episode_index": mapping.output_episode_index,
                    "output_global_index_start": mapping.output_global_index_start,
                    "length": mapping.length,
                }
                for mapping in mappings
            ],
        }
        _write_json(staging / "meta" / "merge_provenance.json", provenance)

        _validate_output(staging, output_info, episode_rows, output_tasks, stats_rows)
        for source in sources:
            if _dataset_file_snapshot(source.root) != source.snapshot:
                raise RuntimeError(f"Source dataset changed while merging: {source.root}")
        staging.rename(output)
        committed = True
    finally:
        if not committed and staging.exists():
            shutil.rmtree(staging)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "sources",
        nargs="+",
        type=Path,
        metavar="SOURCE",
        help="Read-only LeRobot v2.x dataset roots, appended in the order given",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        metavar="OUTPUT",
        help="New merged dataset root; must not exist and must be outside all SOURCE trees",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        source_paths, output = _validate_paths(args.sources, args.output)
        snapshots = {source: _dataset_file_snapshot(source) for source in source_paths}
        sources = [_load_source(source, snapshots[source]) for source in source_paths]
        _validate_compatible(sources)
        output_tasks, task_remaps = _build_tasks(sources)
        _build_output(sources, output, output_tasks, task_remaps)
        print(f"Created merged LeRobot dataset: {output}")
        print(f"Sources: {len(sources)}")
        print(f"Episodes: {sum(len(source.episodes) for source in sources)}")
        print(f"Frames: {sum(int(source.info['total_frames']) for source in sources)}")
        print(f"Provenance: {output / 'meta' / 'merge_provenance.json'}")
        return 0
    except (
        FileNotFoundError,
        NotADirectoryError,
        FileExistsError,
        ValueError,
        RuntimeError,
        OSError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
