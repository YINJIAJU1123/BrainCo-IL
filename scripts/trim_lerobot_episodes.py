#!/usr/bin/env python3
"""Create a new LeRobot v2.x dataset with episode prefixes trimmed by time.

The source dataset is opened read-only. The output must be a new, non-existent
directory outside the source tree and is built in a temporary sibling directory
before it is atomically published.

Examples:

    python scripts/trim_lerobot_episodes.py SOURCE OUTPUT --episode-id 3 --timestamp-s 0.5 --dry-run
    python scripts/trim_lerobot_episodes.py SOURCE OUTPUT --episode-id 3 --timestamp-s 0.5
    python scripts/trim_lerobot_episodes.py SOURCE OUTPUT --trim-start-s 0.5 --dry-run
    python scripts/trim_lerobot_episodes.py SOURCE OUTPUT --trim-start-s 0.5
    python scripts/trim_lerobot_episodes.py SOURCE OUTPUT --trim-manifest trims.json

A JSON manifest can be a mapping from episode index to seconds::

    {"0": 0.5, "3": 0.8}

CSV manifests must contain ``episode_index`` and ``trim_start_s`` columns.
Episodes omitted from a manifest are copied with a trim time of zero. Language
task labels and all state/action values in retained rows are left unchanged.
When ``--episode-id`` is passed, the output is a standalone one-episode dataset;
the selected source episode is reindexed to output episode 0 and the mapping is
recorded in ``meta/trim_provenance.json``.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from dataclasses import dataclass
from datetime import UTC
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any
import uuid

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as parquet

GENERATED_META_FILES = {
    "episodes.jsonl",
    "episodes_stats.jsonl",
    "info.json",
    "trim_provenance.json",
}


@dataclass(frozen=True)
class TrimPlan:
    source_episode_index: int
    output_episode_index: int
    requested_trim_start_s: float
    old_frame_start_inclusive: int
    old_frame_end_exclusive: int
    old_length: int
    new_length: int
    removed_frames: int
    source_timestamp_start_s: float
    source_elapsed_start_s: float
    new_global_index_start: int
    new_global_index_end_exclusive: int


def _read_jsonlines(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write_json(path: Path, value: Any, *, indent: int | None = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as file:
        json.dump(_json_safe(value), file, indent=indent, ensure_ascii=False, allow_nan=False)
        file.write("\n")


def _write_jsonlines(path: Path, values: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as file:
        for value in values:
            json.dump(_json_safe(value), file, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
            file.write("\n")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _format_dataset_path(template: str, info: dict[str, Any], episode_index: int, **extra: Any) -> Path:
    chunks_size = int(info.get("chunks_size", 1000))
    values = {
        "episode_chunk": episode_index // chunks_size,
        "episode_index": episode_index,
        **extra,
    }
    return Path(template.format(**values))


def _validate_trim_value(value: Any, *, context: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{context} must be a non-negative finite number, not {value!r}")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{context} must be a non-negative finite number, not {value!r}") from error
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{context} must be a non-negative finite number, not {value!r}")
    return result


def _load_trim_manifest(path: Path) -> dict[int, float]:
    if not path.is_file():
        raise FileNotFoundError(path)

    entries: list[tuple[Any, Any]]
    if path.suffix.lower() == ".csv":
        with path.open(newline="") as file:
            rows = list(csv.DictReader(file))
        required = {"episode_index", "trim_start_s"}
        if rows and not required.issubset(rows[0]):
            raise ValueError(f"CSV manifest must contain columns {sorted(required)}")
        entries = [(row.get("episode_index"), row.get("trim_start_s")) for row in rows]
    else:
        value = json.loads(path.read_text())
        if isinstance(value, dict) and "episodes" in value:
            value = value["episodes"]
        if isinstance(value, dict):
            entries = list(value.items())
        elif isinstance(value, list):
            entries = []
            for row in value:
                if not isinstance(row, dict) or not {"episode_index", "trim_start_s"}.issubset(row):
                    raise ValueError("JSON manifest list entries need episode_index and trim_start_s")
                entries.append((row["episode_index"], row["trim_start_s"]))
        else:
            raise ValueError("JSON manifest must be an episode mapping or a list of episode records")

    trims: dict[int, float] = {}
    for raw_episode, raw_trim in entries:
        try:
            episode_index = int(raw_episode)
        except (TypeError, ValueError) as error:
            raise ValueError(f"Invalid episode index in manifest: {raw_episode!r}") from error
        if episode_index < 0 or str(raw_episode).strip() not in {str(episode_index), f"{episode_index}.0"}:
            raise ValueError(f"Invalid episode index in manifest: {raw_episode!r}")
        if episode_index in trims:
            raise ValueError(f"Duplicate episode index in manifest: {episode_index}")
        trims[episode_index] = _validate_trim_value(raw_trim, context=f"trim time for episode {episode_index}")
    if not trims:
        raise ValueError("Trim manifest is empty")
    return trims


def _scalar_column(table: Any, key: str, *, dtype: Any) -> np.ndarray:
    if key not in table.column_names:
        raise ValueError(f"Required Parquet column is missing: {key}")
    values = np.asarray(table.column(key).combine_chunks().to_pylist(), dtype=dtype)
    if values.ndim != 1:
        raise ValueError(f"Expected scalar Parquet column {key!r}, got shape {values.shape}")
    return values


def _load_dataset(source: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[int]]:
    info_path = source / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"LeRobot info file not found: {info_path}")
    info = json.loads(info_path.read_text())
    codebase_version = str(info.get("codebase_version", ""))
    if not codebase_version.startswith("v2."):
        raise ValueError(f"Only LeRobot v2.x datasets are supported, got {codebase_version!r}")

    episodes = _read_jsonlines(source / "meta" / "episodes.jsonl")
    if not episodes:
        raise ValueError("meta/episodes.jsonl is missing or empty")
    episode_indices = [int(episode["episode_index"]) for episode in episodes]
    expected = list(range(len(episode_indices)))
    if episode_indices != expected:
        raise ValueError(
            "Episode indices must be ordered and contiguous so output episode numbering remains valid; "
            f"got {episode_indices[:20]}"
        )
    if int(info.get("total_episodes", -1)) != len(episodes):
        raise ValueError("info.json total_episodes does not match episodes.jsonl")
    return info, episodes, episode_indices


def _build_plans(
    source: Path,
    info: dict[str, Any],
    episodes: list[dict[str, Any]],
    trim_by_episode: dict[int, float],
    *,
    reindex_output_episodes: bool = False,
) -> list[TrimPlan]:
    data_template = info.get("data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet")
    global_index = 0
    plans: list[TrimPlan] = []
    for output_position, episode in enumerate(episodes):
        source_episode_index = int(episode["episode_index"])
        output_episode_index = output_position if reindex_output_episodes else source_episode_index
        parquet_path = source / _format_dataset_path(data_template, info, source_episode_index)
        if not parquet_path.is_file():
            raise FileNotFoundError(parquet_path)
        table = parquet.read_table(parquet_path, columns=["timestamp", "frame_index", "episode_index", "index"])
        timestamps = _scalar_column(table, "timestamp", dtype=np.float64)
        frame_indices = _scalar_column(table, "frame_index", dtype=np.int64)
        episode_column = _scalar_column(table, "episode_index", dtype=np.int64)
        old_length = len(timestamps)
        if old_length == 0:
            raise ValueError(f"Episode {source_episode_index} is empty")
        if int(episode.get("length", -1)) != old_length:
            raise ValueError(f"Episode {source_episode_index} metadata length does not match its Parquet rows")
        if not np.all(np.isfinite(timestamps)) or np.any(np.diff(timestamps) <= 0):
            raise ValueError(f"Episode {source_episode_index} timestamps must be finite and strictly increasing")
        if not np.array_equal(frame_indices, np.arange(old_length)):
            raise ValueError(f"Episode {source_episode_index} frame_index is not 0..{old_length - 1}")
        if not np.all(episode_column == source_episode_index):
            raise ValueError(f"Episode {source_episode_index} has inconsistent episode_index values")

        requested = trim_by_episode.get(source_episode_index, 0.0)
        elapsed = timestamps - timestamps[0]
        start = int(np.searchsorted(elapsed, requested, side="left"))
        if start >= old_length:
            raise ValueError(
                f"Episode {source_episode_index}: trim_start_s={requested:g} removes every frame; "
                f"last frame is at {elapsed[-1]:.9g}s"
            )
        new_length = old_length - start
        plans.append(
            TrimPlan(
                source_episode_index=source_episode_index,
                output_episode_index=output_episode_index,
                requested_trim_start_s=requested,
                old_frame_start_inclusive=start,
                old_frame_end_exclusive=old_length,
                old_length=old_length,
                new_length=new_length,
                removed_frames=start,
                source_timestamp_start_s=float(timestamps[start]),
                source_elapsed_start_s=float(elapsed[start]),
                new_global_index_start=global_index,
                new_global_index_end_exclusive=global_index + new_length,
            )
        )
        global_index += new_length
    return plans


def _print_plan(plans: list[TrimPlan], *, fps: float, dry_run: bool) -> None:
    prefix = "DRY RUN - " if dry_run else ""
    print(f"{prefix}trim plan ({len(plans)} episodes, fps={fps:g})")
    print("source->output  requested_s  keep_old_frames  removed  kept  snapped_elapsed_s  new_duration_s")
    for plan in plans:
        duration = (plan.new_length - 1) / fps if plan.new_length > 1 else 0.0
        print(
            f"{plan.source_episode_index:6d}->{plan.output_episode_index:<3d}  "
            f"{plan.requested_trim_start_s:11.6f}  "
            f"[{plan.old_frame_start_inclusive}:{plan.old_frame_end_exclusive})"
            f"{plan.removed_frames:9d}{plan.new_length:6d}  "
            f"{plan.source_elapsed_start_s:17.9f}  {duration:14.6f}"
        )
    print(f"total frames: {sum(plan.old_length for plan in plans)} -> {sum(plan.new_length for plan in plans)}")


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _validate_paths(source: Path, output: Path, *, dry_run: bool) -> tuple[Path, Path]:
    source = source.expanduser().resolve(strict=True)
    output = output.expanduser().resolve(strict=False)
    if not source.is_dir():
        raise NotADirectoryError(source)
    if source == output or _is_relative_to(output, source) or _is_relative_to(source, output):
        raise ValueError("Output must be separate from, and outside, the source dataset tree")
    if output.exists() and not dry_run:
        raise FileExistsError(f"Output already exists; refusing to overwrite it: {output}")
    return source, output


def _dataset_file_snapshot(root: Path) -> dict[str, tuple[int, int, int, int]]:
    snapshot: dict[str, tuple[int, int, int, int]] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            stat = path.stat()
            snapshot[str(path.relative_to(root))] = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
    return snapshot


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _source_hashes(source: Path, relative_paths: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for index, relative in enumerate(relative_paths, start=1):
        print(f"Hashing source file {index}/{len(relative_paths)}: {relative}")
        result[relative] = _sha256(source / relative)
    return result


def _provenance_source_files(
    source_snapshot: dict[str, tuple[int, int, int, int]],
    info: dict[str, Any],
    plans: list[TrimPlan],
) -> list[str]:
    selected = {
        relative
        for relative in source_snapshot
        if not relative.startswith("data/") and not relative.startswith("videos/")
    }
    data_template = info.get("data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet")
    video_template = info.get("video_path")
    video_keys = [key for key, feature in info.get("features", {}).items() if feature.get("dtype") == "video"]
    for plan in plans:
        selected.add(str(_format_dataset_path(data_template, info, plan.source_episode_index)))
        if video_template:
            for video_key in video_keys:
                selected.add(
                    str(_format_dataset_path(str(video_template), info, plan.source_episode_index, video_key=video_key))
                )
    missing = sorted(selected - set(source_snapshot))
    if missing:
        raise FileNotFoundError(f"Selected source files are missing: {missing}")
    return sorted(selected)


def _copy_auxiliary_files(source: Path, staging: Path) -> None:
    for child in source.iterdir():
        if child.name in {"data", "videos", "meta"}:
            continue
        destination = staging / child.name
        if child.is_dir():
            shutil.copytree(child, destination)
        else:
            shutil.copy2(child, destination)

    source_meta = source / "meta"
    destination_meta = staging / "meta"
    destination_meta.mkdir(parents=True, exist_ok=True)
    for child in source_meta.iterdir():
        if child.name in GENERATED_META_FILES:
            continue
        destination = destination_meta / child.name
        if child.is_dir():
            shutil.copytree(child, destination)
        else:
            shutil.copy2(child, destination)


def _replace_scalar_column(table: Any, key: str, values: np.ndarray) -> Any:
    field_index = table.schema.get_field_index(key)
    if field_index < 0:
        raise ValueError(f"Required Parquet column is missing: {key}")
    field = table.schema.field(field_index)
    return table.set_column(field_index, field, pa.array(values, type=field.type))


def _write_trimmed_parquet(
    source_path: Path,
    output_path: Path,
    plan: TrimPlan,
) -> Any:
    table = parquet.read_table(source_path)
    table = table.slice(plan.old_frame_start_inclusive, plan.new_length)
    source_timestamps = _scalar_column(table, "timestamp", dtype=np.float64)
    rebased_timestamps = source_timestamps - source_timestamps[0]
    rebased_timestamps[0] = 0.0
    table = _replace_scalar_column(table, "timestamp", rebased_timestamps)
    table = _replace_scalar_column(table, "frame_index", np.arange(plan.new_length, dtype=np.int64))
    table = _replace_scalar_column(
        table, "episode_index", np.full(plan.new_length, plan.output_episode_index, dtype=np.int64)
    )
    table = _replace_scalar_column(
        table,
        "index",
        np.arange(plan.new_global_index_start, plan.new_global_index_end_exclusive, dtype=np.int64),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    parquet.write_table(table, output_path, compression="snappy")
    return table


def _encode_trimmed_video(
    source_path: Path,
    output_path: Path,
    plan: TrimPlan,
    *,
    ffmpeg: str,
    fps: float,
    codec: str,
    pixel_format: str,
    preset: str,
    crf: int,
) -> None:
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    video_filter = (
        f"trim=start_frame={plan.old_frame_start_inclusive}:end_frame={plan.old_frame_end_exclusive},"
        "setpts=PTS-STARTPTS"
    )
    command = [
        ffmpeg,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source_path),
        "-map",
        "0:v:0",
        "-vf",
        video_filter,
        "-frames:v",
        str(plan.new_length),
        "-an",
        "-c:v",
        codec,
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-pix_fmt",
        pixel_format,
        "-r",
        f"{fps:g}",
        "-fps_mode",
        "cfr",
        "-movflags",
        "+faststart",
        "-y",
        str(output_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg failed for {source_path}:\n{result.stderr.strip()}")


def _estimate_num_samples(
    dataset_len: int,
    min_num_samples: int = 100,
    max_num_samples: int = 10_000,
    power: float = 0.75,
) -> int:
    min_num_samples = min(dataset_len, min_num_samples)
    return max(min_num_samples, min(int(dataset_len**power), max_num_samples))


def _sample_indices(data_len: int) -> list[int]:
    sample_count = _estimate_num_samples(data_len)
    return np.round(np.linspace(0, data_len - 1, sample_count)).astype(int).tolist()


def _downsample_channel_first(image: np.ndarray, target_size: int = 150, max_size_threshold: int = 300) -> np.ndarray:
    _, height, width = image.shape
    if max(width, height) < max_size_threshold:
        return image
    factor = int(width / target_size) if width > height else int(height / target_size)
    return image[:, ::factor, ::factor]


def _feature_stats(array: np.ndarray, *, axis: Any, keepdims: bool) -> dict[str, np.ndarray]:
    return {
        "min": np.min(array, axis=axis, keepdims=keepdims),
        "max": np.max(array, axis=axis, keepdims=keepdims),
        "mean": np.mean(array, axis=axis, keepdims=keepdims),
        "std": np.std(array, axis=axis, keepdims=keepdims),
        "count": np.array([len(array)]),
    }


def _video_stats_and_timestamps(path: Path, expected_frames: int) -> tuple[dict[str, np.ndarray], np.ndarray]:
    wanted = set(_sample_indices(expected_frames))
    sampled: list[np.ndarray] = []
    timestamps: list[float] = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        for frame_index, frame in enumerate(container.decode(stream)):
            if frame.pts is None:
                raise ValueError(f"Decoded video frame has no PTS: {path}")
            timestamps.append(float(frame.pts * frame.time_base))
            if frame_index in wanted:
                image = frame.to_ndarray(format="rgb24").transpose(2, 0, 1)
                sampled.append(_downsample_channel_first(image))
    if len(timestamps) != expected_frames:
        raise ValueError(f"Video {path} has {len(timestamps)} decoded frames, expected {expected_frames}")
    if len(sampled) != len(wanted):
        raise ValueError(f"Could not decode every sampled frame from {path}")
    array = np.stack(sampled)
    stats = _feature_stats(array, axis=(0, 2, 3), keepdims=True)
    normalized = {key: value if key == "count" else np.squeeze(value / 255.0, axis=0) for key, value in stats.items()}
    return normalized, np.asarray(timestamps, dtype=np.float64)


def _numeric_episode_stats(table: Any, features: dict[str, dict[str, Any]]) -> dict[str, dict[str, np.ndarray]]:
    result: dict[str, dict[str, np.ndarray]] = {}
    for key, feature in features.items():
        dtype = feature.get("dtype")
        if dtype in {"string", "image", "video"} or key not in table.column_names:
            continue
        data = np.asarray(table.column(key).combine_chunks().to_pylist())
        result[key] = _feature_stats(data, axis=0, keepdims=data.ndim == 1)
    return result


def _validate_video_timeline(
    video_path: Path,
    video_timestamps: np.ndarray,
    data_timestamps: np.ndarray,
    fps: float,
) -> dict[str, Any]:
    if len(video_timestamps) == 0:
        raise ValueError(f"Video has no frames: {video_path}")
    relative_video = video_timestamps - video_timestamps[0]
    relative_data = data_timestamps - data_timestamps[0]
    max_alignment_error = float(np.max(np.abs(relative_video - relative_data)))
    tolerance = max(1e-6, 0.5 / fps)
    if abs(video_timestamps[0]) > tolerance:
        raise ValueError(f"Video does not start near PTS zero: {video_path} ({video_timestamps[0]:.9g}s)")
    if max_alignment_error > tolerance:
        raise ValueError(
            f"Video/data timeline mismatch for {video_path}: {max_alignment_error:.9g}s > {tolerance:.9g}s"
        )
    return {
        "frame_count": len(video_timestamps),
        "first_pts_s": float(video_timestamps[0]),
        "last_pts_s": float(video_timestamps[-1]),
        "max_data_alignment_error_s": max_alignment_error,
    }


def _validate_output(
    root: Path,
    info: dict[str, Any],
    episodes: list[dict[str, Any]],
    plans: list[TrimPlan],
) -> None:
    written_info = json.loads((root / "meta" / "info.json").read_text())
    written_episodes = _read_jsonlines(root / "meta" / "episodes.jsonl")
    written_stats = _read_jsonlines(root / "meta" / "episodes_stats.jsonl")
    if int(written_info["total_frames"]) != sum(plan.new_length for plan in plans):
        raise ValueError("Output info.json total_frames is incorrect")
    if written_episodes != episodes:
        raise ValueError("Output episodes.jsonl does not match the regenerated episode metadata")
    if [row.get("episode_index") for row in written_stats] != [plan.output_episode_index for plan in plans]:
        raise ValueError("Output episodes_stats.jsonl is incomplete or out of order")

    data_template = info.get("data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet")
    expected_global_index = 0
    for plan in plans:
        path = root / _format_dataset_path(data_template, info, plan.output_episode_index)
        table = parquet.read_table(path)
        if table.num_rows != plan.new_length:
            raise ValueError(f"Output episode {plan.output_episode_index} has the wrong row count")
        timestamps = _scalar_column(table, "timestamp", dtype=np.float64)
        frame_indices = _scalar_column(table, "frame_index", dtype=np.int64)
        episode_indices = _scalar_column(table, "episode_index", dtype=np.int64)
        global_indices = _scalar_column(table, "index", dtype=np.int64)
        if timestamps[0] != 0 or np.any(np.diff(timestamps) <= 0):
            raise ValueError(f"Output episode {plan.output_episode_index} timestamps are not rebased and increasing")
        if not np.array_equal(frame_indices, np.arange(plan.new_length)):
            raise ValueError(f"Output episode {plan.output_episode_index} frame_index is invalid")
        if not np.all(episode_indices == plan.output_episode_index):
            raise ValueError(f"Output episode {plan.output_episode_index} episode_index is invalid")
        if not np.array_equal(
            global_indices, np.arange(expected_global_index, expected_global_index + plan.new_length)
        ):
            raise ValueError(f"Output episode {plan.output_episode_index} global index is invalid")
        expected_global_index += plan.new_length


def _build_output(
    source: Path,
    output: Path,
    info: dict[str, Any],
    source_episodes: list[dict[str, Any]],
    plans: list[TrimPlan],
    *,
    ffmpeg: str,
    video_codec: str,
    video_preset: str,
    video_crf: int,
    source_snapshot: dict[str, tuple[int, int, int, int]],
    source_hashes: dict[str, str],
) -> None:
    fps = float(info["fps"])
    features = info.get("features", {})
    video_keys = [key for key, feature in features.items() if feature.get("dtype") == "video"]
    data_template = info.get("data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet")
    video_template = info.get("video_path")
    if video_keys and not video_template:
        raise ValueError("Dataset has video features but info.json has no video_path template")

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.tmp-{uuid.uuid4().hex}"
    committed = False
    try:
        staging.mkdir()
        _copy_auxiliary_files(source, staging)
        new_episodes: list[dict[str, Any]] = []
        episode_stats_rows: list[dict[str, Any]] = []
        video_validation: dict[str, dict[str, Any]] = {}

        for plan, source_episode in zip(plans, source_episodes, strict=True):
            print(
                f"Writing source episode {plan.source_episode_index} as output episode "
                f"{plan.output_episode_index}: keep {plan.new_length}/{plan.old_length} frames"
            )
            source_parquet = source / _format_dataset_path(data_template, info, plan.source_episode_index)
            output_parquet = staging / _format_dataset_path(data_template, info, plan.output_episode_index)
            output_table = _write_trimmed_parquet(source_parquet, output_parquet, plan)
            data_timestamps = _scalar_column(output_table, "timestamp", dtype=np.float64)
            episode_stats = _numeric_episode_stats(output_table, features)

            for video_key in video_keys:
                source_relative_video = _format_dataset_path(
                    str(video_template), info, plan.source_episode_index, video_key=video_key
                )
                output_relative_video = _format_dataset_path(
                    str(video_template), info, plan.output_episode_index, video_key=video_key
                )
                source_video = source / source_relative_video
                output_video = staging / output_relative_video
                video_feature = features[video_key]
                pixel_format = str(video_feature.get("info", {}).get("video.pix_fmt", "yuv420p"))
                print(f"  Encoding {video_key}")
                _encode_trimmed_video(
                    source_video,
                    output_video,
                    plan,
                    ffmpeg=ffmpeg,
                    fps=fps,
                    codec=video_codec,
                    pixel_format=pixel_format,
                    preset=video_preset,
                    crf=video_crf,
                )
                video_stats, video_timestamps = _video_stats_and_timestamps(output_video, plan.new_length)
                episode_stats[video_key] = video_stats
                validation_key = (
                    f"source_{plan.source_episode_index:06d}_to_output_{plan.output_episode_index:06d}/{video_key}"
                )
                video_validation[validation_key] = _validate_video_timeline(
                    output_video, video_timestamps, data_timestamps, fps
                )

            new_episode = dict(source_episode)
            new_episode["episode_index"] = plan.output_episode_index
            new_episode["length"] = plan.new_length
            new_episodes.append(new_episode)
            episode_stats_rows.append({"episode_index": plan.output_episode_index, "stats": episode_stats})

        new_info = dict(info)
        new_info["total_episodes"] = len(plans)
        new_info["total_frames"] = sum(plan.new_length for plan in plans)
        new_info["total_videos"] = len(plans) * len(video_keys)
        chunks_size = int(new_info.get("chunks_size", 1000))
        new_info["total_chunks"] = math.ceil(len(plans) / chunks_size)
        new_info["splits"] = {"train": f"0:{len(plans)}"}

        _write_json(staging / "meta" / "info.json", new_info, indent=4)
        _write_jsonlines(staging / "meta" / "episodes.jsonl", new_episodes)
        _write_jsonlines(staging / "meta" / "episodes_stats.jsonl", episode_stats_rows)
        _validate_output(staging, new_info, new_episodes, plans)

        final_snapshot = _dataset_file_snapshot(source)
        if final_snapshot != source_snapshot:
            raise RuntimeError("Source dataset changed while trimming; refusing to publish the output")

        provenance = {
            "schema_version": 1,
            "operation": "trim_lerobot_episode_prefix_by_timestamp",
            "created_at_utc": datetime.now(UTC).isoformat(),
            "source_dataset": str(source),
            "output_dataset": str(output),
            "source_unchanged": True,
            "semantics": {
                "removed_interval": "[episode_start, trim_start_s)",
                "first_kept_frame": "first source frame whose episode-local elapsed timestamp is >= trim_start_s",
                "timestamp": "subtract the first kept source timestamp; no resampling",
                "frame_index": "rebuilt from zero per episode",
                "index": "rebuilt contiguously across the output dataset",
                "episode_index": "mapped from source_episode_index to output_episode_index as recorded per episode",
                "task_labels": "preserved",
                "end_trim": "not performed",
            },
            "video_encoding": {
                "codec": video_codec,
                "preset": video_preset,
                "crf": video_crf,
                "fps": fps,
                "frame_selection": "same half-open source frame range as Parquet",
            },
            "episodes": [asdict(plan) for plan in plans],
            "source_file_sha256": source_hashes,
            "video_validation": video_validation,
        }
        _write_json(staging / "meta" / "trim_provenance.json", provenance, indent=2)
        staging.rename(output)
        committed = True
    finally:
        if not committed and staging.exists():
            shutil.rmtree(staging)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", type=Path, help="Read-only source LeRobot dataset root")
    parser.add_argument("output", type=Path, help="New output dataset directory (must not already exist)")
    parser.add_argument(
        "--episode-id",
        type=int,
        help="Quick mode: trim only this source episode and emit it as episode 0 in a standalone dataset",
    )
    trim_group = parser.add_mutually_exclusive_group(required=True)
    trim_group.add_argument(
        "--trim-start-s",
        "--timestamp-s",
        dest="trim_start_s",
        type=float,
        help="Episode-local timestamp in seconds at which retained data begins",
    )
    trim_group.add_argument(
        "--trim-manifest",
        type=Path,
        help="JSON or CSV per-episode trim mapping; omitted episodes use 0 seconds",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and print exact frame ranges only")
    parser.add_argument("--ffmpeg", default="ffmpeg", help="FFmpeg executable used for frame-accurate re-encoding")
    parser.add_argument("--video-codec", default="libx264")
    parser.add_argument("--video-preset", default="medium")
    parser.add_argument("--video-crf", type=int, default=18)
    parser.add_argument(
        "--skip-source-hashes",
        action="store_true",
        help="Skip SHA-256 provenance hashes (source immutability is still checked by file metadata)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        source, output = _validate_paths(args.source, args.output, dry_run=args.dry_run)
        info, episodes, episode_indices = _load_dataset(source)
        selected_episodes = episodes
        reindex_output_episodes = False
        if args.episode_id is not None:
            if args.trim_manifest is not None:
                raise ValueError("--episode-id cannot be combined with --trim-manifest; pass --timestamp-s")
            if args.episode_id not in episode_indices:
                raise ValueError(f"Episode {args.episode_id} not found; available episodes: {episode_indices}")
            selected_episodes = [episodes[args.episode_id]]
            reindex_output_episodes = True

        if args.trim_manifest is not None:
            trim_by_episode = _load_trim_manifest(args.trim_manifest)
            unknown = sorted(set(trim_by_episode) - set(episode_indices))
            if unknown:
                raise ValueError(f"Trim manifest refers to missing episodes: {unknown}")
        else:
            trim = _validate_trim_value(args.trim_start_s, context="--trim-start-s")
            trim_by_episode = dict.fromkeys(episode_indices, trim)

        plans = _build_plans(
            source,
            info,
            selected_episodes,
            trim_by_episode,
            reindex_output_episodes=reindex_output_episodes,
        )
        _print_plan(plans, fps=float(info["fps"]), dry_run=args.dry_run)
        if args.dry_run:
            print("Dry run complete; no files were written.")
            return 0

        if shutil.which(args.ffmpeg) is None:
            raise FileNotFoundError(f"FFmpeg executable was not found: {args.ffmpeg}")
        source_snapshot = _dataset_file_snapshot(source)
        relative_paths = _provenance_source_files(source_snapshot, info, plans)
        source_hashes = {} if args.skip_source_hashes else _source_hashes(source, relative_paths)
        _build_output(
            source,
            output,
            info,
            selected_episodes,
            plans,
            ffmpeg=args.ffmpeg,
            video_codec=args.video_codec,
            video_preset=args.video_preset,
            video_crf=args.video_crf,
            source_snapshot=source_snapshot,
            source_hashes=source_hashes,
        )
        print(f"Created trimmed dataset: {output}")
        print(f"Provenance: {output / 'meta' / 'trim_provenance.json'}")
        return 0
    except (FileNotFoundError, NotADirectoryError, FileExistsError, ValueError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
