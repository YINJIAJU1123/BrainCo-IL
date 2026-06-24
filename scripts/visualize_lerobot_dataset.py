#!/usr/bin/env python3
"""Audit a LeRobot v2.x dataset and visualize it with Rerun.

The script intentionally reads the simple on-disk LeRobot representation directly
(JSON + Parquet + MP4).  It therefore works without downloading a dataset through
the Hugging Face Hub and produces both a JSON quality report and an optional RRD.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from dataclasses import dataclass
from datetime import UTC
from datetime import datetime
import io
import json
import math
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np


@dataclass(frozen=True)
class QualityConfig:
    """Thresholds used by the quality checks (all angles are in radians)."""

    timestamp_tolerance_s: float = 0.005
    sync_tolerance_s: float = 0.02
    max_start_idle_s: float = 0.30
    max_end_idle_s: float = 0.30
    min_motion_velocity: float = 0.02
    max_joint_step: float = 0.35
    max_joint_velocity: float = 6.0
    max_joint_acceleration: float = 100.0
    home_tolerance: float = 0.10
    trim_padding_frames: int = 2


@dataclass
class Issue:
    code: str
    severity: str
    message: str
    episode_index: int
    key: str | None = None
    frame_start: int | None = None
    frame_end: int | None = None
    time_start_s: float | None = None
    time_end_s: float | None = None
    details: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {key: value for key, value in asdict(self).items() if value is not None}


@dataclass
class EpisodeAudit:
    episode_index: int
    parquet_path: Path
    timestamps: np.ndarray
    state: np.ndarray
    action: np.ndarray | None
    state_key: str
    action_key: str | None
    vector_dimensions: list[int]
    frame_indices: np.ndarray
    dataset_indices: np.ndarray
    table: Any
    issues: list[Issue]
    metrics: dict[str, Any]
    suggested_trim: dict[str, int] | None

    def report_dict(self) -> dict[str, Any]:
        duration = float(self.timestamps[-1] - self.timestamps[0]) if len(self.timestamps) > 1 else 0.0
        return {
            "episode_index": self.episode_index,
            "parquet_path": str(self.parquet_path),
            "length": len(self.timestamps),
            "duration_s": duration,
            "state_key": self.state_key,
            "action_key": self.action_key,
            "vector_dimensions": self.vector_dimensions,
            "metrics": _json_safe(self.metrics),
            "suggested_trim": self.suggested_trim,
            "issues": [issue.to_dict() for issue in self.issues],
        }


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _parse_episode_selection(value: str | None, available: list[int]) -> list[int]:
    if not value:
        return available
    selected: set[int] = set()
    for raw_part in value.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", maxsplit=1)
            start, end = int(start_text), int(end_text)
            if end < start:
                raise ValueError(f"Invalid episode range: {part}")
            selected.update(range(start, end + 1))
        else:
            selected.add(int(part))
    missing = sorted(selected - set(available))
    if missing:
        raise ValueError(f"Episodes not found: {missing}; available: {available[:20]}")
    return [index for index in available if index in selected]


def _parse_dimension_selection(value: str | None, dimension: int) -> list[int]:
    if not value:
        return list(range(dimension))
    selected: list[int] = []
    for raw_part in value.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if ":" in part:
            start_text, end_text = part.split(":", maxsplit=1)
            start = int(start_text) if start_text else 0
            end = int(end_text) if end_text else dimension
            if start < 0 or end > dimension or end <= start:
                raise ValueError(f"Invalid dimension slice {part!r} for a {dimension}D vector")
            selected.extend(range(start, end))
        else:
            index = int(part)
            if index < 0 or index >= dimension:
                raise ValueError(f"Dimension {index} is out of range for a {dimension}D vector")
            selected.append(index)
    if not selected:
        raise ValueError("Dimension selection is empty")
    if len(selected) != len(set(selected)):
        raise ValueError(f"Dimension selection contains duplicates: {value}")
    return selected


def _read_jsonlines(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _discover_episode_indices(root: Path, info: dict[str, Any]) -> list[int]:
    episodes = _read_jsonlines(root / "meta" / "episodes.jsonl")
    if episodes:
        return sorted(int(episode["episode_index"]) for episode in episodes)

    indices = []
    for path in root.glob("data/**/*.parquet"):
        match = re.search(r"episode_(\d+)\.parquet$", path.name)
        if match:
            indices.append(int(match.group(1)))
    if indices:
        return sorted(indices)
    total = int(info.get("total_episodes", 0))
    return list(range(total))


def _format_dataset_path(template: str, info: dict[str, Any], episode_index: int, **extra: Any) -> Path:
    chunks_size = int(info.get("chunks_size", 1000))
    values = {
        "episode_chunk": episode_index // chunks_size,
        "episode_index": episode_index,
        **extra,
    }
    return Path(template.format(**values))


def _column_to_numpy(table: Any, key: str, *, dtype: Any = np.float64) -> np.ndarray:
    values = table.column(key).combine_chunks().to_pylist()
    try:
        array = np.asarray(values, dtype=dtype)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Column {key!r} is not a regular numeric array") from error
    if array.ndim == 1:
        array = array[:, None]
    return array


def _scalar_column(table: Any, key: str, *, dtype: Any = np.float64) -> np.ndarray:
    array = _column_to_numpy(table, key, dtype=dtype)
    if array.shape[1:] != (1,):
        raise ValueError(f"Column {key!r} is not scalar; shape={array.shape}")
    return array[:, 0]


def _is_numeric_vector_feature(feature: dict[str, Any]) -> bool:
    dtype = str(feature.get("dtype", ""))
    shape = feature.get("shape", [])
    return dtype not in {"image", "video", "string"} and len(shape) == 1 and int(shape[0]) > 1


def _choose_vector_key(
    features: dict[str, dict[str, Any]],
    requested: str | None,
    candidates: tuple[str, ...],
    contains: str,
    kind: str,
) -> str:
    if requested:
        if requested not in features:
            raise ValueError(f"Requested {kind} key {requested!r} is not present")
        return requested
    for key in candidates:
        if key in features and _is_numeric_vector_feature(features[key]):
            return key
    matches = [
        key for key, feature in features.items() if contains in key.lower() and _is_numeric_vector_feature(feature)
    ]
    if len(matches) == 1:
        return matches[0]
    available = [key for key, feature in features.items() if _is_numeric_vector_feature(feature)]
    raise ValueError(f"Could not uniquely discover {kind} key; candidates={matches or available}. Pass --{kind}-key.")


def _make_issue(
    code: str,
    severity: str,
    message: str,
    episode_index: int,
    timestamps: np.ndarray,
    *,
    key: str | None = None,
    frame_start: int | None = None,
    frame_end: int | None = None,
    details: dict[str, Any] | None = None,
) -> Issue:
    def frame_time(frame: int | None) -> float | None:
        if frame is None or not len(timestamps):
            return None
        return float(timestamps[min(max(frame, 0), len(timestamps) - 1)])

    return Issue(
        code=code,
        severity=severity,
        message=message,
        episode_index=episode_index,
        key=key,
        frame_start=frame_start,
        frame_end=frame_end,
        time_start_s=frame_time(frame_start),
        time_end_s=frame_time(frame_end),
        details=details,
    )


def _valid_dt(timestamps: np.ndarray, fps: float) -> np.ndarray:
    dt = np.diff(timestamps).astype(np.float64)
    fallback = 1.0 / fps
    return np.where(np.isfinite(dt) & (dt > 0), dt, fallback)


def _transition_metrics(values: np.ndarray, timestamps: np.ndarray, fps: float) -> dict[str, np.ndarray | float]:
    if len(values) < 2:
        empty = np.empty((0, values.shape[1]), dtype=np.float64)
        return {
            "step": empty,
            "velocity": empty,
            "acceleration": empty,
            "max_step": 0.0,
            "max_velocity": 0.0,
            "max_acceleration": 0.0,
        }
    dt = _valid_dt(timestamps, fps)
    step = np.diff(values.astype(np.float64), axis=0)
    velocity = step / dt[:, None]
    if len(velocity) > 1:
        velocity_dt = (dt[:-1] + dt[1:]) / 2.0
        acceleration = np.diff(velocity, axis=0) / velocity_dt[:, None]
    else:
        acceleration = np.empty((0, values.shape[1]), dtype=np.float64)
    return {
        "step": step,
        "velocity": velocity,
        "acceleration": acceleration,
        "max_step": float(np.nanmax(np.abs(step))) if step.size else 0.0,
        "max_velocity": float(np.nanmax(np.abs(velocity))) if velocity.size else 0.0,
        "max_acceleration": float(np.nanmax(np.abs(acceleration))) if acceleration.size else 0.0,
    }


def _largest_location(values: np.ndarray) -> tuple[int, int, float]:
    safe = np.nan_to_num(np.abs(values), nan=-1.0)
    flat_index = int(np.argmax(safe))
    frame, dimension = np.unravel_index(flat_index, safe.shape)
    return int(frame), int(dimension), float(safe[frame, dimension])


def _check_kinematics(
    values: np.ndarray,
    timestamps: np.ndarray,
    fps: float,
    episode_index: int,
    key: str,
    config: QualityConfig,
) -> tuple[list[Issue], dict[str, float], dict[str, np.ndarray | float]]:
    issues: list[Issue] = []
    metrics = _transition_metrics(values, timestamps, fps)
    public_metrics = {name: float(metrics[name]) for name in ("max_step", "max_velocity", "max_acceleration")}

    checks = (
        ("step", config.max_joint_step, "joint_step_jump", "angle step", 1),
        ("velocity", config.max_joint_velocity, "velocity_limit", "velocity", 1),
        ("acceleration", config.max_joint_acceleration, "velocity_jump", "acceleration", 2),
    )
    for metric_name, threshold, code, label, frame_offset in checks:
        array = metrics[metric_name]
        if not isinstance(array, np.ndarray) or not array.size or float(metrics[f"max_{metric_name}"]) <= threshold:
            continue
        frame, dimension, value = _largest_location(array)
        issue_frame = frame + frame_offset
        issues.append(
            _make_issue(
                code,
                "error",
                f"{key} {label} jump: dim {dimension}, {value:.4g} > {threshold:.4g}",
                episode_index,
                timestamps,
                key=key,
                frame_start=max(0, issue_frame - 1),
                frame_end=issue_frame,
                details={"dimension": dimension, "observed": value, "threshold": threshold},
            )
        )
    return issues, public_metrics, metrics


def _check_boundary_idle(
    motion_values: np.ndarray,
    timestamps: np.ndarray,
    fps: float,
    episode_index: int,
    key: str,
    config: QualityConfig,
) -> tuple[list[Issue], dict[str, float | int], dict[str, int] | None]:
    issues: list[Issue] = []
    if len(motion_values) < 2:
        issues.append(_make_issue("too_short", "error", "Episode has fewer than two frames", episode_index, timestamps))
        return issues, {"start_idle_s": 0.0, "end_idle_s": 0.0, "moving_transitions": 0}, None

    velocity = _transition_metrics(motion_values, timestamps, fps)["velocity"]
    assert isinstance(velocity, np.ndarray)
    speed = np.nanmax(np.abs(velocity), axis=1)
    moving = np.flatnonzero(speed >= config.min_motion_velocity)
    if not len(moving):
        issues.append(
            _make_issue(
                "episode_static",
                "error",
                f"No motion reaches {config.min_motion_velocity:.4g} rad/s",
                episode_index,
                timestamps,
                key=key,
                frame_start=0,
                frame_end=len(timestamps) - 1,
            )
        )
        return (
            issues,
            {"start_idle_s": float(timestamps[-1] - timestamps[0]), "end_idle_s": 0.0, "moving_transitions": 0},
            None,
        )

    first_transition = int(moving[0])
    last_transition = int(moving[-1])
    start_idle_s = float(timestamps[first_transition] - timestamps[0])
    end_idle_s = float(timestamps[-1] - timestamps[last_transition + 1])
    keep_start = max(0, first_transition - config.trim_padding_frames)
    keep_end = min(len(timestamps) - 1, last_transition + 1 + config.trim_padding_frames)

    if start_idle_s > config.max_start_idle_s:
        issues.append(
            _make_issue(
                "start_idle",
                "error",
                f"Episode starts with {start_idle_s:.3f}s of idle frames",
                episode_index,
                timestamps,
                key=key,
                frame_start=0,
                frame_end=first_transition,
                details={"observed_s": start_idle_s, "threshold_s": config.max_start_idle_s},
            )
        )
    if end_idle_s > config.max_end_idle_s:
        issues.append(
            _make_issue(
                "end_idle",
                "error",
                f"Episode ends with {end_idle_s:.3f}s of idle frames",
                episode_index,
                timestamps,
                key=key,
                frame_start=last_transition + 1,
                frame_end=len(timestamps) - 1,
                details={"observed_s": end_idle_s, "threshold_s": config.max_end_idle_s},
            )
        )
    trim = {"start_frame_inclusive": keep_start, "end_frame_inclusive": keep_end}
    metrics = {"start_idle_s": start_idle_s, "end_idle_s": end_idle_s, "moving_transitions": len(moving)}
    return issues, metrics, trim


def _check_timestamps(
    timestamps: np.ndarray, fps: float, episode_index: int, config: QualityConfig
) -> tuple[list[Issue], dict[str, float | int]]:
    issues: list[Issue] = []
    if not np.isfinite(timestamps).all():
        bad = np.flatnonzero(~np.isfinite(timestamps))
        issues.append(
            _make_issue(
                "timestamp_nonfinite",
                "error",
                f"Timestamp contains {len(bad)} NaN/Inf values",
                episode_index,
                timestamps,
                frame_start=int(bad[0]),
                frame_end=int(bad[-1]),
            )
        )
    dt = np.diff(timestamps)
    expected_dt = 1.0 / fps
    non_increasing = np.flatnonzero(dt <= 0)
    irregular = np.flatnonzero(np.abs(dt - expected_dt) > config.timestamp_tolerance_s)
    dropped = np.flatnonzero(dt > expected_dt + config.timestamp_tolerance_s)
    if len(non_increasing):
        frame = int(non_increasing[0] + 1)
        issues.append(
            _make_issue(
                "timestamp_non_increasing",
                "error",
                f"Timestamp is duplicate or moves backwards at {len(non_increasing)} transition(s)",
                episode_index,
                timestamps,
                frame_start=frame - 1,
                frame_end=frame,
            )
        )
    if len(dropped):
        frame = int(dropped[np.argmax(dt[dropped])] + 1)
        issues.append(
            _make_issue(
                "dropped_sample",
                "error",
                f"Detected {len(dropped)} timestamp gap(s) larger than the {fps:g} Hz period",
                episode_index,
                timestamps,
                frame_start=frame - 1,
                frame_end=frame,
                details={"largest_gap_s": float(np.max(dt[dropped])), "expected_period_s": expected_dt},
            )
        )
    elif len(irregular):
        frame = int(irregular[np.argmax(np.abs(dt[irregular] - expected_dt))] + 1)
        issues.append(
            _make_issue(
                "timestamp_jitter",
                "warning",
                f"Detected {len(irregular)} sampling interval(s) outside timestamp tolerance",
                episode_index,
                timestamps,
                frame_start=frame - 1,
                frame_end=frame,
            )
        )
    return issues, {
        "expected_period_s": expected_dt,
        "min_period_s": float(np.min(dt)) if len(dt) else 0.0,
        "max_period_s": float(np.max(dt)) if len(dt) else 0.0,
        "irregular_intervals": len(irregular),
        "dropped_intervals": len(dropped),
    }


def _infer_timestamp_scale(values: np.ndarray, fps: float) -> float:
    positive_dt = np.diff(values)
    positive_dt = positive_dt[np.isfinite(positive_dt) & (positive_dt > 0)]
    if not len(positive_dt):
        return 1.0
    median_dt = float(np.median(positive_dt))
    expected = 1.0 / fps
    scales = (1.0, 1e-3, 1e-6, 1e-9)
    return min(scales, key=lambda scale: abs(math.log10(max(median_dt * scale, 1e-15) / expected)))


def _check_auxiliary_timestamps(
    table: Any,
    main_timestamps: np.ndarray,
    fps: float,
    episode_index: int,
    config: QualityConfig,
) -> tuple[list[Issue], dict[str, Any], list[str]]:
    issues: list[Issue] = []
    metrics: dict[str, Any] = {}
    timestamp_keys: list[str] = []
    for key in table.column_names:
        if key == "timestamp" or "timestamp" not in key.lower():
            continue
        try:
            raw = _scalar_column(table, key)
        except ValueError:
            continue
        timestamp_keys.append(key)
        scale = _infer_timestamp_scale(raw, fps)
        values = raw * scale
        difference = values - main_timestamps
        median_offset = float(np.nanmedian(difference))
        residual = difference - median_offset
        max_residual = float(np.nanmax(np.abs(residual)))
        metrics[key] = {
            "unit_scale_to_seconds": scale,
            "median_offset_s": median_offset,
            "max_alignment_jitter_s": max_residual,
        }
        if max_residual > config.sync_tolerance_s:
            frame = int(np.nanargmax(np.abs(residual)))
            issues.append(
                _make_issue(
                    "sensor_timestamp_misaligned",
                    "error",
                    f"{key} differs from the main timeline by up to {max_residual:.4g}s after offset removal",
                    episode_index,
                    main_timestamps,
                    key=key,
                    frame_start=frame,
                    frame_end=frame,
                    details={"max_jitter_s": max_residual, "median_offset_s": median_offset},
                )
            )
        if abs(median_offset) > config.sync_tolerance_s and abs(median_offset) < 60.0:
            issues.append(
                _make_issue(
                    "sensor_timestamp_offset",
                    "warning",
                    f"{key} has a median offset of {median_offset:.4g}s from the main timeline",
                    episode_index,
                    main_timestamps,
                    key=key,
                    details={"median_offset_s": median_offset},
                )
            )
    return issues, metrics, timestamp_keys


def _check_finite_vector(values: np.ndarray, timestamps: np.ndarray, episode_index: int, key: str) -> list[Issue]:
    if np.isfinite(values).all():
        return []
    bad = np.argwhere(~np.isfinite(values))
    first_frame, first_dimension = (int(item) for item in bad[0])
    return [
        _make_issue(
            "nonfinite_value",
            "error",
            f"{key} contains {len(bad)} NaN/Inf value(s)",
            episode_index,
            timestamps,
            key=key,
            frame_start=first_frame,
            frame_end=first_frame,
            details={"first_dimension": first_dimension},
        )
    ]


def _decode_video(
    path: Path,
    *,
    images: bool,
    image_stride: int = 1,
    image_max_width: int | None = None,
):
    try:
        import av  # noqa: PLC0415
    except ImportError as error:
        raise RuntimeError("Video checking requires PyAV. Install the project environment first.") from error

    with av.open(str(path), "r") as container:
        stream = container.streams.video[0]
        for frame_index, decoded_frame in enumerate(container.decode(stream)):
            timestamp = (
                float(decoded_frame.pts * decoded_frame.time_base) if decoded_frame.pts is not None else float("nan")
            )
            image = None
            if images and frame_index % image_stride == 0:
                output_frame = decoded_frame
                if image_max_width and output_frame.width > image_max_width:
                    height = round(output_frame.height * image_max_width / output_frame.width)
                    output_frame = output_frame.reformat(width=image_max_width, height=height)
                image = output_frame.to_ndarray(format="rgb24")
            yield frame_index, timestamp, image


def _check_image_column(
    table: Any,
    root: Path,
    camera_key: str,
    timestamps: np.ndarray,
    episode_index: int,
) -> tuple[list[Issue], dict[str, Any]]:
    if camera_key not in table.column_names:
        issue = _make_issue(
            "camera_column_missing",
            "error",
            f"Missing image column for {camera_key}",
            episode_index,
            timestamps,
            key=camera_key,
        )
        return [issue], {"frame_count": 0, "expected_frame_count": len(timestamps)}

    values = table.column(camera_key).combine_chunks().to_pylist()
    missing: list[int] = []
    for index, raw_value in enumerate(values):
        value = raw_value
        if isinstance(raw_value, dict):
            if raw_value.get("bytes") is not None:
                continue
            value = raw_value.get("path")
        if isinstance(value, str):
            path = Path(value)
            path = path if path.is_absolute() else root / path
            if path.is_file():
                continue
        missing.append(index)

    issues: list[Issue] = []
    if missing:
        issues.append(
            _make_issue(
                "camera_frame_missing",
                "error",
                f"{camera_key} is missing {len(missing)} of {len(timestamps)} image frame(s)",
                episode_index,
                timestamps,
                key=camera_key,
                frame_start=missing[0],
                frame_end=missing[-1],
                details={"missing_frames": missing[:100]},
            )
        )
    return issues, {
        "frame_count": len(values) - len(missing),
        "expected_frame_count": len(timestamps),
        "missing_frames": len(missing),
    }


def _check_video(
    path: Path,
    camera_key: str,
    timestamps: np.ndarray,
    fps: float,
    episode_index: int,
    config: QualityConfig,
) -> tuple[list[Issue], dict[str, Any], np.ndarray]:
    issues: list[Issue] = []
    if not path.is_file():
        issues.append(
            _make_issue(
                "camera_file_missing",
                "error",
                f"Missing video for {camera_key}: {path}",
                episode_index,
                timestamps,
                key=camera_key,
            )
        )
        return issues, {"path": str(path), "frame_count": 0}, np.empty(0)
    try:
        video_timestamps = np.asarray([item[1] for item in _decode_video(path, images=False)], dtype=np.float64)
    except Exception as error:  # Decoder errors vary by codec/FFmpeg version.
        issues.append(
            _make_issue(
                "camera_decode_failed",
                "error",
                f"Could not decode {camera_key}: {error}",
                episode_index,
                timestamps,
                key=camera_key,
            )
        )
        return issues, {"path": str(path), "frame_count": 0}, np.empty(0)

    expected_count = len(timestamps)
    if len(video_timestamps) != expected_count:
        issues.append(
            _make_issue(
                "camera_frame_count",
                "error",
                f"{camera_key} has {len(video_timestamps)} frames, expected {expected_count}",
                episode_index,
                timestamps,
                key=camera_key,
                details={"observed": len(video_timestamps), "expected": expected_count},
            )
        )

    dt = np.diff(video_timestamps)
    expected_dt = 1.0 / fps
    gaps = np.flatnonzero(dt > expected_dt + config.timestamp_tolerance_s)
    if len(gaps):
        frame = int(gaps[np.argmax(dt[gaps])] + 1)
        issues.append(
            _make_issue(
                "camera_dropped_frame",
                "error",
                f"{camera_key} has {len(gaps)} video timestamp gap(s)",
                episode_index,
                timestamps,
                key=camera_key,
                frame_start=max(0, frame - 1),
                frame_end=min(frame, len(timestamps) - 1),
                details={"largest_gap_s": float(np.max(dt[gaps])), "expected_period_s": expected_dt},
            )
        )

    compared = min(len(video_timestamps), expected_count)
    max_alignment = 0.0
    if compared:
        relative_video = video_timestamps[:compared] - video_timestamps[0]
        relative_data = timestamps[:compared] - timestamps[0]
        residual = relative_video - relative_data
        max_alignment = float(np.nanmax(np.abs(residual)))
        if max_alignment > config.sync_tolerance_s:
            frame = int(np.nanargmax(np.abs(residual)))
            issues.append(
                _make_issue(
                    "camera_timestamp_misaligned",
                    "error",
                    f"{camera_key} video PTS differs from data timestamps by up to {max_alignment:.4g}s",
                    episode_index,
                    timestamps,
                    key=camera_key,
                    frame_start=frame,
                    frame_end=frame,
                    details={"max_alignment_error_s": max_alignment},
                )
            )
    metrics = {
        "path": str(path),
        "frame_count": len(video_timestamps),
        "expected_frame_count": expected_count,
        "max_pts_alignment_error_s": max_alignment,
        "timestamp_gaps": len(gaps),
    }
    return issues, metrics, video_timestamps


def _load_home_reference(value: str | None) -> np.ndarray | None:
    if not value:
        return None
    path = Path(value).expanduser()
    text = path.read_text() if path.is_file() else value
    parsed = json.loads(text)
    if isinstance(parsed, dict):
        parsed = parsed.get("home", parsed.get("position"))
    array = np.asarray(parsed, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError("Home reference must be a JSON list (or {'home': [...]})")
    return array


def _apply_home_check(
    audits: list[EpisodeAudit], reference: np.ndarray | None, config: QualityConfig
) -> tuple[np.ndarray, str]:
    initial = np.stack([audit.state[0] for audit in audits])
    source = "provided"
    if reference is None:
        reference = np.median(initial, axis=0)
        source = "coordinate_median_of_selected_episodes"
    if reference.shape != (initial.shape[1],):
        raise ValueError(f"Home reference has shape {reference.shape}, expected {(initial.shape[1],)}")
    for audit, position in zip(audits, initial, strict=True):
        difference = np.abs(position - reference)
        maximum = float(np.max(difference))
        dimension = int(np.argmax(difference))
        audit.metrics["home"] = {"max_abs_error": maximum, "worst_dimension": dimension}
        if maximum > config.home_tolerance:
            audit.issues.append(
                _make_issue(
                    "home_position_mismatch",
                    "error",
                    f"Initial state differs from home by {maximum:.4g} rad at dim {dimension}",
                    audit.episode_index,
                    audit.timestamps,
                    key=audit.state_key,
                    frame_start=0,
                    frame_end=0,
                    details={"dimension": dimension, "observed": maximum, "threshold": config.home_tolerance},
                )
            )
    return reference, source


def _load_episode(
    root: Path,
    info: dict[str, Any],
    episode_index: int,
    state_key: str,
    action_key: str | None,
    fps: float,
    config: QualityConfig,
    *,
    vector_dimensions: list[int],
    expected_state_dim: int | None,
    expected_action_dim: int | None,
    skip_video_checks: bool,
) -> EpisodeAudit:
    try:
        import pyarrow.parquet as parquet  # noqa: PLC0415
    except ImportError as error:
        raise RuntimeError("Reading LeRobot data requires pyarrow. Install the project environment first.") from error

    data_template = info.get("data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet")
    parquet_path = root / _format_dataset_path(data_template, info, episode_index)
    if not parquet_path.is_file():
        raise FileNotFoundError(parquet_path)
    table = parquet.read_table(parquet_path)
    timestamps = _scalar_column(table, "timestamp")
    raw_state = _column_to_numpy(table, state_key)
    raw_action = _column_to_numpy(table, action_key) if action_key else None
    if max(vector_dimensions) >= raw_state.shape[1]:
        raise ValueError(f"Episode {episode_index}: selected dimension exceeds state shape {raw_state.shape}")
    if raw_action is not None and max(vector_dimensions) >= raw_action.shape[1]:
        raise ValueError(f"Episode {episode_index}: selected dimension exceeds action shape {raw_action.shape}")
    state = raw_state[:, vector_dimensions]
    action = raw_action[:, vector_dimensions] if raw_action is not None else None
    frame_indices = (
        _scalar_column(table, "frame_index", dtype=np.int64)
        if "frame_index" in table.column_names
        else np.arange(len(timestamps))
    )
    dataset_indices = (
        _scalar_column(table, "index", dtype=np.int64) if "index" in table.column_names else frame_indices.copy()
    )
    issues: list[Issue] = []
    metrics: dict[str, Any] = {
        "dimensions": {
            "raw_state": raw_state.shape[1],
            "raw_action": raw_action.shape[1] if raw_action is not None else None,
            "selected": len(vector_dimensions),
        }
    }

    if len(state) != len(timestamps) or (action is not None and len(action) != len(timestamps)):
        raise ValueError(f"Episode {episode_index}: state/action row count does not match timestamps")
    if expected_state_dim is not None and state.shape[1] != expected_state_dim:
        issues.append(
            _make_issue(
                "state_dimension",
                "error",
                f"State dim is {state.shape[1]}, expected {expected_state_dim}",
                episode_index,
                timestamps,
                key=state_key,
            )
        )
    if action is not None and expected_action_dim is not None and action.shape[1] != expected_action_dim:
        issues.append(
            _make_issue(
                "action_dimension",
                "error",
                f"Action dim is {action.shape[1]}, expected {expected_action_dim}",
                episode_index,
                timestamps,
                key=action_key,
            )
        )

    timestamp_issues, metrics["timestamp"] = _check_timestamps(timestamps, fps, episode_index, config)
    issues.extend(timestamp_issues)
    issues.extend(_check_finite_vector(state, timestamps, episode_index, state_key))
    state_issues, metrics["state"], _ = _check_kinematics(state, timestamps, fps, episode_index, state_key, config)
    issues.extend(state_issues)
    idle_issues, metrics["boundary_idle"], suggested_trim = _check_boundary_idle(
        state, timestamps, fps, episode_index, state_key, config
    )
    issues.extend(idle_issues)

    if action is not None:
        issues.extend(_check_finite_vector(action, timestamps, episode_index, action_key or "action"))
        action_issues, metrics["action"], _ = _check_kinematics(
            action, timestamps, fps, episode_index, action_key or "action", config
        )
        issues.extend(action_issues)

    aux_issues, metrics["auxiliary_timestamps"], aux_keys = _check_auxiliary_timestamps(
        table, timestamps, fps, episode_index, config
    )
    issues.extend(aux_issues)
    camera_features = {
        key: feature for key, feature in info.get("features", {}).items() if feature.get("dtype") in {"video", "image"}
    }
    metrics["sensor_timestamp_coverage"] = {
        "independent_timestamp_keys": aux_keys,
        "camera_keys": list(camera_features),
        "can_prove_capture_time_alignment": bool(aux_keys),
        "note": (
            "Independent sensor timestamps are available."
            if aux_keys
            else "Only the shared LeRobot sample timestamp is present; row/video synchronization can be checked, "
            "but capture-side camera/joint timestamp alignment cannot be proven."
        ),
    }
    if camera_features and not aux_keys:
        issues.append(
            _make_issue(
                "independent_timestamps_missing",
                "warning",
                "No per-sensor timestamp columns: capture-side camera/joint alignment cannot be proven",
                episode_index,
                timestamps,
            )
        )

    metrics["cameras"] = {}
    video_template = info.get("video_path")
    for camera_key, feature in camera_features.items():
        if feature.get("dtype") == "image":
            image_issues, image_metrics = _check_image_column(table, root, camera_key, timestamps, episode_index)
            issues.extend(image_issues)
            metrics["cameras"][camera_key] = image_metrics
        elif not skip_video_checks and video_template:
            video_path = root / _format_dataset_path(video_template, info, episode_index, video_key=camera_key)
            video_issues, video_metrics, _ = _check_video(
                video_path, camera_key, timestamps, fps, episode_index, config
            )
            issues.extend(video_issues)
            metrics["cameras"][camera_key] = video_metrics

    return EpisodeAudit(
        episode_index=episode_index,
        parquet_path=parquet_path,
        timestamps=timestamps,
        state=state,
        action=action,
        state_key=state_key,
        action_key=action_key,
        vector_dimensions=vector_dimensions,
        frame_indices=frame_indices,
        dataset_indices=dataset_indices,
        table=table,
        issues=issues,
        metrics=metrics,
        suggested_trim=suggested_trim,
    )


def _dimension_names(feature: dict[str, Any], dimension: int, selection: list[int]) -> list[str]:
    names = feature.get("names")
    if isinstance(names, list) and max(selection) < len(names):
        return [str(names[index]) for index in selection]
    return [f"dim_{index:02d}" for index in range(dimension)]


def _entity_segment(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("_")
    return value or "unnamed"


def _decode_embedded_image(value: Any, root: Path) -> np.ndarray | None:
    try:
        from PIL import Image  # noqa: PLC0415
    except ImportError as error:
        raise RuntimeError("Image visualization requires Pillow") from error
    if value is None:
        return None
    if isinstance(value, dict):
        if value.get("bytes") is not None:
            return np.asarray(Image.open(io.BytesIO(value["bytes"])).convert("RGB"))
        value = value.get("path")
    if isinstance(value, str):
        path = Path(value)
        if not path.is_absolute():
            path = root / path
        if path.is_file():
            return np.asarray(Image.open(path).convert("RGB"))
    return None


def _encode_jpeg(image: np.ndarray, *, max_width: int, quality: int) -> bytes:
    try:
        from PIL import Image  # noqa: PLC0415
    except ImportError as error:
        raise RuntimeError("Image visualization requires Pillow") from error
    output = Image.fromarray(image)
    if max_width > 0 and output.width > max_width:
        height = round(output.height * max_width / output.width)
        output = output.resize((max_width, height), Image.Resampling.BILINEAR)
    buffer = io.BytesIO()
    output.save(buffer, format="JPEG", quality=quality)
    return buffer.getvalue()


def _issue_severity_by_frame(audit: EpisodeAudit) -> tuple[np.ndarray, dict[int, list[Issue]]]:
    severity = np.zeros(len(audit.timestamps), dtype=np.float64)
    events: dict[int, list[Issue]] = {}
    value_for = {"info": 0.25, "warning": 0.5, "error": 1.0}
    for issue in audit.issues:
        start = issue.frame_start if issue.frame_start is not None else 0
        end = issue.frame_end if issue.frame_end is not None else start
        start = min(max(start, 0), len(severity) - 1)
        end = min(max(end, start), len(severity) - 1)
        severity[start : end + 1] = np.maximum(severity[start : end + 1], value_for.get(issue.severity, 1.0))
        events.setdefault(start, []).append(issue)
    return severity, events


def _log_rerun(
    root: Path,
    info: dict[str, Any],
    audits: list[EpisodeAudit],
    rrd_path: Path | None,
    *,
    spawn: bool,
    image_stride: int,
    image_max_width: int,
    jpeg_quality: int,
    skip_images: bool,
) -> None:
    try:
        import rerun as rr  # noqa: PLC0415
    except ImportError as error:
        raise RuntimeError(
            "Rerun is not installed. Install the project environment (rerun-sdk is included by LeRobot)."
        ) from error

    rr.init("wuji_lerobot_data_quality", spawn=spawn)
    if rrd_path is not None:
        rrd_path.parent.mkdir(parents=True, exist_ok=True)
        rr.save(str(rrd_path))

    state_names = _dimension_names(
        info["features"][audits[0].state_key],
        audits[0].state.shape[1],
        audits[0].vector_dimensions,
    )
    action_names = (
        _dimension_names(
            info["features"][audits[0].action_key],
            audits[0].action.shape[1],
            audits[0].vector_dimensions,
        )
        if audits[0].action_key and audits[0].action is not None
        else []
    )
    camera_features = {
        key: feature for key, feature in info.get("features", {}).items() if feature.get("dtype") in {"video", "image"}
    }
    video_template = info.get("video_path")
    global_time_offset = 0.0
    next_global_frame = 0

    for audit in audits:
        relative_time = audit.timestamps - audit.timestamps[0]
        global_times = relative_time + global_time_offset
        global_frames = np.arange(next_global_frame, next_global_frame + len(audit.timestamps))
        severity, events = _issue_severity_by_frame(audit)
        state_metrics = _transition_metrics(audit.state, audit.timestamps, float(info["fps"]))
        state_velocity = state_metrics["velocity"]
        assert isinstance(state_velocity, np.ndarray)
        padded_velocity = np.vstack([np.zeros((1, audit.state.shape[1])), state_velocity])

        def time_columns(frames=global_frames, times=global_times):
            return [
                rr.TimeColumn("frame", sequence=frames),
                rr.TimeColumn("time", timestamp=times),
            ]

        rr.send_columns(
            "episode/index",
            indexes=time_columns(),
            columns=rr.Scalars.columns(scalars=np.full(len(global_frames), audit.episode_index)),
        )
        rr.send_columns(
            "quality/severity",
            indexes=time_columns(),
            columns=rr.Scalars.columns(scalars=severity),
        )
        for dimension, name in enumerate(state_names):
            segment = f"{dimension:02d}_{_entity_segment(name)}"
            rr.send_columns(
                f"state/position/{segment}",
                indexes=time_columns(),
                columns=rr.Scalars.columns(scalars=audit.state[:, dimension]),
            )
            rr.send_columns(
                f"state/velocity/{segment}",
                indexes=time_columns(),
                columns=rr.Scalars.columns(scalars=padded_velocity[:, dimension]),
            )
        if audit.action is not None:
            for dimension, name in enumerate(action_names):
                segment = f"{dimension:02d}_{_entity_segment(name)}"
                rr.send_columns(
                    f"action/{segment}",
                    indexes=time_columns(),
                    columns=rr.Scalars.columns(scalars=audit.action[:, dimension]),
                )
        for frame, frame_issues in events.items():
            rr.set_time("frame", sequence=int(global_frames[frame]))
            rr.set_time("time", timestamp=float(global_times[frame]))
            for issue in frame_issues:
                rr.log(
                    "quality/issues",
                    rr.TextLog(f"[{issue.severity.upper()}] ep={audit.episode_index} {issue.code}: {issue.message}"),
                )

        video_iterators: dict[str, Any] = {}
        if not skip_images and video_template:
            for camera_key, feature in camera_features.items():
                if feature.get("dtype") != "video":
                    continue
                path = root / _format_dataset_path(video_template, info, audit.episode_index, video_key=camera_key)
                if path.is_file():
                    video_iterators[camera_key] = iter(
                        _decode_video(
                            path,
                            images=True,
                            image_stride=image_stride,
                            image_max_width=image_max_width,
                        )
                    )

        embedded_image_columns = {
            key: audit.table.column(key).combine_chunks()
            for key, feature in camera_features.items()
            if feature.get("dtype") == "image" and key in audit.table.column_names
        }
        for local_frame, (global_frame, global_time) in enumerate(zip(global_frames, global_times, strict=True)):
            if skip_images:
                continue
            for camera_key, iterator in video_iterators.items():
                try:
                    decoded_frame, _, image = next(iterator)
                except StopIteration:
                    continue
                if decoded_frame == local_frame and image is not None:
                    rr.set_time("frame", sequence=int(global_frame))
                    rr.set_time("time", timestamp=float(global_time))
                    rr.log(
                        f"cameras/{_entity_segment(camera_key)}",
                        rr.EncodedImage(
                            contents=_encode_jpeg(image, max_width=image_max_width, quality=jpeg_quality),
                            media_type="image/jpeg",
                        ),
                    )
            if local_frame % image_stride == 0:
                for camera_key, column in embedded_image_columns.items():
                    image = _decode_embedded_image(column[local_frame].as_py(), root)
                    if image is not None:
                        rr.set_time("frame", sequence=int(global_frame))
                        rr.set_time("time", timestamp=float(global_time))
                        rr.log(
                            f"cameras/{_entity_segment(camera_key)}",
                            rr.EncodedImage(
                                contents=_encode_jpeg(image, max_width=image_max_width, quality=jpeg_quality),
                                media_type="image/jpeg",
                            ),
                        )

        step = 1.0 / float(info["fps"])
        global_time_offset = float(global_times[-1] + step) if len(global_times) else global_time_offset
        next_global_frame += len(audit.timestamps)


def _build_report(
    root: Path,
    info: dict[str, Any],
    audits: list[EpisodeAudit],
    config: QualityConfig,
    home_reference: np.ndarray,
    home_source: str,
) -> dict[str, Any]:
    issues = [issue for audit in audits for issue in audit.issues]
    severities = Counter(issue.severity for issue in issues)
    codes = Counter(issue.code for issue in issues)
    return {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "dataset_root": str(root),
        "dataset": {
            "codebase_version": info.get("codebase_version"),
            "robot_type": info.get("robot_type"),
            "fps": info.get("fps"),
            "selected_episodes": [audit.episode_index for audit in audits],
        },
        "config": asdict(config),
        "home_reference": {"source": home_source, "position": home_reference.tolist()},
        "summary": {
            "passed": severities.get("error", 0) == 0,
            "error_count": severities.get("error", 0),
            "warning_count": severities.get("warning", 0),
            "issue_codes": dict(sorted(codes.items())),
        },
        "episodes": [audit.report_dict() for audit in audits],
    }


def _print_summary(report: dict[str, Any], report_path: Path, rrd_path: Path | None) -> None:
    summary = report["summary"]
    status = "PASS" if summary["passed"] else "FAIL"
    print(
        f"[{status}] episodes={len(report['episodes'])} "
        f"errors={summary['error_count']} warnings={summary['warning_count']}"
    )
    for episode in report["episodes"]:
        errors = sum(issue["severity"] == "error" for issue in episode["issues"])
        warnings = sum(issue["severity"] == "warning" for issue in episode["issues"])
        trim = episode.get("suggested_trim")
        trim_text = f" suggested_trim={trim}" if trim else ""
        print(f"  episode {episode['episode_index']}: errors={errors} warnings={warnings}{trim_text}")
    print(f"JSON report: {report_path}")
    if rrd_path is not None:
        print(f"Rerun recording: {rrd_path}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check a LeRobot dataset for idle boundaries, frame/timestamp problems, home mismatch, and joint jumps; visualize it with Rerun."
    )
    parser.add_argument("dataset", type=Path, help="Local LeRobot dataset root (contains meta/, data/, videos/)")
    parser.add_argument("--episodes", help="Episode selection, e.g. 0,2,5-9 (default: all)")
    parser.add_argument("--state-key", help="State vector feature (default: auto-discover observation.state)")
    parser.add_argument(
        "--action-key", help="Action vector feature (default: auto-discover action; pass 'none' to disable)"
    )
    parser.add_argument(
        "--vector-dims",
        help="Dimensions to check from both state/action, e.g. 14:21,28:49,21:28,49:70 (default: all)",
    )
    parser.add_argument("--expected-state-dim", type=int)
    parser.add_argument("--expected-action-dim", type=int)
    parser.add_argument("--home-position", help="JSON list, or path to a JSON file; default: median initial state")
    parser.add_argument("--report", type=Path, default=Path("lerobot_quality_report.json"))
    parser.add_argument("--rrd", type=Path, help="Write an RRD recording to this path")
    parser.add_argument(
        "--spawn", action="store_true", help="Open the Rerun viewer (desktop only; not for SSH servers)"
    )
    parser.add_argument("--no-rerun", action="store_true", help="Only run checks and write the JSON report")
    parser.add_argument("--skip-images", action="store_true", help="Do not put camera frames into Rerun")
    parser.add_argument("--skip-video-checks", action="store_true", help="Skip MP4 frame-count and PTS checks")
    parser.add_argument("--image-stride", type=int, default=1, help="Log every Nth image to Rerun")
    parser.add_argument("--image-max-width", type=int, default=640, help="Downscale images before RRD export")
    parser.add_argument("--jpeg-quality", type=int, default=75, help="JPEG quality used inside the RRD")
    parser.add_argument("--timestamp-tolerance-ms", type=float, default=5.0)
    parser.add_argument("--sync-tolerance-ms", type=float, default=20.0)
    parser.add_argument("--max-start-idle-s", type=float, default=0.30)
    parser.add_argument("--max-end-idle-s", type=float, default=0.30)
    parser.add_argument("--min-motion-velocity", type=float, default=0.02, help="rad/s")
    parser.add_argument("--max-joint-step", type=float, default=0.35, help="rad/frame")
    parser.add_argument("--max-joint-velocity", type=float, default=6.0, help="rad/s")
    parser.add_argument("--max-joint-acceleration", type=float, default=100.0, help="rad/s^2")
    parser.add_argument("--home-tolerance", type=float, default=0.10, help="rad")
    parser.add_argument("--trim-padding-frames", type=int, default=2)
    parser.add_argument("--fail-on-error", action="store_true", help="Exit with status 2 when quality errors are found")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    root = args.dataset.expanduser().resolve()
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"Not a LeRobot dataset: missing {info_path}")
    info = json.loads(info_path.read_text())
    fps = float(info["fps"])
    if fps <= 0:
        raise ValueError(f"Invalid dataset fps: {fps}")
    available_episodes = _discover_episode_indices(root, info)
    selected_episodes = _parse_episode_selection(args.episodes, available_episodes)
    if not selected_episodes:
        raise ValueError("No episodes selected")

    features = info.get("features", {})
    state_key = _choose_vector_key(
        features,
        args.state_key,
        ("observation.state", "observation/joint_position", "joint_position", "state"),
        "state",
        "state",
    )
    action_key = None
    if args.action_key != "none":
        try:
            action_key = _choose_vector_key(features, args.action_key, ("action", "actions"), "action", "action")
        except ValueError:
            if args.action_key:
                raise
            print("Warning: no action vector discovered; action checks are disabled", file=sys.stderr)
    raw_state_dim = int(features[state_key]["shape"][0])
    vector_dimensions = _parse_dimension_selection(args.vector_dims, raw_state_dim)
    if action_key and int(features[action_key]["shape"][0]) != raw_state_dim:
        raise ValueError("--vector-dims currently requires state and action to have the same raw dimension")

    config = QualityConfig(
        timestamp_tolerance_s=args.timestamp_tolerance_ms / 1000.0,
        sync_tolerance_s=args.sync_tolerance_ms / 1000.0,
        max_start_idle_s=args.max_start_idle_s,
        max_end_idle_s=args.max_end_idle_s,
        min_motion_velocity=args.min_motion_velocity,
        max_joint_step=args.max_joint_step,
        max_joint_velocity=args.max_joint_velocity,
        max_joint_acceleration=args.max_joint_acceleration,
        home_tolerance=args.home_tolerance,
        trim_padding_frames=args.trim_padding_frames,
    )
    audits = [
        _load_episode(
            root,
            info,
            episode_index,
            state_key,
            action_key,
            fps,
            config,
            vector_dimensions=vector_dimensions,
            expected_state_dim=args.expected_state_dim,
            expected_action_dim=args.expected_action_dim,
            skip_video_checks=args.skip_video_checks,
        )
        for episode_index in selected_episodes
    ]
    home_reference, home_source = _apply_home_check(audits, _load_home_reference(args.home_position), config)
    report = _build_report(root, info, audits, config, home_reference, home_source)
    report_path = args.report.expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    if not args.no_rerun:
        if not args.rrd and not args.spawn:
            raise ValueError("Choose a Rerun output: pass --rrd FILE, --spawn, or use --no-rerun")
        if args.image_stride < 1:
            raise ValueError("--image-stride must be >= 1")
        if args.image_max_width < 1:
            raise ValueError("--image-max-width must be >= 1")
        if not 1 <= args.jpeg_quality <= 100:
            raise ValueError("--jpeg-quality must be in [1, 100]")
        _log_rerun(
            root,
            info,
            audits,
            args.rrd.expanduser().resolve() if args.rrd else None,
            spawn=args.spawn,
            image_stride=args.image_stride,
            image_max_width=args.image_max_width,
            jpeg_quality=args.jpeg_quality,
            skip_images=args.skip_images,
        )

    rrd_path = args.rrd.expanduser().resolve() if args.rrd else None
    _print_summary(report, report_path, rrd_path)
    return 2 if args.fail_on_error and not report["summary"]["passed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
