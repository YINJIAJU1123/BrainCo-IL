#!/usr/bin/env python3
"""Visually choose LeRobot episode crop ranges and optionally build a new dataset.

This is the interactive companion to ``trim_lerobot_episodes.py``.  It opens the
source LeRobot dataset read-only, plots per-episode motion signals, and lets you
click two points on the time axis:

* first click: crop start
* second click: crop end

The selected timestamps are snapped to source frame boundaries.  If OUTPUT is
provided, the script writes a new LeRobot dataset without modifying SOURCE.  It
reuses the same Parquet/video/metadata rebuilding code as ``trim_lerobot_episodes.py``.

Common workflows:

1. Single-episode visual validation::

       python scripts/interactive_trim_lerobot_episodes.py SOURCE OUTPUT --episode-id 3

   This opens episode 3, lets you click start/end, and writes a standalone
   one-episode output dataset reindexed as episode 0.

2. Pick ranges and save a manifest for final training output::

       python scripts/interactive_trim_lerobot_episodes.py SOURCE --all-episodes --manifest-out trim_ranges.json

3. Build the final combined training dataset from that range manifest::

       python scripts/interactive_trim_lerobot_episodes.py SOURCE FINAL_OUTPUT --range-manifest trim_ranges.json

   Episodes omitted from the manifest are kept unchanged unless ``--selected-only``
   is passed.

4. Preview the exact frame ranges without writing files::

       python scripts/interactive_trim_lerobot_episodes.py SOURCE FINAL_OUTPUT \\
         --range-manifest trim_ranges.json --dry-run
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from dataclasses import dataclass
from datetime import UTC
from datetime import datetime
import json
import math
import os
from pathlib import Path
import shutil
import sys
from typing import Any

import numpy as np
import pyarrow.parquet as parquet


@dataclass(frozen=True)
class EpisodeRange:
    episode_index: int
    trim_start_s: float
    trim_end_s: float
    start_frame: int
    end_frame_exclusive: int
    snapped_start_s: float
    snapped_end_s: float
    original_length: int
    kept_length: int


@dataclass(frozen=True)
class RangeTrimPlan:
    """Runtime-compatible superset of trim_lerobot_episodes.TrimPlan."""

    source_episode_index: int
    output_episode_index: int
    requested_trim_start_s: float
    requested_trim_end_s: float | None
    old_frame_start_inclusive: int
    old_frame_end_exclusive: int
    old_length: int
    new_length: int
    removed_frames: int
    removed_prefix_frames: int
    removed_suffix_frames: int
    source_timestamp_start_s: float
    source_timestamp_end_s: float
    source_elapsed_start_s: float
    source_elapsed_end_s: float
    new_global_index_start: int
    new_global_index_end_exclusive: int


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text())


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


def _write_json(path: Path, value: Any, *, indent: int | None = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as file:
        json.dump(_json_safe(value), file, indent=indent, ensure_ascii=False, allow_nan=False)
        file.write("\n")


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


def _format_dataset_path(template: str, info: dict[str, Any], episode_index: int, **extra: Any) -> Path:
    chunks_size = int(info.get("chunks_size", 1000))
    values = {
        "episode_chunk": episode_index // chunks_size,
        "episode_index": episode_index,
        **extra,
    }
    return Path(template.format(**values))


def _read_jsonlines(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _load_dataset(source: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[int]]:
    info_path = source / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"LeRobot info file not found: {info_path}")
    info = _read_json(info_path)
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


def _scalar_column(table: Any, key: str, *, dtype: Any) -> np.ndarray:
    if key not in table.column_names:
        raise ValueError(f"Required Parquet column is missing: {key}")
    values = np.asarray(table.column(key).combine_chunks().to_pylist(), dtype=dtype)
    if values.ndim != 1:
        raise ValueError(f"Expected scalar Parquet column {key!r}, got shape {values.shape}")
    return values


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


def _load_trim_writer_helpers() -> tuple[Any, Any, Any, Any]:
    from trim_lerobot_episodes import _build_output
    from trim_lerobot_episodes import _dataset_file_snapshot
    from trim_lerobot_episodes import _provenance_source_files
    from trim_lerobot_episodes import _source_hashes

    return _build_output, _dataset_file_snapshot, _provenance_source_files, _source_hashes


def _write_range_manifest(path: Path, source: Path, ranges: dict[int, EpisodeRange]) -> None:
    payload = {
        "schema_version": 1,
        "type": "lerobot_episode_trim_ranges",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source_dataset": str(source),
        "semantics": {
            "trim_start_s": "first kept source frame has episode-local elapsed timestamp >= trim_start_s",
            "trim_end_s": "last kept source frame has episode-local elapsed timestamp <= trim_end_s",
            "frame_range": "half-open source frame interval [start_frame, end_frame_exclusive)",
        },
        "episodes": [asdict(ranges[index]) for index in sorted(ranges)],
    }
    _write_json(path, payload, indent=2)


def _load_range_manifest(path: Path) -> dict[int, tuple[float, float | None]]:
    if not path.is_file():
        raise FileNotFoundError(path)

    value = _read_json(path)
    if isinstance(value, dict) and "episodes" in value:
        rows = value["episodes"]
    elif isinstance(value, dict):
        rows = []
        for episode_index, raw_range in value.items():
            if isinstance(raw_range, dict):
                rows.append(
                    {
                        "episode_index": episode_index,
                        "trim_start_s": raw_range.get("trim_start_s", raw_range.get("start_s")),
                        "trim_end_s": raw_range.get("trim_end_s", raw_range.get("end_s")),
                    }
                )
            elif isinstance(raw_range, list | tuple) and len(raw_range) == 2:
                rows.append({"episode_index": episode_index, "trim_start_s": raw_range[0], "trim_end_s": raw_range[1]})
            else:
                raise ValueError(
                    "Range manifest mappings must use [start_s, end_s] or {trim_start_s, trim_end_s} values"
                )
    elif isinstance(value, list):
        rows = value
    else:
        raise ValueError("Range manifest must be a JSON object or list")

    ranges: dict[int, tuple[float, float | None]] = {}
    for row in rows:
        if not isinstance(row, dict) or "episode_index" not in row:
            raise ValueError("Each range manifest row must contain episode_index")
        episode_index = int(row["episode_index"])
        if episode_index < 0:
            raise ValueError(f"Episode index must be non-negative: {episode_index}")
        if episode_index in ranges:
            raise ValueError(f"Duplicate episode index in range manifest: {episode_index}")
        start_s = _validate_trim_value(row.get("trim_start_s", row.get("start_s", 0.0)), context="trim_start_s")
        raw_end = row.get("trim_end_s", row.get("end_s"))
        end_s = None if raw_end is None else _validate_trim_value(raw_end, context="trim_end_s")
        if end_s is not None and end_s <= start_s:
            raise ValueError(f"Episode {episode_index}: trim_end_s must be greater than trim_start_s")
        ranges[episode_index] = (start_s, end_s)
    if not ranges:
        raise ValueError("Range manifest is empty")
    return ranges


def _parse_episode_selection(value: str | None, available: list[int]) -> list[int]:
    if not value:
        return []
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
        raise ValueError(f"Episodes not found: {missing}; available: {available}")
    return [episode_index for episode_index in available if episode_index in selected]


def _column_to_numpy(table: Any, key: str, *, dtype: Any = np.float64) -> np.ndarray:
    values = table.column(key).combine_chunks().to_pylist()
    try:
        array = np.asarray(values, dtype=dtype)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Column {key!r} is not a regular numeric array") from error
    if array.ndim == 1:
        array = array[:, None]
    return array


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
) -> str | None:
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
    return matches[0] if len(matches) == 1 else None


def _load_episode_table(
    source: Path,
    info: dict[str, Any],
    episode_index: int,
    columns: list[str] | None = None,
) -> Any:
    data_template = info.get("data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet")
    parquet_path = source / _format_dataset_path(data_template, info, episode_index)
    if not parquet_path.is_file():
        raise FileNotFoundError(parquet_path)
    return parquet.read_table(parquet_path, columns=columns)


def _robust_normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return values
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return np.zeros_like(values)
    low = float(np.percentile(finite, 5))
    high = float(np.percentile(finite, 95))
    if high <= low:
        high = float(np.max(finite))
        low = float(np.min(finite))
    if high <= low:
        return np.zeros_like(values)
    return np.clip((values - low) / (high - low), 0.0, 1.0)


def _motion_signal(vector: np.ndarray, fps: float) -> np.ndarray:
    if len(vector) < 2:
        return np.zeros(len(vector), dtype=np.float64)
    delta = np.diff(vector, axis=0, prepend=vector[[0]])
    return np.linalg.norm(delta, axis=1) * fps


def _plot_and_get_clicks(
    *,
    episode_index: int,
    elapsed: np.ndarray,
    frame_indices: np.ndarray,
    series: list[tuple[str, np.ndarray]],
    fps: float,
) -> tuple[float, float]:
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("matplotlib is required for interactive range selection") from error

    backend = plt.get_backend().lower()
    noninteractive_backends = {"agg", "cairo", "pdf", "pgf", "ps", "svg", "template"}
    if backend in noninteractive_backends or (
        sys.platform.startswith("linux") and not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY")
    ):
        raise RuntimeError(
            "No interactive matplotlib display is available. Run this click-based mode from a machine/session with "
            "GUI forwarding, or create a JSON range manifest and pass --range-manifest."
        )

    fig, ax = plt.subplots(figsize=(13, 6))
    for label, values in series:
        ax.plot(elapsed, _robust_normalize(values), linewidth=1.2, label=label)

    duration_s = float(elapsed[-1]) if len(elapsed) else 0.0
    ax.set_title(
        f"Episode {episode_index}: click START then END on this time axis\n"
        "Close the window to cancel this episode selection."
    )
    ax.set_xlabel("episode-local elapsed time (s)")
    ax.set_ylabel("normalized motion signal")
    ax.set_xlim(0, max(duration_s, 1e-9))
    ax.set_ylim(-0.05, 1.05)
    ax.grid(visible=True, alpha=0.25)
    ax.legend(loc="upper right")

    def time_to_frame(seconds: np.ndarray) -> np.ndarray:
        return seconds * fps

    def frame_to_time(frames: np.ndarray) -> np.ndarray:
        return frames / fps

    top = ax.secondary_xaxis("top", functions=(time_to_frame, frame_to_time))
    top.set_xlabel("approx. frame index")
    ax.text(
        0.01,
        0.98,
        f"frames: {int(frame_indices[0])}..{int(frame_indices[-1])} | fps={fps:g}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
    )
    fig.tight_layout()
    points = plt.ginput(2, timeout=0, show_clicks=True)
    plt.close(fig)
    if len(points) != 2:
        raise RuntimeError(f"Episode {episode_index}: selection cancelled before two clicks were recorded")
    start_s, end_s = sorted(float(point[0]) for point in points)
    start_s = max(0.0, min(start_s, duration_s))
    end_s = max(0.0, min(end_s, duration_s))
    if end_s <= start_s:
        raise ValueError(f"Episode {episode_index}: selected range is empty")
    return start_s, end_s


def _snap_range_to_frames(
    *,
    episode_index: int,
    timestamps: np.ndarray,
    requested_start_s: float,
    requested_end_s: float | None,
) -> EpisodeRange:
    if len(timestamps) == 0:
        raise ValueError(f"Episode {episode_index} is empty")
    elapsed = timestamps - timestamps[0]
    start = int(np.searchsorted(elapsed, requested_start_s, side="left"))
    end = len(elapsed) if requested_end_s is None else int(np.searchsorted(elapsed, requested_end_s, side="right"))
    if start >= len(elapsed):
        raise ValueError(
            f"Episode {episode_index}: trim_start_s={requested_start_s:g} removes every frame; "
            f"last frame is at {elapsed[-1]:.9g}s"
        )
    if end <= start:
        raise ValueError(
            f"Episode {episode_index}: selected range is empty after frame snapping ([start={start}, end={end}))"
        )
    snapped_start = float(elapsed[start])
    snapped_end = float(elapsed[end - 1])
    return EpisodeRange(
        episode_index=episode_index,
        trim_start_s=float(requested_start_s),
        trim_end_s=float(requested_end_s if requested_end_s is not None else elapsed[-1]),
        start_frame=start,
        end_frame_exclusive=end,
        snapped_start_s=snapped_start,
        snapped_end_s=snapped_end,
        original_length=len(elapsed),
        kept_length=end - start,
    )


def _select_episode_range(
    *,
    source: Path,
    info: dict[str, Any],
    episode_index: int,
    state_key: str | None,
    action_key: str | None,
    yes: bool,
) -> EpisodeRange:
    features = info.get("features", {})
    state_key = _choose_vector_key(
        features,
        state_key,
        ("observation.state", "observation/state", "state"),
        "state",
        "state",
    )
    action_key = _choose_vector_key(features, action_key, ("action", "actions"), "action", "action")
    columns = ["timestamp", "frame_index"]
    columns.extend(key for key in (state_key, action_key) if key is not None)
    table = _load_episode_table(source, info, episode_index, columns=columns)
    timestamps = _scalar_column(table, "timestamp", dtype=np.float64)
    frame_indices = _scalar_column(table, "frame_index", dtype=np.int64)
    elapsed = timestamps - timestamps[0]
    fps = float(info["fps"])

    series: list[tuple[str, np.ndarray]] = []
    if state_key is not None:
        state = _column_to_numpy(table, state_key)
        series.append((f"{state_key} velocity norm", _motion_signal(state, fps)))
    if action_key is not None:
        action = _column_to_numpy(table, action_key)
        series.append((f"{action_key} velocity norm", _motion_signal(action, fps)))
    if not series:
        series.append(("frame position", np.arange(len(timestamps), dtype=np.float64)))

    while True:
        start_s, end_s = _plot_and_get_clicks(
            episode_index=episode_index,
            elapsed=elapsed,
            frame_indices=frame_indices,
            series=series,
            fps=fps,
        )
        selected = _snap_range_to_frames(
            episode_index=episode_index,
            timestamps=timestamps,
            requested_start_s=start_s,
            requested_end_s=end_s,
        )
        _print_selected_range(selected)
        if yes:
            return selected
        answer = input("Accept this range? [Y/n/r=reselect] ").strip().lower()
        if answer in {"", "y", "yes"}:
            return selected
        if answer in {"r", "reselect"}:
            continue
        raise RuntimeError(f"Episode {episode_index}: selection rejected")


def _build_range_plans(
    source: Path,
    info: dict[str, Any],
    source_episodes: list[dict[str, Any]],
    ranges_by_episode: dict[int, tuple[float, float | None]],
    *,
    reindex_output_episodes: bool,
) -> list[RangeTrimPlan]:
    global_index = 0
    plans: list[RangeTrimPlan] = []
    for output_position, episode in enumerate(source_episodes):
        source_episode_index = int(episode["episode_index"])
        output_episode_index = output_position if reindex_output_episodes else source_episode_index
        start_s, end_s = ranges_by_episode.get(source_episode_index, (0.0, None))
        table = _load_episode_table(
            source,
            info,
            source_episode_index,
            columns=["timestamp", "frame_index", "episode_index", "index"],
        )
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

        snapped = _snap_range_to_frames(
            episode_index=source_episode_index,
            timestamps=timestamps,
            requested_start_s=start_s,
            requested_end_s=end_s,
        )
        removed_prefix = snapped.start_frame
        removed_suffix = old_length - snapped.end_frame_exclusive
        new_length = snapped.kept_length
        plans.append(
            RangeTrimPlan(
                source_episode_index=source_episode_index,
                output_episode_index=output_episode_index,
                requested_trim_start_s=start_s,
                requested_trim_end_s=end_s,
                old_frame_start_inclusive=snapped.start_frame,
                old_frame_end_exclusive=snapped.end_frame_exclusive,
                old_length=old_length,
                new_length=new_length,
                removed_frames=removed_prefix + removed_suffix,
                removed_prefix_frames=removed_prefix,
                removed_suffix_frames=removed_suffix,
                source_timestamp_start_s=float(timestamps[snapped.start_frame]),
                source_timestamp_end_s=float(timestamps[snapped.end_frame_exclusive - 1]),
                source_elapsed_start_s=snapped.snapped_start_s,
                source_elapsed_end_s=snapped.snapped_end_s,
                new_global_index_start=global_index,
                new_global_index_end_exclusive=global_index + new_length,
            )
        )
        global_index += new_length
    return plans


def _print_selected_range(selected: EpisodeRange) -> None:
    print(
        f"Episode {selected.episode_index}: clicked {selected.trim_start_s:.6f}s -> "
        f"{selected.trim_end_s:.6f}s; snapped to frames "
        f"[{selected.start_frame}:{selected.end_frame_exclusive}) "
        f"({selected.kept_length}/{selected.original_length} kept, "
        f"snapped elapsed {selected.snapped_start_s:.6f}s -> {selected.snapped_end_s:.6f}s)"
    )


def _print_range_plan(plans: list[RangeTrimPlan], *, fps: float, dry_run: bool) -> None:
    prefix = "DRY RUN - " if dry_run else ""
    print(f"{prefix}range trim plan ({len(plans)} episodes, fps={fps:g})")
    print(
        "source->output  requested_start_s  requested_end_s  keep_old_frames  "
        "removed_head  removed_tail  kept  snapped_start_s  snapped_end_s"
    )
    for plan in plans:
        requested_end = "episode_end" if plan.requested_trim_end_s is None else f"{plan.requested_trim_end_s:.6f}"
        print(
            f"{plan.source_episode_index:6d}->{plan.output_episode_index:<3d}  "
            f"{plan.requested_trim_start_s:17.6f}  "
            f"{requested_end:>15}  "
            f"[{plan.old_frame_start_inclusive}:{plan.old_frame_end_exclusive})"
            f"{plan.removed_prefix_frames:14d}{plan.removed_suffix_frames:14d}{plan.new_length:7d}  "
            f"{plan.source_elapsed_start_s:15.6f}  {plan.source_elapsed_end_s:13.6f}"
        )
    print(f"total frames: {sum(plan.old_length for plan in plans)} -> {sum(plan.new_length for plan in plans)}")


def _patch_range_provenance(output: Path, source: Path, plans: list[RangeTrimPlan]) -> None:
    provenance_path = output / "meta" / "trim_provenance.json"
    provenance = _read_json(provenance_path)
    provenance["operation"] = "trim_lerobot_episode_range_by_timestamp"
    provenance["range_selection_script"] = str(Path(__file__).name)
    provenance["source_dataset"] = str(source)
    provenance["output_dataset"] = str(output)
    provenance["semantics"] = {
        "kept_interval": "[trim_start_s, trim_end_s] after snapping to source frames",
        "frame_range": "same half-open source frame interval [old_frame_start_inclusive, old_frame_end_exclusive) "
        "for Parquet and every video stream",
        "first_kept_frame": "first source frame whose episode-local elapsed timestamp is >= trim_start_s",
        "last_kept_frame": "last source frame whose episode-local elapsed timestamp is <= trim_end_s",
        "timestamp": "subtract the first kept source timestamp; no resampling",
        "frame_index": "rebuilt from zero per episode",
        "index": "rebuilt contiguously across the output dataset",
        "episode_index": "mapped from source_episode_index to output_episode_index as recorded per episode",
        "task_labels": "preserved",
        "source_dataset": "never modified",
    }
    provenance["episodes"] = [asdict(plan) for plan in plans]
    _write_json(provenance_path, provenance, indent=2)


def _selected_source_episodes(
    episodes: list[dict[str, Any]],
    selected_episode_ids: list[int],
    *,
    include_unselected_full: bool,
) -> tuple[list[dict[str, Any]], bool]:
    if include_unselected_full:
        return episodes, False
    selected = [episodes[episode_index] for episode_index in selected_episode_ids]
    return selected, True


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "source",
        type=Path,
        metavar="SOURCE",
        help="Read-only LeRobot dataset root containing meta/, data/, and videos/",
    )
    parser.add_argument(
        "output",
        nargs="?",
        type=Path,
        metavar="OUTPUT",
        help="Optional new output dataset root; must not exist and must be outside SOURCE",
    )

    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--episode-id", type=int, metavar="ID", help="Interactively select one source episode")
    selection.add_argument(
        "--episodes",
        metavar="IDS",
        help="Interactively select episodes, e.g. '0,3,5-7'",
    )
    selection.add_argument("--all-episodes", action="store_true", help="Interactively select every episode")
    selection.add_argument(
        "--range-manifest",
        type=Path,
        metavar="PATH",
        help="Build/preview from a previously saved JSON range manifest without opening the GUI",
    )

    parser.add_argument(
        "--manifest-out",
        type=Path,
        metavar="PATH",
        help="Save the interactively selected ranges as a JSON manifest",
    )
    parser.add_argument(
        "--include-unselected-full",
        action="store_true",
        help="When interactively selecting a subset and writing OUTPUT, keep unselected episodes unchanged",
    )
    parser.add_argument(
        "--selected-only",
        action="store_true",
        help="With --range-manifest, write only manifest episodes and reindex them from zero",
    )
    parser.add_argument("--state-key", help="Numeric state feature to plot; defaults to observation.state when present")
    parser.add_argument("--action-key", help="Numeric action feature to plot; defaults to action when present")
    parser.add_argument("--yes", action="store_true", help="Accept clicked ranges without asking for terminal confirm")
    parser.add_argument("--dry-run", action="store_true", help="Print exact frame ranges; do not create OUTPUT")
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
        source = args.source.expanduser().resolve(strict=True)
        if not source.is_dir():
            raise NotADirectoryError(source)
        if args.output is not None:
            source, output = _validate_paths(source, args.output, dry_run=args.dry_run)
        else:
            output = None
            if args.dry_run:
                print("--dry-run has no effect without OUTPUT; no files will be written either way.")

        info, episodes, episode_indices = _load_dataset(source)
        if args.episode_id is not None:
            if args.episode_id not in episode_indices:
                raise ValueError(f"Episode {args.episode_id} not found; available episodes: {episode_indices}")
            selected_episode_ids = [args.episode_id]
        elif args.episodes:
            selected_episode_ids = _parse_episode_selection(args.episodes, episode_indices)
        elif args.all_episodes:
            selected_episode_ids = episode_indices
        else:
            selected_episode_ids = []

        if args.range_manifest is not None:
            ranges_by_episode = _load_range_manifest(args.range_manifest)
            unknown = sorted(set(ranges_by_episode) - set(episode_indices))
            if unknown:
                raise ValueError(f"Range manifest refers to missing episodes: {unknown}")
            if args.selected_only:
                selected_episode_ids = [
                    episode_index for episode_index in episode_indices if episode_index in ranges_by_episode
                ]
                source_episodes = [episodes[episode_index] for episode_index in selected_episode_ids]
                reindex_output_episodes = True
            else:
                source_episodes = episodes
                reindex_output_episodes = False
        else:
            selected_ranges: dict[int, EpisodeRange] = {}
            for episode_index in selected_episode_ids:
                selected_ranges[episode_index] = _select_episode_range(
                    source=source,
                    info=info,
                    episode_index=episode_index,
                    state_key=args.state_key,
                    action_key=args.action_key,
                    yes=args.yes,
                )
            if args.manifest_out is not None:
                _write_range_manifest(args.manifest_out, source, selected_ranges)
                print(f"Wrote range manifest: {args.manifest_out}")
            ranges_by_episode = {
                episode_index: (selected.trim_start_s, selected.trim_end_s)
                for episode_index, selected in selected_ranges.items()
            }
            if not ranges_by_episode:
                raise ValueError("No episode ranges were selected")
            source_episodes, reindex_output_episodes = _selected_source_episodes(
                episodes,
                selected_episode_ids,
                include_unselected_full=args.include_unselected_full,
            )

        plans = _build_range_plans(
            source,
            info,
            source_episodes,
            ranges_by_episode,
            reindex_output_episodes=reindex_output_episodes,
        )
        _print_range_plan(plans, fps=float(info["fps"]), dry_run=args.dry_run)

        if output is None:
            if args.manifest_out is None and args.range_manifest is None:
                print("No OUTPUT or --manifest-out was provided, so no files were written.")
            return 0
        if args.dry_run:
            print("Dry run complete; no files were written.")
            return 0
        if shutil.which(args.ffmpeg) is None:
            raise FileNotFoundError(f"FFmpeg executable was not found: {args.ffmpeg}")

        _build_output, _dataset_file_snapshot, _provenance_source_files, _source_hashes = _load_trim_writer_helpers()
        source_snapshot = _dataset_file_snapshot(source)
        relative_paths = _provenance_source_files(source_snapshot, info, plans)
        source_hashes = {} if args.skip_source_hashes else _source_hashes(source, relative_paths)
        _build_output(
            source,
            output,
            info,
            source_episodes,
            plans,
            ffmpeg=args.ffmpeg,
            video_codec=args.video_codec,
            video_preset=args.video_preset,
            video_crf=args.video_crf,
            source_snapshot=source_snapshot,
            source_hashes=source_hashes,
        )
        _patch_range_provenance(output, source, plans)
        print(f"Created visually trimmed dataset: {output}")
        print(f"Provenance: {output / 'meta' / 'trim_provenance.json'}")
        return 0
    except (
        FileNotFoundError,
        NotADirectoryError,
        FileExistsError,
        ValueError,
        RuntimeError,
        KeyboardInterrupt,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
