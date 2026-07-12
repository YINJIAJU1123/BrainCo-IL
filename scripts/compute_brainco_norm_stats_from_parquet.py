#!/usr/bin/env python3
"""Compute BrainCo normalization stats directly from LeRobot parquet files.

This is a fast path for Revo3 datasets whose normalization inputs depend only on
``observation.state`` and ``action``. It deliberately skips the LeRobot
DataLoader and therefore never decodes camera videos.

The transform matches the current BrainCo Pi0.5 training setup:

* raw Revo3 vectors: 70D = two 7D EEF poses + two 7D arms + two 21D hands
* model vectors: 56D = left arm + right arm + left hand + right hand
* action chunks: episode-end clamping to ``action_horizon``
* action representation: arm dimensions are delta; hand dimensions are absolute

This script is positional. It does not validate or repair semantic joint-name /
value permutations in the source parquet vectors.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pyarrow.parquet as pq
import tqdm

from openpi.shared import normalize

REVO3_70D_TO_56D = np.asarray(
    [
        *range(14, 21),
        *range(21, 28),
        *range(28, 49),
        *range(49, 70),
    ],
    dtype=np.int64,
)


def _read_jsonlines(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _episode_paths(dataset: Path) -> list[tuple[int, Path, int | None]]:
    info_path = dataset / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(info_path)
    info = json.loads(info_path.read_text())
    data_path = info.get("data_path")
    if not isinstance(data_path, str):
        raise ValueError(f"Missing string data_path in {info_path}")

    chunks_size = int(info.get("chunks_size", 1000))
    episodes = _read_jsonlines(dataset / "meta" / "episodes.jsonl")
    result = []
    for episode in episodes:
        episode_index = int(episode["episode_index"])
        relative = data_path.format(
            episode_chunk=episode_index // chunks_size,
            episode_index=episode_index,
        )
        expected_length = int(episode["length"]) if "length" in episode else None
        result.append((episode_index, dataset / relative, expected_length))
    return result


def _read_vector_column(path: Path, column: str) -> np.ndarray:
    table = pq.read_table(path, columns=[column])
    values = np.asarray(table[column].to_pylist(), dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"{path}: expected {column!r} to be a vector column, got shape {values.shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{path}: {column!r} contains NaN or Inf")
    return values


def _build_action_chunks(actions: np.ndarray, action_horizon: int) -> np.ndarray:
    if action_horizon <= 0:
        raise ValueError(f"action_horizon must be positive, got {action_horizon}")
    if len(actions) == 0:
        raise ValueError("Cannot build action chunks for an empty episode")
    frame_indices = np.arange(len(actions), dtype=np.int64)[:, None]
    offsets = np.arange(action_horizon, dtype=np.int64)[None, :]
    return actions[np.minimum(frame_indices + offsets, len(actions) - 1)]


def _transform_episode(
    state: np.ndarray,
    actions: np.ndarray,
    *,
    action_horizon: int,
    arm_dims: int,
) -> tuple[np.ndarray, np.ndarray]:
    if state.shape != actions.shape:
        raise ValueError(f"state/action shape mismatch: {state.shape} != {actions.shape}")
    if state.shape[-1] != 70:
        raise ValueError(f"Expected raw Revo3 state/action dimension 70, got {state.shape[-1]}")
    if arm_dims < 0 or arm_dims > len(REVO3_70D_TO_56D):
        raise ValueError(f"arm_dims must be in [0, 56], got {arm_dims}")

    state_56d = state[:, REVO3_70D_TO_56D]
    action_56d = actions[:, REVO3_70D_TO_56D]
    action_chunks = _build_action_chunks(action_56d, action_horizon)
    action_chunks[..., :arm_dims] -= state_56d[:, None, :arm_dims]
    return state_56d, action_chunks


def compute_stats(
    dataset: Path,
    *,
    action_horizon: int,
    arm_dims: int,
) -> tuple[dict[str, normalize.NormStats], int, int]:
    state_stats = normalize.RunningStats()
    action_stats = normalize.RunningStats()
    episodes = _episode_paths(dataset)
    total_frames = 0

    for episode_index, path, expected_length in tqdm.tqdm(episodes, desc="Computing parquet stats"):
        if not path.is_file():
            raise FileNotFoundError(path)
        state = _read_vector_column(path, "observation.state")
        actions = _read_vector_column(path, "action")
        if expected_length is not None and len(state) != expected_length:
            raise ValueError(
                f"Episode {episode_index}: metadata length {expected_length} does not match parquet length {len(state)}"
            )

        transformed_state, transformed_actions = _transform_episode(
            state,
            actions,
            action_horizon=action_horizon,
            arm_dims=arm_dims,
        )
        state_stats.update(transformed_state)
        action_stats.update(transformed_actions)
        total_frames += len(state)

    if total_frames == 0:
        raise ValueError(f"Dataset has no frames: {dataset}")
    return (
        {
            "state": state_stats.get_statistics(),
            "actions": action_stats.get_statistics(),
        },
        len(episodes),
        total_frames,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="LeRobot dataset root containing meta/ and data/")
    parser.add_argument("output_dir", type=Path, help="Directory in which norm_stats.json will be written")
    parser.add_argument("--action-horizon", type=int, default=16)
    parser.add_argument("--arm-dims", type=int, default=14)
    parser.add_argument("--overwrite", action="store_true", help="Replace output_dir/norm_stats.json if it exists")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    dataset = args.dataset.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_path = output_dir / "norm_stats.json"
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite existing stats: {output_path}; pass --overwrite to replace")

    started = time.monotonic()
    norm_stats, episode_count, frame_count = compute_stats(
        dataset,
        action_horizon=args.action_horizon,
        arm_dims=args.arm_dims,
    )
    normalize.save(output_dir, norm_stats)
    elapsed = time.monotonic() - started
    print(f"Wrote: {output_path}")
    print(f"Processed: {episode_count} episodes, {frame_count} frames, horizon={args.action_horizon}")
    print(f"Elapsed: {elapsed:.2f} seconds")


if __name__ == "__main__":
    main()
