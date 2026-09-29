#!/usr/bin/env python3
"""Compute the fixed 1:1 reference norm from reviewed real and unique Ego starts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from openpi.shared import normalize
from scripts.compute_brainco_norm_stats_from_parquet import (
    _episode_paths, _read_vector_columns, _to_56d,
)


def _sha(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            sha.update(block)
    return sha.hexdigest()


def _consume(dataset: Path, selected: np.ndarray | None, state_stats, action_stats,
             *, horizon: int) -> tuple[int, list[dict]]:
    offset = 0
    inputs = []
    count = 0
    for episode, path, expected_length in _episode_paths(dataset):
        if expected_length is None:
            raise ValueError(f"episode {episode} has no declared length")
        lo = np.searchsorted(selected, offset, side="left") if selected is not None else 0
        hi = np.searchsorted(selected, offset + expected_length, side="left") if selected is not None else expected_length
        if hi > lo:
            state, actions = _read_vector_columns(path, ("observation.state", "action"))
            if len(state) != expected_length or len(actions) != expected_length:
                raise ValueError(f"episode {episode} length mismatch")
            starts = selected[lo:hi] - offset if selected is not None else np.arange(expected_length)
            state56 = _to_56d(state, key="observation.state")[starts]
            actions56 = _to_56d(actions, key="action")
            chunk_index = np.minimum(starts[:, None] + np.arange(horizon), expected_length - 1)
            chunks = actions56[chunk_index].copy()
            chunks[..., :14] -= state56[:, None, :14]
            if not np.isfinite(state56).all() or not np.isfinite(chunks).all():
                raise ValueError(f"nonfinite transformed vectors in episode {episode}")
            state_stats.update(state56)
            action_stats.update(chunks)
            count += len(starts)
            inputs.append({"episode": episode, "parquet": str(path), "sha256": _sha(path), "starts": len(starts)})
        offset += expected_length
    if selected is not None and count != len(selected):
        raise ValueError(f"selected {len(selected)} starts but consumed {count}")
    return count, inputs


def compute(ego: Path, real: Path, output: Path, *, horizon: int = 16, seed: int = 42) -> dict:
    if output.exists():
        raise FileExistsError(output)
    ego_episodes = _episode_paths(ego)
    real_episodes = _episode_paths(real)
    ego_count = sum(length for _, _, length in ego_episodes)
    real_count = sum(length for _, _, length in real_episodes)
    if real_count <= 0 or ego_count < real_count:
        raise ValueError("reviewed real must be nonempty and Ego must provide equal unique starts")
    # This is the same seed and permutation as CoverageBatchSampler's Ego prefix.
    ego_order = np.random.default_rng(seed).permutation(ego_count)[:real_count]
    selected = np.sort(ego_order)
    stats = {key: normalize.RunningStats() for key in ("state", "actions")}
    ego_used, ego_inputs = _consume(ego, selected, stats["state"], stats["actions"], horizon=horizon)
    real_used, real_inputs = _consume(real, None, stats["state"], stats["actions"], horizon=horizon)
    result = {key: value.get_statistics() for key, value in stats.items()}
    for key, value in result.items():
        span = np.asarray(value.q99) - np.asarray(value.q01)
        if not np.isfinite(span).all() or (span < 1e-6).any():
            raise ValueError(f"{key}: degenerate quantile span at {np.flatnonzero(span < 1e-6).tolist()}")
    output.mkdir(parents=True)
    np.save(output / "ego_start_indices.npy", ego_order, allow_pickle=False)
    normalize.save(output, result)
    manifest = {"method": "1:1 real+Ego starts, horizon16, 70D/56D to 56D, arm delta, hand absolute",
                "seed": seed, "horizon": horizon, "real_starts": real_used, "ego_starts": ego_used,
                "ego_selected_order_sha256": hashlib.sha256(ego_order.tobytes()).hexdigest(),
                "ego_start_indices_file": "ego_start_indices.npy",
                "ego_start_indices_file_sha256": _sha(output / "ego_start_indices.npy"),
                "ego_metadata_sha256": [_sha(ego / "meta/info.json"), _sha(ego / "meta/episodes.jsonl")],
                "real_metadata_sha256": [_sha(real / "meta/info.json"), _sha(real / "meta/episodes.jsonl")],
                "inputs": {"ego": ego_inputs, "real": real_inputs}}
    (output / "sampling_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    manifest["norm_sha256"] = _sha(output / "norm_stats.json")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ego", type=Path, required=True)
    parser.add_argument("--real", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compute(args.ego, args.real, args.output)
    print(json.dumps({key: result[key] for key in ("real_starts", "ego_starts", "norm_sha256")}, indent=2))


if __name__ == "__main__":
    main()
