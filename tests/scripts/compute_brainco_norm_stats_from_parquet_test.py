from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts.compute_brainco_norm_stats_from_parquet import _transform_episode
from scripts.compute_brainco_norm_stats_from_parquet import compute_stats


def test_transform_removes_first_14_eef_dims_from_70d_state_and_action() -> None:
    state = np.arange(2 * 70, dtype=np.float32).reshape(2, 70)
    actions = state + 500

    state_56d, action_chunks = _transform_episode(state, actions, action_horizon=1, arm_dims=14)

    np.testing.assert_array_equal(state_56d, state[:, 14:70])
    np.testing.assert_array_equal(action_chunks[:, 0, :14], actions[:, 14:28] - state[:, 14:28])
    np.testing.assert_array_equal(action_chunks[:, 0, 14:], actions[:, 28:70])


def test_transform_accepts_70d_state_and_56d_action() -> None:
    state = np.arange(2 * 70, dtype=np.float32).reshape(2, 70)
    actions = np.arange(2 * 56, dtype=np.float32).reshape(2, 56) + 1000

    state_56d, action_chunks = _transform_episode(state, actions, action_horizon=3, arm_dims=14)

    np.testing.assert_array_equal(state_56d, state[:, 14:70])
    assert action_chunks.shape == (2, 3, 56)
    np.testing.assert_array_equal(action_chunks[0, 0, :14], actions[0, :14] - state[0, 14:28])
    np.testing.assert_array_equal(action_chunks[0, 1, :14], actions[1, :14] - state[0, 14:28])
    np.testing.assert_array_equal(action_chunks[0, 2, :14], actions[1, :14] - state[0, 14:28])
    np.testing.assert_array_equal(action_chunks[0, :, 14:], actions[[0, 1, 1], 14:])


def test_transform_accepts_already_converted_56d_vectors() -> None:
    state = np.arange(56, dtype=np.float32)[None, :]
    actions = (state + 10).copy()

    state_56d, action_chunks = _transform_episode(state, actions, action_horizon=2, arm_dims=14)

    np.testing.assert_array_equal(state_56d, state)
    np.testing.assert_array_equal(action_chunks[0, :, :14], np.full((2, 14), 10, dtype=np.float32))
    np.testing.assert_array_equal(action_chunks[0, :, 14:], np.repeat(actions[:, 14:], 2, axis=0))


def test_transform_rejects_mismatched_frame_counts() -> None:
    with pytest.raises(ValueError, match="frame count mismatch"):
        _transform_episode(
            np.zeros((2, 70), dtype=np.float32),
            np.zeros((3, 56), dtype=np.float32),
            action_horizon=16,
            arm_dims=14,
        )


def test_transform_rejects_unknown_vector_dimension() -> None:
    with pytest.raises(ValueError, match="dimension 56 or raw Revo3 dimension 70"):
        _transform_episode(
            np.zeros((2, 69), dtype=np.float32),
            np.zeros((2, 56), dtype=np.float32),
            action_horizon=16,
            arm_dims=14,
        )


def test_compute_stats_reads_70d_parquet_and_emits_56d_stats(tmp_path: Path) -> None:
    dataset = tmp_path / "merged"
    info = {
        "chunks_size": 1000,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
    }
    (dataset / "meta").mkdir(parents=True)
    (dataset / "meta" / "info.json").write_text(json.dumps(info))
    (dataset / "meta" / "episodes.jsonl").write_text(json.dumps({"episode_index": 0, "length": 3}) + "\n")
    values = np.arange(3 * 70, dtype=np.float32).reshape(3, 70)
    table = pa.table(
        {
            "observation.state": pa.array(values.tolist(), type=pa.list_(pa.float32(), 70)),
            "action": pa.array((values + 10).tolist(), type=pa.list_(pa.float32(), 70)),
        }
    )
    parquet_path = dataset / "data/chunk-000/episode_000000.parquet"
    parquet_path.parent.mkdir(parents=True)
    pq.write_table(table, parquet_path)

    stats, episode_count, frame_count = compute_stats(dataset, action_horizon=2, arm_dims=14)

    assert episode_count == 1
    assert frame_count == 3
    for key in ("state", "actions"):
        assert stats[key].mean.shape == (56,)
        assert stats[key].std.shape == (56,)
        assert stats[key].q01.shape == (56,)
        assert stats[key].q99.shape == (56,)
