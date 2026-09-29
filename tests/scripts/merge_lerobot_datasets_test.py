from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as parquet

from scripts.merge_lerobot_datasets import main


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def _write_jsonlines(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _stats(values: np.ndarray) -> dict[str, list]:
    return {
        "min": np.min(values, axis=0, keepdims=values.ndim == 1).tolist(),
        "max": np.max(values, axis=0, keepdims=values.ndim == 1).tolist(),
        "mean": np.mean(values, axis=0, keepdims=values.ndim == 1).tolist(),
        "std": np.std(values, axis=0, keepdims=values.ndim == 1).tolist(),
        "count": [len(values)],
    }


def _create_dataset(
    root: Path,
    *,
    tasks: list[str],
    episode_task_indices: list[int],
    value_offset: int,
) -> None:
    lengths = [3] * len(episode_task_indices)
    features = {
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
        "observation.state": {"dtype": "float32", "shape": [2], "names": ["a", "b"]},
        "action": {"dtype": "float32", "shape": [2], "names": ["a", "b"]},
        "observation.images.cam": {
            "dtype": "video",
            "shape": [24, 32, 3],
            "names": ["height", "width", "channels"],
            "info": {
                "video.height": 24,
                "video.width": 32,
                "video.channels": 3,
                "video.codec": "h264",
                "video.pix_fmt": "yuv420p",
            },
        },
    }
    info = {
        "codebase_version": "v2.1",
        "robot_type": "test",
        "total_episodes": len(lengths),
        "total_frames": sum(lengths),
        "total_tasks": len(tasks),
        "total_videos": len(lengths),
        "total_chunks": 1,
        "chunks_size": 1000,
        "fps": 10,
        "splits": {"train": f"0:{len(lengths)}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
    }
    _write_json(root / "meta" / "info.json", info)
    _write_jsonlines(
        root / "meta" / "tasks.jsonl",
        [{"task_index": index, "task": task} for index, task in enumerate(tasks)],
    )
    _write_jsonlines(
        root / "meta" / "episodes.jsonl",
        [
            {
                "episode_index": episode_index,
                "tasks": [tasks[task_index]],
                "length": length,
            }
            for episode_index, (task_index, length) in enumerate(zip(episode_task_indices, lengths, strict=True))
        ],
    )

    stats_rows = []
    global_index = 0
    for episode_index, (task_index, length) in enumerate(zip(episode_task_indices, lengths, strict=True)):
        values = np.arange(length, dtype=np.float32) + value_offset + 10 * episode_index
        timestamps = np.arange(length, dtype=np.float32) / 10
        frame_indices = np.arange(length, dtype=np.int64)
        episode_indices = np.full(length, episode_index, dtype=np.int64)
        global_indices = np.arange(global_index, global_index + length, dtype=np.int64)
        task_indices = np.full(length, task_index, dtype=np.int64)
        states = np.column_stack([values, values + 0.5]).astype(np.float32)
        actions = np.column_stack([values + 1, values + 1.5]).astype(np.float32)
        table = pa.table(
            {
                "timestamp": pa.array(timestamps, type=pa.float32()),
                "frame_index": pa.array(frame_indices, type=pa.int64()),
                "episode_index": pa.array(episode_indices, type=pa.int64()),
                "index": pa.array(global_indices, type=pa.int64()),
                "task_index": pa.array(task_indices, type=pa.int64()),
                "observation.state": pa.array(states.tolist(), type=pa.list_(pa.float32(), 2)),
                "action": pa.array(actions.tolist(), type=pa.list_(pa.float32(), 2)),
            }
        )
        data_path = root / f"data/chunk-000/episode_{episode_index:06d}.parquet"
        data_path.parent.mkdir(parents=True, exist_ok=True)
        parquet.write_table(table, data_path)

        video_path = root / f"videos/chunk-000/observation.images.cam/episode_{episode_index:06d}.mp4"
        video_path.parent.mkdir(parents=True, exist_ok=True)
        video_path.write_bytes(f"fake-mp4-{value_offset}-{episode_index}".encode())
        stats_rows.append(
            {
                "episode_index": episode_index,
                "stats": {
                    "timestamp": _stats(timestamps),
                    "frame_index": _stats(frame_indices),
                    "episode_index": _stats(episode_indices),
                    "index": _stats(global_indices),
                    "task_index": _stats(task_indices),
                    "observation.state": _stats(states),
                    "action": _stats(actions),
                    "observation.images.cam": {
                        "min": [[[0.0]]],
                        "max": [[[1.0]]],
                        "mean": [[[0.5]]],
                        "std": [[[0.1]]],
                        "count": [length],
                    },
                },
            }
        )
        global_index += length
    _write_jsonlines(root / "meta" / "episodes_stats.jsonl", stats_rows)


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _read_jsonlines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_merge_reindexes_tasks_episodes_frames_and_copies_videos(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    output = tmp_path / "merged"
    _create_dataset(first, tasks=["bread", "shared"], episode_task_indices=[0, 1], value_offset=0)
    _create_dataset(second, tasks=["turtle", "shared"], episode_task_indices=[0, 1], value_offset=100)
    before_first = _tree_hashes(first)
    before_second = _tree_hashes(second)

    result = main([str(first), str(second), "--output", str(output)])

    assert result == 0
    assert _tree_hashes(first) == before_first
    assert _tree_hashes(second) == before_second
    info = json.loads((output / "meta" / "info.json").read_text())
    assert info["total_episodes"] == 4
    assert info["total_frames"] == 12
    assert info["total_tasks"] == 3
    assert info["total_videos"] == 4
    assert info["splits"] == {"train": "0:4"}

    tasks = _read_jsonlines(output / "meta" / "tasks.jsonl")
    assert tasks == [
        {"task_index": 0, "task": "bread"},
        {"task_index": 1, "task": "shared"},
        {"task_index": 2, "task": "turtle"},
    ]
    episodes = _read_jsonlines(output / "meta" / "episodes.jsonl")
    assert [row["episode_index"] for row in episodes] == [0, 1, 2, 3]
    assert [row["tasks"] for row in episodes] == [["bread"], ["shared"], ["turtle"], ["shared"]]

    for episode_index in range(4):
        table = parquet.read_table(output / f"data/chunk-000/episode_{episode_index:06d}.parquet")
        np.testing.assert_array_equal(table["frame_index"].to_numpy(), np.arange(3))
        np.testing.assert_array_equal(table["episode_index"].to_numpy(), np.full(3, episode_index))
        np.testing.assert_array_equal(table["index"].to_numpy(), np.arange(episode_index * 3, episode_index * 3 + 3))
    np.testing.assert_array_equal(
        parquet.read_table(output / "data/chunk-000/episode_000002.parquet")["task_index"].to_numpy(),
        np.full(3, 2),
    )
    np.testing.assert_array_equal(
        parquet.read_table(output / "data/chunk-000/episode_000003.parquet")["task_index"].to_numpy(),
        np.full(3, 1),
    )

    source_video = second / "videos/chunk-000/observation.images.cam/episode_000000.mp4"
    output_video = output / "videos/chunk-000/observation.images.cam/episode_000002.mp4"
    assert output_video.read_bytes() == source_video.read_bytes()
    assert output_video.stat().st_ino != source_video.stat().st_ino
    source_table = parquet.read_table(second / "data/chunk-000/episode_000000.parquet")
    merged_table = parquet.read_table(output / "data/chunk-000/episode_000002.parquet")
    assert merged_table["observation.state"].to_pylist() == source_table["observation.state"].to_pylist()
    assert merged_table["action"].to_pylist() == source_table["action"].to_pylist()

    stats = _read_jsonlines(output / "meta" / "episodes_stats.jsonl")
    assert stats[2]["stats"]["episode_index"]["mean"] == [2.0]
    assert stats[2]["stats"]["index"]["min"] == [6]
    assert stats[2]["stats"]["task_index"]["mean"] == [2.0]
    assert stats[2]["stats"]["observation.images.cam"] == {
        "min": [[[0.0]]],
        "max": [[[1.0]]],
        "mean": [[[0.5]]],
        "std": [[[0.1]]],
        "count": [3],
    }
    provenance = json.loads((output / "meta" / "merge_provenance.json").read_text())
    assert provenance["videos"]["operation"] == "copy2_without_reencoding"
    assert provenance["state_and_action"]["operation"] == "preserve_source_values_and_dimensions"
    assert provenance["task_index_remaps"][str(second)] == {"0": 2, "1": 1}


def test_merge_refuses_existing_output(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    output = tmp_path / "merged"
    _create_dataset(first, tasks=["bread"], episode_task_indices=[0], value_offset=0)
    _create_dataset(second, tasks=["turtle"], episode_task_indices=[0], value_offset=100)
    output.mkdir()

    assert main([str(first), str(second), "--output", str(output)]) == 2
