from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as parquet
import pytest

from scripts.trim_lerobot_episodes import _load_trim_manifest
from scripts.trim_lerobot_episodes import main


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def _write_jsonlines(path: Path, values: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(value) + "\n" for value in values))


def _write_video(path: Path, frame_count: int, fps: int, color: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"color=c={color}:s=32x24:r={fps}",
            "-frames:v",
            str(frame_count),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-y",
            str(path),
        ],
        check=True,
    )


def _create_dataset(root: Path) -> None:
    fps = 10
    lengths = [6, 5]
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
                "video.codec": "libx264",
                "video.pix_fmt": "yuv420p",
            },
        },
    }
    info = {
        "codebase_version": "v2.1",
        "robot_type": "test",
        "total_episodes": 2,
        "total_frames": sum(lengths),
        "total_tasks": 1,
        "total_videos": 2,
        "total_chunks": 1,
        "chunks_size": 1000,
        "fps": fps,
        "splits": {"train": "0:2"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
    }
    _write_json(root / "meta" / "info.json", info)
    _write_jsonlines(
        root / "meta" / "episodes.jsonl",
        [{"episode_index": index, "tasks": ["test"], "length": length} for index, length in enumerate(lengths)],
    )
    _write_jsonlines(root / "meta" / "tasks.jsonl", [{"task_index": 0, "task": "test"}])
    _write_jsonlines(root / "meta" / "episodes_stats.jsonl", [])
    (root / "README.md").write_text("synthetic test dataset\n")

    global_index = 0
    for episode_index, length in enumerate(lengths):
        values = np.arange(length, dtype=np.float32) + episode_index * 10
        table = pa.table(
            {
                "timestamp": pa.array(np.arange(length, dtype=np.float32) / fps, type=pa.float32()),
                "frame_index": pa.array(np.arange(length), type=pa.int64()),
                "episode_index": pa.array(np.full(length, episode_index), type=pa.int64()),
                "index": pa.array(np.arange(global_index, global_index + length), type=pa.int64()),
                "task_index": pa.array(np.zeros(length), type=pa.int64()),
                "observation.state": pa.array(
                    np.column_stack([values, values + 0.5]).tolist(), type=pa.list_(pa.float32(), 2)
                ),
                "action": pa.array(
                    np.column_stack([values + 1, values + 1.5]).tolist(), type=pa.list_(pa.float32(), 2)
                ),
            }
        )
        parquet_path = root / f"data/chunk-000/episode_{episode_index:06d}.parquet"
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        parquet.write_table(table, parquet_path)
        _write_video(
            root / f"videos/chunk-000/observation.images.cam/episode_{episode_index:06d}.mp4",
            length,
            fps,
            "red" if episode_index == 0 else "blue",
        )
        global_index += length


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _video_frame_count(path: Path) -> int:
    with av.open(str(path)) as container:
        return sum(1 for _ in container.decode(container.streams.video[0]))


def test_load_json_manifest_supports_partial_episode_mapping(tmp_path: Path) -> None:
    manifest = tmp_path / "trim.json"
    manifest.write_text(json.dumps({"0": 0.5, "3": 0.8}))

    assert _load_trim_manifest(manifest) == {0: 0.5, 3: 0.8}


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg is required")
def test_trim_builds_independent_valid_dataset(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    _create_dataset(source)
    before = _tree_hashes(source)

    result = main(
        [
            str(source),
            str(output),
            "--trim-start-s",
            "0.2",
            "--video-preset",
            "ultrafast",
            "--skip-source-hashes",
        ]
    )

    assert result == 0
    assert _tree_hashes(source) == before
    info = json.loads((output / "meta" / "info.json").read_text())
    assert info["total_frames"] == 7
    episodes = [json.loads(line) for line in (output / "meta" / "episodes.jsonl").read_text().splitlines()]
    assert [episode["length"] for episode in episodes] == [4, 3]
    assert (output / "meta" / "tasks.jsonl").read_text() == (source / "meta" / "tasks.jsonl").read_text()
    assert (output / "README.md").read_text() == (source / "README.md").read_text()

    first = parquet.read_table(output / "data/chunk-000/episode_000000.parquet")
    second = parquet.read_table(output / "data/chunk-000/episode_000001.parquet")
    np.testing.assert_allclose(first["timestamp"].to_numpy(), [0.0, 0.1, 0.2, 0.3], atol=1e-6)
    np.testing.assert_array_equal(first["frame_index"].to_numpy(), np.arange(4))
    np.testing.assert_array_equal(first["index"].to_numpy(), np.arange(4))
    np.testing.assert_array_equal(second["index"].to_numpy(), np.arange(4, 7))
    np.testing.assert_allclose(first["observation.state"].to_pylist()[0], [2.0, 2.5])

    first_video = output / "videos/chunk-000/observation.images.cam/episode_000000.mp4"
    second_video = output / "videos/chunk-000/observation.images.cam/episode_000001.mp4"
    assert _video_frame_count(first_video) == 4
    assert _video_frame_count(second_video) == 3

    stats = [json.loads(line) for line in (output / "meta" / "episodes_stats.jsonl").read_text().splitlines()]
    assert stats[0]["stats"]["timestamp"]["count"] == [4]
    assert stats[0]["stats"]["observation.images.cam"]["count"] == [4]
    provenance = json.loads((output / "meta" / "trim_provenance.json").read_text())
    assert provenance["source_unchanged"] is True
    assert provenance["episodes"][0]["old_frame_start_inclusive"] == 2


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg is required")
def test_episode_id_mode_emits_one_reindexed_episode(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "single_episode_output"
    _create_dataset(source)
    before = _tree_hashes(source)

    result = main(
        [
            str(source),
            str(output),
            "--episode-id",
            "1",
            "--timestamp-s",
            "0.2",
            "--video-preset",
            "ultrafast",
            "--skip-source-hashes",
        ]
    )

    assert result == 0
    assert _tree_hashes(source) == before
    info = json.loads((output / "meta" / "info.json").read_text())
    assert info["total_episodes"] == 1
    assert info["total_frames"] == 3
    assert info["total_videos"] == 1
    episodes = [json.loads(line) for line in (output / "meta" / "episodes.jsonl").read_text().splitlines()]
    assert episodes == [{"episode_index": 0, "tasks": ["test"], "length": 3}]

    table = parquet.read_table(output / "data/chunk-000/episode_000000.parquet")
    np.testing.assert_array_equal(table["episode_index"].to_numpy(), np.zeros(3, dtype=np.int64))
    np.testing.assert_array_equal(table["index"].to_numpy(), np.arange(3))
    np.testing.assert_allclose(table["observation.state"].to_pylist()[0], [12.0, 12.5])
    assert not (output / "data/chunk-000/episode_000001.parquet").exists()
    video = output / "videos/chunk-000/observation.images.cam/episode_000000.mp4"
    assert _video_frame_count(video) == 3

    provenance = json.loads((output / "meta" / "trim_provenance.json").read_text())
    assert provenance["episodes"][0]["source_episode_index"] == 1
    assert provenance["episodes"][0]["output_episode_index"] == 0


def test_dry_run_does_not_create_output(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    if shutil.which("ffmpeg") is None:
        pytest.skip("FFmpeg is required to construct the test fixture")
    source = tmp_path / "source"
    output = tmp_path / "output"
    _create_dataset(source)

    result = main([str(source), str(output), "--trim-start-s", "0.2", "--dry-run"])

    assert result == 0
    assert not output.exists()
    assert "no files were written" in capsys.readouterr().out
