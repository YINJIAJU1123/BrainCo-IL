#!/usr/bin/env python3
"""Create a read-only real-demonstration review packet; never decide exclusions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import pyarrow.parquet as pq


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            sha.update(chunk)
    return sha.hexdigest()


def inspect(source: Path, output: Path, *, interval_seconds: int = 60) -> dict:
    info_path = source / "meta/info.json"
    info = json.loads(info_path.read_text())
    episodes = [json.loads(line) for line in (source / "meta/episodes.jsonl").read_text().splitlines()]
    output.mkdir(parents=True, exist_ok=True)
    report = {"source": str(source), "info_sha256": digest(info_path), "fps": info["fps"],
              "total_frames": info["total_frames"], "episodes": []}
    manifest = {"schema_version": 1, "source": str(source), "source_info_sha256": report["info_sha256"],
                "status": "awaiting_human_review",
                "task_instruction": "put bread into left basket and put fruit into right basket",
                "timeline_interval_seconds": interval_seconds, "intervals": []}
    for ep in episodes:
        episode = int(ep["episode_index"])
        length = int(ep["length"])
        relative = info["data_path"].format(episode_chunk=episode // info["chunks_size"], episode_index=episode)
        parquet = source / relative
        table = pq.read_table(parquet, columns=["observation.state", "action", "frame_index", "timestamp"])
        state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
        action = np.asarray(table["action"].to_pylist(), dtype=np.float32)
        frames = np.asarray(table["frame_index"].to_pylist(), dtype=np.int64)
        timestamps = np.asarray(table["timestamp"].to_pylist(), dtype=np.float64)
        if len(state) != length or len(action) != length or state.shape[1] not in (56, 70):
            raise ValueError(f"episode {episode}: data length or dimensions disagree with metadata")
        if not np.array_equal(frames, np.arange(length)):
            raise ValueError(f"episode {episode}: frame_index is discontinuous")
        if np.max(np.abs(timestamps - frames / info["fps"])) > 1e-3:
            raise ValueError(f"episode {episode}: timestamps disagree with FPS")
        state_finite = np.isfinite(state).all(axis=1)
        action_finite = np.isfinite(action).all(axis=1)
        joints = state[:, -56:]
        speed = np.linalg.norm(np.diff(joints, axis=0), axis=1) * info["fps"]
        buckets = []
        bucket_size = interval_seconds * info["fps"]
        for start in range(0, length, bucket_size):
            stop = min(start + bucket_size, length)
            movement = speed[start:max(start, stop - 1)]
            buckets.append({"start_frame": start, "end_frame_exclusive": stop,
                            "median_joint_speed_rad_s": float(np.nanmedian(movement)) if len(movement) else 0,
                            "p95_joint_speed_rad_s": float(np.nanpercentile(movement, 95)) if len(movement) else 0,
                            "nonfinite_state_frames": int((~state_finite[start:stop]).sum()),
                            "nonfinite_action_frames": int((~action_finite[start:stop]).sum())})
        report["episodes"].append({"episode_index": episode, "length": length,
                                   "parquet_sha256": digest(parquet), "state_dim": state.shape[1],
                                   "action_dim": action.shape[1], "buckets": buckets})
        manifest["intervals"].append({"episode_index": episode, "start_frame": 0,
                                      "end_frame_exclusive": length, "decision": "pending", "reason": ""})
        for key in ("observation.images.cam_head", "observation.images.cam_left_wrist",
                    "observation.images.cam_right_wrist"):
            video = source / info["video_path"].format(episode_chunk=episode // info["chunks_size"],
                                                        episode_index=episode, video_key=key)
            if not video.is_file():
                raise FileNotFoundError(video)
            target = output / f"episode_{episode:06d}_{key.rsplit('.', 1)[-1]}_timeline.jpg"
            subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(video),
                            "-vf", f"fps=1/{interval_seconds},scale=320:-1,tile=6x6",
                            "-frames:v", "1", str(target)], check=True)
    (output / "signals.json").write_text(json.dumps(report, indent=2) + "\n")
    (output / "review_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return {"episodes": len(episodes), "frames": info["total_frames"], "output": str(output)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--interval-seconds", type=int, default=60)
    args = parser.parse_args()
    print(json.dumps(inspect(args.source, args.output, interval_seconds=args.interval_seconds)))


if __name__ == "__main__":
    main()
