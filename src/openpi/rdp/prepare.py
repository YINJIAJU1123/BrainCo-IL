"""Build a versioned RDP cache without modifying the source LeRobot dataset."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from openpi.deploy.contract import resolve_dataset_interface
from openpi.rdp.tactile import CHANNELS
from openpi.rdp.tactile import FINGERS
from openpi.rdp.tactile import MARKER_KEYS
from openpi.rdp.tactile import TactilePCA
from openpi.rdp.tactile import held_indices
from openpi.rdp.tactile import offsets


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def arrow_numpy(column, shape):
    """Flatten Arrow lists without allocating millions of Python floats."""
    array = column.combine_chunks()
    while pa.types.is_list(array.type) or pa.types.is_large_list(array.type) or pa.types.is_fixed_size_list(array.type):
        array = array.flatten()
    if array.null_count:
        raise ValueError("null values in dataset column")
    return array.to_numpy(zero_copy_only=False).reshape(shape)


def prepare(root, output, *, components=15, tactile_hz=10.0, image_size=128, images=True):
    root, output = Path(root).resolve(), Path(output).resolve()
    if (output / "manifest.json").exists():
        raise FileExistsError("completed cache exists; use a new output directory")
    output.mkdir(parents=True, exist_ok=True)
    (output / "pca").mkdir(exist_ok=True)
    info = json.loads((root / "meta/info.json").read_text())
    interface = resolve_dataset_interface([root], ("left_arm", "right_arm", "left_hand", "right_hand"))
    for key in MARKER_KEYS:
        feature = info["features"][key]
        if tuple(feature["shape"]) != (20, 20, 6) or tuple(feature["names"]) != CHANNELS:
            raise ValueError(f"unsupported marker schema: {key}")
    paths = sorted(root.glob("data/*/episode_*.parquet"))
    if len(paths) < 2:
        raise ValueError("at least two episodes required for an episode-level holdout")
    total = sum(pq.read_metadata(p).num_rows for p in paths)
    raw = [
        np.lib.format.open_memmap(output / f"offset_{f}.npy", mode="w+", dtype="float32", shape=(total, 800))
        for f in FINGERS
    ]
    states, actions, times, segments, samples, audits = [], [], [], [], [], []
    source_hashes = {"meta/info.json": sha256(root / "meta/info.json")}
    start = 0
    for path in paths:
        columns = ["observation.state", "action", "timestamp", "episode_index", *MARKER_KEYS]
        table = pq.read_table(path, columns=columns)
        count = len(table)
        episode = int(table["episode_index"][0].as_py())
        ts = arrow_numpy(table["timestamp"], (-1,)).astype(np.float64)
        held, age = held_indices(ts, tactile_hz)
        states.append(arrow_numpy(table["observation.state"], (count, -1)).astype(np.float32))
        actions.append(arrow_numpy(table["action"], (count, -1)).astype(np.float32))
        times.append(ts)
        segment = {
            "episode": episode,
            "start": start,
            "end": start + count,
            "split": "val" if path == paths[-1] else "train",
        }
        segments.append(segment)
        np.save(output / f"held_{episode}.npy", held)
        np.save(output / f"age_{episode}.npy", age)
        if segment["split"] == "train":
            unique = np.unique(held)
            samples.extend(
                (start + unique[np.linspace(0, len(unique) - 1, min(len(unique), 3000), dtype=int)]).tolist()
            )
        for j, key in enumerate(MARKER_KEYS):
            markers = arrow_numpy(table[key], (count, 20, 20, 6)).astype(np.float32)
            x = offsets(markers)
            raw[j][start : start + count] = x
            audits.append(
                {
                    "episode": episode,
                    "finger": FINGERS[j],
                    "finite": True,
                    "coordinate_points_min": int(
                        np.count_nonzero(np.any(markers[..., :4] != 0, axis=-1), axis=(1, 2)).min()
                    ),
                    "changed_fraction": float(np.any(np.diff(x, axis=0) != 0, axis=1).mean()),
                    "offset_min": float(x.min()),
                    "offset_max": float(x.max()),
                }
            )
        source_hashes[str(path.relative_to(root))] = sha256(path)
        start += count
        print(f"read episode={episode} frames={count} split={segment['split']}", flush=True)
    for key, arrays in (("state", states), ("action", actions), ("timestamp", times)):
        np.save(output / f"{key}.npy", np.concatenate(arrays))
    embedding = np.lib.format.open_memmap(
        output / "tactile.npy", mode="w+", dtype="float32", shape=(total, components * len(FINGERS) + 1)
    )
    pca_report = {}
    for j, finger in enumerate(FINGERS):
        pca, variance = TactilePCA.fit(raw[j][samples], components)
        pca.save(output / "pca" / f"{finger}.npz")
        validation = raw[j][segments[-1]["start"] :: 3]
        mse = float(np.mean((pca.reconstruct(validation) - validation) ** 2))
        baseline = float(np.mean((validation - pca.mean) ** 2))
        pca_report[finger] = {
            "train_explained_variance": variance,
            "val_reconstruction_mse": mse,
            "val_residual_fraction": mse / max(baseline, 1e-12),
        }
        for seg in segments:
            held = np.load(output / f"held_{seg['episode']}.npy")
            x = raw[j][seg["start"] : seg["end"]]
            embedding[seg["start"] : seg["end"], j * components : (j + 1) * components] = pca.encode(x[held])
        print(f"PCA {finger}: {pca_report[finger]}", flush=True)
    for seg in segments:
        embedding[seg["start"] : seg["end"], -1] = np.load(output / f"age_{seg['episode']}.npy") * tactile_hz
    embedding.flush()
    # A read-only linked training view lets the canonical normalization entry
    # point fit on training episodes only, with its original temporal queries.
    view = output / "train_dataset"
    (view / "meta").mkdir(parents=True, exist_ok=True)
    train_ids = {s["episode"] for s in segments if s["split"] == "train"}
    train_info = dict(
        info,
        total_episodes=len(train_ids),
        total_frames=segments[-1]["start"],
        episode_indices=sorted(train_ids),
        next_frame_index=segments[-1]["start"],
        splits={"train": f"0:{len(train_ids)}"},
    )
    train_info["total_videos"] = len(train_ids) * sum(v.get("dtype") == "video" for v in info["features"].values())
    (view / "meta/info.json").write_text(json.dumps(train_info, indent=2))
    for path in (root / "meta").iterdir():
        if path.name == "info.json" or not path.is_file():
            continue
        if path.name in ("episodes.jsonl", "episodes_stats.jsonl"):
            lines = [line for line in path.read_text().splitlines() if json.loads(line)["episode_index"] in train_ids]
            (view / "meta" / path.name).write_text("\n".join(lines) + "\n")
        else:
            shutil.copy2(path, view / "meta" / path.name)
    for path in paths[:-1]:
        dest = view / path.relative_to(root)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            dest.symlink_to(path)
    if not (view / "videos").exists():
        (view / "videos").symlink_to(root / "videos", target_is_directory=True)
    if images:
        cache_images(root, output, segments, interface["cameras"], image_size, info)
    manifest = {
        "schema_version": 1,
        "dataset": str(root),
        "frames": total,
        "segments": segments,
        "interface": interface,
        "components_per_finger": components,
        "tactile_dim": embedding.shape[1],
        "tactile_hz": tactile_hz,
        "image_size": image_size,
        "images_cached": images,
        "pca": pca_report,
        "marker_audit": audits,
        "source_sha256": source_hashes,
        "timing_note": "10 Hz causal adapter polling; marker capture timestamps unavailable; last channel is hold age / 0.1 s, not measured sensor latency",
        "validation_note": "last entire episode held out from PCA, normalization and policy training",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"PREPARED {output}", flush=True)
    return manifest


def cache_images(root, output, segments, cameras, size, info):
    for cam_index, camera in enumerate(cameras):
        dest = output / f"camera_{cam_index}.npy"
        arr = np.lib.format.open_memmap(dest, mode="w+", dtype="uint8", shape=(segments[-1]["end"], size, size, 3))
        for seg in segments:
            episode = seg["episode"]
            path = root / info["video_path"].format(
                episode_chunk=episode // info["chunks_size"], episode_index=episode, video_key=camera
            )
            cap = cv2.VideoCapture(str(path))
            fps = cap.get(cv2.CAP_PROP_FPS)
            if not cap.isOpened() or abs(fps - float(info["fps"])) > 0.1:
                raise ValueError(f"video timing mismatch: {path}: {fps}")
            for index in range(seg["start"], seg["end"]):
                ok, frame = cap.read()
                if not ok:
                    raise ValueError(f"video too short: {path}, frame {index - seg['start']}")
                arr[index] = cv2.cvtColor(
                    cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB
                )
            cap.release()
            print(f"cached {camera} episode={episode}", flush=True)
        arr.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--components", type=int, default=15)
    parser.add_argument("--tactile-hz", type=float, default=10)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--skip-images", action="store_true")
    args = parser.parse_args()
    prepare(
        args.dataset,
        args.output,
        components=args.components,
        tactile_hz=args.tactile_hz,
        image_size=args.image_size,
        images=not args.skip_images,
    )


if __name__ == "__main__":
    main()
