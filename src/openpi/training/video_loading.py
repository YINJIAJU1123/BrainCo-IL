"""Opt-in CPU video I/O for the v2.1 BrainCo post-training datasets.

The coverage sampler and model transforms remain owned by data_loader. Readers
are local to each worker; source objects are copied before enabling this path.
An experiment can use a separately verified profile placed beside its run.
"""

from __future__ import annotations

from collections import OrderedDict, defaultdict
import copy
import dataclasses
import hashlib
import json
import os
from pathlib import Path

import av
import jax
import numpy as np
import torch
from lerobot.common.datasets.video_utils import decode_video_frames

from openpi.training import config_io, data_loader

WRIST_KEYS = frozenset(("observation.images.cam_left_wrist", "observation.images.cam_right_wrist"))


@dataclasses.dataclass(frozen=True)
class VideoLoadingOptions:
    workers: int = 4
    prefetch_factor: int = 2
    skip_masked_wrists: bool = True
    decoder_cache_size: int = 16
    ffmpeg_threads: int = 1
    batch_video_queries: bool = True
    optimized_decode: bool = True

    def __post_init__(self):
        if self.workers < 0 or self.prefetch_factor < 1 or self.decoder_cache_size < 0 or self.ffmpeg_threads < 1:
            raise ValueError("Invalid video loading resource limits")


class PyAVDecoderCache:
    """Bounded container reuse; RGB conversion is deferred until frame selection.

    Seek rounding, RGB conversion, float32 distance/tie selection and timestamp
    tolerance deliberately follow the deployed torchvision/PyAV LeRobot reader.
    Dependency frames still need decoding, but do not need RGB conversion.
    """

    def __init__(self, capacity: int, threads: int):
        self.capacity = capacity
        self.threads = threads
        self.readers = OrderedDict()
        self.stats = dict(opens=0, hits=0, evictions=0, seeks=0, decoded_frames=0, rgb_frames=0)
        self.last_pts = []

    def _reader(self, path):
        key = str(path)
        if key in self.readers:
            self.stats["hits"] += 1
            reader = self.readers.pop(key)
        else:
            reader = av.open(key, metadata_errors="ignore")
            reader.streams.video[0].codec_context.thread_count = self.threads
            self.stats["opens"] += 1
        if self.capacity:
            self.readers[key] = reader
            while len(self.readers) > self.capacity:
                _, old = self.readers.popitem(last=False)
                old.close()
                self.stats["evictions"] += 1
        return reader

    def decode(self, path, timestamps, tolerance_s):
        if not timestamps:
            raise ValueError("Video query must contain timestamps")
        reader = self._reader(path)
        try:
            stream = reader.streams.video[0]
            # This is the same tick rounding used by torchvision VideoReader.seek.
            offset = int(round(max(0.0, min(timestamps)) / stream.time_base))
            reader.seek(offset, backward=True, any_frame=False, stream=stream)
            self.stats["seeks"] += 1
            query = np.asarray(timestamps, dtype=np.float32)
            best_dist = np.full(len(query), np.inf, dtype=np.float32)
            best_frames = [None] * len(query)
            best_pts = [None] * len(query)
            last_ts = max(timestamps)
            for frame in reader.decode(video=0):
                pts = float(frame.pts * frame.time_base)
                self.stats["decoded_frames"] += 1
                distances = np.abs(query - np.float32(pts))
                for i in np.flatnonzero(distances < best_dist):
                    best_dist[i] = distances[i]
                    best_frames[i] = frame
                    best_pts[i] = pts
                if pts >= last_ts:
                    break
            if not np.all(best_dist < tolerance_s):
                raise AssertionError(f"Video timestamp outside tolerance: {path}, {timestamps}, {best_dist}")
            converted = {}
            selected = []
            for frame in best_frames:
                identity = id(frame)
                if identity not in converted:
                    converted[identity] = torch.as_tensor(frame.to_rgb().to_ndarray()).permute(2, 0, 1)
                    self.stats["rgb_frames"] += 1
                selected.append(converted[identity])
            self.last_pts = best_pts
            return torch.stack(selected).type(torch.float32) / 255
        finally:
            if not self.capacity:
                reader.close()

    def close(self):
        for reader in self.readers.values():
            reader.close()
        self.readers.clear()

    def __del__(self):
        self.close()


class VideoLeRobotDataset(data_loader._ExplicitFilesLeRobotDataset):
    """Copied source with the existing HF/action path and grouped video requests."""

    def __getstate__(self):
        state = dict(self.__dict__)
        state.pop("_video_readers", None)
        state.pop("_video_pid", None)
        return state

    def _readers(self):
        if getattr(self, "_video_pid", None) != os.getpid():
            self._video_readers = PyAVDecoderCache(self._video_options.decoder_cache_size,
                                                 self._video_options.ffmpeg_threads)
            self._video_pid = os.getpid()
            self._skipped_video_queries = 0
        return self._video_readers

    def __getitem__(self, index):
        return self.__getitems__([index])[0]

    def __getitems__(self, indices):
        items = []
        requests = defaultdict(list)
        readers = self._readers()
        for position, index in enumerate(indices):
            # Keep LeRobot's action horizon, end padding and float32 timestamps.
            item = self.hf_dataset[index]
            episode = item["episode_index"].item()
            query_indices = None
            if self.delta_indices is not None:
                query_indices, padding = self._get_query_indices(index, episode)
                query_result = self._query_hf_dataset(query_indices)
                item = {**item, **padding, **query_result}
            if self.meta.video_keys:
                timestamps = self._get_query_timestamps(item["timestamp"].item(), query_indices)
                for key, query in timestamps.items():
                    if self._skip_source_wrists and key in WRIST_KEYS:
                        feature = self.features[key]
                        if feature.get("names") != ["height", "width", "channels"]:
                            raise ValueError(f"Unexpected image shape convention for {key}")
                        height, width, channels = feature["shape"]
                        item.setdefault(key, torch.zeros((len(query), channels, height, width),
                                                        dtype=torch.float32).squeeze(0))
                        self._skipped_video_queries += 1
                    else:
                        path = self.root / self.meta.get_video_file_path(episode, key)
                        requests[path].append((position, key, query))
            items.append(item)

        for path, entries in requests.items():
            # Sort internal I/O only. Sparse timestamps use independent seeks, so
            # random samples never cause a decode of the whole intervening video.
            ordered = sorted(entries, key=lambda entry: min(entry[2]))
            for position, key, query in ordered:
                if self._video_options.optimized_decode:
                    frames = readers.decode(path, query, self.tolerance_s)
                else:
                    frames = decode_video_frames(path, query, self.tolerance_s, "pyav")
                items[position].setdefault(key, frames.squeeze(0))

        for item in items:
            if self.image_transforms is not None:
                for key in self.meta.camera_keys:
                    item[key] = self.image_transforms(item[key])
            item["task"] = self.meta.tasks[item["task_index"].item()]
        return items


class BatchedConcatDataset(data_loader.ConcatLeRobotDataset):
    def __getitems__(self, indices):
        groups = defaultdict(list)
        for position, index in enumerate(indices):
            source = int(np.searchsorted(self._cumulative_lengths[1:], index, side="right"))
            groups[source].append((position, int(index - self._cumulative_lengths[source])))
        result = [None] * len(indices)
        for source, entries in groups.items():
            dataset = self._datasets[source]
            local_indices = [index for _, index in entries]
            samples = dataset.__getitems__(local_indices)
            for (position, _), sample in zip(entries, samples, strict=True):
                transform = self._prompt_transforms[source]
                result[position] = transform(sample) if transform is not None else sample
        return result


class BatchedTransformedDataset(data_loader.TransformedDataset):
    def __getitems__(self, indices):
        return [self._transform(sample) for sample in self._dataset.__getitems__(indices)]


def configure_sources(raw, data_config, options):
    if not isinstance(raw, data_loader.ConcatLeRobotDataset):
        raise ValueError("Optimized video loading requires explicit concatenated LeRobot sources")
    if len(raw._datasets) != len(data_config.lerobot_datasets):
        raise ValueError("Source configuration mismatch")
    sources = []
    for source, config in zip(raw._datasets, data_config.lerobot_datasets, strict=True):
        if source.video_backend != "pyav":
            raise ValueError("This verified input path requires the PyAV backend")
        cloned = copy.copy(source)
        cloned.__class__ = VideoLeRobotDataset
        cloned._video_options = options
        cloned._skip_source_wrists = options.skip_masked_wrists and config.mask_wrist_images is True
        for name in ("_video_readers", "_video_pid"):
            cloned.__dict__.pop(name, None)
        sources.append(cloned)
    return BatchedConcatDataset(sources, raw._prompt_transforms)


def create_data_loader(config, options, *, sharding=None, shuffle=False):
    """Use the unchanged coverage plan, transforms and JAX device boundary."""
    data_config = config.data.create(config.assets_dirs, config.model)
    raw = data_loader.create_torch_dataset(data_config, config.model.action_horizon, config.model)
    dataset = configure_sources(raw, data_config, options)
    plan = data_loader._coverage_plan(data_config, config.batch_size, dataset)
    if config.num_train_steps != plan.steps:
        raise ValueError("Training steps differ from coverage sampling plan")
    dataset = data_loader.transform_dataset(dataset, data_config)
    if options.batch_video_queries:
        dataset.__class__ = BatchedTransformedDataset
    loader = data_loader.TorchDataLoader(
        dataset, local_batch_size=config.batch_size // jax.process_count(), sharding=sharding,
        shuffle=shuffle, num_batches=plan.steps, num_workers=options.workers,
        seed=config.seed, batch_sampler=plan,
    )
    if options.workers:
        # PyTorch consults this when constructing the iterator, before workers start.
        loader.torch_loader.prefetch_factor = options.prefetch_factor
    return data_loader.DataLoaderImpl(data_config, loader)


def config_sha256(config):
    return hashlib.sha256(config_io.to_yaml(config).encode()).hexdigest()


def prepare_verified_profile(config):
    """Load a per-experiment profile only after its recorded equality/speed gates.

    The serial launcher reloads scripts.train between groups. Reading here also
    works when that launcher has already constructed the original TrainConfig.
    """
    manifest = Path(config.checkpoint_dir).parent / ".input_pipeline" / f"{config.exp_name}.json"
    saved = Path(config.checkpoint_dir) / "input_pipeline.json"
    if config.resume and saved.exists():
        manifest = saved
    if not manifest.exists():
        return config, None
    profile = json.loads(manifest.read_text())
    options = VideoLoadingOptions(**profile["options"])
    expected_config = dataclasses.replace(config, num_workers=profile["original_workers"]) if config.resume else config
    # Resume/overwrite are execution switches, excluded from the admission fingerprint.
    expected_config = dataclasses.replace(expected_config, resume=False, overwrite=False)
    if profile["experiment"] != config.exp_name or profile["config_sha256"] != config_sha256(expected_config):
        raise ValueError("Verified input profile does not match this experiment config")
    evidence_path = Path(profile["evidence_path"])
    evidence_bytes = evidence_path.read_bytes()
    if hashlib.sha256(evidence_bytes).hexdigest() != profile["evidence_sha256"]:
        raise ValueError("Input benchmark evidence changed")
    evidence = json.loads(evidence_bytes)
    if not evidence["passed"] or evidence["selected_options"] != profile["options"]:
        raise ValueError("Input benchmark gates did not pass")
    root = Path(__file__).resolve().parents[3]
    for relative, digest in profile["source_sha256"].items():
        if hashlib.sha256((root / relative).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Verified input source changed: {relative}")
    return dataclasses.replace(config, num_workers=options.workers), profile


def save_profile(profile, directory):
    if profile is not None:
        path = Path(directory) / "input_pipeline.json"
        path.write_text(json.dumps(profile, indent=2) + "\n")
