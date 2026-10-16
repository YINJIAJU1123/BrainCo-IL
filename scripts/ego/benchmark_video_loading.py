"""CPU-only, fixed-index comparison of the production and opt-in video loaders.

Run from an isolated source copy with CUDA_VISIBLE_DEVICES='' and
JAX_PLATFORMS=cpu. Original datasets, configs and running jobs are read-only.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import pickle
import resource
import time
import traceback
from unittest.mock import patch

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import numpy as np
import torch
import torchvision

from openpi.training import config_io, data_loader, video_loading

BASELINE_STATS = dict(opens=0, hits=0, evictions=0, seeks=0, decoded_frames=0, rgb_frames=0)
ORIGINAL_VIDEO_READER = torchvision.io.VideoReader


class CountingReader:
    def __init__(self, *args, **kwargs):
        self.reader = ORIGINAL_VIDEO_READER(*args, **kwargs)
        BASELINE_STATS["opens"] += 1

    def __iter__(self):
        return self

    def __next__(self):
        value = next(self.reader)
        BASELINE_STATS["decoded_frames"] += 1
        BASELINE_STATS["rgb_frames"] += 1
        return value

    def seek(self, *args, **kwargs):
        self.reader.seek(*args, **kwargs)
        BASELINE_STATS["seeks"] += 1
        return self

    def __getattr__(self, name):
        return getattr(self.reader, name)


def worker_init(worker_id):
    data_loader._worker_init_fn(worker_id)
    torchvision.io.VideoReader = CountingReader


class TaggedDataset:
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitems__(self, indices):
        if hasattr(self.dataset, "__getitems__"):
            samples = self.dataset.__getitems__(indices)
        else:
            samples = [self.dataset[index] for index in indices]
        stats = dict(BASELINE_STATS)
        stats["skipped_video_queries"] = 0
        raw = self.dataset._dataset
        for source in raw._datasets:
            reader = getattr(source, "_video_readers", None)
            if reader is not None:
                for key, value in reader.stats.items():
                    stats[key] += value
                stats["skipped_video_queries"] += getattr(source, "_skipped_video_queries", 0)
        stats["pid"] = os.getpid()
        stats["open_fds"] = len(os.listdir("/proc/self/fd"))
        stats["peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return [(sample, index, stats if i == 0 else None)
                for i, (sample, index) in enumerate(zip(samples, indices, strict=True))]


def collate(samples):
    return (data_loader._collate_fn([sample[0] for sample in samples]),
            [sample[1] for sample in samples], samples[0][2])


def leaf_records(tree):
    pairs, _ = jax.tree_util.tree_flatten_with_path(tree)
    records = []
    for path, value in pairs:
        array = np.asarray(value)
        digest = hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()
        records.append(dict(path=jax.tree_util.keystr(path), shape=list(array.shape),
                            dtype=str(array.dtype), sha256=digest))
    return records


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def close_loader(loader):
    iterator = loader._data_loader.torch_loader._iterator
    if iterator is not None:
        iterator._shutdown_workers()


def make_loader(config, raw, options):
    with patch.object(data_loader, "create_torch_dataset", return_value=raw):
        if options is None:
            loader = data_loader.create_data_loader(config)
        else:
            loader = video_loading.create_data_loader(config, options)
    torch_loader = loader._data_loader.torch_loader
    # This assignment happens before iterator/worker creation.
    torch_loader.dataset.__class__  # force an early error if the loader is malformed
    object.__setattr__(torch_loader, "dataset", TaggedDataset(torch_loader.dataset))
    torch_loader.collate_fn = collate
    torch_loader.worker_init_fn = worker_init
    return loader


def run_case(config, raw, options, output, name, *, warmup, batches, expected=None, start_step=0):
    write(output / "progress.json", dict(phase="benchmark", case=name, options=dataclasses.asdict(options)
                                        if options is not None else None, start_step=start_step))
    print("START", name, flush=True)
    loader = make_loader(config, raw, options)
    loader.set_start_step(start_step)
    started = time.perf_counter()
    iterator = iter(loader._data_loader.torch_loader)
    records, waits, worker_stats = [], [], {}
    steady_start = None
    try:
        for i in range(warmup + batches):
            if i == warmup:
                steady_start = time.perf_counter()
            t0 = time.perf_counter()
            batch, indices, stats = next(iterator)
            elapsed = time.perf_counter() - t0
            record = dict(step=start_step+i, indices=indices, leaves=leaf_records(batch))
            assert indices == loader.coverage_plan.batch_indices(start_step+i)
            if expected is not None:
                assert record == expected[i], f"Data/order mismatch in {name} at batch {i}"
            records.append(record)
            worker_stats[stats["pid"]] = stats
            if i >= warmup:
                waits.append(elapsed)
            if (i + 1) % 10 == 0:
                write(output / "progress.json", dict(phase="benchmark", case=name, completed=i+1,
                                                     total=warmup+batches, elapsed_s=time.perf_counter()-started))
        total = time.perf_counter() - steady_start
        first_batch = time.perf_counter() - started - total
    finally:
        close_loader(loader)
    result = dict(name=name, options=dataclasses.asdict(options) if options is not None else None,
                  warmup_batches=warmup, measured_batches=batches, startup_and_warmup_s=first_batch,
                  steady_seconds=total, batches_per_second=batches/total,
                  samples_per_second=batches*config.batch_size/total,
                  wait_mean_s=float(np.mean(waits)), wait_p50_s=float(np.percentile(waits,50)),
                  wait_p95_s=float(np.percentile(waits,95)), exact_input_order_equal=expected is not None,
                  worker_stats=list(worker_stats.values()), records=records)
    write(output / f"{name}.json", result)
    print("DONE",name, "samples/s",result["samples_per_second"],"p95",result["wait_p95_s"],flush=True)
    return result


def raw_parity(config, raw, output):
    dc = config.data.create(config.assets_dirs, config.model)
    optimized = video_loading.configure_sources(raw, dc, video_loading.VideoLoadingOptions())
    original_classes = [type(source) for source in raw._datasets]
    original_keys = [set(source.__dict__) for source in raw._datasets]
    lengths = [len(source) for source in raw._datasets]
    boundaries = []
    for source_index, source in enumerate(raw._datasets):
        offset = sum(lengths[:source_index])
        candidates = [0, 1, len(source)-16, len(source)-2, len(source)-1]
        for episode in [0, len(source.episode_data_index["to"])//2,
                        len(source.episode_data_index["to"])-1]:
            end = int(source.episode_data_index["to"][episode])
            candidates.extend([max(0,end-16),max(0,end-1)])
        boundaries.extend(offset+index for index in dict.fromkeys(candidates))
    verified = []
    for index in boundaries:
        before = raw[index]
        after = optimized[index]
        assert leaf_records(before) == leaf_records(after), f"Raw boundary mismatch: {index}"
        verified.append(index)
    assert [type(source) for source in raw._datasets] == original_classes
    assert [set(source.__dict__) for source in raw._datasets] == original_keys
    # Direct seek tests include shuffled/duplicate timestamps and tolerance failure.
    real = raw._datasets[1]
    path = real.root / real.meta.get_video_file_path(0, "observation.images.cam_head")
    cache = video_loading.PyAVDecoderCache(2, 1)
    pts_checks = []
    from lerobot.common.datasets.video_utils import decode_video_frames
    for query in [[0.0],[1/30],[100.0],[(lengths[1]-1)/30],
                  [3/30,1/30,3/30], [500.0,500.0+1/30]]:
        expected = decode_video_frames(path, query, real.tolerance_s, "pyav")
        actual = cache.decode(path, query, real.tolerance_s)
        np.testing.assert_array_equal(np.asarray(expected), np.asarray(actual))
        pts_checks.append(dict(query=query, selected_pts=cache.last_pts))
    try:
        cache.decode(path, [1/60], 0.000001)
    except AssertionError:
        pass
    else:
        raise AssertionError("Out-of-tolerance timestamp was accepted")
    cache.close()
    result = dict(passed=True, boundary_indices=verified, raw_leaves_bitwise_equal=True,
                  source_objects_unchanged=True, timestamp_checks=pts_checks,
                  out_of_tolerance_rejected=True)
    write(output / "raw_parity.json", result)
    return result


def plan_parity(config, raw, output):
    dc=config.data.create(config.assets_dirs,config.model)
    optimized=video_loading.configure_sources(raw,dc,video_loading.VideoLoadingOptions())
    before=data_loader._coverage_plan(dc,config.batch_size,raw)
    after=data_loader._coverage_plan(dc,config.batch_size,optimized)
    digest=hashlib.sha256()
    counts=np.zeros(2,dtype=np.int64)
    for step in range(before.steps):
        lhs=before.batch_indices(step);rhs=after.batch_indices(step)
        assert lhs==rhs
        digest.update(np.asarray(lhs,dtype=np.int64).tobytes())
        for index in lhs:counts[int(index>=before.lengths[0])]+=1
    resume=[]
    for step in [0,2,37,2000,before.steps-1,before.steps]:
        before.set_start_step(step);after.set_start_step(step)
        assert len(before)==len(after)
        if step<before.steps:assert next(iter(before))==next(iter(after))
        resume.append(step)
    result=dict(passed=True,steps=before.steps,source_counts=counts.tolist(),
                all_batch_indices_sha256=digest.hexdigest(),resume_steps=resume,
                real_padding=before.padding,real_unique=int(len(np.unique(before.real_order))))
    write(output/'sampling_parity.json',result)
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    parser.add_argument('--raw-cache',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--batches',type=int,default=100)
    args=parser.parse_args()
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    if output.joinpath('benchmark_summary.json').exists():
        raise FileExistsError('Use a fresh benchmark output directory')
    os.nice(15)
    torch.set_num_threads(1)
    config=config_io.load_train_config(args.config)
    with Path(args.raw_cache).open('rb') as stream:raw=pickle.load(stream)
    identity=dict(config_sha256=video_loading.config_sha256(config),
                  raw_cache_bytes=Path(args.raw_cache).stat().st_size,
                  cpu_affinity=sorted(os.sched_getaffinity(0)),devices=[str(d) for d in jax.devices()],
                  config_path=args.config)
    assert all(device.platform=='cpu' for device in jax.devices())
    write(output/'environment.json',identity)
    write(output/'progress.json',dict(phase='raw_and_sampling_parity'))
    raw_parity(config,raw,output)
    plan_parity(config,raw,output)
    screens=[]
    baseline=run_case(config,raw,None,output,'screen_baseline',warmup=2,batches=10)
    screens.append(baseline)
    candidates=[
        ('skip_only_w2_p2',video_loading.VideoLoadingOptions(workers=2,decoder_cache_size=0,
                                                           batch_video_queries=False,optimized_decode=False)),
        ('cached_w2_p2',video_loading.VideoLoadingOptions(workers=2)),
        ('cached_w4_p2',video_loading.VideoLoadingOptions(workers=4)),
        ('cached_w8_p2',video_loading.VideoLoadingOptions(workers=8)),
        ('cached_w8_p4',video_loading.VideoLoadingOptions(workers=8,prefetch_factor=4)),
    ]
    for name,options in candidates:
        screens.append(run_case(config,raw,options,output,'screen_'+name,warmup=2,batches=10,
                                expected=baseline['records']))
    selected=min(screens[1:],key=lambda row:row['steady_seconds'])
    options=video_loading.VideoLoadingOptions(**selected['options'])
    # ABBA controls warm-cache drift; all four consume the same sample prefix.
    # Match the warmup across cases and cover the largest prefetch window.
    full_warmup=max(32,options.workers*options.prefetch_factor)
    full_baseline1=run_case(config,raw,None,output,'baseline_repeat1',warmup=full_warmup,batches=args.batches)
    expected=full_baseline1['records']
    best1=run_case(config,raw,options,output,'optimized_repeat1',warmup=full_warmup,batches=args.batches,expected=expected)
    best2=run_case(config,raw,options,output,'optimized_repeat2',warmup=full_warmup,batches=args.batches,expected=expected)
    full_baseline2=run_case(config,raw,None,output,'baseline_repeat2',warmup=full_warmup,batches=args.batches,expected=expected)
    resumed=run_case(config,raw,options,output,'optimized_resume37',warmup=0,batches=3,
                     start_step=37,expected=expected[37:40])
    speedups=[full_baseline1['steady_seconds']/best1['steady_seconds'],
              full_baseline2['steady_seconds']/best2['steady_seconds']]
    summary=dict(passed=min(speedups)>1.15 and args.batches>=100,
                 selected_options=dataclasses.asdict(options),minimum_speedup=min(speedups),
                 repeat_speedups=speedups,all_compared_input_leaves_bitwise_equal=True,
                 full_sampling_and_resume_equal=True,measured_batches_per_repeat=args.batches,
                 baseline_samples_per_second=[full_baseline1['samples_per_second'],full_baseline2['samples_per_second']],
                 optimized_samples_per_second=[best1['samples_per_second'],best2['samples_per_second']],
                 screening=[{k:v for k,v in row.items() if k not in ['records','worker_stats']} for row in screens],
                 resume_verified=resumed['exact_input_order_equal'],environment=identity)
    write(output/'benchmark_summary.json',summary)
    write(output/'progress.json',dict(phase='complete',passed=summary['passed']))
    print('BENCHMARK_COMPLETE',json.dumps(summary),flush=True)


if __name__=='__main__':
    try:
        main()
    except BaseException as error:
        traceback.print_exc()
        raise
