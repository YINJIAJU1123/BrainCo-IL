"""Check the 100 h EgoDex LeRobot batch and optionally compute train-only stats."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
import pyarrow.parquet as pq

from openpi.training import config_io


ROOT = Path(__file__).resolve().parents[2]
DATASET = Path('/mnt/data_nas/ruibin/dataset/egodex_100h_offset0_lerobot21')
TEMPLATE = ROOT / 'src/openpi/training/training_config_template/ego/egodex_pi05_100h_offset0_20k.yaml'
ASSETS = ROOT / 'assets/egodex_pi05_100h_offset0_20260923/egodex_100h_offset0_train'
CAMERAS = ('observation.images.cam_head', 'observation.images.cam_left_wrist',
           'observation.images.cam_right_wrist')


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open() if line.strip()]


def _check_video(path: Path, frames: int, *, black: bool) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened() or int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) != frames:
        cap.release()
        raise ValueError(f'Video count mismatch: {path}')
    ok, image = cap.read()
    cap.release()
    if not ok or (black and np.any(image)):
        raise ValueError(f'Invalid camera frame: {path}')


def _check_split(root: Path, expected_episodes: int, expected_frames: int) -> dict:
    info = json.loads((root / 'meta/info.json').read_text())
    episodes = _load_jsonl(root / 'meta/episodes.jsonl')
    if (info['codebase_version'] != 'v2.1' or info['fps'] != 30
            or info['total_episodes'] != expected_episodes
            or info['total_frames'] != expected_frames
            or info['total_videos'] != expected_episodes * 3
            or len(episodes) != expected_episodes):
        raise ValueError(f'Invalid metadata totals: {root}')
    features = info['features']
    if (features['action']['shape'] != [56]
            or features['observation.state']['names'] != features['action']['names']
            or any(features[key]['dtype'] != 'video' for key in CAMERAS)):
        raise ValueError(f'Invalid action/camera features: {root}')
    provenance = json.loads((root / 'meta/provenance.json').read_text())
    if provenance['offset_m_per_hand'] != 0:
        raise ValueError(f'Unexpected outward offset: {root}')
    for pos in sorted({0, expected_episodes // 2, expected_episodes - 1}):
        episode = episodes[pos]
        index = episode['episode_index']
        length = episode['length']
        if index != pos or length < 16:
            raise ValueError(f'Invalid episode: {root}, {pos}')
        chunk = index // info['chunks_size']
        parquet = root / info['data_path'].format(episode_chunk=chunk, episode_index=index)
        table = pq.read_table(parquet, columns=['action', 'observation.state'])
        if len(table) != length or len(table['action'][0].as_py()) != 56:
            raise ValueError(f'Invalid parquet dimensions: {parquet}')
        for key in CAMERAS:
            video = root / info['video_path'].format(
                episode_chunk=chunk, video_key=key, episode_index=index)
            _check_video(video, length, black=key != CAMERAS[0])
    return {'episodes': expected_episodes, 'frames': expected_frames,
            'hours': expected_frames / 108000, 'sampled_episodes': 3}


def prepare(dataset: Path, *, compute_stats: bool, loader_check: bool) -> dict:
    report = json.loads((dataset / 'conversion_report.json').read_text())
    if report['status'] != 'complete':
        raise ValueError('LeRobot conversion is not complete')
    expected = {'train': (64502, 10809850), 'val': (3278, 545248)}
    splits = {name: _check_split(dataset / name, *counts) for name, counts in expected.items()}
    plan = json.loads((dataset / 'plan.json').read_text())
    sources = {name: {row['source'] for row in plan['sources'] if row['split'] == name}
               for name in expected}
    if sources['train'] & sources['val']:
        raise ValueError('Source leakage between train and validation')
    cfg = config_io.load_train_config(TEMPLATE)
    if (cfg.num_train_steps != 20000 or cfg.batch_size != 16 or cfg.resume
            or cfg.data.egodex_head_only or not cfg.data.mask_wrist_images
            or cfg.data.policy_io.dataset_state_dim != 56
            or Path(cfg.data.base_config.lerobot_datasets[0].repo_id) != dataset / 'train'):
        raise ValueError('100 h training template does not match the data contract')
    stats = ASSETS / 'norm_stats.json'
    identity = {'train_info_sha256': _sha256(dataset / 'train/meta/info.json'),
                'train_episodes_sha256': _sha256(dataset / 'train/meta/episodes.jsonl'),
                'conversion_report_sha256': _sha256(dataset / 'conversion_report.json'),
                'horizon': 16, 'arm_dims': 14, 'train_frames': 10809850}
    stamp = ASSETS / 'statistics_provenance.json'
    if stats.is_file():
        if not stamp.is_file() or json.loads(stamp.read_text()) != identity:
            raise ValueError('Existing normalization statistics have unknown provenance')
    elif compute_stats:
        subprocess.run([sys.executable, str(ROOT / 'scripts/compute_brainco_norm_stats_from_parquet.py'),
                        str(dataset / 'train'), str(ASSETS), '--action-horizon', '16',
                        '--arm-dims', '14'], cwd=ROOT, check=True)
        stamp.write_text(json.dumps(identity, indent=2) + '\n')
    if loader_check:
        if not stats.is_file():
            raise ValueError('Compute train-only normalization statistics before the loader check')
        from openpi.training import data_loader
        from openpi import transforms
        dc = cfg.data.create(cfg.assets_dirs, cfg.model)
        ds = data_loader.create_torch_dataset(dc, cfg.model.action_horizon, cfg.model)
        pre = transforms.compose([*dc.repack_transforms.inputs, *dc.data_transforms.inputs])
        for index in (0, len(ds) // 2, len(ds) - 1):
            sample = pre(ds[index])
            if sample['actions'].shape != (16, 56):
                raise ValueError(f'Invalid transformed action chunk at {index}')
            masks = sample['image_mask']
            if not masks['base_0_rgb'] or masks['left_wrist_0_rgb'] or masks['right_wrist_0_rgb']:
                raise ValueError(f'Invalid camera attention masks at {index}')
    return {'status': 'ready' if stats.is_file() else 'stats_needed', 'splits': splits,
            'source_counts': {name: len(rows) for name, rows in sources.items()},
            'normalization_stats': str(stats), 'template': str(TEMPLATE)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=DATASET)
    parser.add_argument('--compute-stats', action='store_true')
    parser.add_argument('--loader-check', action='store_true')
    args = parser.parse_args()
    print(json.dumps(prepare(args.dataset, compute_stats=args.compute_stats,
                             loader_check=args.loader_check), indent=2))


if __name__ == '__main__':
    main()
