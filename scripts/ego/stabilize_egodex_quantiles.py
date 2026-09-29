"""Use train-only extrema for EgoDex dimensions with collapsed histogram quantiles."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.compute_brainco_norm_stats_from_parquet import _episode_paths, _read_vector_columns, _transform_episode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', type=Path, required=True)
    parser.add_argument('--stats', type=Path, required=True)
    args = parser.parse_args()
    assert args.train.name == 'train'
    backup = args.stats.with_name('norm_stats.histogram_original.json')
    if not backup.exists():
        backup.write_bytes(args.stats.read_bytes())
    document = json.loads(backup.read_text())
    stats = document['norm_stats']
    minimum = {k: np.full(56, np.inf) for k in stats}
    maximum = {k: np.full(56, -np.inf) for k in stats}
    frames = 0
    for _, path, _ in _episode_paths(args.train):
        state, actions = _read_vector_columns(path, ('observation.state', 'action'))
        state, actions = _transform_episode(state, actions, action_horizon=16, arm_dims=14)
        for k, values in [('state', state), ('actions', actions.reshape(-1,56))]:
            minimum[k] = np.minimum(minimum[k], values.min(axis=0))
            maximum[k] = np.maximum(maximum[k], values.max(axis=0))
        frames += len(state)
    changes = {}
    for k, values in stats.items():
        lo, hi = np.array(values['q01']), np.array(values['q99'])
        selected = np.flatnonzero(hi - lo < 1e-6)
        changes[k] = []
        for index in selected:
            assert maximum[k][index] - minimum[k][index] >= 1e-6, 'Truly constant dimension needs explicit policy'
            changes[k].append(dict(index=int(index), old_q01=float(lo[index]), old_q99=float(hi[index]),
                                   new_q01=float(minimum[k][index]), new_q99=float(maximum[k][index])))
            lo[index], hi[index] = minimum[k][index], maximum[k][index]
        values['q01'], values['q99'] = lo.tolist(), hi.tolist()
        assert np.all(hi - lo >= 1e-6)
    args.stats.write_text(json.dumps(document, indent=2))
    report = dict(train_root=str(args.train), train_frames=frames, action_horizon=16, arm_dims=14,
                  rule='For q99-q01 < 1e-6 only, replace bounds with min/max from training data',
                  validation_data_used=False, changes=changes,
                  original_sha256=hashlib.sha256(backup.read_bytes()).hexdigest(),
                  stabilized_sha256=hashlib.sha256(args.stats.read_bytes()).hexdigest())
    args.stats.with_name('quantile_stabilization.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
