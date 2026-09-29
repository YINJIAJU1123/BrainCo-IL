"""Deterministic source-disjoint EgoDex evaluation, one full horizon per val segment."""
import argparse
import dataclasses
import hashlib
import json
from pathlib import Path
import sys

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import swanlab

from openpi import transforms
from openpi.models import model as model_lib
from openpi.policies import policy as policy_lib
from openpi.training import config_io, config as config_lib, data_loader, sharding

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recipe', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    cfg = config_io.load_train_config(args.recipe)
    val_root = Path(cfg.data.base_config.lerobot_datasets[0].repo_id).parent / 'val'
    val_cfg = dataclasses.replace(cfg.data, base_config=dataclasses.replace(
        cfg.data.base_config, lerobot_datasets=(config_lib.LeRobotDataset(repo_id=str(val_root), weight=1.0),)))
    dc = val_cfg.create(cfg.assets_dirs, cfg.model)
    assert dc.norm_stats is not None and dc.use_quantile_norm
    ds = data_loader.create_torch_dataset(dc, cfg.model.action_horizon, cfg.model)
    full = data_loader.transform_dataset(ds, dc)
    if args.checkpoint:
        model = cfg.model.load(model_lib.restore_params(args.checkpoint / 'params', dtype=jnp.bfloat16))
    else:
        from scripts.train import init_train_state
        _, init_rng = jax.random.split(jax.random.key(cfg.seed))
        state, _ = init_train_state(cfg, init_rng, sharding.make_mesh(cfg.fsdp_devices), resume=False)
        model = nnx.merge(state.model_def, state.params)
        del state
    model.eval()
    evaluate_loss = nnx.jit(lambda m, rng, obs, actions: jnp.mean(m.compute_loss(rng, obs, actions, train=False)))
    policy = policy_lib.Policy(model, rng=jax.random.key(42), transforms=[
        *dc.data_transforms.inputs,
        transforms.Normalize(dc.norm_stats, use_quantiles=dc.use_quantile_norm),
        *dc.model_transforms.inputs,
    ], output_transforms=[*dc.model_transforms.outputs,
        transforms.Unnormalize(dc.norm_stats, use_quantiles=dc.use_quantile_norm),
        *dc.data_transforms.outputs], sample_kwargs={'num_steps': 10})
    episodes = [json.loads(line) for line in (val_root/'meta/episodes.jsonl').read_text().splitlines()]
    records = []; offset = 0
    slices = {'left_arm': slice(0,7), 'right_arm': slice(7,14),
              'left_hand': slice(14,35), 'right_hand': slice(35,56)}
    for ep in episodes:
        n = ep['length']; index = offset + min(n//2, n-16)
        raw = ds[index]; prepared = full[index]
        assert not np.asarray(raw['action_is_pad']).any()
        observation = model_lib.Observation.from_dict(jax.tree.map(lambda x: jnp.asarray(x)[None], prepared))
        loss = float(evaluate_loss(model, jax.random.fold_in(jax.random.key(42), index),
                                  observation, jnp.asarray(prepared['actions'])[None]))
        noise = np.random.default_rng(42 + index).standard_normal((16,56)).astype(np.float32)
        result = policy.infer({'observation/state': np.asarray(raw['observation.state']),
                               'observation/image': np.asarray(raw[cfg.data.head_camera_key]),
                               'prompt': raw['prompt']}, noise=noise)
        predicted = np.asarray(result['actions']); expected = np.asarray(raw['action'])
        assert predicted.shape == expected.shape == (16,56)
        assert np.isfinite(predicted).all() and np.isfinite(loss)
        row = dict(episode=ep['episode_index'], dataset_index=index, task=raw['prompt'], loss=loss,
                   mae_radians={k: float(np.abs(predicted[:,v]-expected[:,v]).mean()) for k,v in slices.items()})
        records.append(row); print(json.dumps(row), flush=True); offset += n
    summary = dict(checkpoint=str(args.checkpoint) if args.checkpoint else 'base with seeded 56D interface',
                   val_root=str(val_root), samples=len(records), horizon=16, padded_positions_scored=False,
                   train_statistics=True, seed=42, inference_steps=10,
                   recipe_sha256=hashlib.sha256(args.recipe.read_bytes()).hexdigest(),
                   loss=float(np.mean([r['loss'] for r in records])),
                   mae_radians={k:float(np.mean([r['mae_radians'][k] for r in records])) for k in slices},
                   per_task={}, records=records)
    for task in sorted({r['task'] for r in records}):
        selected = [r for r in records if r['task']==task]
        summary['per_task'][task] = dict(samples=len(selected),loss=float(np.mean([r['loss'] for r in selected])),
            mae_radians={k:float(np.mean([r['mae_radians'][k] for r in selected])) for k in slices})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2))
    import os
    run = swanlab.init(project=cfg.project_name, experiment_name=cfg.exp_name+('_eval' if args.checkpoint else '_base_eval'),
                      mode=os.environ.get('SWANLAB_MODE','local'), logdir=os.environ.get('SWANLAB_LOG_DIR','./swanlog'),
                      config=dict(validation_samples=len(records), seed=42, checkpoint=summary['checkpoint']))
    swanlab.log({'validation/loss':summary['loss'], **{'validation/'+k+'_mae_rad':v for k,v in summary['mae_radians'].items()}},step=0)
    swanlab.finish()


if __name__ == '__main__':
    main()
