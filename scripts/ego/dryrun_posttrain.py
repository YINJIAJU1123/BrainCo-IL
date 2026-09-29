#!/usr/bin/env python3
"""Read one real mixed batch and run a forward pass without weight updates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import flax.nnx as nnx
import jax
import numpy as np

from openpi.training import config_io, data_loader, sharding
from scripts.train import init_train_state


def dryrun(config_path: Path) -> dict:
    config = config_io.load_train_config(config_path)
    mesh = sharding.make_mesh(config.fsdp_devices)
    batch_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    loader = data_loader.create_data_loader(config, sharding=batch_sharding, shuffle=False, num_batches=1)
    batch = next(iter(loader))
    state, _ = init_train_state(config, jax.random.key(config.seed), mesh, resume=False)
    model = nnx.merge(state.model_def, state.params)
    model.eval()
    observation, actions = batch
    loss = model.compute_loss(jax.random.key(config.seed + 1), observation, actions, train=False)
    loss = np.asarray(jax.device_get(loss))
    action_values = np.asarray(jax.device_get(actions))
    if loss.shape != (config.batch_size, config.model.action_horizon):
        raise ValueError(f"unexpected loss shape {loss.shape}")
    if not np.isfinite(loss).all() or not np.isfinite(action_values).all():
        raise FloatingPointError("nonfinite forward loss or action batch")
    masks = {key: np.asarray(jax.device_get(value)).astype(bool).tolist()
             for key, value in observation.image_masks.items()}
    return {"experiment": config.exp_name, "batch_shape": list(action_values.shape),
            "forward_loss_mean": float(loss.mean()), "image_masks": masks,
            "optimizer_updates": 0}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    print(json.dumps(dryrun(args.config), indent=2))


if __name__ == "__main__":
    main()
