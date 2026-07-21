# BrainCo-IL

BrainCo-IL is the BrainCo training and policy-runtime repository derived from
[OpenPI](https://github.com/Physical-Intelligence/openpi). It intentionally keeps
only the paths used by BrainCo robots:

- JAX PI0 / PI0.5 and ACT models
- LeRobot dataset loading
- BrainCo state, image, action, and delta transforms
- partial pretrained-weight loading
- self-describing `train_config.yaml` checkpoints
- the external deploy plugin used by `revo_deploy`

Upstream robot examples, PI0-FAST, RLDS/DROID, the PyTorch PI0 implementation,
WebSocket serving, and the old BrainCo ROS2 example are not part of this repository.

## Joint Layout

The default 56D layout is:

```text
left_arm(7) + right_arm(7) + left_hand(21) + right_hand(21)
```

The authoritative layout is serialized by
`BrainCoPolicyIOConfig` in each checkpoint. Reduced policies such as right-arm
28D use the same group-based contract.

## Install

```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync
```

## Train

Start from one of the YAML files in
`src/openpi/training/training_config_template/`:

```bash
uv run python scripts/compute_norm_stats.py \
  --config-path \
  src/openpi/training/training_config_template/pi05_brainco_revo3_0712_ght_56d.yaml

uv run python scripts/train.py \
  src/openpi/training/training_config_template/pi05_brainco_revo3_0712_ght_56d.yaml
```

Simple scalar values can also be overridden from the CLI when starting from the
`pi05_brainco_56d` or `act_brainco_56d` recipe.

Training writes the fully resolved config to the run directory and every step
checkpoint as `train_config.yaml`. That file, the checkpoint parameters, and
normalization assets are the complete deployment artifact.

## Deploy Contract

`revo_deploy` imports:

```text
openpi.deploy.plugin
```

The plugin reads `train_config.yaml`, reports the observation/action contract,
loads the JAX model and checkpoint, and owns model-specific preprocessing. The
deploy side sends raw RGB `uint8` HWC images and the configured joint state.

See:

- `vla_trainning_deploy_handshake.md`
- `docs/brainco_il_beginner/README.md`
- `docs/jax_in_brainco_il.md`
- `src/openpi/deploy/plugin.py`

## Core Layout

```text
scripts/train.py                         training loop
scripts/compute_norm_stats.py            dataset normalization statistics
src/openpi/models/                       PI0/PI0.5 and ACT networks
src/openpi/training/                     config, loader, optimizer, checkpoints
src/openpi/policies/brainco_policy.py    BrainCo observation/action transforms
src/openpi/deploy/plugin.py              external deployment boundary
```

## License

Apache-2.0, inherited from upstream OpenPI.
