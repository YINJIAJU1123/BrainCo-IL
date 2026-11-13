# BrainCo-IL

Deployment compatibility: **revo inference architecture v1.1** (`v1.1` tag).

BrainCo training and inference repository derived from
[OpenPI](https://github.com/Physical-Intelligence/openpi). The maintained
production paths are JAX PI0/PI0.5, ACT, Diffusion Policy, LeRobot 2.1 datasets, BrainCo data
transforms, checkpointing, and the Policy Inferencer process used by `revo_deploy`.

## Train

Copy one concise example and edit it:

```bash
cp configs/experiments/pi05.example.yaml my_experiment.yaml
# Select base: pi05, act, or diffusion.
# For diffusion, start with configs/experiments/diffusion.example.yaml.

uv run python scripts/compute_norm_stats.py --config-path my_experiment.yaml
uv run python scripts/train.py my_experiment.yaml
```

The human-authored YAML selects dataset, semantic groups, action horizon and
training hyperparameters. It never contains joint names or numeric state/action
slices. Those are resolved from the LeRobot dataset's `meta/info.json`.

Each saved step contains:

```text
params/
assets/
train_config.yaml       # GENERATED / DO NOT EDIT / content + checkpoint ID checked
policy_contract.json    # GENERATED / DO NOT EDIT / contract hash checked
```

`train_config.yaml` rebuilds the exact preprocessing, normalization, model and
unnormalization pipeline. `policy_contract.json` describes named inputs and
grouped absolute action outputs for deployment. Regenerate these artifacts;
manual edits are rejected. Both generated files carry the same checkpoint ID,
so mixing artifacts from different training steps fails before model loading.

## Policy Inferencer

The only production deployment entry point is an independent process:

```bash
uv run python -m openpi.deploy.policy_inferencer --checkpoint /path/to/checkpoint
```

It implements framed protocol v1:

```text
DESCRIBE -> CONTRACT
LOAD -> READY
INFER -> RESULT
RESET / CLOSE / ERROR
```

The inferencer has no ROS dependency. It accepts named joint groups and raw RGB
images, orders/selects features using checkpoint metadata, runs the full model
pipeline, and returns named grouped `float32[T, D]` absolute actions. PI0.5, ACT and
Diffusion Policy use the same boundary; `revo_deploy` does not know the model type. Runtime
capabilities come from `DESCRIBE`: PI0/PI0.5 advertise `standard` and `rtc`, while
ACT and Diffusion Policy advertise `standard`.

## Diffusion Policy

See [the integration guide](docs/diffusion_policy.md) for architecture, training,
export and `revo_deploy` registration/validation commands. This is a JAX adaptation
of the official CNN + conditional U-Net policy: epsilon MSE, cosine DDPM or DDIM,
and EMA deployment weights. It uses **one observation frame from three cameras**,
BrainCo quantile normalization and internal arm deltas, and returns absolute joint
positions. Multi-frame history, language conditioning, RTC, and importing upstream
PyTorch checkpoints are not supported. Training and inference use float32.

The trainer finalizes saved steps through `openpi.deploy.export_checkpoint`,
which re-reads generated configs and validates IDs, hashes and normalization assets.
To regenerate the standard artifacts for an existing saved step:

```bash
uv run python -m openpi.deploy.export_checkpoint --checkpoint /absolute/checkpoint/step
```

## Core layout

```text
configs/experiments/                  concise pi05/act/diffusion examples
scripts/train.py                      training loop and generated artifacts
src/openpi/models/                    PI0/PI0.5, ACT and Diffusion Policy
src/openpi/training/                  config, data, optimizer, checkpoint IO
src/openpi/policies/brainco_policy.py BrainCo transforms and feature selection
src/openpi/deploy/contract.py         generated PolicyContract
src/openpi/deploy/policy_inferencer.py subprocess inference boundary
src/openpi/deploy/protocol.py         framed ndarray transport
```

## License

Apache-2.0, inherited from upstream OpenPI. Adapted Diffusion Policy portions retain
their MIT notice; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
