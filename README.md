# BrainCo-IL

BrainCo training and inference repository derived from
[OpenPI](https://github.com/Physical-Intelligence/openpi). The maintained
production paths are JAX PI0/PI0.5, ACT, LeRobot 2.1 datasets, BrainCo data
transforms, checkpointing, and the Policy Inferencer process used by `revo_deploy`.

## Train

Copy one concise example and edit it:

```bash
cp configs/experiments/pi05.example.yaml my_experiment.yaml
# Change only `base: pi05` to `base: act` to select ACT.

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
train_config.yaml       # GENERATED / DO NOT EDIT / hash checked
policy_contract.json    # GENERATED / DO NOT EDIT / hash checked
```

`train_config.yaml` rebuilds the exact preprocessing, normalization, model and
unnormalization pipeline. `policy_contract.json` describes named inputs and
grouped absolute action outputs for deployment. Regenerate these artifacts;
manual edits are rejected.

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
pipeline, and returns named grouped `float32[T, D]` absolute actions. PI0.5 and
ACT use the same boundary; `revo_deploy` does not know the model type.

## Core layout

```text
configs/experiments/                  concise pi05/act examples
scripts/train.py                      training loop and generated artifacts
src/openpi/models/                    PI0/PI0.5 and ACT
src/openpi/training/                  config, data, optimizer, checkpoint IO
src/openpi/policies/brainco_policy.py BrainCo transforms and feature selection
src/openpi/deploy/contract.py         generated PolicyContract
src/openpi/deploy/policy_inferencer.py subprocess inference boundary
src/openpi/deploy/protocol.py         framed ndarray transport
```

## License

Apache-2.0, inherited from upstream OpenPI.
