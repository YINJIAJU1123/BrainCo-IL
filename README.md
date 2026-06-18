# brainco-openpi

BrainCo-focused fork of [openpi](https://github.com/Physical-Intelligence/openpi) for supervised fine-tuning and serving pi0 / pi0.5 VLA policies on dual-arm + dual-dexterous-hand robots.

This repository keeps the upstream OpenPI model stack and adds a BrainCo high-dimensional action pipeline:

- BrainCo policy transforms in `src/openpi/policies/brainco_policy.py`
- BrainCo LeRobot data config in `src/openpi/training/config.py`
- Reference pi0.5 training config: `pi05_brainco_multi_58d`
- Optional ROS2/OpenPI deployment example under `examples/brainco/`
- Partial checkpoint loading for action-dimension changes

## Reference Layout

The reference BrainCo layout is 58D:

```text
left_arm(7) + left_hand(22) + right_arm(7) + right_hand(22) = 58
```

Both `observation.state` and `action` are expected to use the same ordering.

## Data Format

The reference `LeRobotBrainCoDataConfig` expects LeRobot samples with:

```text
observation.state                      float32[58]
observation.images.stereo_right         uint8[H, W, 3]
observation.images.cam_left_wrist       uint8[H, W, 3]
observation.images.cam_right_wrist      uint8[H, W, 3]
action                                  float32[action_horizon, 58]
prompt                                  string
```

If your converted dataset uses different image key names, update the `RepackTransform` in `LeRobotBrainCoDataConfig`.

## Training

Edit the dataset paths in `src/openpi/training/config.py` under `pi05_brainco_multi_58d`, then compute normalization statistics and train:

```bash
uv run scripts/compute_norm_stats.py --config-name pi05_brainco_multi_58d

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
  uv run scripts/train.py pi05_brainco_multi_58d \
  --exp-name=my_brainco_run
```

The config uses `PartialCheckpointWeightLoader` to load compatible pi0.5 base weights while randomly initializing shape-mismatched action/state projection layers. This is what makes changing from the upstream action dimension to 58D straightforward.

## Serving

```bash
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_brainco_multi_58d \
  --policy.dir=/path/to/checkpoint
```

The policy server expects observations using the keys produced by `BrainCoInputs`:

```text
observation/state
observation/image
observation/left_wrist_image
observation/right_wrist_image
prompt
```

## Adapting Dimensions

To change the morphology, keep the joint ordering explicit and update these together:

- `model.action_dim` in the training config
- `BRAINCO_ACTION_DIM` in `src/openpi/policies/brainco_policy.py`
- `DeltaActions` mask in `LeRobotBrainCoDataConfig`
- dataset `observation.state` and `action` widths
- normalization stats, recomputed for the new dataset

For example, if each hand has 21 DOF, the dual-arm total is `7 + 21 + 7 + 21 = 56`, and the delta mask should be:

```python
_transforms.make_bool_mask(7, -21, 7, -21)
```

The current reference uses 22 DOF per hand:

```python
_transforms.make_bool_mask(7, -22, 7, -22)
```

## Repository Layout

```text
examples/brainco/                  optional ROS2 deployment example
packages/openpi-client/             client/runtime utilities
scripts/                            train, compute_norm_stats, serve_policy
src/openpi/policies/brainco_policy.py
src/openpi/training/config.py
```

## Install

```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

## Upstream

This project is based on OpenPI and keeps upstream examples for ALOHA, DROID, LIBERO, and the base pi0/pi0.5 models. See the original [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi) repository for upstream model details.

## License

Apache-2.0, inherited from upstream OpenPI.
