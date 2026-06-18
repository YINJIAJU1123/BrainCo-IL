# BrainCo OpenPI Deployment Example

This directory contains an optional ROS2 client example for running a trained OpenPI policy against a BrainCo-style dual-arm + dual-dexterous-hand robot.

The training/model-side reference layout is:

```text
left_arm(7) + left_hand(22) + right_arm(7) + right_hand(22) = 58
```

## Layout

```text
examples/brainco/
├── config/deploy.yaml        deployment configuration
├── core/                     DeployConfig, ROS2Interface, timestamp sync, utilities
├── deploy/                   CLI entry and OpenPI Runtime environment wrapper
├── package.xml               optional ROS2 package manifest
└── README.md
```

## Run

Start the policy server from the repository root:

```bash
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_brainco_multi_58d \
  --policy.dir=/path/to/checkpoint \
  --host=0.0.0.0 \
  --port=8000
```

Then run the ROS2 client in an environment where `rclpy` and `openpi-client` are available:

```bash
PYTHONPATH=examples python3 -m brainco.deploy.main -c examples/brainco/config/deploy.yaml
```

Common overrides:

```bash
PYTHONPATH=examples python3 -m brainco.deploy.main \
  -c examples/brainco/config/deploy.yaml \
  --host 192.168.1.100 \
  --port 8000 \
  --side both \
  --broker-mode rtg
```

## Configuration

Main config: `examples/brainco/config/deploy.yaml`

Important fields:

```yaml
arm_mode: "dual"
arm_dof: 7
hand_dof: 22
server_host: localhost
server_port: 8000
action_horizon: 50
control_hz: 30.0
broker_mode: "rtg"
```

The ROS2 observation state is packed in this order:

```text
left_arm + left_hand + right_arm + right_hand
```

For a different hand dimension, update `hand_dof` here and keep the training config, dataset, norm stats, and policy output dimension aligned.

## Expected Policy Inputs

The policy server expects observations compatible with `BrainCoInputs`:

```text
observation/state
observation/image
observation/left_wrist_image
observation/right_wrist_image
prompt
```

## Troubleshooting

- `ModuleNotFoundError: No module named 'brainco'`: run from the repository root, or add `examples` to `PYTHONPATH`.
- `ModuleNotFoundError: No module named 'rclpy._rclpy_pybind11'`: use the Python interpreter from your ROS2 Humble environment.
- WebSocket disconnects: check the policy server logs and verify that `pi05_brainco_multi_58d`, norm stats, and observation/action dimensions match.
- Missing ROS2 data: verify topic names in `deploy.yaml` with `ros2 topic list` and `ros2 topic hz`.

## Notes

This deployment package is an example client. The training and policy-serving path does not depend on using this ROS2 stack.
