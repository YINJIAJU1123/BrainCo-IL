# brainco-openpi

面向 BrainCo 双臂 + 双灵巧手机器人的 [openpi](https://github.com/Physical-Intelligence/openpi) 分支，用于 pi0 / pi0.5 VLA 策略的监督微调、策略服务和可选 ROS2 部署。

本仓库保留上游 OpenPI 模型栈，并增加 BrainCo 高维动作管线：

- `src/openpi/policies/brainco_policy.py`：BrainCo 输入/输出 transform
- `src/openpi/training/config.py`：BrainCo LeRobot 数据配置
- 参考 pi0.5 训练配置：`pi05_brainco_multi_56d`
- `examples/brainco/`：可选 ROS2/OpenPI 部署示例
- `PartialCheckpointWeightLoader`：支持跨 action 维度加载可兼容的 base 权重

## 参考维度

当前 BrainCo 参考布局为 56 维：

```text
left_arm(7) + left_hand(21) + right_arm(7) + right_hand(21) = 56
```

`observation.state` 和 `action` 必须使用同一拼接顺序。

## 数据格式

参考 `LeRobotBrainCoDataConfig` 期望 LeRobot 样本包含：

```text
observation.state                      float32[56]
observation.images.stereo_right         uint8[H, W, 3]
observation.images.cam_left_wrist       uint8[H, W, 3]
observation.images.cam_right_wrist      uint8[H, W, 3]
action                                  float32[action_horizon, 56]
prompt                                  string
```

如果你的数据转换脚本使用了不同的图像 key，请修改 `LeRobotBrainCoDataConfig` 里的 `RepackTransform`。

## 训练

先在 `src/openpi/training/config.py` 的 `pi05_brainco_multi_56d` 中填写数据集路径，然后计算归一化统计并训练：

```bash
uv run scripts/compute_norm_stats.py --config-name pi05_brainco_multi_56d

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
  uv run scripts/train.py pi05_brainco_multi_56d \
  --exp-name=my_brainco_run
```

该配置使用 `PartialCheckpointWeightLoader`：加载 pi0.5 base checkpoint 时，形状匹配的权重正常加载，动作/状态投影层等形状不匹配的层随机初始化。这是从上游 action 维度迁移到 56 维的核心机制。

## 策略服务

```bash
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_brainco_multi_56d \
  --policy.dir=/path/to/checkpoint
```

策略服务期望输入 key 与 `BrainCoInputs` 对齐：

```text
observation/state
observation/image
observation/left_wrist_image
observation/right_wrist_image
prompt
```

## 修改维度

换构型时，保持关节顺序明确，并同时修改：

- 训练配置里的 `model.action_dim`
- `src/openpi/policies/brainco_policy.py` 里的 `BRAINCO_ACTION_DIM`
- `LeRobotBrainCoDataConfig` 里的 `DeltaActions` mask
- 数据集 `observation.state` 和 `action` 宽度
- 重新计算 norm stats

当前参考配置每只手 21 维：

```python
_transforms.make_bool_mask(7, -21, 7, -21)
```

## 目录

```text
examples/brainco/                  可选 ROS2 部署示例
packages/openpi-client/             client/runtime 工具
scripts/                            train、compute_norm_stats、serve_policy
src/openpi/policies/brainco_policy.py
src/openpi/training/config.py
```

## 安装

```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

## 上游

本项目基于 OpenPI，仍保留 ALOHA、DROID、LIBERO 和 pi0/pi0.5 的上游示例。模型细节请参考 [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi)。

## 许可证

Apache-2.0，继承自上游 OpenPI。
