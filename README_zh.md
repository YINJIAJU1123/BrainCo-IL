# BrainCo-IL

BrainCo-IL 是基于 [OpenPI](https://github.com/Physical-Intelligence/openpi)
改造的 BrainCo 训练与策略运行仓库。当前只保留实际使用的核心链路：

- JAX PI0 / PI0.5 与 ACT 模型
- LeRobot 数据加载
- BrainCo state、图像、action 与 delta transform
- 预训练权重的部分加载
- checkpoint 内自描述的 `train_config.yaml`
- 供 `revo_deploy` 调用的 external deploy plugin

上游 Aloha、DROID、Libero 示例、PI0-FAST、RLDS、PyTorch PI0、
WebSocket policy server 和旧 BrainCo ROS2 部署方案均已删除。

## 关节顺序

默认 56D 顺序为：

```text
left_arm(7) + right_arm(7) + left_hand(21) + right_hand(21)
```

真正的输入输出语义由 checkpoint 中序列化的
`BrainCoPolicyIOConfig` 决定。right-arm 28D 等裁剪模型也使用同一套
group contract。

## 安装

```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync
```

## 训练

优先从 `src/openpi/training/training_config_template/` 中选择 YAML：

```bash
uv run python scripts/compute_norm_stats.py \
  --config-path \
  src/openpi/training/training_config_template/pi05_brainco_revo3_0712_ght_56d.yaml

uv run python scripts/train.py \
  src/openpi/training/training_config_template/pi05_brainco_revo3_0712_ght_56d.yaml
```

也可以从 `pi05_brainco_56d` 或 `act_brainco_56d` 基准 recipe 出发，
通过 CLI 覆写 batch size、action horizon、学习率和训练步数等标量。

训练会把最终展开后的完整配置写入 run 根目录和每个 step checkpoint
中的 `train_config.yaml`。部署需要的完整产物只有：

```text
train_config.yaml + params/ + assets/
```

## 部署握手

`revo_deploy` 只需要导入：

```text
openpi.deploy.plugin
```

plugin 读取 checkpoint 的 `train_config.yaml`，描述输入输出 contract，
加载 JAX 模型与权重，并负责模型相关预处理。部署端只发送原始 RGB
`uint8` HWC 图像和约定顺序的 joint state。

详细说明见：

- `vla_trainning_deploy_handshake.md`
- `docs/brainco_il_beginner/README.md`
- `docs/jax_in_brainco_il.md`
- `src/openpi/deploy/plugin.py`

## 核心目录

```text
scripts/train.py                         训练主循环
scripts/compute_norm_stats.py            数据归一化统计
src/openpi/models/                       PI0/PI0.5 与 ACT 网络
src/openpi/training/                     配置、数据加载、优化器、checkpoint
src/openpi/policies/brainco_policy.py    BrainCo 输入输出 transform
src/openpi/deploy/plugin.py              外部部署边界
```

## 许可证

Apache-2.0，继承自上游 OpenPI。
