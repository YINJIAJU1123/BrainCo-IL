# BrainCo-IL 训练知识链

这套文档面向第一次接触 VLA、JAX 和 BrainCo-IL 训练代码的读者。目标不是逐个解释 API，而是把训练过程中出现的对象按因果关系串起来。

## 1. 两条主线

### 工程训练链

```text
训练 YAML
  -> TrainConfig
  -> DataConfigFactory.create()
  -> DataConfig
  -> LeRobot Dataset + transforms
  -> DataLoader
  -> batch = (Observation, Actions)
  -> TrainState
  -> jitted train_step
  -> checkpoint
```

### PI0.5 模型计算链

```text
三路图像 + 任务 prompt + 当前机器人 state
  -> VLM Prefix，提供环境和任务条件

带噪的未来 action chunk + flow timestep
  -> Action Expert
  -> 预测 flow velocity
  -> 训练时计算 loss / 推理时执行 Euler 去噪
```

这两条链在 `train_step()` 中汇合：DataLoader 提供 batch，TrainState 提供模型参数，模型根据 Observation 和真实 Actions 计算 loss。

## 2. 推荐阅读顺序

| 顺序 | 文档 | 核心问题 |
| --- | --- | --- |
| 1 | [训练入口与 JAX](01_train_entry_and_jax.md) | `main()` 启动时依次做了什么？ |
| 2 | [配置系统](02_config_system.md) | YAML 如何变成 `TrainConfig` 和 `DataConfig`？ |
| 3 | [数据流水线与 batch](03_data_pipeline_and_batch.md) | 一条 LeRobot 样本如何变成模型 batch？ |
| 4 | [PI0.5 模型架构](04_pi05_architecture.md) | VLM 和 Action Expert 各自处理什么？ |
| 5 | [TrainState、JIT 与 checkpoint](05_train_state_jit_checkpoint.md) | 参数如何初始化、更新和恢复？ |
| 6 | [28D 与 56D](06_action_dimensions_28d_56d.md) | 改动作维度时到底改了什么？ |

JAX 的更多概念可继续阅读 [JAX 在 BrainCo-IL 中的作用](../jax_in_brainco_il.md)。

## 3. 全文使用的形状符号

以下例子采用当前 28D 配置中的典型值：

```text
B = 32   batch size
H = 16   action_horizon
D = 28   action_dim
L = 256  max_token_len
```

因此一个 batch 中常见的形状为：

```text
图像             [B, 224, 224, 3]
当前 state       [B, D]
prompt/state IDs [B, L]
未来 action      [B, H, D]
```

## 4. 先记住五个对象

| 对象 | 含义 |
| --- | --- |
| `TrainConfig` | 一次训练实验的总配置 |
| `DataConfig` | 已展开的数据路径、归一化统计量和 transform 链 |
| `Observation` | 图像、当前 state、prompt token 等模型条件 |
| `Actions` | 监督目标，即未来一段动作序列 |
| `TrainState` | 当前 step、模型参数、优化器状态和模型结构 |

最容易混淆的是：`state` 不是 action。`state` 描述机器人现在在哪里，`action` 描述接下来要让机器人怎么动。

## 5. 一句话总览

BrainCo-IL 先用 YAML 决定模型和数据规则，再由 CPU DataLoader 产生 `(Observation, Actions)` batch；PI0.5 用 VLM 理解图像、任务和当前状态，用 Action Expert 把噪声动作逐步变成未来动作，训练循环则通过 JAX 自动求导和 Optax 持续更新允许训练的参数。
