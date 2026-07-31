# BrainCo-IL 中的 VLASH 配置与数据流

VLASH 不是新的 `ModelType`。BrainCo-IL 仍然使用 PI0.5,只增加:

```text
temporal-offset 训练样本
+ future-state condition
+ checkpoint 中的异步执行语义
```

部署端的 active/pending chunk 调度仍由 `revo_deploy` 负责。

## 1. 当前实现范围

已经支持:

- `Pi0Config.state_cond`
- `max_delay_steps`
- `use_state_ground_truth`
- 随机单 offset 训练
- deploy plugin 的 VLASH execution contract

暂不支持:

- `shared_observation=true`
- 一个 prefix 同时连接多个 offset suffix 的联合 attention
- revo-deploy 的 active/pending chunk 状态机

`shared_observation` 是训练加速手段,不是 VLASH future-state-aware
异步推理成立的必要条件。当前 JAX 配置在启用它时会直接报错,避免配置
看似生效但实际仍走普通 attention。

## 2. 与 VLASH 官方配置的对应关系

| 官方字段 | BrainCo-IL 字段 | 当前含义 |
| --- | --- | --- |
| `policy.state_cond` | `model.state_cond` | 连续 state 进入 Action Expert adaRMS |
| `max_delay_steps` | `data.max_delay_steps` | 均匀采样 `offset∈[0,N]` |
| `shared_observation` | `data.shared_observation` | 已保留字段,当前必须为 `false` |
| action proxy future state | `use_state_ground_truth=false` | 使用 `a_(t+δ-1)` 近似 future state |
| ground-truth future state | `use_state_ground_truth=true` | 使用数据集记录的 `s_(t+δ)` |

BrainCo 数据有完整 56D proprioceptive state,因此示例默认:

```yaml
use_state_ground_truth: true
```

## 3. 训练样本

标准 PI0.5:

```text
(image_t, state_t) -> action[t:t+H]
```

VLASH:

```text
offset δ ~ UniformInteger(0, max_delay_steps)

固定:
  image_t
  task_t

平移:
  state  = state_(t+δ)
  target = action[t+δ:t+δ+H]
```

当前训练循环没有 action padding loss mask,因此 VLASH dataset 只保留对
所有 offset 都能形成完整 action chunk 的锚点:

```text
t + max_delay_steps + H <= episode_end
```

这样 loss 的 shape 仍然是:

```text
[B, H]
```

不需要改变 `train_step()`。

## 4. state condition

`state_cond=false` 时沿用原 PI0.5:

```text
离散 state -> prompt -> VLM prefix
```

`state_cond=true` 时与 VLASH 官方 PI0.5 路径一致:

```text
VLM prefix:
  images + "Task: ...; Action:"

Action Expert suffix:
  noisy action tokens

adaRMS condition:
  time_condition + state_condition
```

其中:

```text
state [B,D]
  -> state_proj
  -> state_mlp_in
  -> SiLU
  -> state_mlp_out (zero initialized)
  -> SiLU
  -> [B,1024]
```

最后一层零初始化使原始 PI0.5 checkpoint 刚加载时 state condition 为零,
之后再通过 VLASH 微调学习。

## 5. 56D 示例配置

配置文件:

[`pi05_brainco_revo3_0712_ght_56d_vlash.yaml`](../../src/openpi/training/training_config_template/pi05_brainco_revo3_0712_ght_56d_vlash.yaml)

当前建议值:

```yaml
model:
  state_cond: true
  discrete_state_input: false

data:
  max_delay_steps: 6
  shared_observation: false
  use_state_ground_truth: true
```

这里 `max_delay_steps=6` 对应 30 Hz 下最多约 200 ms 的训练错位范围。
它应根据部署端:

```text
chunk_handoff_timestamp - image_timestamp
```

的 p95/p99 实测值调整,而不是只看模型 forward 时间。

训练前需要用该配置重新计算 norm stats:

```bash
uv run python scripts/compute_norm_stats.py \
  --config-path \
  src/openpi/training/training_config_template/pi05_brainco_revo3_0712_ght_56d_vlash.yaml
```

然后训练:

```bash
uv run python scripts/train.py \
  src/openpi/training/training_config_template/pi05_brainco_revo3_0712_ght_56d_vlash.yaml
```

## 6. 部署 contract

当 `max_delay_steps>0` 时,deploy plugin 会额外声明:

```yaml
execution:
  schema_version: 1
  mode: vlash_async
  supports_future_state: true
  future_state_key: observation.state
  future_state_semantics: predicted_state_at_chunk_handoff
  future_state_space: raw_absolute_joint_position
  requires_unblended_chunk_boundaries: true
  requires_low_pass_disabled: true
  max_trained_delay_steps: 6
  state_conditioning: adarms
```

revo-deploy 应发送未归一化的 raw absolute 56D future state。BrainCo-IL
plugin 继续负责 BrainCoInputs、Normalize 和模型输入变换。

当前 revo-deploy 握手约定只暴露两个运行项:

- 是否启用 VLASH
- `inference_overlap_steps`

VLASH 模式保留 chunk 内从 policy rate 到 control rate 的线性插值,但关闭
startup transition、跨 chunk boundary blend 和 ActionWriter low-pass。
live 与 dataset observation source 都可异步执行。dataset 模式第一次从
`t` 前进 `H-overlap` 帧取得触发时刻图像,之后每次前进 `H` 帧;模型 state
仍由 Actor 使用当前 chunk 的末帧 action 覆盖。
