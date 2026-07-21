# 06. 28D 与 56D：数据语义和网络形状

`action_dim` 不只是一个数字，它同时约束数据、模型和部署接口。

## 1. BrainCo 关节组

[`BrainCoPolicyIOConfig`](../../src/openpi/policies/brainco_policy.py) 定义四组关节：

| 关节组 | 维度 |
| --- | ---: |
| `left_arm` | 7 |
| `right_arm` | 7 |
| `left_hand` | 21 |
| `right_hand` | 21 |

典型 56D 布局：

```text
left_arm(7) + right_arm(7) + left_hand(21) + right_hand(21) = 56
```

当前右侧 28D 布局：

```text
right_arm(7) + right_hand(21) = 28
```

顺序必须在数据集、norm stats、训练配置和部署端保持一致。

## 2. 模型中哪些层直接改变形状

PI0.5 初始化时创建：

```python
action_in_proj  = Linear(action_dim, 1024)
action_out_proj = Linear(1024, action_dim)
```

因此：

| 配置 | `action_in_proj` | `action_out_proj` |
| --- | --- | --- |
| 28D | `Linear(28, 1024)` | `Linear(1024, 28)` |
| 56D | `Linear(56, 1024)` | `Linear(1024, 56)` |

以下主体结构不会因为 D 改变：

- SigLIP 图像编码器宽度
- Gemma 2B VLM 宽度和层数
- Gemma 300M Action Expert 宽度和层数
- time MLP 的 1024 维宽度

## 3. 还有哪些 tensor 跟着改变

模型输入规格将 state 维度也设为 `action_dim`：

```text
state   [B, D]
actions [B, H, D]
noise   [B, H, D]
v_t     [B, H, D]
```

所以改 D 不只是改最终输出层，还改变 action chunk、flow noise、loss 和当前 state 的数据合同。

## 4. VLM 为什么不需要换输入层

PI0.5 的 state 先被离散化并写入 prompt：

```text
28D state -> 28 个离散数值对应的文本内容
56D state -> 56 个离散数值对应的文本内容
```

tokenizer 最后都输出固定：

```text
[B, max_token_len]
```

因此 Gemma 2B 的 token embedding 和 2048 维隐藏层形状不变。变化的是 VLM 看到的内容和有效 token 数，不是网络输入宽度。

## 5. `action_in_proj` 会进入 VLM 吗

不会进入 VLM prefix。两路关系是：

```text
图像 + prompt + state tokens
  -> VLM Prefix
       │
       │ 被 Action Expert attention 读取
       ▼
noisy actions -> action_in_proj -> Action Expert -> action_out_proj
```

`action_in_proj` 是 Action Expert 的动作入口。它的输出会参与联合 Transformer attention，但 noisy action 不会成为 Gemma 2B prefix 的输入。

训练时 action loss 仍可沿 attention 路径传回可训练的 VLM/LoRA 参数，这与“action 被输入 VLM”不是一回事。

## 6. 70D、56D 和 28D 数据转换

仓库提供一个特定转换：

```text
70D = 左EEF(7) + 右EEF(7) + 左臂(7) + 右臂(7) + 左手(21) + 右手(21)
  -> 删除两组 EEF pose
56D = 左臂 + 右臂 + 左手 + 右手
```

对应 [`BrainCoRevo3EefJointHandToJointHand`](../../src/openpi/policies/brainco_policy.py)。

它是明确的 70D→56D 转换，不是通用的 70D→任意 D 裁剪器。

当前 28D YAML 配置的是已经为右臂+右手整理好的 28D 数据集，并关闭该 70D→56D transform：

```yaml
revo3_eef_joint_hand_to_joint_hand: false
dataset_state_dim: 28
dataset_state_indices: null
```

## 7. 为什么预训练动作接口可能重新初始化

如果基础 checkpoint 的动作维度与目标配置不同，下面两层 shape 不匹配：

```text
action_in_proj
action_out_proj
```

当前 28D 配置使用 `PartialCheckpointWeightLoader` 并允许跳过这些不匹配层。主体预训练权重继续加载，新的 28D 动作接口保留随机初始化，再通过训练学会映射。

## 8. 当前 `lora_vlm_only` 实际训练什么

当前 28D YAML 使用：

```text
paligemma_variant   = gemma_2b_lora
action_expert       = gemma_300m
freeze_strategy     = lora_and_action_interface
```

因此允许更新的主要参数是：

- Gemma 2B VLM 中的 LoRA 参数；
- `action_in_proj` 和 `action_out_proj`；
- `time_mlp_in` 和 `time_mlp_out`。

Gemma 300M Action Expert 主体没有选择 LoRA 变体，并被该 freeze strategy 冻结。也就是说，文件名中的 `vlm_only` 主要描述“大模型主体只训练 VLM LoRA”，但新动作接口和时间条件层仍然必须训练。

LoRA/full training 与 28D/56D 是两个不同维度的配置选择：前者决定哪些参数更新，后者决定 state/action 的数据合同和动作投影层 shape。

## 9. 改维度时的检查清单

不能只改 `model.action_dim`。至少检查：

- [ ] 数据集 `observation.state` 的维度和顺序
- [ ] 数据集 `action` 的维度和顺序
- [ ] `policy_io.state_groups` 和 `action_groups`
- [ ] `delta_action_groups` 与 delta mask
- [ ] `model.action_dim`
- [ ] norm stats 是否与新布局对应
- [ ] 权重加载器如何处理不匹配的投影层
- [ ] `max_token_len` 是否容纳离散 state
- [ ] 部署端是否使用同一 action 顺序和 delta/absolute 语义

## 10. 最终结论

从 56D 改到 28D 时：

```text
直接改变参数 shape：action_in_proj、action_out_proj
改变 tensor shape：state、actions、noise、velocity、loss 输入
改变 VLM 内容：离散 state token 的数量和语义
不改变主体宽度：SigLIP、Gemma 2B、Gemma 300M
必须同步改变：数据语义、norm stats、delta 规则和部署合同
```
