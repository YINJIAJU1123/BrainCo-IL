# 02. 配置系统：从 YAML 到 DataConfig

配置系统负责回答三个问题：

1. 训练什么模型？
2. 使用什么数据和 transform？
3. 用什么训练超参数、权重和保存规则？

## 1. YAML 如何进入程序

执行：

```bash
python3 scripts/train.py path/to/config.yaml
```

入口会调用 [`config.cli()`](../../src/openpi/training/config.py)：

```python
if len(sys.argv) == 2 and pathlib.Path(sys.argv[1]).suffix in (".yaml", ".yml"):
    return config_io.load_train_config(sys.argv[1])
```

因此 YAML 不会留在普通字典阶段，而是被反序列化成有类型的对象树：

```text
YAML
  -> TrainConfig
       ├─ model: Pi0Config
       ├─ data: LeRobotBrainCoDataConfig
       ├─ optimizer
       ├─ lr_schedule
       ├─ weight_loader
       └─ batch、step、checkpoint 等设置
```

## 2. `TrainConfig` 是总配置

[`TrainConfig`](../../src/openpi/training/config.py) 包含：

- `model`：`action_dim`、`action_horizon`、PI0.5 变体等。
- `data`：数据集位置及 BrainCo 数据规则。
- `weight_loader`：从哪里加载预训练参数。
- `optimizer`、`lr_schedule`：如何更新参数。
- `freeze_strategy`：哪些参数允许训练。
- `batch_size`、`num_train_steps`：训练规模。
- `checkpoint_dir`：保存位置。

可以把它看成一次实验的“总施工图”。

## 3. `config.data` 还不是最终 DataConfig

YAML 加载完成后：

```python
config.data
```

通常是一个 `LeRobotBrainCoDataConfig`，它是 `DataConfigFactory`，也就是“数据配置工厂”。

真正展开发生在：

```python
data_config = config.data.create(config.assets_dirs, config.model)
```

输入：

- `config.assets_dirs`：归一化统计量等 assets 的默认目录。
- `config.model`：模型类型、`action_dim`、`max_token_len` 等。

输出：

- 一个完整的 `DataConfig`。

注意：这里返回的不是 DataLoader，也没有立即读取数据。它只是把数据路径、归一化统计量和 transform 顺序准备好。

## 4. `LeRobotBrainCoDataConfig.create()` 做什么

[`LeRobotBrainCoDataConfig`](../../src/openpi/training/config.py) 依次完成：

```text
校验 policy_io
  -> 创建原始字段重排规则
  -> 创建 BrainCo 专属输入/输出 transforms
  -> 可选地执行 70D -> 56D 转换
  -> 可选地执行 absolute -> delta 动作转换
  -> 根据 PI0.5/ACT 创建模型 transforms
  -> 加载 norm stats
  -> 返回 DataConfig
```

最终 `DataConfig` 中最重要的是三组 transform：

| transform 组 | 作用 |
| --- | --- |
| `repack_transforms` | 把数据集原始 key 改成统一 key |
| `data_transforms` | BrainCo 关节、相机和 delta/absolute 语义转换 |
| `model_transforms` | resize、tokenize、padding 等模型输入处理 |

## 5. `brainco_policy` 的作用

[`brainco_policy.py`](../../src/openpi/policies/brainco_policy.py) 集中保存 BrainCo 机器人的数据语义：

- `BrainCoPolicyIOConfig`：描述 state/action 由哪些关节组组成。
- `SelectPolicyFeatures`：使用 LeRobot metadata 自动解析出的 indices 选维。
- `BrainCoInputs`：把相机、state、prompt 转成模型统一字段。
- `BrainCoOutputs`：推理时把模型 action 转回 BrainCo 输出格式。

`policy_io` 首先是语义合同和校验器。例如 28D：

```text
state_groups  = right_arm(7) + right_hand(21)
action_groups = right_arm(7) + right_hand(21)
```

新 experiment 中的精确 joint names、维度和 indices 都从 LeRobot
`meta/info.json` 自动解析，不需要用户维护数值切片。

## 6. 单侧 28D 配置的展开结果

简洁 experiment 只需选择：

```yaml
base: pi05
policy:
  groups: [right_arm, right_hand]
  action_horizon: 16
```

加载后会根据 dataset metadata 自动展开为：

```text
model.action_dim       = 28
model.action_horizon   = 16
model.max_token_len    = 256
data.policy_io         = right_arm + right_hand
delta_action_groups    = right_arm
```

训练内部默认对手臂使用 delta transform、对手保持 absolute；推理 output
transform 会恢复成 absolute action，跨仓协议永远不暴露 delta。

下一章继续看最终的 `DataConfig` 如何真正产生 batch。
