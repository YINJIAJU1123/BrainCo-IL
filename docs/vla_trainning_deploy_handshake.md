# BrainCo-IL 训练与部署握手机制

这个分支把 BrainCo-IL 从“部署端通过 Python config 名字去训练仓库里查配置”的模式，改成了“checkpoint 自描述”的模式。

核心原则是：**checkpoint 目录里的 `train_config.yaml` 是模型训练与部署语义的唯一真相源**。

## 改动目标

- BrainCo 训练入口保持稳定，不再为每次实验新增一个 Python `TrainConfig`。
- 每次训练开始时，把最终展开后的完整训练配置保存到 checkpoint。
- `revo_deploy` 不再理解 BrainCo-IL 的训练细节，也不再维护 fallback config。
- PI0.5 和 ACT 通过同一个 BrainCo-IL deploy plugin 对接部署。
- 复制单个 step checkpoint 到部署机器时，也能直接恢复训练时的模型结构、数据 transform 和 runtime 语义。

## 当前训练配置

`src/openpi/training/config.py` 现在只注册 BrainCo recipe 和 debug config：

```text
pi05_brainco_56d
act_brainco_56d
debug
debug_restore
debug_pi05
debug_act
```

其中：

- `pi05_brainco_56d` 是 BrainCo 56D PI0.5 的默认训练 recipe。
- `act_brainco_56d` 是 BrainCo 56D ACT 的默认训练 recipe。

它们不是 Python 意义上的抽象基类，而是**可覆写的基准配置**：本身就是具体的 `TrainConfig` 实例，提供一套默认模型结构、数据 transform、action 维度、chunk 大小、batch、训练步数和学习率。

以后做新实验时，不应该再往 `_CONFIGS` 里新增 `pi05_xxx_yyy` 这种 Python config。应该基于这两个 recipe，通过 CLI 或 YAML 覆写实验参数。

官方 OpenPI 示例 config，比如 Aloha、Droid、Libero、RoboArena、Polaris，已经从 `_CONFIGS` 中移除，因此不会再作为训练选项出现。相关 DataConfig/transform 类暂时还保留在文件里，因为删除它们会牵涉更多 import 和示例代码清理，可以后续单独做。

## 训练方式

训练时先选择一个 recipe：

- PI0.5：`pi05_brainco_56d`
- ACT：`act_brainco_56d`

然后覆写本次实验的参数，例如：

- 数据集路径
- 实验名
- checkpoint 根目录
- action horizon / chunk size
- batch size
- 训练步数
- save interval
- learning rate
- 是否 LoRA
- 模型变体

### CLI 覆写

tyro 支持直接通过命令行覆写 dataclass 字段，例如：

```bash
python scripts/train.py pi05_brainco_56d \
  --exp-name revotron_0712_pi05_chunk16 \
  --checkpoint-base-dir /mnt/data_nas/xiyue/BrainCo-IL/checkpoints \
  --model.action-horizon 16 \
  --batch-size 16 \
  --num-train-steps 40000 \
  --save-interval 4000 \
  --lr-schedule.peak-lr 5e-5
```

CLI 适合覆写简单标量参数，例如 batch、step、chunk、学习率等。

如果参数比较复杂，比如多数据集路径、dataset 权重、transform 组合，建议使用 YAML。

### YAML 训练

训练入口也支持直接读取完整的 `train_config.yaml`：

```bash
python scripts/train.py /path/to/train_config.yaml
```

这适合云端训练和实验复现，因为 YAML 里可以完整记录最终 `TrainConfig`，不会依赖某个 Python config 名字。

### 仓库内训练配置模板

可直接使用和修改的训练 YAML 统一放在：

```text
src/openpi/training/training_config_template/
```

当前模板包括：

| 文件 | 用途 |
| --- | --- |
| `act_brainco_revo3_0708_chunk16.yaml` | Revo3 0708 数据，ACT，56D，action horizon 16 |
| `act_brainco_revo3_0712_ght_56d.yaml` | Revo3 0712 GHT 数据，ACT，56D，action horizon 16 |
| `pi05_brainco_revo3_0712_ght_56d.yaml` | Revo3 0712 GHT 数据，PI0.5 全参数训练，56D，action horizon 16 |
| `pi05_0713_merged_action_interface_only.yaml` | Revo3 0713 merged 数据，只训练 action interface 和 time MLP |
| `pi05_0713_merged_lora_action56.yaml` | Revo3 0713 merged 数据，训练 LoRA、action interface 和 time MLP |

例如：

```bash
uv run python scripts/train.py \
  src/openpi/training/training_config_template/pi05_brainco_revo3_0712_ght_56d.yaml
```

前三个模板由 `main` 中原有的 Revo3 Python config 迁移而来；后两个模板来自阿里云
`/mnt/data_nas/xiyue/BrainCo-IL/configs/experiments/` 中实际使用的训练配置。

模板是训练输入，应保持易于修改。训练启动后，run 根目录和 step checkpoint 中自动生成的
`train_config.yaml` 是完整展开的配置快照，用于复现和部署，不应反向堆回模板目录。

### `train_config.yaml` 训练模板

下面是 PI0.5 BrainCo 56D 的训练模板。使用时通常只需要改：

- `exp_name`
- `checkpoint_base_dir`
- `data.fields.base_config.fields.lerobot_datasets`
- `model.fields.action_horizon`
- `batch_size`
- `num_train_steps`
- `save_interval`
- `lr_schedule`

```yaml
schema_version: 1
train_config:
  __class__: openpi.training.config.TrainConfig
  fields:
    name: pi05_brainco_56d
    project_name: imitation
    exp_name: revotron_0712_pi05_chunk16

    model:
      __class__: openpi.models.pi0_config.Pi0Config
      fields:
        action_dim: 56
        action_horizon: 16
        max_token_len: 256
        dtype: bfloat16
        paligemma_variant: gemma_2b
        action_expert_variant: gemma_300m
        pi05: true
        discrete_state_input: true

    weight_loader:
      __class__: openpi.training.weight_loaders.PartialCheckpointWeightLoader
      fields:
        params_path: gs://openpi-assets/checkpoints/pi05_base/params
        skip_on_mismatch_regex: .*(action_in_proj|action_out_proj|state_proj).*

    pytorch_weight_path: null
    pytorch_training_precision: bfloat16

    lr_schedule:
      __class__: openpi.training.optimizer.CosineDecaySchedule
      fields:
        warmup_steps: 1000
        peak_lr: 5.0e-05
        decay_steps: 40000
        decay_lr: 5.0e-06

    optimizer:
      __class__: openpi.training.optimizer.AdamW
      fields:
        b1: 0.9
        b2: 0.95
        eps: 1.0e-08
        weight_decay: 1.0e-10
        clip_gradient_norm: 1.0

    ema_decay: null
    freeze_strategy: none

    data:
      __class__: openpi.training.config.LeRobotBrainCoDataConfig
      fields:
        repo_id: brainco_56d
        base_config:
          __class__: openpi.training.config.DataConfig
          fields:
            prompt_from_task: true
            lerobot_datasets:
            - __class__: openpi.training.config.LeRobotDataset
              fields:
                repo_id: /path/to/lerobot_dataset_a
                weight: 1.0
            # 多数据集训练时继续追加 LeRobotDataset，并按需调整 weight。
            # - __class__: openpi.training.config.LeRobotDataset
            #   fields:
            #     repo_id: /path/to/lerobot_dataset_b
            #     weight: 0.5
        extra_delta_transform: true
        head_camera_key: observation.images.cam_head
        revo3_eef_joint_hand_to_joint_hand: true

    assets_base_dir: ./assets
    checkpoint_base_dir: /mnt/data_nas/xiyue/BrainCo-IL/checkpoints
    seed: 42
    batch_size: 16
    num_workers: 8
    num_train_steps: 40000
    log_interval: 10
    save_interval: 4000
    keep_period: 4000
    save_final_checkpoint: true
    overwrite: false
    resume: false
    wandb_enabled: true
    policy_metadata: null
    fsdp_devices: 1
```

ACT 训练也可以用同一个结构，只需要把关键字段改成 ACT recipe：

```yaml
name: act_brainco_56d
model:
  __class__: openpi.models.act_config.ACTConfig
  fields:
    action_dim: 56
    action_horizon: 100
weight_loader:
  __class__: openpi.training.weight_loaders.NoOpWeightLoader
  fields: {}
```

ACT 从头训练，不加载 `pi05_base` 权重；其余 BrainCo 数据配置、checkpoint 目录、batch、step、学习率等字段按实验需要调整。

### 可序列化参数冻结策略

BrainCo PI0.5 训练支持在 YAML 中通过 `freeze_strategy` 选择参数冻结方式：

```yaml
# 全参数训练，不额外冻结参数。
freeze_strategy: none

# LoRA 实验：只训练 LoRA、56D action interface 和 Pi0.5 time MLP。
freeze_strategy: lora_and_action_interface

# Action interface 实验：冻结 VLM，只训练 56D action interface 和 Pi0.5 time MLP。
freeze_strategy: action_interface_only
```

使用 `lora_and_action_interface` 时，model 中至少一个 Gemma variant 必须是 LoRA variant；BrainCo 双 LoRA 实验使用：

```yaml
model:
  __class__: openpi.models.pi0_config.Pi0Config
  fields:
    pi05: true
    action_dim: 56
    action_horizon: 16
    max_token_len: 256
    paligemma_variant: gemma_2b_lora
    action_expert_variant: gemma_300m_lora
freeze_strategy: lora_and_action_interface
```

`action_interface_only` 应配合非 LoRA model variant 使用。`freeze_strategy` 是普通字符串字段，会随完整
`TrainConfig` 保存到 run 根目录和每个 step checkpoint 的 `train_config.yaml`；训练恢复时再根据它构造
NNX `freeze_filter`，因此不会丢失参数冻结语义。

## checkpoint 保存规则

训练脚本会自动保存最终展开后的完整配置：

```text
<checkpoint_base_dir>/<config_name>/<exp_name>/train_config.yaml
<checkpoint_base_dir>/<config_name>/<exp_name>/<step>/train_config.yaml
```

也就是说：

- run 根目录会有一份 `train_config.yaml`
- 每个 step checkpoint 目录也会有一份 `train_config.yaml`

step 目录里也保存一份非常重要，因为部署时经常只拷贝一个 step 目录，例如：

```text
.../12000
```

如果这个目录本身带着 `train_config.yaml`，部署端就不需要知道它来自哪个实验、哪个 Python config 名字、训练仓库当时有什么 `_CONFIGS`。

## 部署 checkpoint 目录约定

一个可部署的 BrainCo-IL checkpoint 目录应该长这样：

```text
checkpoint_dir/
  train_config.yaml
  params/                 # JAX checkpoint 参数，或 PyTorch 的 model.safetensors
  assets/                 # norm stats 等资产
  config.yaml             # revo_deploy 使用的部署侧配置
```

其中：

- `train_config.yaml` 由 BrainCo-IL 训练脚本自动生成。
- `config.yaml` 是部署侧写的，作用只是告诉 `revo_deploy` 去哪里加载 plugin、用哪个 Python 解释器。

部署侧 `config.yaml` 应该保持很小：

```yaml
schema_version: 1
runtime: external_policy

plugin:
  repo: /home/phoebe/Brainco/BrainCo-IL
  module: openpi.deploy.plugin
```




## BrainCo-IL plugin 的作用

BrainCo-IL 对部署暴露的 plugin 在：

```text
src/openpi/deploy/plugin.py
```

它提供两个函数：

```python
describe_policy(checkpoint_dir, *, runtime_options=None, overrides=None)
create_policy(checkpoint_dir, *, runtime_options=None)
```

### describe_policy

`describe_policy()` 会读取：

```text
checkpoint_dir/train_config.yaml
```

然后还原出训练时的完整 `TrainConfig`，再返回部署需要的 runtime contract，包括：

- `policy_type`：例如 `pi05`、`pi0`、`act`
- `action_dim`
- `action_horizon`
- observation 输入 key
- 图像尺寸和相机绑定
- state composition
- action 输出 key
- joint group 顺序
- 每个 group 的 action mode
- dataset inference 需要的 metadata
- runtime 默认参数，例如 `policy_rate`

### create_policy

`create_policy()` 也读取同一份 `train_config.yaml`，然后调用：

```python
openpi.policies.policy_config.create_trained_policy(...)
```

来真正创建 policy。

PI0/PI0.5 在推理时需要 denoising steps，所以 plugin 只会对 `pi0` / `pi05` 传：

```python
sample_kwargs={"num_steps": ...}
```

ACT 没有扩散去噪过程，所以 ACT 不会收到 `num_steps`。

## revo_deploy 如何握手

新的 `revo_deploy` 不再知道 BrainCo-IL 的模型细节。它只做通用外部 policy 加载：

1. 读取 checkpoint 目录下的 `config.yaml`
2. 把 `plugin.repo` 加进 `PYTHONPATH`
3. import `plugin.module`
4. 调用 `describe_policy(checkpoint_dir, runtime_options=...)`
5. 根据 plugin 返回的 spec 生成 deploy runtime bundle
6. 调用 `create_policy(checkpoint_dir, runtime_options=...)`
7. 把 observation 喂给 policy
8. 播放 action chunk

也就是说：

- 模型结构由 BrainCo-IL 负责
- 数据 transform 由 BrainCo-IL 负责
- checkpoint norm stats 由 BrainCo-IL policy loader 负责
- robot 控制、action 播放、monitor、ros topic 由 `revo_deploy` 负责

两边通过 `train_config.yaml + deploy plugin spec` 握手。

## 为什么要这么改

旧架构的问题是：

1. 每次云端实验都要新建一个 Python `TrainConfig`
2. checkpoint 本身无法说明自己是怎么训练出来的
3. deploy 需要写 `policy_id`，再回训练仓库 `_CONFIGS` 里查
4. 如果部署代码、训练仓库、checkpoint 三者版本不一致，很容易静默加载错配置
5. deploy 里会残留 fallback config，导致训练细节散落在多个地方
6. ACT 和 PI0.5 的 inference 参数不同，但旧 plugin 容易把所有模型都当 PI0.5 处理

新架构的好处是：

- checkpoint 自己携带完整训练语义
- deploy 不再依赖训练 config 名字
- 单个 step checkpoint 拷到哪里都能被识别
- PI0.5 和 ACT 可以走同一个 external policy runtime
- 模型相关逻辑集中在 BrainCo-IL
- robot 运行逻辑集中在 `revo_deploy`

最终责任边界变成：

```text
BrainCo-IL:
  负责训练、模型、数据 transform、policy 创建、checkpoint 自描述

revo_deploy:
  负责加载 plugin、构造 runtime、获取 observation、播放 action、监控和 ros 通信
```

这样后续如果加入新的模型，比如 GROOT，只要训练仓库提供同样的 deploy plugin 接口和 checkpoint 自描述文件，`revo_deploy` 就不需要再理解新模型内部细节。
