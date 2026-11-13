# Diffusion Policy 在 BrainCo-IL 中的训练与部署

## 接入位置

PI0.5、ACT 和 Diffusion Policy 共用下列路径，算法分支位于模型配置与模型内部：

```text
experiment YAML → config_io → LeRobot metadata names / cameras / fps
    → LeRobot action[t:t+H] → SelectPolicyFeatures → BrainCoInputs
    → arm delta → Normalize → model transforms → Observation
    → scripts/train.py → compute_loss → AdamW / EMA → Orbax
    → export_checkpoint → params/ + assets/ + train_config.yaml + policy_contract.json

revo_deploy → framed stdio → named observation adapter
    → 同一输入 transforms → model.sample_actions
    → Unnormalize → arm absolute → grouped float32[H, D]
```

`base: diffusion` 自动创建 `DiffusionPolicyConfig`；不需要新增 trainer、部署后端协议
或改动下游 ROS 逻辑。26D、56D 和部分 group 都从 metadata 的 joint names 解析，
EEF 等字段不进入硬件动作向量。group 按 `left_arm → right_arm → left_hand → right_hand`
顺序填写。当前三个模型都要求 head、left_wrist、right_wrist 相机；额外相机不参与训练。

## 与官方实现的对应关系

参考 [real-stanford/diffusion_policy](https://github.com/real-stanford/diffusion_policy/tree/5ba07ac6661db573af695b419a7947ecb704690f)，
具体来源与许可证记录在 [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md)。

| 部分 | 本仓库实现 |
| --- | --- |
| 条件网络 | 三路独立、随机初始化的 ResNet-18/GroupNorm，空间平均池化，拼接连续关节状态 |
| 动作网络 | 官方 `ConditionalUnet1D` 的 JAX/NNX 适配；正弦时间编码、Mish、FiLM scale/bias、下采样/上采样和 skip connection |
| 训练目标 | 每个样本随机扩散时间步，`x_t = sqrt(alpha_bar_t) * action + sqrt(1-alpha_bar_t) * noise`；预测 ε 的 MSE |
| 调度器 | 默认 squaredcos_cap_v2、100 步、DDPM fixed-small；可选 linear 或确定性 DDIM（eta=0） |
| 推理步序 | Diffusers 的 leading spacing；训练时间步与推理时间步分别保存 |
| 优化器 | AdamW，betas=(0.95, 0.999)，weight_decay=1e-6，warmup + cosine LR |
| EMA | 复用 BrainCo 的固定 decay EMA，默认 0.999；导出 EMA、续训保留在线参数与 EMA |
| 数据语义 | 复用 BrainCo 的 q01/q99 归一化、图像 resize/增强和 arm delta/absolute 变换 |

这是算法适配，不是逐项复刻官方默认实验。官方示例的双帧历史、独立预测/执行长度、
随机 crop/ImageNet 图像归一化、min/max 动作归一化、动态 EMA 等没有原样搬入。
本版本 `n_obs_steps=1`，预测和返回整个 `action_horizon`，动作从当前帧开始；
数据集尾部动作沿用 LeRobot 的边界重复。`n_obs_steps>1` 会明确报错，不能把相隔一个
chunk 的两次部署请求当成相邻历史帧。任务文本会被丢弃，适合单任务训练。

模型训练与加载保持 float32。默认 U-Net 宽度 `[256,512,1024]`；horizon 必须能被
`2 ** (len(down_dims)-1)` 整除。模型没有 RTC，不接受 `strategy: rtc`。
官方 PyTorch 权重不能直接加载到此 JAX 模型。

## 训练与续训

在仓库根目录执行：

```bash
cp configs/experiments/diffusion.example.yaml my_diffusion.yaml
# 编辑 experiment.dataset、experiment.name、policy.groups 和 training 参数。
uv run python scripts/compute_norm_stats.py --config-path my_diffusion.yaml
uv run python scripts/train.py my_diffusion.yaml
```

训练入口和 PI0.5/ACT 一致。统计量必须针对本次数据、group 和 delta 编码重新计算。
默认实验名含 `56d` 仅是便于识别的文字；真实维度来自数据，26D 数据请相应修改名称。
部署延迟较敏感时，在训练前的人工 YAML 中设置：

```yaml
model:
  sampler: ddim
  num_train_timesteps: 100
  num_inference_steps: 10
```

续训时在同一人工 experiment 中设置 `training.resume: true` 并增加 `training.num_steps`，
再次运行训练命令。不要同时设置 overwrite；不要在续训时改变模型结构、数据、group 或
归一化统计。固定 EMA 可通过 `training.ema_decay` 配置，`null` 表示禁用。

默认 checkpoint 位于 `checkpoints/<experiment.name>/<experiment.name>/<step>/`。
沿用仓库既有零起始保存标签：训练 20000 次更新的最终目录是 `19999`。

## 导出

训练在 Orbax 完成写入后自动调用唯一的部署导出函数 `export_checkpoint`，生成并重新验证：

```text
<step>/
├── params/                  # EMA 权重（禁用 EMA 时为在线权重）
├── train_state/             # 续训需要
├── assets/<asset_id>/norm_stats.json
├── train_config.yaml        # 自动生成，包含完整模型/数据流水线
└── policy_contract.json     # 自动生成，standard 能力、joint names、camera、fps、H
```

重新导出已有保存步的外壳可执行：

```bash
uv run python -m openpi.deploy.export_checkpoint --checkpoint /absolute/checkpoint/19999
```

导出阶段需要源 metadata 来核对接口。数据已搬迁时，可使用兼容入口
`python -m openpi.deploy.export_contract --checkpoint ... --dataset /new/dataset`。
两个入口都经过相同 exporter。推理阶段不需要训练数据集或仓库外的统计量；完整复制
step 目录即可。生成的 YAML/JSON 不可手改，采样器和模型参数也由生成配置恢复。

## 接入 revo_deploy v1.1

先确保独立虚拟环境已安装包，且从仓库外可导入：

```bash
uv pip install --python /home/xyd/BrainCo/BrainCo-IL/.venv/bin/python --no-deps -e /home/xyd/BrainCo/BrainCo-IL
cd /tmp
/home/xyd/BrainCo/BrainCo-IL/.venv/bin/python -c 'import openpi.deploy.policy_inferencer'
```

沿用 BrainCo backend 即可选择 Diffusion checkpoint。如需在面板单独显示一个名称，在
已经 build/source 的 `revo_deploy` 环境中注册同一模块：

```bash
ros2 run vla_deploy register_policy_backend \
  --id brainco_diffusion --name 'BrainCo Diffusion Policy' \
  --python /home/xyd/BrainCo/BrainCo-IL/.venv/bin/python \
  --module openpi.deploy.policy_inferencer
```

按 SOP 重新 build/source 后验收：

```bash
ros2 run vla_deploy validate_policy_backend \
  --backend brainco_diffusion \
  --checkpoint /absolute/checkpoint/19999 \
  --robot-profile revotron_revo3 \
  --dataset /absolute/lerobot_dataset \
  --through infer \
  --output-json /tmp/brainco_diffusion_conformance.json
```

Revomate/Revo2 改用 `revomate_revo2` 和对应数据。也可按 SOP 分别执行
`--through static`、`describe`、`load`。上述步骤仅验收 inferencer，不启动 Actor。

`DESCRIBE` 不导入 JAX/PyTorch；`LOAD` 校验生成配置并加载权重/统计量；`INFER`
按名称重排关节、返回同一 request ID 和 contract hash。stdout 专供 framed protocol，
日志写 stderr。输出会检查 horizon、维度和 finite，再转为分组 float32 absolute position。

## 验证

```bash
JAX_PLATFORMS=cpu uv run pytest -q \
  src/openpi/models/diffusion_test.py \
  src/openpi/deploy/diffusion_contract_test.py \
  src/openpi/deploy/contract_test.py \
  src/openpi/training/config_test.py \
  scripts/diffusion_train_test.py
```

如 shell 已 source ROS，测试前取消其 `PYTHONPATH`，避免自动加载不兼容的 ROS pytest 插件。
端到端测试在临时目录创建带三路 H.264 视频的 LeRobot v2.1 数据，计算统计量、训练两步、
恢复训练一步、导出，再移走源数据和源统计量，从 `/tmp` 启动独立 inferencer 进行真实
`DESCRIBE/LOAD/INFER/CLOSE`。它验证的是工程链路，不代表任务成功率或实机控制效果。

2026-11-14 的本地 CPU 验收使用上述小型测试模型和生成数据，已通过本机
`revo_deploy v1.1` 原生 `backend_conformance` 的 static、describe、load、infer
四阶段，使用 `revomate_revo2` profile 的右臂/右手子集，倒序 names 输入返回
`right_arm: float32[4,7]`、`right_hand: float32[4,6]`。未运行生产规模训练或实机控制。

公共兼容修复同时覆盖了 Orbax 参数树的字符串层序号/空参数恢复，以及 CPU 续训时的
buffer donation 问题；CPU 关闭训练状态 donation，GPU 保留原有优化。端到端测试会
逐张量比较保存前后的 EMA 参数。ACT 共用 ResNet 抽取前后的参数与前向输出也已对照一致。
