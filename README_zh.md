# BrainCo-IL

本仓库保留 BrainCo 使用的 PI0/PI0.5、ACT、Diffusion Policy、LeRobot 2.1 数据流水线、训练与
独立 Policy Inferencer 推理进程。

## 训练

```bash
cp configs/experiments/pi05.example.yaml my_experiment.yaml
# base 可选 pi05、act、diffusion
# Diffusion Policy 推荐复制 configs/experiments/diffusion.example.yaml

uv run python scripts/compute_norm_stats.py --config-path my_experiment.yaml
uv run python scripts/train.py my_experiment.yaml
```

简洁 experiment YAML 只写 dataset、语义 groups、action horizon 和训练参数；
joint name 与数值切片统一从 LeRobot `meta/info.json` 自动解析。

每个 step 自动生成 `train_config.yaml` 与 `policy_contract.json`。二者都标记
`generated: true`、`do_not_edit: true` 并带 hash，禁止手改。前者只供
BrainCo-IL 重建训练时的数据/模型流水线，后者供跨仓握手。

## 部署

生产入口只有独立 Python 推理进程：

```bash
uv run python -m openpi.deploy.policy_inferencer --checkpoint /path/to/checkpoint
```

Policy Inferencer 实现 DESCRIBE、LOAD/READY、INFER/RESULT、RESET/CLOSE。它接收带
joint name 的 observation，在仓库内部完成选维、图像预处理、归一化、模型
推理和反归一化，跨仓只返回按 group 打包的 absolute action chunk。

PI0.5、ACT 与 Diffusion Policy 共用完全相同的部署协议，`revo_deploy` 不解析模型类型，也不
包含任何 JAX/PyTorch/LeRobot runtime。


## Diffusion Policy

训练、导出、部署注册与验收命令见 [Diffusion Policy 接入说明](docs/diffusion_policy.md)。
新增模型是官方 CNN + 条件 U-Net 的 JAX/NNX 适配：ε 预测 MSE、DDPM/DDIM 采样、
EMA 推理权重。首版使用**三相机的当前单帧观测**，复用 BrainCo 的分位数归一化与
臂部 delta 编码，对外恢复为 absolute joint position。暂不支持观测历史、语言条件、
RTC 或官方 PyTorch checkpoint 的直接加载。

训练保存后统一调用 `openpi.deploy.export_checkpoint`，重新读取并校验生成文件、
checkpoint ID、hash 和归一化资产。可对已有保存步重新生成部署外壳：

```bash
uv run python -m openpi.deploy.export_checkpoint --checkpoint /absolute/checkpoint/step
```

算法参考来源和 MIT 声明见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
