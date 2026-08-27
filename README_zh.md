# BrainCo-IL

本仓库保留 BrainCo 使用的 PI0/PI0.5、ACT、LeRobot 2.1 数据流水线、训练与
独立 Policy Inferencer 推理进程。

## 训练

```bash
cp configs/experiments/pi05.example.yaml my_experiment.yaml
# 将 base: pi05 改成 base: act 即切换 ACT

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

PI0.5 与 ACT 共用完全相同的部署协议，`revo_deploy` 不解析模型类型，也不
包含任何 JAX/PyTorch/LeRobot runtime。
