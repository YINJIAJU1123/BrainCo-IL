# BrainCo-IL 与 revo_deploy 握手

## 配置与制品

人工维护：

- BrainCo-IL `experiment.yaml`：训练意图，`base: pi05/act`。
- revo_deploy `robot_profile.yaml`：canonical joint names、ROS wiring、
  controller 与 safety。
- revo_deploy `inference_backends.yaml`：本机 inferencer command。

checkpoint 自动生成且禁止手改：

- `train_config.yaml`：BrainCo-IL 内部复建流水线。
- `policy_contract.json`：跨仓具名 IO 契约。

## 启动时序

```text
revo_deploy -> spawn openpi.deploy.policy_inferencer --checkpoint ...
revo_deploy -> DESCRIBE
inferencer  -> CONTRACT
revo_deploy -> 校验 protocol/hash/group joint names/cameras/controllers
revo_deploy -> LOAD
inferencer  -> 加载 params + norm stats + transforms + model
inferencer  -> READY(checkpoint_id, contract_hash)
```

## 在线数据

gateway 将 ROS JointState 按 `msg.name` 映射成 canonical named groups，与原始
RGB 图像和 task 一起发送 INFER。inferencer 使用训练 dataset 保存的 names 排序
和选维，完成：

```text
feature selection -> preprocess -> normalize -> PI0.5/ACT -> unnormalize
```

RESULT 固定是 named/grouped absolute hardware action。actor 校验 hash、
request ID、names、shape、NaN/Inf，随后负责 chunk 预取/播放、安全处理和
controller 路由。

因此新增推理仓不需要复制 BrainCo 配置，只要实现相同 Policy Inferencer protocol 和
PolicyContract 即可。
