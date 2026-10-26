# BrainCo-IL：本次混训相关提交核对

核对日期：2026-10-26（北京时间）。分支：`ego-train`。本次只补充提交边界及源码审计，不 push。

## 本仓库负责什么

从已验收的机器人动作标签接入 LeRobot/56D 训练接口，处理 task、相机 mask、归一化、混训抽样、checkpoint 和 SwanLab。机器人指尖几何、手部优化目标、双臂 IK 和标签修复实现在 BrainCo-Dataset。

## 本轮三笔提交

| Commit | 原始 message | 具体内容 |
|---|---|---|
| `fbb9ee9` | `feat(ego): support deterministic Ego reuse in coverage sampling` | 显式 `coverage_ego_repeat` 开关；配额超出小 Ego 池时逐遍按固定种子洗牌，每遍内无放回，保留共同前缀及按训练步恢复；含测试 |
| `35b058f` | `feat(ego): add verified MINT-to-LeRobot post-training workflow` | 固定388段准入集、按元数据名称排列56D关节字段、LeRobot导出、三组配置生成、loader/采样/模型保存恢复验收及隔离启动脚本 |
| `55525e1` | `docs(ego): record completed MINT mixture runs and 5090 delivery` | train/val/test 345/16/27 段、统一task、实际三组训练配置/结果、SwanLab和5090交付清单，更新README/TODO |

`deterministic Ego reuse` 的含义是**按固定种子重复抽取 Ego 训练数据**；不是改变机器人动作求解。`35b058f` 中的命名字段排列只解决训练数据列顺序，也不改变指尖坐标、关节轴或 hand IK。

## 复用的已有提交

| Commit | 已有内容 | 本轮关系 |
|---|---|---|
| `ea835f9` | `Optimize post-train video input with verified per-run profiles` | 已有视频读取/解码优化，本轮验证后应用到三组；没有重新实现解码器 |
| `06d79eb` | `Record completed pi05 mixture experiments in repository` | 旧100h Ego三比例实验记录；为本轮recipe、norm和初始化来源提供历史上下文 |
| `7a39fa4` | `docs(ego): record self-captured MINT processing and runtime estimates` | 数据处理状态及耗时记录；不含Dataset指尖算法实现 |

## Dataset 侧的重要提交

以下 commit 属于 **BrainCo-Dataset**，没有迁入 IL：

- `7dc998b`：统一官方 RevoTron 双臂后端和真机 EEF 高度参照。
- `9c5b95c`：有来源和边界的分段批处理。
- `4c1680e`：网格指尖参考统一、跨关节、拇指相对位置、全程手型约束及后处理。
- `83e78bf` / `c6f24b5`：移除未采用的RGB修指路线并清理CLI/历史导入。
- `b3f3c65`：连续性/碰撞/限位修复、活动手与QC/批处理稳定性。
- `71e4cc9`：本次补交Dataset后续记录、两仓库边界说明和逐文件源码审计；不改变算法。

完整文件及验证范围见 [Dataset 提交清单](../../../BrainCo-Dataset/docs/ego/COMMIT_AUDIT_20261026.md)。

## 已核对的实际运行与工作区

[逐文件审计](mint_v3_posttrain_20261023/commit_audit_20261026.json)：四份训练代码及八份实验脚本，共12份，均与Git已提交字节、A100隔离源码/脚本一致。原三组的训练源码SHA仍见[运行快照](mint_v3_posttrain_20261023/training_source_sha256.json)。本次没有修改训练代码或重跑训练。

三组都从 **EgoDex 100h mid-train `19999/params`** 初始化，重新建立 optimizer。训练采用388段版本、完整源视频分组345/16/27；只有train的62,469帧参与重复采样，17段隔离。后续404段候选未替换这些已完成实验的数据。完成及交付结果见[REPORT](mint_v3_posttrain_20261023/REPORT.md)。

本次新增提交 message：`docs(ego): map training commits to Dataset provenance`。包含本清单、12文件审计证据与README索引；重要算法/训练代码均已在上表提交中，未发现漏交代码。
