# EgoDex → Revotron → LeRobot 训练管线

项目记录：A100 是工作环境，Mac 用于调试与 review。Mac 两仓库的历史基线及 Post-train 准备代码已于 2026-09-29 提交并推送到 `ego-train`；2026-10-15 已保护原改动并同步 A100：IL 准备提交 `25c5e5e`、Dataset `c5bb19b`，post-train 脚本与三份配置已到位。

## 文档导航与更新依据

本文件是 Ego 项目的总入口，后文保存 100h 管线与历史运行记录。最新进展见 2026-10-16 的 TODO 与准备 REPORT：1:1 已完成，3:1 运行中，5:2 等待；CPU 输入优化已完成双轮长测及step37恢复，读取吞吐提升4.84–5.85倍；第三组按专用profile在组间启用，实际训练速度待核验。历史环境恢复为 10 月 15 日，准备检查为 9 月 29 日，retarget 审计为 10 月 5 日。

| 文件 | 用途与更新方式 |
|---|---|
| [POSTTRAIN_TODO.md](POSTTRAIN_TODO.md) | 持续维护的完成状态、剩余工作和验收条件；查“还有什么没做”从这里开始 |
| [POSTTRAIN_ARCHITECTURE.md](POSTTRAIN_ARCHITECTURE.md) | 数据、代码、模型与存储之间的关系；设计或审计结论变化时更新 |
| [POSTTRAIN_RUNBOOK.md](POSTTRAIN_RUNBOOK.md) | 审核、同步、norm、验收、启动及恢复的操作顺序 |
| [准备 REPORT（含 10 月 16 日输入优化）](</Users/larrybb/Documents/ChatGPT/Ego data/reports/posttrain_prep_20260929/REPORT.md>) | 日期分节的检查证据：版本、数据、checkpoint、ROS 和兼容性；位于仓库外 |
| [10 月 5 日 retarget 审计 REPORT](</Users/larrybb/Documents/ChatGPT/Ego data/reports/retarget_audit_20261005/REPORT.md>) | 服务器代码及真实样本审计，包括新直出入口的手指字段顺序问题；位于仓库外 |
| `WORKTREE_REVIEW_20260923.md` | 9 月 23 日工作区审阅快照；在 Mac 被本地 exclude 忽略，未纳入 Git |

后续性能工作已收录在 [视频读取与解码优化清单](POSTTRAIN_TODO.md#video-loading-optimization)，官方方案调研及实测依据保存在准备 REPORT。

REPORT 与仓库文档各自独立保存，没有自动同步程序。每次取得新结果后，在对应 REPORT 保存证据，同时更新 TODO 的状态及涉及的设计、操作说明。报告中的远端状态按其核查日期理解；后续远端操作前重新核对。

Post-train 当前记录：三份 YAML、来源级 mask/prompt、覆盖采样和共享 norm 入口已在 Mac 准备，9 月 29 日 A100 隔离副本检查通过；正式 A100 仓库已同步，4 项检查通过。10 月 15 日确认原数据和两端 100h checkpoint 副本保留，5090 三相机/四组关节反馈可订阅；5090 独立副本 `/home/phoebe/Brainco/ego-posttrain/BrainCo-IL` 已修复旧 Ego 腕 mask（`cb25680`，18 项检查通过），共享部署入口保持原状态。用户已确认 task 并选择完整保留 55,243 帧；派生数据和共享 norm 已生成，三组真实混合 loader 的 CPU 验收通过。5090 原 100h 完整权重及 mask 不变性通过；A100 5 卡的三组前向、独立 3 步短测、完整保存恢复、部署加载和 SwanLab 上传补传均通过。用户已授权串行训练；截至 10 月 16 日，第一组 1:1 已完成 6,908 步并生成最终 `6907`，云端 run `6hayni72`；第二组 3:1 运行中，云端 run `uy1203p2`；第三组 5:2 等待。第三组输入优化的 CPU 对照与启用状态见 TODO 和准备 REPORT。10 月 5 日发现的新直出 42D 字段顺序错配继续列入 TODO；既有 100h 使用旧 exporter。

当前状态（2026-09-28）：100 小时批次的 20,000 步训练已完成，最终 checkpoint 为 `checkpoints/egodex_pi05_100h_offset0_20260923/offset0_bs16_steps20000_v1/19999/`；其中已生成 `policy_contract.json` 并封装部署元数据，训练配置内层内容保持不变。固定验证集评估和 5090 部署验收仍待完成。第 8 节训练命令保留为历史复现记录，不应在同名实验目录重跑。

## 1. 目标与流程

`EgoDex 视频/标注 → 手指 retargeting + 双臂 IK → 有效区间筛选 → LeRobot 2.1 → PI0.5 训练 → Revotron 验证`

- 以后从原始 EgoDex 处理动作时，目标是直接写 LeRobot Parquet + 三路 MP4；原始标注仍是 HDF5，但不再生成动作/图像中间 HDF5。手部 retargeting、双臂 IK 和质量检查仍须执行。
- 单片段入口 `BrainCo-Dataset/scripts/ego/direct_to_lerobot.py` 已实现：94 帧样本与旧的 reviewed 双阶段求解逐值完全一致；三路视频由 LeRobot PyAV loader 验证。2026-09-29 又完成两段单核计时转换，同时发现两段在手部关节 NaN 校验处失败。批量调度、断点续跑与代表性吞吐测试待补齐；实测见第 5 节。
- 本次 100 小时批次沿用先 HDF5 后 LeRobot 的旧链路；不可作为未来直出速度的实测值。
- 直出试跑从 `BrainCo-Dataset` 根目录执行 `PYTHONPATH=. .venv-mint/bin/python -m scripts.ego.direct_to_lerobot --annotation <原始.hdf5> --video <原始.mp4> --model /mnt/data_oss/egodex_train_20260917/model/official_tianji.xml --task <任务名> --output <全新LeRobot目录>`；最终 IK 模型必须显式指定，避免与手部初解模型混淆。

## 2. 已完成批次

- 64 workers；2026-09-18 11:12 至 09-21 14:29，约 75.3 小时。
- 成功 97,797 段，失败 12,281 段；按连续 ≥16 有效帧统计 147.96 小时。
- OSS 挂载输出：`/mnt/data_oss/egodex_train_20260917/`，包含 `output/`、`source/`、`receipts/`。
- 当前额外外移量 0；旧约 1 小时批次为每侧 0.025 m。两者分别记录，不能混用元数据。
- 原始 ZIP 是截断前缀；仅处理索引中完整且 CRC 正确的成员。
- 100 小时 LeRobot 导出于 2026-09-23 04:52 完成：训练 **64,502 段 / 10,809,850 帧 / 100.0912 小时**；验证 **3,278 段 / 545,248 帧 / 5.0486 小时**；8 workers 用时 50,571.87 秒。输出：`/mnt/data_nas/ruibin/dataset/egodex_100h_offset0_lerobot21/{train,val}`，报告见同级 `conversion_report.json`。
- 这份 100 小时导出最初只有头部相机；现已在 LeRobot 文件本身补齐两路黑手腕。train/val 分别新增 129,004/6,556 条视频；两份元数据均声明三路相机、总视频数分别为 193,506/9,834。2026-09-23 的真实 loader 预检已通过。

## 3. 数据标准与字段

| 项目 | 约定 |
|---|---|
| 格式 | LeRobot v2.1：Parquet、MP4、meta JSON/JSONL |
| 频率 | 30 FPS；帧号、视频与动作一一对应 |
| state/action | float32，56 维：左臂7、右臂7、左手21、右手21 |
| 存盘动作 | absolute 关节角，rad；名称/顺序与自采70维的最后56维一致 |
| 训练动作 | 双臂减当前 state 变成 delta；手指保持 absolute |
| 相机 | 标准数据内含头部真实视频与左右手腕黑色视频；训练时左右腕 `image_mask=false`。100 小时批次已补齐两路手腕 |
| 合成 state | 同帧 IK 结果，等于 action；不是实机反馈，需单独评估域差异 |
| 有效区间 | 连续有效帧至少达到 action_horizon，当前16；边界 padding 策略待核对 |

## 4. 本轮 LeRobot 导出

- 默认选择约100小时训练集、额外约5小时验证集；按原始source隔离，固定seed=42。
- 按part/任务分配时长比例，保留完整有效区间；允许最后一个区间导致小幅超额。保存固定plan以便续跑。
- 从本地成功回执建立清单；损坏回执回读OSS；导出时再次检查有效帧、关节字段、外移量和时间戳。
- 已完成批次入口：`/mnt/data_nas/ruibin/egodex_lerobot_export_20260922/export_egodex_receipts.py`；8 个进程，每个 FFmpeg 2 线程。新批次使用 `BrainCo-Dataset/scripts/ego/direct_to_lerobot.py`。
- 本地临时空间只放正在处理的原始/切片视频；最终压缩数据持久化到NAS：`/mnt/data_nas/ruibin/dataset/egodex_100h_offset0_lerobot21`。
- 生成 `plan.json`、`progress.json`、`conversion_report.json`；每个source有完成回执，重启复用已完成source。
- 当前出口保留合成state语义，不改变外移量。尚未证明实机闭环效果与自采一致。

## 5. 耗时基准

### 数据处理：EgoDex → 动作 → LeRobot

| 数据批次 | 处理阶段 | 耗时 | 结果与状态 |
|---|---|---:|---|
| 约 1h | 动作 HDF5 → LeRobot，2 workers | 约 75.6 分钟 | **已确认**；导出训练集 67.58 分钟、验证集 8.42 分钟 |
| 100h | EgoDex → 动作 HDF5，64 workers | 75.3 小时 | **已确认**；整个源批得到 147.96 小时可训练动作，供后续选样 |
| 100h | 动作 HDF5 + 原始 MP4 → LeRobot，8 workers | 14 小时 03 分 | **已确认**；导出训练集 100.09 小时、验证集 5.05 小时 |

1h 的首行是历史旧批次，不能与后一行相加当作同一批数据的精确总耗时。未来原始 EgoDex → LeRobot 直出尚无全量测速。

### 原始 EgoDex → LeRobot 单核直出小样本（2026-09-29）

**成功片段的实测外推：约 8.65 单核小时／1 小时原视频（约 8 小时 39 分）。**
在 A100 上将现有 `BrainCo-Dataset/scripts/ego/direct_to_lerobot.py` 固定到物理 CPU 核 78
（该机器每核 1 线程）；NumPy/BLAS 线程数设为 1，FFmpeg 子进程继承同一 CPU affinity，
不使用 GPU。输入为 NAS 上的原始 EgoDex HDF5 标注与 MP4，输出集中于本机 scratch
`/mnt/egodex_work/direct_benchmark_DZXaLW3O/`。求解、三路视频写入和 LeRobot 元数据生成均计入墙钟时间。

| 原始片段 | 原视频帧数 / 时长 | 墙钟耗时 | 结果 |
|---|---:|---:|---|
| `add_remove_lid/1` | 266 帧 / 8.87 秒 | 77.142 秒 | 完整输出 266 帧、三路视频；训练侧 LeRobot loader 可读 |
| `add_remove_lid/4` | 453 帧 / 15.10 秒 | 90.414 秒 | 失败：手部关节 NaN 位置不一致，未输出 LeRobot |
| `arrange_topple_dominoes/1` | 580 帧 / 19.33 秒 | 166.754 秒 | 完整输出 580 帧、三路视频；训练侧 LeRobot loader 可读 |
| `arrange_topple_dominoes/5` | 615 帧 / 20.50 秒 | 149.800 秒 | 同一 NaN 校验失败，未输出 LeRobot |

另以 `add_remove_lid/0` 的 94 帧成功直出作为预热，未纳入计时统计。两段成功样本合计
846 帧，即 28.2 秒原视频，累计耗时 243.896 秒；`243.896 / 28.2 = 8.65`。
这里的分母是**成功片段的原视频时长**，不是完整原始语料或含失败开销的最终可用数据时长。
四段计时中有两段在 `handpose/pipeline/egodex_reviewed.py` 的
`hand_joints` 一致性检查报 `NaN location mismatch`。检查曾把无效源帧的
NaN 与求解器保持的有限上一帧关节逐项比较；现仅比较求解器接受的源帧。
原失败片段的完整回归仍待执行；A100 已于 2026-09-29 通过 `brainco-a100-6007` 连通。2026-10-05 审计另发现新直出入口的手指字段顺序错配；此前求解数值一致与视频可读检查不足以证明出口字段语义正确，修复后须一起回归。
这些样本只涵盖两种任务、数十秒视频，且输出写到本地 scratch；不能把 8.65 小时
视作 NAS 正式批量直出的吞吐承诺。机器可复核的样本路径、哈希与结果见同目录
`benchmark_report.json`；未改动正式 100h 数据和转换脚本。

### 模型训练：LeRobot → PI0.5

| 训练数据 | 训练任务 | 耗时 | 状态与依据 |
|---|---|---:|---|
| 约 1h | 单 A100、batch16、horizon16，首轮 10k 步 | 约 16 小时 05 分 | **已确认**；按训练指标首末时间统计 |
| 约 1h | 同一实验从 10k 续训到 30k，追加 20k 步 | 约 56.2 小时 | **已确认**；续训实测 |
| 100h | 单 A100、batch16、horizon16，完成 20k 步 | 约 54 小时 36 分 | **已确认**；以正式训练日志的 20k 步完成记录为准，训练前准备与后续评估不计入 |

训练耗时不含数据处理、训练前准备和模型评估；续训实测包含运行期间保存 checkpoint 的开销。

## 6. 100 小时训练计划与验收

- 源片段级 train/val 隔离，固定 plan 与代码版本；只用训练集计算归一化统计。2026-09-23 预检报告为 `status=ready`：train 56,099 个 source、val 2,843 个 source，两者无交叉；每份数据抽查首/中/末 3 段的 parquet 与三路视频；训练 loader 的首/中/末样本均给出 16×56 动作块，头部 mask 开、双腕 mask 关。日志：`runs/ego/100h_20k/preflight.log`。
- 训练集 `norm_stats.json` 的 state/action 各 56 维，均为有限数，标准差和 q01–q99 跨度均大于 0；最窄的是左拇指 DIP（state 跨度 0.00439 rad、action 跨度 0.00457 rad），没有退化维度。`statistics_provenance.json` 记录了训练集 info、episodes 和转换报告的 SHA-256、16 帧 horizon、14 个臂维度及 10,809,850 训练帧。已有统计不需重算。
- 50 步短测在单张 A100 上完成：loss 从 1.5990 到 1.4009，50 条指标均为有限数；GPU 占用约 16,908 MiB。随后从第 49 步 checkpoint 恢复并完成第 50 步，loss 1.5643，写出的 `50/` 含 `params/`、`train_state/`、`assets/` 与配置；其中统计文件的 SHA-256 与原始 `norm_stats.json` 一致。51 条逐步指标在该实验 checkpoint 根目录的 `metrics.jsonl`。每次打开 64,502 段 NAS 数据约耗 5 分钟，不计入单步均值。
- 100h 配置沿用 1h Ego 正式训练的 PI0.5 LoRA、action interface、优化器、batch16、horizon16、warmup1000 与 5e-5→5e-6 学习率；改动限于数据/统计路径、0 m offset、20k 总步数及匹配的 decay 周期、实验标识和元数据。它与自采真机 0713 配置并非逐字段一致：自采数据为 70 维 state 输入并启用真实手腕图像，而 Ego 数据是 56 维且屏蔽黑色手腕图像；本批还使用训练集分位数归一化。
- 训练仍调用同一个 `scripts/train.py`；本轮没有改 100h 正式 YAML 的模型和优化器设置。共享 `data_loader.py` 保留了针对新版 HuggingFace `Column` 的 Arrow 标量列读取修复，避免 1080 万行时间戳逐行生成 Python 对象；此前 v1 暂存区还包含 SwanLab/指标记录和 Ego 相机 mask 的共享代码改动，不能把整个仓库描述为“infra 完全未改”。
- 独立实验从 `pi05_base` 开始，batch16、horizon16、20k 步，warmup1000，学习率 5e-5→5e-6。模板：`BrainCo-IL/src/openpi/training/training_config_template/ego/egodex_pi05_100h_offset0_20k.yaml`；本批 offset=0，不能复用 1 小时 offset=0.025m 的统计或 checkpoint。
- 20k × batch16 = 32 万个采样起点，约为 1,080.985 万训练帧的 **2.96% 一轮**；每个起点监督后续 16 帧，窗口可重叠。先作为基线，之后看验证结果是否延长训练。
- 本批正式启动前已补齐黑手腕，检查帧同步、关节顺序、三路视频与真实 loader，并完成 50+1 步 profile/恢复；正式训练于 2026-09-26 跑满 20k 步。SwanLab 网页曲线已由完整的 `metrics.jsonl` 补齐，但本地 SwanLab 原记录只到约 730 步，自动日志链路仍需排查。
- 相机 mask：LeRobot 固定保留三路相机字段；训练配置不再补黑图，设置 `mask_wrist_images=true` 使左右手腕 `image_mask=false`，Transformer 不使用手腕 image token，但 image encoder 仍会执行。
- 配置管理：Ego 模板集中在 `BrainCo-IL/src/openpi/training/training_config_template/ego/`；新运行的日志与 YAML 快照放忽略的 `runs/ego/`。旧 `logs/egodex_*` 保留为历史记录，不再作为新实验的配置来源。
- 部署封装：100 小时最终 checkpoint 已手动生成 `policy_contract.json`；新 checkpoint 的保存流程现在自动生成并校验契约，成功后写 `deploy_ready.json`。序列化配置夹具检查及 A100 既有 checkpoint 契约再生成已通过；完整真实权重加载、保存流程和部署输出验收仍待执行。
- 修改边界：封装只写入目标 checkpoint 目录，不改 5090 的部署代码，也不改 5090 上的 BrainCo-IL 仓库；当前上传的 29999 checkpoint 只新增 contract，并给同目录 `train_config.yaml` 增加外层 artifact 元数据，内部训练配置保留。
- 固定验证集比较初始化/10k/20k；左右臂、左右手MAE及可视化回放。
- 训练端与5090部署端的56维选择、三路相机字段和absolute输出契约统一后再验证实机。
- 待讨论：外移是否统一0.025m；合成state与反馈state差异；padding策略；硬件编码；数据/模型保留策略。

## 7. 项目目录与历史导出命令

`BrainCo-IL/scripts/ego/` 放 Ego 专用入口与 shell 脚本，`scripts/bread_turtle/` 放另一条管线的 shell 脚本；`BrainCo-IL/tests/scripts/` 放脚本测试；通用训练引擎留在原位。Mac 上与 Ego 无关的本地改动单独审阅，不同步进 A100 工作仓库。下面是已完成的 100 小时导出命令，保留作复现记录；不要在同一输出目录再次启动。

旧批次的 64-worker/HDF5 辅助脚本已移至 `/mnt/data_nas/ruibin/egodex_project_archive_20260923/BrainCo-Dataset/`，`manifest.json` 保存原路径与 SHA-256；新直出入口仍在 `BrainCo-Dataset/scripts/ego/`。这些归档脚本不属于本次 Ego v1 提交。

```bash
cd /mnt/data_nas/ruibin/egodex_lerobot_export_20260922
nohup env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /mnt/data_nas/ruibin/BrainCo-Dataset/.venv-mint/bin/python -u export_egodex_receipts.py \
  --output /mnt/data_nas/ruibin/dataset/egodex_100h_offset0_lerobot21 \
  --train-hours 100 --val-hours 5 --workers 8 --offset 0 \
  >> export.log 2>&1 < /dev/null &
echo $!
tail -f /mnt/data_nas/ruibin/egodex_lerobot_export_20260922/export.log
```

- 完成依据：`conversion_report.json` 中 `status=complete`；`progress.json` 提供实时源数量、已导出时长和耗时。
- 重复执行同一命令会校验并跳过已完成source；文件锁阻止同目录重复运行。`Ctrl+C`退出tail不会停止后台导出。
- 数据选择实测约100.09小时训练集、5.05小时验证集；以正式运行保存的plan为准。
- 旧导出器小样本训练/验证各 1 段、共 414 帧，验证了头部视频、56 维字段和断点复用；三路相机由后续补齐步骤负责。直出入口另用 94 帧样本验证了三路相机与动作逐值对齐。
- 正式批量、黑手腕补齐、真实 loader 预检、50 步短测与 checkpoint 恢复均已完成；用户启动的 20k 步正式训练也已完成。

## 8. 100 小时训练操作顺序

以下是已完成批次在 A100 `/mnt/data_nas/ruibin/BrainCo-IL` 的历史复现命令，不要在现有同名实验目录重新启动。黑手腕补齐日志 `/mnt/egodex_work/black_wrist_100h/run.log` 已显示 train 和 val 均 `status=complete`；预检、50+1 步短测和正式 20k 步均已完成。这份配置按单张 A100 设计，历史运行前通过 `nvidia-smi` 选择可用 GPU 并设置 `CUDA_VISIBLE_DEVICES`，避免 JAX 看到机器上的全部 GPU。

```bash
export PYTHONPATH=src OPENPI_DATA_HOME=/mnt/data_nas/ruibin/.cache/openpi
export HF_HUB_OFFLINE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to one available GPU index}"
PY=/mnt/data_nas/ruibin/.venvs/ego-train/bin/python
$PY scripts/ego/prepare_100h_training.py --loader-check
$PY scripts/ego/make_100h_profile.py
```

先用 `runs/ego/100h_20k/profile50/train_config.yaml` 跑 50 步，再用 `runs/ego/100h_20k/profile_resume/train_config.yaml` 恢复 1 步：

```bash
$PY -u scripts/train.py runs/ego/100h_20k/profile50/train_config.yaml \
  > runs/ego/100h_20k/profile50/run.log 2>&1
$PY -u scripts/train.py runs/ego/100h_20k/profile_resume/train_config.yaml \
  > runs/ego/100h_20k/profile_resume/run.log 2>&1
```

当时确认第 50 步 checkpoint、`metrics.jsonl` 的 loss、步耗时和 GPU 占用后，执行了以下正式训练命令：

```bash
mkdir -p runs/ego/100h_20k/formal
nohup env SWANLAB_MODE=local SWANLAB_LOG_DIR="$PWD/runs/ego/100h_20k/swanlog" \
  "$PY" -u scripts/train.py \
  src/openpi/training/training_config_template/ego/egodex_pi05_100h_offset0_20k.yaml \
  > runs/ego/100h_20k/formal/train.log 2>&1 < /dev/null &
echo $!
tail -f runs/ego/100h_20k/formal/train.log
```

SwanLab 本地面板可用 `/mnt/data_nas/ruibin/.venvs/ego-train/bin/swanlab watch runs/ego/100h_20k/swanlog -h 0.0.0.0 -p 5092` 启动。checkpoint 位于 `checkpoints/egodex_pi05_100h_offset0_20260923/offset0_bs16_steps20000_v1/`；最终 `19999/` 已有部署契约。固定验证集比较仍待完成。

## 9. 后续项目改进 TODO

本轮 Post-train 的优先级、依赖和验收状态以 [POSTTRAIN_TODO.md](POSTTRAIN_TODO.md) 为准。以下保留历史待办背景。

- [ ] 完成自动契约封装的既有 checkpoint 真实加载及 5090 部署端验证；训练保存端代码和夹具检查已完成，历史手动入口保留。
- [ ] 在原失败样本 `add_remove_lid/4` 与 `arrange_topple_dominoes/5` 上回归 NaN 检查修复，并重新报告成功率和吞吐。
- [ ] 把单片段 EgoDex 直出扩为批量管线，补齐确定性计划、source 隔离、锁与断点恢复、失败清理、代表性吞吐基准和自动化回归测试，再决定旧 HDF5 中转实现的归档/退役。
- [ ] 为 OSS 原始/历史中间产物、NAS 权威 LeRobot 与 checkpoint、本地 scratch 建立清单和保留策略；先固定引用与校验和，再考虑迁移或清理现有 100 小时数据。
- [ ] 修齐黑手腕补齐后的 train/val `provenance.json`，并验证视频、元数据和报告一致；为数据转换与固定验证集评估补齐版本化结果。
