# EgoDex → Revotron → LeRobot 训练管线

项目记录：A100 是工作环境，Mac 用于调试与 review。代码按可复核的小版本集中提交；本轮目录整理与训练准备暂不提交。

## 1. 目标与流程

`EgoDex 视频/标注 → 手指 retargeting + 双臂 IK → 有效区间筛选 → LeRobot 2.1 → PI0.5 训练 → Revotron 验证`

- 以后从原始 EgoDex 处理动作时，目标是直接写 LeRobot Parquet + 三路 MP4；原始标注仍是 HDF5，但不再生成动作/图像中间 HDF5。手部 retargeting、双臂 IK 和质量检查仍须执行。
- 单片段入口 `BrainCo-Dataset/scripts/ego/direct_to_lerobot.py` 已实现：94 帧样本与旧的 reviewed 双阶段求解逐值完全一致；三路视频由 LeRobot PyAV loader 验证。批量调度、断点续跑与吞吐测试待补齐。
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

### 模型训练：LeRobot → PI0.5

| 训练数据 | 训练任务 | 耗时 | 状态与依据 |
|---|---|---:|---|
| 约 1h | 单 A100、batch16、horizon16，首轮 10k 步 | 约 16 小时 05 分 | **已确认**；按训练指标首末时间统计 |
| 约 1h | 同一实验从 10k 续训到 30k，追加 20k 步 | 约 56.2 小时 | **已确认**；续训实测 |
| 100h | 单 A100、batch16、horizon16，计划训练 20k 步 | 粗估约 54 小时，排期预留 2–3 天 | **短测外推**；本批 50 步均值 9.80 秒/步（前 5 步后 9.67 秒/步），NAS 读取造成明显波动；正式运行以 SwanLab 实测为准 |

训练耗时不含数据处理、训练前准备和模型评估；续训实测包含运行期间保存 checkpoint 的开销。

## 6. 100 小时训练计划与验收

- 源片段级 train/val 隔离，固定 plan 与代码版本；只用训练集计算归一化统计。2026-09-23 预检报告为 `status=ready`：train 56,099 个 source、val 2,843 个 source，两者无交叉；每份数据抽查首/中/末 3 段的 parquet 与三路视频；训练 loader 的首/中/末样本均给出 16×56 动作块，头部 mask 开、双腕 mask 关。日志：`runs/ego/100h_20k/preflight.log`。
- 训练集 `norm_stats.json` 的 state/action 各 56 维，均为有限数，标准差和 q01–q99 跨度均大于 0；最窄的是左拇指 DIP（state 跨度 0.00439 rad、action 跨度 0.00457 rad），没有退化维度。`statistics_provenance.json` 记录了训练集 info、episodes 和转换报告的 SHA-256、16 帧 horizon、14 个臂维度及 10,809,850 训练帧。已有统计不需重算。
- 50 步短测在单张 A100 上完成：loss 从 1.5990 到 1.4009，50 条指标均为有限数；GPU 占用约 16,908 MiB。随后从第 49 步 checkpoint 恢复并完成第 50 步，loss 1.5643，写出的 `50/` 含 `params/`、`train_state/`、`assets/` 与配置；其中统计文件的 SHA-256 与原始 `norm_stats.json` 一致。51 条逐步指标在该实验 checkpoint 根目录的 `metrics.jsonl`。每次打开 64,502 段 NAS 数据约耗 5 分钟，不计入单步均值。
- 100h 配置沿用 1h Ego 正式训练的 PI0.5 LoRA、action interface、优化器、batch16、horizon16、warmup1000 与 5e-5→5e-6 学习率；改动限于数据/统计路径、0 m offset、20k 总步数及匹配的 decay 周期、实验标识和元数据。它与自采真机 0713 配置并非逐字段一致：自采数据为 70 维 state 输入并启用真实手腕图像，而 Ego 数据是 56 维且屏蔽黑色手腕图像；本批还使用训练集分位数归一化。
- 训练仍调用同一个 `scripts/train.py`；本轮没有改 100h 正式 YAML 的模型和优化器设置。共享 `data_loader.py` 保留了针对新版 HuggingFace `Column` 的 Arrow 标量列读取修复，避免 1080 万行时间戳逐行生成 Python 对象；此前 v1 暂存区还包含 SwanLab/指标记录和 Ego 相机 mask 的共享代码改动，不能把整个仓库描述为“infra 完全未改”。
- 独立实验从 `pi05_base` 开始，batch16、horizon16、20k 步，warmup1000，学习率 5e-5→5e-6。模板：`BrainCo-IL/src/openpi/training/training_config_template/ego/egodex_pi05_100h_offset0_20k.yaml`；本批 offset=0，不能复用 1 小时 offset=0.025m 的统计或 checkpoint。
- 20k × batch16 = 32 万个采样起点，约为 1,080.985 万训练帧的 **2.96% 一轮**；每个起点监督后续 16 帧，窗口可重叠。先作为基线，之后看验证结果是否延长训练。
- 正式启动前：黑手腕已补齐；检查帧同步、关节顺序、三路视频和真实 loader；用相同配置跑 50 步 profile、保存 checkpoint 并恢复 1 步；再由用户启动 20k 步，在 SwanLab 监测 loss、学习率、步耗时和 GPU。
- 相机 mask：LeRobot 固定保留三路相机字段；训练配置不再补黑图，设置 `mask_wrist_images=true` 使左右手腕 `image_mask=false`，Transformer 不使用手腕 image token，但 image encoder 仍会执行。
- 配置管理：Ego 模板集中在 `BrainCo-IL/src/openpi/training/training_config_template/ego/`；新运行的日志与 YAML 快照放忽略的 `runs/ego/`。旧 `logs/egodex_*` 保留为历史记录，不再作为新实验的配置来源。
- 部署封装：开环前对 checkpoint 执行 `python scripts/export_policy_contract.py <checkpoint>`，生成部署需要的 `policy_contract.json`，并补齐 `train_config.yaml` 的 artifact 元数据。
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
- 正式批量、黑手腕补齐、真实 loader 预检、50 步短测与 checkpoint 恢复均已完成；由用户手动启动 20k 步正式训练。

## 8. 100 小时训练操作顺序

在 A100 的 `/mnt/data_nas/ruibin/BrainCo-IL` 执行；黑手腕补齐日志 `/mnt/egodex_work/black_wrist_100h/run.log` 已显示 train 和 val 均 `status=complete`。预检和 50+1 步短测已通过，下列前两段保留作复验命令，正式启动只需执行第三段。这份配置按单张 A100 设计，运行前先用 `nvidia-smi` 选择可用 GPU 并设置 `CUDA_VISIBLE_DEVICES`；否则 JAX 可能看到机器上的全部 GPU。正式训练命令由使用者启动。

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

确认第 50 步 checkpoint、`metrics.jsonl` 的 loss、步耗时和 GPU 占用后，启动正式训练：

```bash
mkdir -p runs/ego/100h_20k/formal
nohup env SWANLAB_MODE=local SWANLAB_LOG_DIR="$PWD/runs/ego/100h_20k/swanlog" \
  "$PY" -u scripts/train.py \
  src/openpi/training/training_config_template/ego/egodex_pi05_100h_offset0_20k.yaml \
  > runs/ego/100h_20k/formal/train.log 2>&1 < /dev/null &
echo $!
tail -f runs/ego/100h_20k/formal/train.log
```

SwanLab 本地面板可用 `/mnt/data_nas/ruibin/.venvs/ego-train/bin/swanlab watch runs/ego/100h_20k/swanlog -h 0.0.0.0 -p 5092` 启动。checkpoint 写入 `checkpoints/egodex_pi05_100h_offset0_20260923/offset0_bs16_steps20000_v1/`，可在 10k、20k 对固定验证集做对比；部署前按第 6 节生成 `policy_contract.json`。
