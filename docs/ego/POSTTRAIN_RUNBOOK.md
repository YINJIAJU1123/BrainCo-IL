# Ego 100h Post-train：审核后启动与恢复说明

本文件是后续操作顺序；当前 **使用用户指定的 A100 5 卡，开训验收已通过，第一组正式运行（已核验 step 10）**。状态同步于 2026-10-15，证据入口见 [README](README.md#文档导航与更新依据)，具体完成条件见 [TODO](POSTTRAIN_TODO.md)。CPU 准备和三组真实混合 loader 检查已完成；原 100h 在 5090 独立副本离线前向通过。新共享 norm 三组前向及实际保存恢复、部署加载均已通过。

## 0. 恢复后的代码与部署接口

A100 已同步准备版本：IL 准备提交 `25c5e5e`、Dataset `c5bb19b`。原改动保存在 `/mnt/data_nas/ruibin/ego-posttrain-backups/20261015_140810`，两仓库另有 `refs/backup/ego-pre-sync-20261015`；不要直接重新应用整个旧 stash，原代码已纳入当前提交。5090 根盘约剩 40 GB；后续运行前重新确认空间与 GPU。

5090 共享 backend 仍使用原 `/home/phoebe/Brainco/BrainCo-IL`（`ac65a34`）。本项目后续工作仅在 `/home/phoebe/Brainco/ego-posttrain/BrainCo-IL`，分支 `ego-posttrain-mask`、提交 `cb25680`；该独立副本已修复旧 Ego mask，18 项 CPU 检查通过。运行时从独立目录设置 `PYTHONPATH=src`，使用已有 `/home/phoebe/Brainco/BrainCo-IL/.venv/bin/python` 的依赖；本轮未改共享环境。原 100h 完整权重、固定噪声腕 mask 不变性和离线输出均已通过；三组新 norm 的模型前向已通过。ROS 反馈必须按 `msg.name` 重排。

## 1. 审核与派生数据

10 月 15 日决定：用户确认正确 task，并要求整段保留，不剔除摆物/复位区间。`review_manifest.json` 已批准完整 `[0,55243)`。派生数据已同步 NAS `revotron_item_sorting_reviewed_v1`，原始数据不改。只修正 `meta/tasks.jsonl` 和 `meta/episodes.jsonl` 的文本；三路视频全片解码和传输 SHA 均通过。真机 `tolerance_s=0.0002` 仅用于覆盖已量化的 float32 时间戳舍入误差（最大帧间隔误差 0.000113932 秒），Ego 仍用默认 0.0001 秒。


5090 原始数据：`/home/phoebe/.cache/huggingface/lerobot/001/revotron_mit_revo3_mit_3cam_menghao_item_sorting`。本轮筛查包和 `review_manifest.json` 在 Mac 工作区 `reports/posttrain_prep_20260929/`。每个原 episode 的 `intervals` 必须无缝覆盖 `[0,length)`，逐段写 `decision=keep|exclude`；排除段必须写原因。信号和缩略图只提供候选线索，决定由人工确认。

审核后的 manifest 用 `scripts/ego/validate_review_manifest.py SOURCE MANIFEST` 校验，输出有效 `N`、`R=32×ceil(N/32)`、重复数和三组步数。若存在排除或任务边界，生成新的完整 LeRobot episode，重建 frame/global index、时间戳、视频、tasks 和统计；原始真机目录不改。逐路核对帧数与视频同步后，把派生数据同步到 A100 NAS 的 `/mnt/data_nas/ruibin/dataset/revotron_item_sorting_reviewed_v1`。Ego 与真机保持两个独立目录，由 A100 的同一 DataLoader 分别读取并在内存中组成混合 batch；不做物理合并或跨机实时读取。当前全段保留决定已落实，统计已生成；若将来改变区间，另建数据/统计版本。

## 2. A100 基线、norm 与配置

运行前重新核对代码与路径。10 月 15 日 A100/Mac 代码 HEAD 已一致（IL 准备提交 `25c5e5e`、Dataset `c5bb19b`）；本轮文档另同步至 A100。Ego `/mnt/data_nas/ruibin/dataset/egodex_100h_offset0_lerobot21/train` 和 `.../offset0_bs16_steps20000_v1/19999` 已确认存在，审核派生真机目录和共享 norm 已生成。

本轮继续使用既有 100h 数据。若使用 `direct_to_lerobot.py` 生成的新 Ego 派生数据，先完成 TODO 中的 42D 手指名称/数值顺序修复及逐字段验收；10 月 5 日审计抽查的 100h 旧 exporter 样本逐值正确，新直出样本不能沿用这个结论。

在审核数据上运行 `scripts/ego/compute_shared_norm.py --ego EGO --real REAL --output /mnt/data_nas/ruibin/BrainCo-IL/assets/posttrain_shared_20260929/egodex_real_reference_1to1`。该脚本使用全部有效真机起点和等量 seed=42 无放回 Ego 起点，统计完整 horizon16 chunk；会拒绝退化分位跨度。保存 `norm_stats.json` SHA 与抽样回执，另以真实 loader 抽样比较 normalized 分位数及异常放大；任何维度异常先修正数据/统计。

若有效 `N` 不等于 55,243，更新三份 YAML 的配额、`num_train_steps`、`warmup_steps`、`decay_steps` 和记录字段；三份配置必须仍共用同一个 norm SHA。三份 YAML 共用相同数据、任务和统计，实验标识、比例和相应步数不同。三个计划文件：`posttrain_e1_r1.yaml`、`posttrain_e3_r1.yaml`、`posttrain_e5_r2.yaml`。当前计划分别是 6,908、13,816、12,089 步，均已标记 `approved_full_recording`。

## 3. 无训练验收

代码、数据和 norm 就绪后逐组运行 `scripts/ego/dryrun_posttrain.py CONFIG`：该入口读取真实 batch、完整加载初始化权重并做前向有限值检查，不调用反向、优化器更新，也不创建 SwanLab 训练 run。额外以固定 state、prompt 和噪声验证：改动 Ego 已 mask 的腕图不影响模型输出；真机三路 mask 均 true。对三组全计划模拟 batch 组成、真机覆盖和任意 step 的恢复接续。部署侧在 mask 适配后，通过当前独立 Policy Inferencer 验证 contract、完整权重加载及反归一化后的有限 `16×56` absolute 动作，不启动 actor。`revo_deploy` 的 dataset inference 模式也会发布控制命令，不作为只读验收入口。

## 4. 正式训练（GPU 分配与验收通过后）

同一空闲 A100、seed42、batch16、num_workers2；从 mid-train checkpoint 的 `params/` 完整加载权重，创建新 AdamW 与学习率计划，不恢复 mid-train 优化器。按用户最终指定的实验先行。每组保存独立 run 目录和稳定 SwanLab ID；本地 `metrics.jsonl` 为先持久化的权威记录，上传失败的行保留在 `swanlab_pending.jsonl`，网络恢复后人工核验并运行 `scripts/ego/replay_swanlab_pending.py RUN_DIR`。报告总 loss、来源诊断、梯度、学习率、累计样本和真机覆盖，同时说明三组总计算量不同。

中断恢复只针对同一 post-train run：先确认最后完整 checkpoint 和其 `train_config.yaml`，设 `resume=true`、`overwrite=false`，复用原 run 目录及配置。`TrainState.step` 是已完成优化步数，覆盖采样器从该步重建后续 batch；worker 预取不作为恢复游标。严禁把 mid-train 的优化器状态当成新实验恢复状态。checkpoint 完成后自动检查 norm、生成 contract，最后写 `deploy_ready.json`；缺少标记的 checkpoint 只可用于恢复排障。

## 5. 评价

三个模型使用同一个 Ego 固定验证子集比较左右臂、左右手 absolute MAE 和回放。真机训练样本指标仅是训练诊断。泛化比较依赖另录独立示教或同条件实机测试；每个模型使用自身绑定的 norm 反归一化，原 mid-train 对照继续使用原 norm。

## 2026-10-15 已生成统计

共享 norm SHA-256：`94398dac18e6717128104ff0d0cd5a15665c5fb451ae876c9c863f366a589eeb`。三组 CPU loader 检查已完成；不能用此替代 GPU 模型前向。当前使用用户指定的 5 卡执行第 3 节，再进行实际保存/恢复与日志验收。当前不创建新的 DSW 或连接正在配置的 4090。

## 2026-10-15 云端日志与执行入口

用户已授权验收通过后启动三组实验并上传 SwanLab。按 1:1 → 3:1 → 5:2 在同一张 5 卡串行执行，各组仍从原 19999 权重初始化独立 optimizer；验收用的短测 checkpoint 不作为生产初始化。执行脚本与状态保存在 `/mnt/data_nas/ruibin/ego-posttrain-prep/20261015/`，当前入口 `validate_and_train.py`、日志 `pipeline_v3.log`。任一检查失败立即停止；以 `gpu_validation.json`、`training_status.json` 和各 run 的 `metrics.jsonl` 判断实际进度。

设置 `SWANLAB_MODE=cloud`。用户重新登录后，云端创建实验和稳定 ID 补传已实测通过；SDK 0.8.4 在同一进程连续初始化时需要刷新认证，训练与补传入口现于 cloud 模式显式使用已保存凭据重新认证，不写入或输出密钥。训练记录追加 Ego/真机分别的 `loss_ego` / `loss_real`；待补传记录也执行 flush/fsync。
