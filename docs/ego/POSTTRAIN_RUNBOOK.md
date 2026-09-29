# Ego 100h Post-train：审核后启动与恢复说明

本文件是后续操作顺序；当前 **不得执行训练命令**。本轮只运行了静态采样模拟、真机只读筛查与契约夹具测试。

## 1. 审核与派生数据

5090 原始数据：`/home/phoebe/.cache/huggingface/lerobot/001/revotron_mit_revo3_mit_3cam_menghao_item_sorting`。本轮筛查包和 `review_manifest.json` 在 Mac 工作区 `reports/posttrain_prep_20260929/`。每个原 episode 的 `intervals` 必须无缝覆盖 `[0,length)`，逐段写 `decision=keep|exclude`；排除段必须写原因。信号和缩略图只提供候选线索，决定由人工确认。

审核后的 manifest 用 `scripts/ego/validate_review_manifest.py SOURCE MANIFEST` 校验，输出有效 `N`、`R=32×ceil(N/32)`、重复数和三组步数。若存在排除或任务边界，生成新的完整 LeRobot episode，重建 frame/global index、时间戳、视频、tasks 和统计；原始真机目录不改。逐路核对帧数与视频同步后，把派生数据同步到 A100 NAS 的 `/mnt/data_nas/ruibin/dataset/revotron_item_sorting_reviewed_v1`。Ego 与真机保持两个独立目录，由 A100 的同一 DataLoader 分别读取并在内存中组成混合 batch；不做物理合并或跨机实时读取。审核未通过前不能生成共享 norm。

## 2. A100 基线、norm 与配置

重新核对 A100 `BrainCo-IL` 和 `BrainCo-Dataset` 的 commit、未提交改动及配置；保持本轮 Mac 基线报告中的已有改动。确认 Ego 路径 `/mnt/data_nas/ruibin/dataset/egodex_100h_offset0_lerobot21/train`、初始化 checkpoint `.../offset0_bs16_steps20000_v1/19999` 和派生真机路径都可读，核验源哈希与 56D 名称顺序。

在审核数据上运行 `scripts/ego/compute_shared_norm.py --ego EGO --real REAL --output /mnt/data_nas/ruibin/BrainCo-IL/assets/posttrain_shared_20260929/egodex_real_reference_1to1`。该脚本使用全部有效真机起点和等量 seed=42 无放回 Ego 起点，统计完整 horizon16 chunk；会拒绝退化分位跨度。保存 `norm_stats.json` SHA 与抽样回执，另以真实 loader 抽样比较 normalized 分位数及异常放大；任何维度异常先修正数据/统计。

若有效 `N` 不等于 55,243，更新三份 YAML 的配额、`num_train_steps`、`warmup_steps`、`decay_steps` 和记录字段；三份配置必须仍共用同一个 norm SHA。三份 YAML 仅实验标识、比例和相应步数不同。三个计划文件：`posttrain_e1_r1.yaml`、`posttrain_e3_r1.yaml`、`posttrain_e5_r2.yaml`。当前计划分别是 6,908、13,816、12,089 步，均标记待审核。

## 3. 无训练验收

逐组运行 `scripts/ego/dryrun_posttrain.py CONFIG`：该入口读取真实 batch、完整加载初始化权重并做前向有限值检查，不调用反向、优化器更新，也不创建 SwanLab 训练 run。额外以固定 state、prompt 和噪声验证：改动 Ego 已 mask 的腕图不影响模型输出；真机三路 mask 均 true。对三组全计划模拟 batch 组成、真机覆盖和任意 step 的恢复接续。用既有 checkpoint 检验自动 contract、`openpi.deploy.plugin` 加载及反归一化后的有限 `16×56` absolute 动作；不得驱动真机。

## 4. 正式训练（仅在用户另行授权后）

同一空闲 A100、seed42、batch16、num_workers2；从 mid-train checkpoint 的 `params/` 完整加载权重，创建新 AdamW 与学习率计划，不恢复 mid-train 优化器。按用户最终指定的实验先行。每组保存独立 run 目录和稳定 SwanLab ID；本地 `metrics.jsonl` 为先持久化的权威记录，上传失败的行保留在 `swanlab_pending.jsonl`，网络恢复后人工核验并运行 `scripts/ego/replay_swanlab_pending.py RUN_DIR`。报告总 loss、来源诊断、梯度、学习率、累计样本和真机覆盖，同时说明三组总计算量不同。

中断恢复只针对同一 post-train run：先确认最后完整 checkpoint 和其 `train_config.yaml`，设 `resume=true`、`overwrite=false`，复用原 run 目录及配置。`TrainState.step` 是已完成优化步数，覆盖采样器从该步重建后续 batch；worker 预取不作为恢复游标。严禁把 mid-train 的优化器状态当成新实验恢复状态。checkpoint 完成后自动检查 norm、生成 contract，最后写 `deploy_ready.json`；缺少标记的 checkpoint 只可用于恢复排障。

## 5. 评价

三个模型使用同一个 Ego 固定验证子集比较左右臂、左右手 absolute MAE 和回放。真机训练样本指标仅是训练诊断。泛化比较依赖另录独立示教或同条件实机测试；每个模型使用自身绑定的 norm 反归一化，原 mid-train 对照继续使用原 norm。
