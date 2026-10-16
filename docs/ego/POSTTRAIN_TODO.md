# Ego 100h Post-train TODO

文档状态同步：2026-10-16。入口见 [README](README.md)，完整执行证据保存在仓库外的准备 REPORT 与 `reports/posttrain_execution_20261015/`。

用户已授权三组串行训练，并要求通过独立 CPU 对照后将输入优化用于第三组。第一组 1:1 已完成 6,908 步，最终 checkpoint 为 `posttrain_e1_r1/6907`，deploy_ready 存在且 norm SHA 与共享资产一致；第二组 3:1 继续运行；第三组 5:2 尚未启动。每组仍重新加载原 19999 权重和新 optimizer。真机全部 55,243 帧保留；改变区间需要另生成统计和计划。2026-10-16 正式 CPU 双轮对照与实际 step37 恢复检查通过；优化版 11.61/12.06 样本每秒，原版 2.40/2.06，分别加速 4.84/5.85 倍。第三组采用 worker8/prefetch4、reader缓存16、FFmpeg1线程，组间启动时读取专用 input_pipeline 配置。第二组继续使用原输入路径。实际 GPU 总 step 加速与持续利用率待第三组运行后核验。

| 优先级 | 项目 | 当前状态 | 完成条件 / 依赖 |
|---|---|---|---|
| P0 | Mac/A100 基线 | 准备提交 IL `25c5e5e`、Dataset `c5bb19b` 已推送且两端一致；本轮补齐来源 loss、云端认证刷新与回归测试 | 原备份、stash 和 backup ref 保留；最新发布提交号见 REPORT |
| P0 | 5090 部署版本兼容 | 独立副本 `ego-posttrain/BrainCo-IL` 的旧 Ego mask 修复、18 项检查和原 100h 完整权重离线前向通过 | 共享 backend 未切换，实机闭环评价仍待执行 |
| P0 | 运行环境 | 用户指定 A100 5 卡；只由主进程初始化 GPU，数据 worker 使用 CPU；7 卡未使用 | 正式训练日志中检查吞吐、显存和指标；不占用其他任务 |
| P0 | 真机范围与 task | 完整 `[0,55243)` 保留，task 修正、派生 NAS 数据、SHA 与三路视频全片解码通过 | 已完成；这是全段保留策略，不宣称逐帧行为质量审核 |
| P0 | 来源级相机与 prompt | 三组真实 batch 的来源 mask/prompt、56D/horizon16 通过；改动 masked 腕图后 loss 差值均为 0 | 已完成模型前向验收 |
| P0 | 覆盖式采样 | 三组配额、唯一覆盖、CPU step=2 重建以及实际保存恢复后的 step=3 batch 一致性通过 | 每组真实配额 55,264，完整保留 55,243 起点并补采 21 |
| P0 | 共享 1:1 norm | 真机/Ego 各 55,243 起点；SHA `94398dac18e6717128104ff0d0cd5a15665c5fb451ae876c9c863f366a589eeb`；checkpoint 内文件 SHA 一致 | 已完成；三组共用相同统计 |
| P0 | 三份配置与前向 | 步数 6,908 / 13,816 / 12,089；loss 0.23185 / 0.20573 / 0.20573，全部有限 | 已完成；3:1 与 5:2 首 batch 相同，完整采样配额不同 |
| P0 | checkpoint 保存恢复 | 独立短测完成 3 次优化更新；恢复后 130 个状态数组逐值一致，step=3，norm/contract/deploy_ready 通过 | 第一组最终 checkpoint 已生成且 ready/norm 核对通过；完整模型评价仍待完成 |
| P0 | 部署加载 | 短测封装 checkpoint 经 policy factory 完整加载，真实输入输出有限 16×56 absolute 动作，abs max 1.77356 | 离线检查通过，无机器人命令；实机闭环另行验收 |
| P0 | 指标与上传 | 云端创建、上传及稳定 ID 补传通过；认证刷新修复通过；来源 loss 回归测试确认总 loss/参数更新不变 | 本地 metrics 与 pending 均 fsync；正式训练需持续查看曲线 |
| P0 | 正式训练 | 1:1 已完成 6,908 步，最终目录 `6907`；3:1 运行中，云端 run `uy1203p2`；5:2 等待组间交接 | [1:1 SwanLab](https://swanlab.cn/@laryybb626/imitation/runs/6hayni72)、[3:1 SwanLab](https://swanlab.cn/@laryybb626/imitation/runs/uy1203p2)；固定验证集和实机比较待做 |
| P0 | 数据供给与 GPU 空转 | 40 秒连续采样平均 GPU 利用率 21.6%，75.8% 点为 0%；主线程 83.1% 采样时间在等 DataLoader，worker 主要在视频读取解码 | 已完成官方方案调研；按下方视频优化清单执行 worker/预取 → 有界 decoder 缓存/批量取帧 → 兼容的 TorchCodec/NVDEC → 存储分片；补 loader_wait、compute_time 和平均利用率 |
| P0 | 是否扩轮与验证依据 | 当前真机覆盖 1 轮；近期真机训练 loss 明显下降，但未建立本轮独立真机验证集 | 保留 1 轮基线，建议后续评估 3 轮预算（20,720 / 41,440 / 36,260 步），有效改善时再考虑 5 轮；固定 Ego val、独立真机数据与实机效果决定；扩轮和学习率/采样恢复需统一规划 |
| P1 | 入口、资产和架构清单 | 已列审阅清单；100h 与 checkpoint 基本元数据已核实 | 补 OSS 实际大小、引用、哈希与迁移恢复方案；目录清理先审阅 |
| P1 | Ego 新直出字段顺序 | 10 月 5 日服务器审计发现 42D 手指名称与数值顺序错配，未修复；既有 100h 旧 exporter 抽查样本逐值正确 | 新直出入口按关节语义重排；逐字段与旧出口对照；排查该入口生成的派生数据，在确认前不用于训练 |
| P1 | Ego 直出 NaN | 无效源帧与求解器有限保持帧的比较已修正 | A100 上重跑 `add_remove_lid/4`、`arrange_topple_dominoes/5` 和成功样本；保留限位/FK 检查并重报成功率与吞吐 |
| P1 | 相机 provenance | 新出口明确“缺物理腕机”和“已物化黑色视频” | 100h train/val 旧 provenance 与视频/元数据一致性核验，必要时另附修正回执 |
| P1 | 100h 基线评价与部署 | 20k 训练已完成；固定验证集比较和 5090 部署验收仍待完成 | 在固定验证集比较初始化/10k/20k 的左右臂、左右手 MAE 与回放，再按部署流程验收 |
| P2 | 确定性批量直出 | 未开始 | 锁、断点恢复、原子发布、重试、失败回归和吞吐验收后，再提 HDF5 链路退役方案 |

## 下一步执行顺序（10 月 15 日 GPU 验收后）

1. 在同一张 A100 5 卡串行执行 1:1 → 3:1 → 5:2；各组独立目录、独立 optimizer 和稳定 SwanLab run ID。
2. 跟踪本地 `metrics.jsonl` 与云端 loss、来源 loss、梯度、学习率、覆盖进度和吞吐；上传异常用待补文件恢复。
3. 核验正式 checkpoint 的 `deploy_ready.json`，完成固定验证集比较；之后才进行实机闭环评价。

P1/P2 保留工程与评价待办；新直出字段顺序问题须在使用该入口新产物前修复。当前三组实验使用既有 100h 数据。

## 本轮实测与边界

- 5090 真机元数据：1 episode、55,243 帧、30 FPS；state/action 70D 且名称相同，尾部 56D；三路视频均为 55,243 帧。Parquet 连续索引和时间戳通过，state/action 非有限帧数均为 0。
- 每分钟关节速度的中位数在 0.817–1.281 rad/s 的 56D 向量范数范围内；这不是示教有效性判断。
- 静态计划采用 `R=55,264`，补采 21 个真机起点；Ego 唯一起点数量分别为 55,264、165,792、138,160。采样模拟证明真机 55,243 个原起点均覆盖一次。
- 10 月 15 日 A100 train/val 存在，episode 长度汇总匹配；首尾共 4 段 Parquet 与三路视频流元数据抽查通过。train 为 10,809,850 帧 / 64,502 段，val 为 545,248 帧 / 3,278 段。未全量解码或哈希所有数据/大权重。
- 10 月 15 日 A100 正式工作副本新增实测：覆盖采样、manifest 校验、契约封装共 4 项通过。5090 独立副本 mask/config/contract 共 18 项通过。后续完成共享 norm 与三组真实混合 loader CPU 验收、5090 原基线完整前向；三组新 norm GPU 前向和独立短测实际保存恢复均通过，正式训练保存点待生成。

- [ ] 评估数据路径与训练时 A100 绝对路径解耦，使 5090 读取当地验证集无需改写 checkpoint 的训练配置（保留 A100 原 README 待办）。

本轮新增 norm 分批统计、按清单读取 Parquet、来源级时间戳容差及 YAML 更新纳入 ego-train 提交。提交前 11 个变更文件在 Mac/A100 逐文件 SHA 一致；原始备份保留，发布前另备份至 A100 `ego-posttrain-backups/20261015_publish`。发布后的准确提交号与同步核验记录见外部 REPORT。

GPU 执行日志：A100 `/mnt/data_nas/ruibin/ego-posttrain-prep/20261015/pipeline_v3.log`；`gpu_validation.json` 为开训前验收快照，`training_status.json` 与正式 run 的 `metrics.jsonl` 表示后续运行进度。验收缓存仅复用本轮已校验的数据对象；原数据未改动。

利用率解释补充（2026-10-15）：旧日志的 100% 采自计算完成后的单个时刻；连续采样显示输入供给存在长时间等待。59.9 GiB 显存主要来自 JAX 75% 预分配。诊断证据与建议保存在外部准备 REPORT 的“轮数建议与 GPU 空转诊断”节；当前实验配置继续按 1 轮、batch16、worker2 运行。

<a id="video-loading-optimization"></a>

## 视频读取与解码优化（前三项已验收，2026-10-16）

前三项已完成独立实现、初筛和正式长测，第三组通过专用profile启用；后续阶段继续按下表推进。调研、官方来源及适配限制见外部准备 REPORT 的“视频读取解码方案调研与优化清单”节。已有 persistent_workers 和 Arrow 读取缓存；新增工作重点为视频路径，而非重复启用已有缓存。

| 顺序 | 待办 | 当前项目适配与验收 |
|---|---|---|
| 1 | 补输入链路计时；对比 worker2/4/8 与 prefetch2/4 | 当前 worker2、prefetch2、persistent_workers=true；固定 batch16、seed 和索引，测取 batch 等待 p50/p95、样本吞吐、平均 GPU 利用率及内存；同时限制 FFmpeg 总线程数 |
| 2 | 跳过 masked Ego 双腕视频解码 | 原黑腕视频仍读取；用相同形状/类型的黑占位图，验证 Ego mask=false、真机三路 true、固定噪声模型输出不变 |
| 3 | 每个 worker 复用有界 LRU decoder；按视频批量取帧 | 优先减少反复打开真机三路长视频；按同一视频组织内部 I/O 后恢复原采样顺序；保留帧号、PTS 和动作对齐，观测缓存命中率/文件句柄 |
| 4 | 基准 CPU TorchCodec、精确帧索引与 NVDEC/DALI | 当前 torch2.7.1 / torchcodec0.11.1 不在官方兼容配对内；先在项目独立环境匹配版本。官方较新 API 只作候选；验证 FFmpeg、CUDA、编码格式及 Torch→JAX 数据交接；保持时间戳容差 |
| 5 | 对比 NAS 与有限本地视频缓存 | 先测网络读取与 CPU 解码各占多少；缓存按视频 SHA、容量上限和命中率管理，预解码缓存单独核算空间；不以拷贝文件代替解码优化 |
| 6 | 评估分片存储、预解码帧缓存或 LeRobot v3 派生副本 | WebDataset/更少的大文件可改善顺序 I/O；需保留覆盖配额、索引和断点游标。保留当前 v2.1 原数据，另建派生版本；仅打包 MP4 不会消除视频解码 |

基准采用固定真实索引，先预热再记录至少 100 个 batch，区分启动与稳态；同一条件重复并报告波动。动作/state/prompt/norm/mask 与采样接续必须一致；跨解码后端另外比较帧号/PTS、像素和固定噪声模型输出。依次测量，收益超过波动且输入契约通过后再采用。当前 NumPy→JAX 输入边界需要专门设计设备预取，不能直接照搬 PyTorch CUDA 的 pin_memory/non_blocking 示例。

## 2026-10-16 中午暂停记录（下午已恢复并完成长测）

已停止独立 CPU 基准及其 worker；最后进度 `baseline_repeat1:100/132`，包含 32 个预热，正式双轮结果尚未形成。前三项实现及全部初筛数据逐字段一致，短测候选为 worker8/prefetch4；正式长测、第三组启用与代码提交/push 均待继续。第二组 PID1072324 在暂停核验时 step3930，第三组尚未启动；原串行调度继续，后续状态可能变化。

交接：[HANDOFF](</Users/larrybb/Documents/ChatGPT/Ego data/reports/posttrain_execution_20261015/video_loading_20261016/HANDOFF.md>)；暂停状态和测试进程退出证据见同目录 `PAUSED.json`。仅在用户要求继续后恢复工作；原定时巡检保持取消。

## 2026-10-16 输入优化验收与下一步

2026-10-16 正式 CPU 双轮对照与实际 step37 恢复检查通过；优化版 11.61/12.06 样本每秒，原版 2.40/2.06，分别加速 4.84/5.85 倍。第三组采用 worker8/prefetch4、reader缓存16、FFmpeg1线程，组间启动时读取专用 input_pipeline 配置。第二组继续使用原输入路径。实际 GPU 总 step 加速与持续利用率待第三组运行后核验。

- [x] 14个边界样本、完整12089-batch采样、四轮132-batch逐字段对照与实际step37恢复。
- [x] 前三项优化、六项profile验收检查；保留原代码快照和暂停记录。
- [ ] 第三组开始后核对run内input_pipeline.json、实际worker8和持续GPU利用率/step耗时。
- [ ] 三组最终checkpoint、SwanLab记录完整性和固定验证集/实机评价。

正式证据：A100 `/mnt/data_nas/ruibin/ego-input-bench/20261016/results_v3/`；benchmark_summary、handoff_validation与publication分别记录长测、入口验证和发布状态。CPU吞吐提升不是已实测GPU训练总加速。
