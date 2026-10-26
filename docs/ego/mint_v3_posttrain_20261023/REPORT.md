# 自采 MINT 388 段混训执行记录

更新：2026-10-26（北京时间）。当前阶段：三组训练均正常完成，最终 checkpoint 及部署描述文件已核验；5090 已完成传输、91文件 SHA256 校验及最终目录发布。

## 本次用户确认

- 继续使用旧真机 `revotron_item_sorting_reviewed_v1`，55,243 帧、约 30.7 分钟。
- 用户确认两域任务一致；派生数据和训练 prompt 统一为 `put bread into left basket and put fruit into right basket`。原源文件及旧方向标注保留在 provenance，上一版“按反向任务分别训练”的提案已被本次确认取代。
- 执行 Ego:真机 1:1、3:1、5:2 三组，原 100h checkpoint 19999 初始化、新 optimizer、batch 16、共享 norm 不变。
- 保留按完整源视频划分的 train/val/test；17 段继续排除。
- 使用启动时核实空闲的授权 GPU，每组单卡。10-24 启动时未 commit/push；10-26 用户授权按功能分批 commit，仍不 push。

## 导出和配额

| 集合 | 段数 | 帧数 |
|---|---:|---:|
| train | 345 | 62,469 |
| val | 16 | 3,051 |
| test | 27 | 5,655 |

导出根：`/mnt/data_nas/ruibin/ego-posttrain-prep/mint_v3_388_20261023/lerobot`。视频沿用已验收源片段字节，腕部使用黑图视频并由 mask 排除。state 为当前帧合成关节值，action 为命名重排的绝对 56D；loader 对双臂做 delta，双手保持 absolute。全部动作/state 逐值验证、视频 SHA/帧数通过；真实 loader 验证通过：345 段尾部 padding、6 个跨域抽查起点原版/优化版逐值一致、三组全配额与 step37 恢复一致；两次 GPU 更新有限值通过，完整参数/optimizer 恢复哈希一致，policy_contract.json 与 deploy_ready.json 已生成。原数据质量仍是数值通过、完整视觉验收待完成。

| 比例 | Ego 抽样数 | 真机抽样数 | 更新步数 | Ego 抽样数/训练池 |
|---|---:|---:|---:|---:|
| 1:1 | 55,264 | 55,264 | 6,908 | 0.885 |
| 3:1 | 165,792 | 55,264 | 13,816 | 2.654 |
| 5:2 | 138,160 | 55,264 | 12,089 | 2.212 |

真机完整覆盖 55,243 帧并补 21 个起点。小 Ego 池通过显式 `coverage_ego_repeat=true` 按固定种子逐遍独立洗牌，每遍内无放回；三组共享 Ego 前缀和真机排列，保持按 optimizer step 恢复。默认配置仍拒绝超过唯一帧数的配额，旧实验行为不变。新增采样单测已通过。

## 文件与验证入口

- [导出脚本](export_lerobot.py)
- [三组配置生成](configure_runs.py)
- [loader 与采样验证](validate_loader.py)
- [原准备及来源说明](PREPARATION.md)

隔离源码位于 NAS 准备根 `source/`，配置位于 `configs/`，输出位于 `checkpoints/mint_v3_388_posttrain_20261024/`。三组 SwanLab 云端实验已创建，日志调用无错误，未发现 pending 上传文件；未逐点查询云端历史指标。

## GPU 验证与启动证据

[模型验证](model_validation.json)：两次更新 loss 0.284952 / 0.349116，参数与 optimizer 保存恢复一致，恢复 step=2；该验证目录独立于正式训练。正式三组均重新从原 19999 初始化、新 optimizer。

| 实验 | 物理 GPU | 训练 PID | 总更新步数 |
|---|---:|---:|---:|
| mint388_e1_r1 | 0 | 1200777 | 6,908 |
| mint388_e3_r1 | 4 | 1200783 | 13,816 |
| mint388_e5_r2 | 6 | 1200785 | 12,089 |

每组独立单卡、8 个输入 worker、prefetch4、decoder cache16、FFmpeg 单线程。2/7 卡已有留存或任务未操作。正式进程退出后 supervisor 按原留存要求在自己的卡保留 9GiB，并写入 exit/hold 状态；退出不等于训练成功，须看 returncode 和完整 checkpoint。

源码基于 A100 `06d79eb962dd99afada78c7ac7ab190b740f6a85` 的隔离副本，叠加显式 Ego 重复采样开关。A100 共享 checkout 保持干净，Mac 原有文档改动保留。10-26 按用户授权分批 commit，见下方提交记录；未 push。

## SwanLab

- [1:1 · mint388_e1_r1](https://swanlab.cn/@laryybb626/imitation/runs/53jbnc93)
- [3:1 · mint388_e3_r1](https://swanlab.cn/@laryybb626/imitation/runs/43erelqs)
- [5:2 · mint388_e5_r2](https://swanlab.cn/@laryybb626/imitation/runs/j63wckkr)

验证附件：[export_validation](export_validation.json)、[loader_validation](loader_validation.json)、[model_validation](model_validation.json)、[launch_status](launch_status.json)、[训练源码 SHA](training_source_sha256.json)。正式配置：[1:1](configs/e1_r1.yaml)、[3:1](configs/e3_r1.yaml)、[5:2](configs/e5_r2.yaml)。

## 启动后核验（2026-10-24T10:38:32.922481+08:00）

三组进程存活，均完成 step10，loss/梯度有限，首个10步窗口约1.34–1.35秒/步。初始窗口不代表全程稳定速度。详细指标见 [startup_verification.json](startup_verification.json)。本次任务已完成导出、验证、正式启动与链接交付；训练完成后的最终 checkpoint / SwanLab / 留出及真机评价仍待后续执行。

## 2026-10-26：训练完成核验

三组退出码均为 0，全部本地日志指标有限。最终 checkpoint 均包含 params、train_state、assets、train_config.yaml、policy_contract.json、deploy_ready.json 和 input_pipeline.json；contract hash、56D、horizon16、30Hz 及共享 norm SHA 已核验。

| 比例 | 最终 checkpoint | 完成时间（北京时间） | 末100条日志 real loss均值 | 末100条日志秒/步均值 |
|---|---|---|---:|---:|
| mint388_e1_r1 | 6907 | 10-24 13:17:51 | 0.032903 | 1.375 |
| mint388_e3_r1 | 13815 | 10-24 15:56:27 | 0.032092 | 1.356 |
| mint388_e5_r2 | 12088 | 10-24 15:16:35 | 0.031455 | 1.351 |

这些是训练日志窗口统计，尚未执行统一离线评估和实机任务成功率比较。日志每10步记录，最后日志步6900/13810/12080早于最终保存步；训练完成依据退出码和最终 checkpoint，不能用最后日志步推断少训。SwanLab run ID 与记录一致，未发现 pending 上传文件；本次未逐点读取云端指标。详见 [完成证据](completion_evidence_20261026.json)。

## 初始化、数据划分和本次变更

三组最终配置都核实为从 EgoDex 100h mid-train `offset0_bs16_steps20000_v1/19999/params` 初始化，`resume:false`、新 optimizer；并非接着旧三比例 post-train optimizer 继续更新。

数据按完整源视频固定划分为 345/16/27 段，只有62,469帧训练池参与配额抽样；val/test没有为凑比例放回训练。3:1和5:2依赖新的显式逐遍洗牌开关。新增的是小数据池采样支持、MINT导出/命名重排和验收脚本；视频解码优化复用已有实现，本次将已验证的输入profile用于全部三组。模型结构、冻结策略、共享norm、batch16、真机一轮暴露及三组原步数/学习率预算沿用已确认配方。

## 5090 交付与提交记录

接收位置：`/home/phoebe/Brainco/egodex_pi05_posttrain_20261024_mint388/`，与上一轮 `egodex_pi05_posttrain_20260929/` 并列。包括三组完整最终 checkpoint 与 optimizer 状态。已按源端 [91文件 SHA256 清单](final_checkpoint_manifest.json) 全量校验，17,287,829,209 字节全部一致，并原子发布到最终目录。验证了三组 contract hash、norm SHA、56D、horizon16、30Hz 和描述文件；未在5090执行完整模型加载或实机试验。见 [交付验证](delivery_verification.json) 和 [5090目录说明](DELIVERY_5090.md)。

已提交前两批：`fbb9ee9`（重复采样支持及测试）、`35b058f`（导出、验证和启动脚本）。本实验配置、证据与5090交付记录随第三批 `docs(ego): record completed MINT mixture runs and 5090 delivery` 提交。完整提交说明见 [COMMIT_MESSAGE.txt](COMMIT_MESSAGE.txt)。

5090 最终 checkpoint 根目录：

- `~/Brainco/egodex_pi05_posttrain_20261024_mint388/mint388_e1_r1/6907`
- `~/Brainco/egodex_pi05_posttrain_20261024_mint388/mint388_e3_r1/13815`
- `~/Brainco/egodex_pi05_posttrain_20261024_mint388/mint388_e5_r2/12088`

旧实验目录保留。最终差分传输复用了5090旧权重文件的独立派生副本，实际接收约1.52GB，重建完整17.29GB；校验以源端逐文件SHA为准。旧 checkpoint 只读，临时basis文件已清理。本次在BrainCo-IL按功能分三批commit，未push。
