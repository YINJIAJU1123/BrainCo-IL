# 自采 MINT v3：388段首轮混训准备与待确认方案

> **2026-10-24 状态更新**：用户已确认统一任务文本、旧真机来源及三组比例；下面保留 10-23 提案历史，当前执行以 [REPORT.md](REPORT.md) 为准。

日期：2026-10-23，北京时间。用户授权推进388段数值通过集，17段继续隔离；本轮完成只读来源核对、命名字段重排和可逆数据准备。**训练未启动，配方和split仍是建议，未把建议当成已确认配置。**

## 当前交付

- 388段、71,175帧，约39.54分钟；17段仅在排除ID表中出现，没有读取它们的动作进行本轮准备。
- 每段最终HDF5当前SHA256均与验收清单一致；绝对关节数组按命名重排为训练接口56D，生成388个独立NPZ派生文件。原HDF5、QC、视频与验收报告只读。
- 完整来源、原始SHA、重排规则、派生SHA见NAS `approved_pool_manifest.json`；仓库内[通过集与来源SHA](approved_set.json)、[字段核验](mapping_validation.json)、[split提案](split_proposal.json)、[原norm兼容诊断](old_norm_compatibility.json)。完整manifest的SHA保存在通过集附件。
- NAS准备根：`/mnt/data_nas/ruibin/ego-posttrain-prep/mint_v3_388_20261023/`；派生数组在`mapped_actions/`，新代码在`code/prepare_manifest.py`。源批次仍是 `/mnt/data_nas/ruibin/runs/mint_ego_bread_fruit_v3/full_v3_20261023/`。
- 当前尚未生成本批LeRobot训练/验证/测试目录，也未创建可执行训练配置、模型更新或SwanLab run。不能将NPZ准备完成称为已经开始训练。

## 数据质量与准入口径

来源验收：`full_v3_20261023/acceptance_20261023/REPORT.md`、`evidence/delivery_sets.json`、`evidence/ledger.json`、`delivery/artifact_manifest.json`。仅选择明确的`numerical_pass_visual_pending`388段；保留`quarantine`17段排除清单。

388段数值通过，完整视觉通过集仍为0；实际视觉抽查48/388段、328/71175个帧位置，另340段未抽查。取物阶段偏直、局部手指映射幅度和单目尺度/相机校准局限仍在。用户本次授权进入训练准备，不把状态修改为视觉通过。原CSV方向标签没有被统一或改写。

来源帧率为30000/1001，已生成片段与动作是30Hz一一对应，沿用原验收的帧映射；不能重新按两个时钟独立切片。动作chunk不能越过源segment、无效区间或task边界。所有通过段至少87帧，horizon16可在段内按既有尾部padding规则读取。

## 42手部字段顺序的实际核验

`direct_to_lerobot.py`是原始EgoDex标注求解入口，当前42D字段顺序风险仍未修复；这批已生成MINT机器人HDF5不能直接套用该入口重新求解。

本轮从每个HDF5的`meta.attrs`读取实际关节名：手部原生顺序为thumb→index→middle→ring→little，thumb为CMP/CMR/MCP/PIP/DIP；训练参考是little→ring→middle→index→thumb，thumb为MCP/PIP/DIP/CMP/CMR。左手原生名称虽然带`L_right_`前缀，仍须按元数据中的左手group与L前缀映射为`left_`，不能把`right`字样当成右手标签。

映射时验证双臂`Joint1_L..Joint7_L`与`Joint1_R..Joint7_R`，手部仅允许明确L/R前缀和已知关节后缀；未知、重复或缺失名称拒绝。目标56D名称与旧真机`meta/info.json`去掉前14D EEF后的名称逐项一致。388段全部完成：

1. 当前源SHA验证；来源维度、float32、有限值、arm_valid和连续帧索引检查。
2. 每一目标列与对应原生命名列逐值一致；逆排列还原原56D数组逐值一致。
3. NPZ写入后重新加载，动作逐值一致；保留各段边界和source_frame_index。

数组为robot-base下absolute关节角（rad）。后续LeRobot state拟沿旧Ego约定使用当前帧同一合成关节量，不代表实测机器人反馈；进入模型前双臂才做相对当前state的delta，双手absolute，horizon16。手部碰撞/限位仍沿用同SHA成品的原QC；此命名映射验证不等于物理抓握或视觉GT验证。

## 推荐的首轮最小实验（待用户确认）

结构化参数见[推荐方案JSON](recommended_recipe.json)。

目的：在共同100h初始化、相同真机暴露与训练预算下，观察加入本批自采Ego后的目标任务适应情况。**建议先做一组，避免同时改变配比、步数、norm和初始化。**

| 参数 | 推荐值与依据 |
|---|---|
| 模型 | 沿用π0.5，56D、horizon16、max_token_len256、bfloat16；VLM/AE LoRA及action/time接口冻结策略沿旧三组 |
| 初始化 | 原100h `egodex_pi05_100h_offset0_20260923/offset0_bs16_steps20000_v1/19999/params`，A100存在已核对；新optimizer，从step0开始 |
| 混合比例方向 | **自采Ego:真机=1:1** |
| 真机 | 暂提议旧 `revotron_item_sorting_reviewed_v1`，55,243帧；是否继续用该数据待用户确认 |
| 暴露/预算 | 真机一轮：R=ceil(55243/32)*32=55,264（补21）；Ego抽55,264个无放回训练起点；batch16，6,908次更新 |
| 随机种子 | 训练/coverage seed42；Ego独立排列、真机seed43；sample游标按optimizer step |
| LR/optimizer | 沿旧1:1：warmup691、cosine decay6908、5e-5→5e-6；AdamW(.9,.95)、eps1e-8、weight_decay1e-10、clip1、EMA关闭 |
| norm | 首轮建议固定原共享norm `94398dac18e6717128104ff0d0cd5a15665c5fb451ae876c9c863f366a589eeb`，方便与旧1:1同尺度比较；真实loader归一化与模型前向仍待验收 |
| 相机/prompt | Ego head有效、双腕false；真机三路true；按各自原task读取，绝不把相反方向标签改成同一句 |
| 输入与隔离 | CPU默认8，内部线程1；新训练入口在本准备根的隔离代码/派生目录，冻结源码SHA；不用旧三组run目录或共享正式源码做新修改 |
| 日志/模型保存 | 新exp_name、新checkpoint目录、新SwanLab ID，source loss/曝光/保存恢复沿原训练链路；待确认配方后生成resolved配置 |

原三组末100条训练real loss均值约0.03499/0.03480/0.03394，没有统一真机验证，不能据此认定5:2胜出。旧teleop模板也没有本次已核实的独立真机基线评价。首轮可使用已完成的旧1:1最终6907和共同初始化19999作冻结评价参照；对照差异包括Ego来源、任务组成和标签工程，不能把全部变化归因于单一比例。若要证明“加入自采Ego优于只用真机”，后续需另外批准同预算真机单域对照，本轮不自动添加该实验。

## 分组划分提案（未生效）

| 集合 | 完整源视频组 | 数值通过段 | 帧数 |
|---|---|---:|---:|
| train | GX010158、GX010159、GX010161、GX010165 | 345 | 62,469 |
| val | GX010149 | 16 | 3,051 |
| test | GX010151 | 27 | 5,655 |
| 合计 | 六源 | 388 | 71,175 |

三份集合按完整源视频隔离，同一源视频所有切段只能进入一个集合，且两种task在每份集合均出现。训练池62,469足够无放回抽55,264；旧3:1需165,792、5:2需138,160个Ego起点，当前无放回采样器不能直接套用。改变重复采样策略属于新的实验设计。

六视频没有已核实的录制session/日期分组，故上表只能称按源视频留出，不能宣称跨session独立测试。旧100h和teleop的provenance表明它们来自EgoDex，未出现上述GX源标识；检查范围内未找到本批自采数据已经进入π0.5训练的manifest。不过它们没有完整原始session哈希表，因此不能把“未检出”写成全球无同源泄漏证明。v3_0003等源片段曾用于MINT上游动作/手型调试与视觉选择；全批标签算法也经过六源pilot调试，留出结果须注明这个上游工程选择限制。

若六源实际来自同一session，严格的跨session评估需要另采新session。无需为了凑比例把val/test重新放回训练；最终split需按用户确认的session信息或接受上述范围后锁定。训练过程中只用val选择checkpoint，test留到方案冻结后使用。

## 反向任务文本与负迁移检查

自采Ego task为`put bread into right basket`与`put fruit into left basket`；旧真机task为`put bread into left basket and put fruit into right basket`。这是有条件的不同任务，混训时各保留原文本；相机视角与左右语义尚不能据名称自动推断为同一物理动作。

评价要分域、分物体/目标方向记录，不能只看合并loss。先核对固定样本的task token、相机mask、state/action字段和帧对齐；统一噪声、采样步数与norm绑定，再比较19999/旧1:1/本轮模型的各域flow loss、反归一化关节误差。新源Ego留出只有合成标签，结果反映标签拟合；旧真机单段全用于训练，没有独立真机泛化集。真正“放错篮”的负迁移需独立任务数据或统一实机试验记录，不能靠训练loss排除。

## norm诊断与仍需执行的门槛

只对提案train四源做了CPU诊断：旧norm下state左右手超出q01/q99的比例约4.79%/7.73%，action左右手约5.28%/10.41%，归一化动作绝对值峰值约3.27。没有出现数值爆炸，但存在分布迁移；不据此修改norm、夹断数值或宣布模型前向通过。完整结果见[诊断JSON](old_norm_compatibility.json)。若真实loader/前向出现异常，再单独说明并决定重新统计，仅使用确认后的训练集。

训练前尚需：确认推荐配方、真机来源与split → 从已命名映射的388派生池导出独立LeRobot train/val/test → 验证Parquet命名字段/像素PTS/边界padding/prompt及mask → 训练集norm/模型有限值与保存恢复验收 → 固定全部source/config SHA → 核实GPU2所有权并释放本任务留存，再实际启动。GPU2守护165099的UUID、9GiB和proc启动身份已核实，保持留存；GPU7守护未操作。启动前必须重查PID，历史PID不能直接复用。未启动训练，未commit、未push。

复现本轮CPU准备见[prepare_manifest.py](prepare_manifest.py)。它只读取388段，写到独立NAS准备目录；当前输出已存在，勿原路径重复执行。若需重验，另建attempt目录并明确单一调度，不运行任何退休MINT控制器。
