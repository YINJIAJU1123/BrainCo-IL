# Ego／真机 π0.5 LoRA：三比例实验完整记录

更新：2026-10-19。以下为当前实验记录；文末折叠保留10月17–19日旧快照及跨项目备注。实验结论、完成状态以本节和本次证据为准。

## 1. 目的、假设与可回答的问题

以已完成100h Ego mid-training的π0.5 checkpoint19999为共同起点，在**真机训练起点暴露量相同**的条件下比较Ego:真机=1:1、3:1、5:2。假设是增加Ego回放可能有助于保持人类动作先验，同时仍适应真机分拣任务；增加Ego也可能稀释真机更新占比。需要统一验证集和实机成功率检验这些假设，当前训练loss只能说明拟合与训练健康。

三组已完成6,908／13,816／12,089次更新，最终目录编号6907／13815／12088，均有deploy_ready。三组最终模型已完整复制到5090并通过迁移SHA核验。尚无统一离线验证或新模型实机成功率，**目前不能宣布哪个比例策略最好**。

这不是等更新数或等计算预算实验：总样本数、Ego暴露量、更新数和学习率调度长度均不同。5:2另采用输入优化，不能将墙钟差异归因于配比本身。没有0:1真机单域对照，也没有在本轮完成19999的统一评价，因此当前不能量化“加入Ego相对纯真机”的因果收益。

## 2. 数据、观察与动作接口

| 项目 | Ego | 真机 |
|---|---|---|
| A100路径 | `/mnt/data_nas/ruibin/dataset/egodex_100h_offset0_lerobot21/train` | `/mnt/data_nas/ruibin/dataset/revotron_item_sorting_reviewed_v1` |
| 版本/规模 | LeRobot v2.1；10,809,850帧、64,502 episode、30fps，旧exporter的100h offset0数据 | LeRobot v2.1；55,243帧、1 episode、30fps，约30.7分钟 |
| 任务文本 | 各条meta task，prompt_from_task=true | `put bread into left basket and put fruit into right basket`；由用户确认替换乱码，并在来源配置中override |
| 观察相机mask | head=true、left/right wrist=false；黑腕占位 | head=true、left/right wrist=true |
| 起点选取 | seed42排列中按各组配额取无放回前缀，未遍历整个100h | 全部55,243起点保留，seed43排列，额外补21起点 |
| 时间戳容差 | 0.0001秒 | 0.0002秒 |

两部分数据各在自己的目录，由同一loader读取并混合，无需物理合并。真机原始记录在5090的 `/home/phoebe/.cache/huggingface/lerobot/001/revotron_mit_revo3_mit_3cam_menghao_item_sorting`；本轮训练使用NAS派生版本。全段保留是用户选择，不表示逐帧排除了摆物/复位。未使用生成腕图替换输入。

数据指纹：Ego `meta/info.json` SHA256=`295fb64bdf581d75dedba1528aa39d0cdc63af396e602df62ec5ed7412d5e97c`；真机=`1fc81cb32d03d0375bdc53e3c83ef9b8278f93c08f65e7d033d7ad2dba164ea0`，来自原[snapshot](snapshot.json)。本次保存resolved配置，未重新全量哈希数据集。新的直出入口42D字段顺序问题仍在TODO，不能据本轮旧100h结果放行新出口。

state/action顺序为左臂7、右臂7、左手21、右手21，共56维关节量。真机原70维含14维EEF，输入适配移除EEF，保留56维关节；56维Ego保持对应语义。horizon16、策略配置30Hz。训练前双臂动作转换为**相对当前观测state的关节增量**，不是逐相邻动作差分；双手保持absolute。随后按共享q01/q99统计做quantile归一化：`2*(x-q01)/(q99-q01+1e-6)-1`。部署输出经反归一化并为双臂加回当前state，得到16×56 absolute关节目标。

共享norm来自固定seed42的Ego/真机各55,243起点、各自完整horizon16 chunk，三组都使用 `assets/posttrain_shared_20260929/egodex_real_reference_1to1/norm_stats.json`，SHA256=`94398dac18e6717128104ff0d0cd5a15665c5fb451ae876c9c863f366a589eeb`。每个最终checkpoint内的assets带同一norm。邻近起点chunk重叠，起点数不是独立视频数或互不重叠动作数。

## 3. equal_real_exposure：公式与实际暴露

令真机唯一帧起点N=55,243、coverage_rounds=1、round_to=32、batch B=16，Ego:真机=a:b：

```text
R = ceil(N / 32) * 32 = 55,264       # 真机起点暴露；55,243唯一 + 21补采
E = R * a / b                       # Ego起点暴露
S = (E + R) / 16                    # 总更新次数
第s步真机数量 = floor((s+1)*R/S) - floor(s*R/S)
```

| Ego:真机 / run | 更新S | Ego起点E | 真机起点R | 总样本E+R | 每batch Ego/真机 | warmup / decay |
|---|---:|---:|---:|---:|---|---|
| 1:1 / posttrain_e1_r1 | 6,908 | 55,264 | 55,264 | 110,528 | 8/8 | 691 / 6908 |
| 3:1 / posttrain_e3_r1 | 13,816 | 165,792 | 55,264 | 221,056 | 12/4 | 1382 / 13816 |
| 5:2 / posttrain_e5_r2 | 12,089 | 138,160 | 55,264 | 193,424 | 12/4或11/5 | 1209 / 12089 |

完整运行累计比例严格成立；不是视频时长比、数据集规模比或额外loss系数。coverage模式下两个来源的`weight:0.5`不控制配比。seed42生成相同Ego排列前缀，seed43生成真机排列，补21来自该排列开头；batch内部再用`SeedSequence([42,s,2])`打乱。resume按优化器step恢复，预取不改变游标。

以上完整域暴露由已完成的更新次数和确定性采样计划推得，并非把最后一条日志当作完整曝光计数。全12089-batch索引及step37实际恢复对照已通过。三组指标每10步记录，尾日志6900／13810／12080都早于最终更新，故尾日志累计样本数略少。

## 4. 模型、冻结范围与实际resolved超参

三组各自从 `/mnt/data_nas/ruibin/BrainCo-IL/checkpoints/egodex_pi05_100h_offset0_20260923/offset0_bs16_steps20000_v1/19999/params` 完整加载权重，**各建新optimizer**；不把上一比例的训练权重或optimizer传给下一组。resume=false、overwrite=false。

π0.5：action_dim56、horizon16、max_token_len256、discrete_state_input=true、bfloat16；VLM=`gemma_2b_lora`，action expert=`gemma_300m_lora`。源码`gemma.py`中前者attention/FFN LoRA rank16、alpha16；后者rank32、alpha32；默认rslora=false，缩放alpha/rank=1。

实际`freeze_strategy=lora_and_action_interface`的可训练参数匹配规则为：

```text
.*(lora|action_in_proj|action_out_proj|time_mlp_in|time_mlp_out).*
```

因此训练两套Gemma的LoRA，以及动作输入/输出投影和time MLP（包括匹配的bias）；其余参数冻结，包括视觉encoder、Gemma基础权重、非匹配embedding/norm/调制参数。并非整个action expert全量训练，也并非仅LoRA。π0.5此构造分支使用离散state token，不创建π0专用state_proj。范围由resolved配置与实际源码过滤器核对，未重新初始化全模型统计可训练参数总数。

| 参数 | 三组实际配置 |
|---|---|
| seed / coverage_seed | 42 / 42 |
| batch / FSDP | 16 / 1设备，A100 GPU5串行 |
| AdamW | b1=0.9，b2=0.95，eps=1e-8，weight_decay=1e-10；全局梯度裁剪1.0 |
| 学习率 | Cosine：peak5e-5→5e-6；各组warmup/decay见上表 |
| EMA | 关闭 |
| 保存/日志 | log_interval10；save_interval1000；keep_period2000；save_final_checkpoint=true |
| 实际worker | 1:1=2、3:1=2、5:2=8；第三组run保存配置已核对 |
| 第三组输入优化 | prefetch4、每worker/来源最多16个reader、FFmpeg1线程、跳过masked黑腕、只对选中帧转RGB；原像素/采样/恢复校验通过 |

resolved配置全文及其原始字节SHA、软件与源码证据见[本次核验JSON](documentation-evidence-20261019.json)和[原环境snapshot](snapshot.json)。运行环境记录：jax/jaxlib0.5.3、torch2.7.1、SwanLab0.8.4、Orbax0.11.13、LeRobot0.1.0；输入基准另记录torchvision0.22.1、av17.0.1。本次Mac/A100核对train.py、pi0.py、gemma.py、config.py、coverage_sampler.py、transforms.py、policy_config.py七份源码SHA一致。

代码沿革：1:1 runtime_code记录commit `56152137f8586621b4ff87ff9f5059f6ce19cc2e`，包含当时训练入口SHA和fresh optimizer声明；3:1沿同一串行旧入口运行，没有单独runtime_code文件，不能伪造独立版本记录。第三组输入优化发布commit `ea835f980d9ccb97d14307b31bd2c84b55d20571`，其run中的input_pipeline记录五份源码SHA和验收证据SHA。Dataset历史准备版本为`c5bb19b`；它不是本次数据文件的完整内容哈希。文档核验没有改动训练源码。

## 5. loss的精确定义和记录口径

令A为上文delta/absolute语义处理并归一化后的动作chunk（B×16×56）；ε为同shape标准高斯噪声，每样本采样 `t=0.999*Beta(1.5,1)+0.001`。构造 `x_t=t*ε+(1-t)*A`，目标速度 `u=ε-A`，网络输出v。模型返回每时间步56维均方误差：

```text
L[i,h] = mean_d (v[i,h,d] - u[i,h,d])²
l[i] = mean_h L[i,h]
loss = mean_i l[i]
loss_real = sum_i(real_mask[i]*l[i]) / max(sum_i real_mask[i], 1)
loss_ego  = sum_i(ego_mask[i]*l[i])  / max(sum_i ego_mask[i], 1)
```

这里误差是**归一化flow速度场MSE**，不是关节角MAE、完整动作采样误差或任务成功率。action_horizon尾部补齐按原loader保留；此loss函数没有另外按`action_is_pad`剔除损失。

来源诊断只在明确的coverage来源配置`{(ego,true),(real,false)}`下启用，以left_wrist image mask区分real与ego。复用同一前向和同一噪声，不重算来源目标，不额外加权反向。单步有 `loss=(n_real*loss_real+n_ego*loss_ego)/B`。日志把`infos`中各步指标分别做算术平均，通常每10次更新输出（首条step0只有1次）；5:2的n_real在4/5之间变化，因此**日志窗口均值不能简单套固定5:2权重反推**。图像训练预处理、时间和噪声随机性也影响曲线。源码依据：`pi0.py::compute_loss`、`train.py::train_step`及日志循环，SHA见本次JSON。

## 6. 已观察到的曲线与结果

下表以各run的本地metrics前100条／末100条记录作等记录权平均；通常约覆盖1000次更新，首条窗口较短。是训练诊断，未进行统一held-out前向；不能据下表直接排策略优劣。

| 比例 | 总loss 前→后 | real loss 前→后 | Ego loss 前→后 | 末100条秒/step均值 | 最后记录step / 总loss |
|---|---|---|---|---:|---|
| 1:1 | 0.11184→0.06327 | 0.11155→0.03499 | 0.11212→0.09156 | 5.624 | 6900 / 0.068655 |
| 3:1 | 0.11438→0.07202 | 0.13783→0.03480 | 0.10657→0.08442 | 9.346 | 13810 / 0.068396 |
| 5:2 | 0.11426→0.07021 | 0.13192→0.03394 | 0.10727→0.08471 | 1.577 | 12080 / 0.067538 |

三组训练loss均下降，末段real诊断均值约0.034–0.035，但这不足以证明收敛或泛化。第三组实测末段约1.577秒/step；前两组约5.624/9.346秒，涉及来源组成、硬件共享负载和输入实现差异，不能把它作为控制变量齐全的GPU加速倍数。独立CPU ABBA基准的4.84–5.85倍只描述相同输入下的供给吞吐。日志的GPU 100%来自计算时刻抽样，不能解释为持续平均利用率。

SwanLab ID已与A100各run的`swanlab_id.txt`核对：

- [1:1曲线：6hayni72](https://swanlab.cn/@laryybb626/imitation/runs/6hayni72)
- [3:1曲线：uy1203p2](https://swanlab.cn/@laryybb626/imitation/runs/uy1203p2)
- [5:2曲线：vrd7jl5u](https://swanlab.cn/@laryybb626/imitation/runs/vrd7jl5u)

三组当前未发现本地`swanlab_pending.jsonl`；本次未逐点查询云端API，因此“云端与本地每一条记录完全一致”仍待核对。run链接均为laryybb626/imitation下的本轮π0.5实验。

## 7. 最终checkpoint、迁移与使用

| 比例 | 更新次数 | 两端根目录下的相对路径 | deploy_ready UTC |
|---|---:|---|---|
| 1:1 | 6908 | `posttrain_e1_r1/6907` | 2026-10-15 19:36:50 |
| 3:1 | 13816 | `posttrain_e3_r1/13815` | 2026-10-17 05:02:34 |
| 5:2 | 12089 | `posttrain_e5_r2/12088` | 2026-10-17 10:30:14 |

A100根：`/mnt/data_nas/ruibin/BrainCo-IL/checkpoints/egodex_pi05_posttrain_20260929/`

5090根：`/home/phoebe/Brainco/egodex_pi05_posttrain_20260929/`

例如5090第三组最终checkpoint根为 `/home/phoebe/Brainco/egodex_pi05_posttrain_20260929/posttrain_e5_r2/12088`。迁移于2026-10-19 03:49:09 UTC（北京时间11:49:09）完成：91文件、17,287,304,438字节，每文件SHA256匹配。这里核对的是既有finished/source-manifests/三组verification回执，没有重复传输或重算17GB；[本地回执副本](transfer-verification-20261019.json)。5090原始回执目录：`/home/phoebe/Brainco/ego-posttrain/transfers/20261019-final-checkpoints/`。

加载时传入**最终编号目录根**，让factory从其中读取`params/`和`assets/`；不能将`params`子目录当checkpoint根，也不能将整组run父目录代替最终编号。以下为离线使用示例，本次未执行模型加载或机器人推理：

```python
from pathlib import Path
from openpi.training import config_io
from openpi.policies import policy_config
checkpoint = Path('/home/phoebe/Brainco/egodex_pi05_posttrain_20260929/posttrain_e5_r2/12088')
config = config_io.load_train_config(checkpoint / 'train_config.yaml')
policy = policy_config.create_trained_policy(config, checkpoint)
```

使用5090已有独立工程 `/home/phoebe/Brainco/ego-posttrain/BrainCo-IL` 的相容环境。checkpoint保留`train_config.yaml`、`policy_contract.json`、`deploy_ready.json`、`params/`及自身norm资产，不能只搬权重并套旧teleop norm。训练数据路径仍记录A100位置；做5090离线数据评价时另提供当地数据路径，不覆盖原训练配置。迁移校验通过不代表新三组已在5090完整加载或实机闭环验收。

## 8. 与旧teleop设置的证据边界

可查的旧模板为 `src/openpi/training/training_config_template/ego/egodex_pi05_teleop_20260916_resume_30k.yaml`：单一`egodex_teleop_20260916_lerobot21/train`来源、资产`egodex_teleop_offset25_train`、全局mask_wrist_images=true；名义初始loader为pi05_base的PartialCheckpointWeightLoader，跳过不匹配action/state投影；resume=true、总步数30000，但LR decay_steps仍10000、warmup1000。它与本轮双来源相机mask、100h19999完整初始化、新optimizer、共享1:1 norm、覆盖式配额和按总步数缩放LR的设置不同。

该文件是恢复模板，不是某次实际run的完整证据；不能据模板认定旧run实际从base重置、训练到了30k、用了哪个恢复checkpoint或有多少真机数据。旧run resolved配置、恢复状态与统一评价未在本次核实，不将它混作本轮三组的公平基线。

## 9. 下一步评价与记录入口

1. 固定一份Ego val样本清单（已有545,248帧／3278 episode）和真正独立的真机数据；当前真机只一段训练episode，随机邻帧留出不能当独立泛化验证。
2. 在同一评价协议下比较19999与三组最终模型：固定样本、噪声/seed、推理步数、相机mask、动作语义；按域分别统计flow loss和反归一化的左右臂/左右手关节MAE，记录样本量。19999使用它自己的绑定norm，三组使用各自checkpoint中的共享norm。
3. 核验5090新模型完整加载和有限16×56输出，再设计相同初始物体分布、成功定义、试次数、策略30Hz/chunk16及干预统计的实机比较；这些结果目前未知。不要用单次训练尾loss决定扩轮或胜者。
4. 对齐本地metrics与SwanLab尾部/全记录；最终checkpoint存在、模型加载成功、日志上传完整、离线泛化、实机成功分别记录。

文档体系：[README](../README.md)是入口，[TODO](../POSTTRAIN_TODO.md)是当前清单，[准备与执行记录入口](../README.md)保存日期证据，本COMPARISON维护实验设计/结果，并随BrainCo-IL仓库版本管理。[完成证据](completion-20261019.json)、[本次resolved配置/曲线统计/源码SHA](documentation-evidence-20261019.json)、[迁移校验](transfer-verification-20261019.json)是同目录证据附件，与本文一并随仓库保存。

<details>
<summary>历史记录：10月17日排队/运行中快照与10月19日初次完成核实（以下不是当前状态）</summary>

# Ego／真机 π0.5 LoRA：1:1、3:1、5:2 对比实验记录

## 2026-10-19 周末完成状态核实

核实时间：2026-10-19T02:11:28.687329+00:00。三组均完成，流水线退出，all_training_complete.json存在。以下为实际完成状态，替代旧快照中的排队/运行中描述；旧快照保留为历史。

| 组别 | 完成更新数 | 最终checkpoint | deploy_ready时间（UTC） | 日志入口 |
|---|---:|---|---|---|
| posttrain_e1_r1 | 6,908 | `/mnt/data_nas/ruibin/BrainCo-IL/checkpoints/egodex_pi05_posttrain_20260929/posttrain_e1_r1/6907` | 2026-10-15T19:36:50.458158+00:00 | [SwanLab](https://swanlab.cn/@laryybb626/imitation/runs/6hayni72) |
| posttrain_e3_r1 | 13,816 | `/mnt/data_nas/ruibin/BrainCo-IL/checkpoints/egodex_pi05_posttrain_20260929/posttrain_e3_r1/13815` | 2026-10-17T05:02:34.507512+00:00 | [SwanLab](https://swanlab.cn/@laryybb626/imitation/runs/uy1203p2) |
| posttrain_e5_r2 | 12,089 | `/mnt/data_nas/ruibin/BrainCo-IL/checkpoints/egodex_pi05_posttrain_20260929/posttrain_e5_r2/12088` | 2026-10-17T10:30:14.827868+00:00 | [SwanLab](https://swanlab.cn/@laryybb626/imitation/runs/vrd7jl5u) |

5:2首条真实训练指标在2026-10-17 05:09:07UTC，最终deploy_ready于10:30:14UTC落盘；原队列实际接续成功，非按预计时间推断。worker8输入profile已写入run。完成步数依据最终0起始检查点编号、流水线返回与all_training_complete；本次未重新加载optimizer。指标每10步记录，所以最后一条metrics的6900/13810/12080不是各组总更新数。三组固定验证/实机质量结果本次未核实，不以训练loss判胜负。

WAN本轮新增5000步也完成，2026-10-18 02:38:37UTC结束。第5000步训练五条均值26.1188dB（基线12.1405dB），魔方留出10.7810dB（基线10.3636dB）；倒水24.2498dB，故未满足五条全部≥25dB。训练拟合明显改善，泛化尚未证明。最终六条生成/对照视频均在NAS eval-005000目录，当前本地视频仍为第2000步，不能误当最终可视结果。本次未搬运视频。

证据：[completion-20261019.json](completion-20261019.json)。下一步先查看WAN最终整段动作/物体/连续性，再评估条件语义与独立数据泛化；π0.5三组采用统一固定验证和相同实机口径比较，保留相同真机暴露但不同总步数/计算量的可比性限制。本次未启动新训练或评价。


更新快照：**2026-10-17T03:42:46.363128+00:00**。本记录区分已发生状态与计划值；不会自动随后台训练刷新。用户2026-10-17已明确要求开始5:2并做好对比记录。

## 执行与接续状态

使用既有GPU5串行流水线PID1072324（快照存活），顺序`posttrain_e1_r1 → posttrain_e3_r1 → posttrain_e5_r2`。3:1尚未结束；5:2已经在同一进程的configs列表中排队。前组`train.main`返回并通过最终`deploy_ready.json`存在断言后，才reload训练入口并进入下一组。不停止当前训练，不切卡，不另起竞争作业。

**5:2当前没有run目录、SwanLab ID或真实训练步数，不能称已开训。** 此次已核实自动接续条件和输入配置的准入哈希；未再次运行shell。实际开训验收还需看到`training_status`切换、该组独立目录/ID/input_pipeline.json出现，并连续看到至少两条有限loss/梯度、非零参数更新和递增step。尚未观察到，留作明确待验收项。

输入优化准入已核实：规范化配置SHA256`4e7844567a700efcd72a7e6dfc9a04ececc3ed085bda43bc287b309fc3496993`与profile一致，五份源码SHA及CPU对照证据SHA均匹配。此配置指纹是`config_io.to_yaml`序列化哈希，不能与原始YAML字节哈希混比。验证仅CPU加载/序列化配置，没有创建模型或数据loader。

## 对比设计与计划配额

| 比例（Ego:真机）/run | 计划更新数 | Ego样本起点 | 真机样本起点 | warmup步数 | 实际状态 |
|---|---:|---:|---:|---:|---|
| 1:1 / `posttrain_e1_r1` | 6,908 | 55,264 | 55,264 | 691 | 训练完成；固定验证/实机比较待做 |
| 3:1 / `posttrain_e3_r1` | 13,816 | 165,792 | 55,264 | 1,382 | 运行中 |
| 5:2 / `posttrain_e5_r2` | 12,089 | 138,160 | 55,264 | 1,209 | 已由原流水线排队；尚未开始 |

配比表示**完整运行累计抽取样本起点数之比**，不是视频时长、数据集大小、loss权重或每个batch固定整数比例。实际`coverage`采样器以真机全部55,243个唯一起点为一轮，向32取整为55,264（补21个起点），按比例计算Ego配额。batch16；每步用累计配额的floor差分确定来源数量，再以seed和step打乱，因此5:2每batch真机4或5、Ego12或11。YAML中的两个`weight:0.5`不代表在coverage模式下采用1:1随机混合。

Ego seed42无放回排列，各比例共享同一排列前缀；真机seed43打乱并补采。coverage_seed42、训练seed42；按optimizer step确定batch，worker预取不改变恢复位置。每个起点形成horizon16，邻近起点动作范围重叠，起点暴露量不能误称独立动作或独立视频数量。

**可比性限定：这是相同真机暴露量的比较，并非相同更新步数、Ego暴露量、总样本数或计算预算。** 三组总样本分别110,528／221,056／193,424；3:1相较1:1更新数翻倍。Cosine调度按各组步数缩放，warmup亦不同；不能把结果差异全归因于比例。5:2另用经逐字段/采样及恢复一致性CPU对照通过的输入优化（worker8/prefetch4、decoder缓存16、FFmpeg1线程），另两组worker2；因此吞吐/墙钟也不能直接作为纯比例效应。需要等实际5:2运行测总step耗时，不把CPU吞吐4.84–5.85倍写成GPU总加速。

## 共享数据与版本

- ego_train_info：`/mnt/data_nas/ruibin/dataset/egodex_100h_offset0_lerobot21/train/meta/info.json`；SHA256 `295fb64bdf581d75dedba1528aa39d0cdc63af396e602df62ec5ed7412d5e97c`；元数据 {"total_frames": 10809850, "total_episodes": 64502, "fps": 30, "codebase_version": "v2.1"}。
- real_info：`/mnt/data_nas/ruibin/dataset/revotron_item_sorting_reviewed_v1/meta/info.json`；SHA256 `1fc81cb32d03d0375bdc53e3c83ef9b8278f93c08f65e7d033d7ad2dba164ea0`；元数据 {"total_frames": 55243, "total_episodes": 1, "fps": 30, "codebase_version": "v2.1"}。
- shared_norm：`/mnt/data_nas/ruibin/BrainCo-IL/assets/posttrain_shared_20260929/egodex_real_reference_1to1/norm_stats.json`；SHA256 `94398dac18e6717128104ff0d0cd5a15665c5fb451ae876c9c863f366a589eeb`；元数据 {}。
- shared_norm_sampling：`/mnt/data_nas/ruibin/BrainCo-IL/assets/posttrain_shared_20260929/egodex_real_reference_1to1/sampling_manifest.json`；SHA256 `1d2395231978cabdcb866a763a5a00bc1c6e70e1a0f76ebc016a64b7084ed799`；元数据 {"seed": 42}。

Ego为`egodex_100h_offset0_lerobot21/train`（10,809,850帧，64,502 episode，30fps），不是把100小时全部过一遍。真机为`revotron_item_sorting_reviewed_v1`，1 episode/55,243帧/约30.7分钟，全段保留、任务文本修正为`put bread into left basket and put fruit into right basket`。原视频和Parquet未改；用户明确全段保留不等于逐帧行为审核。共享norm来自固定1:1参考样本，不随三组比例重算；全数据哈希清单沿用原数据审核，本次仅重读元数据和norm指纹。

相机：Ego腕图mask，真机head/双腕有效。双臂delta、双手absolute，70D真机含EEF字段转56D，action_horizon16、30Hz。真机时间戳`tolerance_s=0.0002`，Ego默认；不改变源时间戳。没有用WAN生成腕图替代真实输入。

## 初始化、超参与硬件

三组均重新加载同一原100h检查点`/mnt/data_nas/ruibin/BrainCo-IL/checkpoints/egodex_pi05_100h_offset0_20260923/offset0_bs16_steps20000_v1/19999/params`，新optimizer，互不续用上一比例训练权重；不是从WAN检查点初始化。

- π0.5：`gemma_2b_lora`＋`gemma_300m_lora`，`freeze_strategy=lora_and_action_interface`；action_dim56、horizon16、max_token_len256、discrete_state_input=true、bfloat16。底层LoRA rank/alpha未在本次展开核实，不另填猜测值。
- AdamW：b1=0.9、b2=0.95、eps=1e-8、weight_decay=1e-10、clip_grad_norm=1；EMA关闭。Cosine peak_lr5e-5→5e-6，decay_steps等于各组完整步数，warmup见表。
- batch16、FSDP设备1、seed42，resume=false、overwrite=false、最终检查点开启。save_interval1000、keep_period2000；实际留存目录以快照为准。
- 硬件：`brainco-a100-6007` GPU **5**，A100-SXM4-80GB，UUID `GPU-e2b51ad6-ecf5-ffea-e58d-8a44f85c2da3`。GPU2是独立WAN任务，不操作；共享机器负载影响墙钟。
- 软件实查：{"jax": "0.5.3", "jaxlib": "0.5.3", "torch": "2.7.1", "swanlab": "0.8.4", "orbax-checkpoint": "0.11.13", "lerobot": "0.1.0"}。
- 1:1 runtime记录代码commit`56152137f8586621b4ff87ff9f5059f6ce19cc2e`；3:1由同一旧入口运行。5:2输入优化发布记录为`ea835f9`，实际五份源码hash此次匹配profile。远端Git HEAD/dirty查询被Git目录所有权保护阻止，未改safe.directory或持久权限，不能据历史发布信息声称当前整仓干净；精确源码指纹见snapshot。Mac本次仅补记录，无代码发布。

## 三组实际状态

### 1:1：posttrain_e1_r1

- 快照状态：训练完成；固定验证/实机比较待做；step=6900，loss=0.068655，ego=0.101495，real=0.035815。
- 首条已记录更新（非进程启动时刻）：2026-10-15T08:29:37.126738+00:00。
- 最新指标时间：2026-10-15T19:35:56.531769+00:00；指标每10步记录，不能把最后记录步数当作最终更新总数。
- 完成凭据时间：2026-10-15T19:36:50.458158+00:00。1:1最终目录6907表示0起始编号，共6908次更新。
- 累计已记录域暴露量：Ego 55208，真机 55208；以上为该指标时刻实际值，不是完整计划配额。
- Run目录：`/mnt/data_nas/ruibin/BrainCo-IL/checkpoints/egodex_pi05_posttrain_20260929/posttrain_e1_r1`。日志：该目录`metrics.jsonl`；流水线`/mnt/data_nas/ruibin/ego-posttrain-prep/20261015/pipeline_v3.log`。
- 配置证据：`/mnt/data_nas/ruibin/BrainCo-IL/checkpoints/egodex_pi05_posttrain_20260929/posttrain_e1_r1/train_config.yaml`，SHA256 `0d592a01417550ac04a133501fd0dc4023d5901a23d2e2d576b7c44c8ba28d10`；保存配置全文见[snapshot.json](snapshot.json)。
- 最近检查点：`/mnt/data_nas/ruibin/BrainCo-IL/checkpoints/egodex_pi05_posttrain_20260929/posttrain_e1_r1/6907`；最终计划目录：`/mnt/data_nas/ruibin/BrainCo-IL/checkpoints/egodex_pi05_posttrain_20260929/posttrain_e1_r1/6907`，存在和deploy_ready状态见快照。
- [SwanLab 6hayni72](https://swanlab.cn/@laryybb626/imitation/runs/6hayni72)。本次未查询云端API确认上传新鲜度，run ID来自既有文件；5:2不得复用其他组run链接。

### 3:1：posttrain_e3_r1

- 快照状态：运行中；step=13290，loss=0.072607，ego=0.087671，real=0.027415。
- 首条已记录更新（非进程启动时刻）：2026-10-15T19:39:37.220437+00:00。
- 最新指标时间：2026-10-17T03:41:47.926683+00:00；指标每10步记录，不能把最后记录步数当作最终更新总数。
- 完成凭据时间：尚无最终deploy_ready，结束时间未知。1:1最终目录6907表示0起始编号，共6908次更新。
- 累计已记录域暴露量：Ego 159492，真机 53164；以上为该指标时刻实际值，不是完整计划配额。
- Run目录：`/mnt/data_nas/ruibin/BrainCo-IL/checkpoints/egodex_pi05_posttrain_20260929/posttrain_e3_r1`。日志：该目录`metrics.jsonl`；流水线`/mnt/data_nas/ruibin/ego-posttrain-prep/20261015/pipeline_v3.log`。
- 配置证据：`/mnt/data_nas/ruibin/BrainCo-IL/checkpoints/egodex_pi05_posttrain_20260929/posttrain_e3_r1/train_config.yaml`，SHA256 `7f7b7c301676bc8ad0cf7e992234c2d3a7c353158ecf370a016c3839648d6592`；保存配置全文见[snapshot.json](snapshot.json)。
- 最近检查点：`/mnt/data_nas/ruibin/BrainCo-IL/checkpoints/egodex_pi05_posttrain_20260929/posttrain_e3_r1/13000`；最终计划目录：`/mnt/data_nas/ruibin/BrainCo-IL/checkpoints/egodex_pi05_posttrain_20260929/posttrain_e3_r1/13815`，存在和deploy_ready状态见快照。
- [SwanLab uy1203p2](https://swanlab.cn/@laryybb626/imitation/runs/uy1203p2)。本次未查询云端API确认上传新鲜度，run ID来自既有文件；5:2不得复用其他组run链接。

### 5:2：posttrain_e5_r2

- 快照状态：已由原流水线排队；尚未开始；无实际指标。
- 首条已记录更新（非进程启动时刻）：未知/尚未开始。
- 最新指标时间：未知/尚未开始；指标每10步记录，不能把最后记录步数当作最终更新总数。
- 完成凭据时间：尚无最终deploy_ready，结束时间未知。1:1最终目录6907表示0起始编号，共6908次更新。
- 累计已记录域暴露量：Ego 尚无，真机 尚无；以上为该指标时刻实际值，不是完整计划配额。
- Run目录：`/mnt/data_nas/ruibin/BrainCo-IL/checkpoints/egodex_pi05_posttrain_20260929/posttrain_e5_r2`。日志：该目录`metrics.jsonl`；流水线`/mnt/data_nas/ruibin/ego-posttrain-prep/20261015/pipeline_v3.log`。
- 配置证据：`/mnt/data_nas/ruibin/BrainCo-IL/src/openpi/training/training_config_template/ego/posttrain_e5_r2.yaml`，SHA256 `21da0d29c657038b9270f58d927c41a6a9e88461b9181c72b8e064982faad6f7`；保存配置全文见[snapshot.json](snapshot.json)。
- 最近检查点：`尚无`；最终计划目录：`/mnt/data_nas/ruibin/BrainCo-IL/checkpoints/egodex_pi05_posttrain_20260929/posttrain_e5_r2/12088`，存在和deploy_ready状态见快照。
- 未创建，待该组开始取得实际run ID。本次未查询云端API确认上传新鲜度，run ID来自既有文件；5:2不得复用其他组run链接。


## 统一评价口径与待执行项

训练loss和各来源loss仅用于训练健康及来源诊断。不同配比的总loss权重、噪声、动作与样本组成不同，不能直接以最终训练loss排列策略质量。1:1结束、deploy_ready存在也不代表实机效果已验收。

后续统一使用：同一独立Ego val（已有545,248帧/3,278 episode）和同一独立真机任务评估；相同固定样本列表/seed、相机mask、共享norm、动作坐标与horizon，对每域分别报告样本数与误差，并明确loss聚合和匹配训练暴露量的比较限制。当前真机训练只有一个全段episode，**尚无确认独立真机留出集**，不能将同一片段邻帧称独立泛化验证。

实机评价需统一初始物体分布/任务、策略频率30Hz、chunk16、相机配置和成功定义，比较成功率、完成时间、人工干预/失败类别并保留试次数；所有这些结果均待评估，当前未派发机器人指令，也未启动额外离线评估。固定评价清单、独立真机数据、具体试次数和安全执行安排尚待确定。

## 日志与证据入口

- [机器可读快照与完整配置](snapshot.json)
- 原准备报告：[准备与执行记录入口](../README.md)
- [WAN当前日志接入核验](wan_logging_check.json)：2026-10-17 03:42UTC GPU2运行到2157步，**当前完整视频run未接SwanLab**。fixed-window旧中断项不代表本run中断；没有本run的云端链接，未新建外部日志上传。
- 原混训会话：`codex://threads/01a0ebc1-580c-7f22-a950-19dd5413e65d`；输入优化会话`01a14405-8b68-74b3-823f-5cf8e94eb62d`。

本次已完成：现有队列/启动门核实、数据配置和源码证据整理、报告入口更新。未完成：5:2实际开训与步数推进验收、三组统一离线/实机质量比较。当前3:1仍运行，所以不人为抢占其GPU5。


## 2026-10-17 视频交付与接续补记

更新：2026-10-17T04:07:34.792962+00:00。WAN第2000步六条完整对照的640×160低码率浏览预览全部同步，319/361/470/381/356/310帧逐条完整解码通过；指标沿用原始评估，不以低码率预览重算。见[本地视频报告](/Users/larrybb/Documents/ChatGPT/Ego data/wrist-view-sim/experiments/runs/20261016-wan-fullvideo-gpu2/report.html)与[交付状态](delivery_status.json)。高码率原始六条尚未全部同步，不声称全部原件已交付；无Library上传。

04:04UTC只读核实3:1到13430/13816，loss0.0683755，进程存活；5:2目录尚未创建，原流水线排队不变。按近190步实际墙钟9.24秒/步，估算05:03UTC（北京时间13:03）结束3:1训练，另需最终保存/初始化；此为估算，不是5:2开训承诺。未等待或启动额外训练。


</details>
