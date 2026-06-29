# BrainCo 双臂灵巧手分拣：模型结构诊断与改进计划

更新日期：2026-06-22

## 1. 结论边界

本文把依据分为三档：

- **代码事实**：可以直接从当前 `brainco-openpi` 实现中确认。
- **文献证据**：论文直接验证了某种机制，但实验平台和 BrainCo 不同。
- **工程假设**：机制上合理，但是否能提高本项目成功率，必须通过消融实验确认。

公开的 `wuji-openpi` 仓库目前只有演示视频和训练/部署代码，没有公开分拣任务的重复试验次数、成功率、置信区间或与其他策略的对照。因此不能用公开视频推导它的任务成功率。仓库中的 `examples/open_loop_eval.py` 评估 MSE、MAE 和轨迹抖动，这些指标不能替代实机任务成功率。

## 2. 当前模型结构

当前 Revo3 配置位于 `src/openpi/training/config.py`：

```python
Pi0Config(
    pi05=True,
    action_dim=56,
    action_horizon=100,
    max_token_len=256,
)
```

56D 顺序为：

```text
left_arm(7) + left_hand(21) + right_arm(7) + right_hand(21)
```

模型主体是 π0.5 风格架构：

- SigLIP 图像编码器；
- PaliGemma / Gemma 2B 视觉语言骨干；
- Gemma 300M action expert；
- flow matching 连续动作生成；
- 三路当前时刻 RGB 图像；
- 一次预测 `100 × 56` 的动作块。

需要准确说明的是：当前模型并不是“只有一个 Linear 就直接输出动作”。每个时刻的 56D noisy action 先通过 `action_in_proj: Linear(56, hidden)`，然后由 action-expert Transformer 在动作时间序列和视觉语言条件上继续建模，最后通过 `action_out_proj: Linear(hidden, 56)` 输出。真正缺少的是**显式的左臂、左手、右臂、右手结构 token 或独立协调 head**。

## 3. 六个潜在瓶颈及其证据强度

### 3.1 56D 动作没有显式区分四个身体分组

#### 代码事实

`src/openpi/models/pi0.py` 中：

```python
self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width)
self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim)
```

模型知道 56 个数值，但没有显式知道：

```text
0:7   是左臂
7:28  是左手
28:35 是右臂
35:56 是右手
```

这种语义只能由训练数据隐式学出。

#### 文献证据

- **InterACT** 专门研究双臂之间的 inter-dependency。它使用分层注意力、Multi-arm Decoder 和同步模块，让两条手臂分别产生动作、同时交换中间信息；论文的实机/仿真结果和消融实验支持显式双臂依赖建模的价值。
- **TwinVLA** 将两个预训练单臂 VLA 组合成协调双臂策略，并报告其数据效率和表现优于相近规模的 monolithic RDT-1B，说明模块化双臂结构是一个有实证依据的方向。
- **RDT-1B** 强调双臂动作分布的多模态性，以及保留动作物理含义的 unified action space。但 RDT-1B **不能单独证明**“四个独立输出 head 一定优于一个 56D head”。

#### 能否下结论

只能说这是一个**有文献动机的工程假设**，不能说已经证明是 BrainCo 当前性能瓶颈。需要比较：

1. 原始 flat 56D head；
2. grouped state token，但保留 flat output；
3. grouped state token + 四个输出 head；
4. InterACT 风格的左右臂同步 decoder。


### 3.2 π0.5 将 proprioception 离散为 256 档文本 token

#### 代码事实

`src/openpi/models/tokenizer.py` 中，π0.5 把归一化状态按 `[-1, 1]` 划为 256 档，再把整数写入 prompt：

```python
discretized_state = np.digitize(
    state,
    bins=np.linspace(-1, 1, 256 + 1)[:-1],
) - 1

full_prompt = f"Task: {prompt}, State: {state_str};\nAction: "
```

这意味着当前 π0.5 没有像 π0 那样通过连续 `state_proj` 将关节状态直接送入 action expert。

#### 文献证据

OpenVLA-OFT 使用两层 MLP 将连续 proprioceptive state 投影到语言 embedding 空间。其“腕部图像 + proprioception”组合输入使 LIBERO 平均成功率进一步提高 5.2 个百分点。需要注意：这项实验把腕部图像和 proprioception 一起加入，并没有单独证明连续 proprioception 相对 π0.5 离散状态的收益。

π0.5 本身采用离散状态是经过大规模训练验证的设计。因此“256 档一定不够精确”并没有直接证据；实际误差还取决于 norm stats、各关节量程和控制器容差。

#### 改进假设

保留离散 state token 以兼容 π0.5 预训练，同时增加并行连续状态分支：

```text
56D normalized state
    ├── π0.5 discrete state tokens
    └── continuous grouped state encoder
            ├── left-arm token
            ├── left-hand token
            ├── right-arm token
            └── right-hand token
```

连续分支通过 cross-attention 或 AdaLN/FiLM 条件注入 action expert。是否有效必须与只用离散状态的 baseline 对照。

### 3.3 视觉输入没有显式短时历史

#### 代码事实

当前 observation spec 中每台相机都是：

```text
[batch, 224, 224, 3]
```

没有 `[batch, time, height, width, channels]` 维度，也没有 recurrent memory。模型会生成未来动作序列，但感知输入本身是当前帧。

#### 文献证据

- ACT 讨论了非马尔可夫示教和误差累积，并使用 action chunking 与 overlapping temporal ensemble 缩短有效任务时域、提高闭环平滑性。
- Diffusion Policy 使用 observation/action horizon 与 receding-horizon control，并对 action horizon 和延迟鲁棒性进行了消融。

这些工作支持“时序上下文和闭环重规划可能有帮助”，但不证明增加视觉历史在所有任务上都会提升。OpenVLA-OFT 的强结果本身仍使用单步输入。

#### 何时值得加入

只有当失败与以下现象相关时优先加入：

- 遮挡后忘记目标；
- 无法从单帧判断运动方向或物体是否滑动；
- 抓取/抬升阶段发生 phase ambiguity；
- 相似当前图像对应不同历史状态。

推荐先比较 1、2、4 帧历史，而不是直接引入很长视频模型。

### 3.4 RGB 没有显式 3D 几何表示

#### 代码事实

当前输入只有三路 RGB，没有 depth、点云、相机外参 token 或显式 3D position encoding。

#### 文献证据

- **3D Diffusion Policy（DP3）** 使用稀疏点云编码。论文报告在 72 个仿真任务中相对基线提升 24.2%，在四个真实任务、每项 40 条示教时达到 85% 成功率。
- **SpatialVLA** 使用 Ego3D Position Encoding 将深度构造的 3D 坐标注入视觉 token，并用 spatial action representation 提高空间泛化和跨机器人适配能力。

这是六项中证据相对较强的一项。分拣涉及物体高度、抓取点、箱体边缘、堆叠遮挡和相机视角变化，显式 3D 表示与任务需求直接相关。

#### 改进方案

```text
RGB ───────────────→ SigLIP semantic tokens
Depth / point cloud → compact 3D encoder
camera calibration  → camera-pose tokens
                         ↓
                2D/3D cross-attention fusion
```

应分别验证真实深度、单目估深和纯 RGB baseline，避免把深度噪声误认为结构收益。

### 3.5 100 步 action horizon 可能降低接触阶段反应速度

#### 代码事实

当前：

```text
action_horizon = 100
control_hz = 30 Hz
```

模型预测范围约为：

```text
100 / 30 ≈ 3.33 秒
```

RTG 在默认 `trigger_fraction=0.5` 时会约 50 步后触发下一次推理，所以实际并非完整执行 3.33 秒才观察环境，但一次预测仍覆盖较长未来。

#### 文献证据

- ACT 表明 action chunking 能显著缩短有效任务时域，但也明确讨论了 chunk 越长、反应越开环的权衡；它通过每步查询和 temporal ensemble 融合重叠 action chunk。
- Diffusion Policy 使用 receding-horizon control，并对 action horizon 与推理延迟作了消融。
- OpenVLA-OFT 在真实 ALOHA 双臂实验中使用 25-step chunk、25 Hz，即约 1 秒动作范围。

因此，文献支持的是“需要在时间一致性和闭环反应之间做消融”，不是“100 步必然错误”。ACT 的部分实验本身也使用 100-step chunk，但搭配了不同的控制频率和 temporal ensemble。

#### 推荐实验

保持训练数据和总计算量一致，比较：

| 预测长度 | 执行长度 | 目的 |
|---:|---:|---|
| 100 | RTG 50 | 当前 baseline |
| 50 | 15 | 中等范围闭环 |
| 30 | 10 | 推荐起点 |
| 20 | 5 | 强闭环、接触优先 |

### 3.6 没有显式目标、箱体、阶段和双臂分工输出

#### 代码事实

当前模型直接执行：

```text
images + language + state → 100 × 56 actions
```

没有额外监督：

```text
target_object
destination_bin
task_phase
active_arm
grasp_state
```

#### 文献证据

π0.5 的主要设计之一是混合训练 object detection、semantic subtask prediction、语言和低层连续动作。推理时先产生高层 subtask，再以 subtask 为条件产生动作。论文实验认为这些异构数据和高层预测对开放世界泛化至关重要。

这能支撑“显式语义中间变量值得尝试”，但不能证明手工定义的分拣 phase head 必然优于原始端到端 π0.5。

#### 改进方案

增加辅助 head 或离散 token：

```text
target object → destination bin → phase → arm assignment
                                      ↓
                           low-level action expert
```

阶段可定义为：

```text
locate → approach → grasp → lift → transport → place → release
```

需要同时比较 predicted phase、ground-truth phase 和无 phase baseline，避免“错误高层预测级联破坏低层控制”。

## 4. 推荐的模型结构改进路线

### Phase 0：建立可复现实机 baseline

在改结构前，至少记录：

- 总成功率及 95% 置信区间；
- locate、grasp、lift、transport、place 分阶段成功率；
- 抓错物体率、放错箱率、掉落率、碰撞率；
- 平均周期时间、推理延迟、人工干预次数；
- seen/unseen 物体、光照变化、箱位变化和遮挡条件。

建议每个核心条件至少 30 次试验。公开视频中的一次成功不能作为 baseline。

### Phase 1：低成本 action-decoder 消融

保持视觉骨干、数据和训练预算一致：

| 实验 | Action head | K | 目的 |
|---|---|---:|---|
| B0 | 当前 flow matching | 100 | 原始 baseline |
| B1 | 当前 flow matching | 30 | 验证 horizon |
| B2 | L1 parallel chunk | 30 | 验证简单确定性 head |
| B3 | L1 parallel chunk + temporal ensemble | 30 | 验证重叠闭环融合 |

若 B2/B3 已明显优于 B0，就没有必要立即扩大 diffusion expert。

### Phase 2：连续且结构化的 proprioception

按顺序加入：

1. 一个连续 56D state projector；
2. 四组 state token；
3. side/type/joint embedding；
4. 四组输出 head；
5. InterACT 风格左右臂同步模块。

每一步单独消融。不要一次加入全部模块，否则无法知道增益来源。

### Phase 3：空间与时序感知

先加 3D，再加历史：

1. RGB baseline；
2. RGB + depth；
3. RGB + compact point cloud token；
4. RGB/3D + camera-pose embedding；
5. 最后比较 1、2、4 帧历史。

优先级理由：分拣的目标位置、箱体几何和抓取高度与 3D 直接相关；视觉历史只有在部分可观测和 phase ambiguity 明显时才更重要。

### Phase 4：分层语义辅助任务

增加：

```text
object grounding head
destination-bin head
phase head
arm-assignment head
```

总损失示例：

```math
L = L_{action}
  + \lambda_{obj} L_{obj}
  + \lambda_{bin} L_{bin}
  + \lambda_{phase} L_{phase}
  + \lambda_{arm} L_{arm}
```

先作为 auxiliary loss 使用，不要一开始强制动作完全依赖 predicted phase，以降低级联错误风险。

### Phase 5：只有确认动作多模态问题后，升级生成式 expert

触发条件：

- L1 head 明显产生不同示教策略的平均动作；
- 同一观察存在多个有效抓取方向；
- 左右手选择具有多模态性；
- 简单 flow head 容量不足或跨任务干扰严重。

候选结构：

- RDT 风格 bimanual Diffusion Transformer；
- DexVLA 风格 plug-in diffusion expert；
- lightweight mixture-of-experts action decoder。

推荐训练顺序参考 DexVLA：

1. 独立预训练 action expert；
2. 对齐 VLM 和 embodiment；
3. 对目标分拣任务后训练。

不建议在只有少量单任务数据时直接训练 1B action expert。

## 5. 根据失败类型选择结构

| 实机失败 | 首选结构改进 | 不应优先做什么 |
|---|---|---|
| 抓错物体、放错箱 | object/bin/phase 辅助 head、语言 FiLM | 盲目扩大 action expert |
| 接近位置偏、抓空 | 3D/depth spatial encoder | 只增加训练步数 |
| 抓住后滑落、手指不稳 | 连续 proprioception、短 chunk、可选触觉 | 只改 VLM 规模 |
| 左右臂相互干扰 | grouped token、multi-arm decoder、同步模块 | 单纯提高 action_dim |
| 动作在多种抓法之间取平均 | flow/diffusion expert | 继续使用确定性 L1 head |
| 接触后反应慢 | 短 chunk、receding horizon、temporal ensemble | 继续执行长开环 chunk |
| 遮挡后忘记任务阶段 | 短视觉历史、phase token | 只做轨迹平滑 |

## 6. 推荐的 BrainCo-Sort V1

```text
head RGB + wrist RGBs + optional depth
                  ↓
       2D/3D visual fusion
                  ↓
     PaliGemma semantic backbone
                  ↓
 discrete state + continuous grouped state tokens
                  ↓
 target / bin / phase auxiliary tokens
                  ↓
 bimanual coordination action decoder
                  ↓
 left arm | left hand | right arm | right hand
                  ↓
       20–30 step continuous chunk
```

第一轮同时保留两个 action head 做公平比较：

- deterministic L1 chunk head；
- 原始 flow-matching head。

以实机成功率和分阶段成功率选择，不以训练 loss 单独决定。

## 7. 参考文献

1. Black et al., **π0: A Vision-Language-Action Flow Model for General Robot Control**, 2024. <https://arxiv.org/abs/2410.24164>
2. Black et al., **π0.5: a Vision-Language-Action Model with Open-World Generalization**, 2025. <https://arxiv.org/abs/2504.16054>
3. Zhao et al., **Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware (ACT)**, 2023. <https://arxiv.org/abs/2304.13705>
4. Chi et al., **Diffusion Policy: Visuomotor Policy Learning via Action Diffusion**, 2023. <https://arxiv.org/abs/2303.04137>
5. Kim et al., **Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success (OpenVLA-OFT)**, 2025. <https://arxiv.org/abs/2502.19645>
6. Liu et al., **RDT-1B: a Diffusion Foundation Model for Bimanual Manipulation**, 2024. <https://arxiv.org/abs/2410.07864>
7. Wen et al., **DexVLA: Vision-Language Model with Plug-In Diffusion Expert for General Robot Control**, 2025. <https://arxiv.org/abs/2502.05855>
8. Ze et al., **3D Diffusion Policy: Generalizable Visuomotor Policy Learning via Simple 3D Representations**, 2024. <https://arxiv.org/abs/2403.03954>
9. Qu et al., **SpatialVLA: Exploring Spatial Representations for Visual-Language-Action Model**, 2025. <https://arxiv.org/abs/2501.15830>
10. Lee et al., **InterACT: Inter-dependency Aware Action Chunking with Hierarchical Attention Transformers for Bimanual Manipulation**, 2024. <https://arxiv.org/abs/2409.07914>
11. Im et al., **TwinVLA: Data-Efficient Bimanual Manipulation with Twin Single-Arm Vision-Language-Action Models**, 2025. <https://arxiv.org/abs/2511.05275>

## 8. 阅读这些结果时的注意事项

不同论文的机器人、动作空间、控制频率、示教数量、任务难度和成功判据均不同。ACT 的 80–90%、DP3 的 85%、OpenVLA-OFT 的 97.1% 等数字只能证明其机制在对应实验中有效，不能横向比较，也不能作为 BrainCo 分拣的预期成功率。BrainCo 上的最终结论必须来自相同数据、相同训练预算和相同实机协议下的消融实验。
