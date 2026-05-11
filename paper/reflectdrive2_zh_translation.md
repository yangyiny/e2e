# ReflectDrive-2：面向离散扩散驾驶的强化学习对齐自编辑

作者：Huimin Wang、Yue Wang、Bihao Cui、Pengxiang Li、Ben Lu、Mingqian Wang、Tong Wang、Chuan Tang、Teng Zhang、Kun Zhan  
机构：LiAuto  
arXiv：2605.04647v1  
原文文件：`ReflectDrive-2.pdf`

## 摘要

本文提出 ReflectDrive-2，这是一个用于自动驾驶规划的掩码离散扩散规划器，并配有独立的动作专家分支。它把规划表示为离散轨迹 token，并通过并行掩码解码生成轨迹。离散 token 空间天然支持原地轨迹修订：AutoEdit 可以用同一个模型重写选中的 token，不需要额外的 refinement 网络。

为了训练这种能力，作者采用两阶段流程。第一阶段，沿着纵向进度和横向航向两个方向，对专家轨迹构造结构感知扰动，并监督模型恢复原始专家轨迹。第二阶段，用强化学习微调整个“决策、草拟、反思”的 rollout，把终端驾驶奖励分配给最终编辑后的轨迹，并通过完整 rollout 的转移传播策略梯度信用。完整 rollout 强化学习对耦合 drafting 和 editing 很关键：仅用监督训练时，推理阶段 AutoEdit 最多只提升 0.3 PDMS；加入强化学习后，AutoEdit 的增益提升到 1.9。

作者还共同设计了高效的反思式解码栈，用于 decision-draft-reflect 流程，包括共享前缀 KV 复用、Alternating Step Decode，以及融合的端上 unmasking。在 NA VSIM 上，ReflectDrive-2 使用纯视觉输入达到 91.0 PDMS；在 best-of-6 oracle 设置下达到 94.8 PDMS；在 NVIDIA Thor 上平均延迟为 31.8 ms。

## 1 引言

模仿学习驾驶策略的规划错误很少是随机的。它们通常集中在两个方向：纵向速度判断错误，例如过冲、前进不足、刹车过晚；横向航向漂移，例如偏离车道、转弯切角、驶出可行驶区域。这些方向也是专家示范模仿学习累积分布偏移的主要方向，因此一个能够按结构进行原地修正的规划表示，很适合这个问题。

传统模块化系统和端到端规划器通常一次性提交单条轨迹；自回归 VLA 规划器按序列逐 token 解码，如果要修订已生成 token，往往需要重新 rollout；连续扩散规划器可以并行生成，但反向过程对应的是高斯扰动，而不是训练后驾驶策略的结构性失败模式。掩码离散扩散则天然支持这类修订：可以将任意子集的轨迹 token 重新处理，并在其他 token 条件下由同一模型重写，而不需要辅助网络或独立推理模式。

但是，仅在训练好的 drafter 上加一个自编辑步骤收益很小。drafter 没有动机产生“容易被 editor 改好”的草稿，editor 也没有收到哪些 rewrite 会改善闭环驾驶表现的信号。仅监督训练时，自编辑能力虽然存在于权重中，但两个阶段是解耦的：drafter 优化自己的 token 级损失，editor 优化单独的纠错损失，两者都不知道对最终驾驶结果的影响。

完整 draft-and-edit rollout 上的强化学习填补了这个空隙。当单个终端奖励把策略梯度信用同时分配给 drafting 和 editing 转移时，这两个阶段被耦合起来。drafter 学会产生可修订草稿，也就是 post-edit 轨迹得分高于 pre-edit 的 token 分布；editor 学会把草稿推向闭环奖励，而不仅是降低 token 级不确定性。于是，自纠错不再是事后附加模块，而成为被优化策略 rollout 的一部分。

本文提出的系统称为 ReflectDrive-2，其自编辑机制称为 AutoEdit。ReflectDrive-2 的输入包括环视相机、路线/导航指令 token 和自车状态；输出是离散轨迹 token，其中最终 waypoint token 锚定行为假设，其余轨迹 token 表示 4 秒规划。每个 goal point 表示一个候选行为假设，如车道保持、让行、超车或变道；推理时从 goal posterior 中通过 top-k sampling 和 NMS 选择。AutoEdit 先用覆盖纵向和横向失败轴的结构感知扰动预训练，再通过联合 rollout 强化学习与 drafter 协同训练。

主要贡献如下：

1. **目标条件掩码扩散规划**：提出 ReflectDrive-2，将驾驶 VLA 表述为 decision-draft-reflect 流程。goal posterior 暴露行为级假设，掩码离散扩散为每个假设生成可编辑轨迹，AutoEdit 在同一 token 空间重写草稿。在 NA VSIM 上，纯视觉输入达到 91.0 PDMS，best-of-6 oracle 达到 94.8 PDMS。
2. **奖励耦合的 AutoEdit**：提出 AutoEdit，用匹配模仿学习常见失败轴的结构扰动训练，并在完整 draft-and-edit rollout 上用强化学习让 drafter 和 editor 共同适应奖励，大幅放大推理阶段 AutoEdit 的作用。
3. **高效反思式解码**：利用 decision-draft-reflect 结构设计共享前缀 KV cache、作为时间 AutoEdit 的 ASD，以及融合 CUDA unmasking，在 NVIDIA Thor 上实现 31.8 ms 平均延迟，规划质量几乎无损。

## 2 相关工作

### 2.1 端到端规划与 VLA 规划

端到端规划器直接从传感器映射到轨迹，避免模块间误差传播。SMART 将多智能体轨迹 token 化，用自回归 next-token prediction 建模。VLA 规划器继承语言先验，但通常逐 token 解码，延迟随轨迹长度增长；如果要纠错，往往需要第二次顺序 rollout。连续扩散规划器可以并行生成，但通常需要 20 步以上去噪，带引导的变体还会因逐步梯度传播增加成本。

ReflectDrive-2 用掩码离散扩散替代这两类范式：少量并行 unmasking 轮次即可生成完整轨迹，token 级编辑是原生能力，而不是第二阶段外接模块。这些基线通常不能自然地把原地编辑与同一个策略 rollout 和奖励信号耦合起来，而这正是本文方法的核心。

### 2.2 离散扩散与 token 空间编辑

离散扩散为类别状态空间提供了自然的生成框架。D3PM 将扩散模型扩展到离散变量；MaskGIT 表明 masked-token prediction 可以通过置信度 unmasking 支持并行生成。近期这一路线扩展到语言模型：LLaDA、Seed Diffusion、MDLM、SEDD、Block Diffusion、Fast-dLLM 等改进了离散扩散建模或推理效率。LLaDA 2.0/2.1 进一步扩展规模，并提出 Token-to-Token 编辑，即在解码时重新生成低置信度 token。

任意 token 子集都可被重写，这使离散扩散非常适合可编辑规划。但多数既有 token-editing 机制只是解码时启发式，或是独立训练的 refinement 阶段。比如 LLaDA 2.1 的 T2T 根据模型置信度修订 token，但没有显式针对下游控制的结构性错误训练。AutoEdit 不同：它用纵向进度错误和横向航向偏差等常见驾驶失败模式进行监督，使 editor 在训练时就见到推理时需要修正的错误类型。

DriveFine 是最接近的相关工作，它提出了 refinement-augmented masked-diffusion driving VLA，但其 refiner 与 drafter 分开训练和优化。ReflectDrive-2 则把 drafting 和 editing 当作一个组合 rollout：终端驾驶奖励分配给 post-edit 轨迹，策略梯度信用同时作用于两个阶段的 token 转移。

### 2.3 扩散策略的强化学习

DDPO 和 DPPO 将策略梯度用于连续扩散，把去噪视为多步 MDP。对于离散扩散，d1 使用 GRPO 风格强化学习但忽略多步结构，d2 通过 step-aware gradient 和 group-relative advantage 恢复多步结构，SPG 推导更紧的 ELBO/EUBO 边界。在驾驶中，HDP 和 DriveFine 将 RL 后训练用于扩散规划器。

这些方法大多优化单 pass rollout，即只优化 drafting 或只优化 refining。ReflectDrive-2 的 RL 目标用于组合 rollout，即 draft 到 AutoEdit，因此终端奖励联合分配给两个阶段。仅仅增加扩散步数并不会产生语义上独立的 edit operator，也不会让编辑阶段获得明确奖励信用；本文的组合 rollout 包含一个显式 reflection 阶段。

## 3 预备知识

### 3.1 问题设定

在时刻 t，自车接收观测 `o_t = (v_t, l_t, s_t)`，包括三个通道：

- 全景视觉 token：来自左前、正前、右前三个相机的两个时间帧；
- 导航指令通道：路线级命令和动作提示，例如保持车道、路口左转、直行，以语言 token 形式输入同一 backbone；
- 自车状态通道：速度、加速度、yaw rate 等运动学 token。

目标是生成未来轨迹 `tau = {(x_k, y_k)}_{k=1}^K`，满足安全、舒适、规则合规，并与导航指令一致。需要时，heading 由连续 waypoint 推导。

### 3.2 掩码离散扩散

作者将未来自车轨迹表示为 BEV 坐标 token 序列 `x_0`。前向过程以概率 t 独立地将 token 替换为 `[MASK]`，得到部分掩码序列 `x_t`。一个双向 Transformer `p_theta` 在多模态上下文 `c=(v_t,l_t,s_t)` 条件下预测原始 token。与以往只监督 masked positions 的 masked-diffusion language model 不同，本文监督所有位置：

```text
L_DLM(theta) = - E_{x0,t} [ 1/L * sum_i log p_theta(x0_i | x_t, c) ].
```

经验上，all-position objective 带来更稳定的优化和更连贯的草稿。推理时，从全掩码序列开始，用少量并行去噪步骤生成完整轨迹。

掩码扩散支持选择性再生成：任意编辑 mask 都可指定要重写的 token 子集。AutoEdit 继承这个接口，但将 editor 从解码启发式转为训练过的 operator，并通过共享 RL 奖励与 drafter 耦合。

### 3.3 高效推理的 KV cache

标准 masked diffusion 使用双向注意力，因此普通 KV cache 失效：因为 masked token 会不断变化，每个去噪步骤都需要重算 KV。Block Diffusion 通过分块使完成块可复用 cache，LLaDA 2.1 推广到 block-wise causal attention。本文对场景上下文 prompt 使用 causal attention，对轨迹 token 使用 block-wise attention，从而在保留轨迹块内部双向扩散能力的同时复用 prompt 的 KV。

### 3.4 强化学习微调

监督训练模仿数据分布，但不直接优化驾驶目标。作者把轨迹生成视为 MDP，并用强化学习对齐闭环奖励。目标为最大化 `E[R(tau)]`，用 G 条采样轨迹上的 group-relative advantage 和离散扩散策略梯度优化。关键是：在本文第 4.5 节中，总生成步数 `S = S_draft + S_edit`，因此同一个奖励同时给 drafting 和 AutoEdit 的 token 转移分配信用。

## 4 方法

### 4.1 ReflectDrive-2 总览

ReflectDrive-2 将自动驾驶规划统一为目标提议、掩码轨迹草拟和 token 空间轨迹修正。给定多模态上下文 `c=(v_t,l_t,s_t)`，模型先预测一组 goal-point 假设；每个 goal 条件化一个 masked discrete-diffusion decoder，并在少量并行去噪轮次内生成轨迹；产生初始草稿后，AutoEdit 复用同一个条件 token 模型更新选中的轨迹 token。

该方法有三个耦合组件：

1. goal-point posterior 作为行为级假设的紧凑决策层；
2. goal-conditioned masked diffusion 将每个假设实现为完整轨迹；
3. AutoEdit 通过选择性重写草稿轨迹实现 token 空间纠错。

监督阶段同时训练 masked trajectory generation 和 structure-aware correction：随机 masking 让模型学会草拟轨迹，扰动纠错让模型从纵向和横向规划错误中恢复 clean trajectory。约束感知 field loss 则根据可行驶区域几何约束轨迹 token 的空间分布。

强化学习阶段优化完整 draft-and-edit rollout，而不是只优化 drafting。对每个候选样本，终端驾驶奖励分配给最终 post-edit 轨迹，策略梯度信用作用于 drafting 和 AutoEdit 两个阶段的 token 转移。这是 ReflectDrive-2 的关键：AutoEdit 不是后处理启发式，而是与 drafter 在同一闭环目标下优化的策略 rollout 组成部分。

### 4.2 目标条件掩码轨迹扩散

两个时间相邻的全景帧由 ViT 视觉 backbone 编码并投影到 diffusion Transformer 的 token 空间。视觉 token 与导航指令 token、自车状态 token 拼接后输入共享 backbone。每个 Transformer block 还包含 action-specific FFN 和 action head，用于轨迹 token 预测。

ReflectDrive-2 不直接提交单峰终点预测，而是预测离散 BEV 坐标上的 goal-point posterior。goal point 是离散 `(x,y)` token pair，表示未来规划的行为级假设。训练时，goal head 用专家 endpoint 监督；推理时，使用 top-k sampling 加 BEV 空间 NMS 采样候选 goal。NMS 删除重复 endpoint，同时保留空间上不同的替代方案，使不同 goal 可以对应车道保持、让行、绕行、转弯不同线路等行为。

未来自车轨迹由 8 个 waypoint 表示。每个 waypoint 离散化为一个纵向 token 和一个横向 token，因此长度为 `L=16`：

```text
x0 = [x1, y1, ..., x8, y8]
```

最终坐标对 `(x8,y8)` 对应选中的 goal。训练时随机位置替换为 `[MASK]` 并使用 all-position masked-diffusion objective；推理时固定 goal token，其余轨迹 token 初始化为 `[MASK]`，模型在少量并行去噪轮次中填充 masked positions。每轮提交置信度最高的预测。

### 4.3 AutoEdit 轨迹修正

AutoEdit 是一个 token-to-token 轨迹编辑器，运行在与 masked-diffusion drafter 相同的离散动作空间中。与 masked drafting 不同，AutoEdit 不会把选中的轨迹 token 重新变为 `[MASK]`。它接收当前具体轨迹 token 序列，预测替换 token，并只提交被选中的替换项。因此 AutoEdit 执行直接 token-to-token 重写，而不是 re-masking 和 re-denoising。

给定 clean waypoint 序列 `z0`，作者在 token 化前合成扰动轨迹 `T(z0)`。扰动覆盖两个常见规划错误族：

**纵向进度扰动**：沿轨迹弧长重缩放进度。`beta < 1` 产生保守前进不足，`beta > 1` 产生过冲或减速不足。

**横向航向扰动**：在自车坐标系中旋转轨迹，产生连贯横向偏差，同时保持轨迹平滑。

扰动轨迹 token 化为 `tilde{x0}` 后，AutoEdit 学习从扰动 token 序列直接映射回 clean token 序列：

```text
L_SAP(theta) = - E_{x0,T} [ 1/L * sum_i log q_theta(x0_i | tilde{x0}, c) ].
```

推理时，AutoEdit 从草拟轨迹 `x^(0)` 开始执行 K 轮 token-to-token editing。每一轮，模型根据当前轨迹 token 预测替换序列，再根据 commit mask 只提交低置信度的非 goal 轨迹 token，goal token 保持固定作为行为锚点。commit mask 不是 re-masking mask，输入始终是具体 token 序列。

### 4.4 约束感知监督目标

`L_DLM` 和 `L_SAP` 优化 token 级预测，但不显式编码可行驶区域几何。作者增加 field-based spatial penalty，对坐标 token logits 诱导出的 waypoint 空间分布进行惩罚。给定 BEV cost field `C`，如果模型把概率质量分配到高成本区域，就用 field-weighted log barrier 惩罚。

实现中，`C` 是 drivable-area compliance field。可行驶区域外的点根据距离最近可行驶 cell 的距离构造 cost，边界附近设置安全容忍带。完整监督目标为：

```text
L_sup = L_DLM + lambda_SAP * L_SAP + lambda_field * L_field
```

### 4.5 draft-and-edit rollout 上的强化学习

监督训练让模型模仿专家轨迹并从合成扰动中恢复，但不直接优化闭环驾驶指标。因此作者在组合 draft-and-edit rollout 上进行强化学习微调。关键区别在于：生成过程被显式分为 drafting 阶段和 AutoEdit 阶段。终端奖励分配给最终 post-edit 轨迹，策略梯度 objective 同时给两个阶段的 token 转移分配信用。

每个场景中，作者用 top-k sampling 和 NMS 采样 `N_g` 个 goal point，每个 goal 采样 `I` 个 draft，总计 `G=N_g I` 个候选 rollout。候选 g 的 token 转移序列先经历 `S_draft` 个 masked drafting 转移，再经历 `S_edit` 个 AutoEdit 转移。最终轨迹由最后 token 序列 detokenize 得到。

使用闭环规划分数作为终端奖励 `R(tau_g)`，并计算 group-relative advantage：

```text
A_g = R(tau_g) - 1/G * sum_j R(tau_j)
```

因为只有 post-edit 轨迹收到奖励，drafting 阶段会被优化为产生可被后续修正改善的轨迹，AutoEdit 阶段则被优化为产生能提升闭环分数的修正，而不是仅降低 token 不确定性。

## 5 面向反思式掩码规划的高效推理

作者将部署视为一条优化链。最终栈在 full-step 帧运行完整 decision-draft-reflect，在 lite-step 帧运行轻量 temporal AutoEdit，NVIDIA Thor 上平均每帧 31.8 ms。

**共享前缀 KV 复用**：goal proposal、trajectory drafting 和 AutoEdit 都依赖同一视觉、导航、自车状态前缀。保留共享前缀 cache，并在不同 serving 阶段切换 cache 状态，可将 attention operator 延迟从 0.28 ms 降到 0.08 ms。

**可变 action cache 回退与合并重写**：action-token block 是可变的。每次 masked drafting 或 AutoEdit 更新 token 后，旧 action-token KV 失效，因此 cache 指针回退到共享前缀边界，只重算 mutable action block。多块边界处进一步把 cache rewrite 与下一块的第一次 token 更新合并，使边界延迟从 14.7 ms 降到 11.5 ms。

**Action-expert FFN**：轨迹 token 解码使用受限动作词表和短定长 token block，因此把 action branch 的 full FFN 替换为小型 action-expert FFN，将隐藏维度从 4096 降至 1024，per-block FFN 延迟从 2.47 ms 降到 0.95 ms。

**融合端上 token update**：masked drafting 和 AutoEdit 都需要置信度排序、token selection 和状态更新。CPU 实现会在每步引入设备同步。作者把 token selection、ranking 和 token-state update 融合到端上 CUDA kernel，使单步更新延迟从 0.45 ms 降到 0.06 ms。

**Alternating Step Decode 作为时间 token-to-token AutoEdit**：流式驾驶中，相邻帧场景和未来计划高度相似。ReflectDrive-2 在 full-step 与 lite-step 帧之间交替：full-step 运行完整 pipeline；lite-step 将上一帧规划变换到当前自车坐标系，再执行短 token-to-token AutoEdit，而不是从头生成轨迹。

### 推理优化链

| 优化 | 延迟 before to after | 加速 |
|---|---:|---:|
| 共享前缀 KV 复用 | 0.28 to 0.08 ms | 3.5x |
| Cache rewind 和 merged rewrite | 14.7 to 11.5 ms | 1.28x |
| Action-expert FFN | 2.47 to 0.95 ms | 2.6x |
| 融合 CUDA unmasking | 0.45 to 0.06 ms | 7.5x |
| ASD temporal AutoEdit | 26.2 to 7.6 ms | 3.4x |
| 端到端 planner 平均每帧 | 45.0 to 31.8 ms | 1.42x |

## 6 实验

### 6.1 实验设置

**数据集与指标**：实验在 NA VSIM 上评估。NA VSIM 是基于 nuPlan 的闭环规划 benchmark。任务是以 2 Hz 预测 4 秒自车轨迹。训练集为 navtrain，包含 1,192 个场景；评估集为 navtest，包含 136 个场景。指标为 PDMS，聚合 no at-fault collision、drivable-area compliance、time to collision、comfort 和 ego progress。

**实现**：模型包含 0.7B masked-diffusion language backbone 和 0.1B ViT visual encoder，均从专有预训练权重初始化，并在 NA VSIM 上全量微调。输入为左前、正前、右前三相机的两个时间帧，加导航指令和自车状态 token；输出为 8 个 waypoint，对应 16 个离散坐标 token。训练先 SFT，再使用 PDMS 作为 reward 做 RFT。

**基线**：端到端规划器包括 UniAD、TransFuser、Hydra-MDP、DiffusionDrive、GoalFlow；VLA 规划器包括 AutoVLA、DriveVLA-W0、ReCogDrive。标准评估中所有方法输出单条轨迹。best-of-N 评估中，ReflectDrive-2 采样多个 goal point，并保留闭环分数最高的轨迹，这是 oracle selection，不作为标准 benchmark 结果。

### 6.2 RL 对推理阶段 AutoEdit 的影响

Table 3 是本文最关键的实验。监督训练后，推理阶段 AutoEdit 最多只提升 0.3 PDMS；即使用结构感知扰动训练过 AutoEdit，闭环贡献仍很有限。加入完整 draft-and-edit rollout 的强化学习后，同一个推理阶段 AutoEdit 提升 1.9 PDMS。机制是：共享终端奖励让 drafter 学会输出可修订草稿，让 AutoEdit 学会朝奖励方向修正，而不仅是降低 token 不确定性。

| 训练设置 | 无 AutoEdit | 有 AutoEdit | PDMS 增益 |
|---|---:|---:|---:|
| DLM | 84.8 | 85.0 | +0.2 |
| DLM + DACF | 87.2 | 87.3 | +0.1 |
| DLM + DACF + AutoEdit training | 87.7 | 88.0 | +0.3 |
| DLM + DACF + AutoEdit training + RL | 89.1 | 91.0 | +1.9 |

### 6.3 闭环驾驶表现

标准单轨迹设置下，ReflectDrive-2 纯视觉输入达到 91.0 PDMS，高于 ReCogDrive 的 90.8 和使用相机加 LiDAR 的 GoalFlow 的 90.3。最大优势在 ego progress，达到 89.4，是表中最高；DAC 仍保持 98.1，comfort 达到 100.0。不过 NC 和 TTC 不是所有基线中最高。与其他 camera-only VLA 相比，ReflectDrive-2 的 0.2 到 1.9 PDMS 优势主要来自 Table 3 隔离出的 rollout 级 RL 与 AutoEdit 交互。

| 方法 | 输入 | NC | DAC | TTC | Comfort | EP | PDMS |
|---|---|---:|---:|---:|---:|---:|---:|
| UniAD | Cam | 97.8 | 91.9 | 92.9 | 100.0 | 78.8 | 83.4 |
| TransFuser | C&L | 97.7 | 92.8 | 92.8 | 100.0 | 79.2 | 84.0 |
| Hydra-MDP | C&L | 98.3 | 96.0 | 94.6 | 100.0 | 78.7 | 86.5 |
| DiffusionDrive | C&L | 98.2 | 96.2 | 94.7 | 100.0 | 82.2 | 88.1 |
| GoalFlow | C&L | 98.4 | 98.3 | 94.6 | 100.0 | 85.0 | 90.3 |
| AutoVLA | Cam | 98.4 | 95.6 | 98.0 | 99.9 | 81.9 | 89.1 |
| DriveVLA-W0 | Cam | 98.7 | 99.1 | 95.3 | 99.3 | 83.3 | 90.2 |
| ReCogDrive | Cam | 97.9 | 97.3 | 94.9 | 100.0 | 87.3 | 90.8 |
| ReflectDrive-2 | Cam | 97.3 | 98.1 | 92.5 | 100.0 | 89.4 | 91.0 |

best-of-6 oracle 设置下，ReflectDrive-2 达到 94.8 PDMS，与 NA VSIM human reference 相同。single 与 best-of-6 的 3.8 PDMS 差距说明 goal posterior 捕获了多模态行为分布，而不是同一 endpoint 的噪声副本。

| 设置 | NC | DAC | TTC | Comfort | EP | PDMS |
|---|---:|---:|---:|---:|---:|---:|
| ReflectDrive-2 single | 97.3 | 98.1 | 92.5 | 100.0 | 89.4 | 91.0 |
| ReflectDrive-2 best-of-6 oracle | 98.5 | 99.2 | 95.5 | 99.8 | 93.8 | 94.8 |
| Human reference | 100.0 | 100.0 | 100.0 | 99.9 | 87.5 | 94.8 |

### 6.4 决策多样性与反思

goal point 可视化显示，在转弯场景中，不同 goal 对应不同过弯线，其中部分候选更好地遵守可行驶边界；在交互场景中，模型能围绕邻近交通参与者生成纵向和横向不同的行为，例如保持车道、变道、调整速度。因此 goal point 不是采样噪声，而是不同的行为假设。

AutoEdit 可视化显示，初始草稿经过 AutoEdit 后可以被拉回可行驶区域，或围绕邻近交通参与者调整轨迹。这些修订是在同一 token 空间中进行的结构化重写，不是简单平滑。

### 6.5 消融实验

没有推理阶段 AutoEdit 时，field loss 主要通过 DAC 带来 2.4 PDMS 增益；AutoEdit supervised training 再增加 0.5；完整 rollout RL 将 EP 从 82.2 提升到 89.3，将 PDMS 提升到 89.1；再结合推理阶段 AutoEdit，最终达到 91.0。

| 训练目标 | NC | DAC | TTC | Comfort | EP | PDMS |
|---|---:|---:|---:|---:|---:|---:|
| DLM | 97.5 | 93.9 | 92.8 | 99.5 | 79.5 | 84.8 |
| DLM + DACF | 97.4 | 97.0 | 93.1 | 99.9 | 81.4 | 87.2 |
| DLM + DACF + AutoEdit training | 97.8 | 96.7 | 93.6 | 99.9 | 82.2 | 87.7 |
| DLM + DACF + AutoEdit training + RL | 96.3 | 97.9 | 88.9 | 99.7 | 89.3 | 89.1 |

推理预算方面，扩散生成步数和 AutoEdit 步数增加后性能先提升后平台化，大约 3 到 5 步附近达到稳定。过多重写可能扰动已经较好的草稿。goal-proposal 参数方面，更多 proposal 可暴露更多行为假设；NMS threshold 约 1.2 m 最优，太小会保留重复项，太大则会删除真实替代方案。

## 7 结论

ReflectDrive-2 将自动驾驶规划重构为在共享离散 token 空间中的决策、轨迹草拟和自修正联合过程。goal-point posterior 先暴露行为级假设，masked discrete diffusion 并行生成可编辑轨迹，AutoEdit 用同一策略重写草稿，不需要辅助修复网络。

本文最核心的发现是：驾驶规划中的自修正不仅需要一个训练过的 editor。仅监督训练时，AutoEdit 虽存在于模型权重中，但推理收益很小。对完整 draft-and-edit rollout 使用强化学习后，共享终端奖励使 drafter 和 editor 协同适应：草稿变得可修订，编辑变得面向奖励。该交互将 AutoEdit 的推理增益从 0.3 PDMS 提升到 1.9 PDMS，也是 ReflectDrive-2 纯视觉 91.0 PDMS 的主要来源。

best-of-6 oracle 达到 94.8 PDMS，说明 goal posterior 能捕获真实多模态驾驶行为分布。作者还展示 decision-draft-reflect 不只是建模范式，也能定义高效运行时：共享前缀 KV cache、Alternating Step Decode、轻量 action-expert FFN 和融合端上 unmasking 将 NVIDIA Thor 上的平均 planner 延迟降到 31.8 ms。

### 局限与未来工作

ReflectDrive-2 用固定分辨率 BEV 坐标 token 表示轨迹，这带来可解释、可编辑的动作空间，但 waypoint 精度受坐标 bin 尺寸限制。未来可用更细坐标词表、残差 offset 或离散连续混合 action head 改善精度，同时保留 token 空间可编辑性。当前 RL 阶段优化轻量闭环规划分数，仍是真实驾驶目标的 proxy。更高保真交互仿真器和更丰富安全奖励可能改善对齐，但计算成本更高。当前 AutoEdit 扰动聚焦纵向进度与横向航向错误，未来可扩展到让行时机、cut-in 响应、gap selection 等交互级失败。

## 论文解读问答

### Q1 论文试图解决什么问题？

它试图解决自动驾驶轨迹规划中的结构性错误纠正问题：模仿学习规划器常在纵向速度/进度和横向航向上犯错，而现有端到端、VLA 或连续扩散方法要么一次性提交轨迹，要么纠错成本高，要么没有让“草拟”和“编辑”围绕闭环奖励共同优化。论文希望让规划器能先生成草稿，再在同一离散 token 空间中原地修正，并让这种修正真正提升闭环驾驶分数。

### Q2 这是否是一个新的问题？

不是全新问题。自动驾驶规划、模仿学习分布偏移、轨迹 refinement、扩散策略 RL 都是已有问题。但本文的问题组织方式有新意：把自动驾驶规划中的自编辑明确建模为“离散扩散 token 空间中的 decision-draft-reflect”，并强调通过完整 draft-and-edit rollout 的 RL 来耦合 drafter 和 editor。

### Q3 这篇文章要验证一个什么科学假设？

核心假设是：仅训练一个轨迹 editor 不足以带来显著闭环收益；只有把 drafting 和 editing 放进同一个 rollout，并用 post-edit 轨迹的终端驾驶奖励共同训练，模型才会学到“可被修正的草稿”和“面向奖励的修正”，从而显著提升闭环规划性能。

### Q4 有哪些相关研究？如何归类？谁是这一课题在领域内值得关注的研究员？

相关研究可分为四类：

1. 端到端自动驾驶规划：ChauffeurNet、CIL、TransFuser、ST-P3、UniAD、VAD、GoalFlow 等。
2. VLA/语言动作规划：OpenVLA、DriveVLM、DriveLM、AutoVLA、DriveVLA、ReCogDrive 等。
3. 扩散规划与扩散策略：Diffuser、Diffusion Policy、DiffusionDrive、MotionDiffuser、GoalFlow、HDP 等。
4. 离散扩散、token editing 与 RL 对齐：D3PM、MaskGIT、LLaDA、LLaDA 2.1、Seed Diffusion、Block Diffusion、DDPO、DPPO、d1、d2、SPG、DriveFine。

值得关注的研究员包括：Andreas Geiger、Kashyap Chitta、Marco Pavone、Boris Ivanovic、Holger Caesar、Raquel Urtasun、Drew Bagnell、Sergey Levine、Shuran Song、Michael Janner、Yilun Du，以及 UniAD/VAD/DriveVLA/DriveLM/DriveFine 相关作者群体。就本文直接相关方向，还应关注 LiAuto 该论文作者团队及近期离散扩散语言模型、扩散 RL 后训练团队。

### Q5 论文中提到的解决方案之关键是什么？

关键是三件事合在一起：第一，把轨迹表示为可并行生成、可原地重写的离散 token；第二，用 AutoEdit 在同一模型和同一 token 空间中直接 token-to-token 修正轨迹；第三，用完整 draft-and-edit rollout 的 RL，把 post-edit 闭环奖励同时分配给草拟和编辑阶段，让两者协同优化。

### Q6 论文中的实验是如何设计的？

实验先在 NA VSIM 上做标准单轨迹闭环评估，与端到端方法和 camera-only VLA 方法比较；再做 best-of-6 oracle 评估来衡量 goal posterior 的多模态候选质量；然后用 Table 3 隔离 RL 与 AutoEdit 的交互，比较不同训练阶段下推理 AutoEdit 的收益；最后做训练组件消融、推理步数敏感性、goal proposal 参数敏感性和部署延迟/质量评估。

### Q7 用于定量评估的数据集是什么？代码有没有开源？

定量评估数据集是 NA VSIM，它基于 nuPlan，训练 split 为 navtrain，包含 1,192 scenes；评估 split 为 navtest，包含 136 scenes。主要指标是 PDMS，由 NC、DAC、TTC、comfort 和 EP 聚合。

论文正文没有给出代码仓库链接，也没有声明代码开源。根据截至 2026-05-10 的公开网页检索，暂未找到官方 ReflectDrive-2 代码仓库；CatalyzeX 页面也显示只能 request code。因此应判断为尚未开源，至少论文发布时未公开代码。

### Q8 论文中的实验及结果有没有很好地支持需要验证的科学假设？

总体支持较强。Table 3 直接验证了核心假设：监督训练下 AutoEdit 只有 0.1 到 0.3 PDMS 增益，而完整 rollout RL 后 AutoEdit 增益达到 1.9 PDMS。这非常贴合“自编辑必须通过共享奖励与 drafter 耦合”的主张。Table 4 显示最终系统在 NA VSIM 单轨迹评估中达到最高 PDMS，也支持方法有效性。

但也有保留：数据集规模较小，navtest 只有 136 scenes；模型使用专有预训练权重；best-of-6 是 oracle selection，不是实际部署策略；代码和模型未开源，不利于复现。因此实验支持论文主张，但外部可复现性和更大范围泛化仍需进一步验证。

### Q9 这篇论文到底有什么贡献？

贡献可以概括为：提出了一个可编辑离散扩散驾驶规划框架 ReflectDrive-2；提出 AutoEdit，把轨迹纠错变成同一 token 空间中的直接自编辑；证明完整 draft-and-edit rollout 的 RL 能显著放大自编辑收益；在 NA VSIM 上实现纯视觉 91.0 PDMS 和 best-of-6 oracle 94.8 PDMS；同时给出面向车端部署的高效反思式解码方案，在 NVIDIA Thor 上达到 31.8 ms 平均延迟。
