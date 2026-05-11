# DiffusionDriveV2：端到端自动驾驶中的强化学习约束截断扩散建模

原文：`Zou 等 - 2025 - DiffusionDriveV2 Reinforcement Learning-Constrained Truncated Diffusion Modeling in End-to-End Auto.pdf`  
作者：Jialv Zou、Shaoyu Chen、Bencheng Liao、Zhiyu Zheng、Yuehao Song、Lefei Zhang、Qian Zhang、Wenyu Liu、Xinggang Wang  
机构：华中科技大学、地平线、武汉大学  
代码与模型：<https://github.com/hustvl/DiffusionDriveV2>

## 摘要

用于端到端自动驾驶轨迹规划的扩散模型常受到模式坍缩影响，倾向于生成保守且同质化的行为。DiffusionDrive 通过预定义 anchors 表示不同驾驶意图，并用它们划分动作空间，从而生成多样轨迹。但由于它依赖 imitation learning，约束不足，会陷入“多样性”和“稳定高质量”之间的困境。

本文提出 DiffusionDriveV2，使用强化学习约束低质量模式，同时探索更优轨迹。这显著提升了整体输出质量，同时保留底层 Gaussian Mixture Model 的多模态能力。首先，作者使用适合轨迹规划的 scale-adaptive multiplicative noise，以促进广泛探索。其次，提出 intra-anchor GRPO，在单个 anchor 内部进行 advantage estimation；同时提出 inter-anchor truncated GRPO，从跨 anchor 全局视角提供稳定训练信号，避免把不同驾驶意图，例如左转和直行，错误地相互比较，从而进一步造成模式坍缩。

DiffusionDriveV2 在 NAVSIM v1 上达到 91.2 PDMS，在 NAVSIM v2 上达到 85.5 EPDMS，均使用相同 ResNet-34 backbone，刷新记录。进一步实验验证，本文方法缓解了截断扩散模型在多样性和稳定高质量之间的矛盾。

## 1 引言

随着 3D 检测、多目标跟踪、预训练、在线建图和运动预测等任务成熟，自动驾驶系统研究逐渐转向端到端自动驾驶，即从原始传感器输入直接学习驾驶策略。

早期端到端单模态规划器回归一条轨迹，无法在高不确定复杂场景中提出备选方案。基于选择的方法使用大型静态候选轨迹词表，但离散化灵活性有限。扩散模型可以根据周围场景动态生成少量候选轨迹，因此被用于轨迹生成。但 vanilla diffusion 直接用于多模态轨迹生成时会模式坍缩，收敛到单一高概率模式，无法捕获未来多样性。

DiffusionDrive 使用由多个预定义 trajectory anchors 定义的 Gaussian Mixture Model 作为初始噪声先验。每个 anchor 对应一种驾驶意图，例如变道或直行，从而将生成空间划分为多个子空间，促进多样行为生成。

但 DiffusionDrive 面临一个根本困境：多样性与稳定高质量难以兼得。其 imitation learning 目标在实践中通常只优化最接近专家轨迹的单个正模式，忽略绝大多数负模式。因此模型可能同时生成高质量轨迹和大量未受约束、低质量、甚至碰撞的轨迹。系统被迫依赖下游 selector 从大量候选中筛掉坏轨迹，而 selector 参数更少、泛化更弱，OOD 场景中风险较高。

强化学习为这个问题提供了解法。与只学习单个正样本模式的 imitation learning 不同，RL 可以执行 exploration-constraint：一方面对所有模式施加目标对齐约束，奖励期望行为并惩罚不安全动作，抬高模型下限；另一方面推动模型探索更广动作空间，寻找可能优于专家的策略，抬高模型上限。

已有一些工作将 GRPO 引入端到端自动驾驶，但主要用于 vanilla diffusion。对 anchored truncated diffusion 来说，每个 anchor 表示不同驾驶意图。如果直接在不同意图之间做 advantage comparison，会加剧模式坍缩。例如左转和直行不应该简单比较优劣，而应在各自意图下共存。基于此，DiffusionDriveV2 提出 intra-anchor GRPO 和 inter-anchor truncated GRPO。

## 2 相关工作

**端到端自动驾驶**：UniAD 将感知和规划任务统一；VAD 使用向量化场景表示提升效率；VADv2、Hydra-MDP、GoalFlow 等探索多模态规划和候选选择。

**扩散式轨迹生成**：vanilla diffusion 容易模式坍缩；DiffusionDrive 通过 anchor GMM 提升多模态性，但负模式缺少约束。

**强化学习后训练**：DeepSeek-R1 推动了 GRPO 风格训练，端到端驾驶中也出现了 DIVER 等 RL 方法。但如何将 GRPO 稳定迁移到截断扩散和 GMM anchor 框架，是本文重点。

## 3 预备知识

DiffusionDrive 的截断扩散生成器使用多个 anchors 构造 Gaussian Mixture Model。每个 anchor 附近采样噪声，经过少量去噪步骤生成候选轨迹。它能产生多样驾驶意图，但 imitation learning 只强监督接近专家的正模式，对大量负模式没有足够约束。

GRPO 通过组内样本的相对奖励估计 advantage，不需要额外 value model。直接用于 anchored GMM 时，如果把不同 anchor 的样本放在同一组比较，就可能把不同驾驶意图相互压制，破坏多模态性。

## 4 方法

### 4.1 截断扩散生成器

DiffusionDriveV2 继承 DiffusionDrive 的截断扩散生成器。模型从 anchor-centered 分布采样噪声轨迹，并在少量去噪步骤中生成候选轨迹。不同颜色或不同 anchor 代表不同驾驶意图。

### 4.2 面向扩散生成器的强化学习

训练目标是用闭环规划奖励替代或补充 imitation learning 对单个正模式的监督。RL 不只优化最接近专家的轨迹，而是对多个候选输出进行约束和探索。奖励来自轨迹在规划指标下的表现，例如安全、可行驶区域、进度、舒适性等。

### 4.3 Scale-adaptive multiplicative exploration

普通 additive noise 对轨迹规划不理想，因为近端 waypoint 和远端 waypoint 的尺度不同。远端坐标通常更大，同样幅度的加性噪声对近端和远端影响不一致。DiffusionDriveV2 使用 scale-adaptive multiplicative noise，使探索噪声随轨迹尺度自适应变化，从而保持探索轨迹的平滑性和连贯性。

### 4.4 Intra-Anchor GRPO

Intra-Anchor GRPO 只在同一 anchor 内进行组内 advantage estimation。这样，每个驾驶意图内部的多个样本可以比较好坏，例如同为左转意图的不同速度或路径可以比较；但不同意图之间不会被错误比较，例如左转不会和直行竞争。这可以避免跨意图比较造成的模式坍缩。

### 4.5 Inter-Anchor Truncated GRPO

只做 intra-anchor 比较会缺少全局视角，不利于不同 anchor 之间的整体质量评估。Inter-Anchor Truncated GRPO 引入跨 anchor 的截断式全局信息，让训练既能看到全局质量差异，又不让不同驾驶意图之间的比较信号过强，避免不合理压制多模态。

### 4.6 Mode selector

DiffusionDriveV2 仍使用 mode selector 从多个候选轨迹中选择最终输出。但论文强调，RL 的目标是提升 raw generated trajectories 的整体质量，降低对 selector 的过度依赖。selector 只是最终决策模块，而不是弥补大量低质量候选的唯一防线。

## 5 实验

### 5.1 Benchmark 与实现

实验使用 NA VSIM v1 和 NA VSIM v2。NA VSIM 基于 OpenScene/nuPlan，包含真实规划导向驾驶场景，传感器包括 8 个相机和 5 个 LiDAR 合并点云。数据分为 navtrain 1,192 个训练场景和 navtest 136 个评估场景。

为公平比较，DiffusionDriveV2 使用与 Transfuser 和 DiffusionDrive 相同的 ResNet-34 backbone，并匹配 DiffusionDrive 的 diffusion decoder size。输入为三个裁剪缩放的前向相机图像拼接成 1024 x 256 图像，以及 LiDAR 点云的栅格化 BEV 表示。模型以 DiffusionDrive 预训练权重冷启动，在 navtrain 上用 RL 训练 10 epoch；mode selector 训练 20 epoch。推理时仍只需 2 个去噪步骤。

### 5.2 NA VSIM v1 主结果

DiffusionDriveV2 在 NA VSIM v1 navtest 上达到 91.2 PDMS，高于 DiffusionDrive 的 88.1，EP 从 82.2 提升到 87.5。它还超过了 DIVER、DriveSuprim、GoalFlow、Hydra-MDP 等方法。

| 方法 | Backbone | NC | DAC | TTC | Comf. | EP | PDMS |
|---|---|---:|---:|---:|---:|---:|---:|
| Transfuser | ResNet-34 | 97.7 | 92.8 | 92.8 | 100 | 79.2 | 84.0 |
| DRAMA | ResNet-34 | 98.0 | 93.1 | 94.8 | 100 | 80.1 | 85.5 |
| DiffusionDrive | ResNet-34 | 98.2 | 96.2 | 94.7 | 100 | 82.2 | 88.1 |
| DIVER | ResNet-34 | 98.5 | 96.5 | 94.9 | 100 | 82.6 | 88.3 |
| DriveSuprim | ResNet-34 | 97.8 | 97.3 | 93.6 | 100 | 86.7 | 89.9 |
| DiffusionDriveV2 | ResNet-34 | 98.3 | 97.9 | 94.8 | 99.9 | 87.5 | 91.2 |

### 5.3 NA VSIM v2 主结果

在更难的 NA VSIM v2 上，DiffusionDriveV2 达到 85.5 EPDMS，超过 Transfuser、Hydra-MDP++、DriveSuprim 和 ARTEMIS。

| 方法 | EPDMS |
|---|---:|
| Ego Status MLP | 64.0 |
| Transfuser | 76.7 |
| Hydra-MDP++ | 81.4 |
| DriveSuprim | 83.1 |
| ARTEMIS | 83.1 |
| DiffusionDriveV2 | 85.5 |

### 5.4 多样性与质量

论文引入 Diversity Metric 衡量多模态生成能力，并报告 Top-K PDMS 来评估 raw generated trajectories 的整体质量。每个模型生成 20 条轨迹，评价 selector 之前的原始输出。

| 方法 | Diversity | PDMS@1 | PDMS@5 | PDMS@10 |
|---|---:|---:|---:|---:|
| TransfuserTD | 0.1 | 85.7 | 85.7 | 85.7 |
| DiffusionDrive | 42.3 | 93.5 | 84.3 | 75.3 |
| DiffusionDriveV2 | 30.3 | 94.9 | 91.1 | 84.4 |

结果说明：vanilla/truncated 单调生成方法质量较稳定但几乎没有多样性；DiffusionDrive 多样性很高，但低质量候选较多，PDMS@10 明显下降；DiffusionDriveV2 在保持较高多样性的同时显著提高 Top-K 质量，说明 RL 约束改善了多样性与稳定质量之间的权衡。

### 5.5 消融实验

**探索噪声**：multiplicative noise 优于 additive noise，PDMS 从 89.7 提升到 90.1。

**Intra-Anchor GRPO**：加入 intra-anchor 后，PDMS 从 89.2 提升到 90.1，证明在同一 anchor 内做 advantage estimation 很关键。

**Inter-Anchor Truncated GRPO**：加入 inter-anchor truncated 后，PDMS 从 89.5 提升到 90.1，说明全局信息有助于训练稳定和质量提升。

## 6 结论

DiffusionDriveV2 通过强化学习解决 DiffusionDrive 中由 imitation learning 导致的多样性与稳定高质量冲突。它提出 scale-adaptive multiplicative exploration、Intra-Anchor GRPO 和 Inter-Anchor Truncated GRPO，使截断扩散生成器能约束所有模式并探索更优轨迹。实验表明，它在 NA VSIM v1 和 v2 上刷新性能，并在保持多模态生成能力的同时显著提升候选轨迹整体质量。

## 论文解读问答

### Q1 论文试图解决什么问题？

论文试图解决截断扩散自动驾驶规划中的“多样性与稳定高质量”矛盾。DiffusionDrive 可以生成多样轨迹，但 imitation learning 只强监督接近专家的正模式，导致大量负模式缺少约束，可能产生碰撞或低质量轨迹。DiffusionDriveV2 试图用 RL 同时约束坏模式、探索好模式。

### Q2 这是否是一个新的问题？

不是完全新问题。多模态生成、模式坍缩、imitation learning 局限和 RL 后训练都已有研究。但在 anchored truncated diffusion/GMM 轨迹生成框架中，系统性处理“各模式都要高质量，同时不破坏多样性”的问题，是本文的新颖点。

### Q3 这篇文章要验证一个什么科学假设？

核心假设是：对 anchored truncated diffusion 规划器，仅靠 imitation learning 无法约束全部模式；如果用结构化 RL，尤其是 intra-anchor 与 inter-anchor truncated 的 GRPO 信用分配，就能在保持多模态的同时提高所有候选轨迹质量。

### Q4 有哪些相关研究？如何归类？谁是这一课题在领域内值得关注的研究员？

相关研究可分为端到端规划、anchor/选择式多模态规划、扩散轨迹生成、RL 后训练和 GRPO。代表方法包括 UniAD、VAD、VADv2、Hydra-MDP、GoalFlow、DiffusionDrive、DIVER、DriveSuprim，以及机器人扩散策略相关工作。

值得关注的研究员包括 Sergey Levine、Shuran Song、Michael Janner、Yilun Du、Marco Pavone、Andreas Geiger、Kashyap Chitta、Xinggang Wang、Wenyu Liu，以及 NAVSIM、DiffusionDrive、VAD 系列团队。

### Q5 论文中提到的解决方案之关键是什么？

关键是把 RL 正确接入 anchored truncated diffusion：用 scale-adaptive multiplicative noise 保持轨迹探索合理；用 intra-anchor GRPO 避免不同驾驶意图相互压制；用 inter-anchor truncated GRPO 提供全局质量信号；最后配合 mode selector 输出最终轨迹。

### Q6 论文中的实验是如何设计的？

论文在 NA VSIM v1 和 v2 上做闭环评估，报告 PDMS/EPDMS 和子指标；比较 Transfuser、Hydra-MDP、GoalFlow、DiffusionDrive、DIVER、DriveSuprim 等方法；用 Diversity 与 Top-K PDMS 评估 raw generated trajectories 的多样性和整体质量；再通过探索噪声、Intra-Anchor GRPO、Inter-Anchor Truncated GRPO 消融验证每个设计。

### Q7 用于定量评估的数据集是什么？代码有没有开源？

定量评估使用 NA VSIM v1 和 NA VSIM v2。数据基于 OpenScene/nuPlan，split 为 navtrain 1,192 个训练场景和 navtest 136 个评估场景。代码和模型已公开，论文给出的地址是 <https://github.com/hustvl/DiffusionDriveV2>。

### Q8 论文中的实验及结果有没有很好地支持需要验证的科学假设？

支持较强。NA VSIM v1/v2 主结果显示 RL 后训练显著提升 PDMS/EPDMS；Top-K PDMS 表明 DiffusionDriveV2 不只是 selector 选得好，而是原始候选整体质量更高；消融实验也分别支持 multiplicative noise、intra-anchor 和 inter-anchor truncated 的必要性。不过训练从 DiffusionDrive 冷启动，且仍依赖仿真式闭环指标，真实世界泛化仍需要更多证据。

### Q9 这篇论文到底有什么贡献？

贡献是提出 DiffusionDriveV2，将 RL 引入 anchored truncated diffusion 自动驾驶规划；识别并解决 DiffusionDrive 的多样性-质量困境；提出 scale-adaptive multiplicative exploration、Intra-Anchor GRPO、Inter-Anchor Truncated GRPO；在 NA VSIM v1/v2 上达到新的最优结果，并提升 raw candidate trajectories 的整体质量。
