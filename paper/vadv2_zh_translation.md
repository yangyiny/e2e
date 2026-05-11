# VADv2：通过概率规划实现端到端向量化自动驾驶

原文：`Chen 等 - 2024 - VADv2 End-to-End Vectorized Autonomous Driving via Probabilistic Planning.pdf`  
作者：Shaoyu Chen、Bo Jiang、Hao Gao、Bencheng Liao、Qing Xu、Qian Zhang、Chang Huang、Wenyu Liu、Xinggang Wang  
机构：华中科技大学、地平线  
项目页：<https://hgao-cv.github.io/VADv2>  
相关代码入口：<https://github.com/hustvl/VAD>

## 摘要

从大规模驾驶示范中学习类人驾驶策略很有前景，但规划任务本身具有不确定性和非确定性，这使得从示范中提取驾驶知识变得困难。为了解决不确定性问题，本文提出 VADv2，一个基于概率规划的端到端驾驶模型。

VADv2 以流式方式输入多视角图像序列，将传感器数据转换为环境 token embedding，输出动作的概率分布，并从中采样一个动作控制车辆。仅使用相机传感器，VADv2 在 CARLA Town05 benchmark 上取得了当时最优闭环性能，显著优于已有方法。它甚至在没有 rule-based wrapper 的情况下，也能以完全端到端方式稳定运行。

## 1 引言

端到端自动驾驶试图从大规模人类驾驶示范中学习类人驾驶策略。但规划并不是确定性映射：在同一环境下，人类驾驶员可能有多种合理动作。例如跟车时可以继续跟随，也可以变道超车；面对迎面来车时可以让行，也可以抢先通过。动作的时机和速度受到许多不可观测因素影响，因此具有强随机性。

已有学习式规划方法大多采用确定性回归范式，直接回归未来轨迹或控制信号。这隐含假设环境和动作之间存在确定性关系，但现实驾驶并非如此。当可行解空间非凸时，确定性回归容易输出多个合理动作之间的“平均动作”，这可能导致安全问题。此外，确定性回归倾向于输出训练数据中占比最高的 dominant trajectory，例如停车或直行，从而损害复杂场景中的规划表现。

为了解决这一问题，VADv2 提出概率规划。它将规划策略建模为环境条件下的非平稳随机过程 `p(a|o)`，其中 `o` 是历史和当前驾驶环境观测，`a` 是候选规划动作。相比确定性建模，概率建模可以捕获规划不确定性，并产生更准确、更安全的规划。

规划动作空间是高维连续时空空间，直接拟合并不现实。VADv2 将连续规划动作空间离散化为一个较大的 planning vocabulary。具体做法是收集驾驶示范中的轨迹，并用最远轨迹采样选择 N 条代表性轨迹作为规划词表。模型学习每个候选轨迹在当前环境下的概率分布。

概率规划还有两个额外优势：第一，它不仅监督正样本动作，也能对词表中的所有候选动作提供监督，使训练信号更丰富；第二，它在推理阶段更灵活，可以输出多模态规划结果，也方便与规则或优化式规划方法结合。

## 2 相关工作

**感知与 BEV 表示**：LSS、BEVFormer 等将多视角图像编码到 BEV，HDMapNet、VectorMapNet、MapTR、LaneGAP 等研究向量化地图表示。

**运动预测**：传统方法使用历史轨迹和 HD map，近年端到端方法联合感知与预测；部分方法使用 GMM 回归多模态轨迹，但模式数量有限。

**端到端规划**：CILRS、LBC、Transfuser、ST-P3、UniAD、VAD 等方法从传感器输入学习规划。多数方法仍是确定性轨迹回归，难以表达动作分布的不确定性。

**概率规划**：VADv2 的重点是把规划从确定性回归改成概率分布建模，以覆盖非凸、多模态动作空间。

## 3 方法

### 3.1 场景编码器

VADv2 继承 VAD 的向量化思想，以流式方式接收多视角图像序列。模型将传感器数据编码为环境 token，包括图像 token、智能体 token、地图 token、交通元素 token 等。规划 token 与这些环境 token 交互，学习动态交通参与者、静态道路结构和交通信号信息。

### 3.2 概率规划

VADv2 不直接回归一条轨迹，而是构造一个 planning vocabulary。词表中的每个动作是一条代表性未来轨迹。模型输出候选轨迹上的概率分布，然后从分布中采样或选择动作。

论文将规划策略写作：

```text
p(a | o)
```

其中 `o` 表示环境观测，`a` 表示候选动作。通过这种概率分布，模型可以表达同一场景下多种合理驾驶意图。

### 3.3 训练

训练目标包含 distribution loss 和 conflict loss。distribution loss 让模型学习专家动作在 planning vocabulary 上的概率分布；conflict loss 通过引入驾驶先验，对冲突或不安全候选动作施加约束。场景 token 的监督则来自交通参与者、地图和交通元素等训练标签。

### 3.4 推理

推理时，模型根据当前环境输出动作概率分布，并采样一个动作控制车辆。概率规划也允许输出多模态候选轨迹，供后续规则或优化模块进一步选择。论文强调，VADv2 可以在完全端到端设置下稳定运行，即使不使用 rule-based wrapper 也能完成闭环驾驶。

## 4 实验

### 4.1 数据集与设置

论文使用 CARLA simulator 进行闭环评估，采用 Town05 Long 和 Town05 Short benchmark。Town05 Long 包含 10 条约 1 km 的路线，评估综合驾驶能力；Town05 Short 包含 32 条约 70 m 的路线，更关注特定场景能力，例如路口前变道。

训练数据由 CARLA 官方 autonomous agent 在 Town03、Town04、Town06、Town07、Town10 中随机生成路线采集，采样频率为 2 Hz，共约 300 万帧。每帧保存 6 相机环视图像、交通信号、其他交通参与者信息和自车状态。地图信息仅在训练阶段作为 ground truth 使用，闭环评估时 VADv2 不使用 HD map。

### 4.2 指标

闭环指标使用 CARLA 官方指标：Route Completion 表示完成路线比例；Infraction Score 表示违规程度，红灯、碰撞等违规会降低该分数；Driving Score 是 Route Completion 与 Infraction Score 的乘积，是主指标。开环消融使用 L2 distance 和 collision rate。

### 4.3 主要结果

在 Town05 Long 上，VADv2 达到 85.1 Driving Score、98.4 Route Completion、0.87 Infraction Score。相比使用相机加 LiDAR 的 DriveMLM，Driving Score 提升 9.0；相比此前纯相机方法，优势更大。

| 方法 | 输入 | Driving Score | Route Completion | Infraction Score |
|---|---|---:|---:|---:|
| Roach | C | 41.6 | 96.4 | 0.43 |
| ThinkTwice | C+L | 70.9 | 95.5 | 0.75 |
| DriveAdapter+TCP | C+L | 71.9 | 97.3 | 0.74 |
| DriveMLM | C+L | 76.1 | 98.1 | 0.78 |
| VADv2 | C | 85.1 | 98.4 | 0.87 |

在 Town05 Short 上，VADv2 达到 89.7 Driving Score 和 93.0 Route Completion，相比 VAD 的 64.3 DS、87.3 RC 有显著提升。

### 4.4 消融实验

消融实验显示，distribution loss 对规划精度非常关键；没有它时，L2 和碰撞率都很差。conflict loss 也提供重要驾驶先验。去掉 agent token、map token、traffic element token 或 image token 任意一种，性能都会下降。完整设计取得最佳结果。

### 4.5 可视化

可视化结果显示，VADv2 可以在不同场景下生成多模态规划轨迹，并根据场景选择合理动作。例如在跟车、变道、交互通过等场景中，概率分布能表达不同驾驶意图，而不是只输出单一平均轨迹。

## 5 结论

VADv2 提出概率规划，用动作分布建模自动驾驶规划的不确定性。它将连续轨迹空间离散为 planning vocabulary，从大规模驾驶示范中学习环境条件动作分布，并通过采样控制车辆。实验表明，仅使用相机输入，VADv2 在 CARLA Town05 闭环 benchmark 上显著优于已有方法，证明概率规划是端到端驾驶中处理不确定性的有效方向。

## 论文解读问答

### Q1 论文试图解决什么问题？

论文试图解决端到端规划中的不确定性和多模态问题。确定性轨迹回归会把多个合理动作平均化，尤其在变道、让行、交互通行等非凸决策场景中容易产生不安全动作。VADv2 希望通过概率规划学习动作分布，而不是只预测一条轨迹。

### Q2 这是否是一个新的问题？

不是全新问题。规划不确定性、多模态预测和行为选择长期存在。但在端到端自动驾驶规划中，明确把连续规划动作空间离散为大型 planning vocabulary，并学习环境条件动作概率分布，是该论文的主要新意。

### Q3 这篇文章要验证一个什么科学假设？

核心假设是：自动驾驶规划不是确定性映射，而是条件随机过程。相比单轨迹回归，概率规划能更好地表示多种合理行为，并在闭环驾驶中带来更高安全性和任务完成率。

### Q4 有哪些相关研究？如何归类？谁是这一课题在领域内值得关注的研究员？

相关研究包括 BEV 感知与向量化地图、端到端规划、运动预测中的多模态建模、概率式/采样式规划。代表方法包括 LSS、BEVFormer、MapTR、Transfuser、ST-P3、UniAD、VAD，以及使用 GMM 的多模态预测方法。

值得关注的研究员包括 Andreas Geiger、Kashyap Chitta、Marco Pavone、Raquel Urtasun、Xinggang Wang、Wenyu Liu，以及 CARLA Leaderboard、Transfuser、UniAD、VAD 系列团队。

### Q5 论文中提到的解决方案之关键是什么？

关键是把规划动作空间离散成代表性轨迹词表，并学习 `p(a|o)`。模型不再输出确定性轨迹，而是输出候选动作概率分布，再采样或选择动作。distribution loss、conflict loss 和多类场景 token 是实现概率规划的关键组件。

### Q6 论文中的实验是如何设计的？

论文使用 CARLA Town05 Long 和 Town05 Short 做闭环评估，并与 CILRS、LBC、Roach、Transfuser、ST-P3、VAD、DriveMLM 等方法比较；使用 Driving Score、Route Completion 和 Infraction Score 作为主指标；消融实验用开环 L2 和 collision rate 验证 loss 与 token 设计。

### Q7 用于定量评估的数据集是什么？代码有没有开源？

定量评估使用 CARLA Town05 Long 和 Town05 Short。训练数据由 CARLA 官方 autonomous agent 在多个 Town 中采集，约 300 万帧。论文给出项目页 <https://hgao-cv.github.io/VADv2>，并给出相关代码入口 <https://github.com/hustvl/VAD>；公开检索也能看到 VADv2 相关代码仓库。

### Q8 论文中的实验及结果有没有很好地支持需要验证的科学假设？

支持较强。VADv2 在 Town05 Long 和 Short 上显著超过确定性端到端方法，说明概率规划确实改善了闭环行为。消融也显示 distribution loss 和 conflict loss 有必要。不过评估主要集中在 CARLA 仿真，真实数据集和真实车闭环还需要进一步验证。

### Q9 这篇论文到底有什么贡献？

贡献是提出概率规划范式，将端到端驾驶从确定性轨迹回归转为动作分布学习；构造 planning vocabulary 来离散连续动作空间；提出 VADv2 模型并在 CARLA Town05 上取得强闭环结果；展示了纯视觉端到端模型在复杂闭环场景中稳定运行的潜力。
