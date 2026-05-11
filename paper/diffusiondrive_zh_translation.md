# DiffusionDrive：用于端到端自动驾驶的截断扩散模型

原文：`Liao 等 - 2025 - DiffusionDrive Truncated Diffusion Model for End-to-End Autonomous Driving.pdf`  
作者：Bencheng Liao、Shaoyu Chen、Haoran Yin、Bo Jiang、Cheng Wang、Sixu Yan、Xinbang Zhang、Xiangyu Li、Ying Zhang、Qian Zhang、Xinggang Wang  
机构：华中科技大学、地平线  
代码、模型与 Demo：<https://github.com/hustvl/DiffusionDrive>

## 摘要

扩散模型近年来成为机器人策略学习中的强大生成技术，能够建模多模态动作分布。将这种能力用于端到端自动驾驶是一个有前景的方向。但机器人扩散策略通常需要大量去噪步骤，而交通场景更加动态、开放，这给实时生成多样化驾驶动作带来挑战。

为了解决这些问题，本文提出截断扩散策略。它引入先验多模态 anchor，并截断扩散 schedule，使模型学习从 anchored Gaussian distribution 去噪到多模态驾驶动作分布。作者还设计了高效的级联扩散 decoder，以增强轨迹候选与条件场景上下文之间的交互。

DiffusionDrive 相比 vanilla diffusion policy 将去噪步数减少 10 倍，只需 2 步即可生成更高质量、更具多样性的轨迹。在规划导向的 NAVSIM 数据集上，使用相同 ResNet-34 backbone，DiffusionDrive 达到 88.1 PDMS，并在 NVIDIA 4090 上以 45 FPS 实时运行。定性结果表明，它能在复杂场景中稳健生成多样且合理的驾驶动作。

## 1 引言

端到端自动驾驶希望从原始传感器输入直接学习驾驶策略。主流端到端规划器，如 Transfuser、UniAD、VAD，通常用 ego query 回归一条单模态轨迹。这种范式无法处理驾驶行为固有的不确定性和多模态性。

VADv2 引入大型固定 anchor 词表，例如 4096 或 8192 条候选轨迹，通过对 anchor 打分来覆盖更多驾驶行为。但这种固定词表受 anchor 数量和质量限制，面对词表外场景容易失败；同时，大量 anchor 也带来实时计算压力。

扩散模型可以从高斯噪声中通过迭代去噪生成多模态动作，因此看起来适合自动驾驶。作者首先将 vanilla robotic diffusion policy 接到 Transfuser 上，得到 TransfuserDP。虽然性能有所提升，但出现两个问题：第一，DDIM 扩散策略需要约 20 步去噪，推理开销过大；第二，从不同高斯噪声采样得到的轨迹严重重叠，说明直接使用 vanilla diffusion 在交通场景中容易模式坍缩，难以产生有效多样性。

作者的关键观察是：人类驾驶并不是从完全随机噪声中生成动作，而是遵循一些先验驾驶模式，并根据实时交通条件动态调整。因此，DiffusionDrive 将这些先验驾驶模式嵌入扩散策略中：把高斯分布划分成围绕先验 anchors 的多个子高斯分布，即 anchored Gaussian distribution。通过截断扩散 schedule，只在 anchor 周围引入少量噪声，模型从更合理的初始分布开始去噪，因此只需 2 步即可满足实时驾驶需求。

## 2 相关工作

**端到端自动驾驶**：UniAD 将感知、预测和规划统一为可微框架；VAD 使用向量化场景表示提升效率；Transfuser 使用相机和 LiDAR 融合进行规划。这些方法多为单轨迹回归或固定 anchor 选择，难以同时兼顾多样性和实时性。

**多模态规划与 anchor 方法**：VADv2 和 Hydra-MDP 使用大规模 anchor vocabulary 表示动作空间，但性能依赖 anchor 数量与覆盖质量，且计算成本较高。

**扩散策略**：Diffusion Policy 等机器人方法通过迭代去噪生成动作，但直接迁移到自动驾驶会遇到去噪步数多、轨迹重叠、模式坍缩等问题。

**闭环规划评估**：NAVSIM/NA VSIM 基于 nuPlan，使用 PDM Score 对规划输出进行闭环或非反应式仿真评估，已成为规划导向自动驾驶 benchmark。

## 3 方法

### 3.1 预备知识

vanilla diffusion policy 从标准高斯噪声开始，逐步去噪得到动作。训练时通过前向扩散给真实动作加噪，模型学习预测去噪后的动作；推理时从随机噪声出发，经过多步 DDIM/DDPM 反向过程得到动作。

在自动驾驶中，动作是未来轨迹。DiffusionDrive 认为直接从随机高斯噪声生成轨迹不符合驾驶先验，也导致多样性差和计算成本高。

### 3.2 直接迁移扩散策略的问题

作者构建 TransfuserDP，将 Transfuser 的确定性 MLP head 替换为条件扩散模型。它比 Transfuser 提升 0.6 PDMS，但代价是去噪 20 步，总规划模块时间达到 130 ms，FPS 降到 7。此外，定性结果显示不同噪声生成的轨迹高度重叠，说明 vanilla diffusion 没有产生真正多样化行为。

### 3.3 截断扩散

DiffusionDrive 的核心是 anchored Gaussian distribution。给定一组先验 anchor 轨迹，模型不从标准高斯噪声开始，而是在每个 anchor 附近加入少量高斯噪声，形成多个 anchor-centered 子分布。训练和推理都从这些更接近合理驾驶行为的初始点开始。

这种做法有两个好处：

1. 多个 anchor 对应不同驾驶意图，天然提升多模态覆盖；
2. 初始状态接近真实轨迹分布，因此可以截断扩散 schedule，大幅减少去噪步数。

DiffusionDrive 训练时用 20 个聚类 anchor，并将扩散 schedule 截断到 50/1000；推理时只用 2 个去噪步骤。

### 3.4 架构

DiffusionDrive 可接入已有端到端规划器的感知模块，并支持不同传感器输入。论文实验中沿用 Transfuser 的 ResNet-34 backbone 与感知设置。

扩散 decoder 接收从 anchored Gaussian distribution 采样的 noisy trajectories。它先基于轨迹坐标与 BEV 或 PV 特征做 deformable spatial cross-attention，然后与感知模块输出的 agent/map queries 做 cross-attention，再经过 FFN 和 timestep modulation。每个 decoder layer 预测置信度和相对初始 noisy trajectory 的 offset。

作者还引入 cascade diffusion decoder：在每个去噪步骤内部堆叠多个 decoder layer，逐步细化轨迹重构。推理时选择置信度最高的轨迹作为输出。

### 3.5 训练与推理

训练目标由轨迹重构损失和二分类置信度损失组成。每个 anchor 依据是否接近专家轨迹被标为正/负样本。正样本参与 L1 reconstruction loss，所有样本参与 BCE classification loss。

推理时，模型可灵活设定采样轨迹数量 `N_infer`。尽管训练使用固定数量 anchors，推理可以根据算力和应用需求调整候选轨迹数。

## 4 实验

### 4.1 数据集

主要评估在 NA VSIM navtest 上进行。NA VSIM 基于 OpenScene/nuPlan，面向真实世界规划场景，包含 8 个相机的 360 度视野和 5 个 LiDAR 合并点云。指标为 PDMS，由 NC、DAC、TTC、comfort 和 ego progress 组成。

论文还在 nuScenes 上做开环评估，使用 L2 error、collision rate 和 FPS。

### 4.2 实现细节

为了公平比较，作者采用与 Transfuser 相同的感知模块和 ResNet-34 backbone。输入为三个裁剪缩放后的前向相机图像，拼接为 1024 x 256 图像，以及栅格化 BEV LiDAR 表示。模型在 navtrain 上从零训练 100 epoch，使用 AdamW，总 batch size 512，在 8 张 NVIDIA 4090 上训练。输出是 4 秒 8 waypoint 轨迹。

### 4.3 NA VSIM 主要结果

DiffusionDrive 在 NA VSIM navtest 上达到 88.1 PDMS。相比 Transfuser 的 84.0，提升 4.1；相比 VADv2-V8192 的 80.9，提升 7.2，同时 anchor 数量从 8192 降到 20；相比 Hydra-MDP-V8192-W-EP 的 86.5，也提升 1.6。

| 方法 | 输入 | Backbone | Anchor | NC | DAC | TTC | Comf. | EP | PDMS |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| UniAD | Camera | ResNet-34 | 0 | 97.8 | 91.9 | 92.9 | 100 | 78.8 | 83.4 |
| Transfuser | C&L | ResNet-34 | 0 | 97.7 | 92.8 | 92.8 | 100 | 79.2 | 84.0 |
| Hydra-MDP-V8192-W-EP | C&L | ResNet-34 | 8192 | 98.3 | 96.0 | 94.6 | 100 | 78.7 | 86.5 |
| DiffusionDrive | C&L | ResNet-34 | 20 | 98.2 | 96.2 | 94.7 | 100 | 82.2 | 88.1 |

### 4.4 从 Transfuser 到 DiffusionDrive 的路线图

| 方法 | PDMS | 去噪步数 | 规划模块总时间 | 参数 | FPS |
|---|---:|---:|---:|---:|---:|
| Transfuser | 84.0 | 1 | 0.2 ms | 56M | 60 |
| TransfuserDP | 84.6 | 20 | 130.0 ms | 101M | 7 |
| TransfuserTD | 85.7 | 2 | 13.8 ms | 102M | 27 |
| DiffusionDrive | 88.1 | 2 | 7.6 ms | 60M | 45 |

这张表说明：vanilla diffusion 有性能收益但开销过高；截断扩散显著减少步数；专门设计的 diffusion decoder 进一步提升性能并降低开销。

### 4.5 消融实验

消融显示，spatial cross-attention、agent/map cross-attention 和 cascade decoder 都有贡献。完整模型达到 88.1 PDMS。去噪步数方面，1 步为 87.9，2 步为 88.1，3 步仍为 88.1，说明 2 步已足够。cascade stages 从 1 到 2 提升明显，4 stages 只有微小增益但参数更多。候选噪声数 `N_infer` 从 10 到 20 有明显提升，40 只带来极小收益。

### 4.6 nuScenes 开环结果

在 nuScenes 上，DiffusionDrive 使用 ResNet-50 camera-only 设置达到 Avg L2 0.57、Avg collision 0.08、FPS 8.2。相比 VAD 的 Avg L2 0.72、collision 0.22，误差降低 20.8%，碰撞率降低 63.6%。

## 5 结论

DiffusionDrive 将扩散模型引入端到端自动驾驶，并提出截断扩散策略解决 vanilla diffusion 在交通场景中的高延迟和模式坍缩问题。通过 anchored Gaussian distribution 和级联扩散 decoder，它能在 2 步去噪内生成多样且高质量的轨迹。在 NA VSIM 上达到 88.1 PDMS，并以 45 FPS 实时运行。

## 论文解读问答

### Q1 论文试图解决什么问题？

论文试图解决端到端自动驾驶中多模态轨迹生成与实时性之间的矛盾。单轨迹回归无法表达多种合理驾驶行为；大型 anchor vocabulary 受词表限制且计算重；vanilla diffusion 虽能生成多模态动作，但去噪步数多、容易轨迹重叠，不适合实时驾驶。

### Q2 这是否是一个新的问题？

不是全新问题。多模态规划、动作不确定性和实时推理一直存在。但将扩散策略系统地迁移到端到端自动驾驶，并针对交通场景提出截断扩散和 anchored Gaussian distribution，是本文的新问题表述和解决方向。

### Q3 这篇文章要验证一个什么科学假设？

核心假设是：驾驶动作生成不应从完全随机高斯噪声开始，而应从包含驾驶先验的 anchor-centered 分布开始。这样既能保持扩散模型的多模态表达能力，又能大幅减少去噪步数并提升生成质量。

### Q4 有哪些相关研究？如何归类？谁是这一课题在领域内值得关注的研究员？

相关研究可分为端到端规划、固定 anchor/词表式规划、扩散策略和闭环规划 benchmark。代表方法包括 Transfuser、UniAD、VAD、VADv2、Hydra-MDP、Diffusion Policy、Diffuser、NAVSIM/PDM 系列。

值得关注的研究员包括 Shuran Song、Sergey Levine、Michael Janner、Yilun Du、Marco Pavone、Andreas Geiger、Kashyap Chitta、Xinggang Wang、Wenyu Liu，以及 DiffusionDrive/VAD 系列作者团队。

### Q5 论文中提到的解决方案之关键是什么？

关键是截断扩散策略：用少量先验 anchor 构造 anchored Gaussian distribution，从 anchor 附近开始去噪，而不是从标准高斯噪声开始；再配合专门设计的 cascade diffusion decoder，使模型能以 2 步生成高质量多模态轨迹。

### Q6 论文中的实验是如何设计的？

实验先在 NA VSIM navtest 上进行闭环指标比较，评估 PDMS 及其子指标；再用路线图实验比较 Transfuser、vanilla diffusion、truncated diffusion 和完整 DiffusionDrive；之后做 decoder 组件、去噪步数、cascade stage、采样数量消融；最后在 nuScenes 上做开环 L2、collision rate 和 FPS 评估。

### Q7 用于定量评估的数据集是什么？代码有没有开源？

主要定量评估数据集是 NA VSIM navtest，指标为 PDMS；补充评估使用 nuScenes。代码、模型与 Demo 已开源，论文给出的地址是 <https://github.com/hustvl/DiffusionDrive>。

### Q8 论文中的实验及结果有没有很好地支持需要验证的科学假设？

支持较好。TransfuserDP 说明 vanilla diffusion 虽有增益但开销巨大；TransfuserTD 和 DiffusionDrive 说明截断扩散可以把去噪步数降到 2，并提升 PDMS；消融也显示 2 步已接近饱和。不过该方法仍依赖 anchor 先验，真实闭环部署中的鲁棒性和 OOD 泛化还需要更多验证。

### Q9 这篇论文到底有什么贡献？

贡献是首次系统性地将扩散模型用于端到端自动驾驶规划；提出 anchored Gaussian distribution 与截断扩散策略，解决 vanilla diffusion 的实时性和模式坍缩问题；设计高效 cascade diffusion decoder；在 NA VSIM 和 nuScenes 上取得强结果，并开源代码和模型。
