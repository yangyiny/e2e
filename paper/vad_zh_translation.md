# VAD：用于高效自动驾驶的向量化场景表示

原文：`Jiang 等 - 2023 - VAD Vectorized Scene Representation for Efficient Autonomous Driving.pdf`  
作者：Bo Jiang、Shaoyu Chen、Qing Xu、Bencheng Liao、Jiajie Chen、Helong Zhou、Qian Zhang、Wenyu Liu、Chang Huang、Xinggang Wang  
机构：华中科技大学、地平线  
代码：<https://github.com/hustvl/VAD>

## 摘要

自动驾驶需要对周围环境进行全面理解，才能可靠地完成轨迹规划。已有方法通常依赖稠密栅格化场景表示，例如智能体 occupancy、语义地图、flow map、cost map 等，再基于这些表示进行规划。这类方法计算成本较高，并且容易丢失实例级结构信息。

本文提出 VAD，一个端到端向量化自动驾驶范式。VAD 将驾驶场景建模为完全向量化的表示。该范式有两个主要优点：第一，它使用向量化的智能体运动和地图元素作为显式实例级规划约束，从而有效提升规划安全性；第二，它摆脱了计算密集的栅格化表示和手工后处理步骤，运行速度明显更快。

在 nuScenes 数据集上，VAD 取得了当时最优的端到端规划性能。VAD-Base 将平均碰撞率降低 29.0%，并比此前最优方法快 2.5 倍；轻量版 VAD-Tiny 在保持相近规划性能的同时，把推理速度提升到最高 9.3 倍。作者认为，这种性能和效率对真实自动驾驶系统部署非常关键。

## 1 引言

自动驾驶既要求充分的场景理解以保证安全，也要求较高效率以支持真实部署。传统自动驾驶方法采用模块化范式，将感知和规划拆成独立模块。其缺点是规划模块无法访问原始传感器数据，只能依赖前级感知结果；感知误差会传递到规划阶段，并且规划阶段难以识别和修复这些错误。

端到端自动驾驶方法试图用一个整体模型从传感器输入直接输出感知和规划结果。有些方法不显式学习场景表示，直接从传感器数据回归轨迹，这种方式可解释性较弱，也较难优化。更多方法先把传感器数据转换为栅格化场景表示，再用于规划。栅格表示直观，但计算量大，而且会丢失道路、车道、交通参与者等对象的实例级结构。

VAD 的核心主张是：自动驾驶可以完全基于向量化场景表示完成。向量化地图由边界向量和车道向量表示，提供交通流、可行驶边界、车道方向等道路结构信息；向量化智能体运动表示其他交通参与者的未来运动，为避碰提供实例级约束。这些信息不仅更轻量，也更贴近规划所需的结构。

VAD 同时以隐式和显式方式利用向量化信息。隐式方面，模型使用 map queries 和 agent queries 从传感器数据中学习地图和智能体运动特征，并通过 query interaction 将其传递给规划模块。显式方面，模型提出三类向量化规划约束：自车与智能体碰撞约束、自车越界约束、自车车道方向约束。这些约束改善规划安全性，同时不会引入很大计算开销。

## 2 相关工作

**感知**：论文讨论了相机 3D 检测、BEV 表示和在线地图构建。DETR3D、PETR 使用查询机制完成 3D 检测；LSS、BEVFormer、BEVerse 等推动了 BEV 表示；HDMapNet、VectorMapNet、MapTR、LaneGAP 等将地图元素向量化。

**运动预测**：传统运动预测常用历史轨迹和 HD map 作为输入，部分方法将场景渲染为 BEV 图像，也有方法使用图网络或 Transformer 处理向量化表示。VAD 将运动预测作为端到端驾驶系统的一部分，预测 agent-level future motion，并将结果用于规划约束。

**规划**：传统规划通常依赖规则、搜索或优化；端到端方法则从传感器输入学习规划策略。ST-P3、UniAD 等方法依赖 occupancy、语义地图等栅格化中间表示。VAD 与这些方法不同，它用向量化地图和向量化运动作为主要场景表示，并直接构造实例级约束。

## 3 方法

### 3.1 向量化场景学习

VAD 的整体流程包含四个阶段：图像 backbone 提取多视角图像特征；BEV encoder 将图像信息转换为 BEV queries；map decoder 与 motion decoder 分别输出向量化地图和智能体运动；planning module 基于自车 query、agent query、map query 的交互生成自车未来轨迹。

向量化地图由边界向量和车道向量组成。边界向量刻画道路或可行驶区域边界，车道向量刻画车道中心线或方向信息。智能体运动则以 agent motion vector 表示，包含目标检测属性与未来运动轨迹。

### 3.2 通过交互进行规划

规划模块并不是只用一个 ego query 直接回归轨迹，而是让 ego query 与 agent queries、map queries 交互。ego-agent interaction 使自车规划关注其他交通参与者的运动；ego-map interaction 使自车规划关注道路拓扑、边界和车道方向。通过这种 query interaction，向量化场景信息以隐式方式影响最终轨迹。

### 3.3 向量化规划约束

VAD 提出三种显式规划约束。

**自车-智能体碰撞约束**：要求规划轨迹与其他动态智能体保持安全距离，同时考虑纵向和横向安全阈值。

**自车-边界越界约束**：基于向量化道路边界，惩罚靠近或越过边界的规划点，将轨迹推离不可行驶边界。

**自车-车道方向约束**：在每个未来时刻，寻找规划点附近的车道向量，并约束自车运动方向与车道方向保持一致。

### 3.4 端到端学习目标

训练目标由三部分组成：向量化地图学习损失、向量化运动预测损失、规划相关损失。规划部分包含 imitation learning loss，以及上述碰撞、越界、车道方向三类约束。总损失是这些损失的加权和：

```text
L = w1 Lmap + w2 Lmot + w3 Lcol + w4 Lbd + w5 Ldir + w6 Limi
```

其中 `Limi` 是预测自车轨迹与专家轨迹之间的 L1 imitation loss。

## 4 实验

### 4.1 数据集与指标

主要实验在 nuScenes 上进行。nuScenes 包含 1000 个约 20 秒的驾驶场景，关键帧以 2 Hz 标注。论文使用 Displacement Error 和 Collision Rate 评估开环规划。闭环实验使用 CARLA Town05 benchmark，指标为 Route Completion 和 Driving Score。

### 4.2 主要结果

在 nuScenes 开环评估中，VAD-Base 的平均 L2 误差为 0.72 m，平均碰撞率为 0.22%，FPS 为 4.5；UniAD 的平均 L2 误差为 1.03 m，平均碰撞率为 0.31%，FPS 为 1.8。VAD-Tiny 速度达到 16.8 FPS，平均 L2 误差为 0.78 m。

| 方法 | Avg L2 m | Avg Collision % | FPS |
|---|---:|---:|---:|
| ST-P3 | 2.11 | 0.71 | 1.6 |
| UniAD | 1.03 | 0.31 | 1.8 |
| VAD-Tiny | 0.78 | 0.38 | 16.8 |
| VAD-Base | 0.72 | 0.22 | 4.5 |
| VAD-Tiny + ego status | 0.41 | 0.16 | 16.8 |
| VAD-Base + ego status | 0.37 | 0.14 | 4.5 |

在 CARLA Town05 闭环评估中，VAD-Base 在 Town05 Short 上达到 64.29 DS、87.26 RC；在 Town05 Long 上达到 30.31 DS、75.20 RC。它优于多数纯视觉端到端方法，但在 Town05 Long 的 Driving Score 上低于使用 LiDAR 的 Transfuser。

### 4.3 消融实验

消融结果显示，agent interaction、map interaction 和三类向量化约束都对性能有贡献。完整模型达到 Avg L2 0.72 m、Avg Collision 0.22%。只使用向量化表示但不加方向/越界约束时，平均碰撞率为 0.26%；加入方向和越界约束后，平均碰撞率降为 0.22%。

### 4.4 运行效率

VAD-Tiny 的总延迟为 59.5 ms，其中 backbone 占 39.0%，BEV encoder 占 20.7%，motion module 占 19.3%，map module 占 15.3%，planning module 仅占 5.7%。这说明向量化规划模块本身较轻量，主要耗时仍在视觉特征与 BEV 编码上。

## 5 结论

VAD 提出了一种完全向量化的端到端自动驾驶范式。它用向量化地图和智能体运动替代栅格化场景表示，并通过 query interaction 和显式规划约束提升规划安全性。实验表明，VAD 在 nuScenes 上同时取得较高规划性能和显著速度优势，说明向量化场景表示是端到端自动驾驶的一条有效路线。

## 论文解读问答

### Q1 论文试图解决什么问题？

论文试图解决端到端自动驾驶中过度依赖稠密栅格化场景表示的问题。栅格表示计算成本高，并且容易丢失实例级结构信息，导致规划效率低、安全约束不够直接。VAD 希望用向量化地图和向量化智能体运动来支撑高效、安全、可解释的规划。

### Q2 这是否是一个新的问题？

不是全新问题。自动驾驶规划中的场景表示、安全约束和效率问题一直存在。但本文的新意在于，将场景理解、运动预测和规划统一放在完全向量化表示下，并把向量化结果直接作为规划约束，而不是先转成栅格图。

### Q3 这篇文章要验证一个什么科学假设？

核心假设是：相比栅格化表示，向量化场景表示更适合规划，因为它保留实例级结构信息，能形成更直接的安全约束，并且计算更高效。因此，基于向量化表示的端到端模型可以同时提升规划精度、安全性和速度。

### Q4 有哪些相关研究？如何归类？谁是这一课题在领域内值得关注的研究员？

相关研究可分为四类：BEV 感知与 3D 检测，如 LSS、BEVFormer、DETR3D、PETR；在线向量化地图，如 HDMapNet、VectorMapNet、MapTR、LaneGAP；端到端规划，如 ST-P3、UniAD、Transfuser；运动预测与多模态预测，如 VectorNet、TNT、DenseTNT 等。

值得关注的研究员包括 Andreas Geiger、Raquel Urtasun、Marco Pavone、Kashyap Chitta、Hongyang Li、Xinggang Wang、Wenyu Liu，以及 MapTR、UniAD、VAD 系列作者团队。

### Q5 论文中提到的解决方案之关键是什么？

关键是用向量化 scene representation 替代栅格化 representation，并把它用于两层规划信号：一层是 ego query 与 agent/map query 的隐式交互；另一层是碰撞、越界、车道方向三个显式向量化约束。

### Q6 论文中的实验是如何设计的？

论文在 nuScenes 上做开环规划评估，使用 L2 displacement error 和 collision rate；在 CARLA Town05 上做闭环评估，使用 Driving Score 和 Route Completion；同时通过消融实验验证 query interaction、向量化约束、地图表示形式和运行效率。

### Q7 用于定量评估的数据集是什么？代码有没有开源？

定量评估主要使用 nuScenes，闭环评估使用 CARLA Town05。代码和模型已开源，论文给出的地址是 <https://github.com/hustvl/VAD>。

### Q8 论文中的实验及结果有没有很好地支持需要验证的科学假设？

总体支持较好。VAD 在 nuScenes 上同时降低 L2 误差和碰撞率，并且速度明显快于 UniAD；消融实验也显示向量化约束能进一步降低碰撞率。不过闭环 CARLA 结果相对复杂，VAD 并非所有指标都优于使用 LiDAR 的方法，因此“向量化范式全面优于其他范式”还需要更多闭环和真实车端验证。

### Q9 这篇论文到底有什么贡献？

贡献是提出了端到端自动驾驶的向量化范式 VAD；设计了向量化地图、智能体运动和自车规划的统一框架；提出三类实例级向量化规划约束；在 nuScenes 上证明该范式能兼顾规划性能和推理效率；并开源代码推动后续研究。
