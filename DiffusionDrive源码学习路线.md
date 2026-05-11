# DiffusionDrive 源码学习路线

目标：从已经跑通第一阶段环境出发，逐步吃透 DiffusionDrive 的源码、论文实现关系、训练链路和服务器复现实验流程。建议每读完一节就做一次小实验或断点观察，不要只看代码。

---

## 0. 当前学习目标

- [ ] 能说清楚 `diffusiondrive_agent` 如何接入 NAVSIM 的训练、缓存、评估框架。
- [ ] 能从 `run_training.py` 一路追到 `TrajectoryHead.forward_train()`。
- [ ] 能把论文中的 truncated diffusion policy 对应到代码中的 anchor、noise、DDIM scheduler、2-step denoising。
- [ ] 能独立修改配置，在服务器上完成 cache、sanity train、checkpoint evaluation、全量训练。
- [ ] 能做至少 3 个小实验：去噪步数、anchor 数量/路径、`N_infer` 或 decoder 消融。

---

## 1. 代码总入口地图

### 1.1 Hydra 配置入口

核心文件：

- `DiffusionDrive/navsim/planning/script/config/common/agent/diffusiondrive_agent.yaml`
- `DiffusionDrive/navsim/planning/script/config/training/default_training.yaml`
- `DiffusionDrive/docs/train_eval.md`

论文对应：

- 实验设置：第 4.2 节 Implementation Detail。
- 训练 recipe：navtrain 训练、navtest 评估、100 epoch、AdamW、ResNet-34、20 clustered anchors。

你要看懂的问题：

- `agent=diffusiondrive_agent` 到底实例化哪个 Python 类？
- `checkpoint_path` 如何传入 agent？
- `lr=6e-4`、`batch_size=64`、`precision=16-mixed`、`strategy=ddp` 从哪里来？
- `cache_path` 默认指向哪里？

关键代码点：

```yaml
_target_: navsim.agents.diffusiondrive.transfuser_agent.TransfuserAgent
config:
  _target_: navsim.agents.diffusiondrive.transfuser_config.TransfuserConfig
checkpoint_path: null
lr: 6e-4
```

建议断点：

- `navsim/planning/script/run_training.py` 中 `instantiate(cfg.agent)` 后面。
- `navsim/agents/diffusiondrive/transfuser_agent.py` 的 `TransfuserAgent.__init__()`。

检查命令：

```bash
cd DiffusionDrive
python navsim/planning/script/run_training.py \
  agent=diffusiondrive_agent \
  experiment_name=debug_config_print \
  train_test_split=navtrain \
  trainer.params.fast_dev_run=true
```

如果只想看配置，可以先用 Hydra 的 `--cfg job`：

```bash
cd DiffusionDrive
python navsim/planning/script/run_training.py \
  agent=diffusiondrive_agent \
  train_test_split=navtrain \
  --cfg job
```

---

## 2. 第一条主线：Agent 如何接入 NAVSIM

### 2.1 Agent 封装层

核心文件：

- `DiffusionDrive/navsim/agents/diffusiondrive/transfuser_agent.py`

论文对应：

- 论文没有专门讲这一层，它是工程接入层。
- 对应“DiffusionDrive can integrate various existing perception modules and sensor inputs”这类实现支撑。

关键函数：

- `TransfuserAgent.__init__()`
- `TransfuserAgent.init_from_pretrained()`
- `TransfuserAgent.get_sensor_config()`
- `TransfuserAgent.get_feature_builders()`
- `TransfuserAgent.get_target_builders()`
- `TransfuserAgent.forward()`
- `TransfuserAgent.compute_loss()`
- `TransfuserAgent.get_optimizers()`

阅读顺序：

1. `__init__()`：保存 config、学习率、checkpoint，并创建 `TransfuserModel(config)`。
2. `init_from_pretrained()`：如果给了 checkpoint，就加载 `checkpoint['state_dict']`，并去掉 `agent.` 前缀。
3. `get_sensor_config()`：目前 `SensorConfig.build_all_sensors(include=[3])`，对应使用前向相机子集。
4. `get_feature_builders()`：返回 `TransfuserFeatureBuilder`，负责构建相机、LiDAR、ego status 等模型输入。
5. `get_target_builders()`：返回 `TransfuserTargetBuilder`，负责构建轨迹、检测框、BEV 语义等监督信号。
6. `forward()`：把 features/targets 交给真正的模型 `V2TransfuserModel`。
7. `compute_loss()`：调用 `transfuser_loss()` 汇总轨迹、检测、BEV 语义等 loss。
8. `get_coslr_optimizers()`：构建 AdamW 和 WarmupCosLR。

建议断点：

```python
# transfuser_agent.py
def __init__(...):
    ...
    self._transfuser_model = TransfuserModel(config)

def forward(self, features, targets=None):
    return self._transfuser_model(features, targets=targets)

def compute_loss(...):
    return transfuser_loss(targets, predictions, self._config)
```

观察变量：

- `self._config.bkb_path`
- `self._config.plan_anchor_path`
- `features.keys()`
- `targets.keys()`
- `predictions.keys()`
- `loss_dict` 中各项 loss 的量级

你需要最终能回答：

- NAVSIM 训练框架只认识 `AbstractAgent` 接口，DiffusionDrive 通过 `TransfuserAgent` 适配这个接口。
- 真正的模型不是写在 agent 里，而是在 `transfuser_model_v2.py` 里。
- agent 负责“接入框架”，model 负责“前向推理”，loss 文件负责“训练目标”。

---

## 3. 第二条主线：配置和论文超参数

### 3.1 全局配置

核心文件：

- `DiffusionDrive/navsim/agents/diffusiondrive/transfuser_config.py`

论文对应：

- 第 4.2 节 Implementation Detail。
- 表 1、表 2、表 3-6 的模型设置基础。

重点字段：

```python
trajectory_sampling = TrajectorySampling(time_horizon=4, interval_length=0.5)
image_architecture = "resnet34"
lidar_architecture = "resnet34"
bkb_path = "..."
plan_anchor_path = "..."
num_bounding_boxes = 30
tf_d_model = 256
tf_num_layers = 3
tf_num_head = 8
trajectory_weight = 12.0
trajectory_cls_weight = 10.0
trajectory_reg_weight = 8.0
bev_semantic_weight = 14.0
lr_steps = [70]
optimizer_type = "AdamW"
```

服务器训练前必须改：

```python
bkb_path = "/your/server/path/resnet34.a1_in1k/pytorch_model.bin"
plan_anchor_path = "/your/server/path/kmeans_navsim_traj_20.npy"
```

建议小实验：

- [ ] 打印 `config.trajectory_sampling.num_poses`，确认是 8 个 waypoint。
- [ ] 打印 `np.load(plan_anchor_path).shape`，确认 anchor shape 是 `(20, 8, 2)`。
- [ ] 改 `trajectory_cls_weight` 或 `trajectory_reg_weight`，观察 loss 量级变化。

---

## 4. 第三条主线：模型 forward 数据流

### 4.1 主模型结构

核心文件：

- `DiffusionDrive/navsim/agents/diffusiondrive/transfuser_model_v2.py`
- `DiffusionDrive/navsim/agents/diffusiondrive/transfuser_backbone.py`

论文对应：

- 图 4 Overall architecture。
- 第 3.4 节 Architecture。
- 第 4.2 节中“沿用 Transfuser 感知模块和 ResNet-34 backbone”。

关键类：

- `V2TransfuserModel`
- `TransfuserBackbone`
- `AgentHead`
- `TrajectoryHead`

`V2TransfuserModel.forward()` 数据流：

1. 读取输入：

```python
camera_feature = features["camera_feature"]
lidar_feature = features["lidar_feature"]
status_feature = features["status_feature"]
```

2. Backbone 提取 BEV 特征：

```python
bev_feature_upscale, bev_feature, _ = self._backbone(camera_feature, lidar_feature)
```

3. 处理 BEV token 和 ego status token：

```python
bev_feature = self._bev_downscale(bev_feature).flatten(-2, -1)
status_encoding = self._status_encoding(status_feature)
keyval = torch.concatenate([bev_feature, status_encoding[:, None]], dim=1)
```

4. Transformer decoder 输出 trajectory query 和 agent queries：

```python
query_out = self._tf_decoder(query, keyval)
trajectory_query, agents_query = query_out.split(self._query_splits, dim=1)
```

5. 规划头输出轨迹，检测头输出 agent boxes：

```python
trajectory = self._trajectory_head(...)
agents = self._agent_head(agents_query)
```

建议断点：

- `V2TransfuserModel.forward()` 开头：看 `features` shape。
- `self._backbone(...)` 后：看 BEV 特征 shape。
- `query_out.split(...)` 后：看 `trajectory_query.shape` 和 `agents_query.shape`。
- `_trajectory_head(...)` 前后：看 trajectory 输出。

建议记录的 shape：

```text
camera_feature:
lidar_feature:
status_feature:
bev_feature_upscale:
bev_feature:
trajectory_query:
agents_query:
predictions["trajectory"]:
```

---

## 5. 第四条主线：论文核心 TrajectoryHead

### 5.1 截断扩散训练

核心文件：

- `DiffusionDrive/navsim/agents/diffusiondrive/transfuser_model_v2.py`

核心类：

- `TrajectoryHead`

论文对应：

- 第 3.3 节 Truncated Diffusion。
- 公式 (4)：给 anchors 加截断高斯噪声。
- 公式 (5)：decoder 预测分类分数和去噪轨迹。
- 公式 (6)：轨迹重建和分类损失。

关键代码：

```python
self.diffusion_scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    beta_schedule="scaled_linear",
    prediction_type="sample",
)

plan_anchor = np.load(plan_anchor_path)
self.plan_anchor = nn.Parameter(torch.tensor(plan_anchor), requires_grad=False)
```

训练时：

```python
timesteps = torch.randint(0, 50, (bs,), device=device)
noise = torch.randn(odo_info_fut.shape, device=device)
noisy_traj_points = self.diffusion_scheduler.add_noise(
    original_samples=odo_info_fut,
    noise=noise,
    timesteps=timesteps,
)
```

这对应论文里的“training diffusion schedule is truncated by 50/1000”。

训练 forward 流程：

1. `plan_anchor` 扩展到 batch。
2. `norm_odo(plan_anchor)` 归一化轨迹坐标。
3. 从 `[0, 50)` 采样扩散 timestep。
4. 用 DDIM scheduler 给 anchor 加噪。
5. `denorm_odo()` 转回真实坐标。
6. 用 sine embedding 编码 noisy trajectory。
7. 用 time MLP 编码 diffusion timestep。
8. 进入 cascade diffusion decoder。
9. 每一层 decoder 输出 `poses_reg` 和 `poses_cls`。
10. `LossComputer` 按最近 anchor 选择正样本并计算分类/回归损失。

建议断点：

```python
def forward_train(...):
    plan_anchor = self.plan_anchor.unsqueeze(0).repeat(bs,1,1,1)
    timesteps = torch.randint(0, 50, (bs,), device=device)
    noisy_traj_points = ...
    poses_reg_list, poses_cls_list = self.diff_decoder(...)
```

观察变量：

- `plan_anchor.shape`
- `timesteps.min()` / `timesteps.max()`
- `noise.std()`
- `noisy_traj_points[0, :, :, :2]`
- `poses_reg_list[-1].shape`
- `poses_cls_list[-1].shape`

### 5.2 截断扩散推理

论文对应：

- 第 3.3 节 Inference。
- 表 2 中 2 denoising steps。
- 表 4 中 denoising step number 消融。

关键代码：

```python
step_num = 2
self.diffusion_scheduler.set_timesteps(1000, device)
step_ratio = 20 / step_num
roll_timesteps = (np.arange(0, step_num) * step_ratio).round()[::-1]
trunc_timesteps = torch.ones((bs,), device=device, dtype=torch.long) * 8
img = self.diffusion_scheduler.add_noise(original_samples=img, noise=noise, timesteps=trunc_timesteps)
```

推理 forward 流程：

1. 从固定 `plan_anchor` 开始。
2. 在 timestep 8 上加入少量噪声，得到 anchored Gaussian distribution 的样本。
3. `step_num=2`，执行两轮 denoising。
4. 每轮 decoder 预测 `x_start`。
5. 使用 `diffusion_scheduler.step()` 更新样本。
6. 最后用 `poses_cls.argmax()` 选择置信度最高轨迹。

建议断点：

```python
def forward_test(...):
    step_num = 2
    roll_timesteps = ...
    for k in roll_timesteps[:]:
        ...
        img = self.diffusion_scheduler.step(...).prev_sample
```

建议小实验：

- [ ] 把 `step_num = 2` 改成 `1`，验证速度和 PDMS 变化。
- [ ] 把 `step_num = 2` 改成 `3`，验证是否接近论文表 4 的饱和趋势。
- [ ] 把 `trunc_timesteps = 8` 改成其他值，观察轨迹多样性和稳定性。

注意：

- 训练里截断上限是 50。
- 推理里实际起点使用 `trunc_timesteps=8`，denoising roll timesteps 来自 `20 / step_num`。
- 这两个数字值得重点理解，因为它们是代码实现和论文描述之间最容易忽略的细节。

---

## 6. 第五条主线：Cascade Diffusion Decoder

### 6.1 Decoder 层

核心文件：

- `DiffusionDrive/navsim/agents/diffusiondrive/transfuser_model_v2.py`
- `DiffusionDrive/navsim/agents/diffusiondrive/modules/blocks.py`

核心类：

- `CustomTransformerDecoderLayer`
- `CustomTransformerDecoder`
- `DiffMotionPlanningRefinementModule`
- `ModulationLayer`
- `GridSampleCrossBEVAttention`

论文对应：

- 第 3.4 节 Diffusion decoder。
- 表 3 diffusion decoder design choices。
- 表 5 cascade stages。

你要对应起来的模块：

- Spatial cross-attention：`GridSampleCrossBEVAttention`
- Agent/map cross-attention：代码中主要是和 `agents_query`、`ego_query` 交互
- FFN：decoder layer 内部 feed-forward
- Timestep modulation：`ModulationLayer`
- Score 和 trajectory offset：`DiffMotionPlanningRefinementModule`
- Cascade decoder：`CustomTransformerDecoder(diff_decoder_layer, 2)`

建议断点：

- `CustomTransformerDecoder.forward()`
- `CustomTransformerDecoderLayer.forward()`
- `GridSampleCrossBEVAttention.forward()`
- `DiffMotionPlanningRefinementModule.forward()`

建议观察：

- 每一层 decoder 的 `poses_reg` 是否逐步更接近 target。
- `poses_cls` 是否对某些 anchor/mode 给出更高分。
- `traj_feature` 在 modulation 前后数值分布是否稳定。

建议小实验：

- [ ] 将 cascade 层数从 2 改为 1，复现表 5 的趋势。
- [ ] 将 cascade 层数从 2 改为 4，观察参数量、显存、速度变化。
- [ ] 临时跳过某个 cross-attention，观察是否接近表 3 的性能下降。

---

## 7. 第六条主线：Feature / Target 构建

### 7.1 输入输出是什么

核心文件：

- `DiffusionDrive/navsim/agents/diffusiondrive/transfuser_features.py`
- `DiffusionDrive/navsim/planning/training/dataset.py`
- `DiffusionDrive/navsim/common/dataclasses.py`

论文对应：

- 第 4.2 节 Implementation Detail。
- 输入：3 张前向 camera 拼接为 `1024 x 256`，BEV LiDAR raster。
- 输出：4 秒 8 waypoint trajectory，辅助检测和 BEV semantic。

你要看懂的问题：

- `features["camera_feature"]` 从哪里来？
- `features["lidar_feature"]` 是怎么 rasterize 的？
- `features["status_feature"]` 包含哪些 ego status？
- `targets["trajectory"]` 的坐标系是什么？
- `targets["agent_states"]` 和 `targets["bev_semantic_map"]` 如何构建？

建议断点：

- `TransfuserFeatureBuilder.get_features_from_scenario()`
- `TransfuserTargetBuilder.get_targets()`
- `Dataset.__getitem__()`

建议打印：

```python
print(features.keys())
print({k: v.shape for k, v in features.items() if hasattr(v, "shape")})
print(targets.keys())
print({k: v.shape for k, v in targets.items() if hasattr(v, "shape")})
```

---

## 8. 第七条主线：Loss 如何对应论文

### 8.1 总 loss

核心文件：

- `DiffusionDrive/navsim/agents/diffusiondrive/transfuser_loss.py`
- `DiffusionDrive/navsim/agents/diffusiondrive/modules/multimodal_loss.py`

论文对应：

- 公式 (6)：轨迹重建 loss + 分类 loss。
- 第 4.2 节辅助感知任务：3D object detection、2D BEV semantic segmentation。

总 loss：

```python
loss = (
    config.trajectory_weight * trajectory_loss
    + config.diff_loss_weight * diffusion_loss
    + config.agent_class_weight * agent_class_loss
    + config.agent_box_weight * agent_box_loss
    + config.bev_semantic_weight * bev_semantic_loss
)
```

注意：

- 当前 DiffusionDrive 主要使用 `predictions["trajectory_loss"]`，由 `TrajectoryHead.forward_train()` 内部计算。
- `diffusion_loss` 在当前路径下通常是 0，代码保留了字段。
- 检测 loss 使用 Hungarian matching。
- BEV semantic loss 使用 cross entropy。

### 8.2 多模态轨迹 loss

核心逻辑：

```python
dist = torch.linalg.norm(target_traj.unsqueeze(1)[...,:2] - plan_anchor, dim=-1)
mode_idx = torch.argmin(dist.mean(dim=-1), dim=-1)
best_reg = torch.gather(poses_reg, 1, mode_idx)
loss_cls = focal_loss(poses_cls, one_hot(mode_idx))
reg_loss = F.l1_loss(best_reg, target_traj)
```

含义：

- 找到距离专家轨迹最近的 anchor/mode。
- 这个 mode 是正样本。
- 只对最近 mode 的轨迹做回归。
- 对所有 mode 做分类监督。

建议断点：

- `LossComputer.forward()`

观察变量：

- `dist.shape`
- `mode_idx`
- `target_classes_onehot.sum(dim=1)`
- `loss_cls`
- `reg_loss`

你最终要能说清楚：

- 论文里的“closest anchor positive, others negative”在这里通过 `mode_idx` 和 one-hot classification target 实现。
- 轨迹回归只回归与 GT 最近的 mode。

---

## 9. 训练命令清单

### 9.1 环境变量

服务器上建议先明确这几个路径：

```bash
export NAVSIM_DEVKIT_ROOT=/path/to/DiffusionDrive
export OPENSCENE_DATA_ROOT=/path/to/openscene
export NAVSIM_EXP_ROOT=/path/to/experiments/diffusiondrive
```

检查：

```bash
echo $NAVSIM_DEVKIT_ROOT
echo $OPENSCENE_DATA_ROOT
echo $NAVSIM_EXP_ROOT
```

### 9.2 安装额外依赖

```bash
cd DiffusionDrive
conda activate navsim
pip install diffusers einops
pip install -e .
```

如果服务器不能联网，需要提前下载：

- `diffusers`
- `einops`
- `resnet34.a1_in1k/pytorch_model.bin`
- `kmeans_navsim_traj_20.npy`
- 官方 checkpoint：`diffusiondrive_navsim_88p1_PDMS.pth`

### 9.3 Dataset cache

```bash
cd $NAVSIM_DEVKIT_ROOT
python navsim/planning/script/run_dataset_caching.py \
  agent=diffusiondrive_agent \
  experiment_name=training_diffusiondrive_agent \
  train_test_split=navtrain
```

### 9.4 Metric cache

```bash
cd $NAVSIM_DEVKIT_ROOT
python navsim/planning/script/run_metric_caching.py \
  train_test_split=navtest \
  cache.cache_path=$NAVSIM_EXP_ROOT/metric_cache
```

### 9.5 官方 checkpoint 评估

```bash
cd $NAVSIM_DEVKIT_ROOT
export CKPT=/path/to/diffusiondrive_navsim_88p1_PDMS.pth
python navsim/planning/script/run_pdm_score.py \
  train_test_split=navtest \
  agent=diffusiondrive_agent \
  worker=ray_distributed \
  agent.checkpoint_path=$CKPT \
  experiment_name=diffusiondrive_agent_eval
```

### 9.6 Sanity train

先跑一个极小训练，确认链路和显存：

```bash
cd $NAVSIM_DEVKIT_ROOT
python navsim/planning/script/run_training.py \
  agent=diffusiondrive_agent \
  experiment_name=sanity_diffusiondrive_agent \
  train_test_split=navtrain \
  split=trainval \
  trainer.params.max_epochs=1 \
  trainer.params.limit_train_batches=10 \
  trainer.params.limit_val_batches=2 \
  cache_path="${NAVSIM_EXP_ROOT}/training_cache/" \
  use_cache_without_dataset=True \
  force_cache_computation=False
```

### 9.7 全量训练

```bash
cd $NAVSIM_DEVKIT_ROOT
python navsim/planning/script/run_training.py \
  agent=diffusiondrive_agent \
  experiment_name=training_diffusiondrive_agent \
  train_test_split=navtrain \
  split=trainval \
  trainer.params.max_epochs=100 \
  cache_path="${NAVSIM_EXP_ROOT}/training_cache/" \
  use_cache_without_dataset=True \
  force_cache_computation=False
```

---

## 10. 服务器训练注意事项

### 10.1 路径问题

最常见报错来自硬编码路径：

- `bkb_path` 仍然是作者机器路径。
- `plan_anchor_path` 仍然是作者机器路径。
- `OPENSCENE_DATA_ROOT` 没有指到数据根目录。
- `NAVSIM_EXP_ROOT` 没有写权限或空间不足。

训练前检查：

```bash
test -f /your/path/to/pytorch_model.bin
test -f /your/path/to/kmeans_navsim_traj_20.npy
test -d $OPENSCENE_DATA_ROOT
test -d $NAVSIM_EXP_ROOT
```

### 10.2 显存与 batch size

默认 `batch_size=64`，`precision=16-mixed`，`strategy=ddp`。如果显存不足：

```bash
trainer.params.accumulate_grad_batches=2
dataloader.params.batch_size=16
```

单卡调试可尝试：

```bash
trainer.params.strategy=auto
dataloader.params.batch_size=4
trainer.params.limit_train_batches=10
```

### 10.3 多卡训练

PyTorch Lightning 会根据可见 GPU 使用 DDP。建议用 `CUDA_VISIBLE_DEVICES` 明确卡：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python navsim/planning/script/run_training.py ...
```

观察：

- GPU 利用率是否均衡。
- dataloader 是否成为瓶颈。
- `num_workers` 是否过大导致 CPU 或共享内存压力。

### 10.4 Cache 优先

全量训练前务必先 cache。直接从原始数据训练会让调试周期非常长，也更难定位问题。

建议顺序：

1. dataset cache。
2. 官方 checkpoint eval。
3. sanity train。
4. 全量 train。
5. 自己 checkpoint eval。

---

## 11. 建议断点总表

| 目标 | 文件 | 断点位置 | 看什么 |
|---|---|---|---|
| Hydra agent 实例化 | `run_training.py` | `instantiate(cfg.agent)` 后 | agent 类型、config |
| Agent 初始化 | `transfuser_agent.py` | `__init__()` | checkpoint、lr、config |
| 输入构建 | `transfuser_features.py` | feature/target builder | camera/lidar/status/trajectory |
| 主模型输入 | `transfuser_model_v2.py` | `V2TransfuserModel.forward()` 开头 | features shape |
| Backbone 输出 | `transfuser_model_v2.py` | `_backbone(...)` 后 | BEV feature shape |
| Query 分裂 | `transfuser_model_v2.py` | `query_out.split(...)` 后 | ego/agent query shape |
| 截断加噪训练 | `transfuser_model_v2.py` | `TrajectoryHead.forward_train()` | timestep、noise、noisy traj |
| 两步去噪推理 | `transfuser_model_v2.py` | `TrajectoryHead.forward_test()` | roll_timesteps、img 更新 |
| Decoder 交互 | `transfuser_model_v2.py` | `CustomTransformerDecoderLayer.forward()` | cross-attn 输入输出 |
| BEV cross attention | `blocks.py` | `GridSampleCrossBEVAttention.forward()` | 轨迹点如何采样 BEV |
| 轨迹 loss | `multimodal_loss.py` | `LossComputer.forward()` | mode_idx、cls/reg loss |
| 总 loss | `transfuser_loss.py` | `transfuser_loss()` | loss_dict 量级 |

---

## 12. 推荐小实验路线

### 实验 A：复现去噪步数消融

目的：对应论文表 4。

改动位置：

- `TrajectoryHead.forward_test()` 中 `step_num = 2`

实验组：

- `step_num = 1`
- `step_num = 2`
- `step_num = 3`

记录：

- PDMS
- FPS 或总评估时间
- top-1 轨迹可视化

### 实验 B：观察 anchor 先验作用

目的：理解 anchored Gaussian distribution。

改动位置：

- `TransfuserConfig.plan_anchor_path`
- `TrajectoryHead.forward_train()`
- `TrajectoryHead.forward_test()`

实验组：

- 官方 `kmeans_navsim_traj_20.npy`
- 随机打乱 anchor 顺序
- 用一条外推轨迹复制成 20 个 mode

记录：

- `mode_idx` 分布
- trajectory loss
- PDMS
- 多样性可视化

### 实验 C：Cascade decoder 层数

目的：对应论文表 5。

改动位置：

```python
self.diff_decoder = CustomTransformerDecoder(diff_decoder_layer, 2)
```

实验组：

- 1 层
- 2 层
- 4 层

记录：

- 参数量
- 显存
- 推理耗时
- PDMS

### 实验 D：Loss 观察

目的：看训练是否稳定。

观察项：

- `trajectory_loss`
- `agent_class_loss`
- `agent_box_loss`
- `bev_semantic_loss`
- `trajectory_loss_0`
- `trajectory_loss_1`

判断：

- 如果 `trajectory_loss` 不下降，先看 anchor 路径和 `mode_idx`。
- 如果 `bev_semantic_loss` 异常，先看 target map shape 和 class index。
- 如果检测 loss 爆炸，先看 Hungarian matching 输入是否有 NaN。

---

## 13. 一周学习节奏建议

### Day 1：读入口

- [ ] 读 `diffusiondrive_agent.yaml`
- [ ] 读 `transfuser_agent.py`
- [ ] 跑 `--cfg job`
- [ ] 跑 `fast_dev_run=true`

### Day 2：读数据

- [ ] 读 `transfuser_features.py`
- [ ] 打印 features/targets shape
- [ ] 打开 cache 文件看结构

### Day 3：读主模型

- [ ] 读 `V2TransfuserModel.forward()`
- [ ] 画出 camera/lidar/status 到 BEV/query 的数据流
- [ ] 记录所有关键 tensor shape

### Day 4：读扩散规划头

- [ ] 读 `TrajectoryHead.__init__()`
- [ ] 读 `forward_train()`
- [ ] 读 `forward_test()`
- [ ] 对照论文第 3.3 节写一页笔记

### Day 5：读 decoder 和 loss

- [ ] 读 `CustomTransformerDecoderLayer`
- [ ] 读 `GridSampleCrossBEVAttention`
- [ ] 读 `LossComputer`
- [ ] 跑一次 sanity train 并记录 loss

### Day 6：跑官方 checkpoint eval

- [ ] 准备 metric cache
- [ ] 下载官方 checkpoint
- [ ] 跑 navtest PDMS
- [ ] 保存日志和输出 csv

### Day 7：做第一个消融

- [ ] 改 `step_num`
- [ ] 分别评估 1/2/3 step
- [ ] 写一页结论，对照论文表 4

---

## 14. 最终吃透标准

当你能做到下面这些，基本就真正吃透了：

- [ ] 不看 README，也能写出 cache、train、eval 三条命令。
- [ ] 能解释 `agent=diffusiondrive_agent` 如何一路实例化到 `TrajectoryHead`。
- [ ] 能画出 `features -> backbone -> queries -> diffusion decoder -> trajectory` 的数据流。
- [ ] 能解释训练时为什么 `timesteps` 是 `[0, 50)`。
- [ ] 能解释推理时为什么只做 `step_num=2`。
- [ ] 能解释 `plan_anchor` 如何决定正样本 mode。
- [ ] 能根据 loss 曲线判断是数据问题、anchor 问题、模型问题还是显存/训练配置问题。
- [ ] 能在服务器上从零完成官方 checkpoint 评估和自己的训练 checkpoint 评估。
- [ ] 能完成至少一个复现实验和一个改进实验。

---

## 15. 后续深入方向

建议优先做这三个方向：

1. 动态 anchor：不再固定使用 K-Means anchor，而是根据地图拓扑、导航意图或场景上下文生成 anchor。
2. 显式安全约束：在 diffusion decoder 输出后加入规则约束、可行驶区域约束或碰撞风险过滤。
3. 更强闭环验证：从 NAVSIM 非反应式评估扩展到 nuPlan closed-loop 或 CARLA 交互式仿真。

这三个方向都能直接继承当前代码框架，且和论文 Q10 中的后续研究问题一致。
