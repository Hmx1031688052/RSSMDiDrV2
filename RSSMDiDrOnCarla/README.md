# RSSMDiDrOnCarla

这是一个面向 CARLA `roundabout` 任务的实验管线：

```text
PolyPlanner 专家数据 -> 真实 ego future 替换 -> DreamerV3 RSSM latent -> DiffusionDrive-style planner
```

目标是拔掉原 DiffusionDrive 的 scene encoder，只保留：

- plan anchors
- truncated diffusion on anchors
- trajectory decoder
- mode selector
- RSSM latent 条件输入

最终 planner 从 `rssm_latent` 生成未来 4s 的 8 个 ego-frame 规划路点。

## 原理

RSSM-DiffusionDrive 可以基于专家数据集生成规划路点，核心是两阶段建模：

1. DreamerV3 RSSM 负责学习专家数据中的时序驾驶状态。
   RSSM 输入来自 replay 中的相机、底盘 / ego 状态、动作、reward、episode boundary 等信息，并把历史上下文压缩成 `rssm_latent`。
2. DiffusionDrive-style planner 负责从 `rssm_latent` 生成未来轨迹。
   planner 不再看 camera / lidar / BEV / agent target，而是用 RSSM latent 作为唯一条件输入，对 anchor 加截断扩散噪声后做 refinement。

注意：Dreamer 的 action 仍然是环境控制动作，例如 `[acc, steer]`。`expert_waypoints8` 是监督 / 重建信号，不是 RSSM action。

当前文件夹已经实现 planner-side pipeline。仍需要 DreamerV3 侧补齐 offline RSSM world model training 和 RSSM latent exporter。

## 路径约定

以下命令以 `roundabout` 为例。建议先统一设置这些路径：

```bash
export TASK=carla_roundabout
export RUN_ROOT=./outputs/rssm_didr_roundabout
export COLLECT_LOGDIR=${RUN_ROOT}/expert_collect
export REPLAY_DIR=${COLLECT_LOGDIR}/replay
export LATENT_DIR=${RUN_ROOT}/rssm_latents
export PLANNER_DATASET_DIR=${RUN_ROOT}/planner_dataset
export ANCHOR_PATH=${RUN_ROOT}/anchors.npy
export PLANNER_CKPT_DIR=${RUN_ROOT}/planner_checkpoints_0.03std
```

## 顺序执行命令

### 0. 启动外部依赖

在采集前，需要先启动：

- CARLA server
- roundabout 对应地图 / route 配置
- ROS2 PolyPlanner 外部专家
- `/obs_info`、`/ctrl_info`、`/global_info`、`/reset_info`、`/traj_best_vis` 等 topic

`collect_polyplanner.py` 会通过 ROS2 topic 接收 PolyPlanner 控制和轨迹，并将成功 episode 写入 Dreamer replay。

### 1. 采集 roundabout 专家 replay

`car_dreamer/configs/tasks.yaml` 中 roundabout 的任务名是 `carla_roundabout`，默认专家采集 episode 数为 `1000`。

```bash
python dreamerv3/collect_polyplanner.py \
  --task ${TASK} \
  --dreamerv3.logdir ${COLLECT_LOGDIR} \
  --env.expert_collection.episodes 1000 \
  --env.planner_target.use_waypoint_action False
```

采集完成后，replay 默认位于：

```text
${COLLECT_LOGDIR}/replay
```

也就是本文后续使用的：

```text
${REPLAY_DIR}
```

采集阶段保存的是专家成功 episode。此时 replay 中的 `expert_waypoints8` 仍来自 PolyPlanner 规划线，不应直接作为最终 planner 监督。

### 2. 将 `expert_waypoints8` 替换为真实 ego future

```bash
python -m RSSMDiDrOnCarla.scripts.prepare_polyplanner_replay \
  --replay_dir ${REPLAY_DIR} \
  --waypoint_scale 30.0 \
  --dt 0.1 \
  --waypoint_interval 5
```

这一步会复用 `dreamerv3/replace_expert_with_ego_traj.py` 的逻辑，原地替换 replay chunk 中的 `expert_waypoints8`。

替换后，`expert_waypoints8` 的语义固定为：

```text
ego-frame future trajectory
8 个 waypoint
每 0.5s 一个点
总 horizon 4s
xy 已除以 waypoint_scale
source = ego_future
```

planner target 的构造方式是：

```text
xy = expert_waypoints8.reshape(8, 2) * waypoint_scale
heading = finite_difference_heading(xy)
trajectory = concat(xy, heading)  # [8, 3]
```

从这一步之后，planner 的唯一监督轨迹来源就是替换后的 `expert_waypoints8`，不再使用旧 PolyPlanner best trajectory。

### 3. 训练 DreamerV3 offline RSSM world model

当前仓库已有 JAX DreamerV3 在线训练入口，但还没有一个完整的“只读专家 replay、离线训练 RSSM world model”的专用脚本。

需要补齐的上游入口建议是：

```bash
conda activate cardreamer
python -m RSSMDiDrOnCarla.scripts.train_offline_rssm \
  --task carla_roundabout \
  --replay_dir ${REPLAY_DIR} \
  --logdir ${RUN_ROOT}/offline_rssm \
  --batch_length 64 \
  --batch_size 16 \
  --env.planner_target.use_waypoint_action False \
  --dreamerv3.encoder.cnn_keys "birdeye_wpt" \
  --dreamerv3.encoder.mlp_keys "ego_speed|ego_yawrate|ego_x|ego_y|ego_yaw" \
  --dreamerv3.decoder.cnn_keys "birdeye_wpt" \
  --dreamerv3.decoder.mlp_keys "ego_speed|ego_yawrate"
```

该脚本应满足：

- 不再在线采样环境。
- 只从 `${REPLAY_DIR}` 读取专家 replay。
- Dreamer action 使用 replay 中的环境动作 `[acc, steer]`。
- `expert_waypoints8` 可以作为 observation reconstruction head，但不作为 action。
- 训练 RSSM posterior / prior、decoder、reward、continue 等 world model loss。

在该上游入口完成前，planner-side 只能使用已经导出的 latent 或 synthetic latent 做 smoke test。

### 4. 导出 RSSM latent

当前仓库也还没有完整的 RSSM latent exporter。需要补齐的上游入口建议是：

```bash
python -m RSSMDiDrOnCarla.scripts.export_rssm_latents \
  --checkpoint ${RUN_ROOT}/offline_rssm/checkpoint.ckpt \
  --replay_dir ${REPLAY_DIR} \
  --output_dir ${LATENT_DIR} \
  --env.planner_target.use_waypoint_action False \
  --dreamerv3.encoder.cnn_keys "birdeye_wpt" \
  --dreamerv3.encoder.mlp_keys "ego_speed|ego_yawrate|ego_x|ego_y|ego_yaw" \
  --dreamerv3.decoder.cnn_keys "birdeye_wpt" \
  --dreamerv3.decoder.mlp_keys "ego_speed|ego_yawrate"
```

导出的 latent chunk 最好和 replay chunk 同名。每个 `.npz` 至少包含：

```text
rssm_latent: [T, D]
```

也可以导出 RSSM component keys：

```text
deter, stoch, logit, mean, std
```

如果提供 component keys，planner exporter 会将它们 flatten 后拼接成一个 `rssm_latent`。

### 5. 导出 planner dataset

当 `${LATENT_DIR}` 中已经存在和 replay 对齐的 latent chunk 后，执行：

```bash
python -m RSSMDiDrOnCarla.scripts.export_planner_dataset \
  --replay_dir ${REPLAY_DIR} \
  --latent_dir ${LATENT_DIR} \
  --output_dir ${PLANNER_DATASET_DIR} \
  --waypoint_scale 30.0
```

如果 latent 文件名和 replay 文件名不同，但排序后一一对应，可以加：

```bash
--pair_by_order
```

如果 latent 长度和 replay chunk 长度存在轻微差异，并希望截断到较短长度，可以加：

```bash
--allow_length_mismatch
```

每个输出 planner chunk 包含：

```text
rssm_latent: [T, D]
expert_waypoints8: [T, 16]
future_ego_waypoints8: [T, 16]
trajectory: [T, 8, 3]
waypoint_scale
dt
waypoint_interval
target_source = ego_future
```

### 6. 聚类 roundabout anchors

```bash
python -m RSSMDiDrOnCarla.tools.kmeans_polyplanner_anchors \
  --replay_dir ${REPLAY_DIR} \
  --output ${ANCHOR_PATH} \
  --num_modes 20 \
  --waypoint_scale 30.0
```

输出：

```text
anchors.npy: [20, 8, 2]
```

anchor 坐标是未归一化的 ego-frame 米制 xy，来源是替换后的真实 ego future trajectory。

### 7. 训练 RSSM-DiffusionDrive planner

```bash
conda activate vwm
python -m RSSMDiDrOnCarla.scripts.train_rssm_didr_planner \
  --dataset_dir ${PLANNER_DATASET_DIR} \
  --anchor_path ${ANCHOR_PATH} \
  --output_dir "$RUN_ROOT/planner_checkpoints_0.03std" \
  --batch_size 128 \
  --epochs 200 \
  --lr 1e-4 \
  --waypoint_scale 30.0 \
  --device auto \
  --latent_noise_std 0.03
```

训练输出：

```text
${PLANNER_CKPT_DIR}/run_config.json
${PLANNER_CKPT_DIR}/history.json
${PLANNER_CKPT_DIR}/epoch_*.pt
${PLANNER_CKPT_DIR}/best.pt
${PLANNER_CKPT_DIR}/last.pt
```

### 8. planner-side 一键命令

如果你已经有 `${REPLAY_DIR}` 和 `${LATENT_DIR}`，可以用总控脚本串起 prepare、planner dataset export、anchor clustering、planner training：

```bash
python -m RSSMDiDrOnCarla.scripts.run_full_training_pipeline \
  --replay_dir ${REPLAY_DIR} \
  --latent_dir ${LATENT_DIR} \
  --work_dir ${RUN_ROOT}/planner_run \
  --waypoint_scale 30.0 \
  --dt 0.1 \
  --waypoint_interval 5 \
  --num_modes 20 \
  --batch_size 128 \
  --epochs 20 \
  --lr 1e-4 \
  --device auto
```

### 9. open-loop eval 一键命令
完整 RSSM + planner 联合测试命令：
```bash
python -m RSSMDiDrOnCarla.scripts.eval_open_loop_rssm_didr \
  --replay_dir $REPLAY_DIR \
  --rssm_checkpoint "$RUN_ROOT/offline_rssm/checkpoint.ckpt" \
  --planner_checkpoint "$PLANNER_CKPT_DIR/best.pt" \
  --anchor_path $ANCHOR_PATH \
  --output_dir "$RUN_ROOT/open_loop_eval" \
  --waypoint_scale 30.0 \
  --eval_noise clean \
  --device auto \
  --jax_platform cpu \
  --dreamerv3.jax.train_devices=0
```
如果你已经提前导出了 RSSM latent，也可以跳过 RSSM 前向，测试 cached latent + planner：
```bash
python -m RSSMDiDrOnCarla.scripts.eval_open_loop_rssm_didr \
  --replay_dir "$REPLAY_DIR" \
  --latent_dir "$LATENT_DIR" \
  --planner_checkpoint "/media/pc/T7/Hmx_RssmDIDR/RSSMDiDrV2/outputs/rssm_didr_roundabout/selector_ranking_finetune/best.pt" \
  --anchor_path "$ANCHOR_PATH" \
  --output_dir "$RUN_ROOT/selector_open_loop_eval_cached" \
  --waypoint_scale 30.0 \
  --eval_noise clean \
  --device auto \
  --save_plots \
  --plot_samples 20 \
  --plot_stride 20 \
  --plot_modes 6 \
  --save_animation \
  --no_plot_neighbors
```
闭环测试
```bash
python -m RSSMDiDrOnCarla.scripts.eval_close_loop_rssm_didr \
  --task carla_roundabout \
  --rssm_checkpoint "$RUN_ROOT/offline_rssm/checkpoint.ckpt" \
  --planner_checkpoint "/media/pc/T7/Hmx_RssmDIDR/RSSMDiDrV2/outputs/rssm_didr_roundabout/selector_ranking_finetune/best.pt" \
  --anchor_path "$ANCHOR_PATH" \
  --output_dir "$RUN_ROOT/closed_loop_eval_rssm" \
  --episodes 10 \
  --max_steps 1000 \
  --device auto \
  --jax_platform cpu \
  --dreamerv3.jax.train_devices=0 \
  --env.planner_target.use_waypoint_action False \
  --dreamerv3.encoder.cnn_keys "birdeye_wpt" \
  --dreamerv3.encoder.mlp_keys "ego_speed|ego_yawrate|ego_x|ego_y|ego_yaw" \
  --dreamerv3.decoder.cnn_keys "birdeye_wpt" \
  --dreamerv3.decoder.mlp_keys "ego_speed|ego_yawrate" \
  --live_plot
  --carla_live_draw --carla_spectator_topdown

```
<!-- python -m RSSMDiDrOnCarla.scripts.train_torch_selector_ranking_with_jax_rssm \
  --offline_replay_dir "$REPLAY_DIR" \
  --rssm_checkpoint "$JAX_RSSM_CKPT" \
  --planner_checkpoint "$PLANNER_CKPT_DIR/best.pt" \
  --anchor_path "$ANCHOR_PATH" \
  --output_dir "$RUN_ROOT/torch_selector_ranking_jax_rssm" \
  --iterations 10000 \
  --batch_length 64 \
  --batch_size 16 \
  --imag_horizon 8 \
  --lr 1e-5 -->
python -m RSSMDiDrOnCarla.scripts.export_torch_selector_ranking_dataset_with_jax_rssm \
  --offline_replay_dir "$REPLAY_DIR" \
  --rssm_checkpoint "/media/pc/T7/Hmx_RssmDIDR/RSSMDIDR/outputs/rssm_didr_roundabout/offline_rssm/checkpoint.ckpt" \
  --planner_checkpoint "/media/pc/T7/Hmx_RssmDIDR/RSSMDIDR/outputs/rssm_didr_roundabout/planner_checkpoints_0.03std/epoch_0200.pt" \
  --anchor_path "$ANCHOR_PATH" \
  --output_dir "$RUN_ROOT/selector_ranking_dataset_0200pt" \
  --batch_length 64 \
  --batch_size 16 \
  --sequence_stride 1 \
  --imag_horizon 8 \
  --eval_timestep 8 \
  --dreamerv3.encoder.cnn_keys "birdeye_wpt" \
  --dreamerv3.encoder.mlp_keys "ego_speed|ego_yawrate|ego_x|ego_y|ego_yaw" \
  --dreamerv3.decoder.cnn_keys "birdeye_wpt" \
  --dreamerv3.decoder.mlp_keys "ego_speed|ego_yawrate" \
  --env.planner_target.use_waypoint_action False 

python -m RSSMDiDrOnCarla.scripts.train_torch_selector_from_ranking_dataset \
  --dataset_dir "$RUN_ROOT/selector_ranking_dataset" \
  --planner_checkpoint "/media/pc/T7/Hmx_RssmDIDR/RSSMDIDR/outputs/rssm_didr_roundabout/planner_checkpoints_0.03std/epoch_0100.pt" \
  --anchor_path "$ANCHOR_PATH" \
  --output_dir "$RUN_ROOT/selector_ranking_finetune" \
  --iterations 200000 \
  --batch_size 1024 \
  --lr 1e-5 \
  --ranking_weight 1.0 \
  --rl_weight 0.1 \
  --kl_weight 0.05 \
  --entropy_weight 0.01 
```bash
--skip_prepare
```
GT Vector State:
python -m RSSMDiDrOnCarla.scripts.export_gt_history_planner_dataset \
  --replay_dir "$REPLAY_DIR" \
  --output_dir "$RUN_ROOT/planner_dataset_gt_history" \
  --waypoint_scale 30.0 \
  --history_length 10

python -m RSSMDiDrOnCarla.scripts.train_rssm_didr_planner \
  --dataset_dir "$RUN_ROOT/planner_dataset_gt_history" \
  --anchor_path "$ANCHOR_PATH" \
  --output_dir "$RUN_ROOT/planner_checkpoints_gt_history" \
  --condition_type gt_history \
  --condition_key gt_history \
  --history_length 10 \
  --batch_size 512 \
  --epochs 200 \
  --lr 1e-4 \
  --waypoint_scale 30.0 \
  --save_every 2 \
  --device auto

python -m RSSMDiDrOnCarla.scripts.eval_open_loop_rssm_didr \
  --replay_dir "$REPLAY_DIR" \
  --planner_checkpoint "$RUN_ROOT/planner_checkpoints_gt_history/best.pt" \
  --anchor_path "$ANCHOR_PATH" \
  --output_dir "$RUN_ROOT/open_loop_eval_gt_history" \
  --waypoint_scale 30.0 \
  --eval_noise clean \
  --device auto \
  --save_plots \
  --plot_modes 6


## Planner 训练样本

每个 planner sample 是：

```text
features["rssm_latent"]: [D]
features["expert_waypoints8"]: [16]
targets["trajectory"]: [8, 3]
```

其中：

```text
targets["trajectory"]
  = replaced expert_waypoints8
  -> unnormalize xy
  -> finite-difference heading
  -> [8, 3]
```

它不来自旧 PolyPlanner 轨迹，也不来自 NAVSIM human trajectory。

## Planner 模型

输入：

```text
rssm_latent: [B, D]
plan_anchor: [20, 8, 2]
```

训练 forward：

```text
1. 将 20 条 anchor repeat 到 batch 维度。
2. 用 waypoint_scale 将 anchor xy 归一化。
3. 从 [0, 50) 采样 truncated diffusion timestep。
4. 给归一化 anchor 加 Gaussian noise。
5. 将 noisy anchor 反归一化回米制坐标。
6. 编码 noisy_anchor + diffusion timestep + rssm_latent。
7. decoder 输出多模态轨迹和 selector score。
```

输出：

```text
poses_reg: [B, 20, 8, 3]
poses_cls: [B, 20]
trajectory: [B, 8, 3]  # argmax(poses_cls) 选出的最终轨迹
```

因此它不是单模态直接回归，而是基于 anchor 的多模态 refinement。

## Loss 细节

首先把 GT 分配给最近的 anchor：

```text
dist[m] = mean_t || target_xy[t] - anchor[m, t] ||_2
mode_idx = argmin(dist)
```

只回归该 mode：

```text
best_reg = poses_reg[:, mode_idx]
reg_loss = L1(best_reg, target_trajectory)
```

selector 分类监督：

```text
cls_loss = CrossEntropy(poses_cls, mode_idx)
```

总 loss：

```text
loss = reg_loss_weight * reg_loss + cls_loss_weight * cls_loss
```

当前默认：

```text
reg_loss_weight = 1.0
cls_loss_weight = 0.5
```

额外记录：

```text
ADE = mean_t || pred_xy[t] - target_xy[t] ||_2
FDE = || pred_xy[-1] - target_xy[-1] ||_2
```

planner loss 只依赖：

```text
targets["trajectory"]
```

不使用：

- BEV semantic loss
- agent box loss
- 旧 PolyPlanner trajectory loss

## 相比原 DiffusionDrive 保留了什么

保留：

- plan anchors
- truncated diffusion noisy-anchor refinement
- multi-modal trajectory prediction
- selector / mode classification
- closest-anchor assignment
- selected-mode trajectory regression
- 4s / 8 waypoint 输出格式

## 相比原 DiffusionDrive 删除了什么

删除：

- camera scene encoder
- lidar / BEV backbone
- BEV semantic head
- agent detection head
- BEV cross-attention
- agent query cross-attention
- NAVSIM feature / target builders
- 原 PolyPlanner best trajectory 监督
- NAVSIM BEV / agent targets

当前 planner 的唯一条件输入是：

```text
rssm_latent
```

当前 planner 的唯一轨迹监督是：

```text
替换后的 expert_waypoints8 -> trajectory [8, 3]
```

## Smoke Test

不依赖真实 CARLA 数据的 synthetic pipeline 检查：

```bash
python -m RSSMDiDrOnCarla.scripts.smoke_test_pipeline
```

如果当前 Python 环境安装了 PyTorch，这个 smoke test 还会跑一个 1 epoch planner train step。

如果希望没有 PyTorch 时直接失败，而不是跳过训练段：

```bash
python -m RSSMDiDrOnCarla.scripts.smoke_test_pipeline --require_torch
```
