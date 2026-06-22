# `train_jax_stable_online_finetune.py` 技术报告

## 1. 摘要

`JAXRSSMJAXDiDr/scripts/train_jax_stable_online_finetune.py` 实现了一套面向 CARLA 闭环驾驶任务的稳定在线微调流程。它将已有的 JAX DreamerV3 RSSM 世界模型和 JAX DiffusionDrive 风格规划器连接起来，通过“在线采集 replay、更新 RSSM、冻结 RSSM 评估候选轨迹、只更新规划器 selector head”的方式逐轮提升规划器在环境中的模式选择能力。

该脚本的核心思想不是端到端更新完整规划器，也不是重新训练扩散解码器，而是保持 planner 的扩散主体、轨迹解码器、latent encoder、anchor encoder 和 plan anchors 固定，仅训练 `selector_head`。这样可以避免在线闭环训练早期由于世界模型误差、控制器误差或稀疏奖励导致的轨迹生成分布崩塌，同时允许策略逐步偏向 RSSM imagination 中回报更高、安全性更好的候选 mode。

整体闭环如下：

```text
offline replay + online replay
        |
        v
更新 DreamerV3/JAX RSSM 世界模型
        |
        v
冻结当前 RSSM 变量为 target_varibs
        |
        v
用 planner 生成所有候选 mode
        |
        v
每个 mode 单独通过控制器 + RSSM imagination 打分
        |
        v
用候选回报训练 planner.selector_head
        |
        v
使用新 selector 继续 CARLA 闭环采集
```

## 2. 文件定位与设计目标

脚本路径：

```text
JAXRSSMJAXDiDr/scripts/train_jax_stable_online_finetune.py
```

它依赖三个已有模块：

- `train_offline_rssm.py`：复用 DreamerV3/JAX 运行时、环境构造、RSSM agent 和 checkpoint 机制。
- `eval_close_loop_rssm_didr.py`：复用闭环评估中的控制器、轨迹合法性检查、动作归一化等逻辑。
- `JAXRSSMJAXDiDr.models.jax_didr_planner`：复用 JAX planner 的 `_decode()` 和 `deterministic_anchors()`，获得每个 mode 的轨迹和 selector logits。

脚本开头的 docstring 已明确其保守训练策略：

```text
collect CARLA replay
-> update online DreamerV3/JAX RSSM
-> snapshot/freeze RSSM variables
-> score every planner candidate mode
-> update only selector_head
-> collect again
```

因此，该脚本适合用于已有 supervised planner 已经能产生合理候选轨迹，但闭环 selector 仍需要适配场景、交通交互和 RSSM 价值偏好的阶段。

## 3. 输入、输出与运行入口

### 3.1 关键输入

命令行参数由 `parse_args()` 定义，主要分为以下几类。

基础路径：

- `--offline_replay_dir`：离线 expert 或历史 replay 目录。
- `--rssm_checkpoint`：已有 JAX DreamerV3 RSSM checkpoint。
- `--planner_checkpoint`：已有 JAX DiDr planner checkpoint。
- `--output_dir`：在线微调输出目录。
- `--online_replay_dir`：在线采集 replay 输出目录；默认是 `output_dir/online_replay`。
- `--anchor_path`：可选，用于替换 checkpoint 中的 plan anchors。

在线循环参数：

- `--outer_iterations`：外层 collect/update 轮数。
- `--collect_episodes`：每轮采集 episode 数。
- `--max_steps`：每个 episode 最大步数。
- `--plan_interval_steps`：闭环控制中重新规划的间隔，默认 5 步。若环境 `dt=0.1`，即每 0.5 秒重新规划一次。

RSSM 更新参数：

- `--wm_updates`：每轮世界模型更新次数。
- `--batch_length`：RSSM 训练序列长度。
- `--batch_size`：训练 batch 大小。
- `--offline_ratio`：混合 replay 中离线数据比例。
- `--structured_world_model`：是否使用结构化观测配置。

selector 更新参数：

- `--selector_updates`：每轮 selector 更新次数。
- `--selector_lr`、`--selector_weight_decay`、`--grad_clip`：selector optimizer 参数。
- `--eval_timestep`：planner deterministic denoising 使用的扩散 timestep。
- `--score_horizon_steps`：RSSM imagination 打分 horizon。
- `--discount`：imagination return 折扣。
- `--policy_temperature`：selector logits softmax 温度。
- `--ranking_weight`、`--kl_weight`、`--entropy_weight`：selector loss 权重。

控制器与轨迹过滤参数：

- `--dt`、`--waypoint_dt`：控制步长和 waypoint 时间间隔。
- `--planner_output_unit`：planner xy 输出单位，支持 `meters` 或 `normalized`。
- `--plan_x_sign`、`--plan_y_sign`：轨迹坐标符号修正。
- `--target_speed_min/max`、`--fallback_speed`、`--stop_speed`：速度估计与控制边界。
- `--wheelbase`、`--max_steer_rad`、`--steer_gain`、`--steer_sign`：pure pursuit 转向参数。
- `--min_valid_points`、`--min_forward_x`、`--max_abs_y`、`--max_step_distance`：轨迹合法性过滤。

TTC 安全惩罚参数：

- `--ttc_penalty_weight`：风险惩罚权重，设为 0 可关闭。
- `--ttc_threshold`：前车 TTC 风险阈值。
- `--ttc_lateral_width`：同车道/走廊横向半宽。
- `--ttc_min_gap`、`--ttc_max_distance`：前车距离过滤。
- `--ttc_cpa_distance`、`--ttc_cpa_time`：最近点接近 CPA 风险参数。

### 3.2 输出文件

每次运行会在 `output_dir` 下产生：

- `run_config.json`：完整参数、额外 Dreamer/env 参数、planner checkpoint epoch、训练策略说明。
- `history.json`：每个 outer iteration 的 collect、RSSM、selector 指标。
- `online_replay/*.npz`：在线闭环采集 replay。
- `rssm_online/checkpoint.ckpt`：在线更新后的 DreamerV3/RSSM checkpoint。
- `planner_selector_online.pkl.gz`：最新 planner checkpoint。
- `planner_selector_online_outer_XXXX.pkl.gz`：按 outer iteration 周期保存的 planner checkpoint。

注意：保存的 planner checkpoint 中只有 `selector_head` 被在线训练过，其他 planner 参数保持来自初始 checkpoint 或 anchor override 后的值。

### 3.3 程序入口

入口函数为：

```python
def main() -> None:
    args, extra = parse_args()
    if args.jax_platform:
        jax.config.update("jax_platform_name", args.jax_platform)
    trainer = StableOnlineTrainer(args, extra)
    trainer.run()
```

`parse_known_args()` 会保留未知参数作为 `extra`，传给 Dreamer/env 配置。`ensure_control_action_extra()` 会强制设置：

```text
--env.planner_target.use_waypoint_action False
```

这意味着环境执行低层 `[acc, steer]` 控制，而不是直接执行 waypoint action。

## 4. 核心类与模块职责

### 4.1 `ReplaySequenceDataset`

`ReplaySequenceDataset` 是一个基于 `.npz` replay chunk 的 numpy 序列采样器。它读取 Dreamer row-format replay，并按 `batch_length` 抽取连续序列。

关键逻辑：

- 遍历 replay 目录下所有 `.npz`。
- 过滤掉没有 `action` 字段或长度不足 `batch_length` 的 chunk。
- `sample(batch_size, rng)` 随机抽取多个序列，并对所有样本共同拥有的 keys 做 stack。
- `_get(index)` 从对应 chunk 中截取 `[start:end]`。
- 如果 replay 中有 `executed_control`，则用它覆盖 `action`，保证世界模型训练看到真实执行的低层控制。
- 自动补齐 `is_first`、`is_last`、`is_terminal`、`reward` 和若干事件字段，避免部分 replay 缺字段导致训练失败。

这使得离线 replay 和在线 replay 可以被统一喂给 Dreamer world model。

### 4.2 `MixedReplaySampler`

`MixedReplaySampler` 用于混合离线和在线 replay：

```text
batch = offline_part + online_part
offline_count = round(batch_size * offline_ratio)
online_count = batch_size - offline_count
```

如果在线 replay 目录还没有可用 chunk，则退化为纯离线采样。该设计能降低在线初期数据质量差、分布窄导致 RSSM 快速遗忘的问题。

### 4.3 `EpisodeReplayWriter`

`EpisodeReplayWriter` 负责把在线闭环 episode 写成 `.npz` replay chunk。每一步保存：

- 环境 obs 原始字段。
- `action` / `executed_control`：归一化 `[acc, steer]`。
- `planner_waypoints8`：当前执行的 8 点轨迹。
- `selected_mode`：selector 选中的 mode index。
- `selector_logits`：当时 planner 输出的所有 mode logits。
- `plan_age`：当前 plan 已经执行了多少 control step。

episode 结束时将所有共同 keys stack 后保存为：

```text
online_o{outer_idx}_e{episode_idx}_{timestamp}.npz
```

### 4.4 `StableOnlineTrainer`

`StableOnlineTrainer` 是主控制类，负责初始化、采集、RSSM 更新、selector 更新和保存。

初始化阶段完成：

1. 构建 DreamerV3/CARLA 环境。
2. 创建 DreamerV3 agent 并加载 RSSM checkpoint。
3. 创建 `rssm_online` checkpoint 管理器。
4. 加载 planner checkpoint，可选覆盖 anchors。
5. 将 planner 控制相关参数同步到 `planner_config`。
6. 拆出 `selector_head` 作为唯一可训练参数。
7. 建立 selector optimizer。
8. 创建 JIT 函数：
   - world model train step
   - RSSM episode init
   - RSSM observation update
   - batch 起点 latent 提取
   - mode returns imagination
   - selector update step
   - 单步规划函数

## 5. Planner 与 RSSM 的接口

### 5.1 RSSM latent 作为 planner condition

Planner 的输入 condition 来自 RSSM posterior state：

```python
def flatten_state_feat(state):
    deter = state["deter"].reshape((state["deter"].shape[0], -1))
    stoch = state["stoch"].reshape((state["stoch"].shape[0], -1))
    return jnp.concatenate([deter, stoch], axis=-1)
```

也就是说，planner 不直接读取 raw observation，而是读取 Dreamer RSSM 的 compact latent feature：

```text
feat = concat(flatten(deter), flatten(stoch))
```

### 5.2 生成所有候选 mode

`planner_modes_logits()` 使用 deterministic anchors 和 planner `_decode()`：

```text
latent feat
  -> deterministic_anchors(...)
  -> _decode(...)
  -> poses_reg [B, M, P, 3]
  -> poses_cls [B, M]
```

其中：

- `B`：batch size。
- `M`：候选 mode 数，一般等于 anchor 数。
- `P`：waypoint 数，默认 8。
- `poses_reg[..., :2]`：xy 轨迹。
- `poses_reg[..., 2]`：heading。
- `poses_cls`：selector logits。

在线训练中，`poses_reg` 由冻结 planner body 产生，`poses_cls` 由可训练 selector head 产生。

### 5.3 单步闭环规划

在线采集时，`plan_np()` 会：

1. 调用 `_plan_fn` 得到所有 candidate trajectories 和 logits。
2. 用 `argmax(logits)` 选择一个 mode。
3. 按需要把 normalized xy 转为 meters。
4. 返回 selected trajectory、selected mode index、logits 和全部 `poses_reg`。

闭环执行不使用 soft trajectory averaging，而是执行单个离散 mode。这个设计对稳定性很重要，因为 soft average 容易把多个合理但拓扑不同的轨迹平均成不可行轨迹。

## 6. 在线采集流程

`collect(outer_idx)` 在每个 outer iteration 中运行若干 episode。

单个 episode 的流程：

1. reset 环境。
2. 初始化 RSSM latent。
3. 进入控制循环，最多 `max_steps`。
4. 每步用当前 obs 和上一动作调用 `encode_obs()`，更新 RSSM posterior。
5. 若 `step_idx % plan_interval_steps == 0`，用 planner 重新规划。
6. 将 ego-frame selected trajectory 转到 world frame 缓存。
7. 每个 0.1s 控制步将缓存 world plan 转回当前 ego frame。
8. 调用闭环评估同款轨迹过滤和 pure pursuit + PID 控制器，得到 `[acc, steer]`。
9. 执行动作，记录 replay。
10. episode 终止时保存 `.npz`。

重要细节：

- 规划频率低于控制频率。默认 `plan_interval_steps=5`，`dt=0.1`，所以每 0.5 秒重新规划一次。
- 缓存的是 world-frame waypoints；每个 control step 根据最新 ego pose 转回 ego-frame，因此 receding horizon 执行更接近真实闭环。
- 如果轨迹非法，控制器不会直接崩溃，而是保守减速并限制转向。

## 7. RSSM 在线更新

`update_rssm(outer_idx)` 负责更新 DreamerV3 world model。

流程：

1. 构建 `MixedReplaySampler`。
2. 每次更新采样一个 batch。
3. 首次更新时初始化 Dreamer train state。
4. 调用 `_wm_train_fn`：

```text
data -> agent.preprocess
     -> agent.wm.train(data, state)
     -> metrics["stable_online_wm_loss"] = model_loss_mean
```

5. 累加 Dreamer counter。
6. 每 `log_every` 打印 RSSM loss。
7. 保存 `rssm_online/checkpoint.ckpt`。
8. 将当前 `agent.varibs` 深拷贝到 `target_varibs`。

`target_varibs` 是 selector 更新时的冻结 RSSM 快照。这样 selector 的每个 update 都基于固定世界模型打分，不会在同一轮 selector 更新中目标函数漂移。

## 8. 候选 mode 的 RSSM imagination 打分

`_make_returns_fn()` 是整份脚本中最关键的部分。它对每个样本的每个 planner mode 独立 rollout，计算 imagined return。

输入：

```text
state      当前 RSSM state
speed      当前 ego speed
neighbors  当前 neighbor_vehicles_local
modes_xy   planner 所有 candidate mode 的 xy 轨迹 [B, M, P, 2]
```

内部先将 batch 和 mode 展平：

```text
[B, M, ...] -> [B*M, ...]
```

然后每个 mode 都维护一套虚拟车辆状态：

- `pose_x`
- `pose_y`
- `pose_yaw`
- `cur_speed`
- `prev_steer`
- PID integral / previous error / ready flag
- accumulated returns / discounts

每个 imagined step 执行：

```text
候选轨迹转换到当前 ego 坐标
        |
        v
eval_like_controller 计算 action
        |
        v
RSSM img_step(state, action)
        |
        v
reward head 和 cont head 预测 reward / continuation
        |
        v
减去 TTC risk penalty
        |
        v
积分 discounted return
        |
        v
用简化自行车模型更新虚拟 ego pose 和 speed
```

最后返回：

```text
returns [B, M]
```

### 8.1 控制器一致性

`eval_like_controller()` 在 JIT 内部复刻闭环评估使用的轨迹验证、速度估计和 pure pursuit 逻辑：

- 检查 finite。
- 检查最大横向偏移 `max_abs_y`。
- 检查相邻 waypoint 距离 `max_step_distance`。
- 过滤前向点 `x >= min_forward_x`。
- 使用前几个 segment 的长度估计目标速度。
- 根据速度计算 lookahead。
- 取 `0.75*ld, ld, 1.35*ld` 三个 lookahead 目标点并加权。
- 使用 pure pursuit curvature 转成 steering。
- 使用 PID 计算 acceleration。
- 将物理加速度映射为环境归一化 action。

这样做保证候选 mode 的 imagination 打分与实际 collect 时的控制行为尽量一致。

### 8.2 RSSM reward 与 continuation

每个 imagined step：

```text
cur_state = rssm.img_step(cur_state, action)
reward = reward_head(cur_state).mode()
cont = cont_head(cur_state).mean()
returns += discounts * reward
discounts *= discount * cont
```

其中 `cont` 代表 episode continuation 概率，能让模型在预测终止风险高的状态降低后续回报权重。

### 8.3 TTC / CPA 安全惩罚

脚本额外引入显式安全 shaping：

```text
reward = reward - ttc_penalty_weight * ttc_cost
```

TTC 风险分两类：

1. 前车跟驰风险：
   - 将邻车转换到 ego heading 对齐的局部坐标。
   - 只考虑前方、距离有限、横向在 corridor 内、正在 closing 的车辆。
   - 若 gap 已经过小则 TTC 视为 0。
   - 风险随 TTC 低于阈值而二次增长。

2. CPA 最近点风险：
   - 使用短时常速度模型预测 ego 与邻车的 closest point of approach。
   - 对侧方、后方、交汇/并入等不在同一 corridor 的互动更敏感。
   - 同时考虑距离风险和时间风险。

最终 TTC cost 取 leading risk 和 CPA risk 的最大值。

## 9. Selector 更新目标

`update_selector()` 每次迭代会：

1. 从 mixed replay 采样 batch。
2. 用冻结 `target_varibs` 从序列末尾提取 RSSM state、speed、feat、neighbors。
3. 用当前 planner 生成所有 candidate trajectories。
4. 将 candidate xy 转成 meters，并 `stop_gradient`。
5. 用冻结 RSSM imagination 得到 `returns [B, M]`。
6. 调用 `_selector_step()` 更新 `selector_head`。

### 9.1 Advantage 标准化

在 `_selector_step_impl()` 中，候选 mode 的回报先做 per-sample 标准化：

```text
adv = returns - mean(returns, axis=mode)
adv = adv / max(std(returns, axis=mode), adv_eps)
adv = clip(adv, -adv_clip, adv_clip)
adv = stop_gradient(adv)
```

这意味着 selector 学的是“同一个状态下哪个 mode 相对更好”，而不是不同状态之间绝对 return 尺度的差异。

### 9.2 Ranking loss

核心 ranking objective：

```text
logp = log_softmax(logits / temperature)
ranking_loss = -mean(sum_m adv_m * logp_m)
```

当某个 mode 的 return 高于本状态平均值时，其 advantage 为正，优化会提高它的 log probability；反之降低。

### 9.3 KL 保守项

脚本保存初始 planner 参数为 `ref_planner_params`，并在 selector 更新时计算参考 logits：

```text
ref_logp = log_softmax(ref_logits / temperature)
ref_prob = softmax(ref_logits / temperature)
kl_loss = mean(sum_m ref_prob_m * (ref_logp_m - logp_m))
```

这是从 reference selector 到 current selector 的 KL 约束。它防止在线 selector 过快偏离 supervised checkpoint，属于稳定在线微调的关键保护项。

### 9.4 Entropy 奖励

脚本还计算：

```text
entropy = -sum_m prob_m * logp_m
loss = ranking_weight * ranking_loss
     + kl_weight * kl_loss
     - entropy_weight * entropy
```

entropy 项鼓励保留一定探索/不确定性，避免 selector 很快塌缩到少数 mode。

### 9.5 训练指标

每次 selector update 输出：

- `selector_loss`
- `ranking_loss`
- `kl_loss`
- `entropy`
- `return_mean`
- `return_std`
- `selected_return`：当前 selector argmax mode 的平均 return。
- `oracle_return`：所有 mode 中最佳 return。
- `rank_acc`：selector argmax 是否等于 return argmax。

`selected_return` 与 `oracle_return` 的差距可以衡量 selector 是否学会接近 RSSM imagination 下的最优候选。

## 10. 为什么只训练 `selector_head`

完整 planner 通常包含：

- plan anchors
- anchor encoder
- time encoder
- latent encoder
- decoder layers
- delta/trajectory head
- selector head

该脚本只训练 `selector_head`，原因包括：

1. 候选轨迹分布来自 supervised pretrain，通常已经具备可行几何结构。
2. 在线 RSSM imagination 仍存在模型误差，不适合直接反向塑形轨迹几何。
3. selector 是离散 mode 排序问题，优化目标与 imagined returns 更直接对齐。
4. 固定 trajectory generator 可减少闭环训练中的分布漂移。
5. KL 到 reference selector 可以进一步限制策略更新幅度。

因此，该脚本更像“在线重排序/选择器适配”，而不是“在线生成器再训练”。

## 11. 稳定性机制汇总

脚本中的稳定性设计主要包括：

- 混合 replay：RSSM 更新使用 offline + online，降低遗忘。
- 冻结 target RSSM：selector 更新时世界模型固定。
- 只训练 selector：避免轨迹生成器被不稳定 reward 破坏。
- 每个 mode 独立打分：不使用 soft average trajectory。
- KL 正则：限制 selector 偏离原 checkpoint。
- entropy 正则：避免过早确定性塌缩。
- advantage 标准化：减少 return 尺度漂移影响。
- gradient clipping：限制 selector 更新步长。
- 轨迹合法性过滤：非法轨迹触发保守减速。
- TTC/CPA 惩罚：补偿 RSSM reward head 对安全风险建模不足。
- world-frame plan cache：闭环执行中重投影到当前 ego 坐标，减少计划坐标过期问题。

## 12. 主要数据流

### 12.1 Collect 数据流

```text
obs_t, action_{t-1}, rssm_state_{t-1}
        |
        v
RSSM obs_step -> posterior latent feat_t
        |
        v
planner decode -> candidate modes + logits
        |
        v
argmax selector -> selected trajectory
        |
        v
trajectory validation + pure pursuit + PID
        |
        v
env.step([acc, steer])
        |
        v
write online replay
```

### 12.2 RSSM Update 数据流

```text
offline replay sequences
online replay sequences
        |
        v
MixedReplaySampler
        |
        v
Dreamer preprocess
        |
        v
world model train
        |
        v
agent.varibs updated
        |
        v
target_varibs snapshot
```

### 12.3 Selector Update 数据流

```text
mixed replay sequence batch
        |
        v
frozen RSSM observe -> final state / feat / speed / neighbors
        |
        v
planner all modes -> modes_xy
        |
        v
frozen RSSM imagination return for each mode
        |
        v
advantage ranking target
        |
        v
update selector_head only
```

## 13. 与普通 `train_jax_online_finetune.py` 的区别

从 RUNBOOK 中的普通 Dreamer-style planner/critic finetune 看，早期版本目标是通过 RSSM imagination 更新 planner 和 critic。`train_jax_stable_online_finetune.py` 更保守：

- 不引入单独 critic。
- 不通过 soft selected trajectory 直接优化 actor。
- 不更新 planner 轨迹生成主体。
- 使用 RSSM reward/cont + TTC penalty 对所有 candidate mode 打分。
- 把问题转化为 selector ranking。

因此它更适合作为 supervised planner 到闭环在线性能之间的稳定过渡阶段。

## 14. 运行示例

示例命令：

```bash
python -m JAXRSSMJAXDiDr.scripts.train_jax_stable_online_finetune \
  --offline_replay_dir "$REPLAY_DIR" \
  --rssm_checkpoint "$JAX_RSSM_CKPT" \
  --planner_checkpoint "$RUN_ROOT/jax_didr_planner/best.pkl.gz" \
  --output_dir "$RUN_ROOT/jax_didr_stable_online" \
  --task carla_roundabout \
  --outer_iterations 10 \
  --collect_episodes 2 \
  --max_steps 1000 \
  --plan_interval_steps 5 \
  --wm_updates 200 \
  --selector_updates 200 \
  --batch_length 64 \
  --batch_size 16 \
  --offline_ratio 0.7 \
  --selector_lr 1e-5 \
  --eval_timestep 8 \
  --score_horizon_steps 15 \
  --structured_world_model
```

调试时可以先缩小：

```bash
python -m JAXRSSMJAXDiDr.scripts.train_jax_stable_online_finetune \
  --offline_replay_dir "$REPLAY_DIR" \
  --rssm_checkpoint "$JAX_RSSM_CKPT" \
  --planner_checkpoint "$RUN_ROOT/jax_didr_planner/best.pkl.gz" \
  --output_dir "$RUN_ROOT/debug_stable_online" \
  --outer_iterations 1 \
  --collect_episodes 1 \
  --max_steps 100 \
  --wm_updates 5 \
  --selector_updates 5 \
  --batch_size 4 \
  --batch_length 32 \
  --structured_world_model
```

如果只想测试 RSSM/selector 更新链路而不启动在线采集，可加：

```bash
--no_collect
```

## 15. 关键日志解读

Collect 阶段：

```text
[collect] outer=1 episode=001 steps=... return=... replay=...
```

表示该 episode 已保存为 online replay。

RSSM 更新阶段：

```text
[wm] outer=1 update=10 loss=...
```

这里的 loss 来自 Dreamer world model 的 `model_loss_mean`。

Selector 更新阶段：

```text
[selector] outer=1 update=10 selector_loss=... selected_return=... oracle_return=... rank_acc=...
```

建议重点看：

- `selected_return` 是否逐渐接近 `oracle_return`。
- `rank_acc` 是否上升。
- `kl_loss` 是否过大，过大说明 selector 偏离 reference 太多。
- `entropy` 是否快速降为很低，过低可能表示 mode collapse。

Outer 总结：

```text
[stable_online] outer=1 mean_return=... wm_loss=... selected_return=...
```

可用于观察闭环 return、RSSM loss 和 selector imagination return 的同步变化。

## 16. 潜在风险与注意事项

1. RSSM reward head 的偏差会直接影响 selector 排序。TTC penalty 能补偿一部分安全风险，但不能完全替代真实环境评估。
2. `neighbor_vehicles_local` 的字段格式被假设为每车 11 维；如果环境编码变化，TTC/CPA 逻辑需要同步调整。
3. `planner_output_unit` 必须与 checkpoint 输出一致。当前注释认为 JAX planner checkpoint 输出 meters。
4. `plan_x_sign` / `plan_y_sign` 符号错误会导致闭环控制完全失效。
5. `offline_ratio` 太低可能导致 RSSM 遗忘，太高则在线适配慢。
6. `selector_lr` 太大可能使 KL/entropy 控制不住，出现 selector 快速塌缩。
7. `score_horizon_steps` 越大，计算越慢，且 RSSM 长期预测误差越明显。
8. `plan_interval_steps` 过大时 plan cache 会过旧，过小时规划调用频繁、JIT/控制开销增加。

## 17. 建议的实验监控指标

建议记录并对比：

- CARLA episode return。
- 成功率 `is_success` / `destination_reached`。
- 碰撞率 `is_collision`。
- 出车道率 `out_of_lane`。
- RSSM `stable_online_wm_loss`。
- selector `selected_return`、`oracle_return`、`rank_acc`。
- selector `kl_loss` 和 `entropy`。
- invalid plan 打印频率。
- selected mode 分布熵。

如果出现闭环性能下降但 `selected_return` 上升，通常说明 RSSM imagination reward 与真实环境 reward 出现错配，需要检查 RSSM 更新、TTC penalty 或 reward head。

## 18. 总结

`train_jax_stable_online_finetune.py` 是一个保守、工程化程度较高的在线微调脚本。它将复杂的在线 planner finetune 问题拆成三件相对可控的事：

1. 用 offline + online replay 持续适配 RSSM 世界模型。
2. 用冻结 RSSM 对 planner 的所有离散候选 mode 做短期 imagination 排序。
3. 只训练 selector head，让 planner 更倾向选择高回报、低风险的候选轨迹。

这种设计牺牲了一部分轨迹生成自由度，但显著降低了在线训练不稳定性，特别适合已经拥有可用 supervised planner checkpoint、希望进一步提升 CARLA 闭环表现的场景。
