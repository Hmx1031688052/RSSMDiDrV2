# DreamerV3 + DiffusionDrive 稳定在线微调方案

## 1. 目标与核心结论

本文档整理一套面向 **CARLA 闭环自动驾驶规划** 的最终稳定版微调方案，适用于当前系统：

```text
DreamerV3 RSSM world model
    +
RSSM-conditioned DiffusionDrive planner
    +
8 个未来 waypoint，每个间隔 0.5s
    +
低层 controller，每 0.1s 输出 acc / steer
    +
CARLA 闭环执行
```

最终推荐方案不是直接全量强化学习微调整个 diffusion planner，而是：

```text
Online CARLA replay
    ↓
Online update RSSM world model
    ↓
Freeze / EMA target RSSM
    ↓
Candidate-level RSSM imagination scoring
    ↓
Selector-only ranking finetune
    ↓
Closed-loop evaluation
```

核心原则：

\[
\boxed{
\text{RSSM 可以在线更新，但 planner 更新时必须使用冻结的 target RSSM}
}
\]

\[
\boxed{
\text{planner 先只微调 selector/head，不要一开始全量微调 diffusion body}
}
\]

\[
\boxed{
\text{RSSM imagination 先用 15-step return + value bootstrap，不要直接 40-step rollout}
}
\]

---

## 2. 为什么不能直接用当前 Dreamer-style actor-critic 版本作为最终版

当前版本的优化逻辑大致是：

```text
posterior latent s_t
    ↓
planner 每个 imagined step 重新输出 waypoint
    ↓
soft_select 多模态轨迹
    ↓
differentiable PID pure-pursuit controller
    ↓
RSSM img_step
    ↓
reward / continue / critic lambda-return
    ↓
更新 planner + critic
```

它可以作为 baseline，但不建议作为最终稳定版，原因包括：

### 2.1 RSSM 没有在线更新

当前脚本虽然支持 `offline_replay_dir` 和 `online_replay_dir`，但本质只是从已有 replay 文件中采样，并没有执行：

```text
CARLA collect
    ↓
append online replay
    ↓
update RSSM
    ↓
sync target RSSM
```

如果换了新的 controller，planner + controller 会访问新的状态动作分布。仅依赖离线 expert replay 训练好的 RSSM，容易出现 OOD imagination：

\[
(s_t,a_t)\sim d_{\pi_{\text{planner}}+C_{\text{new}}}
\not\sim
d_{\text{offline}}
\]

此时 RSSM reward / continue / dynamics 预测可能失真，planner 容易钻 world model 漏洞。

### 2.2 当前版本几乎更新整个 planner

当前 actor optimizer 对整个 `planner_params` 求梯度，仅冻结了 `plan_anchor`。这会更新：

```text
selector_head
delta_head
latent_encoder
time_encoder
anchor_encoder
decoder_layers
diffusion body
```

这比 selector-only 微调激进得多。对于已经预训练较好的 diffusion planner，闭环差往往不是“所有候选都差”，而是：

```text
候选轨迹中存在较好轨迹
但 selector / poses_cls 没有选出来
```

因此第一阶段最稳的是只训练 selector/head。

### 2.3 soft_select 会平均多模态轨迹

当前方式使用：

\[
\hat{\tau}
=
\sum_m
\operatorname{softmax}(l_m/T)\tau_m
\]

如果不同 mode 分别表示左偏、直行、右偏，soft average 可能生成一条原本不存在的中间轨迹，导致控制不稳定。

最终版应改成：

```text
每条 candidate trajectory 单独进入 controller
每条 candidate trajectory 单独 RSSM rollout
每条 candidate trajectory 得到一个 return J_m
用 J_m 训练 selector
```

---

## 3. 最终稳定版总体框架

### 3.1 真实闭环执行频率

CARLA 环境：

\[
\Delta t_{\text{env}}=0.1s
\]

planner 输出：

\[
8 \text{ waypoints},\quad \Delta t_{\text{wp}}=0.5s
\]

总规划时域：

\[
8\times0.5s=4.0s
\]

但真实闭环执行时，建议：

```text
每 0.5s 调用一次 planner
每 0.1s 调用一次 controller
每次规划后只执行前 5 个 0.1s 控制
然后重新观测、重新规划
```

即：

```text
当前观测 o_t
    ↓
encoder + RSSM observe
    ↓
posterior latent s_t
    ↓
DiffusionDrive planner 输出 8 个 0.5s waypoint
    ↓
controller 插值并跟踪轨迹
    ↓
CARLA 执行 5 步，每步 0.1s
    ↓
重新观测
```

### 3.2 RSSM 训练频率

RSSM 必须按真实控制频率训练：

\[
\boxed{
\Delta t_{\text{RSSM}}=0.1s
}
\]

RSSM 的 action 不应是 waypoint，而应是真实执行控制：

\[
a_t=(acc_t, steer_t)
\]

或者：

\[
a_t=(throttle_t, brake_t, steer_t)
\]

RSSM 学习：

\[
s_{t+1}
=
\text{RSSM}_\theta(s_t,a_t)
\]

这样 RSSM imagination 才能反映 controller 执行误差和车辆短时动力学。

---

## 4. Online replay 设计

每个 0.1s 环境步保存：

```text
obs_t
executed_control_t
reward_t
continue_t / done_t
obs_{t+1}
planner_waypoints_t
controller_tracking_error_t
```

其中必须项：

```text
obs_t
executed_control_t
reward_t
continue_t / done_t
obs_{t+1}
```

建议额外保存：

```text
planner_waypoints_t
selected_mode_t
selector_logits_t
controller_ref_t
controller_tracking_error_t
ego_speed_t
ego_yawrate_t
ego_pose_t
collision_t
offroad_t
route_progress_t
```

这些额外字段可以用于后续分析 controller 误差、selector 崩溃、RSSM OOD、闭环失败原因。

---

## 5. RSSM online update

### 5.1 replay 混合采样

维护两个 replay：

```text
offline expert replay
online planner/controller replay
```

world model batch 混合采样：

\[
\mathcal{D}
=
\alpha\mathcal{D}_{\text{offline}}
+
(1-\alpha)\mathcal{D}_{\text{online}}
\]

前期建议：

\[
\alpha=0.7
\]

即：

```text
70% offline expert replay
30% online planner replay
```

后期逐渐调整为：

\[
\alpha=0.5
\]

即：

```text
50% offline
50% online
```

这样 RSSM 既不会遗忘 expert 正常驾驶分布，又能逐步覆盖 planner + new controller 产生的新状态分布。

### 5.2 world model loss

RSSM 使用 DreamerV3 原始 world model loss：

\[
\mathcal{L}_{\text{WM}}
=
\mathcal{L}_{\text{obs}}
+
\mathcal{L}_{\text{reward}}
+
\mathcal{L}_{\text{continue}}
+
\beta_{\text{dyn}}\mathcal{L}_{\text{dyn-kl}}
+
\beta_{\text{rep}}\mathcal{L}_{\text{rep-kl}}
\]

观测重构：

\[
\mathcal{L}_{\text{obs}}
=
-\log p_\theta(o_t|s_t)
\]

reward 预测：

\[
\mathcal{L}_{\text{reward}}
=
-\log p_\theta(r_t|s_t)
\]

continue 预测：

\[
\mathcal{L}_{\text{continue}}
=
-\log p_\theta(c_t|s_t)
\]

RSSM prior：

\[
p_\theta(z_t|h_t)
\]

RSSM posterior：

\[
q_\theta(z_t|h_t,o_t)
\]

dyn KL：

\[
\mathcal{L}_{\text{dyn-kl}}
=
\text{KL}
\left[
\operatorname{sg}(q_\theta(z_t|h_t,o_t))
\Vert
p_\theta(z_t|h_t)
\right]
\]

rep KL：

\[
\mathcal{L}_{\text{rep-kl}}
=
\text{KL}
\left[
q_\theta(z_t|h_t,o_t)
\Vert
\operatorname{sg}(p_\theta(z_t|h_t))
\right]
\]

其中 \(\operatorname{sg}(\cdot)\) 表示 stop-gradient。

### 5.3 可选：multi-step latent consistency

如果后续希望 RSSM 支持更长 horizon，可以加入多步一致性约束：

\[
\mathcal{L}_{\text{multi}}
=
\sum_{k\in\{5,10,15,20\}}
\lambda_k
\left\|
\operatorname{sg}(z_{t+k}^{\text{post}})
-
z_{t+k}^{\text{imag}}
\right\|_2^2
\]

其中：

\[
z_{t+k}^{\text{post}}
=
q_\theta(z_{t+k}|h_{t+k},o_{t+k})
\]

是用真实未来观测得到的 posterior latent。

\[
z_{t+k}^{\text{imag}}
\]

是从 \(s_t\) 开始连续 img_step 得到的 imagined latent。

初期建议：

\[
\lambda_k=0.01\sim0.05
\]

不要一开始加太大，否则可能干扰 RSSM 表征学习。

---

## 6. target RSSM 机制

不要用正在快速更新的 RSSM 直接优化 planner。维护两个 world model：

```text
rssm_online：持续用 replay 更新
rssm_target：用于 planner imagination scoring
```

### 6.1 hard update

每隔若干次 world model update：

\[
\theta_{\text{target}}
\leftarrow
\theta_{\text{online}}
\]

### 6.2 EMA update

或者使用指数滑动平均：

\[
\theta_{\text{target}}
\leftarrow
\rho\theta_{\text{target}}
+
(1-\rho)\theta_{\text{online}}
\]

建议：

\[
\rho=0.99\sim0.995
\]

planner 微调时：

\[
\boxed{
\text{RSSM target frozen}
}
\]

也就是 planner loss 的梯度不能更新 RSSM。

---

## 7. planner 参数冻结策略

第一阶段只训练 selector/head。

冻结：

```text
plan_anchor
anchor_encoder
time_encoder
latent_encoder
decoder_layers
delta_head
diffusion denoising body
```

只训练：

```text
selector_head / poses_cls
```

即：

\[
\theta_{\text{train}}
=
\theta_{\text{selector}}
\]

不要一开始训练：

\[
\theta_{\text{planner all}}
\]

这样可以最大程度保留预训练 diffusion planner 的轨迹生成能力，只调整“从候选中选哪一条”。

---

## 8. Candidate-level RSSM imagination scoring

### 8.1 planner 输出候选轨迹

对于当前 posterior latent \(s_t\)，planner 输出 \(M\) 条候选轨迹：

\[
\{\tau_1,\tau_2,\ldots,\tau_M\}
\]

第 \(m\) 条轨迹：

\[
\tau_m
=
\{p_{m,1},p_{m,2},\ldots,p_{m,8}\}
\]

其中：

\[
p_{m,i}=(x_{m,i},y_{m,i},\theta_{m,i})
\]

selector 输出 logits：

\[
l_m
\]

selector 概率：

\[
\pi_\phi(m|s_t)
=
\frac{\exp(l_m)}
{\sum_j \exp(l_j)}
\]

### 8.2 插值到 0.1s reference

每条候选轨迹先从 8 个 0.5s waypoint 插值成 40 个 0.1s reference：

\[
\tilde{\tau}_m
=
\text{Interp}(\tau_m,0.1s)
\]

即：

\[
8\times0.5s=40\times0.1s=4.0s
\]

### 8.3 只 rollout 前 15 步

因为当前 RSSM 训练 horizon 是 15，直接用 40-step reward 不可靠。最终稳定版先用：

\[
H=15
\]

即：

\[
15\times0.1s=1.5s
\]

虽然 trajectory 覆盖 4 秒，但 scoring 只评估前 1.5 秒，剩余长期收益通过 value bootstrap 补。

### 8.4 controller 进入 imagination

对于每条候选轨迹，第 \(k\) 个 RSSM imagined step：

\[
a_{m,k}
=
C_{\text{ctrl}}
(
\hat{e}_{m,k},
\tilde{\tau}_m
)
\]

这里 controller 可以看到完整 reference path，而不是只看单个点。

然后：

\[
\hat{s}_{m,k+1}
=
\text{RSSM}_{\text{target}}
(
\hat{s}_{m,k},
a_{m,k}
)
\]

reward head：

\[
\hat{r}_{m,k}
=
R_{\text{rssm}}(\hat{s}_{m,k})
\]

continue head：

\[
\hat{c}_{m,k}
=
C_{\text{rssm}}(\hat{s}_{m,k})
\]

---

## 9. 15-step return + value bootstrap

对每条候选轨迹 \(\tau_m\)，计算 imagined return：

\[
J_m
=
\sum_{k=0}^{H-1}
\gamma^k
\left(
\prod_{j=0}^{k-1}\hat{c}_{m,j}
\right)
\hat{r}_{m,k}
+
\gamma^H
\left(
\prod_{j=0}^{H-1}\hat{c}_{m,j}
\right)
V(\hat{s}_{m,H})
\]

其中：

\[
H=15
\]

\[
\gamma=0.997
\]

\(V(\hat{s}_{m,H})\) 是 critic / value head 对 horizon 之后长期收益的估计。

不要一开始使用：

\[
H=40
\]

因为这会超出当前 RSSM 较可靠的 imagination horizon，容易产生 model exploitation。

---

## 10. selector ranking loss

### 10.1 group advantage

对同一个 latent 下的 \(M\) 条候选轨迹，得到 return：

\[
J_1,J_2,\ldots,J_M
\]

计算组内均值：

\[
\bar{J}
=
\frac{1}{M}\sum_{m=1}^{M}J_m
\]

标准差：

\[
\sigma_J
=
\sqrt{
\frac{1}{M}
\sum_{m=1}^{M}
(J_m-\bar{J})^2
}
\]

advantage：

\[
A_m
=
\frac{
J_m-\bar{J}
}{
\sigma_J+\epsilon
}
\]

clip：

\[
\tilde{A}_m
=
\operatorname{clip}(A_m,-A_{\max},A_{\max})
\]

建议：

\[
A_{\max}=2\sim3
\]

### 10.2 ranking loss

selector ranking loss：

\[
\mathcal{L}_{\text{rank}}
=
-
\sum_{m=1}^{M}
\operatorname{sg}(\tilde{A}_m)
\log\pi_\phi(m|s_t)
\]

含义：

```text
RSSM imagined return 高的候选 → selector 概率提高
RSSM imagined return 低的候选 → selector 概率降低
```

### 10.3 top-1 简化版本

也可以使用 top-1 简化版本：

\[
m^*
=
\arg\max_m J_m
\]

\[
\mathcal{L}_{\text{top1}}
=
-\log\pi_\phi(m^*|s_t)
\]

但推荐优先使用 advantage ranking，因为它能利用所有候选之间的相对优劣。

---

## 11. reference regularization

保留预训练 planner 作为 reference：

\[
\pi_{\text{ref}}(m|s_t)
\]

加入 KL 保护：

\[
\mathcal{L}_{\text{ref}}
=
\text{KL}
\left[
\pi_\phi(\cdot|s_t)
\Vert
\pi_{\text{ref}}(\cdot|s_t)
\right]
\]

或者 cross entropy：

\[
\mathcal{L}_{\text{ref-ce}}
=
-\sum_m
\pi_{\text{ref}}(m|s_t)
\log \pi_\phi(m|s_t)
\]

再加 entropy：

\[
\mathcal{L}_{\text{ent}}
=
-
\mathcal{H}(\pi_\phi(\cdot|s_t))
\]

最终 selector loss：

\[
\boxed{
\mathcal{L}_{\text{selector}}
=
\mathcal{L}_{\text{rank}}
+
\lambda_{\text{ref}}\mathcal{L}_{\text{ref}}
+
\lambda_{\text{ent}}\mathcal{L}_{\text{ent}}
}
\]

建议：

\[
\lambda_{\text{ref}}=0.05\sim0.2
\]

\[
\lambda_{\text{ent}}=0.001\sim0.01
\]

---

## 12. 可选 expert anchor CE 保护

如果 replay 中保留 expert trajectory，可以额外计算：

\[
m_{\text{gt}}
=
\arg\min_m
\text{ADE}(\tau_m,\tau_{\text{expert}})
\]

加入分类保护：

\[
\mathcal{L}_{\text{gt-cls}}
=
-\log\pi_\phi(m_{\text{gt}}|s_t)
\]

最终：

\[
\mathcal{L}_{\text{planner}}
=
\mathcal{L}_{\text{rank}}
+
\lambda_{\text{ref}}\mathcal{L}_{\text{ref}}
+
\lambda_{\text{ent}}\mathcal{L}_{\text{ent}}
+
\lambda_{\text{gt}}\mathcal{L}_{\text{gt-cls}}
\]

建议：

\[
\lambda_{\text{gt}}=0.05\sim0.1
\]

这个项只是保护预训练分布，不要太大，否则会抵消 RSSM ranking 的作用。

---

## 13. critic / value 训练

critic 输入 RSSM feature：

\[
f_t=[h_t,z_t]
\]

输出：

\[
V_\psi(f_t)
\]

critic target 使用 lambda-return：

\[
G_k^\lambda
=
\hat{r}_k
+
\gamma\hat{c}_k
\left[
(1-\lambda)V_\psi(\hat{s}_{k+1})
+
\lambda G_{k+1}^\lambda
\right]
\]

critic loss：

\[
\mathcal{L}_{\text{critic}}
=
\left\|
V_\psi(\hat{s}_k)
-
\operatorname{sg}(G_k^\lambda)
\right\|_2^2
\]

建议训练节奏：

```text
先更新 RSSM + critic
再冻结 rssm_target + critic_target
再更新 selector
```

planner 更新时：

\[
\boxed{
\text{RSSM frozen, critic frozen}
}
\]

selector 只接收来自 \(J_m\) 的 stop-gradient ranking signal。

---

## 14. 完整训练循环

```text
初始化：
    1. 加载离线 RSSM checkpoint
    2. 加载预训练 DiffusionDrive planner checkpoint
    3. 初始化 critic / value head
    4. 保存 ref_planner = pretrained planner
    5. rssm_target = rssm_online
    6. 冻结 planner body，只开放 selector_head

循环：
    A. CARLA 在线收集
        - 当前 planner + controller 执行闭环
        - 每 0.5s 规划一次
        - 每 0.1s 执行一次控制
        - 保存 online replay

    B. 更新 RSSM online
        - batch = offline replay + online replay
        - 优化 world model loss
        - 更新 reward / continue / value

    C. 同步 target RSSM
        - hard update 或 EMA update
        - planner 更新时固定 rssm_target

    D. selector ranking finetune
        - 从 replay 取 posterior latent s_t
        - planner 输出 M 条候选轨迹
        - 每条候选轨迹插值成 0.1s reference
        - controller 跟踪前 H=15 步
        - rssm_target imagination rollout
        - reward + continue + value 得到 J_m
        - 归一化得到 advantage A_m
        - 更新 selector_head

    E. 闭环评估
        - CARLA closed-loop success rate
        - collision rate
        - route progress
        - offroad rate
        - controller tracking error
        - open-loop ADE / FDE
```

整体形式：

```text
Collect
    ↓
Update RSSM
    ↓
Freeze target RSSM
    ↓
Score candidates
    ↓
Update selector
    ↓
Closed-loop eval
    ↓
Collect again
```

---

## 15. 分阶段实施路线

### 阶段 0：检查当前 baseline

先保留当前 Dreamer-style actor-critic 版本作为 baseline：

```text
full planner actor finetune
RSSM fixed
soft_select
lambda-return actor loss
```

记录其 closed-loop 指标，作为后续对比。

### 阶段 1：selector-only ranking finetune

冻结 planner body，只训练：

```text
selector_head / poses_cls
```

使用：

\[
H=15
\]

loss：

\[
\mathcal{L}_{\text{planner}}
=
\mathcal{L}_{\text{rank}}
+
\lambda_{\text{ref}}\mathcal{L}_{\text{ref}}
+
\lambda_{\text{ent}}\mathcal{L}_{\text{ent}}
\]

这是最稳的第一步。

### 阶段 2：打开 delta_head

如果 selector-only 稳定但提升有限，打开：

```text
selector_head
delta_head
```

加入轨迹 reference regularization：

\[
\mathcal{L}_{\text{traj-ref}}
=
\left\|
\tau_\theta
-
\operatorname{sg}(\tau_{\text{ref}})
\right\|_1
\]

总 loss：

\[
\mathcal{L}
=
\mathcal{L}_{\text{rank}}
+
\lambda_{\text{ref}}\mathcal{L}_{\text{ref}}
+
\lambda_{\text{traj}}\mathcal{L}_{\text{traj-ref}}
\]

建议：

\[
\lambda_{\text{traj}}=0.1\sim0.5
\]

### 阶段 3：打开最后几层 decoder

如果仍然不够，可以打开最后几层 decoder：

```text
selector_head
delta_head
last decoder layer
```

不建议一开始 full planner fine-tune。

推荐顺序：

```text
selector_head
→ selector_head + delta_head
→ selector_head + delta_head + last decoder layer
→ more decoder layers
```

---

## 16. horizon curriculum

初始：

\[
H=15
\]

如果 RSSM rollout 指标稳定，再逐步增加：

\[
15
\rightarrow
20
\rightarrow
25
\rightarrow
30
\]

暂时不建议直接用：

\[
H=40
\]

### 16.1 需要监控的 RSSM rollout 指标

latent rollout error：

\[
E_{\text{latent}}(k)
=
\left\|
z_{t+k}^{\text{imag}}
-
z_{t+k}^{\text{post}}
\right\|
\]

reward prediction error：

\[
E_{\text{reward}}(k)
=
\left|
\hat{r}_{t+k}
-
r_{t+k}
\right|
\]

ego state prediction error：

\[
E_{\text{ego}}(k)
=
\left\|
\hat{e}_{t+k}
-
e_{t+k}
\right\|
\]

只有当 \(k=20,25,30\) 的误差还能接受，才把 selector scoring horizon 加长。

---

## 17. 推荐默认超参数

| 模块 | 参数 | 建议值 |
|---|---:|---:|
| RSSM env step | \(\Delta t\) | 0.1s |
| Planner waypoint dt | \(\Delta t_{\text{wp}}\) | 0.5s |
| Planner waypoints | \(N\) | 8 |
| Initial imagine horizon | \(H\) | 15 |
| Discount | \(\gamma\) | 0.997 |
| Lambda return | \(\lambda\) | 0.95 |
| Offline replay ratio | \(\alpha\) | 0.7 → 0.5 |
| Advantage clip | \(A_{\max}\) | 2 ~ 3 |
| Reference KL weight | \(\lambda_{\text{ref}}\) | 0.05 ~ 0.2 |
| Entropy weight | \(\lambda_{\text{ent}}\) | 0.001 ~ 0.01 |
| Expert CE weight | \(\lambda_{\text{gt}}\) | 0.05 ~ 0.1 |
| Multi-step consistency | \(\lambda_k\) | 0.01 ~ 0.05 |
| Target RSSM EMA | \(\rho\) | 0.99 ~ 0.995 |

---

## 18. 与当前代码相比的关键改造点

当前版本：

```text
planner 作为 actor
每个 imagined step 重新调用 planner
soft_select 多模态轨迹
RSSM reward + critic lambda return
更新大部分 planner 参数
RSSM 不在线更新
```

最终稳定版：

```text
固定一条 candidate trajectory 进行 RSSM rollout
不用 soft average，显式 candidate scoring
只训练 selector_head
RSSM 单独 online update
planner 更新时使用 frozen target RSSM
15-step return + value bootstrap
真实执行和 imagination 对齐 0.5s planner / 0.1s controller
```

核心改造：

\[
\boxed{
\text{soft\_select actor loss}
\rightarrow
\text{candidate ranking selector loss}
}
\]

\[
\boxed{
\text{full planner update}
\rightarrow
\text{selector-only update}
}
\]

\[
\boxed{
\text{fixed offline RSSM}
\rightarrow
\text{online RSSM + frozen target RSSM}
}
\]

---

## 19. 最终推荐伪代码

```python
for outer_iter in range(num_outer_iters):

    # ============================================================
    # A. collect online replay
    # ============================================================
    for episode in range(num_collect_episodes):
        obs = env.reset()
        rssm_state = rssm_online.init_state()

        while not done:
            # observe current posterior
            rssm_state = rssm_online.observe_step(obs, prev_action)

            # planner runs every 0.5s
            candidate_trajs, selector_logits = planner(rssm_state)
            selected_idx = argmax(selector_logits)
            selected_traj = candidate_trajs[selected_idx]

            # interpolate 8 waypoints into 40 low-level references
            ref_path = interpolate(selected_traj, src_dt=0.5, dst_dt=0.1)

            # execute first 5 low-level actions
            for k in range(5):
                action = controller(ego_state, ref_path)
                next_obs, reward, done, info = env.step(action)

                online_replay.add(
                    obs=obs,
                    action=action,
                    reward=reward,
                    done=done,
                    next_obs=next_obs,
                    planner_waypoints=selected_traj,
                    selected_idx=selected_idx,
                    tracking_error=info["tracking_error"],
                )

                obs = next_obs
                if done:
                    break

    # ============================================================
    # B. update RSSM online
    # ============================================================
    for wm_step in range(num_wm_updates):
        batch = mixed_sample(
            offline_replay,
            online_replay,
            offline_ratio=0.7,
        )
        wm_loss = dreamerv3_world_model_loss(rssm_online, batch)
        rssm_online.update(wm_loss)

    # ============================================================
    # C. sync target RSSM
    # ============================================================
    rssm_target.ema_update(rssm_online, rho=0.995)

    # ============================================================
    # D. update critic / value
    # ============================================================
    for critic_step in range(num_critic_updates):
        batch = mixed_sample(offline_replay, online_replay)
        states = rssm_target.posterior_states(batch)

        imagined = imagine_with_current_selector(
            rssm_target,
            planner,
            controller,
            states,
            horizon=15,
        )

        target_return = lambda_return(
            rewards=imagined.rewards,
            continues=imagined.continues,
            values=critic(imagined.states),
            gamma=0.997,
            lambda_=0.95,
        )

        critic_loss = mse(
            critic(imagined.states),
            stop_gradient(target_return),
        )
        critic.update(critic_loss)

    # ============================================================
    # E. selector ranking update
    # ============================================================
    for planner_step in range(num_selector_updates):
        batch = mixed_sample(offline_replay, online_replay)
        states = rssm_target.posterior_states(batch)

        candidate_trajs, logits = planner(states)
        probs = softmax(logits)

        returns = []
        for traj_m in candidate_trajs:
            ref_m = interpolate(traj_m, src_dt=0.5, dst_dt=0.1)

            J_m = rssm_score_candidate(
                rssm_target=rssm_target,
                critic=critic_target,
                controller=controller,
                init_state=states,
                ref_path=ref_m,
                horizon=15,
            )
            returns.append(J_m)

        returns = stack(returns, axis=-1)
        advantage = normalize(returns, axis=-1)
        advantage = clip(advantage, -3.0, 3.0)

        rank_loss = -sum(
            stop_gradient(advantage) * log_softmax(logits),
            axis=-1,
        ).mean()

        ref_logits = ref_planner.selector_logits(states)
        ref_probs = softmax(ref_logits)

        ref_loss = kl_divergence(probs, ref_probs).mean()
        ent_loss = -entropy(probs).mean()

        planner_loss = (
            rank_loss
            + lambda_ref * ref_loss
            + lambda_ent * ent_loss
        )

        selector_head.update(planner_loss)
```

---

## 20. 最终一句话总结

最终稳定版不是：

\[
\text{DreamerV3 直接全量 RL 微调整个 Diffusion planner}
\]

而是：

\[
\boxed{
\text{DreamerV3 online RSSM}
+
\text{target RSSM imagination scoring}
+
\text{DiffusionDrive candidate ranking}
+
\text{selector-only finetune}
+
\text{15-step return bootstrap}
}
\]

这样可以同时避免三个主要风险：

```text
1. RSSM 长 horizon 不准
2. planner 钻 world model 漏洞
3. 全量 RL 微调把 diffusion planner 拉崩
```

这也是当前 DreamerV3 + DiffusionDrive + CARLA 闭环系统中最稳、最可控、最适合逐步实现的方案。
