# JAXRSSMJAXDiDr Runbook

This version keeps the original `dreamerv3/` JAX RSSM/world model and ports the
DiffusionDrive planner side to JAX.

Important rule:

```text
RSSM: reuse existing JAX DreamerV3 checkpoint, no Torch rewrite required.
Planner: train a new JAX DiffusionDrive checkpoint from RSSM latents.
Online finetune: update JAX planner + JAX critic through Dreamer RSSM imagination.
```

All commands below are Ubuntu/bash style.

## 0. Environment

Run from the repository root:

```bash
cd /media/pc/T7/Hmx_RssmDIDR/TorchWMDIDR
```

Set paths:

```bash
export RUN_ROOT=./outputs/jaxrssm_jaxdidr
export COLLECT_LOGDIR=${RUN_ROOT}/expert_collect
export REPLAY_DIR=${COLLECT_LOGDIR}/replay
export LATENT_DIR=${RUN_ROOT}/rssm_latents
export PLANNER_DATASET_DIR=${RUN_ROOT}/planner_dataset
export ANCHOR_PATH=${RUN_ROOT}/anchors.npy
export PLANNER_CKPT_DIR=${RUN_ROOT}/planner_checkpoints


mkdir -p "$RUN_ROOT"
```

Expected replay fields include at least:

```text
action
reward
is_first
is_terminal
birdeye_wpt
ego_x
ego_y
ego_yaw
ego_speed
ego_yawrate
neighbor_vehicles_local
expert_waypoints8
```

## 1. Expert Replay

If you already have prepared PolyPlanner expert replay in `$REPLAY_DIR`, skip
this step.

Use your existing expert collection/preparation pipeline under `dreamerv3/` and
`RSSMDiDrOnCarla/`. The JAX planner pipeline consumes the same replay format as
the previous RSSM/DiDr scripts.

Quick check:

```bash
python - <<'PY'
from pathlib import Path
import numpy as np
import os
root = Path(os.environ["REPLAY_DIR"]).expanduser()
path = sorted(root.glob("*.npz"))[0]
data = np.load(path, allow_pickle=True)
print(path)
for key in ["action", "reward", "birdeye_wpt", "ego_speed", "neighbor_vehicles_local", "expert_waypoints8"]:
    print(key, data[key].shape, data[key].dtype)
PY
```

## 2. JAX RSSM Checkpoint

If you already have a good JAX DreamerV3 RSSM checkpoint, reuse it:

```bash
export JAX_RSSM_CKPT=/media/pc/T7/Hmx_RssmDIDR/RSSMDIDR/outputs/rssm_didr_roundabout/offline_rssm/checkpoint.ckpt

If not, train one with the existing JAX RSSM implementation:

```bash
python -m RSSMDiDrOnCarla.scripts.train_offline_rssm \
  --replay_dir "$REPLAY_DIR" \
  --logdir "$RUN_ROOT/jax_rssm" \
  --batch_length 64 \
  --batch_size 16 \
  --updates 100000

export JAX_RSSM_CKPT=$RUN_ROOT/jax_rssm/checkpoint.ckpt
```

This checkpoint is the source of truth for RSSM. Do not train the Torch RSSM for
this JAX planner path.

## 3. Export JAX RSSM Latents

Export posterior latents from the JAX RSSM checkpoint:

```bash
python -m RSSMDiDrOnCarla.scripts.export_rssm_latents \
  --checkpoint "$JAX_RSSM_CKPT" \
  --replay_dir "$REPLAY_DIR" \
  --output_dir "$RUN_ROOT/jax_latents" \
  --task carla_roundabout
```

Expected output:

```text
$RUN_ROOT/jax_latents/*.npz
$RUN_ROOT/jax_latents/rssm_latents_summary.json
```

Each latent chunk should contain:

```text
rssm_latent: [T, D]
deter / stoch components unless --no_save_components was used
```

## 4. Build Planner Dataset

Pair replay targets with exported RSSM latents:

```bash
python -m RSSMDiDrOnCarla.scripts.export_planner_dataset \
  --replay_dir "$REPLAY_DIR" \
  --latent_dir "$RUN_ROOT/jax_latents" \
  --output_dir "$RUN_ROOT/jax_planner_dataset"
```

Expected output chunks:

```text
rssm_latent: [T, D]
expert_waypoints8: [T, 16]
trajectory: [T, 8, 3]
```

Quick check:

```bash
python - <<'PY'
from pathlib import Path
import numpy as np
import os
root = Path(os.environ["RUN_ROOT"]) / "jax_planner_dataset"
path = sorted(root.glob("*.npz"))[0]
data = np.load(path, allow_pickle=True)
print(path)
for key in ["rssm_latent", "expert_waypoints8", "trajectory"]:
    print(key, data[key].shape, data[key].dtype)
PY
```

## 5. Pretrain JAX DiffusionDrive Planner

Train the JAX planner from scratch:

```bash
python -m JAXRSSMJAXDiDr.scripts.train_jax_planner_pretrain \
  --dataset_dir "$RUN_ROOT/jax_planner_dataset" \
  --anchor_path "$ANCHOR_PATH" \
  --output_dir "$RUN_ROOT/jax_didr_planner" \
  --epochs 200 \
  --batch_size 128 \
  --lr 1e-4 \
  --weight_decay 1e-4 \
  --dropout 0.1 \
  --latent_noise_std 0.03

```
export RUN_ROOT=./outputs/jaxrssm_jaxdidr
export REPLAY_DIR="$RUN_ROOT/expert_collect/replay"
export JAX_RSSM_CKPT="$RUN_ROOT/jax_rssm/checkpoint.ckpt"
export ANCHOR_PATH="$RUN_ROOT/anchors.npy"
export PLANNER_CKPT="$RUN_ROOT/jax_didr_planner/best.pkl.gz"
python -m JAXRSSMJAXDiDr.scripts.train_jax_stable_online_finetune \
  --task carla_roundabout \
  --offline_replay_dir "$REPLAY_DIR" \
  --rssm_checkpoint "/media/pc/T7/Hmx_RssmDIDR/RSSMDiDrV2/outputs/jaxrssm_jaxdidr/jax_stable_online/rssm_online/checkpoint.ckpt" \
  --planner_checkpoint "/media/pc/T7/Hmx_RssmDIDR/RSSMDiDrV2/outputs/jaxrssm_jaxdidr/jax_stable_online/planner_selector_online_outer_0007.pkl.gz" \
  --anchor_path "$ANCHOR_PATH" \
  --output_dir "$RUN_ROOT/jax_stable_online_continue616" \
  --outer_iterations 30 \
  --collect_episodes 20 \
  --max_steps 1000 \
  --plan_interval_steps 5 \
  --wm_updates 200 \
  --selector_updates 300 \
  --batch_length 64 \
  --batch_size 16 \
  --score_horizon_steps 15 \
  --offline_ratio 0.7 \
  --eval_timestep 8 \
  --jax_platform gpu \
  --dreamerv3.jax.train_devices=0 \
  --env.planner_target.use_waypoint_action False \
  --dreamerv3.encoder.cnn_keys "birdeye_wpt" \
  --dreamerv3.encoder.mlp_keys "ego_speed|ego_yawrate|ego_x|ego_y|ego_yaw" \
  --dreamerv3.decoder.cnn_keys "birdeye_wpt" \
  --dreamerv3.decoder.mlp_keys "ego_speed|ego_yawrate"
The supervised planner loss matches the TorchDiDr planner:

```text
loss =
  reg_loss
  + cls_loss
  + 0.05 * path_length_loss
  + 0.05 * step_length_loss
  + 0.01 * smooth_loss
```

Expected outputs:

```text
$RUN_ROOT/jax_didr_planner/best.pkl.gz
$RUN_ROOT/jax_didr_planner/last.pkl.gz
$RUN_ROOT/jax_didr_planner/history.json
$RUN_ROOT/jax_didr_planner/run_config.json
```

## 6. Open-Loop Evaluation

Evaluate ADE/FDE on the planner dataset:

```bash
python -m JAXRSSMJAXDiDr.scripts.eval_open_loop \
  --dataset_dir "$RUN_ROOT/jax_planner_dataset" \
  --planner_checkpoint "$RUN_ROOT/jax_didr_planner/best.pkl.gz" \
  --output_dir "$RUN_ROOT/jax_didr_eval" \
  --eval_timestep 0 \
  --save_predictions
```
python -m JAXRSSMJAXDiDr.scripts.eval_close_loop \
  --task carla_roundabout \
  --rssm_checkpoint "/media/pc/T7/Hmx_RssmDIDR/RSSMDiDrV2/outputs/jaxrssm_jaxdidr/jax_rssm/checkpoint.ckpt" \
  --planner_checkpoint "/media/pc/T7/Hmx_RssmDIDR/RSSMDiDrV2/outputs/jaxrssm_jaxdidr/jax_didr_planner/best.pkl.gz" \
  --anchor_path "/media/pc/T7/Hmx_RssmDIDR/RSSMDiDrV2/outputs/rssm_didr_roundabout/anchors.npy" \
  --output_dir "/media/pc/T7/Hmx_RssmDIDR/RSSMDiDrV2/outputs/jaxrssm_jaxdidr/jax_didr_closed_loop_vis" \
  --episodes 10 \
  --max_steps 1000 \
  --device auto \
  --jax_platform cpu \
  --eval_timestep 0 \
  --env.planner_target.use_waypoint_action False \
  --dreamerv3.jax.train_devices=0 \
  --dreamerv3.encoder.cnn_keys "birdeye_wpt" \
  --dreamerv3.encoder.mlp_keys "ego_speed|ego_yawrate|ego_x|ego_y|ego_yaw" \
  --dreamerv3.decoder.cnn_keys "birdeye_wpt" \
  --dreamerv3.decoder.mlp_keys "ego_speed|ego_yawrate" \
  --live_plot \
  --plot_modes 6 \
  --carla_live_draw \
  --carla_spectator_topdown
Expected outputs:

```text
$RUN_ROOT/jax_didr_eval/metrics.json
$RUN_ROOT/jax_didr_eval/predictions.npz
```

## 7. Dreamer-Style JAX Planner/Critic Finetune

This step uses the frozen/reused JAX Dreamer RSSM checkpoint for imagination and
updates:

```text
JAX DiffusionDrive planner
JAX value critic
```

It does not reimplement or retrain the RSSM.

Run:

```bash
python -m JAXRSSMJAXDiDr.scripts.train_jax_online_finetune \
  --offline_replay_dir "$REPLAY_DIR" \
  --rssm_checkpoint "$JAX_RSSM_CKPT" \
  --planner_checkpoint "$RUN_ROOT/jax_didr_planner/best.pkl.gz" \
  --output_dir "$RUN_ROOT/jax_didr_online" \
  --task carla_roundabout \
  --iterations 1000 \
  --batch_length 64 \
  --batch_size 16 \
  --imag_horizon 15 \
  --actor_updates 1 \
  --critic_updates 1 \
  --actor_lr 3e-5 \
  --critic_lr 3e-5 \
  --discount 0.997 \
  --lambda_ 0.95 \
  --bc_weight 0.1 \
  --wp_smooth_weight 0.05 \
  --ctrl_smooth_weight 0.05
```

Actor update data flow:

```text
replay sequence
 -> DreamerV3 encoder + RSSM observe
 -> posterior final state
 -> JAX DiDr deterministic planner
 -> soft mode selection
 -> JAX differentiable PID pure-pursuit controller
 -> DreamerV3 RSSM img_step
 -> DreamerV3 reward / continue heads
 -> JAX critic lambda return
 -> actor loss backprop to planner
```

Actor loss:

```text
actor_loss =
  - imagined_return
  + bc_weight * pretrained_planner_waypoint_bc
  + wp_smooth_weight * waypoint_smoothness
  + ctrl_smooth_weight * action_smoothness
```

Critic loss:

```text
critic_loss = symlog-MSE(value(features), lambda_return)
```

Expected outputs:

```text
$RUN_ROOT/jax_didr_online/planner_online.pkl.gz
$RUN_ROOT/jax_didr_online/critic_online.pkl.gz
$RUN_ROOT/jax_didr_online/history.json
$RUN_ROOT/jax_didr_online/run_config.json
```

## 8. Optional Online Replay Mixing

If you have CARLA online replay chunks saved in Dreamer row format, pass them:

```bash
python -m JAXRSSMJAXDiDr.scripts.train_jax_online_finetune \
  --offline_replay_dir "$REPLAY_DIR" \
  --online_replay_dir "$RUN_ROOT/jax_online_replay" \
  --rssm_checkpoint "$JAX_RSSM_CKPT" \
  --planner_checkpoint "$RUN_ROOT/jax_didr_planner/best.pkl.gz" \
  --output_dir "$RUN_ROOT/jax_didr_online_mixed"
```

Evaluate the pretrained JAX planner:

```bash
python -m JAXRSSMJAXDiDr.scripts.eval_close_loop \
  --task carla_roundabout \
  --rssm_checkpoint "$JAX_RSSM_CKPT" \
  --planner_checkpoint "$RUN_ROOT/jax_didr_planner/best.pkl.gz" \
  --output_dir "$RUN_ROOT/jax_didr_closed_loop" \
  --episodes 10 \
  --max_steps 1000 \
  --plan_interval_steps 5 \
  --eval_timestep 0 \
  --env.planner_target.use_waypoint_action False \
  --dreamerv3.encoder.cnn_keys "birdeye_wpt" \
  --dreamerv3.encoder.mlp_keys "ego_speed|ego_yawrate|ego_x|ego_y|ego_yaw" \
  --dreamerv3.decoder.cnn_keys "birdeye_wpt" \
  --dreamerv3.decoder.mlp_keys "ego_speed|ego_yawrate" \
  --live_plot \
  --plot_modes 6 \
```

Evaluate the selector-ranking finetuned planner:

```bash
python -m JAXRSSMJAXDiDr.scripts.eval_close_loop \
  --task carla_roundabout \
  --rssm_checkpoint "$JAX_RSSM_CKPT" \
  --planner_checkpoint "$RUN_ROOT/jax_didr_selector_ranking/planner_selector_ranking.pkl.gz" \
  --output_dir "$RUN_ROOT/jax_didr_selector_ranking_closed_loop" \
  --episodes 10 \
  --max_steps 1000 \
  --plan_interval_steps 5 \
  --eval_timestep 8
```
Online replay must keep:

```text
row t = obs_t + action_t + reward_t + is_terminal_t
action == executed_control
```

Do not store waypoint as `action`.

## 9. Smoke Tests

Compile:

```bash
python -m compileall JAXRSSMJAXDiDr
```

Minimal JAX component smoke:

```bash
python - <<'PY'
import tempfile
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from JAXRSSMJAXDiDr.models import (
    JAXDiDrConfig, init_planner, loss_and_metrics,
    differentiable_pidpp, init_critic, critic_value, lambda_return,
)

with tempfile.TemporaryDirectory() as tmp:
    anchors = np.random.randn(20, 8, 2).astype(np.float32)
    anchor_path = Path(tmp) / "anchors.npy"
    np.save(anchor_path, anchors)
    cfg = JAXDiDrConfig(latent_dim=64, plan_anchor_path=str(anchor_path), hidden_dim=64, decoder_ffn_dim=128)
    params = init_planner(jax.random.PRNGKey(0), cfg)
    batch = {"rssm_latent": jnp.zeros((3, 64)), "trajectory": jnp.zeros((3, 8, 3))}
    loss, _ = loss_and_metrics(params, cfg, jax.random.PRNGKey(1), batch)
    assert jnp.isfinite(loss)
    action = differentiable_pidpp(jnp.zeros((3, 8, 2)), jnp.zeros((3, 1)), cfg)
    assert action.shape == (3, 2)
    critic = init_critic(jax.random.PRNGKey(2), 64, hidden_dim=32)
    value = critic_value(critic, jnp.zeros((4, 3, 64)))
    assert value.shape == (4, 3)
    ret = lambda_return(jnp.ones((5, 3)), jnp.zeros((5, 3)), jnp.ones((5, 3)) * 0.99, jnp.zeros((3,)))
    assert ret.shape == (5, 3)
print("JAXRSSMJAXDiDr smoke OK")
PY
```

## 10. Output Summary

Main reusable artifacts:

```text
JAX RSSM checkpoint:
  $JAX_RSSM_CKPT

JAX planner pretrain checkpoint:
  $RUN_ROOT/jax_didr_planner/best.pkl.gz

JAX online planner checkpoint:
  $RUN_ROOT/jax_didr_online/planner_online.pkl.gz

JAX online critic checkpoint:
  $RUN_ROOT/jax_didr_online/critic_online.pkl.gz
```

Use the JAX RSSM checkpoint plus JAX planner checkpoint together for future
closed-loop CARLA evaluation.
