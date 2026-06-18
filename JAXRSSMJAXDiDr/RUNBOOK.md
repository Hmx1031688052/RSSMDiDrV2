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

Use your existing expert collection pipeline under `dreamerv3/`, then prepare
the replay through the JAX package:

```bash
python -m JAXRSSMJAXDiDr.scripts.prepare_polyplanner_replay \
  --replay_dir "$REPLAY_DIR"
```

The JAX planner pipeline consumes the same Dreamer row-format replay fields as
the previous RSSM/DiDr scripts.

Optional: replace `expert_waypoints8` with the ego vehicle's actually executed
future trajectory. Use this when you want the planner target to be the
closed-loop ego path from replay rather than the PolyPlanner/global-route
target. The script edits replay chunks in place, so copy the replay directory
first if you want to keep the original expert targets:

```bash
cp -a "$REPLAY_DIR" "${REPLAY_DIR}_polyplanner_backup"

python -u dreamerv3/replace_expert_with_ego_traj.py \
  --replay_dir "$REPLAY_DIR" \
  --waypoint_scale 30.0 \
  --dt 0.1 \
  --waypoint_interval 5
```

This writes:

```text
expert_waypoints8: [T, 16]
```

as 8 future ego-frame waypoints, spaced by `waypoint_interval * dt`
seconds. With the defaults above, this is 8 waypoints at 0.5s intervals,
covering a 4.0s horizon. The replay chunks must contain `ego_x`, `ego_y`, and
preferably `ego_yaw`; chunks without ego positions are skipped. Run this before
building anchors or exporting the planner dataset. If you already built
`$ANCHOR_PATH` or `$RUN_ROOT/jax_planner_dataset`, regenerate them after this
replacement.

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
```

If not, train one with the existing JAX RSSM implementation:

```bash
python -m JAXRSSMJAXDiDr.scripts.train_offline_rssm \
  --task carla_roundabout \
  --replay_dir "$REPLAY_DIR" \
  --logdir "$RUN_ROOT/jax_rssm" \
  --batch_length 64 \
  --batch_size 16 \
  --updates 100000 \
  --structured_world_model

export JAX_RSSM_CKPT=$RUN_ROOT/jax_rssm/checkpoint.ckpt
```

This checkpoint is the source of truth for RSSM. Do not train the Torch RSSM for
this JAX planner path.

## 3. Export JAX RSSM Latents

Export posterior latents from the JAX RSSM checkpoint:

```bash
python -m JAXRSSMJAXDiDr.scripts.export_rssm_latents \
  --checkpoint "$JAX_RSSM_CKPT" \
  --replay_dir "$REPLAY_DIR" \
  --output_dir "$RUN_ROOT/jax_latents" \
  --task carla_roundabout \
  --structured_world_model
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
python -m JAXRSSMJAXDiDr.scripts.export_planner_dataset \
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

Build plan anchors from the prepared replay:

```bash
python -m JAXRSSMJAXDiDr.tools.kmeans_polyplanner_anchors \
  --replay_dir "$REPLAY_DIR" \
  --output "$ANCHOR_PATH" \
  --num_modes 30
python -m JAXRSSMJAXDiDr.tools.checkanchors \
  --anchor_path /media/pc/T7/Hmx_RssmDIDR/RSSMDiDrV2/outputs/jaxrssm_jaxdidr/anchors.npy
```

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
  --trajectory_cls_weight 10.0 \
  --trajectory_reg_weight 8.0 \
  --latent_noise_std 0.03

```

The supervised planner loss follows `DiffusionDriveV2/modules/multimodal_loss.py`:

```text
positive mode = argmin mean L2(target_xy, plan_anchor_xy)
cls_loss = 10.0 * sigmoid_focal_loss(poses_cls, one_hot(positive mode), gamma=2, alpha=0.25)
reg_loss = 8.0 * L1(poses_reg[positive mode], target_trajectory)
loss = cls_loss + reg_loss
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
  --dataset_dir "/media/pc/T7/Hmx_RssmDIDR/RSSMDiDrV2/outputs/jaxrssm_jaxdidr/jax_planner_dataset" \
  --planner_checkpoint "/media/pc/T7/Hmx_RssmDIDR/RSSMDiDrV2/outputs/jaxrssm_jaxdidr/jax_didr_planner/epoch_0068.pkl.gz" \
  --output_dir "/media/pc/T7/Hmx_RssmDIDR/RSSMDiDrV2/outputs/jaxrssm_jaxdidr/jax_didr_eval_openloop" \
  --eval_timestep 0 \
  --save_predictions
```

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
  --ctrl_smooth_weight 0.05 \
  --structured_world_model
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
  --output_dir "$RUN_ROOT/jax_didr_online_mixed" \
  --structured_world_model
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
  --dreamerv3.encoder.cnn_keys none \
  --dreamerv3.decoder.cnn_keys none \
  --dreamerv3.encoder.mlp_keys "ego_.*|neighbor_vehicles_local|route_waypoints8|global_path_ego|global_path_ego_mask|target_region|route_remaining" \
  --dreamerv3.decoder.mlp_keys "ego_.*|neighbor_vehicles_local|route_waypoints8|global_path_ego|global_path_ego_mask|target_region|route_remaining" \
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
  --eval_timestep 8 \
  --dreamerv3.encoder.cnn_keys none \
  --dreamerv3.decoder.cnn_keys none \
  --dreamerv3.encoder.mlp_keys "ego_.*|neighbor_vehicles_local|route_waypoints8|global_path_ego|global_path_ego_mask|target_region|route_remaining" \
  --dreamerv3.decoder.mlp_keys "ego_.*|neighbor_vehicles_local|route_waypoints8|global_path_ego|global_path_ego_mask|target_region|route_remaining"
```
Online replay must keep:

```text
row t = obs_t + action_t + reward_t + is_terminal_t
action == executed_control
```

Do not store waypoint as `action`.

## 9. VAD Six-Camera Perception Latents

All VAD/CARLA helper programs for the JAX-only path live under
`JAXRSSMJAXDiDr/`.

Collect replay with VAD-style six RGB cameras:

```bash
python -m JAXRSSMJAXDiDr.scripts.collect_vad_sixcam_replay \
  --task carla_roundabout \
  --dreamerv3.logdir "$RUN_ROOT/vad_sixcam_collect" \
  --env.expert_collection.episodes 1000 \
  --env.planner_target.use_waypoint_action False
```

The camera keys are written in VAD/nuScenes order:

```text
camera_front
camera_front_right
camera_front_left
camera_back
camera_back_left
camera_back_right
```

Export compact VAD scene latents from collected replay:

```bash
python -m JAXRSSMJAXDiDr.scripts.export_vad_latents \
  --replay_dir "$RUN_ROOT/vad_sixcam_collect/replay" \
  --output_dir "$RUN_ROOT/vad_latent_replay" \
  --vad_root ./VAD \
  --vad_model tiny \
  --vad_checkpoint ./VAD/ckpts/VAD_tiny_stage_2.pth
```

This writes `vad_scene_latent` to each replay chunk. By default it concatenates
pooled VAD BEV features, updated agent query tokens, updated map query tokens,
and the ego query. Use `--latent_components bev` if you want the earlier
BEV-only latent. Use `vad_scene_latent` as an RSSM MLP observation key rather
than feeding VAD's final ego trajectory into RSSM.

Recommended starting point:

```text
VAD-Tiny first: 100x100 BEV, 3 encoder layers, queue_length=3, about 16.8 FPS in the VAD README.
VAD-Base later: 200x200 BEV, 6 encoder layers, queue_length=4, better open-loop metrics but about 4.5 FPS.
```

## 10. Smoke Tests

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

## 11. Output Summary

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
