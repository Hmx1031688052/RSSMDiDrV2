# GRPO Online Debug Order

This note records the current debug sequence for `train_jax_grpo_world_model_finetune.py`.
Do not reuse a collapsed GRPO checkpoint between experiments. Each run below should start from the same original RSSM and planner checkpoints.

Bench2Drive note: external Bench2Drive reference directories should not be pushed. The root `.gitignore` ignores `/Bench2Drive/` and `/Bench2DriveZoo-uniad-vad/`, while keeping `JAXRSSMJAXDiDr/bench2drive/*.py` available for this repo.

## 0. Common Variables

Edit these paths first in Ubuntu bash:

```bash
export OFFLINE_REPLAY="/path/to/offline_replay"
export ONLINE_REPLAY="/path/to/online_replay"
export RSSM_CKPT="/path/to/rssm/checkpoint.ckpt"
export PLANNER_CKPT="/path/to/planner.pkl.gz"
export RUN_ROOT="/path/to/ForDebug/runs/grpo_debug"
```

Append your usual Dreamer/CARLA extra args at the end of each command if your normal training command needs them.

## 1. RSSM-only Control

Purpose: check whether online mixed replay RSSM training alone already breaks the posterior/reward model before any planner update.

```bash
python -m JAXRSSMJAXDiDr.scripts.train_jax_grpo_world_model_finetune \
  --online_schedule train_only \
  --outer_iterations 1 \
  --offline_replay_dir "$OFFLINE_REPLAY" \
  --online_replay_dir "$ONLINE_REPLAY" \
  --rssm_checkpoint "$RSSM_CKPT" \
  --planner_checkpoint "$PLANNER_CKPT" \
  --output_dir "$RUN_ROOT/01_rssm_only" \
  --wm_updates 200 \
  --grpo_updates 0 \
  --selector_after_grpo_updates 0
```

Interpretation: if closed-loop behavior degrades after this run, the main suspect is RSSM online update drift, not GRPO.

## 2. Minimal GRPO-only Control

Purpose: freeze RSSM updates and test whether one tiny trajectory-body GRPO update immediately damages basic forward motion.

```bash
python -m JAXRSSMJAXDiDr.scripts.train_jax_grpo_world_model_finetune \
  --online_schedule train_only \
  --outer_iterations 1 \
  --offline_replay_dir "$OFFLINE_REPLAY" \
  --online_replay_dir "$ONLINE_REPLAY" \
  --rssm_checkpoint "$RSSM_CKPT" \
  --planner_checkpoint "$PLANNER_CKPT" \
  --output_dir "$RUN_ROOT/02_grpo_only_min" \
  --wm_updates 0 \
  --grpo_updates 1 \
  --grpo_lr 1e-6 \
  --selector_after_grpo_updates 0 \
  --grpo_reward_viz_every 1 \
  --grpo_reward_viz_samples 1 \
  --grpo_reward_viz_topk 5
```

Check:

```bash
ls -lh "$RUN_ROOT/02_grpo_only_min/grpo_reward_viz"
head -n 1 "$RUN_ROOT/02_grpo_only_min/grpo_reward_viz/rewards.jsonl"
```

Interpretation: if this already triggers early `too_few_forward_points`, fixed-RSSM GRPO itself has a reward/sign/scale problem.

## 3. RSSM + Minimal GRPO

Purpose: add RSSM online update back while keeping GRPO tiny. This isolates the RSSM reward drift plus GRPO coupling.

```bash
python -m JAXRSSMJAXDiDr.scripts.train_jax_grpo_world_model_finetune \
  --online_schedule train_only \
  --outer_iterations 1 \
  --offline_replay_dir "$OFFLINE_REPLAY" \
  --online_replay_dir "$ONLINE_REPLAY" \
  --rssm_checkpoint "$RSSM_CKPT" \
  --planner_checkpoint "$PLANNER_CKPT" \
  --output_dir "$RUN_ROOT/03_rssm_plus_grpo_min" \
  --wm_updates 200 \
  --grpo_updates 1 \
  --grpo_lr 1e-6 \
  --selector_after_grpo_updates 0 \
  --grpo_reward_viz_every 1 \
  --grpo_reward_viz_samples 1 \
  --grpo_reward_viz_topk 5
```

Interpretation: if step 2 is stable but this run degrades, RSSM online update drift is likely feeding bad rewards into GRPO.

## 4. Selector-only Control

Purpose: test whether RSSM-ranked selector tuning alone chooses stop-like modes.

```bash
python -m JAXRSSMJAXDiDr.scripts.train_jax_grpo_world_model_finetune \
  --online_schedule train_only \
  --outer_iterations 1 \
  --offline_replay_dir "$OFFLINE_REPLAY" \
  --online_replay_dir "$ONLINE_REPLAY" \
  --rssm_checkpoint "$RSSM_CKPT" \
  --planner_checkpoint "$PLANNER_CKPT" \
  --output_dir "$RUN_ROOT/04_selector_only" \
  --wm_updates 0 \
  --grpo_updates 0 \
  --selector_after_grpo_updates 2
```

Interpretation: if this degrades, selector ranking with RSSM return can pick a stop-like mode even without trajectory-body GRPO.

## 5. What To Compare

For each run, inspect:

- training metrics: `finite_step`, `target_used`, `positive_ratio`, `unsafe_ratio`, `rl_loss`, `il_loss`, `reward_mean`, `reward_positive_mean`
- closed-loop symptoms: early `invalid_plan=too_few_forward_points`, selected mode, short decoded trajectory x values
- reward visualization: whether high-return trajectories are short/stop-like or genuinely forward-progressing

Expected diagnosis:

- Step 1 fails: RSSM online update/posterior/reward drift is already enough to hurt behavior.
- Step 2 fails: GRPO objective under fixed RSSM is unsafe; check reward sign, coordinate sign, advantage gating, and IL strength.
- Step 2 passes but step 3 fails: RSSM online drift plus GRPO reward coupling is the main issue.
- Step 4 fails: selector-light ranking is independently biased toward stop-like modes.
