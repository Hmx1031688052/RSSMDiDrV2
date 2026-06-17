"""Pure JAX DiffusionDrive-style planner conditioned on JAX RSSM latents.

The module intentionally avoids Flax/Haiku so it can live beside the local
DreamerV3 code without introducing another module system. Parameters are plain
JAX pytrees and checkpoints are small pickle payloads with a JSON-like config.
"""

from __future__ import annotations

import gzip
import pickle
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Dict, Optional

import jax
import jax.numpy as jnp
import numpy as np


PyTree = Dict[str, Any]


@dataclass
class JAXDiDrConfig:
    latent_dim: int
    plan_anchor_path: str
    condition_key: str = "rssm_latent"
    hidden_dim: int = 256
    num_modes: int = 20
    num_poses: int = 8
    waypoint_scale: float = 30.0
    diffusion_train_steps: int = 1000
    truncated_train_steps: int = 50
    truncated_eval_step: int = 8
    decoder_layers: int = 2
    decoder_heads: int = 4
    decoder_ffn_dim: int = 512
    dropout: float = 0.1
    reg_loss_weight: float = 8.0
    cls_loss_weight: float = 10.0
    actor_softmax_temperature: float = 1.0
    waypoint_dt: float = 0.5
    wheelbase: float = 2.875
    max_steer_rad: float = 0.65
    steer_sign: float = -1.0
    steer_gain: float = 0.7
    lookahead_min: float = 4.5
    lookahead_max: float = 14.0
    lookahead_gain: float = 1.0
    speed_kp: float = 1.0
    ctrl_acc_min: float = -3.0
    ctrl_acc_max: float = 3.0
    ctrl_target_speed_max: float = 8.0
    ctrl_soft_lookup_temp: float = 0.75
    latent_noise_std: float = 0.0
    latent_dropout: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "JAXDiDrConfig":
        valid = {item.name for item in fields(cls)}
        return cls(**{key: value for key, value in dict(data).items() if key in valid})


def _split(rng, count: int):
    return jax.random.split(rng, count)


def _uniform(rng, shape, bound: float):
    return jax.random.uniform(rng, shape, dtype=jnp.float32, minval=-bound, maxval=bound)


def _xavier_uniform(rng, shape):
    fan_in, fan_out = shape[0], shape[1]
    bound = jnp.sqrt(6.0 / float(fan_in + fan_out))
    return _uniform(rng, shape, bound)


def _linear_init(rng, in_dim: int, out_dim: int, zero_bias: bool = False):
    w_rng, b_rng = _split(rng, 2)
    bound = jnp.sqrt(1.0 / float(in_dim))
    bias = jnp.zeros((out_dim,), jnp.float32) if zero_bias else _uniform(b_rng, (out_dim,), bound)
    return {"w": _uniform(w_rng, (in_dim, out_dim), bound), "b": bias}


def _mha_in_proj_init(rng, dim: int):
    bound = jnp.sqrt(6.0 / float(dim + 3 * dim))
    return {"w": _uniform(rng, (dim, dim), bound), "b": jnp.zeros((dim,), jnp.float32)}


def _ln_init(dim: int):
    return {"scale": jnp.ones((dim,), jnp.float32), "bias": jnp.zeros((dim,), jnp.float32)}


def _linear(params, x):
    return x @ params["w"] + params["b"]


def _layer_norm(params, x, eps: float = 1e-5):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)
    return (x - mean) * jax.lax.rsqrt(var + eps) * params["scale"] + params["bias"]


def _mish(x):
    return x * jnp.tanh(jax.nn.softplus(x))


def _gelu(x):
    return 0.5 * x * (1.0 + jax.lax.erf(x / jnp.sqrt(2.0)))


def _dropout(rng, x, rate: float, training: bool):
    rate = float(rate)
    if rng is None or (not training) or rate <= 0.0:
        return x
    keep = 1.0 - rate
    mask = jax.random.bernoulli(rng, keep, x.shape)
    return jnp.where(mask, x / keep, 0.0)


def _sinusoidal_pos_emb(timestep, dim: int):
    half = dim // 2
    denom = max(half - 1, 1)
    freqs = jnp.exp(jnp.arange(half, dtype=jnp.float32) * -(jnp.log(10000.0) / denom))
    emb = timestep.astype(jnp.float32)[:, None] * freqs[None]
    emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=-1)
    if dim % 2:
        emb = jnp.pad(emb, ((0, 0), (0, 1)))
    return emb


def _mlp2_init(rng, in_dim: int, hidden_dim: int, out_dim: int, layernorm: bool = False):
    keys = _split(rng, 4)
    params = {
        "l1": _linear_init(keys[0], in_dim, hidden_dim),
        "l2": _linear_init(keys[1], hidden_dim, out_dim),
    }
    if layernorm:
        params["ln"] = _ln_init(hidden_dim)
    return params


def _mlp2_apply(params, x, act=_mish):
    x = _linear(params["l1"], x)
    if "ln" in params:
        x = _layer_norm(params["ln"], x)
    x = act(x)
    return _linear(params["l2"], x)


def _attn_init(rng, hidden_dim: int):
    keys = _split(rng, 4)
    return {
        "q": _mha_in_proj_init(keys[0], hidden_dim),
        "k": _mha_in_proj_init(keys[1], hidden_dim),
        "v": _mha_in_proj_init(keys[2], hidden_dim),
        "o": _linear_init(keys[3], hidden_dim, hidden_dim, zero_bias=True),
    }


def _attention(params, query, key_value, heads: int, rng=None, dropout: float = 0.0, training: bool = False):
    q = _linear(params["q"], query)
    k = _linear(params["k"], key_value)
    v = _linear(params["v"], key_value)
    batch, q_len, hidden = q.shape
    kv_len = k.shape[1]
    head_dim = hidden // heads
    q = q.reshape(batch, q_len, heads, head_dim).transpose(0, 2, 1, 3)
    k = k.reshape(batch, kv_len, heads, head_dim).transpose(0, 2, 1, 3)
    v = v.reshape(batch, kv_len, heads, head_dim).transpose(0, 2, 1, 3)
    logits = jnp.einsum("bhqd,bhkd->bhqk", q, k) / jnp.sqrt(float(head_dim))
    weights = jax.nn.softmax(logits, axis=-1)
    weights = _dropout(rng, weights, dropout, training) if rng is not None else weights
    out = jnp.einsum("bhqk,bhkd->bhqd", weights, v)
    out = out.transpose(0, 2, 1, 3).reshape(batch, q_len, hidden)
    return _linear(params["o"], out)


def _decoder_layer_init(rng, hidden_dim: int, ffn_dim: int):
    keys = _split(rng, 6)
    return {
        "ln_self": _ln_init(hidden_dim),
        "self_attn": _attn_init(keys[0], hidden_dim),
        "ln_cross": _ln_init(hidden_dim),
        "cross_attn": _attn_init(keys[1], hidden_dim),
        "ln_ffn": _ln_init(hidden_dim),
        "ff1": _linear_init(keys[2], hidden_dim, ffn_dim),
        "ff2": _linear_init(keys[3], ffn_dim, hidden_dim),
    }


def _decoder_layer_apply(params, x, memory, heads: int, rng=None, dropout: float = 0.0, training: bool = False):
    rngs = [None] * 6 if rng is None else list(_split(rng, 6))
    dropout = float(dropout)
    self_attn = _attention(params["self_attn"], x, x, heads, rngs[0], dropout, training)
    x = _layer_norm(params["ln_self"], x + _dropout(rngs[1], self_attn, dropout, training))
    cross_attn = _attention(params["cross_attn"], x, memory, heads, rngs[2], dropout, training)
    x = _layer_norm(params["ln_cross"], x + _dropout(rngs[3], cross_attn, dropout, training))
    y = _linear(params["ff1"], x)
    y = _gelu(y)
    y = _dropout(rngs[4], y, dropout, training)
    y = _linear(params["ff2"], y)
    return _layer_norm(params["ln_ffn"], x + _dropout(rngs[5], y, dropout, training))


def freeze_plan_anchor_updates(updates: PyTree) -> PyTree:
    """Keep checkpoint-compatible anchors in params while making them non-trainable."""

    if not isinstance(updates, dict) or "plan_anchor" not in updates:
        return updates
    updates = dict(updates)
    updates["plan_anchor"] = jax.tree_util.tree_map(jnp.zeros_like, updates["plan_anchor"])
    return updates


def diffusion_alpha_cumprod(train_steps: int):
    betas = jnp.linspace(1e-4, 2e-2, int(train_steps), dtype=jnp.float32)
    alphas = 1.0 - betas
    return jnp.cumprod(alphas, axis=0)


def add_noise(sample, noise, timesteps, train_steps: int):
    alpha = diffusion_alpha_cumprod(train_steps)[timesteps]
    alpha = alpha.reshape((alpha.shape[0],) + (1,) * (sample.ndim - 1))
    return jnp.sqrt(alpha) * sample + jnp.sqrt(1.0 - alpha) * noise


def init_planner(rng, config: JAXDiDrConfig) -> PyTree:
    if config.hidden_dim % config.decoder_heads != 0:
        raise ValueError("hidden_dim must be divisible by decoder_heads")
    anchors = np.load(config.plan_anchor_path).astype(np.float32)
    expected = (config.num_modes, config.num_poses, 2)
    if tuple(anchors.shape) != expected:
        raise ValueError(f"Expected anchors {expected}, got {anchors.shape}")
    keys = _split(rng, 12 + int(config.decoder_layers))
    params = {
        "plan_anchor": jnp.asarray(anchors, dtype=jnp.float32),
        "anchor_encoder": _mlp2_init(keys[0], config.num_poses * 2, config.hidden_dim, config.hidden_dim, layernorm=True),
        "time_encoder": _mlp2_init(keys[1], config.hidden_dim, config.hidden_dim * 4, config.hidden_dim),
        "latent_encoder": _mlp2_init(keys[2], config.latent_dim, config.hidden_dim, config.hidden_dim, layernorm=True),
        "decoder_layers": [
            _decoder_layer_init(keys[3 + idx], config.hidden_dim, config.decoder_ffn_dim)
            for idx in range(int(config.decoder_layers))
        ],
        "delta_head": _mlp2_init(keys[8], config.hidden_dim, config.hidden_dim, config.num_poses * 3, layernorm=True),
        "selector_head": _mlp2_init(keys[9], config.hidden_dim, config.hidden_dim, 1, layernorm=True),
    }
    return params


def _decode(params: PyTree, config: JAXDiDrConfig, latent, noisy_xy, timesteps, rng=None, training: bool = False):
    batch, modes, poses, dim = noisy_xy.shape
    if poses != config.num_poses or dim != 2:
        raise ValueError(f"Expected noisy_xy [B,M,{config.num_poses},2], got {noisy_xy.shape}")
    latent = latent.reshape((batch, -1))
    latent_token = _mlp2_apply(params["latent_encoder"], latent)[:, None]
    time_token = _mlp2_apply(params["time_encoder"], _sinusoidal_pos_emb(timesteps, config.hidden_dim))[:, None]
    query = _mlp2_apply(params["anchor_encoder"], noisy_xy.reshape((batch, modes, -1)))
    query = query + latent_token + time_token
    memory = latent_token
    layer_rngs = [None] * len(params["decoder_layers"]) if rng is None else list(_split(rng, len(params["decoder_layers"])))
    for layer, layer_rng in zip(params["decoder_layers"], layer_rngs):
        query = _decoder_layer_apply(
            layer,
            query,
            memory,
            int(config.decoder_heads),
            rng=layer_rng,
            dropout=float(config.dropout),
            training=training,
        )
    delta = _mlp2_apply(params["delta_head"], query).reshape((batch, modes, config.num_poses, 3))
    poses_reg = delta.at[..., :2].add(noisy_xy)
    heading = jnp.tanh(poses_reg[..., 2]) * jnp.pi
    poses_reg = poses_reg.at[..., 2].set(heading)
    poses_cls = _mlp2_apply(params["selector_head"], query).squeeze(-1)
    return poses_reg, poses_cls


def sample_noisy_anchors(params: PyTree, config: JAXDiDrConfig, rng, batch_size: int):
    anchor = jnp.repeat(jax.lax.stop_gradient(params["plan_anchor"])[None], batch_size, axis=0)
    normalized = anchor / float(config.waypoint_scale)
    t_rng, n_rng = _split(rng, 2)
    timesteps = jax.random.randint(t_rng, (batch_size,), 0, int(config.truncated_train_steps), dtype=jnp.int32)
    noise = jax.random.normal(n_rng, normalized.shape, dtype=jnp.float32)
    noisy = jnp.clip(add_noise(normalized, noise, timesteps, int(config.diffusion_train_steps)), -1.0, 1.0)
    return noisy * float(config.waypoint_scale), timesteps


def deterministic_anchors(params: PyTree, config: JAXDiDrConfig, batch_size: int, timestep: int = 0):
    anchor = jnp.repeat(jax.lax.stop_gradient(params["plan_anchor"])[None], batch_size, axis=0)
    timesteps = jnp.full((batch_size,), int(timestep), dtype=jnp.int32)
    normalized = anchor / float(config.waypoint_scale)
    zero_noise = jnp.zeros_like(normalized)
    noisy = add_noise(normalized, zero_noise, timesteps, int(config.diffusion_train_steps))
    return noisy * float(config.waypoint_scale), timesteps


def select_best(poses_reg, poses_cls):
    idx = jnp.argmax(poses_cls, axis=-1)
    return poses_reg[jnp.arange(poses_reg.shape[0]), idx]


def soft_select(poses_reg, poses_cls, temperature: float = 1.0):
    prob = jax.nn.softmax(poses_cls / max(float(temperature), 1e-6), axis=-1)
    return jnp.einsum("bm,bmpd->bpd", prob, poses_reg[..., :2])


def predict(params: PyTree, config: JAXDiDrConfig, latent, timestep: int = 0, soft: bool = False, temperature: float = 1.0):
    noisy_xy, timesteps = deterministic_anchors(params, config, latent.shape[0], timestep=timestep)
    poses_reg, poses_cls = _decode(params, config, latent, noisy_xy, timesteps, training=False)
    traj = soft_select(poses_reg, poses_cls, temperature) if soft else select_best(poses_reg, poses_cls)
    return {"trajectory": traj, "poses_reg": poses_reg, "poses_cls": poses_cls, "timesteps": timesteps}


def _binary_cross_entropy_with_logits(logits, target):
    return jnp.maximum(logits, 0.0) - logits * target + jnp.log1p(jnp.exp(-jnp.abs(logits)))


def _sigmoid_focal_loss(logits, target, gamma: float = 2.0, alpha: float = 0.25):
    pred_sigmoid = jax.nn.sigmoid(logits)
    target = target.astype(logits.dtype)
    pt = (1.0 - pred_sigmoid) * target + pred_sigmoid * (1.0 - target)
    focal_weight = (alpha * target + (1.0 - alpha) * (1.0 - target)) * jnp.power(pt, gamma)
    loss = _binary_cross_entropy_with_logits(logits, target) * focal_weight
    return loss.mean()


def _regularize_latent(config: JAXDiDrConfig, rng, latent, training: bool):
    if not training:
        return latent
    noise_rng, dropout_rng = _split(rng, 2)
    if float(config.latent_noise_std) > 0.0:
        scale = jnp.maximum(jnp.std(latent, axis=-1, keepdims=True), 1e-3)
        noise = jax.random.normal(noise_rng, latent.shape, dtype=latent.dtype)
        latent = latent + noise * scale * float(config.latent_noise_std)
    if float(config.latent_dropout) > 0.0:
        latent = _dropout(dropout_rng, latent, float(config.latent_dropout), training=True)
    return latent


def _planner_losses(params, config: JAXDiDrConfig, rng, latent, target, training: bool = True):
    anchor_rng, latent_rng, decode_rng = _split(rng, 3)
    latent = _regularize_latent(config, latent_rng, latent, training=training)
    noisy_xy, timesteps = sample_noisy_anchors(params, config, anchor_rng, latent.shape[0])
    poses_reg, poses_cls = _decode(params, config, latent, noisy_xy, timesteps, rng=decode_rng, training=training)
    anchor = jnp.repeat(jax.lax.stop_gradient(params["plan_anchor"])[None], target.shape[0], axis=0)
    anchor_dist = jnp.linalg.norm(target[:, None, :, :2] - anchor, axis=-1).mean(axis=-1)
    mode_idx = jnp.argmin(anchor_dist, axis=-1)
    best_reg = poses_reg[jnp.arange(target.shape[0]), mode_idx]

    target_onehot = jax.nn.one_hot(mode_idx, poses_cls.shape[-1], dtype=poses_cls.dtype)
    cls_loss_raw = _sigmoid_focal_loss(poses_cls, target_onehot, gamma=2.0, alpha=0.25)
    reg_loss_raw = jnp.mean(jnp.abs(best_reg - target))
    cls_loss = float(config.cls_loss_weight) * cls_loss_raw
    reg_loss = float(config.reg_loss_weight) * reg_loss_raw
    loss = cls_loss + reg_loss
    ade = jnp.linalg.norm(best_reg[..., :2] - target[..., :2], axis=-1).mean()
    fde = jnp.linalg.norm(best_reg[..., -1, :2] - target[..., -1, :2], axis=-1).mean()
    selected_idx = jnp.argmax(poses_cls, axis=-1)
    metrics = {
        "loss": loss,
        "reg_loss": reg_loss,
        "reg_loss_raw": reg_loss_raw,
        "cls_loss": cls_loss,
        "cls_loss_raw": cls_loss_raw,
        "mode_cls_acc": (selected_idx == mode_idx).astype(jnp.float32).mean(),
        "ade": ade,
        "fde": fde,
    }
    return loss, metrics


def loss_and_metrics(params: PyTree, config: JAXDiDrConfig, rng, batch: Dict[str, jnp.ndarray], training: bool = True):
    key = config.condition_key
    latent = batch[key] if key in batch else batch["condition"]
    target = batch["trajectory"]
    return _planner_losses(params, config, rng, latent.astype(jnp.float32), target.astype(jnp.float32), training=training)


def save_checkpoint(path: str | Path, config: JAXDiDrConfig, params: PyTree, opt_state: Optional[Any] = None, epoch: int = 0):
    payload = {
        "config": config.to_dict(),
        "params": jax.device_get(params),
        "opt_state": jax.device_get(opt_state) if opt_state is not None else None,
        "epoch": int(epoch),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb") as f:
        pickle.dump(payload, f)


def load_checkpoint(path: str | Path):
    with gzip.open(path, "rb") as f:
        payload = pickle.load(f)
    return JAXDiDrConfig.from_dict(payload["config"]), payload["params"], payload.get("opt_state"), int(payload.get("epoch", 0))
