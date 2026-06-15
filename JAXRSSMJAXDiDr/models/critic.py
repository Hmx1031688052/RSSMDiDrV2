"""JAX value critic and Dreamer lambda-return utilities."""

from __future__ import annotations

from typing import Any, Dict

import jax
import jax.numpy as jnp

from .jax_didr_planner import _linear, _linear_init, _ln_init, _mish, _split, _layer_norm


def symlog(x):
    return jnp.sign(x) * jnp.log(jnp.abs(x) + 1.0)


def symexp(x):
    return jnp.sign(x) * (jnp.exp(jnp.abs(x)) - 1.0)


def init_critic(rng, feat_dim: int, hidden_dim: int = 512, layers: int = 2) -> Dict[str, Any]:
    keys = _split(rng, int(layers) + 2)
    params = {"layers": [], "norms": []}
    dim = int(feat_dim)
    for idx in range(int(layers)):
        params["layers"].append(_linear_init(keys[idx], dim, int(hidden_dim)))
        params["norms"].append(_ln_init(int(hidden_dim)))
        dim = int(hidden_dim)
    params["out"] = _linear_init(keys[-1], dim, 1, scale=0.0)
    return params


def critic_logits(params, feat):
    x = feat
    for layer, norm in zip(params["layers"], params["norms"]):
        x = _mish(_layer_norm(norm, _linear(layer, x)))
    return _linear(params["out"], x).squeeze(-1)


def critic_value(params, feat):
    return symexp(critic_logits(params, feat))


def critic_loss(params, feat, target):
    pred = critic_logits(params, feat)
    return jnp.mean(jnp.square(pred - symlog(jax.lax.stop_gradient(target))))


def lambda_return(reward, value, discount, bootstrap, lambda_=0.95):
    next_values = jnp.concatenate([value[1:], bootstrap[None]], axis=0)
    inputs = reward + discount * next_values * (1.0 - float(lambda_))

    def step(last, inp):
        out = inp[0] + inp[1] * float(lambda_) * last
        return out, out

    _, returns_rev = jax.lax.scan(step, bootstrap, (inputs[::-1], discount[::-1]))
    return returns_rev[::-1]

