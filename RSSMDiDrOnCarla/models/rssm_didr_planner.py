"""RSSM latent conditioned DiffusionDrive-style trajectory planner."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class RSSMDiDrConfig:
    latent_dim: int
    plan_anchor_path: str
    condition_type: str = "rssm_latent"
    condition_key: str = "rssm_latent"
    gt_history_length: int = 10
    gt_history_include_neighbors: bool = True
    gt_history_align_neighbor_ids: bool = True
    hidden_dim: int = 256
    num_modes: int = 20
    num_poses: int = 8
    waypoint_scale: float = 30.0
    diffusion_train_steps: int = 1000
    truncated_train_steps: int = 50
    truncated_eval_step: int = 8
    eval_refine_steps: int = 2
    decoder_layers: int = 2
    decoder_heads: int = 4
    decoder_ffn_dim: int = 512
    dropout: float = 0.1
    reg_loss_weight: float = 1.0
    cls_loss_weight: float = 0.5
    latent_noise_std: float = 0.0
    latent_dropout: float = 0.0

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        scale = np.log(10000.0) / max(half_dim - 1, 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device, dtype=torch.float32) * -scale)
        emb = x.float()[:, None] * emb[None]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        if emb.shape[-1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[-1]))
        return emb


class TruncatedDiffusion:
    """Small scheduler with the add-noise behavior needed by this planner."""

    def __init__(self, train_steps: int = 1000, beta_start: float = 1e-4, beta_end: float = 2e-2):
        betas = torch.linspace(beta_start, beta_end, train_steps, dtype=torch.float32)
        alphas = 1.0 - betas
        self.alpha_cumprod = torch.cumprod(alphas, dim=0)

    def add_noise(self, sample: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        alpha = self.alpha_cumprod.to(sample.device)[timesteps].view(-1, *([1] * (sample.ndim - 1)))
        return alpha.sqrt() * sample + (1.0 - alpha).sqrt() * noise


class RSSMTrajectoryDecoder(nn.Module):
    """Trajectory decoder and selector conditioned by one RSSM latent token."""

    def __init__(self, config: RSSMDiDrConfig):
        super().__init__()
        self.config = config
        anchor_dim = config.num_poses * 2

        self.anchor_encoder = nn.Sequential(
            nn.Linear(anchor_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.Mish(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )
        self.time_encoder = nn.Sequential(
            SinusoidalPosEmb(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.hidden_dim * 4),
            nn.Mish(),
            nn.Linear(config.hidden_dim * 4, config.hidden_dim),
        )
        self.latent_encoder = nn.Sequential(
            nn.Linear(config.latent_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.Mish(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )

        layer = nn.TransformerDecoderLayer(
            d_model=config.hidden_dim,
            nhead=config.decoder_heads,
            dim_feedforward=config.decoder_ffn_dim,
            dropout=config.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=config.decoder_layers)
        self.delta_head = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.Mish(),
            nn.Linear(config.hidden_dim, config.num_poses * 3),
        )
        self.selector_head = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.Mish(),
            nn.Linear(config.hidden_dim, 1),
        )

    def decode_features(
        self,
        rssm_latent: torch.Tensor,
        noisy_xy: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        batch, modes, poses, dim = noisy_xy.shape
        if poses != self.config.num_poses or dim != 2:
            raise ValueError(f"Expected noisy_xy [B, M, {self.config.num_poses}, 2], got {noisy_xy.shape}")

        latent = rssm_latent.reshape(batch, -1)
        latent_token = self.latent_encoder(latent).unsqueeze(1)
        time_token = self.time_encoder(timesteps).unsqueeze(1)

        query = self.anchor_encoder(noisy_xy.reshape(batch, modes, -1))
        query = query + latent_token + time_token
        return self.decoder(query, latent_token)

    def forward(
        self,
        rssm_latent: torch.Tensor,
        noisy_xy: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, modes, poses, dim = noisy_xy.shape
        decoded = self.decode_features(rssm_latent, noisy_xy, timesteps)
        delta = self.delta_head(decoded).reshape(batch, modes, self.config.num_poses, 3)
        poses_reg = delta.clone()
        poses_reg[..., :2] = poses_reg[..., :2] + noisy_xy
        poses_reg[..., 2] = torch.tanh(poses_reg[..., 2]) * np.pi
        poses_cls = self.selector_head(decoded).squeeze(-1)
        return poses_reg, poses_cls


class RSSMDiffusionDrivePlanner(nn.Module):
    """Anchor + truncated diffusion + decoder + selector, conditioned on RSSM latent."""

    def __init__(self, config: RSSMDiDrConfig):
        super().__init__()
        self.config = config
        anchors = np.load(config.plan_anchor_path).astype(np.float32)
        if anchors.shape != (config.num_modes, config.num_poses, 2):
            raise ValueError(
                f"Expected anchors [{config.num_modes}, {config.num_poses}, 2], got {anchors.shape}"
            )
        self.register_buffer("plan_anchor", torch.from_numpy(anchors), persistent=True)
        self.scheduler = TruncatedDiffusion(config.diffusion_train_steps)
        self.decoder = RSSMTrajectoryDecoder(config)

    def _normalize_xy(self, xy: torch.Tensor) -> torch.Tensor:
        return xy / float(self.config.waypoint_scale)

    def _denormalize_xy(self, xy: torch.Tensor) -> torch.Tensor:
        return xy * float(self.config.waypoint_scale)

    def _sample_noisy_anchors(
        self,
        batch_size: int,
        device: torch.device,
        timesteps: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        anchor = self.plan_anchor.to(device).unsqueeze(0).repeat(batch_size, 1, 1, 1)
        normalized = self._normalize_xy(anchor)
        if timesteps is None:
            timesteps = torch.randint(
                0,
                self.config.truncated_train_steps,
                (batch_size,),
                device=device,
                dtype=torch.long,
            )
        noise = torch.randn_like(normalized)
        noisy = self.scheduler.add_noise(normalized, noise, timesteps).clamp(-1.0, 1.0)
        return self._denormalize_xy(noisy), timesteps

    def forward(self, features: Dict[str, torch.Tensor], targets: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        key = self.config.condition_key
        if key in features:
            latent = features[key].float()
        elif "condition" in features:
            latent = features["condition"].float()
        else:
            raise KeyError(f"Missing planner condition feature `{key}`")
        if self.training:
            if self.config.latent_noise_std > 0.0:
                # print('add   std!')
                scale = latent.detach().std(dim=-1, keepdim=True).clamp_min(1e-3)
                latent = latent + torch.randn_like(latent) * scale * float(self.config.latent_noise_std)
            if self.config.latent_dropout > 0.0:
                latent = F.dropout(latent, p=float(self.config.latent_dropout), training=True)
        batch_size = latent.shape[0]
        noisy_xy, timesteps = self._sample_noisy_anchors(batch_size, latent.device)
        poses_reg, poses_cls = self.decoder(latent, noisy_xy, timesteps)
        best = self.select_best(poses_reg, poses_cls)
        output = {
            "trajectory": best,
            "poses_reg": poses_reg,
            "poses_cls": poses_cls,
            "timesteps": timesteps,
        }
        if targets is not None:
            output.update(self.compute_loss(output, targets))
        return output

    def select_best(self, poses_reg: torch.Tensor, poses_cls: torch.Tensor) -> torch.Tensor:
        mode_idx = poses_cls.argmax(dim=-1)
        gather_idx = mode_idx[:, None, None, None].repeat(1, 1, self.config.num_poses, 3)
        return torch.gather(poses_reg, 1, gather_idx).squeeze(1)

    def compute_loss(self, predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        poses_reg = predictions["poses_reg"]
        poses_cls = predictions["poses_cls"]
        target_traj = targets["trajectory"].float()
        plan_anchor = self.plan_anchor.to(target_traj.device).unsqueeze(0).repeat(target_traj.shape[0], 1, 1, 1)

        dist = torch.linalg.norm(target_traj.unsqueeze(1)[..., :2] - plan_anchor, dim=-1).mean(dim=-1)
        mode_idx = torch.argmin(dist, dim=-1)
        gather_idx = mode_idx[:, None, None, None].repeat(1, 1, self.config.num_poses, 3)
        best_reg = torch.gather(poses_reg, 1, gather_idx).squeeze(1)

        reg_loss = F.l1_loss(best_reg, target_traj)
        cls_loss = F.cross_entropy(poses_cls, mode_idx)
        loss = self.config.reg_loss_weight * reg_loss + self.config.cls_loss_weight * cls_loss
        ade = torch.linalg.norm(best_reg[..., :2] - target_traj[..., :2], dim=-1).mean()
        fde = torch.linalg.norm(best_reg[..., -1, :2] - target_traj[..., -1, :2], dim=-1).mean()

        return {
            "loss": loss,
            "reg_loss": reg_loss.detach(),
            "cls_loss": cls_loss.detach(),
            "ade": ade.detach(),
            "fde": fde.detach(),
        }

    def save_checkpoint(self, path: str | Path, optimizer: Optional[torch.optim.Optimizer] = None, epoch: int = 0) -> None:
        payload = {
            "config": self.config.to_dict(),
            "model": self.state_dict(),
            "epoch": int(epoch),
        }
        if optimizer is not None:
            payload["optimizer"] = optimizer.state_dict()
        torch.save(payload, path)
