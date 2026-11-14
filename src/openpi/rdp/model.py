"""Two-stage RDP: action-only variational encoder, causal tactile GRU, latent DP.

Architecture follows Xue et al. RDP (2025). BrainCo adaptation uses joint
actions, five independent PCA sensors and an explicit causal hold-age input.
"""

from dataclasses import dataclass
import math

from diffusers import DDIMScheduler
from diffusers import DDPMScheduler
import torch
from torch import nn
from torchvision.models import resnet18

from openpi.rdp.vendor.conditional_unet1d import ConditionalUnet1D


@dataclass(frozen=True)
class RDPConfig:
    action_dim: int = 56
    horizon: int = 16
    tactile_dim: int = 76
    latent_dim: int = 16
    hidden_dim: int = 256
    image_size: int = 128
    vision_dim: int = 128
    down_dims: tuple = (128, 256, 512)
    diffusion_steps: int = 100
    inference_steps: int = 10
    kl_weight: float = 1e-6

    def __post_init__(self):
        object.__setattr__(self, "down_dims", tuple(self.down_dims))
        if self.horizon % 4 or (self.horizon // 4) % (2 ** (len(self.down_dims) - 1)):
            raise ValueError("horizon must support AT /4 and latent U-Net downsampling")
        if min(self.action_dim, self.tactile_dim, self.latent_dim, self.hidden_dim) <= 0:
            raise ValueError("dimensions must be positive")
        if not 1 <= self.inference_steps <= self.diffusion_steps:
            raise ValueError("invalid diffusion inference steps")


class AsymmetricTokenizer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder = nn.Sequential(
            nn.Conv1d(config.action_dim, 128, 5, 2, 2), nn.ReLU(), nn.Conv1d(128, config.latent_dim * 2, 5, 2, 2)
        )
        self.decoder = nn.GRU(
            config.latent_dim * (config.horizon // 4) + config.tactile_dim, config.hidden_dim, batch_first=True
        )
        self.output = nn.Linear(config.hidden_dim, config.action_dim)

    def encode(self, actions, *, sample=False):
        mean, logvar = self.encoder(actions.transpose(1, 2)).transpose(1, 2).chunk(2, -1)
        logvar = logvar.clamp(-20, 10)
        latent = mean + torch.exp(0.5 * logvar) * torch.randn_like(mean) if sample else mean
        kl = 0.5 * (mean.square() + logvar.exp() - 1 - logvar).mean()
        return latent, kl

    def decode(self, latent, tactile, hidden=None):
        # A global latent describes the entire chunk; feedback remains causal.
        z = latent.flatten(1).unsqueeze(1).expand(-1, tactile.shape[1], -1)
        out, hidden = self.decoder(torch.cat((z, tactile), dim=-1), hidden)
        return self.output(out), hidden

    def loss(self, actions, tactile):
        latent, kl = self.encode(actions, sample=True)
        prediction, _ = self.decode(latent, tactile)
        reconstruction = (prediction - actions).abs().mean()
        return reconstruction + self.config.kl_weight * kl, {"reconstruction": reconstruction, "kl": kl}


def group_norm(module):
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            setattr(module, name, nn.GroupNorm(math.gcd(8, child.num_features), child.num_features))
        else:
            group_norm(child)
    return module


class LatentDiffusionPolicy(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoders = nn.ModuleList(
            [group_norm(resnet18(weights=None, num_classes=config.vision_dim)) for _ in range(3)]
        )
        self.unet = ConditionalUnet1D(
            config.latent_dim,
            global_cond_dim=config.action_dim + config.tactile_dim + 3 * config.vision_dim,
            diffusion_step_embed_dim=128,
            down_dims=list(config.down_dims),
            kernel_size=5,
            n_groups=8,
            cond_predict_scale=True,
        )
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=config.diffusion_steps,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=False,
            prediction_type="epsilon",
        )
        self.register_buffer("latent_mean", torch.zeros(config.latent_dim))
        self.register_buffer("latent_std", torch.ones(config.latent_dim))

    def condition(self, images, state, tactile):
        # images: [B,3 cameras,H,W,3], uint8; only current-frame inputs.
        x = images.permute(0, 1, 4, 2, 3).float() / 255.0
        x = (x - 0.5) / 0.5
        return torch.cat([state, tactile, *(net(x[:, i]) for i, net in enumerate(self.encoders))], dim=-1)

    def loss(self, latent, images, state, tactile):
        z = (latent - self.latent_mean) / self.latent_std
        noise = torch.randn_like(z)
        t = torch.randint(self.config.diffusion_steps, (len(z),), device=z.device)
        noisy = self.noise_scheduler.add_noise(z, noise, t)
        predicted = self.unet(noisy, t, global_cond=self.condition(images, state, tactile))
        return (predicted - noise).square().mean()

    @torch.inference_mode()
    def sample(self, images, state, tactile):
        scheduler = DDIMScheduler.from_config(self.noise_scheduler.config)
        scheduler.set_timesteps(self.config.inference_steps, device=state.device)
        z = torch.randn((len(state), self.config.horizon // 4, self.config.latent_dim), device=state.device)
        condition = self.condition(images, state, tactile)
        for t in scheduler.timesteps:
            noise = self.unet(z, t, global_cond=condition)
            z = scheduler.step(noise, t, z, eta=0.0).prev_sample
        return z * self.latent_std + self.latent_mean
