"""JAX/NNX adaptation of real-stanford/diffusion_policy's image U-Net policy.

See THIRD_PARTY_NOTICES.md for provenance. The temporal network follows
ConditionalUnet1D with global observation conditioning and epsilon prediction.
BrainCo supplies single-frame observations, normalization, augmentation, EMA
and absolute-action reconstruction through its existing training/policy stack.
"""

import itertools
import math

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
from typing_extensions import override

from openpi.models import diffusion_config
from openpi.models import model as _model
from openpi.models.resnet import ResNet18


def diffusion_betas(config: diffusion_config.DiffusionPolicyConfig) -> np.ndarray:
    """Diffusers squaredcos_cap_v2 schedule, or the configured linear schedule."""
    if config.beta_schedule == "linear":
        return np.linspace(config.beta_start, config.beta_end, config.num_train_timesteps, dtype=np.float32)

    def alpha_bar(t):
        return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2

    count = config.num_train_timesteps
    return np.array(
        [min(1 - alpha_bar((i + 1) / count) / alpha_bar(i / count), 0.999) for i in range(count)],
        dtype=np.float32,
    )


def denoise_step(sample, epsilon, alpha, previous_alpha, noise, *, sampler: str, clip_sample: bool):
    """One DDPM fixed-small posterior step or deterministic DDIM (eta=0).

    alpha and previous_alpha are cumulative alpha products. At the final step
    previous_alpha=1, so no noise is added and the predicted clean sample exits.
    """
    beta = 1 - alpha
    clean = (sample - jnp.sqrt(beta) * epsilon) / jnp.sqrt(alpha)
    if clip_sample:
        clean = jnp.clip(clean, -1.0, 1.0)
    if sampler == "ddim":
        return jnp.sqrt(previous_alpha) * clean + jnp.sqrt(1 - previous_alpha) * epsilon
    current_alpha = alpha / previous_alpha
    current_beta = 1 - current_alpha
    mean = (jnp.sqrt(previous_alpha) * current_beta / beta) * clean
    mean += (jnp.sqrt(current_alpha) * (1 - previous_alpha) / beta) * sample
    variance = jnp.maximum((1 - previous_alpha) / beta * current_beta, 0.0)
    return mean + jnp.sqrt(variance) * noise


def _mish(x):
    return x * jnp.tanh(jax.nn.softplus(x))


class Conv1dBlock(nnx.Module):
    def __init__(self, in_dim, out_dim, config, *, rngs):
        self.conv = nnx.Conv(in_dim, out_dim, (config.kernel_size,), padding="SAME", rngs=rngs)
        self.norm = nnx.GroupNorm(out_dim, num_groups=config.n_groups, epsilon=1e-5, rngs=rngs)

    def __call__(self, x):
        return _mish(self.norm(self.conv(x)))


class ConditionalResidualBlock1D(nnx.Module):
    def __init__(self, in_dim, out_dim, cond_dim, config, *, rngs):
        self.block1 = Conv1dBlock(in_dim, out_dim, config, rngs=rngs)
        self.block2 = Conv1dBlock(out_dim, out_dim, config, rngs=rngs)
        self.predict_scale = config.cond_predict_scale
        self.condition = nnx.Linear(cond_dim, out_dim * (2 if self.predict_scale else 1), rngs=rngs)
        self.residual = nnx.Conv(in_dim, out_dim, (1,), rngs=rngs) if in_dim != out_dim else None

    def __call__(self, x, condition):
        out = self.block1(x)
        embedding = self.condition(_mish(condition))[:, None, :]
        if self.predict_scale:
            scale, bias = jnp.split(embedding, 2, axis=-1)
            out = scale * out + bias
        else:
            out = out + embedding
        return self.block2(out) + (self.residual(x) if self.residual is not None else x)


class ConditionalUnet1D(nnx.Module):
    """Temporal convolutions use [batch, horizon, channels] throughout."""

    def __init__(self, config, global_cond_dim, *, rngs):
        self.embed_dim = config.diffusion_step_embed_dim
        self.time_in = nnx.Linear(self.embed_dim, self.embed_dim * 4, rngs=rngs)
        self.time_out = nnx.Linear(self.embed_dim * 4, self.embed_dim, rngs=rngs)
        cond_dim = self.embed_dim + global_cond_dim
        dims = (config.action_dim, *config.down_dims)
        pairs = list(itertools.pairwise(dims))

        def block(in_dim, out_dim):
            return ConditionalResidualBlock1D(in_dim, out_dim, cond_dim, config, rngs=rngs)

        self.down = [
            (
                block(in_dim, out_dim),
                block(out_dim, out_dim),
                # Explicit padding matches PyTorch Conv1d(k=3,s=2,p=1).
                nnx.Conv(out_dim, out_dim, (3,), strides=2, padding=((1, 1),), rngs=rngs)
                if i < len(pairs) - 1
                else None,
            )
            for i, (in_dim, out_dim) in enumerate(pairs)
        ]
        self.mid = [block(dims[-1], dims[-1]), block(dims[-1], dims[-1])]
        self.up = [
            (
                block(out_dim * 2, in_dim),
                block(in_dim, in_dim),
                nnx.ConvTranspose(in_dim, in_dim, (4,), strides=2, padding="SAME", rngs=rngs),
            )
            for in_dim, out_dim in reversed(pairs[1:])
        ]
        self.final_block = Conv1dBlock(dims[1], dims[1], config, rngs=rngs)
        self.final = nnx.Conv(dims[1], config.action_dim, (1,), rngs=rngs)

    def __call__(self, sample, timestep, global_condition):
        timestep = jnp.broadcast_to(jnp.asarray(timestep), (sample.shape[0],))
        half = self.embed_dim // 2
        frequencies = jnp.exp(-math.log(10000) * jnp.arange(half, dtype=jnp.float32) / (half - 1))
        phase = timestep[:, None] * frequencies[None, :]
        embedding = jnp.concatenate([jnp.sin(phase), jnp.cos(phase)], axis=-1)
        embedding = self.time_out(_mish(self.time_in(embedding)))
        condition = jnp.concatenate([embedding, global_condition], axis=-1)
        x = sample
        skips = []
        for first, second, downsample in self.down:
            x = second(first(x, condition), condition)
            skips.append(x)
            if downsample is not None:
                x = downsample(x)
        for block in self.mid:
            x = block(x, condition)
        for first, second, upsample in self.up:
            x = jnp.concatenate([x, skips.pop()], axis=-1)
            x = upsample(second(first(x, condition), condition))
        return self.final(self.final_block(x))


class DiffusionPolicy(_model.BaseModel):
    def __init__(self, config: diffusion_config.DiffusionPolicyConfig, *, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.config = config
        self.encoders = {
            key: ResNet18(config.vision_feature_dim, width=config.vision_width, rngs=rngs) for key in _model.IMAGE_KEYS
        }
        self.unet = ConditionalUnet1D(
            config, config.action_dim + len(_model.IMAGE_KEYS) * config.vision_feature_dim, rngs=rngs
        )
        # Derived from config, not trainable parameters or EMA state.
        self._alpha_cumprod = tuple(float(x) for x in np.cumprod(1 - diffusion_betas(config)))

    def _encode_observation(self, observation):
        features = [observation.state]
        for key in _model.IMAGE_KEYS:
            image_features = jnp.mean(self.encoders[key](observation.images[key]), axis=(1, 2))
            features.append(image_features * observation.image_masks[key][:, None])
        return jnp.concatenate(features, axis=-1)

    def _preprocess(self, rng, observation, *, train=False):
        return _model.preprocess_observation(
            rng, observation, train=train, image_resolution=(self.config.image_size, self.config.image_size)
        )

    @override
    def compute_loss(self, rng, observation, actions, *, train=False):
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = self._preprocess(preprocess_rng, observation, train=train)
        condition = self._encode_observation(observation)
        noise = jax.random.normal(noise_rng, actions.shape, dtype=jnp.float32)
        timesteps = jax.random.randint(time_rng, (actions.shape[0],), 0, self.config.num_train_timesteps)
        alpha = jnp.asarray(self._alpha_cumprod)[timesteps, None, None]
        noisy_actions = jnp.sqrt(alpha) * actions + jnp.sqrt(1 - alpha) * noise
        prediction = self.unet(noisy_actions, timesteps, condition)
        return jnp.mean(jnp.square(prediction - noise), axis=-1)

    @override
    def sample_actions(self, rng, observation, *, noise=None):
        observation = self._preprocess(None, observation)
        condition = self._encode_observation(observation)
        rng, initial_rng = jax.random.split(rng)
        shape = (observation.state.shape[0], self.action_horizon, self.action_dim)
        sample = jax.random.normal(initial_rng, shape) if noise is None else jnp.asarray(noise, dtype=jnp.float32)
        if sample.shape != shape:
            raise ValueError(f"diffusion noise must have shape {shape}, got {sample.shape}")
        # Match Diffusers' default 'leading' timestep spacing (including when
        # num_train_timesteps is not divisible by num_inference_steps).
        ratio = self.config.num_train_timesteps // self.config.num_inference_steps
        timesteps = jnp.arange(self.config.num_inference_steps - 1, -1, -1) * ratio
        alphas = jnp.asarray(self._alpha_cumprod)

        def step(carry, timestep):
            trajectory, key = carry
            key, noise_key = jax.random.split(key)
            epsilon = self.unet(trajectory, timestep, condition)
            previous = timestep - ratio
            previous_alpha = jnp.where(previous >= 0, alphas[jnp.maximum(previous, 0)], 1.0)
            trajectory = denoise_step(
                trajectory,
                epsilon,
                alphas[timestep],
                previous_alpha,
                jax.random.normal(noise_key, shape),
                sampler=self.config.sampler,
                clip_sample=self.config.clip_sample,
            )
            return (trajectory, key), None

        (sample, _), _ = jax.lax.scan(step, (sample, rng), timesteps)
        return sample.astype(jnp.float32)
