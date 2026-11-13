"""Diffusion Policy (CNN + conditional temporal U-Net) configuration.

The first BrainCo variant uses one observation and predicts the entire future
chunk. It shares the LeRobot transforms and JAX trainer with PI0.5 and ACT.
"""

import dataclasses
from typing import TYPE_CHECKING, Literal

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.shared import array_typing as at

if TYPE_CHECKING:
    from openpi.models.diffusion import DiffusionPolicy


@dataclasses.dataclass(frozen=True)
class DiffusionPolicyConfig(_model.BaseModelConfig):
    dtype: str = "float32"
    action_dim: int = 56
    action_horizon: int = 32
    max_token_len: int = 1
    n_obs_steps: int = 1

    # ResNet-18 + GroupNorm, initialized from scratch, one encoder per camera.
    image_size: int = 224
    vision_width: int = 64
    vision_feature_dim: int = 128
    diffusion_step_embed_dim: int = 128
    down_dims: tuple[int, ...] = (256, 512, 1024)
    kernel_size: int = 5
    n_groups: int = 8
    cond_predict_scale: bool = True

    num_train_timesteps: int = 100
    num_inference_steps: int = 100
    sampler: Literal["ddpm", "ddim"] = "ddpm"
    beta_schedule: Literal["squaredcos_cap_v2", "linear"] = "squaredcos_cap_v2"
    beta_start: float = 0.0001
    beta_end: float = 0.02
    clip_sample: bool = True

    def __post_init__(self):
        # Normalize YAML lists so the NNX graph definition remains hashable.
        object.__setattr__(self, "down_dims", tuple(self.down_dims))
        if self.dtype != "float32":
            raise ValueError("Diffusion Policy currently requires dtype=float32")
        if self.n_obs_steps != 1:
            raise ValueError(
                "Diffusion Policy currently supports n_obs_steps=1; observation history is not implemented"
            )
        if self.action_dim <= 0 or self.action_horizon <= 0:
            raise ValueError("action_dim and action_horizon must be positive")
        if len(self.down_dims) < 2 or self.n_groups <= 0:
            raise ValueError("down_dims needs at least two levels and n_groups must be positive")
        if any(dim <= 0 or dim % self.n_groups for dim in self.down_dims):
            raise ValueError("every down_dims entry must be positive and divisible by n_groups")
        divisor = 2 ** (len(self.down_dims) - 1)
        if self.action_horizon % divisor:
            raise ValueError(f"action_horizon must be divisible by {divisor} for this U-Net")
        if self.diffusion_step_embed_dim < 4 or self.diffusion_step_embed_dim % 2:
            raise ValueError("diffusion_step_embed_dim must be even and >= 4")
        if self.kernel_size <= 0 or self.kernel_size % 2 == 0:
            raise ValueError("kernel_size must be positive and odd")
        if self.image_size < 32 or self.vision_width <= 0 or self.vision_feature_dim <= 0:
            raise ValueError("image_size must be >= 32 and vision dimensions must be positive")
        if not 1 <= self.num_inference_steps <= self.num_train_timesteps:
            raise ValueError("require 1 <= num_inference_steps <= num_train_timesteps")
        if self.sampler not in ("ddpm", "ddim"):
            raise ValueError(f"unsupported sampler: {self.sampler}")
        if self.beta_schedule not in ("squaredcos_cap_v2", "linear"):
            raise ValueError(f"unsupported beta_schedule: {self.beta_schedule}")
        if not 0 < self.beta_start < self.beta_end < 1:
            raise ValueError("require 0 < beta_start < beta_end < 1")

    @property
    @override
    def model_type(self) -> _model.ModelType:
        return _model.ModelType.DIFFUSION

    @override
    def create(self, rng: at.KeyArrayLike) -> "DiffusionPolicy":
        from openpi.models.diffusion import DiffusionPolicy  # noqa: PLC0415

        return DiffusionPolicy(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        with at.disable_typechecking():
            observation = _model.Observation(
                images={
                    key: jax.ShapeDtypeStruct((batch_size, self.image_size, self.image_size, 3), jnp.float32)
                    for key in _model.IMAGE_KEYS
                },
                image_masks={key: jax.ShapeDtypeStruct((batch_size,), jnp.bool_) for key in _model.IMAGE_KEYS},
                state=jax.ShapeDtypeStruct((batch_size, self.action_dim), jnp.float32),
            )
        actions = jax.ShapeDtypeStruct((batch_size, self.action_horizon, self.action_dim), jnp.float32)
        return observation, actions

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        return nnx.Nothing
