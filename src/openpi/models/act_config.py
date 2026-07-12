"""Config for the Action Chunking Transformer (ACT) model.

ACT (Zhao et al., 2023, "Learning Fine-Grained Bimanual Manipulation with Low-Cost
Hardware") is a CVAE-style policy that maps a few camera views + the proprioceptive
state to a *chunk* of future actions.

Unlike pi0 / pi05 it has **no language / VLM component**, so:
- there is no tokenized prompt (the model transforms must not tokenize for ACT), and
- it is usually trained *per task* (or with an explicit task-id conditioning), because
  it cannot disambiguate tasks from a language instruction.

It also trains **from scratch** -- there is no pretrained checkpoint to load, so configs
should use a no-op weight loader.

This lives alongside Pi0Config as a peer `ModelType`, so switching algorithm is just a
matter of selecting a different training config name.
"""

import dataclasses
from typing import TYPE_CHECKING

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.shared import array_typing as at

if TYPE_CHECKING:
    from openpi.models.act import ACT


@dataclasses.dataclass(frozen=True)
class ACTConfig(_model.BaseModelConfig):
    dtype: str = "float32"

    # Shared BaseModelConfig fields (redefined with ACT defaults).
    action_dim: int = 56
    action_horizon: int = 100  # = action chunk size
    # ACT has no language input; max_token_len is unused but kept for interface parity.
    max_token_len: int = 1

    # Transformer hyperparameters.
    hidden_dim: int = 512
    num_heads: int = 8
    dim_feedforward: int = 3200
    num_encoder_layers: int = 4
    num_decoder_layers: int = 7
    # Number of layers in the CVAE "style" encoder (over the action chunk).
    num_cvae_layers: int = 4
    dropout: float = 0.1

    # CVAE latent.
    latent_dim: int = 32
    kl_weight: float = 10.0

    # Vision backbone. Scaffold ships a from-scratch ResNet-18-style trunk implemented in
    # NNX. Production may want to swap in an ImageNet-pretrained ResNet-18.
    vision_backbone: str = "resnet18"

    @property
    @override
    def model_type(self) -> _model.ModelType:
        return _model.ModelType.ACT

    @override
    def create(self, rng: at.KeyArrayLike) -> "ACT":
        from openpi.models.act import ACT

        return ACT(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            # Note: no tokenized_prompt -- ACT has no language input.
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """ACT trains end-to-end from scratch; nothing is frozen."""
        return nnx.Nothing
