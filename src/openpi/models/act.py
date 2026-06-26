"""Action Chunking Transformer (ACT), implemented natively in JAX/Flax NNX.

ACT is a conditional VAE (CVAE):

  Training:
    - A "style" encoder (BERT-like transformer) reads the proprioceptive state + the
      ground-truth action chunk and produces a latent z (reparameterized Gaussian).
    - A DETR-style encoder-decoder reads [z, state, image features] as memory and decodes
      a chunk of `action_horizon` actions from learned query embeddings.
    - Loss = L1(reconstruction) + kl_weight * KL(z || N(0, I)).

  Inference:
    - z is set to the prior mean (zeros); the style encoder is unused.

This implements the `BaseModel` interface so it slots into the same training loop, norm
stats, serving and checkpointing as pi0 / pi05. It has no language component.
"""

import flax.linen as nn
import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import act_config as _act_config
from openpi.models import model as _model
from openpi.shared import array_typing as at


def _num_groups(channels: int, max_groups: int = 32) -> int:
    """Largest divisor of `channels` that is <= max_groups (for GroupNorm)."""
    g = min(max_groups, channels)
    while channels % g != 0:
        g -= 1
    return g


class _BasicBlock(nnx.Module):
    """ResNet basic residual block (two 3x3 convs) with GroupNorm."""

    def __init__(self, in_ch: int, out_ch: int, stride: int, *, rngs: nnx.Rngs):
        self.conv1 = nnx.Conv(in_ch, out_ch, (3, 3), strides=stride, padding="SAME", use_bias=False, rngs=rngs)
        self.norm1 = nnx.GroupNorm(out_ch, num_groups=_num_groups(out_ch), rngs=rngs)
        self.conv2 = nnx.Conv(out_ch, out_ch, (3, 3), strides=1, padding="SAME", use_bias=False, rngs=rngs)
        self.norm2 = nnx.GroupNorm(out_ch, num_groups=_num_groups(out_ch), rngs=rngs)

        self.has_downsample = stride != 1 or in_ch != out_ch
        if self.has_downsample:
            self.down_conv = nnx.Conv(in_ch, out_ch, (1, 1), strides=stride, padding="SAME", use_bias=False, rngs=rngs)
            self.down_norm = nnx.GroupNorm(out_ch, num_groups=_num_groups(out_ch), rngs=rngs)

    def __call__(self, x):
        identity = x
        out = nnx.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        if self.has_downsample:
            identity = self.down_norm(self.down_conv(x))
        return nnx.relu(out + identity)


class _ResNet18(nnx.Module):
    """ResNet-18 trunk (from scratch) projecting to `out_dim` channels.

    Input:  [b, H, W, 3] in [-1, 1].
    Output: [b, H/32, W/32, out_dim] feature map.
    """

    def __init__(self, out_dim: int, *, rngs: nnx.Rngs):
        self.conv1 = nnx.Conv(3, 64, (7, 7), strides=2, padding="SAME", use_bias=False, rngs=rngs)
        self.norm1 = nnx.GroupNorm(64, num_groups=_num_groups(64), rngs=rngs)

        chans = [64, 128, 256, 512]
        blocks_per_stage = [2, 2, 2, 2]
        strides = [1, 2, 2, 2]
        self.stages: list[list[_BasicBlock]] = []
        in_ch = 64
        for out_ch, n_blocks, stage_stride in zip(chans, blocks_per_stage, strides, strict=True):
            stage = []
            for i in range(n_blocks):
                stage.append(_BasicBlock(in_ch, out_ch, stride=stage_stride if i == 0 else 1, rngs=rngs))
                in_ch = out_ch
            self.stages.append(stage)

        self.proj = nnx.Conv(512, out_dim, (1, 1), rngs=rngs)

    def __call__(self, x):
        x = nnx.relu(self.norm1(self.conv1(x)))
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding="SAME")
        for stage in self.stages:
            for block in stage:
                x = block(x)
        return self.proj(x)


class _EncoderLayer(nnx.Module):
    """Pre-norm transformer self-attention encoder layer."""

    def __init__(self, dim: int, num_heads: int, ff: int, dropout: float, *, rngs: nnx.Rngs):
        self.self_attn = nnx.MultiHeadAttention(
            num_heads=num_heads, in_features=dim, dropout_rate=dropout, decode=False, rngs=rngs
        )
        self.norm1 = nnx.LayerNorm(dim, rngs=rngs)
        self.norm2 = nnx.LayerNorm(dim, rngs=rngs)
        self.linear1 = nnx.Linear(dim, ff, rngs=rngs)
        self.linear2 = nnx.Linear(ff, dim, rngs=rngs)
        self.dropout = nnx.Dropout(dropout, rngs=rngs)

    def __call__(self, x, *, deterministic: bool):
        h = self.norm1(x)
        x = x + self.self_attn(h, deterministic=deterministic)
        h = self.norm2(x)
        h = self.linear2(self.dropout(nnx.relu(self.linear1(h)), deterministic=deterministic))
        return x + h


class _DecoderLayer(nnx.Module):
    """Pre-norm transformer decoder layer (self-attn + cross-attn to memory)."""

    def __init__(self, dim: int, num_heads: int, ff: int, dropout: float, *, rngs: nnx.Rngs):
        self.self_attn = nnx.MultiHeadAttention(
            num_heads=num_heads, in_features=dim, dropout_rate=dropout, decode=False, rngs=rngs
        )
        self.cross_attn = nnx.MultiHeadAttention(
            num_heads=num_heads, in_features=dim, dropout_rate=dropout, decode=False, rngs=rngs
        )
        self.norm1 = nnx.LayerNorm(dim, rngs=rngs)
        self.norm2 = nnx.LayerNorm(dim, rngs=rngs)
        self.norm3 = nnx.LayerNorm(dim, rngs=rngs)
        self.linear1 = nnx.Linear(dim, ff, rngs=rngs)
        self.linear2 = nnx.Linear(ff, dim, rngs=rngs)
        self.dropout = nnx.Dropout(dropout, rngs=rngs)

    def __call__(self, x, memory, *, deterministic: bool):
        h = self.norm1(x)
        x = x + self.self_attn(h, deterministic=deterministic)
        h = self.norm2(x)
        x = x + self.cross_attn(h, memory, deterministic=deterministic)
        h = self.norm3(x)
        h = self.linear2(self.dropout(nnx.relu(self.linear1(h)), deterministic=deterministic))
        return x + h


class ACT(_model.BaseModel):
    def __init__(self, config: _act_config.ACTConfig, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.hidden_dim = config.hidden_dim
        self.latent_dim = config.latent_dim
        self.kl_weight = config.kl_weight

        dim = config.hidden_dim
        ad = config.action_dim
        ah = config.action_horizon

        # --- vision ---
        self.backbone = _ResNet18(dim, rngs=rngs)
        self._num_cams = len(_model.IMAGE_KEYS)
        h_feat = _model.IMAGE_RESOLUTION[0] // 32
        w_feat = _model.IMAGE_RESOLUTION[1] // 32
        self._tokens_per_img = h_feat * w_feat
        num_img_tokens = self._num_cams * self._tokens_per_img

        # --- CVAE "style" encoder (state + action chunk -> latent) ---
        self.cvae_state_proj = nnx.Linear(ad, dim, rngs=rngs)
        self.cvae_action_proj = nnx.Linear(ad, dim, rngs=rngs)
        self.cls_token = nnx.Param(jax.random.normal(rngs.params(), (dim,)) * 0.02)
        self.cvae_pos_emb = nnx.Param(jax.random.normal(rngs.params(), (2 + ah, dim)) * 0.02)
        self.cvae_layers = [
            _EncoderLayer(dim, config.num_heads, config.dim_feedforward, config.dropout, rngs=rngs)
            for _ in range(config.num_cvae_layers)
        ]
        self.cvae_norm = nnx.LayerNorm(dim, rngs=rngs)
        self.latent_head = nnx.Linear(dim, 2 * config.latent_dim, rngs=rngs)

        # --- main DETR-style encoder/decoder ---
        self.latent_proj = nnx.Linear(config.latent_dim, dim, rngs=rngs)
        self.enc_state_proj = nnx.Linear(ad, dim, rngs=rngs)
        self.enc_pos_emb = nnx.Param(jax.random.normal(rngs.params(), (2 + num_img_tokens, dim)) * 0.02)
        self.encoder_layers = [
            _EncoderLayer(dim, config.num_heads, config.dim_feedforward, config.dropout, rngs=rngs)
            for _ in range(config.num_encoder_layers)
        ]
        self.enc_norm = nnx.LayerNorm(dim, rngs=rngs)

        self.query_embed = nnx.Param(jax.random.normal(rngs.params(), (ah, dim)) * 0.02)
        self.decoder_layers = [
            _DecoderLayer(dim, config.num_heads, config.dim_feedforward, config.dropout, rngs=rngs)
            for _ in range(config.num_decoder_layers)
        ]
        self.dec_norm = nnx.LayerNorm(dim, rngs=rngs)
        self.action_head = nnx.Linear(dim, ad, rngs=rngs)

        # Toggled by model.train() / model.eval().
        self.deterministic = True

    def _encode_images(self, obs: _model.Observation) -> at.Float[at.Array, "b n emb"]:
        tokens = []
        for key in _model.IMAGE_KEYS:
            feat = self.backbone(obs.images[key])  # [b, h, w, dim]
            b, h, w, c = feat.shape
            tokens.append(jnp.reshape(feat, (b, h * w, c)))
        return jnp.concatenate(tokens, axis=1)  # [b, num_cams * h*w, dim]

    def _cvae_encode(
        self, state: at.Float[at.Array, "b ad"], actions: _model.Actions, *, deterministic: bool
    ) -> tuple[at.Float[at.Array, "b z"], at.Float[at.Array, "b z"]]:
        b = state.shape[0]
        cls = jnp.broadcast_to(self.cls_token.value[None, None, :], (b, 1, self.hidden_dim))
        qpos = self.cvae_state_proj(state)[:, None, :]
        act = self.cvae_action_proj(actions)  # [b, ah, dim]
        x = jnp.concatenate([cls, qpos, act], axis=1)
        x = x + self.cvae_pos_emb.value[None, : x.shape[1], :]
        for layer in self.cvae_layers:
            x = layer(x, deterministic=deterministic)
        x = self.cvae_norm(x)
        stats = self.latent_head(x[:, 0])  # use CLS token
        mu, logvar = jnp.split(stats, 2, axis=-1)
        return mu, logvar

    def _decode(
        self,
        z: at.Float[at.Array, "b z"],
        state: at.Float[at.Array, "b ad"],
        img_tokens: at.Float[at.Array, "b n emb"],
        *,
        deterministic: bool,
    ) -> _model.Actions:
        b = state.shape[0]
        latent = self.latent_proj(z)[:, None, :]
        qpos = self.enc_state_proj(state)[:, None, :]
        memory = jnp.concatenate([latent, qpos, img_tokens], axis=1)
        memory = memory + self.enc_pos_emb.value[None, : memory.shape[1], :]
        for layer in self.encoder_layers:
            memory = layer(memory, deterministic=deterministic)
        memory = self.enc_norm(memory)

        x = jnp.broadcast_to(self.query_embed.value[None], (b, self.action_horizon, self.hidden_dim))
        for layer in self.decoder_layers:
            x = layer(x, memory, deterministic=deterministic)
        x = self.dec_norm(x)
        return self.action_head(x)  # [b, ah, ad]

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        preprocess_rng, sample_rng = jax.random.split(rng)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)
        deterministic = not train

        img_tokens = self._encode_images(observation)
        mu, logvar = self._cvae_encode(observation.state, actions, deterministic=deterministic)

        # Reparameterize.
        eps = jax.random.normal(sample_rng, mu.shape)
        z = mu + jnp.exp(0.5 * logvar) * eps

        pred = self._decode(z, observation.state, img_tokens, deterministic=deterministic)

        recon = jnp.mean(jnp.abs(pred - actions), axis=-1)  # [b, ah]  L1 per timestep
        kl = -0.5 * jnp.sum(1.0 + logvar - jnp.square(mu) - jnp.exp(logvar), axis=-1)  # [b]
        # Broadcasting kl (constant over the chunk) so that jnp.mean over [b, ah] gives
        # exactly mean(L1) + kl_weight * mean(KL).
        return recon + self.kl_weight * kl[:, None]

    @override
    def sample_actions(self, rng: at.KeyArrayLike, observation: _model.Observation, **kwargs) -> _model.Actions:
        observation = _model.preprocess_observation(None, observation, train=False)
        img_tokens = self._encode_images(observation)
        batch_size = observation.state.shape[0]
        # At inference the latent is the prior mean (zeros); the CVAE encoder is unused.
        z = jnp.zeros((batch_size, self.latent_dim), dtype=observation.state.dtype)
        return self._decode(z, observation.state, img_tokens, deterministic=True)
