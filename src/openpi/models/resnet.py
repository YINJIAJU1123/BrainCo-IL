"""Shared ResNet-18 / GroupNorm image trunk for ACT and Diffusion Policy."""

import flax.linen as nn
import flax.nnx as nnx


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


class ResNet18(nnx.Module):
    """ResNet-18 trunk (from scratch) projecting to `out_dim` channels.

    Input:  [b, H, W, 3] in [-1, 1].
    Output: [b, H/32, W/32, out_dim] feature map.
    """

    def __init__(self, out_dim: int, *, width: int = 64, rngs: nnx.Rngs):
        self.conv1 = nnx.Conv(3, width, (7, 7), strides=2, padding="SAME", use_bias=False, rngs=rngs)
        self.norm1 = nnx.GroupNorm(width, num_groups=_num_groups(width), rngs=rngs)

        chans = [width, width * 2, width * 4, width * 8]
        blocks_per_stage = [2, 2, 2, 2]
        strides = [1, 2, 2, 2]
        self.stages: list[list[_BasicBlock]] = []
        in_ch = width
        for out_ch, n_blocks, stage_stride in zip(chans, blocks_per_stage, strides, strict=True):
            stage = []
            for i in range(n_blocks):
                stage.append(_BasicBlock(in_ch, out_ch, stride=stage_stride if i == 0 else 1, rngs=rngs))
                in_ch = out_ch
            self.stages.append(stage)

        self.proj = nnx.Conv(width * 8, out_dim, (1, 1), rngs=rngs)

    def __call__(self, x):
        x = nnx.relu(self.norm1(self.conv1(x)))
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding="SAME")
        for stage in self.stages:
            for block in stage:
                x = block(x)
        return self.proj(x)
