"""Action Chunking Transformer(ACT)模型配置.

ACT(Zhao et al., 2023)是一种 CVAE 风格策略,将多个相机视角和本体状态
映射为未来动作 chunk.

与 PI0/PI0.5 不同,ACT 不包含语言/VLM 模块,因此:
- 不存在 tokenized prompt,ACT 的模型 transforms 不能执行语言 tokenization;
- 通常按单任务训练,或显式加入 task-id 条件,因为它无法通过语言区分任务.

ACT 通常从头训练,不加载预训练 checkpoint,因此配置应使用 NoOpWeightLoader.

ACTConfig 与 Pi0Config 作为并列 ModelType 接入公共训练接口,
切换算法只需选择不同模型配置.
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

    # BaseModelConfig 共享字段,这里使用 ACT 默认值重新定义.
    action_dim: int = 56
    action_horizon: int = 100  # 即 action chunk size.
    # ACT 没有语言输入;max_token_len 不使用,仅为保持公共接口一致.
    max_token_len: int = 1

    # Transformer 超参数.
    hidden_dim: int = 512
    num_heads: int = 8
    dim_feedforward: int = 3200
    num_encoder_layers: int = 4
    num_decoder_layers: int = 7
    # 对 action chunk 编码的 CVAE style encoder 层数.
    num_cvae_layers: int = 4
    dropout: float = 0.1

    # CVAE 隐变量配置.
    latent_dim: int = 32
    kl_weight: float = 10.0

    # 视觉 backbone.当前提供 NNX 实现的随机初始化 ResNet-18 风格主干;
    # 生产训练可替换为 ImageNet 预训练 ResNet-18.
    vision_backbone: str = "resnet18"

    @property
    @override
    def model_type(self) -> _model.ModelType:
        return _model.ModelType.ACT

    @override
    def create(self, rng: at.KeyArrayLike) -> "ACT":
        """通过与 PI0.5 相同的 BaseModel 接口实例化 ACT."""
        from openpi.models.act import ACT

        return ACT(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            # ACT 没有语言输入,因此不创建 tokenized_prompt.
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
        """ACT 从头进行端到端训练,不冻结任何参数."""
        return nnx.Nothing
