"""PI0.5、ACT 和策略加载共享的模型无关 JAX 接口.

训练代码只依赖 BaseModel.compute_loss,推理代码只依赖
BaseModel.sample_actions.模型专属实现保留在 pi0.py 和 act.py 中,
Observation 则定义它们共同的结构化输入.
"""

import abc
from collections.abc import Sequence
import dataclasses
import enum
import logging
import pathlib
from typing import Generic, TypeVar

import augmax
from flax import nnx
from flax import struct
from flax import traverse_util
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp

from openpi.shared import image_tools
import openpi.shared.array_typing as at

logger = logging.getLogger("openpi")

# 模型输入使用 JAX 或 NumPy 数组表示.
ArrayT = TypeVar("ArrayT", bound=jax.Array | np.ndarray)


class ModelType(enum.Enum):
    """当前支持的模型类型."""

    PI0 = "pi0"
    PI05 = "pi05"
    ACT = "act"


# 模型固定期望以下三个相机视角.
IMAGE_KEYS = (
    "base_0_rgb",
    "left_wrist_0_rgb",
    "right_wrist_0_rgb",
)


# 如果后续发布更小模型,图像分辨率可能需要调整.
IMAGE_RESOLUTION = (224, 224)


# 数据格式
#
# Data transforms 先生成嵌套字典,随后转换为 `Observation` 和 `Actions` 对象.
#
# 字典结构如下:
# {
#     # Observation 数据.
#     "image": {
#         "base_0_rgb": (float32|uint8)[*b, h, w, 3],  # RGB 图像,范围为 [-1, 1] 或 [0, 255]
#         ...  # 其他相机视角
#     },
#     "image_mask": {
#         "base_0_rgb": bool[*b],  # true 表示图像有效
#         ...  # 其他视角的 mask
#     },
#     "state": float32[*b, s],  # 低维机器人状态
#     "tokenized_prompt": int32[*b, l],  # 可选:tokenize 后的语言指令
#     "tokenized_prompt_mask": bool[*b, l],  # 可选:语言 token mask
#
#      # Actions 数据.
#      "actions": float32[*b ah ad]
# }
# 其中:
#   *b = batch 维度
#   h,w = 图像高和宽
#   s = state 维度
#   l = token 序列长度
#
@at.typecheck
@struct.dataclass
class Observation(Generic[ArrayT]):
    """保存 observation,即模型输入.

    预期字典结构参见 `Observation.from_dict`;data transforms 应输出该格式.
    """

    # 范围为 [-1, 1] 的 float32 图像.
    images: dict[str, at.Float[ArrayT, "*b h w c"]]
    # 图像有效性 mask,键与 images 一致.
    image_masks: dict[str, at.Bool[ArrayT, "*b"]]
    # 低维机器人状态.
    state: at.Float[ArrayT, "*b s"]

    # tokenize 后的 prompt.
    tokenized_prompt: at.Int[ArrayT, "*b l"] | None = None
    # prompt token 的有效性 mask.
    tokenized_prompt_mask: at.Bool[ArrayT, "*b l"] | None = None

    @classmethod
    def from_dict(cls, data: at.PyTree[ArrayT]) -> "Observation[ArrayT]":
        """定义嵌套字典到结构化 Observation 的映射."""
        # tokenized_prompt 与 tokenized_prompt_mask 必须同时提供.
        if ("tokenized_prompt" in data) != ("tokenized_prompt_mask" in data):
            raise ValueError("tokenized_prompt and tokenized_prompt_mask must be provided together.")
        # uint8 图像统一转换为范围 [-1, 1] 的 float32.
        for key in data["image"]:
            if data["image"][key].dtype == np.uint8:
                data["image"][key] = data["image"][key].astype(np.float32) / 255.0 * 2.0 - 1.0
        return cls(
            images=data["image"],
            image_masks=data["image_mask"],
            state=data["state"],
            tokenized_prompt=data.get("tokenized_prompt"),
            tokenized_prompt_mask=data.get("tokenized_prompt_mask"),
        )

    def to_dict(self) -> at.PyTree[ArrayT]:
        """将 Observation 转回嵌套字典."""
        result = dataclasses.asdict(self)
        result["image"] = result.pop("images")
        result["image_mask"] = result.pop("image_masks")
        return result


# 定义动作格式.Data transforms 生成的字典通过 "actions" 字段携带该数据.
Actions = at.Float[ArrayT, "*b ah ad"]


def preprocess_observation(
    rng: at.KeyArrayLike | None,
    observation: Observation,
    *,
    train: bool = False,
    image_keys: Sequence[str] = IMAGE_KEYS,
    image_resolution: tuple[int, int] = IMAGE_RESOLUTION,
) -> Observation:
    """预处理 observation.

    训练时执行图像增强,按需调整图像尺寸,并为缺失视角补默认 image mask.
    """

    if not set(image_keys).issubset(observation.images):
        raise ValueError(f"images dict missing keys: expected {image_keys}, got {list(observation.images)}")

    batch_shape = observation.state.shape[:-1]

    # 本函数在模型计算内部执行.数据集 transforms 通常已完成 resize,
    # 这里保留尺寸检查,使直接调用模型时仍然安全.
    out_images = {}
    for key in image_keys:
        image = observation.images[key]
        if image.shape[1:3] != image_resolution:
            logger.info(f"Resizing image {key} from {image.shape[1:3]} to {image_resolution}")
            image = image_tools.resize_with_pad(image, *image_resolution)

        if train:
            # augmax 要求 [0, 1] 输入,先从 [-1, 1] 转换.
            image = image / 2.0 + 0.5

            transforms = []
            if "wrist" not in key:
                height, width = image.shape[1:3]
                transforms += [
                    augmax.RandomCrop(int(width * 0.95), int(height * 0.95)),
                    augmax.Resize(width, height),
                    augmax.Rotate((-5, 5)),
                ]
            transforms += [
                augmax.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5),
            ]
            sub_rngs = jax.random.split(rng, image.shape[0])
            image = jax.vmap(augmax.Chain(*transforms))(sub_rngs, image)

            # 图像增强后转换回 [-1, 1].
            image = image * 2.0 - 1.0

        out_images[key] = image

    # 整理各相机视角的有效性 mask.
    out_masks = {}
    for key in out_images:
        if key not in observation.image_masks:
            # 未显式提供 mask 时默认该视角有效.
            out_masks[key] = jnp.ones(batch_shape, dtype=jnp.bool)
        else:
            out_masks[key] = jnp.asarray(observation.image_masks[key])

    return Observation(
        images=out_images,
        image_masks=out_masks,
        state=observation.state,
        tokenized_prompt=observation.tokenized_prompt,
        tokenized_prompt_mask=observation.tokenized_prompt_mask,
    )


@dataclasses.dataclass(frozen=True)
class BaseModelConfig(abc.ABC):
    """所有模型共享的配置基类.

    具体模型配置应继承本类,并实现 `create` 来构造对应网络.
    """

    # 动作空间维度.
    action_dim: int
    # 动作序列长度.
    action_horizon: int
    # tokenize 后 prompt 的最大长度.
    max_token_len: int

    @property
    @abc.abstractmethod
    def model_type(self) -> ModelType:
        """返回模型类型."""

    @abc.abstractmethod
    def create(self, rng: at.KeyArrayLike) -> "BaseModel":
        """创建新模型并初始化参数."""

    def load(self, params: at.Params, *, remove_extra_params: bool = True) -> "BaseModel":
        """重建配置指定的网络图,并写入从 checkpoint 恢复的参数叶子."""
        model = nnx.eval_shape(self.create, jax.random.key(0))
        graphdef, state = nnx.split(model)
        if remove_extra_params:
            params = ocp.transform_utils.intersect_trees(state.to_pure_dict(), params)
        at.check_pytree_equality(expected=state.to_pure_dict(), got=params, check_shapes=True, check_dtypes=False)
        state.replace_by_pure_dict(params)
        return nnx.merge(graphdef, state)

    @abc.abstractmethod
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[Observation, Actions]:
        """返回模型输入规格,各叶子值为 jax.ShapeDtypeStruct."""

    def fake_obs(self, batch_size: int = 1) -> Observation:
        observation_spec, _ = self.inputs_spec(batch_size=batch_size)
        return jax.tree.map(lambda x: jnp.ones(x.shape, x.dtype), observation_spec)

    def fake_act(self, batch_size: int = 1) -> Actions:
        _, action_spec = self.inputs_spec(batch_size=batch_size)
        return jax.tree.map(lambda x: jnp.ones(x.shape, x.dtype), action_spec)


@dataclasses.dataclass
class BaseModel(nnx.Module, abc.ABC):
    """所有模型实现的公共基类.

    具体模型应继承本类,并调用 super().__init__() 初始化 action_dim、
    action_horizon 和 max_token_len 等共享字段.
    """

    action_dim: int
    action_horizon: int
    max_token_len: int

    @abc.abstractmethod
    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        observation: Observation,
        actions: Actions,
        *,
        train: bool = False,
    ) -> at.Float[at.Array, "*b ah"]: ...

    @abc.abstractmethod
    def sample_actions(self, rng: at.KeyArrayLike, observation: Observation, **kwargs) -> Actions: ...


def restore_params(
    params_path: pathlib.Path | str,
    *,
    restore_type: type[np.ndarray] | type[jax.Array] = jax.Array,
    dtype: jnp.dtype | None = None,
    sharding: jax.sharding.Sharding | None = None,
) -> at.Params:
    """从 checkpoint 恢复非结构化参数 PyTree.

    既支持训练期间由 `save_state` 保存的 checkpoint(参见
    `training/checkpoints.py`),也支持发布的预训练 checkpoint.

    参数:
        params_path:checkpoint 参数目录的本地路径.
        restore_type:参数恢复后的类型;可设为 `np.ndarray`.
        dtype:统一恢复为指定 dtype;未设置时保留 checkpoint 原始 dtype.
        sharding:参数使用的分片;未设置时在全部设备上复制.

    返回:
        恢复后的参数树.
    """
    params_path = pathlib.Path(params_path).resolve() if not str(params_path).startswith("gs://") else params_path

    if restore_type is jax.Array and sharding is None:
        mesh = jax.sharding.Mesh(jax.devices(), ("x",))
        sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    with ocp.PyTreeCheckpointer() as ckptr:
        metadata = ckptr.metadata(params_path)
        item = {"params": metadata["params"]}

        params = ckptr.restore(
            params_path,
            ocp.args.PyTreeRestore(
                item=item,
                restore_args=jax.tree.map(
                    lambda _: ocp.ArrayRestoreArgs(sharding=sharding, restore_type=restore_type, dtype=dtype), item
                ),
            ),
        )["params"]

    # 训练时由 `save_state` 保存的参数路径会带有 nnx.State 添加的 "value" 后缀.
    # 这里统一移除该后缀,始终返回 NNX 所称的 pure dict.
    flat_params = traverse_util.flatten_dict(params)
    if all(kp[-1] == "value" for kp in flat_params):
        flat_params = {kp[:-1]: v for kp, v in flat_params.items()}
    return traverse_util.unflatten_dict(flat_params)
