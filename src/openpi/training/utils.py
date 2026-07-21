"""在 jitted 训练步骤之间传递的共享结构化状态."""

from collections.abc import Callable
from typing import Any

from flax import nnx
from flax import struct
import jax
import optax

from openpi.models import model as _model
from openpi.shared import array_typing as at


@at.typecheck
@struct.dataclass
class TrainState:
    """模型图、参数、优化器状态、训练步数和可选 EMA 参数."""
    step: at.Int[at.ArrayLike, ""]
    params: nnx.State
    model_def: nnx.GraphDef[_model.BaseModel]
    opt_state: optax.OptState
    tx: optax.GradientTransformation = struct.field(pytree_node=False)

    ema_decay: float | None = struct.field(pytree_node=False)
    ema_params: nnx.State | None = None


@at.typecheck
def tree_to_info(tree: at.PyTree, interp_func: Callable[[Any], str] = str) -> str:
    """将 PyTree 转为便于日志阅读的字符串.

    可通过 `interp_func` 将叶子值转换为更有意义的描述.
    """
    tree, _ = jax.tree_util.tree_flatten_with_path(tree)
    return "\n".join(f"{jax.tree_util.keystr(path)}: {interp_func(value)}" for path, value in tree)


@at.typecheck
def array_tree_to_info(tree: at.PyTree) -> str:
    """将数组 PyTree 转为包含 shape 和 dtype 的日志字符串."""
    return tree_to_info(tree, lambda x: f"{x.shape}@{x.dtype}")
