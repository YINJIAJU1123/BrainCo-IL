"""训练使用的 JAX device mesh 与 FSDP 放置规则.

batch 轴负责数据并行;fsdp 轴可在每个数据并行组内继续切分大型参数张量.
"""

import contextlib
import logging

import jax
import numpy as np

BATCH_AXIS = "batch"
FSDP_AXIS = "fsdp"
# 启用 FSDP 时,数据同时沿 batch 轴和 FSDP 轴切分.
DATA_AXIS = (BATCH_AXIS, FSDP_AXIS)


class _MeshState:
    active_mesh: jax.sharding.Mesh | None = None


def make_mesh(num_fsdp_devices: int) -> jax.sharding.Mesh:
    """将全部可见 JAX 设备组织为二维逻辑 mesh."""
    if jax.device_count() % num_fsdp_devices != 0:
        raise ValueError(
            f"Number of devices {jax.device_count()} must be divisible by the number of FSDP devices {num_fsdp_devices}."
        )
    mesh_shape = (jax.device_count() // num_fsdp_devices, num_fsdp_devices)
    return jax.make_mesh(mesh_shape, (BATCH_AXIS, FSDP_AXIS))


@contextlib.contextmanager
def set_mesh(mesh: jax.sharding.Mesh):
    """在上下文中保存当前全局 mesh.

    将 mesh 参数逐层传入网络非常繁琐;在 JAX 提供更合适的 API 前,
    使用该上下文管理器维护全局引用,并仅供下方
    `activation_sharding_constraint` 使用.
    """
    if _MeshState.active_mesh is not None:
        raise ValueError("Cannot nest set_mesh context managers.")
    _MeshState.active_mesh = mesh
    try:
        yield
    finally:
        _MeshState.active_mesh = None


def activation_sharding_constraint(pytree):
    """约束大型中间激活值,使其与数据分片布局保持一致."""
    if _MeshState.active_mesh is None:
        return pytree
    return jax.lax.with_sharding_constraint(
        pytree, jax.sharding.NamedSharding(_MeshState.active_mesh, jax.sharding.PartitionSpec(DATA_AXIS))
    )


def fsdp_sharding(
    pytree,
    mesh: jax.sharding.Mesh,
    *,
    min_size_mbytes: int = 4,  # 4 MiB
    log: bool = False,
):
    """根据 mesh 形状为数组 PyTree 应用 FSDP 分片.

    参数:
        pytree:需要应用 mesh 分片的 PyTree;只有带 shape 属性的数组对象会参与.
        mesh:应用到 PyTree 的设备 mesh.
        min_size_mbytes:参与分片的最小数组大小,单位 MiB;更小的数组保持复制.
        log:为 true 时记录各数组的分片决策.

    返回:
        与输入 PyTree 结构一致的 sharding PyTree.
    """
    min_size_bytes = min_size_mbytes * 2**20

    def _shard_arr(kp, array: jax.ShapeDtypeStruct):
        # 未真正启用 FSDP 时全部复制,避免无意义的分片计算和日志.
        if mesh.shape[FSDP_AXIS] == 1:
            return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
        # 标量和向量保持复制.
        if not hasattr(array, "shape"):
            return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
        if len(array.shape) < 2:
            return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
        # 小数组保持复制.
        if (arr_size := np.prod(array.shape) * np.dtype(array.dtype).itemsize) < min_size_bytes:
            return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

        # 大型张量沿可整除的最大维度切分;标量、向量和小数组保持复制,
        # 因为对它们分片的通信与管理开销更高.
        axes = np.argsort(array.shape)[::-1]
        spec = [None] * len(axes)
        for i in axes:
            if array.shape[i] % mesh.shape[FSDP_AXIS] == 0:
                if log:
                    logging.info(
                        f"Sharding {jax.tree_util.keystr(kp)} of shape {array.shape} ({arr_size / 2**20:.2f} MiB) along axis {i}"
                    )
                spec[i] = FSDP_AXIS
                return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(*spec))

        # 找不到可整除维度时保持复制.
        if log:
            logging.warning(
                f"Could not find a valid sharding for {jax.tree_util.keystr(kp)} of shape {array.shape} with mesh of shape {mesh.shape}"
            )
        return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    return jax.tree_util.tree_map_with_path(_shard_arr, pytree)
