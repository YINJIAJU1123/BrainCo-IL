"""第一个训练步骤前的初始参数加载逻辑.

WeightLoader 用于初始化新的 TrainState.恢复中断训练时不会调用它,
而是由 checkpoints.py 恢复完整训练状态.
"""

import dataclasses
import logging
import re
from typing import Protocol, runtime_checkable

import flax.traverse_util
import numpy as np

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.download as download

logger = logging.getLogger(__name__)


@runtime_checkable
class WeightLoader(Protocol):
    def load(self, params: at.Params) -> at.Params:
        """加载模型权重.

        参数:
            params:模型参数,由表示各层参数的数组对象组成的嵌套结构.

        返回:
            加载后的参数,结构必须与 `params` 一致.如果只加载参数子集,
            loader 必须将其与 `params` 合并后返回.
        """


@dataclasses.dataclass(frozen=True)
class NoOpWeightLoader(WeightLoader):
    """保留随机初始化,供 ACT 从头训练等场景使用."""

    def load(self, params: at.Params) -> at.Params:
        return params


@dataclasses.dataclass(frozen=True)
class CheckpointWeightLoader(WeightLoader):
    """从 checkpoint 加载完整权重.

    支持:
      训练生成的 checkpoint:
        示例:"./checkpoints/<config>/<exp>/<step>/params"
      发布的 checkpoint:
        示例:"gs://openpi-assets/checkpoints/<model>/params"
    """

    params_path: str

    def load(self, params: at.Params) -> at.Params:
        # 先加载为 np.ndarray,再由训练初始化逻辑转换并按目标布局分片.
        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        # 补回 checkpoint 中不存在的 LoRA 参数.
        return _merge_params(loaded_params, params, missing_regex=".*lora.*")


@dataclasses.dataclass(frozen=True)
class PartialCheckpointWeightLoader(WeightLoader):
    """从 checkpoint 加载权重,并跳过允许 shape 不匹配的层.

    适用于目标模型 action_dim 与预训练模型不同的情况.例如将 32 维权重
    加载到 58 维模型时,可以跳过 action_in_proj 和 action_out_proj,
    同时加载其余 shape 兼容的层.

    支持:
      训练生成的 checkpoint:
        示例:"./checkpoints/<config>/<exp>/<step>/params"
      发布的 checkpoint:
        示例:"gs://openpi-assets/checkpoints/<model>/params"
    """

    params_path: str
    # shape 不匹配时允许跳过的层名正则,默认跳过动作/状态投影层.
    skip_on_mismatch_regex: str = ".*(action_in_proj|action_out_proj|state_proj|state_mlp_in|state_mlp_out).*"

    def load(self, params: at.Params) -> at.Params:
        # 先加载为 np.ndarray,再由训练初始化逻辑转换并按目标布局分片.
        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)

        # 合并参数,并允许指定层存在 shape 不匹配.
        flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
        flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")

        skip_pattern = re.compile(self.skip_on_mismatch_regex)
        result = {}
        skipped_keys = []
        missing_keys = []

        # 加载 shape 兼容的预训练叶子;对于明确允许发生维度变化的
        # action/state 投影层,保留目标模型自身的初始化.
        for k, v in flat_loaded.items():
            if k in flat_ref:
                ref_shape = getattr(flat_ref[k], "shape", None)
                loaded_shape = getattr(v, "shape", None)

                # 检查 shape 是否匹配,或当前层是否允许跳过.
                if ref_shape == loaded_shape:
                    # shape 匹配,加载 checkpoint 权重.
                    result[k] = v.astype(flat_ref[k].dtype) if v.dtype != flat_ref[k].dtype else v
                elif skip_pattern.fullmatch(k):
                    # shape 不匹配但位于跳过列表,保留目标模型初始值.
                    skipped_keys.append(k)
                    logger.info(f"Skipping layer {k}: shape mismatch ({loaded_shape} -> {ref_shape})")
                else:
                    # shape 不匹配且不允许跳过,视为配置或 checkpoint 错误.
                    raise ValueError(
                        f"Shape mismatch at {k}: expected {ref_shape}, got {loaded_shape}. "
                        f"Layer does not match skip_on_mismatch_regex pattern."
                    )

        # 从目标模型补回缺失的 LoRA/VLASH state-condition 参数和被跳过层.
        # state_cond 层在原始 PI0.5 checkpoint 中不存在,因此仅处理 shape
        # mismatch 还不够,也要允许 skip regex 匹配的目标参数完全缺失.
        lora_pattern = re.compile(".*lora.*")
        for k in flat_ref:
            if k not in result and (lora_pattern.fullmatch(k) or skip_pattern.fullmatch(k)):
                result[k] = flat_ref[k]
                if k not in skipped_keys and not lora_pattern.fullmatch(k):
                    missing_keys.append(k)

        if skipped_keys:
            logger.warning(
                f"Partially loaded checkpoint: skipped {len(skipped_keys)} layers with shape mismatches. "
                f"These layers will use random initialization: {', '.join(skipped_keys)}"
            )
        if missing_keys:
            logger.warning(
                f"Partially loaded checkpoint: initialized {len(missing_keys)} target-only layers. "
                f"These layers are absent from the checkpoint: {', '.join(missing_keys)}"
            )

        return flax.traverse_util.unflatten_dict(result, sep="/")


@dataclasses.dataclass(frozen=True)
class PaliGemmaWeightLoader(WeightLoader):
    """从官方 PaliGemma checkpoint 加载权重.

    同名权重会被覆盖,目标模型额外参数保持不变,
    因而可以保留 PI0 使用的 action expert 参数.
    """

    def load(self, params: at.Params) -> at.Params:
        path = download.maybe_download(
            "gs://vertex-model-garden-paligemma-us/paligemma/pt_224.npz", gs={"token": "anon"}
        )
        with path.open("rb") as f:
            flat_params = dict(np.load(f, allow_pickle=False))
        loaded_params = {"PaliGemma": flax.traverse_util.unflatten_dict(flat_params, sep="/")["params"]}
        # 从目标模型补回 checkpoint 中缺失的全部权重.
        return _merge_params(loaded_params, params, missing_regex=".*")


def _merge_params(loaded_params: at.Params, params: at.Params, *, missing_regex: str) -> at.Params:
    """将加载参数与目标模型参考参数合并.

    参数:
        loaded_params:从 checkpoint 加载的参数.
        params:目标模型参考参数.
        missing_regex:需要从参考参数补回的缺失键正则.

    返回:
        合并后的新参数字典.
    """
    flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
    flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")

    # 首先接收所有存在于目标参数树中的加载权重.
    result = {}
    for k, v in flat_loaded.items():
        if k in flat_ref:
            result[k] = v.astype(flat_ref[k].dtype) if v.dtype != flat_ref[k].dtype else v

    flat_loaded.clear()

    # 然后按 missing_regex 从目标参数树补回缺失权重.
    pattern = re.compile(missing_regex)
    for k in {k for k in flat_ref if pattern.fullmatch(k)}:
        if k not in result:
            result[k] = flat_ref[k]

    return flax.traverse_util.unflatten_dict(result, sep="/")
