"""从 LeRobot 到 JAX 的训练输入流水线.

本模块负责框架边界:
  LeRobot/PyTorch worker 在 CPU 上解码并组装 batch;
  BrainCo/model transforms 将单条样本转换为模型输入约定;
  最后将 batch 转为带分片布局的 JAX 数组,交给编译后的训练步骤.

机器人语义保留在 config.py 和 brainco_policy.py 中,不写入通用 DataLoader.
"""

from collections.abc import Iterator, Sequence
import logging
import multiprocessing
import os
import typing
from typing import Protocol, SupportsIndex, TypeVar

import jax
import jax.numpy as jnp
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import numpy as np
import torch

import openpi.models.model as _model
import openpi.training.config as _config
import openpi.transforms as _transforms

T_co = TypeVar("T_co", covariant=True)


def _patch_torch_stack_for_datasets_column() -> None:
    """兼容旧版 LeRobot 与新版 HuggingFace datasets 的 Column 对象."""
    if getattr(torch.stack, "_openpi_datasets_column_compat", False):
        return

    original_stack = torch.stack

    def stack_compat(input, *stack_args, **stack_kwargs):
        if input.__class__.__name__ == "Column" and input.__class__.__module__.startswith("datasets."):
            values = list(input)
            if values and torch.is_tensor(values[0]):
                return original_stack(values, *stack_args, **stack_kwargs)
            return torch.as_tensor(values)
        return original_stack(input, *stack_args, **stack_kwargs)

    stack_compat._openpi_datasets_column_compat = True  # noqa: SLF001
    torch.stack = stack_compat


_patch_torch_stack_for_datasets_column()


def _create_lerobot_dataset_compat(*args, **kwargs) -> lerobot_dataset.LeRobotDataset:
    """在安装兼容补丁后创建 LeRobotDataset."""
    dataset_cls = kwargs.pop("dataset_cls", lerobot_dataset.LeRobotDataset)
    kwargs.setdefault("video_backend", os.environ.get("OPENPI_LEROBOT_VIDEO_BACKEND", "pyav"))
    return dataset_cls(*args, **kwargs)


class Dataset(Protocol[T_co]):
    """支持随机访问的数据集接口."""

    def __getitem__(self, index: SupportsIndex) -> T_co:
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class DataLoader(Protocol[T_co]):
    """数据加载器接口."""

    def data_config(self) -> _config.DataConfig:
        """返回当前 DataLoader 使用的数据配置."""
        raise NotImplementedError("Subclasses of DataLoader should implement data_config.")

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of DataLoader should implement __iter__.")


class TransformedDataset(Dataset[T_co]):
    """在组装 batch 前,对单条样本执行配置好的 transform 链."""

    def __init__(self, dataset: Dataset, transforms: Sequence[_transforms.DataTransformFn]):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)

    def __getitem__(self, index: SupportsIndex) -> T_co:
        return self._transform(self._dataset[index])

    def __len__(self) -> int:
        return len(self._dataset)


class FakeDataset(Dataset):
    def __init__(self, model_config: _model.BaseModelConfig, num_samples: int):
        self._num_samples = num_samples
        self._observation_spec, self._action_spec = model_config.inputs_spec()

    def __getitem__(self, index: SupportsIndex) -> dict:
        rng = jax.random.key(index.__index__())

        def make_from_spec(spec: jax.ShapeDtypeStruct):
            nonlocal rng
            rng, data_rng = jax.random.split(rng)
            # FakeDataset 生成单样本时去掉临时 batch 维.
            shape = spec.shape[1:]
            if spec.dtype == jnp.float32:
                return jax.random.uniform(data_rng, shape=shape, minval=-1.0, maxval=1.0)
            if spec.dtype == jnp.int32:
                return jax.random.randint(data_rng, shape=shape, minval=0, maxval=2048)
            return jnp.zeros(shape=shape, dtype=spec.dtype)

        observation = jax.tree.map(make_from_spec, self._observation_spec)
        action = jax.tree.map(make_from_spec, self._action_spec)

        return {
            **observation.to_dict(),
            "actions": action,
        }

    def __len__(self) -> int:
        return self._num_samples


def _create_sequence_dataset(
    repo_id: str,
    *,
    dataset_meta: lerobot_dataset.LeRobotDatasetMetadata,
    data_config: _config.DataConfig,
    action_horizon: int,
) -> lerobot_dataset.LeRobotDataset:
    delta_timestamps = {
        key: [t / dataset_meta.fps for t in range(action_horizon)] for key in data_config.action_sequence_keys
    }
    return _create_lerobot_dataset_compat(repo_id, delta_timestamps=delta_timestamps)


class ConcatLeRobotDataset(Dataset):
    """拼接多个 LeRobot 数据集,并使用每个数据集中的全部样本."""

    def __init__(
        self,
        datasets: list[Dataset],
        prompt_transforms: list[_transforms.DataTransformFn] | None = None,
    ):
        """创建拼接式多数据集.

        参数:
            datasets:需要拼接的数据集列表.
            prompt_transforms:每个数据集可选的 prompt transform.
        """
        self._datasets = datasets
        self._prompt_transforms = prompt_transforms or [lambda x: x] * len(datasets)

        # 计算累计长度,用于将全局索引映射回具体数据集.
        self._dataset_lengths = [len(d) for d in datasets]
        self._cumulative_lengths = np.cumsum([0, *self._dataset_lengths])
        self._total_length = sum(self._dataset_lengths)

        logging.info(f"Created ConcatLeRobotDataset with {len(datasets)} datasets:")
        for i, dataset in enumerate(datasets):
            percentage = len(dataset) / self._total_length * 100
            logging.info(f"  Dataset {i}: {len(dataset)} samples ({percentage:.1f}%)")
        logging.info(f"  Total samples: {self._total_length} (all data included)")

    def __getitem__(self, index: SupportsIndex) -> dict:
        idx = index.__index__()

        # 定位当前全局索引所属的数据集.
        dataset_idx = int(np.searchsorted(self._cumulative_lengths[1:], idx, side="right"))
        local_idx = int(idx - self._cumulative_lengths[dataset_idx])

        sample = self._datasets[dataset_idx][local_idx]

        # 如果配置了数据集专属 prompt transform,则在这里应用.
        if self._prompt_transforms[dataset_idx] is not None:
            sample = self._prompt_transforms[dataset_idx](sample)

        return sample

    def __len__(self) -> int:
        return self._total_length


class MultiLeRobotDataset(Dataset):
    """按给定权重从多个 LeRobot 数据集中采样."""

    def __init__(
        self,
        datasets: list[Dataset],
        weights: list[float],
        prompt_transforms: list[_transforms.DataTransformFn] | None = None,
    ):
        """创建多数据集加权采样器.

        参数:
            datasets:参与采样的数据集列表.
            weights:每个数据集的采样权重,总和必须为 1.
            prompt_transforms:每个数据集可选的 prompt transform.
        """
        if len(datasets) != len(weights):
            raise ValueError(f"Number of datasets ({len(datasets)}) must match number of weights ({len(weights)})")

        if not np.isclose(sum(weights), 1.0):
            raise ValueError(f"Dataset weights must sum to 1.0, got {sum(weights)}")

        self._datasets = datasets
        self._weights = np.array(weights)
        self._prompt_transforms = prompt_transforms or [lambda x: x] * len(datasets)

        # 记录各数据集长度并计算虚拟数据集总长度.
        self._dataset_lengths = [len(d) for d in datasets]
        self._total_length = sum(self._dataset_lengths)

        # 预先构建符合采样权重的虚拟索引映射.
        self._dataset_indices = []
        self._local_indices = []

        # 根据权重创建索引映射,使虚拟数据集中的索引按权重分布.
        for dataset_idx, (dataset, weight) in enumerate(zip(datasets, weights, strict=False)):
            num_samples = int(self._total_length * weight)
            dataset_len = len(dataset)

            # 为当前数据集生成索引;样本不足时允许重复.
            local_idxs = np.random.choice(dataset_len, size=num_samples, replace=True)

            self._dataset_indices.extend([dataset_idx] * num_samples)
            self._local_indices.extend(local_idxs.tolist())

        logging.info(f"Created MultiLeRobotDataset with {len(datasets)} datasets (weighted sampling):")
        for i, (dataset, weight) in enumerate(zip(datasets, weights, strict=False)):
            logging.info(f"  Dataset {i}: {len(dataset)} samples, weight {weight:.2f}")
        logging.info(f"  Total virtual samples: {len(self._dataset_indices)}")

    def __getitem__(self, index: SupportsIndex) -> dict:
        idx = index.__index__() % len(self._dataset_indices)
        dataset_idx = self._dataset_indices[idx]
        local_idx = self._local_indices[idx]

        sample = self._datasets[dataset_idx][local_idx]

        # 如果存在数据集专属 prompt transform,则在返回前应用.
        if self._prompt_transforms[dataset_idx] is not None:
            sample = self._prompt_transforms[dataset_idx](sample)

        return sample

    def __len__(self) -> int:
        return len(self._dataset_indices)


def create_torch_dataset(
    data_config: _config.DataConfig, action_horizon: int, model_config: _model.BaseModelConfig
) -> Dataset:
    """打开 LeRobot 数据集,并按时间戳请求未来 action chunk."""
    repo_id = data_config.repo_id

    # 检查是否启用多数据集训练.
    if data_config.lerobot_datasets:
        multi_dataset_mode = data_config.multi_dataset_mode
        logging.info(
            f"Creating multi-dataset loader with {len(data_config.lerobot_datasets)} datasets (mode: {multi_dataset_mode})"
        )

        datasets = []
        weights = []
        prompt_transforms = []

        for ds_config in data_config.lerobot_datasets:
            # 创建单个 LeRobot 数据集.
            dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(ds_config.repo_id)
            # LeRobot 根据数据集 FPS,从当前 observation 时刻开始,
            # 取得 action_horizon 个连续动作目标.
            dataset = _create_sequence_dataset(
                ds_config.repo_id,
                dataset_meta=dataset_meta,
                data_config=data_config,
                action_horizon=action_horizon,
            )

            # 按需创建该数据集对应的 prompt transform.
            prompt_transform = None
            if data_config.prompt_from_task:
                prompt_transform = _transforms.PromptFromLeRobotTask(dataset_meta.tasks)

            datasets.append(dataset)
            weights.append(ds_config.weight)
            prompt_transforms.append(prompt_transform)

            logging.info(f"  - {ds_config.repo_id}: weight={ds_config.weight:.2f}, samples={len(dataset)}")

        # 根据模式选择多数据集组合方式.
        if multi_dataset_mode == "concat":
            # concat 模式:使用所有数据集中的全部样本.
            return ConcatLeRobotDataset(datasets, prompt_transforms)
        if multi_dataset_mode == "weighted":
            # weighted 模式:按权重采样,并校验权重和为 1.
            total_weight = sum(weights)
            if not np.isclose(total_weight, 1.0):
                raise ValueError(f"Dataset weights must sum to 1.0 in weighted mode, got {total_weight}")
            return MultiLeRobotDataset(datasets, weights, prompt_transforms)
        raise ValueError(f"Unknown multi_dataset_mode: {multi_dataset_mode}")

    # 单数据集路径.
    if repo_id is None:
        raise ValueError("Repo ID is not set. Cannot create dataset.")
    if repo_id == "fake":
        return FakeDataset(model_config, num_samples=1024)

    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)
    dataset = _create_sequence_dataset(
        data_config.repo_id,
        dataset_meta=dataset_meta,
        data_config=data_config,
        action_horizon=action_horizon,
    )

    if data_config.prompt_from_task:
        dataset = TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)])

    return dataset


def transform_dataset(dataset: Dataset, data_config: _config.DataConfig, *, skip_norm_stats: bool = False) -> Dataset:
    """构建与策略推理一致的样本 transform 顺序."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Run `scripts/compute_norm_stats.py --config-path experiment.yaml` "
                "with the same concise experiment file."
            )
        norm_stats = data_config.norm_stats

    return TransformedDataset(
        dataset,
        [
            # 原始数据集键 -> BrainCo 通用键.
            *data_config.repack_transforms.inputs,
            # 关节布局、相机映射,以及可选的 absolute -> delta 动作转换.
            *data_config.data_transforms.inputs,
            # state/action 归一化必须发生在 padding 和 tokenization 之前.
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            # 模型专属的图像缩放、prompt tokenization 和 shape padding.
            *data_config.model_transforms.inputs,
        ],
    )


def create_data_loader(
    config: _config.TrainConfig,
    *,
    sharding: jax.sharding.Sharding | None = None,
    shuffle: bool = False,
    num_batches: int | None = None,
    skip_norm_stats: bool = False,
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """创建以 LeRobot 为数据源的 JAX 训练 DataLoader."""
    data_config = config.data.create(config.assets_dirs, config.model)
    logging.info(f"data_config: {data_config}")

    return create_torch_data_loader(
        data_config,
        model_config=config.model,
        action_horizon=config.model.action_horizon,
        batch_size=config.batch_size,
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=config.num_workers,
        seed=config.seed,
        skip_norm_stats=skip_norm_stats,
    )


def create_torch_data_loader(
    data_config: _config.DataConfig,
    model_config: _model.BaseModelConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    num_workers: int = 0,
    seed: int = 0,
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """创建训练 DataLoader.

    参数:
        data_config:数据配置.
        action_horizon:动作序列长度.
        batch_size:全局 batch 大小.
        sharding:DataLoader 使用的设备分片;为空时使用默认单设备分片.
        skip_norm_stats:是否跳过数据归一化.
        shuffle:是否打乱数据.
        num_batches:最多返回的 batch 数;超过单轮数据量时会循环数据集,
            未设置时无限迭代.
        num_workers:DataLoader worker 进程数;为 0 时在主进程执行.
        seed:打乱数据使用的随机种子.
    """
    dataset = create_torch_dataset(data_config, action_horizon, model_config)
    dataset = transform_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats)

    local_batch_size = batch_size // jax.process_count()

    logging.info(f"local_batch_size: {local_batch_size}")
    data_loader = TorchDataLoader(
        dataset,
        local_batch_size=local_batch_size,
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=num_workers,
        seed=seed,
    )

    return DataLoaderImpl(data_config, data_loader)


class TorchDataLoader:
    """先在 CPU 上用 PyTorch 组装 batch,再显式放置到 JAX 设备."""

    def __init__(
        self,
        dataset,
        local_batch_size: int,
        *,
        sharding: jax.sharding.Sharding | None = None,
        shuffle: bool = False,
        num_batches: int | None = None,
        num_workers: int = 0,
        seed: int = 0,
    ):
        """创建底层 PyTorch DataLoader.

        参数:
            dataset:待加载的数据集.
            local_batch_size:每个本地进程的 batch 大小.
            sharding:目标 JAX 设备分片.
            shuffle:是否打乱数据.
            num_batches:最多返回的 batch 数;超出单轮数据量时循环数据集,
                未设置时无限迭代.
            num_workers:数据加载 worker 进程数;为 0 时在主进程执行.
            seed:打乱数据使用的随机种子.
        """
        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if len(dataset) < local_batch_size:
            raise ValueError(f"Local batch size ({local_batch_size}) is larger than the dataset size ({len(dataset)}).")

        self._sharding = sharding
        if sharding is None:
            self._sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )
        self._num_batches = num_batches

        mp_context = None
        if num_workers > 0:
            mp_context = multiprocessing.get_context("spawn")

        generator = torch.Generator()
        generator.manual_seed(seed)
        self._data_loader = torch.utils.data.DataLoader(
            typing.cast(torch.utils.data.Dataset, dataset),
            batch_size=local_batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            multiprocessing_context=mp_context,
            persistent_workers=num_workers > 0,
            collate_fn=_collate_fn,
            worker_init_fn=_worker_init_fn,
            drop_last=True,
            generator=generator,
        )

    @property
    def torch_loader(self) -> torch.utils.data.DataLoader:
        return self._data_loader

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._data_loader)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # 当前轮数据已耗尽,重新创建迭代器并继续.
                num_items += 1
                # 这里是 CPU -> JAX device 的边界.batch 会直接按照
                # jitted train_step 期望的分片布局创建.
                yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)


def _collate_fn(items):
    """将多条样本整理为带 batch 维的 NumPy 数组."""
    # stack 前统一转换为 NumPy,避免输入中混有 JAX 数组.
    return jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0), *items)


def _worker_init_fn(worker_id: int) -> None:
    """禁止 DataLoader worker 中的 JAX 预占 GPU 显存."""
    # 注意:worker 中此时已经 import jax,因此这里不能用于选择 JAX backend.
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


class DataLoaderImpl(DataLoader):
    """向模型无关的训练循环提供结构化 Observation 对象."""

    def __init__(self, data_config: _config.DataConfig, data_loader: TorchDataLoader):
        self._data_config = data_config
        self._data_loader = data_loader

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    def __iter__(self):
        for batch in self._data_loader:
            yield _model.Observation.from_dict(batch), batch["actions"]
