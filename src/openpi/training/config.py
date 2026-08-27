"""BrainCo 策略可序列化的训练配置图.

TrainConfig 是 scripts/train.py 使用的根配置对象.内部 factory 会展开模型结构、
数据 transforms、优化器设置和权重加载方式.最终展开的对象也会保存进每个
checkpoint,使部署端无需查询配置注册表即可重建同一策略.
"""

import abc
from collections.abc import Sequence
import dataclasses
import difflib
import importlib
import logging
import pathlib
import sys
from typing import Any, Literal, Protocol, TypeAlias

import etils.epath as epath
import flax.nnx as nnx
from typing_extensions import override
import tyro

import openpi.models.act_config as act_config
import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.models.tokenizer as _tokenizer
import openpi.policies.brainco_policy as brainco_policy
import openpi.shared.download as _download
import openpi.shared.nnx_utils as nnx_utils
import openpi.shared.normalize as _normalize
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms

ModelType: TypeAlias = _model.ModelType
# 规避 tyro 直接处理 nnx.filterlib.Filter 时的兼容问题.
Filter: TypeAlias = nnx.filterlib.Filter
FreezeStrategy: TypeAlias = Literal["none", "lora_and_action_interface", "action_interface_only"]

_TRAINABLE_PARAMETER_REGEX_BY_FREEZE_STRATEGY: dict[FreezeStrategy, str] = {
    "lora_and_action_interface": (
        ".*(lora|action_in_proj|action_out_proj|time_mlp_in|time_mlp_out|state_proj).*"
    ),
    "action_interface_only": (
        ".*(action_in_proj|action_out_proj|time_mlp_in|time_mlp_out|state_proj).*"
    ),
}


@dataclasses.dataclass(frozen=True)
class AssetsConfig:
    """指定构建数据流水线所需 assets(例如 norm stats)的位置.

    这些 assets 会复制到 checkpoint 的 `assets/asset_id` 目录中.

    该机制可从其他 checkpoint(例如基础模型 checkpoint)或统一位置加载
    assets.例如微调时从基础模型 checkpoint 加载 Trossen 机器人的 norm stats:

    ```
    AssetsConfig(
        assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
        asset_id="trossen",
    )
    ```
    """

    # assets 根目录;未设置时使用 TrainConfig.assets_dirs.
    # 可用于从基础模型 checkpoint 或其他统一位置加载 assets.
    assets_dir: str | None = None

    # asset 标识;未设置时使用 repo_id,便于引用不同机器人平台的 assets.
    asset_id: str | None = None


@dataclasses.dataclass(frozen=True)
class LeRobotDataset:
    """多数据集训练中单个 LeRobot 数据集的配置."""

    # LeRobot repo_id 或本地路径.
    repo_id: str
    # 当前数据集的采样权重;所有数据集权重之和必须为 1.
    weight: float
    # 可选:覆盖当前数据集使用的 asset_id.
    asset_id: str | None = None


@dataclasses.dataclass(frozen=True)
class DataConfig:
    # LeRobot repo_id;为空时创建 fake data.
    repo_id: str | None = None
    # 数据 assets 在 assets 根目录下的子目录名.
    asset_id: str | None = None
    # 预计算的归一化统计量;为空时不执行归一化.
    norm_stats: dict[str, _transforms.NormStats] | None = None

    # 将数据集特有的键结构重排为 data transforms 期望的通用结构.
    repack_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # 数据 transforms,通常包含机器人专属变换,在归一化前执行.
    # 归一化后的结构参见 `model.Observation` 和 `model.Actions`.
    data_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # 模型专属 transforms,在数据归一化后执行.
    model_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # 为 true 时使用分位数归一化,否则使用标准 z-score 归一化.
    use_quantile_norm: bool = False

    # DataLoader 用于生成动作序列的原始键名.序列长度由模型配置中的
    # action_horizon 决定;如果 LeRobot 数据集使用其他动作键,需要修改此项.
    action_sequence_keys: Sequence[str] = ("actions",)

    # 为 true 时,使用 LeRobot 数据集 task 生成 prompt.
    prompt_from_task: bool = False

    # 多数据集联合训练使用的 LeRobot 数据集列表.
    lerobot_datasets: Sequence[LeRobotDataset] = ()
    # 多数据集模式:
    # - concat:拼接全部数据集,使用每一条样本.
    # - weighted:按权重采样,部分样本可能重复或被跳过.
    multi_dataset_mode: Literal["concat", "weighted"] = "concat"

class GroupFactory(Protocol):
    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        """创建 transform 组."""


@dataclasses.dataclass(frozen=True)
class ModelTransformFactory(GroupFactory):
    """根据 ModelType 创建最终面向模型的 transforms."""

    # 设置后作为模型使用的默认 prompt.
    default_prompt: str | None = None

    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        match model_config.model_type:
            case _model.ModelType.PI0:
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI05:
                assert isinstance(model_config, pi0_config.Pi0Config)
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                            discrete_state_input=model_config.discrete_state_input,
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.ACT:
                # ACT 不包含语言/VLM 模块,因此不执行 prompt tokenization.
                # 它只需要缩放图像,并将 state/actions padding 到 action_dim.
                # 在放入 JAX 设备前丢弃 LeRobot task metadata 携带的 prompt.
                return _transforms.Group(
                    inputs=[
                        _transforms.ResizeImages(224, 224),
                        _transforms.DropPrompt(),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )


@dataclasses.dataclass(frozen=True)
class DataConfigFactory(abc.ABC):
    # LeRobot repo_id.
    repo_id: str = tyro.MISSING
    # 指定 assets 的加载方式.
    assets: AssetsConfig = dataclasses.field(default_factory=AssetsConfig)
    # factory 展开时使用的基础 DataConfig.
    base_config: tyro.conf.Suppress[DataConfig | None] = None

    @abc.abstractmethod
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        """创建完整的数据配置."""

    def create_base_config(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # 在此解析归一化配置,使训练和策略创建从同一 DataConfig
        # 获得相同的统计量与归一化模式.
        repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None
        asset_id = self.assets.asset_id or repo_id
        return dataclasses.replace(
            self.base_config or DataConfig(),
            repo_id=repo_id,
            asset_id=asset_id,
            norm_stats=self._load_norm_stats(epath.Path(self.assets.assets_dir or assets_dirs), asset_id),
            use_quantile_norm=model_config.model_type != ModelType.PI0,
        )

    def _load_norm_stats(self, assets_dir: epath.Path, asset_id: str | None) -> dict[str, _transforms.NormStats] | None:
        if asset_id is None:
            return None
        try:
            data_assets_dir = str(assets_dir / asset_id)
            norm_stats = _normalize.load(_download.maybe_download(data_assets_dir))
            logging.info(f"Loaded norm stats from {data_assets_dir}")
            return norm_stats
        except FileNotFoundError:
            logging.info(f"Norm stats not found in {data_assets_dir}, skipping.")
        return None


@dataclasses.dataclass(frozen=True)
class FakeDataConfig(DataConfigFactory):
    repo_id: str = "fake"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return DataConfig(repo_id=self.repo_id)


@dataclasses.dataclass(frozen=True)
class LeRobotBrainCoDataConfig(DataConfigFactory):
    """BrainCo 机器人数据集配置.

    state/action 的语义布局由 ``policy_io`` 描述.既支持完整的 56D 双臂配置,
    也支持 28D 右臂加右手等裁剪布局.

    常见数据集字段:
    - observation.state: joint state vector described by ``policy_io``
    - action: action vector described by ``policy_io``
    - observation.images.cam_left_wrist: (480, 640, 3)
    - observation.images.cam_right_wrist: (480, 640, 3)
    - observation.images.stereo_right or observation.images.cam_head

    action groups 必须完整覆盖 ``model.action_dim``.当该维度与预训练
    checkpoint 不同时,输入输出投影层可能需要重新初始化.
    """

    extra_delta_transform: bool = False
    policy_io: brainco_policy.BrainCoPolicyIOConfig = dataclasses.field(
        default_factory=brainco_policy.BrainCoPolicyIOConfig
    )
    arm_dof: int = 7
    hand_dof: int = 21
    head_camera_key: str = "observation.images.stereo_right"
    revo3_eef_joint_hand_to_joint_hand: bool = False
    # 原始数据集中的 action 键名;LeRobot loader 会在 repack transform 前使用.
    action_sequence_keys: Sequence[str] = ("action",)
    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        """将 BrainCo 数据集语义展开为有序 transform 流水线."""
        self.policy_io.validate(model_config.action_dim)
        # 将数据集原始键映射为 transforms 期望的键.
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": self.head_camera_key,
                        "observation/left_wrist_image": "observation.images.cam_left_wrist",
                        "observation/right_wrist_image": "observation.images.cam_right_wrist",
                        "observation/state": "observation.state",
                        "actions": "action",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        input_transforms = []
        if self.policy_io.dataset_state_indices is not None:
            input_transforms.append(
                brainco_policy.SelectPolicyFeatures(
                    state_indices=self.policy_io.dataset_state_indices,
                    action_indices=(
                        self.policy_io.dataset_action_indices
                        if self.policy_io.dataset_action_indices is not None
                        else self.policy_io.dataset_state_indices
                    ),
                )
            )
        elif self.revo3_eef_joint_hand_to_joint_hand:
            input_transforms.append(
                brainco_policy.BrainCoRevo3EefJointHandToJointHand(
                    arm_dof=self.arm_dof,
                    hand_dof=self.hand_dof,
                )
            )
        input_transforms.append(brainco_policy.BrainCoInputs(model_type=model_config.model_type))

        data_transforms = _transforms.Group(
            inputs=input_transforms,
            outputs=[brainco_policy.BrainCoOutputs(action_dim=model_config.action_dim)],
        )

        # 按语义 group 在训练输入侧执行 delta 转换;
        # 输出 transform 始终将模型 delta 恢复为 absolute 关节目标.
        if self.extra_delta_transform:
            delta_action_mask = _transforms.make_bool_mask(*self.policy_io.delta_mask_dims())

            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        # PI0.5 对 prompt/state 做 tokenization;ACT 丢弃语言并保留连续 state.
        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    """训练和部署流水线使用的完整、可序列化输入配置."""

    # 配置名称,必须唯一,用于引用该配置.
    name: tyro.conf.Suppress[str]
    # 项目名称.
    project_name: str = "openpi"
    # 实验名称,用于命名 metadata 和 checkpoint 目录.
    exp_name: str = tyro.MISSING

    # 模型配置.action_dim、action_horizon、max_token_len 等字段由所有模型共享,
    # 参见 BaseModelConfig;具体模型配置(如 Pi0Config)可增加额外字段.
    model: _model.BaseModelConfig = dataclasses.field(default_factory=pi0_config.Pi0Config)

    # 模型初始化后,可通过 weight loader 从磁盘加载完整或部分权重.
    weight_loader: weight_loaders.WeightLoader = dataclasses.field(default_factory=weight_loaders.NoOpWeightLoader)

    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(default_factory=_optimizer.CosineDecaySchedule)
    optimizer: _optimizer.OptimizerConfig = dataclasses.field(default_factory=_optimizer.AdamW)
    ema_decay: float | None = 0.99

    # 指定需要冻结的参数.
    freeze_filter: tyro.conf.Suppress[Filter] = dataclasses.field(default_factory=nnx.Nothing)
    # 可序列化的 BrainCo 微调策略;与 freeze_filter 不同,该字段可以写入并恢复自 train_config.yaml.
    freeze_strategy: FreezeStrategy = "none"

    # 指定训练数据及其处理方式.
    data: DataConfigFactory = dataclasses.field(default_factory=FakeDataConfig)

    # 配置 assets 的根目录,例如 norm stats.
    assets_base_dir: str = "./assets"
    # checkpoint 根目录.
    checkpoint_base_dir: str = "./checkpoints"

    # 训练随机数生成器使用的 seed.
    seed: int = 42
    # 全局 batch 大小.
    batch_size: int = 32
    # DataLoader worker 数量.增加该值可提高数据加载速度,
    # 同时也会增加内存和 CPU 占用.
    num_workers: int = 2
    # 训练总步数,每一步对应一个 batch.
    num_train_steps: int = 30_000

    # 每隔多少步记录一次训练指标.
    log_interval: int = 100
    # 每隔多少步保存一次 checkpoint.
    save_interval: int = 1000
    # 设置后,满足 step % keep_period == 0 的 checkpoint 会长期保留.
    keep_period: int | None = 5000
    # 即使最终步不落在 save_interval 上,也保存最终 checkpoint.
    save_final_checkpoint: bool = True

    # 为 true 时覆盖已存在的 checkpoint 目录.
    overwrite: bool = False
    # 为 true 时从最新 checkpoint 恢复训练.
    resume: bool = False

    # 为 true 时启用实验指标记录;当前实现使用 SwanLab.
    wandb_enabled: bool = True

    # 传递给策略服务的附加 metadata.
    policy_metadata: dict[str, Any] | None = None

    # 大于 1 时启用 FSDP,并在指定数量的设备间切分模型,可降低单卡显存,
    # 但训练可能变慢.例如共有 4 张卡且 fsdp_devices=2 时,
    # 每 2 张卡组成一个模型分片组,两个组之间执行数据并行.
    fsdp_devices: int = 1

    @property
    def assets_dirs(self) -> pathlib.Path:
        """返回当前配置对应的 assets 目录."""
        return (pathlib.Path(self.assets_base_dir) / self.name).resolve()

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        """返回当前配置对应的 checkpoint 目录."""
        if not self.exp_name:
            raise ValueError("--exp_name must be set")
        return (pathlib.Path(self.checkpoint_base_dir) / self.name / self.exp_name).resolve()

    @property
    def trainable_filter(self) -> nnx.filterlib.Filter:
        """选择允许自动求导和 Optax 更新的参数."""
        return nnx.All(nnx.Param, nnx.Not(self.effective_freeze_filter))

    @property
    def effective_freeze_filter(self) -> nnx.filterlib.Filter:
        """将可序列化冻结策略解析为训练使用的 NNX filter."""
        if self.freeze_strategy == "none":
            return self.freeze_filter
        trainable_regex = _TRAINABLE_PARAMETER_REGEX_BY_FREEZE_STRATEGY[self.freeze_strategy]
        return nnx.Not(nnx_utils.PathRegex(trainable_regex))

    def __post_init__(self) -> None:
        if self.resume and self.overwrite:
            raise ValueError("Cannot resume and overwrite at the same time.")
        if self.freeze_strategy != "none" and not isinstance(self.freeze_filter, nnx.Nothing):
            raise ValueError("freeze_strategy and the legacy freeze_filter cannot both be set")
        if self.freeze_strategy == "lora_and_action_interface":
            if not isinstance(self.model, pi0_config.Pi0Config):
                raise ValueError("lora_and_action_interface requires a Pi0Config model")
            if "lora" not in self.model.paligemma_variant and "lora" not in self.model.action_expert_variant:
                raise ValueError("lora_and_action_interface requires at least one LoRA model variant")
        if (
            self.freeze_strategy == "action_interface_only"
            and isinstance(self.model, pi0_config.Pi0Config)
            and ("lora" in self.model.paligemma_variant or "lora" in self.model.action_expert_variant)
        ):
            raise ValueError("action_interface_only requires non-LoRA model variants")


# 在代码中需要按名称获取配置时,请使用 `get_config`.
_CONFIGS = [
    #
    # 调试配置.
    #
    TrainConfig(
        name="debug",
        data=FakeDataConfig(),
        batch_size=2,
        model=pi0_config.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy"),
        save_interval=100,
        overwrite=True,
        exp_name="debug",
        num_train_steps=10,
        wandb_enabled=False,
    ),
    TrainConfig(
        name="debug_restore",
        data=FakeDataConfig(),
        batch_size=2,
        model=pi0_config.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy"),
        weight_loader=weight_loaders.CheckpointWeightLoader("./checkpoints/debug/debug/9/params"),
        overwrite=True,
        exp_name="debug",
        num_train_steps=10,
        wandb_enabled=False,
    ),
    TrainConfig(
        name="debug_pi05",
        model=pi0_config.Pi0Config(pi05=True, paligemma_variant="dummy", action_expert_variant="dummy"),
        data=FakeDataConfig(),
        batch_size=2,
        num_train_steps=10,
        overwrite=True,
        exp_name="debug_pi05",
        wandb_enabled=False,
    ),
    TrainConfig(
        name="debug_act",
        model=act_config.ACTConfig(
            action_dim=24,
            action_horizon=10,
            hidden_dim=64,
            dim_feedforward=128,
            num_encoder_layers=1,
            num_decoder_layers=1,
            num_cvae_layers=1,
            num_heads=2,
            latent_dim=8,
        ),
        data=FakeDataConfig(),
        batch_size=2,
        num_train_steps=4,
        ema_decay=None,
        overwrite=True,
        exp_name="debug_act",
        wandb_enabled=False,
    ),
]

if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("Config names must be unique.")
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def cli() -> TrainConfig:
    if len(sys.argv) == 2 and pathlib.Path(sys.argv[1]).suffix in (".yaml", ".yml"):
        config_io = importlib.import_module("openpi.training.config_io")
        return config_io.load_train_config(sys.argv[1])
    return tyro.extras.overridable_config_cli({k: (k, v) for k, v in _CONFIGS_DICT.items()})


def get_config(config_name: str) -> TrainConfig:
    """按名称获取配置."""
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0)
        closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
        raise ValueError(f"Config '{config_name}' not found.{closest_str}")

    return _CONFIGS_DICT[config_name]
