"""BrainCo PI0/PI0.5 与 ACT 策略的 JAX 训练入口.

主调用链:
  config.cli -> main -> data_loader.create_data_loader
             -> init_train_state -> jax.jit(train_step)
             -> model.compute_loss -> optimizer update
             -> checkpoints.save_state

模型差异封装在 BaseModel.compute_loss 之后,因此这套训练循环不需要为
PI0.5 和 ACT 分别编写分支.
"""

import dataclasses
import functools
import logging
import os
import pathlib
import platform
from typing import Any
import warnings

warnings.filterwarnings(
    "ignore",
    message="The pynvml package is deprecated.*",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    message="The video decoding and encoding capabilities of torchvision are deprecated.*",
    category=UserWarning,
)

import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import flax.traverse_util as traverse_util
import jax
import jax.experimental
import jax.numpy as jnp
import numpy as np
import optax
import swanlab
import tqdm_loggable.auto as tqdm

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.config_io as _config_io
import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders


def init_logging():
    """设置便于阅读的日志格式."""
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(formatter)


def init_swanlab(config: _config.TrainConfig, *, resuming: bool, enabled: bool = True):
    """创建或恢复与当前 checkpoint 目录关联的 SwanLab 实验."""
    if not enabled:
        swanlab.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")

    init_kwargs = {
        "project": config.project_name,
        "workspace": os.environ.get("SWANLAB_WORKSPACE", "YinJiaju"),
        "experiment_name": config.exp_name,
        "name": config.exp_name,
        "mode": os.environ.get("SWANLAB_MODE", "offline"),
        "config": dataclasses.asdict(config),
    }
    if api_key_file := os.environ.get("SWANLAB_API_KEY_FILE"):
        api_key = pathlib.Path(api_key_file).expanduser().read_text().strip()
        if not api_key:
            raise ValueError(f"SWANLAB_API_KEY_FILE is empty: {api_key_file}")
        swanlab.login(api_key=api_key, relogin=True, save=False)
        init_kwargs["settings"] = swanlab.Settings(api_key=api_key, interactive=False)
    if resuming:
        run_id_path = ckpt_dir / "swanlab_id.txt"
        if run_id_path.exists():
            init_kwargs["id"] = run_id_path.read_text().strip()
            init_kwargs["resume"] = "must"

    run = swanlab.init(**init_kwargs)
    run_id = getattr(run, "id", None)
    if run_id and not resuming:
        (ckpt_dir / "swanlab_id.txt").write_text(str(run_id))


def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    """加载并校验权重,返回成功加载的参数子集."""
    loaded_params = loader.load(params_shape)
    at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)

    # 移除未实际加载的 jax.ShapeDtypeStruct,确保这里只返回真实权重.
    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )


@at.typecheck
def init_train_state(
    config: _config.TrainConfig, init_rng: at.KeyArrayLike, mesh: jax.sharding.Mesh, *, resume: bool
) -> tuple[training_utils.TrainState, Any]:
    """构建模型、优化器状态、初始权重和设备分片布局."""
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        # ModelConfig.create 是选择 PI0.5 或 ACT 具体网络的边界.
        model = config.model.create(model_rng)

        # 将预训练权重子集写入刚创建的模型.
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            # 如果待加载参数不是目标模型参数树的子集,这里会直接报错.
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        # 冻结参数不参与优化器更新,转成 bfloat16 可以减少设备显存占用.
        params = nnx_utils.state_map(
            params, config.effective_freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16))
        )

        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    # 在不分配完整模型数组的情况下取得参数树形状,据此生成 FSDP 布局,
    # 并用目标形状校验预训练权重.
    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        return train_state_shape, state_sharding

    partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # 编译初始化过程,让参数直接按最终分片方式创建到设备上,
    # 避免先在主机侧构造一份完整副本.
    train_state = jax.jit(
        init,
        donate_argnums=(1,),  # 允许 JAX 复用预训练参数缓冲区.
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding


@at.typecheck
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    """执行一次可编译的前向、反向和参数更新."""
    # NNX 在 TrainState 中分别保存网络结构和参数状态;
    # 这里重新合并,得到 compute_loss 所需的完整模型对象.
    model = nnx.merge(state.model_def, state.params)
    model.train()

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions
    ):
        chunked_loss = model.compute_loss(rng, observation, actions, train=True)
        return jnp.mean(chunked_loss)

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch

    # 只有 TrainConfig 选中的参数参与自动求导.
    # 全参数训练、LoRA 和仅训练动作接口的差异在这里生效.
    diff_state = nnx.DiffState(0, config.trainable_filter)
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model, train_rng, observation, actions)

    # Optax 将梯度转换为参数更新:梯度裁剪 -> AdamW -> 学习率缩放.
    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    # 将更新后的可训练参数写回模型,再生成新的完整参数状态.
    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
            ),
        )

    # 仅保留矩阵类 kernel 参数,用于统计参数范数.
    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
    }
    return new_state, info


def main(config: _config.TrainConfig):
    """组装训练子系统并运行编译后的训练循环."""
    init_logging()
    logging.info(f"Running on: {platform.node()}")

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )

    # 当代码和静态 shape 一致时,跨进程启动复用已编译的 XLA 程序.
    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    # JAX 显式管理随机数:模型初始化和逐步训练使用从根 seed
    # 派生出的独立随机 key.
    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    # 逻辑 mesh 同时表达 batch 数据并行和可选的 FSDP 模型分片;
    # 数据 batch 与 TrainState 分别使用各自的分片规则.
    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # 在监控和模型初始化前先处理 checkpoint 目录,
    # 让 overwrite/resume 配置错误在昂贵的 GPU 初始化前暴露.
    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    if jax.process_index() == 0:
        _config_io.save_train_config(config, config.checkpoint_dir)
    init_swanlab(config, resuming=resuming, enabled=config.wandb_enabled)

    # DataLoader 在 CPU 上通过 LeRobot/PyTorch 解码并执行配置好的 transforms,
    # 随后按照 data_sharding 将 batch 创建为设备上的 JAX 数组.
    data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        shuffle=True,
    )
    data_iter = iter(data_loader)
    batch = next(data_iter)
    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")

    # 记录首个 batch 的图像,快速检查相机顺序和预处理结果.
    try:
        images_to_log = [
            swanlab.Image(np.concatenate([np.array(img[i]) for img in batch[0].images.values()], axis=1))
            for i in range(min(5, len(next(iter(batch[0].images.values())))))
        ]
        swanlab.log({"camera_views": images_to_log}, step=0)
    except Exception as e:
        logging.warning(f"Failed to log images to SwanLab: {e}")

    # 恢复训练时先构建预期的抽象 TrainState 结构,
    # 再由 Orbax 将参数和优化器的具体数值恢复进去.
    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")

    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)

    # 将完整的前向、反向和优化器更新编译为一个 XLA 程序.
    # donate_argnums 允许 JAX 原地复用旧 TrainState 的缓冲区.
    ptrain_step = jax.jit(
        functools.partial(train_step, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )

    start_step = int(train_state.step)
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )

    infos = []
    for step in pbar:
        # mesh 上下文让深层网络可以添加激活值分片约束,
        # 无需在每一层函数调用中显式传递 mesh.
        with sharding.set_mesh(mesh):
            train_state, info = ptrain_step(train_rng, train_state, batch)
        infos.append(info)
        if step % config.log_interval == 0:
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
            info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
            pbar.write(f"Step {step}: {info_str}")
            swanlab.log(reduced_info, step=step)
            infos = []
        batch = next(data_iter)

        should_save_interval = step % config.save_interval == 0 and step > start_step
        should_save_final = config.save_final_checkpoint and step == config.num_train_steps - 1
        if should_save_interval or should_save_final:
            # params/ 面向部署;train_state/ 还包含恢复训练所需的
            # 优化器状态和训练步数.
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)
            checkpoint_manager.wait_until_finished()
            if jax.process_index() == 0:
                _config_io.save_train_config(config, config.checkpoint_dir / str(step))

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()
    swanlab.finish()


if __name__ == "__main__":
    main(_config.cli())
