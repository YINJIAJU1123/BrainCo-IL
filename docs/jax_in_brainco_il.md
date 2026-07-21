# JAX 在 BrainCo-IL 中的作用

## 1. JAX 是什么

JAX 是面向高性能数值计算的 Python 框架。它提供与 NumPy 相似的数组
接口，同时增加：

- 自动求导
- JIT 编译
- GPU/TPU 执行
- 显式随机数管理
- 多设备数据和参数切分
- 面向函数式程序的计算图优化

最简单的对比：

```python
import numpy as np

x = np.array([1.0, 2.0])
y = x**2
```

JAX 的写法非常接近：

```python
import jax.numpy as jnp

x = jnp.array([1.0, 2.0])
y = x**2
```

区别是 JAX 可以把这一组计算编译后放到 GPU 上执行，还可以自动计算
`y` 对 `x` 的梯度。

## 2. BrainCo-IL 中各框架的分工

BrainCo-IL 当前不是纯 PyTorch 项目，也不是所有数据操作都在 JAX 中
完成。实际分工是：

```text
LeRobot
  -> PyTorch DataLoader
  -> NumPy batch
  -> JAX Array / device sharding
  -> Flax NNX model
  -> Optax optimizer
  -> Orbax checkpoint
```

对应职责：

| 组件 | 作用 |
| --- | --- |
| LeRobot | 读取 parquet、视频、任务信息和时间序列 action |
| PyTorch DataLoader | 多进程取样、shuffle、batch 和 collate |
| NumPy | DataLoader worker 中的中间数组格式 |
| JAX | GPU 计算、自动求导、JIT、多卡和随机数 |
| Flax NNX | 定义 PI0.5、ACT 网络及参数树 |
| Optax | AdamW、gradient clipping 和学习率 schedule |
| Orbax | 保存和恢复 JAX checkpoint |

因此仓库中仍然依赖 `torch`，不代表模型使用 PyTorch runtime。Torch
只负责训练数据加载，模型训练和部署推理都使用 JAX。

## 3. 训练调用链

训练入口是：

```text
scripts/train.py
```

完整主链为：

```text
config.cli()
  -> TrainConfig
  -> main(config)
  -> sharding.make_mesh()
  -> data_loader.create_data_loader()
  -> init_train_state()
  -> jax.jit(train_step)
  -> model.compute_loss()
  -> nnx.value_and_grad()
  -> Optax update
  -> checkpoints.save_state()
```

配置、数据、模型和参数初始化是四条相对独立的支线，最终在
`train_step()` 汇合。

## 4. `jax.numpy`

代码中通常写为：

```python
import jax.numpy as jnp
```

`jnp` 提供与 NumPy 类似的 API：

```python
jnp.mean(...)
jnp.square(...)
jnp.concatenate(...)
jnp.cumsum(...)
```

PI0.5 的 flow matching loss 位于：

```text
src/openpi/models/pi0.py
Pi0.compute_loss()
```

核心计算是：

```python
return jnp.mean(jnp.square(v_t - u_t), axis=-1)
```

其中：

- `v_t` 是网络预测的 flow velocity。
- `u_t` 是根据真实 action 和随机 noise 构造的训练目标。
- JAX 在 GPU 上计算每个 action step 的均方误差。

## 5. 自动求导

训练反向传播的核心位于：

```text
scripts/train.py
train_step()
```

```python
loss, grads = nnx.value_and_grad(
    loss_fn,
    argnums=diff_state,
)(model, train_rng, observation, actions)
```

它完成：

1. 调用 `model.compute_loss()` 前向计算。
2. 构建 loss 到可训练参数的计算关系。
3. 自动反向传播。
4. 返回标量 `loss` 和参数梯度树 `grads`。

`diff_state` 决定哪些参数参与求导：

```python
diff_state = nnx.DiffState(0, config.trainable_filter)
```

因此全参数训练、LoRA 和仅训练 action interface 的区别，最终会反映在
这里传入的参数 filter 上。冻结参数不会产生梯度，也不会交给优化器更新。

## 6. JIT 编译

JIT 是 Just-In-Time compilation，即即时编译：

```python
ptrain_step = jax.jit(train_step, ...)
```

JAX 会将训练 step 中的一系列 Python/JAX 运算整理成一个 XLA 计算图：

```text
forward
  -> loss
  -> backward
  -> gradient clipping
  -> AdamW
  -> parameter update
  -> EMA update
```

第一次遇到某组输入 shape 时需要编译，因此首次训练 step 或首次推理
通常明显较慢。相同 shape 的后续调用会复用已编译程序。

这也是 JAX 训练偏好固定 shape 的原因：

- batch size 固定
- action dimension 固定
- action horizon 固定
- 图像分辨率固定
- token 最大长度固定

如果这些 shape 在运行中变化，JAX 可能重新编译。

BrainCo-IL 将编译缓存写到：

```text
~/.cache/jax
```

相同程序以后启动时可以减少部分编译成本。

## 7. `jax.eval_shape`

模型初始化前，代码会先运行：

```python
train_state_shape = jax.eval_shape(init, init_rng)
```

`eval_shape` 不真正创建全部 GPU 参数，而是计算参数树中每个叶子的：

- key path
- shape
- dtype

这份抽象参数树有两个用途：

1. `sharding.fsdp_sharding()` 根据 shape 决定参数如何分布到 GPU。
2. `weight_loader` 根据目标 shape 校验预训练参数能否加载。

之后才通过 `jax.jit(init)` 真正创建并放置训练状态。

## 8. 随机数 key

JAX 不推荐依赖全局随机状态，而是显式传递随机 key：

```python
rng = jax.random.key(config.seed)
train_rng, init_rng = jax.random.split(rng)
```

模型初始化使用 `init_rng`，训练使用 `train_rng`。

每一步训练又执行：

```python
train_rng = jax.random.fold_in(rng, state.step)
```

这样每个 step 的随机数不同，但给定相同 seed 和 step 时仍然可以复现。

PI0.5 中继续拆分 key：

```python
preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
```

分别用于：

- 图像 augmentation
- flow matching noise
- flow matching timestep

## 9. PyTree

JAX 使用 PyTree 表示嵌套参数和数据，例如：

```text
TrainState
  params
    PaliGemma
      llm
      img
    action_in_proj
    action_out_proj
  opt_state
  ema_params
```

PyTree 可以是 dataclass、dict、tuple 和 list 的组合，只要叶子是数组或
JAX 能处理的对象。

项目中常见：

```python
jax.tree.map(...)
jax.tree_util.tree_map_with_path(...)
```

这些操作会遍历完整参数树，而不用手写每一层的参数名称。

## 10. 数据何时进入 GPU

`data_loader.py` 先通过 PyTorch DataLoader 在 CPU 上得到 NumPy batch。

真正进入 JAX device 的位置是：

```python
jax.make_array_from_process_local_data(self._sharding, batch)
```

流程是：

```text
LeRobot sample
  -> BrainCo transforms
  -> normalization
  -> resize/tokenize/pad
  -> NumPy batch
  -> JAX Array
  -> GPU sharding
```

这样做可以避免 DataLoader worker 提前初始化 JAX GPU runtime，也能让
视频解码和 CPU 数据处理与 GPU 训练并行。

## 11. 多 GPU 和 sharding

多卡入口位于：

```text
src/openpi/training/sharding.py
```

设备 mesh 使用两个逻辑轴：

```text
batch
fsdp
```

`batch` 轴用于数据并行，`fsdp` 轴用于切分较大的参数。

例如四张 GPU，`fsdp_devices=2` 时：

```text
mesh shape = (2, 2)

第一维：两个 data-parallel group
第二维：每个 group 内两张卡切分模型参数
```

主要 sharding：

- `data_sharding`：batch 沿数据轴切分。
- `replicated_sharding`：随机 key、日志等小对象在设备间复制。
- `train_state_sharding`：大参数按 FSDP 规则切分，小参数复制。

`fsdp_sharding()` 默认只考虑超过 4 MiB 的矩阵或高维 tensor。无法均匀
切分或体积较小的参数会复制到每张设备。

## 12. 参数更新

得到梯度后：

```python
updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
new_params = optax.apply_updates(params, updates)
```

`state.tx` 是 Optax optimizer transformation。当前 AdamW 配置的实际顺序：

```text
gradient
  -> global norm clipping
  -> Adam moments
  -> weight decay
  -> learning-rate scaling
  -> parameter update
```

配置位置：

```text
src/openpi/training/optimizer.py
```

默认 gradient clipping 为：

```yaml
clip_gradient_norm: 1.0
```

## 13. EMA

如果 `ema_decay` 不为 `None`，每一步还会更新指数滑动平均参数：

```python
ema = decay * old_ema + (1 - decay) * new_params
```

EMA 不参与梯度计算，它是训练参数的平滑副本。保存部署参数时，如果存在
EMA，checkpoint 会优先把 EMA 写入独立的 `params/`。

BrainCo 当前常用 PI0.5 和 ACT 配置将 `ema_decay` 设置为 `None`，因此
直接部署当前模型参数。

## 14. checkpoint

checkpoint 由 Orbax 管理：

```text
src/openpi/training/checkpoints.py
```

每个 step 保存：

```text
<step>/
  train_state/
  params/
  assets/
  train_config.yaml
```

含义：

- `train_state/`：恢复训练所需的 step、参数、optimizer state 和 EMA。
- `params/`：部署推理需要的纯模型参数。
- `assets/`：训练使用的归一化统计。
- `train_config.yaml`：重建模型、transform 和输入输出语义。

部署不需要 optimizer state，所以只读取 `params/`。

## 15. PI0.5 去噪推理

PI0.5 推理从随机 action noise 开始：

```python
noise = jax.random.normal(
    rng,
    (batch_size, action_horizon, action_dim),
)
```

随后使用：

```python
jax.lax.while_loop(...)
```

重复执行 action expert，根据网络预测的 flow velocity 将 noise 更新为
action chunk。

这里使用 `jax.lax.while_loop` 而不是普通 Python `while`，因为整个去噪
循环需要进入 JIT/XLA 计算图，在 GPU 上执行。

`num_inference_steps` 决定迭代次数：

- 步数增加：通常更接近完整数值积分，但推理更慢。
- 步数减少：推理更快，但 action 质量可能下降。

## 16. ACT 中的 JAX

ACT 同样实现统一接口：

```python
compute_loss(...)
sample_actions(...)
```

所以 `train.py` 不需要知道当前模型是 ACT 还是 PI0.5。

区别只发生在模型内部：

- PI0.5 训练 flow matching，推理执行多步去噪。
- ACT 训练 CVAE + chunk reconstruction，推理一次前向输出 action chunk。

JAX 对两种模型提供相同的：

- 参数树
- 自动求导
- JIT
- sharding
- optimizer update
- checkpoint

## 17. JAX 和 PyTorch 的主要区别

| 方面 | JAX | PyTorch |
| --- | --- | --- |
| 数组 | `jax.Array` | `torch.Tensor` |
| 自动求导 | `jax.grad` / `value_and_grad` | `loss.backward()` |
| 编译 | `jax.jit` 是核心使用方式 | `torch.compile` 通常可选 |
| 随机数 | 显式传递 key | 通常使用全局随机状态 |
| 参数更新 | 偏函数式，生成新状态 | 常见原地更新 |
| 多卡 | mesh、NamedSharding、FSDP axis | DDP、FSDP |
| 网络框架 | Flax NNX 等 | `torch.nn` |

JAX 更强调纯函数、固定 shape 和显式状态，这让编译器更容易优化整个训练
step，但也意味着动态 Python 控制流和运行中 shape 变化需要更加谨慎。

## 18. 阅读代码的推荐顺序

建议按以下顺序阅读：

1. `scripts/train.py`
2. `src/openpi/training/config.py`
3. `src/openpi/training/data_loader.py`
4. `src/openpi/transforms.py`
5. `src/openpi/policies/brainco_policy.py`
6. `src/openpi/models/model.py`
7. `src/openpi/models/pi0.py` 或 `src/openpi/models/act.py`
8. `src/openpi/training/optimizer.py`
9. `src/openpi/training/sharding.py`
10. `src/openpi/training/checkpoints.py`

对应的完整知识链见：

```text
docs/brainco_il_beginner/README.md
```
