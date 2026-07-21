# 05. TrainState、JIT 与 checkpoint

DataLoader 准备好 batch 后，训练还需要一个保存“模型当前状态”的对象，这就是 `TrainState`。

## 1. TrainState 包含什么

[`TrainState`](../../src/openpi/training/utils.py) 可以概括为：

```text
TrainState
  ├─ step          当前训练步数
  ├─ params        模型参数
  ├─ model_def     模型结构定义
  ├─ tx            Optax 优化器
  ├─ opt_state     AdamW 动量等优化器状态
  └─ ema_params    可选的参数滑动平均
```

它不只是模型权重。要完整恢复训练，step 和优化器状态同样重要。

## 2. `init_train_state()` 的初始化链

[`init_train_state()`](../../scripts/train.py) 依次执行：

```text
创建 Optax optimizer
  -> config.model.create() 创建 PI0.5/ACT 网络
  -> 可选加载预训练参数子集
  -> 根据 freeze strategy 转换冻结参数 dtype
  -> 创建 optimizer state
  -> 组装 TrainState
```

在真正创建全部参数前，代码先执行：

```python
train_state_shape = jax.eval_shape(init, init_rng)
```

它只推导参数树的 shape 和 dtype，主要用于：

- 生成 FSDP sharding。
- 检查预训练参数能否装入目标模型。

之后再通过 JIT 让参数直接按最终 sharding 创建在设备上。

## 3. 为什么要 `jax.block_until_ready()`

```python
jax.block_until_ready(train_state)
```

JAX 的设备计算通常异步提交。这里显式等待模型初始化真正完成，保证之后的日志和恢复操作不会只看到一个尚未执行完的任务。

## 4. resume 时为什么先构造结构

```python
if resuming:
    train_state = restore_state(...)
```

Orbax 恢复参数时需要知道目标 PyTree 的结构、shape、dtype 和 sharding。因此代码先用配置构造“预期 TrainState”，再把 checkpoint 中的具体数值填进去。

恢复后通常包括：

- 已训练参数
- 当前 step
- 优化器状态
- 可选 EMA 参数

## 5. `jax.jit(train_step)` 编译什么

```python
ptrain_step = jax.jit(
    functools.partial(train_step, config),
    in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
    out_shardings=(train_state_sharding, replicated_sharding),
    donate_argnums=(1,),
)
```

被编译的是一次完整参数更新：

```text
模型前向
  -> compute_loss
  -> 自动求导
  -> 梯度裁剪
  -> AdamW 更新
  -> 写回参数
  -> 可选 EMA 更新
```

三个输入依次是：

| 输入 | sharding |
| --- | --- |
| 随机 key | replicated |
| TrainState | 参数专属 sharding |
| batch | 沿数据轴切分 |

`donate_argnums=(1,)` 表示旧 TrainState 的设备缓冲区可以被新 TrainState 复用，以减少显存峰值和拷贝。

## 6. 一次 `train_step()` 做什么

```text
TrainState.model_def + params
  -> 合并成可调用模型
  -> observation, actions = batch
  -> model.compute_loss(...)
  -> nnx.value_and_grad(...)
  -> 只对 trainable_filter 选中的参数求梯度
  -> Optax 生成 updates
  -> 写回参数并 step + 1
```

`freeze_strategy` 在这里真正生效：冻结参数仍参与前向计算，但不产生用于更新的梯度。

## 7. checkpoint 中保存什么

[`checkpoints.save_state()`](../../src/openpi/training/checkpoints.py) 每个保存步写入：

```text
<step>/
  ├─ train_state/   恢复训练所需的状态
  ├─ params/        部署推理所需的模型参数
  ├─ assets/        norm stats
  └─ train_config.yaml
```

- 部署通常只需要 `params`、assets 和配置。
- 继续训练需要 `train_state` 中的优化器状态和 step。

到这里，工程训练链已经闭环：batch 进入模型，参数被更新，结果被 checkpoint 保存。
