# 01. 训练入口与 JAX

训练入口是 [`scripts/train.py`](../../scripts/train.py)。文件末尾只有一条主入口：

```python
if __name__ == "__main__":
    main(_config.cli())
```

它可以拆成两步：

```text
_config.cli()  -> 读取配置，得到 TrainConfig
main(config)   -> 根据配置组装并运行训练
```

## 1. `main()` 的启动顺序

```text
检查 batch size
  -> 设置 JAX 编译缓存
  -> 创建随机 key
  -> 创建设备 mesh 和 sharding
  -> 初始化 checkpoint 目录
  -> 保存训练配置并启动 SwanLab
  -> 创建 DataLoader，取第一个 batch
  -> 初始化或恢复 TrainState
  -> JIT 编译 train_step
  -> 进入训练循环
```

这个顺序有一个重要目的：先检查配置、checkpoint 和数据，再进行昂贵的模型初始化与训练。

## 2. JAX 编译缓存

```python
jax.config.update(
    "jax_compilation_cache_dir",
    str(epath.Path("~/.cache/jax").expanduser()),
)
```

JAX 不只是逐条执行 Python 运算。`jax.jit()` 会把前向、反向和优化器更新转换为 XLA 可执行程序，再交给 GPU 执行。

```text
Python/JAX 函数
  -> 计算图
  -> XLA 优化与编译
  -> GPU 可执行程序
```

第一次遇到一组新 shape 时通常需要编译，后续相同 shape 可以复用。缓存目录让之后的进程也有机会复用编译结果。

这也是训练时固定以下 shape 的原因之一：

- batch size
- 图像分辨率
- token 最大长度
- action horizon
- action dimension

## 3. 为什么训练需要随机数

```python
rng = jax.random.key(config.seed)
train_rng, init_rng = jax.random.split(rng)
```

随机数在当前训练中至少用于：

- 随机初始化没有预训练权重的参数
- 打乱数据顺序
- 图像数据增强
- 为 flow matching 生成随机噪声
- 随机采样 flow timestep

JAX 不依赖一个隐式的全局随机状态，而是显式传递 key。`split()` 从父 key 派生互不相同的子 key：

```text
root key
  ├─ init_rng   模型初始化
  └─ train_rng  逐步训练
```

每个训练 step 又执行：

```python
train_rng = jax.random.fold_in(rng, state.step)
```

因此不同 step 得到不同随机数；相同 seed 和 step 又能得到可复现的结果。

## 4. mesh 和 sharding

```python
mesh = sharding.make_mesh(config.fsdp_devices)
data_sharding = NamedSharding(mesh, PartitionSpec(DATA_AXIS))
replicated_sharding = NamedSharding(mesh, PartitionSpec())
```

可以先把 mesh 理解为“训练使用的 GPU 排列表”。sharding 描述数组怎样放在这张表上：

- `data_sharding`：沿 batch 维分给不同设备。
- `replicated_sharding`：每张设备都持有相同的小对象，例如随机 key 和指标。
- `train_state_sharding`：根据参数形状决定参数是否进行 FSDP 切分。

## 5. 为什么先初始化 checkpoint 目录

```python
checkpoint_manager, resuming = initialize_checkpoint_dir(...)
```

这里会提前决定：

- 新建训练目录；
- `overwrite=True` 时清空已有目录；
- `resume=True` 时从已有 checkpoint 恢复；
- 两者都没有但目录已存在时直接报错。

把这个检查放在模型初始化之前，可以让路径或恢复配置错误尽早暴露。

随后：

```python
save_train_config(...)
init_swanlab(...)
```

训练使用的完整配置会写入 checkpoint 目录，SwanLab 则记录 loss、梯度范数、参数范数和样例图像。

## 6. 第一个 batch 和训练循环

```python
data_iter = iter(data_loader)
batch = next(data_iter)
```

这一步不是创建一个空壳，而是实际读取、解码、transform、组 batch，并把第一批数据放到 JAX 设备。之后它还会被用于第一次训练更新。

训练循环的核心只有：

```python
train_state, info = ptrain_step(train_rng, train_state, batch)
batch = next(data_iter)
```

即：用当前 batch 更新一次参数，然后获取下一批数据。

下一章继续追踪 `_config.cli()` 如何构造 `TrainConfig`。
