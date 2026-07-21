# 03. 数据流水线与 batch

DataLoader 的目标是把 LeRobot 中的一组原始样本变成模型直接使用的：

```python
batch = (observation, actions)
```

## 1. 完整数据链

[`create_data_loader()`](../../src/openpi/training/data_loader.py) 的主链为：

```text
config.data.create(...)
  -> DataConfig
  -> create_torch_dataset(...)
  -> transform_dataset(...)
  -> PyTorch DataLoader
  -> NumPy batch
  -> JAX device array + sharding
  -> DataLoaderImpl
  -> (Observation, Actions)
```

这里各框架的分工是：

| 组件 | 工作 |
| --- | --- |
| LeRobot | 读取 parquet、视频、task 和时间序列 |
| PyTorch DataLoader | 多进程取样、shuffle、collate、batch |
| NumPy | CPU 侧统一的中间数组格式 |
| JAX | 把 batch 放到设备并执行训练 |

## 2. 如何取得未来 action chunk

`create_torch_dataset()` 根据数据集 FPS 构造时间偏移：

```python
[t / dataset_meta.fps for t in range(action_horizon)]
```

如果 `action_horizon=16`，一条样本不只是一个 action，而是从当前时刻开始的 16 个连续动作：

```text
当前 observation_t
  配对
[action_t, action_t+1, ..., action_t+15]
```

这就是 action chunk。

## 3. 单条样本的 transform 顺序

`transform_dataset()` 的顺序不能随意交换：

```text
原始 LeRobot 样本
  -> repack：统一字段名
  -> BrainCo data transforms：相机/state/action 语义转换
  -> Normalize：归一化 state 和 action
  -> model transforms：resize、tokenize、padding
```

对于 PI0.5，最终会得到类似：

```text
images                    三路 224x224 RGB
image_masks               三路图像是否有效
state                     当前机器人状态
tokenized_prompt          prompt + 离散 state 的 token IDs
tokenized_prompt_mask     有效 token mask
actions                   未来 action chunk
```

## 4. `create_torch_data_loader()` 返回什么

它最终返回：

```python
DataLoaderImpl(data_config, TorchDataLoader(...))
```

内部两层职责不同：

- `TorchDataLoader`：CPU 取样、组 batch，并转换为带 sharding 的 JAX 数组。
- `DataLoaderImpl`：把普通字典转换成结构化 `Observation`，并返回 actions。

因此训练循环看见的是：

```python
yield Observation.from_dict(batch), batch["actions"]
```

## 5. `iter()` 和 `next()` 的含义

```python
data_iter = iter(data_loader)
batch = next(data_iter)
```

- `iter(data_loader)` 创建迭代器，表示准备按顺序取数据。
- `next(data_iter)` 真正请求下一批数据。

第一次 `next()` 会触发完整链路：读取视频/数组、执行 transforms、stack、CPU 到设备搬运。所以日志里的 `Initialized data loader` 更准确地说是：

> 已成功跑通 DataLoader，并拿到了第一个可供模型训练的 batch。

## 6. batch 是 observation/action pair 吗

是，但不是单个 pair，而是 `B` 个样本组成的一批：

```text
batch
  ├─ Observation_batch
  │    ├─ images       [B, 224, 224, 3]
  │    ├─ state        [B, D]
  │    └─ prompt IDs   [B, L]
  └─ Actions_batch     [B, H, D]
```

以 `B=32, H=16, D=28` 为例：

```text
state   [32, 28]
actions [32, 16, 28]
```

## 7. batch size 意味着什么

`batch_size=32` 表示一次参数更新同时参考 32 条训练样本：

```text
32 个样本
  -> 分别计算 loss
  -> 对 batch、时间和动作维度汇总
  -> 得到一个标量 loss
  -> 反向传播一次
  -> 参数更新一次
```

它不是拿数字 32 去训练，也不是连续更新 32 次，而是一次更新综合 32 条样本的梯度。

下一章把这个 batch 放进 PI0.5，观察 VLM 和 Action Expert 的数据流。
