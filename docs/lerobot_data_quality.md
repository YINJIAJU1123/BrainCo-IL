# LeRobot 数据质检与 Rerun 可视化

`scripts/visualize_lerobot_dataset.py` 面向 LeRobot v2.x 数据集，同时完成自动质检、JSON 报告和 Rerun 可视化。它直接读取 `meta/info.json`、episode Parquet 和相机 MP4，不会修改原始数据。

本项目的数据在无图形界面的训练服务器上，因此标准流程是：**服务器生成 `.rrd` 和 JSON → `rsync` 到 Mac → 只在 Mac 打开 Rerun Viewer**。不要在 SSH 服务器上使用 `--spawn`。

## 快速使用

当前 Revo3 数据原始 state/action 是 70D，其中前 14 维是双侧 EEF pose。关节质量检查应抽取与训练变换一致的 56D（左臂、左手、右臂、右手）。在服务器运行：

```bash
cd /home/ruibin/wuji-openpi
source .venv/bin/activate

DATA=/mnt/data_nas/ruibin/dataset/original-revomate_revo3_pick_and_place/original/lerobot_v21/revomate_revo3_mit_3cam_test
OUT=/home/ruibin/wuji-openpi/outputs/data_quality/revo3_pick_place
mkdir -p "$OUT"

python scripts/visualize_lerobot_dataset.py "$DATA" \
  --episodes 0-9 \
  --vector-dims '14:21,28:49,21:28,49:70' \
  --expected-state-dim 56 \
  --expected-action-dim 56 \
  --image-stride 3 \
  --image-max-width 640 \
  --jpeg-quality 75 \
  --rrd "$OUT/data_quality.rrd" \
  --report "$OUT/data_quality.json"
```

在 Mac 上拉取结果：

```bash
mkdir -p /Users/larrybb/Downloads/wuji_data_quality/revo3_pick_place

rsync -azP \
  -e 'ssh -i ~/.ssh/id_rsa_remote -p 6007' \
  ruibin@8.130.44.94:/home/ruibin/wuji-openpi/outputs/data_quality/revo3_pick_place/ \
  /Users/larrybb/Downloads/wuji_data_quality/revo3_pick_place/
```

然后在 Mac 打开：

```bash
cd /Users/larrybb/PycharmProjects/wuji-openpi
uv run rerun /Users/larrybb/Downloads/wuji_data_quality/revo3_pick_place/data_quality.rrd
```

只生成检查报告、不加载 Rerun：

```bash
python scripts/visualize_lerobot_dataset.py "$DATA" \
  --no-rerun \
  --report outputs/data_quality.json \
  --fail-on-error
```

`--fail-on-error` 在发现 error 时返回退出码 2，适合放进数据上传或训练前的流水线。

## 检查内容

- episode 边界静止：用状态向量的逐关节速度识别开头、结尾静止时间，并在 JSON 中给出 `suggested_trim`，但不自动洗数据。
- 掉帧与时间戳：检查共享时间戳的倒退、重复、周期偏差和缺帧间隔；逐路解码 MP4，检查帧数、PTS 间隔和与 episode 时间轴的相对对齐。
- reset/home 一致性：默认以所选 episodes 首帧的逐维中位数为 home；生产检查建议用 `--home-position '[...]'` 或传入 JSON 文件，避免整批数据都偏离时“互相证明正确”。
- 关节连续性：对 state 和 action 检查单帧角度步进、速度上限和速度突变（加速度）。
- 数值有效性：检查 state/action 中的 NaN、Inf，以及可选的预期维数。

Rerun 内包含三路相机、56D joint/hand state position、state velocity、action、episode index、质量严重度曲线和问题事件。拖动统一时间轴即可对照图像与关节曲线。

## 时间戳对齐的边界

标准 LeRobot 数据通常只有每个样本一个共享 `timestamp`。这种数据可以验证：Parquet 行连续、相机视频帧数正确、视频 PTS 与共享时间轴一致；但它不能反向证明录制时各 ROS topic 的原始时间戳真的对齐。

脚本会自动查找名字中含 `timestamp` 的额外标量列。若没有找到，会生成 `independent_timestamps_missing` warning，并在报告中令 `can_prove_capture_time_alignment=false`。要严格落实“所有相机和关节角不能掉帧、时间戳对齐”，录制/转换阶段应额外保留下列字段（命名可自定）：

```text
timestamp.left_arm
timestamp.right_arm
timestamp.left_hand
timestamp.right_hand
timestamp.cam_high
timestamp.cam_left_wrist
timestamp.cam_right_wrist
```

它们应来自同一时钟域。脚本会自动推断秒/毫秒/微秒/纳秒单位，检查相对主时间轴的 offset 和 jitter。

## 阈值调整

默认值偏向发现明显坏数据，实际阈值应根据控制频率、手部噪声和任务速度标定：

```text
--max-start-idle-s 0.30
--max-end-idle-s 0.30
--min-motion-velocity 0.02       # rad/s
--max-joint-step 0.35            # rad/frame
--max-joint-velocity 6.0         # rad/s
--max-joint-acceleration 100.0   # rad/s^2
--home-tolerance 0.10            # rad
--timestamp-tolerance-ms 5
--sync-tolerance-ms 20
```

脚本默认把图像缩放到最大宽度 640 并以 JPEG 写入 RRD，避免把原始 720p RGB 帧无压缩塞进文件。还可用 `--image-stride 3` 降低图像采样，或用 `--skip-images` 只保留曲线。`--skip-video-checks` 会加快纯关节数据检查，但也会放弃相机帧数与 PTS 质检。
