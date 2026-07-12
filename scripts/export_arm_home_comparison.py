#!/usr/bin/env python3
"""Compare first-frame arm positions across LeRobot episodes."""

from __future__ import annotations

import argparse
import base64
import csv
import json
from pathlib import Path

from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as parquet

plt.switch_backend("Agg")


def _format_path(template: str, info: dict, episode: int) -> Path:
    chunk = episode // int(info.get("chunks_size", 1000))
    return Path(template.format(episode_chunk=chunk, episode_index=episode))


def _image_uri(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def export(dataset: Path, output: Path, tolerance: float) -> None:
    root = dataset.expanduser().resolve()
    info = json.loads((root / "meta" / "info.json").read_text())
    dimensions = list(range(14, 28))
    names = [info["features"]["observation.state"]["names"][index] for index in dimensions]
    episodes = list(range(int(info["total_episodes"])))
    positions = []
    template = info.get("data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet")
    for episode in episodes:
        table = parquet.read_table(root / _format_path(template, info, episode), columns=["observation.state"])
        first_state = np.asarray(table["observation.state"].to_pylist()[0], dtype=np.float64)
        positions.append(first_state[dimensions])
    positions = np.stack(positions)
    reference = np.median(positions, axis=0)
    deviation = positions - reference
    output.mkdir(parents=True, exist_ok=True)

    with (output / "arm_first_frame_positions.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["episode_index", *names])
        for episode, row in zip(episodes, positions, strict=True):
            writer.writerow([episode, *row.tolist()])

    figure, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    for axis, start, title in ((axes[0], 0, "Left arm"), (axes[1], 7, "Right arm")):
        for local_dimension in range(7):
            dimension = start + local_dimension
            axis.plot(episodes, positions[:, dimension], marker="o", linewidth=1.8, label=names[dimension])
        axis.set_title(title)
        axis.set_ylabel("First-frame position (rad)")
        axis.grid(visible=True, alpha=0.25)
        axis.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
    axes[-1].set_xlabel("Episode index")
    axes[-1].set_xticks(episodes)
    figure.suptitle("Revo3 arm first-frame positions across episodes")
    figure.tight_layout(rect=(0, 0, 0.82, 0.96))
    figure.savefig(output / "arm_first_frame_positions.png", dpi=180)
    plt.close(figure)

    limit = max(float(np.max(np.abs(deviation))), tolerance)
    figure, axis = plt.subplots(figsize=(18, 8))
    image = axis.imshow(deviation, cmap="coolwarm", norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit))
    axis.set_xticks(np.arange(len(names)), labels=names, rotation=50, ha="right")
    axis.set_yticks(np.arange(len(episodes)), labels=[f"ep{episode}" for episode in episodes])
    axis.set_title(f"First-frame deviation from per-joint median (red boxes: |error| > {tolerance:g} rad)")
    for row in range(len(episodes)):
        for column in range(len(names)):
            value = deviation[row, column]
            axis.text(column, row, f"{value:+.2f}", ha="center", va="center", fontsize=7)
            if abs(value) > tolerance:
                axis.add_patch(Rectangle((column - 0.48, row - 0.48), 0.96, 0.96, fill=False, edgecolor="red", lw=1.5))
    figure.colorbar(image, ax=axis, label="Position difference (rad)")
    figure.tight_layout()
    figure.savefig(output / "arm_first_frame_deviation_heatmap.png", dpi=180)
    plt.close(figure)

    position_uri = _image_uri(output / "arm_first_frame_positions.png")
    heatmap_uri = _image_uri(output / "arm_first_frame_deviation_heatmap.png")
    (output / "index_standalone.html").write_text(
        f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>机械臂首帧位置对比</title>
<style>body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;margin:24px}}img{{width:100%;height:auto}}</style></head><body>
<h1>机械臂 episode 首帧位置对比</h1>
<p>第一张图把 10 个 episode 的首帧直接叠加。第二张图显示每个首帧相对逐关节中位数的差, 红框表示绝对差超过 {tolerance:g} rad。</p>
<p>这只是跨 episode 起始位置一致性检查, 不代表统计中位数就是真实硬件 home。</p>
<img src="{position_uri}"><img src="{heatmap_uri}"></body></html>\n"""
    )
    print(f"Exported first-frame comparison for {len(episodes)} episodes to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=0.1)
    args = parser.parse_args()
    export(args.dataset, args.output, args.tolerance)


if __name__ == "__main__":
    main()
