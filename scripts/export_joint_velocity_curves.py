#!/usr/bin/env python3
"""Export shareable joint/velocity plots around quality-report anomalies."""

from __future__ import annotations

import argparse
import base64
from collections import defaultdict
import csv
import html
import io
import json
from pathlib import Path
import re
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as parquet

plt.switch_backend("Agg")


KINEMATIC_CODES = {"joint_step_jump", "velocity_limit", "velocity_jump"}
CODE_LABELS = {
    "joint_step_jump": "angle step",
    "velocity_limit": "velocity",
    "velocity_jump": "acceleration",
}


def _format_dataset_path(template: str, info: dict[str, Any], episode_index: int) -> Path:
    chunk = episode_index // int(info.get("chunks_size", 1000))
    return Path(template.format(episode_chunk=chunk, episode_index=episode_index))


def _column_to_numpy(table: Any, key: str) -> np.ndarray:
    array = np.asarray(table.column(key).combine_chunks().to_pylist(), dtype=np.float64)
    return array[:, None] if array.ndim == 1 else array


def _kinematics(values: np.ndarray, timestamps: np.ndarray, fps: float) -> dict[str, np.ndarray]:
    dt = np.diff(timestamps)
    dt = np.where(np.isfinite(dt) & (dt > 0), dt, 1.0 / fps)
    raw_step = np.diff(values, axis=0)
    raw_velocity = raw_step / dt[:, None]
    if len(raw_velocity) > 1:
        velocity_dt = (dt[:-1] + dt[1:]) / 2.0
        raw_acceleration = np.diff(raw_velocity, axis=0) / velocity_dt[:, None]
    else:
        raw_acceleration = np.empty((0, values.shape[1]))
    return {
        "position": values,
        "step": np.vstack([np.zeros((1, values.shape[1])), raw_step]),
        "velocity": np.vstack([np.zeros((1, values.shape[1])), raw_velocity]),
        "acceleration": np.vstack([np.zeros((2, values.shape[1])), raw_acceleration])[: len(values)],
    }


def _safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("_") or "joint"


def _event_frame(issue: dict[str, Any]) -> int:
    return int(issue.get("frame_end", issue.get("frame_start", 0)))


def _write_window_csv(
    path: Path,
    timestamps: np.ndarray,
    frame_indices: np.ndarray,
    state: dict[str, np.ndarray],
    action: dict[str, np.ndarray],
    dimension: int,
    start: int,
    end: int,
) -> None:
    fields = [
        "frame_index",
        "timestamp_s",
        "state_position_rad",
        "action_position_rad",
        "state_step_rad",
        "action_step_rad",
        "state_velocity_rad_s",
        "action_velocity_rad_s",
        "state_acceleration_rad_s2",
        "action_acceleration_rad_s2",
    ]
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for index in range(start, end + 1):
            writer.writerow(
                {
                    "frame_index": int(frame_indices[index]),
                    "timestamp_s": float(timestamps[index]),
                    "state_position_rad": float(state["position"][index, dimension]),
                    "action_position_rad": float(action["position"][index, dimension]),
                    "state_step_rad": float(state["step"][index, dimension]),
                    "action_step_rad": float(action["step"][index, dimension]),
                    "state_velocity_rad_s": float(state["velocity"][index, dimension]),
                    "action_velocity_rad_s": float(action["velocity"][index, dimension]),
                    "state_acceleration_rad_s2": float(state["acceleration"][index, dimension]),
                    "action_acceleration_rad_s2": float(action["acceleration"][index, dimension]),
                }
            )


def _plot_group(
    path: Path,
    timestamps: np.ndarray,
    state: dict[str, np.ndarray],
    action: dict[str, np.ndarray],
    dimension: int,
    joint_name: str,
    episode_index: int,
    issues: list[dict[str, Any]],
    start: int,
    end: int,
    config: dict[str, Any],
) -> None:
    view = slice(start, end + 1)
    time = timestamps[view]
    figure, axes = plt.subplots(4, 1, figsize=(15, 12), sharex=True)
    panels = (
        ("position", "Position (rad)", None),
        ("step", "Angle step (rad/frame)", float(config["max_joint_step"])),
        ("velocity", "Velocity (rad/s)", float(config["max_joint_velocity"])),
        ("acceleration", "Acceleration (rad/s²)", float(config["max_joint_acceleration"])),
    )
    for axis, (metric, ylabel, threshold) in zip(axes, panels, strict=True):
        axis.plot(time, state[metric][view, dimension], color="#1976d2", linewidth=1.7, label="state (measured)")
        axis.plot(time, action[metric][view, dimension], color="#ef6c00", linewidth=1.4, label="action (target)")
        if threshold is not None:
            axis.axhline(threshold, color="#c62828", linestyle="--", linewidth=1.0, label=f"threshold ±{threshold:g}")
            axis.axhline(-threshold, color="#c62828", linestyle="--", linewidth=1.0)
        for issue in issues:
            event = _event_frame(issue)
            color = "#1565c0" if issue["key"] == "observation.state" else "#e65100"
            axis.axvline(timestamps[event], color=color, linestyle=":", linewidth=1.3, alpha=0.9)
        axis.set_ylabel(ylabel)
        axis.grid(visible=True, alpha=0.25)
    axes[0].legend(loc="upper right", ncol=3)
    axes[-1].set_xlabel("Episode time (s)")
    event_text = "; ".join(
        f"{issue['key'].replace('observation.', '')} {CODE_LABELS[issue['code']]} "
        f"f{_event_frame(issue)}={issue['details']['observed']:.4g}"
        for issue in issues
    )
    event_text = event_text or "No kinematic threshold exceedance; full-episode review curve"
    figure.suptitle(
        f"Episode {episode_index} · selected dim {dimension} · {joint_name}\n{event_text}",
        fontsize=13,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "episode_index",
        "selected_dimension",
        "raw_dimension",
        "joint_name",
        "source",
        "issue_code",
        "frame_start",
        "frame_end",
        "time_start_s",
        "time_end_s",
        "observed",
        "threshold",
        "plot",
        "window_csv",
    ]
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _data_uri(path: Path, media_type: str) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{media_type};base64,{encoded}"


def _image_preview_data_uri(path: Path) -> str:
    from PIL import Image

    image = Image.open(path).convert("RGB")
    if image.width > 1100:
        height = round(image.height * 1100 / image.width)
        image = image.resize((1100, height), Image.Resampling.BILINEAR)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=58, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _write_html(
    path: Path,
    groups: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    *,
    standalone: bool,
) -> None:
    issue_rows: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        issue_rows[(int(row["episode_index"]), int(row["selected_dimension"]))].append(row)
    cards = []
    for group in groups:
        key = (group["episode_index"], group["selected_dimension"])
        plot_href = group["plot"]
        csv_href = group["window_csv"]
        if standalone:
            plot_href = _image_preview_data_uri(path.parent / plot_href)
        csv_link = f'<a href="{csv_href}">下载该窗口 CSV</a>' if not standalone else "逐帧 CSV 请见完整目录压缩包"
        descriptions = "<br>".join(
            f"{html.escape(row['source'])} · {html.escape(row['issue_code'])} · frame {row['frame_end']} · "
            f"{float(row['observed']):.4g} &gt; {float(row['threshold']):.4g}"
            for row in issue_rows[key]
        )
        descriptions = descriptions or "当前阈值下没有关节角、速度或加速度超限; 此处展示完整 episode 曲线供人工复盘。"
        cards.append(
            f"""
            <section>
              <h2>Episode {group["episode_index"]} · {html.escape(group["joint_name"])}</h2>
              <p>{descriptions}</p>
              <p>{csv_link}</p>
              <a href="{plot_href}"><img src="{plot_href}" loading="lazy"></a>
            </section>
            """
        )
    summary_href = _data_uri(path.parent / "summary.csv", "text/csv") if standalone else "summary.csv"
    path.write_text(
        """<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<title>Revo3 关节异常曲线</title><style>
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;margin:24px;max-width:1500px}
section{border-top:1px solid #ddd;padding:18px 0}img{width:100%;height:auto}code{background:#eee;padding:2px 5px}
</style></head><body>
<h1>Revo3 关节角 / 速度异常曲线</h1>
<p>蓝线是 <code>observation.state</code> (实测), 橙线是 <code>action</code> (目标)。红色虚线是当前阈值, 竖向点线是 JSON 报告记录的异常帧。</p>
<p>state 与 action 仅叠加用于对照; 没有进行控制延迟补偿, 不能把两条线的瞬时差直接解释为跟踪误差。</p>
<p><a href="""
        + summary_href
        + '" download="summary.csv">下载全部异常摘要 CSV</a></p>\n'
        + """
"""
        + "\n".join(cards)
        + "</body></html>\n"
    )


def export_curves(
    dataset: Path,
    report_path: Path,
    output_dir: Path,
    context_s: float,
    *,
    all_joints: bool,
) -> None:
    root = dataset.expanduser().resolve()
    report = json.loads(report_path.expanduser().read_text())
    info = json.loads((root / "meta" / "info.json").read_text())
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    csv_dir = output_dir / "csv"
    plots_dir.mkdir(exist_ok=True)
    csv_dir.mkdir(exist_ok=True)

    episode_reports = {int(item["episode_index"]): item for item in report["episodes"]}
    first_episode = next(iter(episode_reports.values()))
    dimensions = [int(index) for index in first_episode["vector_dimensions"]]
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for episode in report["episodes"]:
        for issue in episode["issues"]:
            if issue["code"] in KINEMATIC_CODES:
                grouped[(int(episode["episode_index"]), int(issue["details"]["dimension"]))].append(issue)
    if all_joints:
        for episode_index in episode_reports:
            for dimension in range(len(dimensions)):
                grouped.setdefault((episode_index, dimension), [])

    raw_names = info["features"][first_episode["state_key"]].get("names")
    joint_names = [str(raw_names[index]) if raw_names else f"dim_{index:02d}" for index in dimensions]
    fps = float(info["fps"])
    context_frames = max(1, round(context_s * fps))
    data_template = info.get("data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet")
    summary_rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []

    for episode_index in sorted({key[0] for key in grouped}):
        episode_report = episode_reports[episode_index]
        table = parquet.read_table(root / _format_dataset_path(data_template, info, episode_index))
        timestamps = _column_to_numpy(table, "timestamp")[:, 0]
        frame_indices = _column_to_numpy(table, "frame_index")[:, 0].astype(np.int64)
        state_values = _column_to_numpy(table, episode_report["state_key"])[:, dimensions]
        action_values = _column_to_numpy(table, episode_report["action_key"])[:, dimensions]
        state = _kinematics(state_values, timestamps, fps)
        action = _kinematics(action_values, timestamps, fps)

        for (group_episode, dimension), issues in sorted(grouped.items()):
            if group_episode != episode_index:
                continue
            if issues:
                event_frames = [_event_frame(issue) for issue in issues]
                start = max(0, min(event_frames) - context_frames)
                end = min(len(timestamps) - 1, max(event_frames) + context_frames)
            else:
                start, end = 0, len(timestamps) - 1
            joint_name = joint_names[dimension]
            stem = f"ep{episode_index:02d}_dim{dimension:02d}_{_safe_name(joint_name)}"
            plot_rel = f"plots/{stem}.png"
            csv_rel = f"csv/{stem}.csv"
            _plot_group(
                output_dir / plot_rel,
                timestamps,
                state,
                action,
                dimension,
                joint_name,
                episode_index,
                issues,
                start,
                end,
                report["config"],
            )
            _write_window_csv(
                output_dir / csv_rel,
                timestamps,
                frame_indices,
                state,
                action,
                dimension,
                start,
                end,
            )
            group_rows.append(
                {
                    "episode_index": episode_index,
                    "selected_dimension": dimension,
                    "joint_name": joint_name,
                    "plot": plot_rel,
                    "window_csv": csv_rel,
                }
            )
            summary_rows.extend(
                {
                    "episode_index": episode_index,
                    "selected_dimension": dimension,
                    "raw_dimension": dimensions[dimension],
                    "joint_name": joint_name,
                    "source": issue["key"],
                    "issue_code": issue["code"],
                    "frame_start": issue.get("frame_start"),
                    "frame_end": issue.get("frame_end"),
                    "time_start_s": issue.get("time_start_s"),
                    "time_end_s": issue.get("time_end_s"),
                    "observed": issue["details"]["observed"],
                    "threshold": issue["details"]["threshold"],
                    "plot": plot_rel,
                    "window_csv": csv_rel,
                }
                for issue in issues
            )

    _write_summary_csv(output_dir / "summary.csv", summary_rows)
    _write_html(output_dir / "index.html", group_rows, summary_rows, standalone=False)
    _write_html(output_dir / "index_standalone.html", group_rows, summary_rows, standalone=True)
    (output_dir / "README.txt").write_text(
        "Open index.html to browse all anomaly curves.\n"
        "Blue: observation.state (measured). Orange: action (target). Red dashed: threshold.\n"
        "The plots use the same formulas and thresholds as data_quality.json.\n"
        "State/action delay is not compensated, so their instantaneous difference is not tracking error.\n"
    )
    print(f"Exported {len(group_rows)} plots and {len(summary_rows)} issue rows to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--context-s", type=float, default=2.0, help="Seconds shown before/after each anomaly")
    parser.add_argument("--all-joints", action="store_true", help="Export every joint even without anomalies")
    args = parser.parse_args()
    export_curves(args.dataset, args.report, args.output, args.context_s, all_joints=args.all_joints)


if __name__ == "__main__":
    main()
