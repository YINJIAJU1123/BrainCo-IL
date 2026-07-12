#!/usr/bin/env python3
"""Browser UI for visually selecting LeRobot episode trim ranges.

This script is meant for remote-server workflows: the dataset stays on the
server, while you interact with a small web page from your local browser.  It
does not modify SOURCE.  When asked to build OUTPUT, it reuses the same range
trim writer used by ``interactive_trim_lerobot_episodes.py``.

Example:

    SOURCE=/mnt/data_nas/ruibin/dataset/.../revomate_revo3_mit_3cam_test
    OUTPUT=/mnt/data_nas/ruibin/dataset/trimmed/revo3_visual_trimmed

    python scripts/web_trim_lerobot_episodes.py "$SOURCE" "$OUTPUT" --host 127.0.0.1 --port 7860

For a remote server, forward the port from your local machine:

    ssh -i ~/.ssh/id_rsa_remote -p 6007 -L 7860:127.0.0.1:7860 ruibin@8.130.44.94

Then open http://127.0.0.1:7860 in a browser.  Pick an episode, choose a camera,
use the video as the main reference, and drag on the curve to select the kept
interval.  Click "Snap & save episode" before moving to another episode.

Output modes:

* selected-only off: write the full dataset; saved episodes are cropped and
  unsaved episodes stay complete.
* selected-only on: write only the saved episodes and reindex them from zero.

Use "Dry run" before "Build output" to verify the exact output episode count and
frame ranges.  The source dataset is always read-only.
"""

from __future__ import annotations

import argparse
import contextlib
from dataclasses import asdict
from datetime import UTC
from datetime import datetime
from http.server import BaseHTTPRequestHandler
from http.server import ThreadingHTTPServer
import io
import json
from pathlib import Path
import shutil
import sys
import threading
import time
import traceback
from typing import Any
from urllib.parse import parse_qs
from urllib.parse import urlparse

from interactive_trim_lerobot_episodes import EpisodeRange
from interactive_trim_lerobot_episodes import _build_range_plans
from interactive_trim_lerobot_episodes import _choose_vector_key
from interactive_trim_lerobot_episodes import _column_to_numpy
from interactive_trim_lerobot_episodes import _format_dataset_path
from interactive_trim_lerobot_episodes import _json_safe
from interactive_trim_lerobot_episodes import _load_dataset
from interactive_trim_lerobot_episodes import _load_episode_table
from interactive_trim_lerobot_episodes import _load_range_manifest
from interactive_trim_lerobot_episodes import _load_trim_writer_helpers
from interactive_trim_lerobot_episodes import _motion_signal
from interactive_trim_lerobot_episodes import _patch_range_provenance
from interactive_trim_lerobot_episodes import _print_range_plan
from interactive_trim_lerobot_episodes import _robust_normalize
from interactive_trim_lerobot_episodes import _scalar_column
from interactive_trim_lerobot_episodes import _snap_range_to_frames
from interactive_trim_lerobot_episodes import _validate_paths
from interactive_trim_lerobot_episodes import _write_range_manifest
import numpy as np

HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>LeRobot Visual Trim</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f7f7fb;
      --card: #ffffff;
      --text: #151721;
      --muted: #667085;
      --line: #d7dce7;
      --blue: #2563eb;
      --orange: #f97316;
      --green: #16a34a;
      --red: #dc2626;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    main {
      max-width: 1180px;
      margin: 28px auto;
      padding: 0 18px 48px;
    }
    h1 {
      margin: 0 0 4px;
      font-size: 26px;
      letter-spacing: -0.02em;
    }
    .sub {
      margin: 0 0 18px;
      color: var(--muted);
      font-size: 14px;
    }
    .grid {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 340px;
      gap: 16px;
    }
    .card {
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 16px;
      box-shadow: 0 10px 26px rgba(15, 23, 42, 0.06);
      padding: 16px;
    }
    .row {
      display: flex;
      align-items: center;
      gap: 10px;
      flex-wrap: wrap;
    }
    label {
      font-size: 13px;
      color: var(--muted);
    }
    select, input {
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 8px 10px;
      background: white;
      font: inherit;
      min-height: 38px;
    }
    input[type="number"] { width: 120px; }
    input[type="checkbox"] { min-height: auto; }
    button {
      border: 0;
      border-radius: 10px;
      padding: 9px 12px;
      background: var(--blue);
      color: white;
      font-weight: 650;
      cursor: pointer;
    }
    button.secondary { background: #475569; }
    button.green { background: var(--green); }
    button.red { background: var(--red); }
    button:disabled {
      cursor: not-allowed;
      opacity: 0.55;
    }
    #chart {
      width: 100%;
      height: 270px;
      user-select: none;
      touch-action: none;
      border: 1px solid var(--line);
      border-radius: 14px;
      background: linear-gradient(#fff, #fbfdff);
    }
    .video-toolbar {
      justify-content: space-between;
      margin-bottom: 10px;
    }
    .video-shell {
      position: relative;
      width: 100%;
      background: #020617;
      border: 1px solid var(--line);
      border-radius: 16px;
      overflow: hidden;
      min-height: 380px;
      display: grid;
      place-items: center;
    }
    #videoPlayer {
      width: 100%;
      max-height: 620px;
      background: #020617;
      display: block;
    }
    .video-placeholder {
      position: absolute;
      inset: 0;
      display: grid;
      place-items: center;
      color: #cbd5e1;
      padding: 20px;
      text-align: center;
      pointer-events: none;
    }
    .chart-title {
      display: flex;
      justify-content: space-between;
      gap: 10px;
      align-items: center;
      margin: 12px 0 8px;
    }
    .hint {
      margin-top: 10px;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.45;
    }
    .kv {
      display: grid;
      grid-template-columns: 80px minmax(0, 1fr);
      gap: 6px 10px;
      font-size: 13px;
      margin-top: 12px;
    }
    .kv div:nth-child(odd) { color: var(--muted); }
    code, pre {
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    }
    code {
      overflow-wrap: anywhere;
      color: #334155;
      background: #f1f5f9;
      border-radius: 6px;
      padding: 1px 5px;
    }
    pre {
      min-height: 160px;
      max-height: 360px;
      overflow: auto;
      white-space: pre-wrap;
      background: #0f172a;
      color: #dbeafe;
      border-radius: 14px;
      padding: 12px;
      font-size: 12px;
      line-height: 1.45;
    }
    .pill {
      display: inline-flex;
      border-radius: 999px;
      padding: 3px 8px;
      background: #e0f2fe;
      color: #075985;
      font-size: 12px;
      font-weight: 700;
    }
    .small { font-size: 12px; color: var(--muted); }
    .selection-list {
      display: grid;
      gap: 8px;
      margin-top: 10px;
    }
    .selection-item {
      border: 1px solid var(--line);
      border-radius: 12px;
      background: #f8fafc;
      padding: 9px;
      font-size: 12px;
    }
    .selection-item .row {
      justify-content: space-between;
      gap: 8px;
    }
    .selection-item button {
      padding: 5px 8px;
      font-size: 12px;
    }
    .empty {
      color: var(--muted);
      font-size: 13px;
      background: #f8fafc;
      border: 1px dashed var(--line);
      border-radius: 12px;
      padding: 10px;
      margin-top: 10px;
    }
    @media (max-width: 900px) {
      .grid { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <main>
    <h1>LeRobot Visual Trim</h1>
    <p class="sub">Drag on the curve to select the interval you want to keep. Source data is read-only.</p>

    <div class="grid">
      <section class="card">
        <div class="row">
          <label for="episodeSelect">Episode</label>
          <select id="episodeSelect"></select>
          <button id="loadBtn" class="secondary">Load</button>
          <span id="episodeInfo" class="pill">loading…</span>
        </div>
        <div style="height: 12px"></div>
        <div class="row video-toolbar">
          <label for="cameraSelect">Camera <select id="cameraSelect"></select></label>
          <span id="videoTime" class="small">video: --</span>
        </div>
        <div class="video-shell">
          <video id="videoPlayer" controls playsinline preload="metadata"></video>
          <div id="videoPlaceholder" class="video-placeholder">Loading video…</div>
        </div>
        <div class="chart-title">
          <strong>Motion curve / trim range</strong>
          <span class="small">Drag here to choose start/end; video follows the pointer.</span>
        </div>
        <svg id="chart" role="img" aria-label="motion signal chart"></svg>
        <p class="hint">
          操作: 先看上方视频定位动作, 再在下方曲线上按住并横向拖动, 蓝色区域就是保留区间。
          拖动曲线时视频会跳到对应时间。也可以手动输入 start/end 秒数。
          曲线是 state/action 的归一化速度范数, 用来帮助你找开头停顿和结束点。
        </p>
      </section>

      <aside class="card">
        <div class="row">
          <label>Start s <input id="startInput" type="number" min="0" step="0.001"></label>
          <label>End s <input id="endInput" type="number" min="0" step="0.001"></label>
        </div>
        <div style="height: 10px"></div>
        <div class="row">
          <button id="saveSelectionBtn">Snap & save episode</button>
          <button id="clearBtn" class="secondary">Clear episode</button>
        </div>
        <div id="snapInfo" class="hint"></div>

        <hr style="border:0;border-top:1px solid var(--line);margin:16px 0">

        <div class="row" style="justify-content:space-between">
          <strong>Selected episodes</strong>
          <button id="clearAllBtn" class="red">Clear all</button>
        </div>
        <div id="selectedSummary" class="hint"></div>
        <div id="selectedList" class="selection-list"></div>

        <div class="kv">
          <div>Source</div><div><code id="sourcePath"></code></div>
          <div>Output</div><div><code id="outputPath"></code></div>
          <div>Manifest</div><div><code id="manifestPath"></code></div>
        </div>

        <hr style="border:0;border-top:1px solid var(--line);margin:16px 0">

        <label class="row" style="justify-content:flex-start">
          <input id="selectedOnlyInput" type="checkbox">
          <span>selected-only: 只输出已选择 episode, 并从 0 重新编号</span>
        </label>
        <div id="outputModeInfo" class="hint"></div>
        <label class="row" style="justify-content:flex-start;margin-top:8px">
          <input id="skipHashesInput" type="checkbox">
          <span>skip source hashes: 更快, 但 provenance 少 SHA-256</span>
        </label>

        <div style="height: 12px"></div>
        <div class="row">
          <button id="saveManifestBtn" class="secondary">Save manifest</button>
          <button id="dryRunBtn" class="secondary">Dry run</button>
          <button id="buildBtn" class="green">Build output</button>
        </div>
        <p class="small">
          最终训练集一般不要勾 selected-only; 单条 episode 验证才勾。
        </p>
      </aside>
    </div>

    <section class="card" style="margin-top:16px">
      <div class="row" style="justify-content:space-between">
        <strong>Log</strong>
        <span id="statusText" class="small">idle</span>
      </div>
      <pre id="log"></pre>
    </section>
  </main>

  <script>
    const svg = document.getElementById("chart");
    const episodeSelect = document.getElementById("episodeSelect");
    const cameraSelect = document.getElementById("cameraSelect");
    const videoPlayer = document.getElementById("videoPlayer");
    const videoPlaceholder = document.getElementById("videoPlaceholder");
    const videoTime = document.getElementById("videoTime");
    const logEl = document.getElementById("log");
    const statusText = document.getElementById("statusText");
    const startInput = document.getElementById("startInput");
    const endInput = document.getElementById("endInput");
    const snapInfo = document.getElementById("snapInfo");
    const selectedSummary = document.getElementById("selectedSummary");
    const selectedList = document.getElementById("selectedList");
    const selectedOnlyInput = document.getElementById("selectedOnlyInput");
    const outputModeInfo = document.getElementById("outputModeInfo");
    const skipHashesInput = document.getElementById("skipHashesInput");

    let config = null;
    let episodes = [];
    let current = null;
    let selection = null;
    let playheadTime = 0;
    let dragging = false;
    let dragStart = 0;
    let statusTimer = null;

    const colors = ["#2563eb", "#f97316", "#16a34a", "#9333ea"];
    const margin = {left: 58, right: 24, top: 24, bottom: 42};

    function log(text) {
      logEl.textContent = text;
    }

    function outputModeText() {
      const selectedCount = document.querySelectorAll("[data-clear-episode]").length;
      if (selectedOnlyInput.checked) {
        return `MODE: selected-only output (${selectedCount} saved selected episode(s), reindexed from 0).`;
      }
      const total = episodes?.episodes?.length || 0;
      return `MODE: full dataset output (${total} episode(s); ` +
        "saved selections are cropped, unselected episodes stay full).";
    }

    function updateOutputModeInfo() {
      outputModeInfo.textContent = outputModeText();
    }

    async function api(path, options = {}) {
      const response = await fetch(path, {
        headers: {"content-type": "application/json"},
        ...options,
      });
      const data = await response.json();
      if (!response.ok || data.ok === false) {
        throw new Error(data.error || `HTTP ${response.status}`);
      }
      return data;
    }

    function svgEl(name, attrs = {}) {
      const node = document.createElementNS("http://www.w3.org/2000/svg", name);
      for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
      return node;
    }

    function chartSize() {
      const rect = svg.getBoundingClientRect();
      return {
        width: Math.max(720, rect.width || 720),
        height: Math.max(360, rect.height || 480),
      };
    }

    function xScale(t) {
      const {width} = chartSize();
      const plotWidth = width - margin.left - margin.right;
      const duration = Math.max(current?.duration_s || 1, 1e-9);
      return margin.left + (t / duration) * plotWidth;
    }

    function xInvert(x) {
      const {width} = chartSize();
      const plotWidth = width - margin.left - margin.right;
      const duration = Math.max(current?.duration_s || 1, 1e-9);
      return Math.max(0, Math.min(duration, ((x - margin.left) / plotWidth) * duration));
    }

    function yScale(v) {
      const {height} = chartSize();
      const plotHeight = height - margin.top - margin.bottom;
      return margin.top + (1 - v) * plotHeight;
    }

    function pointerTime(event) {
      const pt = svg.createSVGPoint();
      pt.x = event.clientX;
      pt.y = event.clientY;
      const local = pt.matrixTransform(svg.getScreenCTM().inverse());
      return xInvert(local.x);
    }

    function drawChart() {
      const {width, height} = chartSize();
      svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
      svg.innerHTML = "";
      if (!current) return;

      const plotWidth = width - margin.left - margin.right;
      const plotHeight = height - margin.top - margin.bottom;

      svg.appendChild(svgEl("rect", {
        x: margin.left,
        y: margin.top,
        width: plotWidth,
        height: plotHeight,
        fill: "#ffffff",
        stroke: "#d7dce7",
      }));

      for (let i = 0; i <= 5; i++) {
        const x = margin.left + (plotWidth * i) / 5;
        const t = (current.duration_s * i) / 5;
        svg.appendChild(svgEl("line", {x1: x, y1: margin.top, x2: x, y2: margin.top + plotHeight, stroke: "#eef2f7"}));
        const text = svgEl("text", {x, y: height - 14, "text-anchor": "middle", fill: "#64748b", "font-size": "12"});
        text.textContent = t.toFixed(2) + "s";
        svg.appendChild(text);
      }
      for (let i = 0; i <= 4; i++) {
        const y = margin.top + (plotHeight * i) / 4;
        svg.appendChild(svgEl("line", {
          x1: margin.left,
          y1: y,
          x2: margin.left + plotWidth,
          y2: y,
          stroke: "#eef2f7",
        }));
      }

      current.series.forEach((series, seriesIndex) => {
        const points = current.time_s
          .map((t, i) => `${xScale(t).toFixed(2)},${yScale(series.values[i]).toFixed(2)}`)
          .join(" ");
        svg.appendChild(svgEl("polyline", {
          points,
          fill: "none",
          stroke: colors[seriesIndex % colors.length],
          "stroke-width": "1.8",
          "stroke-linejoin": "round",
          "stroke-linecap": "round",
        }));
        const legend = svgEl("text", {
          x: margin.left + 10,
          y: margin.top + 18 + 18 * seriesIndex,
          fill: colors[seriesIndex % colors.length],
          "font-size": "13",
          "font-weight": "700",
        });
        legend.textContent = series.label;
        svg.appendChild(legend);
      });

      if (selection) {
        const start = Math.min(selection.start, selection.end);
        const end = Math.max(selection.start, selection.end);
        const x1 = xScale(start);
        const x2 = xScale(end);
        svg.appendChild(svgEl("rect", {
          x: x1,
          y: margin.top,
          width: Math.max(1, x2 - x1),
          height: plotHeight,
          fill: "#2563eb",
          opacity: "0.13",
        }));
        svg.appendChild(svgEl("line", {
          x1,
          y1: margin.top,
          x2: x1,
          y2: margin.top + plotHeight,
          stroke: "#2563eb",
          "stroke-width": "2",
        }));
        svg.appendChild(svgEl("line", {
          x1: x2,
          y1: margin.top,
          x2,
          y2: margin.top + plotHeight,
          stroke: "#2563eb",
          "stroke-width": "2",
        }));
      }

      if (Number.isFinite(playheadTime)) {
        const playheadX = xScale(playheadTime);
        svg.appendChild(svgEl("line", {
          x1: playheadX,
          y1: margin.top,
          x2: playheadX,
          y2: margin.top + plotHeight,
          stroke: "#f97316",
          "stroke-width": "2",
          "stroke-dasharray": "5 4",
        }));
      }
    }

    function setVideoTime(seconds) {
      if (!current) return;
      const duration = current.duration_s;
      const next = Math.max(0, Math.min(duration, seconds));
      playheadTime = next;
      if (Number.isFinite(videoPlayer.duration)) {
        videoPlayer.currentTime = next;
      }
      videoTime.textContent = `video: ${next.toFixed(3)}s / ${duration.toFixed(3)}s`;
      drawChart();
    }

    function setSelection(start, end, syncVideo = false) {
      if (!current) return;
      const duration = current.duration_s;
      const a = Math.max(0, Math.min(duration, start));
      const b = Math.max(0, Math.min(duration, end));
      selection = {start: Math.min(a, b), end: Math.max(a, b)};
      startInput.value = selection.start.toFixed(6);
      endInput.value = selection.end.toFixed(6);
      if (syncVideo) setVideoTime(b);
      drawChart();
    }

    function loadVideo() {
      if (!current) return;
      if (!config.video_keys || !config.video_keys.length) {
        videoPlayer.removeAttribute("src");
        videoPlaceholder.textContent = "This dataset has no video feature in info.json.";
        videoPlaceholder.style.display = "grid";
        videoTime.textContent = "video: unavailable";
        return;
      }
      const videoKey = cameraSelect.value || config.video_keys[0];
      const url = `/media/video?episode_index=${current.episode_index}&video_key=${encodeURIComponent(videoKey)}`;
      videoPlayer.src = url;
      videoPlaceholder.style.display = "none";
      videoTime.textContent = `video: ${playheadTime.toFixed(3)}s / ${current.duration_s.toFixed(3)}s`;
    }

    async function loadEpisode() {
      const episodeId = Number(episodeSelect.value);
      snapInfo.textContent = "";
      current = await api(`/api/episode?episode_index=${episodeId}`);
      playheadTime = 0;
      document.getElementById("episodeInfo").textContent =
        `ep ${episodeId} · ${current.length} frames · ${current.duration_s.toFixed(2)}s · fps ${current.fps}`;
      loadVideo();
      if (current.selection) {
        setSelection(current.selection.trim_start_s, current.selection.trim_end_s);
        snapInfo.textContent =
          `Loaded saved selection: frames [${current.selection.start_frame}:${current.selection.end_frame_exclusive})`;
      } else {
        setSelection(0, current.duration_s);
      }
      drawChart();
    }

    async function refreshSelections() {
      const data = await api("/api/selections");
      const rows = data.episodes || [];
      const selectedIds = rows.map(r => r.episode_index).join(", ");
      selectedSummary.textContent = rows.length
        ? `${rows.length} saved crop selection(s): ${selectedIds}. ` +
          "Unchecked selected-only = output all episodes; checked = output only these."
        : "No saved selections. Full mode outputs all episodes unchanged; selected-only would output nothing.";
      if (!rows.length) {
        selectedList.innerHTML = `<div class="empty">No selected episodes yet.</div>`;
        updateOutputModeInfo();
        return;
      }
      selectedList.innerHTML = rows.map(r => `
        <div class="selection-item">
          <div class="row">
            <strong>episode ${r.episode_index}</strong>
            <span>
              <button class="secondary" data-load-episode="${r.episode_index}">Load</button>
              <button class="red" data-clear-episode="${r.episode_index}">Clear</button>
            </span>
          </div>
          <div>
            ${r.trim_start_s.toFixed(6)}s → ${r.trim_end_s.toFixed(6)}s<br>
            frames [${r.start_frame}:${r.end_frame_exclusive}), kept ${r.kept_length}/${r.original_length}
          </div>
        </div>
      `).join("");
      updateOutputModeInfo();
    }

    async function saveSelection() {
      if (!current || !selection) return;
      const data = await api("/api/selection", {
        method: "POST",
        body: JSON.stringify({
          episode_index: current.episode_index,
          trim_start_s: Number(startInput.value),
          trim_end_s: Number(endInput.value),
        }),
      });
      const r = data.range;
      current.selection = r;
      snapInfo.textContent =
        `Saved ep ${r.episode_index}: clicked ${r.trim_start_s.toFixed(6)}s→${r.trim_end_s.toFixed(6)}s, ` +
        `snapped frames [${r.start_frame}:${r.end_frame_exclusive}), kept ${r.kept_length}/${r.original_length}.`;
      log(JSON.stringify(data, null, 2));
      await refreshSelections();
    }

    async function clearSelection() {
      if (!current) return;
      const data = await api("/api/selection", {
        method: "DELETE",
        body: JSON.stringify({episode_index: current.episode_index}),
      });
      snapInfo.textContent = data.message;
      setSelection(0, current.duration_s);
      log(JSON.stringify(data, null, 2));
      await refreshSelections();
    }

    async function clearAllSelections() {
      if (!window.confirm("Clear all saved episode selections in this server session?")) return;
      const data = await api("/api/selections", {method: "DELETE", body: JSON.stringify({})});
      snapInfo.textContent = data.message;
      if (current) {
        current.selection = null;
        setSelection(0, current.duration_s);
      }
      log(JSON.stringify(data, null, 2));
      await refreshSelections();
    }

    async function saveManifest() {
      const data = await api("/api/save_manifest", {method: "POST", body: JSON.stringify({})});
      log(JSON.stringify(data, null, 2));
    }

    async function dryRun() {
      const data = await api("/api/dry_run", {
        method: "POST",
        body: JSON.stringify({selected_only: selectedOnlyInput.checked}),
      });
      log(`${outputModeText()}\n\n${data.plan}`);
    }

    async function buildOutput() {
      const data = await api("/api/build", {
        method: "POST",
        body: JSON.stringify({
          selected_only: selectedOnlyInput.checked,
          skip_source_hashes: skipHashesInput.checked,
        }),
      });
      log(`${outputModeText()}\n\n${JSON.stringify(data, null, 2)}`);
      pollStatus();
      if (!statusTimer) statusTimer = setInterval(pollStatus, 2000);
    }

    async function pollStatus() {
      const data = await api("/api/status");
      statusText.textContent = data.status.state;
      if (data.status.log) log(data.status.log);
      if (["succeeded", "failed"].includes(data.status.state) && statusTimer) {
        clearInterval(statusTimer);
        statusTimer = null;
      }
    }

    svg.addEventListener("pointerdown", (event) => {
      if (!current) return;
      dragging = true;
      dragStart = pointerTime(event);
      setSelection(dragStart, dragStart, true);
      svg.setPointerCapture(event.pointerId);
    });
    svg.addEventListener("pointermove", (event) => {
      if (!dragging || !current) return;
      setSelection(dragStart, pointerTime(event), true);
    });
    svg.addEventListener("pointerup", (event) => {
      if (!dragging) return;
      dragging = false;
      setSelection(dragStart, pointerTime(event), true);
      svg.releasePointerCapture(event.pointerId);
    });
    startInput.addEventListener("change", () => setSelection(Number(startInput.value), Number(endInput.value)));
    endInput.addEventListener("change", () => setSelection(Number(startInput.value), Number(endInput.value)));
    selectedOnlyInput.addEventListener("change", updateOutputModeInfo);
    cameraSelect.addEventListener("change", loadVideo);
    videoPlayer.addEventListener("timeupdate", () => {
      if (!current) return;
      playheadTime = Math.max(0, Math.min(current.duration_s, videoPlayer.currentTime || 0));
      videoTime.textContent = `video: ${playheadTime.toFixed(3)}s / ${current.duration_s.toFixed(3)}s`;
      drawChart();
    });
    videoPlayer.addEventListener("loadedmetadata", () => {
      if (current) setVideoTime(playheadTime);
    });
    videoPlayer.addEventListener("error", () => {
      videoPlaceholder.textContent = "Could not load this video stream.";
      videoPlaceholder.style.display = "grid";
    });
    window.addEventListener("resize", drawChart);

    document.getElementById("loadBtn").addEventListener(
      "click", () => loadEpisode().catch(err => log(err.stack || String(err))));
    document.getElementById("saveSelectionBtn").addEventListener(
      "click", () => saveSelection().catch(err => log(err.stack || String(err))));
    document.getElementById("clearBtn").addEventListener(
      "click", () => clearSelection().catch(err => log(err.stack || String(err))));
    document.getElementById("clearAllBtn").addEventListener(
      "click", () => clearAllSelections().catch(err => log(err.stack || String(err))));
    document.getElementById("saveManifestBtn").addEventListener(
      "click", () => saveManifest().catch(err => log(err.stack || String(err))));
    document.getElementById("dryRunBtn").addEventListener(
      "click", () => dryRun().catch(err => log(err.stack || String(err))));
    document.getElementById("buildBtn").addEventListener(
      "click", () => buildOutput().catch(err => log(err.stack || String(err))));
    selectedList.addEventListener("click", (event) => {
      const target = event.target;
      if (!(target instanceof HTMLElement)) return;
      const loadEpisodeId = target.dataset.loadEpisode;
      const clearEpisodeId = target.dataset.clearEpisode;
      if (loadEpisodeId !== undefined) {
        episodeSelect.value = loadEpisodeId;
        loadEpisode().catch(err => log(err.stack || String(err)));
      }
      if (clearEpisodeId !== undefined) {
        api("/api/selection", {
          method: "DELETE",
          body: JSON.stringify({episode_index: Number(clearEpisodeId)}),
        })
          .then(data => {
            log(JSON.stringify(data, null, 2));
            if (current && current.episode_index === Number(clearEpisodeId)) {
              current.selection = null;
              setSelection(0, current.duration_s);
            }
            return refreshSelections();
          })
          .catch(err => log(err.stack || String(err)));
      }
    });

    async function init() {
      config = await api("/api/config");
      episodes = await api("/api/episodes");
      document.getElementById("sourcePath").textContent = config.source;
      document.getElementById("outputPath").textContent = config.output || "(not set)";
      document.getElementById("manifestPath").textContent = config.manifest_out;
      selectedOnlyInput.checked = config.selected_only_default;
      cameraSelect.innerHTML = (config.video_keys || []).map(
        key => `<option value="${key}">${key}</option>`
      ).join("");
      cameraSelect.disabled = !(config.video_keys || []).length;
      episodeSelect.innerHTML = episodes.episodes.map(
        ep => `<option value="${ep.episode_index}">episode ${ep.episode_index} · ${ep.length} frames</option>`
      ).join("");
      if (episodes.episodes.length) episodeSelect.value = episodes.episodes[0].episode_index;
      await loadEpisode();
      await refreshSelections();
      log("Ready. Drag on the chart, then click “Snap & save episode”.");
    }

    init().catch(err => log(err.stack || String(err)));
  </script>
</body>
</html>
"""


class AppState:
    def __init__(self, args: argparse.Namespace) -> None:
        self.source = args.source.expanduser().resolve(strict=True)
        if not self.source.is_dir():
            raise NotADirectoryError(self.source)
        self.output = args.output.expanduser().resolve(strict=False) if args.output else None
        self.manifest_out = args.manifest_out.expanduser().resolve(strict=False)
        self.state_key = args.state_key
        self.action_key = args.action_key
        self.selected_only_default = args.selected_only
        self.ffmpeg = args.ffmpeg
        self.video_codec = args.video_codec
        self.video_preset = args.video_preset
        self.video_crf = args.video_crf

        self.info, self.episodes, self.episode_indices = _load_dataset(self.source)
        self.video_keys = [
            key for key, feature in self.info.get("features", {}).items() if feature.get("dtype") == "video"
        ]
        self.ranges_by_episode: dict[int, tuple[float, float]] = {}
        if args.range_manifest is not None:
            loaded = _load_range_manifest(args.range_manifest)
            self.ranges_by_episode = {
                episode_index: (start, self._episode_duration_s(episode_index) if end is None else end)
                for episode_index, (start, end) in loaded.items()
            }

        self.lock = threading.Lock()
        self.build_status: dict[str, Any] = {
            "state": "idle",
            "log": "",
            "started_at_utc": None,
            "finished_at_utc": None,
        }

    def _episode_duration_s(self, episode_index: int) -> float:
        table = _load_episode_table(self.source, self.info, episode_index, columns=["timestamp"])
        timestamps = _scalar_column(table, "timestamp", dtype=np.float64)
        return float(timestamps[-1] - timestamps[0]) if len(timestamps) else 0.0

    def config_payload(self) -> dict[str, Any]:
        return {
            "ok": True,
            "source": str(self.source),
            "output": None if self.output is None else str(self.output),
            "manifest_out": str(self.manifest_out),
            "fps": float(self.info["fps"]),
            "selected_only_default": self.selected_only_default,
            "video_keys": self.video_keys,
        }

    def video_path(self, episode_index: int, video_key: str) -> Path:
        if episode_index not in self.episode_indices:
            raise ValueError(f"Episode {episode_index} not found")
        if video_key not in self.video_keys:
            raise ValueError(f"Video key {video_key!r} is not present; available={self.video_keys}")
        video_template = self.info.get("video_path")
        if not video_template:
            raise ValueError("Dataset has video features but info.json has no video_path template")
        relative = _format_dataset_path(str(video_template), self.info, episode_index, video_key=video_key)
        path = (self.source / relative).resolve()
        try:
            path.relative_to(self.source)
        except ValueError as error:
            raise ValueError(f"Resolved video path escapes the dataset root: {path}") from error
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    def episodes_payload(self) -> dict[str, Any]:
        rows = []
        for episode in self.episodes:
            episode_index = int(episode["episode_index"])
            length = int(episode["length"])
            rows.append(
                {
                    "episode_index": episode_index,
                    "length": length,
                    "duration_s": self._episode_duration_s(episode_index),
                }
            )
        return {"ok": True, "episodes": rows}

    def episode_payload(self, episode_index: int, *, max_points: int = 1800) -> dict[str, Any]:
        if episode_index not in self.episode_indices:
            raise ValueError(f"Episode {episode_index} not found")

        features = self.info.get("features", {})
        state_key = _choose_vector_key(
            features,
            self.state_key,
            ("observation.state", "observation/state", "state"),
            "state",
            "state",
        )
        action_key = _choose_vector_key(features, self.action_key, ("action", "actions"), "action", "action")
        columns = ["timestamp", "frame_index"]
        columns.extend(key for key in (state_key, action_key) if key is not None)
        table = _load_episode_table(self.source, self.info, episode_index, columns=columns)
        timestamps = _scalar_column(table, "timestamp", dtype=np.float64)
        frame_indices = _scalar_column(table, "frame_index", dtype=np.int64)
        elapsed = timestamps - timestamps[0]
        fps = float(self.info["fps"])

        indices = _downsample_indices(len(elapsed), max_points=max_points)
        series: list[dict[str, Any]] = []
        if state_key is not None:
            state = _column_to_numpy(table, state_key)
            series.append(
                {
                    "label": f"{state_key} velocity norm",
                    "values": _robust_normalize(_motion_signal(state, fps))[indices],
                }
            )
        if action_key is not None:
            action = _column_to_numpy(table, action_key)
            series.append(
                {
                    "label": f"{action_key} velocity norm",
                    "values": _robust_normalize(_motion_signal(action, fps))[indices],
                }
            )
        if not series:
            series.append({"label": "frame position", "values": _robust_normalize(np.arange(len(elapsed)))[indices]})

        selection = None
        with self.lock:
            if episode_index in self.ranges_by_episode:
                start_s, end_s = self.ranges_by_episode[episode_index]
                selection = asdict(self._resolve_range(episode_index, start_s, end_s))

        return {
            "ok": True,
            "episode_index": episode_index,
            "fps": fps,
            "length": len(elapsed),
            "duration_s": float(elapsed[-1]),
            "time_s": elapsed[indices],
            "frame_index": frame_indices[indices],
            "series": series,
            "selection": selection,
        }

    def save_selection(self, episode_index: int, start_s: float, end_s: float) -> EpisodeRange:
        selected = self._resolve_range(episode_index, start_s, end_s)
        with self.lock:
            self.ranges_by_episode[episode_index] = (selected.trim_start_s, selected.trim_end_s)
        return selected

    def clear_selection(self, episode_index: int) -> bool:
        with self.lock:
            return self.ranges_by_episode.pop(episode_index, None) is not None

    def selections_payload(self) -> dict[str, Any]:
        with self.lock:
            raw_ranges = dict(self.ranges_by_episode)
        rows = []
        for episode_index in sorted(raw_ranges):
            start_s, end_s = raw_ranges[episode_index]
            rows.append(asdict(self._resolve_range(episode_index, start_s, end_s)))
        return {"ok": True, "episodes": rows}

    def clear_all_selections(self) -> int:
        with self.lock:
            count = len(self.ranges_by_episode)
            self.ranges_by_episode.clear()
        return count

    def save_manifest(self) -> dict[str, Any]:
        ranges = self._resolved_ranges()
        _write_range_manifest(self.manifest_out, self.source, ranges)
        return {
            "ok": True,
            "manifest": str(self.manifest_out),
            "episodes": [asdict(ranges[index]) for index in sorted(ranges)],
        }

    def dry_run_plan(self, *, selected_only: bool) -> str:
        plans = self._plans(selected_only=selected_only)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            mode = "selected-only" if selected_only else "full-dataset"
            print(f"mode: {mode} ({len(plans)} output episode(s))")
            _print_range_plan(plans, fps=float(self.info["fps"]), dry_run=True)
        return output.getvalue()

    def start_build(self, *, selected_only: bool, skip_source_hashes: bool) -> dict[str, Any]:
        if self.output is None:
            raise ValueError("OUTPUT was not provided when starting the web app")
        with self.lock:
            if self.build_status["state"] == "running":
                return {"ok": True, "message": "Build is already running", "status": self.build_status}
            self.build_status = {
                "state": "running",
                "log": "Starting build…\n",
                "started_at_utc": datetime.now(UTC).isoformat(),
                "finished_at_utc": None,
            }
        thread = threading.Thread(
            target=self._build_worker,
            kwargs={"selected_only": selected_only, "skip_source_hashes": skip_source_hashes},
            daemon=True,
        )
        thread.start()
        return {"ok": True, "message": "Build started", "status": self.build_status}

    def status(self) -> dict[str, Any]:
        with self.lock:
            return {"ok": True, "status": dict(self.build_status)}

    def _resolve_range(self, episode_index: int, start_s: float, end_s: float) -> EpisodeRange:
        table = _load_episode_table(self.source, self.info, episode_index, columns=["timestamp"])
        timestamps = _scalar_column(table, "timestamp", dtype=np.float64)
        return _snap_range_to_frames(
            episode_index=episode_index,
            timestamps=timestamps,
            requested_start_s=start_s,
            requested_end_s=end_s,
        )

    def _resolved_ranges(self) -> dict[int, EpisodeRange]:
        with self.lock:
            raw_ranges = dict(self.ranges_by_episode)
        if not raw_ranges:
            raise ValueError("No episode ranges have been selected yet")
        return {
            episode_index: self._resolve_range(episode_index, start_s, end_s)
            for episode_index, (start_s, end_s) in raw_ranges.items()
        }

    def _plans(self, *, selected_only: bool) -> list[Any]:
        ranges = self._resolved_ranges()
        ranges_by_episode = {
            episode_index: (selected.trim_start_s, selected.trim_end_s) for episode_index, selected in ranges.items()
        }
        if selected_only:
            selected_episode_ids = [
                episode_index for episode_index in self.episode_indices if episode_index in ranges_by_episode
            ]
            source_episodes = [self.episodes[episode_index] for episode_index in selected_episode_ids]
            reindex_output_episodes = True
        else:
            source_episodes = self.episodes
            reindex_output_episodes = False
        return _build_range_plans(
            self.source,
            self.info,
            source_episodes,
            ranges_by_episode,
            reindex_output_episodes=reindex_output_episodes,
        )

    def _build_worker(self, *, selected_only: bool, skip_source_hashes: bool) -> None:
        output = io.StringIO()
        try:
            if shutil.which(self.ffmpeg) is None:
                raise FileNotFoundError(f"FFmpeg executable was not found: {self.ffmpeg}")
            if self.output is None:
                raise ValueError("OUTPUT was not provided")
            source, output_path = _validate_paths(self.source, self.output, dry_run=False)
            plans = self._plans(selected_only=selected_only)
            if selected_only:
                selected_episode_ids = [plan.source_episode_index for plan in plans]
                source_episodes = [self.episodes[episode_index] for episode_index in selected_episode_ids]
            else:
                source_episodes = self.episodes
            _build_output, _dataset_file_snapshot, _provenance_source_files, _source_hashes = (
                _load_trim_writer_helpers()
            )
            with contextlib.redirect_stdout(output):
                mode = "selected-only" if selected_only else "full-dataset"
                print(f"mode: {mode} ({len(plans)} output episode(s))")
                _print_range_plan(plans, fps=float(self.info["fps"]), dry_run=False)
                source_snapshot = _dataset_file_snapshot(source)
                relative_paths = _provenance_source_files(source_snapshot, self.info, plans)
                source_hashes = {} if skip_source_hashes else _source_hashes(source, relative_paths)
                _build_output(
                    source,
                    output_path,
                    self.info,
                    source_episodes,
                    plans,
                    ffmpeg=self.ffmpeg,
                    video_codec=self.video_codec,
                    video_preset=self.video_preset,
                    video_crf=self.video_crf,
                    source_snapshot=source_snapshot,
                    source_hashes=source_hashes,
                )
                _patch_range_provenance(output_path, source, plans)
                print(f"Created visually trimmed dataset: {output_path}")
                print(f"Provenance: {output_path / 'meta' / 'trim_provenance.json'}")
            with self.lock:
                self.build_status = {
                    "state": "succeeded",
                    "log": output.getvalue(),
                    "started_at_utc": self.build_status["started_at_utc"],
                    "finished_at_utc": datetime.now(UTC).isoformat(),
                }
        except Exception:
            with self.lock:
                self.build_status = {
                    "state": "failed",
                    "log": output.getvalue() + "\n" + traceback.format_exc(),
                    "started_at_utc": self.build_status["started_at_utc"],
                    "finished_at_utc": datetime.now(UTC).isoformat(),
                }


def _downsample_indices(length: int, *, max_points: int) -> np.ndarray:
    if length <= max_points:
        return np.arange(length, dtype=np.int64)
    return np.unique(np.round(np.linspace(0, length - 1, max_points)).astype(np.int64))


def _parse_float(value: Any, *, key: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{key} must be a number") from error
    if not np.isfinite(result):
        raise ValueError(f"{key} must be finite")
    return result


def _handler_factory(state: AppState) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "LeRobotTrimHTTP/1.0"

        def log_message(self, fmt: str, *args: Any) -> None:
            message = fmt % args if args else fmt
            sys.stderr.write(f"[{self.log_date_time_string()}] {message}\n")

        def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            try:
                parsed = urlparse(self.path)
                query = parse_qs(parsed.query)
                if parsed.path == "/":
                    self._send_bytes(HTML.encode(), content_type="text/html; charset=utf-8")
                elif parsed.path == "/api/config":
                    self._send_json(state.config_payload())
                elif parsed.path == "/api/episodes":
                    self._send_json(state.episodes_payload())
                elif parsed.path == "/api/selections":
                    self._send_json(state.selections_payload())
                elif parsed.path == "/api/episode":
                    episode_index = int(query.get("episode_index", ["0"])[0])
                    max_points = int(query.get("max_points", ["1800"])[0])
                    self._send_json(state.episode_payload(episode_index, max_points=max_points))
                elif parsed.path == "/api/status":
                    self._send_json(state.status())
                elif parsed.path == "/media/video":
                    episode_index = int(query.get("episode_index", ["0"])[0])
                    video_key = query.get("video_key", [""])[0]
                    self._send_file(state.video_path(episode_index, video_key), content_type="video/mp4")
                else:
                    self._send_json({"ok": False, "error": f"Not found: {parsed.path}"}, status=404)
            except Exception as error:
                self._send_json({"ok": False, "error": str(error), "traceback": traceback.format_exc()}, status=500)

        def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            try:
                parsed = urlparse(self.path)
                body = self._read_json_body()
                if parsed.path == "/api/selection":
                    episode_index = int(body["episode_index"])
                    start_s = _parse_float(body.get("trim_start_s"), key="trim_start_s")
                    end_s = _parse_float(body.get("trim_end_s"), key="trim_end_s")
                    selected = state.save_selection(episode_index, start_s, end_s)
                    self._send_json({"ok": True, "range": asdict(selected)})
                elif parsed.path == "/api/save_manifest":
                    self._send_json(state.save_manifest())
                elif parsed.path == "/api/dry_run":
                    plan = state.dry_run_plan(selected_only=bool(body.get("selected_only", False)))
                    self._send_json({"ok": True, "plan": plan})
                elif parsed.path == "/api/build":
                    self._send_json(
                        state.start_build(
                            selected_only=bool(body.get("selected_only", False)),
                            skip_source_hashes=bool(body.get("skip_source_hashes", False)),
                        )
                    )
                else:
                    self._send_json({"ok": False, "error": f"Not found: {parsed.path}"}, status=404)
            except Exception as error:
                self._send_json({"ok": False, "error": str(error), "traceback": traceback.format_exc()}, status=500)

        def do_DELETE(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            try:
                parsed = urlparse(self.path)
                body = self._read_json_body()
                if parsed.path == "/api/selection":
                    episode_index = int(body["episode_index"])
                    existed = state.clear_selection(episode_index)
                    message = (
                        f"Cleared episode {episode_index}" if existed else f"Episode {episode_index} had no selection"
                    )
                    self._send_json({"ok": True, "message": message})
                elif parsed.path == "/api/selections":
                    count = state.clear_all_selections()
                    self._send_json({"ok": True, "message": f"Cleared {count} saved episode selection(s)"})
                else:
                    self._send_json({"ok": False, "error": f"Not found: {parsed.path}"}, status=404)
            except Exception as error:
                self._send_json({"ok": False, "error": str(error), "traceback": traceback.format_exc()}, status=500)

        def _send_file(self, path: Path, *, content_type: str) -> None:
            file_size = path.stat().st_size
            start, end = self._parse_range(file_size)
            status = 206 if self.headers.get("Range") else 200
            content_length = end - start + 1
            self.send_response(status)
            self.send_header("content-type", content_type)
            self.send_header("accept-ranges", "bytes")
            self.send_header("content-length", str(content_length))
            if status == 206:
                self.send_header("content-range", f"bytes {start}-{end}/{file_size}")
            self.send_header("cache-control", "no-store")
            self.end_headers()

            with path.open("rb") as file:
                file.seek(start)
                remaining = content_length
                while remaining > 0:
                    chunk = file.read(min(1024 * 1024, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)

        def _parse_range(self, file_size: int) -> tuple[int, int]:
            header = self.headers.get("Range")
            if not header:
                return 0, file_size - 1
            if not header.startswith("bytes=") or "," in header:
                raise ValueError(f"Unsupported Range header: {header}")
            raw_start, raw_end = header.removeprefix("bytes=").split("-", maxsplit=1)
            if raw_start == "":
                suffix_length = int(raw_end)
                if suffix_length <= 0:
                    raise ValueError(f"Invalid Range header: {header}")
                start = max(0, file_size - suffix_length)
                end = file_size - 1
            else:
                start = int(raw_start)
                end = file_size - 1 if raw_end == "" else int(raw_end)
            if start < 0 or end < start or start >= file_size:
                raise ValueError(f"Invalid Range header for {file_size} bytes: {header}")
            return start, min(end, file_size - 1)

        def _read_json_body(self) -> dict[str, Any]:
            length = int(self.headers.get("content-length", "0") or "0")
            if length == 0:
                return {}
            return json.loads(self.rfile.read(length).decode())

        def _send_json(self, value: Any, *, status: int = 200) -> None:
            payload = json.dumps(_json_safe(value), ensure_ascii=False, allow_nan=False).encode()
            self._send_bytes(payload, status=status, content_type="application/json; charset=utf-8")

        def _send_bytes(self, payload: bytes, *, status: int = 200, content_type: str) -> None:
            self.send_response(status)
            self.send_header("content-type", content_type)
            self.send_header("content-length", str(len(payload)))
            self.send_header("cache-control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

    return Handler


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", type=Path, metavar="SOURCE", help="Read-only LeRobot dataset root")
    parser.add_argument(
        "output",
        nargs="?",
        type=Path,
        metavar="OUTPUT",
        help="Optional new output dataset root; required only when clicking Build output",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind address; use 0.0.0.0 for remote browser access",
    )
    parser.add_argument("--port", type=int, default=7860, help="HTTP port")
    parser.add_argument(
        "--manifest-out",
        type=Path,
        default=Path("trim_ranges.json"),
        help="Where Save manifest writes JSON",
    )
    parser.add_argument("--range-manifest", type=Path, help="Preload existing range manifest")
    parser.add_argument(
        "--selected-only",
        action="store_true",
        help="Default UI checkbox to write only selected episodes",
    )
    parser.add_argument("--state-key", help="Numeric state feature to plot; defaults to observation.state when present")
    parser.add_argument("--action-key", help="Numeric action feature to plot; defaults to action when present")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--video-codec", default="libx264")
    parser.add_argument("--video-preset", default="medium")
    parser.add_argument("--video-crf", type=int, default=18)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        state = AppState(args)
        handler = _handler_factory(state)
        server = ThreadingHTTPServer((args.host, args.port), handler)
        host_for_display = "127.0.0.1" if args.host in {"0.0.0.0", "::"} else args.host
        print(f"LeRobot visual trim UI: http://{host_for_display}:{server.server_port}")
        print(f"SOURCE: {state.source}")
        print(f"OUTPUT: {state.output if state.output is not None else '(not set; manifest-only mode)'}")
        print("Press Ctrl+C to stop.")
        server.serve_forever(poll_interval=0.5)
        return 0
    except KeyboardInterrupt:
        print("\nStopped.")
        return 0
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    finally:
        time.sleep(0.05)


if __name__ == "__main__":
    raise SystemExit(main())
