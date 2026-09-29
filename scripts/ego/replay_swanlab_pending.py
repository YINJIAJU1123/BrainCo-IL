#!/usr/bin/env python3
"""Replay persisted metric uploads to the stable SwanLab run after an outage."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import swanlab

from openpi.training.config_io import load_train_config


def replay(run_dir: Path) -> int:
    run_id = (run_dir / "swanlab_id.txt").read_text().strip()
    if not run_id:
        raise ValueError("stable SwanLab run ID is empty")
    config = load_train_config(run_dir)
    pending = run_dir / "swanlab_pending.jsonl"
    if not pending.is_file():
        return 0
    rows = [json.loads(line) for line in pending.read_text().splitlines() if line.strip()]
    run = swanlab.init(project=config.project_name, experiment_name=config.exp_name,
                       name=config.exp_name, id=run_id, resume="must",
                       mode=os.environ.get("SWANLAB_MODE", "local"),
                       logdir=os.environ.get("SWANLAB_LOG_DIR", "./swanlog"))
    uploaded = 0
    try:
        for row in rows:
            step = int(row.pop("step"))
            swanlab.log(row, step=step)
            uploaded += 1
    finally:
        swanlab.finish()
    # Keep the original file until all log calls and finish() complete.
    if uploaded == len(rows):
        pending.replace(run_dir / "swanlab_pending.replayed.jsonl")
    return uploaded


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    print(json.dumps({"uploaded": replay(args.run_dir)}))


if __name__ == "__main__":
    main()
