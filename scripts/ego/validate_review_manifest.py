#!/usr/bin/env python3
"""Validate a human-approved review manifest and calculate final quotas."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def validate(source: Path, manifest_path: Path, *, horizon: int = 16) -> dict:
    manifest = json.loads(manifest_path.read_text())
    info_path = source / "meta/info.json"
    source_hash = hashlib.sha256(info_path.read_bytes()).hexdigest()
    if manifest.get("schema_version") != 1 or manifest.get("source_info_sha256") != source_hash:
        raise ValueError("review manifest schema or source fingerprint differs")
    episodes = {int(row["episode_index"]): int(row["length"]) for row in (
        json.loads(line) for line in (source / "meta/episodes.jsonl").read_text().splitlines()
    )}
    by_episode = {episode: [] for episode in episodes}
    for interval in manifest.get("intervals", []):
        episode = int(interval["episode_index"])
        if episode not in by_episode:
            raise ValueError(f"unknown episode {episode}")
        by_episode[episode].append(interval)
    kept = 0
    for episode, intervals in by_episode.items():
        cursor = 0
        for interval in sorted(intervals, key=lambda x: x["start_frame"]):
            start, stop = int(interval["start_frame"]), int(interval["end_frame_exclusive"])
            if start != cursor or stop <= start or stop > episodes[episode]:
                raise ValueError(f"episode {episode}: review intervals overlap, leave a gap, or exceed length")
            decision = interval.get("decision")
            if decision not in ("keep", "exclude"):
                raise ValueError(f"episode {episode}: every interval needs a human keep/exclude decision")
            if decision == "exclude" and not str(interval.get("reason", "")).strip():
                raise ValueError(f"episode {episode}: excluded interval has no reason")
            if decision == "keep":
                if stop - start < horizon:
                    raise ValueError(f"episode {episode}: kept interval is shorter than horizon {horizon}")
                kept += stop - start
            cursor = stop
        if cursor != episodes[episode]:
            raise ValueError(f"episode {episode}: review does not cover all source frames")
    if kept == 0:
        raise ValueError("no reviewed training starts")
    rounded = 32 * math.ceil(kept / 32)
    return {"reviewed_real_starts": kept, "rounded_real_starts": rounded,
            "real_padding_repeats": rounded - kept,
            "steps": {"posttrain_e1_r1": rounded * 2 // 16,
                      "posttrain_e3_r1": rounded * 4 // 16,
                      "posttrain_e5_r2": rounded * 7 // 2 // 16}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    print(json.dumps(validate(args.source, args.manifest), indent=2))


if __name__ == "__main__":
    main()
