#!/usr/bin/env python3
"""Package a BrainCo checkpoint for the vla_deploy checkpoint-artifact contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import yaml


# The schema module is intentionally dependency-free so this legacy manual
# exporter also works without importing the JAX policy runtime.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from openpi.policies.brainco_contract import JOINT_GROUPS


CAMERAS = [
    "observation.images.cam_head",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
]


def _atomic_write(path: Path, text: str) -> None:
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(text, encoding="utf-8")
    os.replace(temp, path)


def _sha256_json(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def package_checkpoint(root: Path, *, robot_interface: str, contract_only: bool) -> dict:
    train_path = root / "train_config.yaml"
    if not train_path.is_file() or not (root / "params").is_dir():
        raise FileNotFoundError(f"not a checkpoint directory: {root}")
    raw = yaml.safe_load(train_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise ValueError(f"unsupported train_config.yaml format: {train_path}")
    train_payload = raw.get("train_config")
    if not isinstance(train_payload, dict):
        raise ValueError("train_config.yaml has no train_config payload")
    fields = train_payload.get("fields", {})
    model_fields = fields.get("model", {}).get("fields", {})
    data_fields = fields.get("data", {}).get("fields", {})
    io_fields = data_fields.get("policy_io", {}).get("fields", {})
    action_dim = int(model_fields.get("action_dim", 0))
    chunk_steps = int(model_fields.get("action_horizon", 0))
    action_groups = io_fields.get("action_groups", list(JOINT_GROUPS))
    state_groups = io_fields.get("state_groups", list(JOINT_GROUPS))
    if action_groups != list(JOINT_GROUPS) or state_groups != list(JOINT_GROUPS):
        raise ValueError("deployment protocol requires the ordered dual-arm 56D joint groups")
    expected_dim = sum(len(names) for names in JOINT_GROUPS.values())
    if action_dim != expected_dim or chunk_steps <= 0:
        raise ValueError(f"expected action_dim={expected_dim} and positive action_horizon, got {action_dim}/{chunk_steps}")
    checkpoint_id = str(raw.get("checkpoint_id") or f"{fields.get('name', root.parent.name)}_step{root.name}")
    head_camera = data_fields.get("head_camera_key", CAMERAS[0])
    cameras = [head_camera, *CAMERAS[1:]]
    policy_rate_hz = float(fields.get("policy_rate_hz", 30.0))
    if policy_rate_hz <= 0:
        raise ValueError("policy_rate_hz must be positive")
    unsigned = {
        "generated": True,
        "do_not_edit": True,
        "generated_by": "BrainCo-IL",
        "protocol_version": 1,
        "checkpoint_id": checkpoint_id,
        "robot_interface": robot_interface,
        "required_joint_groups": JOINT_GROUPS,
        "required_cameras": cameras,
        "output_joint_groups": JOINT_GROUPS,
        "chunk_steps": chunk_steps,
        "policy_rate_hz": policy_rate_hz,
    }
    contract = dict(unsigned)
    contract["contract_hash"] = _sha256_json(unsigned)
    _atomic_write(root / "policy_contract.json", json.dumps(contract, ensure_ascii=False, indent=2) + "\n")

    if not contract_only:
        wrapped = {
            "schema_version": 1,
            "generated": True,
            "do_not_edit": True,
            "generated_by": "BrainCo-IL",
            "checkpoint_id": checkpoint_id,
            "content_sha256": _sha256_json(train_payload),
            "train_config": train_payload,
        }
        _atomic_write(train_path, yaml.safe_dump(wrapped, sort_keys=False, allow_unicode=False))
    return {
        "checkpoint": str(root),
        "checkpoint_id": checkpoint_id,
        "contract_hash": contract["contract_hash"],
        "action_dim": action_dim,
        "chunk_steps": chunk_steps,
        "policy_rate_hz": policy_rate_hz,
        "train_config_wrapped": not contract_only,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--robot-interface", default="revotron_mit_revo3_mit_3cam")
    parser.add_argument("--contract-only", action="store_true", help="do not add deployment metadata to train_config.yaml")
    args = parser.parse_args()
    print(json.dumps(package_checkpoint(args.checkpoint.resolve(), robot_interface=args.robot_interface, contract_only=args.contract_only), indent=2))


if __name__ == "__main__":
    main()
