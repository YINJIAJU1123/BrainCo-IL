"""BrainCo OpenPI deployment plugin.

This module is imported by deployment systems that need to recover the runtime
contract for a trained checkpoint without knowing BrainCo-IL internals.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from openpi.policies import policy_config
from openpi.training import config as training_config

API_VERSION = 1
DEFAULT_CAMERA_BINDINGS = {
    "base": "observation.images.cam_head",
    "left_wrist": "observation.images.cam_left_wrist",
    "right_wrist": "observation.images.cam_right_wrist",
}


def describe_policy(
    checkpoint_dir: str | Path,
    *,
    policy_id: str,
    runtime_options: dict[str, Any] | None = None,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the deployment contract for a BrainCo policy checkpoint."""

    del checkpoint_dir, runtime_options
    train_config = training_config.get_config(policy_id)
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    spec = _base_spec(
        policy_id=policy_id,
        action_dim=int(train_config.model.action_dim),
        action_horizon=int(train_config.model.action_horizon),
        asset_id=data_config.asset_id,
    )
    return _deep_update(spec, overrides or {})


def create_policy(
    checkpoint_dir: str | Path,
    *,
    policy_id: str,
    runtime_options: dict[str, Any] | None = None,
):
    """Load a trained BrainCo policy checkpoint."""

    opts = dict(runtime_options or {})
    train_config = training_config.get_config(policy_id)
    return policy_config.create_trained_policy(
        train_config,
        checkpoint_dir,
        sample_kwargs={"num_steps": int(opts.get("num_inference_steps", 10) or 10)},
        default_prompt=str(opts.get("default_prompt", "") or "") or None,
        pytorch_device=str(opts.get("device", "") or "") or None,
    )


def _base_spec(
    *,
    policy_id: str,
    action_dim: int,
    action_horizon: int,
    asset_id: str | None,
) -> dict[str, Any]:
    if action_dim != 56:
        raise RuntimeError(f"BrainCo deploy plugin expects 56D actions, got {action_dim}")
    return {
        "schema_version": API_VERSION,
        "policy_id": policy_id,
        "policy_type": "pi05",
        "action_dim": action_dim,
        "action_horizon": action_horizon,
        "asset_id": asset_id,
        "inputs": {
            "state_key": "observation.state",
            "policy_input_map": {
                "observation/state": "observation.state",
                "observation/image": "observation.images.cam_head",
                "observation/left_wrist_image": "observation.images.cam_left_wrist",
                "observation/right_wrist_image": "observation.images.cam_right_wrist",
            },
            "state_composition": [
                {"kind": "joint_pos", "group": "left_arm", "dim": 7},
                {"kind": "joint_pos", "group": "right_arm", "dim": 7},
                {"kind": "joint_pos", "group": "left_hand", "dim": 21},
                {"kind": "joint_pos", "group": "right_hand", "dim": 21},
            ],
            "image_shape": [3, 224, 224],
            "camera_bindings": dict(DEFAULT_CAMERA_BINDINGS),
        },
        "outputs": {
            "action_key": "action",
            "space": "joint_position",
            "joint_groups": {
                "left_arm": {"indices": list(range(7)), "action_mode": "absolute"},
                "right_arm": {"indices": list(range(7, 14)), "action_mode": "absolute"},
                "left_hand": {"indices": list(range(14, 35)), "action_mode": "absolute"},
                "right_hand": {"indices": list(range(35, 56)), "action_mode": "absolute"},
            },
        },
        "dataset": {
            "state_key": "observation.state",
            "state_dim": 70,
            "state_slice": [14, 70],
            "task_key": "task",
            "task_index_key": "task_index",
            "tasks_path": "meta/tasks.parquet",
        },
        "runtime_options": {
            "num_inference_steps": 10,
            "policy_rate": 30.0,
        },
    }


def _deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_update(out[key], value)
        else:
            out[key] = value
    return out
