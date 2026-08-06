"""BrainCo OpenPI deployment plugin.

This module is imported by deployment systems that need to recover the runtime
contract for a trained checkpoint without knowing BrainCo-IL internals.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from openpi import transforms
from openpi.models import model as model_lib
from openpi.policies import brainco_policy
from openpi.policies import policy_config
from openpi.training import config_io

API_VERSION = 1
DEFAULT_CAMERA_BINDINGS = {
    "base": "observation.images.cam_head",
    "left_wrist": "observation.images.cam_left_wrist",
    "right_wrist": "observation.images.cam_right_wrist",
}
RAW_IMAGE_CONTRACT = {
    "shape": [-1, -1, 3],
    "layout": "hwc",
    "dtype": "uint8",
    "color": "rgb",
    "scale": "0to255",
    "preprocessing_owner": "plugin",
}
POLICY_IMAGE_KEYS = (
    "observation/image",
    "observation/left_wrist_image",
    "observation/right_wrist_image",
)


class ValidateRawImages:
    """Fail fast unless the deploy caller follows the raw image contract."""

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        for key in POLICY_IMAGE_KEYS:
            if key not in data:
                raise ValueError(f"raw observation missing required image '{key}'")
            image = np.asarray(data[key])
            if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
                raise ValueError(
                    f"raw observation '{key}' must be RGB uint8 HWC, got shape={image.shape} dtype={image.dtype}"
                )
        return data


def describe_policy(
    checkpoint_dir: str | Path,
    *,
    runtime_options: dict[str, Any] | None = None,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the deployment contract for a BrainCo policy checkpoint."""

    del runtime_options
    train_config = config_io.load_train_config(checkpoint_dir)
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    policy_io = _policy_io(train_config)
    spec = _base_spec(
        config_name=train_config.name,
        policy_type=_policy_type(train_config),
        action_dim=int(train_config.model.action_dim),
        action_horizon=int(train_config.model.action_horizon),
        asset_id=data_config.asset_id,
        policy_io=policy_io,
    )
    if execution := _vlash_execution_spec(train_config, data_config):
        spec["execution"] = execution
    return _deep_update(spec, overrides or {})


def create_policy(
    checkpoint_dir: str | Path,
    *,
    runtime_options: dict[str, Any] | None = None,
):
    """Load a trained BrainCo policy checkpoint."""

    opts = dict(runtime_options or {})
    train_config = config_io.load_train_config(checkpoint_dir)
    return policy_config.create_trained_policy(
        train_config,
        checkpoint_dir,
        repack_transforms=transforms.Group(inputs=(ValidateRawImages(),)),
        sample_kwargs=_sample_kwargs(train_config, opts),
        default_prompt=str(opts.get("default_prompt", "") or "") or None,
    )


def _base_spec(
    *,
    config_name: str,
    policy_type: str,
    action_dim: int,
    action_horizon: int,
    asset_id: str | None,
    policy_io: brainco_policy.BrainCoPolicyIOConfig,
) -> dict[str, Any]:
    policy_io.validate(action_dim)
    action_groups: dict[str, dict[str, Any]] = {}
    offset = 0
    for group in policy_io.action_groups:
        group_dim = policy_io.group_dim(group)
        action_groups[group] = {
            "indices": list(range(offset, offset + group_dim)),
            "action_mode": "absolute",
        }
        offset += group_dim
    dataset = {
        "state_key": "observation.state",
        "state_dim": policy_io.dataset_state_dim,
        "task_key": "task",
        "task_index_key": "task_index",
        "tasks_path": "meta/tasks.jsonl",
    }
    if policy_io.dataset_state_indices is not None:
        dataset["state_indices"] = list(policy_io.dataset_state_indices)
    return {
        "schema_version": API_VERSION,
        "config_name": config_name,
        "policy_type": policy_type,
        "action_dim": action_dim,
        "action_horizon": action_horizon,
        "asset_id": asset_id,
        "observation_contract": {
            "version": 1,
            "images": "raw_rgb_uint8_hwc",
            "preprocessing_owner": "plugin",
        },
        "inputs": {
            "state_key": "observation.state",
            "state_dim": policy_io.state_dim,
            "policy_input_map": {
                "observation/state": "observation.state",
                "observation/image": "observation.images.cam_head",
                "observation/left_wrist_image": "observation.images.cam_left_wrist",
                "observation/right_wrist_image": "observation.images.cam_right_wrist",
            },
            "state_composition": [
                {"kind": "joint_pos", "group": group, "dim": policy_io.group_dim(group)}
                for group in policy_io.state_groups
            ],
            "image_contract": dict(RAW_IMAGE_CONTRACT),
            "camera_bindings": dict(DEFAULT_CAMERA_BINDINGS),
        },
        "outputs": {
            "action_key": "action",
            "space": "joint_position",
            "joint_groups": action_groups,
        },
        "dataset": dataset,
        "runtime_options": {
            "policy_rate": 30.0,
            **({"num_inference_steps": 10} if policy_type in ("pi0", "pi05") else {}),
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


def _policy_type(train_config) -> str:
    model_type = train_config.model.model_type
    if isinstance(model_type, model_lib.ModelType):
        return model_type.value
    return str(model_type)


def _policy_io(train_config) -> brainco_policy.BrainCoPolicyIOConfig:
    policy_io = getattr(train_config.data, "policy_io", None)
    if not isinstance(policy_io, brainco_policy.BrainCoPolicyIOConfig):
        raise RuntimeError("BrainCo checkpoint train_config.data must provide a BrainCoPolicyIOConfig")
    return policy_io


def _sample_kwargs(train_config, opts: dict[str, Any]) -> dict[str, Any]:
    policy_type = _policy_type(train_config)
    if policy_type in ("pi0", "pi05"):
        return {"num_steps": int(opts.get("num_inference_steps", 10) or 10)}
    return {}


def _vlash_execution_spec(train_config, data_config) -> dict[str, Any] | None:
    """描述 checkpoint 的 VLASH future-state/异步执行语义.

    policy 仍然是 PI0.5;revo_deploy 只需把交接时刻的 raw absolute
    future state 放进标准 observation.state,模型/归一化接口无需变化.
    """
    max_delay_steps = int(getattr(data_config, "max_delay_steps", 0) or 0)
    if max_delay_steps <= 0:
        return None

    model = train_config.model
    use_state_ground_truth = bool(getattr(data_config, "use_state_ground_truth", True))
    return {
        "schema_version": 1,
        "mode": "vlash_async",
        "supports_future_state": True,
        "future_state_key": "observation.state",
        "future_state_semantics": "predicted_state_at_chunk_handoff",
        "future_state_space": "raw_absolute_joint_position",
        "requires_unblended_chunk_boundaries": True,
        "requires_low_pass_disabled": True,
        "max_trained_delay_steps": max_delay_steps,
        "training_future_state_source": (
            "ground_truth" if use_state_ground_truth else "action_proxy"
        ),
        "state_conditioning": "adarms" if bool(getattr(model, "state_cond", False)) else "discrete_prompt",
        "offset_sampling": "uniform_inclusive",
        "shared_observation_training": bool(getattr(data_config, "shared_observation", False)),
    }
