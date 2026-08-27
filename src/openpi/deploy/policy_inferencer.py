"""Model-agnostic BrainCo Policy Inferencer for revo_deploy."""

from __future__ import annotations

import argparse
import contextlib
from pathlib import Path
import sys
import time
import traceback
from typing import Any

import numpy as np

from openpi.deploy.contract import load_policy_contract
from openpi.deploy.protocol import recv_frame
from openpi.deploy.protocol import send_frame


class _ValidateImages:
    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        for key in (
            "observation/image",
            "observation/left_wrist_image",
            "observation/right_wrist_image",
        ):
            image = np.asarray(data.get(key))
            if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
                raise ValueError(f"{key} must be RGB uint8 HWC, got {image.shape}/{image.dtype}")
        return data


def serve(checkpoint: str | Path) -> None:
    checkpoint = Path(checkpoint).expanduser().resolve()
    contract = load_policy_contract(checkpoint)
    train_config = None
    policy = None

    while True:
        message = recv_frame(sys.stdin.buffer)
        message_type = str(message.get("type", "")).upper()
        try:
            if message_type == "DESCRIBE":
                send_frame(sys.stdout.buffer, {"type": "CONTRACT", "contract": contract})
            elif message_type == "LOAD":
                if policy is None:
                    # Keep DESCRIBE lightweight: importing JAX/model modules is
                    # intentionally delayed until the LOAD phase.
                    from openpi import transforms  # noqa: PLC0415
                    from openpi.policies import policy_config  # noqa: PLC0415
                    from openpi.training import config_io  # noqa: PLC0415

                    train_config = config_io.load_train_config(checkpoint)
                    policy = policy_config.create_trained_policy(
                        train_config,
                        checkpoint,
                        repack_transforms=transforms.Group(inputs=(_ValidateImages(),)),
                        sample_kwargs=_sample_kwargs(train_config),
                    )
                    _warmup_if_requested(policy, contract, message)
                send_frame(
                    sys.stdout.buffer,
                    {
                        "type": "READY",
                        "checkpoint_id": contract["checkpoint_id"],
                        "contract_hash": contract["contract_hash"],
                    },
                )
            elif message_type == "INFER":
                if policy is None:
                    raise RuntimeError("inferencer is not loaded; send LOAD first")
                started = time.perf_counter()
                observation = _build_policy_observation(message["observation"], contract)
                result = policy.infer(observation)
                actions = np.asarray(result["actions"], dtype=np.float32)
                grouped = _group_actions(actions, contract)
                timing = dict(result.get("policy_timing", {}) or {})
                timing["inferencer_total_ms"] = (time.perf_counter() - started) * 1000.0
                send_frame(
                    sys.stdout.buffer,
                    {
                        "type": "RESULT",
                        "request_id": int(message["request_id"]),
                        "actions": grouped,
                        "timing": timing,
                    },
                )
            elif message_type == "RESET":
                send_frame(sys.stdout.buffer, {"type": "RESET_DONE"})
            elif message_type == "CLOSE":
                send_frame(sys.stdout.buffer, {"type": "CLOSED"})
                return
            else:
                raise ValueError(f"unsupported message type: {message_type!r}")
        except Exception as exc:
            send_frame(
                sys.stdout.buffer,
                {
                    "type": "ERROR",
                    "request_id": message.get("request_id"),
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                },
            )


def _build_policy_observation(observation: dict[str, Any], contract: dict[str, Any]) -> dict[str, Any]:
    received_groups = observation.get("joint_groups", {})
    parts: list[np.ndarray] = []
    for group, expected_names in contract["required_joint_groups"].items():
        received = received_groups.get(group)
        if not isinstance(received, dict):
            raise ValueError(f"observation missing joint group {group!r}")
        names = [str(name) for name in received.get("names", [])]
        positions = np.asarray(received.get("positions"), dtype=np.float32)
        if positions.shape != (len(names),):
            raise ValueError(f"joint group {group!r} names/positions mismatch")
        index = {name: i for i, name in enumerate(names)}
        missing = [name for name in expected_names if name not in index]
        if missing:
            raise ValueError(f"joint group {group!r} missing names: {missing}")
        parts.append(np.asarray([positions[index[name]] for name in expected_names], dtype=np.float32))

    images = observation.get("images", {})
    result: dict[str, Any] = {
        "observation/state": np.concatenate(parts).astype(np.float32, copy=False),
        "prompt": str(observation.get("task", "")),
    }
    camera_bindings = {
        "observation.images.cam_head": "observation/image",
        "observation.images.cam_left_wrist": "observation/left_wrist_image",
        "observation.images.cam_right_wrist": "observation/right_wrist_image",
    }
    for key in contract["required_cameras"]:
        if key not in images:
            raise ValueError(f"observation missing camera {key!r}")
        try:
            policy_key = camera_bindings[key]
        except KeyError as exc:
            raise ValueError(f"BrainCo inferencer does not support camera key {key!r}") from exc
        result[policy_key] = np.asarray(images[key])
    return result


def _group_actions(actions: np.ndarray, contract: dict[str, Any]) -> dict[str, np.ndarray]:
    if actions.ndim != 2:
        raise ValueError(f"policy actions must be [steps, dim], got {actions.shape}")
    groups: dict[str, np.ndarray] = {}
    offset = 0
    for group, names in contract["output_joint_groups"].items():
        width = len(names)
        groups[group] = np.ascontiguousarray(actions[:, offset : offset + width], dtype=np.float32)
        offset += width
    if offset != actions.shape[1]:
        raise ValueError(f"contract action dim {offset} does not match policy output {actions.shape[1]}")
    return groups


def _sample_kwargs(train_config) -> dict[str, Any]:
    if str(train_config.model.model_type.value) in ("pi0", "pi05"):
        return {"num_steps": 10}
    return {}


def _warmup_if_requested(policy, contract: dict[str, Any], message: dict[str, Any]) -> None:
    observation = message.get("warmup_observation")
    if observation is not None:
        policy.infer(_build_policy_observation(observation, contract))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    # EOF is the normal fallback when the parent gateway exits before it can
    # complete a framed CLOSE handshake.
    with contextlib.suppress(KeyboardInterrupt, EOFError):
        serve(args.checkpoint)


if __name__ == "__main__":
    main()
