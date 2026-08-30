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
    inference_strategy = "standard"
    rtc_config = None
    rtc_initial_delay_steps = 0
    rtc_warmed = False

    while True:
        message = recv_frame(sys.stdin.buffer)
        message_type = str(message.get("type", "")).upper()
        try:
            if message_type == "DESCRIBE":
                send_frame(
                    sys.stdout.buffer,
                    {
                        "type": "CONTRACT",
                        "contract": contract,
                        "capabilities": {
                            "inference_strategies": ["standard", "rtc"],
                            "rtc_modes": ["guided"],
                            "rtc_schedules": ["exp"],
                        },
                    },
                )
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
                    inference_strategy, rtc_config, rtc_initial_delay_steps = _parse_inference_config(
                        message.get("inference_config"), policy, train_config
                    )
                    _warmup_if_requested(policy, contract, message)
                send_frame(
                    sys.stdout.buffer,
                    {
                        "type": "READY",
                        "checkpoint_id": contract["checkpoint_id"],
                        "contract_hash": contract["contract_hash"],
                        "inference_strategy": inference_strategy,
                    },
                )
            elif message_type == "INFER":
                if policy is None:
                    raise RuntimeError("inferencer is not loaded; send LOAD first")
                started = time.perf_counter()
                observation = _build_policy_observation(message["observation"], contract)
                rtc_context = message.get("rtc_context")
                if inference_strategy == "rtc" and rtc_context is not None:
                    previous_actions = np.asarray(rtc_context.get("prev_chunk_left_over"), dtype=np.float32)
                    if previous_actions.size > 0:
                        result = policy.infer_rtc(
                            observation,
                            previous_actions=previous_actions,
                            inference_delay=int(rtc_context.get("predicted_delay_steps", 0)),
                            rtc_config=rtc_config,
                        )
                    else:
                        result = policy.infer(observation)
                else:
                    result = policy.infer(observation)
                if inference_strategy == "rtc" and not rtc_warmed:
                    # The first request has no previous chunk.  Compile the RTC
                    # VJP path before the first chunk is released to the actor,
                    # while startup is still stationary, and discard its output.
                    policy.infer_rtc(
                        observation,
                        previous_actions=np.asarray(result["actions"], dtype=np.float32),
                        inference_delay=rtc_initial_delay_steps,
                        rtc_config=rtc_config,
                    )
                    rtc_warmed = True
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


def _parse_inference_config(payload: Any, policy, train_config):
    payload = dict(payload or {})
    strategy = str(payload.get("strategy", "standard")).strip().lower()
    if strategy not in ("standard", "rtc"):
        raise ValueError(f"unsupported inference strategy: {strategy!r}")
    if strategy == "standard":
        return strategy, None, 0
    if str(train_config.model.model_type.value) not in ("pi0", "pi05") or not policy.supports_rtc:
        raise RuntimeError("inference-time RTC is only supported for PI0/PI0.5 policies")

    from openpi.policies.rtc import RTCConfig  # noqa: PLC0415

    rtc_payload = dict(payload.get("rtc", {}) or {})
    rtc_config = RTCConfig(
        prefix_horizon=int(rtc_payload.get("prefix_horizon", 8)),
        max_guidance_weight=float(rtc_payload.get("max_guidance_weight", 5.0)),
        schedule=str(rtc_payload.get("schedule", "exp")),
    )
    rtc_config.validate(int(train_config.model.action_horizon))
    initial_delay_steps = int(rtc_payload.get("initial_delay_steps", 0))
    if initial_delay_steps < 0:
        raise ValueError("RTC initial_delay_steps must be >= 0")
    return strategy, rtc_config, initial_delay_steps


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
