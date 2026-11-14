# ruff: noqa: PLC0415
"""Stateful RDP inference: one fresh observation -> one absolute action.

The caller must provide tactile fields and timestamps in the same clock as
observation timestamp. There is no fallback to a zero tactile observation.
"""

import argparse
import contextlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

import cv2
import numpy as np

from openpi.deploy.contract import load_policy_contract
from openpi.deploy.policy_inferencer import _build_policy_observation
from openpi.deploy.policy_inferencer import _group_actions
from openpi.deploy.protocol import recv_frame
from openpi.deploy.protocol import send_frame
from openpi.rdp.prepare import sha256
from openpi.rdp.tactile import FINGERS
from openpi.rdp.tactile import MARKER_KEYS
from openpi.rdp.tactile import TactilePCA
from openpi.rdp.tactile import offsets


class StreamingRDP:
    def __init__(self, checkpoint, *, device="cuda"):
        import torch

        from openpi import transforms
        from openpi.rdp.model import AsymmetricTokenizer
        from openpi.rdp.model import LatentDiffusionPolicy
        from openpi.rdp.model import RDPConfig
        from openpi.shared import normalize
        from openpi.training import config_io

        self.torch = torch
        self.root = Path(checkpoint)
        integrity = json.loads((self.root / "_CHECKPOINT_METADATA").read_text())
        if not integrity.get("complete"):
            raise ValueError("incomplete checkpoint")
        for name, expected in integrity["sha256"].items():
            if sha256(self.root / name) != expected:
                raise ValueError(f"checkpoint hash mismatch: {name}")
        self.contract = load_policy_contract(self.root)
        self.description = json.loads((self.root / "rdp_model.json").read_text())
        self.cfg = RDPConfig(**self.description["config"])
        self.device = torch.device(device)
        self.at = AsymmetricTokenizer(self.cfg).to(self.device).eval()
        self.ldp = LatentDiffusionPolicy(self.cfg).to(self.device).eval()
        for name, model in (("at", self.at), ("ldp", self.ldp)):
            saved = torch.load(self.root / f"params/{name}.pt", map_location="cpu", weights_only=False)
            model.load_state_dict(saved["model"])
        io = config_io.load_train_config(self.root)
        data = io.data.create(io.assets_dirs, io.model)
        stats = normalize.load(self.root / "assets")
        self.input_transform = transforms.compose(
            (*data.data_transforms.inputs, transforms.Normalize(stats, use_quantiles=data.use_quantile_norm))
        )
        self.output_transform = transforms.compose(
            (transforms.Unnormalize(stats, use_quantiles=data.use_quantile_norm), *data.data_transforms.outputs)
        )
        self.pca = [TactilePCA.load(self.root / "assets/pca" / f"{f}.npz") for f in FINGERS]
        self.reset()

    def reset(self):
        self.latent = self.hidden = self.anchor = self.held = None
        self.step = 0
        self.last_poll = self.last_time = None
        self.last_marker_stamps = None

    def tactile(self, observation):
        now = float(observation["timestamp"])
        if not np.isfinite(now) or (self.last_time is not None and now <= self.last_time):
            raise ValueError("observation timestamp must increase; RESET between episodes")
        supplied = observation.get("tactile", {})
        stamps = np.array([float(supplied[key]["timestamp"]) for key in MARKER_KEYS])
        if not np.isfinite(stamps).all() or np.any(stamps > now + 1e-6):
            raise ValueError("invalid/future tactile timestamp")
        if np.any(now - stamps > self.description["max_tactile_age_s"]):
            raise ValueError("stale tactile input")
        if self.last_marker_stamps is not None and np.any(stamps < self.last_marker_stamps):
            raise ValueError("tactile clock moved backward")
        if self.last_poll is None or now - self.last_poll >= 1 / self.description["tactile_hz"] - 1e-5:
            self.held = np.concatenate(
                [pca.encode(offsets(supplied[key]["markers"])) for key, pca in zip(MARKER_KEYS, self.pca, strict=True)]
            )
            self.last_poll = now
        self.last_time, self.last_marker_stamps = now, stamps
        return np.r_[self.held, (now - self.last_poll) * self.description["tactile_hz"]].astype(np.float32)

    def infer(self, observation):
        torch = self.torch
        if self.last_time is not None and float(observation["timestamp"]) - self.last_time > 0.5:
            self.reset()
        tactile = self.tactile(observation)
        raw = _build_policy_observation(observation, self.contract)
        transformed = self.input_transform(raw)
        feedback = torch.as_tensor(tactile, device=self.device)[None, None]
        started = time.perf_counter()
        replanned = self.latent is None or self.step >= self.cfg.horizon
        with torch.inference_mode():
            if replanned:
                images = np.stack(
                    [
                        cv2.resize(raw[key], (self.cfg.image_size, self.cfg.image_size), interpolation=cv2.INTER_AREA)
                        for key in (
                            "observation/image",
                            "observation/left_wrist_image",
                            "observation/right_wrist_image",
                        )
                    ]
                )
                self.latent = self.ldp.sample(
                    torch.as_tensor(images, device=self.device)[None],
                    torch.as_tensor(transformed["state"], dtype=torch.float32, device=self.device)[None],
                    feedback[:, 0],
                )
                self.hidden = None
                self.anchor = transformed["state"].copy()
                self.step = 0
            actions, self.hidden = self.at.decode(self.latent, feedback, self.hidden)
            self.step += 1
        output = self.output_transform({"state": self.anchor.copy(), "actions": actions[0].cpu().numpy()})
        return {
            "actions": output["actions"],
            "timing": {
                "total_ms": (time.perf_counter() - started) * 1000,
                "replanned": replanned,
                "chunk_position": self.step - 1,
            },
        }


def serve(checkpoint, *, protocol_in=None, protocol_out=None):
    stream_in = sys.stdin.buffer if protocol_in is None else protocol_in
    stream_out = sys.stdout.buffer if protocol_out is None else protocol_out
    contract = load_policy_contract(checkpoint)
    policy = None
    while True:
        message = recv_frame(stream_in)
        kind = message.get("type", "").upper()
        try:
            if kind == "DESCRIBE":
                reply = {
                    "type": "CONTRACT",
                    "contract": contract,
                    "capabilities": {"inference_strategies": ["standard"], "streaming_tactile": True},
                }
            elif kind == "LOAD":
                if policy is None:
                    policy = StreamingRDP(
                        checkpoint, device=message.get("device", os.environ.get("BRAINCO_RDP_DEVICE", "cuda"))
                    )
                reply = {
                    "type": "READY",
                    "checkpoint_id": contract["checkpoint_id"],
                    "contract_hash": contract["contract_hash"],
                    "inference_strategy": "standard",
                }
            elif kind == "INFER":
                if policy is None:
                    raise RuntimeError("send LOAD first")
                result = policy.infer(message["observation"])
                reply = {
                    "type": "RESULT",
                    "request_id": int(message["request_id"]),
                    "contract_hash": contract["contract_hash"],
                    "actions": _group_actions(result["actions"], contract),
                    "timing": result["timing"],
                }
            elif kind == "RESET":
                if policy is not None:
                    policy.reset()
                reply = {"type": "RESET_DONE"}
            elif kind == "CLOSE":
                send_frame(stream_out, {"type": "CLOSED"})
                return
            else:
                raise ValueError(f"unsupported message {kind}")
            send_frame(stream_out, reply)
        except Exception as exc:
            if policy is not None:
                policy.reset()
            send_frame(
                stream_out,
                {
                    "type": "ERROR",
                    "request_id": message.get("request_id"),
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                },
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    with contextlib.redirect_stdout(sys.stderr):
        serve(args.checkpoint, protocol_in=sys.stdin.buffer, protocol_out=sys.__stdout__.buffer)
