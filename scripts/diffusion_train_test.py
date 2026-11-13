"""Exercise real LeRobot IO, training/resume, export and subprocess inference."""

import dataclasses
import functools
import io
import json
import os
from pathlib import Path
import subprocess
import sys

from flax import nnx
import jax
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.video_utils import encode_video_frames
import numpy as np
import pytest
import yaml

from openpi.deploy import contract
from openpi.deploy import protocol
from openpi.deploy.export_checkpoint import export_checkpoint
from openpi.models import model as model_lib
from openpi.training import config_io

from . import compute_norm_stats
from . import train


def test_diffusion_train_resume_export_and_infer(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "lerobot.common.datasets.lerobot_dataset.encode_video_frames",
        functools.partial(encode_video_frames, vcodec="h264"),
    )
    root = tmp_path / "dataset"
    # Interspersed EEF metadata must not leak into the named hardware interface.
    names = [
        "eef_pose_x",
        *(f"revoarm_left_joint{i}" for i in range(1, 8)),
        *(f"revoarm_right_joint{i}" for i in range(1, 8)),
        *(
            f"left_{finger}_joint"
            for finger in (
                "thumb_proximal",
                "thumb_metacarpal",
                "index_proximal",
                "middle_proximal",
                "ring_proximal",
                "pinky_proximal",
            )
        ),
        *(
            f"right_{finger}_joint"
            for finger in (
                "thumb_proximal",
                "thumb_metacarpal",
                "index_proximal",
                "middle_proximal",
                "ring_proximal",
                "pinky_proximal",
            )
        ),
    ]
    cameras = [f"observation.images.cam_{name}" for name in ("head", "left_wrist", "right_wrist")]
    features = {key: {"dtype": "float32", "shape": (27,), "names": names} for key in ("observation.state", "action")}
    features.update(
        {key: {"dtype": "video", "shape": (32, 32, 3), "names": ["height", "width", "channels"]} for key in cameras}
    )
    dataset = LeRobotDataset.create("test/diffusion", fps=10, root=root, features=features, use_videos=True)
    rng = np.random.default_rng(0)
    for frame in range(8):
        state = np.linspace(0, 1, 27, dtype=np.float32) + frame * 0.01
        dataset.add_frame(
            {
                "observation.state": state,
                "action": state + np.float32(0.05 + frame * 0.002),
                "task": "test diffusion",
                **{key: rng.integers(0, 256, (32, 32, 3), dtype=np.uint8) for key in cameras},
            }
        )
    dataset.save_episode()
    assert json.loads((root / "meta/info.json").read_text())["codebase_version"] == "v2.1"
    experiment = tmp_path / "diffusion.yaml"
    experiment.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "base": "diffusion",
                "experiment": {"name": "smoke", "dataset": str(root)},
                "policy": {"groups": ["right_arm", "right_hand"], "action_horizon": 4},
                "model": {
                    "down_dims": [8, 16],
                    "kernel_size": 3,
                    "diffusion_step_embed_dim": 8,
                    "image_size": 32,
                    "vision_width": 8,
                    "vision_feature_dim": 8,
                    "num_train_timesteps": 8,
                    "num_inference_steps": 2,
                    "sampler": "ddim",
                },
                "training": {
                    "batch_size": 2,
                    "num_workers": 0,
                    "num_steps": 2,
                    "warmup_steps": 1,
                    "log_interval": 1,
                    "save_interval": 1,
                    "wandb_enabled": False,
                    "assets_base_dir": str(tmp_path / "assets"),
                    "checkpoint_base_dir": str(tmp_path / "checkpoints"),
                },
            }
        )
    )
    compute_norm_stats.main(config_path=str(experiment))
    config = config_io.load_train_config(experiment)
    assert config.model.action_dim == 13
    original_save = train._checkpoints.save_state  # noqa: SLF001

    def checked_save(manager, state, loader, step):
        expected = state.ema_params if state.ema_params is not None else state.params
        expected = jax.tree.map(lambda x: np.array(x, copy=True), expected.to_pure_dict())
        original_save(manager, state, loader, step)
        manager.wait_until_finished()
        loaded = model_lib.restore_params(config.checkpoint_dir / str(step) / "params")
        actual = nnx.state(config.model.load(loaded)).to_pure_dict()
        for (path, before), after in zip(
            jax.tree_util.tree_flatten_with_path(expected)[0], jax.tree.leaves(actual), strict=True
        ):
            assert np.isfinite(before).all(), str(path)
            np.testing.assert_array_equal(before, after, err_msg=f"checkpoint step {step}: {path}")

    monkeypatch.setattr(train._checkpoints, "save_state", checked_save)  # noqa: SLF001
    train.main(config)
    first_checkpoint = config.checkpoint_dir / "1"
    assert (first_checkpoint / "params").is_dir()
    assert (first_checkpoint / "train_state").is_dir()
    assert contract.load_policy_contract(first_checkpoint)["checkpoint_id"] == "smoke_step1"
    # Resume includes optimizer, online parameters and EMA state.
    train.main(dataclasses.replace(config, resume=True, num_train_steps=3))
    checkpoint = config.checkpoint_dir / "2"
    payload = contract.load_policy_contract(checkpoint)
    assert payload["checkpoint_id"] == "smoke_step2"
    assert payload["inference_strategies"] == ["standard"]
    assert list(payload["output_joint_groups"]) == ["right_arm", "right_hand"]
    with pytest.raises(FileNotFoundError, match="parameters missing"):
        export_checkpoint(config, tmp_path / "empty")
    # Both generated artifacts must still load after source data/assets move.
    root.rename(tmp_path / "moved_dataset")
    (tmp_path / "assets").rename(tmp_path / "moved_assets")
    observation = {
        "joint_groups": {
            group: {
                "names": group_names[::-1],
                "positions": np.linspace(0, 1, len(group_names), dtype=np.float32)[::-1],
            }
            for group, group_names in payload["required_joint_groups"].items()
        },
        "images": {key: np.zeros((32, 32, 3), np.uint8) for key in cameras},
        "task": "test diffusion",
    }
    stream = io.BytesIO()
    for message in (
        {"type": "DESCRIBE"},
        {"type": "LOAD", "inference_config": {"strategy": "rtc"}},
        {"type": "INFER", "request_id": 40, "observation": observation},
        {"type": "LOAD", "warmup_observation": observation},
        {"type": "INFER", "request_id": 42, "observation": observation},
        {"type": "CLOSE"},
    ):
        protocol.send_frame(stream, message)
    env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    env["JAX_PLATFORMS"] = "cpu"
    jax.clear_caches()
    result = subprocess.run(
        [sys.executable, "-m", "openpi.deploy.policy_inferencer", "--checkpoint", str(checkpoint)],
        input=stream.getvalue(),
        capture_output=True,
        cwd=Path("/tmp"),
        env=env,
        timeout=120,
        check=True,
    )
    output = io.BytesIO(result.stdout)
    assert protocol.recv_frame(output)["type"] == "CONTRACT"
    assert protocol.recv_frame(output)["type"] == "ERROR"  # RTC is unsupported.
    assert "not loaded" in protocol.recv_frame(output)["error"]
    ready = protocol.recv_frame(output)
    assert ready["type"] == "READY", ready
    reply = protocol.recv_frame(output)
    assert reply["type"] == "RESULT", reply
    assert reply["request_id"] == 42
    assert reply["contract_hash"] == payload["contract_hash"]
    assert set(reply["actions"]) == set(payload["output_joint_groups"])
    for group, values in reply["actions"].items():
        assert values.shape == (4, len(payload["output_joint_groups"][group]))
        assert values.dtype == np.float32
        assert np.isfinite(values).all()
    assert protocol.recv_frame(output)["type"] == "CLOSED"
    assert output.read() == b""
