from dataclasses import asdict
import io
import json

import numpy as np
import pytest
import torch
import yaml

from openpi.deploy import policy_inferencer
from openpi.deploy import protocol
from openpi.deploy.contract_test import _dataset
from openpi.deploy.contract_test import _experiment
from openpi.rdp.export import export
from openpi.rdp.inferencer import StreamingRDP
from openpi.rdp.model import AsymmetricTokenizer
from openpi.rdp.model import LatentDiffusionPolicy
from openpi.rdp.model import RDPConfig
from openpi.rdp.tactile import FINGERS
from openpi.rdp.tactile import MARKER_KEYS
from openpi.rdp.tactile import TactilePCA
from openpi.shared import normalize
from openpi.training import config_io


@pytest.fixture
def artifact(tmp_path):
    source = tmp_path / "dataset"
    _dataset(source)
    io_config = tmp_path / "io.yaml"
    _experiment(io_config, source, "diffusion")
    raw = yaml.safe_load(io_config.read_text())
    raw["policy"]["action_horizon"] = 16
    io_config.write_text(yaml.safe_dump(raw))
    io_cfg = config_io.load_train_config(io_config)
    cfg = RDPConfig(action_dim=io_cfg.model.action_dim, down_dims=(32, 64, 128), vision_dim=16, image_size=32)
    run, cache, norm = tmp_path / "run", tmp_path / "cache", tmp_path / "norm"
    run.mkdir()
    (cache / "pca").mkdir(parents=True)
    for finger in FINGERS:
        TactilePCA(np.zeros(800), np.eye(800, 15), np.ones(15)).save(cache / "pca" / f"{finger}.npz")
    (cache / "manifest.json").write_text(json.dumps({"tactile_hz": 10, "pca": {}}))
    stats = normalize.NormStats(
        mean=np.zeros(cfg.action_dim),
        std=np.ones(cfg.action_dim),
        q01=-np.ones(cfg.action_dim),
        q99=np.ones(cfg.action_dim),
    )
    normalize.save(norm, {"state": stats, "actions": stats})
    for name, model in [("at", AsymmetricTokenizer(cfg)), ("ldp", LatentDiffusionPolicy(cfg))]:
        torch.save(
            {"config": asdict(cfg), "model": model.state_dict(), "step": 0, "record": {"commit": "unit-test"}},
            run / f"{name}_best.pt",
        )
    return export(run, cache, io_config, norm, run / "at_best.pt")


def observation(policy, frame):
    return {
        "timestamp": frame / 30,
        "joint_groups": {
            key: {"names": names, "positions": np.zeros(len(names), np.float32)}
            for key, names in policy.contract["required_joint_groups"].items()
        },
        "images": {key: np.zeros((32, 32, 3), np.uint8) for key in policy.contract["required_cameras"]},
        "tactile": {
            key: {"markers": np.zeros((20, 20, 6), np.float32), "timestamp": (frame // 3) / 10} for key in MARKER_KEYS
        },
    }


def test_export_streaming_reset_staleness_and_hashes(artifact):
    policy = StreamingRDP(artifact, device="cpu")
    assert policy.contract["chunk_steps"] == 1
    assert policy.contract["model_chunk_steps"] == 16
    for frame in range(17):
        result = policy.infer(observation(policy, frame))
        assert result["actions"].shape == (1, policy.cfg.action_dim)
        assert np.isfinite(result["actions"]).all()
        assert result["timing"]["replanned"] == (frame in (0, 16))
    stale = observation(policy, 40)
    for value in stale["tactile"].values():
        value["timestamp"] = 0
    with pytest.raises(ValueError, match="stale tactile"):
        policy.infer(stale)
    policy.reset()
    assert policy.infer(observation(policy, 0))["timing"]["replanned"]
    (artifact / "assets/pca/thumb.npz").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="hash mismatch"):
        StreamingRDP(artifact, device="cpu")


def test_standard_entry_dispatches_rdp_describe(artifact):
    incoming, outgoing = io.BytesIO(), io.BytesIO()
    for message in [{"type": "DESCRIBE"}, {"type": "CLOSE"}]:
        protocol.send_frame(incoming, message)
    incoming.seek(0)
    policy_inferencer.serve(artifact, protocol_in=incoming, protocol_out=outgoing)
    outgoing.seek(0)
    reply = protocol.recv_frame(outgoing)
    assert reply["type"] == "CONTRACT"
    assert reply["capabilities"]["streaming_tactile"]
    assert protocol.recv_frame(outgoing)["type"] == "CLOSED"
