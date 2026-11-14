import numpy as np
import pytest
import torch

from openpi.rdp.model import AsymmetricTokenizer
from openpi.rdp.model import LatentDiffusionPolicy
from openpi.rdp.model import RDPConfig
from openpi.rdp.tactile import TactilePCA
from openpi.rdp.tactile import held_indices
from openpi.rdp.tactile import offsets


def test_pca_roundtrip_and_saved_transform(tmp_path):
    rng = np.random.default_rng(42)
    raw = rng.normal(size=(80, 3)) @ rng.normal(size=(3, 800)) + 4
    pca, fraction = TactilePCA.fit(raw, components=3)
    assert fraction > 0.99999
    np.testing.assert_allclose(pca.reconstruct(raw), raw, atol=1e-5)
    pca.save(tmp_path / "pca.npz")
    np.testing.assert_array_equal(TactilePCA.load(tmp_path / "pca.npz").encode(raw), pca.encode(raw))


def test_tactile_hold_is_causal_and_episode_local():
    time = np.arange(10) / 30
    indices, age = held_indices(time, 10)
    np.testing.assert_array_equal(indices, [0, 0, 0, 3, 3, 3, 6, 6, 6, 9])
    assert np.all(time[indices] <= time)
    np.testing.assert_allclose(age[:3], [0, 1 / 30, 2 / 30])
    with pytest.raises(ValueError, match="strictly increasing"):
        held_indices([0, 1, 0], 10)
    bad = np.zeros((20, 20, 6), np.float32)
    bad[0, 0, 4] = np.nan
    with pytest.raises(ValueError, match="finite"):
        offsets(bad)


def test_at_streaming_equals_sequence_and_future_touch_cannot_leak():
    torch.manual_seed(12)
    at = AsymmetricTokenizer(RDPConfig()).eval()
    actions, tactile = torch.randn(2, 16, 56), torch.randn(2, 16, 76)
    latent, _ = at.encode(actions)
    sequence, _ = at.decode(latent, tactile)
    hidden, parts = None, []
    for i in range(16):
        current, hidden = at.decode(latent, tactile[:, i : i + 1], hidden)
        parts.append(current)
    torch.testing.assert_close(torch.cat(parts, 1), sequence, atol=1e-6, rtol=1e-5)
    changed = tactile.clone()
    changed[:, 8:] += 100
    other, _ = at.decode(latent, changed)
    torch.testing.assert_close(sequence[:, :8], other[:, :8])
    assert not torch.allclose(sequence[:, 8:], other[:, 8:])


def test_ldp_updates_without_updating_frozen_at():
    torch.manual_seed(12)
    cfg = RDPConfig(down_dims=(32, 64, 128), vision_dim=16, image_size=32)
    at = AsymmetricTokenizer(cfg).eval().requires_grad_(requires_grad=False)
    policy = LatentDiffusionPolicy(cfg)
    actions, tactile = torch.randn(2, 16, 56), torch.randn(2, 76)
    with torch.no_grad():
        latent, _ = at.encode(actions)
    images = torch.randint(256, (2, 3, 32, 32, 3), dtype=torch.uint8)
    loss = policy.loss(latent, images, torch.zeros(2, 56), tactile)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is None for p in at.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in policy.unet.parameters())
    sampled = policy.sample(images, torch.zeros(2, 56), tactile)
    assert sampled.shape == (2, 4, 16)
    assert torch.isfinite(sampled).all()
