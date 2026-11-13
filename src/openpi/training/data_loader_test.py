from types import SimpleNamespace

import datasets
import jax
import numpy as np
import pytest
import torch

from openpi.models import pi0_config
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader


def test_lerobot_dataset_uses_float32_safe_timestamp_tolerance(monkeypatch):
    captured = {}

    def dataset_cls(*args, **kwargs):
        captured.update(kwargs)
        return args

    monkeypatch.delenv("OPENPI_LEROBOT_TOLERANCE_S", raising=False)
    _data_loader._create_lerobot_dataset_compat("dataset", dataset_cls=dataset_cls)  # noqa: SLF001

    assert captured["tolerance_s"] == pytest.approx(2e-4)


def test_lerobot_dataset_timestamp_tolerance_can_be_overridden(monkeypatch):
    captured = {}

    def dataset_cls(*args, **kwargs):
        captured.update(kwargs)
        return args

    monkeypatch.setenv("OPENPI_LEROBOT_TOLERANCE_S", "0.0005")
    _data_loader._create_lerobot_dataset_compat("dataset", dataset_cls=dataset_cls)  # noqa: SLF001

    assert captured["tolerance_s"] == pytest.approx(5e-4)


@pytest.mark.parametrize("value", ["not-a-number", "0", "-0.1", "nan", "inf"])
def test_lerobot_dataset_rejects_invalid_timestamp_tolerance(monkeypatch, value):
    monkeypatch.setenv("OPENPI_LEROBOT_TOLERANCE_S", value)

    with pytest.raises(ValueError, match="positive finite float"):
        _data_loader._create_lerobot_dataset_compat("dataset", dataset_cls=lambda *args, **kwargs: None)  # noqa: SLF001


def test_norm_stats_dataset_skips_video_decoding():
    dataset = object.__new__(_data_loader._NormStatsLeRobotDataset)  # noqa: SLF001

    images = dataset._query_videos(  # noqa: SLF001
        {
            "observation.images.cam_high": [0.0],
            "observation.images.cam_left_wrist": [0.0],
        },
        ep_idx=3,
    )

    assert set(images) == {"observation.images.cam_high", "observation.images.cam_left_wrist"}
    assert all(torch.equal(image, torch.zeros((3, 1, 1))) for image in images.values())


@pytest.mark.parametrize("decode_videos", [False, True])
def test_projected_dataset_preserves_actions_and_padding_without_reading_tactile(tmp_path, monkeypatch, decode_videos):
    camera = "observation.images.head"
    tactile_video = "observation.tactile_rgb.left_thumb"
    tactile_numeric = "observation.tactile_pressure.left_palm"
    raw = datasets.Dataset.from_dict({
        "timestamp": np.array([0, 1 / 30, 0, 1 / 30, 2 / 30], np.float32),
        "episode_index": [0, 0, 1, 1, 1],
        "frame_index": [0, 1, 0, 1, 2],
        "index": list(range(5)),
        "task_index": [0] * 5,
        "observation.state": [[float(i), float(i + 10)] for i in range(5)],
        "action": [[float(i + 1), float(i + 11)] for i in range(5)],
        tactile_numeric: [[float("nan")] * 128 for _ in range(5)],
    })
    (tmp_path / "data").mkdir()
    raw.to_parquet(tmp_path / "data" / "episode.parquet")
    features = {key: {"dtype": "float32"} for key in raw.column_names}
    features.update({camera: {"dtype": "video"}, tactile_video: {"dtype": "video"}})
    metadata = SimpleNamespace(
        features=features, video_keys=[camera, tactile_video], tasks={0: "test"},
        get_video_file_path=lambda ep, key: f"{key}/{ep}.mp4",
    )
    cls = _data_loader._PolicyLeRobotDataset if decode_videos else _data_loader._NormStatsLeRobotDataset  # noqa: SLF001
    projected = object.__new__(cls)
    projected._dataset_feature_keys = frozenset(["observation.state", "action", camera])  # noqa: SLF001
    projected.root = tmp_path
    projected.episodes = None
    projected.meta = metadata
    projected.hf_dataset = projected.load_hf_dataset()
    assert set(projected.hf_dataset.column_names) == set(raw.column_names) - {tactile_numeric}
    assert tactile_numeric not in projected.hf_dataset[0]

    baseline = object.__new__(_data_loader.lerobot_dataset.LeRobotDataset)
    raw.set_transform(_data_loader.lerobot_dataset.hf_transform_to_torch)
    baseline.hf_dataset = raw
    baseline.meta = metadata
    baseline.root = tmp_path
    for dataset in [baseline, projected]:
        dataset.episode_data_index = {"from": torch.tensor([0, 2]), "to": torch.tensor([2, 5])}
        dataset.delta_indices = {"action": [0, 1, 2]}
        dataset.image_transforms = None
        dataset.tolerance_s = 2e-4
        dataset.video_backend = "pyav"

    decoded = []
    def decode(path, *args):
        decoded.append(str(path))
        return torch.ones((1, 3, 2, 2))
    monkeypatch.setattr(_data_loader.lerobot_dataset, "decode_video_frames", decode)
    before, after = [], []
    for index in range(5):
        original = baseline[index]
        decoded.clear()
        selected = projected[index]
        assert tactile_numeric not in selected and tactile_video not in selected
        assert selected["task"] == original["task"]
        for key in ["observation.state", "action", "action_is_pad"]:
            torch.testing.assert_close(selected[key], original[key])
        assert all(camera in path and tactile_video not in path for path in decoded)
        assert len(decoded) == (1 if decode_videos else 0)
        before.append(original["action"].numpy())
        after.append(selected["action"].numpy())
    # The final action in episode 0 pads within that episode, not into episode 1.
    np.testing.assert_array_equal(after[1], np.repeat([[2.0, 12.0]], 3, axis=0))
    np.testing.assert_array_equal(np.mean(before, axis=(0, 1)), np.mean(after, axis=(0, 1)))
    np.testing.assert_array_equal(np.std(before, axis=(0, 1)), np.std(after, axis=(0, 1)))


def test_torch_data_loader():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 16)

    loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=4,
        num_batches=2,
    )
    batches = list(loader)

    assert len(batches) == 2
    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_torch_data_loader_infinite():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 4)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4)
    data_iter = iter(loader)

    for _ in range(10):
        _ = next(data_iter)


def test_torch_data_loader_parallel():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 10)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4, num_batches=2, num_workers=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_with_fake_dataset():
    config = _config.get_config("debug")

    loader = _data_loader.create_data_loader(config, skip_norm_stats=True, num_batches=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == config.batch_size for x in jax.tree.leaves(batch))

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)
