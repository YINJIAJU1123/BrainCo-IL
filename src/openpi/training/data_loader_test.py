import jax
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
