import numpy as np

from openpi.training import weight_loaders


def test_partial_loader_initializes_allowed_target_only_layers(monkeypatch):
    reference = {
        "Pi0": {
            "base": {"kernel": np.zeros((2, 2), dtype=np.float32)},
            "action_out_proj": {"kernel": np.zeros((2, 2), dtype=np.float32)},
        }
    }
    checkpoint = {
        "Pi0": {
            "base": {"kernel": np.ones((2, 2), dtype=np.float32)},
        }
    }
    monkeypatch.setattr(weight_loaders.download, "maybe_download", lambda path: path)
    monkeypatch.setattr(weight_loaders._model, "restore_params", lambda *args, **kwargs: checkpoint)  # noqa: SLF001

    loaded = weight_loaders.PartialCheckpointWeightLoader("checkpoint").load(reference)

    assert np.array_equal(loaded["Pi0"]["base"]["kernel"], checkpoint["Pi0"]["base"]["kernel"])
    assert np.array_equal(
        loaded["Pi0"]["action_out_proj"]["kernel"],
        reference["Pi0"]["action_out_proj"]["kernel"],
    )
