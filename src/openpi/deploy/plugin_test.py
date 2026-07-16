import types

import numpy as np
import pytest

from openpi.deploy import plugin


def _raw_observation() -> dict:
    return {key: np.zeros((8, 12, 3), dtype=np.uint8) for key in plugin.POLICY_IMAGE_KEYS}


def test_validate_raw_images_accepts_uint8_hwc():
    observation = _raw_observation()
    assert plugin.ValidateRawImages()(observation) is observation


def test_validate_raw_images_rejects_legacy_float_chw():
    observation = _raw_observation()
    observation[plugin.POLICY_IMAGE_KEYS[0]] = np.zeros((3, 224, 224), dtype=np.float32)

    with pytest.raises(ValueError, match="must be RGB uint8 HWC"):
        plugin.ValidateRawImages()(observation)


@pytest.mark.parametrize(
    ("options", "expected"),
    [
        ({}, None),
        ({"use_compiled_executable": None}, None),
        ({"use_compiled_executable": False}, False),
        ({"use_compiled_executable": True}, True),
    ],
)
def test_optional_bool(options, expected):
    assert plugin._optional_bool(options, "use_compiled_executable") is expected  # noqa: SLF001


@pytest.mark.parametrize("value", [0, 1, "false", "true", [], {}])
def test_optional_bool_rejects_non_bool(value):
    with pytest.raises(TypeError, match="must be a bool"):
        plugin._optional_bool(  # noqa: SLF001
            {"use_compiled_executable": value}, "use_compiled_executable"
        )


@pytest.mark.parametrize("override", [None, False, True])
def test_create_policy_forwards_compiled_executable_override(monkeypatch, override):
    train_config = types.SimpleNamespace(model=types.SimpleNamespace(model_type="act"))
    captured = {}

    monkeypatch.setattr(plugin.config_io, "load_train_config", lambda checkpoint_dir: train_config)

    def fake_create_trained_policy(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return object()

    monkeypatch.setattr(plugin.policy_config, "create_trained_policy", fake_create_trained_policy)
    runtime_options = {} if override is None else {"use_compiled_executable": override}
    plugin.create_policy("/tmp/checkpoint", runtime_options=runtime_options)

    assert captured["args"] == (train_config, "/tmp/checkpoint")
    assert captured["kwargs"]["use_compiled_executable"] is override
