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
