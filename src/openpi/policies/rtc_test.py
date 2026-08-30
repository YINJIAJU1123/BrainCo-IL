import numpy as np
import pytest

from openpi.policies.rtc import RTCConfig
from openpi.policies.rtc import build_exp_prefix_weights


def test_exp_prefix_weights() -> None:
    weights = build_exp_prefix_weights(
        action_horizon=16,
        inference_delay=4,
        prefix_horizon=8,
        available_prefix_steps=10,
    )[:, 0]

    np.testing.assert_array_equal(weights[:4], np.ones((4,), dtype=np.float32))
    np.testing.assert_array_equal(weights[8:], np.zeros((8,), dtype=np.float32))
    assert np.all(np.diff(weights[3:8]) < 0)
    assert weights[4] == pytest.approx(0.571, abs=0.01)
    assert weights[7] == pytest.approx(0.026, abs=0.01)


def test_exp_prefix_weights_respect_available_prefix() -> None:
    weights = build_exp_prefix_weights(
        action_horizon=16,
        inference_delay=5,
        prefix_horizon=8,
        available_prefix_steps=3,
    )[:, 0]
    np.testing.assert_array_equal(weights[:3], np.ones((3,), dtype=np.float32))
    np.testing.assert_array_equal(weights[3:], np.zeros((13,), dtype=np.float32))


def test_rtc_config_rejects_unsupported_schedule() -> None:
    with pytest.raises(ValueError, match="currently supports 'exp'"):
        RTCConfig(schedule="linear").validate(action_horizon=16)
