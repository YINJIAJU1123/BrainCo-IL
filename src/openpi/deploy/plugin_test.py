import numpy as np
import pytest

from openpi.deploy import plugin
from openpi.policies import brainco_policy


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


def test_base_spec_describes_right_arm_and_hand_policy():
    policy_io = brainco_policy.BrainCoPolicyIOConfig(
        state_groups=("right_arm", "right_hand"),
        action_groups=("right_arm", "right_hand"),
        delta_action_groups=("right_arm",),
        dataset_state_dim=28,
        dataset_state_indices=None,
    )

    spec = plugin._base_spec(  # noqa: SLF001
        config_name="right28",
        policy_type="pi05",
        action_dim=28,
        action_horizon=16,
        asset_id="right28",
        policy_io=policy_io,
    )

    assert spec["inputs"]["state_dim"] == 28
    assert [entry["group"] for entry in spec["inputs"]["state_composition"]] == [
        "right_arm",
        "right_hand",
    ]
    assert list(spec["outputs"]["joint_groups"]) == ["right_arm", "right_hand"]
    assert spec["dataset"] == {
        "state_key": "observation.state",
        "state_dim": 28,
        "task_key": "task",
        "task_index_key": "task_index",
        "tasks_path": "meta/tasks.parquet",
    }
