import io
import json

import numpy as np
import pytest

from openpi.deploy import contract
from openpi.deploy import policy_inferencer
from openpi.deploy import protocol
from openpi.models import act_config
from openpi.models import pi0_config
from openpi.training import config_io


def _dataset(root):
    names = [
        *(f"left_arm_joint{i}" for i in range(1, 8)),
        *(f"right_arm_joint{i}" for i in range(1, 8)),
        *(f"left_finger{i}_joint" for i in range(1, 7)),
        *(f"right_finger{i}_joint" for i in range(1, 7)),
    ]
    features = {
        "observation.state": {"dtype": "float32", "shape": [26], "names": names},
        "action": {"dtype": "float32", "shape": [26], "names": names},
        "observation.images.cam_head": {"dtype": "video", "shape": [8, 8, 3]},
        "observation.images.cam_left_wrist": {"dtype": "video", "shape": [8, 8, 3]},
        "observation.images.cam_right_wrist": {"dtype": "video", "shape": [8, 8, 3]},
    }
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(json.dumps({"robot_type": "test", "fps": 30, "features": features}))


def _experiment(path, dataset, base):
    path.write_text(
        f"""schema_version: 1
base: {base}
experiment:
  name: test_{base}
  dataset: {dataset}
policy:
  groups: [left_arm, right_arm, left_hand, right_hand]
  action_horizon: 12
training:
  num_steps: 20
"""
    )


def test_concise_experiment_switches_pi05_and_act(tmp_path):
    dataset = tmp_path / "dataset"
    _dataset(dataset)
    pi_path = tmp_path / "pi.yaml"
    act_path = tmp_path / "act.yaml"
    _experiment(pi_path, dataset, "pi05")
    _experiment(act_path, dataset, "act")

    pi = config_io.load_train_config(pi_path)
    act = config_io.load_train_config(act_path)
    assert isinstance(pi.model, pi0_config.Pi0Config)
    assert pi.model.pi05 is True
    assert isinstance(act.model, act_config.ACTConfig)
    assert pi.model.action_dim == act.model.action_dim == 26
    assert pi.model.action_horizon == act.model.action_horizon == 12
    assert contract.build_policy_contract(pi, "pi")["inference_strategies"] == [
        "standard",
        "rtc",
    ]
    assert contract.build_policy_contract(act, "act")["inference_strategies"] == [
        "standard"
    ]


def test_concise_experiment_rejects_action_mode_fields(tmp_path):
    dataset = tmp_path / "dataset"
    _dataset(dataset)
    experiment = tmp_path / "pi.yaml"
    _experiment(experiment, dataset, "pi05")
    experiment.write_text(
        experiment.read_text().replace(
            "  action_horizon: 12\n",
            "  action_horizon: 12\n  internal_delta: false\n",
        )
    )

    with pytest.raises(ValueError, match="always absolute"):
        config_io.load_train_config(experiment)


def test_generated_contract_and_protocol_round_trip(tmp_path):
    dataset = tmp_path / "dataset"
    _dataset(dataset)
    experiment = tmp_path / "pi.yaml"
    _experiment(experiment, dataset, "pi05")
    config = config_io.load_train_config(experiment)
    generated = contract.save_policy_contract(config, tmp_path / "checkpoint")
    loaded = contract.load_policy_contract(generated)
    assert loaded["do_not_edit"] is True
    assert sum(map(len, loaded["output_joint_groups"].values())) == 26
    assert loaded["inference_strategies"] == ["standard", "rtc"]

    stream = io.BytesIO()
    protocol.send_frame(stream, {"type": "TEST", "array": np.zeros((2, 3), np.float32)})
    stream.seek(0)
    result = protocol.recv_frame(stream)
    assert result["type"] == "TEST"
    assert result["array"].shape == (2, 3)


def test_inferencer_named_adapter_accepts_reordered_names(tmp_path):
    dataset = tmp_path / "dataset"
    _dataset(dataset)
    experiment = tmp_path / "pi.yaml"
    _experiment(experiment, dataset, "pi05")
    config = config_io.load_train_config(experiment)
    payload = contract.build_policy_contract(config, "test")
    observation = {
        "joint_groups": {
            group: {
                "names": list(reversed(names)),
                "positions": np.arange(len(names), dtype=np.float32)[::-1],
            }
            for group, names in payload["required_joint_groups"].items()
        },
        "images": {key: np.zeros((8, 8, 3), np.uint8) for key in payload["required_cameras"]},
        "task": "test",
    }

    result = policy_inferencer._build_policy_observation(observation, payload)  # noqa: SLF001

    expected = np.concatenate(
        [np.arange(len(names), dtype=np.float32) for names in payload["required_joint_groups"].values()]
    )
    np.testing.assert_array_equal(result["observation/state"], expected)


def test_inferencer_builds_training_transform_input_keys(tmp_path):
    dataset = tmp_path / "dataset"
    _dataset(dataset)
    experiment = tmp_path / "pi.yaml"
    _experiment(experiment, dataset, "pi05")
    config = config_io.load_train_config(experiment)
    payload = contract.build_policy_contract(config, "test")
    observation = {
        "joint_groups": {
            group: {"names": names, "positions": np.zeros(len(names), np.float32)}
            for group, names in payload["required_joint_groups"].items()
        },
        "images": {key: np.zeros((8, 8, 3), np.uint8) for key in payload["required_cameras"]},
        "task": "test",
    }

    result = policy_inferencer._build_policy_observation(observation, payload)  # noqa: SLF001

    assert result["observation/state"].shape == (26,)
    assert result["observation/image"].shape == (8, 8, 3)
    assert result["observation/left_wrist_image"].shape == (8, 8, 3)
    assert result["observation/right_wrist_image"].shape == (8, 8, 3)
