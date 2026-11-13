import copy
import io
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import yaml

from openpi.deploy import contract
from openpi.deploy import policy_inferencer
from openpi.deploy import protocol
from openpi.deploy.contract_test import _dataset
from openpi.deploy.contract_test import _experiment
from openpi.models.diffusion_config import DiffusionPolicyConfig
from openpi.training import config_io


@pytest.fixture
def resolved(tmp_path):
    dataset = tmp_path / "dataset"
    _dataset(dataset)
    path = tmp_path / "diffusion.yaml"
    _experiment(path, dataset, "diffusion")
    config = config_io.load_train_config(path)
    return config, contract.build_policy_contract(config, "checkpoint")


def test_diffusion_experiment_and_generated_config(resolved, tmp_path):
    config, payload = resolved
    assert isinstance(config.model, DiffusionPolicyConfig)
    assert config.model.action_dim == 26
    assert config.model.action_horizon == 12
    assert config.ema_decay == 0.999
    assert payload["inference_strategies"] == ["standard"]
    path = config_io.save_train_config(config, tmp_path / "checkpoint")
    restored = config_io.load_train_config(path)
    assert restored.model == config.model
    config_io.validate_checkpoint_id(path, "checkpoint")
    with pytest.raises(RuntimeError, match="differs"):
        config_io.validate_checkpoint_id(path, "wrong_step")
    raw = yaml.safe_load(path.read_text())
    raw["train_config"]["fields"]["model"]["fields"]["num_inference_steps"] = 1
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(RuntimeError, match="modified"):
        config_io.load_train_config(path)


def test_contract_tampering(resolved, tmp_path):
    config, _ = resolved
    path = contract.save_policy_contract(config, tmp_path)
    raw = json.loads(path.read_text())
    raw["chunk_steps"] = 999
    path.write_text(json.dumps(raw))
    with pytest.raises(RuntimeError, match="modified"):
        contract.load_policy_contract(path)


def test_hand_only_experiment_needs_no_delta_groups(tmp_path):
    dataset = tmp_path / "dataset"
    _dataset(dataset)
    path = tmp_path / "diffusion.yaml"
    _experiment(path, dataset, "diffusion")
    experiment = yaml.safe_load(path.read_text())
    experiment["policy"]["groups"] = ["right_hand"]
    path.write_text(yaml.safe_dump(experiment))
    config = config_io.load_train_config(path)
    assert config.model.action_dim == 6
    assert config.data.policy_io.delta_action_groups == ()
    config.data.create(config.assets_dirs, config.model)
    assert list(contract.build_policy_contract(config, "x")["output_joint_groups"]) == ["right_hand"]


def test_dataset_permutation_is_applied_even_when_width_is_unchanged(tmp_path):
    from openpi import transforms  # noqa: PLC0415

    dataset = tmp_path / "dataset"
    _dataset(dataset)
    info_path = dataset / "meta/info.json"
    info = json.loads(info_path.read_text())
    for key in ("observation.state", "action"):
        info["features"][key]["names"].reverse()
    info_path.write_text(json.dumps(info))
    path = tmp_path / "diffusion.yaml"
    _experiment(path, dataset, "diffusion")
    config = config_io.load_train_config(path)
    data = config.data.create(config.assets_dirs, config.model)
    sample = {
        "observation.state": np.arange(26, dtype=np.float32),
        "action": np.tile(np.arange(26, dtype=np.float32), (12, 1)),
        "prompt": "test",
        **{
            key: np.zeros((32, 32, 3), np.uint8)
            for key in contract.build_policy_contract(config, "x")["required_cameras"]
        },
    }
    repacked = transforms.compose(data.repack_transforms.inputs)(sample)
    expected = np.arange(26)[list(config.data.policy_io.dataset_state_indices)]
    np.testing.assert_array_equal(repacked["observation/state"], expected)
    np.testing.assert_array_equal(repacked["actions"][0], expected)


@pytest.mark.parametrize("problem", ["groups", "fps", "camera", "indices"])
def test_export_rejects_inconsistent_metadata(tmp_path, problem):
    dataset = tmp_path / "dataset"
    _dataset(dataset)
    path = tmp_path / "diffusion.yaml"
    _experiment(path, dataset, "diffusion")
    config = config_io.load_train_config(path)
    info_path = dataset / "meta/info.json"
    info = json.loads(info_path.read_text())
    if problem == "groups":
        with pytest.raises(ValueError, match="unique and ordered"):
            contract.resolve_dataset_interface([dataset], ["right_arm", "left_arm"])
        return
    if problem == "fps":
        info["fps"] = 0
    elif problem == "camera":
        del info["features"]["observation.images.cam_head"]
    else:
        for key in ("observation.state", "action"):
            info["features"][key]["names"].reverse()
    info_path.write_text(json.dumps(info))
    with pytest.raises(ValueError, match=r"fps|cameras|names/indices"):
        contract.build_policy_contract(config, "x")


def named_observation(payload):
    return {
        "joint_groups": {
            group: {"names": names, "positions": np.arange(len(names), dtype=np.float32)}
            for group, names in payload["required_joint_groups"].items()
        },
        "images": {key: np.zeros((32, 32, 3), np.uint8) for key in payload["required_cameras"]},
        "task": "unused by diffusion",
    }


def test_reversed_names_and_absolute_action_round_trip(resolved):
    config, payload = resolved
    normal = named_observation(payload)
    reversed_obs = copy.deepcopy(normal)
    for group in reversed_obs["joint_groups"].values():
        group["names"] = group["names"][::-1]
        group["positions"] = group["positions"][::-1]
    original = policy_inferencer._build_policy_observation(normal, payload)  # noqa: SLF001
    reordered = policy_inferencer._build_policy_observation(reversed_obs, payload)  # noqa: SLF001
    np.testing.assert_array_equal(original["observation/state"], reordered["observation/state"])
    # Test the actual delta -> normalization -> inverse -> absolute chain.
    from openpi import transforms  # noqa: PLC0415
    from openpi.shared.normalize import NormStats  # noqa: PLC0415

    data = config.data.create(config.assets_dirs, config.model)
    stats = {
        key: NormStats(mean=np.zeros(26), std=np.ones(26), q01=np.full(26, -2.0), q99=np.full(26, 3.0))
        for key in ("state", "actions")
    }
    actions = np.random.default_rng(7).normal(size=(12, 26)).astype(np.float32)
    inputs = {**original, "actions": actions.copy()}
    encoded = transforms.compose(
        [
            *data.data_transforms.inputs,
            transforms.Normalize(stats, use_quantiles=True),
            *data.model_transforms.inputs,
        ]
    )(inputs)
    assert "prompt" not in encoded
    assert "tokenized_prompt" not in encoded
    decoded = transforms.compose(
        [
            transforms.Unnormalize(stats, use_quantiles=True),
            *data.data_transforms.outputs,
        ]
    )({"state": encoded["state"], "actions": encoded["actions"]})
    np.testing.assert_allclose(decoded["actions"], actions, atol=2e-5)


@pytest.mark.parametrize("problem", ["missing", "duplicate", "nan", "inf", "length"])
def test_invalid_named_input(resolved, problem):
    _, payload = resolved
    obs = named_observation(payload)
    group = obs["joint_groups"]["left_arm"]
    if problem == "missing":
        group["names"] = ["unknown", *group["names"][1:]]
    elif problem == "duplicate":
        group["names"] = [group["names"][1], *group["names"][1:]]
    elif problem == "length":
        group["positions"] = group["positions"][:-1]
    else:
        group["positions"][0] = float(problem)
    with pytest.raises(ValueError, match="joint group"):
        policy_inferencer._build_policy_observation(obs, payload)  # noqa: SLF001


@pytest.mark.parametrize("problem", ["horizon", "width", "nan"])
def test_invalid_output(resolved, problem):
    _, payload = resolved
    actions = np.zeros((12, 26), dtype=np.float32)
    if problem == "horizon":
        actions = actions[:1]
    elif problem == "width":
        actions = actions[:, :1]
    else:
        actions[0, 0] = np.nan
    with pytest.raises(ValueError, match=r"chunk length|action dim|non-finite"):
        policy_inferencer._group_actions(actions, payload)  # noqa: SLF001


def test_partial_reads_and_stdout_pollution():
    class PartialReader(io.BytesIO):
        def read(self, size=-1):
            return super().read(min(size, 3))

    stream = io.BytesIO()
    array = np.arange(12, dtype=np.float32).reshape(3, 4)
    protocol.send_frame(stream, {"type": "RESULT", "actions": {"left_arm": array}})
    result = protocol.recv_frame(PartialReader(stream.getvalue()))
    np.testing.assert_array_equal(result["actions"]["left_arm"], array)
    with pytest.raises(ValueError, match="stdout"):
        protocol.recv_frame(io.BytesIO(b"unexpected log output\n"))


def test_describe_is_lightweight_from_outside_repo(resolved, tmp_path):
    config, _ = resolved
    checkpoint = tmp_path / "checkpoint"
    contract.save_policy_contract(config, checkpoint)
    # No weights/config exist. DESCRIBE must still work, without JAX/Torch imports.
    command = """
import sys
from openpi.deploy.policy_inferencer import main
main()
assert not any(k in sys.modules for k in ('jax', 'torch', 'openpi.models.diffusion'))
"""
    stream = io.BytesIO()
    protocol.send_frame(stream, {"type": "DESCRIBE"})
    protocol.send_frame(stream, {"type": "CLOSE"})
    env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    result = subprocess.run(
        [sys.executable, "-c", command, "--checkpoint", str(checkpoint)],
        input=stream.getvalue(),
        capture_output=True,
        cwd=Path("/tmp"),
        env=env,
        timeout=15,
        check=True,
    )
    output = io.BytesIO(result.stdout)
    assert protocol.recv_frame(output)["capabilities"]["inference_strategies"] == ["standard"]
    assert protocol.recv_frame(output)["type"] == "CLOSED"
    assert output.read() == b""
