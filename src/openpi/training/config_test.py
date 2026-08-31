import pytest
import yaml

from openpi.models import pi0_config
from openpi.training import config as _config
from openpi.training import config_io


@pytest.mark.parametrize(
    ("strategy", "trainable_path", "frozen_path"),
    [
        ("lora_and_action_interface", ("PaliGemma", "llm", "lora", "kernel"), ("PaliGemma", "img", "kernel")),
        ("action_interface_only", ("Pi0", "action_in_proj", "kernel"), ("PaliGemma", "llm", "kernel")),
    ],
)
def test_freeze_strategy_yaml_round_trip(tmp_path, strategy, trainable_path, frozen_path):
    model = pi0_config.Pi0Config(
        pi05=True,
        paligemma_variant="gemma_2b_lora" if strategy == "lora_and_action_interface" else "gemma_2b",
        action_expert_variant="gemma_300m_lora" if strategy == "lora_and_action_interface" else "gemma_300m",
    )
    original = _config.TrainConfig(name="test", exp_name="test", model=model, freeze_strategy=strategy)

    path = config_io.save_train_config(original, tmp_path)
    restored = config_io.load_train_config(path)

    assert restored.freeze_strategy == strategy
    assert not restored.effective_freeze_filter(trainable_path, None)
    assert restored.effective_freeze_filter(frozen_path, None)


def test_lora_strategy_requires_lora_variant():
    with pytest.raises(ValueError, match="requires at least one LoRA model variant"):
        _config.TrainConfig(
            name="test",
            exp_name="test",
            model=pi0_config.Pi0Config(pi05=True),
            freeze_strategy="lora_and_action_interface",
        )


def test_save_final_checkpoint_yaml_round_trip(tmp_path):
    original = _config.TrainConfig(
        name="test",
        exp_name="test",
        save_final_checkpoint=False,
    )

    path = config_io.save_train_config(original, tmp_path)
    restored = config_io.load_train_config(path)

    assert restored.save_final_checkpoint is False


def test_saved_train_config_is_bound_to_checkpoint_directory(tmp_path):
    original = _config.TrainConfig(name="run", exp_name="experiment")
    checkpoint = tmp_path / "42"

    path = config_io.save_train_config(original, checkpoint)
    payload = yaml.safe_load(path.read_text())

    assert payload["checkpoint_id"] == "experiment_step42"
    config_io.validate_checkpoint_id(checkpoint, "experiment_step42")
    with pytest.raises(RuntimeError, match="differs from policy_contract"):
        config_io.validate_checkpoint_id(checkpoint, "other")


def test_removed_fields_in_existing_checkpoint_are_ignored(tmp_path):
    original = _legacy_compatible_config()
    payload = yaml.safe_load(config_io.to_yaml(original))
    train_fields = payload["train_config"]["fields"]
    train_fields["pytorch_weight_path"] = None
    train_fields["pytorch_training_precision"] = "bfloat16"
    data_fields = train_fields["data"]["fields"]["base_config"]["fields"]
    data_fields["rlds_data_dir"] = None
    data_fields["action_space"] = None
    data_fields["datasets"] = []
    # Files without a content hash are legacy generated configs. They remain
    # loadable so the one checkpoint selected for migration can be upgraded.
    payload.pop("content_sha256")

    path = tmp_path / "train_config.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    restored = config_io.load_train_config(path)

    assert restored.name == original.name
    assert restored.model.action_dim == original.model.action_dim


def test_generated_train_config_rejects_manual_edits(tmp_path):
    payload = yaml.safe_load(config_io.to_yaml(_legacy_compatible_config()))
    payload["train_config"]["fields"]["batch_size"] = 999
    path = tmp_path / "train_config.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(RuntimeError, match="was modified"):
        config_io.load_train_config(path)


def _legacy_compatible_config():
    return _config.TrainConfig(
        name="legacy",
        exp_name="legacy",
        data=_config.LeRobotBrainCoDataConfig(
            repo_id="legacy",
            base_config=_config.DataConfig(
                lerobot_datasets=(_config.LeRobotDataset(repo_id="/legacy/dataset", weight=1.0),),
            ),
        ),
    )
