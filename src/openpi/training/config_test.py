from pathlib import Path

import pytest

from openpi.models import pi0_config
from openpi.training import config as _config
from openpi.training import config_io

_TEMPLATE_DIR = Path(__file__).parent / "training_config_template"


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


@pytest.mark.parametrize("template_path", sorted(_TEMPLATE_DIR.glob("*.yaml")), ids=lambda path: path.stem)
def test_training_config_template_loads(template_path):
    restored = config_io.load_train_config(template_path)

    assert restored.model.action_dim == 56
    assert restored.data.base_config is not None
    assert restored.data.base_config.lerobot_datasets
