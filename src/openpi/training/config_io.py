"""YAML IO for self-describing training checkpoints."""

from __future__ import annotations

import dataclasses
import hashlib
import importlib
import inspect
import json
import logging
from pathlib import Path
from typing import Any

import flax.nnx as nnx
import yaml

from openpi.deploy.contract import checkpoint_id_for
from openpi.deploy.contract import resolve_dataset_interface
from openpi.models import act_config
from openpi.models import pi0_config
from openpi.policies import brainco_policy
from openpi.training import config as training_config
from openpi.training import optimizer
from openpi.training import weight_loaders

LOGGER = logging.getLogger(__name__)
TRAIN_CONFIG_YAML = "train_config.yaml"
CONFIG_SCHEMA_VERSION = 1
_UNSUPPORTED = object()
_REMOVED_FIELDS = {
    "action_space",
    "datasets",
    "pytorch_training_precision",
    "pytorch_weight_path",
    "rlds_data_dir",
}


def save_train_config(config: training_config.TrainConfig, directory: str | Path) -> Path:
    """Write an immutable, fully resolved training config into a directory."""

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / TRAIN_CONFIG_YAML
    path.write_text(
        to_yaml(config, checkpoint_id=checkpoint_id_for(config, directory)),
        encoding="utf-8",
    )
    return path


def load_train_config(path_or_dir: str | Path) -> training_config.TrainConfig:
    """Load a TrainConfig from a YAML file or a checkpoint directory."""

    path = find_train_config_yaml(path_or_dir)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"train config YAML must parse to a mapping: {path}")
    if "base" in payload:
        return _from_experiment_payload(payload, path)
    schema_version = int(payload.get("schema_version", 0) or 0)
    if schema_version != CONFIG_SCHEMA_VERSION:
        raise RuntimeError(
            f"unsupported train config schema_version={schema_version} in {path}; expected {CONFIG_SCHEMA_VERSION}"
        )
    if payload.get("generated") is not True or payload.get("do_not_edit") is not True:
        raise RuntimeError(f"checkpoint train_config.yaml must be generated and immutable: {path}")
    if not str(payload.get("generated_by", "")).strip():
        raise RuntimeError(f"checkpoint train_config.yaml requires generated_by: {path}")
    config_payload = payload.get("train_config")
    expected_hash = str(payload.get("content_sha256", ""))
    if expected_hash:
        actual_hash = _plain_hash(config_payload)
        if actual_hash != expected_hash:
            raise RuntimeError(f"generated train_config.yaml was modified: {path}; regenerate it instead of editing it")
    config = _from_plain(config_payload)
    if not isinstance(config, training_config.TrainConfig):
        raise RuntimeError(f"YAML did not describe a TrainConfig: {path}")
    return config


def find_train_config_yaml(path_or_dir: str | Path) -> Path:
    """Find train_config.yaml in a step checkpoint or its parent run directory."""

    path = Path(path_or_dir).expanduser().resolve()
    if path.is_file():
        return path
    candidates = [
        path / TRAIN_CONFIG_YAML,
        path.parent / TRAIN_CONFIG_YAML,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"{TRAIN_CONFIG_YAML} not found in {path} or its parent; new checkpoints must save it")


def has_train_config_yaml(path_or_dir: str | Path) -> bool:
    try:
        find_train_config_yaml(path_or_dir)
    except FileNotFoundError:
        return False
    return True


def validate_checkpoint_id(path_or_dir: str | Path, expected: str) -> None:
    """Bind a generated train config to the contract used for LOAD."""

    path = find_train_config_yaml(path_or_dir)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    checkpoint_id = str(payload.get("checkpoint_id", "")).strip()
    if checkpoint_id and checkpoint_id != str(expected):
        raise RuntimeError(
            f"train_config.yaml checkpoint_id differs from policy_contract.json: {checkpoint_id!r} != {expected!r}"
        )
    if not checkpoint_id:
        LOGGER.warning(
            "Legacy train_config.yaml has no checkpoint_id: %s; regenerate new checkpoints",
            path,
        )


def to_yaml(config: training_config.TrainConfig, *, checkpoint_id: str | None = None) -> str:
    config_payload = _to_plain(config)
    payload = {
        "schema_version": CONFIG_SCHEMA_VERSION,
        "generated": True,
        "do_not_edit": True,
        "generated_by": "BrainCo-IL",
        "checkpoint_id": checkpoint_id,
        "content_sha256": _plain_hash(config_payload),
        "train_config": config_payload,
    }
    return yaml.safe_dump(payload, sort_keys=False, allow_unicode=False)


def _to_plain(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple | list):
        items = [_to_plain(item) for item in value]
        return [item for item in items if item is not _UNSUPPORTED]
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            plain_item = _to_plain(item)
            if plain_item is not _UNSUPPORTED:
                out[str(key)] = plain_item
        return out
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _dataclass_to_plain(value)
    if value is nnx.Nothing:
        return _UNSUPPORTED
    return _UNSUPPORTED


def _dataclass_to_plain(value: Any) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    for field in dataclasses.fields(value):
        plain_value = _to_plain(getattr(value, field.name))
        if plain_value is _UNSUPPORTED:
            LOGGER.debug(
                "Skipping unsupported TrainConfig field %s.%s while writing YAML",
                type(value).__name__,
                field.name,
            )
            continue
        fields[field.name] = plain_value
    return {
        "__class__": _class_path(type(value)),
        "fields": fields,
    }


def _from_plain(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_from_plain(item) for item in value)
    if not isinstance(value, dict):
        return value
    class_name = value.get("__class__")
    if class_name:
        cls = _import_class(str(class_name))
        fields = value.get("fields", {})
        if not dataclasses.is_dataclass(cls):
            raise RuntimeError(f"serialized class is not a dataclass: {class_name}")
        if not isinstance(fields, dict):
            raise RuntimeError(f"serialized fields for {class_name} must be a mapping")
        accepted_fields = set(inspect.signature(cls).parameters)
        unknown_fields = sorted(set(fields) - accepted_fields)
        if unknown_fields:
            log = LOGGER.info if set(unknown_fields).issubset(_REMOVED_FIELDS) else LOGGER.warning
            log("Ignoring removed or unknown fields for %s: %s", class_name, ", ".join(unknown_fields))
        kwargs = {str(key): _from_plain(item) for key, item in fields.items() if str(key) in accepted_fields}
        return cls(**kwargs)
    return {key: _from_plain(item) for key, item in value.items()}


def _class_path(cls: type) -> str:
    return f"{cls.__module__}.{cls.__qualname__}"


def _plain_hash(payload: Any) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(data).hexdigest()


def _from_experiment_payload(payload: dict[str, Any], path: Path) -> training_config.TrainConfig:
    """Expand the concise, human-authored experiment schema."""

    if int(payload.get("schema_version", CONFIG_SCHEMA_VERSION)) != CONFIG_SCHEMA_VERSION:
        raise RuntimeError(f"unsupported experiment schema in {path}")
    base = str(payload.get("base", "")).strip().lower()
    if base not in ("pi05", "act"):
        raise ValueError(f"experiment base must be 'pi05' or 'act', got {base!r} ({path})")

    experiment = dict(payload.get("experiment", {}) or {})
    policy = dict(payload.get("policy", {}) or {})
    training = dict(payload.get("training", {}) or {})
    name = str(experiment.get("name", path.stem)).strip()
    dataset_value = experiment.get("dataset") or experiment.get("datasets")
    if isinstance(dataset_value, str):
        dataset_roots = [dataset_value]
    elif isinstance(dataset_value, list):
        dataset_roots = [str(item) for item in dataset_value]
    else:
        raise ValueError(f"experiment.dataset must be a path or list of paths ({path})")
    groups = tuple(str(group) for group in policy.get("groups", ("left_arm", "right_arm", "left_hand", "right_hand")))
    unknown_policy = sorted(set(policy) - {"groups", "action_horizon"})
    if unknown_policy:
        raise ValueError(
            f"unknown concise policy fields: {unknown_policy} ({path}); deployment actions are always absolute"
        )
    interface = resolve_dataset_interface(dataset_roots, groups)
    action_dim = len(interface["action_names"])
    action_horizon = int(policy.get("action_horizon", 32))

    model_overrides = dict(payload.get("model", {}) or {})
    if base == "pi05":
        model = pi0_config.Pi0Config(
            action_dim=action_dim,
            action_horizon=action_horizon,
            pi05=True,
            max_token_len=int(model_overrides.pop("max_token_len", 256)),
            discrete_state_input=bool(model_overrides.pop("discrete_state_input", True)),
            **model_overrides,
        )
        weight_loader = weight_loaders.PartialCheckpointWeightLoader(
            str(training.pop("pretrained_params", "gs://openpi-assets/checkpoints/pi05_base/params")),
            skip_on_mismatch_regex=r".*(action_in_proj|action_out_proj|state_proj).*",
        )
        lr_schedule = optimizer.CosineDecaySchedule(
            warmup_steps=int(training.pop("warmup_steps", 1000)),
            peak_lr=float(training.pop("peak_lr", 5e-5)),
            decay_steps=int(training.get("num_steps", 30_000)),
            decay_lr=float(training.pop("decay_lr", 5e-6)),
        )
    else:
        model = act_config.ACTConfig(
            action_dim=action_dim,
            action_horizon=action_horizon,
            **model_overrides,
        )
        weight_loader = weight_loaders.NoOpWeightLoader()
        lr_schedule = optimizer.CosineDecaySchedule(
            warmup_steps=int(training.pop("warmup_steps", 1000)),
            peak_lr=float(training.pop("peak_lr", 1e-4)),
            decay_steps=int(training.get("num_steps", 30_000)),
            decay_lr=float(training.pop("decay_lr", 1e-5)),
        )

    datasets = tuple(
        training_config.LeRobotDataset(
            repo_id=root,
            weight=1.0 / len(dataset_roots),
        )
        for root in dataset_roots
    )
    policy_io = brainco_policy.BrainCoPolicyIOConfig(
        state_groups=groups,
        action_groups=groups,
        delta_action_groups=tuple(group for group in groups if group.endswith("_arm")),
        dataset_state_dim=int(interface["state_dim"]),
        dataset_state_indices=tuple(interface["state_indices"]),
        dataset_action_dim=int(interface["action_dim"]),
        dataset_action_indices=tuple(interface["action_indices"]),
        group_dims=dict(interface["group_dims"]),
    )
    data = training_config.LeRobotBrainCoDataConfig(
        repo_id=name,
        assets=training_config.AssetsConfig(asset_id=name),
        base_config=training_config.DataConfig(
            lerobot_datasets=datasets,
            prompt_from_task=True,
            multi_dataset_mode=str(experiment.get("multi_dataset_mode", "concat")),
        ),
        # Delta encoding is an implementation detail of the training pipeline.
        # The experiment schema and deployment protocol are always absolute.
        extra_delta_transform=True,
        policy_io=policy_io,
        head_camera_key=_head_camera(interface["cameras"]),
        revo3_eef_joint_hand_to_joint_hand=False,
        action_sequence_keys=("action",),
    )
    allowed_training = {
        "seed",
        "batch_size",
        "num_workers",
        "log_interval",
        "save_interval",
        "keep_period",
        "save_final_checkpoint",
        "overwrite",
        "resume",
        "wandb_enabled",
        "fsdp_devices",
    }
    unknown = sorted(set(training) - allowed_training - {"num_steps", "checkpoint_base_dir", "assets_base_dir"})
    if unknown:
        raise ValueError(f"unknown concise training fields: {unknown} ({path})")
    train_kwargs = {key: training[key] for key in allowed_training if key in training}
    return training_config.TrainConfig(
        name=name,
        project_name=str(experiment.get("project_name", "imitation")),
        exp_name=str(experiment.get("exp_name", name)),
        model=model,
        weight_loader=weight_loader,
        lr_schedule=lr_schedule,
        ema_decay=None,
        data=data,
        checkpoint_base_dir=str(training.get("checkpoint_base_dir", "./checkpoints")),
        assets_base_dir=str(training.get("assets_base_dir", "./assets")),
        num_train_steps=int(training.get("num_steps", 30_000)),
        **train_kwargs,
    )


def _head_camera(cameras: list[str]) -> str:
    for key in cameras:
        if key.endswith(".cam_head"):
            return key
    raise ValueError(f"dataset must provide observation.images.cam_head, got {cameras}")


def _import_class(class_path: str) -> type:
    if not class_path.startswith("openpi."):
        raise RuntimeError(f"refusing to import non-openpi config class: {class_path}")
    module_name, _, qualname = class_path.rpartition(".")
    module = importlib.import_module(module_name)
    obj: Any = module
    for part in qualname.split("."):
        obj = getattr(obj, part)
    if not isinstance(obj, type):
        raise RuntimeError(f"serialized object is not a class: {class_path}")
    return obj
