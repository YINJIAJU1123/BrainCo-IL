"""YAML IO for self-describing training checkpoints."""

from __future__ import annotations

import dataclasses
import importlib
import inspect
import logging
from pathlib import Path
from typing import Any

import flax.nnx as nnx
import yaml

from openpi.training import config as training_config

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
    """Write the fully resolved training config into a directory."""

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / TRAIN_CONFIG_YAML
    path.write_text(to_yaml(config), encoding="utf-8")
    return path


def load_train_config(path_or_dir: str | Path) -> training_config.TrainConfig:
    """Load a TrainConfig from a YAML file or a checkpoint directory."""

    path = find_train_config_yaml(path_or_dir)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"train config YAML must parse to a mapping: {path}")
    schema_version = int(payload.get("schema_version", 0) or 0)
    if schema_version != CONFIG_SCHEMA_VERSION:
        raise RuntimeError(
            f"unsupported train config schema_version={schema_version} in {path}; "
            f"expected {CONFIG_SCHEMA_VERSION}"
        )
    config_payload = payload.get("train_config")
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
    raise FileNotFoundError(
        f"{TRAIN_CONFIG_YAML} not found in {path} or its parent; new checkpoints must save it"
    )


def has_train_config_yaml(path_or_dir: str | Path) -> bool:
    try:
        find_train_config_yaml(path_or_dir)
    except FileNotFoundError:
        return False
    return True


def to_yaml(config: training_config.TrainConfig) -> str:
    payload = {
        "schema_version": CONFIG_SCHEMA_VERSION,
        "train_config": _to_plain(config),
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
        kwargs = {
            str(key): _from_plain(item)
            for key, item in fields.items()
            if str(key) in accepted_fields
        }
        return cls(**kwargs)
    return {key: _from_plain(item) for key, item in value.items()}


def _class_path(cls: type) -> str:
    return f"{cls.__module__}.{cls.__qualname__}"


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
