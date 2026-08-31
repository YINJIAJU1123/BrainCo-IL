"""Generated deployment contract for BrainCo checkpoints.

The contract is a build artifact. Human-authored experiment YAML selects
semantic groups; exact names and ordering are resolved from LeRobot metadata.
"""

from __future__ import annotations

from collections.abc import Iterable
import hashlib
import json
from pathlib import Path
from typing import Any

CONTRACT_FILENAME = "policy_contract.json"
PROTOCOL_VERSION = 1
_GROUP_ORDER = ("left_arm", "right_arm", "left_hand", "right_hand")


def resolve_dataset_interface(dataset_roots: Iterable[str | Path], groups: Iterable[str]) -> dict[str, Any]:
    roots = [Path(root).expanduser() for root in dataset_roots]
    if not roots:
        raise ValueError("at least one LeRobot dataset is required")
    selected_groups = tuple(str(group) for group in groups)
    if not selected_groups:
        raise ValueError("policy groups must not be empty")
    unknown = sorted(set(selected_groups) - set(_GROUP_ORDER))
    if unknown:
        raise ValueError(f"unsupported policy groups: {unknown}")

    resolved = [_read_dataset_interface(root, selected_groups) for root in roots]
    reference = resolved[0]
    for root, current in zip(roots[1:], resolved[1:], strict=True):
        for key in (
            "state_names",
            "action_names",
            "state_indices",
            "action_indices",
            "cameras",
            "fps",
        ):
            if current[key] != reference[key]:
                raise ValueError(f"LeRobot datasets have incompatible {key}: {roots[0]} vs {root}")
    return reference


def build_policy_contract(
    train_config,
    checkpoint_id: str,
    *,
    dataset_roots: Iterable[str | Path] | None = None,
) -> dict[str, Any]:
    policy_io = train_config.data.policy_io
    roots = list(dataset_roots or _dataset_roots(train_config))
    interface = resolve_dataset_interface(roots, policy_io.state_groups)
    if tuple(policy_io.state_groups) != tuple(policy_io.action_groups):
        raise ValueError("protocol v1 requires identical state_groups and action_groups")
    if len(interface["action_names"]) != int(train_config.model.action_dim):
        raise ValueError(
            f"resolved action dim {len(interface['action_names'])} does not match model "
            f"action_dim {train_config.model.action_dim}"
        )

    joint_groups = {
        group: [name for name in interface["state_names"] if _joint_group(name) == group]
        for group in policy_io.state_groups
    }
    output_groups = {
        group: [name for name in interface["action_names"] if _joint_group(name) == group]
        for group in policy_io.action_groups
    }
    model_type = str(train_config.model.model_type.value)
    contract: dict[str, Any] = {
        "generated": True,
        "do_not_edit": True,
        "generated_by": "BrainCo-IL",
        "protocol_version": PROTOCOL_VERSION,
        "checkpoint_id": str(checkpoint_id),
        "robot_interface": interface["robot_type"],
        "required_joint_groups": joint_groups,
        "required_cameras": interface["cameras"],
        "output_joint_groups": output_groups,
        "chunk_steps": int(train_config.model.action_horizon),
        "policy_rate_hz": float(interface["fps"]),
        "inference_strategies": (["standard", "rtc"] if model_type in ("pi0", "pi05") else ["standard"]),
    }
    contract["contract_hash"] = _payload_hash(contract)
    return contract


def save_policy_contract(
    train_config,
    directory: str | Path,
    *,
    dataset_roots: Iterable[str | Path] | None = None,
) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    contract = build_policy_contract(
        train_config,
        checkpoint_id_for(train_config, directory),
        dataset_roots=dataset_roots,
    )
    path = directory / CONTRACT_FILENAME
    path.write_text(json.dumps(contract, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def load_policy_contract(path_or_dir: str | Path) -> dict[str, Any]:
    path = Path(path_or_dir).expanduser()
    if path.is_dir():
        path = path / CONTRACT_FILENAME
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"policy contract must be a JSON object: {path}")
    if payload.get("generated") is not True or payload.get("do_not_edit") is not True:
        raise RuntimeError(f"policy contract must be generated and immutable: {path}")
    if not str(payload.get("generated_by", "")).strip():
        raise RuntimeError(f"policy contract requires generated_by: {path}")
    expected = str(payload.get("contract_hash", ""))
    actual = _payload_hash(payload)
    if not expected or expected != actual:
        raise RuntimeError(
            f"generated policy contract was modified or is incomplete: {path}; regenerate it instead of editing it"
        )
    if int(payload.get("protocol_version", 0)) != PROTOCOL_VERSION:
        raise RuntimeError(
            f"unsupported policy protocol {payload.get('protocol_version')}; expected {PROTOCOL_VERSION}"
        )
    strategies = payload.get("inference_strategies")
    if strategies is not None and (
        not isinstance(strategies, list)
        or not strategies
        or len(strategies) != len(set(strategies))
        or set(strategies) - {"standard", "rtc"}
    ):
        raise RuntimeError("policy contract has invalid inference_strategies")
    return payload


def _read_dataset_interface(root: Path, groups: tuple[str, ...]) -> dict[str, Any]:
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"LeRobot metadata not found: {info_path}")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    features = info.get("features", {})
    state_feature = features.get("observation.state", {})
    action_feature = features.get("action", {})
    state_all = _feature_names(state_feature, "observation.state", info_path)
    action_all = _feature_names(action_feature, "action", info_path)
    # Group order is the model vector order. Within each group, retain the
    # exact order recorded by LeRobot metadata.
    state_indices = tuple(i for group in groups for i, name in enumerate(state_all) if _joint_group(name) == group)
    action_indices = tuple(i for group in groups for i, name in enumerate(action_all) if _joint_group(name) == group)
    state_names = [state_all[i] for i in state_indices]
    action_names = [action_all[i] for i in action_indices]
    if state_names != action_names:
        raise ValueError(
            f"selected observation.state/action names differ in {info_path}: state={state_names}, action={action_names}"
        )
    missing = [group for group in groups if not any(_joint_group(name) == group for name in state_names)]
    if missing:
        raise ValueError(f"LeRobot metadata {info_path} has no joints for groups {missing}")
    cameras = [str(key) for key in features if str(key).startswith("observation.images.")]
    if not cameras:
        raise ValueError(f"LeRobot metadata has no observation.images.* features: {info_path}")
    return {
        "robot_type": str(info.get("robot_type", "")),
        "fps": float(info.get("fps", 0.0)),
        "state_dim": len(state_all),
        "action_dim": len(action_all),
        "state_indices": state_indices,
        "action_indices": action_indices,
        "state_names": state_names,
        "action_names": action_names,
        "cameras": cameras,
        "group_dims": {group: sum(_joint_group(name) == group for name in state_names) for group in groups},
    }


def _feature_names(feature: Any, key: str, info_path: Path) -> list[str]:
    if not isinstance(feature, dict) or not isinstance(feature.get("names"), list):
        raise ValueError(f"LeRobot feature {key!r} has no names list: {info_path}")
    names = [str(name) for name in feature["names"]]
    shape = feature.get("shape", [])
    if shape and int(shape[-1]) != len(names):
        raise ValueError(f"LeRobot feature {key!r} shape/names mismatch: {info_path}")
    if len(set(names)) != len(names):
        raise ValueError(f"LeRobot feature {key!r} contains duplicate names: {info_path}")
    return names


def _joint_group(name: str) -> str | None:
    if name.startswith("left_arm_joint"):
        return "left_arm"
    if name.startswith("right_arm_joint"):
        return "right_arm"
    if name.startswith("left_") and name.endswith("_joint"):
        return "left_hand"
    if name.startswith("right_") and name.endswith("_joint"):
        return "right_hand"
    return None


def _dataset_roots(train_config) -> list[str]:
    base_config = getattr(train_config.data, "base_config", None)
    datasets = tuple(getattr(base_config, "lerobot_datasets", ()) or ())
    roots = [str(dataset.repo_id) for dataset in datasets]
    if not roots:
        repo_id = getattr(train_config.data, "repo_id", None)
        if repo_id:
            roots.append(str(repo_id))
    return roots


def checkpoint_id_for(train_config, directory: str | Path) -> str:
    directory = Path(directory)
    if directory.name.isdigit():
        return f"{train_config.exp_name}_step{directory.name}"
    return directory.name


def _payload_hash(payload: dict[str, Any]) -> str:
    unsigned = {key: value for key, value in payload.items() if key != "contract_hash"}
    data = json.dumps(unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(data).hexdigest()
