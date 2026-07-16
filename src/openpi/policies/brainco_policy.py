"""Policy transforms and IO metadata for BrainCo robots."""

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model

# BrainCo robot action dimension: dual-arm dual-dexterous-hand
BRAINCO_ACTION_DIM = 56
JOINT_GROUP_DIMS = {
    "left_arm": 7,
    "right_arm": 7,
    "left_hand": 21,
    "right_hand": 21,
}


@dataclasses.dataclass(frozen=True)
class BrainCoPolicyIOConfig:
    """Serializable semantic contract shared by training and deployment."""

    state_groups: tuple[str, ...] = ("left_arm", "right_arm", "left_hand", "right_hand")
    action_groups: tuple[str, ...] = ("left_arm", "right_arm", "left_hand", "right_hand")
    delta_action_groups: tuple[str, ...] = ("left_arm", "right_arm")
    dataset_state_dim: int = 70
    dataset_state_indices: tuple[int, ...] | None = tuple(range(14, 70))

    @property
    def state_dim(self) -> int:
        return sum(self.group_dim(group) for group in self.state_groups)

    @property
    def action_dim(self) -> int:
        return sum(self.group_dim(group) for group in self.action_groups)

    def group_dim(self, group: str) -> int:
        try:
            return JOINT_GROUP_DIMS[group]
        except KeyError as exc:
            raise ValueError(
                f"unsupported BrainCo joint group {group!r}; expected one of {tuple(JOINT_GROUP_DIMS)}"
            ) from exc

    def delta_mask_dims(self) -> tuple[int, ...]:
        delta_groups = set(self.delta_action_groups)
        return tuple(
            self.group_dim(group) if group in delta_groups else -self.group_dim(group) for group in self.action_groups
        )

    def validate(self, model_action_dim: int) -> None:
        _validate_unique_groups("state_groups", self.state_groups)
        _validate_unique_groups("action_groups", self.action_groups)
        _validate_unique_groups("delta_action_groups", self.delta_action_groups)
        unknown_delta_groups = set(self.delta_action_groups) - set(self.action_groups)
        if unknown_delta_groups:
            raise ValueError(
                f"delta_action_groups must be a subset of action_groups, got {sorted(unknown_delta_groups)}"
            )
        if self.action_dim != model_action_dim:
            raise ValueError(
                f"policy IO action groups describe {self.action_dim} dims, model expects {model_action_dim}"
            )
        if self.dataset_state_dim <= 0:
            raise ValueError(f"dataset_state_dim must be > 0, got {self.dataset_state_dim}")
        if self.dataset_state_indices is not None:
            if len(self.dataset_state_indices) != self.state_dim:
                raise ValueError(
                    "dataset_state_indices length must match policy state_dim: "
                    f"{len(self.dataset_state_indices)} vs {self.state_dim}"
                )
            if len(set(self.dataset_state_indices)) != len(self.dataset_state_indices):
                raise ValueError("dataset_state_indices contains duplicate indices")
            if (
                min(self.dataset_state_indices, default=0) < 0
                or max(self.dataset_state_indices, default=-1) >= self.dataset_state_dim
            ):
                raise ValueError(f"dataset_state_indices must be within [0, {self.dataset_state_dim})")


def _validate_unique_groups(name: str, groups: tuple[str, ...]) -> None:
    if not groups:
        raise ValueError(f"{name} must not be empty")
    if len(set(groups)) != len(groups):
        raise ValueError(f"{name} contains duplicate groups: {groups}")
    for group in groups:
        if group not in JOINT_GROUP_DIMS:
            raise ValueError(f"{name} contains unsupported group {group!r}; expected one of {tuple(JOINT_GROUP_DIMS)}")


def make_brainco_example() -> dict:
    """Creates a random input example for the BrainCo policy."""
    return {
        "observation/state": np.random.rand(BRAINCO_ACTION_DIM).astype(np.float32),
        "observation/image": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        "observation/left_wrist_image": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        "observation/right_wrist_image": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        "prompt": "pick up the object with both hands",
    }


def _parse_image(image) -> np.ndarray:
    """Parse image to uint8 (H, W, C) format."""
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class BrainCoRevo3EefJointHandToJointHand(transforms.DataTransformFn):
    """Convert Revo3 70D EEF+joint+hand vectors to the original 56D dataset layout.

    Source layout:
    - left EEF pose: 7 dims
    - right EEF pose: 7 dims
    - left arm joints: 7 dims
    - right arm joints: 7 dims
    - left hand joints: 21 dims
    - right hand joints: 21 dims

    Target layout:
    - left arm joints
    - right arm joints
    - left hand joints
    - right hand joints
    """

    arm_dof: int = 7
    hand_dof: int = 21
    eef_pose_dof: int = 7

    def _target_indices(self) -> np.ndarray:
        left_eef_start = 0
        right_eef_start = left_eef_start + self.eef_pose_dof
        left_arm_start = right_eef_start + self.eef_pose_dof
        right_arm_start = left_arm_start + self.arm_dof
        left_hand_start = right_arm_start + self.arm_dof
        right_hand_start = left_hand_start + self.hand_dof

        return np.asarray(
            [
                *range(left_arm_start, left_arm_start + self.arm_dof),
                *range(right_arm_start, right_arm_start + self.arm_dof),
                *range(left_hand_start, left_hand_start + self.hand_dof),
                *range(right_hand_start, right_hand_start + self.hand_dof),
            ],
            dtype=np.int64,
        )

    def _convert(self, value: np.ndarray, key: str) -> np.ndarray:
        value = np.asarray(value)
        indices = self._target_indices()
        source_dim = 2 * self.eef_pose_dof + 2 * self.arm_dof + 2 * self.hand_dof
        target_dim = 2 * (self.arm_dof + self.hand_dof)

        if value.shape[-1] == target_dim:
            return value
        if value.shape[-1] != source_dim:
            raise ValueError(f"{key} has dim {value.shape[-1]}, expected {source_dim} or {target_dim}")
        return value[..., indices]

    def __call__(self, data: dict) -> dict:
        data["observation/state"] = self._convert(data["observation/state"], "observation/state")
        if "actions" in data:
            data["actions"] = self._convert(data["actions"], "actions")
        return data


@dataclasses.dataclass(frozen=True)
class BrainCoInputs(transforms.DataTransformFn):
    """Convert inputs from BrainCo dataset to the format expected by Pi0 models.

    Used for both training and inference.
    """

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        stereo_right_image = _parse_image(data["observation/image"])
        left_wrist_image = _parse_image(data["observation/left_wrist_image"])
        right_wrist_image = _parse_image(data["observation/right_wrist_image"])

        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": stereo_right_image,
                "left_wrist_0_rgb": left_wrist_image,
                "right_wrist_0_rgb": right_wrist_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }

        if "actions" in data:
            inputs["actions"] = data["actions"]

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class BrainCoOutputs(transforms.DataTransformFn):
    """Convert model outputs back to BrainCo action format.

    Used for inference only.
    """

    action_dim: int = BRAINCO_ACTION_DIM

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"])
        return {"actions": actions[:, : self.action_dim]}
