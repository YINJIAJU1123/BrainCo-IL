"""Policy transforms for Wuji robot (dual-arm dual-dexterous-hand).

Dataset structure:
- observation.state: configurable dims (dual arms + dual hands)
- action: configurable dims
- observation.images.cam_left_wrist: (480, 640, 3)
- observation.images.cam_right_wrist: (480, 640, 3)
- observation.images.stereo_right or cam_head: head / external camera
"""

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model

# Default Wuji robot action dimension for this deployment:
# dual arms 14 + left hand 21 + right hand 21.
WUJI_ACTION_DIM = 56


def make_wuji_example() -> dict:
    """Creates a random input example for the Wuji policy."""
    return {
        "observation/state": np.random.rand(WUJI_ACTION_DIM).astype(np.float32),
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
class WujiRevo3EefJointHandToJointHand(transforms.DataTransformFn):
    """Convert Revo3 70D EEF+joint+hand vectors to the deployed 56D joint+hand layout.

    Source layout:
    - left EEF pose: 7 dims
    - right EEF pose: 7 dims
    - left arm joints: 7 dims
    - right arm joints: 7 dims
    - left hand joints: 21 dims
    - right hand joints: 21 dims

    Target layout:
    - left arm joints
    - left hand joints
    - right arm joints
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
                *range(left_hand_start, left_hand_start + self.hand_dof),
                *range(right_arm_start, right_arm_start + self.arm_dof),
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
class WujiInputs(transforms.DataTransformFn):
    """Convert inputs from Wuji dataset to the format expected by Pi0 models.

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
class WujiOutputs(transforms.DataTransformFn):
    """Convert model outputs back to Wuji action format.

    Used for inference only.
    """

    action_dim: int = WUJI_ACTION_DIM

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"])
        return {"actions": actions[:, : self.action_dim]}
