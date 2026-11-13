"""Resolve known recorder camera names without rewriting dataset metadata."""

from collections.abc import Iterable


_CAMERA_ALIASES = {
    "observation/image": ("observation.images.cam_head", "observation.images.head"),
    "observation/left_wrist_image": ("observation.images.cam_left_wrist", "observation.images.wrist_left"),
    "observation/right_wrist_image": ("observation.images.cam_right_wrist", "observation.images.wrist_right"),
}


def resolve_camera_bindings(cameras: Iterable[str]) -> dict[str, str]:
    """Bind each model view to exactly one metadata key, rejecting ambiguity."""
    available = set(cameras)
    bindings = {}
    for model_key, aliases in _CAMERA_ALIASES.items():
        matches = available.intersection(aliases)
        if len(matches) != 1:
            raise ValueError(f"expected exactly one camera for {model_key} from {aliases}, got {sorted(matches)}")
        bindings[model_key] = matches.pop()
    unsupported = available - set(bindings.values())
    if unsupported:
        raise ValueError(f"unsupported BrainCo cameras: {sorted(unsupported)}")
    return bindings
