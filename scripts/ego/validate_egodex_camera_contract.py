"""Regression checks for missing-sensor masks and existing three-camera inputs."""
import json
import sys
from pathlib import Path
import numpy as np
from openpi.models import model
from openpi.models.pi0 import make_attn_mask
from openpi.policies.brainco_policy import BrainCoInputs, EgoDexMissingCameras
from openpi.training import config_io

def main(output):
    rgb=np.arange(12*16*3,dtype=np.uint8).reshape(12,16,3)
    state=np.zeros(56,np.float32)
    sample={"observation/state":state,"observation/image":rgb,
            "observation/left_wrist_image":np.full_like(rgb,91),
            "observation/right_wrist_image":np.full_like(rgb,173)}
    normal=BrainCoInputs(model.ModelType.PI05)(sample)
    assert all(normal["image_mask"].values())
    np.testing.assert_array_equal(normal["image"]["left_wrist_0_rgb"],sample["observation/left_wrist_image"])
    np.testing.assert_array_equal(normal["image"]["right_wrist_0_rgb"],sample["observation/right_wrist_image"])
    # Filling occurs before repack, and policy inputs independently enforce the same convention.
    raw=EgoDexMissingCameras()({"observation.images.cam_head":rgb})
    for side in ("left","right"):
        image=raw[f"observation.images.cam_{side}_wrist"]
        assert image.shape==rgb.shape and image.dtype==np.uint8 and not image.any()
    ego=BrainCoInputs(model.ModelType.PI05,egodex_head_only=True)(
        {"observation/state":state,"observation/image":rgb})
    assert ego["image_mask"]["base_0_rgb"]
    assert not ego["image_mask"]["left_wrist_0_rgb"] and not ego["image_mask"]["right_wrist_0_rgb"]
    np.testing.assert_array_equal(normal["image"]["base_0_rgb"],ego["image"]["base_0_rgb"])
    # Real model attention helper: the action position cannot attend missing wrist positions.
    valid=np.array([[True,False,False,True]],dtype=bool)
    attn=np.asarray(make_attn_mask(valid,np.array([False,False,False,True])))
    assert not attn[0,3,1:3].any() and attn[0,3,0] and attn[0,3,3]
    root=Path(__file__).resolve().parents[2]
    cfg=config_io.load_train_config(root/"src/openpi/training/training_config_template/ego/egodex_pi05_smoke_20260916.yaml")
    import yaml
    restored=config_io._from_plain(yaml.safe_load(config_io.to_yaml(cfg))["train_config"])
    assert restored.data.mask_wrist_images and not restored.data.egodex_head_only
    masked=BrainCoInputs(model.ModelType.PI05,mask_wrist_images=True)(sample)
    assert masked["image_mask"]["base_0_rgb"]
    assert not masked["image_mask"]["left_wrist_0_rgb"] and not masked["image_mask"]["right_wrist_0_rgb"]
    np.testing.assert_array_equal(masked["image"]["left_wrist_0_rgb"],sample["observation/left_wrist_image"])
    report=dict(three_camera_pixels_and_masks_unchanged=True,pre_repack_uint8_black=True,
                inference_head_only_supported=True,missing_wrist_attention_blocked=True,
                recipe_roundtrip_preserves_camera_contract=True)
    Path(output).write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=="__main__":
    main(*sys.argv[1:])
