"""Validate actual EgoDex LeRobot videos, loader chunks and training transforms."""
import json
from pathlib import Path
import sys
import av
import jax
import numpy as np
import pyarrow.parquet as pq
from openpi.training import config_io, data_loader
from openpi import transforms
from openpi.models import model

def main(recipe, output):
    cfg = config_io.load_train_config(recipe)
    dc = cfg.data.create(cfg.assets_dirs, cfg.model)
    root = Path(dc.lerobot_datasets[0].repo_id)
    meta = json.loads((root/"meta/info.json").read_text())
    names = [f"{side}_arm_joint{i}" for side in ("left","right") for i in range(1,8)]
    names += [f"{side}_{finger}_{joint}_joint" for side in ("left","right")
              for finger,joints in [("little",("MPR","MCP","PIP","DIP")),("ring",("MPR","MCP","PIP","DIP")),
                                   ("middle",("MPR","MCP","PIP","DIP")),("index",("MPR","MCP","PIP","DIP")),
                                   ("thumb",("MCP","PIP","DIP","CMP","CMR"))] for joint in joints]
    for key in ("action","observation.state"):
        assert meta["features"][key]["names"][-56:] == names
    ds = data_loader.create_torch_dataset(dc,16,cfg.model)
    pre = transforms.compose([*dc.repack_transforms.inputs,*dc.data_transforms.inputs])
    full = data_loader.transform_dataset(ds,dc)
    results=[]; start=0
    for ep,path in enumerate(sorted(root.glob("data/**/*.parquet"))):
        tab=pq.read_table(path).to_pydict()
        a=np.asarray(tab["action"],np.float32); state=np.asarray(tab["observation.state"],np.float32)
        n=len(a)
        assert a.shape==state.shape and a.shape in ((n,70),(n,56))
        assert n>=16 and np.isfinite(a).all() and np.isfinite(state).all()
        joint_start=a.shape[1]-56
        np.testing.assert_allclose(tab["timestamp"],np.arange(n)/30,atol=1e-5)
        np.testing.assert_array_equal(tab["frame_index"],np.arange(n))
        np.testing.assert_array_equal(tab["episode_index"],np.full(n,ep))
        video=next(root.glob(f"videos/**/episode_{ep:06d}.mp4"))
        pts=[]
        with av.open(str(video)) as container:
            for frame in container.decode(video=0):
                pts.append(float(frame.pts*frame.time_base))
                rgb=frame.to_ndarray(format="rgb24")
                assert rgb.shape==tuple(meta["features"]["observation.images.cam_head"]["shape"])
        assert len(pts)==n
        np.testing.assert_allclose(pts,np.arange(n)/30,atol=1e-5)
        future_max=0.
        for i in (0,n//2,n-16,n-1):
            raw=ds[start+i]; out=pre(raw.copy())
            assert raw["prompt"]
            expected=a[np.minimum(i+np.arange(16),n-1),joint_start:].copy()
            expected[:,:14]-=state[i,joint_start:joint_start+14]
            np.testing.assert_allclose(out["state"],state[i,joint_start:])
            np.testing.assert_allclose(out["actions"],expected,atol=1e-6)
            np.testing.assert_array_equal(np.asarray(raw["action_is_pad"]),i+np.arange(16)>=n)
            future_max=max(future_max,float(np.max(np.abs(expected[1:,:14]))))
            for wrist in ("left_wrist_0_rgb","right_wrist_0_rgb"):
                assert out["image"][wrist].dtype==np.uint8 and not out["image"][wrist].any()
                assert not out["image_mask"][wrist]
            transformed=full[start+i]
            obs=model.Observation.from_dict(jax.tree.map(lambda x: np.asarray(x)[None],transformed))
            augmented=model.preprocess_observation(jax.random.key(i),obs,train=True)
            for wrist in ("left_wrist_0_rgb","right_wrist_0_rgb"):
                assert np.all(np.asarray(augmented.images[wrist])==-1)
            assert transformed["actions"].shape==(16,56)
        results.append(dict(episode=ep,frames=n,video_frames=len(pts),future_arm_delta_max=future_max,
                            task=raw["prompt"],boundary_padding="passed"))
        start+=n
    assert start==len(ds)==meta["total_frames"]
    assert any(r["future_arm_delta_max"]>0 for r in results), "All sampled future arm chunks are stationary"
    report=dict(frames=start,fps=30,episodes=results,joint_names=names,
                camera_masks=dict(head=True,left_wrist=False,right_wrist=False),
                augmentation_black_preserved=True,
                offset_m_per_hand=(cfg.policy_metadata or {}).get("offset_m_per_hand"),
                semantics="arm14 delta relative to current synthetic state; hand42 absolute",
                loader=type(ds._datasets[0]).__module__)
    Path(output).write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=="__main__":
    main(*sys.argv[1:])
