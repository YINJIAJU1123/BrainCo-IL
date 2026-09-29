"""Offline EgoDex inference using the saved config and normalization assets."""
import json
from pathlib import Path
import sys
import numpy as np
from openpi.training import config_io, data_loader
from openpi.policies import policy_config

def main(checkpoint, output):
    cfg=config_io.load_train_config(checkpoint)
    dc=cfg.data.create(cfg.assets_dirs,cfg.model)
    ds=data_loader.create_torch_dataset(dc,16,cfg.model)
    raw=ds[0]
    policy=policy_config.create_trained_policy(cfg,checkpoint,sample_kwargs={"num_steps":10})
    result=policy.infer({"observation/state":np.asarray(raw["observation.state"]),
                        "observation/image":np.asarray(raw[cfg.data.head_camera_key]),
                        "prompt":raw["prompt"]})
    actions=np.asarray(result["actions"])
    assert actions.shape==(16,56) and np.isfinite(actions).all()
    np.save(Path(output).with_suffix(".npy"),actions)
    report=dict(checkpoint=str(checkpoint),shape=list(actions.shape),finite=True,
                min=float(actions.min()),max=float(actions.max()),task=raw["prompt"],
                output_semantics="absolute arm and hand joint targets after policy output transforms",
                camera_masks=dict(head=True,left_wrist=False,right_wrist=False),
                timings=result.get("policy_timing",{}))
    Path(output).write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
if __name__=="__main__":
    main(*sys.argv[1:])
