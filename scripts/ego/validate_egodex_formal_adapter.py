"""Exercise formal recipe generation and 56D real-loader compatibility on the smoke data."""
import dataclasses
import json
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from openpi.training import config_io
from openpi.training import config as training_config
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.ego.train_egodex_formal import make_config, ROOT

root=ROOT/"logs/egodex_smoke_20260916/formal_adapter_check"
root.mkdir(parents=True,exist_ok=True)
source=ROOT.parent/"dataset/egodex_rviz_three_lerobot21"
dataset=root/"lerobot56"
(dataset/"meta").mkdir(parents=True,exist_ok=True)
info=json.loads((source/"meta/info.json").read_text())
for key in ("action","observation.state"):
    info["features"][key]["shape"]=[56]
    info["features"][key]["names"]=info["features"][key]["names"][14:]
(dataset/"meta/info.json").write_text(json.dumps(info))
for name in ("episodes.jsonl","tasks.jsonl"):
    shutil.copyfile(source/"meta"/name,dataset/"meta"/name)
episode_stats=[]
for line in (source/"meta/episodes_stats.jsonl").read_text().splitlines():
    item=json.loads(line)
    for key in ("action","observation.state"):
        for stat in ("min","max","mean","std"):
            item["stats"][key][stat]=item["stats"][key][stat][14:]
    episode_stats.append(item)
(dataset/"meta/episodes_stats.jsonl").write_text("".join(json.dumps(i)+"\n" for i in episode_stats))
if not (dataset/"videos").exists():
    (dataset/"videos").symlink_to(source/"videos",target_is_directory=True)
for path in source.glob("data/**/*.parquet"):
    table=pq.read_table(path)
    for key in ("action","observation.state"):
        values=np.asarray(table[key].to_pylist(),np.float32)[:,14:].copy()
        idx=table.schema.get_field_index(key)
        table=table.set_column(idx,key,pa.FixedSizeListArray.from_arrays(pa.array(values.ravel()),56))
    metadata=dict(table.schema.metadata or {})
    hf=json.loads(metadata[b"huggingface"])
    for key in ("action","observation.state"):
        hf["info"]["features"][key]["length"]=56
    metadata[b"huggingface"]=json.dumps(hf).encode()
    table=table.replace_schema_metadata(metadata)
    target=dataset/path.relative_to(source); target.parent.mkdir(parents=True,exist_ok=True)
    pq.write_table(table,target)
cfg=make_config(dataset,954)
assert cfg.num_train_steps==477 and cfg.batch_size==2 and cfg.resume is False
assert cfg.data.policy_io.dataset_state_dim==56 and cfg.data.policy_io.dataset_state_indices==tuple(range(56))
assert "pi05_base/params" in str(cfg.weight_loader.params_path)
assert "smoke" not in cfg.name
# This is only a shape/loader regression fixture; formal statistics remain untouched.
cfg=dataclasses.replace(cfg,data=dataclasses.replace(cfg.data,assets=training_config.AssetsConfig(
    assets_dir=str(ROOT/"assets/egodex_pi05_smoke_20260916"),asset_id="egodex_three_offset0")),
    policy_metadata=dict(offset_m_per_hand=0.0,purpose="56D compatibility regression using smoke data"))
recipe=config_io.save_train_config(cfg,root/"recipe")
subprocess.run([sys.executable,str(ROOT/"scripts/ego/validate_egodex_smoke.py"),str(recipe),str(root/"validation56.json")],check=True)
incomplete=root/"incomplete"; incomplete.mkdir(exist_ok=True)
(incomplete/"report.json").write_text(json.dumps(dict(status="running",processed=1,total_episodes=258,failed=0)))
result=subprocess.run([sys.executable,str(ROOT/"scripts/ego/train_egodex_formal.py"),"--batch",str(incomplete),"--prepare-only"],capture_output=True,text=True)
assert result.returncode!=0 and "Upstream batch not accepted" in result.stderr
(root/"result.json").write_text(json.dumps(dict(formal_recipe_generated=True,base_initialization=True,
    steps_derived_from_frames=True,real_lerobot56_validation=True,incomplete_batch_refused=True,
    note="This fixture is not formal dataset acceptance"),indent=2))
print("Formal recipe and real 56D loader regression passed",flush=True)
