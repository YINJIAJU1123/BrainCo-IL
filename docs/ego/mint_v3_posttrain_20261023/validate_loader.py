import os
os.environ.update(CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
from pathlib import Path
import json,time,hashlib,dataclasses
import numpy as np
import torch
from openpi.training import config_io,data_loader,video_loading
R=Path('/mnt/data_nas/ruibin/ego-posttrain-prep/mint_v3_388_20261023');torch.set_num_threads(1)
def save(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
def equal(a,b):
 assert a.keys()==b.keys()
 for k in a:
  if isinstance(a[k],dict):equal(a[k],b[k])
  elif torch.is_tensor(a[k]):assert torch.equal(a[k],b[k]),k
  else:np.testing.assert_array_equal(a[k],b[k],err_msg=k)
def main():
 c=config_io.load_train_config(R/'configs/e1_r1.yaml');d=c.data.create(c.assets_dirs,c.model)
 raw=data_loader.create_torch_dataset(d,16,c.model)
 print('raw loader ready',flush=True)
 opt=video_loading.VideoLoadingOptions(workers=8,prefetch_factor=4)
 fast=video_loading.configure_sources(raw,d,opt)
 # Check every new episode end against final mapped action, preserving segment boundaries.
 m=json.loads((R/'approved_pool_manifest.json').read_text());byid={r['episode']:r for r in m['episodes']}
 prov=json.loads((R/'lerobot/train/meta/provenance.json').read_text())
 ego=fast._datasets[0];offset=0
 for r in prov['episodes']:
  a=np.load(byid[r['source_episode']]['mapped_actions']['path'])['actions'];idx=offset+len(a)-1
  q,pad=ego._get_query_indices(idx,r['episode_index']);v=ego._query_hf_dataset(q)['action'].numpy()
  np.testing.assert_array_equal(v,np.repeat(a[-1:],16,axis=0));assert pad['action_is_pad'].sum()==15
  offset+=len(a)
 # Actual raw video/action/prompt equality at beginning, interior and end of both sources.
 indices=[0,31,62468,62469,62469+31,62469+55242]
 durations=[]
 for ds in (raw,fast):
  t=time.perf_counter();samples=[ds[i] for i in indices];durations.append(time.perf_counter()-t)
  if ds is raw:base=samples
  else:
   for a,b in zip(base,samples):equal(a,b)
 transformed=data_loader.transform_dataset(fast,d)
 for idx in [0,62469]:
  sample=transformed[idx];assert np.isfinite(sample['actions']).all();print('transformed',idx,{k:np.asarray(v).shape for k,v in sample.items() if not isinstance(v,dict)},flush=True)
 plans={}
 for name in ['e1_r1','e3_r1','e5_r2']:
  cfg=config_io.load_train_config(R/f'configs/{name}.yaml');dc=cfg.data.create(cfg.assets_dirs,cfg.model);p=data_loader._coverage_plan(dc,16,raw)
  assert p.steps==cfg.num_train_steps
  allidx=np.asarray([p.batch_indices(i) for i in range(p.steps)])
  assert allidx.shape==(p.steps,16) and allidx.min()>=0 and allidx.max()<len(raw)
  assert (allidx>=62469).sum()==55264 and len(np.unique(p.real_order))==55243
  for start in range(0,p.ego_count,62469):assert len(np.unique(p.ego_order[start:start+62469]))==len(p.ego_order[start:start+62469])
  p.set_start_step(37);assert next(iter(p))==allidx[37].tolist()
  plans[name]={'steps':p.steps,'ego_count':p.ego_count,'ego_unique':len(np.unique(p.ego_order)),'real_count':p.real_count,'padding':p.padding,'sample_plan_sha256':hashlib.sha256(allidx.tobytes()).hexdigest()}
 # Equality gates above are mandatory. Small timing is diagnostic, not a broad throughput claim.
 evidence={'passed':True,'selected_options':dataclasses.asdict(opt),'all_train_episode_tails_checked':345,'raw_video_action_prompt_bitwise_equal_indices':indices,'raw_and_optimized_seconds':durations,'sampling':plans,'split':'source-video grouped; session independence unverified'}
 save(R/'loader_validation.json',evidence)
 # Profile opt-in uses exactly the checked config/source and these evidence bytes.
 source_sha={str(p.relative_to(R/'source')):hashlib.sha256(p.read_bytes()).hexdigest() for p in (R/'source/src/openpi/training').glob('*.py')}
 for name in plans:
  cfg=config_io.load_train_config(R/f'configs/{name}.yaml');folder=cfg.checkpoint_dir.parent/'.input_pipeline';folder.mkdir(parents=True,exist_ok=True)
  save(folder/f'{cfg.exp_name}.json',{'experiment':cfg.exp_name,'config_sha256':video_loading.config_sha256(cfg),'original_workers':cfg.num_workers,'options':dataclasses.asdict(opt),'evidence_path':str(R/'loader_validation.json'),'evidence_sha256':hashlib.sha256((R/'loader_validation.json').read_bytes()).hexdigest(),'source_sha256':source_sha})
 print(json.dumps(evidence),flush=True)
if __name__=='__main__':main()
