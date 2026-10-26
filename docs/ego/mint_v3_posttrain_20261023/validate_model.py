"""Real GPU update and checkpoint roundtrip in isolated validation directory."""
from pathlib import Path
import json,dataclasses,functools,hashlib
import jax,numpy as np
from openpi.training import config_io,data_loader,video_loading,sharding,checkpoints
from scripts.train import init_train_state,train_step,_finalize_deployment_checkpoint
R=Path('/mnt/data_nas/ruibin/ego-posttrain-prep/mint_v3_388_20261023')
def main():
 assert json.loads((R/'loader_validation.json').read_text())['passed']
 cfg=config_io.load_train_config(R/'configs/e1_r1.yaml');cfg=dataclasses.replace(cfg,name='validation',exp_name='model_roundtrip',wandb_enabled=False)
 mesh=sharding.make_mesh(1);ds=jax.sharding.NamedSharding(mesh,jax.sharding.PartitionSpec(sharding.DATA_AXIS));rep=jax.sharding.NamedSharding(mesh,jax.sharding.PartitionSpec())
 opts=video_loading.VideoLoadingOptions(workers=2,prefetch_factor=2)
 loader=video_loading.create_data_loader(cfg,opts,sharding=ds)
 rng,init_rng=jax.random.split(jax.random.key(cfg.seed));state,ss=init_train_state(cfg,init_rng,mesh,resume=False);jax.block_until_ready(state)
 fn=jax.jit(functools.partial(train_step,cfg),in_shardings=(rep,ss,ds),out_shardings=(ss,rep),donate_argnums=(1,))
 it=iter(loader);results=[]
 for i in range(2):
  batch=next(it)
  masks={k:np.asarray(v).astype(bool).tolist() for k,v in batch[0].image_masks.items()}
  assert sum(masks['left_wrist_0_rgb'])==8 and all(masks['base_0_rgb'])
  with sharding.set_mesh(mesh):state,info=fn(rng,state,batch)
  metrics={k:float(v) for k,v in jax.device_get(info).items()};assert all(np.isfinite(v) for v in metrics.values());results.append(metrics);print('update',i,metrics,flush=True)
 manager,_=checkpoints.initialize_checkpoint_dir(cfg.checkpoint_dir,keep_period=None,overwrite=False,resume=False)
 config_io.save_train_config(cfg,cfg.checkpoint_dir);checkpoints.save_state(manager,state,loader,1);manager.wait_until_finished();config_io.save_train_config(cfg,cfg.checkpoint_dir/'1');_finalize_deployment_checkpoint(cfg,1)
 # Compare full restored state including optimizer using streaming hashes, then check deploy metadata.
 def digest(tree):
  h=hashlib.sha256()
  for a in jax.tree.leaves(tree):h.update(np.asarray(a).tobytes())
  return h.hexdigest()
 expected=digest(state);abstract,_=init_train_state(cfg,init_rng,mesh,resume=True)
 restored=checkpoints.restore_state(manager,abstract,loader,1);assert int(restored.step)==2 and digest(restored)==expected
 assert (cfg.checkpoint_dir/'1/policy_contract.json').is_file() and (cfg.checkpoint_dir/'1/deploy_ready.json').is_file()
 result={'passed':True,'updates':2,'finite_metrics':results,'full_state_optimizer_restore_equal':True,'restored_step':2,'deployment_contract_and_ready_present':True,'checkpoint':str(cfg.checkpoint_dir/'1'),'masks':masks}
 (R/'model_validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
