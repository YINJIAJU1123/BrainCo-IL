from pathlib import Path
import yaml,json,hashlib
R=Path('/mnt/data_nas/ruibin/ego-posttrain-prep/mint_v3_388_20261023');S=R/'source'
(R/'configs').mkdir(exist_ok=True)
for name in ['e1_r1','e3_r1','e5_r2']:
 p=S/f'src/openpi/training/training_config_template/ego/posttrain_{name}.yaml';v=yaml.safe_load(p.read_text());f=v['train_config']['fields']
 f.update(name='mint_v3_388_posttrain_20261024',exp_name='mint388_'+name,checkpoint_base_dir=str(R/'checkpoints'),num_workers=8)
 b=f['data']['fields']['base_config']['fields'];b['coverage_ego_repeat']=True
 for i,d in enumerate(b['lerobot_datasets']):
  d['fields']['prompt_override']='put bread into left basket and put fruit into right basket'
  if i==0:d['fields']['repo_id']=str(R/'lerobot/train')
 f['policy_metadata'].update(ego_pool_frames=62469,ego_repeat='successive seeded permutations',approved_episodes=388,train_episodes=345,excluded_episodes=17,task_authorized='2026-10-24 unified by user',source_manifest_sha256=hashlib.sha256((R/'approved_pool_manifest.json').read_bytes()).hexdigest())
 (R/'configs'/f'{name}.yaml').write_text(yaml.safe_dump(v,sort_keys=False))
print('three configs written')
