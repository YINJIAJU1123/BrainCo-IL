"""CPU-only derivative preparation for the explicit 388 approved MINT episodes.

Reads the acceptance manifest and final HDF5s without changing them. Stores a
named 56D permutation for future LeRobot export, not a training split or launch.
"""
from __future__ import annotations
import os
os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
import concurrent.futures
import hashlib
import json
from pathlib import Path
import re
import collections
import h5py
import numpy as np
ROOT=Path('/mnt/data_nas/ruibin/runs/mint_ego_bread_fruit_v3/full_v3_20261023')
ACCEPT=ROOT/'acceptance_20261023'
OUT=Path('/mnt/data_nas/ruibin/ego-posttrain-prep/mint_v3_388_20261023')
REFERENCE=Path('/mnt/data_nas/ruibin/dataset/revotron_item_sorting_reviewed_v1/meta/info.json')
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as stream:
  for block in iter(lambda:stream.read(4*1024**2),b''):h.update(block)
 return h.hexdigest()
def save(path,value):
 path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)
def canonical_names():
 n=[f'{s}_arm_joint{j}' for s in ['left','right'] for j in range(1,8)]
 for s in ['left','right']:
  n += [f'{s}_{finger}_{j}_joint' for finger in ['little','ring','middle','index'] for j in ['MPR','MCP','PIP','DIP']]
  n += [f'{s}_thumb_{j}_joint' for j in ['MCP','PIP','DIP','CMP','CMR']]
 return n
def string(x):return x.decode() if isinstance(x,bytes) else str(x)
def process(row,names):
 sid=row['episode'];entry=row['entry'];artifacts=row['artifacts'];p=Path(artifacts['hdf5']['path'])
 assert sid in str(p) and p.is_file()
 expected=artifacts['hdf5']['sha256'];assert sha(p)==expected,(sid,'source HDF5 changed')
 values=[];native=[];mapped=[]
 with h5py.File(p,'r') as f:
  meta=f['meta'];assert meta.attrs['coordinate_frame']=='robot_base';assert float(meta.attrs['fps'])==30
  for kind,side in [('arm','left'),('arm','right'),('hand','left'),('hand','right')]:
   block=f[f'action/{kind}_joints_{side}'][:];source_names=[string(x) for x in meta.attrs[f'{kind}_joint_names_{side}']]
   assert len(source_names)==block.shape[1]
   for name in source_names:
    if kind=='arm':
     match=re.fullmatch(r'Joint([1-7])_([LR])',name);assert match and match[2]==('L' if side=='left' else 'R'),name
     target=f'{side}_arm_joint{match[1]}'
    else:
     prefix=('L' if side=='left' else 'R')+'_right_';assert name.startswith(prefix),name
     target=side+'_'+name[len(prefix):]
    native.append(name);mapped.append(target)
   values.append(block)
  raw=np.concatenate(values,axis=1)
  assert raw.shape==(entry['frames'],56) and raw.dtype==np.float32 and np.isfinite(raw).all()
  assert f['meta/arm_valid'][:].all()
  assert len(set(mapped))==56 and set(mapped)==set(names)
  permutation=[mapped.index(name) for name in names];ordered=raw[:,permutation]
  inverse=np.argsort(permutation);np.testing.assert_array_equal(ordered[:,inverse],raw)
  for j,name in enumerate(names):np.testing.assert_array_equal(ordered[:,j],raw[:,mapped.index(name)])
  frames=f['meta/source_frame_index'][:];assert len(frames)==len(raw) and (np.diff(frames)==1).all()
  provenance=json.loads(meta.attrs['segment_provenance_json']);assert provenance['episode_id']==sid
  assert provenance['sha256']==entry['sha256']
  action_sha=hashlib.sha256(ordered.tobytes()).hexdigest()
  out=OUT/'mapped_actions'/f'{sid}.npz'
  with out.open('xb') as stream:
   np.savez_compressed(stream,actions=ordered,joint_names=np.asarray(names),source_frame_index=frames)
  with np.load(out,allow_pickle=False) as check:np.testing.assert_array_equal(check['actions'],ordered)
  return dict(episode=sid,frames=len(raw),source_video_group=entry['source_segment']['source'],
              session_id=None,task=entry['task_description'],source_segment=entry['source_segment'],
              source_hdf5=dict(path=str(p),bytes=p.stat().st_size,sha256=expected),
              source_clip=artifacts['source'],prediction=artifacts['prediction'],
              mapped_actions=dict(path=str(out),bytes=out.stat().st_size,sha256=sha(out),array_sha256=action_sha),
              mapping=dict(source_names=native,source_semantics=mapped,target_names=names,permutation=permutation,
                           roundtrip_bitwise_equal=True,named_columns_bitwise_equal=True,
                           positional_copy_would_be_wrong=mapped!=names),
              action_semantics='absolute robot-base joint angles in radians; training arm delta applied later',
              proposed_state_semantics='synthetic state equal to mapped action at current frame; not measured feedback',
              numerical_status=row['status'],visual_status=row['visual_status'],
              split='unassigned_pending_source_group_decision',
              source_episode_boundary_preserved=True,head_mask=True,wrist_masks=[False,False])
def main():
 (OUT/'mapped_actions').mkdir(parents=True,exist_ok=False)
 paths={'sets':ACCEPT/'evidence/delivery_sets.json','ledger':ACCEPT/'evidence/ledger.json',
        'artifacts':ACCEPT/'delivery/artifact_manifest.json','reference_info':REFERENCE}
 # The archived ledger may be the acceptance root or its evidence subdirectory.
 if not paths['ledger'].exists():paths['ledger']=ACCEPT/'ledger.json'
 sets=json.loads(paths['sets'].read_text());ledger=json.loads(paths['ledger'].read_text());approved=set(sets['numerical_pass_visual_pending']);excluded=set(sets['quarantine'])
 assert len(approved)==388 and len(excluded)==17 and not approved&excluded
 rows=[r for r in ledger if r['episode'] in approved];assert len(rows)==388
 names=canonical_names();reference=json.loads(REFERENCE.read_text());reference_names=reference['features']['action']['names']
 assert reference_names[14:]==names and reference['features']['observation.state']['names']==reference_names
 with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
  records=list(pool.map(lambda row:process(row,names),rows))
 records.sort(key=lambda x:x['episode']);assert {r['episode'] for r in records}==approved
 groups=collections.defaultdict(lambda:dict(episodes=0,frames=0,tasks=collections.Counter()))
 for r in records:
  g=groups[r['source_video_group']];g['episodes']+=1;g['frames']+=r['frames'];g['tasks'][r['task']]+=1
 manifest=dict(schema_version=1,status='approved_pool_prepared_recipe_and_split_pending',training_started=False,
               acceptance=dict(numerical_pass_count=388,quarantine_count=17,visually_accepted_count=0,
                               quality_limit='partial visual sampling; grasp may remain too straight; no full visual admission'),
               source_manifests={k:dict(path=str(v),sha256=sha(v)) for k,v in paths.items()},
               approved_episode_ids=sorted(approved),excluded_episode_ids=sorted(excluded),
               total_frames=sum(r['frames'] for r in records),source_groups=dict(groups),
               proposed_split=dict(status='proposal_only',train=['GX010158','GX010159','GX010161','GX010165'],
                                   val=['GX010149'],test=['GX010151'],session_independence_unconfirmed=True),
               sessions='Unknown; grouping by source video prevents adjacent cuts crossing splits but does not prove session independence',
               joint_names=names,episodes=records)
 save(OUT/'approved_pool_manifest.json',manifest)
 save(OUT/'mapping_validation.json',dict(passed=True,episodes=388,frames=manifest['total_frames'],
      source_hdf5_current_sha_matches=True,named_mapping_and_roundtrip_bitwise_equal=True,
      affected_hand_fields=42,old_direct_exporter_not_used=True,reference_names_match=True,
      source_metadata_names_used=True,quarantine_actions_read=False,gpu_used=False))
 print(json.dumps(dict(passed=True,episodes=388,frames=manifest['total_frames'],groups=dict(groups),output=str(OUT))))
if __name__=='__main__':main()
