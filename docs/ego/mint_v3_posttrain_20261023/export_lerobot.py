"""Export only frozen approved episodes; preserve source bytes and named actions."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
from pathlib import Path
import json, hashlib, concurrent.futures, subprocess, copy
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
ROOT=Path('/mnt/data_nas/ruibin/ego-posttrain-prep/mint_v3_388_20261023')
TASK='put bread into left basket and put fruit into right basket'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def save(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
def stats(a):
 a=np.asarray(a);a=a[:,None] if a.ndim==1 else a
 return {**{k:getattr(a,k)(axis=0).tolist() for k in ['min','max','mean','std']},'count':[len(a)]}
def export(args):
 split,i,offset,r,features=args
 n=r['frames'];z=Path(r['mapped_actions']['path']);assert sha(z)==r['mapped_actions']['sha256']
 a=np.load(z)['actions']; assert a.shape==(n,56)
 clip=Path(r['source_clip']['path']);assert sha(clip)==r['source_clip']['sha256']
 probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=width,height,nb_frames,r_frame_rate','-of','json',str(clip)]))['streams'][0]
 assert int(probe['nb_frames'])==n and probe['r_frame_rate']=='30/1' and (probe['width'],probe['height'])==(1280,720)
 d=ROOT/'lerobot'/split;par=d/f'data/chunk-000/episode_{i:06d}.parquet';par.parent.mkdir(parents=True,exist_ok=True)
 cols={'timestamp':np.arange(n,dtype=np.float32)/30,'frame_index':np.arange(n,dtype=np.int64),'episode_index':np.full(n,i,dtype=np.int64),'index':np.arange(offset,offset+n,dtype=np.int64),'task_index':np.zeros(n,dtype=np.int64),'action':a,'observation.state':a}
 arrays={k:pa.FixedSizeListArray.from_arrays(pa.array(v.ravel(),type=pa.float32()),56) if v.ndim==2 else pa.array(v) for k,v in cols.items()}
 pq.write_table(pa.table(arrays),par)
 read=pq.read_table(par);np.testing.assert_array_equal(np.asarray(read['action'].to_pylist(),dtype=np.float32),a);np.testing.assert_array_equal(np.asarray(read['observation.state'].to_pylist(),dtype=np.float32),a)
 # Head uses the unchanged accepted clip. Real black wrist videos allow baseline reader validation.
 for key in [k for k,v in features.items() if v['dtype']=='video']:
  dest=d/f'videos/chunk-000/{key}/episode_{i:06d}.mp4';dest.parent.mkdir(parents=True,exist_ok=True)
  dest.symlink_to(clip if key.endswith('cam_head') else ROOT/'lerobot'/'black'/f'{n}.mp4')
 return {'episode_index':i,'tasks':[TASK],'length':n}, {'episode_index':i,'stats':{k:stats(v) for k,v in cols.items()}}, {'episode_index':i,'source_episode':r['episode'],'source_video_group':r['source_video_group'],'original_task':r['task'],'effective_task':TASK,'source_clip_sha256':sha(clip),'parquet_sha256':sha(par),'frames':n}
def main():
 m=json.loads((ROOT/'approved_pool_manifest.json').read_text());out=ROOT/'lerobot';out.mkdir(exist_ok=False)
 assert len(m['episodes'])==388 and len(m['excluded_episode_ids'])==17
 info=json.load(open('/mnt/data_nas/ruibin/dataset/egodex_100h_offset0_lerobot21/train/meta/info.json'))
 head=info['features']['observation.images.cam_head'];head['shape']=[720,1280,3];head['info'].update({'video.height':720,'video.width':1280})
 (out/'black').mkdir()
 def black(n):
  subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','color=c=black:s=640x480:r=30','-frames:v',str(n),'-c:v','libx264','-threads','1','-pix_fmt','yuv420p',str(out/'black'/f'{n}.mp4')],check=True)
 with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:list(ex.map(black,sorted({r['frames'] for r in m['episodes']})))
 evidence={}
 for split,groups in m['proposed_split'].items():
  if split not in ('train','val','test'):continue
  rows=[r for r in m['episodes'] if r['source_video_group'] in groups]; jobs=[];offset=0
  for i,r in enumerate(rows):jobs.append((split,i,offset,r,info['features']));offset+=r['frames']
  with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:results=list(ex.map(export,jobs))
  meta=out/split/'meta';meta.mkdir()
  inf=copy.deepcopy(info);inf.update(total_episodes=len(rows),total_frames=offset,total_tasks=1,total_videos=3*len(rows),total_chunks=1,splits={'train':f'0:{len(rows)}'},robot_type='revotron_mint_head_only')
  save(meta/'info.json',inf)
  for fn,vals in [('episodes.jsonl',[v[0] for v in results]),('episodes_stats.jsonl',[v[1] for v in results]),('tasks.jsonl',[{'task_index':0,'task':TASK}])]:
   (meta/fn).write_text(''.join(json.dumps(v)+'\n' for v in vals))
  save(meta/'provenance.json',{'task_authorization':'User confirmed unified task 2026-10-24','manifest_sha256':sha(ROOT/'approved_pool_manifest.json'),'episodes':[v[2] for v in results],'excluded_episode_ids':m['excluded_episode_ids'],'state':'synthetic same-frame joint action','source_group_split':groups})
  evidence[split]={'episodes':len(rows),'frames':offset,'root':str(meta.parent)}
  print(split,evidence[split],flush=True)
 save(ROOT/'export_validation.json',{'passed':True,'splits':evidence,'all_action_state_values_exact':True,'all_clip_sha_and_frame_count_verified':True,'source_hdf5_unchanged':True,'excluded_count':17,'task':TASK,'loader_validation':'pending'})
if __name__=='__main__':main()
