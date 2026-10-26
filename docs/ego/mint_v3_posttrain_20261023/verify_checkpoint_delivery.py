"""Manifest final checkpoints and verify a byte-exact deployment delivery."""
from pathlib import Path
import argparse,hashlib,json,time
RUNS={'mint388_e1_r1':6907,'mint388_e3_r1':13815,'mint388_e5_r2':12088}
NORM='94398dac18e6717128104ff0d0cd5a15665c5fb451ae876c9c863f366a589eeb'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
def contracts(root):
 result={}
 for name,step in RUNS.items():
  p=root/name/str(step)
  for sub in ['params','train_state','assets']:assert (p/sub).is_dir(),(p,sub)
  c=json.loads((p/'policy_contract.json').read_text());d=json.loads((p/'deploy_ready.json').read_text())
  unsigned={k:v for k,v in c.items() if k!='contract_hash'}
  actual=hashlib.sha256(json.dumps(unsigned,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
  assert actual==c['contract_hash']==d['contract_hash']
  assert c['checkpoint_id']==d['checkpoint_id'] and c['chunk_steps']==d['chunk_steps']==16 and c['policy_rate_hz']==d['policy_rate_hz']==30
  assert sum(map(len,c['output_joint_groups'].values()))==d['action_dim']==56 and d['train_config_wrapped'] is True
  assert (p/'train_config.yaml').is_file()
  norm=list((p/'assets').rglob('norm_stats.json'));assert len(norm)==1 and sha(norm[0])==NORM
  result[name]={'step':step,'contract_hash':actual,'norm_sha256':NORM,'descriptor_files_present':True,'optimizer_state_included':True}
 return result
def main():
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['build','verify']);ap.add_argument('root',type=Path);ap.add_argument('manifest',type=Path);ap.add_argument('--result',type=Path);a=ap.parse_args()
 info=contracts(a.root)
 if a.mode=='build':
  files=[]
  for name,step in RUNS.items():
   for p in sorted((a.root/name/str(step)).rglob('*')):
    assert not p.is_symlink(),p
    if p.is_file():files.append({'path':str(p.relative_to(a.root)),'bytes':p.stat().st_size,'sha256':sha(p)})
  out={'created_unix':time.time(),'runs':info,'files':files,'total_bytes':sum(x['bytes'] for x in files),'total_files':len(files)}
  a.manifest.write_text(json.dumps(out,indent=2)+'\n')
 else:
  m=json.loads(a.manifest.read_text());actual=[]
  for name,step in RUNS.items():actual.extend(str(p.relative_to(a.root)) for p in (a.root/name/str(step)).rglob('*') if p.is_file())
  assert set(actual)=={x['path'] for x in m['files']}
  for row in m['files']:
   p=a.root/row['path'];assert p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],p
  out={'passed':True,'checked_unix':time.time(),'root':str(a.root),'files':m['total_files'],'bytes':m['total_bytes'],'runs':info,'all_sha256_match':True,'full_model_load_tested':False}
  if a.result:a.result.write_text(json.dumps(out,indent=2)+'\n')
 print(json.dumps({k:v for k,v in out.items() if k!='files'} if a.mode=='build' else out),flush=True)
if __name__=='__main__':main()
