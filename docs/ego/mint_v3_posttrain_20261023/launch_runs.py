"""One launch owner; fresh single-GPU experiments after recorded validation."""
import os,json,subprocess,hashlib,time
from pathlib import Path
R=Path('/mnt/data_nas/ruibin/ego-posttrain-prep/mint_v3_388_20261023')
GPUS={'e1_r1':(0,'GPU-412f73ba-2ae4-4b78-512f-0bb002a01569'),'e3_r1':(4,'GPU-02bab9e1-3f3c-9871-ee5f-ff6f3b34b1e3'),'e5_r2':(6,'GPU-0bc4d417-338f-f420-aa4f-ae2057ce49a1')}
PYTHON='/mnt/data_nas/ruibin/.venvs/ego-train/bin/python'
def main():
 for f in ['export_validation.json','loader_validation.json','model_validation.json']:assert json.loads((R/f).read_text())['passed'],f
 # Exclusive file prevents double launch even after interruption.
 with (R/'launch_claim.json').open('x') as f:json.dump({'pid':os.getpid(),'time':time.time()},f)
 source={str(p.relative_to(R/'source')):hashlib.sha256(p.read_bytes()).hexdigest() for p in (R/'source').rglob('*.py')}
 (R/'training_source_sha256.json').write_text(json.dumps(source,indent=2)+'\n')
 rows=[]
 for name,(index,uuid) in GPUS.items():
  usage=int(subprocess.check_output(['nvidia-smi',f'--id={uuid}','--query-gpu=memory.used','--format=csv,noheader,nounits']).decode().strip());assert usage<100,(index,usage)
  env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=uuid,JAX_PLATFORMS='cuda',XLA_PYTHON_CLIENT_PREALLOCATE='true',XLA_PYTHON_CLIENT_MEM_FRACTION='0.75',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONPATH=str(R/'source/src')+':'+str(R/'source'),SWANLAB_MODE='cloud',SWANLAB_LOG_DIR=str(R/'swanlog'/name))
  with (R/f'{name}.log').open('xb') as log:
   p=subprocess.Popen([PYTHON,'-u',str(R/'code/run_one.py'),name],cwd=R/'source',env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
  rows.append({'experiment':'mint388_'+name,'ratio':name,'gpu_index':index,'gpu_uuid':uuid,'supervisor_pid':p.pid,'proc_start_ticks':Path(f'/proc/{p.pid}/stat').read_text().split()[21],'config_sha256':hashlib.sha256((R/f'configs/{name}.yaml').read_bytes()).hexdigest()})
  (R/'launch_status.json').write_text(json.dumps({'runs':rows,'time':time.time()},indent=2)+'\n')
 print(json.dumps(rows),flush=True)
if __name__=='__main__':main()
