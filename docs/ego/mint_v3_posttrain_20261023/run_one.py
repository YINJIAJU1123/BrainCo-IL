"""Run one authorized experiment and retain its GPU after training exits."""
import subprocess,sys,os,json,time
from pathlib import Path
R=Path('/mnt/data_nas/ruibin/ego-posttrain-prep/mint_v3_388_20261023');name=sys.argv[1]
p=subprocess.Popen([sys.executable,'-u','scripts/train.py',str(R/f'configs/{name}.yaml')])
(R/f'{name}_process.json').write_text(json.dumps({'supervisor_pid':os.getpid(),'training_pid':p.pid,'training_start_ticks':Path(f'/proc/{p.pid}/stat').read_text().split()[21],'gpu_uuid':os.environ['CUDA_VISIBLE_DEVICES'],'started':time.time()},indent=2))
rc=p.wait();(R/f'{name}_exit.json').write_text(json.dumps({'returncode':rc,'time':time.time()}));print('TRAIN_EXIT',rc,flush=True)
# Resource retention follows the existing user request. Only this assigned GPU.
import torch
torch.cuda.set_device(0)
x=torch.empty(9*1024**3,dtype=torch.uint8,device='cuda');x.zero_()
(R/f'{name}_hold.json').write_text(json.dumps({'pid':os.getpid(),'proc_start_ticks':Path(f'/proc/{os.getpid()}/stat').read_text().split()[21],'gpu_uuid':os.environ['CUDA_VISIBLE_DEVICES'],'bytes':x.numel(),'training_returncode':rc},indent=2))
while not (R/f'STOP_HOLD_{name}').exists():time.sleep(30)
