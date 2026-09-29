"""Execute the reviewed EgoDex recipe after validation, profiling and restore checks."""
import dataclasses
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from openpi.training import config_io


def main():
    root = Path(__file__).resolve().parents[2]
    logs = root/'logs/egodex_teleop_formal'
    logs.mkdir(parents=True, exist_ok=True)
    lock = (logs/'reviewed_training.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    recipe = root/'logs/egodex_teleop_review_20260916/train_config.yaml'
    cfg = config_io.load_train_config(recipe)
    stats = root/'assets/egodex_pi05_teleop_20260916/egodex_teleop_offset25_train/norm_stats.json'
    cfg = dataclasses.replace(cfg, policy_metadata={**(cfg.policy_metadata or {}),
        'normalization_sha256': hashlib.sha256(stats.read_bytes()).hexdigest(),
        'degenerate_quantile_fallback': 'training min/max for dimension 32 (left_thumb_DIP_joint)'})
    config_io.save_train_config(cfg, recipe.parent)
    env = dict(os.environ, PYTHONPATH=str(root/'src'), OPENPI_DATA_HOME=str(root.parent/'.cache/openpi'),
               HF_HUB_OFFLINE='1', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1', PYTHONUNBUFFERED='1',
               XLA_PYTHON_CLIENT_PREALLOCATE='false', SWANLAB_MODE='local',
               SWANLAB_LOG_DIR=str(root/'swanlog/egodex_teleop_20260916'))
    status = {'started_unix': time.time(), 'recipe': str(recipe), 'pid': os.getpid()}
    def update(stage, **fields):
        status.update(stage=stage, updated_unix=time.time(), **fields)
        tmp = logs/'reviewed_training_status.tmp'
        tmp.write_text(json.dumps(status, indent=2)); tmp.replace(logs/'reviewed_training_status.json')
        print(json.dumps(status), flush=True)
    def run(stage, args):
        update(stage)
        with (logs/(stage+'.log')).open('a') as output:
            p = subprocess.Popen([sys.executable, '-u', *map(str,args)], cwd=root, env=env,
                                 stdout=output, stderr=subprocess.STDOUT)
            update(stage, child_pid=p.pid)
            rc = p.wait()
        if rc:
            raise RuntimeError(f'{stage} failed with exit code {rc}; see {logs/(stage+".log")}')
    try:
        run('base_eval_stabilized', ['scripts/ego/evaluate_egodex_holdout.py','--recipe',recipe,
                                     '--output',logs/'base_evaluation_stabilized.json'])
        base = json.loads((logs/'base_evaluation_stabilized.json').read_text())
        assert base['samples']==59 and np.isfinite(base['loss']) and base['loss']<100
        profile = dataclasses.replace(cfg, exp_name='offset25_bs16_profile50_stabilized_v1',
            num_train_steps=50, log_interval=1, save_interval=49, keep_period=49)
        profile_recipe = config_io.save_train_config(profile, logs/'profile_stabilized_recipe')
        run('profile_stabilized', ['scripts/train.py', profile_recipe])
        metrics = [json.loads(line) for line in (profile.checkpoint_dir/'metrics.jsonl').read_text().splitlines()]
        assert len(metrics)==50 and metrics[-1]['step']==49
        assert all(np.isfinite(m['loss']) and m['applied_update_norm']>0 for m in metrics)
        peak = max(m.get('gpu_memory_mib',0) for m in metrics)
        assert 0<peak<78000, f'Insufficient memory headroom: {peak} MiB'
        resume = dataclasses.replace(profile, resume=True, num_train_steps=51)
        resume_recipe = config_io.save_train_config(resume,logs/'profile_stabilized_resume_recipe')
        run('profile_stabilized_resume', ['scripts/train.py',resume_recipe])
        restored = json.loads((profile.checkpoint_dir/'metrics.jsonl').read_text().splitlines()[-1])
        assert restored['step']==50 and restored['applied_update_norm']>0
        assert (profile.checkpoint_dir/'50/params').is_dir()
        (logs/'profile_results.json').write_text(json.dumps(dict(batch_size=16, steps=50, restored_steps=1,
            peak_sampled_gpu_memory_mib=peak, steady_seconds_per_step=float(np.mean([m['seconds_per_step'] for m in metrics[5:]])),
            loss_min=min(m['loss'] for m in metrics),loss_max=max(m['loss'] for m in metrics),
            checkpoint=str(profile.checkpoint_dir/'50')),indent=2))
        update('waiting_for_full_data_validation')
        deadline=time.monotonic()+7200
        for split,frames,episodes in [('train',121651,387),('val',15158,59)]:
            path=logs/(split+'_validation.json')
            while not path.exists():
                pid=int((logs/('validate_'+split+'.pid')).read_text())
                assert Path(f'/proc/{pid}/cmdline').exists() and Path(f'/proc/{pid}/cmdline').read_bytes(), f'{split} validator stopped'
                assert time.monotonic()<deadline, 'Data validation timeout'
                time.sleep(10)
            result=json.loads(path.read_text())
            assert result['frames']==frames and len(result['episodes'])==episodes
            assert result['augmentation_black_preserved']
        assert cfg.batch_size==16 and cfg.num_train_steps==10000 and not cfg.resume
        update('formal_training', checkpoint=str(cfg.checkpoint_dir))
        run('formal_training',['scripts/train.py',recipe])
        run('formal_evaluation',['scripts/ego/evaluate_egodex_holdout.py','--recipe',recipe,
             '--checkpoint',cfg.checkpoint_dir/'9999','--output',logs/'formal_evaluation.json'])
        update('complete')
    except BaseException as error:
        update('failed', error=str(error))
        raise


if __name__=='__main__':
    main()
