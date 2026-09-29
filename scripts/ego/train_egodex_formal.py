"""Run EgoDex formal acceptance, conversion, train-only statistics, validation and training."""
import argparse
import dataclasses
import fcntl
import json
import hashlib
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
DATA_REPO=ROOT.parent/"BrainCo-Dataset"
BATCH=ROOT.parent/"egodex_train_action_gap25_20260916_teleop"
DATASET=ROOT.parent/"dataset/egodex_teleop_20260916_lerobot21"
NAME="egodex_pi05_teleop_20260916"
ASSET_ID="egodex_teleop_offset25_train"

def make_config(train_root, train_frames, *, batch_size=2, epochs=1, steps=None):
    from openpi.training import config_io
    from openpi.training import config as training_config
    template=ROOT/"src/openpi/training/training_config_template/ego/egodex_pi05_teleop_20260916_resume_30k.yaml"
    base=config_io.load_train_config(template)
    if steps is None:
        steps=math.ceil(train_frames/batch_size)*epochs
    if batch_size<1 or steps<1 or epochs<1:
        raise ValueError("Batch size, epochs and steps must be positive")
    policy_io=dataclasses.replace(base.data.policy_io,dataset_state_dim=56,dataset_state_indices=tuple(range(56)))
    data=dataclasses.replace(
        base.data,repo_id=ASSET_ID,policy_io=policy_io,
        assets=training_config.AssetsConfig(assets_dir=str(ROOT/"assets"/NAME),asset_id=ASSET_ID),
        base_config=dataclasses.replace(base.data.base_config,lerobot_datasets=(
            training_config.LeRobotDataset(repo_id=str(train_root),weight=1.0),)))
    interval=min(1000,max(1,steps//5))
    cfg=dataclasses.replace(
        base,name=NAME,exp_name=f"offset25_bs{batch_size}_steps{steps}",data=data,batch_size=batch_size,
        num_train_steps=steps,num_workers=2,log_interval=10,save_interval=interval,keep_period=interval,
        lr_schedule=dataclasses.replace(base.lr_schedule,warmup_steps=min(1000,max(1,steps//10)),decay_steps=steps),
        resume=False,overwrite=False,wandb_enabled=False,
        policy_metadata=dict(dataset_version="egodex_20260916_teleop",offset_m_per_hand=0.025,
                             camera_masks=dict(head=True,left_wrist=False,right_wrist=False),
                             source_disjoint_validation=True,train_frames=train_frames,
                             effective_epochs=steps*batch_size/train_frames))
    return cfg

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batch",type=Path,default=BATCH)
    p.add_argument("--output",type=Path,default=DATASET)
    p.add_argument("--batch-size",type=int,default=2,help="Default is the verified single-A100 batch size")
    p.add_argument("--epochs",type=int,default=1,help="Initial formal baseline; steps are derived from accepted training frames")
    p.add_argument("--steps",type=int,help="Explicit optimization budget, overrides epochs")
    p.add_argument("--prepare-only",action="store_true")
    p.add_argument("--resume",action="store_true")
    args=p.parse_args()
    # Fail before conversion or training if the upstream batch is incomplete.
    report=json.loads((args.batch/"report.json").read_text())
    if report["status"]!="complete" or report["processed"]!=report["total_episodes"] or report["failed"]:
        raise RuntimeError(f"Upstream batch not accepted: status={report['status']}, processed={report['processed']}/{report['total_episodes']}, failed={report['failed']}")
    logdir=ROOT/"runs/egodex/teleop_formal"
    logdir.mkdir(parents=True,exist_ok=True)
    lock=(logdir/"pipeline.lock").open("a")
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    env=dict(os.environ,OPENPI_DATA_HOME=str(ROOT.parent/".cache/openpi"),XLA_PYTHON_CLIENT_PREALLOCATE="false",
             SWANLAB_MODE="disabled",HF_HUB_OFFLINE="1",OMP_NUM_THREADS="2",OPENBLAS_NUM_THREADS="1",PYTHONUNBUFFERED="1")
    python=str(ROOT.parent/".venvs/ego-train/bin/python")
    mint=str(DATA_REPO/".venv-mint/bin/python")
    def run(stage,command,cwd=ROOT):
        print(stage+": "+" ".join(map(str,command)),flush=True)
        with (logdir/(stage+".log")).open("a") as log:
            log.write("\nCOMMAND: "+" ".join(map(str,command))+"\n"); log.flush()
            subprocess.run(list(map(str,command)),cwd=cwd,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    audit=logdir/"acceptance.json"
    run("acceptance",[mint,DATA_REPO/"scripts/audit_egodex_training_batch.py","--batch",args.batch,"--output",audit],DATA_REPO)
    run("conversion",[mint,DATA_REPO/"scripts/export_egodex_training_lerobot21.py","--audit",audit,
                      "--reference",ROOT.parent/"dataset/egodex_rviz_three_lerobot21/meta/info.json",
                      "--output",args.output,"--workers","2"],DATA_REPO)
    conversion=json.loads((args.output/"conversion_report.json").read_text())
    assert conversion["status"]=="complete"
    train_root=args.output/"train"
    frames=conversion["splits"]["train"]["frames"]
    from openpi.training import config_io
    from openpi.training import config as training_config
    cfg=make_config(train_root,frames,batch_size=args.batch_size,epochs=args.epochs,steps=args.steps)
    if args.resume:
        cfg=dataclasses.replace(cfg,resume=True)
    norm=ROOT/"assets"/NAME/ASSET_ID
    identity=dict(
        export_identity_sha256=hashlib.sha256((args.output/"export_identity.json").read_bytes()).hexdigest(),
        train_root=str(train_root.resolve()),train_frames=frames,action_horizon=16,arm_dims=14,
        estimator="openpi.shared.normalize.RunningStats: 5000-bin approximate quantiles",
    )
    provenance=norm/"statistics_provenance.json"
    if (norm/"norm_stats.json").is_file():
        if not provenance.is_file() or json.loads(provenance.read_text())!=identity:
            raise RuntimeError("Existing formal statistics do not match the accepted training dataset")
    else:
        run("norm_stats",[python,ROOT/"scripts/compute_brainco_norm_stats_from_parquet.py",train_root,norm,
                          "--action-horizon","16","--arm-dims","14"])
        provenance.write_text(json.dumps(identity,indent=2))
    recipe=config_io.save_train_config(cfg,logdir/"recipe")
    run("validate_train",[python,ROOT/"scripts/ego/validate_egodex_smoke.py",recipe,logdir/"train_validation.json"])
    val_cfg=dataclasses.replace(cfg,data=dataclasses.replace(cfg.data,base_config=dataclasses.replace(
        cfg.data.base_config,lerobot_datasets=(training_config.LeRobotDataset(repo_id=str(args.output/"val"),weight=1.0),))))
    val_recipe=config_io.save_train_config(val_cfg,logdir/"val_validation_recipe")
    run("validate_val",[python,ROOT/"scripts/ego/validate_egodex_smoke.py",val_recipe,logdir/"val_validation.json"])
    prepared=dict(status="prepared",recipe=str(recipe),train_frames=frames,steps=cfg.num_train_steps,
                  batch_size=cfg.batch_size,checkpoint=str(cfg.checkpoint_dir),
                  train_only_statistics=str(norm/"norm_stats.json"),resume=args.resume)
    (logdir/"pipeline_status.json").write_text(json.dumps(prepared,indent=2))
    print(json.dumps(prepared,indent=2),flush=True)
    if args.prepare_only:
        return
    run("train",[python,ROOT/"scripts/train.py",recipe])
    checkpoint=cfg.checkpoint_dir/str(cfg.num_train_steps-1)
    run("inference",[python,ROOT/"scripts/ego/infer_egodex_smoke.py",checkpoint,logdir/"inference.json"])
    prepared["status"]="trained"
    (logdir/"pipeline_status.json").write_text(json.dumps(prepared,indent=2))

if __name__=="__main__":
    main()
