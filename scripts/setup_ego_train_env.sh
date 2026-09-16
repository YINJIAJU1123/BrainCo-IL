#!/usr/bin/env bash
set -uo pipefail
cd /mnt/data_nas/ruibin/BrainCo-IL
logdir=logs/egodex_smoke_20260916
export UV_CACHE_DIR=/mnt/data_nas/ruibin/.cache/uv
export GIT_LFS_SKIP_SMUDGE=1
python=/mnt/data_nas/ruibin/.venvs/ego-train/bin/python
rc=0
uv pip install --python "$python" --no-deps --editable . || rc=$?
if [[ "$rc" == 0 ]]; then
  "$python" -c 'import sys,jax,torch,flax,lerobot,orbax.checkpoint,optax,augmax,swanlab,openpi; print("Python:",sys.executable); print("JAX:",jax.__version__,jax.devices()); print("Torch:",torch.__version__); print("Flax:",flax.__version__); print("LeRobot:",lerobot.__file__); print("OpenPI:",openpi.__file__); assert any(d.platform=="gpu" for d in jax.devices())' || rc=$?
fi
echo "$rc" > "$logdir/ego-train-setup.exitcode"
echo "Setup finished; exit code=$rc"
exit "$rc"
