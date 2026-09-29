#!/usr/bin/env bash
set -euo pipefail
cd /mnt/data_nas/ruibin/BrainCo-IL
export PATH=/mnt/data_nas/ruibin/.venvs/ego-train/bin:$PATH
export OPENPI_DATA_HOME=/mnt/data_nas/ruibin/.cache/openpi
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export SWANLAB_MODE=disabled
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
recipe=src/openpi/training/training_config_template/ego/egodex_pi05_smoke_20260916.yaml
logdir=runs/egodex/smoke_20260916
mkdir -p "$logdir"
python -c 'import sys,jax,torch,lerobot; print(sys.version); print("JAX",jax.__version__,jax.devices()); print("Torch",torch.__version__); print("LeRobot",lerobot.__file__); assert any(d.platform=="gpu" for d in jax.devices())' > "$logdir/runtime.log" 2>&1
stats=assets/egodex_pi05_smoke_20260916/egodex_three_offset0/norm_stats.json
if [[ ! -f "$stats" ]]; then
  python scripts/compute_brainco_norm_stats_from_parquet.py /mnt/data_nas/ruibin/dataset/egodex_rviz_three_lerobot21 "$(dirname "$stats")" --action-horizon 16 --arm-dims 14 > "$logdir/norm_stats.log" 2>&1
fi
python scripts/ego/validate_egodex_smoke.py "$recipe" "$logdir/validation.json" > "$logdir/validation.log" 2>&1
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv -l 2 > "$logdir/gpu.csv" &
monitor_pid=$!
trap 'kill "$monitor_pid" 2>/dev/null || true' EXIT
python scripts/train.py "$recipe" > "$logdir/train.log" 2>&1
python scripts/train.py src/openpi/training/training_config_template/ego/egodex_pi05_smoke_20260916_resume.yaml > "$logdir/resume.log" 2>&1
python scripts/ego/infer_egodex_smoke.py checkpoints/egodex_pi05_smoke_20260916/three_offset0_bs2/25 "$logdir/inference.json" > "$logdir/inference.log" 2>&1
echo "Smoke training, restore and inference complete"
