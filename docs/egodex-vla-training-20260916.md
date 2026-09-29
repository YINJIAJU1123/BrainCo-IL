# EgoDex → BrainCo-IL training record — 2026-09-16

This file is the historical 1 h / offset 0.025 m run record. For the completed 100 h / offset 0 batch and its 20k-step training plan, use `/mnt/data_nas/ruibin/egodex_lerobot_export_20260922/README.md` on A100. New Ego scripts are under `scripts/ego/`, versioned recipes under `src/openpi/training/training_config_template/ego/`, and new runtime files under ignored `runs/ego/`. The old `logs/egodex_*` paths below remain historical snapshots.

## Current formal run — 2026-09-16 evening

The teleop batch is complete, embedded and independently audited; the old HDF5 directory has been retired. LeRobot conversion is complete: train 387 segments / 121651 frames / 221 sources; val 59 segments / 15158 frames / 25 sources.

The reviewed run is PI0.5 LoRA plus action interface, 56D, horizon16, batch16, 10000 steps, from base. SwanLab local monitoring is enabled at server 127.0.0.1:5092, with logs in swanlog/egodex_teleop_20260916. Formal experiment: offset25_bs16_steps10000_v1. Reviewed YAML: logs/egodex_teleop_review_20260916/train_config.yaml.

Train-only histogram quantiles collapsed for left_thumb_DIP_joint (index32). Only this dimension uses train min/max bounds; all other statistics remain identical. Original stats and quantile_stabilization.json are retained beside norm_stats.json. Fixed holdout baseline now has mean loss1.460399668 over59 full-horizon samples. No validation data was used to estimate normalization bounds.

Batch16 profile passed50 updates plus1 restored update, peak sampled GPU memory16932MiB, loss1.401678–1.527411, measured steady mean6.169s/step including data loading. Formal run starts only after both full-loader checks pass. Supervised process: scripts/ego/run_egodex_reviewed_training.py. Live authoritative state: logs/egodex_teleop_formal/reviewed_training_status.json. Training completion automatically runs the same fixed holdout evaluation.

Detailed operation/recovery notes: logs/egodex_teleop_formal/运行说明.md. Code/config/stats hashes archived under logs/egodex_teleop_formal/run_provenance/.

Earlier sections below are historical execution records; use the current status file for live progress.

## Status

The three-episode engineering smoke PASSED: real LeRobot v2.1 loading and production transforms, 25 optimization steps, checkpoint restore followed by one more optimization step, and offline inference.
This establishes pipeline operability, not model quality, convergence, or robot readiness.

Full-batch acceptance/conversion/training is PENDING. At the post-smoke gate the teleop reprocessing report was still running (66/258 in the saved smoke summary check; 68/258 when testing the acceptance gate), with no reported failures in completed episodes. The acceptance script refused the incomplete batch. No formal dataset has been exported and no formal training has started. Reprocessing was neither restarted nor modified.

## Environment

Host: A100 instance dsw-l03y9n6cauzipk57is, IP 10.0.0.108 at execution.
Training repository: /mnt/data_nas/ruibin/BrainCo-IL
Data repository: /mnt/data_nas/ruibin/BrainCo-Dataset
Python: /mnt/data_nas/ruibin/.venvs/ego-train/bin/python (3.11.11)
JAX 0.5.3, Flax 0.10.2, Torch 2.7.1+cu126; JAX detected CudaDevice(id=0).
Dependencies were pinned by the repository uv.lock; mirror installation used its exported version/hash requirements.
Runtime versions and environment setup outcome are in logs/egodex_smoke_20260916/runtime.log and ego-train-setup.log (exit code 0).
The existing .venv-mint and repository .venv symlink were not changed.

Common environment:

```bash
cd /mnt/data_nas/ruibin/BrainCo-IL
export PATH=/mnt/data_nas/ruibin/.venvs/ego-train/bin:$PATH
export OPENPI_DATA_HOME=/mnt/data_nas/ruibin/.cache/openpi
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export SWANLAB_MODE=disabled
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
```

## Smoke data and semantics

Input: /mnt/data_nas/ruibin/dataset/egodex_rviz_three_lerobot21
30 FPS, 954 frames (31.8 seconds), three readable videos/parquets:

| Episode | Task | Frames | Sampled future arm delta maximum |
|---|---|---:|---:|
| 0 | add remove lid | 288 | 1.162180 |
| 1 | assemble disassemble structures | 508 | 0.733382 |
| 2 | assemble disassemble legos | 158 | 0.949118 |

All video frames were decoded and timestamps compared to parquet. Per-episode samples at start, middle, last full horizon and last frame verified loader chunk values and padding against parquet, including episode-end clamping.
The exact joint names were checked: arms Joint1–7 for each side; hands little/ring/middle/index MPR/MCP/PIP/DIP, thumb MCP/PIP/DIP/CMP/CMR.
Existing BrainCoRevo3EefJointHandToJointHand extracts 14:70 once. The transformed state is 56D; actions have horizon 16, arm14 relative to current state, hand42 absolute.
State is synthetic same-frame IK state, not robot feedback.
Smoke data uses 0 m extra outward offset per hand. Formal teleop batch uses 0.025 m per hand and is a separate data version.

## Camera contract

The LeRobot export contains three camera streams: a real head video and explicit all-zero wrist videos.
The training config does not generate wrist images. It sets `data.mask_wrist_images=true` so `BrainCoInputs` reads the wrist streams and emits masks: base_0_rgb=True, left_wrist_0_rgb=False, right_wrist_0_rgb=False.
Observation conversion maps uint8 black to float32 -1. Masked wrist images retain their supplied pixels through augmentation.
Pi0.embed_prefix repeats these masks over visual tokens. make_attn_mask excludes wrist tokens as attention keys/queries; inference suffix attention also uses the prefix mask. Image encoding still runs for placeholders.
Default BrainCoInputs remains three-camera with all masks True. Regression checks verified the wrist video streams are read, the wrist masks block attention, and YAML roundtrip retains `mask_wrist_images`.

## Training and measured results

Recipe: src/openpi/training/training_config_template/ego/egodex_pi05_smoke_20260916.yaml
Resume recipe: src/openpi/training/training_config_template/ego/egodex_pi05_smoke_20260916_resume.yaml
Formal resume template: src/openpi/training/training_config_template/ego/egodex_pi05_teleop_20260916_resume_30k.yaml
Runtime-generated config snapshots remain under `logs/egodex_teleop_formal/` and are ignored by Git.

## Deployment checkpoint artifact

Before opening a checkpoint with `vla_deploy`, package its deployment metadata:

```bash
python scripts/export_policy_contract.py /path/to/checkpoint
```

The command writes `policy_contract.json` and wraps `train_config.yaml` with the generated checkpoint metadata required by the deployment validator. The contract describes the 56D joint groups, three camera keys, 16-step action chunk, and 30 Hz policy rate. `--contract-only` leaves an already packaged `train_config.yaml` unchanged.

Artifact boundary: packaging only writes files inside the target checkpoint directory. It does not modify the 5090 deployment code or the 5090 BrainCo-IL repository. For the current uploaded checkpoint, the modified files are `/home/phoebe/Brainco/egodex_pi05_teleop_20260916/29999/policy_contract.json` and that directory's `train_config.yaml` metadata envelope; the embedded training configuration is preserved.
Experiment: egodex_pi05_smoke_20260916 / three_offset0_bs2
PI0.5, gemma_2b_lora + gemma_300m_lora, lora_and_action_interface, bfloat16, 56D, horizon16, batch2, workers2, seed42, peak LR 5e-5, smoke warmup2.
wandb_enabled=false; external experiment uploads disabled.
Base params: /mnt/data_nas/ruibin/.cache/openpi/openpi-assets/checkpoints/pi05_base/params
Smoke statistics: assets/egodex_pi05_smoke_20260916/egodex_three_offset0/norm_stats.json

| Item | Result |
|---|---|
| Optimization steps | 25 fresh + 1 restored |
| Loss range across all steps | 1.2568–1.5672 |
| First step | loss 1.4219, grad norm 1.5772, applied update norm 0.1164 |
| Last initial step (index24) | loss 1.4570, grad norm 1.5180, applied update norm 0.0990 |
| Restored step (index25) | loss 1.2568, grad norm 0.9916, applied update norm 0.0976 |
| Finite checks | loss/gradient/parameter/update norms finite at every logged step |
| Parameter change | nonzero actual applied parameter delta at every step |
| Peak sampled GPU memory | 16,886 MiB, sampled every 2 seconds |
| Offline inference | (16,56), all finite; range -2.835324 to 2.656865 |
| Inference output convention | absolute arms and hands after output transforms |

The checkpoint directory name is the zero-based loop index; saved train_state.step is index+1.
Checkpoint24 contains step25. Restore resumed at step25 and performed an update to step26, saving checkpoint25.
Inference used checkpoint25 and a real episode0 head image/state/prompt. Its first call including compilation took about 30.9 seconds.

Checkpoint roots:
- /mnt/data_nas/ruibin/BrainCo-IL/checkpoints/egodex_pi05_smoke_20260916/three_offset0_bs2/24
- /mnt/data_nas/ruibin/BrainCo-IL/checkpoints/egodex_pi05_smoke_20260916/three_offset0_bs2/25

Each contains params/, train_state/, assets/, train_config.yaml. The run root also has train_config.yaml.
Full logs and evidence:
- logs/egodex_smoke_20260916/validation.json, validation.log
- logs/egodex_smoke_20260916/camera_regression.json
- logs/egodex_smoke_20260916/norm_stats.log
- logs/egodex_smoke_20260916/train.log, resume.log, gpu.csv
- logs/egodex_smoke_20260916/inference.json, inference.npy, inference.log
- logs/egodex_smoke_20260916/smoke_results.json

## Commands used

With the common environment above:

```bash
python scripts/compute_brainco_norm_stats_from_parquet.py \
  /mnt/data_nas/ruibin/dataset/egodex_rviz_three_lerobot21 \
  assets/egodex_pi05_smoke_20260916/egodex_three_offset0 \
  --action-horizon 16 --arm-dims 14

python scripts/ego/validate_egodex_smoke.py \
  src/openpi/training/training_config_template/ego/egodex_pi05_smoke_20260916.yaml \
  logs/egodex_smoke_20260916/validation.json

python scripts/train.py src/openpi/training/training_config_template/ego/egodex_pi05_smoke_20260916.yaml
python scripts/train.py src/openpi/training/training_config_template/ego/egodex_pi05_smoke_20260916_resume.yaml

python scripts/ego/infer_egodex_smoke.py \
  checkpoints/egodex_pi05_smoke_20260916/three_offset0_bs2/25 \
  logs/egodex_smoke_20260916/inference.json
```

The completed end-to-end invocation was bash scripts/ego/train_egodex_smoke.sh. Re-running the fresh recipe against its existing checkpoint directory intentionally fails; choose a new experiment for another fresh smoke.

To continue the existing completed smoke by one additional step (26 → 27), create a resume config without overwriting the recorded recipes:

```bash
python - <<'PY'
import dataclasses
from pathlib import Path
from openpi.training import config_io
root = Path("checkpoints/egodex_pi05_smoke_20260916/three_offset0_bs2")
cfg = config_io.load_train_config(root / "25")
cfg = dataclasses.replace(cfg, resume=True, overwrite=False, num_train_steps=27)
config_io.save_train_config(cfg, Path("logs/egodex_smoke_20260916/resume_to27"))
PY
python scripts/train.py logs/egodex_smoke_20260916/resume_to27/train_config.yaml
```

## Formal batch gate and prepared conversion

Batch: /mnt/data_nas/ruibin/egodex_train_action_gap25_20260916_teleop
Old observations dependency: /mnt/data_nas/ruibin/egodex_train_action_gap25_20260915_1h
Keep the old directory: new HDF5 obs uses external links.

New preparation scripts (syntax checked; full-batch execution still pending):
- BrainCo-Dataset/scripts/audit_egodex_training_batch.py
- BrainCo-Dataset/scripts/export_egodex_training_lerobot21.py

The acceptance gate requires final complete report with all258 records, zero unresolved failures, matching model/URDF/offset/provenance, hand equality, joint limits, finite values, resolved observation links, recomputed serialized FK consistency, and accepted_segments matching current validity masks.
If failures occur, review them before deciding how to proceed. The old hand-conversion failure part1/boil_serve_egg/0 remains excluded from the258 input episodes.

Retain only continuous valid ranges with length>=16. The deterministic split hashes original source episode with seed label egodex-42 and selects26 original sources for val before segmentation, preventing source leakage.
The proposed exporter writes separate train/ and val/ LeRobot roots, independent native-resolution head videos from BGR HDF5 input, and directly exports named56D joint state/action (no EEF pose fields). The existing70D→56D transform explicitly accepts56D as a no-op; formal policy_io must declare dataset_state_dim=56 and indices0:56.
Export identity and per-episode progress allow resuming the identical export; a lock prevents duplicate exporters. Conversion is complete only when conversion_report.json exists with status complete.
Formal normalization must be computed from train/ only with horizon16 and arm14 delta. Formal recipe must use independent assets/experiment/checkpoints, load base params anew, resume=false, and not reuse smoke optimizer state or statistics.
The formal entry defaults to the verified batch2 and an initial one-epoch baseline, deriving steps as ceil(training_frames / batch_size). --batch-size, --epochs and --steps allow explicit budgets; larger batch sizes still require profiling.

Once reprocessing finishes, execute the acceptance gate first:

```bash
cd /mnt/data_nas/ruibin/BrainCo-Dataset
.venv-mint/bin/python scripts/audit_egodex_training_batch.py \
  --batch /mnt/data_nas/ruibin/egodex_train_action_gap25_20260916_teleop \
  --output /mnt/data_nas/ruibin/BrainCo-IL/logs/egodex_teleop_formal/acceptance.json
```

After successful acceptance and review, the prepared conversion command is:

```bash
.venv-mint/bin/python scripts/export_egodex_training_lerobot21.py \
  --audit /mnt/data_nas/ruibin/BrainCo-IL/logs/egodex_teleop_formal/acceptance.json \
  --reference /mnt/data_nas/ruibin/dataset/egodex_rviz_three_lerobot21/meta/info.json \
  --output /mnt/data_nas/ruibin/dataset/egodex_teleop_20260916_lerobot21 \
  --workers 2
```

## Remaining work and limitations

- Full reprocessing is incomplete. Accepted full-batch duration and split/conversion statistics are not yet available; do not reuse the old "about one hour" estimate.
- Run full audit and conversion, then verify real LeRobot loading/video timing/joint mapping on both exported splits.
- Compute training-only statistics, profile formal batch size, set a justified training budget and start from base.
- Smoke inference values are an engineering check only. No held-out model-quality evaluation or physical execution has been performed.
- IK acceptance does not establish bimanual, whole-body, or environment collision safety.
- Torchvision emitted its video API deprecation warning; pyav decoding succeeded. Orbax warned that saved sharding was used during restore; same single-A100 topology restored successfully.
- Existing user changes were preserved; no commits were created. 5090 system/control code was untouched.


## Unified formal entry and statistics clarification

The completion criterion is a usable full-data path: accepted HDF5 batch → source-disjoint train/val LeRobot → train-only normalization → real loader validation → base-initialized formal training. The new entry is scripts/ego/train_egodex_formal.py. After the batch completes:

```bash
cd /mnt/data_nas/ruibin/BrainCo-IL
/mnt/data_nas/ruibin/.venvs/ego-train/bin/python scripts/ego/train_egodex_formal.py
```

This runs acceptance, conversion, training-only statistics, train/val validation, recipe generation, training and offline inference. The default budget is a first one-epoch baseline at the verified batch2; explicit --steps or --epochs selects a different budget. --prepare-only stops after data and recipe validation; --resume explicitly requests formal checkpoint recovery.
The same entry refuses incomplete or failed upstream batches. Statistics provenance binds the normalization asset to the export identity, training root, horizon16 and arm14-delta convention.
The recipe generator and real LeRobot 56D path were regression-tested using the existing three episodes in an isolated fixture, including incomplete-batch refusal. Evidence: logs/egodex_smoke_20260916/formal_adapter_check/result.json. This does not constitute acceptance of the pending full batch.

Smoke q01/q99 were calculated for both state and actions, 56 values each, together with mean/std.
scripts/compute_brainco_norm_stats_from_parquet.py reads parquet via PyArrow, applies the 70D→56D conversion, builds clamped horizon16 chunks, subtracts current arm14 state from each future action, and updates openpi.shared.normalize.RunningStats.
RunningStats uses 5000-bin histograms per dimension and estimates the 1%/99% quantiles from cumulative counts. These are approximate histogram quantiles, not exact numpy.quantile over concatenated samples. State statistics contain954 samples; action statistics flatten954×16=15,264 action positions, including episode-end clamping consistent with training.
PI0.5 uses quantile normalization: (x-q01)/(q99-q01+1e-6)*2-1. This transform does not clip values outside the quantile interval.

## Historical old-batch cost

Source records: egodex_train_action_gap25_20260915_1h/nohup.log and report.json.
Wall clock: 2026-09-15 16:39:13+08:00 to 2026-09-16 06:11:40+08:00 =13h32m27s.
report.elapsed_this_run_seconds=48,528.873s =13h28m48.873s (inner processing timer).
The invocation used workers1 and CUDA_VISIBLE_DEVICES empty.
258 successful HDF5 episodes, one failed hand conversion;170,286 frames =94.60min source duration, with3603.067s =60.05min accepted under the old validity criteria.
Measured disk usage: output144,214,411,776 bytes (144.21GB /134.31GiB); source3,931,194,368 bytes (3.93GB /3.66GiB); entire batch148,147,237,888 bytes (148.15GB /137.97GiB).
The HDF5 files retain full episodes, including invalid frames and images; the output size is not a size estimate for an already-filtered one-hour LeRobot export. The archive zip is outside this batch directory and not included.
Correction: per-episode report.results records do include peak_rss_gib. Across259 records the largest logged value is25.9171GiB; this is a reported process RSS measurement, not the output file size or necessarily a whole-job simultaneous-memory peak.


## YAML training entry and worker increase — 2026-09-16 16:20

The canonical BrainCo-IL training entry remains scripts/train.py <recipe.yaml>. Smoke invoked that entry directly for both the initial25 steps and restore. The shell wrapper only sequences validation/training/inference and redirects logs. The optional formal orchestration utility can be run with --prepare-only to generate data/statistics and the formal YAML, then training is launched through the same scripts/train.py entry. Model/horizon/LoRA/camera/dataset/optimizer/checkpoint settings reside in YAML.

At the user's request, current arm-only reprocessing was increased from4 to10 workers on the10-core A100 instance. CPU cgroup quota=1,000,000/100,000=10 cores, cpuset0-9.
The old supervisor7365 was paused to prevent new dispatches; all active children finished.196 complete HDF5/JSON pairs were preserved. The supervisor then exited and the identical command resumed with --workers10, PID137166, same output directory and single-thread BLAS settings. The manifest-listed processing code hashes remain unchanged.
The196 HDF5 files were checked after restart: sizes and mtime_ns all unchanged.10 active solver children were observed, and report reconstructed196 successful records with0 failures.
Audit trail: /mnt/data_nas/ruibin/egodex_train_action_gap25_20260916_teleop/latest_worker_change.json and worker_changes/20260916_161939/.
This change applies to arm-only reprocessing, which reuses hand actions and linked images; it does not imply that10 simultaneous full decode/hand-retargeting jobs fit the same memory budget.

Exact old-batch source consumption:
-259 attempted source episodes:259 MP4 files totalling3,111,123,266 bytes and259 annotation HDF5 files totalling819,002,324 bytes; combined3,930,125,590 bytes (3.93GB /3.66GiB),5709.6 seconds of source video.
-The258 successful episodes alone consumed3,086,099,059 bytes of MP4 plus814,242,972 bytes of annotations =3,900,342,031 bytes;5676.2 seconds.
-The output HDF5 stores native1080x1920 uint8 BGR images as per-frame chunks with gzip level4, rather than retaining MP4 inter-frame compression. Output retains full successful episodes including invalid frames; the validity criteria identify3603.067 usable seconds.
-The local part1.zip is83,886,080,000 bytes (83.89GB /78.125GiB); only the259 attempted episode pairs above were consumed for this one-hour target.


## Branch synchronization and reprocessing completion

Both NAS BrainCo-IL and /Users/larrybb/PycharmProjects/BrainCo-IL now use the ego-train branch, created from refactor/code-redundancy-cleanup at d5ce5a4d10a58c608ce5321edd0d1a09160e2b36.
Only the14 EgoDex task files were copied from NAS to Mac and SHA256-compared. Existing user changes in statistics, merge, bread/turtle and local bctrl files were retained. Changes remain uncommitted; no push was performed.

The reprocessing report reached complete at2026-09-16 16:32:37+08:00:258 successful,0 failed,170286 frames. Ten-worker processing of the62 remaining episodes took about12m24s after the16:20:13 restart.
Current mask-valid duration is4579.5 seconds (76m19.5s), before discarding valid runs shorter than16 frames. Full acceptance was started after completion.
The original nohup.log remains the correct log for tail -f; it was appended across worker restart.
The134.31GiB historical size describes full old HDF5 data, not final LeRobot output. Final size must be measured after retained-segment video conversion.
