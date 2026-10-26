# MINT 388-episode post-training checkpoints

Three completed Ego:real mixture experiments, initialized independently from
EgoDex 100h mid-train checkpoint `offset0_bs16_steps20000_v1/19999` with fresh
optimizers. The real source is the original reviewed 55,243-frame recording.

| Ego:real | Final checkpoint | SwanLab |
|---|---|---|
| 1:1 | `mint388_e1_r1/6907` | https://swanlab.cn/@laryybb626/imitation/runs/53jbnc93 |
| 3:1 | `mint388_e3_r1/13815` | https://swanlab.cn/@laryybb626/imitation/runs/43erelqs |
| 5:2 | `mint388_e5_r2/12088` | https://swanlab.cn/@laryybb626/imitation/runs/j63wckkr |

Use each final checkpoint directory as the policy checkpoint root. Each contains
`params/`, `train_state/`, `assets/`, `train_config.yaml`, `policy_contract.json`,
`deploy_ready.json` and `input_pipeline.json`.

The original absolute NAS paths in the saved training metadata describe the
source experiment. The actual local checkpoint locations are the paths above
under `/home/phoebe/Brainco/egodex_pi05_posttrain_20261024_mint388/`.

The 388 approved episodes are split by complete source video into 345 train,
16 validation and 27 test episodes. Only the 62,469 training frames participate
in repeated Ego sampling; 17 quarantined episodes are excluded. The confirmed
prompt is `put bread into left basket and put fruit into right basket`.

Check `final_checkpoint_manifest.json` and `delivery_verification.json` for the
91-file SHA256 verification. Full model loading and robot performance evaluation
are pending. Training completion and transfer checks do not establish policy
success rate.
