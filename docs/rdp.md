# Experimental tactile RDP

This baseline implements per-finger marker PCA, an action-only variational CNN
encoder with a tactile-conditioned causal GRU decoder (AT), and an image/state/
tactile-conditioned diffusion policy over the frozen AT's latent actions (LDP).
It uses the official RDP temporal U-Net (see THIRD_PARTY_NOTICES.md).

The lamp dataset provides five left-finger `20x20x6` marker arrays. Only
`offset_x, offset_y` enter PCA. Each finger has 15 components; train-only score
standardization produces 75 features. The last feature is causal adapter hold
age, scaled by 0.1 s. Pressure, zero wrench fields and tactile RGB are unused.
PCA is fit on six training episodes; the final entire episode is held out from
PCA, canonical normalization and policy fitting. PCA has no action supervision.
Marker source capture timestamps are absent, so recorded 10 Hz polling is a
controlled sample-and-hold approximation, not a measured sensor-latency model.

The action rate remains the dataset's 30 Hz; the latent action chunk has 16
steps (0.533 s). AT downsamples time by four, with a 16-dimensional Gaussian
latent per remaining time step, L1 reconstruction and KL weight 1e-6. The GRU
receives the flattened latent and current touch. LDP uses three independent
ResNet18+GroupNorm encoders at 128px, a [128,256,512] temporal U-Net, 100 diffusion
training steps and 10-step DDIM inference. LDP targets posterior means from the
frozen AT; latent mean/std are fitted on training windows. LDP uses EMA 0.999.
These are BrainCo baseline choices, not paper-exact reproduction settings.

## Data and training

```bash
python -m openpi.rdp.prepare --dataset /path/to/original_lerobot --output /path/to/cache
```

The original dataset is not modified. Cache includes PCA assets, an audit,
per-frame state/action/tactile arrays, three resized RGB arrays and a linked
`train_dataset` view. Use a concise `base: diffusion` I/O experiment pointing to
that training view, with semantic groups and action_horizon=16. This experiment
only selects the existing BrainCo joint transforms and normalization; the RDP
trainer is selected explicitly by the command below, not by that `base` field.

```bash
python scripts/compute_norm_stats.py --config-path /path/to/io_config.yaml
python -m openpi.rdp.train --stage both --cache /path/to/cache \
  --io-config /path/to/io_config.yaml --norm /path/to/canonical_norm_directory \
  --output /path/to/run --at-steps 20000 --ldp-steps 20000 --swanlab
```

`--stage both` completes AT, loads its best held-out checkpoint, freezes it,
trains LDP, then generates an immutable deployment export. `--resume` restores
model, optimizer, RNG and EMA state; batch ordering after resume is not promised
to be bitwise identical. Validation includes AT L1 with actual and zero tactile
input. A lower reconstruction loss alone does not establish control success or
that tactile feedback is causally useful on the robot.

## Streaming deployment

The existing `python -m openpi.deploy.policy_inferencer --checkpoint ...` entry
point dispatches RDP exports automatically. DESCRIBE remains lightweight.
`rdp_model.json` is the RDP architecture authority; `train_config.yaml` records
canonical joint I/O transforms. Export hashes cover weights, PCA, normalization
and both generated specifications.

The protocol contract deliberately reports `chunk_steps: 1` and
`model_chunk_steps: 16`. Every INFER consumes current feedback and emits one
absolute action; returning all 16 actions in one call would remove the intended
feedback within the chunk. The wrapper retains the latent and GRU state until
the next plan, anchors arm deltas to the chunk-start state, and supports RESET.
A pause longer than 0.5 s resets cached policy state. LDP planning is synchronous
at chunk boundaries in this first version; deployed end-to-end deadline behavior
still needs hardware measurement. Fast GRU calls are separate from slow planning.

Observations must additionally contain:

```python
{
    "timestamp": monotonic_seconds,
    "tactile": {
        "observation.tactile_markers.left_thumb": {
            "markers": array_of_shape_20_20_6,
            "timestamp": marker_receive_time_in_same_clock,
        },
        # index, middle, ring, little are also required
    },
}
```

Missing, nonfinite, future, stale (>0.5 s), or backward-clock input is rejected.
The revo_deploy companion branch adds five PointCloud2 subscriptions and bounded
Arrow-batch dataset playback, resetting the inferencer at episode/loop boundaries.
Use its `revotron_revo3_vitai_left.yaml` profile, whose camera aliases and ViTai
marker topics match the lamp recorder. Receive timestamps measure delivery
freshness, not camera capture latency. No physical robot motion is initiated by
training or these automated interface checks.
