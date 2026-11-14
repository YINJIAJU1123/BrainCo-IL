# ruff: noqa: PLC0415
"""Durable AT -> LDP experiment. Resume and export never refit tactile PCA."""

import argparse
import copy
from dataclasses import asdict
import json
import os
from pathlib import Path
import random
import subprocess
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Subset

from openpi.rdp.data import RDPDataset
from openpi.rdp.model import AsymmetricTokenizer
from openpi.rdp.model import LatentDiffusionPolicy
from openpi.rdp.model import RDPConfig


def atomic_save(value, path):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temp)
    os.replace(temp, path)


def to_device(batch, device):
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def evaluate(model, at, loader, device, stage, *, max_batches=20):
    model.eval()
    values = []
    with torch.inference_mode():
        for index, batch in enumerate(loader):
            if index >= max_batches:
                break
            b = to_device(batch, device)
            if stage == "at":
                z, _ = model.encode(b["actions"])
                actions, _ = model.decode(z, b["tactile"])
                zero, _ = model.decode(z, torch.zeros_like(b["tactile"]))
                values.append([float((actions - b["actions"]).abs().mean()), float((zero - b["actions"]).abs().mean())])
            else:
                z, _ = at.encode(b["actions"])
                loss = model.loss(z, b["images"], b["state"], b["tactile"][:, 0])
                values.append([float(loss)])
    result = np.mean(values, axis=0)
    return {"val_loss": float(result[0]), **({"val_zero_tactile_l1": float(result[1])} if stage == "at" else {})}


def train_stage(args, stage, *, at_path=None):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction, 0 if device.index is None else device.index)
    dataset = RDPDataset(args.cache, args.io_config, args.norm, images=stage == "ldp")
    val = RDPDataset(args.cache, args.io_config, args.norm, split="val", images=stage == "ldp")
    # Evenly cover the entire held-out episode, not just its first static frames.
    val = Subset(val, np.linspace(0, len(val) - 1, min(1280, len(val)), dtype=int).tolist())
    cfg = RDPConfig(
        action_dim=dataset.config.model.action_dim,
        horizon=dataset.horizon,
        tactile_dim=dataset.manifest["tactile_dim"],
        image_size=dataset.manifest["image_size"],
    )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    val_loader = DataLoader(val, batch_size=args.batch_size, num_workers=0)
    if len(loader) == 0:
        raise ValueError("training split smaller than batch size")
    at = AsymmetricTokenizer(cfg).to(device)
    if stage == "ldp":
        saved = torch.load(at_path, map_location="cpu", weights_only=False)
        if saved["config"] != asdict(cfg):
            raise ValueError("AT and LDP configurations differ")
        at.load_state_dict(saved["model"])
        at.eval().requires_grad_(requires_grad=False)
        model = LatentDiffusionPolicy(cfg).to(device)
        # A fixed latent normalizer, fitted only on training action windows.
        chunks = []
        with torch.inference_mode():
            for index, batch in enumerate(loader):
                z, _ = at.encode(batch["actions"].to(device))
                chunks.append(z.flatten(0, 1))
                if index >= 63:
                    break
            z = torch.cat(chunks)
            model.latent_mean.copy_(z.mean(0))
            model.latent_std.copy_(z.std(0).clamp_min(1e-3))
    else:
        model = at
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-6)
    ema = copy.deepcopy(model).eval().requires_grad_(requires_grad=False) if stage == "ldp" else None
    latest = output / f"{stage}_latest.pt"
    step, best = 0, float("inf")
    if latest.exists():
        if not args.resume:
            raise FileExistsError(f"{latest} exists; use --resume")
        saved = torch.load(latest, map_location="cpu", weights_only=False)
        if saved["config"] != asdict(cfg):
            raise ValueError("resume model config mismatch")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        if ema is not None:
            ema.load_state_dict(saved["ema"])
        step, best = saved["step"], saved["best"]
        torch.set_rng_state(saved["rng"])
        if device.type == "cuda" and saved.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
    tracker = None
    if args.swanlab:
        from packaging.version import Version
        import swanlab

        if Version(swanlab.__version__) < Version("0.9.0"):
            raise RuntimeError("SwanLab >=0.9.0 required")
        key_file = os.environ.get("SWANLAB_API_KEY_FILE")
        if key_file:
            swanlab.login(api_key=Path(key_file).read_text().strip(), save=False)
        tracker = swanlab
        tracker.init(
            project="imitation",
            experiment_name=f"{output.name}_{stage}",
            config={**asdict(cfg), "stage": stage, "cache": args.cache},
            mode=os.environ.get("SWANLAB_MODE", "offline"),
        )
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    record = {
        "algorithm": "rdp",
        "stage": stage,
        "config": asdict(cfg),
        "commit": commit,
        "cache": str(Path(args.cache).resolve()),
        "io_config": str(Path(args.io_config).resolve()),
        "at_checkpoint": str(at_path) if at_path else None,
        "seed": args.seed,
    }
    (output / f"{stage}_run.json").write_text(json.dumps(record, indent=2))
    total = args.at_steps if stage == "at" else args.ldp_steps
    started = time.monotonic()
    iterator = iter(loader)
    while step < total:
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        b = to_device(batch, device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        if stage == "at":
            loss, details = model.loss(b["actions"], b["tactile"])
        else:
            with torch.no_grad():
                z, _ = at.encode(b["actions"])
            loss = model.loss(z, b["images"], b["state"], b["tactile"][:, 0])
            details = {}
        if not torch.isfinite(loss):
            raise FloatingPointError(f"{stage} nonfinite loss at step {step}")
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        for group in optimizer.param_groups:
            group["lr"] = args.lr * min((step + 1) / 500, 1.0)
        optimizer.step()
        if ema is not None:
            with torch.no_grad():
                for target, value in zip(ema.parameters(), model.parameters(), strict=True):
                    target.lerp_(value, 1 - 0.999)
        step += 1
        if step == 1 or step % 100 == 0:
            metrics = {
                "step": step,
                "loss": float(loss),
                "grad_norm": float(grad),
                **{key: float(value) for key, value in details.items()},
            }
            print(
                json.dumps({"stage": stage, **metrics, "elapsed_s": round(time.monotonic() - started, 2)}), flush=True
            )
            if tracker:
                tracker.log(metrics, step=step)
        if step % args.save_interval == 0 or step == total:
            candidate = ema if ema is not None else model
            metrics = evaluate(candidate, at, val_loader, device, stage)
            print(json.dumps({"stage": stage, "step": step, **metrics}), flush=True)
            improved = metrics["val_loss"] < best
            best = min(best, metrics["val_loss"])
            state = {
                "model": model.state_dict(),
                "config": asdict(cfg),
                "step": step,
                "best": best,
                "optimizer": optimizer.state_dict(),
                "rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
                "ema": ema.state_dict() if ema is not None else None,
                "record": record,
            }
            atomic_save(state, latest)
            if improved:
                atomic_save(
                    {
                        "model": candidate.state_dict(),
                        "config": asdict(cfg),
                        "step": step,
                        "metrics": metrics,
                        "record": record,
                    },
                    output / f"{stage}_best.pt",
                )
            if tracker:
                tracker.log(metrics, step=step)
    if tracker:
        tracker.finish()
    (output / f"{stage}_complete.json").write_text(json.dumps({"step": step, "best": best, "commit": commit}))
    print(f"STAGE_COMPLETE {stage} step={step} best={best}", flush=True)
    return output / f"{stage}_best.pt"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--io-config", required=True)
    p.add_argument("--norm", required=True, help="canonical norm_stats directory, fitted on train_dataset")
    p.add_argument("--output", required=True)
    p.add_argument("--stage", choices=("at", "ldp", "both"), default="both")
    p.add_argument("--at-checkpoint")
    p.add_argument("--at-steps", type=int, default=20000)
    p.add_argument("--ldp-steps", type=int, default=20000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--save-interval", type=int, default=1000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--memory-fraction", type=float, default=0.4)
    p.add_argument("--swanlab", action="store_true")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    at_path = args.at_checkpoint
    if args.stage in ("at", "both"):
        at_path = train_stage(args, "at")
    if args.stage in ("ldp", "both"):
        if not at_path:
            p.error("--at-checkpoint is required for --stage ldp")
        train_stage(args, "ldp", at_path=at_path)
        from openpi.rdp.export import export

        export(args.output, args.cache, args.io_config, args.norm, at_path)


if __name__ == "__main__":
    main()
