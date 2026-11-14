"""Generate an immutable streaming Policy Inferencer v1 checkpoint."""

import json
from pathlib import Path
import shutil

import torch

from openpi.deploy.contract import _payload_hash
from openpi.deploy.contract import build_policy_contract
from openpi.deploy.contract import checkpoint_id_for
from openpi.rdp.prepare import sha256
from openpi.rdp.tactile import MARKER_KEYS
from openpi.training import config_io


def export(run, cache, io_config, norm, at_path):
    run, cache = Path(run), Path(cache)
    final = run / "export"
    if final.exists():
        raise FileExistsError(f"immutable export exists: {final}")
    temp = run / ".export.incomplete"
    temp.mkdir(exist_ok=True)
    (temp / "params").mkdir(exist_ok=True)
    (temp / "assets").mkdir(exist_ok=True)
    cfg = config_io.load_train_config(io_config)
    manifest = json.loads((cache / "manifest.json").read_text())
    saved = torch.load(run / "ldp_best.pt", map_location="cpu", weights_only=False)
    shutil.copy2(at_path, temp / "params/at.pt")
    shutil.copy2(run / "ldp_best.pt", temp / "params/ldp.pt")
    shutil.copytree(cache / "pca", temp / "assets/pca", dirs_exist_ok=True)
    shutil.copy2(Path(norm) / "norm_stats.json", temp / "assets/norm_stats.json")
    # train_config is the canonical joint selection/normalization specification.
    # RDP's actual architecture is explicitly recorded in rdp_model.json.
    config_io.save_train_config(cfg, temp)
    contract = build_policy_contract(cfg, checkpoint_id_for(cfg, temp))
    contract.update(
        algorithm="rdp",
        model_chunk_steps=saved["config"]["horizon"],
        chunk_steps=1,
        required_tactile={key: [20, 20, 6] for key in MARKER_KEYS},
        tactile_rate_hz=manifest["tactile_hz"],
        streaming=True,
    )
    contract["contract_hash"] = _payload_hash(contract)
    (temp / "policy_contract.json").write_text(json.dumps(contract, indent=2))
    description = {
        "algorithm": "rdp",
        "config": saved["config"],
        "training_commit": saved["record"]["commit"],
        "tactile_hz": manifest["tactile_hz"],
        "max_tactile_age_s": 0.5,
        "io_config_only": "train_config.yaml describes canonical joint transforms; rdp_model.json describes the model",
        "pca": manifest["pca"],
        "streaming": True,
    }
    (temp / "rdp_model.json").write_text(json.dumps(description, indent=2))
    hashes = {
        str(p.relative_to(temp)): sha256(p)
        for p in sorted(temp.rglob("*"))
        if p.is_file() and p.name != "_CHECKPOINT_METADATA"
    }
    (temp / "_CHECKPOINT_METADATA").write_text(json.dumps({"complete": True, "sha256": hashes}, indent=2))
    temp.rename(final)
    print(f"EXPORTED {final}", flush=True)
    return final
