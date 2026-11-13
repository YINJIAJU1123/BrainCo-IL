"""Finalize and validate a saved BrainCo checkpoint for Policy Inferencer v1.

The trainer calls export_checkpoint only after Orbax finishes saving params
and normalization assets. The CLI regenerates artifacts for an existing step.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from openpi.deploy import contract
from openpi.models.model import ModelType
from openpi.shared import normalize
from openpi.training import config_io


def _validate_assets(config, checkpoint: Path) -> None:
    params = checkpoint / "params"
    if not params.is_dir() or not any(params.iterdir()):
        raise FileNotFoundError(f"saved model parameters missing or empty: {params}")
    asset_id = config.data.assets.asset_id or config.data.repo_id
    stats = normalize.load(checkpoint / "assets" / asset_id)
    for key in ("state", "actions"):
        if key not in stats:
            raise ValueError(f"checkpoint normalization missing {key}")
        fields = ("mean", "std") if config.model.model_type == ModelType.PI0 else ("mean", "std", "q01", "q99")
        for field in fields:
            value = np.asarray(getattr(stats[key], field))
            if value.shape != (config.model.action_dim,) or not np.all(np.isfinite(value)):
                raise ValueError(f"checkpoint normalization {key}.{field} must be finite [{config.model.action_dim}]")


def export_checkpoint(config, checkpoint: str | Path, *, dataset_roots=None) -> Path:
    """Write both generated artifacts, then re-read and validate their binding."""
    checkpoint = Path(checkpoint).expanduser().resolve()
    _validate_assets(config, checkpoint)
    # Resolving metadata again catches changed datasets before marking a step
    # deployable. The generated training indices must describe that metadata.
    payload = contract.build_policy_contract(
        config, contract.checkpoint_id_for(config, checkpoint), dataset_roots=dataset_roots
    )
    config_io.save_train_config(config, checkpoint)
    contract.save_policy_contract(config, checkpoint, dataset_roots=dataset_roots)
    restored = config_io.load_train_config(checkpoint)
    saved_contract = contract.load_policy_contract(checkpoint)
    config_io.validate_checkpoint_id(checkpoint, saved_contract["checkpoint_id"])
    if saved_contract != payload:
        raise RuntimeError("checkpoint contract changed during export")
    _validate_assets(restored, checkpoint)
    return checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="An existing saved step containing params/ and assets/")
    parser.add_argument("--config", help="Source experiment YAML; defaults to the checkpoint's generated config")
    args = parser.parse_args()
    config = config_io.load_train_config(args.config or args.checkpoint)
    print(export_checkpoint(config, args.checkpoint))


if __name__ == "__main__":
    main()
