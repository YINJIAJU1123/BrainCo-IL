"""Episode-safe action/tactile windows using the canonical BrainCo transforms."""

import json
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

from openpi import transforms
from openpi.shared import normalize
from openpi.training import config_io


class RDPDataset(Dataset):
    def __init__(self, cache, io_config, norm_path, *, split="train", images=False):
        self.cache = Path(cache)
        self.manifest = json.loads((self.cache / "manifest.json").read_text())
        self.config = config_io.load_train_config(io_config)
        data_config = self.config.data.create(self.config.assets_dirs, self.config.model)
        stats = normalize.load(Path(norm_path))
        self.transform = transforms.compose(
            (
                *data_config.data_transforms.inputs,
                transforms.Normalize(stats, use_quantiles=data_config.use_quantile_norm),
            )
        )
        self.horizon = self.config.model.action_horizon
        self.state = np.load(self.cache / "state.npy", mmap_mode="r")
        self.action = np.load(self.cache / "action.npy", mmap_mode="r")
        self.tactile = np.load(self.cache / "tactile.npy", mmap_mode="r")
        self.cameras = [np.load(self.cache / f"camera_{i}.npy", mmap_mode="r") for i in range(3)] if images else []
        self.indices = np.concatenate(
            [np.arange(s["start"], s["end"]) for s in self.manifest["segments"] if s["split"] == split]
        )
        self.ends = np.zeros(len(self.state), dtype=np.int64)
        for seg in self.manifest["segments"]:
            self.ends[seg["start"] : seg["end"]] = seg["end"]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        i = self.indices[index]
        window = np.minimum(i + np.arange(self.horizon), self.ends[i] - 1)
        images = [np.asarray(cam[i]) for cam in self.cameras] if self.cameras else [np.zeros((8, 8, 3), np.uint8)] * 3
        data = {
            "observation/state": self.state[i].copy(),
            "actions": self.action[window].copy(),
            "observation/image": images[0],
            "observation/left_wrist_image": images[1],
            "observation/right_wrist_image": images[2],
        }
        out = self.transform(data)
        result = {
            "state": out["state"].astype(np.float32),
            "actions": out["actions"].astype(np.float32),
            "tactile": self.tactile[window].copy(),
        }
        if self.cameras:
            result["images"] = np.stack(images)
        return result
