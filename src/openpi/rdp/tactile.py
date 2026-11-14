"""Fixed per-finger PCA and causal sample-and-hold. No action supervision."""

from pathlib import Path

import numpy as np

FINGERS = ("thumb", "index", "middle", "ring", "little")
MARKER_KEYS = tuple(f"observation.tactile_markers.left_{finger}" for finger in FINGERS)
CHANNELS = ("origin_x", "origin_y", "current_x", "current_y", "offset_x", "offset_y")


def offsets(markers):
    markers = np.asarray(markers, dtype=np.float32)
    if markers.shape[-3:] != (20, 20, 6) or not np.isfinite(markers).all():
        raise ValueError("expected finite [...,20,20,6] markers")
    return markers[..., 4:6].reshape(*markers.shape[:-3], 800)


def held_indices(timestamps, rate_hz):
    """Poll recorded markers at <= rate_hz; never use a future observation.

    The age is adapter hold age, NOT measured sensor capture latency. The
    recorder has no independent marker capture timestamp in this dataset.
    """
    timestamps = np.asarray(timestamps, dtype=np.float64)
    if rate_hz <= 0 or np.any(np.diff(timestamps) <= 0):
        raise ValueError("positive rate and strictly increasing timestamps required")
    ticks = np.floor((timestamps - timestamps[0]) * rate_hz + 1e-5).astype(np.int64)
    changes = np.r_[True, np.diff(ticks) > 0]
    indices = np.maximum.accumulate(np.where(changes, np.arange(len(ticks)), 0))
    return indices, (timestamps - timestamps[indices]).astype(np.float32)


class TactilePCA:
    def __init__(self, mean, basis, scale):
        self.mean = np.asarray(mean, np.float32)
        self.basis = np.asarray(basis, np.float32)
        self.scale = np.asarray(scale, np.float32)

    @classmethod
    def fit(cls, data, components=15):
        x = np.asarray(data, np.float64)
        if x.ndim != 2 or x.shape[1] != 800 or not np.isfinite(x).all():
            raise ValueError("PCA expects finite [frames,800] offsets")
        if not 1 <= components <= min(x.shape):
            raise ValueError("invalid PCA dimension")
        mean = x.mean(0)
        centered = x - mean
        cov = centered.T @ centered / max(len(x) - 1, 1)
        values, vectors = np.linalg.eigh(cov)
        order = np.argsort(values)[::-1][:components]
        basis = vectors[:, order]
        # Fix arbitrary eigenvector signs for reproducible assets.
        sign = np.sign(basis[np.argmax(np.abs(basis), axis=0), np.arange(components)])
        basis *= np.where(sign == 0, 1, sign)
        scale = np.maximum(np.sqrt(np.maximum(values[order], 0)), 1e-3)
        return cls(mean, basis, scale), float(values[order].sum() / max(values.sum(), 1e-12))

    def encode(self, data):
        return ((np.asarray(data, np.float32) - self.mean) @ self.basis / self.scale).astype(np.float32)

    def reconstruct(self, data):
        return (self.encode(data) * self.scale) @ self.basis.T + self.mean

    def save(self, path):
        np.savez(path, mean=self.mean, basis=self.basis, scale=self.scale)

    @classmethod
    def load(cls, path):
        with np.load(Path(path), allow_pickle=False) as data:
            return cls(data["mean"], data["basis"], data["scale"])
