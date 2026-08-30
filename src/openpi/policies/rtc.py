"""Inference-time Real-Time Chunking helpers.

This module intentionally contains only runtime configuration and mask
construction.  The ordinary policy path does not import or call these helpers
unless RTC is explicitly selected by the deployment frontend.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np


@dataclasses.dataclass(frozen=True)
class RTCConfig:
    """Configuration for guided inference-time RTC."""

    prefix_horizon: int = 8
    max_guidance_weight: float = 5.0
    schedule: str = "exp"

    def validate(self, action_horizon: int) -> None:
        if action_horizon <= 1:
            raise ValueError("RTC requires action_horizon > 1")
        if not 1 <= self.prefix_horizon <= action_horizon:
            raise ValueError(f"RTC prefix_horizon must be in [1, {action_horizon}], got {self.prefix_horizon}")
        if not math.isfinite(self.max_guidance_weight) or self.max_guidance_weight <= 0:
            raise ValueError("RTC max_guidance_weight must be finite and > 0")
        if self.schedule.lower() != "exp":
            raise ValueError(f"unsupported RTC schedule {self.schedule!r}; this deployment currently supports 'exp'")


def build_exp_prefix_weights(
    *,
    action_horizon: int,
    inference_delay: int,
    prefix_horizon: int,
    available_prefix_steps: int,
) -> np.ndarray:
    """Build the RTC exponential soft mask as ``float32[H, 1]``.

    The first ``inference_delay`` available actions receive full weight.  The
    rest decay exponentially until ``prefix_horizon``; actions outside the
    available previous-chunk prefix receive zero weight.
    """

    action_horizon = int(action_horizon)
    if action_horizon <= 0:
        raise ValueError(f"action_horizon must be > 0, got {action_horizon}")
    available = min(max(0, int(available_prefix_steps)), action_horizon)
    end = min(max(0, int(prefix_horizon)), available)
    start = min(max(0, int(inference_delay)), end)

    weights = np.zeros((action_horizon,), dtype=np.float32)
    weights[:start] = 1.0
    decay_steps = end - start
    if decay_steps > 0:
        linear = np.linspace(1.0, 0.0, decay_steps + 2, dtype=np.float32)[1:-1]
        weights[start:end] = linear * np.expm1(linear) / np.float32(math.e - 1.0)
    return weights[:, None]
