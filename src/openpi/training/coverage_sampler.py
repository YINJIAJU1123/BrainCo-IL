"""Deterministic, quota based two-source batches addressed by optimizer step.

Only the two source permutations are stored as NumPy arrays. Batch membership is
computed from the step, so worker prefetch never advances the resume position.
"""

from __future__ import annotations

import math

import numpy as np


class CoverageBatchSampler:
    def __init__(self, lengths: tuple[int, int], *, coverage_index: int, ego_ratio: int,
                 real_ratio: int, round_to: int, rounds: int, batch_size: int, seed: int, ego_repeat: bool = False):
        if len(lengths) != 2 or coverage_index not in (0, 1):
            raise ValueError("coverage requires exactly two sources and a valid coverage_index")
        if min(*lengths, ego_ratio, real_ratio, round_to, rounds, batch_size) <= 0:
            raise ValueError("coverage lengths, ratios, rounds, round_to and batch_size must be positive")
        self.lengths = lengths
        self.coverage_index = coverage_index
        self.ego_index = 1 - coverage_index
        self.batch_size = batch_size
        self.seed = seed
        self.start_step = 0
        real_required = lengths[coverage_index] * rounds
        self.real_count = round_to * math.ceil(real_required / round_to)
        if self.real_count % real_ratio:
            raise ValueError("rounded coverage count must divide real_ratio")
        self.ego_count = self.real_count * ego_ratio // real_ratio
        if self.ego_count * real_ratio != self.real_count * ego_ratio:
            raise ValueError("rounded coverage count does not give an integer Ego quota")
        total = self.real_count + self.ego_count
        if total % batch_size:
            raise ValueError("source quotas do not fill whole batches")
        if self.ego_count > lengths[self.ego_index] and not ego_repeat:
            raise ValueError("Ego quota exceeds unique Ego training starts")
        self.steps = total // batch_size
        # Independent seeds make the Ego prefix identical for every ratio.
        ego_rng = np.random.default_rng(seed)
        self.ego_order = np.concatenate([
            ego_rng.permutation(lengths[self.ego_index])
            for _ in range(math.ceil(self.ego_count / lengths[self.ego_index]))
        ])[:self.ego_count]
        real_rng = np.random.default_rng(seed + 1)
        real_orders = [real_rng.permutation(lengths[coverage_index]) for _ in range(rounds)]
        real_order = np.concatenate(real_orders)
        padding = self.real_count - len(real_order)
        self.real_order = np.concatenate((real_order, np.resize(real_orders[0], padding)))
        self.padding = padding
        self.offsets = (0, lengths[0])

    def set_start_step(self, step: int) -> None:
        if not 0 <= step <= self.steps:
            raise ValueError(f"resume step {step} outside [0, {self.steps}]")
        self.start_step = step

    def batch_indices(self, step: int) -> list[int]:
        if not 0 <= step < self.steps:
            raise IndexError(step)
        real_start = step * self.real_count // self.steps
        real_end = (step + 1) * self.real_count // self.steps
        ego_start = step * self.batch_size - real_start
        ego_end = (step + 1) * self.batch_size - real_end
        real = self.real_order[real_start:real_end] + self.offsets[self.coverage_index]
        ego = self.ego_order[ego_start:ego_end] + self.offsets[self.ego_index]
        indices = np.concatenate((real, ego))
        np.random.default_rng(np.random.SeedSequence([self.seed, step, 2])).shuffle(indices)
        return indices.tolist()

    def __iter__(self):
        for step in range(self.start_step, self.steps):
            yield self.batch_indices(step)

    def __len__(self) -> int:
        return self.steps - self.start_step
