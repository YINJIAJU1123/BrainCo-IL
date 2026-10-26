import unittest
from collections import Counter

from openpi.training.coverage_sampler import CoverageBatchSampler


class CoverageBatchSamplerTest(unittest.TestCase):
    def _plan(self, ratio, real_ratio=1):
        return CoverageBatchSampler((1000, 65), coverage_index=1, ego_ratio=ratio,
                                    real_ratio=real_ratio, round_to=32, rounds=1,
                                    batch_size=16, seed=42)

    def test_ratios_coverage_prefix_and_resume(self):
        plans = [self._plan(1), self._plan(3), self._plan(5, 2)]
        for plan, ego_ratio, real_ratio in zip(plans, (1, 3, 5), (1, 1, 2), strict=True):
            batches = list(plan)
            self.assertEqual(len(batches), plan.steps)
            self.assertTrue(all(len(batch) == 16 for batch in batches))
            real = [i - 1000 for batch in batches for i in batch if i >= 1000]
            ego = [i for batch in batches for i in batch if i < 1000]
            self.assertEqual(len(real), 96)
            self.assertEqual(len(ego) * real_ratio, len(real) * ego_ratio)
            self.assertEqual(set(real), set(range(65)))
            self.assertEqual(len(set(ego)), len(ego))
            resumed = self._plan(ego_ratio, real_ratio)
            resumed.set_start_step(3)
            self.assertEqual(list(resumed), batches[3:])
            self.assertEqual(list(self._plan(ego_ratio, real_ratio)), batches)
        self.assertEqual(plans[0].ego_order.tolist(), plans[1].ego_order[:plans[0].ego_count].tolist())
        self.assertEqual(plans[2].ego_order.tolist(), plans[1].ego_order[:plans[2].ego_count].tolist())
        self.assertEqual(plans[0].real_order.tolist(), plans[1].real_order.tolist())
        self.assertEqual(Counter(sum(index >= 1000 for index in batch)
                                 for batch in list(plans[2])[:7]), {4: 3, 5: 4})

    def test_explicit_repeat_preserves_passes_prefix_and_resume(self):
        import numpy as np
        plans = [CoverageBatchSampler((70, 65), coverage_index=1, ego_ratio=e,
                 real_ratio=r, round_to=32, rounds=1, batch_size=16, seed=42,
                 ego_repeat=True) for e, r in [(1, 1), (3, 1), (5, 2)]]
        for plan in plans:
            for start in range(0, plan.ego_count, 70):
                part = plan.ego_order[start:start + 70]
                self.assertEqual(len(set(part)), len(part))
            original = list(plan)
            plan.set_start_step(3)
            self.assertEqual(list(plan), original[3:])
            self.assertEqual(set(plan.real_order), set(range(65)))
        np.testing.assert_array_equal(plans[0].ego_order, plans[1].ego_order[:96])
        np.testing.assert_array_equal(plans[2].ego_order, plans[1].ego_order[:240])
        for p in plans[1:]:
            np.testing.assert_array_equal(p.real_order, plans[0].real_order)

    def test_invalid_quota_rejected(self):
        with self.assertRaisesRegex(ValueError, "Ego quota"):
            CoverageBatchSampler((10, 65), coverage_index=1, ego_ratio=3,
                                 real_ratio=1, round_to=32, rounds=1, batch_size=16, seed=42)


if __name__ == "__main__":
    unittest.main()
