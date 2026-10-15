import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scripts.ego import compute_shared_norm as norm


class Collector:
    def __init__(self):
        self.batches = []

    def update(self, values):
        self.batches.append(values.copy())


class SharedNormConsumeTest(unittest.TestCase):
    def consume(self, lengths, selected):
        episodes = [(i, Path(str(i)), n) for i, n in enumerate(lengths)]
        offset = 0
        vectors = {}
        for i, _, n in episodes:
            state = np.repeat((10 * np.arange(offset + 1, offset + n + 1))[:, None], 56, axis=1).astype(np.float32)
            vectors[str(i)] = (state, state + 2)
            offset += n
        states, actions = Collector(), Collector()
        with patch.object(norm, '_episode_paths', return_value=episodes), \
             patch.object(norm, '_read_vector_columns', side_effect=lambda p, _: vectors[str(p)]), \
             patch.object(norm, '_sha', return_value='fixture'):
            count, inputs = norm._consume(Path('.'), selected, states, actions, horizon=2)
        return count, inputs, states, actions

    def test_selected_starts_and_episode_end_clamping(self):
        count, inputs, states, actions = self.consume([3, 2], np.array([0, 2, 4]))
        self.assertEqual(count, 3)
        self.assertEqual([x['starts'] for x in inputs], [2, 1])
        np.testing.assert_array_equal(np.concatenate(states.batches)[:, 0], [10, 30, 50])
        chunks = np.concatenate(actions.batches)
        np.testing.assert_array_equal(chunks[:, :, 0], [[2, 12], [2, 2], [2, 2]])
        np.testing.assert_array_equal(chunks[:, :, 14], [[12, 22], [32, 32], [52, 52]])

    def test_buffer_flush_preserves_final_partial_batch(self):
        count, _, states, actions = self.consume([4096, 2], None)
        self.assertEqual(count, 4098)
        self.assertEqual([len(x) for x in states.batches], [4096, 2])
        self.assertEqual([len(x) for x in actions.batches], [4096, 2])
        self.assertEqual(actions.batches[-1][-1, -1, 0], 2)


if __name__ == '__main__':
    unittest.main()
