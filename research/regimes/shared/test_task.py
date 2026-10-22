"""Independent checks of R2's oracle and splits: labels recomputed from tuples
built with itertools and Python booleans, not from task.py's bit arithmetic.

uv run --locked python -m unittest -v research/regimes/shared/test_task.py
"""
from itertools import product
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import task  # noqa: E402


def rows():
    # product varies the last position fastest, so reverse to make x1 the least significant bit.
    return [tuple(reversed(t)) for t in product((0, 1), repeat=10)]


class Oracle(unittest.TestCase):
    def test_domain_order_and_labels(self):
        R = rows()
        self.assertEqual(task.X.shape, (1024, 10))
        self.assertTrue(np.array_equal(task.B, np.array(R)))
        for i, x in enumerate(R):
            x1, x2, x3, x4, x5, x6, x7 = (bool(v) for v in x[:7])
            z = x1 != x2
            want = dict(y1=z != x3, y2=z and x4, part=(x2 != x3) != x4, unrel=(x5 != x6) != x7,
                        split=2*((x1 != x3) != x5) + ((x2 != x6) != x7))
            for t, v in want.items():
                self.assertEqual(task.LABELS[t][i], int(v), (i, t))
            self.assertEqual(task.PROBES['x1^x2'][i], int(z))
        self.assertEqual(sum(i == sum(b << k for k, b in enumerate(x)) for i, x in enumerate(R)), 1024)

    def test_label_balance(self):
        self.assertEqual(task.LABELS['y2'].sum(), 256)  # imbalanced: 1 on a quarter
        for t in ('y1', 'part', 'unrel'):
            self.assertEqual(task.LABELS[t].sum(), 512)
        self.assertEqual(np.bincount(task.LABELS['split']).tolist(), [256]*4)

    def test_relevant_bits(self):
        # A label depends on a bit iff flipping that bit changes it for some input.
        idx = np.arange(1024)
        for t, y in task.LABELS.items():
            dep = tuple(k+1 for k in range(10) if np.any(y[idx ^ (1 << k)] != y))
            self.assertEqual(dep, task.RELEVANT[t], t)

    def test_y2_is_z_on_the_x4_half(self):
        on = task.B[:, 3] == 1
        self.assertTrue(np.array_equal(task.LABELS['y2'][on], task.Z[on]))
        self.assertFalse(task.LABELS['y2'][~on].any())


class Splits(unittest.TestCase):
    def test_disjoint_complete_nested(self):
        for seed in (52100, 52101, 52102):
            pool, test = task.split(seed)
            self.assertEqual(len(set(pool.tolist())), 512)
            self.assertEqual(len(set(test.tolist())), 512)
            self.assertFalse(set(pool.tolist()) & set(test.tolist()))
            self.assertEqual(set(pool.tolist()) | set(test.tolist()), set(range(1024)))
            # Inputs are distinct rows, so disjoint indices mean no duplicate input crosses the split.
            self.assertEqual(len({tuple(r) for r in task.B[test]} & {tuple(r) for r in task.B[pool]}), 0)
            again, _ = task.split(seed)
            self.assertTrue(np.array_equal(pool, again))

    def test_coverage_counts(self):
        pool, _ = task.split(52100)
        self.assertEqual(task.coverage(pool, 'y2'), 8)
        self.assertEqual(task.coverage(pool[:1], 'y2'), 1)


if __name__ == '__main__':
    unittest.main()
