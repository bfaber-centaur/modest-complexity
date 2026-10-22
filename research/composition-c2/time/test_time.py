"""Checks of the time lane's threshold rule and error-compatibility counts on hand-made inputs,
plus consistency of its neighbour label sets with c2task. Code checks only, no scientific claim.

uv run --locked python -m unittest -v research/composition-c2/time/test_time.py
"""
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import time_analyze as ta  # noqa: E402


class Threshold(unittest.TestCase):
    def steps(self, n):
        return list(range(0, 50*n, 50))

    def test_first_of_five_consecutive(self):
        acc = [.90, .95, .96, .97, .95, .99, .50]
        self.assertEqual(ta.first_sustained(self.steps(7), acc), 50)

    def test_a_dip_restarts_the_window(self):
        acc = [.96]*4 + [.90] + [.96]*5
        self.assertEqual(ta.first_sustained(self.steps(10), acc), 250)

    def test_censored_when_the_window_cannot_complete(self):
        self.assertIsNone(ta.first_sustained(self.steps(6), [.9, .9, .96, .96, .96, .96]))

    def test_logging_gap_breaks_the_window(self):
        self.assertIsNone(ta.first_sustained([0, 50, 100, 200, 250], [.96]*5))

    def test_counts_at_the_boundary(self):
        self.assertEqual(ta.first_sustained(self.steps(5), [122/128]*5), 0)
        self.assertIsNone(ta.first_sustained(self.steps(5), [121/128]*5))
        self.assertEqual(ta.first_sustained(self.steps(5), [40/42]*5), 0)
        self.assertIsNone(ta.first_sustained(self.steps(5), [39/42]*5))


class Compatibility(unittest.TestCase):
    def test_hand_made_rows(self):
        y = np.array([0, 1, 2])
        every = [{0, 4}, {1}, {3, 5}]; prefix = [{4}, set(), {5}]
        c = ta.compatibility('013', every, prefix, y)  # only row 2 is wrong, predicting 3
        self.assertEqual(c['n_wrong'], 1)
        self.assertEqual((c['any_nbr'], c['same_prefix']), (1.0, 0.0))
        self.assertAlmostEqual(c['any_nbr_uniform'], 2/7)
        self.assertAlmostEqual(c['same_prefix_uniform'], 1/7)
        self.assertIsNone(ta.compatibility('012', every, prefix, y)['any_nbr'])

    def test_label_sets_agree_with_c2task(self):
        self.assertEqual(len(ta.EVERY_NBR), 128)
        has_true = np.array([int(y) in s for y, s in zip(ta.Y, ta.EVERY_NBR)])
        self.assertTrue((has_true == ta.NBR).all())
        self.assertEqual(int((~has_true).sum()), 42)
        for e, p in zip(ta.EVERY_NBR, ta.PREFIX_NBR):
            self.assertTrue(p <= e and 1 <= len(p) <= 3)

    def test_subsets_of_the_true_answers(self):
        truth = ''.join(map(str, ta.Y))
        self.assertEqual(ta.subsets(truth), dict(N=1.0, notN=1.0, n_N=86, n_notN=42))


if __name__ == '__main__':
    unittest.main()
