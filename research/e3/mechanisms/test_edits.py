"""Checks for edits.py that would change its conclusions if wrong.

uv run --locked --group analysis python -m unittest discover -s research/e3/mechanisms -v
"""
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from edits import ROOT, dominant, filter_units, greedy, input_energy, keep_frequencies, sham, sham_orders  # noqa: E402
sys.path.insert(0, str(ROOT))
from experiment import init  # noqa: E402


class EditTests(unittest.TestCase):
    def setUp(self):
        self.w = init(31, 40, 7)

    def test_projection_reproduces_the_e2_filter(self):
        k = dominant(self.w)[0]
        keep = np.zeros((16, 40), bool)
        keep[0] = True
        keep[k, np.arange(40)] = True
        for x, y in zip(keep_frequencies(self.w, keep), filter_units(self.w, k)):
            np.testing.assert_allclose(x, y, atol=1e-12)

    def test_orders_are_seeded_and_never_dominant(self):
        k = dominant(self.w)[0]
        a, b = sham_orders(k, 37000), sham_orders(k, 37000)
        self.assertEqual(a, b)
        self.assertNotEqual(a, sham_orders(k, 37001))
        for j, order in enumerate(a):
            self.assertEqual(sorted(order), sorted(set(range(1, 16)) - {k[j]}))

    def test_greedy_stopping_rule(self):
        e = np.zeros((15, 1))
        e[0, 0], e[1:4, 0] = 10., [4., 4., 3.]  # dominant 1; others 2, 3, 4
        orders = [[2, 3, 4]]
        self.assertEqual(greedy(e, np.array([1]), orders, 1.)[0], [[2, 3, 4]])  # 11 is closer to 10 than 8
        self.assertEqual(greedy(e, np.array([1]), orders, .9)[0], [[2, 3]])     # 8 is closer to 9 than 11
        self.assertEqual(greedy(e, np.array([1]), orders, .5)[0], [[2]])        # 4 is closer to 5 than 8

    def test_sham_drops_dominant_frequency_and_matches_total_input_energy(self):
        k = dominant(self.w)[0]
        c, info = sham(self.w, 37000)
        e, ec = input_energy(self.w), input_energy(c)
        self.assertTrue(np.all(ec[k - 1, np.arange(40)] < 1e-20))
        self.assertAlmostEqual(ec.sum()/e[k - 1, np.arange(40)].sum(), 1, delta=.02)
        self.assertAlmostEqual(info['kept_over_goal'], ec.sum()/e[k - 1, np.arange(40)].sum(), places=9)
        np.testing.assert_allclose(c[0].mean(0), self.w[0].mean(0), atol=1e-12)  # DC kept
        w_out = np.abs(np.fft.rfft(c[2], axis=1))**2
        self.assertTrue(np.all(w_out[np.arange(40), k] < 1e-20))


if __name__ == '__main__':
    unittest.main()
