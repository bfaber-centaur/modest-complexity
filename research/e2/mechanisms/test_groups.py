"""Checks for groups.py that would change its conclusions if wrong.

uv run --locked --group analysis python -m unittest discover -s research/e2/mechanisms -v
"""
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from groups import ROOT, contribution, d_shares, pizza_reference  # noqa: E402
sys.path.insert(0, str(ROOT))
from experiment import forward, init  # noqa: E402


def split():
    raw = np.load(ROOT/'research/data/e1-split.npz')
    return dict(a=raw['a'], b=raw['b'], y=raw['yadd'], yadd=raw['yadd'], train=raw['train'], cal=raw['cal'],
                audit=raw['audit'])


class ShareTests(unittest.TestCase):
    def setUp(self):
        self.d, self.p = split(), 31
        self.all = np.arange(31*31)

    def term(self, f, k):
        a, b, c = self.d['a'][:, None], self.d['b'][:, None], np.arange(self.p)
        t = f(a, b)*np.cos(2*np.pi*k*(a + b - c)/self.p)
        return t - t.mean(1, keepdims=True)

    def test_clock_term_is_fully_explained_and_peaks_at_correct_label(self):
        out = d_shares(self.term(lambda a, b: 1., 5), self.d, self.all, 5)
        self.assertAlmostEqual(out['d_share'], 1.)
        self.assertAlmostEqual(out['k_share'], 1.)
        self.assertTrue(out['peak_d0'])

    def test_pizza_term_matches_derived_share(self):
        # Over all pairs, the |cos(pi k (a-b)/p)| factor leaves E|cos|^2/E cos^2 = 8/pi^2 up to discreteness.
        ref = pizza_reference(self.d, self.all, self.p)
        self.assertAlmostEqual(ref, 8/np.pi**2, delta=.01)
        out = d_shares(self.term(lambda a, b: np.abs(np.cos(np.pi*3*(a - b)/self.p)), 3), self.d, self.all, 3)
        self.assertGreater(out['k_share'], .78)
        self.assertTrue(out['peak_d0'])

    def test_noise_share_is_small(self):
        x = np.random.default_rng(0).standard_normal((961, 31))
        self.assertLess(d_shares(x - x.mean(1, keepdims=True), self.d, self.all, 2)['d_share'], .01)

    def test_all_unit_contributions_are_the_centred_logits(self):
        w = init(31, 24, 5)
        z = forward(w, self.d['a'], self.d['b'])[0]
        np.testing.assert_allclose(contribution(w, self.d, np.arange(24)), z - z.mean(1, keepdims=True), atol=1e-12)
        halves = contribution(w, self.d, np.arange(12)) + contribution(w, self.d, np.arange(12, 24))
        np.testing.assert_allclose(halves, z - z.mean(1, keepdims=True), atol=1e-12)


if __name__ == '__main__':
    unittest.main()
