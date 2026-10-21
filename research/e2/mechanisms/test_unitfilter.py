"""Checks for unitfilter.py that would change its conclusions if wrong.

uv run --locked --group analysis python -m unittest discover -s research/e2/mechanisms -v
"""
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from unitfilter import ROOT, dominant, filter_units, logits, other_frequency, rotate_phases, scale_units  # noqa: E402
sys.path.insert(0, str(ROOT))
from experiment import init  # noqa: E402

P = 31
PAIRS = dict(a=np.repeat(np.arange(P), P), b=np.tile(np.arange(P), P))


def addition_network(n=6):
    """Units cos(k a + al), cos(k b + be) -> cos(k c + al + be) on a full phase grid, all k."""
    t = 2*np.pi*np.arange(P)[:, None]/P
    k, al, be = (x.ravel() for x in np.meshgrid(np.arange(1, 16), 2*np.pi*np.arange(n)/n, 2*np.pi*np.arange(n)/n,
                                                 indexing='ij'))
    return [np.cos(k*t + al), np.cos(k*t + be), np.cos(k*t + al + be).T], k


class FilterTests(unittest.TestCase):
    def test_filter_recovers_a_hidden_addition_network(self):
        # Positive control: other-frequency parts added to every unit hide the computation;
        # the per-unit filter, which never sees a label, removes them exactly.
        w, k = addition_network()
        y = (PAIRS['a'] + PAIRS['b']) % P
        self.assertEqual(np.mean(logits(w, PAIRS).argmax(1) == y), 1.)
        rng = np.random.default_rng(0)
        t = 2*np.pi*np.arange(P)[:, None]/P
        noise = lambda: sum(.6*np.cos(((k + o) % 15 + 1)*t + rng.uniform(0, 2*np.pi, len(k))) for o in (2, 6, 11))
        dirty = [w[0] + noise(), w[1] + noise(), w[2] + noise().T]
        self.assertLess(np.mean(logits(dirty, PAIRS).argmax(1) == y), .5)
        np.testing.assert_array_equal(dominant(dirty)[0], k)
        clean = filter_units(dirty, dominant(dirty)[0])
        for x, x0 in zip(clean, w):
            np.testing.assert_allclose(x, x0, atol=1e-10)
        self.assertEqual(np.mean(logits(clean, PAIRS).argmax(1) == y), 1.)

    def test_filter_is_a_projection(self):
        w = init(P, 20, 3)
        k = dominant(w)[0]
        once = filter_units(w, k)
        for x, x1, x2 in zip(w, once, filter_units(once, k)):
            np.testing.assert_allclose(x1, x2, atol=1e-12)
            self.assertLess(np.sum(x1**2), np.sum(x**2))
        np.testing.assert_array_equal(dominant(once)[0], k)

    def test_phase_scramble_keeps_frequencies_and_energies(self):
        w, k = addition_network()
        rng = np.random.default_rng(2)
        r = rotate_phases(w, k, rng)
        for x, y, axis in zip(w, r, (0, 0, 1)):
            np.testing.assert_allclose(np.abs(np.fft.rfft(x, axis=axis)), np.abs(np.fft.rfft(y, axis=axis)), atol=1e-9)
        y = (PAIRS['a'] + PAIRS['b']) % P
        self.assertLess(np.mean(logits(r, PAIRS).argmax(1) == y), .2)

    def test_random_frequency_is_never_the_dominant_one(self):
        rng = np.random.default_rng(1)
        k = np.repeat(np.arange(1, 16), 400)
        other = other_frequency(k, rng)
        self.assertFalse(np.any(other == k))
        for f in range(1, 16):
            self.assertEqual(set(other[k == f]), set(range(1, 16)) - {f})

    def test_scaling(self):
        w = init(P, 12, 4)
        z = logits(w, PAIRS)
        np.testing.assert_allclose(logits(scale_units(w, np.arange(12), 3.), PAIRS), 3*z)
        only = logits(scale_units(w, np.arange(5), np.inf), PAIRS)
        np.testing.assert_allclose(only, np.maximum(w[0][PAIRS['a']][:, :5] + w[1][PAIRS['b']][:, :5], 0) @ w[2][:5])


if __name__ == '__main__':
    unittest.main()
