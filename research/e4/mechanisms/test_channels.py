"""Checks of channels.py on synthetic networks and on the filtered start files (no run needed).

uv run --locked --group analysis python -m unittest discover -s research/e4/mechanisms -v
"""
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import channels as c  # noqa: E402

P = c.P
A, B = np.divmod(np.arange(P*P), P)
RNG = np.random.default_rng(0)
COMMON = set(c.e3.COMMON.tolist())
D = dict(a=A, b=B, train=np.array(sorted(RNG.choice([i for i in range(P*P) if i not in COMMON], 299, replace=False))))


def ideal(shift=0.):
    """Units on frequencies 1..4, 8 phases each; readout phase phi + phi' + shift."""
    t = 2*np.pi*np.arange(P)/P
    cols, rows, k = [], [], []
    for f in range(1, 5):
        for phi in np.linspace(0, 2*np.pi, 8, endpoint=False):
            phi2 = 2*phi + 1.
            cols.append((np.cos(f*t + phi), np.cos(f*t + phi2)))
            rows.append(np.cos(f*t + phi + phi2 + shift))
            k.append(f)
    U, V = (np.stack([x[i] for x in cols], 1) for i in range(2))
    return [U, V, np.stack(rows)], np.array(k)


def random_net(h=16):
    return [RNG.normal(size=(P, h)), RNG.normal(size=(P, h)), RNG.normal(size=(h, P))]


class Channels(unittest.TestCase):
    def test_ideal_network_is_aligned_and_correct(self):
        w, k = ideal()
        live = np.ones(len(k), bool)
        al, err = c.alignment(w, k, live)
        self.assertLess(err, 1e-12)
        self.assertGreater(al['alignment_R'], .99)
        self.assertGreater(al['coherence'], .99)
        z = c.hidden(w, A, B) @ w[2]
        self.assertEqual(np.mean(z.argmax(1) == (A + B) % P), 1.)
        ch, _, id_err = c.channels(w, D, k, live)
        self.assertLess(id_err, 1e-12)
        self.assertLess(np.abs(ch['off']).max(), 1e-10)  # the readout is wholly at the reference

    def test_readout_phase_sign_and_cancellation(self):
        w, k = ideal(np.pi)
        al, _ = c.alignment(w, k, np.ones(len(k), bool))
        self.assertLess(al['alignment_R'], -.99)  # anti-aligned: the mode disfavours c = a + b
        self.assertGreater(al['coherence'], .99)
        w, k = ideal()
        w = [np.hstack([w[0]]*2), np.hstack([w[1]]*2), np.vstack([w[2], -w[2]])]
        k = np.tile(k, 2)  # every unit paired with a copy whose readout is negated: the groups cancel
        al, _ = c.alignment(w, k, np.ones(len(k), bool))
        self.assertLess(abs(al['alignment_R']), 1e-9)
        self.assertLess(al['coherence'], 1e-9)

    def test_identities_on_a_random_network(self):
        w = random_net()
        k = RNG.integers(1, P//2 + 1, 16)
        live = RNG.random(16) < .8
        ch, z, err = c.channels(w, D, k, live)
        self.assertLess(err, 1e-12)
        _, err = c.alignment(w, k, live)
        self.assertLess(err, 1e-12)
        parts, err = c.weight_parts(w, k, live)
        self.assertLess(err, 1e-12)
        self.assertAlmostEqual(parts['W_ref_norm']**2 + parts['W_off_norm']**2 + parts['W_dc_norm']**2, np.sum(w[2]**2))
        W_ref, W_off = c.split_readout(w[2], k, live)
        self.assertAlmostEqual(np.sum(W_ref**2), parts['W_ref_norm']**2)  # same basis rule as the channel split
        self.assertTrue(np.allclose(W_ref[~live], 0))
        self.assertTrue(np.allclose(W_ref + W_off + w[2].mean(1, keepdims=True), w[2]))
        y = (A[c.e3.COMMON] + B[c.e3.COMMON]) % P
        M = RNG.random(z.shape) < .5
        M[np.arange(len(y)), y] = False
        M[:, 0] = True
        parts = sum(c.linear_margin(x, y, M) for x in ch.values())
        self.assertTrue(np.allclose(parts, c.linear_margin(z, y, M)))  # margins add across channels

    def test_filtered_start_readout_is_all_reference(self):
        """filter_units keeps each W row at the unit's frequency only, so at update 0 a filtered
        start has no off-reference readout and its update-0 dominant frequency is the kept one."""
        for s in c.SOURCES:
            with np.load(c.ROOT/f'research/e3/mechanisms/starts/e3-me-filt-w128-i{s}-a1.npz') as f:
                w = [f[x] for x in 'UVW']
            k, _, live = c.e3.dominant(w)
            _, W_off = c.split_readout(w[2], k, live)
            self.assertLess(np.abs(W_off[live]).max(), 1e-12*np.abs(w[2]).max())


if __name__ == '__main__':
    unittest.main()
