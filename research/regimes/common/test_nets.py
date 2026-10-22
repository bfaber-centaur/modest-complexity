"""Code checks for the shared regime network: gradients, unlabeled rows, Adam
against an independent calculation, frozen parameters, optimizer reset and
checkpoint round-trip. They test the implementation, not any scientific claim.

uv run --locked python -m unittest -v research/regimes/common/test_nets.py
"""
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nets  # noqa: E402


def problem(seed=0):
    r = np.random.default_rng(seed)
    X = r.integers(0, 2, (13, 6)).astype(float)
    targets = dict(a=r.integers(0, 3, 13), b=r.integers(0, 2, 13))
    targets['b'][[1, 4, 7]] = -1  # unlabeled rows
    p = nets.init(6, (5, 4), dict(a=3, b=2), seed=3)
    return p, X, targets


class NetTests(unittest.TestCase):
    def test_finite_difference_gradients(self):
        p, X, targets = problem()
        # Binary inputs with zero biases put the all-zero row exactly on the ReLU
        # kink, where a central difference is not the subgradient; move off it.
        for k in ('b1', 'b2'):
            p[k] = np.random.default_rng(2).normal(0, .1, p[k].shape)
        weights = dict(a=1., b=.5)
        _, _, g = nets.loss_grad(p, X, targets, weights)
        r = np.random.default_rng(1)
        for k in p:
            for _ in range(3):
                ix = tuple(r.integers(0, n) for n in p[k].shape)
                old = p[k][ix]; eps = 1e-6
                p[k][ix] = old+eps; lp = nets.loss_grad(p, X, targets, weights)[0]
                p[k][ix] = old-eps; lm = nets.loss_grad(p, X, targets, weights)[0]
                p[k][ix] = old
                self.assertTrue(np.isclose((lp-lm)/(2*eps), g[k][ix], rtol=1e-4, atol=1e-8), k)

    def test_unlabeled_rows_and_absent_heads(self):
        p, X, targets = problem()
        keep = targets['b'] >= 0
        _, per_head, _ = nets.loss_grad(p, X, dict(b=targets['b']))
        _, sub, _ = nets.loss_grad(p, X[keep], dict(b=targets['b'][keep]))
        self.assertAlmostEqual(per_head['b'], sub['b'], places=12)
        _, _, g = nets.loss_grad(p, X, dict(b=targets['b']))
        self.assertFalse(np.any(g['W_a']) or np.any(g['b_a']))

    def test_head_init_independent_of_other_heads(self):
        one = nets.init(6, (5,), dict(final=8), seed=9)
        more = nets.init(6, (5,), dict(step1=8, step2=8, final=8), seed=9)
        for k in one:
            np.testing.assert_array_equal(one[k], more[k])

    def test_adam_matches_independent_calculation(self):
        p, X, targets = problem()
        ref = {k: v.copy() for k, v in p.items()}
        st = nets.adam_state(p)
        m = {k: 0. for k in p}; v = {k: 0. for k in p}
        lr, wd = .01, .1
        for t in range(1, 6):
            _, _, g = nets.loss_grad(p, X, targets)
            _, _, gr = nets.loss_grad(ref, X, targets)
            nets.adam_step(p, g, st, lr, wd)
            for k in ref:
                m[k] = .9*m[k] + .1*gr[k]; v[k] = .999*v[k] + .001*gr[k]**2
                mh = m[k]/(1-.9**t); vh = v[k]/(1-.999**t)
                ref[k] = ref[k] - lr*wd*ref[k] - lr*mh/(np.sqrt(vh)+1e-8)
        for k in p:
            np.testing.assert_allclose(p[k], ref[k], rtol=1e-12, atol=1e-14)

    def test_frozen_parameters_get_no_update_or_decay(self):
        p, X, targets = problem()
        frozen = {k: p[k].copy() for k in ('W1', 'b1', 'W2', 'b2')}
        st = nets.adam_state(p)
        heads = [k for k in p if k not in frozen]
        for _ in range(3):
            _, _, g = nets.loss_grad(p, X, targets)
            nets.adam_step(p, g, st, .01, wd=.5, trainable=heads)
        for k, v in frozen.items():
            np.testing.assert_array_equal(p[k], v); self.assertFalse(np.any(st['m'][k]))

    def test_training_decreases_loss(self):
        p, X, targets = problem()
        before = nets.loss_grad(p, X, targets)[0]; st = nets.adam_state(p)
        for _ in range(100):
            nets.adam_step(p, nets.loss_grad(p, X, targets)[2], st, .01)
        self.assertLess(nets.loss_grad(p, X, targets)[0], .5*before)

    def test_checkpoint_round_trip_continues_bitwise(self):
        p, X, targets = problem(); st = nets.adam_state(p)
        for _ in range(4):
            nets.adam_step(p, nets.loss_grad(p, X, targets)[2], st, .01)
        with tempfile.TemporaryDirectory() as d:
            nets.save(Path(d)/'c.npz', p, st, step=4, phase='A')
            q, sq, meta = nets.load(Path(d)/'c.npz')
        self.assertEqual(meta, dict(step=4, phase='A')); self.assertEqual(sq['t'], 4)
        for _ in range(3):
            nets.adam_step(p, nets.loss_grad(p, X, targets)[2], st, .01)
            nets.adam_step(q, nets.loss_grad(q, X, targets)[2], sq, .01)
        for k in p:
            np.testing.assert_array_equal(p[k], q[k])
        fresh = nets.adam_state(q)
        self.assertEqual(fresh['t'], 0); self.assertFalse(any(np.any(x) for x in fresh['m'].values()))

    def test_forward_from_hidden_layer_matches_forward(self):
        p, X, _ = problem()
        acts, logits = nets.forward(p, X)
        acts1, logits1 = nets.forward_from(p, 1, acts[1])
        for h in logits:
            np.testing.assert_array_equal(logits[h], logits1[h])
        np.testing.assert_array_equal(acts[2], acts1[1])


if __name__ == '__main__':
    unittest.main()
