"""Checks for history.py written independently of its label code: oracle by
enumeration, split integrity, the depth-0 readout gradient, and the branching
semantics (checkpoint round-trip, optimizer reset, frozen A head).

uv run --locked python -m unittest -v research/regimes/history/test_history.py
"""
import itertools
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import history as H  # noqa: E402
import nets  # noqa: E402  (on sys.path through history)

BITS = {'A': '012', 'Brel': '013', 'Bunrel': '456'}


class Oracle(unittest.TestCase):
    def test_labels_by_enumeration(self):
        X = H.domain()
        for task, bits in BITS.items():
            expected = np.zeros(1024, int)
            for x in itertools.product('01', repeat=10):  # x[k] is bit k
                index = sum(int(b) << k for k, b in enumerate(x))
                self.assertEqual(X[index].astype(int).tolist(), [int(b) for b in x])
                expected[index] = ''.join(x[int(k)] for k in bits).count('1') % 2
            np.testing.assert_array_equal(H.labels(X, task), expected)
        np.testing.assert_array_equal(H.labels(X, 'notA'), 1 - H.labels(X, 'A'))

    def test_balance_and_pairwise_independence(self):
        X = H.domain(); Y = {t: H.labels(X, t) for t in BITS}
        for t in BITS:
            self.assertEqual(Y[t].sum(), 512)
        for a, b in itertools.combinations(BITS, 2):
            self.assertEqual((Y[a] == Y[b]).sum(), 512)
        self.assertEqual(len({tuple(r) for r in X}), 1024)


class Split(unittest.TestCase):
    def test_disjoint_cover_and_reproducible(self):
        train, test = H.split()
        self.assertEqual((len(train), len(test)), (256, 768))
        self.assertEqual(len(np.intersect1d(train, test)), 0)
        np.testing.assert_array_equal(np.sort(np.concatenate([train, test])), np.arange(1024))
        self.assertEqual(len(np.unique(train)), 256)
        np.testing.assert_array_equal(train, H.split()[0])


class Readout(unittest.TestCase):
    def test_depth0_gradient_by_finite_differences(self):
        r = np.random.default_rng(0)
        F = r.normal(size=(20, 6)); y = r.integers(0, 2, 20)
        p = nets.init(6, (), {'R': 2}, 1); p['b_R'] += r.normal(size=2)
        _, _, g = nets.loss_grad(p, F, {'R': y})
        for k in p:
            num = np.zeros_like(p[k])
            for i in np.ndindex(p[k].shape):
                old = p[k][i]
                p[k][i] = old + 1e-6; up = nets.loss_grad(p, F, {'R': y})[0]
                p[k][i] = old - 1e-6; dn = nets.loss_grad(p, F, {'R': y})[0]
                p[k][i] = old; num[i] = (up - dn) / 2e-6
            np.testing.assert_allclose(g[k], num, rtol=1e-6, atol=1e-8)

    def test_null_cannot_generalize_on_independent_features(self):
        r = np.random.default_rng(1); train, test = H.split()
        Hf = r.normal(size=(1024, 16)); y = H.labels(H.domain(), 'A')
        res = H.readout(Hf, y, train, test, permute=True)
        self.assertLess(abs(res['2000']['test_acc'] - .5), .1)


class Branching(unittest.TestCase):
    def test_branch_from_checkpoint_equals_in_process_sequence(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            H.run(d/'a', [('A', 5)], init_seed=7)
            ck = d/'a'/'ckpt_t5.npz'
            branched = H.run(d/'b', [('Brel', 4)], source=ck, source_sha256=nets.sha256_file(ck))
            whole = H.run(d/'c', [('A', 5), ('Brel', 4)], init_seed=7)
            for k in whole:
                np.testing.assert_array_equal(branched[k], whole[k])
            a_end = nets.load(ck)[0]
            for k in ('W_A', 'b_A'):  # the B phase never touches the A head
                np.testing.assert_array_equal(branched[k], a_end[k])
            self.assertFalse(np.array_equal(branched['W1'], a_end['W1']))
            with self.assertRaises(ValueError):
                H.run(d/'e', [('A', 1)], source=ck, source_sha256='0'*64)

    def test_reset_head_restores_the_initial_draw_only(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            H.run(d/'a', [('A', 5), ('Brel', 3)], init_seed=7)
            ck = d/'a'/'ckpt_t8.npz'; trained = nets.load(ck)[0]
            out = H.run(d/'r', [('A', 1)], source=ck, reset_head_seed=7)
            init = nets.init(10, (128,), {'A': 2, 'B': 2}, 7)
            cfg = json.loads((d/'r'/'config.json').read_text())
            self.assertEqual(cfg['reset_head_A_to_init_seed'], 7)
            start = nets.load(d/'r'/'ckpt_t0.npz')[0]
            for k in ('W_A', 'b_A'):
                np.testing.assert_array_equal(start[k], init[k])
            for k in ('W1', 'b1', 'W_B', 'b_B'):
                np.testing.assert_array_equal(start[k], trained[k])
            self.assertFalse(np.array_equal(out['W_B'], init['W_B']))

    def test_optimizer_is_reset_at_each_phase(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            two = H.run(d/'two', [('A', 3), ('A', 3)], init_seed=7)
            one = H.run(d/'one', [('A', 6)], init_seed=7)
            p = nets.init(10, (128,), {'A': 2, 'B': 2}, 7)
            X = H.domain(); tr = H.split()[0]
            for _ in range(2):  # independent re-implementation of two reset phases
                st = nets.adam_state(p)
                for _ in range(3):
                    g = nets.loss_grad(p, X[tr], {'A': H.labels(X, 'A')[tr]})[2]
                    nets.adam_step(p, g, st, H.LR, H.WD, ['W1', 'b1', 'W_A', 'b_A'])
            for k in p:
                np.testing.assert_array_equal(two[k], p[k])
            self.assertFalse(np.array_equal(two['W1'], one['W1']))


if __name__ == '__main__':
    unittest.main()
