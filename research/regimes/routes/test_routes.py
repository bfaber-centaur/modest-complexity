"""Independent checks of the R4 oracle, split, conditions and intervention pairs.

The domain is rebuilt here with itertools and pure-Python parity over bit
tuples, rather than by calling task.py's index arithmetic, and every property
in QUESTION.md "Validation" is checked against it.

uv run --locked python -m unittest -v research/regimes/routes/test_routes.py
"""
from itertools import product
from pathlib import Path
import sys
import unittest
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent/'common'))
import task  # noqa: E402
import nets  # noqa: E402


def bits_of(row):
    return tuple(int(v > 0) for v in row)


def parity(bits):  # rule bits are inputs 0-2
    return (bits[0] + bits[1] + bits[2]) % 2


class RouteTaskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d = task.domain()
        cls.rows = [bits_of(r) for r in cls.d['X']]
        cls.ev = task.evaluation_sets()
        cls.sets = {c: task.training_set(c) for c in task.CONDITIONS}

    def test_domain_is_every_input_once_with_parity_labels(self):
        self.assertEqual(sorted(self.rows), sorted(product((0, 1), repeat=10)))
        self.assertTrue(set(np.unique(self.d['X'])) == {-1., 1.})
        for bits, y in zip(self.rows, self.d['y']):
            self.assertEqual(parity(bits), y)

    def test_split_is_by_context_disjoint_and_closed(self):
        held = self.ev['held']
        base = lambda b: b[:3] + b[4:]  # everything but s
        train_bases = {base(self.rows[i]) for i in np.flatnonzero(~held)}
        held_bases = {base(self.rows[i]) for i in np.flatnonzero(held)}
        self.assertEqual((len(train_bases), len(held_bases)), (256, 256))
        self.assertFalse(train_bases & held_bases)
        train_ctx = {b[4:] for b in (self.rows[i] for i in np.flatnonzero(~held))}
        held_ctx = {b[4:] for b in (self.rows[i] for i in np.flatnonzero(held))}
        self.assertEqual((len(train_ctx), len(held_ctx)), (32, 32))
        self.assertFalse(train_ctx & held_ctx)
        held_rows = {self.rows[i] for i in np.flatnonzero(held)}
        for b in held_rows:  # closed under s and rule-bit flips
            for col in (0, 1, 2, 3):
                self.assertIn(b[:col] + (1 - b[col],) + b[col+1:], held_rows)
        for cond, t in self.sets.items():
            self.assertFalse({self.rows[i] for i in t['idx']} & held_rows, cond)
            self.assertFalse({base(self.rows[i]) for i in t['idx']} & held_bases, cond)

    def test_irrelevant_bits_independent_of_label_in_training(self):
        t = self.sets['rho1']
        X = np.array([self.rows[i] for i in t['idx']]); y = t['target']
        for col in range(4, 10):
            for v in (0, 1):
                self.assertEqual(y[X[:, col] == v].mean(), 0.5)

    def test_conditions_realize_reliability_balance_and_nesting(self):
        prev = None
        for cond, k in task.EXCEPTIONS.items():
            t = self.sets[cond]
            X = np.array([self.rows[i] for i in t['idx']]); y = t['target']
            self.assertEqual(len(set(map(tuple, X))), 256, 'duplicate training inputs')
            self.assertTrue(all(parity(r) == v for r, v in zip(X, y)))
            s = X[:, 3]
            self.assertEqual((s == y).mean(), 1 - k/32)
            self.assertEqual((y.mean(), s.mean()), (0.5, 0.5))
            self.assertTrue(np.array_equal(t['exception'], s != y))
            pattern = X[:, 0] + 2*X[:, 1] + 4*X[:, 2]
            for a in range(8):
                self.assertEqual(int(t['exception'][pattern == a].sum()), k)
            if prev is not None:
                self.assertTrue(np.all(t['exception'] >= prev), 'exceptions must nest')
            prev = t['exception']
            # Pairing: same bases and labels as rho1; only s on exception rows differs.
            X1 = np.array([self.rows[i] for i in self.sets['rho1']['idx']])
            diff = X != X1
            self.assertTrue(np.array_equal(diff.any(1), t['exception']))
            self.assertFalse(np.delete(diff, 3, axis=1).any())
            self.assertTrue(np.array_equal(y, self.sets['rho1']['target']))

    def test_shortcut_only_target_is_s_and_independent_of_rule(self):
        t, half = self.sets['shortcut_only'], self.sets['rho1of2']
        self.assertTrue(np.array_equal(t['idx'], half['idx']))
        X = np.array([self.rows[i] for i in t['idx']])
        self.assertTrue(np.array_equal(t['target'], X[:, 3]))
        pattern = X[:, 0] + 2*X[:, 1] + 4*X[:, 2]
        for a in range(8):
            self.assertEqual(t['target'][pattern == a].mean(), 0.5)

    def test_environments_and_flip_pairs(self):
        ev, rows, y = self.ev, self.rows, self.d['y']
        self.assertEqual((ev['reliable'].sum(), ev['reversed'].sum(), ev['held'].sum()), (256, 256, 512))
        for i in np.flatnonzero(ev['reliable']):
            self.assertEqual(rows[i][3], y[i])
        for i in np.flatnonzero(ev['reversed']):
            self.assertEqual(rows[i][3], 1 - y[i])
        expected_col = {'s': 3, **{f'rule{j}': j for j in range(3)}, **{f'irr{j}': 4+j for j in range(6)}}
        for name, P in ev['pairs'].items():
            col = expected_col[name]
            self.assertEqual(len({tuple(p) for p in P}), len(P))
            if name == 's' or name.startswith('rule'):
                self.assertEqual(len(P), 256, name)
            else:
                self.assertGreater(len(P), 50, name)
            for i, j in P:
                self.assertTrue(ev['held'][i] and ev['held'][j], name)
                changed = [k for k in range(10) if rows[i][k] != rows[j][k]]
                self.assertEqual(changed, [col], name)
                if name == 's':
                    self.assertEqual((rows[i][3], rows[j][3]), (0, 1))
                if name.startswith('rule'):
                    self.assertEqual((y[i], y[j]), (0, 1))

    def test_metrics_on_oracle_functions(self):
        y, s = self.d['y'], self.d['s']
        t = self.sets['rho7of8']
        rule = task.metrics(2.*y - 1, t, self.ev, y)
        short = task.metrics(2.*s - 1, t, self.ev, y)
        self.assertEqual((rule['acc_reliable'], rule['acc_reversed'], rule['acc_independent']), (1, 1, 1))
        self.assertEqual((rule['flip_rule'], rule['flip_s'], rule['flip_irr']), (1, 0, 0))
        self.assertEqual((rule['train_acc'], rule['delta_rule'], rule['delta_s']), (1, 2, 0))
        self.assertEqual((short['acc_reliable'], short['acc_reversed'], short['acc_independent']), (1, 0, .5))
        self.assertEqual((short['flip_rule'], short['flip_s'], short['flip_irr']), (0, 1, 0))
        self.assertEqual((short['train_acc'], short['train_acc_exceptions'], short['delta_s']), (28/32, 0, 2))

    def test_finite_difference_gradients_on_actual_configuration(self):
        t = self.sets['rho3of4']; X = self.d['X'][t['idx']]
        p = nets.init(10, (128,), {'y': 2}, 54399)
        r = np.random.default_rng(54398)
        for k in p:
            if k.startswith('b'):
                p[k] = r.normal(0, .1, p[k].shape)
        _, _, g = nets.loss_grad(p, X, {'y': t['target']})
        eps = 1e-6
        for k in p:
            for flat in r.choice(p[k].size, min(8, p[k].size), replace=False):
                idx = np.unravel_index(flat, p[k].shape); old = p[k][idx]
                p[k][idx] = old + eps; up = nets.loss_grad(p, X, {'y': t['target']})[0]
                p[k][idx] = old - eps; dn = nets.loss_grad(p, X, {'y': t['target']})[0]
                p[k][idx] = old
                self.assertAlmostEqual((up - dn)/(2*eps), g[k][idx], delta=1e-7 + 1e-5*abs(g[k][idx]))


if __name__ == '__main__':
    unittest.main()
