"""Checks of the correction lane's sets, centroids, map gradient and wrapper use. No fit on a base.

uv run --locked python -m unittest -v research/composition-c3/correction/test_corr.py
"""
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import corr_core as cc  # noqa: E402
import c2rnn  # noqa: E402


class TestCorrection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.T, cls.R, cls.pre = cc.task()

    def test_prefixes_and_rows(self):
        R, pre = self.R, self.pre
        self.assertEqual(len(pre['x']), 128)
        self.assertTrue(np.all(np.bincount(pre['x'], minlength=8) == 16))
        self.assertEqual(int(R['held'].sum()), 128)
        for i in range(128):  # each prefix: 1 held-out op 3 and 3 training ones, same h_2 symbol
            rows = np.flatnonzero(R['prefix'] == i)
            self.assertEqual(len(rows), 4)
            self.assertEqual(int(R['held'][rows].sum()), 1)
            self.assertTrue(np.all(R['s'][rows] == pre['s'][i]) and np.all(R['ops'][rows, :2] == pre['ops'][i]))
            self.assertTrue(np.all(R['states'][rows, 1] == pre['x'][i]))

    def test_split(self):
        sp = cc.split(self.pre)
        self.assertEqual({k: len(v) for k, v in sp.items()}, dict(fit=64, val=32, eval=32))
        allidx = np.concatenate(list(sp.values()))
        self.assertEqual(sorted(allidx.tolist()), list(range(128)))
        for k, n in cc.GROUPS.items():
            self.assertTrue(np.all(np.bincount(self.pre['x'][sp[k]], minlength=8) == n))
        again = cc.split(self.pre)
        self.assertTrue(all(np.array_equal(sp[k], again[k]) for k in sp))

    def test_unique_weighted_centroids_equal_row_weighted(self):
        b = cc.bases()[3]
        p, rec = cc.load_base(b)
        C = cc.centroids(cc.states(p, self.pre), self.pre['x'])
        tr = ~self.R['held']
        H2 = c2rnn.forward(p, self.R['ops'][tr], s=self.R['s'][tr])[0][2]
        y = self.R['states'][tr, 1]
        Crow = np.stack([H2[y == k].mean(0) for k in range(8)])
        np.testing.assert_allclose(C, Crow, rtol=0, atol=1e-12)
        self.assertEqual(rec['params_sha256'], cc.c3correct.params_sha256(p))

    def test_frozen_base_and_identity(self):
        p, _ = cc.load_base(cc.bases()[0])
        with self.assertRaises(ValueError):
            p['Ws'][0, 0] = 1.
        sel = np.arange(512)
        full = c2rnn.forward(p, self.R['ops'], s=self.R['s'])[1][-1]
        a = cc.accuracy(p, self.R, sel, None)
        self.assertEqual(a, float(np.mean(full.argmax(1) == self.R['states'][:, -1])))

    def test_map_gradient(self):
        rng = np.random.default_rng(0)
        H = np.maximum(rng.normal(0.5, 1, (20, cc.WIDTH)), 0)
        Y = np.maximum(rng.normal(0.5, 1, (20, cc.WIDTH)), 0)
        q = cc.map_init(8, 1)
        q['b2'] += 0.3
        loss, g = cc.map_loss_grad(q, H, Y)
        for k in q:
            for _ in range(3):
                i = tuple(rng.integers(0, n) for n in q[k].shape)
                e = 1e-6
                q[k][i] += e; up = cc.map_mse(q, H, Y)
                q[k][i] -= 2*e; dn = cc.map_mse(q, H, Y)
                q[k][i] += e
                self.assertAlmostEqual(g[k][i], (up - dn)/(2*e), delta=1e-6 + 1e-4*abs(g[k][i]))

    def test_map_same_draw_for_B_and_C(self):
        a, b = cc.map_init(16, cc.MAP_SEED), cc.map_init(16, cc.MAP_SEED)
        self.assertTrue(all(np.array_equal(a[k], b[k]) for k in a))
        out = cc.map_forward(a, np.ones((3, cc.WIDTH)))[1]
        self.assertTrue(np.all(out >= 0))

    def test_corrections_see_only_h(self):
        """B, C and D are functions of the state matrix alone: permuting rows permutes outputs."""
        rng = np.random.default_rng(1)
        H = np.maximum(rng.normal(0.5, 1, (16, cc.WIDTH)), 0)
        x = np.repeat(np.arange(8), 2)
        mods, _ = cc.fit_arms(H, x, 8, 1e-2)
        perm = rng.permutation(16)
        for arm in 'BCD':
            fn = cc.correction(arm, mods)
            np.testing.assert_array_equal(fn(H)[perm], fn(H[perm]))


if __name__ == '__main__':
    unittest.main()
