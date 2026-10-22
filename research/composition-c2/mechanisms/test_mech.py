"""Tests for the mechanisms lane. Run in its own process:
uv run --locked python -m unittest -v research/composition-c2/mechanisms/test_mech.py
"""
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mech_core as mc  # noqa: E402
import c2rnn  # noqa: E402
import c2task  # noqa: E402


class TestMech(unittest.TestCase):
    def setUp(self):
        self.T = c2task.load(c2task.ORIGINAL)
        self.rng = np.random.default_rng(0)

    def test_enumeration_orders(self):
        R = c2task.rows(self.T['P'], c2task.programs(3))
        pre = mc.prefixes(3)
        self.assertTrue(np.array_equal(pre['s'], R['s']) and np.array_equal(pre['ops'], R['ops']))
        self.assertTrue(np.array_equal(mc.apply_ops(self.T['P'], R['s'], R['ops']), R['states'][:, -1]))
        for i in range(4):
            pid = mc.lex(R['ops'][:, :i]) * mc.N_SYM + R['s']
            p = mc.prefixes(i)
            self.assertTrue(np.array_equal(p['s'][pid], R['s']) and np.array_equal(p['ops'][pid], R['ops'][:, :i]))
            self.assertTrue(np.array_equal(mc.suffixes(3-i)[mc.lex(R['ops'][:, i:])], R['ops'][:, i:]))

    def test_patch_is_recombined_forward(self):
        """Whole-state interchange == plain forward on donor prefix ++ base suffix, in all schemes
        with a shared transition (the recovered conditions), for every (i, j)."""
        for tie_readout in (True, False):
            p = c2rnn.init(128, 1, tie_readout=tie_readout)
            for k in p:  # move away from the init's identical copies and zero biases
                p[k] = p[k] + 0.1*self.rng.normal(size=p[k].shape)
            for i in range(4):
                for j in range(4):
                    pre, sfx = mc.prefixes(i), mc.suffixes(3-j)
                    sel = self.rng.integers(len(pre['s']), size=40); ss = self.rng.integers(len(sfx), size=40)
                    h = mc.state(p, pre['s'][sel], pre['ops'][sel])
                    patched = mc.run_suffix(p, h, sfx[ss], j)
                    ref = mc.recombined_state(p, pre['s'][sel], pre['ops'][sel], sfx[ss])
                    np.testing.assert_allclose(patched, ref, rtol=0, atol=1e-12)
                    if i == j == 0:  # whole program from the start state is the ordinary forward pass
                        H, logits = c2rnn.forward(p, sfx[ss], s=pre['s'][sel])
                        np.testing.assert_array_equal(mc.readout(p, patched), logits[-1])

    def test_untied_transition_is_not_recombination(self):
        """Guard: the equivalence needs a shared transition; with per-step transitions a
        cross-position patch differs from the recombined forward pass."""
        p = c2rnn.init(128, 1, tie_transition=False)
        for k in p:
            p[k] = p[k] + 0.1*self.rng.normal(size=p[k].shape)
        pre, sfx = mc.prefixes(2), mc.suffixes(3)
        h = mc.state(p, pre['s'][:8], pre['ops'][:8])
        patched = mc.run_suffix(p, h, sfx[:8], 0)
        ref = c2rnn.forward(p, np.concatenate([pre['ops'][:8], sfx[:8]], 1), s=pre['s'][:8], step_map=[1, 2, 3, 3, 3])[0][-1]  # any frame but the patch's own
        self.assertGreater(np.abs(patched - ref).max(), 1e-3)

    def test_readout_map(self):
        p = c2rnn.init(128, 2, tie_readout=False)
        p['V3'] = p['V3'] + 1.0
        h = self.rng.normal(size=(5, 128))
        np.testing.assert_array_equal(mc.readout(p, h), h @ p['V3'] + p['bv3'])
        np.testing.assert_array_equal(mc.readout(p, h, 1), h @ p['V1'] + p['bv1'])

    def test_agreement(self):
        pred = np.array([[0, 1], [0, 2], [0, 1], [5, 5]])
        group = np.array([0, 0, 0, 1])  # group 1 has one member: skipped
        # condition 0: 3 of 3 agree -> 1; condition 1: labels 1,2,1 -> pairs agreeing 2 of 6 ordered = 1/3
        self.assertAlmostEqual(mc.agreement(pred, group), (1 + 1/3) / 2)
        self.assertAlmostEqual(mc.cross_agreement(pred[:2], group[:2], pred[2:3], group[2:3]), (1 + 1/2) / 2)

    def test_centroid_basis(self):
        import mech_residual as mr
        H = self.rng.normal(size=(80, 128)); y = np.repeat(np.arange(8), 10)
        c, U = mr.centroid_basis(H, y)
        np.testing.assert_allclose(U.T @ U, np.eye(7), atol=1e-12)
        centered = c - c.mean(0)
        np.testing.assert_allclose(mr.proj(U, centered), centered, atol=1e-12)  # U spans the centered means
        r = self.rng.normal(size=(5, 128))
        np.testing.assert_allclose(mr.proj(U, mr.proj(U, r)), mr.proj(U, r), atol=1e-12)
        Q = mr.random_bases(np.random.default_rng(1), 128)[0]
        np.testing.assert_allclose(Q.T @ Q, np.eye(7), atol=1e-12)

    def test_checkpoint_hash_refused(self):
        ck = dict(mc.checkpoints([0])[0]); ck['sha256'] = '0'*64
        with self.assertRaises(ValueError):
            mc.load_verified(ck)


if __name__ == '__main__':
    unittest.main()
