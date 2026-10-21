"""Code checks for research/mechanisms/freq.py. They test the analysis, not any
scientific hypothesis.

uv run --locked --group analysis python -m unittest discover -s research/mechanisms -v
"""
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from freq import (ROOT, curve, evaluate, filter_inputs, kendall_tau, prune_order, ranking, rescale,  # noqa: E402
                  spectrum, synthetic, targeted)
from drift import drift  # noqa: E402
from gauge import balance  # noqa: E402
from relational import absent_labels, relational, rotation, subspace_removal  # noqa: E402
from experiment import forward, init, interventions  # noqa: E402


def split():
    raw = np.load(ROOT/'research/data/e1-split.npz')
    return dict(a=raw['a'], b=raw['b'], y=raw['yadd'], yadd=raw['yadd'], train=raw['train'], cal=raw['cal'],
                audit=raw['audit'])


class FrequencyTests(unittest.TestCase):
    def setUp(self):
        self.d = split()
        self.w = init(31, 24, 5)

    def test_full_retention_and_empty_removal_reconstruct(self):
        all_freqs = np.arange(1, 16)
        ref = forward(self.w, self.d['a'], self.d['b'])[0]
        for w in (filter_inputs(self.w, all_freqs, True), filter_inputs(self.w, [], False)):
            np.testing.assert_allclose(forward(w, self.d['a'], self.d['b'])[0], ref, rtol=0, atol=1e-12)

    def test_full_removal_leaves_token_independent_embeddings(self):
        w = filter_inputs(self.w, np.arange(1, 16), False)
        for x, orig in zip(w[:2], self.w[:2]):
            np.testing.assert_allclose(x, np.broadcast_to(orig.mean(0), x.shape), atol=1e-12)

    def test_matches_historical_top4_intervention(self):
        # The historical intervention at K=4 (experiment.interventions) on the audit split.
        w = synthetic(self.d)
        w[0] = w[0] + init(31, w[0].shape[1], 9)[0]*.3  # break ties and add every frequency
        hist = interventions(w, self.d['a'], self.d['b'], self.d['y'], self.d['audit'], 1, 1)
        rows = {(r['mode'], r['K']): r for r in curve(w, self.d, ranking(w))}
        self.assertEqual(hist['frequencies'], ranking(w)[:4].tolist())
        self.assertEqual(hist['retained'], rows['retain', 4]['audit_accuracy'])
        self.assertEqual(hist['removed'], rows['remove', 4]['audit_accuracy'])


class RescalingTests(unittest.TestCase):
    def setUp(self):
        self.d = split()
        self.w = init(31, 24, 5)
        self.c = np.exp(np.random.default_rng(3).normal(0, 1, 24))

    def test_rescaling_preserves_logits_but_not_energy(self):
        wr = rescale(self.w, self.c)
        ref = forward(self.w, self.d['a'], self.d['b'])[0]
        np.testing.assert_allclose(forward(wr, self.d['a'], self.d['b'])[0], ref, rtol=1e-12, atol=1e-12)
        self.assertFalse(np.allclose(spectrum(wr).sum(1), spectrum(self.w).sum(1)))
        np.testing.assert_array_equal(prune_order(wr), prune_order(self.w))

    def test_fixed_frequency_filter_commutes_with_rescaling(self):
        # Only the selection can change under rescaling; filtering a fixed set cannot.
        chosen = [2, 5, 9]
        for keep in (True, False):
            a = forward(filter_inputs(rescale(self.w, self.c), chosen, keep), self.d['a'], self.d['b'])[0]
            b = forward(filter_inputs(self.w, chosen, keep), self.d['a'], self.d['b'])[0]
            np.testing.assert_allclose(a, b, rtol=1e-11, atol=1e-11)

    def test_kendall_tau(self):
        r = np.arange(1, 16)
        self.assertEqual(kendall_tau(r, r), 1.0)
        self.assertEqual(kendall_tau(r, r[::-1]), -1.0)


class GaugeTests(unittest.TestCase):
    def setUp(self):
        self.d = split()
        self.w = init(31, 24, 5)
        self.w[2][3] = 0  # a unit that computes nothing keeps gain 1
        self.c = np.exp(np.random.default_rng(4).normal(0, 1, 24))

    def test_balanced_units_preserve_logits(self):
        wb, c = balance(self.w)
        np.testing.assert_allclose(np.sum(wb[0]**2, 0) + np.sum(wb[1]**2, 0),
                                   np.where(np.arange(24) == 3, np.sum(self.w[0][:, 3]**2 + self.w[1][:, 3]**2),
                                            np.sum(wb[2]**2, 1)), rtol=1e-12)
        self.assertEqual(c[3], 1.)
        np.testing.assert_allclose(forward(wb, self.d['a'], self.d['b'])[0], forward(self.w, self.d['a'], self.d['b'])[0],
                                   rtol=1e-12, atol=1e-12)

    def test_rescaled_copies_normalize_to_the_same_weights(self):
        # Units that compute nothing are left as they are, so this holds for live units only.
        w = init(31, 24, 5)
        wb = balance(w)[0]
        for x, y in zip(balance(rescale(w, self.c))[0], wb):
            np.testing.assert_allclose(x, y, rtol=1e-12, atol=1e-15)
        for x, y in zip(balance(wb)[0], wb):  # idempotent
            np.testing.assert_allclose(x, y, rtol=1e-12, atol=1e-15)

    def test_balanced_copy_has_minimum_norm_on_the_orbit(self):
        wb = balance(self.w)[0]
        norm = lambda w: sum(np.sum(x*x) for x in w)
        rng = np.random.default_rng(6)
        for _ in range(20):
            self.assertLessEqual(norm(wb), norm(rescale(wb, np.exp(rng.normal(0, .3, 24)))) + 1e-12)


class RelationalTests(unittest.TestCase):
    def setUp(self):
        self.d = split()
        self.held, self.absent = absent_labels(self.d)

    def test_true_label_is_always_absent(self):
        rows = np.arange(len(self.held))
        self.assertTrue(self.absent[rows, self.d['yadd'][self.held]].all())
        self.assertEqual(len(self.held), 662)

    def test_rank_statistic_extremes(self):
        y = self.d['yadd']
        logits = np.zeros((961, 31))
        logits[np.arange(961), y] = 1.  # true label strictly above everything
        r = relational(logits, self.d, self.held, self.absent)
        self.assertEqual((r['true_rank_among_absent'], r['true_top_among_absent'], r['pred_is_present']), (0., 1., 0.))
        logits = -logits  # strictly below everything
        r = relational(logits, self.d, self.held, self.absent)
        self.assertEqual((r['true_rank_among_absent'], r['true_top_among_absent']), (1., 0.))

    def test_null_rotation_matches_norm_rank_and_spectrum(self):
        r = rotation(31, np.random.default_rng(1))
        np.testing.assert_allclose(r @ r.T, np.eye(31), atol=1e-12)
        np.testing.assert_allclose(r @ np.ones(31), np.ones(31), atol=1e-12)
        w = init(31, 24, 5)
        delta = w[0] - filter_inputs(w, [2, 5, 9], False)[0]
        np.testing.assert_allclose(np.linalg.svd(r @ delta, compute_uv=False), np.linalg.svd(delta, compute_uv=False),
                                   atol=1e-12)
        np.testing.assert_allclose((r @ delta).mean(0), 0, atol=1e-12)

    def test_random_subspace_null(self):
        w, r = init(31, 24, 5), rotation(31, np.random.default_rng(2))
        # With R = I it is top-K removal; with every frequency it leaves DC only, for any R.
        for x, y in zip(subspace_removal(w, [2, 5], np.eye(31)), filter_inputs(w, [2, 5], False)):
            np.testing.assert_allclose(x, y, atol=1e-12)
        for x, y in zip(subspace_removal(w, np.arange(1, 16), r), filter_inputs(w, np.arange(1, 16), False)):
            np.testing.assert_allclose(x, y, atol=1e-12)
        # It is a projection: it removes energy, and its change has rank 2K in token space.
        wn = subspace_removal(w, [2, 5, 9], r)
        self.assertLess(np.sum(wn[0]**2), np.sum(w[0]**2))
        self.assertEqual(np.linalg.matrix_rank(w[0] - wn[0], tol=1e-10), 6)
        for x, y in zip(subspace_removal(wn, [2, 5, 9], r), wn):  # idempotent
            np.testing.assert_allclose(x, y, atol=1e-12)


class DriftTests(unittest.TestCase):
    def test_change_at_one_frequency(self):
        w = init(31, 24, 5)
        moved = [w[0] + (w[0] - filter_inputs(w, [7], False)[0]), w[1], 2*w[2]]
        out = drift(w, moved)
        self.assertEqual(out['change_ranking'][0], 7)
        self.assertAlmostEqual(out['change_energy_share'][6], 1.)
        self.assertEqual(out['relative_change']['V'], 0.)
        self.assertAlmostEqual(out['relative_change']['W'], 1.)


class SyntheticTests(unittest.TestCase):
    """Known ground truth: frequencies 3, 7, 11, 14 with input gains 1, .9, .8, .7."""

    def setUp(self):
        self.d = split()
        self.w = synthetic(self.d)

    def test_ground_truth(self):
        energy = spectrum(self.w).sum(1)
        self.assertEqual(ranking(self.w)[:4].tolist(), [3, 7, 11, 14])
        self.assertLess(energy[[k for k in range(16) if k not in (3, 7, 11, 14)]].max(), 1e-20*energy.max())
        metrics, logits = evaluate(self.w, self.d)
        self.assertEqual(metrics['audit_accuracy'], 1.0)
        rows = {(r['mode'], r['K']): r for r in curve(self.w, self.d, ranking(self.w))}
        np.testing.assert_allclose(forward(filter_inputs(self.w, [3, 7, 11, 14], True), self.d['a'], self.d['b'])[0],
                                   logits, atol=1e-12)
        # Removing all four zeroes every hidden unit: logits 0, loss ln 31.
        self.assertAlmostEqual(rows['remove', 4]['audit_cross_entropy'], np.log(31), places=10)

    def test_targeted_rescaling_reorders_without_changing_function(self):
        info, c = targeted(self.w, 1)
        self.assertEqual(info['target'], 7)
        wr = rescale(self.w, c)
        self.assertEqual(ranking(wr)[0], 7)
        np.testing.assert_allclose(evaluate(wr, self.d)[1], evaluate(self.w, self.d)[1], rtol=1e-12, atol=1e-12)


if __name__ == '__main__':
    unittest.main()
