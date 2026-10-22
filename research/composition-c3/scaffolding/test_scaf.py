"""Checks of the scaffolding lane's plan, schedule restatement, threshold and retention rules,
checkpoint inventory and frozen forward. Code checks only, no scientific claim.

uv run --locked python -m unittest -v research/composition-c3/scaffolding/test_scaf.py
"""
import json
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scaf_analyze as sa  # noqa: E402

PLAN = {e['run_id']: e['argv'] for e in json.loads((sa.HERE/'plan.json').read_text())['runs']}


def flag(argv, name, n=1):
    i = argv.index(name)
    return argv[i+1:i+1+n] if n > 1 else argv[i+1]


def flag_list(argv, name):
    i, out = argv.index(name) + 1, []
    while i < len(argv) and not argv[i].startswith('--'):
        out.append(argv[i]); i += 1
    return out


class Plan(unittest.TestCase):
    def test_twelve_branches_with_the_declared_constants(self):
        ids = [sa.run_id(k, S, s) for S in sa.SWITCH for k in sa.KINDS for s in sa.SEEDS]
        self.assertEqual(sorted(i for i in PLAN if '-calib-' not in i and '-reset-' not in i and '-verify-' not in i), sorted(ids))
        for kind, D in sa.KINDS.items():
            for S in sa.SWITCH:
                for seed in sa.SEEDS:
                    a = PLAN[sa.run_id(kind, S, seed)]
                    self.assertEqual([int(flag(a, '--switch-step')), int(flag(a, '--taper-updates')), int(flag(a, '--init-seed')),
                                      int(flag(a, '--updates'))], [S, D, seed, sa.END])
                    self.assertEqual(flag(a, '--switch-weights', 3), ['0', '0', '1'])
                    self.assertEqual(flag(a, '--step-weights', 3), ['1', '1', '1'])
                    self.assertEqual(Path(flag(a, '--resume')), sa.ckpt_path('keep', seed, S).relative_to(sa.REPO))
                    self.assertEqual(flag(a, '--resume-sha256'), sa.nets.sha256_file(sa.ckpt_path('keep', seed, S)))
                    self.assertIn('--diag', a)
                    saved = {int(x) for x in flag_list(a, '--checkpoint-steps')} | {sa.END}
                    self.assertEqual(saved, set(sa.ckpt_steps(kind, S)) - {S})


def without(argv, flags):
    """argv with each named flag and its values removed (run ID and purpose always dropped)."""
    out, skip = [], False
    for v in argv:
        if v.startswith('--'):
            skip = v in flags + ('--run-id', '--purpose')
        if not skip:
            out.append(v)
    return out


class FollowupPlan(unittest.TestCase):
    def test_reset_arms_equal_their_carried_matches_except_the_reset(self):
        switch = ('--switch-step', '--switch-weights', '--taper-updates')
        for seed in sa.SEEDS:
            drop = PLAN[sa.run_id('drop', 1000, seed)]
            for arm, want in (('drop', without(drop, ())), ('keep', without(drop, switch))):
                a = PLAN[f'c3-scaf-{arm}-reset-at1k-s{seed}-a1']
                self.assertEqual(flag(a, '--reset-optimizer-at'), '1000')
                self.assertEqual(without(a, ('--reset-optimizer-at',)), want)
        for arm in ('keep', 'drop'):
            a = PLAN[f'c3-scaf-verify-carried-{arm}-at1k-s51200-a1']
            self.assertNotIn('--reset-optimizer-at', a)
            self.assertEqual([flag(a, '--updates'), flag(a, '--checkpoint-steps')], ['3000', '3000'])


class Schedule(unittest.TestCase):
    def test_restatement_equals_c3train(self):
        for S in sa.SWITCH:
            for D in sa.KINDS.values():
                sched = sa.schedule('drop' if D == 1 else 'taper', S)
                for t in list(range(0, 50)) + list(range(S - 5, S + D + 5)) + [sa.END - 1, sa.END]:
                    w = sa.c3train.weights_at(t, sched)
                    self.assertEqual(w[:2], [sa.weights_recount(t, S, D)]*2, (S, D, t))
                    self.assertEqual(w[2], 1.)

    def test_taper_end_points(self):
        self.assertEqual(sa.weights_recount(9999, 10000, 2000), 1.)
        self.assertEqual(sa.weights_recount(10000, 10000, 2000), 1 - 1/2000)
        self.assertAlmostEqual(sa.weights_recount(11998, 10000, 2000), 1/2000, places=14)
        self.assertEqual(sa.weights_recount(11999, 10000, 2000), 0.)
        self.assertEqual(sa.weights_recount(1000, 1000, 1), 0.)


class Threshold(unittest.TestCase):
    def steps(self, n):
        return list(range(0, 50*n, 50))

    def test_first_of_five_consecutive(self):
        self.assertEqual(sa.first_sustained(self.steps(7), [.90, .95, .96, .97, .95, .99, .50]), 50)

    def test_dip_restarts_and_gap_breaks(self):
        self.assertEqual(sa.first_sustained(self.steps(10), [.96]*4 + [.90] + [.96]*5), 250)
        self.assertIsNone(sa.first_sustained([0, 50, 100, 200, 250], [.96]*5))
        self.assertEqual(sa.first_sustained(self.steps(5), [122/128]*5), 0)
        self.assertIsNone(sa.first_sustained(self.steps(5), [121/128]*5))


class Retention(unittest.TestCase):
    def test_status_at_the_horizon(self):
        fro = []
        for arm, S, v in (('keep', '', .9), ('final', '', .2)):
            for t in sa.FIXED:
                fro.append(dict(arm=arm, S=S, seed=51200, step=t, **{m: v for m in ('held_L3', 'held_notN', 'L1', 'L2', 'L4', 'L5', 'L6')}))
        for t in sa.FIXED[1:]:
            fro.append(dict(arm='drop', S=10000, seed=51200, step=t, held_L3=.85, held_notN=.851, L1=.95, L2=.9, L4=.8, L5=.9, L6=.95))
        out = {(r['measure'], r['step']): r for r in sa.retention(fro)}
        self.assertEqual(len(out), 7*5)
        self.assertEqual(out[('held_L3', 50000)]['status'], 'degraded')   # d_keep = -0.05 exactly
        self.assertEqual(out[('held_notN', 50000)]['status'], 'retained')
        self.assertEqual(out[('L1', 50000)]['status'], 'above keep')   # +0.05 up to float rounding
        self.assertEqual(out[('L4', 20000)]['status'], '')
        self.assertAlmostEqual(out[('L6', 30000)]['d_final'], .75)
        self.assertAlmostEqual(out[('L4', 50000)]['d_branch'], -.1)


class Inventory(unittest.TestCase):
    def test_reused_checkpoints_exist_and_branches_start_from_the_switch_state(self):
        for seed in sa.SEEDS:
            for arm in ('keep', 'final'):
                for t in sa.FIXED:
                    self.assertTrue(sa.ckpt_path(arm, seed, t).exists(), (arm, seed, t))
            for S in sa.SWITCH:
                self.assertEqual(sa.ckpt_path('drop', seed, S, S), sa.ckpt_path('keep', seed, S))
                self.assertEqual(sa.ckpt_path('taper', seed, 20000, S), sa.RUNS/sa.run_id('taper', S, seed)/'ckpt-20000.npz')
        self.assertEqual(sa.ckpt_steps('drop', 1000), (1000, 3000, 10000, 20000, 30000, 40000, 50000))
        self.assertEqual(sa.ckpt_steps('taper', 10000), (10000, 12000, 20000, 30000, 40000, 50000))


class Frozen(unittest.TestCase):
    def test_length_3_equals_the_training_forward_and_the_logged_row(self):
        p = sa.nets.load(sa.ckpt_path('keep', 51201, 10000))[0]
        H, logits = sa.fwd(p, sa.HELD['ops'], sa.HELD['s'])
        H0, logits0 = sa.c2rnn.forward(p, sa.HELD['ops'], s=sa.HELD['s'])
        for a, b in zip(logits, logits0):
            np.testing.assert_array_equal(a, b)
        e = sa.frozen(p)
        row = {r['step']: r for r in sa.r1_rows('inter', 51201)}[10000]
        self.assertEqual(e['held_final_pred'], row['held_final_pred'])
        self.assertEqual(e['held_L3'], row['held']['final']['acc'])

    def test_frozen_lengths_equal_a_plain_per_row_loop(self):
        p = sa.nets.load(sa.ckpt_path('keep', 51201, 10000))[0]
        e = sa.frozen(p)
        rng = np.random.default_rng(72000)
        relu = lambda x: np.maximum(x, 0)
        for L in (1, 2, 4, 5, 6):
            d = sa.length_rows(L)
            correct = 0
            for i in range(len(d['s'])):
                h = relu(np.eye(8)[d['s'][i]] @ p['Ws'] + p['bs'])
                for op in d['ops'][i]:
                    h = relu(h @ p['A'] + np.eye(4)[op] @ p['B'] + p['bh'])
                correct += int(np.argmax(h @ p['V'] + p['bv']) == d['states'][i, -1])
            self.assertAlmostEqual(e[f'L{L}'], correct/len(d['s']), places=12)
            i = int(rng.integers(len(d['s'])))   # the answer itself, from the permutations
            x = d['s'][i]
            for op in d['ops'][i]:
                x = sa._T['P'][op, x]
            self.assertEqual(x, d['states'][i, -1])

    def test_subsets_and_program_counts(self):
        self.assertEqual((int(sa.NBR.sum()), int((~sa.NBR).sum())), (86, 42))
        self.assertEqual([len(sa.length_rows(L)['s']) for L in (1, 2, 3)], [32, 128, 512])
        truth = ''.join(map(str, sa.HELD['states'][:, -1]))
        self.assertEqual(sa.subsets(truth), dict(N=1., notN=1.))


if __name__ == '__main__':
    unittest.main()
