"""Checks of the c3 shared code (c3train, c3correct). Run in its own process:
uv run --locked python -m unittest -v research/composition-c3/common/test_c3.py

c2train.py and c3train.py are run as subprocesses; this process imports only uniquely named
modules (c2rnn, c2task, c3correct, c3train, nets).
"""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np

HERE = Path(__file__).resolve().parent
C2 = HERE.parents[1]/'composition-c2'/'common'
sys.path[:0] = [str(HERE), str(C2), str(HERE.parents[1]/'regimes'/'common')]
import c2rnn  # noqa: E402
import c2task  # noqa: E402
import c3correct  # noqa: E402
import c3train  # noqa: E402
import nets  # noqa: E402

ROOT = HERE.parents[2]
REC = ROOT/'research'/'composition-c2'/'recovered'
R1_RUNS = ROOT/'research'/'regimes'/'composition'/'runs'
KEYS = ('step', 'train', 'held', 'held_final_pred', 'held_subset')
SCHEMES = [(a, b) for a in (True, False) for b in (True, False)]


def lines(path, keys=KEYS):
    return [{k: r[k] for k in keys if k in r} for r in map(json.loads, Path(path).read_text().splitlines())]


def run(script, *argv, check=True):
    return subprocess.run([sys.executable, str(script), *map(str, argv)], cwd=ROOT, check=check, capture_output=True, text=True)


def ckpt_equal(a, b):
    (pa, sa, _), (pb, sb, _) = nets.load(a), nets.load(b)
    return (sa['t'] == sb['t'] and pa.keys() == pb.keys()
            and all(np.array_equal(x[k], y[k]) for x, y in ((pa, pb), (sa['m'], sb['m']), (sa['v'], sb['v'])) for k in x))


class Schedule(unittest.TestCase):
    def test_boundaries(self):
        keep = dict(before=[1., 1., 1.], switch_step=None, after=None, taper=1)
        drop = dict(keep, switch_step=1000, after=[0., 0., 1.])
        taper = dict(drop, taper=4)
        self.assertEqual(c3train.weights_at(10**6, keep), [1., 1., 1.])
        self.assertEqual(c3train.weights_at(999, drop), [1., 1., 1.])
        self.assertEqual(c3train.weights_at(1000, drop), [0., 0., 1.])
        got = [c3train.weights_at(t, taper) for t in range(998, 1006)]
        self.assertEqual(got, [[1., 1., 1.]]*2 + [[.75, .75, 1.], [.5, .5, 1.], [.25, .25, 1.]] + [[0., 0., 1.]]*3)
        self.assertTrue(all(w[2] == 1. for w in got))  # the final coefficient stays exactly 1
        self.assertEqual(c3train.weights_at(1000, dict(drop, taper=1)), c3train.weights_at(1000, drop))
        ex = c3train.exposure(384, taper, 0, 1010)
        self.assertEqual(ex['updates_with_weight'], [1003, 1003, 1010])
        self.assertEqual(ex['weighted_row_presentations'], [384*(1000 + 1.5)]*2 + [384*1010.])

    def test_loss_gradient_is_linear_in_the_weights(self):
        T = c2task.load(); d = c2task.rows(T['P'], T['train']); y = [d['states'][:, t] for t in range(3)]
        p = c2rnn.init(128, 51200)
        unit = [c2rnn.loss_grad(p, d['s'], d['ops'], y, e)[2] for e in np.eye(3).tolist()]
        for w in ([.75, .75, 1.], [0., 0., 1.], [.25, .5, 1.]):
            total, per_step, g = c2rnn.loss_grad(p, d['s'], d['ops'], y, w)
            self.assertAlmostEqual(total, sum(wk*c for wk, c in zip(w, per_step) if c is not None), places=12)
            for k in g:
                np.testing.assert_allclose(g[k], sum(wk*u[k] for wk, u in zip(w, unit)), rtol=0, atol=1e-13)


class Recipe(unittest.TestCase):
    def test_r1_and_c2_configs(self):
        r, s = c3train.recipe_of(json.loads((R1_RUNS/'r1-comp-rnn-final-s51200-a1'/'config.json').read_text()))
        self.assertEqual(r, dict(task_id='ops51000-ls51100', transition='tied', readout='tied', hidden=128, lr=1e-3, wd=.1, adam=[.9, .999, 1e-8]))
        self.assertEqual(s['before'], [0., 0., 1.])
        r, s = c3train.recipe_of(json.loads((ROOT/'research'/'regimes'/'review'/'runs'/'r1-rev-comp-rnn-untied-s51200-a1'/'config.json').read_text()))
        self.assertEqual((r['readout'], s['before']), ('untied', [1., 1., 1.]))
        r, s = c3train.recipe_of(json.loads((ROOT/'research'/'composition-c2'/'coordinator'/'runs'/'c2-co-verify-inter-s51200-a1'/'config.json').read_text()))
        self.assertEqual((r['task_id'], r['readout'], s['before']), ('ops51000-ls51100', 'tied', [1., 1., 1.]))


class Training(unittest.TestCase):
    """Short runs through the command line."""
    def test_no_change_mode_is_c2train(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp); common = ['--init-seed', 51200, '--updates', 200, '--diag', '--checkpoint-steps', 0, 100]
            for w in ('1 1 1', '0 0 1'):
                run(C2/'c2train.py', '--out', tmp/f'c2-{w[0]}', '--step-weights', *w.split(), *common)
                run(HERE/'c3train.py', '--out', tmp/f'c3-{w[0]}', '--step-weights', *w.split(), *common)
                self.assertEqual(lines(tmp/f'c3-{w[0]}'/'metrics.jsonl', KEYS + ('diag',)), lines(tmp/f'c2-{w[0]}'/'metrics.jsonl', KEYS + ('diag',)))
                self.assertTrue(ckpt_equal(tmp/f'c2-{w[0]}'/'ckpt-200.npz', tmp/f'c3-{w[0]}'/'ckpt-200.npz'))
                self.assertEqual({tuple(r['step_weights']) for r in map(json.loads, (tmp/f'c3-{w[0]}'/'metrics.jsonl').read_text().splitlines())},
                                 {tuple(map(float, w.split()))})

    def test_drop_branch_is_final_only_from_the_same_state(self):
        """Switching at the checkpoint step equals c2train resumed with the final-only weights."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            run(HERE/'c3train.py', '--out', tmp/'src', '--init-seed', 51201, '--updates', 100, '--run-id', 'src')
            src = tmp/'src'/'ckpt-100.npz'; sha = nets.sha256_file(src)
            run(HERE/'c3train.py', '--out', tmp/'drop', '--init-seed', 51201, '--updates', 250, '--switch-step', 100,
                '--switch-weights', 0, 0, 1, '--resume', src, '--resume-sha256', sha, '--resume-config', tmp/'src'/'config.json')
            run(C2/'c2train.py', '--out', tmp/'c2', '--init-seed', 51201, '--updates', 250, '--step-weights', 0, 0, 1,
                '--resume', src, '--resume-sha256', sha)
            self.assertTrue(ckpt_equal(tmp/'drop'/'ckpt-250.npz', tmp/'c2'/'ckpt-250.npz'))
            self.assertEqual(lines(tmp/'drop'/'metrics.jsonl'), lines(tmp/'c2'/'metrics.jsonl'))
            cfg = json.loads((tmp/'drop'/'config.json').read_text())
            self.assertEqual(cfg['exposure']['labels_this_run']['updates_with_weight'], [0, 0, 150])
            self.assertEqual(cfg['exposure']['labels_from_init']['updates_with_weight'], [100, 100, 250])

    def test_scheduled_continuation_is_bitwise(self):
        """A resume in the middle of a taper continues the uninterrupted run exactly."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            sch = ['--switch-step', 100, '--switch-weights', 0, 0, 1, '--taper-updates', 80]
            common = ['--init-seed', 65010, '--hidden', 32, '--diag', *sch]
            run(HERE/'c3train.py', '--out', tmp/'whole', '--updates', 300, '--checkpoint-steps', 0, 150, *common)
            src = tmp/'whole'/'ckpt-150.npz'; sha = nets.sha256_file(src)
            run(HERE/'c3train.py', '--out', tmp/'cont', '--updates', 300, '--resume', src, '--resume-sha256', sha,
                '--resume-config', tmp/'whole'/'config.json', *common)
            self.assertTrue(ckpt_equal(tmp/'whole'/'ckpt-300.npz', tmp/'cont'/'ckpt-300.npz'))
            keys = KEYS + ('step_weights', 'diag')
            self.assertEqual(lines(tmp/'cont'/'metrics.jsonl', keys), [r for r in lines(tmp/'whole'/'metrics.jsonl', keys) if r['step'] >= 150])
            w = {r['step']: r['step_weights'] for r in lines(tmp/'whole'/'metrics.jsonl', KEYS + ('step_weights',))}
            self.assertEqual((w[50], w[100], w[150], w[200]), ([1.]*3, [1 - 1/80]*2 + [1.], [1 - 51/80]*2 + [1.], [0., 0., 1.]))
            # the resume is refused if the schedule before the checkpoint differs, or the recipe does
            for bad in (['--switch-step', 120, '--switch-weights', 0, 0, 1, '--taper-updates', 80], ['--wd', 0, *sch], ['--lr', 2e-3, *sch]):
                r = run(HERE/'c3train.py', '--out', tmp/f'bad{len(bad)}{bad[1]}', '--updates', 300, '--init-seed', 65010, '--hidden', 32,
                        '--resume', src, '--resume-sha256', sha, '--resume-config', tmp/'whole'/'config.json', *bad, check=False)
                self.assertNotEqual(r.returncode, 0); self.assertIn('recipe check failed', r.stderr)

    def test_recovered_r1_checkpoints(self):
        """r1's checkpoints resume against r1's configs and published manifests, and only under r1's recipe."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            src = REC/'r1-comp-rnn-inter-s51200-a1'/'ckpt-1000.npz'; sha = nets.sha256_file(src)
            cfg = R1_RUNS/'r1-comp-rnn-inter-s51200-a1'/'config.json'
            run(HERE/'c3train.py', '--out', tmp/'ok', '--init-seed', 51200, '--updates', 1100, '--switch-step', 1000,
                '--switch-weights', 0, 0, 1, '--taper-updates', 2000, '--resume', src, '--resume-sha256', sha, '--resume-config', cfg)
            source = json.loads((tmp/'ok'/'config.json').read_text())['resumed_from']['source']
            self.assertTrue(source['manifest_listed']); self.assertEqual(source['run_id'], 'r1-comp-rnn-inter-s51200-a1')
            published = lines(R1_RUNS/'r1-comp-rnn-inter-s51200-a1'/'metrics.jsonl', ('step', 'train', 'held', 'held_final_pred'))
            self.assertEqual(lines(tmp/'ok'/'metrics.jsonl', ('step', 'train', 'held', 'held_final_pred'))[:1], [r for r in published if r['step'] == 1000])
            for name, bad in (('weights', ['--step-weights', 0, 0, 1]), ('final-config', ['--resume-config', R1_RUNS/'r1-comp-rnn-final-s51200-a1'/'config.json'])):
                argv = ['--out', tmp/name, '--init-seed', 51200, '--updates', 1100, '--resume', src, '--resume-sha256', sha]
                if name == 'weights':
                    argv += ['--resume-config', cfg]
                r = run(HERE/'c3train.py', *argv, *bad, check=False)
                self.assertNotEqual(r.returncode, 0); self.assertIn('recipe check failed', r.stderr)


class Correction(unittest.TestCase):
    def test_identity_is_the_plain_forward_and_the_base_stays_frozen(self):
        r = np.random.default_rng(65011)
        P = c2task.load()['P']
        for L, progs in ((3, c2task.programs(3)), (5, c2task.programs(5)[::29])):
            d = c2task.rows(P, progs)
            for tt, tr in SCHEMES:
                p = c3correct.frozen({k: v + r.normal(0, .1, v.shape) for k, v in c2rnn.init(16, 65012, tt, tr).items()})
                before = c3correct.params_sha256(p)
                smap = None if (L == 3 or (tt and tr)) else [1, 2, 3, 3, 3]
                ref = c2rnn.forward(p, d['ops'], s=d['s'], step_map=c3correct.step_map_for(p, L, smap))
                for depth in range(1, L):
                    out = c3correct.corrected_forward(p, d['s'], d['ops'], depth, step_map=smap)
                    self.assertTrue(all(np.array_equal(a, b) for a, b in zip(out['H'] + out['logits'], ref[0] + ref[1])))
                    shifted = c3correct.corrected_forward(p, d['s'], d['ops'], depth, correct=lambda h: h[::-1].copy(), step_map=smap)
                    self.assertTrue(all(np.array_equal(a, b) for a, b in zip(shifted['logits'][:depth], ref[1][:depth])))
                    self.assertTrue(np.array_equal(shifted['H'][depth], ref[0][depth][::-1]))
                    self.assertFalse(np.array_equal(shifted['logits'][-1], ref[1][-1]))
                self.assertEqual(c3correct.params_sha256(p), before)
                with self.assertRaises(ValueError):
                    p['A' if tt else 'A1'][0, 0] = 0.
                if L > 3 and not (tt and tr):
                    with self.assertRaises(ValueError):
                        c3correct.corrected_forward(p, d['s'], d['ops'], 2)


if __name__ == '__main__':
    unittest.main()
