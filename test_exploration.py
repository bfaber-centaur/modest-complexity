"""Stage 0 code checks for the exploratory map (research/SWARM.md). These test the
implementation, not any scientific hypothesis."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
from experiment import adam, array_sha256, data, forward, init, lossgrad, objective_metrics
from instrumentation import THREAD_VARS

ROOT = Path(__file__).parent


class SplitTests(unittest.TestCase):
    def test_historical_data_unchanged(self):
        # Raw-byte hash of data(31, 1000) at e0a6e0f, before seeds were separated.
        legacy = hashlib.sha256()
        a, b, y, yr, masks = data(31, 1000)
        for x in (a, b, y, yr, *masks):
            legacy.update(np.ascontiguousarray(x).tobytes())
        self.assertEqual(legacy.hexdigest(), 'a3a8b1bc6047a70e10007587f017023740c18d219d417cea12b9ce270c3973e7')

    def test_control_seed_changes_only_shuffled_labels(self):
        base = data(31, 31000, 32000)
        other = data(31, 31000, 32001)
        legacy = data(31, 31000)
        for x, z, l in zip(base[:3], other[:3], legacy[:3]):
            np.testing.assert_array_equal(x, z); np.testing.assert_array_equal(x, l)
        for x, z in zip(base[4], other[4]):
            np.testing.assert_array_equal(x, z)
        self.assertFalse(np.array_equal(base[3], other[3]))
        self.assertFalse(np.array_equal(base[4][0], data(31, 31001, 32000)[4][0]))

    def test_train_fraction_split(self):
        a, b, y, yr, masks = data(31, 31000, 32000, train_fraction=.5)
        self.assertEqual(sorted(np.concatenate(masks).tolist()), list(range(31**2)))
        for ix, groups in zip(masks, (8, 2, 6)):
            self.assertEqual(set((a[ix]*31+b[ix]).tolist()), set((b[ix]*31+a[ix]).tolist()))
            unordered = {(min(i, j), max(i, j)) for i, j in zip(a[ix], b[ix])}
            self.assertEqual(len(unordered), 31*groups)
        plan = json.loads((ROOT/'research/plan-followups.json').read_text())
        for run in (r for r in plan['runs'] if 'train_fraction' in r):
            self.assertEqual(array_sha256(*data(31, 31000, 32000, run['train_fraction'])[:3],
                                          *data(31, 31000, 32000, run['train_fraction'])[4]), run['split_sha256'])

    def test_empty_split_is_rejected(self):
        # p = 7 has 4 groups per class; a share of .8 leaves no audit pairs.
        with self.assertRaises(ValueError):
            data(7, 8, train_fraction=.8)

    def test_first_map_split(self):
        plan = json.loads((ROOT/'research/plan.json').read_text())
        c = plan['common']
        a, b, y, yr, masks = data(c['p'], c['data_seed'], c['control_seed'])
        self.assertEqual(array_sha256(a, b, y, *masks), c['split_sha256'])
        self.assertEqual(array_sha256(yr), plan['targets']['random'])
        self.assertEqual(sorted(np.concatenate(masks).tolist()), list(range(c['p']**2)))
        for ix in masks:
            # No (a,b)/(b,a) pair crosses a split; class counts survive shuffling.
            self.assertEqual(set((a[ix]*31+b[ix]).tolist()), set((b[ix]*31+a[ix]).tolist()))
            np.testing.assert_array_equal(np.bincount(y[ix], minlength=31), np.bincount(yr[ix], minlength=31))
        np.testing.assert_array_equal(y, (a+b) % 31)
        self.assertEqual(len({r['run_id'] for r in plan['runs']}), len(plan['runs']))
        with np.load(ROOT/'research/data/e1-split.npz') as frozen:
            self.assertEqual(array_sha256(frozen['a'], frozen['b'], frozen['yadd'], frozen['train'],
                                          frozen['cal'], frozen['audit']), c['split_sha256'])
            np.testing.assert_array_equal(frozen['yshuffled'], yr)


class OptimizerTests(unittest.TestCase):
    def test_adamw_matches_independent_formulation_over_many_steps(self):
        # Reference written in PyTorch's AdamW order, element by element:
        # decay first, then bias-corrected step size and denominator.
        p, h, lr, wd, b1, b2, eps = 7, 5, .01, 1., .9, .999, 1e-8
        a, b = np.indices((p, p)).reshape(2, -1); y = (a+b) % p
        w = init(p, h, 3); ref = [x.copy() for x in w]
        m = [np.zeros_like(x) for x in w]; v = [x.copy() for x in m]
        rm = [x.ravel().tolist() for x in m]; rv = [x.ravel().tolist() for x in v]
        for t in range(1, 26):
            _, g = lossgrad(w, a, b, y); adam(w, g, m, v, t, lr, wd)
            _, rg = lossgrad(ref, a, b, y)
            for i in range(3):
                flat = ref[i].ravel()
                for k, gk in enumerate(rg[i].ravel()):
                    flat[k] *= 1 - lr*wd
                    rm[i][k] = b1*rm[i][k] + (1-b1)*gk
                    rv[i][k] = b2*rv[i][k] + (1-b2)*gk*gk
                    denom = (rv[i][k]**.5)/((1-b2**t)**.5) + eps
                    flat[k] -= (lr/(1-b1**t))*rm[i][k]/denom
            for x, r in zip(w, ref):
                np.testing.assert_allclose(x, r, rtol=1e-11, atol=1e-14)
        self.assertGreater(max(np.abs(x-init(p, h, 3)[i]).max() for i, x in enumerate(w)), 1e-2)


class ShortRunTests(unittest.TestCase):
    """A representative short job, including checkpoint and metrics I/O."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        env = {**os.environ, **{k: '1' for k in THREAD_VARS}}
        cls.runs = {}
        for name, init_seed in (('a', 30000), ('b', 30001)):
            out = Path(cls.tmp.name)/name
            subprocess.run([sys.executable, str(ROOT/'experiment.py'), '--out', str(out), '--hidden', '16',
                            '--steps', '300', '--log-every', '100', '--checkpoint-every', '100',
                            '--diagnostics', 'none', '--data-seed', '31000', '--init-seed', str(init_seed),
                            '--control-seed', '32000', '--purpose', 'test'],
                           env=env, check=True, capture_output=True)
            cls.runs[name] = out

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def load(self, name, step):
        with np.load(self.runs[name]/f'checkpoint_{step:06d}.npz') as z:
            return {k: z[k] for k in z.files}

    def test_status_files_and_split_identity(self):
        cfg = {k: json.loads((v/'config.json').read_text()) for k, v in self.runs.items()}
        runtime = json.loads((self.runs['a']/'runtime.json').read_text())
        self.assertEqual((runtime['status'], runtime['completed_steps']), ('complete', 300))
        self.assertEqual(cfg['a']['data_hashes'], cfg['b']['data_hashes'])
        self.assertEqual(cfg['a']['seeds']['init_seed'], 30000)
        self.assertFalse(np.array_equal(self.load('a', 0)['U'], self.load('b', 0)['U']))
        with np.load(self.runs['a']/'data.npz') as d:
            digest = array_sha256(d['a'], d['b'], d['yadd'], d['train'], d['cal'], d['audit'])
        self.assertEqual(digest, cfg['a']['data_hashes']['split_sha256'])
        steps = sorted(int(p.stem.split('_')[1]) for p in self.runs['a'].glob('checkpoint_*.npz'))
        self.assertEqual(steps, [0, 100, 200, 300])

    def test_checkpoints_reproduce_logged_metrics(self):
        rows = {r['step']: r for r in map(json.loads, (self.runs['a']/'metrics.jsonl').read_text().splitlines())}
        with np.load(self.runs['a']/'data.npz') as d:
            a, b, y = d['a'], d['b'], d['y']; masks = [d['train'], d['cal'], d['audit']]
        for step in (0, 100, 200, 300):
            ck = self.load('a', step)
            self.assertEqual(int(ck['step']), step)
            logits = forward([ck['U'], ck['V'], ck['W']], a, b)[0]
            for name, ix in zip(('train', 'cal', 'audit'), masks):
                self.assertEqual(rows[step][name], float(np.mean(logits[ix].argmax(1) == y[ix])))
                self.assertEqual(rows[step]['objective'][name], objective_metrics(logits, y, ix))

    def test_checkpoint_holds_complete_optimizer_state(self):
        # Continuing from step 100's weights and moments must reproduce step 200
        # exactly. This checks checkpoint contents; it is not a resume command.
        ck = self.load('a', 100)
        with np.load(self.runs['a']/'data.npz') as d:
            tr = d['train']; a, b, y = d['a'][tr], d['b'][tr], d['y'][tr]
        w = [ck['U'], ck['V'], ck['W']]; m = [ck['mU'], ck['mV'], ck['mW']]; v = [ck['vU'], ck['vV'], ck['vW']]
        for t in range(101, 201):
            adam(w, lossgrad(w, a, b, y)[1], m, v, t, .001, 1.)
        later = self.load('a', 200)
        for x, key in zip(w+m+v, ['U', 'V', 'W', 'mU', 'mV', 'mW', 'vU', 'vV', 'vW']):
            np.testing.assert_array_equal(x, later[key])

    def continue_from(self, name, weights, steps, expect=None):
        out = Path(self.tmp.name)/name
        cmd = [sys.executable, str(ROOT/'experiment.py'), '--out', str(out), '--hidden', '16', '--steps', str(steps),
               '--log-every', '10', '--checkpoint-every', '50', '--diagnostics', 'none', '--data-seed', '31000',
               '--init-seed', '30000', '--control-seed', '32000', '--purpose', 'test', '--init-weights', str(weights),
               '--init-age', '100', '--checkpoint-steps', '30']
        if expect:
            cmd += ['--expect-init-sha256', expect]
        env = {**os.environ, **{k: '1' for k in THREAD_VARS}}
        return out, subprocess.run(cmd, env=env, capture_output=True)

    def test_continuation_from_given_weights_restarts_the_optimizer(self):
        # An edited start is just weights: the run evaluates them before any update (step 0)
        # and then trains with zero Adam moments and the update count restarting at 1.
        ck = self.load('a', 100)
        start = Path(self.tmp.name)/'start.npz'
        np.savez(start, U=ck['U'], V=ck['V'], W=ck['W'])
        digest = hashlib.sha256(start.read_bytes()).hexdigest()
        out, proc = self.continue_from('cont', start, 100, digest)
        self.assertEqual(proc.returncode, 0, proc.stderr.decode()[-500:])
        cfg = json.loads((out/'config.json').read_text())
        self.assertEqual((cfg['init_weights_sha256'], cfg['init_age']), (digest, 100))
        self.assertTrue((out/'checkpoint_000030.npz').exists())
        with np.load(out/'checkpoint_000000.npz') as z0:
            for k in 'UVW':
                np.testing.assert_array_equal(z0[k], ck[k])
            for k in ('mU', 'mV', 'mW', 'vU', 'vV', 'vW'):
                self.assertFalse(np.any(z0[k]))
        rows = [json.loads(r) for r in (out/'metrics.jsonl').read_text().splitlines()]
        with np.load(out/'data.npz') as d:
            a, b, y, tr = d['a'], d['b'], d['y'], d['train']
        logits = forward([ck['U'], ck['V'], ck['W']], a, b)[0]
        self.assertEqual(rows[0]['step'], 0)
        self.assertEqual(rows[0]['train'], float(np.mean(logits[tr].argmax(1) == y[tr])))
        w = [ck['U'].copy(), ck['V'].copy(), ck['W'].copy()]
        m = [np.zeros_like(x) for x in w]; v = [np.zeros_like(x) for x in w]
        for t in range(1, 101):
            adam(w, lossgrad(w, a[tr], b[tr], y[tr])[1], m, v, t, .001, 1.)
        with np.load(out/'checkpoint_000100.npz') as z:
            for x, k in zip(w, 'UVW'):
                np.testing.assert_array_equal(x, z[k])
            # Not the uninterrupted trajectory, which kept its moments and update count.
            self.assertFalse(np.array_equal(z['U'], self.load('a', 200)['U']))

    def test_continuation_refuses_a_different_weights_file(self):
        ck = self.load('a', 100)
        start = Path(self.tmp.name)/'start-other.npz'
        np.savez(start, U=ck['U'], V=ck['V'], W=ck['W'])
        out, proc = self.continue_from('cont-bad', start, 10, '0'*64)
        self.assertNotEqual(proc.returncode, 0)
        self.assertEqual(json.loads((out/'runtime.json').read_text())['status'], 'failed')
        self.assertFalse((out/'checkpoint_000000.npz').exists())

    def test_resume_continues_training_exactly(self):
        # Resuming run 'a' at update 100 must reproduce its uninterrupted updates 200 and 300:
        # weights, moments and every logged metric apart from wall-clock fields.
        start = self.runs['a']/'checkpoint_000100.npz'
        digest = hashlib.sha256(start.read_bytes()).hexdigest()
        out = Path(self.tmp.name)/'resumed'
        env = {**os.environ, **{k: '1' for k in THREAD_VARS}}
        proc = subprocess.run([sys.executable, str(ROOT/'experiment.py'), '--out', str(out), '--hidden', '16',
                               '--steps', '300', '--log-every', '100', '--checkpoint-every', '100',
                               '--diagnostics', 'none', '--data-seed', '31000', '--init-seed', '30000',
                               '--control-seed', '32000', '--purpose', 'test', '--resume', str(start),
                               '--expect-init-sha256', digest], env=env, capture_output=True)
        self.assertEqual(proc.returncode, 0, proc.stderr.decode()[-500:])
        self.assertEqual(json.loads((out/'config.json').read_text())['resume_sha256'], digest)
        for step in (100, 200, 300):
            with np.load(out/f'checkpoint_{step:06d}.npz') as z:
                for k, x in self.load('a', step).items():
                    np.testing.assert_array_equal(z[k], x)
        clock = ('seconds', 'elapsed_after_diagnostics')
        rows = [{k: v for k, v in json.loads(r).items() if k not in clock} for r in
                (out/'metrics.jsonl').read_text().splitlines()]
        original = {r['step']: r for r in ({k: v for k, v in json.loads(r).items() if k not in clock} for r in
                    (self.runs['a']/'metrics.jsonl').read_text().splitlines())}
        self.assertEqual([r['step'] for r in rows], [100, 200, 300])
        self.assertIsNone(rows[0].pop('optimizer_update'))  # no update is taken at the resumed step
        self.assertEqual(rows[0], {k: v for k, v in original[100].items() if k != 'optimizer_update'})
        self.assertEqual(rows[1:], [original[200], original[300]])


if __name__ == '__main__':
    unittest.main()
