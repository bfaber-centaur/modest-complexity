"""Checks of the c2 shared code (c2task, c2rnn, c2train). Run in its own process:
uv run --locked python -m unittest -v research/composition-c2/common/test_c2.py

r1's train.py is only ever run as a subprocess, so its `task`/`train` module names never meet
this process's imports.
"""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1]/'regimes'/'common')]
import c2rnn  # noqa: E402
import c2task  # noqa: E402
import nets  # noqa: E402

ROOT = HERE.parents[2]
R1 = ROOT/'research'/'regimes'
R1_KEYS = ('step', 'train', 'held', 'held_final_pred')
SCHEMES = [(a, b) for a in (True, False) for b in (True, False)]


def lines(path, keys=R1_KEYS):
    return [{k: r[k] for k in keys} for r in map(json.loads, Path(path).read_text().splitlines())]


def run(*argv):
    subprocess.run([sys.executable, *map(str, argv)], cwd=ROOT, check=True, capture_output=True)


def ckpt_equal(a, b):
    (pa, sa, _), (pb, sb, _) = nets.load(a), nets.load(b)
    return (sa['t'] == sb['t'] and pa.keys() == pb.keys()
            and all(np.array_equal(x[k], y[k]) for x, y in ((pa, pb), (sa['m'], sb['m']), (sa['v'], sb['v'])) for k in x))


class Task(unittest.TestCase):
    def test_rows_and_ids(self):
        T = c2task.load()
        self.assertEqual(T['id'], 'ops51000-ls51100')
        for progs in (T['train'], T['held']):
            ours, theirs = c2task.rows(T['P'], progs), c2task.from_r1(c2task.r1.rows(T['P'], progs))
            for k in ours:
                np.testing.assert_array_equal(ours[k], theirs[k])
        nbr, all_train = c2task.neighbour_answerable(T['P'], T['train'], T['held'])
        self.assertEqual(((~nbr).sum(), len(nbr), all_train), (42, 128, True))

    def test_longer_rows_against_plain_oracle(self):
        P = c2task.load()['P']
        d = c2task.rows(P, c2task.programs(5))
        self.assertEqual(len(d['s']), 8*4**5)
        for i in range(0, len(d['s']), 97):
            x = int(d['s'][i])
            for t, k in enumerate(d['ops'][i]):
                x = [int(v) for v in P[k]][x]
                self.assertEqual(x, d['states'][i, t])


class Model(unittest.TestCase):
    def setUp(self):
        T = c2task.load(); self.d = c2task.rows(T['P'], T['train'])

    def test_schemes_start_as_one_function_with_independent_copies(self):
        ref = c2rnn.forward(c2rnn.init(128, 51200), self.d['ops'], s=self.d['s'])
        for tt, tr in SCHEMES:
            p = c2rnn.init(128, 51200, tt, tr)
            H, logits = c2rnn.forward(p, self.d['ops'], s=self.d['s'])
            self.assertTrue(all(np.array_equal(x, y) for x, y in zip(H + logits, ref[0] + ref[1])))
            copies = [v for k, v in p.items() if k[-1].isdigit()]
            self.assertEqual(len(copies), 9*(not tt) + 6*(not tr))
            self.assertFalse(any(np.shares_memory(a, b) for i, a in enumerate(copies) for b in copies[i+1:]))

    def test_untied_gradients_split_the_tied_gradient_at_init(self):
        y = [self.d['states'][:, t] for t in range(3)]
        g0 = c2rnn.loss_grad(c2rnn.init(128, 51200), self.d['s'], self.d['ops'], y, [1, 1, 1])[2]
        for tt, tr in SCHEMES[1:]:
            g = c2rnn.loss_grad(c2rnn.init(128, 51200, tt, tr), self.d['s'], self.d['ops'], y, [1, 1, 1])[2]
            for k in g0:
                parts = [g[k]] if k in g else [g[f'{k}{i}'] for i in (1, 2, 3)]
                np.testing.assert_allclose(sum(parts), g0[k], rtol=0, atol=1e-13)
                if k not in g:  # each copy gets its own step's term, so the copies differ
                    self.assertFalse(np.array_equal(parts[0], parts[2]))

    def test_finite_differences_all_schemes(self):
        d = {k: v[::9] for k, v in self.d.items()}; y = [d['states'][:, t] for t in range(3)]
        for (tt, tr), w in zip(SCHEMES, ([1, 1, 1], [.5, 0, 2], [0, 0, 1], [2, 1, .25])):
            r = np.random.default_rng(65000)
            p = {k: v + r.normal(0, .3, v.shape) for k, v in c2rnn.init(6, 65001, tt, tr).items()}
            g = c2rnn.loss_grad(p, d['s'], d['ops'], y, w)[2]
            worst = 0.
            for k, v in p.items():
                for i in np.ndindex(v.shape):
                    old = v[i]; v[i] = old + 1e-6; up = c2rnn.loss_grad(p, d['s'], d['ops'], y, w)[0]
                    v[i] = old - 1e-6; down = c2rnn.loss_grad(p, d['s'], d['ops'], y, w)[0]; v[i] = old
                    worst = max(worst, abs((up - down)/2e-6 - g[k][i]))
            self.assertLess(worst, 1e-7, (tt, tr, w))

    def test_any_length_and_continuation(self):
        r = np.random.default_rng(65002)
        P = c2task.load()['P']; d = c2task.rows(P, c2task.programs(5)[::37])
        for tt, tr in SCHEMES:
            p = {k: v + r.normal(0, .1, v.shape) for k, v in c2rnn.init(16, 65003, tt, tr).items()}
            smap = [1, 2, 3, 3, 3]
            if not (tt and tr):
                with self.assertRaises(ValueError):
                    c2rnn.forward(p, d['ops'], s=d['s'])
            H, logits = c2rnn.forward(p, d['ops'], s=d['s'], step_map=smap)
            h = np.maximum(np.eye(8)[d['s']] @ p['Ws'] + p['bs'], 0)  # plain loop, written separately
            for t, k in enumerate(smap):
                A, B, bh, V, bv = (n if n in p else f'{n}{k}' for n in ('A', 'B', 'bh', 'V', 'bv'))
                h = np.maximum(h @ p[A] + np.eye(4)[d['ops'][:, t]] @ p[B] + p[bh], 0)
                np.testing.assert_array_equal(h, H[t+1]); np.testing.assert_array_equal(h @ p[V] + p[bv], logits[t])
            H2, logits2 = c2rnn.forward(p, d['ops'][:, 2:], h0=H[2], step_map=smap[2:])
            self.assertTrue(all(np.array_equal(a, b) for a, b in zip(H2 + logits2, H[2:] + logits[2:])))
            if not (tt and tr):
                with self.assertRaises(ValueError):
                    c2rnn.forward(p, d['ops'][:, 2:], h0=H[2])


class Probe(unittest.TestCase):
    def test_probe_and_table_shape(self):
        import c2probe
        y = np.arange(64) % 8
        self.assertEqual(c2probe.accuracy(c2probe.fit_probe(np.eye(8)[y], y, 65005), np.eye(8)[y], y), 1.0)
        T = c2task.load()
        out = c2probe.decode_table(nets.load(HERE.parent/'recovered'/'r1-comp-rnn-inter-s51200-a1'/'ckpt-10000.npz')[0], T, 65006)
        self.assertEqual(np.array(out['table']).shape, (4, 4))
        self.assertTrue(all(0 <= x <= 1 for row in out['table'] for x in row))


class Training(unittest.TestCase):
    """Short runs through the command line: r1 arithmetic in the no-change mode, the reviewer's
    untied-readout arm, and bitwise continuation."""
    def test_no_change_mode_reproduces_r1(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            for cond, w in (('inter', '1 1 1'), ('final', '0 0 1')):
                run(R1/'composition'/'train.py', '--out', tmp/f'r1-{cond}', '--model', 'rnn', '--condition', cond,
                    '--init-seed', 51200, '--updates', 200)
                run(HERE/'c2train.py', '--out', tmp/f'c2-{cond}', '--init-seed', 51200, '--updates', 200, '--step-weights', *w.split())
                ours = lines(tmp/f'c2-{cond}'/'metrics.jsonl')
                self.assertEqual(ours, lines(tmp/f'r1-{cond}'/'metrics.jsonl'))
                published = lines(R1/'composition'/'runs'/f'r1-comp-rnn-{cond}-s51200-a1'/'metrics.jsonl')
                self.assertEqual(ours, published[:len(ours)])
                p1, s1, _ = nets.load(tmp/f'r1-{cond}'/'ckpt-200.npz'); p2, s2, _ = nets.load(tmp/f'c2-{cond}'/'ckpt-200.npz')
                self.assertTrue(all(np.array_equal(p1[k], p2[k]) and np.array_equal(s1['v'][k], s2['v'][k]) for k in p1))

    def test_untied_readout_reproduces_reviewer_arm(self):
        with tempfile.TemporaryDirectory() as tmp:
            run(HERE/'c2train.py', '--out', Path(tmp)/'u', '--init-seed', 51200, '--updates', 200, '--readout', 'untied')
            published = lines(R1/'review'/'runs'/'r1-rev-comp-rnn-untied-s51200-a1'/'metrics.jsonl')
            ours = lines(Path(tmp)/'u'/'metrics.jsonl')
            self.assertEqual(ours, published[:len(ours)])

    def test_continuation_is_bitwise(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp); common = ['--init-seed', 65004, '--transition', 'untied', '--readout', 'untied',
                                       '--step-weights', .5, 1, 2, '--hidden', 32, '--diag']
            run(HERE/'c2train.py', '--out', tmp/'whole', '--updates', 300, '--checkpoint-steps', 0, 150, *common)
            src = tmp/'whole'/'ckpt-150.npz'
            run(HERE/'c2train.py', '--out', tmp/'cont', '--updates', 300, '--resume', src, '--resume-sha256', nets.sha256_file(src), *common)
            self.assertTrue(ckpt_equal(tmp/'whole'/'ckpt-300.npz', tmp/'cont'/'ckpt-300.npz'))
            whole = [r for r in lines(tmp/'whole'/'metrics.jsonl', R1_KEYS + ('held_subset',)) if r['step'] >= 150]
            self.assertEqual(lines(tmp/'cont'/'metrics.jsonl', R1_KEYS + ('held_subset',)), whole)
            diag = [json.loads(x)['diag'] for x in (tmp/'cont'/'metrics.jsonl').read_text().splitlines()[:-1]]
            self.assertEqual(diag, [json.loads(x)['diag'] for x in (tmp/'whole'/'metrics.jsonl').read_text().splitlines()[3:-1]])
            with self.assertRaises(subprocess.CalledProcessError):
                run(HERE/'c2train.py', '--out', tmp/'bad', '--updates', 300, '--resume', src, '--resume-sha256', '0'*64, *common)


if __name__ == '__main__':
    unittest.main()
