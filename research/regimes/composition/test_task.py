"""Independent checks of the R1 oracle, split and encodings, written differently
from task.py (permutation matrices, array counts, brute decoding), plus a
smoke test of train.py's run record. Code checks only, no scientific claim.

uv run --locked python -m unittest -v research/regimes/composition/test_task.py
"""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent/'common'), str(HERE)]
import nets  # noqa: E402
import task  # noqa: E402


class TaskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.P, _ = task.draw_ops()
        cls.L, cls.train, cls.held = task.split()

    def test_labels_match_matrix_products(self):
        # M[k] @ e_s = e_{P[k][s]}; a program applies M[a] first, so its matrix is M[c] @ M[b] @ M[a].
        M = np.zeros((task.N_OP, task.N_SYM, task.N_SYM))
        for k in range(task.N_OP):
            M[k][self.P[k], np.arange(task.N_SYM)] = 1
        for programs in (self.train, self.held):
            d = task.rows(self.P, programs)
            for i in range(len(d['s'])):
                e = np.eye(task.N_SYM)[d['s'][i]]
                s1 = M[d['a'][i]] @ e; s2 = M[d['b'][i]] @ s1; s3 = M[d['c'][i]] @ s2
                self.assertEqual([d['step1'][i], d['step2'][i], d['final'][i]], [s1.argmax(), s2.argmax(), s3.argmax()])

    def test_ops_are_valid_permutations_and_pass_the_rule(self):
        for p in self.P:
            self.assertEqual(sorted(p.tolist()), list(range(task.N_SYM)))
            self.assertFalse((p == np.arange(task.N_SYM)).all())
        self.assertEqual(len(np.unique(self.P, axis=0)), task.N_OP)

    def test_latin_square_and_split_counts(self):
        self.assertEqual(len(task.latin_squares()), 576)
        for line in (*self.L, *self.L.T):
            self.assertEqual(sorted(line.tolist()), list(range(task.N_OP)))
        H = np.zeros((task.N_OP,)*3, int)
        for q in self.held:
            H[q] += 1
        self.assertEqual(H.sum(), 16)
        # Exactly one held-out program per value of any two slots, so training keeps 3 of 4.
        for axis in range(3):
            self.assertTrue((H.sum(axis) == 1).all())
        T = np.zeros_like(H)
        for q in self.train:
            T[q] += 1
        self.assertTrue(((T + H) == 1).all())
        for axis in range(3):
            self.assertTrue((T.sum(axis) == 3).all())
            self.assertTrue((T.sum(axis=tuple(i for i in range(3) if i != axis)) == 12).all())

    def test_no_row_leakage_and_encoding_round_trip(self):
        tr, he = task.rows(self.P, self.train), task.rows(self.P, self.held)
        Xt, Xh = task.encode(tr), task.encode(he)
        self.assertEqual((len(Xt), len(Xh)), (384, 128))
        self.assertTrue((Xt.sum(1) == 4).all() and (Xh.sum(1) == 4).all())
        cuts = np.cumsum([0, task.N_SYM] + [task.N_OP]*3)
        for X, d in ((Xt, tr), (Xh, he)):
            decoded = [X[:, cuts[i]:cuts[i+1]].argmax(1) for i in range(4)]
            for col, name in zip(decoded, 'sabc'):
                self.assertTrue((col == d[name]).all())
        self.assertFalse({r.tobytes() for r in Xt} & {r.tobytes() for r in Xh})

    def test_heuristic_baseline_independently(self):
        he = task.rows(self.P, self.held)
        drop3 = np.mean([self.P[b][self.P[a][s]] == y for s, a, b, y in zip(he['s'], he['a'], he['b'], he['final'])])
        checks = task.checks(self.P, self.train, self.held)
        self.assertAlmostEqual(checks['heldout_baselines']['drop op 3: P[b](P[a](s))'], drop3)

    def test_rnn_finite_difference_gradients(self):
        import train
        d = {k: v[::7] for k, v in task.rows(self.P, self.train).items()}  # 55 rows
        r = np.random.default_rng(5)
        p = train.rnn_init(6, 3)
        for k in ('bs', 'bh', 'bv'):  # off the ReLU kinks that zero biases put one-hot inputs on
            p[k] = r.normal(0, .3, p[k].shape)
        for steps in (('final',), ('step1', 'step2', 'final')):
            targets = {h: d[h] for h in steps}
            g = train.rnn_loss_grad(p, d, targets)[1]
            for k, v in p.items():
                for idx in list(np.ndindex(v.shape))[:12]:
                    old = v[idx]; v[idx] = old + 1e-6
                    up = train.rnn_loss_grad(p, d, targets)[0]; v[idx] = old - 1e-6
                    down = train.rnn_loss_grad(p, d, targets)[0]; v[idx] = old
                    self.assertAlmostEqual(g[k][idx], (up - down)/2e-6, delta=1e-6 + 1e-4*abs(g[k][idx]), msg=f'{k}{idx}')

    def test_train_smoke_record_and_shuffle(self):
        with tempfile.TemporaryDirectory() as tmp:
            for cond, model in (('final', 'mlp'), ('shuf', 'mlp'), ('interp', 'mlp'), ('inter', 'rnn')):
                out = Path(tmp)/(cond + model)
                subprocess.run([sys.executable, str(HERE/'train.py'), '--out', str(out), '--condition', cond, '--model', model,
                                '--init-seed', '900099', '--updates', '3', '--eval-every', '1', '--hidden', '8', '8'],
                               check=True, capture_output=True)
                config = json.loads((out/'config.json').read_text())
                lines = [json.loads(x) for x in (out/'metrics.jsonl').read_text().splitlines()]
                self.assertEqual([x['step'] for x in lines], [0, 1, 2, 3])
                self.assertEqual(len(lines[-1]['held_final_pred']), 128)
                self.assertEqual(json.loads((out/'runtime.json').read_text())['status'], 'complete')
                if cond == 'interp':
                    self.assertEqual(config['exposure']['fit_rows'], 32)  # every (state, op) transition
                if cond == 'shuf':
                    # Shuffled targets keep the label counts but no longer match the true states.
                    self.assertEqual(config['exposure']['labels_per_row'], 3)
                    self.assertLess(lines[-1]['fit_targets']['step1']['acc'], 1)
                if model == 'rnn':
                    self.assertEqual(config['exposure']['total_parameters'], 8*8+8 + 8*8+4*8+8 + 8*8+8)
            a = nets.load(Path(tmp)/'finalmlp'/'ckpt-0.npz')[0]; b = nets.load(Path(tmp)/'shufmlp'/'ckpt-0.npz')[0]
            for k in a:  # same init seed: trunk and final head start bitwise identical across conditions
                self.assertTrue(np.array_equal(a[k], b[k]), k)


if __name__ == '__main__':
    unittest.main()
