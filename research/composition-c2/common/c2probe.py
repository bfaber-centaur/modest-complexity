"""Decoder-across-position table, shared by the sharing and mechanisms lanes (DECISIONS D5).

For positions i, j in 0..3 (h0 is the start state, h_t the state after op t): fit a linear
softmax probe on h_i of the TRAINING-program rows against the true symbol at i, then score it
on h_j of the HELD-OUT-program rows against the true symbol at j. The probe is r1's
(composition/analyze.py fit_probe): no hidden layer, nets.init with this seed, Adam lr 1e-2,
no decay, 1000 full-batch updates. Descriptive only: a low off-diagonal entry does not show
that position j lacks the symbol (an invertible position-specific code remains possible), and
a high entry does not show the network uses that code.
"""
import numpy as np

import c2rnn
import c2task
import nets

POSITIONS = (0, 1, 2, 3)


def symbols(d):
    """True symbol at positions 0..3: the start symbol, then the state after each op."""
    return [d['s'], *(d['states'][:, t] for t in range(d['states'].shape[1]))]


def fit_probe(H, y, seed, updates=1000, lr=1e-2):
    p = nets.init(H.shape[1], [], dict(y=c2task.N_SYM), seed); st = nets.adam_state(p)
    for _ in range(updates):
        nets.adam_step(p, nets.loss_grad(p, H, dict(y=y))[2], st, lr)
    return p


def accuracy(p, H, y):
    return float(np.mean(nets.forward(p, H)[1]['y'].argmax(1) == y))


def decode_table(p_model, T, seed, step_map=None):
    """dict(train_fit[i], table[i][j]) for a model p_model on task T (c2task.load)."""
    tr, he = c2task.rows(T['P'], T['train']), c2task.rows(T['P'], T['held'])
    Htr = c2rnn.forward(p_model, tr['ops'], s=tr['s'], step_map=step_map)[0]
    Hhe = c2rnn.forward(p_model, he['ops'], s=he['s'], step_map=step_map)[0]
    ytr, yhe = symbols(tr), symbols(he)
    fit, table = [], []
    for i in POSITIONS:
        probe = fit_probe(Htr[i], ytr[i], seed)
        fit.append(accuracy(probe, Htr[i], ytr[i]))
        table.append([accuracy(probe, Hhe[j], yhe[j]) for j in POSITIONS])
    return dict(positions=list(POSITIONS), probe_seed=seed, train_fit=fit, table=table)
