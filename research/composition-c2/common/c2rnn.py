"""c2 recurrent model: r1's composition RNN (research/regimes/composition/train.py, rnn_*)
with configurable transition and readout tying, programs of any length, and continuation
from any hidden state. Float64, own backpropagation.

    h0 = relu(onehot(s) @ Ws + bs)
    h_t = relu(h_{t-1} @ A[k] + onehot(op_t) @ B[k] + bh[k]),   k = step_map[t-1]
    logits_t = h_t @ V[r] + bv[r],                              r = step_map[t-1]

Tying scheme (declared in `scheme`): the start embedding (Ws, bs) is always one copy. The
transition group is A (state weights), B (op embedding) and bh (bias); the readout group is
V and bv. A tied group is stored once under its bare name; an untied group is stored as
copies '<name><k>' for k = 1..3, one per training step. `init` draws r1's tied parameters
exactly (rnn_init) and copies them, so all four schemes start as the same function and the
untied copies are independent arrays. In the tied scheme a shared parameter receives the sum
of the per-step gradients; in an untied scheme copy k receives step k's term alone.

`step_map` gives, for each step t of a program, which per-step copy k (1..3) is used. It is
required for an untied group on programs longer than 3 or for a continuation that starts
mid-program, so that every cross-position use is an explicit choice. A tied group ignores it.

Tied/tied arithmetic is r1's, operation for operation (test_c2.py checks bitwise identity
with r1's train.py and its published metrics).
"""
import numpy as np

from c2task import N_OP, N_SYM, TRAIN_LENGTH

TRANSITION, READOUT = ('A', 'B', 'bh'), ('V', 'bv')
EYE_S, EYE_O = np.eye(N_SYM), np.eye(N_OP)


def scheme(p):
    """{'transition': tied?, 'readout': tied?} read from parameter names."""
    return dict(transition='A' in p, readout='V' in p)


def init(width, seed, tie_transition=True, tie_readout=True):
    """r1's rnn_init: He-normal start embedding and transition (fan-in 8 and width + 4),
    LeCun-normal readout, zero biases; then untied groups as identical, independent copies."""
    r = np.random.default_rng(seed)
    p = dict(Ws=r.normal(0, np.sqrt(2/N_SYM), (N_SYM, width)), bs=np.zeros(width),
             A=r.normal(0, np.sqrt(2/(width+N_OP)), (width, width)), B=r.normal(0, np.sqrt(2/(width+N_OP)), (N_OP, width)),
             bh=np.zeros(width), V=r.normal(0, np.sqrt(1/width), (width, N_SYM)), bv=np.zeros(N_SYM))
    for tied, group in ((tie_transition, TRANSITION), (tie_readout, READOUT)):
        if not tied:
            for name in group:
                v = p.pop(name)
                for k in range(1, TRAIN_LENGTH+1):
                    p[f'{name}{k}'] = v.copy()
    return p


def _keys(p, group, k):
    return tuple(name if name in p else f'{name}{k}' for name in group)


def _step_map(p, n_steps, step_map):
    if step_map is None:
        tied = scheme(p)
        if n_steps > TRAIN_LENGTH and not (tied['transition'] and tied['readout']):
            raise ValueError(f'an untied model needs an explicit step_map for {n_steps} steps')
        step_map = range(1, n_steps+1)
    step_map = [int(k) for k in step_map]
    assert len(step_map) == n_steps and all(1 <= k <= TRAIN_LENGTH for k in step_map), step_map
    return step_map


def start(p, s):
    return np.maximum(EYE_S[s] @ p['Ws'] + p['bs'], 0)


def forward(p, ops, s=None, h0=None, step_map=None):
    """Hidden states [h0, h1, ..., hL] and per-step logits [logits_1, ..., logits_L].
    ops: (n, L) op indices. Give the start symbols s, or a hidden state h0 to continue from
    (then step_map is required unless both groups are tied)."""
    ops = np.asarray(ops).reshape(len(ops), -1)
    if h0 is None:
        H = [start(p, s)]
    else:
        if step_map is None and not all(scheme(p).values()):
            raise ValueError('continuing an untied model from a hidden state needs an explicit step_map')
        H = [np.asarray(h0)]
    smap = _step_map(p, ops.shape[1], step_map)
    logits = []
    for t, k in enumerate(smap):
        A, B, bh = _keys(p, TRANSITION, k)
        H.append(np.maximum(H[-1] @ p[A] + EYE_O[ops[:, t]] @ p[B] + p[bh], 0))
        V, bv = _keys(p, READOUT, k)
        logits.append(H[-1] @ p[V] + p[bv])
    return H, logits


def loss_grad(p, s, ops, targets, weights):
    """sum_t weights[t] * mean cross-entropy of logits_t against targets[t], over training-length
    rows with steps 1..3; targets/weights are lists indexed by step (weight 0 = step not trained,
    its target unused). Backpropagation through all steps. Returns (total, per_step, grads)."""
    H, logits = forward(p, ops, s=s)
    g = {k: np.zeros_like(v) for k, v in p.items()}
    dH = [np.zeros_like(h) for h in H]
    total, per_step = 0., []
    for t in range(1, len(H)):
        w = weights[t-1]
        if w == 0:
            per_step.append(None); continue
        y = targets[t-1]; n = len(y)
        z = logits[t-1] - logits[t-1].max(1, keepdims=True)
        e = np.exp(z); sm = e / e.sum(1, keepdims=True)
        ce = float(np.mean(np.log(e.sum(1)) - z[np.arange(n), y]))
        total += ce if w == 1 else w*ce; per_step.append(ce)
        sm[np.arange(n), y] -= 1; sm /= n
        if w != 1:
            sm *= w
        V, bv = _keys(p, READOUT, t)
        g[V] += H[t].T @ sm; g[bv] += sm.sum(0); dH[t] += sm @ p[V].T
    for t in range(len(H)-1, 0, -1):
        A, B, bh = _keys(p, TRANSITION, t)
        dz = dH[t] * (H[t] > 0)
        g[A] += H[t-1].T @ dz; g[B] += EYE_O[ops[:, t-1]].T @ dz; g[bh] += dz.sum(0)
        dH[t-1] += dz @ p[A].T
    dz = dH[0] * (H[0] > 0)
    g['Ws'] = EYE_S[s].T @ dz; g['bs'] = dz.sum(0)
    return total, per_step, g


def ce_acc(logits, y):
    """r1's train.ce_acc."""
    z = logits - logits.max(1, keepdims=True)
    lse = np.log(np.exp(z).sum(1))
    return dict(acc=float(np.mean(z.argmax(1) == y)), ce=float(np.mean(lse - z[np.arange(len(y)), y])))


def margin(logits, y):
    """Correct-class logit minus the largest other logit, per row."""
    n = np.arange(len(y)); other = logits.copy(); other[n, y] = -np.inf
    return logits[n, y] - other.max(1)
