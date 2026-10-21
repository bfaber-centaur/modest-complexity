"""Independently written reference computations for the e1 audit.

Deliberately imports nothing from experiment.py, launch_runs.py, report_map.py or
publish_run.py. Formulas are rewritten from the stated specification
(research/SWARM.md Stage 0 item 3) in a different style: one-hot matrix products
instead of indexing/np.add.at, np.logaddexp.reduce instead of max-shifted exp,
an explicit "all other logits" matrix for margins, and AdamW in PyTorch's
parameter-update order. Agreement therefore checks the shared implementation;
it is not a byte-level copy of it.
"""
import hashlib
import numpy as np

B1, B2, EPS = .9, .999, 1e-8


def hash_arrays(*arrays):
    # The project's split/target hash is *defined* as SHA-256 over, for each
    # array in order, the text f"{dtype.str}{shape}" followed by C-order bytes.
    h = hashlib.sha256()
    for x in arrays:
        x = np.asarray(x)
        h.update((x.dtype.str + str(x.shape)).encode('ascii'))
        h.update(x.copy(order='C').data)
    return h.hexdigest()


def regenerate_split(p, data_seed, control_seed):
    """Rebuild the grouped split from the written rule: for each residue c,
    list unordered pairs i<=j with i+j = c (mod p) in lexicographic order,
    permute them with the data generator, assign the first round(.3n) to train,
    the next round(.1n) to cal and the rest to audit; both orders of a pair go
    together. The shuffled-label table permutes the addition labels within each
    split (train, cal, audit in that order) with a generator seeded by
    control_seed."""
    rng = np.random.default_rng(data_seed)
    member = np.full((p, p), -1)
    for c in range(p):
        pairs = [(i, j) for i in range(p) for j in range(i, p) if (i + j) % p == c]
        n = len(pairs)
        n_train, n_cal = max(1, int(np.floor(.3*n + .5))), max(1, int(np.floor(.1*n + .5)))
        for rank, k in enumerate(rng.permutation(n)):
            i, j = pairs[k]
            member[i, j] = member[j, i] = 0 if rank < n_train else 1 if rank < n_train + n_cal else 2
    flat = member.reshape(-1)
    a = np.repeat(np.arange(p), p)
    b = np.tile(np.arange(p), p)
    y = (a + b) % p
    splits = [np.nonzero(flat == s)[0] for s in range(3)]
    shuffled = y.copy()
    ctrl = np.random.default_rng(control_seed)
    for ix in splits:
        shuffled[ix] = ctrl.permutation(y[ix])
    return a, b, y, shuffled, splits


def initial_weights(p, h, init_seed):
    """Spec: U, V ~ N(0, .2^2) of shape (p, h), then W ~ N(0, 1/h) of shape (h, p),
    drawn in that order from one generator."""
    g = np.random.default_rng(init_seed)
    U = g.standard_normal((p, h)) * .2
    V = g.standard_normal((p, h)) * .2
    W = g.standard_normal((h, p)) * (1/np.sqrt(h))
    return U, V, W


def one_hot_inputs(a, b, p):
    eye = np.eye(p)
    return eye[a], eye[b]


def forward(params, Xa, Xb):
    U, V, W = params
    pre = Xa @ U + Xb @ V
    act = pre * (pre > 0)
    return act @ W, act, pre


def evaluate(logits, labels):
    """Accuracy (strict: correct logit beats every other), mean cross-entropy and
    signed margins (correct minus best other) for one split."""
    n, p = logits.shape
    hit = np.eye(p, dtype=bool)[labels]
    correct = logits[hit]
    others = logits[~hit].reshape(n, p - 1)
    margin = correct - others.max(axis=1)
    ce = float(np.mean(np.logaddexp.reduce(logits, axis=1) - correct))
    s = np.sort(margin)
    def q(f):  # linear interpolation between order statistics
        pos = f * (n - 1); lo = int(np.floor(pos)); hi = min(lo + 1, n - 1)
        return float(s[lo] + (pos - lo) * (s[hi] - s[lo]))
    return dict(acc=float(np.count_nonzero(margin > 0)) / n, ties=int(np.count_nonzero(margin == 0)),
                cross_entropy=ce, margin_min=float(s[0]), margin_median=q(.5), margin_p05=q(.05))


def loss_and_grad(params, Xa, Xb, labels):
    U, V, W = params
    logits, act, pre = forward(params, Xa, Xb)
    n, p = logits.shape
    lse = np.logaddexp.reduce(logits, axis=1, keepdims=True)
    Y = np.eye(p)[labels]
    loss = float(np.mean(lse[:, 0] - np.sum(logits * Y, axis=1)))
    d_logits = (np.exp(logits - lse) - Y) / n
    d_pre = (d_logits @ W.T) * (pre > 0)
    return loss, [Xa.T @ d_pre, Xb.T @ d_pre, act.T @ d_logits]


def adamw_step(params, grads, m, v, t, lr, wd):
    """PyTorch AdamW order: decay the parameter, update moments, then subtract
    lr/(1-b1^t) * m / (sqrt(v)/sqrt(1-b2^t) + eps). Returns the displacement
    of this step split into adaptive and decay parts (norms over all tensors)."""
    sq_adapt = sq_decay = cross = 0.
    for k in range(len(params)):
        decay = -lr * wd * params[k]
        params[k] = params[k] * (1 - lr*wd)
        m[k] = B1*m[k] + (1 - B1)*grads[k]
        v[k] = B2*v[k] + (1 - B2)*grads[k]*grads[k]
        adapt = -(lr / (1 - B1**t)) * m[k] / (np.sqrt(v[k]) / np.sqrt(1 - B2**t) + EPS)
        params[k] = params[k] + adapt
        sq_adapt += float(np.vdot(adapt, adapt)); sq_decay += float(np.vdot(decay, decay))
        cross += float(np.vdot(adapt, decay))
    cos = cross / np.sqrt(sq_adapt*sq_decay) if sq_adapt*sq_decay > 0 else None
    return dict(adaptive_norm=np.sqrt(sq_adapt), decay_norm=np.sqrt(sq_decay), cosine=cos)
