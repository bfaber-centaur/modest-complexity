"""Small NumPy ReLU network with named softmax heads, Adam with explicit state,
and checkpoints. Shared by the four regime lanes (research/regimes/PLAN.md).

No task logic lives here. Every head is a softmax classifier over its own
classes (a binary task is a two-class head) reading the last hidden layer.
A target of -1 marks an unlabeled row: it contributes nothing to that head's
loss or gradient. Float64 throughout.
"""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from instrumentation import inventory  # noqa: E402


def _head_seed(seed, name):
    # Each head's draw depends only on (seed, name), so adding an auxiliary head
    # leaves the trunk and every other head bitwise unchanged.
    return [seed, int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], 'big')]


def init(n_in, hidden, heads, seed):
    """hidden: widths of the ReLU layers, in order. heads: {name: n_classes}.
    He-normal trunk, LeCun-normal heads, zero biases."""
    rng = np.random.default_rng(seed)
    p, sizes = {}, [n_in, *hidden]
    for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:]), 1):
        p[f'W{i}'] = rng.normal(0, np.sqrt(2/a), (a, b)); p[f'b{i}'] = np.zeros(b)
    for name, k in heads.items():
        r = np.random.default_rng(_head_seed(seed, name))
        p[f'W_{name}'] = r.normal(0, np.sqrt(1/sizes[-1]), (sizes[-1], k)); p[f'b_{name}'] = np.zeros(k)
    return p


def depth(p):
    return sum(1 for k in p if k[0] == 'W' and k[1:].isdigit())


def head_names(p):
    return [k[2:] for k in p if k.startswith('W_')]


def forward_from(p, layer, H):
    """Continue from activations H of `layer` (0 = the input). Returns
    (acts, logits): acts[j] is layer layer+j's activation, logits is {head: array}.
    Use it to run a network on edited (patched, ablated) hidden activations."""
    acts = [H]
    for i in range(layer+1, depth(p)+1):
        acts.append(np.maximum(acts[-1] @ p[f'W{i}'] + p[f'b{i}'], 0))
    return acts, {h: acts[-1] @ p[f'W_{h}'] + p[f'b_{h}'] for h in head_names(p)}


def forward(p, X):
    return forward_from(p, 0, X)


def softmax(z):
    e = np.exp(z - z.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def loss_grad(p, X, targets, weights=None):
    """Per-head mean cross-entropy over that head's labeled rows, summed with
    `weights` (default 1 per head). Heads absent from `targets` get no loss.
    Returns (total, {head: loss}, grads)."""
    acts, logits = forward(p, X)
    weights = weights or {}
    grads = {k: np.zeros_like(v) for k, v in p.items()}
    total, per_head = 0., {}
    dH = np.zeros_like(acts[-1])
    for h, y in targets.items():
        y = np.asarray(y); rows = np.flatnonzero(y >= 0)
        if len(rows) == 0:
            continue
        sel = slice(None) if len(rows) == len(y) else rows  # a slice avoids copies
        z = logits[h][sel]; yl = y[sel]; n = len(rows)
        zs = z - z.max(1, keepdims=True)
        lse = np.log(np.exp(zs).sum(1))
        per_head[h] = float(np.mean(lse - zs[np.arange(n), yl]))
        w = weights.get(h, 1.)
        total += w * per_head[h]
        d = softmax(z); d[np.arange(n), yl] -= 1; d *= w / n
        grads[f'W_{h}'] = acts[-1][sel].T @ d; grads[f'b_{h}'] = d.sum(0)
        dH[sel] += d @ p[f'W_{h}'].T
    for i in range(depth(p), 0, -1):
        dpre = dH * (acts[i] > 0)
        grads[f'W{i}'] = acts[i-1].T @ dpre; grads[f'b{i}'] = dpre.sum(0)
        if i > 1:  # no gradient with respect to the input
            dH = dpre @ p[f'W{i}'].T
    return total, per_head, grads


def adam_state(p):
    """Fresh optimizer state. A phase boundary that resets the optimizer calls this."""
    return dict(t=0, m={k: np.zeros_like(v) for k, v in p.items()},
                v={k: np.zeros_like(v) for k, v in p.items()})


def adam_step(p, g, state, lr, wd=0., trainable=None, b1=.9, b2=.999, eps=1e-8):
    """One Adam update with decoupled decay, in place, as experiment.adam. Names
    outside `trainable` (default: all) get neither update nor decay, and their
    moments are left untouched. The bias-correction count is shared."""
    state['t'] += 1; t = state['t']
    for k in (p if trainable is None else trainable):
        m, v = state['m'][k], state['v'][k]
        m *= b1; m += (1-b1)*g[k]
        v *= b2; v += (1-b2)*g[k]**2
        p[k] *= 1 - lr*wd
        p[k] -= lr*(m/(1-b1**t)) / (np.sqrt(v/(1-b2**t)) + eps)


def save(path, p, state, **meta):
    """Weights, optimizer state and JSON-serializable metadata in one .npz."""
    arrays = {f'p.{k}': v for k, v in p.items()}
    arrays.update({f'm.{k}': v for k, v in state['m'].items()})
    arrays.update({f'v.{k}': v for k, v in state['v'].items()})
    np.savez_compressed(path, t=state['t'], meta=json.dumps(meta), **arrays)


def load(path):
    with np.load(path) as z:
        get = lambda prefix: {k[len(prefix):]: z[k] for k in z.files if k.startswith(prefix)}
        p, m, v = get('p.'), get('m.'), get('v.')
        state = dict(t=int(z['t']), m=m, v=v)
        meta = json.loads(str(z['meta']))
    # forward indexes parameters by name, so key order does not matter.
    return p, state, meta


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def provenance(out, *files):
    """Runtime inventory (instrumentation.inventory) plus hashes of the lane's own
    files, which inventory does not cover because it hashes top-level *.py only."""
    return dict(inventory(out), lane_source_sha256={str(Path(f).resolve().relative_to(ROOT)): sha256_file(f)
                                                    for f in (*files, __file__)})
