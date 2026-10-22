"""c3 correction lane: frozen bases, depth-2 prefix sets, centroids, the map and the arms (QUESTION.md).

Units. A prefix is a unique (start, op1, op2); its state h_2 does not depend on op 3. Prefix
index = lex(op1, op2)*8 + s (c2 mechanisms order). Rows are the 512 length-3 programs of
c2task.rows(P, programs(3)), index = lex(op1, op2, op3)*8 + s, so row r has prefix
(r // 32)*8 + r % 8.

Arms, each a function of the hidden-state matrix alone except the oracle E:
  A identity; B map trained toward c(x); C the same map trained to reconstruct h_2;
  D linear decoder then c(x_hat); E c(x) of the true symbol (oracle reference).
"""
import json
from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
C3 = HERE.parent
REPO = C3.parents[1]
sys.path[:0] = [str(C3/'common'), str(REPO/'research'/'composition-c2'/'common'), str(REPO/'research'/'regimes'/'common')]
import c3correct  # noqa: E402
import c3train  # noqa: E402
import c2probe  # noqa: E402
import c2rnn  # noqa: E402
import c2task  # noqa: E402
import nets  # noqa: E402

N_SYM, N_OP = c2task.N_SYM, c2task.N_OP
DEPTH, STEP, WIDTH = 2, 10000, 128
SEEDS = (51200, 51201, 51202)
MAP_SEED, SPLIT_SEED, PROBE_SEED, CALIB_SEED = 71200, 71300, 71301, 900200
UPDATES = 2000
GROUPS = dict(fit=8, val=4, eval=4)  # prefixes per symbol in the grouped split
ARMS = ('A', 'B', 'C', 'D', 'E')
F2_TASKS = ('ops63000-ls63100', 'ops63000-ls63101', 'ops63001-ls63102', 'ops63001-ls63103')
F2_SEEDS = (63200, 63201)
# condition: (run-id template, published r1 runs directory, readout tying, r1 condition)
BASES = {
    'per-step readout': ('r1-rev-comp-rnn-untied-s{seed}-a1', 'research/regimes/review/runs', 'untied', 'inter'),
    'tied/final': ('r1-comp-rnn-final-s{seed}-a1', 'research/regimes/composition/runs', 'tied', 'final'),
}


def bases(which='original'):
    """'original': the six D4 bases (SHA-256 from c2 RECOVERY.json). 'f2': the follow-up's eight c2
    generalization tied/final runs on new task draws (SHA-256 from each run's MANIFEST.json)."""
    out = []
    if which == 'original':
        rec = {(f['run_id'], f['file']): f['sha256']
               for f in json.loads((REPO/'research/composition-c2/recovered/RECOVERY.json').read_text())['files']}
        for cond, (tmpl, published, readout, r1cond) in BASES.items():
            for seed in SEEDS:
                run = tmpl.format(seed=seed)
                out.append(dict(condition=cond, seed=seed, run_id=run, readout=readout, r1_condition=r1cond,
                                task_id=c2task.ORIGINAL, path=REPO/'research/composition-c2/recovered'/run/f'ckpt-{STEP}.npz',
                                sha256=rec[(run, f'ckpt-{STEP}.npz')], config=REPO/published/run/'config.json'))
    elif which == 'f2':
        for tid in F2_TASKS:
            for seed in F2_SEEDS:
                run = f'c2-gen-final-{tid}-s{seed}-a1'
                d = REPO/'research/composition-c2/generalization/runs'/run
                sha = [e['sha256'] for e in json.loads((d/'MANIFEST.json').read_text()) if e['file'] == f'ckpt-{STEP}.npz']
                out.append(dict(condition='tied/final', seed=seed, run_id=run, readout='tied', r1_condition='final',
                                task_id=tid, path=d/f'ckpt-{STEP}.npz', sha256=sha[0], config=d/'config.json'))
    else:
        raise ValueError(which)
    return out


def expected_recipe(b):
    recipe = dict(task_id=b['task_id'], transition='tied', readout=b['readout'], hidden=WIDTH,
                  lr=1e-3, wd=0.1, adam=[0.9, 0.999, 1e-8])
    sched = dict(before=c3train.R1_WEIGHTS[b['r1_condition']], switch_step=None, after=None, taper=1)
    return recipe, sched


def load_base(b):
    """Frozen parameters and a record of the checks: file SHA-256, meta and Adam step, recipe."""
    sha = nets.sha256_file(b['path'])
    if sha != b['sha256']:
        raise ValueError(f"{b['path']}: SHA-256 {sha} is not the recovered {b['sha256']}")
    p, st, meta = nets.load(b['path'])
    recipe, sched = expected_recipe(b)
    checked = c3train.check_source(b['path'], sha, b['config'], meta, st['t'], recipe, sched)
    assert st['t'] == STEP, (st['t'], STEP)
    p = c3correct.frozen(p)
    rec = dict(condition=b['condition'], seed=b['seed'], run_id=b['run_id'], ckpt=str(b['path'].relative_to(REPO)),
               ckpt_sha256=sha, adam_step=st['t'], meta_step=meta.get('step'), recipe=recipe, schedule=sched,
               source=dict(checked, config=str(Path(checked['config']).relative_to(REPO))),
               params_sha256=c3correct.params_sha256(p))
    return p, rec


def task(tid=c2task.ORIGINAL):
    """Rows of all 512 length-3 programs with held/prefix flags, and the 128 prefixes with symbol x."""
    T = c2task.load(tid)
    R = c2task.rows(T['P'], c2task.programs(3))
    held = {tuple(q) for q in T['held']}
    R['held'] = np.array([tuple(int(o) for o in q) in held for q in R['ops']])
    R['prefix'] = (np.arange(len(R['s'])) // (N_SYM*N_OP)) * N_SYM + np.arange(len(R['s'])) % N_SYM
    first = np.unique(R['prefix'], return_index=True)[1]
    pre = dict(s=R['s'][first], ops=R['ops'][first, :DEPTH], x=R['states'][first, DEPTH-1])
    assert np.array_equal(R['ops'][first, 2], np.zeros(len(first), dtype=R['ops'].dtype))
    return T, R, pre


def split(pre, seed=SPLIT_SEED):
    """Grouped-prefix split: within each symbol (in order 0..7), its prefixes in index order are
    permuted by one generator and cut fit/val/eval = 8/4/4. Returns {name: sorted prefix indices}."""
    rng = np.random.default_rng(seed)
    out = {k: [] for k in GROUPS}
    for x in range(N_SYM):
        idx = rng.permutation(np.flatnonzero(pre['x'] == x))
        a = 0
        for k, n in GROUPS.items():
            out[k] += list(idx[a:a+n]); a += n
        assert a == len(idx)
    return {k: np.array(sorted(v)) for k, v in out.items()}


def states(p, pre):
    """h_2 of every prefix (rows of pre)."""
    return c2rnn.forward(p, pre['ops'], s=pre['s'], step_map=list(range(1, DEPTH+1)))[0][DEPTH]


def centroids(H, x):
    """Unweighted mean of h_2 per symbol over the given prefixes (each prefix once)."""
    return np.stack([H[x == k].mean(0) for k in range(N_SYM)])


def centroid_scale(C):
    """Mean squared distance between distinct centroids."""
    d = ((C[:, None] - C[None]) ** 2).sum(-1)
    return float(d[~np.eye(N_SYM, dtype=bool)].mean())


def nearest(C, H):
    return ((H[:, None] - C[None]) ** 2).sum(-1).argmin(1)


# ---- the map g(h) = relu(relu(h W1 + b1) W2 + b2) ----

def map_init(k, seed, width=WIDTH):
    """He-normal W1, LeCun-normal W2 (as nets.init), zero biases."""
    rng = np.random.default_rng(seed)
    return dict(W1=rng.normal(0, np.sqrt(2/width), (width, k)), b1=np.zeros(k),
                W2=rng.normal(0, np.sqrt(1/k), (k, width)), b2=np.zeros(width))


def map_forward(q, H):
    Z = np.maximum(H @ q['W1'] + q['b1'], 0)
    return Z, np.maximum(Z @ q['W2'] + q['b2'], 0)


def map_loss_grad(q, H, Y):
    """Mean squared error over prefixes and units, and its gradient."""
    Z, G = map_forward(q, H)
    D = G - Y
    dG = 2*D/D.size * (G > 0)
    dZ = dG @ q['W2'].T * (Z > 0)
    return float(np.mean(D**2)), dict(W1=H.T @ dZ, b1=dZ.sum(0), W2=Z.T @ dG, b2=dG.sum(0))


def map_mse(q, H, Y):
    return float(np.mean((map_forward(q, H)[1] - Y) ** 2))


def fit_map(H, Y, k, lr, seed, updates=UPDATES):
    """Full-batch Adam (nets.adam_step defaults, no decay). Returns the map and its fit record."""
    q = map_init(k, seed); st = nets.adam_state(q)
    first = None
    for _ in range(updates):
        loss, g = map_loss_grad(q, H, Y)
        first = loss if first is None else first
        nets.adam_step(q, g, st, lr)
    pre_out = map_forward(q, H)[0] @ q['W2'] + q['b2']
    return q, dict(k=k, lr=lr, seed=seed, updates=updates, loss_first=first, loss_last=map_mse(q, H, Y),
                   dead_output_units=int(((pre_out <= 0).all(0) & (Y > 0).any(0)).sum()))


# ---- arms ----

def fit_arms(H, x, k, lr):
    """Fit B, C and D on the fit prefixes (H, x); returns the centroids, modules and fit records."""
    C = centroids(H, x)
    qB, fB = fit_map(H, C[x], k, lr, MAP_SEED)
    qC, fC = fit_map(H, H, k, lr, MAP_SEED)
    probe = c2probe.fit_probe(H, x, PROBE_SEED)
    fD = dict(probe_seed=PROBE_SEED, train_acc=c2probe.accuracy(probe, H, x))
    return dict(C=C, B=qB, Cmap=qC, D=probe), dict(B=fB, C=fC, D=fD)


def decode(probe, H):
    return nets.forward(probe, H)[1]['y'].argmax(1)


def correction(arm, mods, x_true=None):
    """The function applied to h_2. Only E uses x_true (the oracle reference)."""
    if arm == 'A':
        return None
    if arm == 'B':
        return lambda H: map_forward(mods['B'], H)[1]
    if arm == 'C':
        return lambda H: map_forward(mods['Cmap'], H)[1]
    if arm == 'D':
        return lambda H: mods['C'][decode(mods['D'], H)]
    if arm == 'E':
        return lambda H: mods['C'][x_true]
    raise ValueError(arm)


def accuracy(p, R, sel, fn):
    out = c3correct.corrected_forward(p, R['s'][sel], R['ops'][sel], DEPTH, correct=fn)
    return float(np.mean(out['logits'][-1].argmax(1) == R['states'][sel, -1]))
