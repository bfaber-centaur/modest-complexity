"""R4 task: a 3-bit parity rule, one shortcut bit and six irrelevant bits.

Design and rationale: QUESTION.md. Everything here is deterministic given the
two data seeds. The complete domain is 8 rule patterns x 64 contexts x 2 values
of s = 1,024 inputs, indexed i = (a*64 + c)*2 + s, where a holds the rule bits
(bit j of a is input j), c the irrelevant bits (bit j of c is input 4+j) and s
is input 3. Inputs are encoded as +-1. Training sets are subsets of the domain,
so one forward pass over the domain scores train, held-out and every flip.
"""
import numpy as np

SPLIT_SEED, EXCEPTION_SEED = 54100, 54101
RULE_COLS, S_COL, IRR_COLS = (0, 1, 2), 3, (4, 5, 6, 7, 8, 9)
N_PATTERNS, N_CONTEXTS, N_TRAIN_CONTEXTS = 8, 64, 32
# Exceptions (s = 1 - y) per rule pattern, out of the 32 training bases of that pattern.
EXCEPTIONS = {'rho1': 0, 'rho31of32': 1, 'rho15of16': 2, 'rho7of8': 4, 'rho3of4': 8, 'rho1of2': 16}
CONDITIONS = (*EXCEPTIONS, 'shortcut_only')  # shortcut_only: rho1of2 inputs, target s


def index(a, c, s):
    return (np.asarray(a)*N_CONTEXTS + np.asarray(c))*2 + np.asarray(s)


def domain():
    """Every input. Returns X (1024, 10) in +-1, and integer arrays a, c, s, y."""
    i = np.arange(N_PATTERNS*N_CONTEXTS*2)
    s, c, a = i % 2, (i//2) % N_CONTEXTS, i//(2*N_CONTEXTS)
    bits = np.zeros((len(i), 10), dtype=int)
    for j, col in enumerate(RULE_COLS):
        bits[:, col] = (a >> j) & 1
    for j, col in enumerate(IRR_COLS):
        bits[:, col] = (c >> j) & 1
    bits[:, S_COL] = s
    y = bits[:, list(RULE_COLS)].sum(1) % 2
    return dict(X=2.*bits - 1, a=a, c=c, s=s, y=y)


def train_contexts():
    return np.sort(np.random.default_rng(SPLIT_SEED).permutation(N_CONTEXTS)[:N_TRAIN_CONTEXTS])


def training_bases():
    """The 256 training bases as arrays (a, c), ordered by pattern then context,
    with each base's exception rank within its pattern (0 = first to become an
    exception). Ranks come from one uniform draw, so exceptions nest across rho."""
    ctx = train_contexts()
    a = np.repeat(np.arange(N_PATTERNS), len(ctx)); c = np.tile(ctx, N_PATTERNS)
    u = np.random.default_rng(EXCEPTION_SEED).random(len(a))
    rank = np.empty(len(a), dtype=int)
    for pattern in range(N_PATTERNS):
        rows = np.flatnonzero(a == pattern)
        rank[rows[np.argsort(u[rows], kind='stable')]] = np.arange(len(rows))
    return a, c, rank


def training_set(condition):
    """Domain indices of the 256 training inputs, their targets and exception mask."""
    a, c, rank = training_bases()
    exception = rank < EXCEPTIONS['rho1of2' if condition == 'shortcut_only' else condition]
    y = np.array([bin(int(v)).count('1') % 2 for v in a])
    s = np.where(exception, 1 - y, y)
    target = s if condition == 'shortcut_only' else y
    return dict(idx=index(a, c, s), target=target, exception=exception)


def evaluation_sets():
    """Held-out masks over the domain and the intervention pairs (i, j).
    Rule pairs are oriented (y = 0 member, y = 1 member); s pairs (s = 0, s = 1).
    Irrelevant pairs keep only those whose two endpoints are both held out."""
    d = domain()
    held = ~np.isin(d['c'], train_contexts())
    pairs = {'s': np.flatnonzero(held & (d['s'] == 0))}
    pairs['s'] = np.stack([pairs['s'], pairs['s'] + 1], 1)
    for j in range(len(RULE_COLS)):
        lo = np.flatnonzero(held & ((d['a'] >> j) & 1 == 0))
        hi = index(d['a'][lo] | (1 << j), d['c'][lo], d['s'][lo])
        pairs[f'rule{j}'] = np.where((d['y'][lo] == 0)[:, None], np.stack([lo, hi], 1), np.stack([hi, lo], 1))
    for j in range(len(IRR_COLS)):
        lo = np.flatnonzero(held & ((d['c'] >> j) & 1 == 0))
        hi = index(d['a'][lo], d['c'][lo] | (1 << j), d['s'][lo])
        keep = held[hi]
        pairs[f'irr{j}'] = np.stack([lo[keep], hi[keep]], 1)
    return dict(held=held, reliable=held & (d['s'] == d['y']), reversed=held & (d['s'] != d['y']),
                pairs=pairs)


def metrics(margin, train, ev, y):
    """margin: z1 - z0 over the whole domain. Prediction is class 1 iff margin > 0
    (argmax breaks ties toward class 0)."""
    pred = (margin > 0).astype(int)
    tp = pred[train['idx']]; ok = tp == train['target']; exc = train['exception']
    out = dict(train_acc=ok.mean(), train_acc_exceptions=ok[exc].mean() if exc.any() else None,
               train_acc_conforming=ok[~exc].mean())
    for env in ('reliable', 'reversed', 'held'):
        out[f'acc_{"independent" if env == "held" else env}'] = (pred[ev[env]] == y[ev[env]]).mean()
    for route, names in (('s', ['s']), ('rule', [f'rule{j}' for j in range(3)]),
                         ('irr', [f'irr{j}' for j in range(6)])):
        P = [ev['pairs'][n] for n in names]
        out[f'flip_{route}'] = float(np.mean([(pred[p[:, 0]] != pred[p[:, 1]]).mean() for p in P]))
        out[f'absdelta_{route}'] = float(np.mean([np.abs(margin[p[:, 1]] - margin[p[:, 0]]).mean() for p in P]))
        if route != 'irr':  # oriented: positive = moves the way this route predicts
            out[f'delta_{route}'] = float(np.mean([(margin[p[:, 1]] - margin[p[:, 0]]).mean() for p in P]))
    return {k: (None if v is None else float(v)) for k, v in out.items()}
