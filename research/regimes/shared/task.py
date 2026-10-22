"""R2 shared subcomputations: inputs, label functions, splits and coverage.

Ten binary inputs x1..x10; column j of X holds x_{j+1}, and row i is the
binary expansion of i (x1 is the least significant bit). Bobby's family:
    z  = x1 XOR x2
    y1 = z XOR x3          (related pretraining task)
    y2 = z AND x4          (the new task; 1 on a quarter of the domain)
Pretraining tasks of the other arms are 3-bit parities on other bit sets
(see QUESTION.md): 'part' = x2^x3^x4 shares two of y2's relevant bits but
cannot contain x1^x2; 'unrel' = x5^x6^x7 shares none. The follow-up 'split' is a
4-class pair of parities, one through x1 and one through x2, with no x1^x2.
"""
import numpy as np

N_BITS = 10
X = ((np.arange(2**N_BITS)[:, None] >> np.arange(N_BITS)) & 1).astype(float)
B = X.astype(int)  # B[:, k] is x_{k+1}


def parity(*bits):
    """XOR of the 1-indexed input bits, over the whole domain."""
    out = np.zeros(len(B), int)
    for k in bits:
        out ^= B[:, k-1]
    return out


Z = parity(1, 2)
LABELS = {
    'y1': parity(1, 2, 3),
    'y2': Z & B[:, 3],
    'part': parity(2, 3, 4),
    'unrel': parity(5, 6, 7),
    # Follow-up control (QUESTION.md): 4 classes, 2*(x1^x3^x5) + (x2^x6^x7).
    'split': 2*parity(1, 3, 5) + parity(2, 6, 7),
}
RELEVANT = {'y1': (1, 2, 3), 'y2': (1, 2, 4), 'part': (2, 3, 4), 'unrel': (5, 6, 7), 'split': (1, 2, 3, 5, 6, 7)}
# Candidate intermediates probed in the hidden layers; z first.
PROBES = {'x1^x2': Z, 'x1^x3': parity(1, 3), 'x2^x3': parity(2, 3), 'x1^x4': parity(1, 4),
          'x5^x6': parity(5, 6), 'x1&x2': B[:, 0] & B[:, 1], 'x1': B[:, 0], 'x4': B[:, 3]}


def split(data_seed, pool=512):
    """A random permutation of the 1,024 inputs: the first `pool` indices are the
    training pool (nested subsets D_n = pool[:n]); the rest is the fixed test set."""
    order = np.random.default_rng(data_seed).permutation(len(X))
    return order[:pool], order[pool:]


def coverage(idx, task):
    """Number of distinct patterns of the task's relevant bits present in rows idx."""
    bits = [k-1 for k in RELEVANT[task]]
    return len({tuple(r) for r in B[idx][:, bits]})
