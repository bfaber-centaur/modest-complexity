"""Scale-free pairwise preference between label sets, and the S / D / N label sets of a
nested ladder (definitions for e3; research/e3/audit/QUESTION.md).

For one held-out pair with logits z (length p) and disjoint label sets A, B:

    P(A over B) = mean over a in A, b in B of 1[z_a > z_b] + 0.5 * 1[z_a = z_b]

It depends only on the order of the logits within the pair, so any per-pair strictly
increasing transform (positive scaling, a common offset) leaves it unchanged. It is
undefined (NaN) when A or B is empty. P(a+b over B) is the fraction of B below the
correct label, a rank analogue of the margin over B.

Run-level summaries in this audit average the per-pair values with equal weight per pair,
dropping and counting undefined pairs. Transposes share their label sets, so pairs are not
independent replicates.
"""
import numpy as np


def preference(z, A, B):
    """Per-pair P(A over B) for logits z (pairs, p) and boolean masks A, B (pairs, p).
    Returns (values with NaN where A or B is empty, number of tied comparisons)."""
    gt = z[:, :, None] > z[:, None, :]
    eq = z[:, :, None] == z[:, None, :]
    mask = A[:, :, None] & B[:, None, :]
    n = mask.sum((1, 2))
    wins = (gt & mask).sum((1, 2)) + 0.5*(eq & mask).sum((1, 2))
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(n > 0, wins/n, np.nan), int((eq & mask).sum())


def operand_labels(a, b, train, p):
    """(p, p) boolean: label (a + b) % p occurs on a training pair containing operand x,
    in either position."""
    m = np.zeros((p, p), bool)
    y = (a[train] + b[train]) % p
    m[a[train], y] = True
    m[b[train], y] = True
    return m


def ladder_sets(a, b, smallest_train, largest_train, pairs, p):
    """S, D, N masks (len(pairs), p) for held-out `pairs` of a nested ladder: S = labels seen
    with an operand in the smallest training set, D = those seen only in the largest, N = the
    other wrong labels. The correct label is in none of them (asserted)."""
    assert set(smallest_train.tolist()) <= set(largest_train.tolist())
    assert not set(pairs.tolist()) & set(largest_train.tolist())
    ca, cb = a[pairs], b[pairs]
    y = (ca + cb) % p
    small, large = operand_labels(a, b, smallest_train, p), operand_labels(a, b, largest_train, p)
    S = small[ca] | small[cb]
    D = (large[ca] | large[cb]) & ~S
    N = ~(S | D)
    rows = np.arange(len(y))
    N[rows, y] = False
    assert not (S | D)[rows, y].any()  # for addition, a+b never co-occurs with a or b in training
    return S, D, N, y
