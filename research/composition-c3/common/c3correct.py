"""c3 correction wrapper: run a frozen c2 RNN to depth d, pass h_d through a correction, and
continue the unchanged remaining steps from the corrected state.

    H, logits = forward(p, ops[:, :d], s)            # steps 1..d, untouched
    h = correct(H[d])                                # the only change
    continue forward(p, ops[:, d:], h0=h)            # steps d+1..L, the same weights

`correct` is any function of the hidden-state matrix alone (n, width) -> (n, width). The
wrapper never passes it symbols, ops, suffixes or answers; what a correction was fit on is
the lane's to declare. With correct=None the outputs equal c2rnn.forward on the whole program,
bitwise (test_c3.py), because c2rnn's continuation is bitwise.

`frozen(p)` returns a write-protected copy of the parameters, so any in-place change raises,
and `params_sha256(p)` hashes them by name, dtype, shape and bytes, for before/after records.

Step maps: for programs longer than 3 the tied/tied model gets [1, 2, 3, 3, ...] (its groups
ignore the map); an untied model needs an explicit step_map, as in c2rnn.
"""
import hashlib
from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[1]/'composition-c2'/'common')]
import c2rnn  # noqa: E402
from c2task import TRAIN_LENGTH  # noqa: E402


def frozen(p):
    out = {k: np.array(v, copy=True) for k, v in p.items()}
    for v in out.values():
        v.setflags(write=False)
    return out


def params_sha256(p):
    h = hashlib.sha256()
    for k in sorted(p):
        v = np.ascontiguousarray(p[k])
        h.update(f'{k}|{v.dtype.str}|{v.shape}|'.encode()); h.update(v.tobytes())
    return h.hexdigest()


def step_map_for(p, length, step_map=None):
    if step_map is not None:
        return [int(k) for k in step_map]
    if length <= TRAIN_LENGTH:
        return list(range(1, length+1))
    if all(c2rnn.scheme(p).values()):
        return [min(t, TRAIN_LENGTH) for t in range(1, length+1)]
    raise ValueError(f'an untied model needs an explicit step_map for {length} steps')


def corrected_forward(p, s, ops, depth, correct=None, step_map=None):
    """dict(h_in, h_out, H, logits): the state at `depth` before and after the correction, the
    hidden states [h0..hL] with H[depth] = h_out (the state the remaining steps start from), and
    the per-step logits; logits at steps 1..depth are the uncorrected model's."""
    ops = np.asarray(ops).reshape(len(ops), -1)
    L = ops.shape[1]
    assert 1 <= depth < L, (depth, L)
    smap = step_map_for(p, L, step_map)
    H, logits = c2rnn.forward(p, ops[:, :depth], s=s, step_map=smap[:depth])
    h = H[depth] if correct is None else np.asarray(correct(H[depth]))
    assert h.shape == H[depth].shape, (h.shape, H[depth].shape)
    H2, logits2 = c2rnn.forward(p, ops[:, depth:], h0=h, step_map=smap[depth:])
    return dict(h_in=H[depth], h_out=h, H=H[:depth] + H2, logits=logits + logits2)
