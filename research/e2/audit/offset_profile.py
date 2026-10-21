#!/usr/bin/env python3
"""Offset profile of the restricted-rank statistic (QUESTION.md).

For a held-out pair (a, b), the absent labels are those never seen with operand a
or operand b (either position) among the run's training targets; this is the mask
of research/audit/heldout_rank.py. For each offset k the probe is a+b+k (mod p).
Only pairs whose probe is absent are used. Its normalized rank among the other
absent labels is (number of absent labels with a strictly larger logit) / (number
of other absent labels): 0 = above all, 0.5 = no preference. Pairs with no other
absent label are left out and counted. For every k, the rank of a+b itself is
also computed on the same pairs (with a+b+k still in the comparison set), so
offsets are compared on matched pairs. Because a high a+b sits above every other
probe, offsets k != 0 are also ranked with a+b left out of the comparison set
(rank_mean_without_sum). Independent code only.

uv run --locked python research/e2/audit/offset_profile.py STEP RUN_DIR ... > OUT.jsonl
"""
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'audit'))
import independent as ind  # noqa: E402


def rank(logits, probe, absent):
    """Normalized rank of `probe` among absent labels other than itself; NaN where none."""
    rows = np.arange(len(probe))
    other = absent.copy(); other[rows, probe] = False
    true = logits[rows, probe][:, None]
    n = other.sum(1)
    above = ((logits > true) & other).sum(1)
    ties = ((logits == true) & other).sum(1)
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(n > 0, above / n, np.nan), above == 0, n, ties


def profile(run_dir, step):
    with np.load(run_dir/'data.npz') as d:
        A, B, Y, yadd, tr = d['a'], d['b'], d['y'], d['yadd'], d['train']
        held = np.concatenate([d['cal'], d['audit']])
    p = int(yadd.max()) + 1
    present = np.zeros((p, p), bool)
    present[A[tr], Y[tr]] = True; present[B[tr], Y[tr]] = True
    absent = ~(present[A[held]] | present[B[held]])
    with np.load(run_dir/f'checkpoint_{step:06d}.npz') as z:
        logits = ind.forward((z['U'], z['V'], z['W']), *ind.one_hot_inputs(A, B, p))[0][held]
    s = yadd[held]; rows = []
    for k in range(p):
        probe = (s + k) % p
        use = absent[np.arange(len(held)), probe] & absent[np.arange(len(held)), s]  # probe and a+b both absent
        r, top, n, ties = rank(logits[use], probe[use], absent[use])
        r0, _, _, _ = rank(logits[use], s[use], absent[use])
        # Leaving a+b out of the comparison set removes the mechanical shift a high a+b gives other offsets.
        wo = absent[use].copy(); wo[np.arange(use.sum()), s[use]] = k == 0
        rw, _, nw, _ = rank(logits[use], probe[use], wo)
        ok = ~np.isnan(r)
        rows.append(dict(k=k, pairs=int(ok.sum()), singletons_left_out=int((~ok).sum()), ties=int(ties.sum()),
                         rank_mean=round(float(r[ok].mean()), 4) if ok.any() else None,
                         top_frac=round(float(top[ok].mean()), 4) if ok.any() else None,
                         chance_top=round(float(np.mean(1/(n[ok] + 1))), 4) if ok.any() else None,
                         sum_rank_mean_same_pairs=round(float(np.nanmean(r0[ok])), 4) if ok.any() else None,
                         rank_mean_without_sum=round(float(np.nanmean(rw)), 4) if (~np.isnan(rw)).any() else None,
                         pairs_without_sum=int((~np.isnan(rw)).sum())))
    return dict(run=run_dir.name, step=step, heldout_pairs=len(held), trained_on_addition=bool(np.array_equal(Y, yadd)),
                absent_per_pair=(int(absent.sum(1).min()), float(np.median(absent.sum(1))), int(absent.sum(1).max())),
                rows=rows)


if __name__ == '__main__':
    step = int(sys.argv[1])
    for run in map(Path, sys.argv[2:]):
        print(json.dumps(profile(run, step)), flush=True)
