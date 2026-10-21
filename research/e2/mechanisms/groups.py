#!/usr/bin/env python3
"""Unit groups by dominant input frequency: what each group adds to the logits, and
what removing it does, against random unit sets matched in size and norm.

Question and predictions: QUESTION.md. For a set of live units S at one checkpoint:
- contribution C_S(a, b, c) = sum over S of relu(U[a,j] + V[b,j]) W[j,c], minus its mean
  over c for each pair (softmax-invariant);
- d_share: share of C_S's variance on a set of pairs explained by a function of
  d = a + b - c mod p (the within-d mean); k_share: by cos/sin(2 pi k d / p) alone;
  peak_d0: whether that function of d is largest at d = 0 (the correct label);
- ablation: logits without S. Accuracy on the common never-trained set (483 pairs,
  research/e2/data), on the run's own cal + audit and on train; the restricted rank
  among absent labels (research/mechanisms/relational.py) on the common set; and the
  RMS size of the (pair-centred) logit change.
Groups are read from the analysed checkpoint (checkpoint-local). Random sets: units
drawn uniformly from live units, same count, total |u_j||w_j| within 10% of the group's
(|u_j|^2 = |U[:,j]|^2 + |V[:,j]|^2; the product is rescaling-invariant).
Reference: the ideal "pizza" term |cos(pi k (a-b)/p)| cos(2 pi k d / p), averaged over k.

uv run --locked --group analysis python research/e2/mechanisms/groups.py --out research/e2/mechanisms/figures/groups \
    --runs research/coordinator/runs/e1f2-co-add-w128-i30000-tf50-a1:0,5000,50000 ...
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'research/mechanisms'))
from freq import load_run, sha256, write_csv  # noqa: E402
from gauge import live_units  # noqa: E402
from relational import absent_labels, relational  # noqa: E402
from units import unit_shares  # noqa: E402

SEED = 35000
COMMON = ROOT/'research/e2/data/common-never-trained-30-50.npz'


def contribution(w, d, units):
    c = np.maximum(w[0][d['a']][:, units] + w[1][d['b']][:, units], 0) @ w[2][units]
    return c - c.mean(1, keepdims=True)


def d_shares(c, d, rows, k):
    """Shares of variance of c[rows] explained by a function of d, and by cos/sin(k d)."""
    p = c.shape[1]
    x = c[rows]
    dd = (d['a'][rows][:, None] + d['b'][rows][:, None] - np.arange(p)) % p
    g = np.bincount(dd.ravel(), x.ravel(), p)/np.bincount(dd.ravel(), minlength=p)
    basis = np.stack([np.cos(2*np.pi*k*np.arange(p)/p), np.sin(2*np.pi*k*np.arange(p)/p)], 1)
    gk = basis @ np.linalg.lstsq(basis, g, rcond=None)[0]
    total = np.sum(x**2)
    return dict(d_share=float(1 - np.sum((x - g[dd])**2)/total), k_share=float(1 - np.sum((x - gk[dd])**2)/total),
                peak_d0=bool(g.argmax() == 0))


def pizza_reference(d, rows, p):
    """d_share of the ideal pizza term, mean over k = 1..15, on these pairs."""
    a, b = d['a'][:, None], d['b'][:, None]
    out = []
    for k in range(1, (p + 1)//2):
        t = np.abs(np.cos(np.pi*k*(a - b)/p))*np.cos(2*np.pi*k*(a + b - np.arange(p))/p)
        out.append(d_shares(t - t.mean(1, keepdims=True), d, rows, k)['d_share'])
    return float(np.mean(out))


def ablation(w, d, units, held, absent, base):
    keep = np.ones(w[2].shape[0], bool)
    keep[units] = False
    z = np.maximum(w[0][d['a']][:, keep] + w[1][d['b']][:, keep], 0) @ w[2][keep]
    change = (z - z.mean(1, keepdims=True)) - (base - base.mean(1, keepdims=True))
    acc = lambda ix: float(np.mean(z[ix].argmax(1) == d['y'][ix]))
    rel = relational(z, d, held, absent)
    return dict(common_acc=acc(d['common']), own_heldout_acc=acc(np.concatenate([d['cal'], d['audit']])),
                train_acc=acc(d['train']), common_rank=rel['true_rank_among_absent'],
                common_top=rel['true_top_among_absent'], logit_change_rms=float(np.sqrt(np.mean(change**2))))


def measure(w, d, units, k, held, absent, base):
    c = contribution(w, d, units)
    out = dict(n_units=len(units))
    for name, rows in () if len(units) == 0 else (('train', d['train']), ('common', d['common']),
                       ('own_heldout', np.concatenate([d['cal'], d['audit']]))):
        out.update({f'{key}_{name}': v for key, v in d_shares(c, d, rows, k).items()})
    out.update(ablation(w, d, units, held, absent, base))
    return out


def analyse(run, step, draws, rng):
    w, d, meta = load_run(run, step)
    d['common'] = np.load(COMMON)['index']
    assert not set(d['common'].tolist()) & set(d['train'].tolist())
    held, absent = absent_labels(dict(d, cal=d['common'], audit=d['common'][:0]))
    base = np.maximum(w[0][d['a']] + w[1][d['b']], 0) @ w[2]
    live = np.flatnonzero(live_units(w))
    dom = unit_shares(w)[0][:, live].argmax(0) + 1
    size = np.sqrt(np.sum(w[0]**2, 0) + np.sum(w[1]**2, 0))*np.linalg.norm(w[2], axis=1)
    info = dict(run_id=meta['run_id'], step=step, train_pairs=len(d['train']))
    rows = [dict(info, kind='none', k=0, draw=0, **measure(w, d, [], 1, held, absent, base)),
            dict(info, kind='all units', k=0, draw=0, **measure(w, d, live, 1, held, absent, base))]
    for k in range(1, w[0].shape[0]//2 + 1):
        group = live[dom == k]
        if len(group) == 0:
            continue
        rows.append(dict(info, kind='group', k=k, draw=0, **measure(w, d, group, k, held, absent, base)))
        target, found = size[group].sum(), 0
        for _ in range(200*draws):
            pick = rng.choice(live, len(group), replace=False)
            if abs(size[pick].sum()/target - 1) <= .1:
                rows.append(dict(info, kind='random set', k=k, draw=found, **measure(w, d, pick, k, held, absent, base)))
                found += 1
                if found == draws:
                    break
    reference = dict(run_id=meta['run_id'], step=step, **{f'pizza_d_share_{n}': pizza_reference(d, r, w[0].shape[0])
                     for n, r in (('train', d['train']), ('common', d['common']),
                                  ('own_heldout', np.concatenate([d['cal'], d['audit']])))})
    return rows, meta, reference


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--runs', nargs='+', required=True, help='RUN_DIR:STEP,STEP,...')
    ap.add_argument('--draws', type=int, default=16)
    ap.add_argument('--source-commit', help='Commit the published runs were read from (recorded only)')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)
    rows, inputs, references = [], [], []
    for spec in args.runs:
        run, steps = spec.split(':')
        for step in map(int, steps.split(',')):
            r, meta, ref = analyse(ROOT/run, step, args.draws, rng)
            rows += r
            references.append(ref)
            inputs.append({k: meta[k] for k in ('run_id', 'step', 'checkpoint_sha256', 'split_sha256',
                                                'max_abs_diff_vs_logged_metrics', 'manifest_hashes_verified')})
    git = lambda *a: subprocess.run(['git', *a], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    args.out.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    write_csv(args.out/'groups.csv', [{k: r.get(k, '') for k in keys} for r in rows])
    (args.out/'groups_summary.json').write_text(json.dumps(dict(
        inputs=inputs, pizza_reference=references, common_set_sha256=sha256(COMMON), seed=SEED, draws=args.draws,
        read_from_commit=args.source_commit, session_label='e2-me-cloud-1', analysis_code_commit=git('rev-parse', 'HEAD'),
        analysis_tree_status=git('status', '--porcelain'), analysis_script_sha256=sha256(__file__),
        command=' '.join(['python', *sys.argv])), indent=1) + '\n')
    for r in rows:
        if r['kind'] in ('none', 'all units'):
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})


if __name__ == '__main__':
    main()
