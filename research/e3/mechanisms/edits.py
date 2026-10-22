#!/usr/bin/env python3
"""Start files for the e3 mechanisms continuation runs, and the update-0 report.

Arms (research/e3/mechanisms/QUESTION.md; DECISIONS D5), applied to each source checkpoint:
- keep (A): U, V, W of the source, byte-identical arrays.
- filt (B): e2 `filter_units(w, dominant(w)[0])` (research/e2/mechanisms/unitfilter.py), unchanged:
  each unit's U, V columns and W row projected onto its dominant input frequency, DC of U, V kept.
- sham (C): input-energy-matched non-dominant sham. Each unit gets a random order of its 14
  non-dominant frequencies (numpy default_rng(sham_seed), one permutation of 1..15 per unit in
  unit order, the dominant one dropped). It keeps frequencies in that order until its kept
  non-DC input (U and V) energy reaches lam times its dominant-frequency energy; the last
  frequency is kept only if that brings the total closer to the target. One global lam is
  bisected so that the network's total kept non-DC input energy equals B's. The same set is
  kept in U, V and W; the DC of U and V is kept. No pair, label or accuracy enters.

Writes research/e3/mechanisms/starts/<run_id>.npz (np.savez, keys U, V, W only, float64) and
starts/update0.csv, starts/update0.json: per arm, displacement and kept energy per matrix, gate
flips over all 961 pairs, RMS pair-centred logit change, train loss and accuracy, and accuracy on
the 483 common pairs and the run's own cal + audit; for C also lam and per-unit mismatches.

uv run --locked --group analysis python research/e3/mechanisms/edits.py
"""
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'research/mechanisms'))
sys.path.insert(0, str(ROOT/'research/e2/mechanisms'))
from freq import load_run  # noqa: E402
from unitfilter import dominant, filter_units  # noqa: E402

OUT = ROOT/'research/e3/mechanisms/starts'
COMMON = ROOT/'research/e2/data/common-never-trained-30-50.npz'


def input_energy(w):
    """Non-DC input energy per frequency (1..15) and unit, shape (15, hidden)."""
    return sum(np.abs(np.fft.rfft(x, axis=0))**2 for x in w[:2])[1:]


def keep_frequencies(w, keep):
    """Project each unit's U, V columns and W row onto the frequencies keep[:, j] (rows 0..15;
    row 0 is the DC of U and V, which W's projection drops)."""
    p = w[0].shape[0]
    out_keep = keep.copy()
    out_keep[0] = False
    u, v = (np.fft.irfft(np.fft.rfft(x, axis=0)*keep, n=p, axis=0) for x in w[:2])
    return [u, v, np.fft.irfft(np.fft.rfft(w[2], axis=1)*out_keep.T, n=p, axis=1)]


def sham_orders(k, seed):
    rng = np.random.default_rng(seed)
    return [[int(f) for f in rng.permutation(np.arange(1, 16)) if f != k[j]] for j in range(len(k))]


def greedy(e, k, orders, lam):
    """Per unit, frequencies kept in order until the total reaches lam * dominant energy,
    keeping the last one only if it brings the total closer to the target."""
    sets, kept = [], np.zeros(len(k))
    for j in range(len(k)):
        target, chosen, total = lam*e[k[j] - 1, j], [], 0.
        for f in orders[j]:
            new = total + e[f - 1, j]
            if new >= target:
                if abs(new - target) <= abs(total - target):
                    chosen.append(f)
                    total = new
                break
            chosen.append(f)
            total = new
        sets.append(chosen)
        kept[j] = total
    return sets, kept


def sham(w, seed):
    k = dominant(w)[0]
    e = input_energy(w)
    goal = e[k - 1, np.arange(len(k))].sum()
    orders = sham_orders(k, seed)
    lo, hi = 0., 1.
    while greedy(e, k, orders, hi)[1].sum() < goal:
        hi *= 2
    for _ in range(60):
        mid = (lo + hi)/2
        lo, hi = (mid, hi) if greedy(e, k, orders, mid)[1].sum() < goal else (lo, mid)
    lam = min((lo, hi), key=lambda x: abs(greedy(e, k, orders, x)[1].sum() - goal))
    sets, kept = greedy(e, k, orders, lam)
    keep = np.zeros((w[0].shape[0]//2 + 1, len(k)), bool)
    keep[0] = True
    for j, s in enumerate(sets):
        keep[s, j] = True
    ratio = kept/e[k - 1, np.arange(len(k))]
    info = dict(sham_seed=seed, lam=float(lam), kept_over_goal=float(kept.sum()/goal),
                frequencies_kept_median=float(np.median([len(s) for s in sets])),
                unit_ratio_median=float(np.median(ratio)), unit_ratio_p05=float(np.quantile(ratio, .05)),
                unit_ratio_p95=float(np.quantile(ratio, .95)),
                units_unmatchable=int(np.sum(e.sum(0) - e[k - 1, np.arange(len(k))] < e[k - 1, np.arange(len(k))])))
    return keep_frequencies(w, keep), info


def report(w0, w, d, common):
    """Update-0 measurements of edited weights w against the source w0."""
    z0, z = w0[0][d['a']] + w0[1][d['b']], w[0][d['a']] + w[1][d['b']]
    base, logits = np.maximum(z0, 0) @ w0[2], np.maximum(z, 0) @ w[2]
    change = (logits - logits.mean(1, keepdims=True)) - (base - base.mean(1, keepdims=True))
    tr = d['train']
    x = logits[tr] - logits[tr].max(1, keepdims=True)
    loss = np.mean(np.log(np.exp(x).sum(1)) - x[np.arange(len(tr)), d['y'][tr]])
    acc = lambda ix: float(np.mean(logits[ix].argmax(1) == d['yadd'][ix]))
    out = {}
    for name, i in (('U', 0), ('V', 1), ('W', 2)):
        out[f'displacement_{name}'] = float(np.linalg.norm(w[i] - w0[i])/np.linalg.norm(w0[i]))
        out[f'kept_energy_{name}'] = float(np.sum(w[i]**2)/np.sum(w0[i]**2))
    out.update(gate_flips=float(np.mean((z > 0) != (z0 > 0))), logit_change_rms=float(np.sqrt(np.mean(change**2))),
               train_loss=float(loss), train_acc=float(np.mean(logits[tr].argmax(1) == d['y'][tr])),
               common_acc=acc(common), own_heldout_acc=acc(np.concatenate([d['cal'], d['audit']])))
    return out


def main():
    plan = json.loads((ROOT/'research/e3/plan.json').read_text())
    entries = [r for r in plan['reserved'] + plan['runs'] if r['run_id'].startswith('e3-me-')]
    common = np.load(COMMON)['index']
    OUT.mkdir(parents=True, exist_ok=True)
    rows, sources = [], {}
    for r in entries:
        src = ROOT/r['source_checkpoint']
        sha = hashlib.sha256(src.read_bytes()).hexdigest()
        if sha != r['source_checkpoint_sha256']:
            raise ValueError(f"{src}: SHA-256 {sha} != plan {r['source_checkpoint_sha256']}")
        if r['source_run'] not in sources:
            w, d, meta = load_run(src.parent, 50000)  # manifest, split and logged-metric checks
            assert meta['checkpoint_sha256'] == sha and meta['split_sha256'] == r['split_sha256']
            assert not set(common.tolist()) & set(d['train'].tolist())
            sources[r['source_run']] = (w, d, meta)
        w0, d, meta = sources[r['source_run']]
        arm = r['run_id'].split('-')[2]
        info = {}
        if arm == 'keep':
            with np.load(src) as z:
                w = [z[k] for k in 'UVW']
        elif arm == 'filt':
            w = filter_units(w0, dominant(w0)[0])
        elif arm == 'sham':
            w, info = sham(w0, r['sham_seed'])
        else:
            raise ValueError(r['run_id'])
        w = [np.ascontiguousarray(x, dtype=np.float64) for x in w]
        path = OUT/f"{r['run_id']}.npz"
        np.savez(path, U=w[0], V=w[1], W=w[2])
        rows.append(dict(run_id=r['run_id'], arm=arm, source_run=r['source_run'], source_checkpoint_sha256=sha,
                         init_age=r['init_age'], start_file=str(path.relative_to(ROOT)),
                         start_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                         **report(w0, w, d, common), **info))
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(OUT/'update0.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows({k: r.get(k, '') for k in keys} for r in rows)
    git = lambda *a: subprocess.run(['git', *a], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    (OUT/'update0.json').write_text(json.dumps(dict(
        rows=rows, common_set_sha256=hashlib.sha256(COMMON.read_bytes()).hexdigest(),
        source_checks={k: dict(checkpoint_sha256=v[2]['checkpoint_sha256'], split_sha256=v[2]['split_sha256'],
                               max_abs_diff_vs_logged_metrics=v[2]['max_abs_diff_vs_logged_metrics'])
                       for k, v in sources.items()},
        analysis_code_commit=git('rev-parse', 'HEAD'), analysis_tree_status=git('status', '--porcelain'),
        command=' '.join(['python', *sys.argv])), indent=1) + '\n')
    for r in rows:
        print(r['run_id'], r['start_sha256'])


if __name__ == '__main__':
    main()
