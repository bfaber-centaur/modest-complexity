#!/usr/bin/env python3
"""e4 mechanisms: what changes between the post-filter peak and the collapse (QUESTION.md).

Objects: the e3 continuation runs from the three filtered start files (B) at weight decay 1 and
0, and the unedited runs (A) as reference, on their published checkpoints (manifest-verified by
research/e3/mechanisms/continuation.py); and the six D7 dense replays of B (updates 0-5,000), on
every checkpoint, read from the local run directory after each file matches its published manifest
hash (`source` column: e3 or replay); and the six D15 intervention runs (`source` intervention,
`decay_on` rest or reference: decoupled decay restricted away from, or onto, the reference parts),
checked against their claimed decay settings and against this analysis's reference frequencies;
and the dynamics lane's six D13 decay-off runs (`source` offswitch, `switch_step` 300 or 1,000):
resumed from the e3 wd 1 runs of B and continued at wd 0, on their published checkpoints, with the
reference frequencies and the update-100 gates of the run they resume. Per checkpoint:

- behaviour on the 483 common never-trained pairs: accuracy and the D6 pairwise measures;
- an exact split of the logits z = h W (h = relu(U[a] + V[b]), no biases) into three channels:
    label offset   mu W            (mu = mean hidden activation over the 299 training pairs; shared
                                    by every pair: the mean term of Janati et al. 2026)
    reference      (h - mu) W_ref  (each unit's readout row at its reference frequency k_j)
    off-reference  (h - mu) W_off  (the rest of the row; the row mean is dropped: it is constant
                                    over labels)
  The channels sum to the label-centred logits (checked; maximum error recorded). Linear margins on
  the common pairs, z_{a+b} minus the mean logit over S (or N), add exactly across channels and
  are reported per channel in logit units;
- weights at k_j: Frobenius norms of the parts of U, V (token axis) and W (label axis) at each
  unit's k_j, at DC and at the other frequencies (squares add to the squared norm; checked);
- alignment of the units' contributions to the a+b mode exp(2 pi i k (a + b - c) / p) of the
  logit table over all 961 pairs: A_j = H_j(k_j, k_j) conj(W_j(k_j)) (2D DFT of h_j at (k, k),
  1D DFT of the readout row), whose real part is the unit's support for c = a + b. Reported:
  sum_j Re A_j (useful amplitude), sum_j |A_j|, and their ratio R in [-1, 1]. The A_j summed over
  all units equal the 3D DFT of the logit table at (k, k, -k) (checked);
- gates: fraction of (pair, live unit) ReLU signs over 961 pairs that differ from update 100.

Reference frequency k_j: each unit's dominant input frequency at update 0 (for B, the one the filter
kept; for A, the source's), fixed for the whole run. The reference and off-reference channels are
also recomputed with each checkpoint's own dominant frequencies (`*_cur` columns) to show whether
the answer depends on the definition.

uv run --locked --group analysis python research/e4/mechanisms/channels.py [--local runs/e4] [--out DIR]
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'research/e3/mechanisms'))
import continuation as e3  # noqa: E402  (checks, loaders, D6 measures, label sets)

P = e3.P
HERE = ROOT/'research/e4/mechanisms'
REFERENCE_STEP = 100  # first published checkpoint with train accuracy 1 in all six B runs (label-free)
SOURCES = (30000, 30001, 30002)
RUNS = {('filt', 1): 'e3-me-filt-w128-i{s}-a1', ('filt', 0): 'e3-me-filt-wd0-w128-i{s}-a1',
        ('keep', 1): 'e3-me-keep-w128-i{s}-a1'}
REPLAYS = {('filt', 1): 'e4-me-filt-wd1-replay-w128-i{s}-a1', ('filt', 0): 'e4-me-filt-wd0-replay-w128-i{s}-a1'}
INTERVENTION = ('e4-me-filt-wd1-xref-w128-i{s}-a1', 'e4-me-filt-wd1-onlyref-w128-i{s}-a1')  # D15, decay_on rest / reference
OFFSWITCH = 'e4-dy-filt-wd1to0-u{u}-w128-i{s}-a1'  # dynamics D13: wd 1 to update u, then wd 0 (resumed)


def hidden(w, a, b):
    return np.maximum(w[0][a] + w[1][b], 0)


def split_readout(W, k, live):
    """W_ref (each live row at its frequency k[j] along labels) and W_off (the rest, row mean
    removed). Units that computed nothing at update 0 have no reference and go wholly to W_off."""
    F = np.fft.rfft(W, axis=1)
    keep = np.zeros_like(F, dtype=bool)
    keep[np.arange(len(k)), k] = live
    W_ref = np.fft.irfft(np.where(keep, F, 0), n=W.shape[1], axis=1)
    return W_ref, W - W.mean(1, keepdims=True) - W_ref


def channels(w, d, k, live):
    """Label offset, reference and off-reference logits on the common pairs, and the identity error."""
    h = hidden(w, d['a'][e3.COMMON], d['b'][e3.COMMON])
    mu = hidden(w, d['a'][d['train']], d['b'][d['train']]).mean(0)
    W_ref, W_off = split_readout(w[2], k, live)
    ch = dict(offset=np.broadcast_to(mu @ w[2], h.shape[:1] + (w[2].shape[1],)),
              ref=(h - mu) @ W_ref, off=(h - mu) @ W_off)
    z = h @ w[2]
    centre = lambda x: x - x.mean(1, keepdims=True)
    err = float(np.abs(centre(sum(ch.values())) - centre(z)).max()/np.abs(centre(z)).max())
    return ch, z, err


def linear_margin(z, y, M):
    """z_{a+b} minus the mean logit over the label set M, per pair (exactly additive in z)."""
    rows = np.arange(len(y))
    return z[rows, y] - (z*M).sum(1)/M.sum(1)


def alignment(w, k, live):
    """Per-unit a+b-mode coefficients A_j at k_j over all p^2 pairs, summarised as useful amplitude
    sum Re A_j, total sum |A_j|, alignment R = sum Re A_j / sum |A_j| and coherence
    sum_k |sum_{k_j = k} A_j| / sum |A_j| (1: units of a frequency group add, 0: they cancel; R also
    needs the common phase to favour c = a + b), and the identity error against the 3D DFT of the
    whole logit table."""
    p = w[0].shape[0]
    a, b = np.divmod(np.arange(p*p), p)
    h = hidden(w, a, b).reshape(p, p, -1)                      # (a, b, unit)
    Hk = np.fft.fft2(h, axes=(0, 1))/p**2                      # H_j(k1, k2)
    Wk = np.fft.fft(w[2], axis=1)/p                            # W_j(k)
    units = np.arange(len(k))
    A = Hk[k, k, units]*np.conj(Wk[units, k])
    # Identity: summing every unit's coefficient at (k, k) for one k gives the table's 3D DFT there.
    z = (h.reshape(p*p, -1) @ w[2]).reshape(p, p, p)
    Z = np.fft.fftn(z)/p**3
    err = 0.
    for f in range(1, p//2 + 1):
        total = (Hk[f, f, :]*np.conj(Wk[:, f])).sum()
        err = max(err, abs(total - Z[f, f, (-f) % p]))
    A, k = A[live], k[live]
    total = np.abs(A).sum()
    groups = sum(abs(A[k == f].sum()) for f in np.unique(k))
    return dict(useful_amplitude=float(A.real.sum()), total_amplitude=float(total),
                alignment_R=float(A.real.sum()/total), coherence=float(groups/total)), float(err/np.abs(Z).max())


def weight_parts(w, k, live):
    """Frobenius norms of each weight matrix's part at the units' reference frequencies, of its DC
    part and of the rest (token axis for U and V, label axis for W; units without weights at update
    0 count wholly as rest). By Parseval the squares add to the matrix's squared norm (checked)."""
    p = w[0].shape[0]
    units = np.arange(len(k))
    out, err = {}, 0.
    for name, x, axis in (('U', w[0], 0), ('V', w[1], 0), ('W', w[2].T, 0)):
        e = np.abs(np.fft.rfft(x, axis=axis))**2/p
        e[1:] *= 2                                   # p odd: every non-DC bin stands for +k and -k
        ref = np.zeros(e.shape, bool)
        ref[k, units] = live                         # k >= 1, so DC is never a reference bin
        parts = dict(ref=e[ref].sum(), dc=e[0].sum(), off=e[1:][~ref[1:]].sum())
        err = max(err, abs(sum(parts.values()) - np.sum(x**2))/np.sum(x**2))
        out.update({f'{name}_{part}_norm': float(np.sqrt(v)) for part, v in parts.items()})
    return out, float(err)


def measures(w, d, k_ref, live, sets, gates100):
    S, D, N, y = sets
    ch, z, id_err = channels(w, d, k_ref, live)
    out = dict(e3.common_measures(w, d, sets), channel_identity_rel_err=id_err)
    for setname, M in (('S', S), ('N', N)):
        total = linear_margin(z, y, M)
        out[f'margin_{setname}_total'] = float(total.mean())
        for name, x in ch.items():
            out[f'margin_{setname}_{name}'] = float(linear_margin(x, y, M).mean())
    out['acc_without_offset'] = float(np.mean((z - ch['offset']).argmax(1) == y))
    out['acc_ref_only'] = float(np.mean(ch['ref'].argmax(1) == y))
    out['acc_off_only'] = float(np.mean(ch['off'].argmax(1) == y))
    k_cur = e3.dominant(w)[0]
    ch_cur, _, _ = channels(w, d, k_cur, live)
    for setname, M in (('S', S), ('N', N)):
        for name in ('ref', 'off'):
            out[f'margin_{setname}_{name}_cur'] = float(linear_margin(ch_cur[name], y, M).mean())
    al, al_err = alignment(w, k_ref, live)
    parts, parts_err = weight_parts(w, k_ref, live)
    out.update(al, alignment_identity_rel_err=al_err, **parts, weight_parts_identity_rel_err=parts_err)
    g = e3.gates(w, d)[:, live]
    out['gates_changed_from_update100'] = float(np.mean(g != gates100))
    out['norm'] = float(np.sqrt(sum(np.sum(x**2) for x in w)))
    return out


def checkpoints(run_dir, local_dir):
    """Every checkpoint in the manifest: published ones from Git, the others from local_dir (if
    given), each verified against its manifest hash."""
    manifest = json.loads((run_dir/'manifest.json').read_text())
    out = {}
    for name, f in manifest['files'].items():
        if not name.startswith('checkpoint_'):
            continue
        path = run_dir/name if f['published'] else (local_dir/name if local_dir else None)
        if path is None or not path.exists():
            continue
        if e3.sha256(path) != f['sha256']:
            raise ValueError(f'{path}: hash differs from {run_dir}/manifest.json')
        out[int(name[11:17])] = path
    return manifest, dict(sorted(out.items()))


def load(path):
    with np.load(path) as z:
        return [z[k] for k in 'UVW']


def check_resume(run_dir, r):
    """A --resume run (dynamics D13): config as claimed; the resume file is the parent run's
    published checkpoint at the switch update, with the claimed hash; the run's own first checkpoint
    equals it and replays its logged metrics. Returns (meta, data, the parent's start weights, the
    parent's checkpoints), so that the reference frequencies and gates are the parent's."""
    config = json.loads((run_dir/'config.json').read_text())
    for key, want in (('resume', r['resume']), ('expect_init_sha256', r['resume_sha256']), ('init_weights', None),
                      ('steps', r['steps']), ('wd', r['wd']), ('hidden', r['width']), ('lr', .001)):
        if config.get(key) != want:
            raise ValueError(f'{run_dir.name}: config {key} = {config.get(key)!r}, plan {want!r}')
    parent_dir = ROOT/'research/e3/mechanisms/runs'/r['resume_of']
    _, d0, ws = e3.check_run(parent_dir, next(x for x in e3.entries() if x['run_id'] == r['resume_of']))
    _, parent_paths = checkpoints(parent_dir, None)
    resume, step = ROOT/r['resume'], r['resume_step']
    if parent_paths.get(step) != resume or e3.sha256(resume) != r['resume_sha256']:
        raise ValueError(f'{run_dir.name}: resume file is not the claimed parent checkpoint')
    w, d, meta = e3.load_run(run_dir, step)
    if meta['split_sha256'] != r['split_sha256'] or meta['max_abs_diff_vs_logged_metrics'] > 1e-12:
        raise ValueError(f'{run_dir.name}: split or switch-update replay mismatch {meta}')
    if not all(np.array_equal(x, y) for x, y in zip(w, load(resume))):
        raise ValueError(f'{run_dir.name}: first checkpoint differs from the resume file')
    if not all(np.array_equal(d[k], d0[k]) for k in d):
        raise ValueError(f'{run_dir.name}: data differ from the parent run')
    return meta, d, ws, parent_paths


def analyse(sources):
    """sources: (source label, run directory with manifest, local directory or None, plan entry)."""
    rows, inputs = [], []
    for source, run_dir, local_dir, r in sources:
        if 'resume' in r:  # decay switched off mid-decline: reference and gates from the resumed run
            meta, d, ws, ref_paths = check_resume(run_dir, r)
        else:
            meta, d, ws = e3.check_run(run_dir, r)
        sets = e3.ladder_sets(d['a'], d['b'], e3.tr30, e3.tr50, e3.COMMON, P)
        k_ref, _, live = e3.dominant(ws)
        if 'decay_on' in r:  # restricted decay: config as claimed, projection = this analysis's reference
            config = json.loads((run_dir/'config.json').read_text())
            if (config['decay_on'], config['decay_projection_sha256']) != (r['decay_on'], r['decay_projection_sha256']):
                raise ValueError(f'{run_dir.name}: decay settings differ from the plan')
            with np.load(ROOT/r['decay_projection']) as z:
                if not (np.array_equal(z['k'], k_ref) and np.array_equal(z['live'], live)):
                    raise ValueError(f'{run_dir.name}: projection file differs from the update-0 reference')
        manifest, paths = checkpoints(run_dir, local_dir)
        gates100 = e3.gates(load((ref_paths if 'resume' in r else paths)[REFERENCE_STEP]), d)[:, live]
        arm = 'keep' if '-keep-' in r['run_id'] else 'filt'
        for step, path in paths.items():
            rows.append(dict(source=source, run_id=r['run_id'], arm=arm, wd=r['wd'], decay_on=r.get('decay_on', ''),
                             switch_step=r.get('resume_step', ''), init_seed=r['init_seed'], step=step,
                             **measures(load(path), d, k_ref, live, sets, gates100)))
        inputs.append(dict(source=source, run_id=r['run_id'], resume_of=r.get('resume_of', ''),
                           checkpoints=list(paths), live_units=int(live.sum()),
                           producing_code_commit=manifest['producing_code_commit'],
                           producing_tree_dirty=manifest['producing_tree_dirty']))
    return rows, inputs


def sources(local):
    e3_plan = {r['run_id']: r for r in e3.entries()}
    e4 = json.loads((ROOT/'research/e4/plan.json').read_text())
    e4_plan = {r['run_id']: {**e4['common'], **r} for r in e4['runs']}
    out = [('e3', ROOT/'research/e3/mechanisms/runs'/pattern.format(s=s), None, e3_plan[pattern.format(s=s)])
           for pattern in RUNS.values() for s in SOURCES]
    for source, patterns in (('replay', REPLAYS.values()), ('intervention', INTERVENTION)):
        for pattern in patterns:
            for s in SOURCES:
                run_id = pattern.format(s=s)
                if (HERE/'runs'/run_id).exists():
                    out.append((source, HERE/'runs'/run_id, local/run_id, e4_plan[run_id]))
    for u in (300, 1000):  # every checkpoint of these runs is published
        for s in SOURCES:
            run_id = OFFSWITCH.format(u=u, s=s)
            out.append(('offswitch', ROOT/'research/e4/dynamics/runs'/run_id, None, e4_plan[run_id]))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--local', type=Path, default=ROOT/'runs/e4', help='Local root of the replay run directories')
    ap.add_argument('--out', type=Path, default=HERE/'results')
    args = ap.parse_args()
    git = lambda *a: subprocess.run(['git', *a], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    commit, status = git('rev-parse', 'HEAD'), git('status', '--porcelain')  # before any output is written
    rows, inputs = analyse(sources(args.local))
    worst = max(max(r['channel_identity_rel_err'], r['alignment_identity_rel_err'], r['weight_parts_identity_rel_err'])
                for r in rows)
    if worst > 1e-9:
        raise AssertionError(f'decomposition identity off by {worst:g}')
    args.out.mkdir(parents=True, exist_ok=True)
    e3.write_csv(args.out/'channels.csv', rows)
    (args.out/'channels_inputs.json').write_text(json.dumps(dict(
        runs=inputs, local_root=str(args.local), reference_step=REFERENCE_STEP,
        max_identity_rel_err=worst, analysis_code_commit=commit, analysis_tree_status=status,
        analysis_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        command=' '.join(['python', *sys.argv])), indent=1) + '\n')
    print(len(rows), 'rows; max identity error', worst)


if __name__ == '__main__':
    main()
