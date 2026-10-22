#!/usr/bin/env python3
"""Start files for the e4 audit control C1 (research/e4/audit/QUESTION.md, DECISIONS D11):
random features through the e3 B protocol. For each init seed of B's three sources:

1. U, V: experiment.init for that seed (untrained). They are checked equal to every published
   step-0 checkpoint of that init at width 128 that load_run accepts (manifest, split and
   logged-metric checks): B's source run where its step 0 is published (30000, 30001; it is
   missing for 30002) and the e1 no-decay run, whose step 0 e2 used;
2. W: closed-form ridge readout on relu(U[a] + V[b]) over the 299 training pairs, fitted to
   one-hot a+b minus 1/p, lambda = 1e-6 * trace(G)/128 (e2 audit cycle 3, its first value);
3. the e2 filter exactly as edits.py made B: filter_units(w, dominant(w)[0]);
4. U and V scaled by one factor to B's ||U||^2 + ||V||^2, W by another to B's ||W||. With no
   biases, logits scale by the product, so argmax, ranks and sigma-margins are unchanged
   (checked below). The point is to match Adam's step size relative to the weights.

Writes starts/<run_id>.npz (U, V, W, float64) and starts/update0.{csv,json}.

uv run --locked python research/e4/audit/rf_starts.py
"""
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'research/mechanisms'))
sys.path.insert(0, str(ROOT/'research/e2/mechanisms'))
sys.path.insert(0, str(ROOT/'research/e3/audit'))
import experiment  # noqa: E402
from freq import load_run  # noqa: E402
from unitfilter import dominant, filter_units  # noqa: E402
from units import concentration  # noqa: E402
import pairwise  # noqa: E402

P, RIDGE = 31, 1e-6
OUT = Path(__file__).parent/'starts'
COMMON = ROOT/'research/e2/data/common-never-trained-30-50.npz'
E2_FILTERED_COMMON = {30000: .195, 30001: .143, 30002: .277}  # e2 audit cycle 3, r = 1e-6
sha = lambda f: hashlib.sha256(Path(f).read_bytes()).hexdigest()


def logits(w, d):
    return np.maximum(w[0][d['a']] + w[1][d['b']], 0) @ w[2]


def measures(w, d, common):
    """Update-0 scores: accuracies, train cross-entropy, and the D5 scale-free measures on the common pairs."""
    z = logits(w, d)
    acc = lambda ix, y: float(np.mean(z[ix].argmax(1) == y[ix]))
    tr = d['train']
    x = z[tr] - z[tr].max(1, keepdims=True)
    S = pairwise.operand_labels(d['a'], d['b'], tr, P)
    zc, a, b, y = z[common], d['a'][common], d['b'][common], d['yadd'][common]
    right = np.eye(P, dtype=bool)[y]
    s_set = (S[a] | S[b]) & ~right
    n_set = ~(S[a] | S[b]) & ~right
    wrong = zc[~right].reshape(len(common), P - 1)
    margin = (zc[right] - np.where(n_set, zc, -np.inf).max(1))/wrong.std(1)
    pref = lambda A, B: float(np.nanmean(pairwise.preference(zc, A, B)[0]))
    return dict(common_acc=acc(common, d['yadd']), own_heldout_acc=acc(np.concatenate([d['cal'], d['audit']]), d['yadd']),
                train_acc=acc(tr, d['y']), train_ce=float(np.mean(np.log(np.exp(x).sum(1)) - x[np.arange(len(tr)), d['y'][tr]])),
                P_yN=pref(right, n_set), P_yS=pref(right, s_set), P_SN=pref(s_set, n_set),
                margin_N_sigma=float(np.median(margin)))


def main():
    e3 = {r['run_id']: r for r in json.loads((ROOT/'research/e3/plan.json').read_text())['runs']}
    e4 = json.loads((ROOT/'research/e4/plan.json').read_text())
    common = np.load(COMMON)['index']
    OUT.mkdir(exist_ok=True)
    rows, checks = [], {}
    for seed in (30000, 30001, 30002):
        b_entry = e3[f'e3-me-filt-w128-i{seed}-a1']
        b_file = ROOT/f'research/e3/mechanisms/starts/e3-me-filt-w128-i{seed}-a1.npz'
        assert sha(b_file) == b_entry['init_weights_sha256'], b_file
        with np.load(b_file) as f:
            wb = [f[k] for k in 'UVW']
        src = ROOT/b_entry['source_checkpoint']
        _, d, meta = load_run(src.parent, 50000)
        assert json.loads((src.parent/'config.json').read_text())['init_seed'] == seed
        assert meta['split_sha256'] == e4['common']['split_sha256'] == b_entry['split_sha256']
        assert np.array_equal(d['y'], d['yadd']) and len(d['train']) == 299
        assert not set(common.tolist()) & set(d['train'].tolist())

        U, V = experiment.init(P, 128, seed)[:2]
        step0 = [r for r in (src.parent, ROOT/f'research/size-time/runs/e1-st-add0-w128-i{seed}-a1')
                 if (r/'checkpoint_000000.npz').exists()]
        for r in step0:
            w0, d0, m0 = load_run(r, 0)
            assert np.array_equal(w0[0], U) and np.array_equal(w0[1], V), r
            assert m0['split_sha256'] == meta['split_sha256'] and all(np.array_equal(d0[q], d[q]) for q in d)
        tr = d['train']
        H = np.maximum(U[d['a'][tr]] + V[d['b'][tr]], 0)
        G = H.T @ H
        W = np.linalg.solve(G + RIDGE*np.trace(G)/len(G)*np.eye(len(G)), H.T @ (np.eye(P)[d['yadd'][tr]] - 1/P))
        ridge = [U, V, W]
        k = dominant(ridge)[0]
        filt = filter_units(ridge, k)
        s = np.sqrt((np.sum(wb[0]**2) + np.sum(wb[1]**2))/(np.sum(filt[0]**2) + np.sum(filt[1]**2)))
        t = np.linalg.norm(wb[2])/np.linalg.norm(filt[2])
        w = [np.ascontiguousarray(x, dtype=np.float64) for x in (s*filt[0], s*filt[1], t*filt[2])]

        m_ridge, m_filt, m = measures(ridge, d, common), measures(filt, d, common), measures(w, d, common)
        z_f, z = logits(filt, d), logits(w, d)
        pre_r, pre_f = U[d['a']] + V[d['b']], filt[0][d['a']] + filt[1][d['b']]
        run_id = f'e4-au-rf-w128-i{seed}-a1'
        checks[run_id] = dict(
            reproduces_e2_filtered_common=round(m_filt['common_acc'], 3) == E2_FILTERED_COMMON[seed],
            argmax_unchanged_by_scaling_all_961=bool(np.array_equal(z_f.argmax(1), z.argmax(1))),
            max_rel_logit_error_vs_s_t_times_filtered=float(np.abs(z - s*t*z_f).max()/np.abs(z).max()),
            scale_free_measures_max_abs_change=max(abs(m[q] - m_filt[q]) for q in ('common_acc', 'own_heldout_acc', 'train_acc',
                                                                                    'P_yN', 'P_yS', 'P_SN', 'margin_N_sigma')),
            norms_match_B=bool(np.isclose(np.sum(w[0]**2) + np.sum(w[1]**2), np.sum(wb[0]**2) + np.sum(wb[1]**2), rtol=1e-12)
                               and np.isclose(np.linalg.norm(w[2]), np.linalg.norm(wb[2]), rtol=1e-12)))
        path = OUT/f'{run_id}.npz'
        np.savez(path, U=w[0], V=w[1], W=w[2])
        rows.append(dict(run_id=run_id, init_seed=seed, source_run=meta['run_id'], U_V_equal_to_step0_of=[r.name for r in step0],
                         B_start_sha256=b_entry['init_weights_sha256'], start_file=str(path.relative_to(ROOT)), start_sha256=sha(path),
                         ridge_r=RIDGE, scale_UV=float(s), scale_W=float(t),
                         input_concentration_ridge=concentration(ridge)['input_concentration'],
                         kept_energy_U=float(np.sum(filt[0]**2)/np.sum(U**2)), kept_energy_W=float(np.sum(filt[2]**2)/np.sum(W**2)),
                         gate_flips_vs_ridge=float(np.mean((pre_f > 0) != (pre_r > 0))),
                         norm_UV=float(np.sqrt(np.sum(w[0]**2) + np.sum(w[1]**2))), norm_W=float(np.linalg.norm(w[2])),
                         **{f'ridge_{q}': v for q, v in m_ridge.items()}, **{f'start_{q}': v for q, v in m.items()},
                         **{f'B_{q}': v for q, v in measures(wb, d, common).items()}))
    with open(OUT/'update0.csv', 'w', newline='') as f:
        out = csv.DictWriter(f, list(rows[0]))
        out.writeheader()
        out.writerows(rows)
    git = lambda *a: subprocess.run(['git', *a], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    (OUT/'update0.json').write_text(json.dumps(dict(rows=rows, checks=checks, common_set_sha256=sha(COMMON),
                                                    analysis_code_commit=git('rev-parse', 'HEAD'),
                                                    analysis_tree_status=git('status', '--porcelain'),
                                                    command=' '.join(['python', *sys.argv])), indent=1) + '\n')
    print(json.dumps(checks, indent=1))
    for r in rows:
        print(r['run_id'], r['start_sha256'], {q: round(r[q], 4) for q in r if q.startswith(('start_', 'B_')) and isinstance(r[q], float)})
    if not all(all(v if isinstance(v, bool) else v < 1e-12 for v in c.values()) for c in checks.values()):
        sys.exit('a start-file check failed')


if __name__ == '__main__':
    main()
