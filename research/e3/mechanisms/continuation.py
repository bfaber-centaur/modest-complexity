#!/usr/bin/env python3
"""e3 mechanisms: what continued training does after the edit (QUESTION.md; DECISIONS D5, D6).

Rows, for each continuation run in the plan (arms keep / filt / sham x three source checkpoints at
wd 1, and filt / sham at wd 0 for follow-up 1; `branch` names arm and decay, e.g. filt-wd0),
on every published checkpoint, plus one row per run for its source checkpoint at 50,000 updates
(phase "source", before the edit):
- train cross-entropy and accuracy (299 pairs), accuracy on the run's own cal + audit;
- on the 483 common never-trained pairs (research/e2/data/common-never-trained-30-50.npz):
  accuracy and cross-entropy, and the D6 measures from research/e3/audit/pairwise.py with the
  e2 ladder's S, D, N (30% and 50% training sets): P(S over N), P(a+b over N), P(a+b over S),
  median (a+b - max N)/sigma and (max S - max N)/sigma, sigma = std of the 30 wrong-label logits;
- frequency structure: research/mechanisms/units.py `concentration`, the mean share of each live
  unit's non-DC input energy at its source-checkpoint dominant frequency, and the fraction of live
  units whose dominant input frequency is the one they had at update 0;
- the weight norm over U, V, W together, and the relative distance ||w - w0|| / ||w0||, from the start file (update 0) and
  from the source checkpoint, and the fraction of the 961 x 128 ReLU gates that differ from update 0.
Dense rows (every 10 updates) are each run's metrics.jsonl, unchanged.

`step` counts continuation updates from 0; the checkpoint age (50,000) is the separate column
`init_age`. Every run is checked before use: published files against their manifest, config
against the plan (start-file hash, age, recipe), step-0 weights equal to the start file, and the
step-0 logged metrics against a replay.

uv run --locked --group analysis python research/e3/mechanisms/continuation.py [--runs DIR] [--out DIR]
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
for d in ('research/mechanisms', 'research/e3/audit', 'research/e2/mechanisms', 'research/e2/audit'):
    sys.path.insert(0, str(ROOT/d))
from freq import evaluate, load_run, load_weights, sha256  # noqa: E402
from units import concentration, unit_shares  # noqa: E402
from gauge import live_units  # noqa: E402
from unitfilter import dominant  # noqa: E402
from pairwise import preference, ladder_sets  # noqa: E402
from margins_check import tr30, tr50  # noqa: E402  (frozen e1 30% and e1f2 50% training sets)

P = 31
HERE = ROOT/'research/e3/mechanisms'
COMMON = np.load(ROOT/'research/e2/data/common-never-trained-30-50.npz')['index']
ARMS = ('keep', 'filt', 'sham')


def entries():
    plan = json.loads((ROOT/'research/e3/plan.json').read_text())
    return [r for r in plan['runs'] if r['run_id'].startswith('e3-me-')]


def gates(w, d):
    return w[0][d['a']] + w[1][d['b']] > 0


def distance(w, ref):
    return float(np.sqrt(sum(np.sum((x - y)**2) for x, y in zip(w, ref))/sum(np.sum(y**2) for y in ref)))


def common_measures(w, d, sets):
    S, D, N, y = sets
    ca, cb = d['a'][COMMON], d['b'][COMMON]
    z = np.maximum(w[0][ca] + w[1][cb], 0) @ w[2]
    rows = np.arange(len(y))
    x = z - z.max(1, keepdims=True)
    wrong = np.ones_like(z, bool)
    wrong[rows, y] = False
    sigma = z[wrong].reshape(len(y), P - 1).std(1)
    mx = lambda m: np.where(m, z, -np.inf).max(1)
    out = dict(common_acc=float(np.mean(z.argmax(1) == y)),
               common_ce=float(np.mean(np.log(np.exp(x).sum(1)) - x[rows, y])))
    C = np.eye(P, dtype=bool)[y]
    for name, (A, B) in dict(P_SN=(S, N), P_cN=(C, N), P_cS=(C, S)).items():
        v, ties = preference(z, A, B)
        out[name] = float(np.nanmean(v))
        out[f'{name}_undefined'] = int(np.isnan(v).sum())
        out[f'{name}_ties'] = ties
    out.update(margin_N_sigma=float(np.median((z[rows, y] - mx(N))/sigma)),
               gap_SN_sigma=float(np.median((mx(S) - mx(N))/sigma)), median_sigma=float(np.median(sigma)))
    return out


def structure(w, k_source, k_start):
    live = live_units(w)
    with np.errstate(invalid='ignore'):
        s_in = unit_shares(w)[0]
    units = np.arange(s_in.shape[1])
    c = concentration(w)
    return dict(live_units=c['live_units'], input_concentration=c['input_concentration'],
                output_concentration=c['output_concentration'], in_out_agreement=c['in_out_agreement'],
                frequencies_used=c['frequencies_used'],
                share_at_source_dominant=float(s_in[k_source - 1, units][live].mean()),
                dominant_as_at_update0=float(np.mean((s_in.argmax(0) + 1 == k_start)[live])))


def check_run(run_dir, r):
    """Manifest, plan and start-file checks; returns (step-0 meta, data, start weights)."""
    config = json.loads((run_dir/'config.json').read_text())
    start = ROOT/r['init_weights']
    for key, want in (('init_weights_sha256', r['init_weights_sha256']), ('init_age', r['init_age']),
                      ('steps', r['steps']), ('wd', r['wd']), ('hidden', r['width']), ('lr', .001)):
        if config.get(key) != want:
            raise ValueError(f'{run_dir.name}: config {key} = {config.get(key)!r}, plan {want!r}')
    if sha256(start) != r['init_weights_sha256']:
        raise ValueError(f'{start}: hash differs from the plan')
    w0, d, meta = load_run(run_dir, 0)  # manifest hashes, split hash, step-0 replay of the logged row
    if meta['split_sha256'] != r['split_sha256'] or meta['max_abs_diff_vs_logged_metrics'] > 1e-12:
        raise ValueError(f'{run_dir.name}: split or step-0 replay mismatch {meta}')
    with np.load(start) as z:
        ws = [z[k] for k in 'UVW']
    if not all(np.array_equal(x, y) for x, y in zip(w0, ws)):
        raise ValueError(f'{run_dir.name}: step-0 weights differ from the start file')
    if set(COMMON.tolist()) & set(d['train'].tolist()):
        raise ValueError('common pairs overlap training pairs')
    return meta, d, ws


def published_steps(run_dir):
    manifest = json.loads((run_dir/'manifest.json').read_text())
    return manifest, sorted(int(n[11:17]) for n, f in manifest['files'].items()
                            if n.startswith('checkpoint_') and f['published'])


def analyse(runs_root):
    rows, dense, inputs = [], [], []
    sets = None
    for r in entries():
        run_dir = runs_root/r['run_id']
        meta, d, ws = check_run(run_dir, r)
        if sets is None:
            sets = ladder_sets(d['a'], d['b'], tr30, tr50, COMMON, P)
            assert np.array_equal(d['train'], tr30)
        src_path = ROOT/r['source_checkpoint']
        if sha256(src_path) != r['source_checkpoint_sha256']:
            raise ValueError(f'{src_path}: hash differs from the plan')
        with np.load(src_path) as z:
            wsrc = [z[k] for k in 'UVW']
        k_source, k_start = dominant(wsrc)[0], dominant(ws)[0]
        arm = r['run_id'].split('-')[2]
        base = dict(run_id=r['run_id'], arm=arm, wd=r['wd'], branch=arm + ('-wd0' if r['wd'] == 0 else ''),
                    source_run=r['source_run'], init_seed=r['init_seed'], init_age=r['init_age'])
        manifest, steps = published_steps(run_dir)
        g0 = gates(ws, d)
        for phase, step, w in [('source', '', wsrc)] + [('continuation', s, None) for s in steps]:
            if w is None:
                w = load_weights(run_dir, step)
            m, _ = evaluate(w, d)
            rows.append(dict(base, phase=phase, step=step, train_ce=m['train_cross_entropy'], train_acc=m['train_accuracy'],
                             own_heldout_acc=own_heldout(w, d), **common_measures(w, d, sets), **structure(w, k_source, k_start),
                             norm=float(np.sqrt(sum(np.sum(x**2) for x in w))),
                             dist_from_update0=distance(w, ws), dist_from_source=distance(w, wsrc),
                             gates_changed_from_update0=float(np.mean(gates(w, d) != g0))))
        for line in (run_dir/'metrics.jsonl').read_text().splitlines():
            x = json.loads(line)
            dense.append(dict(base, step=x['step'], train_ce=x['objective']['train']['cross_entropy'], train_acc=x['train'],
                              cal_acc=x['cal'], audit_acc=x['audit'], norm=x['norm']))
        runtime = json.loads((run_dir/'runtime.json').read_text())
        inputs.append(dict(run_id=r['run_id'], status=manifest['status'], completed_steps=manifest['completed_steps'],
                           wall_seconds=runtime.get('wall_seconds'), producing_code_commit=manifest['producing_code_commit'],
                           producing_tree_dirty=manifest['producing_tree_dirty'], init_weights_sha256=r['init_weights_sha256'],
                           step0_replay_max_abs_diff=meta['max_abs_diff_vs_logged_metrics'], checkpoints=steps))
    return rows, dense, inputs


def own_heldout(w, d):
    ix = np.concatenate([d['cal'], d['audit']])
    z = np.maximum(w[0][d['a'][ix]] + w[1][d['b'][ix]], 0) @ w[2]
    return float(np.mean(z.argmax(1) == d['yadd'][ix]))


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows({k: (float(f'{v:.10g}') if isinstance(v, float) else v) for k, v in r.items()} for r in rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--runs', type=Path, default=HERE/'runs')
    ap.add_argument('--out', type=Path, default=HERE/'results')
    args = ap.parse_args()
    rows, dense, inputs = analyse(args.runs)
    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out/'checkpoints.csv', rows)
    write_csv(args.out/'dense.csv', dense)
    git = lambda *a: subprocess.run(['git', *a], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    (args.out/'inputs.json').write_text(json.dumps(dict(
        runs=inputs, runs_root=str(args.runs.resolve().relative_to(ROOT)),
        common_set_sha256=hashlib.sha256((ROOT/'research/e2/data/common-never-trained-30-50.npz').read_bytes()).hexdigest(),
        analysis_code_commit=git('rev-parse', 'HEAD'), analysis_tree_status=git('status', '--porcelain'),
        analysis_script_sha256=sha256(__file__), command=' '.join(['python', *sys.argv])), indent=1) + '\n')
    print(len(rows), 'checkpoint rows,', len(dense), 'dense rows ->', args.out)


if __name__ == '__main__':
    main()
