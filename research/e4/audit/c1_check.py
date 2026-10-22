#!/usr/bin/env python3
"""e4 audit control C1 (research/e4/audit/QUESTION.md, section 1): the six random-feature runs
against B (e3-me-filt[-wd0]-w128-i<seed>-a1), with the pre-stated rule.

- provenance: config against plan and start file, step-0 weights against the start file,
  registry events, completed steps, producing commit;
- every published checkpoint scored with my forward pass (rf_starts.measures: common and own
  held-out accuracy, train accuracy and cross-entropy, P measures, sigma-margin);
- own held-out accuracy every 10 updates from metrics.jsonl (cal and audit pooled by size);
- rise R = common accuracy at update 100 minus update 0 (wd 1 runs), R_B the same for B;
- the dynamics lane's decline times (peak = max own held-out over 0-1,000; first logged update
  after it below theta, theta = own update-0 accuracy and 0.10; censored if none; re-crossings).

uv run --locked python research/e4/audit/c1_check.py OUT_DIR
"""
import json
from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from rf_starts import COMMON, ROOT, measures, sha  # noqa: E402

SEEDS = (30000, 30001, 30002)
AU, ME = HERE/'runs', ROOT/'research/e3/mechanisms/runs'
PLAN = {r['run_id']: r for r in json.loads((ROOT/'research/e4/plan.json').read_text())['runs']}
REG = [json.loads(l) for l in (HERE/'registry.jsonl').read_text().splitlines() if l.strip()]


def rf(seed, wd):
    return f"e4-au-rf{'-wd0' if wd == 0 else ''}-w128-i{seed}-a1"


def b(seed, wd):
    return f"e3-me-filt{'-wd0' if wd == 0 else ''}-w128-i{seed}-a1"


def load_dir(run_dir):
    with np.load(run_dir/'data.npz') as z:
        d = {k: z[k] for k in ('a', 'b', 'y', 'yadd', 'train', 'cal', 'audit')}
    d = {k: (np.flatnonzero(v) if k in ('train', 'cal', 'audit') and v.dtype == bool else v) for k, v in d.items()}
    steps = sorted(int(f.stem.split('_')[1]) for f in run_dir.glob('checkpoint_*.npz'))
    return d, steps


def weights(run_dir, step):
    with np.load(run_dir/f'checkpoint_{step:06d}.npz') as z:
        return [z[k] for k in 'UVW']


def dense(run_dir, d):
    n_cal, n_aud = len(d['cal']), len(d['audit'])
    rows = [json.loads(l) for l in (run_dir/'metrics.jsonl').read_text().splitlines()]
    return {r['step']: dict(own=(r['cal']*n_cal + r['audit']*n_aud)/(n_cal + n_aud), train=r['train']) for r in rows}


def decline(acc):
    steps = sorted(acc)
    peak = max((s for s in steps if s <= 1000), key=lambda s: (acc[s], -s))
    out = dict(own_update0=acc[0], peak=acc[peak], peak_step=peak, last_step=steps[-1])
    for name, theta in (('update0', acc[0]), ('fixed010', .10)):
        after = [s for s in steps if s > peak]
        below = [s for s in after if acc[s] < theta]
        t = below[0] if below else None
        out[f'T_{name}'] = t if t is not None else f'> {steps[-1]}'
        out[f'recross_{name}'] = 0 if t is None else sum(1 for s0, s1 in zip(after, after[1:]) if s0 >= t and acc[s0] < theta <= acc[s1])
    return out


def provenance(run_id, d, steps):
    run_dir, e = AU/run_id, PLAN[run_id]
    cfg = json.loads((run_dir/'config.json').read_text())
    man = json.loads((run_dir/'manifest.json').read_text())
    rt = json.loads((run_dir/'runtime.json').read_text())
    start = ROOT/e['init_weights']
    with np.load(start) as z:
        same0 = all(np.array_equal(z[k], x) for k, x in zip('UVW', weights(run_dir, 0)))
    hashes_ok = all(sha(run_dir/f) == v['sha256'] for f, v in man['files'].items() if (run_dir/f).exists())
    out = dict(init_weights_sha_config=cfg.get('init_weights_sha256'), init_weights_sha_plan=e['init_weights_sha256'],
               start_file_sha=sha(start), step0_equals_start_file=bool(same0), wd=cfg['wd'], plan_wd=e['wd'],
               lr=cfg['lr'], steps=cfg['steps'], init_age=cfg.get('init_age'), resume=cfg.get('resume'),
               completed=rt['completed_steps'], events=[r['status'] for r in REG if r['run_id'] == run_id],
               producing_code_commit=man['producing_code_commit'][:7], published_hashes_match_manifest=bool(hashes_ok),
               split_train_299=len(d['train']) == 299, checkpoints=steps)
    out['PASS'] = bool(out['init_weights_sha_config'] == out['init_weights_sha_plan'] == out['start_file_sha'] and same0
                       and out['wd'] == out['plan_wd'] and out['lr'] == .001 and out['steps'] == out['completed'] == 20000
                       and out['init_age'] == 0 and not out['resume'] and out['events'] == ['running', 'complete'] and hashes_ok
                       and out['split_train_299'])
    return out


def main(out_dir):
    common = np.load(COMMON)['index']
    res = dict(runs={}, rule={}, provenance={})
    for seed in SEEDS:
        for wd in (1., 0.):
            arms = [('rf', AU/rf(seed, wd)), ('B', ME/b(seed, wd))] + ([('A', ME/f'e3-me-keep-w128-i{seed}-a1')] if wd else [])
            for arm, run_dir in arms:  # A (unedited, wd 1): a reference for the figure, not in the rule
                d, steps = load_dir(run_dir)
                assert np.array_equal(d['y'], d['yadd']) and not set(common.tolist()) & set(d['train'].tolist())
                ck = {s: measures(weights(run_dir, s), d, common) for s in steps}
                dn = dense(run_dir, d)
                t_fit = min(s for s, r in dn.items() if r['train'] == 1.)
                res['runs'][run_dir.name] = dict(arm=arm, seed=seed, wd=wd, checkpoints=ck, t_fit=t_fit,
                                                 first_fit_checkpoint=min(s for s in steps if ck[s]['train_acc'] >= .99),
                                                 dense_own_heldout={s: r['own'] for s, r in dn.items()},
                                                 dense_train={s: r['train'] for s, r in dn.items()},
                                                 decline=decline({s: r['own'] for s, r in dn.items()}))
                if arm == 'rf':
                    res['provenance'][run_dir.name] = provenance(run_dir.name, d, steps)
                    logged0 = json.loads((run_dir/'metrics.jsonl').read_text().splitlines()[0])
                    res['provenance'][run_dir.name]['step0_logged_vs_mine_max_abs'] = max(
                        abs(logged0['train'] - ck[0]['train_acc']),
                        abs((logged0['cal']*len(d['cal']) + logged0['audit']*len(d['audit']))/(len(d['cal']) + len(d['audit'])) - ck[0]['own_heldout_acc']))
    R = res['runs']
    for seed in SEEDS:
        r1, r0, b1 = R[rf(seed, 1.)], R[rf(seed, 0.)], R[b(seed, 1.)]
        rise_rf = r1['checkpoints'][100]['common_acc'] - r1['checkpoints'][0]['common_acc']
        rise_b = b1['checkpoints'][100]['common_acc'] - b1['checkpoints'][0]['common_acc']
        rise = 'generic' if rise_rf >= rise_b/2 else ('inconclusive' if rise_rf >= .1 else 'specific')
        collapse = all(isinstance(r1['decline'][f'T_{n}'], int) for n in ('update0', 'fixed010'))
        hold = not isinstance(r0['decline']['T_update0'], int)
        fit = r1['first_fit_checkpoint']
        res['rule'][seed] = dict(
            rise_rf=rise_rf, rise_B=rise_b, half_rise_B=rise_b/2, rise=rise, collapse_wd1=collapse, hold_wd0=hold,
            T_rf_wd1={n: r1['decline'][f'T_{n}'] for n in ('update0', 'fixed010')},
            T_rf_wd0={n: r0['decline'][f'T_{n}'] for n in ('update0', 'fixed010')},
            reported=dict(rise_rf_own_heldout_0_100=r1['dense_own_heldout'][100] - r1['dense_own_heldout'][0],
                          rise_B_own_heldout_0_100=b1['dense_own_heldout'][100] - b1['dense_own_heldout'][0],
                          rf_first_fit_checkpoint=fit, rf_t_fit=r1['t_fit'],
                          rise_rf_common_0_first_fit=r1['checkpoints'][fit]['common_acc'] - r1['checkpoints'][0]['common_acc'],
                          rf_wd0_common_100=r0['checkpoints'][100]['common_acc']))
    rules = res['rule'].values()
    res['verdict'] = ('G SUPPORTED LOCALLY' if all(r['rise'] == 'generic' and r['collapse_wd1'] and r['hold_wd0'] for r in rules)
                      else 'F SUPPORTED LOCALLY' if all(r['rise'] == 'specific' for r in rules) else 'AMBIGUOUS')
    res['provenance_all_PASS'] = all(p['PASS'] for p in res['provenance'].values())
    Path(out_dir, 'c1-check.json').write_text(json.dumps(res, indent=1, default=str) + '\n')
    print(json.dumps(dict(provenance=res['provenance_all_PASS'], rule=res['rule'], verdict=res['verdict'],
                          decline={k: v['decline'] for k, v in R.items()}, t_fit={k: v['t_fit'] for k, v in R.items()}), indent=1, default=str))


if __name__ == '__main__':
    main(sys.argv[1])
