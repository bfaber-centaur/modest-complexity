#!/usr/bin/env python3
"""Software checks of the dynamics lane's six nested-share runs (QUESTION-4.md, part A),
with the e1 audit's independent code (research/audit/independent.py, audit_preview.py).

- splits at 30, 40, 45 and 50% regenerated from the written rule; hashes against the e1
  frozen files and research/e2/plan.json; nesting and the common set;
- audit_preview.py's per-run checks (claim, config, manifest, registry, logging, every
  published checkpoint recomputed) with replays from an independent initialization
  through first fit to 5,000 updates (`early`);
- independent AdamW from the published 40,000 checkpoint to 50,000 (`late`), every
  logged row compared, and the state at 50,000.

uv run --locked python research/e2/audit/runs_check.py OUT_DIR {splits|early|late} [SHARE ...]
"""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import numpy as np

root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(root/'research/audit'))
import independent as ind  # noqa: E402
import audit_preview as ap  # noqa: E402

P, LAUNCH = 31, 'dbfcf2148705e3e81fa91dbec929157ddab8d592'
PLAN = json.loads(subprocess.run(['git', 'show', f'{LAUNCH}:research/e2/plan.json'], capture_output=True, check=True,
                                 cwd=root).stdout)
TARGETS = json.loads((root/'research/plan.json').read_text())['targets']
DYN = Path('research/e2/dynamics')  # relative: run from the repository root, as audit_preview.py needs
COMMON = np.load(root/'research/e2/data/common-never-trained-30-50.npz')['index']
IDS = {s: [r['run_id'] for r in PLAN['runs'] if r['train_fraction'] == s] for s in (.4, .45)}


def split(frac):
    c = PLAN['common']
    a, b, y, _, (tr, cal, au) = ind.regenerate_split(P, c['data_seed'], c['control_seed'], frac)
    return dict(a=a, b=b, yadd=y, train=tr, cal=cal, audit=au)


def groups(ix, s):
    return {(min(int(s['a'][i]), int(s['b'][i])), max(int(s['a'][i]), int(s['b'][i]))) for i in ix}


def check_splits():
    sp = {f: split(f) for f in (.3, .4, .45, .5)}
    h = {f: ind.hash_arrays(*(sp[f][k] for k in ('a', 'b', 'yadd', 'train', 'cal', 'audit'))) for f in sp}
    out = dict(hash=h)
    for f, name in ((.3, 'e1-split.npz'), (.5, 'e1f2-split-tf50.npz')):
        with np.load(root/'research/data'/name) as z:
            out[f'regenerated_{f}_equals_{name}'] = all(np.array_equal(sp[f][k], z[k]) for k in sp[f])
    claimed = {r['train_fraction']: r['split_sha256'] for r in PLAN['runs']}
    out['hash_matches_claim'] = {f: h[f] == claimed[f] for f in (.4, .45)}
    tr = {f: set(sp[f]['train'].tolist()) for f in sp}
    out['nested_30_40_45_50'] = tr[.3] < tr[.4] < tr[.45] < tr[.5]
    out['train_sizes'] = {f: len(tr[f]) for f in sp}
    out['split_sizes'] = {f: [len(sp[f][k]) for k in ('train', 'cal', 'audit')] for f in sp}
    out['common_untrained_at_every_share'] = all(not (tr[f] & set(COMMON.tolist())) for f in sp)
    out['common_equals_40_audit'] = set(COMMON.tolist()) == set(sp[.4]['audit'].tolist())
    out['common_minus_45_audit_groups'] = len(groups(sorted(set(COMMON.tolist()) - set(sp[.45]['audit'].tolist())), sp[.45]))
    out['common_minus_45_audit_in_45_cal'] = set(COMMON.tolist()) - set(sp[.45]['audit'].tolist()) <= set(sp[.45]['cal'].tolist())
    out['groups_per_class'] = {f: [sorted(set(np.bincount([(u + v) % P for u, v in groups(sp[f][k], sp[f])], minlength=P).tolist()))
                                   for k in ('train', 'cal', 'audit')] for f in sp}
    e1 = sp[.3]
    out['added_over_30_from_e1'] = {f: dict(cal=len((tr[f] - tr[.3]) & set(e1['cal'].tolist())),
                                            audit=len((tr[f] - tr[.3]) & set(e1['audit'].tolist()))) for f in (.4, .45, .5)}
    return out


def early(frac, out_json):
    """audit_preview.py on one share with a plan built from the launch commit's claims."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp/'plan.json').write_text(json.dumps({**PLAN, 'targets': TARGETS}))
        np.savez(tmp/'split.npz', **split(frac))
        sys.argv = ['audit_preview.py', '--published', str(DYN), '--plan', str(tmp/'plan.json'), '--frozen', str(tmp/'split.npz'),
                    '--code-commit', LAUNCH, '--ids', *IDS[frac], '--replay', '--replay-max-steps', '5000', '--out', str(out_json)]
        ap.main()


def late(frac, start=40000, stop=50000):
    sp = split(frac); c = PLAN['common']; out = {}
    for run_id in IDS[frac]:
        rd = DYN/'runs'/run_id
        rows = {z['step']: z for z in map(json.loads, (rd/'metrics.jsonl').read_text().splitlines())}
        t0 = time.perf_counter()
        state, worst, trace = ap.replay(ap.load_ck(rd/f'checkpoint_{start:06d}.npz'), start, stop, sp, rows, P, c['lr'], 1.0, sp['yadd'])
        target = ap.load_ck(rd/f'checkpoint_{stop:06d}.npz')
        got = state['params'] + state['m'] + state['v']
        out[run_id] = dict(start=start, stop=stop, seconds=round(time.perf_counter() - t0, 1),
                           state_rel_diff={k: ap.rel(x, target[k]) for x, k in zip(got, ap.KEYS)}, logged_rows=worst,
                           train_ce_rel_trace=trace[::10])
        print(run_id, out[run_id]['seconds'], 's', json.dumps(worst), flush=True)
    return out


if __name__ == '__main__':
    out, what = Path(sys.argv[1]), sys.argv[2]
    out.mkdir(parents=True, exist_ok=True)
    shares = [float(s) for s in sys.argv[3:]] or [.4, .45]
    if what == 'splits':
        (out/'splits.json').write_text(json.dumps(check_splits(), indent=1) + '\n')
    for s in shares:
        tag = f'tf{round(100*s)}'
        if what == 'early':
            early(s, out/f'{tag}-early.json')
        elif what == 'late':
            (out/f'{tag}-late.json').write_text(json.dumps(late(s), indent=1, default=float) + '\n')
