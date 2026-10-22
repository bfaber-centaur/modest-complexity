#!/usr/bin/env python3
"""Mechanisms gate, part 1 (research/e3/audit/QUESTION-2.md, C1-C6): the e3 continuation
code at the pin.

Builds an arm-A start file from a real source checkpoint, runs experiment.py for 300
updates with a reserved plan entry's settings in a scratch directory (a reproducibility
check, not a research run: no run ID, not published), and compares it with an
independent replay (research/audit/independent.py) with fresh and with kept Adam state.

uv run --locked python research/e3/audit/continuation_check.py SCRATCH_DIR OUT.json
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(root))
sys.path.insert(0, str(root/'research/audit'))
import launch_runs  # noqa: E402  (command() only)
from instrumentation import THREAD_VARS  # noqa: E402
import independent as ind  # noqa: E402

P, H, STEPS, CHECK = 31, 128, 300, (10, 30, 100, 300)
plan = json.loads((root/'research/e3/plan.json').read_text())
RESERVED = plan['reserved']
ENV = {**os.environ, **{k: '1' for k in THREAD_VARS}}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metric_rows(run):
    return {r['step']: r for r in map(json.loads, (run/'metrics.jsonl').read_text().splitlines())}


def flags(cmd):
    return {k: v for k, v in zip(cmd, cmd[1:]) if k.startswith('--')}


def replay(params, m, v, t0, steps, Xa, Xb, y, lr, wd, keep=()):
    params, m, v = [x.copy() for x in params], [x.copy() for x in m], [x.copy() for x in v]
    kept = {}
    for t in range(t0, t0 + steps):
        ind.adamw_step(params, ind.loss_and_grad(params, Xa, Xb, y)[1], m, v, t, lr, wd)
        if t - t0 + 1 in keep:
            kept[t - t0 + 1] = ([x.copy() for x in params], [x.copy() for x in m], [x.copy() for x in v])
    return kept


def main(scratch, out_json):
    scratch = Path(scratch)
    scratch.mkdir(parents=True, exist_ok=False)
    res = {}

    # C6: each arm of a source names the same checkpoint, and its hash and step are as claimed.
    by_source = {}
    for e in RESERVED:
        by_source.setdefault(e['source_run'], set()).add((e['source_checkpoint'], e['source_checkpoint_sha256']))
    c6 = {}
    for run, claims in by_source.items():
        (f, h), = claims
        with np.load(root/f) as z:
            c6[run] = dict(checkpoint=f, arms=sum(e['source_run'] == run for e in RESERVED),
                           sha256_matches=sha(root/f) == h, step=int(z['step']))
    res['C6'] = dict(sources=c6, PASS=len(c6) == 3 and all(
        v['arms'] == 3 and v['sha256_matches'] and v['step'] == 50000 for v in c6.values()))

    # C1-C4: a 300-update continuation of arm A for the first source, built by launch_runs.command.
    entry = RESERVED[0]
    assert entry['run_id'] == 'e3-me-keep-w128-i30000-a1'
    src_run = root/Path(entry['source_checkpoint']).parent
    with np.load(root/entry['source_checkpoint']) as z:
        src = {k: z[k] for k in z.files}
    start = scratch/'start-A.npz'
    np.savez(start, U=src['U'], V=src['V'], W=src['W'])
    trial = {**entry, 'run_id': 'audit-scratch-continuation-check', 'purpose': 'audit-check', 'steps': STEPS,
             'init_weights': str(start), 'init_weights_sha256': sha(start)}
    run = scratch/'run'
    cmd = launch_runs.command(plan, trial, run, 'e3-au-cloud-1')
    t = time.perf_counter()
    proc = subprocess.run(cmd, env=ENV, cwd=root, capture_output=True, text=True)
    res['continuation_seconds'] = round(time.perf_counter() - t, 2)
    assert proc.returncode == 0, proc.stderr[-2000:]
    ck = {s: dict(np.load(run/f'checkpoint_{s:06d}.npz')) for s in (0,) + CHECK}

    a, b, y, _, splits = ind.regenerate_split(P, entry['data_seed'], plan['common']['control_seed'], .3)
    with np.load(run/'data.npz') as d:
        same_split = all(np.array_equal(d[k], x) for k, x in zip(('a', 'b', 'yadd', 'train', 'cal', 'audit'), (a, b, y, *splits)))
    tr = splits[0]
    Xa, Xb = ind.one_hot_inputs(a[tr], b[tr], P)
    lr, wd = plan['common']['lr'], entry['wd']
    w0 = [src[k] for k in 'UVW']
    scale = max(float(np.abs(x).max()) for x in w0)
    fresh = replay(w0, [np.zeros_like(x) for x in w0], [np.zeros_like(x) for x in w0], 1, STEPS, Xa, Xb, y[tr], lr, wd, CHECK)
    diff = {s: max(float(np.abs(p - ck[s][k]).max()) for p, k in zip(fresh[s][0], 'UVW')) for s in CHECK}
    mdiff = max(float(np.abs(x - ck[STEPS][k]).max()) for x, k in zip(fresh[STEPS][1] + fresh[STEPS][2],
                                                                      ('mU', 'mV', 'mW', 'vU', 'vV', 'vW')))
    res['C1'] = dict(
        split_regenerated_equals_run=same_split,
        step0_weights_equal_source=all(np.array_equal(ck[0][k], src[k]) for k in 'UVW'),
        step0_moments_zero=not any(np.any(ck[0][k]) for k in ('mU', 'mV', 'mW', 'vU', 'vV', 'vW')),
        max_abs_weight=scale, tolerance=1e-9*scale,
        independent_fresh_replay_max_abs_diff={str(s): diff[s] for s in CHECK},
        independent_fresh_replay_moment_max_abs_diff_at_300=mdiff)
    res['C1']['PASS'] = bool(same_split and res['C1']['step0_weights_equal_source'] and res['C1']['step0_moments_zero']
                             and max(diff.values()) <= 1e-9*scale)

    kept = replay(w0, [src[k] for k in ('mU', 'mV', 'mW')], [src[k] for k in ('vU', 'vV', 'vW')], 50001, 10,
                  Xa, Xb, y[tr], lr, wd, (10,))
    kept_diff = max(float(np.abs(p - ck[10][k]).max()) for p, k in zip(kept[10][0], 'UVW'))
    moved = max(float(np.abs(ck[10][k] - src[k]).max()) for k in 'UVW')
    res['C2'] = dict(kept_moments_replay_max_abs_diff_at_10=kept_diff, run_displacement_max_abs_at_10=moved,
                     PASS=kept_diff > 1e3*1e-9*scale)

    rows, srows = metric_rows(run), metric_rows(src_run)
    r0, rs = rows[0], srows[50000]
    logits = ind.forward(w0, *ind.one_hot_inputs(a, b, P))[0]
    own = {name: ind.evaluate(logits[ix], y[ix]) for name, ix in zip(('train', 'cal', 'audit'), splits)}
    rel = lambda x, y: abs(x - y)/max(abs(y), 1e-300)
    res['C3'] = dict(
        step0_row={k: r0[k] for k in ('train', 'cal', 'audit', 'norm')} | {
            f'{s}_ce': r0['objective'][s]['cross_entropy'] for s in ('train', 'cal', 'audit')},
        source_row_50000={k: rs[k] for k in ('train', 'cal', 'audit', 'norm')} | {
            f'{s}_ce': rs['objective'][s]['cross_entropy'] for s in ('train', 'cal', 'audit')},
        own_forward={s: dict(acc=own[s]['acc'], ce=own[s]['cross_entropy']) for s in own},
        own_forward_ce_rel_diff={s: rel(r0['objective'][s]['cross_entropy'], own[s]['cross_entropy']) for s in own},
        own_forward_ce_abs_diff={s: abs(r0['objective'][s]['cross_entropy'] - own[s]['cross_entropy']) for s in own},
        # The train cross-entropy is a mean of logsumexp minus the correct logit, both of this size:
        train_mean_abs_correct_logit=float(np.mean(np.abs(logits[tr, y[tr]]))))
    res['C3']['row_equals_source_exactly'] = bool(
        all(r0[s] == rs[s] for s in own) and r0['norm'] == rs['norm']
        and all(r0['objective'][s]['cross_entropy'] == rs['objective'][s]['cross_entropy'] for s in own))
    res['C3']['PASS'] = bool(res['C3']['row_equals_source_exactly'] and all(r0[s] == own[s]['acc'] for s in own)
                             and all(v <= 1e-12 for v in res['C3']['own_forward_ce_rel_diff'].values()))
    res['arm_A_train_ce_after_reset'] = {str(s): rows[s]['objective']['train']['cross_entropy'] for s in (0, 10, 30, 100, 300)}
    res['arm_A_train_acc_after_reset'] = {str(s): rows[s]['train'] for s in (0, 10, 30, 100, 300)}

    cfg, scfg = (json.loads((d/'config.json').read_text()) for d in (run, src_run))
    keys = ('task', 'hidden', 'p', 'lr', 'wd', 'diagnostics')  # e1 configs predate --train-fraction; the split hash covers it
    res['C4'] = dict(
        same={k: (cfg[k], scfg[k]) for k in keys}, train_fraction=(cfg['train_fraction'], scfg.get('train_fraction')),
        data_seed=(cfg['seeds']['data_seed'], scfg['seeds']['data_seed']),
        hashes_equal=cfg['data_hashes'] == scfg['data_hashes'],
        init_rule=cfg['init_rule'], init_age=cfg['init_age'], init_weights_sha256_recorded=cfg['init_weights_sha256'] == sha(start))
    res['C4']['PASS'] = bool(all(x == y for x, y in res['C4']['same'].values()) and res['C4']['hashes_equal']
                             and cfg['seeds']['data_seed'] == scfg['seeds']['data_seed']
                             and cfg['init_age'] == 50000 and res['C4']['init_weights_sha256_recorded'])

    # C5: shared tests, the nine launch commands, and refusal of a reserved ID.
    tests = subprocess.run([sys.executable, '-m', 'unittest', 'test_exploration', 'test_instrumentation'],
                           env=ENV, cwd=root, capture_output=True, text=True)
    selftest = subprocess.run([sys.executable, 'experiment.py', '--selftest'], env=ENV, cwd=root, capture_output=True, text=True)
    commands = {}
    for e in RESERVED:
        f = flags(launch_runs.command(plan, e, Path('OUT'), 'LABEL'))
        commands[e['run_id']] = all([
            f['--init-weights'] == e['init_weights'] == f"research/e3/mechanisms/starts/{e['run_id']}.npz",
            f['--expect-init-sha256'] == e['init_weights_sha256'], f['--init-age'] == '50000',
            f['--checkpoint-steps'] == '10,30,100,300', f['--steps'] == '20000', f['--log-every'] == '10',
            f['--checkpoint-every'] == '1000', f['--expect-split-sha256'] == e['split_sha256'] == cfg['data_hashes']['split_sha256'],
            f['--data-seed'] == '31000', f['--task'] == 'add', f['--wd'] == '1', f['--lr'] == '0.001', f['--hidden'] == '128',
            f['--diagnostics'] == 'none', '--train-fraction' not in f,
            f['--init-seed'] == e['source_run'].split('-i')[1][:5]])
    refuse = subprocess.run([sys.executable, 'launch_runs.py', '--plan', 'research/e3/plan.json', '--root', str(scratch/'refuse'),
                             '--registry', str(scratch/'refuse.jsonl'), '--session-label', 'audit', '--ids', RESERVED[0]['run_id'],
                             '--timeout', '1'], env=ENV, cwd=root, capture_output=True, text=True)
    res['C5'] = dict(unittest_returncode=tests.returncode, unittest_summary=tests.stderr.strip().splitlines()[-1],
                     unittest_ran=[ln for ln in tests.stderr.splitlines() if ln.startswith('Ran ')],
                     selftest_returncode=selftest.returncode, nine_commands_correct=commands,
                     reserved_id_refused=refuse.returncode != 0 and 'not in plan' in refuse.stderr,
                     init_weights_sha256_still_pending=sorted({e['init_weights_sha256'] for e in RESERVED}))
    res['C5']['PASS'] = bool(tests.returncode == 0 and selftest.returncode == 0 and len(commands) == 9
                             and all(commands.values()) and res['C5']['reserved_id_refused'])
    Path(out_json).write_text(json.dumps(res, indent=1) + '\n')
    print(json.dumps({k: v['PASS'] for k, v in res.items() if isinstance(v, dict) and 'PASS' in v}))


if __name__ == '__main__':
    main(*sys.argv[1:3])
