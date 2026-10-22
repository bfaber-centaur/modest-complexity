"""c3 scaffolding lane: publish run records, check branches, evaluate checkpoints frozen, measure, draw.

  uv run --locked python research/composition-c3/scaffolding/scaf_analyze.py publish --root runs/c3 --ids ID... [--commit-checkpoints ID/FILE...]
  uv run --locked python research/composition-c3/scaffolding/scaf_analyze.py check     # joins, schedules, exposure, frozen-eval reference -> checks.json
  uv run --locked python research/composition-c3/scaffolding/scaf_analyze.py measure   # CSVs in figures/
  uv run --locked python research/composition-c3/scaffolding/scaf_analyze.py figure    # figures/scaf_main.{png,svg} from those CSVs

Arms (QUESTION.md): A keep = r1 tied/inter 0-10k + c2-time-tt-inter 10k-50k; D final-only =
r1 tied/final + c2-time-tt-final; B drop and C taper = this lane's branches from r1 tied/inter
ckpt-S. Everything reads committed records and checkpoints.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
C2 = REPO/'research'/'composition-c2'
sys.path[:0] = [str(HERE.parent/'common'), str(C2/'common'), str(REPO/'research'/'regimes'/'common')]
import c2rnn  # noqa: E402
import c2task  # noqa: E402
import c3train  # noqa: E402
import nets  # noqa: E402
_spec = importlib.util.spec_from_file_location('c2_time_analyze', C2/'time'/'time_analyze.py')
c2time = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(c2time)  # its compatibility measure

RUNS, FIGS = HERE/'runs', HERE/'figures'
R1_RUNS = REPO/'research'/'regimes'/'composition'/'runs'
SEEDS = (51200, 51201, 51202)
SWITCH = {1000: '1k', 10000: '10k'}
KINDS = {'drop': 1, 'taper': 2000}
FIXED = (1000, 10000, 20000, 30000, 40000, 50000)
RESET, RESET_AT = {'keep-reset': 'keep', 'drop-reset': 'drop'}, 1000   # follow-up arm -> its carried match
END, EVERY, THRESHOLD, SUSTAIN, DEGRADE = 50000, 50, 0.95, 5, 0.05
R1_KEYS = ('train', 'held', 'held_final_pred')
SMALL = ('config.json', 'metrics.jsonl', 'runtime.json')
HEADS = ('step1', 'step2', 'final')
EPS = c3train.c2train.EPS
SCHEDULE_POINTS = (-1, 0, 1, 1998, 1999, 2000)


def run_id(kind, S, seed, attempt=1):
    return f'c3-scaf-{kind}-at{SWITCH[S]}-s{seed}-a{attempt}'


def read_jsonl(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines()] if Path(path).exists() else None


def publish(root, ids, committed_checkpoints=()):
    for rid in ids:
        src, dst = root/rid, RUNS/rid
        dst.mkdir(parents=True, exist_ok=True)
        manifest = []
        for f in sorted(src.iterdir()):
            committed = f.name in SMALL or f'{rid}/{f.name}' in committed_checkpoints
            if committed:
                shutil.copy2(f, dst/f.name)
            manifest.append(dict(file=f.name, bytes=f.stat().st_size, sha256=hashlib.sha256(f.read_bytes()).hexdigest(), committed=committed))
        log = root/f'{rid}.log'
        if log.exists():
            shutil.copy2(log, dst/'stdout.log')
            manifest.append(dict(file='stdout.log', bytes=log.stat().st_size, sha256=hashlib.sha256(log.read_bytes()).hexdigest(), committed=True))
        (dst/'MANIFEST.json').write_text(json.dumps(manifest, indent=1)+'\n')
        print(rid, len(manifest), 'files')


# ---- arms: records, checkpoints, schedules ---------------------------------------------------

def r1_rows(cond, seed):
    return read_jsonl(R1_RUNS/f'r1-comp-rnn-{cond}-s{seed}-a1'/'metrics.jsonl')


def same_r1_keys(a, b):
    return all(a[k] == b[k] for k in R1_KEYS)


def rows_of(arm, seed, S=None):
    """Logged rows of an arm from step 0 (keep, final) or from its switch S (drop, taper); None if absent."""
    if arm in ('keep', 'final'):
        r1 = r1_rows('inter' if arm == 'keep' else 'final', seed)
        cont = read_jsonl(C2/'time'/'runs'/f'c2-time-tt-{"inter" if arm == "keep" else "final"}-s{seed}-a1'/'metrics.jsonl')
        assert same_r1_keys(cont[0], r1[-1]) and r1[-1]['step'] == cont[0]['step'] == 10000
        return r1[:-1] + cont  # the c2 row at 10k equals r1's in every r1 key and also carries diag
    return read_jsonl(RUNS/run_id(arm, S, seed)/'metrics.jsonl')


def schedule(arm, S=None):
    if arm == 'keep':
        return dict(before=[1., 1., 1.], switch_step=None, after=None, taper=1)
    if arm == 'final':
        return dict(before=[0., 0., 1.], switch_step=None, after=None, taper=1)
    if arm in RESET:
        return dict(schedule(RESET[arm], S if RESET[arm] == 'drop' else None), reset_at=RESET_AT)
    return dict(before=[1., 1., 1.], switch_step=S, after=[0., 0., 1.], taper=KINDS[arm])


def ckpt_path(arm, seed, step, S=None):
    cond = 'final' if arm == 'final' else 'inter'
    if step in (0, 1000, 10000) and (arm in ('keep', 'final') or step == S):
        return C2/'recovered'/f'r1-comp-rnn-{cond}-s{seed}-a1'/f'ckpt-{step}.npz'
    if arm in ('keep', 'final'):
        return C2/'time'/'runs'/f'c2-time-tt-{cond}-s{seed}-a1'/f'ckpt-{step}.npz'
    return RUNS/run_id(arm, S, seed)/f'ckpt-{step}.npz'


def ckpt_steps(arm, S=None):
    if arm in ('keep', 'final'):
        return FIXED
    return tuple(sorted({S, S + 2000, *[t for t in FIXED if t >= S]}))


# ---- frozen evaluation ------------------------------------------------------------------------

_T = c2task.load()
HELD = c2task.rows(_T['P'], _T['held'])
NBR = c2task.neighbour_answerable(_T['P'], _T['train'], _T['held'])[0]
LENGTHS = (1, 2, 3, 4, 5, 6)
_ROWS = {}


def length_rows(L):
    if L not in _ROWS:
        _ROWS[L] = c2task.rows(_T['P'], c2task.programs(L))
    return _ROWS[L]


def fwd(p, ops, s):
    """Tied/tied forward of any length (tied groups ignore the step map; c2 generalization's fwd)."""
    assert c2rnn.scheme(p) == dict(transition=True, readout=True)
    L = np.asarray(ops).reshape(len(ops), -1).shape[1]
    return c2rnn.forward(p, ops, s=s, step_map=[min(t, c2task.TRAIN_LENGTH) for t in range(1, L+1)])


def frozen(p):
    """Accuracy of the last step on every program of length 1..6 (all starts), and length-3 held-out / N / notN."""
    out = {}
    for L in LENGTHS:
        d = length_rows(L)
        out[f'L{L}'] = float(np.mean(fwd(p, d['ops'], d['s'])[1][-1].argmax(1) == d['states'][:, -1]))
    pred = fwd(p, HELD['ops'], HELD['s'])[1][-1].argmax(1)
    ok = pred == HELD['states'][:, -1]
    out.update(held_L3=float(ok.mean()), held_N=float(ok[NBR].mean()), held_notN=float(ok[~NBR].mean()),
               held_final_pred=''.join(map(str, pred)))
    return out


# ---- checks ------------------------------------------------------------------------------------

def weights_recount(t, S, D):
    """Independent restatement of QUESTION.md's schedule: intermediate weight of the update at step t."""
    if t < S:
        return 1.
    k = t - S
    return 0. if k >= D - 1 else 1 - (k + 1)/D


def schedule_points(cfg, S, D):
    """The recorded schedule (config.json) against the restatement at S-1, S, S+1, S+1998, S+1999 and S+2000,
    which covers the taper's last nonzero update and its first zero one (S+1999 is not a logged row)."""
    return {S + k: [c3train.weights_at(S + k, cfg['schedule']), [weights_recount(S + k, S, D)]*2 + [1.]] for k in SCHEDULE_POINTS}


def check():
    out = dict(branches={}, frozen_reference={})
    for S in SWITCH:
        for kind, D in KINDS.items():
            for seed in SEEDS:
                rid = run_id(kind, S, seed)
                rows = rows_of(kind, seed, S)
                if rows is None:
                    continue
                cfg = json.loads((RUNS/rid/'config.json').read_text())
                r1 = {r['step']: r for r in r1_rows('inter', seed)}
                bad_w = [r['step'] for r in rows if any(abs(w - x) > 1e-12 for w, x in
                         zip(r['step_weights'], (weights_recount(r['step'], S, D),)*2 + (1.,)))]
                n_inter = sum(weights_recount(t, S, D) != 0 for t in range(S, END))
                pts = schedule_points(cfg, S, D)
                expo = cfg['exposure']['labels_this_run']
                out['branches'][rid] = dict(
                    first_step=rows[0]['step'], last_step=rows[-1]['step'], logged=len(rows),
                    join_equals_r1=same_r1_keys(rows[0], r1[S]),
                    step_weights_differ_at=bad_w[:5], n_step_weights_differ=len(bad_w),
                    schedule_points=pts, schedule_points_equal=all(a == b for a, b in pts.values()),
                    exposure_updates_recount=[n_inter, n_inter, END - S], exposure_updates_logged=expo['updates_with_weight'],
                    exposure_matches=expo['updates_with_weight'] == [n_inter, n_inter, END - S],
                    recipe=dict(task=cfg['task_id'], scheme=cfg['scheme'], lr=cfg['lr'], wd=cfg['wd'], hidden=cfg['hidden'],
                                schedule=cfg['schedule'], resumed_step=cfg['resumed_from']['step'],
                                resumed_sha256=cfg['resumed_from']['sha256'], source_run=cfg['resumed_from']['source']['run_id']))
    # The frozen evaluation against c2 generalization's published length.csv (10k recovered, 50k time lane).
    pub = {}
    for name in ('c2-gen-frozen-orig-a2', 'c2-gen-frozen-50k-a1'):
        with (C2/'generalization'/'runs'/name/'length.csv').open() as f:
            for r in csv.DictReader(f):
                if r['subset'] == 'all' and r['task_id'] == c2task.ORIGINAL:
                    pub[(r['condition'], int(r['init_seed']), int(r['ckpt_step']), int(r['length']))] = float(r['accuracy'])
    for arm, cond in (('keep', 'inter'), ('final', 'final')):
        for seed in SEEDS:
            for step in (10000, 50000):
                got = frozen(nets.load(ckpt_path(arm, seed, step))[0])
                ref = [pub.get((cond, seed, step, L)) for L in LENGTHS]
                out['frozen_reference'][f'{arm}-s{seed}-{step}'] = dict(
                    equal=all(r is not None and got[f'L{L}'] == r for L, r in zip(LENGTHS, ref)), ours=[got[f'L{L}'] for L in LENGTHS], published=ref)
    (HERE/'checks.json').write_text(json.dumps(out, indent=1)+'\n')
    print(json.dumps({k: {i: {x: v[x] for x in v if x in ('join_equals_r1', 'n_step_weights_differ', 'schedule_points_equal', 'exposure_matches', 'equal')}
                          for i, v in d.items()} for k, d in out.items()}, indent=1))


# ---- measures ----------------------------------------------------------------------------------

def first_sustained(steps, acc, threshold=THRESHOLD, k=SUSTAIN, every=EVERY):
    ok = np.asarray(acc) >= threshold - 1e-12
    for i in range(len(steps) - k + 1):
        if ok[i:i+k].all() and steps[i+k-1] - steps[i] == (k-1)*every:
            return int(steps[i])
    return None


def subsets(pred_string):
    ok = np.array([int(c) for c in pred_string]) == HELD['states'][:, -1]
    return dict(N=float(ok[NBR].mean()), notN=float(ok[~NBR].mean()))


def arms():
    """(arm, S, seed) for every arm with records: keep and final once per seed (S None), branches per S."""
    out = [(a, None, s) for a in ('keep', 'final') for s in SEEDS]
    out += [(k, S, s) for S in SWITCH for k in KINDS for s in SEEDS if (RUNS/run_id(k, S, s)/'metrics.jsonl').exists()]
    return out


def optimizer_row(arm, S, seed, rows):
    """Eps-regime onsets (first logged step with median, and with every coordinate's, sqrt(v-hat) below eps)
    and the update norm just after the switch relative to the switch row."""
    d = [(r['step'], r['diag']) for r in rows if 'diag' in r]
    first = lambda f: next((t for t, x in d if f(x)), '')
    out = dict(arm=arm, S=S or '', seed=seed, first_diag_step=d[0][0] if d else '',
               median_below_eps_step=first(lambda x: x['adam_sqrt_vhat']['median'] < EPS),
               all_below_eps_step=first(lambda x: x['adam_sqrt_vhat']['frac_below_eps'] >= 1.))
    if S:
        u = {t: x['update_norm'] for t, x in d}
        for k in (50, 500, 2000, 5000):
            out[f'update_norm_ratio_S+{k}'] = u[S + k]/u[S] if S + k in u else ''
    return out


def compatibility_rows(arm, S, seed, rows):
    """Neighbour compatibility of wrong held-out length-3 predictions at the fixed times (c2 time lane's measure)."""
    logged = {r['step']: r for r in rows}
    return [dict(arm=arm, S=S or '', seed=seed, step=t, **c2time.compatibility(logged[t]['held_final_pred']))
            for t in ckpt_steps(arm, S) if t in logged]


def traj_rows(arm, S, seed, rows):
    sched, traj = schedule(arm, S), []
    for r in rows:
        w = r.get('step_weights') or c3train.weights_at(r['step'], sched)
        sub = r.get('held_subset') or subsets(r['held_final_pred'])
        d = r.get('diag', {})
        v = d.get('adam_sqrt_vhat', {})
        traj.append(dict(arm=arm, S=S or '', seed=seed, step=r['step'], w_inter=w[0],
                         **{f'train_{h}_{m}': r['train'][h][m] for h in HEADS for m in ('acc', 'ce')},
                         weighted_train_loss=sum(wk*r['train'][h]['ce'] for wk, h in zip(w, HEADS)),
                         held_final_acc=r['held']['final']['acc'], held_final_ce=r['held']['final']['ce'],
                         held_N_acc=sub['N'], held_notN_acc=sub['notN'],
                         grad_norm=d.get('grad_norm', ''), update_norm=d.get('update_norm', ''),
                         sqrt_vhat_median=v.get('median', ''), sqrt_vhat_max=v.get('max', ''), frac_below_eps=v.get('frac_below_eps', ''),
                         train_margin_min=d.get('train_final_margin', {}).get('min', ''),
                         train_margin_median=d.get('train_final_margin', {}).get('median', '')))
    return traj


def measure_arm(arm, S, seed, rows=None):
    """Trajectory, threshold, optimizer, compatibility and frozen rows of one arm and seed."""
    rows = rows or rows_of(arm, seed, S)
    traj, thr, fro = traj_rows(arm, S, seed, rows), [], []
    # Thresholds are read on the whole trajectory from init: a branch's rows follow r1 tied/inter's rows before S,
    # so a crossing before the switch (S = 10k) is the keep arm's crossing.
    full = rows if S is None else [r for r in r1_rows('inter', seed) if r['step'] < S] + rows
    steps = [r['step'] for r in full]
    for name, acc in (('held_final', [r['held']['final']['acc'] for r in full]),
                      ('held_notN', [(r.get('held_subset') or subsets(r['held_final_pred']))['notN'] for r in full])):
        t = first_sustained(steps, acc)
        thr.append(dict(arm=arm, S=S or '', seed=seed, measure=name, first_step=steps[0], last_step=steps[-1],
                        crossing_step=t if t is not None else '', censored=t is None, before_switch=t is not None and S is not None and t < S,
                        value_at_switch=acc[steps.index(S)] if S else '', value_at_end=acc[-1]))
    logged = {r['step']: r for r in rows}
    for step in ckpt_steps(arm, S):
        path = ckpt_path(arm, seed, step, S)
        if not path.exists():
            continue
        e = frozen(nets.load(path)[0])
        assert step not in logged or e['held_final_pred'] == logged[step]['held_final_pred'], (arm, S, seed, step)
        fro.append(dict(arm=arm, S=S or '', seed=seed, step=step, **{k: v for k, v in e.items() if k != 'held_final_pred'}))
    return dict(traj=traj, thr=thr, fro=fro, opt=[optimizer_row(arm, S, seed, rows)], comp=compatibility_rows(arm, S, seed, rows))


def collect(arm_list):
    out = dict(traj=[], thr=[], fro=[], opt=[], comp=[])
    for arm, S, seed in arm_list:
        for k, v in measure_arm(arm, S, seed).items():
            out[k] += v
    return out


def measure():
    FIGS.mkdir(exist_ok=True)
    m = collect(arms())
    write_csv(FIGS/'scaf_main.csv', m['traj'])
    write_csv(FIGS/'scaf_frozen.csv', m['fro'])
    write_csv(FIGS/'scaf_threshold.csv', m['thr'])
    write_csv(FIGS/'scaf_retention.csv', retention(m['fro']))
    write_csv(FIGS/'scaf_optimizer.csv', m['opt'])
    write_csv(FIGS/'scaf_compatibility.csv', m['comp'])


def retention(fro):
    """Per branch, measure and fixed time: value, switch state, keep, final-only and the three deltas;
    the endpoint status uses the 5-point rule of QUESTION.md."""
    at = {(r['arm'], r['S'], r['seed'], r['step']): r for r in fro}
    out = []
    for S in SWITCH:
        for kind in KINDS:
            for seed in SEEDS:
                for m in ('held_L3', 'held_notN', 'L1', 'L2', 'L4', 'L5', 'L6'):
                    sw = at.get(('keep', '', seed, S))
                    for t in [x for x in FIXED if x >= S]:
                        b, k, f = at.get((kind, S, seed, t)), at.get(('keep', '', seed, t)), at.get(('final', '', seed, t))
                        if b is None:
                            continue
                        row = dict(arm=kind, S=S, seed=seed, measure=m, step=t, value=b[m], at_switch=sw[m], keep=k[m], final_only=f[m],
                                   d_branch=b[m] - sw[m], d_keep=b[m] - k[m], d_final=b[m] - f[m], status='')
                        if t == END:
                            row['status'] = ('degraded' if row['d_keep'] <= -DEGRADE + 1e-12 else 'above keep' if row['d_keep'] >= DEGRADE - 1e-12 else 'retained')
                        out.append(row)
    return out


def write_csv(path, rows):
    if not rows:
        print(path.name, 'no rows, not written'); return
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)), restval='')
        w.writeheader(); w.writerows(rows)
    print(path.name, len(rows), 'rows')


# ---- figure ------------------------------------------------------------------------------------

COLOR = {'keep': '#eb6834', 'drop': '#7a4fd6', 'taper': '#1baf7a', 'final': '#2a78d6'}
LABEL = {'keep': 'A keep [1,1,1]', 'drop': 'B drop to [0,0,1] at S', 'taper': 'C taper over 2,000 from S', 'final': 'D final-only from init'}
MARKER = dict(zip(SEEDS, 'osD'))


def read_csv(path):
    with path.open() as f:
        rows = list(csv.DictReader(f))
    keep_str = ('arm', 'status', 'measure', 'censored', 'before_switch')
    return [{k: (float(v) if v != '' and k not in keep_str else v) for k, v in r.items()} for r in rows]


def figure():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    traj, fro, thr = read_csv(FIGS/'scaf_main.csv'), read_csv(FIGS/'scaf_frozen.csv'), read_csv(FIGS/'scaf_threshold.csv')
    rows_k = ['sched', 'held', 'notN', 'train', 'readout', 'L6', 'diag', 'vhat', 'lengths']
    titles = dict(sched='intermediate-loss weight of the update', held='held-out length-3 final accuracy (128 rows)',
                  notN='held-out ¬N accuracy (42 rows)', train='train final CE (solid), weighted train loss (dotted)',
                  readout='lengths 1–2: shared readout on h1 (solid), h2 (dotted), training rows',
                  L6='frozen: L = 6 (solid), L = 5 (dashed), L = 4 (dotted), all programs',
                  diag='global L2: update (solid), gradient (dotted); keep and final-only logged from 10k', vhat='Adam √v̂: median (solid), max (dotted); eps dashed',
                  lengths='frozen accuracy by length at 50k (8·4^L rows; L = 3 includes training); grey: at S')
    heights = [.45, 1, 1, .9, .9, 1, .9, .9, .9]
    fig, axes = plt.subplots(len(rows_k), 2, figsize=(13, 28), gridspec_kw=dict(height_ratios=heights, hspace=.4, wspace=.12, top=.89))
    for j in range(2):
        for a in axes[:-2, j]:
            a.sharex(axes[-2, j]); a.tick_params(labelbottom=False)
    present = {(r['arm'], r['S'], int(r['seed'])) for r in traj}

    def series(arm, S, seed, key, src=traj):
        return [(r['step'], r[key]) for r in src if r['arm'] == arm and r['S'] == S and int(r['seed']) == seed and r[key] != '']

    for j, S in enumerate(SWITCH):
        ax = dict(zip(rows_k, axes[:, j]))
        shown = [('keep', ''), ('final', ''), ('drop', float(S)), ('taper', float(S))]
        for arm, s_key in shown:
            for seed in SEEDS:
                if (arm, s_key, seed) not in present:
                    continue
                kw = dict(color=COLOR[arm], marker=MARKER[seed], ms=4)
                def plot(a, key, ls='-', lw=1.3, src=traj, me=None, **extra):
                    xy = series(arm, s_key, seed, key, src)
                    if xy:
                        x, y = zip(*xy)
                        a.plot(x, y, ls=ls, lw=lw, markevery=me or max(1, len(x)//8), **kw, **extra)
                plot(ax['sched'], 'w_inter', lw=1.6)
                plot(ax['held'], 'held_final_acc')
                plot(ax['notN'], 'held_notN_acc')
                plot(ax['train'], 'train_final_ce'); plot(ax['train'], 'weighted_train_loss', ls=':', lw=1)
                plot(ax['readout'], 'train_step1_acc'); plot(ax['readout'], 'train_step2_acc', ls=':', lw=1)
                for L, ls in ((6, '-'), (5, '--'), (4, ':')):
                    plot(ax['L6'], f'L{L}', ls=ls, lw=1.4 if L == 6 else 1, src=fro, me=1)
                plot(ax['diag'], 'update_norm'); plot(ax['diag'], 'grad_norm', ls=':', lw=1)
                plot(ax['vhat'], 'sqrt_vhat_median'); plot(ax['vhat'], 'sqrt_vhat_max', ls=':', lw=1)
                end = [r for r in fro if r['arm'] == arm and r['S'] == s_key and int(r['seed']) == seed and r['step'] == END]
                if end:
                    ax['lengths'].plot(LENGTHS, [end[0][f'L{L}'] for L in LENGTHS], lw=1.3, **kw)
        for seed in SEEDS:
            sw = [r for r in fro if r['arm'] == 'keep' and int(r['seed']) == seed and r['step'] == S]
            if sw:
                ax['lengths'].plot(LENGTHS, [sw[0][f'L{L}'] for L in LENGTHS], color='0.55', lw=1, ls='--', marker=MARKER[seed], ms=3)
        for r in thr:
            if r['measure'] == 'held_final' and r['crossing_step'] != '' and r['S'] in ('', float(S)):
                ax['held'].scatter([r['crossing_step']], [THRESHOLD], marker='v', s=40, color=COLOR[r['arm']], edgecolor='k', lw=.5, zorder=5)
        for k in ('held', 'notN', 'readout', 'L6', 'lengths'):
            ax[k].axhline(THRESHOLD, color='0.45', lw=.9, ls=(0, (4, 3))); ax[k].axhline(1/8, color='0.65', lw=.8, ls=':')
            ax[k].set_ylim(-.02, 1.04)
        ax['vhat'].axhline(1e-8, color='0.3', lw=1, ls=(0, (4, 3)))
        for k in ('train', 'diag', 'vhat'):
            ax[k].set_yscale('log')
        for k, a in ax.items():
            if k != 'lengths':
                a.axvline(S, color='0.3', lw=1.1); a.set_xlim(0, END)
            a.grid(alpha=.25, lw=.5)
            a.set_title(titles[k], loc='left', fontsize=9)
        ax['lengths'].set_xticks(LENGTHS); ax['lengths'].set_xlabel('program length L (1–2 supervised before S; 4–6 novel, shaded)')
        ax['lengths'].axvspan(3.5, 6.5, color='0.92', zorder=0); ax['lengths'].set_xlim(.7, 6.3)
        ax['sched'].set_title(f'switch S = {S:,}   ·   ' + titles['sched'], loc='left', fontsize=10, fontweight='bold', pad=18)
        top = ax['sched'].secondary_xaxis('top', functions=(lambda x, S=S: x - S, lambda x, S=S: x + S))
        top.tick_params(labelsize=7); top.set_xlabel('updates since the switch', fontsize=7, labelpad=1)
        ax['vhat'].set_xlabel('total updates from init (switch S marked)', fontsize=9)
        missing = [f'{k} s{s}' for k in KINDS for s in SEEDS if (k, float(S), s) not in present]
        if missing:
            ax['held'].text(.99, .05, 'not run: ' + ', '.join(missing), transform=ax['held'].transAxes, ha='right', fontsize=7, color='crimson')
    handles = [Line2D([], [], color=COLOR[a], lw=2, label=LABEL[a]) for a in ('keep', 'drop', 'taper', 'final')]
    handles += [Line2D([], [], color='0.3', marker=MARKER[s], lw=0, label=f'init seed {s}') for s in SEEDS]
    handles += [Line2D([], [], color='0.3', marker='v', lw=0, label='first of 5 logged evals ≥ 0.95'),
                Line2D([], [], color='0.45', ls=(0, (4, 3)), label='0.95'), Line2D([], [], color='0.65', ls=':', label='chance 1/8')]
    fig.legend(handles=handles, loc='upper center', ncol=5, fontsize=8, frameon=False, bbox_to_anchor=(.5, .935))
    fig.suptitle('c3 scaffolding: tied/tied RNN, intermediate supervision removed at S (drop) or tapered, against keep and final-only;\n'
                 'original task, three init seeds, unchanged recipe, logged every 50 updates; frozen lengths at fixed checkpoints\n'
                 'where A, B and C all reach 1.0 their lines overplot (the last drawn, C taper, shows on top)',
                 y=.97, fontsize=11)
    for ext in ('png', 'svg'):
        fig.savefig(FIGS/f'scaf_main.{ext}', dpi=120, bbox_inches='tight')
    print('wrote', FIGS/'scaf_main.png')


# ---- follow-up: reset vs carried Adam state at S = 1,000 (QUESTION.md addendum) -------------------

def verify_id(arm, seed):
    return f'c3-scaf-verify-carried-{arm}-at1k-s{seed}-a1'


def fu_arms():
    """A keep, A' keep-reset, B drop, B' drop-reset at S = 1k, and D final-only; reset arms only where run."""
    out = [(a, None, s) for a in ('keep', 'final') for s in SEEDS] + [('drop', RESET_AT, s) for s in SEEDS]
    return out + [(a, RESET_AT, s) for a in RESET for s in SEEDS if (RUNS/run_id(a, RESET_AT, s)/'metrics.jsonl').exists()]


def arrays_equal(a, b):
    pa, sa_, _ = nets.load(a); pb, sb, _ = nets.load(b)
    same = lambda x, y: x.keys() == y.keys() and all(np.array_equal(x[k], y[k]) for k in x)
    return same(pa, pb) and same(sa_['m'], sb['m']) and same(sa_['v'], sb['v']) and sa_['t'] == sb['t']


def followup_check():
    """Matched-control checks (no reset under the re-pin equals the reused carried arm, row for row) and, per
    reset run: join at 1k, logged step weights, recorded reset step, exposure and the Adam count after the reset."""
    out = dict(carried_checks={}, reset_runs={})
    for seed in SEEDS:
        rows = read_jsonl(RUNS/verify_id('keep', seed)/'metrics.jsonl')
        if rows:
            r1 = {r['step']: r for r in r1_rows('inter', seed)}
            out['carried_checks'][verify_id('keep', seed)] = dict(
                logged=len(rows), steps=[rows[0]['step'], rows[-1]['step']], equals_r1_rows=all(same_r1_keys(r, r1[r['step']]) for r in rows))
    rows = read_jsonl(RUNS/verify_id('drop', 51200)/'metrics.jsonl')
    if rows:
        ref = {r['step']: r for r in rows_of('drop', 51200, RESET_AT)}
        out['carried_checks'][verify_id('drop', 51200)] = dict(
            # Every key of every row, diag included; a run's last row has no diag (no update follows it), so the
            # check's row at 3,000 is compared without it.
            logged=len(rows), steps=[rows[0]['step'], rows[-1]['step']],
            equals_branch_rows=all(r == ref[r['step']] for r in rows[:-1]) and 'diag' not in rows[-1]
            and rows[-1] == {k: v for k, v in ref[rows[-1]['step']].items() if k != 'diag'},
            ckpt_3000_equals_branch=arrays_equal(RUNS/verify_id('drop', 51200)/'ckpt-3000.npz', ckpt_path('drop', 51200, 3000, RESET_AT)))
    for arm, match in RESET.items():
        for seed in SEEDS:
            rid = run_id(arm, RESET_AT, seed)
            rows = rows_of(arm, seed, RESET_AT)
            if rows is None:
                continue
            cfg = json.loads((RUNS/rid/'config.json').read_text())
            sched = schedule(arm, RESET_AT)
            r1 = {r['step']: r for r in r1_rows('inter', seed)}
            t3 = nets.load(RUNS/rid/'ckpt-3000.npz')[1]['t']
            n = [sum(c3train.weights_at(t, sched)[k] != 0 for t in range(RESET_AT, END)) for k in range(3)]
            out['reset_runs'][rid] = dict(
                join_equals_r1=same_r1_keys(rows[0], r1[RESET_AT]), recorded_reset_at=cfg['schedule'].get('reset_at'),
                n_step_weights_differ=sum(r['step_weights'] != c3train.weights_at(r['step'], sched) for r in rows),
                exposure_matches=cfg['exposure']['labels_this_run']['updates_with_weight'] == n,
                adam_count_at_3000=t3, adam_count_expected=3000 - RESET_AT,
                recipe=dict(lr=cfg['lr'], wd=cfg['wd'], hidden=cfg['hidden'], scheme=cfg['scheme'], source_run=cfg['resumed_from']['source']['run_id']))
    (HERE/'followup_checks.json').write_text(json.dumps(out, indent=1)+'\n')
    keys = ('equals_r1_rows', 'equals_branch_rows', 'ckpt_3000_equals_branch', 'join_equals_r1', 'recorded_reset_at',
            'n_step_weights_differ', 'exposure_matches', 'adam_count_at_3000')
    print(json.dumps({k: {i: {x: v[x] for x in v if x in keys} for i, v in d.items()} for k, d in out.items()}, indent=1))


FU_MEASURES = ('held_L3', 'held_notN', 'L4', 'L5', 'L6')


def reset_effects(fro):
    """Per seed, measure and fixed time from 3k: A, A', B, B', D, the two reset effects and their difference.
    At 50k each arm pair gets the 5-point status of the addendum's rules."""
    at = {(r['arm'], r['seed'], r['step']): r for r in fro}
    out = []
    for seed in SEEDS:
        for m in FU_MEASURES:
            for t in (3000,) + FIXED[1:]:
                v = {a: at.get((a, seed, t), {}).get(m) for a in ('keep', 'keep-reset', 'drop', 'drop-reset', 'final')}
                if v['keep-reset'] is None and v['drop-reset'] is None:
                    continue
                d = lambda x, y: x - y if x is not None and y is not None else ''
                row = dict(seed=seed, measure=m, step=t, A=v['keep'], A_reset=v['keep-reset'], B=v['drop'], B_reset=v['drop-reset'], D=v['final'],
                           d_reset_keep=d(v['keep-reset'], v['keep']), d_reset_drop=d(v['drop-reset'], v['drop']))
                row['difference'] = d(row['d_reset_drop'] if row['d_reset_drop'] != '' else None, row['d_reset_keep'] if row['d_reset_keep'] != '' else None)
                row['B_reset_minus_D'] = d(v['drop-reset'], v['final'])
                for k in ('d_reset_keep', 'd_reset_drop'):
                    row[k + '_status'] = '' if t != END or row[k] == '' else ('below carried' if row[k] <= -DEGRADE + 1e-12 else
                                                                            'above carried' if row[k] >= DEGRADE - 1e-12 else 'within 5 pp')
                out.append(row)
    return out


def transient(traj):
    """Update L2 of reset runs over 1,000-1,500 against the carried match at the same step (keep's from the check runs)."""
    u = {(r['arm'], int(r['seed']), r['step']): r['update_norm'] for r in traj if r['update_norm'] != ''}
    out = []
    for seed in SEEDS:
        for arm, match in RESET.items():
            for t in range(RESET_AT, RESET_AT + 550, 50):
                a, b = u.get((arm, seed, t)), u.get((match, seed, t))
                if a is not None and b is not None:
                    out.append(dict(seed=seed, arm=arm, step=t, update_norm=a, carried=b, ratio=a/b))
    return out


def followup_measure():
    FIGS.mkdir(exist_ok=True)
    m = collect(fu_arms())
    for seed in SEEDS:   # keep's diag over 1k-3k from the carried-keep check run (its r1 keys equal r1's rows)
        rows = read_jsonl(RUNS/verify_id('keep', seed)/'metrics.jsonl')
        if rows:
            extra = {r['step']: r for r in traj_rows('keep', None, seed, rows)}
            m['traj'] = [extra.get(r['step'], r) if r['arm'] == 'keep' and int(r['seed']) == seed else r for r in m['traj']]
    write_csv(FIGS/'scaf_followup.csv', m['traj'])
    write_csv(FIGS/'scaf_followup_frozen.csv', m['fro'])
    write_csv(FIGS/'scaf_followup_threshold.csv', m['thr'])
    write_csv(FIGS/'scaf_followup_optimizer.csv', m['opt'])
    write_csv(FIGS/'scaf_followup_compatibility.csv', m['comp'])
    write_csv(FIGS/'scaf_followup_effects.csv', reset_effects(m['fro']))
    write_csv(FIGS/'scaf_followup_transient.csv', transient(m['traj']))


FU_STYLE = {'keep': (COLOR['keep'], '-'), 'keep-reset': (COLOR['keep'], (0, (5, 2))), 'drop': (COLOR['drop'], '-'),
            'drop-reset': (COLOR['drop'], (0, (5, 2))), 'final': (COLOR['final'], '-')}
FU_LABEL = {'keep': "A keep, moments carried", 'keep-reset': "A′ keep, Adam reset at 1k", 'drop': 'B drop at 1k, moments carried',
            'drop-reset': "B′ drop at 1k, Adam reset at 1k", 'final': 'D final-only from init'}


def followup_figure():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    traj, fro, thr = (read_csv(FIGS/f'scaf_followup{x}.csv') for x in ('', '_frozen', '_threshold'))
    tra = read_csv(FIGS/'scaf_followup_transient.csv') if (FIGS/'scaf_followup_transient.csv').exists() else []
    rows_k = ['sched', 'held', 'notN', 'L6', 'train', 'readout', 'diag', 'vhat', 'transient']
    titles = dict(sched='intermediate-loss weight; Adam reset at 1k in A′, B′', held='held-out length-3 final accuracy (128 rows)',
                  notN='held-out ¬N accuracy (42 rows)', L6='frozen: L = 6 (solid marker), L = 4 (open marker), all programs',
                  train='train final CE', readout='shared readout on training h1 and h2: the lower of the two accuracies',
                  diag='global L2 of the update', vhat='Adam √v̂, median; eps dashed',
                  transient=['update L2 over 1,000–1,500 (logged every 50): reset dashed, carried solid',
                             'update L2 of the reset run ÷ its carried match, same step'])
    heights = [.45, 1, 1, 1, .9, .8, .9, .9, .9]
    fig, axes = plt.subplots(len(rows_k), 2, figsize=(13, 27), gridspec_kw=dict(height_ratios=heights, hspace=.46, wspace=.14, top=.905))
    xlim = [(0, END), (900, 6000)]
    for j in range(2):
        for a in axes[:-2, j]:
            a.sharex(axes[-2, j]); a.tick_params(labelbottom=False)
        ax = dict(zip(rows_k, axes[:, j]))
        for arm, (c, ls) in FU_STYLE.items():
            for seed in SEEDS:
                tr = [r for r in traj if r['arm'] == arm and int(r['seed']) == seed]
                if not tr:
                    continue
                kw = dict(color=c, ls=ls, lw=1.3, marker=MARKER[seed], ms=3.5, markevery=max(1, len(tr)//8) if j == 0 else 10)
                def plot(a, key, src=tr, **k2):
                    xy = [(r['step'], r[key]) for r in src if r[key] != '']
                    if xy:
                        a.plot(*zip(*xy), **{**kw, **k2})
                plot(ax['sched'], 'w_inter')
                plot(ax['held'], 'held_final_acc'); plot(ax['notN'], 'held_notN_acc')
                plot(ax['train'], 'train_final_ce')
                low = [dict(step=r['step'], v=min(r['train_step1_acc'], r['train_step2_acc'])) for r in tr]
                plot(ax['readout'], 'v', src=low)
                plot(ax['diag'], 'update_norm'); plot(ax['vhat'], 'sqrt_vhat_median')
                fr = sorted((r for r in fro if r['arm'] == arm and int(r['seed']) == seed), key=lambda r: r['step'])
                if fr:
                    ax['L6'].plot([r['step'] for r in fr], [r['L6'] for r in fr], color=c, ls=ls, lw=1.3, marker=MARKER[seed], ms=4)
                    ax['L6'].plot([r['step'] for r in fr], [r['L4'] for r in fr], color=c, ls=ls, lw=.8, marker=MARKER[seed], ms=4, mfc='none')
        for r in thr:
            if r['measure'] == 'held_final' and r['crossing_step'] != '':
                ax['held'].scatter([r['crossing_step']], [THRESHOLD], marker='v', s=40, color=FU_STYLE[r['arm']][0], edgecolor='k', lw=.5, zorder=5)
        for k in ('held', 'notN', 'L6', 'readout'):
            ax[k].axhline(THRESHOLD, color='0.45', lw=.9, ls=(0, (4, 3))); ax[k].axhline(1/8, color='0.65', lw=.8, ls=':')
            ax[k].set_ylim(-.02, 1.04)
        ax['vhat'].axhline(1e-8, color='0.3', lw=1, ls=(0, (4, 3)))
        for k in ('train', 'diag', 'vhat'):
            ax[k].set_yscale('log')
        for k, a in ax.items():
            if k == 'transient':
                continue
            a.axvline(RESET_AT, color='0.3', lw=1.1); a.set_xlim(*xlim[j]); a.grid(alpha=.25, lw=.5)
            a.set_title(titles[k], loc='left', fontsize=9)
        ax['sched'].set_title(('all updates · ' if j == 0 else 'zoom 900–6,000 · ') + titles['sched'], loc='left', fontsize=10, fontweight='bold')
        ax['vhat'].set_xlabel('total updates from init (switch and reset at 1,000 marked)', fontsize=9)
        a = ax['transient']; a.set_title(titles['transient'][j], loc='left', fontsize=9)
        for arm, match in RESET.items():
            for seed in SEEDS:
                xs = [r for r in tra if r['arm'] == arm and int(r['seed']) == seed]
                if not xs:
                    continue
                c = FU_STYLE[arm][0]
                if j == 0:
                    a.plot([r['step'] for r in xs], [r['update_norm'] for r in xs], color=c, ls=(0, (5, 2)), marker=MARKER[seed], ms=3.5)
                    a.plot([r['step'] for r in xs], [r['carried'] for r in xs], color=c, ls='-', marker=MARKER[seed], ms=3.5, lw=.9)
                else:
                    a.plot([r['step'] for r in xs], [r['ratio'] for r in xs], color=c, ls=(0, (5, 2)), marker=MARKER[seed], ms=3.5)
        a.set_yscale('log'); a.set_xlim(990, 1510); a.grid(alpha=.25, lw=.5)
        if j == 1:
            a.axhline(1, color='0.3', lw=.9)
        a.set_xlabel('update step (the reset precedes the update at 1,000)', fontsize=9)
    handles = [Line2D([], [], color=c, ls=ls, lw=2, label=FU_LABEL[a]) for a, (c, ls) in FU_STYLE.items()]
    handles += [Line2D([], [], color='0.3', marker=MARKER[s], lw=0, label=f'init seed {s}') for s in SEEDS]
    handles += [Line2D([], [], color='0.3', marker='v', lw=0, label='first of 5 logged evals ≥ 0.95')]
    fig.legend(handles=handles, loc='upper center', ncol=5, fontsize=8, frameon=False, bbox_to_anchor=(.5, .95))
    fig.suptitle('c3 scaffolding follow-up: Adam state reset vs carried at S = 1,000, in keep and in drop; tied/tied RNN, original task,\n'
                 'three init seeds, unchanged recipe; keep\'s 1k–3k diagnostics from the no-reset check runs, r1 logged none before 10k',
                 y=.972, fontsize=11)
    for ext in ('png', 'svg'):
        fig.savefig(FIGS/f'scaf_followup.{ext}', dpi=120, bbox_inches='tight')
    print('wrote', FIGS/'scaf_followup.png')


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('publish'); p.add_argument('--root', type=Path, required=True); p.add_argument('--ids', nargs='+', required=True)
    p.add_argument('--commit-checkpoints', nargs='*', default=[])
    cmds = {'check': check, 'measure': measure, 'figure': figure,
            'followup-check': followup_check, 'followup-measure': followup_measure, 'followup-figure': followup_figure}
    for c in cmds:
        sub.add_parser(c)
    a = ap.parse_args()
    if a.cmd == 'publish':
        publish(a.root, a.ids, set(a.commit_checkpoints))
    else:
        cmds[a.cmd]()


if __name__ == '__main__':
    main()
