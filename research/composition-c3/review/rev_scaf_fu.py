"""c3 scaffolding follow-up review: Adam reset against carried at S = 1k (QUESTION.md section 5: F1-F4,
CX-F1), then the NOTE's section F numbers. Same forward, oracle and measures as rev_scaf.py; my fresh-state
Adam update is rev_repin.py's. The lane's records are read from its commit with git archive.

uv run --locked python research/composition-c3/review/rev_scaf_fu.py --out OUT --lane 5317c54
"""
import argparse
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE)]
from rev_c3pin import C2, R1, REC, ROOT, SEEDS, my_exposure, my_w, rows  # noqa: E402
from rev_pin import FAIL, check  # noqa: E402
from rev_repin import my_adam  # noqa: E402
from rev_scaf import HEADS, NB_ANY, NB_PRE, TR, Y, close, crossing, frozen, num, read_csv, score, subsets  # noqa: E402
import c2rnn  # noqa: E402
import nets  # noqa: E402

R, END, FIXED = 1000, 50000, (3000, 10000, 20000, 30000, 40000, 50000)
ARMS = ('keep', 'keep-reset', 'drop', 'drop-reset', 'final')
FU = ('held_L3', 'held_notN', 'L4', 'L5', 'L6')
W = {'keep': [1., 1., 1.], 'drop': [0., 0., 1.]}


def run_id(arm, seed):
    return {'drop': f'c3-scaf-drop-at1k-s{seed}-a1', 'drop-reset': f'c3-scaf-drop-reset-at1k-s{seed}-a1',
            'keep-reset': f'c3-scaf-keep-reset-at1k-s{seed}-a1'}[arm]


def norm(d):
    return float(np.sqrt(sum(float(np.sum(v*v)) for v in d.values())))


def rng(xs, d):
    return [round(min(xs), d), round(max(xs), d)]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', type=Path, required=True); ap.add_argument('--lane', required=True)
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=False)
    tar = subprocess.run(['git', 'archive', a.lane, 'research/composition-c3/scaffolding'], cwd=ROOT, capture_output=True, check=True).stdout
    tarfile.open(fileobj=io.BytesIO(tar)).extractall(a.out/'lane', filter='data')
    LANE = a.out/'lane'/'research'/'composition-c3'/'scaffolding'; FIG, RUNS = LANE/'figures', LANE/'runs'

    r1 = {(c, s): rows(R1/'composition'/'runs'/f'r1-comp-rnn-{c}-s{s}-a1'/'metrics.jsonl') for c in ('inter', 'final') for s in SEEDS}
    ver = {('keep', s): rows(RUNS/f'c3-scaf-verify-carried-keep-at1k-s{s}-a1'/'metrics.jsonl') for s in SEEDS}
    ver[('drop', 51200)] = rows(RUNS/'c3-scaf-verify-carried-drop-at1k-s51200-a1'/'metrics.jsonl')
    traj, ck, cfgs = {}, {}, {}
    for arm, c in (('keep', 'inter'), ('final', 'final')):
        for s in SEEDS:
            cont = rows(C2/'time'/'runs'/f'c2-time-tt-{c}-s{s}-a1'/'metrics.jsonl')
            traj[(arm, s)] = [r for r in r1[(c, s)] if r['step'] < 10000] + cont
            for t in (1000,) + FIXED[1:]:
                ck[(arm, s, t)] = REC/f'r1-comp-rnn-{c}-s{s}-a1'/f'ckpt-{t}.npz' if t <= 10000 else C2/'time'/'runs'/f'c2-time-tt-{c}-s{s}-a1'/f'ckpt-{t}.npz'
    for arm in ('drop', 'drop-reset', 'keep-reset'):
        for s in SEEDS:
            d = RUNS/run_id(arm, s)
            traj[(arm, s)] = rows(d/'metrics.jsonl'); cfgs[(arm, s)] = json.loads((d/'config.json').read_text())
            ck[(arm, s, R)] = REC/f'r1-comp-rnn-inter-s{s}-a1'/'ckpt-1000.npz'
            for t in FIXED:
                ck[(arm, s, t)] = d/f'ckpt-{t}.npz'
    resets = [(arm, s) for arm in ('keep-reset', 'drop-reset') for s in SEEDS]

    # F1. Branch points and the reset.
    rec = {(e['run_id'], e['file']): e['sha256'] for e in json.loads((C2/'recovered'/'RECOVERY.json').read_text())['files']}
    bad, bad_w, bad_t, upd = [], [], [], []
    for arm, s in resets:
        c, src, sw = cfgs[(arm, s)], f'r1-comp-rnn-inter-s{s}-a1', arm == 'drop-reset'
        rf = c['resumed_from']
        ok = (rf['path'] == f'research/composition-c2/recovered/{src}/ckpt-1000.npz' and rf['sha256'] == rec[(src, 'ckpt-1000.npz')] == nets.sha256_file(REC/src/'ckpt-1000.npz')
              and rf['step'] == R and rf['source']['run_id'] == src and c['schedule']['reset_at'] == R and c['run_id'] == run_id(arm, s)
              and c['task_id'] == 'ops51000-ls51100' and c['transition'] == c['readout'] == 'tied' and c['step_weights'] == [1., 1., 1.]
              and c['hidden'] == 128 and c['lr'] == 1e-3 and c['wd'] == .1 and c['updates'] == END
              and (c['switch_step'], c['switch_weights'], c['taper_updates']) == ((R, [0., 0., 1.], 1) if sw else (None, None, 1)))
        r1row = next(r for r in r1[('inter', s)] if r['step'] == R)
        ok &= traj[(arm, s)][0]['step'] == R and all(traj[(arm, s)][0][k] == r1row[k] for k in r1row)
        ex = c['exposure']
        for key, start in (('labels_this_run', R), ('labels_from_init', 0)):
            m = my_exposure(384, start, END, R if sw else None, 1)
            ok &= ex[key]['updates_with_weight'] == m['updates_with_weight'] and np.allclose(ex[key]['weighted_row_presentations'], m['weighted_row_presentations'], rtol=1e-12, atol=0)
        if not ok:
            bad.append((arm, s))
        bad_w += [(arm, s, r['step']) for r in traj[(arm, s)] if r['step_weights'] != my_w(r['step'], R if sw else None, 1)]
        for t in FIXED:
            _, st, meta = nets.load(ck[(arm, s, t)])
            if st['t'] != t - R or meta['step'] != t:
                bad_t.append((arm, s, t, st['t']))
        p0 = nets.load(ck[(arm, s, R)])[0]
        g = c2rnn.loss_grad(p0, TR['s'], TR['ops'], [TR['states'][:, t] for t in range(3)], W['drop' if sw else 'keep'])[2]
        p1 = my_adam(p0, g, {n: np.zeros_like(x) for n, x in p0.items()}, {n: np.zeros_like(x) for n, x in p0.items()}, 1)[0]
        dg = traj[(arm, s)][0]['diag']
        upd.append((abs(norm({n: p1[n] - p0[n] for n in p0}) / dg['update_norm'] - 1), abs(norm(g) / dg['grad_norm'] - 1)))
    check('F1 the 6 reset configs resume r1 tied/inter ckpt-1000 at its RECOVERY.json hash, record reset_at 1000, the right flags and run ID; first row equals r1 at 1000; exposure equals my recount', not bad, str(bad))
    check('F1 step_weights equal my w(t) at every logged row of the 6 reset runs', not bad_w, str(bad_w[:3]))
    check('F1 the saved Adam count is step - 1000 at every committed checkpoint (2000 at 3k, 49000 at 50k), meta step global', not bad_t, str(bad_t[:3]))
    check("F1 the logged update and gradient norms at row 1000 equal my fresh-state Adam update's (1e-12 relative)", max(max(u) for u in upd) <= 1e-12, f'max rel {max(max(u) for u in upd):.1e}')

    # F2. Matched controls.
    vk = all(all(r[k] == next(x for x in r1[('inter', s)] if x['step'] == r['step'])[k] for k in ('step', 'train', 'held', 'held_final_pred')) for s in SEEDS for r in ver[('keep', s)])
    check('F2 the three carried-keep verify runs (1k-3k) equal r1 tied/inter in every r1 key at every logged row', vk and all(len(ver[('keep', s)]) == 41 for s in SEEDS))
    bd = {r['step']: r for r in traj[('drop', 51200)]}; vd = ver[('drop', 51200)]
    same = all(r == bd[r['step']] for r in vd[:-1]) and 'diag' not in vd[-1] and vd[-1] == {k: v for k, v in bd[3000].items() if k != 'diag'}
    (pa, sa, _), (pb, sb, _) = nets.load(RUNS/'c3-scaf-verify-carried-drop-at1k-s51200-a1'/'ckpt-3000.npz'), nets.load(ck[('drop', 51200, 3000)])
    bit = sa['t'] == sb['t'] and all(np.array_equal(x[n], y[n]) for x, y in ((pa, pb), (sa['m'], sb['m']), (sa['v'], sb['v'])) for n in pa)
    check('F2 the carried-drop verify run equals the drop branch in every key incl. diag (its last row has none), and its ckpt-3000 equals the branch bitwise', same and bit, f'rows {same}, ckpt {bit}')

    # F3. Records, rescoring, frozen evaluation.
    plan = {e['run_id'] for e in json.loads((LANE/'plan.json').read_text())['runs']}
    ev = {}
    for e in (json.loads(x) for x in (LANE/'registry.jsonl').read_text().splitlines()):
        ev.setdefault(e['run_id'], []).append(e['status'])
    check('F3 registry: each of the plan entries has exactly one running and one complete event', set(ev) == plan and all(v == ['running', 'complete'] for v in ev.values()), f'{len(plan)} entries')
    bad_m, bad_s, n = [], [], 0
    dirs = [RUNS/run_id(arm, s) for arm, s in resets] + [RUNS/f'c3-scaf-verify-carried-{k}-at1k-s{s}-a1' for k, s in [('keep', 51200), ('keep', 51201), ('keep', 51202), ('drop', 51200)]]
    for d in dirs:
        for e in json.loads((d/'MANIFEST.json').read_text()):
            if e['committed'] and nets.sha256_file(d/e['file']) != e['sha256']:
                bad_m.append((d.name, e['file']))
        logged = {r['step']: r for r in rows(d/'metrics.jsonl')}
        for f in sorted(d.glob('ckpt-*.npz')):
            n += 1; t = int(f.stem.split('-')[1]); mine, row = score(nets.load(f)[0]), logged[t]
            if not (mine['held_final_pred'] == row['held_final_pred'] and all(mine[x][h]['acc'] == row[x][h]['acc'] and close(mine[x][h]['ce'], row[x][h]['ce'], 1e-9) for x in ('train', 'held') for h in HEADS)):
                bad_s.append((d.name, t))
    check('F3 every committed file of the 10 follow-up run directories matches its MANIFEST.json', not bad_m, str(bad_m))
    check(f'F3 every committed follow-up checkpoint ({n}) rescores with my forward to its logged row', not bad_s and n == 37, f'{n}; {bad_s[:3]}')
    fz = {k: frozen(nets.load(p)[0]) for k, p in ck.items()}
    lf = read_csv(FIG/'scaf_followup_frozen.csv')
    key = lambda r: (r['arm'], int(r['seed']), int(r['step']))
    bad4 = [key(r) for r in lf if any(float(r[m]) != fz[key(r)][m] for m in ('L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'held_L3', 'held_N', 'held_notN'))]
    check(f'F3 every value of scaf_followup_frozen.csv ({len(lf)} rows x 9) equals my exhaustive evaluation; same checkpoint set', not bad4 and {key(r) for r in lf} == set(fz), f'{bad4[:3]}; mine {len(fz)}')

    # F4. Arithmetic: effects, thresholds, optimizer, transient, compatibility, trajectories.
    v = lambda arm, s, t, m: fz[(arm, s, t)][m] if (arm, s, t) in fz else None
    eff = {}
    for s in SEEDS:
        for m in FU:
            for t in FIXED:
                x = {arm: v(arm, s, t, m) for arm in ARMS}
                dk = x['keep-reset'] - x['keep'] if x['keep'] is not None else None
                dd = x['drop-reset'] - x['drop']
                st = lambda d: '' if t != END or d is None else 'below carried' if d <= -.05 else 'above carried' if d >= .05 else 'within 5 pp'
                eff[(s, m, t)] = dict(A=x['keep'], A_reset=x['keep-reset'], B=x['drop'], B_reset=x['drop-reset'], D=x['final'], d_reset_keep=dk, d_reset_drop=dd,
                                      difference=dd - dk if dk is not None else None, B_reset_minus_D=x['drop-reset'] - x['final'] if x['final'] is not None else None,
                                      d_reset_keep_status=st(dk), d_reset_drop_status=st(dd))
    le = read_csv(FIG/'scaf_followup_effects.csv')
    bade = [r for r in le if not all((close(num(r[k]), eff[(int(r['seed']), r['measure'], int(r['step']))][k]) if not k.endswith('status') else r[k] == eff[(int(r['seed']), r['measure'], int(r['step']))][k])
                                     for k in eff[(51200, 'L6', END)])]
    check(f'F4 every value of scaf_followup_effects.csv ({len(le)} rows) recomputes from my frozen values, statuses included', not bade and len(le) == len(eff), f'{len(bade)} differ')
    full = {}
    for (arm, s), rs in traj.items():
        full[(arm, s)] = rs if arm in ('keep', 'final') else [r for r in r1[('inter', s)] if r['step'] < R] + rs
    thr = {}
    for (arm, s), rs in full.items():
        steps = [r['step'] for r in rs]
        for name, acc in (('held_final', [r['held']['final']['acc'] for r in rs]), ('held_notN', [subsets(r['held_final_pred'])['notN'] for r in rs])):
            thr[(arm, s, name)] = crossing(steps, acc)
    lt = read_csv(FIG/'scaf_followup_threshold.csv')
    badt = [(r['arm'], r['seed'], r['measure']) for r in lt if num(r['crossing_step']) != thr[(r['arm'], int(r['seed']), r['measure'])]]
    check(f'F4 every crossing in scaf_followup_threshold.csv ({len(lt)}) recomputes from the joined trajectories', not badt and len(lt) == len(thr), str(badt))
    # keep's rows over 1k-3k carry diag from the verify runs (r1 keys equal, F2), as in the lane's figure CSV
    keep_diag = {(s, r['step']): r for s in SEEDS for r in ver[('keep', s)]}
    tr_rows = {(arm, s, r['step']): (keep_diag.get((s, r['step']), r) if arm == 'keep' else r) for (arm, s), rs in traj.items() for r in rs}
    un = lambda arm, s, t: tr_rows[(arm, s, t)].get('diag', {}).get('update_norm')
    lr_ = read_csv(FIG/'scaf_followup_transient.csv')
    badr = []
    for r in lr_:
        s, t = int(r['seed']), int(r['step']); x, y = un(r['arm'], s, t), un(r['arm'][:-6], s, t)
        if not (close(float(r['update_norm']), x) and close(float(r['carried']), y) and close(float(r['ratio']), x / y)):
            badr.append((r['arm'], s, t))
    check(f'F4 every value of scaf_followup_transient.csv ({len(lr_)} rows) recomputes from the logged diag (keep carried from the verify runs)', not badr and len(lr_) == 66, str(badr[:2]))
    lo = read_csv(FIG/'scaf_followup_optimizer.csv')
    bado = []
    for r in lo:
        dg = [(x['step'], x['diag']) for x in traj[(r['arm'], int(r['seed']))] if 'diag' in x]
        mine = dict(first_diag_step=dg[0][0], median_below_eps_step=next((t for t, x in dg if x['adam_sqrt_vhat']['median'] < 1e-8), None),
                    all_below_eps_step=next((t for t, x in dg if x['adam_sqrt_vhat']['frac_below_eps'] >= 1.), None))
        if r['S']:
            u = dict((t, x['update_norm']) for t, x in dg)
            mine.update({f'update_norm_ratio_S+{k}': u[R + k] / u[R] for k in (50, 500, 2000, 5000)})
        if not all(close(num(r[k]), x) for k, x in mine.items()):
            bado.append((r['arm'], r['seed']))
    check(f'F4 every value of scaf_followup_optimizer.csv ({len(lo)} rows) recomputes from the logged diag', not bado, str(bado))
    lc = read_csv(FIG/'scaf_followup_compatibility.csv')
    badc = []
    for r in lc:
        pred = np.array([int(c) for c in tr_rows[(r['arm'], int(r['seed']), int(r['step']))]['held_final_pred']]); wrong = np.flatnonzero(pred != Y)
        f = lambda sets: (float(np.mean([pred[i] in sets[i] for i in wrong])), float(np.mean([len(sets[i] - {int(Y[i])})/7 for i in wrong]))) if len(wrong) else (None, None)
        (x1, u1), (x2, u2) = f(NB_ANY), f(NB_PRE)
        if not all(close(num(r[k]), x) for k, x in dict(n_wrong=len(wrong), any_nbr=x1, any_nbr_uniform=u1, same_prefix=x2, same_prefix_uniform=u2).items()):
            badc.append((r['arm'], r['seed'], r['step']))
    check(f'F4 every value of scaf_followup_compatibility.csv ({len(lc)} rows) recomputes with my neighbour sets', not badc, str(badc[:3]))
    lm = read_csv(FIG/'scaf_followup.csv')
    badm, nv = [], 0
    for r in lm:
        arm, s, t = r['arm'], int(r['seed']), int(r['step']); row = tr_rows[(arm, s, t)]
        w = row.get('step_weights') or ([1., 1., 1.] if arm == 'keep' else [0., 0., 1.])
        sub, dg = subsets(row['held_final_pred']), row.get('diag', {}); vh = dg.get('adam_sqrt_vhat', {}); mg = dg.get('train_final_margin', {})
        mine = dict(w_inter=w[0], weighted_train_loss=sum(wk*row['train'][h]['ce'] for wk, h in zip(w, HEADS)), held_final_acc=row['held']['final']['acc'],
                    held_final_ce=row['held']['final']['ce'], held_N_acc=sub['N'], held_notN_acc=sub['notN'], grad_norm=dg.get('grad_norm'),
                    update_norm=dg.get('update_norm'), sqrt_vhat_median=vh.get('median'), sqrt_vhat_max=vh.get('max'), frac_below_eps=vh.get('frac_below_eps'),
                    train_margin_min=mg.get('min'), train_margin_median=mg.get('median'), **{f'train_{h}_{k}': row['train'][h][k] for h in HEADS for k in ('acc', 'ce')})
        nv += len(mine)
        if not all(close(num(r[k]), x) for k, x in mine.items()):
            badm.append((arm, s, t))
    check(f'F4 every value of scaf_followup.csv ({len(lm)} rows, {nv} values) recomputes from the logged rows', not badm and len(lm) == len(tr_rows), f'{badm[:3]}; mine {len(tr_rows)}')
    lab = {}
    for s in SEEDS:
        dd = [eff[(s, m, END)]['d_reset_drop'] for m in ('held_L3', 'L6')]; dk = [eff[(s, m, END)]['d_reset_keep'] for m in ('held_L3', 'L6')]
        lab[s] = ('weights at S suffice' if all(abs(x) < .05 for x in dd) else 'reset perturbation' if min(dd) <= -.05 and min(dk) <= -.05
                  else 'moments implicated' if min(dd) <= -.05 else 'mixed')
    check("F4 interpretation label per seed: 'weights at S suffice' in all three", all(x == 'weights at S suffice' for x in lab.values()), str(lab))

    # CX-F1. The reset's burst does not slow early acquisition.
    a3 = {s: next(r for r in r1[('inter', s)] if r['step'] == 3000)['held']['final']['acc'] for s in SEEDS}
    cmp_ = {}
    for t in (3000, 10000):
        cmp_[('B', t)] = [v('drop-reset', s, t, 'held_L3') - v('drop', s, t, 'held_L3') for s in SEEDS]
        cmp_[('A', t)] = [v('keep-reset', s, t, 'held_L3') - (a3[s] if t == 3000 else v('keep', s, t, 'held_L3')) for s in SEEDS]
    for k_, x in cmp_.items():
        print(f'     CX-F1 {k_[0]}-prime minus {k_[0]} on held-out L3 at {k_[1]}: {[round(y, 4) for y in x]}')
    check('CX-F1 (prediction) B-prime >= B - 0.02 on held-out L3 at 3k and at 10k in >= 2 of 3 seeds', all(sum(y >= -.02 for y in cmp_[('B', t)]) >= 2 for t in (3000, 10000)))
    check('CX-F1 (prediction) A-prime >= A - 0.02 likewise (A at 3k from its logged row)', all(sum(y >= -.02 for y in cmp_[('A', t)]) >= 2 for t in (3000, 10000)))

    # NOTE section F numbers.
    e50 = [eff[(s, m, END)] for s in SEEDS for m in FU]
    check('NOTE F1: A-prime and B-prime at 50k are 1.000 on held L3, notN, L4; >= 0.992 on L5, L6; both reset effects within 1 pp on every measure',
          all(fz[(arm, s, END)][m] == 1. for arm in ('keep-reset', 'drop-reset') for s in SEEDS for m in ('held_L3', 'held_notN', 'L4'))
          and round(min(fz[(arm, s, END)][m] for arm in ('keep-reset', 'drop-reset') for s in SEEDS for m in ('L5', 'L6')), 3) >= .992
          and max(max(abs(x['d_reset_keep']), abs(x['d_reset_drop'])) for x in e50) <= .01,
          f"min L5/L6 {min(fz[(arm, s, END)][m] for arm in ('keep-reset', 'drop-reset') for s in SEEDS for m in ('L5', 'L6')):.4f}; max |effect| {max(max(abs(x['d_reset_keep']), abs(x['d_reset_drop'])) for x in e50):.4f}")
    tab = {(arm, m): [thr[(arm, s, m)] for s in SEEDS] for arm in ('keep', 'keep-reset', 'drop', 'drop-reset') for m in ('held_final', 'held_notN')}
    want = {('keep', 'held_final'): [7700, 9150, 8200], ('keep', 'held_notN'): [8650, 11200, 9000], ('keep-reset', 'held_final'): [8150, 8750, 8950],
            ('keep-reset', 'held_notN'): [9050, 11000, 9250], ('drop', 'held_final'): [14600, 20000, 20450], ('drop', 'held_notN'): [16650, 24100, 22800],
            ('drop-reset', 'held_final'): [12250, 14100, 16350], ('drop-reset', 'held_notN'): [12550, 17950, 18600]}
    check('NOTE F2 crossing table', tab == want, str({k: x for k, x in tab.items() if x != want[k]}))
    dd_all = [eff[(s, m, t)]['d_reset_drop'] for s in SEEDS for m in FU for t in FIXED]
    d10 = [eff[(s, m, 10000)]['d_reset_drop'] for s in SEEDS for m in FU]
    h10 = [round(eff[(s, 'held_L3', 10000)]['d_reset_drop'], 3) for s in SEEDS]
    l20 = [round(eff[(s, 'L6', 20000)]['d_reset_drop'], 2) for s in SEEDS]
    dk_all = [eff[(s, m, t)]['d_reset_keep'] for s in SEEDS for m in FU for t in FIXED if eff[(s, m, t)]['d_reset_keep'] is not None]
    df20 = [round(eff[(s, 'L6', 20000)]['difference'], 2) for s in SEEDS]
    check('NOTE F2: B-prime - B >= 0 everywhere and > 0 on every measure at 10k; held 10k +0.055/+0.125/+0.125; L6 20k +0.25/+0.33/+0.18; A-prime - A in [-0.06, +0.09]; difference L6 20k +0.25/+0.31/+0.18',
          min(dd_all) >= 0 and min(d10) > 0 and h10 == [.055, .125, .125] and l20 == [.25, .33, .18] and rng(dk_all, 2)[0] >= -.06 and rng(dk_all, 2)[1] <= .09 and df20 == [.25, .31, .18],
          f'min {min(dd_all):.4f}; 10k min {min(d10):.4f}; {h10} {l20}; A-prime - A {rng(dk_all, 4)} (A from 10k on); {df20}')
    h6 = [round(fz[('drop-reset', 51201, 20000)]['L6'], 3), round(fz[('drop', 51201, 20000)]['L6'], 3)]
    check('NOTE handoff 6: L6 at 20k s51201, B-prime 0.766 against B 0.438', h6 == [.766, .438], str(h6))
    lag = all(thr[('drop-reset', s, 'held_final')] > max(thr[('keep', s, 'held_final')], thr[('keep-reset', s, 'held_final')]) for s in SEEDS)
    check('NOTE F2: B-prime crosses after A and A-prime in every seed; D censored', lag and all(thr[('final', s, m)] is None for s in SEEDS for m in ('held_final', 'held_notN')))
    first = [x['update_norm'] / y for x, y in ((r[0]['diag'], un(arm[:-6], s, R)) for (arm, s), r in ((k, traj[k]) for k in resets))]
    later = [un(arm, s, t) / un(arm[:-6], s, t) for arm, s in resets for t in range(1050, 1501, 50)]
    early = [un(arm, s, t) / un(arm[:-6], s, t) for arm, s in resets for t in (1050, 1100)]
    check('NOTE F3: first update after the reset 17-22x the carried one; 1.15-1.75 over 1,050-1,500; 1.37-1.75 at 1,050 and 1,100, above 1.5 in 7 of 12',
          rng(first, 0) == [17, 22] and rng(later, 2) == [1.15, 1.75] and rng(early, 2) == [1.37, 1.75] and sum(x > 1.5 for x in early) == 7,
          f'{rng(first, 2)} {rng(later, 3)} {rng(early, 3)} {sum(x > 1.5 for x in early)}')
    ce = lambda arm, s, t: tr_rows[(arm, s, t)]['train']['final']['ce']
    rise = [(arm, s, t) for arm, s in resets for t in range(1050, 1301, 50) if ce(arm, s, t) > ce(arm, s, t - 50)]
    above = [(arm, s, t) for arm, s in resets for t in range(1050, 6001, 50) if ce(arm, s, t) >= ce(arm[:-6], s, t)]
    check('NOTE F3: train final CE never rises over 1,000-1,300 in a reset run, and is below carried at every logged row over 1,050-6,000 except 1,050 in s51202',
          not rise and sorted(above) == [('drop-reset', 51202, 1050), ('keep-reset', 51202, 1050)], f'rises {rise}; not below {above}')
    on = {arm: [next(t for t, x in ((r['step'], r['diag']) for r in traj[(arm, s)] if 'diag' in r) if x['adam_sqrt_vhat']['frac_below_eps'] >= 1.) for s in SEEDS] for arm in ('keep-reset', 'drop-reset', 'drop')}
    on['keep'] = [next(t for t, x in ((r['step'], r['diag']) for r in traj[('keep', s)] if 'diag' in r) if x['adam_sqrt_vhat']['frac_below_eps'] >= 1.) for s in SEEDS]
    check('NOTE F4: eps onsets A-prime 28,600/28,750/30,200 (A 29,250/29,050/29,300), B-prime 28,450/28,650/30,250 (B same as A)',
          on['keep-reset'] == [28600, 28750, 30200] and on['keep'] == [29250, 29050, 29300] and on['drop-reset'] == [28450, 28650, 30250] and on['drop'] == on['keep'], str(on))
    f50 = {arm: [ce(arm, s, END) for s in SEEDS] for arm in ('keep-reset', 'drop-reset')}
    print(f"     NOTE F4 train final CE at 50k: A-prime {f50['keep-reset']}; B-prime {f50['drop-reset']}")
    low = lambda arm, s: min(min(r['train']['step1']['acc'], r['train']['step2']['acc']) for r in traj[(arm, s)] if r['step'] >= R)
    lw = [round(min(low(arm, s) for s in SEEDS), 3) for arm in ('drop-reset', 'drop')]
    at50 = all(min(tr_rows[(arm, s, END)]['train']['step1']['acc'], tr_rows[(arm, s, END)]['train']['step2']['acc']) == 1. for arm in ('drop', 'drop-reset') for s in SEEDS)
    check('NOTE F5: lowest h1/h2 training accuracy from 1k is 0.906 in B-prime and 0.922 in B; 1.0 at 50k in both', lw == [.906, .922] and at50, str(lw))
    o7 = [float(r['update_norm_ratio_S+2000']) for r in read_csv(FIG/'scaf_optimizer.csv') if r['arm'] == 'drop' and r['S'] == '1000']
    check('NOTE interpretation: carried B update norm at S+2000 is 0.53-0.56 of its value at S (O7)', rng(o7, 2) == [.53, .56], str(rng(o7, 4)))

    (a.out/'rev_scaf_fu.json').write_text(json.dumps(dict(
        frozen={'|'.join(map(str, k)): {m: x for m, x in d.items() if m != 'pred'} for k, d in fz.items()},
        effects={'|'.join(map(str, k)): x for k, x in eff.items()}, threshold={'|'.join(map(str, k)): x for k, x in thr.items()},
        cx_f1={f'{k[0]}|{k[1]}': x for k, x in cmp_.items()}), indent=1))


if __name__ == '__main__':
    main()
    print('ALL PASS' if not FAIL else f'{len(FAIL)} FAILED: {FAIL}')
