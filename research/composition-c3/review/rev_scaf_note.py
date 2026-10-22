"""The scaffolding NOTE's numbers (4512bce), checked against my recomputation (rev_scaf.json) and the
lane's CSVs that rev_scaf.py found equal to the logged rows (S4-S7). Reads only.

python3 research/composition-c3/review/rev_scaf_note.py runs/c3/c3-rev-scaf-a1
"""
import csv
import json
from pathlib import Path
import sys

FAIL = []
SEEDS, END = (51200, 51201, 51202), 50000
BR = [(k, S, s) for S in (1000, 10000) for k in ('drop', 'taper') for s in SEEDS]


def check(name, ok, detail=''):
    print(('PASS ' if ok else 'FAIL ') + name + (f'  [{detail}]' if detail else ''), flush=True)
    if not ok:
        FAIL.append(name)


def rng(xs, d):
    return [round(min(xs), d), round(max(xs), d)]


def main(out):
    out = Path(out); J = json.loads((out/'rev_scaf.json').read_text())
    fig = out/'lane'/'research'/'composition-c3'/'scaffolding'/'figures'
    rd = lambda n: list(csv.DictReader((fig/n).open()))
    fz = lambda arm, S, s, t: J['frozen'][f'{arm}|{S}|{s}|{t}']
    ret = lambda k, S, s, m, t: J['retention'][f'{k}|{S}|{s}|{m}|{t}']
    thr = lambda arm, S, s, m: J['threshold'][f'{arm}|{S}|{s}|{m}']['crossing_step']
    main_ = rd('scaf_main.csv'); opt = rd('scaf_optimizer.csv'); cmp_ = rd('scaf_compatibility.csv')
    row = lambda arm, S, s: [r for r in main_ if r['arm'] == arm and r['S'] == (str(S) if S else '') and int(r['seed']) == s]

    # O1.
    st = [ret(k, S, s, m, END)['status'] for k, S, s in BR for m in ('held_L3', 'held_notN', 'L1', 'L2', 'L4', 'L5', 'L6')]
    check('O1 84 of 84 endpoint labels are retained', st.count('retained') == 84 == len(st))
    e = {m: [fz(k, S, s, END)[m] for k, S, s in BR] for m in ('held_L3', 'held_notN', 'L4', 'L5', 'L6')}
    check('O1 branches at 50k: held L3, notN and L4 all 1.000; L5 >= 0.998; L6 0.990-1.000',
          min(e['held_L3']) == min(e['held_notN']) == min(e['L4']) == 1. and round(min(e['L5']), 3) >= .998 and rng(e['L6'], 3) == [.99, 1.], f"L5 min {min(e['L5']):.4f}; L6 {rng(e['L6'], 4)}")
    D = {m: [fz('final', None, s, END)[m] for s in SEEDS] for m in ('held_L3', 'held_notN', 'L4', 'L5', 'L6')}
    check('O1 final-only at 50k: held 0.56 / 0.70 / 0.66, notN 0.12 / 0.26 / 0.14, L4-6 0.10-0.15',
          [round(x, 2) for x in D['held_L3']] == [.56, .70, .66] and [round(x, 2) for x in D['held_notN']] == [.12, .26, .14]
          and rng(D['L4'] + D['L5'] + D['L6'], 2) == [.10, .15], f"{D['held_L3']} {D['held_notN']} {rng(D['L4'] + D['L5'] + D['L6'], 4)}")
    dd = [ret(k, S, s, 'held_L3', END)['d_final'] for k, S, s in BR]; d46 = [ret(k, S, s, m, END)['d_final'] for k, S, s in BR for m in ('L4', 'L5', 'L6')]
    check('O1 delta_D(50k): +0.30 to +0.44 on held L3 and +0.85 to +0.91 on L4-6', rng(dd, 2) == [.30, .44] and rng(d46, 2) == [.85, .91], f'{rng(dd, 4)} {rng(d46, 4)}')

    # O2 (S = 1k) and the crossings.
    tab = {(arm, m): [thr(arm, None if arm in ('keep', 'final') else 1000, s, m) for s in SEEDS] for arm in ('keep', 'drop', 'taper', 'final') for m in ('held_final', 'held_notN')}
    want = {('keep', 'held_final'): [7700, 9150, 8200], ('keep', 'held_notN'): [8650, 11200, 9000], ('drop', 'held_final'): [14600, 20000, 20450],
            ('drop', 'held_notN'): [16650, 24100, 22800], ('taper', 'held_final'): [12050, 21400, 15300], ('taper', 'held_notN'): [14800, 23400, 18550],
            ('final', 'held_final'): [None]*3, ('final', 'held_notN'): [None]*3}
    check('O2 crossing table (keep, drop and taper at 1k, final-only censored)', tab == want, str({k: v for k, v in tab.items() if v != want[k]}))
    x = ret('drop', 1000, 51202, 'held_L3', 10000)['d_keep'], ret('drop', 1000, 51202, 'held_notN', 10000)['d_keep'], ret('drop', 1000, 51202, 'L6', 20000)['d_keep']
    check('O2 drop s51202: held delta_keep -0.39 and notN -0.86 at 10k, L6 -0.57 at 20k', [round(v, 2) for v in x] == [-.39, -.86, -.57], str(x))
    where = {(k, s, m): min((ret(k, 1000, s, m, t)['d_keep'], t) for t in (10000, 20000, 30000, 40000, 50000))[1]
             for k in ('drop', 'taper') for s in SEEDS for m in ('held_L3', 'held_notN', 'L4', 'L5', 'L6')}
    check('O2 at S=1k the most negative delta_keep of each branch and measure is at 10k or 20k', all(t in (10000, 20000) for t in where.values()), str({k: t for k, t in where.items() if t not in (10000, 20000)}))
    l6 = [fz(k, 1000, s, t)['L6'] for k in ('drop', 'taper') for s in SEEDS for t in (40000, 50000)]
    kp = [fz('keep', None, s, 20000)['L6'] for s in SEEDS]
    check('O2 L6 >= 0.97 by 40k in every S=1k branch; keep L6 rounds to 1.00 at 20k', min(l6) >= .97 and all(round(v, 2) == 1. for v in kp), f'{min(l6):.4f}; keep {kp}')
    db = [ret(k, 1000, s, 'held_L3', END)['d_branch'] for k in ('drop', 'taper') for s in SEEDS]; d6 = [ret(k, 1000, s, 'L6', END)['d_branch'] for k in ('drop', 'taper') for s in SEEDS]
    check('O2 delta_branch(50k) at S=1k: +0.64 to +0.66 held, +0.85 to +0.86 L6', rng(db, 2) == [.64, .66] and rng(d6, 2) == [.85, .86], f'{rng(db, 4)} {rng(d6, 4)}')

    # O3 (S = 10k, length 3).
    falls = []
    for k in ('drop', 'taper'):
        for s in SEEDS:
            for col in ('held_final_acc', 'held_notN_acc'):
                v = [float(r[col]) for r in row(k, 10000, s)]
                i = next((j for j, y in enumerate(v) if y == 1.), None)
                if i is None or min(v[i:]) < 1.:
                    falls.append((k, s, col))
    check('O3 at S=10k no held-out or notN logged row falls below 1.0 after its first 1.0', not falls, str(falls))
    at = [fz('keep', None, s, 10000)['held_L3'] for s in SEEDS], fz('keep', None, 51201, 10000)['held_notN']
    nn = [thr('drop', 10000, 51201, 'held_notN'), thr('taper', 10000, 51201, 'held_notN'), thr('keep', None, 51201, 'held_notN')]
    check('O3 at S: 1.0 / 0.961 / 1.0 held (s51201 notN 0.88); s51201 notN crossings 11,300 drop, 11,150 taper, 11,200 keep',
          [round(v, 3) for v in at[0]] == [1., .961, 1.] and round(at[1], 2) == .88 and nn == [11300, 11150, 11200], f'{at} {nn}')

    # O4 (S = 10k, lengths 4-6).
    L6 = lambda arm, S, t: [round(fz(arm, S, s, t)['L6'], 2) for s in SEEDS]
    check('O4 L6 at 10k 0.86 / 0.56 / 0.72; at 20k drop 0.93 / 0.65 / 0.98, taper 0.95 / 0.73 / 0.95, keep 1.00 / 0.99 / 1.00',
          L6('keep', None, 10000) == [.86, .56, .72] and L6('drop', 10000, 20000) == [.93, .65, .98] and L6('taper', 10000, 20000) == [.95, .73, .95]
          and L6('keep', None, 20000) == [1., .99, 1.], f"{L6('keep', None, 10000)} {L6('drop', 10000, 20000)} {L6('taper', 10000, 20000)} {L6('keep', None, 20000)}")
    m6 = min((ret(k, 10000, s, 'L6', t)['d_keep'], k, s, t) for k in ('drop', 'taper') for s in SEEDS for t in (20000, 30000, 40000, 50000))
    m5 = min(ret(k, 10000, s, 'L5', t)['d_keep'] for k in ('drop', 'taper') for s in SEEDS for t in (20000, 30000, 40000, 50000))
    check('O4 lowest delta_keep at S=10k: L6 -0.34 (drop s51201, 20k), L5 -0.13', round(m6[0], 2) == -.34 and m6[1:] == ('drop', 51201, 20000) and round(m5, 2) == -.13, f'{m6} {m5:.4f}')
    a40 = [fz(k, 10000, s, 40000)['L6'] for k in ('drop', 'taper') for s in SEEDS]; a50 = [fz(k, 10000, s, END)['L6'] for k in ('drop', 'taper') for s in SEEDS]
    dip = [round(fz('drop', 10000, 51200, t)['L6'], 3) for t in (12000, 20000, 30000)]
    check('O4 S=10k L6 >= 0.993 at 40k and >= 0.998 at 50k; drop s51200 L6 0.950 / 0.929 / 0.984 at 12k / 20k / 30k',
          round(min(a40), 3) >= .993 and round(min(a50), 3) >= .998 and dip == [.95, .929, .984], f'{min(a40):.4f} {min(a50):.4f} {dip}')
    h = [round(fz('drop', 10000, 51201, 20000)['L6'], 3), round(fz('drop', 1000, 51202, END)['L6'], 3)]
    check('handoff 2: drop-at10k-s51201 L6 at 20k 0.649; drop-at1k-s51202 L6 at 50k 0.993', h == [.649, .993], str(h))

    # O5 (lengths 1-2).
    after = [(float(r['train_step1_acc']), float(r['train_step2_acc']), k, S, s, int(r['step'])) for k, S, s in BR for r in row(k, S, s) if int(r['step']) > S]
    m1, m2 = min(after, key=lambda x: x[0]), min(after, key=lambda x: x[1])
    ten = [x for x in after if x[3] == 10000]
    check('O5 lowest logged train accuracy after S: step 1 0.97, step 2 0.92 at drop 1k s51201 step 7,100; 1.0 at every row after a 10k switch',
          round(m1[0], 2) == .97 and round(m2[1], 2) == .92 and m2[2:] == ('drop', 1000, 51201, 7100) and all(x[0] == x[1] == 1. for x in ten), f'step1 {m1}; step2 {m2}')
    lf = [fz(k, S, s, t)[m] for k, S, s in BR for t in sorted({S, S + 2000, *[x for x in (10000, 20000, 30000, 40000, END) if x >= S]}) for m in ('L1', 'L2')]
    check('O5 frozen L1, L2 >= 0.95 at every fixed time of every branch, 1.0 at 50k', min(lf) >= .95 and all(fz(k, S, s, END)[m] == 1. for k, S, s in BR for m in ('L1', 'L2')), f'min {min(lf):.4f}')
    ce = lambda arm, S, s, h: float(row(arm, S, s)[-1][f'train_{h}_ce'])
    c1, c2 = [ce(k, S, s, 'step1') for k, S, s in BR], [ce(k, S, s, 'step2') for k, S, s in BR]
    k1, k2 = [ce('keep', None, s, 'step1') for s in SEEDS], [ce('keep', None, s, 'step2') for s in SEEDS]
    f1, f2 = [ce('final', None, s, 'step1') for s in SEEDS], [ce('final', None, s, 'step2') for s in SEEDS]
    print(f'     O5 train CE at 50k: branches step1 {min(c1):.2e}-{max(c1):.2e}, step2 {min(c2):.2e}-{max(c2):.2e}; keep {min(k1):.1e}-{max(k1):.1e} / {min(k2):.1e}-{max(k2):.1e}; final-only {f1} / {f2}')
    check('O5 CE at 50k: branches about 1e-5 / 2e-7 (within 2x), keep about 5e-9 / 5e-9 (within 2x), final-only 8.6 / 11',
          all(5e-6 <= v <= 2e-5 for v in c1) and all(1e-7 <= v <= 4e-7 for v in c2) and all(2.5e-9 <= v <= 1e-8 for v in k1 + k2)
          and round(sum(f1)/3, 1) == 8.6 and round(sum(f2)/3) == 11)

    # O6 and O7.
    gap = max(abs(fz('taper', S, s, END)[m] - fz('drop', S, s, END)[m]) for S in (1000, 10000) for s in SEEDS for m in ('held_L3', 'held_notN', 'L1', 'L2', 'L3', 'L4', 'L5', 'L6'))
    check('O6 taper within 0.05 of drop on every measure at 50k, both S', gap < .05, f'max gap {gap:.4f}')
    o = {(r['arm'], r['S'], int(r['seed'])): r for r in opt}
    r50 = {k: float(v['update_norm_ratio_S+50']) for k, v in o.items() if v['S']}
    rest = [v for k, v in r50.items() if k != ('taper', '1000', 51202)]
    r2k = [float(v['update_norm_ratio_S+2000']) for v in o.values() if v['S']]
    check('O7 update norm at S+50 / S: 0.76-0.98 in 11 branches, 1.01 in taper-at1k-s51202; at S+2000 0.45-0.90',
          rng(rest, 2) == [.76, .98] and round(r50[('taper', '1000', 51202)], 2) == 1.01 and rng(r2k, 2) == [.45, .90], f'{rng(rest, 4)} {r50[("taper", "1000", 51202)]:.4f} {rng(r2k, 4)}')
    allb = lambda arm: [int(v['all_below_eps_step']) for k, v in o.items() if k[0] == arm]
    med = [int(v['median_below_eps_step']) for k, v in o.items() if k[0] in ('drop', 'taper')]
    medall = [int(v['median_below_eps_step']) for v in o.values()]
    check('O7 every coordinate below eps from 29,050-29,300 in every branch and in keep; 27,800-29,450 final-only; median below eps 24,900-25,200',
          rng(allb('drop') + allb('taper'), 0) == [29050, 29300] and rng(allb('keep'), 0) == [29050, 29300] and rng(allb('final'), 0) == [27800, 29450] and rng(med, 0) == [24900, 25200],
          f"{rng(allb('drop') + allb('taper'), 0)} {rng(allb('keep'), 0)} {rng(allb('final'), 0)} branches {rng(med, 0)}, all arms {rng(medall, 0)}")
    fce = lambda arm, S: [ce(arm, S, s, 'final') for s in SEEDS]
    bfin = [v for k, S, s in BR for v in [ce(k, S, s, 'final')]]
    print(f"     O7 train final CE at 50k: branches {min(bfin):.2e}-{max(bfin):.2e}; keep {fce('keep', None)}; final-only {fce('final', None)}")

    # O8.
    cb = [r for r in cmp_ if r['arm'] in ('drop', 'taper') and int(r['n_wrong']) > 0]
    check('O8 every wrong held-out prediction of every branch at the fixed times is a one-slot neighbour label; uniform 0.67-0.75; same prefix 0.90-1.0 vs 0.29-0.39',
          all(float(r['any_nbr']) == 1. for r in cb) and rng([float(r['any_nbr_uniform']) for r in cb], 2) == [.67, .75]
          and rng([float(r['same_prefix']) for r in cb], 2) == [.90, 1.] and rng([float(r['same_prefix_uniform']) for r in cb], 2) == [.29, .39],
          f"{len(cb)} rows; uniform {rng([float(r['any_nbr_uniform']) for r in cb], 4)}; same prefix {rng([float(r['same_prefix']) for r in cb], 4)} vs {rng([float(r['same_prefix_uniform']) for r in cb], 4)}")


if __name__ == '__main__':
    main(sys.argv[1])
    print('ALL PASS' if not FAIL else f'{len(FAIL)} FAILED: {FAIL}')
