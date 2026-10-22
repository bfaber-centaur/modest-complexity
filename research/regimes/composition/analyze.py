#!/usr/bin/env python3
"""R1 composition analysis.

  publish  copy each run's small files into runs/<ID>/ and write a SHA-256 manifest
           of every file in the local run directory (checkpoints included)
  figure   the first figure (figures/heldout.png/.svg) and its CSVs, from published runs

uv run --locked python research/regimes/composition/analyze.py publish --root runs/regimes --ids ID ...
uv run --locked python research/regimes/composition/analyze.py figure
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import task  # noqa: E402

RUNS, FIGS = HERE/'runs', HERE/'figures'
CONDS = ('final', 'inter', 'shuf')
SEEDS = (51200, 51201, 51202)
COLOR = dict(final='#2a78d6', inter='#eb6834', shuf='#1baf7a')  # dataviz reference categorical slots 1-3
MARKER = dict(zip(SEEDS, 'osD'))
SMALL = ('config.json', 'metrics.jsonl', 'runtime.json')


def run_id(cond, seed, attempt=1):
    return f'r1-comp-{cond}-s{seed}-a{attempt}'


def publish(root, ids, committed_checkpoints=()):
    for rid in ids:
        src, dst = root/rid, RUNS/rid
        dst.mkdir(parents=True, exist_ok=True)
        manifest = []
        for f in sorted(src.iterdir()):
            committed = f.name in SMALL or f'{rid}/{f.name}' in committed_checkpoints
            if committed:
                shutil.copy2(f, dst/f.name)
            manifest.append(dict(file=f.name, bytes=f.stat().st_size, sha256=hashlib.sha256(f.read_bytes()).hexdigest(),
                                 committed=committed))
        log = root/f'{rid}.log'
        if log.exists():
            shutil.copy2(log, dst/'stdout.log')
        (dst/'MANIFEST.json').write_text(json.dumps(manifest, indent=1)+'\n')
        print(rid, len(manifest), 'files')


def load(rid):
    path = RUNS/rid/'metrics.jsonl'
    if not path.exists():
        return None
    return [json.loads(x) for x in path.read_text().splitlines()]


def statuses():
    out = {}
    reg = HERE/'registry.jsonl'
    for line in reg.read_text().splitlines() if reg.exists() else []:
        e = json.loads(line); out[e['run_id']] = e['status']
    return out


def pattern(q):
    a, b, c = q
    if a == b == c:
        return 'aaa'
    return 'a=b' if a == b else 'b=c' if b == c else 'a=c' if a == c else 'distinct'


def figure():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    FIGS.mkdir(exist_ok=True)
    P, _ = task.draw_ops(); L, train, held = task.split()
    base = task.checks(P, train, held)['heldout_baselines']
    he = task.rows(P, held)
    order = sorted(range(len(held)), key=lambda i: (['aaa', 'a=b', 'b=c', 'a=c', 'distinct'].index(pattern(held[i])), held[i]))
    status = statuses()
    runs = {(c, s): load(run_id(c, s)) for c in CONDS for s in SEEDS}
    interp = load('r1-comp-interp-s51200-a1')
    p2 = interp[-1]['held']['final']['acc'] if interp else None

    # Long table: every logged accuracy and cross-entropy.
    with (FIGS/'heldout_trajectories.csv').open('w', newline='') as f:
        w = csv.writer(f); w.writerow(['run_id', 'condition', 'init_seed', 'step', 'split', 'head', 'acc', 'ce'])
        for (c, s), m in runs.items():
            for row in m or []:
                for split in ('train', 'held'):
                    for h, v in row[split].items():
                        w.writerow([run_id(c, s), c, s, row['step'], split, h, v['acc'], v['ce']])
    # Per held-out program at the last logged step, and what wrong answers are.
    nb = neighbour_labels(P, train, he)
    with (FIGS/'heldout_by_program.csv').open('w', newline='') as f:
        w = csv.writer(f); w.writerow(['run_id', 'condition', 'init_seed', 'step', 'program', 'pattern', 'acc'])
        for (c, s), m in runs.items():
            if not m:
                continue
            pred = np.array(list(m[-1]['held_final_pred']), int)
            for i, q in enumerate(held):
                rows = slice(8*i, 8*i+8)
                w.writerow([run_id(c, s), c, s, m[-1]['step'], ''.join(map(str, q)), pattern(q),
                            float(np.mean(pred[rows] == he['final'][rows]))])
    # Wrong answers: are they labels of training rows that differ from the held-out row in one op slot?
    # (No training row differs only in the start symbol: held-out programs are held out at every start.)
    with (FIGS/'heldout_error_types.csv').open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['run_id', 'condition', 'init_seed', 'step', 'held_acc', 'n_wrong', 'measure', 'observed', 'expected_if_wrong_answers_uniform'])
        for (c, s), m in runs.items():
            if not m:
                continue
            pred = np.array(list(m[-1]['held_final_pred']), int)
            wrong = np.flatnonzero(pred != he['final'])
            for name, sets in [('neighbour in any op slot', [set().union(*x.values()) for x in nb]),
                               *[(f'neighbour differing in op slot {k}', [x[k] for x in nb]) for k in 'abc']]:
                obs = np.mean([pred[i] in sets[i] for i in wrong]) if len(wrong) else np.nan
                exp = np.mean([len(sets[i] - {he['final'][i]})/(task.N_SYM-1) for i in wrong]) if len(wrong) else np.nan
                w.writerow([run_id(c, s), c, s, m[-1]['step'], float(np.mean(pred == he['final'])), len(wrong), name, obs, exp])

    fig, ax = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    refs = [(base['chance'], 'chance 1/8', (0, (4, 3))), (base['3-of-4-slot neighbour vote'], '3-of-4-slot neighbour vote', ':'),
            (base['nearest seen program (oracle-selected)'], 'nearest seen program, oracle-selected bound', '-.')]
    handles = []
    for j, c in enumerate(CONDS):
        a = ax[0, j]
        for y, label, ls in refs:
            h = a.axhline(y, color='0.45', lw=1, ls=ls, zorder=1, label=label)
            if j == 0:
                handles.append(h)
        if p2 is not None:
            h = a.scatter([10000], [p2], marker='*', s=140, color='0.3', zorder=4, clip_on=False,
                          label='P2 positive control, held-out (trained transitions, composition hard-wired)')
            if j == 0:
                handles.append(h)
        missing, seed_lines = [], []
        for s in SEEDS:
            m = runs[(c, s)]
            if not m:
                missing.append(f'{run_id(c, s)}: {status.get(run_id(c, s), "not run")}'); continue
            st = [r['step'] for r in m]
            a.plot(st, [r['train']['final']['acc'] for r in m], color=COLOR[c], lw=1, ls='--', alpha=.7)
            seed_lines += a.plot(st, [r['held']['final']['acc'] for r in m], color=COLOR[c], lw=1.6, marker=MARKER[s],
                                 markevery=20, ms=5, label=f'init {s}')
        if missing:
            a.text(.02, .5, 'missing:\n'+'\n'.join(missing), transform=a.transAxes, fontsize=8, color='crimson')
        a.set(title=f'{c}: final head', xlabel='full-batch updates', ylim=(0, 1.04), xlim=(0, 10000))
        a.legend(handles=seed_lines, loc='center right', fontsize=8, title='held-out (solid)\ntrain (dashed)', title_fontsize=8)
    ax[0, 0].set_ylabel('accuracy, final answer')

    a = ax[1, 0]
    a.axhline(0, color='0.45', lw=1)
    for c in ('inter', 'shuf'):
        for s in SEEDS:
            m, m0 = runs[(c, s)], runs[('final', s)]
            if not (m and m0):
                continue
            a.plot([r['step'] for r in m], [r['held']['final']['acc'] - r0['held']['final']['acc'] for r, r0 in zip(m, m0)],
                   color=COLOR[c], lw=1.4, marker=MARKER[s], markevery=20, ms=5, label=f'{c} − final, init {s}')
    a.set(title='paired difference in held-out final accuracy\n(same init seed)', xlabel='full-batch updates',
          ylabel='accuracy difference', xlim=(0, 10000), ylim=(-.25, .25))
    a.legend(fontsize=7, ncol=2, loc='lower right')

    a = ax[1, 1]
    for c in ('inter', 'shuf'):
        for s in SEEDS:
            m = runs[(c, s)]
            if not m:
                continue
            st = [r['step'] for r in m]
            for h, ls in (('step1', ':'), ('step2', '-')):
                a.plot(st, [r['held'][h]['acc'] for r in m], color=COLOR[c], lw=1.4, ls=ls, marker=MARKER[s],
                       markevery=20, ms=4, label=f'{c} {h}' if s == SEEDS[0] else None)
    a.axhline(base['chance'], color='0.45', lw=1, ls=(0, (4, 3)))
    a.set(title='held-out intermediate heads (true labels)', xlabel='full-batch updates', ylabel='accuracy',
          ylim=(0, 1.04), xlim=(0, 10000))
    a.legend(fontsize=8, loc='center right', title='markers = init seed', title_fontsize=8)

    a = ax[1, 2]
    a.axhline(base['chance'], color='0.45', lw=1, ls=(0, (4, 3)))
    for k, c in enumerate(CONDS):
        for n, s in enumerate(SEEDS):
            m = runs[(c, s)]
            if not m:
                continue
            pred = np.array(list(m[-1]['held_final_pred']), int)
            acc = [np.mean(pred[8*i:8*i+8] == he['final'][8*i:8*i+8]) for i in order]
            a.scatter(np.arange(16) + (k-1)*.25 + (n-1)*.07, acc, s=18, color=COLOR[c], marker=MARKER[s],
                      label=c if n == 0 else None, zorder=3)
    a.set_xticks(range(16), [''.join(map(str, held[i])) for i in order], rotation=90, fontsize=8)
    groups = [pattern(held[i]) for i in order]
    for g in dict.fromkeys(groups):
        idx = [i for i, x in enumerate(groups) if x == g]
        a.text(np.mean(idx), 1.03, g, ha='center', fontsize=8, color='0.3')
        a.axvline(idx[-1]+.5, color='0.85', lw=.8)
    a.set(title='held-out accuracy per program at the last update\n(8 start symbols each)', xlabel='held-out program (ops a b c); marker = init seed as above',
          ylabel='accuracy, final answer', ylim=(-.03, 1.1))
    a.legend(fontsize=8, loc='upper right', bbox_to_anchor=(1, .95))
    fig.legend(handles=handles, loc='outside lower center', ncol=4, fontsize=8, frameon=False, title='reference levels (top row)',
               title_fontsize=8)
    fig.suptitle('R1: 20→128→128 ReLU MLP, 48 training programs, 16 held-out programs (Latin-square split); '
                 'full-batch Adam lr 1e-3, wd 0.1; each line is one init seed', fontsize=10)
    for ext in ('png', 'svg'):
        fig.savefig(FIGS/f'heldout.{ext}', dpi=130)
    print('wrote', FIGS/'heldout.png')


def neighbour_labels(P, train, he):
    """Per held-out row, {op slot: final labels of the training rows that differ from it in that slot only}."""
    tr = task.rows(P, train)
    lookup = {k: int(y) for k, y in zip(zip(tr['s'], tr['a'], tr['b'], tr['c']), tr['final'])}
    out = []
    for x in zip(*(map(int, he[k]) for k in 'sabc')):
        labels = {}
        for slot, name in zip((1, 2, 3), 'abc'):
            keys = [x[:slot] + (v,) + x[slot+1:] for v in range(task.N_OP) if v != x[slot]]
            labels[name] = {lookup[k] for k in keys if k in lookup}
        out.append(labels)
    return out


def fit_probe(H, y, Hh, yh, seed, updates=1000, lr=1e-2):
    """Linear softmax probe (nets with no hidden layer) fitted on training-program rows.
    Returns (weights, train accuracy, held-out accuracy)."""
    sys.path.insert(0, str(HERE.parent/'common')); import nets
    p = nets.init(H.shape[1], [], dict(y=task.N_SYM), seed); st = nets.adam_state(p)
    for _ in range(updates):
        nets.adam_step(p, nets.loss_grad(p, H, dict(y=y))[2], st, lr)
    acc = lambda X, t: float(np.mean(nets.forward(p, X)[1]['y'].argmax(1) == t))
    return p['W_y'], acc(H, y), acc(Hh, yh)


def probe(root, out):
    """Descriptive decoding of step1/step2/final from each layer (input, hidden 1, hidden 2) at
    initialization and at the last update; then interchange interventions on the probe subspaces.
    Checkpoints are read from the local run root and checked against the published manifests."""
    sys.path.insert(0, str(HERE.parent/'common')); import nets
    out.mkdir(parents=True, exist_ok=False)
    P, _ = task.draw_ops(); L, train, held = task.split()
    data = dict(train=task.rows(P, train), held=task.rows(P, held))
    X = {k: task.encode(d) for k, d in data.items()}
    rng = np.random.default_rng(51320)
    # Source rows for each base row: 8 per base, same split, different step2 (or step1) from the base.
    probes, inter = [], []
    for c in CONDS:
        for s in SEEDS:
            rid = run_id(c, s)
            manifest = {m['file']: m['sha256'] for m in json.loads((RUNS/rid/'MANIFEST.json').read_text())}
            for step in (0, 10000):
                f = root/rid/f'ckpt-{step}.npz'
                if hashlib.sha256(f.read_bytes()).hexdigest() != manifest[f.name]:
                    raise SystemExit(f'{f} does not match its published SHA-256')
                p = nets.load(f)[0]
                acts = {k: nets.forward(p, X[k])[0] for k in X}  # [input, h1, h2]
                for layer in (0, 1, 2):
                    if layer == 0 and (c, s, step) != (CONDS[0], SEEDS[0], 0):
                        continue  # the input layer does not depend on the run
                    for target in STEPS_ALL:
                        W, a_tr, a_he = fit_probe(acts['train'][layer], data['train'][target], acts['held'][layer],
                                                  data['held'][target], seed=51310)
                        probes.append(dict(run_id=rid if layer else 'input', condition=c if layer else '', init_seed=s if layer else '',
                                           step=step, layer=layer, target=target, probe_train_acc=a_tr, probe_held_acc=a_he))
                        if step == 10000 and layer in (1, 2) and target in ('step1', 'step2'):
                            inter += interchange(nets, p, P, data, acts, layer, target, W, rng, rid, c, s)
                print(rid, step, flush=True)
    for name, rows_ in (('probes', probes), ('interventions', inter)):
        with (out/f'{name}.csv').open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows_[0])); w.writeheader(); w.writerows(rows_)


STEPS_ALL = ('step1', 'step2', 'final')


def interchange(nets, p, P, data, acts, layer, target, W, rng, rid, cond, seed):
    """Swap the component of a layer's activation that lies in a subspace between a base row and a
    source row whose `target` state differs, then read the final head. The counterfactual answer is
    the base program's remaining ops applied to the source's state; rows whose counterfactual equals
    the base or the source answer are excluded so the three readings are distinct."""
    Wc = W - W.mean(1, keepdims=True)
    U, sv, _ = np.linalg.svd(Wc, full_matrices=False)
    U = U[:, sv > 1e-8*sv.max()]
    k = U.shape[1]
    out = []
    for split in ('train', 'held'):
        d, H = data[split], acts[split][layer]
        n = len(d['s'])
        base, src = [], []
        for i in range(n):
            cand = np.flatnonzero(d[target] != d[target][i])
            for j in rng.choice(cand, 8, replace=False):
                base.append(i); src.append(j)
        base, src = np.array(base), np.array(src)
        state = d[target][src]
        cf = P[d['c'][base], state] if target == 'step2' else P[d['c'][base], P[d['b'][base], state]]
        keep = (cf != d['final'][base]) & (cf != d['final'][src])
        base, src, cf = base[keep], src[keep], cf[keep]
        subspaces = [('probe', U)] + [(f'random-{r}', np.linalg.qr(rng.normal(size=(H.shape[1], k)))[0]) for r in range(5)]
        subspaces.append(('whole layer', None))
        for name, B in subspaces:
            Hb, Hs = H[base], H[src]
            Hp = Hs.copy() if B is None else Hb + (Hs - Hb) @ B @ B.T
            pred = nets.forward_from(p, layer, Hp)[1]['final'].argmax(1)
            out.append(dict(run_id=rid, condition=cond, init_seed=seed, layer=layer, variable=target, subspace=name,
                            dim=H.shape[1] if B is None else k, pairs_from=split, n_pairs=len(base),
                            follows_counterfactual=float(np.mean(pred == cf)), keeps_base_answer=float(np.mean(pred == d['final'][base])),
                            gives_source_answer=float(np.mean(pred == d['final'][src])),
                            unpatched_base_acc=float(np.mean(nets.forward_from(p, layer, Hb)[1]['final'].argmax(1) == d['final'][base]))))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('publish'); p.add_argument('--root', type=Path, required=True); p.add_argument('--ids', nargs='+', required=True)
    p.add_argument('--commit-checkpoints', nargs='*', default=[], help='RUN_ID/ckpt-N.npz files to copy into git')
    sub.add_parser('figure')
    p = sub.add_parser('probe'); p.add_argument('--root', type=Path, required=True); p.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    if args.cmd == 'publish':
        publish(args.root, args.ids, set(args.commit_checkpoints))
    elif args.cmd == 'probe':
        probe(args.root, args.out)
    else:
        figure()


if __name__ == '__main__':
    main()
