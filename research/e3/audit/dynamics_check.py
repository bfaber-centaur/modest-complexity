#!/usr/bin/env python3
"""Dynamics checks (research/e3/audit/QUESTION-2.md, Y1-Y2) from the split rule alone.

Rebuilds every e3 ladder split with the e1 audit's independent rewrite of the split rule
(research/audit/independent.py, which imports nothing from experiment.py) and checks it
against the plan, the coordinator's ladders.json and the saved evaluation sets: hashes and
actual training counts, nesting, transpose closure, partition, the evaluation pairs as the
complement of the 50% training set, the label-set sizes, and the crossed design.

The `runs` mode checks the landed runs (QUESTION-5.md: Y2', Y3, Y5): a replay from the
independent init to each run's first published checkpoint (no step-0 checkpoint is published),
registry and manifest accounting, and my own forward pass on every published checkpoint. The
`replay` mode (Y4) replays one run's first 2,000 updates and its updates 30,001-50,000.

uv run --locked python research/e3/audit/dynamics_check.py splits OUT.json
uv run --locked python research/e3/audit/dynamics_check.py runs RUNS_DIR REGISTRY OUT.json
uv run --locked python research/e3/audit/dynamics_check.py replay RUNS_DIR RUN_ID OUT.json
uv run --locked python research/e3/audit/dynamics_check.py sensitivity RUNS_DIR RUN_ID OUT.json  (post hoc)
uv run --locked python research/e3/audit/dynamics_check.py track RUNS_DIR RUN_ID OTHER_RUN_DIR OUT.json  (post hoc)
uv run --locked python research/e3/audit/dynamics_check.py onestep RUNS_DIR RUN_ID OUT.json [SHARED_CHECKPOINT_30001]  (post hoc)
"""
import hashlib
import itertools
import json
from pathlib import Path
import sys
import numpy as np

root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(root/'research/audit'))
import independent as ind  # noqa: E402
sys.path.insert(0, str(root))
from experiment import init as exp_init  # noqa: E402  (compared with the independent rule only)
from pairwise import ladder_sets  # noqa: E402

P, CONTROL = 31, 32000
SHARES = (.3, .4, .45, .5)
SEEDS = (31000, 41001, 41002, 41003)
INITS = (30000, 30001)
plan = json.loads((root/'research/e3/plan.json').read_text())
ladders = {r['data_seed']: r for r in json.loads((root/'research/e3/coordinator/ladders.json').read_text())}
REFERENCE = {.3: ['research/coordinator/runs/e1-co-add-w128-i30000-a1', 'research/size-time/runs/e1-st-add-w128-i30001-a1'],
             .4: [f'research/e2/dynamics/runs/e2-dy-add-w128-i{i}-tf40-a1' for i in INITS],
             .45: [f'research/e2/dynamics/runs/e2-dy-add-w128-i{i}-tf45-a1' for i in INITS],
             .5: [f'research/coordinator/runs/e1f2-co-add-w128-i{i}-tf50-a1' for i in INITS]}


def transpose_closed(a, b, ix):
    return set((a[ix]*P + b[ix]).tolist()) == set((b[ix]*P + a[ix]).tolist())


def splits(out_json):
    entries = [e for e in plan['runs'] if e['run_id'].startswith('e3-dy-')]
    by_key = {(e['data_seed'], e['init_seed'], e['train_fraction']): e for e in entries}
    res = dict(design=dict(
        entries=len(entries), distinct=len(by_key),
        crossed=sorted(by_key) == sorted(itertools.product(SEEDS[1:], INITS, SHARES)),
        ids_match_fields=all(e['run_id'] == f"e3-dy-add-w128-i{e['init_seed']}-tf{round(100*e['train_fraction'])}-d{e['data_seed']}-a1"
                             for e in entries),
        recipe=sorted({json.dumps({k: {**plan['common'], **e}[k] for k in ('task', 'wd', 'width', 'steps', 'lr', 'log_every',
                                                                         'checkpoint_every', 'diagnostics', 'control_seed')})
                       for e in entries}),
        queue_positions_are_1_to_24=sorted(e['queue_position'] for e in entries) == list(range(1, 25)),
        init_weights_independent_of_data_seed='init(p, hidden, seed) takes no data seed (experiment.py); '
                                              'checked on landed step-0 checkpoints in Y2'), ladders={})
    for seed in SEEDS:
        lad, prev = dict(shares={}), None
        for share in SHARES:
            a, b, y, _, sp = ind.regenerate_split(P, seed, CONTROL, share)
            h = ind.hash_arrays(a, b, y, *sp)
            claims = [by_key[(seed, i, share)] for i in INITS] if seed != 31000 else []
            tr = set(sp[0].tolist())
            lad['shares'][str(share)] = dict(
                train=len(sp[0]), split_sha256=h, matches_ladders_json=h == ladders[seed]['shares'][str(share)]['split_sha256'],
                matches_plan=all(e['split_sha256'] == h and e['train_pairs'] == len(sp[0]) for e in claims) if claims else None,
                partition=sorted(np.concatenate(sp).tolist()) == list(range(P*P)),
                transpose_closed=all(transpose_closed(a, b, ix) for ix in sp),
                nested_in_this=prev is None or prev <= tr)
            prev = tr
        ev = np.array(sorted(set(range(P*P)) - prev))
        saved = np.load(root/f'research/e3/data/eval-d{seed}.npz')['index']
        S, D, N, yy = ladder_sets(a, b, ind.regenerate_split(P, seed, CONTROL, .3)[4][0], np.array(sorted(prev)), ev, P)
        lad['evaluation'] = dict(
            pairs=len(ev), equals_saved=np.array_equal(ev, saved), sha256=ind.hash_arrays(ev),
            matches_ladders_json=ind.hash_arrays(ev) == ladders[seed]['evaluation']['sha256'],
            transpose_closed=transpose_closed(a, b, ev),
            unordered_groups=len({(min(i, j), max(i, j)) for i, j in zip(a[ev], b[ev])}),
            per_class=[int(np.bincount(yy, minlength=P).min()), int(np.bincount(yy, minlength=P).max())],
            S=[int(S.sum(1).min()), round(float(S.sum(1).mean()), 2), int(S.sum(1).max())],
            D=[int(D.sum(1).min()), int(D.sum(1).max()), int((D.sum(1) == 0).sum())],
            N=[int(N.sum(1).min()), int(N.sum(1).max()), int((N.sum(1) == 0).sum()), int((N.sum(1) == 1).sum())])
        lad['evaluation']['N_matches_ladders_json'] = (
            lad['evaluation']['N'][0] == ladders[seed]['evaluation']['N_size']['min']
            and lad['evaluation']['N'][1] == ladders[seed]['evaluation']['N_size']['max']
            and lad['evaluation']['N'][3] == ladders[seed]['evaluation']['N_size']['singleton'])
        res['ladders'][str(seed)] = lad
    ref = {}
    for share, runs in REFERENCE.items():
        h = res['ladders']['31000']['shares'][str(share)]['split_sha256']
        for r in runs:
            c = json.loads((root/r/'config.json').read_text())
            ref[Path(r).name] = c['data_hashes']['split_sha256'] == h and c['seeds']['data_seed'] == 31000 \
                and c['seeds']['init_seed'] == int(Path(r).name.split('-i')[1][:5]) and c['hidden'] == 128 and c['wd'] == 1
    res['reference_ladder_31000_runs_match'] = ref
    res['init_rule_matches_independent'] = {i: all(np.array_equal(p, q) for p, q in zip(exp_init(P, 128, i), ind.initial_weights(P, 128, i)))
                                            for i in INITS}
    Path(out_json).write_text(json.dumps(res, indent=1) + '\n')
    lad_ok = all(all(s['matches_ladders_json'] and s['partition'] and s['transpose_closed'] and s['nested_in_this']
                     and s['matches_plan'] in (True, None) for s in l['shares'].values())
                 and l['evaluation']['equals_saved'] and l['evaluation']['matches_ladders_json'] and l['evaluation']['transpose_closed']
                 and l['evaluation']['N_matches_ladders_json'] for l in res['ladders'].values())
    d = res['design']
    print(dict(Y1=lad_ok, design=d['entries'] == d['distinct'] == 24 and d['crossed'] and d['ids_match_fields']
               and len(d['recipe']) == 1 and d['queue_positions_are_1_to_24'],
               reference=all(ref.values()), init=all(res['init_rule_matches_independent'].values())))


def runs(runs_dir, registry, out_json):
    """Y2' (init pairing by replay to the first published checkpoint; no step-0 checkpoint is
    published), Y3 accounting and Y5 checkpoint predictions for every planned dynamics run."""
    runs_dir = Path(runs_dir)
    entries = {e['run_id']: e for e in plan['runs'] if e['run_id'].startswith('e3-dy-')}
    events = [json.loads(l) for l in Path(registry).read_text().splitlines() if l.strip()]
    sha = lambda f: hashlib.sha256(Path(f).read_bytes()).hexdigest()
    init = {i: ind.initial_weights(P, 128, i) for i in INITS}
    res, splits_cache = dict(runs={}), {}
    for rid, e in sorted(entries.items(), key=lambda kv: kv[1]['queue_position']):
        ev = [x['status'] for x in events if x['run_id'] == rid]
        d = runs_dir/rid
        r = res['runs'][rid] = dict(events=ev, published=d.exists())
        if not d.exists():
            continue
        cfg = json.loads((d/'config.json').read_text())
        man = json.loads((d/'manifest.json').read_text())
        rt = json.loads((d/'runtime.json').read_text())
        key = (e['data_seed'], e['train_fraction'])
        if key not in splits_cache:
            splits_cache[key] = ind.regenerate_split(P, key[0], CONTROL, key[1])
        a, b, y, _, sp = splits_cache[key]
        XA, XB = ind.one_hot_inputs(a, b, P)
        logged = {x['step']: x for x in map(json.loads, (d/'metrics.jsonl').read_text().splitlines())}
        cks = {int(f.stem.split('_')[1]): f for f in sorted(d.glob('checkpoint_*.npz'))}
        exact, rel, ab = True, 0., 0.
        for step, f in cks.items():
            with np.load(f) as z:
                w = [z[k] for k in 'UVW']
            zl = ind.forward(w, XA, XB)[0]
            for name, ix in zip(('train', 'cal', 'audit'), sp):
                exact &= float(np.mean(zl[ix].argmax(1) == y[ix])) == logged[step][name]
                ce = float(np.mean(np.logaddexp.reduce(zl[ix], axis=1) - zl[ix, y[ix]]))
                lg = logged[step]['objective'][name]['cross_entropy']
                rel, ab = max(rel, abs(ce - lg)/abs(lg)), max(ab, abs(ce - lg))
        first = min(cks)
        w = [x.copy() for x in init[e['init_seed']]]
        m, v = [np.zeros_like(x) for x in w], [np.zeros_like(x) for x in w]
        XT = ind.one_hot_inputs(a[sp[0]], b[sp[0]], P)
        for t in range(1, first + 1):
            ind.adamw_step(w, ind.loss_and_grad(w, *XT, y[sp[0]])[1], m, v, t, .001, 1.)
        with np.load(cks[first]) as z:
            diff, scale = max(float(np.abs(x - z[k]).max()) for x, k in zip(w, 'UVW')), max(float(np.abs(z[k]).max()) for k in 'UVW')
        r.update(completed=rt['completed_steps'], status=rt['status'],
                 config_matches_plan=(cfg['data_hashes']['split_sha256'] == e['split_sha256'] and cfg['seeds']['data_seed'] == e['data_seed']
                                      and cfg['seeds']['init_seed'] == e['init_seed'] and cfg['train_fraction'] == e['train_fraction']
                                      and cfg['steps'] == 50000 and cfg['wd'] == 1 and cfg['hidden'] == 128 and cfg['lr'] == .001),
                 train_pairs=len(sp[0]) == e['train_pairs'],
                 manifest_ok=all(sha(d/n) == h['sha256'] for n, h in man['files'].items() if h.get('published') and (d/n).exists()),
                 manifest_missing=[n for n, h in man['files'].items() if h.get('published') and not (d/n).exists()],
                 checkpoints=sorted(cks), accuracies_exact=bool(exact), worst_ce_rel_diff=rel, worst_ce_abs_diff=ab,
                 replay_to_first_checkpoint=dict(step=first, max_abs_diff=diff, tolerance=1e-9*scale, PASS=diff <= 1e-9*scale))
    landed = [r for r in res['runs'].values() if r['published']]
    res['Y2_prime'] = all(r['replay_to_first_checkpoint']['PASS'] for r in landed) and len(landed) == 24
    res['Y3'] = dict(launched=sum(bool(r['events']) for r in res['runs'].values()), landed=len(landed),
                     complete=sum(r.get('status') == 'complete' and r.get('completed') == 50000 for r in landed),
                     event_patterns=sorted({'/'.join(r['events']) for r in res['runs'].values()}),
                     unplanned_ids_in_registry=sorted({x['run_id'] for x in events} - set(entries)),
                     all_landed_consistent=all(r['config_matches_plan'] and r['train_pairs'] and r['manifest_ok'] and not r['manifest_missing']
                                               for r in landed))
    res['Y3']['PASS'] = (res['Y3']['launched'] == res['Y3']['landed'] == res['Y3']['complete'] == 24 and res['Y3']['all_landed_consistent']
                         and res['Y3']['event_patterns'] == ['running/complete'] and not res['Y3']['unplanned_ids_in_registry'])
    res['Y5'] = dict(accuracies_exact=all(r['accuracies_exact'] for r in landed),
                     worst_ce_rel_diff=max(r['worst_ce_rel_diff'] for r in landed), worst_ce_abs_diff=max(r['worst_ce_abs_diff'] for r in landed))
    res['Y5']['PASS_as_prestated'] = res['Y5']['accuracies_exact'] and res['Y5']['worst_ce_rel_diff'] <= 1e-12
    Path(out_json).write_text(json.dumps(res, indent=1) + '\n')
    print({k: res[k] for k in ('Y2_prime', 'Y3', 'Y5')})


def replay(runs_dir, run_id, out_json):
    """Y4: from the independent init to update 2,000, then from the 30k checkpoint with its
    moments (Adam step counter continuing) to 50k; the 50k measures of the replay against the
    published checkpoint."""
    import ladder_check as lc
    e = next(x for x in plan['runs'] if x['run_id'] == run_id)
    d = Path(runs_dir)/run_id
    a, b, y, _, sp = ind.regenerate_split(P, e['data_seed'], CONTROL, e['train_fraction'])
    XT = ind.one_hot_inputs(a[sp[0]], b[sp[0]], P)
    cks = {int(f.stem.split('_')[1]): f for f in d.glob('checkpoint_*.npz')}
    load = lambda f, keys='UVW': [np.load(f)[k] for k in keys]
    w = [x.copy() for x in ind.initial_weights(P, 128, e['init_seed'])]
    m, v = [np.zeros_like(x) for x in w], [np.zeros_like(x) for x in w]
    early = {}
    for t in range(1, 2001):
        ind.adamw_step(w, ind.loss_and_grad(w, *XT, y[sp[0]])[1], m, v, t, .001, 1.)
        if t in cks:
            q = load(cks[t])
            early[t] = dict(max_abs_diff=max(float(np.abs(x - z).max()) for x, z in zip(w, q)),
                            tolerance=1e-9*max(float(np.abs(z).max()) for z in q))
    w, m, v = load(cks[30000]), load(cks[30000], ('mU', 'mV', 'mW')), load(cks[30000], ('vU', 'vV', 'vW'))
    for t in range(30001, 50001):
        ind.adamw_step(w, ind.loss_and_grad(w, *XT, y[sp[0]])[1], m, v, t, .001, 1.)
    pub = load(cks[50000])
    lad = lc.Ladder(e['data_seed'])
    res = dict(run=run_id, early=early, early_PASS=all(x['max_abs_diff'] <= x['tolerance'] for x in early.values()),
               late=dict(start=30000, end=50000, max_abs_diff=max(float(np.abs(x - z).max()) for x, z in zip(w, pub)),
                         max_abs_weight=max(float(np.abs(z).max()) for z in pub),
                         relative_weight_distance=float(np.sqrt(sum(np.sum((x - z)**2) for x, z in zip(w, pub))/sum(np.sum(z**2) for z in pub))),
                         replay=lad.score(w, e['train_fraction']), published=lad.score(pub, e['train_fraction'])))
    Path(out_json).write_text(json.dumps(res, indent=1) + '\n')
    print(json.dumps(res, indent=1))


def sensitivity(runs_dir, run_id, out_json):
    """Post hoc (not in QUESTION-5.md): how far my own replays of updates 30,001-50,000 drift
    apart under rounding-level perturbations: each entry times 1 + 2**-52 * random sign, once in
    the starting weights or in every gradient; or the training pairs in another row order, which
    changes the rounding of every sum over pairs. Compared with the replay-against-published
    distance in Y4."""
    e = next(x for x in plan['runs'] if x['run_id'] == run_id)
    d = Path(runs_dir)/run_id
    a, b, y, _, sp = ind.regenerate_split(P, e['data_seed'], CONTROL, e['train_fraction'])
    XT = ind.one_hot_inputs(a[sp[0]], b[sp[0]], P)
    ck = np.load(d/'checkpoint_030000.npz')
    rng = np.random.default_rng(0)
    finals = {}
    perm = rng.permutation(len(sp[0]))
    XTp, yp = (XT[0][perm], XT[1][perm]), y[sp[0]][perm]
    for name in ('plain', 'start perturbed', 'every gradient perturbed', 'pairs permuted'):
        eps = 0. if name in ('plain', 'pairs permuted') else 2.**-52
        w = [ck[k]*(1 + (eps if name == 'start perturbed' else 0.)*rng.choice([-1., 1.], ck[k].shape)) for k in 'UVW']
        m, v = [ck[k].copy() for k in ('mU', 'mV', 'mW')], [ck[k].copy() for k in ('vU', 'vV', 'vW')]
        for t in range(30001, 50001):
            # 'pairs permuted': the same full batch in another row order, so every sum over pairs is rounded differently
            g = ind.loss_and_grad(w, *XTp, yp)[1] if name == 'pairs permuted' else ind.loss_and_grad(w, *XT, y[sp[0]])[1]
            if name == 'every gradient perturbed':  # a stand-in for a different summation order at every update
                g = [x*(1 + eps*rng.choice([-1., 1.], x.shape)) for x in g]
            ind.adamw_step(w, g, m, v, t, .001, 1.)
        finals[name] = w
    pub = [np.load(d/'checkpoint_050000.npz')[k] for k in 'UVW']
    rel = lambda p_, q: float(np.sqrt(sum(np.sum((x - z)**2) for x, z in zip(p_, q))/sum(np.sum(z**2) for z in q)))
    mx = lambda p_, q: max(float(np.abs(x - z).max()) for x, z in zip(p_, q))
    res = dict(run=run_id, **{f'mine_vs_mine_{n.replace(" ", "_")}': dict(relative_weight_distance=rel(finals[n], finals['plain']),
                                                                         max_abs_diff=mx(finals[n], finals['plain']))
                              for n in ('start perturbed', 'every gradient perturbed', 'pairs permuted')},
               vs_published={n: dict(relative_weight_distance=rel(w, pub), max_abs_diff=mx(w, pub)) for n, w in finals.items()})
    Path(out_json).write_text(json.dumps(res, indent=1) + '\n')
    print(json.dumps(res, indent=1))


def track(runs_dir, run_id, other_dir, out_json):
    """Post hoc: my replay of updates 30,001-50,000 against every checkpoint of another
    trajectory from the same 30k checkpoint (the shared code resumed in scratch), update by
    update at that trajectory's checkpoints."""
    e = next(x for x in plan['runs'] if x['run_id'] == run_id)
    d = Path(runs_dir)/run_id
    a, b, y, _, sp = ind.regenerate_split(P, e['data_seed'], CONTROL, e['train_fraction'])
    XT = ind.one_hot_inputs(a[sp[0]], b[sp[0]], P)
    other = {int(f.stem.split('_')[1]): f for f in Path(other_dir).glob('checkpoint_*.npz')}
    ck = np.load(d/'checkpoint_030000.npz')
    w, m, v = [ck[k].copy() for k in 'UVW'], [ck[k].copy() for k in ('mU', 'mV', 'mW')], [ck[k].copy() for k in ('vU', 'vV', 'vW')]
    rel = lambda p_, q: float(np.sqrt(sum(np.sum((x - z)**2) for x, z in zip(p_, q))/sum(np.sum(z**2) for z in q)))
    rows = []
    for t in range(30001, max(other) + 1):
        ind.adamw_step(w, ind.loss_and_grad(w, *XT, y[sp[0]])[1], m, v, t, .001, 1.)
        if t in other:
            with np.load(other[t]) as z:
                q = [z[k] for k in 'UVW']
            rows.append(dict(step=t, relative_weight_distance=rel(w, q), max_abs_diff=max(float(np.abs(x - z).max()) for x, z in zip(w, q))))
    res = dict(run=run_id, mine_vs_other=rows)
    if 50000 in other:
        with np.load(d/'checkpoint_050000.npz') as z:
            pub = [z[k] for k in 'UVW']
        with np.load(other[50000]) as z:
            oth = [z[k] for k in 'UVW']
        res.update(other_vs_published_50k=dict(relative_weight_distance=rel(oth, pub), bitwise=all(np.array_equal(x, q) for x, q in zip(oth, pub))),
                   mine_vs_published_50k=rel(w, pub))
    Path(out_json).write_text(json.dumps(res, indent=1) + '\n')
    print(json.dumps(res, indent=1))


def onestep(runs_dir, run_id, out_json, shared_step=None):
    """Post hoc: size of a pure summation-order difference over 1 and 10 updates from the 30k
    checkpoint (my replay against my replay with the training pairs in another row order),
    for comparison with my replay against the shared code over the same updates (track); with
    the shared code's checkpoint at 30,001, one update with the logit gradient rounded its way."""
    e = next(x for x in plan['runs'] if x['run_id'] == run_id)
    a, b, y, _, sp = ind.regenerate_split(P, e['data_seed'], CONTROL, e['train_fraction'])
    XT, yt = ind.one_hot_inputs(a[sp[0]], b[sp[0]], P), y[sp[0]]
    perm = np.random.default_rng(0).permutation(len(sp[0]))
    ck = np.load(Path(runs_dir)/run_id/'checkpoint_030000.npz')
    states = []
    for xt, yy in ((XT, yt), ((XT[0][perm], XT[1][perm]), yt[perm])):
        w, m, v = [ck[k].copy() for k in 'UVW'], [ck[k].copy() for k in ('mU', 'mV', 'mW')], [ck[k].copy() for k in ('vU', 'vV', 'vW')]
        out = {}
        for t in range(30001, 30011):
            ind.adamw_step(w, ind.loss_and_grad(w, *xt, yy)[1], m, v, t, .001, 1.)
            if t in (30001, 30010):
                out[t] = [x.copy() for x in w]
        states.append(out)
    rel = lambda p_, q: float(np.sqrt(sum(np.sum((x - z)**2) for x, z in zip(p_, q))/sum(np.sum(z**2) for z in q)))
    res = dict(run=run_id, mine_vs_mine_pairs_permuted={t: dict(relative_weight_distance=rel(states[1][t], states[0][t]),
                                                                max_abs_diff=max(float(np.abs(x - z).max()) for x, z in zip(states[1][t], states[0][t])))
                                                        for t in (30001, 30010)})
    if shared_step:
        # One update with the logit gradient rounded the shared code's way (max-shifted softmax, then
        # subtract 1 at the label) instead of exp(logits - logsumexp) - onehot; everything else mine.
        w, m, v = [ck[k].copy() for k in 'UVW'], [ck[k].copy() for k in ('mU', 'mV', 'mW')], [ck[k].copy() for k in ('vU', 'vV', 'vW')]
        z, act, pre = ind.forward(w, *XT)
        ex = np.exp(z - z.max(1, keepdims=True)); pr = ex/ex.sum(1, keepdims=True)
        pr[np.arange(len(yt)), yt] -= 1; pr /= len(yt)
        dp = (pr @ w[2].T)*(pre > 0)
        ind.adamw_step(w, [XT[0].T @ dp, XT[1].T @ dp, act.T @ pr], m, v, 30001, .001, 1.)
        with np.load(Path(shared_step)) as q:
            sh = [q[k] for k in 'UVW']
        res['shared_rounding_of_logit_gradient_vs_shared_code_30001'] = dict(
            relative_weight_distance=rel(w, sh), max_abs_diff=max(float(np.abs(x - q).max()) for x, q in zip(w, sh)))
        pc = np.exp(z - np.logaddexp.reduce(z, axis=1, keepdims=True))[np.arange(len(yt)), yt]
        res['one_minus_p_label_at_30k'] = dict(median=float(np.median(1 - pc)), min=float(np.min(1 - pc)))
    Path(out_json).write_text(json.dumps(res, indent=1) + '\n')
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    {'splits': splits, 'runs': runs, 'replay': replay, 'sensitivity': sensitivity, 'track': track, 'onestep': onestep}[sys.argv[1]](*sys.argv[2:])
