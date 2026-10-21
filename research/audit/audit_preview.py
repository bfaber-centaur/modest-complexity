#!/usr/bin/env python3
"""Priority-1 audit of published e1 runs with independently written code.

Reads published run directories (config, runtime, metrics, data, checkpoints,
manifest), the plan, the frozen split and a lane registry, and writes one JSON
report. Uses research/audit/independent.py for all numerics; nothing here
imports the shared training/reporting code.

uv run --locked python research/audit/audit_preview.py \
    --published PUB/research/coordinator --code-commit ebd874e... \
    --out research/audit/results/preview-1.json [--replay]
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import independent as ind  # noqa: E402

KEYS = ['U', 'V', 'W', 'mU', 'mV', 'mW', 'vU', 'vV', 'vW']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rel(x, ref):
    return float(np.max(np.abs(x - ref)) / max(float(np.max(np.abs(ref))), 1e-300))


def load_ck(path):
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def check_split(c, frozen_path):
    """`c` holds p, seeds, split_sha256 and, for a follow-up split, train_fraction."""
    p = c['p']
    a, b, y, ysh, splits = ind.regenerate_split(p, c['data_seed'], c['control_seed'], c.get('train_fraction', .3))
    with np.load(frozen_path) as z:
        fr = {k: z[k] for k in z.files}
    regen = dict(zip(('a', 'b', 'yadd', 'yshuffled', 'train', 'cal', 'audit'), (a, b, y, ysh, *splits)))
    out = dict(frozen_keys=sorted(fr),
               regenerated_equals_frozen=all(np.array_equal(regen[k], fr[k]) for k in fr) and
                                         {'a', 'b', 'yadd', 'train', 'cal', 'audit'} <= set(fr),
               split_hash=ind.hash_arrays(fr['a'], fr['b'], fr['yadd'], fr['train'], fr['cal'], fr['audit']),
               add_target_hash=ind.hash_arrays(fr['yadd']))
    out['split_hash_matches_plan'] = out['split_hash'] == c['split_sha256']
    out['target_hashes_match_plan'] = out['add_target_hash'] == c['targets']['add']
    if 'yshuffled' in fr:  # follow-up splits freeze no shuffled table
        out['shuffled_target_hash'] = ind.hash_arrays(fr['yshuffled'])
        out['target_hashes_match_plan'] &= out['shuffled_target_hash'] == c['targets']['random']
    else:
        fr['yshuffled'] = ysh
    sets = [set(fr[k].tolist()) for k in ('train', 'cal', 'audit')]
    out['sizes'] = [len(s) for s in sets]
    out['partition'] = (sum(out['sizes']) == p*p and set().union(*sets) == set(range(p*p))
                        and all(not (sets[i] & sets[j]) for i, j in ((0, 1), (0, 2), (1, 2))))
    out['inputs_row_major'] = bool(np.array_equal(fr['a']*p + fr['b'], np.arange(p*p)))
    out['labels_are_sum_mod_p'] = bool(np.array_equal(fr['yadd'], (fr['a'] + fr['b']) % p))
    out['transpose_closed'] = all({int(fr['b'][i]*p + fr['a'][i]) for i in s} == s for s in sets)
    # Unordered groups per residue in each split (each residue has (p+1)/2 groups).
    per_class = []
    for s in sets:
        groups = {(min(int(fr['a'][i]), int(fr['b'][i])), max(int(fr['a'][i]), int(fr['b'][i]))) for i in s}
        per_class.append(sorted(set(np.bincount([(u+v) % p for u, v in groups], minlength=p).tolist())))
    out['groups_per_class_by_split'] = per_class
    out['shuffled_counts_preserved'] = all(
        np.array_equal(np.bincount(fr['yadd'][list(s)], minlength=p), np.bincount(fr['yshuffled'][list(s)], minlength=p))
        for s in sets)
    out['shuffled_fraction_changed'] = float(np.mean(fr['yshuffled'] != fr['yadd']))
    return out, fr


def label_absence(fr, p):
    """For each held-out pair, which labels co-occur with either operand (in either
    position) among training pairs. The true label never does in modular addition."""
    present = [set() for _ in range(p)]
    for i in fr['train']:
        present[fr['a'][i]].add(int(fr['yadd'][i])); present[fr['b'][i]].add(int(fr['yadd'][i]))
    held = np.concatenate([fr['cal'], fr['audit']])
    true_present = sum(int(fr['yadd'][i]) in (present[fr['a'][i]] | present[fr['b'][i]]) for i in held)
    n_absent = [p - len(present[fr['a'][i]] | present[fr['b'][i]]) for i in held]
    return dict(heldout_pairs=len(held), true_label_cooccurs_with_an_operand_in_train=int(true_present),
                absent_labels_per_heldout_pair=dict(min=min(n_absent), median=float(np.median(n_absent)),
                                                    max=max(n_absent)),
                distinct_train_labels_per_operand=sorted({len(s) for s in present}),
                present=present)


def perfect_lookup(p):
    """Exact modular addition in the trained architecture (no biases): one hidden
    unit per input pair (i, j), U[a] = +1 if a == i else -1, V[b] = 0 if b == j
    else -2, so the unit is 1 only on (i, j); W sends it to (i + j) mod p."""
    U = np.full((p, p*p), -1.); V = np.full((p, p*p), -2.); W = np.zeros((p*p, p))
    for i in range(p):
        for j in range(p):
            u = i*p + j; U[i, u] = 1.; V[j, u] = 0.; W[u, (i + j) % p] = 1.
    return U, V, W


def compare_row(mine, logged, split):
    lo = logged['objective'][split]
    return dict(acc=abs(mine['acc'] - logged[split]),
                **{k: abs(mine[k] - lo[k]) / max(1., abs(lo[k]))
                   for k in ('cross_entropy', 'margin_min', 'margin_median', 'margin_p05')})


def recompute_checkpoints(run_dir, fr, rows, p, labels):
    Xa, Xb = ind.one_hot_inputs(fr['a'], fr['b'], p)
    out = {}
    for path in sorted(run_dir.glob('checkpoint_*.npz')):
        ck = load_ck(path); step = int(ck['step'])
        logits = ind.forward((ck['U'], ck['V'], ck['W']), Xa, Xb)[0]
        res = {}
        for split in ('train', 'cal', 'audit'):
            ix = fr[split]; ev = ind.evaluate(logits[ix], labels[ix])
            res[split] = dict(mine=ev, diff=compare_row(ev, rows[step], split))
        norm = float(np.sqrt(sum(np.vdot(ck[k], ck[k]) for k in ('U', 'V', 'W'))))
        res['norm_rel_diff'] = abs(norm - rows[step]['norm']) / rows[step]['norm']
        res['file_step_matches'] = int(path.stem.split('_')[1]) == step
        res['finite'] = all(np.isfinite(ck[k]).all() for k in KEYS)
        res['dtypes'] = sorted({str(ck[k].dtype) for k in KEYS})
        res['v_nonnegative'] = all((ck[k] >= 0).all() for k in ('vU', 'vV', 'vW'))
        out[step] = res
    return out


def replay(ck, start, stop, fr, rows, p, lr, wd, labels):
    """Independent AdamW training from checkpoint `start` to `stop`; compares every
    logged row in between and returns the state at `stop`."""
    Xa, Xb = ind.one_hot_inputs(fr['a'], fr['b'], p); tr = fr['train']
    Xat, Xbt, yt = Xa[tr], Xb[tr], labels[tr]
    params = [ck['U'].copy(), ck['V'].copy(), ck['W'].copy()]
    m = [ck['mU'].copy(), ck['mV'].copy(), ck['mW'].copy()]
    v = [ck['vU'].copy(), ck['vV'].copy(), ck['vW'].copy()]
    worst = dict(acc_examples=0, cross_entropy=0., margins=0., norm=0., update_norms=0., cosine=0.)
    trace = []
    for t in range(start + 1, stop + 1):
        _, g = ind.loss_and_grad(params, Xat, Xbt, yt)
        upd = ind.adamw_step(params, g, m, v, t, lr, wd)
        if t % 100 == 0 or t == stop:
            row = rows[t]; logits = ind.forward(params, Xa, Xb)[0]
            for split in ('train', 'cal', 'audit'):
                ix = fr[split]; ev = ind.evaluate(logits[ix], labels[ix]); d = compare_row(ev, row, split)
                worst['acc_examples'] = max(worst['acc_examples'], round(d['acc']*len(ix)))
                worst['cross_entropy'] = max(worst['cross_entropy'], d['cross_entropy'])
                worst['margins'] = max(worst['margins'], d['margin_min'], d['margin_median'], d['margin_p05'])
            norm = float(np.sqrt(sum(np.vdot(x, x) for x in params)))
            worst['norm'] = max(worst['norm'], abs(norm - row['norm'])/row['norm'])
            lu = row['optimizer_update']
            worst['update_norms'] = max(worst['update_norms'],
                                        abs(upd['adaptive_norm'] - lu['adaptive_norm'])/lu['adaptive_norm'],
                                        abs(upd['decay_norm'] - lu['decay_norm'])/max(lu['decay_norm'], 1e-300))
            if (upd['cosine'] is None) != (lu['cosine'] is None):
                worst['cosine'] = float('inf')
            elif upd['cosine'] is not None:
                worst['cosine'] = max(worst['cosine'], abs(upd['cosine'] - lu['cosine']))
            trace.append(dict(step=t, train_ce_rel=compare_row(ind.evaluate(logits[tr], yt), row, 'train')['cross_entropy']))
    return dict(params=params, m=m, v=v), worst, trace


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--published', type=Path, required=True, help='Lane directory holding runs/ and registry.jsonl')
    ap.add_argument('--figures', type=Path, help='report_map.py output directory to cross-check')
    ap.add_argument('--plan', type=Path, default=Path('research/plan.json'))
    ap.add_argument('--frozen', type=Path, default=Path('research/data/e1-split.npz'))
    ap.add_argument('--code-commit', required=True, help='Commit whose *.py hashes the runs must record')
    ap.add_argument('--ids', nargs='+')
    ap.add_argument('--replay', action='store_true', help='Replay every gap between published checkpoints')
    ap.add_argument('--replay-max-steps', type=int, default=50000)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    plan = json.loads(args.plan.read_text()); c = plan['common']; p = c['p']
    planned = {r['run_id']: {**c, **r} for r in plan['runs']}
    # The audited runs must share one split; a follow-up plan entry may override it.
    split_keys = {(planned[i].get('train_fraction', .3), planned[i]['split_sha256']) for i in args.ids or planned}
    assert len(split_keys) == 1, split_keys
    (train_fraction, split_sha256), = split_keys
    report = dict(code_commit=args.code_commit, published=str(args.published), train_fraction=train_fraction)
    report['split'], fr = check_split({**c, 'train_fraction': train_fraction, 'split_sha256': split_sha256,
                                       'targets': plan['targets']}, args.frozen)
    absence = label_absence(fr, p); present = absence.pop('present')
    report['label_absence'] = absence

    # Positive control: an exact solution scores 1 on every split with these labels.
    Xa, Xb = ind.one_hot_inputs(fr['a'], fr['b'], p)
    lookup = ind.forward(perfect_lookup(p), Xa, Xb)[0]
    report['positive_control_lookup_acc'] = {s: ind.evaluate(lookup[fr[s]], fr['yadd'][fr[s]])['acc']
                                             for s in ('train', 'cal', 'audit')}

    src = {}
    for name in ('experiment.py', 'launch_runs.py', 'instrumentation.py', 'publish_run.py', 'report_map.py'):
        blob = subprocess.run(['git', 'show', f'{args.code_commit}:{name}'], capture_output=True, check=True).stdout
        src[name] = hashlib.sha256(blob).hexdigest()
    env = {}
    for name in ('pyproject.toml', 'uv.lock', '.python-version'):
        blob = subprocess.run(['git', 'show', f'{args.code_commit}:{name}'], capture_output=True, check=True).stdout
        env[name] = hashlib.sha256(blob).hexdigest()

    registry = [json.loads(s) for s in (args.published/'registry.jsonl').read_text().splitlines() if s.strip()]
    ids = args.ids or sorted({e['run_id'] for e in registry})
    report['runs'] = {}
    for run_id in ids:
        rd = args.published/'runs'/run_id; r = {}
        cfg = json.loads((rd/'config.json').read_text()); rt = json.loads((rd/'runtime.json').read_text())
        man = json.loads((rd/'manifest.json').read_text())
        rows_list = [json.loads(s) for s in (rd/'metrics.jsonl').read_text().splitlines()]
        rows = {z['step']: z for z in rows_list}
        pl = planned[run_id]
        target_key = 'yadd' if pl['task'] == 'add' else 'yshuffled'  # shuffled-label control
        labels = fr[target_key]
        # Handoff: every published file matches the manifest; flags match presence.
        local = {f.name for f in rd.iterdir() if f.is_file() and f.name != 'manifest.json'}
        r['manifest'] = dict(
            published_hashes_match=all(man['files'][n]['sha256'] == sha(rd/n) and man['files'][n]['bytes'] == (rd/n).stat().st_size
                                       for n in local),
            published_flags_match=local == {n for n, f in man['files'].items() if f['published']},
            n_listed=len(man['files']), n_published=len(local))
        ck_listed = sorted(int(n[11:17]) for n in man['files'] if n.startswith('checkpoint_'))
        expected_ck = sorted(set(range(0, c['steps'] + 1, c['checkpoint_every'])) |
                             ({rt['first_interpolation']} if rt['first_interpolation'] is not None else set()))
        r['checkpoint_schedule_complete'] = ck_listed == expected_ck
        # Provenance and configuration.
        r['config'] = dict(
            seeds_match_plan=cfg['seeds'] == dict(data_seed=c['data_seed'], init_seed=pl['init_seed'],
                                                   control_seed=c['control_seed'], diagnostic_seed=c['diagnostic_seed']),
            recipe_matches_plan=(cfg['hidden'], cfg['steps'], cfg['lr'], cfg['wd'], cfg['p'], cfg['log_every'],
                                 cfg['checkpoint_every'], cfg['diagnostics'], cfg['task']) ==
                                (pl['width'], c['steps'], c['lr'], float(pl['wd']), p, c['log_every'],
                                 c['checkpoint_every'], c['diagnostics'], pl['task']) and
                                cfg.get('train_fraction', .3) == train_fraction,
            split_hash_matches_plan=cfg['data_hashes']['split_sha256'] == pl['split_sha256'],
            target_hash_matches_plan=cfg['data_hashes']['target_sha256'] == plan['targets'][pl['task']],
            git_commit=cfg['runtime']['git_commit'],
            source_hashes_match_commit=all(cfg['runtime']['source_sha256'][n] == h for n, h in src.items()),
            environment_hashes_match_commit=cfg['runtime']['environment_sha256'] == env,
            single_thread=set(cfg['runtime']['thread_environment'].values()) == {'1'},
            python_numpy=(cfg['runtime']['python'], cfg['runtime']['numpy']))
        with np.load(rd/'data.npz') as d:
            r['data_npz_equals_frozen'] = all(np.array_equal(d[k], fr[k2]) for k, k2 in
                                              (('a', 'a'), ('b', 'b'), ('y', target_key), ('yadd', 'yadd'),
                                               ('train', 'train'), ('cal', 'cal'), ('audit', 'audit')))
        # Logging semantics.
        steps = [z['step'] for z in rows_list]
        first_fit = next((z['step'] for z in rows_list if z['train'] >= .99), None)
        r['logging'] = dict(
            steps_complete=steps == list(range(0, c['steps'] + 1, c['log_every'])),
            no_update_at_step0=rows_list[0]['optimizer_update'] is None,
            seconds_monotone=all(x['seconds'] <= y['seconds'] for x, y in zip(rows_list, rows_list[1:])),
            runtime_complete=(rt['status'], rt['completed_steps']) == ('complete', c['steps']),
            first_interpolation_consistent=rt['first_interpolation'] == first_fit, first_fit_step=first_fit,
            # Without decay the decay displacement is zero and the logged cosine is null by design.
            cosine_in_range=all((z['optimizer_update']['cosine'] is None) if pl['wd'] == 0 else
                                -1 <= z['optimizer_update']['cosine'] <= 1 for z in rows_list[1:]))
        # Values quoted in the coordinator's INDEX.md table, recomputed from metrics.jsonl.
        late = [z for z in rows_list if z['step'] >= 1000]; last = rows_list[-1]
        r['index_values'] = dict(
            fit_sustained_after_first=all(z['train'] >= .99 for z in rows_list if z['step'] >= first_fit),
            max_audit_acc_from_1000=max(z['audit'] for z in late), max_cal_acc_from_1000=max(z['cal'] for z in late),
            final_train_ce=last['objective']['train']['cross_entropy'], final_audit_ce=last['objective']['audit']['cross_entropy'],
            final_train_margin_min_median=(last['objective']['train']['margin_min'], last['objective']['train']['margin_median']),
            final_audit_margin_median=last['objective']['audit']['margin_median'],
            norm_first_last=(rows_list[0]['norm'], last['norm']), wall_seconds=rt['wall_seconds'],
            audit_acc_step0=rows_list[0]['audit'],
            first_step_audit_below_chance=next((z['step'] for z in rows_list if z['audit'] < 1/p), None))
        ev = [e for e in registry if e['run_id'] == run_id]
        r['registry'] = dict(events=[e['status'] for e in ev],
                             terminal_matches_runtime=ev[-1]['status'] == rt['status'] and ev[-1]['completed_steps'] == rt['completed_steps'],
                             command_matches_plan=ev[0]['command'][ev[0]['command'].index('--hidden') + 1] == str(pl['width'])
                             and ev[0]['command'][ev[0]['command'].index('--init-seed') + 1] == str(pl['init_seed']))
        # Initialization from the recorded seed, written independently.
        if (rd/'checkpoint_000000.npz').exists():  # lanes may publish only some checkpoints
            ck0 = load_ck(rd/'checkpoint_000000.npz')
            init = ind.initial_weights(p, pl['width'], pl['init_seed'])
            r['init'] = dict(weights_equal_independent_draw=all(np.array_equal(x, ck0[k]) for x, k in zip(init, 'UVW')),
                             moments_zero=all(not ck0[k].any() for k in KEYS[3:]))
        else:
            r['init'] = 'not published'
        # Recompute metrics at every published checkpoint.
        rc = recompute_checkpoints(rd, fr, rows, p, labels)
        r['recompute'] = dict(
            steps=sorted(rc),
            max_acc_diff=max(rc[s][sp]['diff']['acc'] for s in rc for sp in ('train', 'cal', 'audit')),
            max_rel_ce_or_margin_diff=max(v for s in rc for sp in ('train', 'cal', 'audit')
                                          for k, v in rc[s][sp]['diff'].items() if k != 'acc'),
            max_norm_rel_diff=max(rc[s]['norm_rel_diff'] for s in rc),
            ties=sum(rc[s][sp]['mine']['ties'] for s in rc for sp in ('train', 'cal', 'audit')),
            checkpoints_ok=all(rc[s]['file_step_matches'] and rc[s]['finite'] and rc[s]['v_nonnegative']
                               and rc[s]['dtypes'] == ['float64'] for s in rc))
        # The surprising result: where do held-out predictions and true labels sit?
        if pl['task'] == 'add':  # label absence is a property of the addition table
            final = load_ck(rd/f'checkpoint_{c["steps"]:06d}.npz')
            logits = ind.forward((final['U'], final['V'], final['W']), Xa, Xb)[0]
            held = np.concatenate([fr['cal'], fr['audit']])
            pred = logits[held].argmax(1); true = fr['yadd'][held]
            rank = (logits[held] > logits[held, true][:, None]).sum(1)  # 0 = top
            r['heldout_final'] = dict(
                acc=float(np.mean(pred == true)),
                pred_cooccurs_with_operand_in_train=float(np.mean([int(pr) in (present[fr['a'][i]] | present[fr['b'][i]])
                                                                   for pr, i in zip(pred, held)])),
                true_label_rank_median=float(np.median(rank)), true_label_rank_min=int(rank.min()),
                true_label_rank_pctile_median=float(np.median(rank)/(p - 1)))
        if args.replay:
            ck_steps = sorted(rc)
            starts = {}
            if 0 not in rc:  # step 0 unpublished: start from an independent draw, which also checks init
                U, V, W = ind.initial_weights(p, pl['width'], pl['init_seed'])
                starts[0] = dict(zip(KEYS, (U, V, W, *(np.zeros_like(x) for x in (U, V, W, U, V, W)))))
                ck_steps = [0] + ck_steps
            r['replay'] = []
            budget = args.replay_max_steps
            for s0, s1 in zip(ck_steps, ck_steps[1:]):
                if s1 - s0 > budget:
                    r['replay'].append(dict(start=s0, stop=s1, skipped='replay budget')); continue
                budget -= s1 - s0
                t0 = time.perf_counter()
                start = starts[s0] if s0 in starts else load_ck(rd/f'checkpoint_{s0:06d}.npz')
                state, worst, trace = replay(start, s0, s1, fr, rows, p, c['lr'], float(pl['wd']), labels)
                target = load_ck(rd/f'checkpoint_{s1:06d}.npz')
                got = state['params'] + state['m'] + state['v']
                r['replay'].append(dict(start=s0, stop=s1, from_independent_init=s0 in starts,
                                        seconds=round(time.perf_counter() - t0, 1),
                                        state_rel_diff={k: rel(x, target[k]) for x, k in zip(got, KEYS)},
                                        logged_rows=worst, train_ce_rel_trace=trace[::max(1, len(trace)//10)]))
                print(run_id, s0, s1, r['replay'][-1]['seconds'], 's', max(r['replay'][-1]['state_rel_diff'].values()), flush=True)
        report['runs'][run_id] = r

    if args.figures:
        with (args.figures/'curves.csv').open() as stream:
            table = list(csv.DictReader(stream))
        bad = 0
        for run_id in ids:
            rows_list = [json.loads(s) for s in (args.published/'runs'/run_id/'metrics.jsonl').read_text().splitlines()]
            mine = [t for t in table if t['run_id'] == run_id]
            bad += len(mine) != len(rows_list)
            for t, z in zip(mine, rows_list):
                vals = [z['step'], z['train'], z['cal'], z['audit'], z['norm']] + \
                       [z['objective'][s][k] for s in ('train', 'cal', 'audit') for k in ('cross_entropy', 'margin_min', 'margin_p05', 'margin_median')]
                keys = ['step', 'train', 'cal', 'audit', 'norm'] + \
                       [f'{s}_{k}' for s in ('train', 'cal', 'audit') for k in ('cross_entropy', 'margin_min', 'margin_p05', 'margin_median')]
                bad += any(float(t[k]) != float(x) for k, x in zip(keys, vals))
        with (args.figures/'status.csv').open() as stream:
            status = {s['run_id']: s for s in csv.DictReader(stream)}
        report['figures'] = dict(curves_rows_mismatched=int(bad), curves_run_ids=sorted({t['run_id'] for t in table}),
                                 status=[(k, v['status'], v['last_logged_step']) for k, v in sorted(status.items())],
                                 files={f.name: sha(f) for f in sorted(args.figures.iterdir())})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, default=float) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'runs'}, indent=1, default=float)[:4000])
    for run_id, r in report['runs'].items():
        print(run_id, json.dumps({k: v for k, v in r.items() if k != 'replay'}, default=float))


if __name__ == '__main__':
    main()
