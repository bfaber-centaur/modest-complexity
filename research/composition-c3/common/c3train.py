#!/usr/bin/env python3
"""Train or continue one c3 RNN run: c2train.py with a step-dependent loss-weight schedule and a
recipe check on resume.

Schedule. The update taken at step t (from the state after t updates to the one after t+1)
uses step weights w(t). Before --switch-step S, w(t) = --step-weights (w_b). From S on,
    w(t) = w_b + f(t) * (w_a - w_b),   f(t) = min(1, (t - S + 1) / D),
with w_a = --switch-weights and D = --taper-updates. D = 1 (the default) switches at once: the
update at S already uses w_a. With D > 1, the update at S moves 1/D of the way and the update
at S + D - 1 is the first to use w_a in full. A weight that is equal in w_b and w_a stays
exactly that value. Without --switch-step the run is c2train.py's, bitwise (test_c3.py).

Optimizer reset. --reset-optimizer-at R replaces the Adam state (both moments and the step
count used for bias correction) with a fresh one just before the update taken at step R; the
weights are carried. The checkpoint saved at step R holds the state before the reset. Checkpoint
meta `step` is the global update count; the saved Adam count is step - R after a reset.

Resume. --resume CKPT continues from a checkpoint with its weights and full Adam state, to
--updates total updates, as c2train.py does. It needs --resume-sha256 and --resume-config, the
config.json of the run that wrote CKPT (r1's research/regimes/composition/train.py, c2train.py or
this script). Checked before any update:
- the checkpoint's SHA-256 equals --resume-sha256, and equals the entry for that file in a
  MANIFEST.json beside the config, if there is one;
- the checkpoint's step and run ID equal the source config's;
- this run's recipe equals the source's: task ID, tying scheme, width, lr, wd, Adam constants,
  the step weights of every update before the checkpoint's step, and any optimizer reset
  before that step; and the checkpoint's Adam count is the one that history implies.
A branch that changes the weights from the checkpoint's step on therefore passes, and any other
recipe change fails. A c2 source's own resume chain was checked by its lane (c2 D9).

metrics.jsonl: c2train.py's rows, plus `step_weights`, the weights w(t) of the update taken at
that step (on the last row, the weights that would apply next).

uv run --locked python research/composition-c3/common/c3train.py --out DIR --init-seed 51200
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
C2 = HERE.parents[1]/'composition-c2'/'common'
sys.path[:0] = [str(HERE), str(C2), str(HERE.parents[1]/'regimes'/'common')]
import c2rnn  # noqa: E402
import c2task  # noqa: E402
import c2train  # noqa: E402
import nets  # noqa: E402  (also puts the repository root on sys.path)
from experiment import array_sha256  # noqa: E402
from instrumentation import Timings  # noqa: E402

R1_WEIGHTS = {'inter': [1., 1., 1.], 'final': [0., 0., 1.]}


def weights_at(t, sched):
    """Step weights of the update taken at step t under sched = dict(before, switch_step, after, taper[, reset_at])."""
    S = sched['switch_step']
    if S is None or t < S:
        return list(sched['before'])
    f = min(1., (t - S + 1) / sched['taper'])
    if f == 1.:
        return list(sched['after'])
    return [b + f*(a - b) for b, a in zip(sched['before'], sched['after'])]


def reset_before(sched, step):
    """The optimizer reset step if one happened before step (strictly), else None."""
    R = sched.get('reset_at')
    return R if R is not None and R < step else None


def adam_count(sched, step):
    """Adam's step count in a checkpoint saved at global step `step` under sched."""
    R = reset_before(sched, step)
    return step if R is None else step - R


def recipe_of(cfg):
    """(recipe, schedule) of a run, read from its config.json in r1's, c2's or c3's format."""
    if 'task_id' in cfg:  # c2train.py or c3train.py
        recipe = {k: cfg[k] for k in ('task_id', 'transition', 'readout', 'hidden', 'lr', 'wd')}
        recipe['adam'] = [cfg['recipe'][k] for k in ('b1', 'b2', 'eps')]
        sched = cfg.get('schedule') or dict(before=cfg['step_weights'], switch_step=None, after=None, taper=1)
    else:  # r1 composition/train.py; the RNN uses the first --hidden width and nets.adam_step's defaults
        assert cfg['model'] == 'rnn' and cfg['seeds_by_role'].get('label_shuffle') is None, 'not an r1 RNN run'
        mode = cfg.get('reviewer', {}).get('mode')
        assert mode in (None, 'untied'), f'unknown r1 reviewer mode {mode}'
        recipe = dict(task_id=c2task.task_id(cfg['task_seed_used'], cfg['split_seed']), transition='tied',
                      readout='untied' if mode == 'untied' else 'tied', hidden=cfg['hidden'][0],
                      lr=cfg['lr'], wd=cfg['wd'], adam=[c2train.B1, c2train.B2, c2train.EPS])
        sched = dict(before=R1_WEIGHTS[cfg['condition']], switch_step=None, after=None, taper=1)
    return recipe, sched


def check_source(ckpt, sha, cfg_path, meta, step, adam_t, recipe, sched):
    """Raise ValueError unless the checkpoint belongs to the config's run and this run's recipe
    equals the source's up to `step`. Returns what was checked, for config.json."""
    cfg = json.loads(Path(cfg_path).read_text())
    src_recipe, src_sched = recipe_of(cfg)
    problems = []
    manifest = Path(cfg_path).parent/'MANIFEST.json'
    listed = [e['sha256'] for e in json.loads(manifest.read_text()) if e['file'] == Path(ckpt).name] if manifest.exists() else []
    if listed and listed != [sha]:
        problems.append(f'{manifest} lists {listed} for {Path(ckpt).name}')
    if adam_t != adam_count(src_sched, step):
        problems.append(f'checkpoint Adam count {adam_t} != {adam_count(src_sched, step)} implied by the source at step {step}')
    if meta.get('run_id') and cfg.get('run_id') and meta['run_id'] != cfg['run_id']:
        problems.append(f"checkpoint run ID {meta['run_id']} != config run ID {cfg['run_id']}")
    for k in recipe:
        if recipe[k] != src_recipe[k]:
            problems.append(f'{k}: {recipe[k]} here, {src_recipe[k]} in the source')
    differ = [t for t in range(step) if weights_at(t, sched) != weights_at(t, src_sched)]
    if differ:
        problems.append(f'step weights differ from the source at {len(differ)} updates before {step}, first at {differ[0]}')
    if reset_before(sched, step) != reset_before(src_sched, step):
        problems.append(f'optimizer reset before {step}: {reset_before(sched, step)} here, {reset_before(src_sched, step)} in the source')
    if problems:
        raise ValueError('--resume recipe check failed: ' + '; '.join(problems))
    return dict(config=str(cfg_path), config_sha256=nets.sha256_file(cfg_path), run_id=cfg.get('run_id'),
                manifest_listed=bool(listed), recipe=src_recipe, schedule=src_sched)


def exposure(n_rows, sched, start, end):
    """Labels per head over updates start..end-1: updates with nonzero weight, and weighted row presentations."""
    W = np.array([weights_at(t, sched) for t in range(start, end)]).reshape(-1, 3)
    return dict(updates_with_weight=[int(x) for x in (W != 0).sum(0)],
                weighted_row_presentations=[float(x) for x in n_rows * W.sum(0)])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--task-id', default=c2task.ORIGINAL)
    ap.add_argument('--transition', choices=('tied', 'untied'), default='tied')
    ap.add_argument('--readout', choices=('tied', 'untied'), default='tied')
    ap.add_argument('--step-weights', type=float, nargs=3, default=[1., 1., 1.], help='w_b, before --switch-step')
    ap.add_argument('--switch-step', type=int, help='S: first update that moves toward --switch-weights')
    ap.add_argument('--switch-weights', type=float, nargs=3, help='w_a')
    ap.add_argument('--taper-updates', type=int, default=1, help='D: updates from S until w_a is reached (1 = at once)')
    ap.add_argument('--reset-optimizer-at', type=int, help='R: fresh Adam state (moments and count) before the update at R')
    ap.add_argument('--init-seed', type=int, required=True)
    ap.add_argument('--hidden', type=int, default=128)
    ap.add_argument('--updates', type=int, default=10000, help='Total updates, counted from init (also on --resume)')
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--wd', type=float, default=0.1)
    ap.add_argument('--eval-every', type=int, default=50)
    ap.add_argument('--diag', action='store_true', help='Log optimization diagnostics at every evaluation')
    ap.add_argument('--checkpoint-steps', type=int, nargs='*', default=[0], help='The final update is always saved')
    ap.add_argument('--resume', type=Path); ap.add_argument('--resume-sha256'); ap.add_argument('--resume-config', type=Path)
    ap.add_argument('--run-id'); ap.add_argument('--purpose', default='research')
    args = ap.parse_args()
    if (args.switch_step is None) != (args.switch_weights is None):
        ap.error('--switch-step and --switch-weights go together')
    if args.taper_updates < 1 or (args.switch_step is not None and args.switch_step < 0):
        ap.error('--taper-updates must be >= 1 and --switch-step >= 0')
    if args.resume and not (args.resume_sha256 and args.resume_config):
        ap.error('--resume needs --resume-sha256 and --resume-config')
    sched = dict(before=args.step_weights, switch_step=args.switch_step, after=args.switch_weights, taper=args.taper_updates,
                 reset_at=args.reset_optimizer_at)
    recipe = dict(task_id=args.task_id, transition=args.transition, readout=args.readout, hidden=args.hidden,
                  lr=args.lr, wd=args.wd, adam=[c2train.B1, c2train.B2, c2train.EPS])
    args.out.mkdir(parents=True, exist_ok=False)
    timing = Timings()

    T = c2task.load(args.task_id)
    data = dict(train=c2task.rows(T['P'], T['train']), held=c2task.rows(T['P'], T['held']))
    nbr, nbr_all_train = c2task.neighbour_answerable(T['P'], T['train'], T['held'])
    fit = data['train']
    targets = [fit['states'][:, t] for t in range(3)]
    tie = dict(transition=args.transition == 'tied', readout=args.readout == 'tied')
    if args.resume:
        sha = nets.sha256_file(args.resume)
        if sha != args.resume_sha256:
            ap.error(f'--resume {args.resume}: SHA-256 {sha} does not equal --resume-sha256 {args.resume_sha256}')
        p, st, src_meta = nets.load(args.resume)
        assert c2rnn.scheme(p) == tie, (c2rnn.scheme(p), tie)
        assert p['bs'].shape == (args.hidden,), 'width differs from the checkpoint'
        start = src_meta.get('step', st['t'])
        try:
            source = check_source(args.resume, sha, args.resume_config, src_meta, start, st['t'], recipe, sched)
        except ValueError as e:
            ap.error(str(e))
        resumed = dict(path=str(args.resume), sha256=sha, step=start, meta=src_meta, source=source)
        assert start < args.updates
    else:
        p = c2rnn.init(args.hidden, args.init_seed, tie['transition'], tie['readout'])
        st, start, resumed = nets.adam_state(p), 0, None

    config = dict(
        vars(args), out=str(args.out), resume=str(args.resume) if args.resume else None,
        resume_config=str(args.resume_config) if args.resume_config else None,
        model='rnn', scheme=c2rnn.scheme(p), resumed_from=resumed, schedule=sched,
        task=dict(id=T['id'], task_seed_used=T['task_seed_used'], ops=T['P'].tolist(), latin_square=T['L'].tolist(),
                  held_out_programs=T['held'], n_notN=int((~nbr).sum()), all_neighbours_are_training_rows=bool(nbr_all_train)),
        seeds_by_role=dict(task=c2task.parse_id(T['id'])[0], split=c2task.parse_id(T['id'])[1], init=args.init_seed),
        recipe=dict(optimizer='Adam, decoupled decay on every parameter', b1=c2train.B1, b2=c2train.B2, eps=c2train.EPS,
                    full_batch=True, loss='sum over steps of w(t) * mean cross-entropy (schedule above)'),
        exposure=dict(fit_rows=len(fit['s']), updates=args.updates, updates_this_run=args.updates - start,
                      row_presentations=len(fit['s'])*args.updates,
                      labels_this_run=exposure(len(fit['s']), sched, start, args.updates),
                      labels_from_init=exposure(len(fit['s']), sched, 0, args.updates),
                      parameters={k: int(v.size) for k, v in p.items()}, total_parameters=int(sum(v.size for v in p.values()))),
        sha256=dict(ops=array_sha256(T['P']), **{f'{k}_rows': array_sha256(d['s'], d['ops'], d['states']) for k, d in data.items()}),
        provenance=nets.provenance(args.out, C2/'c2task.py', C2/'c2rnn.py', C2/'c2train.py', __file__))
    (args.out/'config.json').write_text(json.dumps(config, indent=1, default=int))

    row = None
    with (args.out/'metrics.jsonl').open('w') as log:
        for t in range(start, args.updates+1):
            w = weights_at(t, sched)
            logged = t % args.eval_every == 0 or t == args.updates
            if logged:
                with timing.measure('evaluate'):
                    row, train_logits = c2train.evaluate(p, data, nbr)
                    row = dict(step=t, **row, step_weights=w)
            if t in args.checkpoint_steps or t == args.updates:
                nets.save(args.out/f'ckpt-{t}.npz', p, st, step=t, run_id=args.run_id, scheme=c2rnn.scheme(p))
            if t == args.updates:
                if logged:
                    log.write(json.dumps(row)+'\n')
                break
            if t == sched['reset_at']:
                st = nets.adam_state(p)
            with timing.measure('update'):
                g = c2rnn.loss_grad(p, fit['s'], fit['ops'], targets, w)[2]
                before = {k: v.copy() for k, v in p.items()} if logged and args.diag else None
                nets.adam_step(p, g, st, args.lr, args.wd)
            if logged:
                if args.diag:
                    row['diag'] = c2train.diagnostics(before, p, g, st, train_logits, targets[2])
                log.write(json.dumps(row)+'\n')
    (args.out/'runtime.json').write_text(json.dumps(dict(status='complete', **timing.snapshot()), indent=1))


if __name__ == '__main__':
    main()
