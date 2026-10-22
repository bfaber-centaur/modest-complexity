"""Where the sharing lane's 12 runs live, and r1-style metric rows with N / notN for each.

TT and TU are the original r1 runs (published metrics, recovered checkpoints); UT and UU are
this lane's c2train.py runs. The follow-up variant 'wd0' (all four cells at wd 0) is this lane's
c2-share-wd0-* runs for every cell. For r1 runs, N / notN at each logged step are computed from the
logged held_final_pred and c2task.neighbour_answerable, since r1 did not log them.
"""
import json
from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path[:0] = [str(HERE.parent/'common'), str(REPO/'research'/'regimes'/'common')]
import c2task  # noqa: E402

CELLS = ('TT', 'TU', 'UT', 'UU')
SEEDS = (51200, 51201, 51202)
TIE = dict(TT=('tied', 'tied'), TU=('tied', 'untied'), UT=('untied', 'tied'), UU=('untied', 'untied'))
RUNS_ROOT = REPO/'runs'/'c2'          # local run output (git-ignored), as in the plan's {out}
LANE_RUNS = HERE/'runs'               # committed run records
RECOVERED = HERE.parent/'recovered'
R1_RUN = dict(TT=REPO/'research/regimes/composition/runs/r1-comp-rnn-inter-s{}-a1',
              TU=REPO/'research/regimes/review/runs/r1-rev-comp-rnn-untied-s{}-a1')
R1_ID = dict(TT='r1-comp-rnn-inter-s{}-a1', TU='r1-rev-comp-rnn-untied-s{}-a1')


def run_id(cell, seed, variant='main'):
    if variant == 'wd0':
        return f'c2-share-wd0-{cell.lower()}-s{seed}-a1'
    assert variant == 'main', variant
    return R1_ID[cell].format(seed) if cell in R1_ID else f'c2-share-{cell.lower()}-s{seed}-a1'


def replay_id(cell, seed):
    return f'c2-share-replay-{cell.lower()}-s{seed}-a1'


def run_dir(rid):
    """Committed record if present, else the local output directory."""
    return LANE_RUNS/rid if (LANE_RUNS/rid).exists() else RUNS_ROOT/rid


def find(rid, name):
    """A run file, committed record first, then the local output (checkpoints are mostly local)."""
    for d in (LANE_RUNS/rid, RUNS_ROOT/rid):
        if (d/name).exists():
            return d/name
    raise FileNotFoundError(f'{rid}/{name}')


def metrics_path(cell, seed, variant='main'):
    if variant != 'main':
        return run_dir(run_id(cell, seed, variant))/'metrics.jsonl'
    if cell in R1_RUN:
        return Path(str(R1_RUN[cell]).format(seed))/'metrics.jsonl'
    return run_dir(run_id(cell, seed))/'metrics.jsonl'


def checkpoint(cell, seed, step, variant='main'):
    """Path of a checkpoint: recovered original for TT/TU, local or committed otherwise."""
    if variant != 'main':
        return find(run_id(cell, seed, variant), f'ckpt-{step}.npz')
    if cell in R1_ID:
        path = RECOVERED/R1_ID[cell].format(seed)/f'ckpt-{step}.npz'
        if not path.exists():
            raise FileNotFoundError(path)
        return path
    return find(run_id(cell, seed), f'ckpt-{step}.npz')


def task():
    T = c2task.load(c2task.ORIGINAL)
    nbr = c2task.neighbour_answerable(T['P'], T['train'], T['held'])[0]
    return T, nbr


def read_metrics(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def subset_from_pred(row, held_final, nbr):
    """N / notN held-out final accuracy from a logged held_final_pred string."""
    ok = np.array([int(c) for c in row['held_final_pred']]) == held_final
    return dict(N=float(ok[nbr].mean()), notN=float(ok[~nbr].mean()))


def trajectory(cell, seed, T=None, nbr=None, variant='main'):
    """List of dicts: step, train_final, train_step1, train_step2, held_final, N, notN."""
    if T is None:
        T, nbr = task()
    held_final = c2task.rows(T['P'], T['held'])['states'][:, -1]
    out = []
    for row in read_metrics(metrics_path(cell, seed, variant)):
        sub = row.get('held_subset') or subset_from_pred(row, held_final, nbr)
        out.append(dict(step=row['step'], train_final=row['train']['final']['acc'], train_step1=row['train']['step1']['acc'],
                        train_step2=row['train']['step2']['acc'], held_final=row['held']['final']['acc'],
                        N=sub['N'], notN=sub['notN']))
    return out
