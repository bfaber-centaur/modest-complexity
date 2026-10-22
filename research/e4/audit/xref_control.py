#!/usr/bin/env python3
"""Projection files for the xref control (QUESTION.md addendum 4): exempt-reference decay from starts
that lack B's edited structure, and from B with the frequencies permuted across units.

- init: the untrained e1 init (experiment.init, no start file); k from its update-0 weights.
- rf: the C1 random-feature start files (research/e4/audit/starts). They keep the init's U and V filtered at
  their dominant frequency, so the gate is that dominant() of each equals the init's k, and the init file is used.
- keep (A): the unedited 50k start files; gate: the lane's dominant() of each equals B's projection k,
  so B's own files are used unchanged (no file is written).
- sham: B's k permuted across the live units, numpy seed 41100, first draw per source.

k and live come from research/e2/mechanisms/unitfilter.py dominant() (the lane's projection.py rule);
no labels are used.

uv run --locked --group analysis python research/e4/audit/xref_control.py
"""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'research/e2/mechanisms'))
sys.path.insert(0, str(ROOT/'research/mechanisms'))
from experiment import init  # noqa: E402
from unitfilter import dominant  # noqa: E402

SEEDS = (30000, 30001, 30002)
OUT = HERE/'intervention'
B_FILES = ROOT/'research/e4/mechanisms/intervention'
NOTE = ('P keeps, for each live unit j, both the cosine and sine components at +-k[j] (one rfft bin) along the token axis '
        'of U and V and the label axis of W; DC and all other frequencies belong to the rest. {}')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    with np.load(path) as z:
        return [z[x] for x in 'UVW']


def write(name, k, live, note, start):
    path = OUT/name
    np.savez(path, k=k.astype(np.int64), live=live, note=np.array(NOTE.format(note)), start_file=np.array(start))
    return dict(file=str(path.relative_to(ROOT)), sha256=sha(path), live_units=int(live.sum()),
                units_per_frequency=np.bincount(k[live], minlength=16)[1:].tolist())


def main():
    OUT.mkdir(exist_ok=True)
    rng = np.random.default_rng(41100)
    rows = []
    for s in SEEDS:
        with np.load(B_FILES/f'decay_projection_i{s}.npz') as z:
            kB, liveB = z['k'], z['live']
        k, _, live = dominant(list(init(31, 128, s)[:3]))
        rows.append(dict(source=s, start='init', **write(f'decay_projection_init_i{s}.npz', k, live,
                    f'k[j] is the dominant input frequency of unit j in experiment.init(31, 128, {s}).', f'experiment.init(31, 128, {s})')))
        start = HERE/f'starts/e4-au-rf-w128-i{s}-a1.npz'
        k_rf, _, live_rf = dominant(load(start))
        rows.append(dict(source=s, start='rf', start_sha256=sha(start), gate_k_equals_init=bool(np.array_equal(k_rf, k)),
                         gate_live_equals_init=bool(np.array_equal(live_rf, live)), file=rows[-1]['file'], sha256=rows[-1]['sha256']))
        start = ROOT/f'research/e3/mechanisms/starts/e3-me-keep-w128-i{s}-a1.npz'
        k, _, live = dominant(load(start))
        rows.append(dict(source=s, start='keep', start_sha256=sha(start), gate_k_equals_B=bool(np.array_equal(k, kB)),
                         gate_live_equals_B=bool(np.array_equal(live, liveB)),
                         file=str((B_FILES/f'decay_projection_i{s}.npz').relative_to(ROOT)), sha256=sha(B_FILES/f'decay_projection_i{s}.npz')))
        kp = kB.copy()
        kp[liveB] = rng.permutation(kB[liveB])
        rows.append(dict(source=s, start='sham', same_k_share=float(np.mean(kp[liveB] == kB[liveB])), **write(
            f'decay_projection_sham_i{s}.npz', kp, liveB, 'k is B\'s per-unit k permuted across the live units (numpy seed 41100).',
            f'research/e3/mechanisms/starts/e3-me-filt-w128-i{s}-a1.npz')))
    (OUT/'decay_projection.json').write_text(json.dumps(dict(files=rows), indent=1) + '\n')
    print(json.dumps(rows, indent=1))


if __name__ == '__main__':
    main()
