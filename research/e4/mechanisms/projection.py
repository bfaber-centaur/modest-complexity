#!/usr/bin/env python3
"""Decay-projection files for the intervention (INTERVENTION.md; DECISIONS D14), one per source.

Each file holds, for the filtered start file of that source, every unit's dominant input frequency
at update 0 (`k`, research/e2/mechanisms/unitfilter.py `dominant`: the frequency the filter kept;
no labels are used) and the live mask (`live`), the same basis rule as channels.py. experiment.py
`decay_parts` reads them: P keeps each live unit's cosine and sine components at +-k_j along the
token axis of U and V and the label axis of W; DC and every other frequency belong to "the rest".

uv run --locked --group analysis python research/e4/mechanisms/projection.py
"""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'research/e2/mechanisms'))
sys.path.insert(0, str(ROOT/'research/mechanisms'))
from unitfilter import dominant  # noqa: E402

OUT = Path(__file__).resolve().parent/'intervention'
NOTE = ('P keeps, for each live unit j, both the cosine and sine components at +-k[j] (one rfft bin) along '
        'the token axis of U and V and the label axis of W; DC and all other frequencies belong to the rest. '
        'k[j] is the dominant input frequency of unit j in the filtered start file at update 0.')


def main():
    OUT.mkdir(exist_ok=True)
    rows = []
    for s in (30000, 30001, 30002):
        start = ROOT/f'research/e3/mechanisms/starts/e3-me-filt-w128-i{s}-a1.npz'
        with np.load(start) as z:
            w = [z[x] for x in 'UVW']
        k, _, live = dominant(w)
        path = OUT/f'decay_projection_i{s}.npz'
        np.savez(path, k=k.astype(np.int64), live=live, note=np.array(NOTE),
                 start_file=np.array(str(start.relative_to(ROOT))),
                 start_sha256=np.array(hashlib.sha256(start.read_bytes()).hexdigest()))
        rows.append(dict(source=s, file=str(path.relative_to(ROOT)), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                         live_units=int(live.sum()), units_per_frequency=np.bincount(k[live], minlength=16)[1:].tolist()))
    (OUT/'decay_projection.json').write_text(json.dumps(dict(note=NOTE, files=rows), indent=1) + '\n')
    print(json.dumps(rows, indent=1))


if __name__ == '__main__':
    main()
