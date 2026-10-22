#!/usr/bin/env python3
"""Host calibration for the regime expedition: full-batch Adam updates of the
shared network at sizes like each regime's first design. Random inputs and
labels (calibration seeds 900000+), so nothing here is a scientific result.
A short calibration does not forecast long runs; each lane recalibrates.

uv run --locked python research/regimes/coordinator/calibrate.py --out {out} --updates 2000
"""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'common'))
import nets  # noqa: E402
from instrumentation import Timings  # noqa: E402

SHAPES = {  # name: (rows, inputs, hidden widths, heads)
    'composition-mlp': (384, 20, (128, 128), dict(final=8, step1=8, step2=8)),
    'shared-mlp': (1024, 10, (64,), dict(y1=2, y2=2)),
    'history-mlp': (1024, 11, (128,), dict(A=2, B=2)),
    'routes-mlp': (512, 10, (128,), dict(y=2)),
    'wide-mlp': (1024, 20, (512, 512), dict(y=8)),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--updates', type=int, default=2000)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    timing = Timings(); rows = {}
    for i, (name, (n, d, hidden, heads)) in enumerate(SHAPES.items()):
        r = np.random.default_rng(900000+i)
        X = r.integers(0, 2, (n, d)).astype(float)
        targets = {h: r.integers(0, k, n) for h, k in heads.items()}
        p = nets.init(d, hidden, heads, 900100+i); st = nets.adam_state(p)
        start = time.perf_counter()
        updates = args.updates//5 if name == 'wide-mlp' else args.updates
        with timing.measure(name):
            for _ in range(updates):
                nets.adam_step(p, nets.loss_grad(p, X, targets)[2], st, 1e-3)
        with timing.measure(name+'-checkpoint'):
            nets.save(args.out/f'{name}.npz', p, st)
        sec = time.perf_counter()-start
        rows[name] = dict(rows=n, inputs=d, hidden=hidden, heads=heads, updates=updates,
                          seconds=round(sec, 3), updates_per_second=round(updates/sec, 1))
        print(name, rows[name], flush=True)
    report = dict(purpose='calibration only; random inputs and labels', shapes=rows,
                  provenance=nets.provenance(args.out, __file__), **timing.snapshot())
    (args.out/'calibration.json').write_text(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
