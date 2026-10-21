#!/usr/bin/env python3
"""Is a long-chunk replay mismatch an implementation difference or sensitivity?

From one published checkpoint, run three trajectories for N updates on the same
data and record the largest weight difference every 100 steps:
  shared       experiment.lossgrad + experiment.adam (the code under test)
  independent  research/audit/independent.py
  shared+ulp   the shared code from the same state with every weight moved by
               one unit in the last place (a pure sensitivity control)
If shared reproduces the next checkpoint exactly, and independent-vs-shared grows
like shared-vs-(shared+ulp), the gap is amplification of rounding differences,
not a formula difference.

uv run --locked python research/audit/divergence.py RUN_DIR START STOP OUT.json
"""
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import independent as ind  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiment import adam, lossgrad  # noqa: E402  (objects under test)

run_dir, start, stop, out = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), Path(sys.argv[4])
cfg = json.loads((run_dir/'config.json').read_text())
lr, wd = cfg['lr'], cfg['wd']
with np.load(run_dir/'data.npz') as d:
    tr = d['train']; a, b, y = d['a'][tr], d['b'][tr], d['y'][tr]
with np.load(run_dir/f'checkpoint_{start:06d}.npz') as z:
    ck = {k: z[k] for k in z.files}
target = run_dir/f'checkpoint_{stop:06d}.npz'
p = ck['U'].shape[0]
Xa, Xb = ind.one_hot_inputs(a, b, p)


def state():
    return ([ck[k].copy() for k in 'UVW'], [ck['m'+k].copy() for k in 'UVW'], [ck['v'+k].copy() for k in 'UVW'])


S, I, E = state(), state(), state()
E = ([np.nextafter(x, np.inf) for x in E[0]], E[1], E[2])
rows = []
for t in range(start + 1, stop + 1):
    adam(S[0], lossgrad(S[0], a, b, y)[1], S[1], S[2], t, lr, wd)
    adam(E[0], lossgrad(E[0], a, b, y)[1], E[1], E[2], t, lr, wd)
    ind.adamw_step(I[0], ind.loss_and_grad(I[0], Xa, Xb, y)[1], I[1], I[2], t, lr, wd)
    if t % 100 == 0:
        loss = lossgrad(S[0], a, b, y)[0]
        rows.append(dict(step=t, shared_train_loss=loss,
                         independent_vs_shared=max(float(np.max(np.abs(i - s))) for i, s in zip(I[0], S[0])),
                         ulp_vs_shared=max(float(np.max(np.abs(e - s))) for e, s in zip(E[0], S[0]))))
result = dict(run=run_dir.name, start=start, stop=stop, rows=rows)
if target.exists():
    with np.load(target) as z:
        result['shared_reproduces_stop_checkpoint_bitwise'] = all(
            np.array_equal(x, z[k]) for x, k in zip(S[0] + S[1] + S[2], ['U', 'V', 'W', 'mU', 'mV', 'mW', 'vU', 'vV', 'vW']))
        result['independent_max_abs_diff_at_stop'] = max(float(np.max(np.abs(x - z[k]))) for x, k in zip(I[0], 'UVW'))
out.write_text(json.dumps(result, indent=1) + '\n')
print(json.dumps({k: v for k, v in result.items() if k != 'rows'}))
for r in rows[::max(1, len(rows)//25)]:
    print(r)
