#!/usr/bin/env python3
"""Do the c2-native replays reproduce the original TT / TU runs? Writes out/replays.txt.

For each replay: every r1 metric key (step, train, held, held_final_pred) at every logged step
equals the original's metrics.jsonl exactly; the replay's logged held_subset equals N / notN
computed from the original's held_final_pred; and every array of ckpt-1000 and ckpt-10000
(weights, both Adam moments, step count) equals the recovered original's. File hashes differ by
design (run metadata), so arrays are compared.

uv run --locked python research/composition-c2/sharing/share_replays.py
"""
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import share_runs as S  # noqa: E402
import c2task  # noqa: E402
import nets  # noqa: E402

R1_KEYS = ('step', 'train', 'held', 'held_final_pred')
T, nbr = S.task()
held_final = c2task.rows(T['P'], T['held'])['states'][:, -1]
lines, ok_all = [], True
for cell in ('TT', 'TU'):
    for seed in S.SEEDS:
        rid = S.replay_id(cell, seed)
        rep = S.read_metrics(S.run_dir(rid)/'metrics.jsonl')
        orig = S.read_metrics(S.metrics_path(cell, seed))
        same_metrics = len(rep) == len(orig) and all({k: a[k] for k in R1_KEYS} == {k: b[k] for k in R1_KEYS} for a, b in zip(rep, orig))
        same_subset = all(a['held_subset']['N'] == S.subset_from_pred(b, held_final, nbr)['N'] and
                          a['held_subset']['notN'] == S.subset_from_pred(b, held_final, nbr)['notN'] for a, b in zip(rep, orig))
        same_ckpt = {}
        for step in (1000, 10000):
            p1, s1, _ = nets.load(S.run_dir(rid)/f'ckpt-{step}.npz')
            p0, s0, _ = nets.load(S.checkpoint(cell, seed, step))
            same_ckpt[step] = (s1['t'] == s0['t'] and set(p1) == set(p0) and
                               all(np.array_equal(d1[k], d0[k]) for d1, d0 in ((p1, p0), (s1['m'], s0['m']), (s1['v'], s0['v'])) for k in d0))
        ok = same_metrics and same_subset and all(same_ckpt.values())
        ok_all &= ok
        lines.append(f'{"PASS" if ok else "FAIL"} {rid} vs {S.run_id(cell, seed)}: {len(rep)} logged rows, r1 metric keys identical={same_metrics}, '
                     f'held_subset = N/notN from original predictions={same_subset}, ckpt-1000 arrays identical={same_ckpt[1000]}, '
                     f'ckpt-10000 arrays identical={same_ckpt[10000]}')
        print(lines[-1])
out = Path(__file__).resolve().parent/'out'; out.mkdir(exist_ok=True)
(out/'replays.txt').write_text('\n'.join(lines)+'\n')
sys.exit(0 if ok_all else 1)
