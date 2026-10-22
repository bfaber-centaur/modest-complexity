"""Compare the reviewer's `gate` arm and `part` replay (runs/regimes/r1-rev-share-*)
with the lane's published final.csv at a commit (REVIEW.md §2, M4).

uv run --locked python research/regimes/review/compare_gate.py LANE_COMMIT RUN_ROOT
"""
import csv
import io
import json
from pathlib import Path
import subprocess
import sys

from lanes import check, FAIL, ROOT

commit, root = sys.argv[1], Path(sys.argv[2])
rows = list(csv.DictReader(io.StringIO(subprocess.check_output(
    ['git', '-C', str(ROOT), 'show', f'{commit}:research/regimes/shared/figures/final.csv'], text=True))))
lane = {(r['arm'], r['branch'], r['init_seed'], int(r['n'])): r for r in rows if r['n']}


def mine(run):
    s = json.loads((root/run/'summary.json').read_text())['per_n']
    ro = {}
    for line in (root/run/'metrics.jsonl').read_text().splitlines():
        r = json.loads(line)
        if r.get('tag') == 'readout' and r['step'] == 2000:
            ro[r['n']] = r['new_test_acc_x4is1']
    return {int(n): v['new_test_acc_x4is1'] for n, v in s.items()}, ro


ft, ro = mine('r1-rev-share-replay-part-s52200-a1')
check('replay of lane part/52200 reproduces final.csv to its 6 printed digits (fine-tuned and readout, every n)',
      all(abs(ft[n] - float(lane['part', 'train', '52200', n]['new_test_acc_x4is1'])) < 5e-6 and
          abs(ro[n] - float(lane['part', 'readout', '52200', n]['new_test_acc_x4is1'])) < 5e-6 for n in ft))
print('fine-tuned (train branch) and frozen readout, test x4=1 accuracy; replicates 52200/52201/52202')
for br, idx in (('train', 0), ('readout', 1)):
    g = [mine(f'r1-rev-share-gate-s{s}-a1')[idx] for s in ('52200', '52201', '52202')]
    for n in (16, 32, 64, 128):
        vals = {a: [float(lane[a, br, s, n]['new_test_acc_x4is1']) for s in ('52200', '52201', '52202')]
                for a in ('fresh', 'rel', 'part', 'unrel', 'split') if (a, br, '52200', n) in lane}
        gv = [x[n] for x in g]
        d = lambda a: '/'.join(f'{x-y:+.2f}' for x, y in zip(gv, vals[a]))
        print(f'{br:7s} n={n:3d} gate {"/".join(f"{x:.2f}" for x in gv)} | ' +
              ' '.join(f'{a} {"/".join(f"{x:.2f}" for x in v)}' for a, v in vals.items()) +
              f' | gate-fresh {d("fresh")} gate-part {d("part")} gate-unrel {d("unrel")}')
print('ALL PASS' if not FAIL else f'{len(FAIL)} FAILED: {FAIL}')
