"""Run R2's own train.py (read from a lane commit, unedited) with one extra
pretraining label added in memory, for the reviewer's `gate` counterexample arm
(QUESTION.md, Revisions). Usage, through run_jobs.py:
  arm_shared.py LANE_COMMIT NAME BITS -- <train.py arguments>
NAME=none adds nothing (replay check). BITS are 1-indexed, comma-separated.
"""
import json
import sys

from lanes import lane_module

commit, name, bits = sys.argv[1:4]
rest = sys.argv[sys.argv.index('--')+1:]
T = lane_module(commit, 'shared', 'train.py')
if name != 'none':
    ks = tuple(int(b) for b in bits.split(','))
    T.task.LABELS[name] = T.task.parity(*ks); T.task.RELEVANT[name] = ks
sys.argv = ['train.py', *rest]
T.main()
out = rest[rest.index('--out')+1]
cfg = json.load(open(f'{out}/config.json'))
cfg['reviewer'] = dict(lane_commit=commit, added_label=None if name == 'none' else dict(name=name, parity_of=bits),
                       note='seeds reused from the lane replicates on purpose, to pair this arm with theirs')
json.dump(cfg, open(f'{out}/config.json', 'w'), indent=1)
