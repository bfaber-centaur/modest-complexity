"""Run R3's history.py (read unedited from a lane commit) from a published ckA,
for the reviewer's replay of the B phase (QUESTION.md, Revisions).
  replay_history.py LANE_COMMIT CKA_RUN_ID CKA_SHA256 -- <history.py arguments without --from>
The ckA file is read from the commit into the run's output parent, then passed
with --from and --from-sha256, so history.py itself checks the hash.
"""
from pathlib import Path
import subprocess
import sys
import tempfile

from lanes import lane_module, ROOT

commit, src_run, sha = sys.argv[1:4]
rest = sys.argv[sys.argv.index('--')+1:]
H = lane_module(commit, 'history', 'history.py')
ck = Path(tempfile.mkdtemp())/'ckA.npz'
ck.write_bytes(subprocess.check_output(['git', '-C', str(ROOT), 'show',
                                        f'{commit}:research/regimes/history/runs/{src_run}/ckpt_t2000.npz']))
sys.argv = ['history.py', *rest, '--from', str(ck), '--from-sha256', sha]
H.main()
