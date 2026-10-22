"""Two helpers shared by the reviewer's scripts: PASS/FAIL printing, and loading
a lane module from a given commit (git archive into a temp dir, read-only)."""
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[3]
FAIL = []


def check(name, ok, detail=''):
    print(('PASS ' if ok else 'FAIL ') + name + (f'  [{detail}]' if detail else ''))
    if not ok:
        FAIL.append(name)


def lane_module(commit, lane, filename='task.py'):
    tmp = Path(tempfile.mkdtemp())
    with open(tmp/'a.tar', 'wb') as f:
        subprocess.run(['git', '-C', str(ROOT), 'archive', commit, f'research/regimes/{lane}', 'research/regimes/common',
                        'instrumentation.py', 'experiment.py'], stdout=f, check=True)
    subprocess.run(['tar', '-xf', str(tmp/'a.tar'), '-C', str(tmp)], check=True)
    for p in (tmp, tmp/'research/regimes/common', tmp/f'research/regimes/{lane}'):
        sys.path.insert(0, str(p))
    spec = importlib.util.spec_from_file_location(f'{lane}_{Path(filename).stem}', tmp/f'research/regimes/{lane}'/filename)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m
