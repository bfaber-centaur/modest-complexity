#!/usr/bin/env python3
"""Copy finished runs from the local output root into runs/<run ID>/ with a SHA-256 manifest.

Committed: config.json, metrics.jsonl, runtime.json, the runner log (stdout.log) and the last
checkpoint (weights + Adam state, the input to every frozen evaluation). ckpt-0 is listed by hash
only: it is r1's deterministic init for the recorded seed (c2rnn.init) and is regenerated bitwise.

uv run --locked python research/composition-c2/generalization/gen_publish.py --root runs/c2 RUN_ID ...
"""
import argparse
import json
from pathlib import Path
import shutil
import sys

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[1]/'regimes'/'common')]
import nets  # noqa: E402


def publish(root, rid):
    src, dst = Path(root)/rid, HERE/'runs'/rid
    dst.mkdir(parents=True, exist_ok=True)
    last = max(src.glob('ckpt-*.npz'), key=lambda f: int(f.stem.split('-')[1]))
    commit = {'config.json', 'metrics.jsonl', 'runtime.json', last.name}
    manifest = []
    for f in sorted(src.iterdir()):
        manifest.append(dict(file=f.name, bytes=f.stat().st_size, sha256=nets.sha256_file(f), committed=f.name in commit))
        if f.name in commit:
            shutil.copy2(f, dst/f.name)
    log = Path(root)/f'{rid}.log'
    shutil.copy2(log, dst/'stdout.log')
    manifest.append(dict(file='stdout.log', bytes=log.stat().st_size, sha256=nets.sha256_file(log), committed=True))
    (dst/'MANIFEST.json').write_text(json.dumps(manifest, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('ids', nargs='+')
    args = ap.parse_args()
    for rid in args.ids:
        publish(args.root, rid)


if __name__ == '__main__':
    main()
