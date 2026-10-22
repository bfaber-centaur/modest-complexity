#!/usr/bin/env python3
"""Copy run records from the local run root into sharing/runs/<run_id>/ with a MANIFEST.json
(r1's composition/analyze.py publish): config, metrics, runtime and the runner log are
committed; checkpoints are hashed in the manifest and committed only where listed in
--commit-checkpoints (as RUN_ID/FILE).

uv run --locked python research/composition-c2/sharing/share_publish.py --ids ... [--commit-checkpoints ...]
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
SMALL = ('config.json', 'metrics.jsonl', 'runtime.json')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', type=Path, default=HERE.parents[2]/'runs'/'c2')
    ap.add_argument('--ids', nargs='+', required=True)
    ap.add_argument('--commit-checkpoints', nargs='*', default=[])
    args = ap.parse_args()
    for rid in args.ids:
        src, dst = args.root/rid, HERE/'runs'/rid
        dst.mkdir(parents=True, exist_ok=True)
        manifest = []
        for f in sorted(src.iterdir()):
            committed = f.name in SMALL or f'{rid}/{f.name}' in args.commit_checkpoints
            if committed:
                shutil.copy2(f, dst/f.name)
            manifest.append(dict(file=f.name, bytes=f.stat().st_size, sha256=hashlib.sha256(f.read_bytes()).hexdigest(), committed=committed))
        log = args.root/f'{rid}.log'
        if log.exists():
            shutil.copy2(log, dst/'stdout.log')
        (dst/'MANIFEST.json').write_text(json.dumps(manifest, indent=1)+'\n')
        print(rid, len(manifest), 'files')


if __name__ == '__main__':
    main()
