#!/usr/bin/env python3
"""Copy finished run outputs from the local run root (outside git) into correction/runs/RUN_ID/,
with the job log as stdout.log and a MANIFEST.json of file sizes and SHA-256.

uv run --locked python research/composition-c3/correction/corr_collect.py --root runs/c3 RUN_ID ...
"""
import argparse
import json
from pathlib import Path
import shutil

import corr_core as cc
import nets


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('run_ids', nargs='+')
    args = ap.parse_args()
    for run in args.run_ids:
        src, dst = args.root/run, cc.HERE/'runs'/run
        dst.mkdir(parents=True, exist_ok=False)
        for f in sorted(src.iterdir()):
            shutil.copy2(f, dst/f.name)
        shutil.copy2(args.root/f'{run}.log', dst/'stdout.log')
        manifest = [dict(file=f.name, bytes=f.stat().st_size, sha256=nets.sha256_file(f), committed=True)
                    for f in sorted(dst.iterdir())]
        (dst/'MANIFEST.json').write_text(json.dumps(manifest, indent=1))
        print(run, len(manifest), 'files')


if __name__ == '__main__':
    main()
