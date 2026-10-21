#!/usr/bin/env python3
"""Publish more checkpoints of an e1 run that this session produced and still holds locally.

Each checkpoint's SHA-256 must match the entry in the run's published e1 manifest
(which lists every local file, published or not); otherwise nothing is copied. The e1
run directory is left untouched: the extra checkpoints go to DEST/RUN_ID/ with
supplement.json recording the manifest they were checked against.

uv run --locked python research/e2/dynamics/publish_e1_checkpoints.py LOCAL_RUN_DIR E1_PUBLISHED_RUN_DIR DEST \
    --steps 1000,2000,3000
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('local_run', type=Path)
    ap.add_argument('published_run', type=Path)
    ap.add_argument('dest_root', type=Path)
    ap.add_argument('--steps', required=True)
    args = ap.parse_args()
    manifest_path = args.published_run/'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    assert manifest['run_id'] == args.local_run.name == args.published_run.name
    names = [f'checkpoint_{int(s):06d}.npz' for s in args.steps.split(',')]
    for name in names:
        expected = manifest['files'][name]
        assert not expected['published'], f'{name} is already published in {args.published_run}'
        got = sha256(args.local_run/name)
        assert got == expected['sha256'], f'{name}: local {got} differs from manifest {expected["sha256"]}'
    dest = args.dest_root/manifest['run_id']
    dest.mkdir(parents=True, exist_ok=True)
    for name in names:
        shutil.copy2(args.local_run/name, dest/name)
        assert sha256(dest/name) == manifest['files'][name]['sha256']
    supplement = dest/'supplement.json'
    old = json.loads(supplement.read_text())['files'] if supplement.exists() else {}
    supplement.write_text(json.dumps(dict(
        run_id=manifest['run_id'], e1_manifest=str(manifest_path), e1_manifest_sha256=sha256(manifest_path),
        note='Checkpoints of the e1 run not published in its e1 directory; each SHA-256 equals the e1 manifest entry.',
        files={**old, **{n: manifest['files'][n]['sha256'] for n in names}}), indent=1) + '\n')
    print(manifest['run_id'], len(names), 'checkpoints verified and copied to', dest)


if __name__ == '__main__':
    main()
