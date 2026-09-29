#!/usr/bin/env python3
"""SHA-256 checksum of every trained model file, so that each can be cited exactly.

Run from the repository root after the experiments:
    python tools/checkpoint_manifest.py > checkpoints_sha256.txt
"""
import hashlib
from pathlib import Path

PATTERNS = [
    'gnn_outputs_shared/checkpoints/*.pt',               # graph encoders, step 3
    'TimeXer/checkpoints*/*/checkpoint.pth',             # TimeXer runs, steps 4-9
    'fnspid_transformer/results_*/*/model.pt',           # FNSPID Transformer runs, steps 8-9
]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


for pattern in PATTERNS:
    for path in sorted(Path('.').glob(pattern)):
        print(f'{sha256(path)}  {path.as_posix()}')
