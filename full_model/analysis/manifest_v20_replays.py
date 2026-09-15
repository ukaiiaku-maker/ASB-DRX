#!/usr/bin/env python3
"""Write compact checksums for local V20 trajectory replay evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024*1024), b""):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = []
    for path in sorted(args.root.rglob("*")):
        if path.is_file():
            records.append({
                "path": str(path), "bytes": path.stat().st_size,
                "sha256": digest(path)})
    result = {
        "schema": "asb-drx/v20-local-replay-manifest/v1",
        "root": str(args.root), "file_count": len(records), "files": records,
    }
    args.output.write_text(json.dumps(result, indent=2)+"\n")


if __name__ == "__main__":
    main()
