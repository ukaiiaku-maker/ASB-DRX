#!/usr/bin/env python3
"""Checksum the ignored raw local evidence underlying the V21 decisions."""

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
    parser.add_argument("--root", type=Path, action="append", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    files = []
    for root in args.root:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                files.append({"path": str(path), "bytes": path.stat().st_size,
                              "sha256": digest(path)})
    result = {
        "schema": "asb-drx/v21-local-evidence-manifest/v1",
        "scientific_source_commit": args.source_commit,
        "execution": "local_same_host",
        "roots": [str(root) for root in args.root],
        "file_count": len(files), "files": files,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({"file_count": len(files)}, indent=2))


if __name__ == "__main__":
    main()
