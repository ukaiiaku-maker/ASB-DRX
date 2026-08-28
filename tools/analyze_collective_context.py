#!/usr/bin/env python3
"""Analyze selected complete DDD histories as structural context."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from asb_drx.collective_diagnostics import (
    depin_count_diagnostics,
    native_audit_branching_diagnostics,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_named_path(value: str) -> tuple[str, Path]:
    name, separator, raw_path = value.partition("=")
    if not separator or not name or not raw_path:
        raise argparse.ArgumentTypeError("expected NAME=PATH")
    return name, Path(raw_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--depin-counts", action="append", default=[], type=parse_named_path)
    parser.add_argument("--native-audit", action="append", default=[], type=parse_named_path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    report = {
        "schema": "asb-drx-collective-context-diagnostics/v1",
        "scientific_disposition": (
            "structural falsification context only; no reported value may parameterize production physics"
        ),
        "depin_count_cases": {},
        "native_audit_cases": {},
        "input_sha256": {},
    }
    for name, path in args.depin_counts:
        report["input_sha256"][name] = sha256(path)
        report["depin_count_cases"][name] = depin_count_diagnostics(path)
    for name, path in args.native_audit:
        report["input_sha256"][name] = sha256(path)
        report["native_audit_cases"][name] = native_audit_branching_diagnostics(path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
