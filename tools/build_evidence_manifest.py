#!/usr/bin/env python3
"""Build a complete immutable file inventory without interpreting numeric data."""

from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def classify(path: Path) -> tuple[str, str]:
    lower = str(path).lower()
    suffix = path.suffix.lower()
    name = path.name.lower()
    if suffix in {".py", ".m", ".sh", ".bash"}:
        role = "code"
    elif suffix in {".csv", ".tsv", ".mat", ".npz", ".npy", ".h5", ".hdf5", ".enl"}:
        role = "derived_data" if any(x in lower for x in ("result", "summary", "sweep", "out_")) else "raw_evidence_or_data_unknown"
    elif suffix in {".png", ".tif", ".tiff", ".fig", ".avi"}:
        role = "derived_visualization"
    elif suffix in {".pdf", ".ris", ".enw"}:
        role = "external_literature" if "reference" in lower or name.startswith("s") else "project_document_or_literature"
    elif suffix in {".md", ".docx", ".txt"}:
        role = "prior_conclusion_or_hypothesis"
    elif name == ".ds_store":
        role = "filesystem_metadata"
    else:
        role = "unclassified_evidence"

    if any(x in lower for x in ("avalanche", "opendis", "ddd", "multi_hit", "multihit")):
        topic = "DD_or_collective_events"
    elif any(x in lower for x in ("v32", "v33", "v34", "shear_band", "asb")):
        topic = "legacy_DRX_ASB"
    elif "garofalo" in lower:
        topic = "flow_stress_material_data"
    elif "poliak" in lower or "jonas" in lower:
        topic = "DRX_critical_condition"
    elif "prasad" in lower:
        topic = "processing_map"
    elif "texture" in lower:
        topic = "texture"
    elif "potts" in lower:
        topic = "Potts_DRX"
    elif "recryst" in lower or "recrys" in lower or "drx" in lower:
        topic = "recrystallization"
    else:
        topic = "project_context"
    return role, topic


def provenance(path: Path, root: Path) -> str:
    rel = path.relative_to(root)
    if rel.parts and rel.parts[0] == "HPC3":
        return "project_folder_HPC3_copy_or_result"
    if "references" in rel.parts:
        return "external_literature_in_project_folder"
    if any(part.startswith("results") or part.startswith("out_") for part in rel.parts):
        return "project_derived_output"
    return "project_folder_supplied_source"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--reuse-hashes", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output.resolve()
    prior: dict[str, dict[str, object]] = {}
    if args.reuse_hashes and output.exists():
        existing = json.loads(output.read_text())
        prior = {str(item["path"]): item for item in existing.get("files", [])}
    entries: list[dict[str, object]] = []
    errors: list[dict[str, str]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        try:
            stat = path.stat()
            role, topic = classify(path.relative_to(root))
            mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            old = prior.get(str(path))
            digest = str(old["sha256"]) if old and int(old.get("size_bytes", -1)) == stat.st_size else sha256(path)
            entries.append(
                {
                    "path": str(path),
                    "relative_path": str(path.relative_to(root)),
                    "sha256": digest,
                    "file_type": mime,
                    "size_bytes": stat.st_size,
                    "modified_at": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
                    "date_basis": "filesystem_mtime; embedded publication/run date not inferred",
                    "source_provenance": provenance(path, root),
                    "scientific_topic": topic,
                    "evidence_role": role,
                    "classification_method": "path/type heuristic; review required for scientific interpretation",
                }
            )
        except (OSError, ValueError) as exc:
            errors.append({"path": str(path), "error": str(exc)})
    roles = Counter(str(item["evidence_role"]) for item in entries)
    topics = Counter(str(item["scientific_topic"]) for item in entries)
    payload = {
        "schema": "asb-drx-evidence-manifest/v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "scope": "all regular non-symlink files under supplied project evidence root",
        "hash_algorithm": "SHA-256",
        "file_count": len(entries),
        "total_size_bytes": sum(int(item["size_bytes"]) for item in entries),
        "category_counts": dict(sorted(roles.items())),
        "topic_counts": dict(sorted(topics.items())),
        "errors": errors,
        "files": entries,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + os.linesep)
    print(json.dumps({k: payload[k] for k in ("file_count", "total_size_bytes", "category_counts", "topic_counts", "errors")}, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
