# Evidence inventory

`evidence_manifest.json` is generated from the immutable supplied root `/Users/sdillon/DRX-ASB` using `tools/build_evidence_manifest.py`. It records every regular file, not just selected successes. Automated roles/topics are discovery labels and do not replace the reviewed claim matrix or DD inventory.

The raw evidence is intentionally not copied into Git. Absolute source paths, SHA-256 hashes, sizes, types, and timestamps preserve identity; run manifests will stage exact selected sources by hash.
