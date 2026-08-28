# Immutable legacy regression snapshots

These files are byte-for-byte copies of canonical supplied sources. They are never imported by the new production package. Any regression override is explicit in the adjacent `source_provenance.json`; the original driver/launcher is not edited.

- v32 is the principal ASB-like reference.
- v33 deliberately reuses the v32 driver with the aggressive v33 launcher and is a false-grain negative control.
- v34 is the candidate-bookkeeping failure-analysis control.

Numerical execution is restricted to HPC3. Output is compressed by the runner and fetched with verified checksums.
