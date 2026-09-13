# Full v34 recovery campaign

This directory is the production campaign selected by Full-Model Recovery
Directive v5.  `reference_sources/` contains immutable, byte-identical copies
of the supplied v32--v34 evidence.  Production changes belong in a separate
`production/` tree and must be compared against these references.

The Mission-v3 reduced package remains importable for component regressions,
but it is not called by the full-model driver and is not a production DRX/ASB
architecture.

Evidence is intentionally not copied into Git.  Exact absolute evidence roots,
the fetched HPC3 roots, and representative result hashes are recorded in
`provenance_manifest.json`.

Current reference interpretation:

- v32 is the immutable ASB-like reference.
- v33 is the false-grain/over-eager-label negative control and is not routinely
  rerun.
- v34 is the production architecture.  In the reproduced coupled baseline its
  site and rate fields are nonzero, but cumulative hazard never reaches a
  stochastic threshold.  Consequently no raw event, candidate, promotion, or
  physical grain occurs.  The first observed failure is hazard exposure, not
  candidate persistence or promotion.
