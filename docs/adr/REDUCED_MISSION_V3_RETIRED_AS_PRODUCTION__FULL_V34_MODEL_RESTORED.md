# ADR: REDUCED_MISSION_V3_RETIRED_AS_PRODUCTION__FULL_V34_MODEL_RESTORED

Status: accepted, 2026-09-12

## Decision

The coupled two-dimensional v34 phase-field model is restored as the production
DRX/ASB architecture.  v32 is retained unchanged as the ASB-like regression
reference and v33 as the false-grain negative control.

Mission-v3 reduced models remain importable analytical and numerical fixtures.
They must not replace, delay, or be called implicitly by the production driver.
The final delayed-memory closure is classified
`HARD_NONNEGATIVITY_VALIDITY_STOP_AFTER_HORIZON`: delayed history can amplify
heterogeneity, but that closure did not produce a bounded wall or DRX/ASB state.

## Evidence

The final reduced checkpoint is commit `fb464e1` and annotated tag
`reduced-v3-final-validity-stop-20260912`.  Its fetched HPC records remain in
the prior campaign manifest and result root.

Exact v32, v33, and v34 source copies, configurations, evidence roots, hashes,
and verified HPC3 archives are enumerated in
`full_model/provenance_manifest.json`.

The restored v34 baseline has nonzero eligible site and hazard-rate fields, but
its cumulative hazard remains far below every realized stochastic threshold.
It therefore produces no raw event, candidate, promotion, or physical grain.
The observed chain first fails at hazard exposure.  Separately, the v34 restart
writer and loader omit all candidate arrays, so candidate restart is not exact.

## Consequences

No full-model physics is changed during reference restoration.  The first
production patches are mechanism-specific entropy-separated EXP-floor kinetics
with an exact zero-entropy regression, followed by exact candidate state and RNG
restart.  Long scientific runs occur on HPC3 and preserve unrelated jobs.
