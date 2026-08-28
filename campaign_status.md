# Campaign status

Updated: 2026-08-27 (America/Los_Angeles)

## Current gate: Gate 0 -- no-go

- Repository remote verified as `https://github.com/ukaiiaku-maker/ASB-DRX.git`.
- Isolated branch/worktree created from remote `main` at `a5dd798096e3896f319d314e8e4c60f5b277e589`.
- Source evidence root located at `/Users/sdillon/DRX-ASB` (33,358 files before filtering; about 20 GB).
- Six supplied files located at the evidence-root top level with upload suffixes normalized away.
- Legacy v32, v33 controls, and v34 sources/results located under `recrysyallization_PF-2D/shear_banding`.
- HPC3 aliases and runner verified. Existing unrelated local and Slurm campaigns were observed and left untouched.
- Exact raw DD datasets supporting the asserted density/temperature Poisson-to-multi-hit transition have **not yet been found**. The located v6 package is a simulated first-avalanche/hazard model, not a qualifying multi-condition raw DD dataset.
- DD fitting is stopped by design until event definitions and metadata are established.
- First-pass legacy audit and candidate thermodynamic architecture drafted.
- Full evidence inventory completed: 33,358 files, 21,536,785,369 bytes, no hash errors.
- Campaign-specific HPC3 smoke job `55633650` completed, fetched, and checksum-verified.
- Exact-source HPC3 legacy controls completed and were fetched with verified checksums: v32 job `55633674`, v33 job `55633691`, and v34 job `55633694`.
- v32 reproduces all finite numerical diagnostics within `atol=1e-10`, `rtol=1e-9`; 18 degenerate zero-variance correlations differ only as `NaN` versus roundoff near `1e-17`.
- v33 reproduces the false-grain mechanism but not its exact trajectory: 165 hazard births and 177 allocated labels versus 12 unchanged topology components. These are rejected as physical grains.
- v34 produces zero active, new, or promotable candidates and zero births throughout. This reproduces the stored zero-candidate failure, not the requested candidate-without-promotion premise. Its detailed trajectory is not numerically reproducible from the supplied source/configuration.

## Active no-go conditions

1. Gate 0 failed because the exact DD evidence and required metadata were not found. Its absence cannot be corrected scientifically by substituting the located synthetic first-avalanche model.
2. Gate 1 fitting cannot begin without observables, units, condition axes, seed/cell-size metadata, event definitions, censoring, uncertainty, and a calibration/validation split.
3. Production solver implementation cannot begin before the free-energy/dissipation review and closure choice.
4. The production material target is unresolved: legacy “Fe” naming conflicts with embedded chromium metadata and the validation folder mixes several alloys.

## Required external resolutions

1. Supply or identify the raw DD event trajectories and their event-definition/condition/seed/cell-size/censoring metadata.
2. Resolve whether the production target is Fe, Cr, or a named alloy and identify the authoritative material/validation dataset.
3. If trajectory-level v33/v34 reproduction is required, supply the original runtime environment and immutable source snapshot; the stored result folders do not contain them.

No closure fitting, architecture selection, production solver implementation, or production array is scientifically authorized while these no-go conditions remain.
