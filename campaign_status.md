# Campaign status

Updated: 2026-08-27 (America/Los_Angeles)

## Current gate: Gate 0 -- in progress

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

## Active no-go conditions

1. Gate 0 cannot pass until the exact DD evidence is located or its absence is formally resolved.
2. Gate 1 fitting cannot begin without observables, units, condition axes, seed/cell-size metadata, event definitions, censoring, uncertainty, and a calibration/validation split.
3. Production solver implementation cannot begin before the free-energy/dissipation review and closure choice.

## Next executable actions

1. Complete and hash the evidence inventory.
2. Commit and push the Gate 0 audit milestone.
3. Continue searching project/remote campaign storage for raw DD event trajectories and metadata.
4. Prepare exact legacy v32/v33/v34 HPC3 regression submissions, without arrays.
