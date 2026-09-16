# V31 live decision log

## 2026-09-16 V30 closure

- Audited canonical worktree and remote branch at V30 evidence SHA `67c8b11dbd6775542e894fd3cde3700c3ffad626`.
- Slurm jobs `56070183`, `56070184`, and `56070295` were terminal. Unrelated job `55950433` was observed and left untouched.
- Fetched 408 front/Mura files (577 MiB) without changing the remote archive. The local and remote sorted SHA-256 streams both hash to `dcd6dd00c3105b3846671480a0095c8db089e1fadb37d5191515e76caed569df`.
- Preserved the stale V30 planning manifest and reconciled it separately at commit `512b8f0`.

## Mura B1 decision

- Four homogeneous/broadband cases completed matched 10% and 20% strain horizons with the accepted-step kinematic gates intact.
- The 64 and 128 mechanically heterogeneous cases stopped at 4.05498% and 3.18866% strain because Mura line storage exceeded available plastic work.
- At the last common 3% strain horizon, physical organization observables did not meet 5% grid agreement. B2 is not authorized; no coefficient was tuned and no case was resubmitted.
- Decision: `TIER_B1_MECHANICAL_HETEROGENEITY_THERMODYNAMIC_ADMISSIBILITY_FAILURE`.

## Production repairs

- Commit `6d2a025` replaced ray-crossing identity with periodic contour-component topology. The former 64/128 favorable failures replay through the old exception with one persistent component and no topology event.
- Commit `c2b9fa4` moved physical ASB orientation and junction ownership onto the common Mura/wall path. Numerical constraints no longer drive orientation or enter physical energy/heat.
- The merged repository passes 599 tests. Front focused tests pass 80/80; ASB local gates pass at 32/64/128 with relative first-law residual `1.2149e-5` in the initial production smoke and `1.80e-9`--`8.53e-9` in the matched anchor preflight.

## Long continuations

- Pushed immutable source `c643afe00b4ea6f7f30e021675d7648ecdb6422c`.
- A restartable local front continuation is running for ten failed discriminating cases only, with concurrency four and checkpoints no farther than 840 seconds apart. Equal and mobility-off controls were not replayed. One near-equal 64 case has already reached a named physical terminal; the adapter did not crash.
- A four-case 128-square ASB anchor is running locally from the same immutable source: heterogeneous/homogeneous crossed with adiabatic/isothermal at 1100 K, 3e4/s, seed 43. Its resolved scientific outcome is pending.
- No V31 Slurm job was submitted. The unrelated Slurm job remains untouched.

