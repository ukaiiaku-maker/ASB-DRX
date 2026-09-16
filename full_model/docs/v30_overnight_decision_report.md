# V30 overnight production-integration and qualification checkpoint

Date: 2026-09-15/16

## Executive decision

V30 moved the front, Mura-Nye, and dissipation-ledger work into production.
This is no longer a standalone-fixture checkpoint.

| branch | local production gate | long-run state |
|---|---|---|
| coupled front | `PRODUCTION_COUPLED_BIDIRECTIONAL_FRONT_QUALIFIED_LOCAL` | running controls; driven cases expose `SIBM_GEOMETRIC_TRACKER_PRODUCTION_ADAPTER_FAILURE` |
| Mura-Nye | `PRODUCTION_MURA_NYE_KINEMATICS_QUALIFIED` | B1 job `56070295` running |
| ASB ledger | `ASB_NUMERICAL_CONSTRAINT_CONTAMINATES_PHYSICAL_LEDGER` | blocked; not submitted |
| fully integrated | blocked by ASB | not authorized |

The combined source checkpoint is
`675d74183baf2043ad0f7c055fe6e3370435ae65`. Local and GitHub remote SHAs
matched before staging. The combined canonical suite passes 572 tests in
62.95 seconds.

## Production front

Production defaults to the atomic bidirectional adapter. The former
phase-first afterburner requires two explicit reproduction switches and was
called zero times in V30 scientific cases. Phase motion and the directional
defect transaction are accepted atomically; restart includes signed component
identity, two directional envelopes, wake/revisit state, attempts, acceptances,
rejections, and complete line/Burgers/energy/heat ledgers.

The merged 32/64 short gate passes equal, mobility-off, favorable, and reversed
controls. The longer A1 calculation is more discriminating. At this checkpoint
the 64-grid equal case completed 2000 steps, but many driven and near-equal
cases generated a change in resolved contour-crossing count. The tracker stops
rather than silently rematching a changed topology. The first exception is in
`signed_front_geometry._match`, reached from
`coupled_front_production.accept_coupled_front_candidate`.

This is classified `SIBM_GEOMETRIC_TRACKER_PRODUCTION_ADAPTER_FAILURE`. It is
not repaired by a threshold or by restoring diffuse phase-volume sweep.
Remaining equal/mobility/cycle tasks are allowed to finish and all partial
checkpoints are retained.

## Production Mura-Nye

One family-resolved accepted flux now owns signed population transport,
alignment, plastic slip, plastic distortion, tensorial Nye, orientation,
plastic work, heat, conservative locking/unlocking, capture, and declared
topology sources. There is no post-step Nye or alignment projection.

Manufactured 16/32/64/128 controls pass. Actual transition-band dual-Nye RMS is
0.2167%, 0.1702%, and 0.0566% at 16/32/64; normalized line-continuity residual
is zero; energy closure is at most `7.7e-17` relative; restart is bitwise exact.

The B1 matrix is running as job `56070295`. Attempts `56070185` and `56070276`
are preserved infrastructure failures: the manual wrapper omitted the declared
Anaconda module, producing `ModuleNotFoundError: matplotlib` before scientific
execution. The corrected wrapper loads `anaconda/2025.12`; all six case
processes started with empty error logs.

## Production ASB energy and dissipation

The production driver now carries typed plastic, recovery, annihilation,
junction, boundary, thermal, and sink channels through diagnostics and exact
restart. Numerical compatibility penalties remain excluded from physical
energy and heat, and no residual-defined dissipation exists.

The default legacy junction-relaxation law has no chemical potential, reverse
rate, or conjugate state, so no independent junction affinity can be derived.
More decisively, the orientation/boundary operator changes numerical constraint
energy by `-1.31e22`, `-4.01e22`, and `-7.77e22 J m-3` at 32/64/128 while it is
still driven by derivatives of `A_alpha/A_GB`. First-law residuals are 63.1%,
3.95%, and 108.9%; the isolated 64-grid value is not converged. The branch is
therefore `ASB_NUMERICAL_CONSTRAINT_CONTAMINATES_PHYSICAL_LEDGER`, with missing
junction ownership secondary. No ASB long run was submitted.

## Jobs and provenance

- Front anchor array `56070183`: submitted, mixed completed/running/scientific-failure cases.
- Front delta/history array `56070184`: submitted; partial scientific failures retained.
- Mura B1 `56070295`: running on `hpc3-15-16`.
- Superseded infrastructure attempts: `56070185`, `56070276`.
- Unrelated job `55950433`: observed and untouched.
- Remote run root:
  `/pub/sdillon1/codex-runs/asb-drx-full-v34-recovery/v30-20260916T052139Z-675d741`.

Field figures, structure factors, convergence plots, and terminal long-run
classifications remain pending the running bundles. The restartable jobs should
remain active; their absence from this checkpoint would not authorize replacing
partial evidence with a rerun.

## Claims remaining false

- generic stored-energy SIBM;
- front grid convergence and physical hysteresis;
- current-capacity cone and persistent physical LAGB precursor;
- physically conditioned ASB ledger and resolved-grid ASB;
- strict ASB;
- full-model DRX;
- fully integrated front/Mura/ASB production model.
