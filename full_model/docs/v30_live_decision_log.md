# V30 live decision log

## 2026-09-15 restart

- Local and remote canonical SHA both equal `45a0f9b3c79e69ab2fdec0914f5bd76fd8a43ff3`.
- Canonical worktree was clean before V30 evidence/controller creation.
- V29 decision artifacts were frozen by SHA-256.
- HPC3 queue access through configured host `uci-hpc3` succeeded.
- Unrelated job `55950433` was observed and left untouched.
- Three isolated worktrees were created for front, Mura-Nye, and ASB ledger production integration.
- All long-run matrices remain dependency-blocked until their own production hard gate passes.

## 2026-09-15 merged production integration

- Front production integration passed the merged 32/64 gate as
  `PRODUCTION_COUPLED_BIDIRECTIONAL_FRONT_QUALIFIED_LOCAL`; equal and
  mobility-off ledgers remain zero, favorable/reversed motion has opposite
  direction, and the legacy afterburner call count is zero.
- Mura-Nye production integration passed as
  `PRODUCTION_MURA_NYE_KINEMATICS_QUALIFIED`; transition-band dual-Nye RMS is
  0.0566--0.2167%, normalized line divergence is zero, and restart is bitwise
  exact.
- ASB production ledger software and restart integration pass, but the
  scientific gate is `ASB_DISSIPATION_CHANNEL_MISSING`. The default legacy
  junction-relaxation path lacks an independent affinity, and 32/64/128
  first-law residuals do not converge below 5%. Long ASB remains blocked.
- The merged canonical suite passes 565 tests in 63.56 seconds; 78 focused
  cross-branch tests pass.
- Front and Mura-Nye long bundles are READY pending immutable bundle scripts,
  clean-source push, checksums, and a fresh queue audit.

## 2026-09-16 overnight submissions

- Combined source and bundle scripts pass 572 canonical tests in 62.95 s.
- Exact clean source `675d74183baf2043ad0f7c055fe6e3370435ae65` was pushed,
  cloned on HPC3, and verified against the remote branch before submission.
- Front A1 anchor array `56070183` and delta/history array `56070184` were
  submitted with four-case concurrency and restartable 8-hour allocations.
- Mura-Nye B1 job `56070185` was submitted with 8 CPUs, 32 GiB, and a
  restartable 16-hour allocation.
- ASB was not submitted because its production hard gate failed as
  `ASB_NUMERICAL_CONSTRAINT_CONTAMINATES_PHYSICAL_LEDGER`.
- Queue audit before and after submission preserved unrelated job `55950433`.

## 2026-09-16 Mura launcher recovery

- Mura attempts `56070185` and `56070276` failed before scientific execution.
  Preserved runner logs identify `ModuleNotFoundError: matplotlib`: the manual
  submission wrapper omitted the bundle's declared `anaconda/2025.12` module.
- Corrected job `56070295` loads that module and is running on `hpc3-15-16`.
  This is an infrastructure correction only; source, parameters, and cases are
  unchanged.
