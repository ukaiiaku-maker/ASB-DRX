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
