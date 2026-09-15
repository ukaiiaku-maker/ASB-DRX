# V25 reaction-cone, phase-symmetry, and ASB grid decision report

Date: 2026-09-15

V24 evidence is frozen by SHA-256 in
`full_model/verification/v25_frozen_v24_evidence.json`. No V24 file or archive
was overwritten, no HPC3 campaign job was launched, and unrelated live job
`55950433` was not modified.

## Wall topology and authoritative Nye

The capacity-limited algebraic cone formed from the four signed BCC transport
families spans the missing local Frank--Bilby vector in both 32-grid V24
candidate states. With the measured transport exposure credited only to the
transport channel, however, only about 9.7--10.1% of the required optimized
extent is exposed. The algebraic sub-decision is therefore
`FB_TARGET_REACHABLE_BUT_KINETICALLY_UNEXPOSED`.

That result is not yet an authoritative physical-cone decision. The V24
finite-segment reorientation events rotate stored Nye locally without swept
plastic area, an explicit loop/source/sink, or a matching plastic-distortion
increment. They are now rejected by the audit. Moreover, the V24
Curl-beta/reservoir-Nye relative RMS mismatch remains 1.24--1.39. The governing
classification is consequently:

`REACTION_CONE_TEST_INVALID_DUE_TO_STATE_INCONSISTENCY`.

No topology-rate tuning or long wall calculation is authorized. The next wall
repair must make every accepted reaction obey total authoritative Nye
continuity and then repeat the cone with current, rather than initial-bound,
donor capacity.

Machine result: `full_model/verification/v25_reaction_cone_local.json`.

## Clean symmetric planar SIBM

The production driver now supports a fresh periodic planar bicrystal built
without a restart. It starts with zero total strain, plastic strain, stress,
history, boundary reservoir, and seeded curvature. Compatibility and drag
energy are interface-owned common contributions rather than child-lineage
charges. An exact all-defect freeze restores every defect, thermal,
mechanical, front, tracker, and history state bitwise after each phase step.

The common functional's analytical planar derivative agrees with a centered
finite difference, equal complete states are stationary, and reversing the
physical defect contrast reverses both the derivative and motion. In the
actual full production driver at 32 square and 1000 steps:

- equal: `1.1e-9` interface widths (stationary);
- favorable: `+0.26156` interface widths with stable sign;
- reversed: `-0.26156` interface widths with stable sign;
- mobility off: exactly zero displacement.

All four branches have the identical pre-step phase-geometry hash, remain at
exact zero load, and preserve all frozen non-phase fields bitwise. This passes
the fixed 0.25-interface-width phase-only requirement:

`CLEAN_PRODUCTION_PHASE_ONLY_SIBM_SIGN_QUALIFIED`.

This is not yet a complete dynamic SIBM pass because conservative transfer,
boundary storage, recovery/heat, and constitutive rehardening have not been
activated sequentially in the repaired driver. HPC3 SIBM execution remains
unauthorized until the first sign-breaking stage is identified or all stages
pass.

Machine results: `full_model/verification/v25_symmetric_sibm_local.json` and
`full_model/verification/v25_clean_sibm_matrix.json`.

## ASB physical grid audit

The 64/128 seed-43 evidence uses equal domain size, intrinsic and resolved
interface width, requested heat-kernel width, thermal diffusivity, heat
capacity, conductivity, mechanical correlation length, hotspot width, and
represented thickness. The two-pixel heat-kernel floor produces 0.3125 versus
0.3000 micrometres effective width, a 4.0% difference below the provisional
5% threshold. The restricted 128-grid grain partition differs in only 3.125%
of permutation-invariant neighbor relations.

Over the common physical interval 2.974--5.529 microseconds, heat-to-external-
work normalization differs by only 1.5%, so the discrepancy is not classified
as a heat-deposition normalization defect. The physical response itself is
not converged: external work differs by 8.0%, deposited heat by 9.4%, stored
thermal-energy change by 11.2%, and raw plastic work by 41.3%.

Decision:

`ASB_PHYSICAL_RESPONSE_NOT_YET_GRID_CONVERGED`.

The strain-rate bracket remains unauthorized. The next comparison must use a
matched refinement trajectory and resolve the mechanical/plastic-work response,
not weaken the strict ASB criteria.

Machine result: `full_model/verification/v25_asb_grid_audit.json`.

## Verification and campaign state

The canonical configured suite passes 439 tests. The local V25 additions also
passed direct compilation and production smoke checks. No mechanism-specific
prerequisite currently authorizes a long HPC3 job, so none was submitted.
