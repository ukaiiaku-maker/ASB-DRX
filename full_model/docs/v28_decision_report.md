# V28 discrete-symmetry, Nye, and ASB energy decision report

Date: 2026-09-15

## Decision summary

V28 advanced all three V27 failures locally without running HPC3. V27 evidence
is frozen by SHA-256 in `v28_frozen_v27_evidence.json`. The new decisions are:

| workstream | classification | fixture | scientific gate |
|---|---|---:|---:|
| SIBM | `SIBM_FRONT_FIRST_PASSAGE_ASYMMETRY` | pass | fail |
| authoritative Nye | `DUAL_NYE_STATE_STILL_INCONSISTENT` | pass | fail |
| ASB energy | `ASB_ENERGY_GAUGE_OR_DISSIPATION_FAILURE` | pass | fail |

No 192-square SIBM run, wall cone, long wall run, ASB refinement/rate bracket,
or grain allocation is authorized.

## A. Projection-free SIBM

Two hidden state-construction inconsistencies were repaired. The old
initializer relaxed a tangent-projected phase equation even though production
used an independent clip-and-normalize map. It also copied the already mixed
defect field into both sparse material slots, erasing the declared parent/child
contrast. V28 now solves the actual production map with temporary phase-volume
gauge fixing and initializes manufactured parent and child states explicitly.
The V27 equal-state projection remains available only as a reproduction switch
and was disabled in all V28 qualification runs.

At 64 square, with front processing disabled:

- exact equal displacement is `-1.05e-5` interface widths over 30 steps;
- virgin swept volume and projection activations are exactly zero;
- all five perturbation pairs from `1e-10` through `1e-2` have opposite
  pressure and velocity signs;
- maximum velocity oddness residual is `5.16e-4` and maximum pressure oddness
  residual is `3.31e-7`;
- favorable, reversed, mobility-off, and complete label-exchange signed contour
  responses are respectively positive, negative, zero, and negative;
- the discrete registry potential has only `2.46e-9` relative amplitude and a
  stationary half-cell offset.

Thus the discrete phase residual and declared-state ownership pass the local
projection-free symmetry tests.

When irreversible front processing is enabled, the favorable, reversed,
mobility-off, and label-exchange signs remain correct, but the equal branch
amplifies machine-scale contour/profile motion into 0.675 interface widths and
`9.96e-22 m3` of processed volume in 30 steps. The same instability is present
in S4, so the first biased suboperator is no longer S5 constitutive storage. It
is the contour/first-passage handoff followed by irreversible transfer,
storage, and recovery. Increasing a roundoff cutoff merely changes the onset
and would create an impermissible finite dead zone; it was rejected.

The projection-free phase operator is locally qualified, but the complete
stored-energy SIBM mechanism is not.

## B. Authoritative Nye state

The V27 wall fixture independently initialized uniform signed reservoirs, which
is inadmissible in a periodic domain. V28 balances the initial plus/minus mobile
and forest populations. This reduces the initial dual-Nye relative RMS to
`3.94e-12`; the periodic curl zero mode remains below `1.1e-27` relative at the
first accepted step.

The accepted dynamics immediately destroy the identity:

| accepted step | total relative RMS | zero-mode part | nonzero part |
|---:|---:|---:|---:|
| 1 | `3.94e-12` | `8.05e-15` | `3.94e-12` |
| 2 | 1.413 | `3.67e-16` | 1.413 |
| 3 | 1.432 | `1.91e-16` | 1.432 |
| 100 | 1.212 | `5.24e-17` | 1.212 |

The mismatch is therefore not a remaining mean signed-density initialization
problem. It is a dynamically generated nonzero-mode inconsistency accompanied
by a line-continuity divergence of order `5.6e10 m-2`. The accepted transport
map still evolves reservoir alignment and plastic-distortion Nye as independent
authoritative states. Neither authoritative-state option is yet qualified, so
the Frank--Bilby cone was correctly not rerun.

## C. ASB energy conditioning and exact-time localization

All restart fields were linearly interpolated to the exact common time
`5.842589456e-6 s`. This removes the V27 2.2--5.3% checkpoint-time mismatch.
The 128-square maximum temperature excess is 229.2 K and the 192-square excess
is 291.1 K, a 21.28% relative difference. The exact-time 128 heterogeneous
minus homogeneous maximum-temperature difference is 215.2 K.

The defect energy was decomposed independently. At 128 square,
`F_comp_alpha` changes by `+1.35e23 J/m3` while `F_comp_GB` changes by
`-1.43e23 J/m3`. At 192 square those changes are `+1.02e23` and `-2.47e23
J/m3`. Individual internal terms exceed external work by approximately
`2.6e15` and `4.8e15`, respectively. The old `other_dissipation = plastic -
defect - heat` construction is explicitly rejected: no independently computed
nonnegative other-dissipation channel exists in the saved evidence.

Threshold-free localization also fails fine-grid convergence:

| exact-time observable | 128/192 difference |
|---|---:|
| temperature excess above T0 | 21.28% |
| inverse-participation effective fraction | 19.49% |
| entropy effective fraction | 20.14% |
| minor second moment | 34.21% |
| connected-band FWHM | 78.97% |

At the common time, the thermal diffusion length is 0.152 micrometers, smaller
than the 0.300 micrometer process-zone sigma and 0.316 micrometer phase width;
the cell sizes are 0.0781 and 0.0521 micrometers. Strong heterogeneous heating
is real, but neither the energy ledger nor localization geometry is physically
qualified.

## Verification and claim boundary

The canonical suite passes 462 tests. Source and evidence are committed
separately. The retained claims are:

- projection-free discrete phase symmetry and near-equal oddness pass locally;
- complete SIBM fails at the geometric first-passage/irreversible-processing
  handoff;
- periodic consistent Nye initialization passes, but the accepted step breaks
  the dual-Nye identity immediately;
- ASB heterogeneous heating remains present, while energy conditioning and
  threshold-free grid scaling fail.

The next admissible work is a sign-aware contour first-passage map with no
finite dead zone, a single accepted flux map updating both Nye representations,
and a unit/ownership derivation of the compatibility energies plus independent
dissipation channels. Longer simulations cannot resolve these failures.
