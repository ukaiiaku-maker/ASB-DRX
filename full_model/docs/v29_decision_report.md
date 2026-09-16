# V29 coupled-front, Mura-Nye, and conditioned-ASB decision report

Date: 2026-09-15

## Decision summary

V29 established three verified atomic building blocks but did not promote any
of them past the production hard gates.  No long resolved-grid case and no HPC3
job was launched.

| workstream | classification | fixture | scientific gate |
|---|---|---:|---:|
| signed front geometry | `SIGNED_FRONT_GEOMETRY_CONVENTION_QUALIFIED` | pass | pass |
| coupled front | `COUPLED_FRONT_EVENT_NOT_YET_INTEGRATED` | pass | fail |
| authoritative Nye | `DUAL_NYE_STATE_STILL_INCONSISTENT` | pass | fail |
| ASB energy | `ASB_DISSIPATION_NOT_INDEPENDENTLY_CLOSED` | pass | fail |

The overall decision is
`REFACTOR_PRODUCTION_FRONT_AND_CDD_AROUND_VERIFIED_ATOMIC_OPERATORS`.

## A. Signed geometry and coupled front

The authoritative geometric convention is
`phi = eta_receiver - eta_donor`, with positive swept area denoting receiver
growth.  The tracker interpolates the inverse tanh profile coordinate, matches
crossings by orientation and periodic minimum image, and never integrates a
diffuse phase-volume change as transformed material.

Manufactured translations at `+/-0.01`, `+/-0.1`, `+/-0.5`, and `+/-1` cell
width pass with maximum relative area error `1.56e-13`. Broadening and
narrowing over interface widths 0.45--7 cells produce exactly zero swept area.
Closed-cycle, active-window, label-exchange, reflection, and two-interface
controls pass.

A lineage-free bidirectional transaction kernel now constructs both `A -> B`
and `B -> A` conservative transfers before evaluating the rate.  Its bounded
EXP-floor rate pair satisfies detailed balance.  The equal-state velocity is
exactly zero, the favorable example is positive, the log detailed-balance
residual is `-5.55e-17`, line closure is exact, and defect-energy closure is
`2.94e-39 J` or smaller.

This kernel has not replaced the production phase-first afterburner.  The
scientific coupled-front gate therefore remains failed; the verified kernel is
not used to reinterpret V28 production trajectories.

## B. First violating Nye suboperator

The accepted V24 step now reports incremental reservoir-Nye and plastic-curl
changes after each suboperator.  The first non-roundoff violation is
`slip_orientation_kinematics`: at accepted step 2 it changes plastic-curl Nye
by `598.86 m-1` RMS while reservoir Nye is unchanged.  The following physical
advection/capture operator changes reservoir Nye independently by `467.96
m-1` RMS while plastic distortion is unchanged.  Ordering/topology reactions
remain at roundoff in this comparator.

A one-flux spectral Mura operator was implemented and verified.  It advances
`beta_dot = J` and `alpha_dot = -Curl(J)` atomically, passes signed-population
exchange covariance, and satisfies discrete `Div Curl = 0` at machine-relative
precision.  It is not yet spliced into the old scalar finite-volume transport,
because doing so by reconstructing or projecting alignment after the step
would violate the directive.  Production remains dual-path and the current
capacity cone remains unauthorized.

## C. ASB energy ownership

The compatibility coefficients have consistent units: `(J m)(m^-2)^2 =
J m^-3`.  Their enormous scale is therefore not a dimensional typo.  The
quadratic `A_alpha` and `A_GB` terms are now structurally classified as
augmented numerical constraints, separate from physical Helmholtz energy and
ineligible for conversion to heat.  The old full diagnostic sum is retained
for regression only.

An explicit seven-channel nonnegative dissipation type was added, and negative
channels are rejected.  The saved production trajectories, however, do not
contain independently computed rates for every required channel.  V28's
largest internal/external scale ratio remains `4.80e15`; a new first-law claim
cannot be formed by assigning the residual to “other dissipation.”  The ASB
gate remains `ASB_DISSIPATION_NOT_INDEPENDENTLY_CLOSED`, so no 128/192/256
continuation or rate grid is authorized.

## Verification and provenance

- frozen V28 hashes: `v29_frozen_v28_evidence.json`;
- scientific source: `d0e95f8`;
- local audit runner: `504e17e`;
- machine result: `v29_local_hard_gates.json`;
- canonical tests: 499 passed in 69.55 s;
- execution: local only; no HPC3 submission.

The next admissible implementation step is to replace the production SIBM
afterburner with the verified atomic bidirectional event and refactor signed
population transport so the same accepted Mura face flux owns population,
alignment, plastic distortion, slip/work, heat, and tensorial Nye increments.
Only then can the ASB compatibility audit be repeated against a consistent Nye
state and resolved-grid continuations reconsidered.
