# V39 stiff-ordering and coupled-horizon decision report

## Evidence boundary

V39 continues the V38 checkpoint `5a4b992` on
`exp/full-v34-recovery-v1`. The production stiff-ordering implementation is
`53cc80a`; the first coupled, Mura, thermal-recovery, and interface evidence
checkpoint is `90bdd45`. V37 frozen-source results remain attributed to their
original source and do not inherit V39 numerical accuracy. All evidence named
below is retained in `full_model/verification`, not reconstructed from prose.

## Stiff ordering and the former post-front terminal

The production ordering channel now dispatches low Arrhenius exposures to the
complete-time resolved oracle and stiff exposures to a bounded convex
asymptotic solve of the existing rate/energy basin. The sparse production case
uses sign- and family-resolved active-set obstacle solves. The general sparse
Hessian, ADMM, and nonlinear fallbacks remain available. No kinetic barrier,
entropy, attempt frequency, or physical ordering rate was reduced to obtain
the result.

The stiff endpoint conserves tangle plus ordered line separately by sign and
family, moves signed alignment with the same transfer, is nonnegative and
energy descending, and consumes the complete requested duration. Its ledger
records method, endpoint inventory-change bound, attempts, elapsed time, and
zero discarded reaction time. Near the old explicit limit, the selected
method dispatches exactly to the resolved reference.

The exact V38 post-front failure was reproduced as a typed
`POST_MURA_PENDING` partial checkpoint. Resuming it consumed the remaining
0.5 ms, reached an external clock of 1 ms, and did not repeat the committed
front transaction. The ordering remainder bound was `3.14e-10` relative and
discarded time was zero. Continuous and partial-stage-restarted endpoints are
byte-identical over their shared fields. This closes the ordering-stiffness
terminal rather than relabeling it as a physical arrest.

## Common-state refinement and continuation decision

The production map remains A(H/2), B(H), A(H/2), where A owns Mura, thermal,
and reaction evolution and B owns one complete front transaction. Operator
exposure and the external clock are recorded separately. The complete n16
1 ms comparison gives contour displacement `3.23349e-10` m at H = 62.5 us and
`3.27374e-10` m at H = 31.25 us, a 1.23% difference; the selected 62.5 us
macro interval therefore passes the provisional 5% temporal criterion.

Spatial promotion does not pass. At the same first 62.5 us, n64 and n128 give
front displacements `1.57644e-10` and `1.40799e-10` m, a 12.0% difference.
The ordered-line reservoir is also near extinction and changes from
`1.019998e-4` to `1.244879e-5` m per metre thickness. Stress, slip, beta,
temperature, and total line are much closer, but those agreements cannot
override the unresolved front and ordered-reservoir observables.

Consequently the planned 30 ms *coupled* calculation was not launched. The
precise residual blocker is grid dependence of the production capture/
ordering surface measure and front displacement, not ordering time
integration. The executable next action is a surface-measure-consistent audit
and repair of the ordered-gradient/capture and diffuse-front discretization,
followed by the existing n64/n128 62.5 us overlap. No kinetic retuning or
drift subtraction is authorized by this result. This is a numerical
qualification failure, not a material or physical no-go.

## Interface residual

A compact audit translated the same 3.2 um periodic slab at four subcell
offsets with fixed 0.40 um physical interface width. The relative phase-energy
range decreases from 0.2858% at n32 to 0.1289% at n64 and 0.0624% at n128.
However, the central translation derivative remains about
`3.47e-11 J/m` at n128, and the slab-growth derivative remains about
`8.48e-11 J/m`. A five-percent profile-width perturbation lowers the energy
on the wider side, showing that the initialized profile is not a stationary
discrete profile. The small equal-owner motion therefore remains a declared
diffuse-profile/representation uncertainty. No equal-state projection,
fitted drift subtraction, artificial pressure, or frozen-profile workaround
was applied.

## One-grain Mura evidence

The latest checksum-verified rescue from frozen V37 source contains a valid
checkpoint at step 13212 and strain 0.04737725. It has a 4.54072 degree
orientation span, `3.026997 m/m` ordered line, and `2971.673 m/m` tangle line;
its hard invariants pass. Thirteen history records ahead of the accepted
checkpoint were excluded. This is a valid active-run partial showing a
persistent legacy-source orientation gradient, not a current-source LAGB or
DRX result.

At the same accepted state and `5.22593e-11 s` interval, legacy and current
ordering continuations preserve the same total line, Nye RMS, and orientation
span, but give `2.75645` and `2.58358 m/m` ordered line respectively, a 6.69%
difference relative to current. The rescued ordered observable therefore does
not inherit current V39 accuracy. A current-source restart from it remains a
declared initial-value problem, not proof that current source generated the
precursor.

## Thermal/localization evidence

The repaired V37 exact-source conduction rerun retained and checksum-verified
all six cases. Every case is a valid finite-conduction response-family member,
but every one is classified `BROAD_OR_NONPERSISTENT_HEATING`; strict ASB is
false. At 0.2501 strain, the selected broad negative `rate_low` has peak and
mean rises of 268.876 and 259.001 K, `heterogeneity_short` has 297.523 and
267.752 K, and the strongest concentration `heterogeneity_long` has 325.723
and 267.470 K. Their effective heated widths remain broad, about 9.9 um.

Those frozen-source checkpoints predate the saved plastic-power and actual
irreversible-heat fields. The absence is recorded as `NOT_SAVED_BY_SOURCE_CHECKPOINT`;
absolute shear rate is not substituted as a proxy. The compact current-source
n64 transfer comparison reaches 0.0501 strain in all three selected cases.
`rate_low`, `heterogeneity_short`, and `heterogeneity_long` have peak/mean
temperature rises of 40.343/36.463, 42.743/36.850, and 43.842/36.567 K,
respectively. All remain `BROAD_OR_NONPERSISTENT_HEATING`. Each endpoint
contains independently saved work-conjugate plastic power, actual irreversible
heat production, and a complete per-channel energy ledger. This supports
transfer of the broad early thermal response to current source; its shorter
horizon does not replace or extend the frozen-source 0.2501-strain family.

## Decisions and claim boundary

- Stiff ordering integration and exact post-front recovery: **passed**.
- n16 common-state temporal refinement at H = 62.5 us: **passed**.
- n64/n128 common-state spatial promotion: **failed numerical qualification**.
- Coupled 30 ms continuation: **not launched because its prerequisite failed**.
- Interface residual: **profile/representation uncertainty remains**.
- Frozen-source Mura result: **valid active partial; ordered observable is legacy**.
- Frozen-source thermal family: **valid broad-heating negative; no strict ASB**.
- Selected current-source thermal transfer: **complete at 0.0501 strain;
  field-ledgered and broad in all three cases**.
- DRX, a current-source persistent LAGB, calibrated material behavior, and
  strict ASB are **not claimed**.

The production regression boundary passes 732 tests in 197.51 seconds.
