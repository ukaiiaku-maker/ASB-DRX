# V41 direction-energy, gradient-transfer, and physical-response decision

## Executive decision

V41 replaces the production front's subtraction of two independent outgoing
channel activities with a deterministic complete-dissipation law.  At the
archived first-sign-change state the repaired law admits the available
downhill advance and has no rate/energy conflict.  At the archived 1 ms state,
both signed complete trials are uphill over four decreasing extents, so the
later zero velocity is supported by the represented complete energy rather
than by opposing rate and publication gates.  No fitted mobility, external
pressure, drift subtraction, or removal of the complete-energy guard was used.

The apparent post-front Nye-transfer discrepancy is also resolved at its
declared ownership boundary.  It was the separately stored support-gradient
product-rule term, not missing Burgers content.  Adding that already-owned
term to the diagnostic closes the post-front identity to about 7.1e-6 of the
curl norm on both n128 and n192.  The second Mura half-stage remains above the
provisional strong-norm grid criterion (7.31%); only its common physical modes
pass (4.30%).  Wall-scale spatial accuracy is therefore not promoted.

The one-grain compact continuation and matched thermal control are reported
below at their actually attained endpoints.  V41 does not claim DRX, a
persistent LAGB, strict ASB, or material calibration.

## Production implementation

For each distinct directional transaction, production constructs its complete
endpoint and evaluates the available change

\[
\Delta\mathcal A_d=F(q_d)-F(q_0)-W_d+E_{\mathrm{sink},d}.
\]

The retained mechanism-specific EXP-floor coefficient is multiplied by

\[
\max\{1-\exp(\Delta\mathcal A_d/k_BT),0\}.
\]

Consequently, a nonzero directional activity owns a nonpositive complete
affinity.  The two outgoing endpoints are not mislabeled as a microscopic
reverse pair when their reaction/storage transactions differ.  The legacy
independent-Metropolis traffic difference remains selectable for exact
comparison; `complete_dissipation` is the I3 and monolithic production default.
The finite complete-candidate energy guard remains mandatory.

The declared reconstructed plastic distortion and Nye split are

\[
\bar\beta^p=\sum_a w_a\beta_a^p,\qquad
-\operatorname{Curl}\bar\beta^p=
\sum_a w_a(-\operatorname{Curl}\beta_a^p)
+\sum_a[-\nabla w_a\times\beta_a^p]+\epsilon_h.
\]

`CommonFrontState.interface_nye_m1` owns the second term.  V41 changes the
diagnostic identity, not the authoritative field, and applies no filtering,
target-Nye reconstruction, or projection.

## Directional audit and continuation

Both signed complete candidates were constructed from immutable copies of the
first-sign-change and stationary checkpoints at proposal fractions 1/16,
1/32, 1/64, and 1/128.  At first sign change, the legacy rule has four admitted
uphill conflicts; the selected rule has zero.  At the smallest extent the
forward proposal has an A-to-B event change of -3.526e-24 J and publishes,
whereas its B-to-A channel is +4.551e-24 J.  At 1 ms, both complete directional
endpoints are positive at every tested extent; at the smallest extent the
forward/reverse A-to-B endpoint changes are +2.082e-23/+2.156e-23 J.

The repaired zero-work continuation result is recorded in
`v41_front_continuation_decision.json`.  It distinguishes additional published
motion from the later complete-energy stationary tail and reports the actual
physical horizon reached.  Starting after the archived interval 15 state, it
published six additional intervals and advanced 7.107e-12 m.  Its last
publication ended at 0.265625 ms, followed by 47 stationary intervals through
the completed 1 ms horizon.  All common clocks close.  At the terminal state,
the A-to-B and B-to-A channel endpoint changes are +2.086e-23 and +2.799e-23 J.

The V40 interpretation is superseded by
`NO_PUBLICATION_UNDER_LEGACY_RATE_AND_COMPLETE_ENERGY_GATES;
LEGACY_PHYSICAL_PINNING_UNRESOLVED`.  The V41 selected-law interpretation is
`COMPLETE_DISSIPATION_REPAIR_PASSED; PHYSICAL_STATIONARITY_SUPPORTED`.

## Gradient-transfer and ordering accuracy

Matched one-current-source n128/n192 stage replays each accepted one front
interval and closed the common clocks.  The post-front declared-identity
residual fractions are 7.076e-6 and 7.172e-6.  Post-front curl common modes
differ by 0.0275%.  After the second Mura half-stage the common-mode curl
difference is 4.30%, but the strong RMS difference is 7.314%; the strong
spatial criterion therefore fails.

The trace ordered-line n128/n192 scaling exponent is 3.019.  The inventory is
already present after the first Mura half-stage, before the front, which rules
out front transfer as its cause.  It remains a near-extinction ordering or
tolerance accuracy issue and is not promoted as a wall.

The nominal coupled legacy/current 1 microsecond comparison actually accepted
37.616 ps.  The isolated legacy solve completes but increases the unforced
energy by 7.2523e-4 J/m and is not an accurate thermodynamic comparator.  The
current complete-time solve lowers it by 7.0904e-6 J/m.  No 1 microsecond
legacy/current overlap claim is made.

## One-grain organization

The protected compact continuation is classified from its checksum-verified
retained endpoint in `v41_one_grain_decision.json`.  Relative Frank--Bilby
closure is evaluated only if a geometrically qualified, angle-bearing
candidate exists; otherwise the classification is
`NO_QUALIFIED_BOUNDARY_FOR_RELATIVE_CLOSURE`.  No grain or phase label is
allocated by this branch.

The bounded retry reaches exactly 5.000% strain after 22,595 accepted
intervals.  The retained state has a 9.779 degree global orientation span but
only 0.113 degree between its fifth and ninety-fifth percentiles.  Its selected
wall support has a 0.045 micrometre participation-equivalent width, below the
0.156 micrometre cell size.  The independent section has only 0.0282 degree
between plateaus, reports no candidate wall, and therefore does not admit a
relative Frank--Bilby closure claim.  The large global outlier is not promoted
to a LAGB.  Hard balances pass, normalized line continuity is 9.02e-15, and no
phase or grain state is present.

An interim compact audit exposed a provenance defect in the analysis path,
not in the retained state: the atomically named checkpoint was newer than the
copied `status.json`, and the copied history could contain records written
after that checkpoint.  V41 now treats checkpoint metadata as authoritative,
truncates history at the retained step and strain, and labels a nonmatching
status explicitly.  A regression test prevents later-than-checkpoint history
from entering persistence or invariant decisions.

The exact disabling comparison then advances the retained 5% state to 5.01%
with the explicit topology route off and on.  The topology-off control remains
hard-valid and has no qualified boundary.  Enabling the route drives the local
ordered fraction to about 0.9999, but it also produces an authoritative
source-offset residual of 0.6887 and normalized line-continuity residual of
0.6595, both far above the declared 5% tolerance.  It remains without an
independent boundary candidate and is quarantined as
`TOPOLOGY_ROUTE_HARD_INVALID`, rather than being used as an apparent ordering
success or retuned.

## Matched thermal response

The V40 strongest finite-conduction case is retained as the baseline.  V41
runs the same geometry and transport with flow rates evaluated at the initial
temperature while heat and recovery temperature still evolve.  The result and
trajectory first-law ledger are recorded in
`v41_thermal_response_decision.json`.  This is a causal flow-temperature
feedback test, not a zero-conduction or changed-source-geometry comparison.
Both trajectories reach 0.2501 nominal strain and pass their hard ledgers.  The
full-feedback endpoint has 58.253 K peak-minus-mean temperature and 0.90178
plastic-power participation; the frozen-flow control has 18.159 K and 0.98361.
Thus freezing flow temperature lowers the peak by 29.430 K and the temperature
contrast by 40.094 K while making plastic power substantially broader.  The
control first-law residual is 1.40e-11, all channels are available, and no
numerical constraint enters physical energy or drives state.
Strict ASB requires localized connected structure, persistence, and refinement;
it is not inferred from peak-minus-mean temperature alone.

## Verification and claim boundary

Machine-readable controller and case records separate solver completion,
invariants, temporal/spatial accuracy, constitutive interpretation, and
scientific qualification.  The final merged regression result is recorded in
`v41_campaign_controller.json`.

The front constitutive repair and declared gradient ownership pass their
operator-level decisions.  Strong wall-scale refinement remains unresolved.
The bounded response calculations do not establish DRX, a persistent LAGB,
strict ASB, or calibration to a material class; their absence does not reject
the full campaign premise.
