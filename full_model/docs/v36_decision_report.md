# V36 channel-resolved kinetics and physical response decision

## Current outcome

V36 removes the V35 saturated-channel cancellation from the production front
path without changing the recurrent common state, complete energy gate, or
physical event geometry.  The frozen V35 result remains exactly reproducible,
but its interpretation is narrowed to
`ZERO_DRIFT_UNDER_SATURATED_DIRECTIONAL_CLOSURE_PHYSICAL_ARREST_UNRESOLVED`.
It is not evidence of physical equilibrium or pinning.

The selected V36 closure gives a finite, sign-correct stored-energy response
at zero applied front work.  The response reverses when the material contrast
is exchanged, the equal complete-state fixture is stationary, and the
nonphysical geometry probe has no rate, work, or directional effect.  The
current generic parameters are a physical-response hypothesis, not a material
calibration or a DRX prediction.

The recurrent response is timestep-qualified locally, and both its initial
rate and matched 0.05 ms trajectory are spatially qualified at n128/n192.
Their cumulative sweeps differ by 3.72%, below the unchanged provisional 5%
threshold. Lower grids are retained as fixtures and convergence evidence only.

## Production kinetic object

For each actual outgoing channel `r` from the complete state `z`, production
now records the complete endpoint and evaluates

```
P_r       = |Delta F_r| / V_event
Delta H*r = H0 [f + (1-f) exp(-a (P_r/Pc)^n)]
Delta G*r = Delta H*r - kB T (Delta S*r/kB)
nu_r      = nu0 exp[-max(Delta G*r,0)/(kB T)]
A_r       = exp[-max(Delta F_r,0)/(kB T)]
j_r       = q_r nu_r A_r
v_n       = b (j_plus - j_minus)
```

The negative-barrier branch uses the declared drag policy.  `q_r` is physical
availability, `A_r` is uphill acceptance, and `nu_r` is the transition-state
rate; none is silently folded into another.  The activation entropy appears
once.  Attempt frequency and constant activation entropy remain identifiable
only through their product unless separately constrained.

Opposite geometric directions are not labeled reverse edges when both
transactions irreversibly process line.  Local detailed balance is asserted
only for a represented true reverse edge.  For the frozen V35 discriminator,
both nonreverse endpoints are downhill, so both acceptances equal one, but
their distinct endpoint magnitudes produce rates of 2,491,670.7255 and
2,491,419.6211 per second and a net velocity of `+6.22739e-8 m/s`.  The old
shared-base comparator remains exactly zero.

This is a deterministic net-front coarse graining.  Gross channel activity is
reported, but cancelled hypothetical turnover is not used to update internal
state.  Line/moment transfer, reaction heat, and exports are committed only
through the accepted atomic net transaction.  Stationary material recovery is
handled by the explicit Mura/reaction operators rather than inferred from
cancelled front events.

## Compact front discriminators

All hard discriminator checks in `v36_front_channel_kinetics.json` pass:

- external front work and probe work are exactly zero;
- actual start/end energies and reaction increments are recorded;
- equal complete states are stationary;
- exchanged endpoints reverse the velocity to numerical odd symmetry;
- `+0.1` and `-0.9 GPa` geometry probes give bitwise-identical physical
  decisions and state hashes;
- 0.025/0.05/0.10-cell proposals preserve the physical direction;
- true reversible edges retain exact detailed balance;
- the accepted transaction closes first law to `-1.32e-27 J` and records
  `1.41712e-19 J` generated heat and `2.85326e-21 J` sink export.

## Recurrent physical response

Two protocols are kept separate.  `hold` fixes total strain and applies zero
front work; `continued_deformation` advances the mean shear at the declared
rate.  Mura and front operators must cover the same accepted physical
interval.  Requested steps beyond the evolving Mura clock limit are rejected,
not silently assigned different clocks.

Local controls show:

- n16 timestep refinement from `1e-5` to `5e-6 s` changes the 0.2 ms sweep by
  1.07%; n16 is nevertheless not spatially resolved;
- the three bounded proposal amplitudes span 2.19% in accepted sweep;
- n16 to n32 and n32 to n64 do not converge and are not promoted;
- the n128/n192 initial velocity differs by 3.79%, below the provisional 5%
  threshold;
- the matched n128/n192 0.05 ms cumulative sweeps are `1.06925e-26` and
  `1.02952e-26 m3`, a 3.72% difference;
- exchanging the line-density contrast reverses the signed sweep;
- the front-disabled control has zero sweep;
- fixed-strain and continued-deformation responses diverge as their common
  state evolves, rather than from an imposed front pressure.

The finite response is stored-energy driven and occurs without applied front
work.  It is not yet a DRX demonstration: this is an existing bicrystal,
nucleation and grain-label allocation remain disabled, and no independent
grain-recognition event is claimed.

## Mura rate limit

The feasible-extent implementation retains the exact step-2971 accepted
scales `[1,1,0,0.0625]` and joint affinity while reducing trial evaluations
from 48 to 16.  The two-mode audit improves from 27.21 to 13.25 seconds
(2.05x).  Extent levels 8/12/16 are bit-exact at matched time.  Refining
`1e-10` to `5e-11 s` reduces the reported density, beta/slip, Nye, and
temperature differences while preserving positivity, realizability, energy,
and nonnegative heat.  This is a bounded rate-limit result, not long-time wall
pattern convergence.

## Thermal causality

Only exact, attributable matched comparisons are numeric.  Invalid legacy
selective cases and unmatched endpoints are null.

The corrected frozen-flow intervention at step 2500 establishes strong
temperature-flow feedback: relative to the full law, stress rises 267.66 MPa
and maximum temperature falls 528.06 K, while mean temperature and total
deposited heat increase.  This indicates redistribution and broader
participation, not simply less heat.

The exact-temperature thermostat is now also matched at 8.3367 microseconds
and nominal strain 0.2501.  It exports `1.0593932949e9 J/m3`, equal to
deposited heat within roundoff; thermal change is zero.  Relative to full law,
maximum temperature falls 909.16 K, stress rises 300.934 MPa, and the
participation measure broadens.  First law and Burgers/line/energy invariants
remain within their fixed tolerances.

The corrected frozen-recovery intervention is exactly matched at the same
step, time, and strain.  Relative to full law, maximum temperature falls
90.741 K, stress rises 18.973 MPa, active fraction rises 0.057223, softening
falls 0.006136, and effective width rises `2.084e-8 m`.  Its flow input evolves
to 1166.84 K while its recovery input remains at 900 K, confirming selective
accepted-channel routing.  Endpoint first-law, Burgers, line, and energy
residuals pass their fixed limits.

The finite-conduction/no-bath case is exactly matched and terminal.  Relative
to local adiabatic full law, maximum temperature falls 604.433 K, stress rises
25.261 MPa, active fraction rises 0.536001, softening falls 0.008175, and
effective width rises `3.690e-8 m`.  Its first-law residual is `2.52e-12`; its
maximum Burgers, line, and energy residuals are below `8.0e-17`, `4.8e-17`,
and `1.5e-18`, respectively.  The selected production matrix is therefore
`V36_THERMAL_VALID_MATRIX_COMPLETE_WITH_QUARANTINED_LEGACY`: all selected
comparisons are exact and valid, while the two old misrouted selective rows
remain explicitly invalid and are replaced only in the production selection
by their corrected reruns.  The result distinguishes localized adiabatic
heating from broad conductive heating; no strict-ASB claim is made.

## Decision boundaries

1. V35 zero drift is kinetic cancellation under a frozen shared closure.
2. Gross channel activity is not represented as stochastic stationary
   recovery; explicit reaction operators own that evolution.
3. A complete-energy rejection after transient motion is a candidate physical
   force/capacity arrest only after timestep and grid qualification.
4. The V36 compact and recurrent calculations demonstrate zero-work
   stored-energy-driven migration in the generic bicrystal.
5. The qualified 0.05 ms trajectory remains a bounded exposure, not evidence
   of one-interface-width migration or an independently stable long-time
   arrest.
6. No hard invariant has been relaxed or tuned around.
7. Thermal interventions distinguish broad heating/localization measures from
   strict ASB, which remains unqualified.
8. The hold protocol is static recovery/migration; continued deformation is a
   coupled response but not yet independently recognized DRX.

## Validation and provenance

The final merged V36 branch passed 706 tests in 279.88 seconds.  The core
kinetic commit independently passed 696 tests; the front evidence tests passed
13; the merged Mura tests passed 12; thermal postprocessing/routing tests
passed 9; and recurrent restart tests pass.  The selected HPC production source is
`890cb8906a9772d8bd5c5eb43164ecd44ad2720f`; n192 Slurm job `56099919`
completed in 8m41s, was fetched with verified checksums, and agrees with the
local n128 trajectory. Live and final identities, hashes, and terminal states
are recorded in `v36_case_manifest.json`.
