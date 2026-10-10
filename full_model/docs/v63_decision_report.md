# V63 decision report: rate accuracy, late spatial response, and coupled DRX/ASB continuation

## Decision

V63 completes the unchanged-source n64 trajectory through 2.0 microseconds,
establishes fresh-state temporal accuracy at three dynamically distinct states,
and performs an exact current-state DRX energy replay.  The principal result is
a valid negative: the coupled model has conservative prepared-boundary DRX and
strong thermomechanical localization, but the late localization morphology is
not spatially converged and no registered one-microsecond strict ASB episode is
present.  Spontaneous grain birth remains absent.

The physical production kernels are unchanged from V62 commit
`1c0cb4ca0e8a9a30b0c1a320817a0b35b4d14499`.  Diagnostic source
`ad6171b87a0ec5e3ef588be2484934356514443b` adds copied-state eighth-step
temporal refinement and analysis only.  No constitutive parameter, interface
length, threshold, or applied migration pressure was retuned against the
outcome.

## Unchanged-source n64 completion

The checksum-bound n64 continuation advances V62 step 210 to step 320 with
832 cumulative accepted intervals, zero rejected events, and 0.1903% maximum
complete-energy closure.  At 2.0 microseconds and engineering shear 0.092 it
reaches 2.35254 GPa, 1034.07 K mean temperature, and 283.35 K temperature
range.  Grain volumes are 2.9112, 2.0239, 2.3926, and 5.0722 in units of
1e-21 cubic metres.  Thus grain 40 grows strongly while the historically
tracked initially-lowest-density grain 20 shrinks; the historical scoped DRX
Boolean cannot be generalized to “no boundary transformation.”

The late component can be followed continuously in the saved states from step
265 to 320, a sampled duration of 343.75 ns.  At step 320 its n48/n64 mask
Jaccard index is 0.668, but the fixed-axis transverse-profile error is 45.7%.
The n48 and n64 FWHM values are 1.016 and 0.742 micrometres (9.75 and 9.5
native cells), respectively.  Adequate native-cell sampling therefore does not
imply agreement in physical width.  The observed duration is also far below
the registered 1 microsecond persistence threshold.  V63 does not claim a
mesh-converged or strict ASB.

Phase-sensitive band-15 errors at step 320 are 8.24% for temperature, 61.51%
for exact reconstructed Nye content, and 73.66% for instantaneous plastic
power.  The unchanged EXP-floor substitution audit attributes 76.4% of the
finite rate discrepancy to raw resolved stress, 5.5% to resistance, and 0.5%
to temperature.  This is a nonlinear diagnostic attribution, not a production
rate replacement or a justification to retune the barrier.

## Fresh-state time accuracy

Three one-caller-interval n64 forks separate state-dependent splitting error
from the repaired front clock.  At the early step-185 state, the 3.125/1.5625
ns comparison gives 0.92% signed-density, 0.84% exact-Nye, 0.41% temperature,
and 0.22% plastic-power increment-relative errors.  Near the step-210 divergence, the same
comparison gives 2.69%, 2.46%, 0.57%, and 3.07%.  Both are below 5%.

At late step 275, 3.125 versus 1.5625 ns still gives 5.56% signed-density and
4.92% Nye errors.  Tightening to 1.5625 versus 0.78125 ns reduces the
corresponding increment-relative errors to 2.57% and 2.34%, with 0.75% temperature and
2.40% plastic-power errors.  Every requested mechanical and front clock
closes and no event is rejected.  The selected 6.25 ns production cadence is
therefore locally inaccurate in the late activated state even though the
tighter fork converges.  The completed n64 reference remains valid evidence
for its declared discretization, not a continuum-time morphology certificate.

## What drives the accepted DRX event

An exact replay of the physical n48 step-268 to step-269 front transaction
finds a cold Helmholtz decrease of 2.2966e-14 J.  Defect storage falls by
1.5641e-14 J and recoverable elastic energy by 7.4734e-15 J; boundary excess
rises by 1.4783e-16 J and phase terms are negligible.  The generated front
heat is 2.2966e-14 J and the complete internal-energy residual is about
3.1e-19 J.

Grain 40 gains 1.357e-23 cubic metres and 1.414e-6 metres of represented line
content during the coupled transaction, increasing its own assigned defect
storage by 6.787e-15 J.  Other donors lose more defect storage and all grains'
recoverable elastic shares fall.  Growth is consequently selected by the
constrained multigrain stored-energy and mechanical transaction, not by the
initial scalar defect ranking or by a synthetic pressure.  This qualifies
prepared-boundary redistribution; it does not qualify spontaneous nucleation.

## Rate-selected response

V63 also selects 80,000/s at fixed geometry, coefficients, and 900 K initial
temperature as a bounded response discriminator.  The n32 physical/frozen
pair reaches 1.0 microsecond with zero rejection and complete-energy closure
below 0.613%.  The physical member is slightly stronger, has lower temperature
contrast and lower power concentration, while being hotter on average and
showing slightly more grain-40 growth.  Freezing the declared Arrhenius
arguments therefore does not suppress a persistent ASB at this grid; the
causal result is negative rather than the expected simple thermal-runaway
ordering.

The matched n48 intervention certificate passes, with zero rejected events in
either member and maximum energy closure of 0.374% physical and 0.429% frozen.
At 1.0 microsecond the physical member is stronger (2.946 versus 2.803 GPa)
and hotter on average (1006.03 versus 1003.62 K), but has lower temperature
range (255.79 versus 260.45 K).  Its maximum matched component excess is only
16.21 K against the registered 50 K threshold.  No strict snapshot or episode
exists.  This is an additional fixed-parameter causal check, not a material
calibration or a rate/temperature regime map.

The coarse n32 pair was additionally continued, without parameter changes,
through 2.0 microseconds.  Its physical member reaches 1.248 GPa, 1128.29 K
mean temperature, and 480.86 K temperature range; the frozen member reaches
1.331 GPa, 1114.15 K, and 411.46 K.  Both trajectories have zero rejected
events, and their maximum complete-energy closures are 1.884% and 1.350%.
The matched intervention remains valid and the physical component eventually
exceeds the control by 67.17 K, while stress softening reaches 70.91%.
Nevertheless, no conjunctive snapshot and no one-microsecond same-component
episode is present.  This longer coarse-grid result is therefore an
informative negative, not an ASB qualification, especially given the failed
n32/n48 spatial comparison.

The physical n32/n48 high-rate endpoint is itself not spatially converged.
On common Fourier band 15, phase-sensitive errors are 16.65% for temperature,
81.81% for exact reconstructed Nye, and 82.23% for instantaneous power.  The
n48 causal comparison can therefore diagnose the Arrhenius intervention at
that discretization, but it cannot establish a grid-independent ASB regime.

## Claim boundary and next decision

V63 establishes a coupled conservative production trajectory, resolved
current-state DRX energetics, state-local temporal convergence at a tighter
late cadence, and a high-rate causal experiment.  It does not establish a
spatially converged persistent ASB, spontaneous grain birth, a universal
time-step certificate, or a calibrated material response.

The next model decision should address the work-conjugate spatial stress/rate
pathway and late caller splitting together.  A candidate nonlocal closure is
justified only if it is introduced as a separate energy-consistent physical
hypothesis and compared with the unchanged EXP-floor reference.  Simply
adding a smoothing length, weakening the barrier, or changing the registered
width and persistence criteria would not resolve the evidence exposed here.
