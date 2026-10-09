# V62 decision report: repaired front clock, fresh n48 causal pair, and repaired-clock spatial audit

## Decision

V62 completes the requested numerical repair, clean n48 causal experiment, and
a fresh repaired-clock n32/n48 physical comparison through 2.0 microseconds.
It does **not** qualify strict ASB or spontaneous grain birth. The repaired
production model develops large post-peak softening and short, causally shifted
heat/power localization episodes, but no state satisfies the registered strict
conjunction. Prepared-boundary DRX remains active, conservative, and
source-attributable, but neither fresh member reaches the existing endpoint
threshold for substantial boundary-driven DRX by 2.0 microseconds.

The numerical source is immutable commit
`1c0cb4ca0e8a9a30b0c1a320817a0b35b4d14499`. Both fresh trajectories start
from the same analytic four-grain state and use the same 5 micrometre domain,
312.5 nm interface scale, 6.25 ns caller cadence, 40,000/s loading, local
interface temperature routing, local realized-event heat deposition, and
physical heat equation. The control freezes only the declared flow, recovery,
and front Arrhenius temperature arguments at 900 K.

## Step-68 obstruction and numerical repair

The original step-68 calculation was expensive for two independent reasons.
Each constitutive residual redundantly rebuilt state-invariant rotated fields,
wall derivatives, and owner frames, while positivity forced 1,132 residual
calls over 283 attempted common substeps. Exact caching reduced copied-state
wall time from 183.890 to 123.238 seconds (1.492x) with every checkpoint array,
decision, and ledger bitwise identical.

The more important temporal defect was the front clock. The old operator
clipped every requested contour fraction at 0.015, so changing caller cadence
changed the number of capped physical events. At n48, a 25 ns full/two-half
comparison differed by 23.50% in swept-volume increment and 23.01% in front
heat; quarter/eighth refinement worsened the swept discrepancy to 43.65%.
V62 replaces clipping with atomic rollback and predictive adaptive subdivision
until every accepted event is below the unchanged contour-CFL bound. The
physical force, rate, material transfer, energy, and heat laws are unchanged.

At the selected 6.25 ns cadence, the copied n48 two-half/four-quarter audit has
2.095% maximum signal-relative L2 error, 0.283% swept-increment error, 0.489%
front-heat error, and 1.407% stress error. Increment-relative signed-density and
Nye errors remain 25.53% and 10.42%. This qualifies the tested interval only;
V62 does not claim complete-trajectory temporal convergence.

## Cross-grid interpretation

The apparent initial 33% n32/n48 support-gradient Nye discrepancy was a raw
unnormalized L2 scaling artifact. Physical RMS values at initialization agree
to 2.8e-6 relative. The old histories first separate by more than 5% in
temperature contrast at 1.1 microseconds and normalized support-gradient Nye
at 1.275 microseconds, but those histories use the rejected clipped clock.
Their divergence therefore cannot be assigned uniquely to spatial resolution.
That diagnosis was subsequently tested with a fresh n32 trajectory using the
same repaired source, analytic origin, dimensions, coefficients, and 6.25 ns
cadence as the fresh n48 member. At 1.25 microseconds its mean-temperature and
temperature-range differences are 0.22% and 0.80%; at 1.53125 microseconds
they are 0.04% and 0.98%. The former large bulk divergence therefore does not
reappear after the clock repair.

At the common 2.0 microsecond endpoint, post-front stress differs by 3.91%,
mean temperature by 0.070%, peak-minus-mean temperature by 1.68%, temperature
range by 1.36%, and power participation by 4.53%. Fresh sweep, revisit, and
largest-grain growth differ by 1.72%, 0.72%, and 2.84%. Those bulk response
and prepared-boundary observables pass the provisional 5% screen. Spatial
content and morphology do not: mean Nye differs by 13.43%, power width by
54.34%, aspect ratio by 58.05%, and endpoint heat/power overlap is zero at n32
versus 0.0714 at n48. An explicitly unmatched localization screen begins at
1.56875 and 1.58125 microseconds, only two caller intervals apart, but this is
not a causal ASB criterion.

No spatial refinement certificate is issued. The physical-only pair lacks a
repaired-clock n32 Arrhenius control and a third grid, and the morphology fails
the registered comparison. An n64 run was also not launched: only 5.4 GiB of
disk remained while the authoritative n32 history was still growing, so a
speculative finer run would have endangered checkpoint and delivery integrity.

A common-physical-mode Fourier audit further localizes the discrepancy. At
1.0 microsecond, common-band total-variation distances are 0.15% for
temperature, 0.26% for instantaneous plastic power, 0.95% for preceding front
heat, and 2.48% for Nye magnitude. At 1.65 microseconds, temperature and front
heat remain at 4.52% and 4.17%, while Nye and power have separated to 12.49%
and 19.58%. At 2.0 microseconds, front heat is still 2.34%, but Nye is 34.17%
and power is 16.34%; the n32 fractions in common radial modes 8--15 are 0.432
for Nye and 0.421 for power, versus 0.117 and 0.314 at n48. This evidence
places the first unresolved length-selection problem in signed-content and
instantaneous mechanical organization rather than in the repaired front-heat
field or thermal diffusion. It is a diagnostic localization, not proof of one
unique offending term.

The saved-state operator ordering sharpens that statement. A family-resolved
Nye spectral warning first exceeds 5% at 1.0625 microseconds, but this is a
near-coarse-Nyquist derivative signal: its modes 1--7 differ by only 2.60%,
whereas n32 carries 7.59% of Nye power in modes 8--15 versus 0.54% at n48.
The support-gradient and exact reconstructed Nye tensors remain within 0.52%
and 0.62% at that time; the high-mode discrepancy is in the support-weighted
owner curl/reservoir representation. Owner-curl and owner-reservoir tensors
remain mutually consistent to roundoff, so this is not a second Nye update.

More importantly, Nye is not the rate-driving state in this implementation.
The first causal response separation occurs at 1.15625 microseconds. There,
support spectra differ by 0.019%, common-stress spectra by 1.57%, and weighted
owner effective-stress spectra by only 0.042%, but the same EXP-floor mapping
amplifies these differences to 6.89% in weighted owner speed, 6.87% in slip
rate, and 9.53% in plastic power. Its independently evaluated local logarithmic
sensitivity is essentially grid-independent: weighted medians are 20.83 and
20.85 and 90th percentiles are 21.56 on both grids. Signed mobile polarization
then separates at 1.25 microseconds, chemical backstress at 1.28125
microseconds, accumulated family slip and plastic distortion at 1.3125
microseconds, effective stress at 1.53125 microseconds, and temperature at
1.6875 microseconds. Total line density, Taylor resistance, supports, and
orientation never fail the 5% spectral screen.

Thus the remaining causal problem is a steep but consistently evaluated
owner-level stress-to-rate map acting on a spatial stress field without a
qualified rate-regularizing length. The next repair must resolve that stress
field or add a declared thermodynamically conjugate nonlocal closure. Retuning
the EXP-floor barrier, filtering a plotted result, or changing the ASB width
criterion would hide rather than repair the mechanism.

## Repaired physical continuations

The transitioned V61 n48 state was advanced from its verified 1.675
microsecond checkpoint to 2.0 microseconds with zero rejection. It reaches
2.084 GPa, 1036.43 K mean temperature, a 276.64 K range, and 48.72% segment
softening. It is valid evidence for the repaired continuation, but it is not a
clean causal partner because its prefix used the old clock.

The fresh n48 physical member completes 320 caller intervals and 697 accepted
internal intervals with zero rejection. Its endpoint is 2.356 GPa, 1033.68 K,
and a 289.08 K range. Maximum relative complete-energy closure is 0.2125%, and
relative owner-Nye mismatch is 1.81e-14. The corresponding frozen member
completes 320 caller intervals and 670 accepted internal intervals with zero
rejection. Its endpoint is 2.385 GPa, 1025.90 K, and a 252.95 K range; maximum
energy closure is 0.3220%, and owner-Nye mismatch is 6.52e-15.

The matched intervention certificate passes at all 320 common states. At the
late joint-localization onset, the physical member enters the simple
heat/power-band screen at step 265 (1.65625 microseconds), whereas the frozen
member enters at step 274 (1.7125 microseconds). Freezing Arrhenius arguments
therefore delays onset by 56.25 ns. The corresponding sampled spans are only
56.25 ns for physical and 31.25 ns for frozen, so neither approaches the
registered 1 microsecond persistence requirement.

The strict fixed-component classifier is more restrictive and finds no
qualifying snapshot or episode. Maximum matched component temperature excess
is 37.565 K, below the registered 50 K threshold. During physical steps
265--274, softening is 27.6--31.8% and heat/power overlap is 0.80--0.90, but
minor width is only 1.10--1.13 interface widths and aspect ratio only
1.30--1.35. The required limits are at least two interface widths and aspect
ratio at least three. Spatial refinement also remains unqualified. Strict ASB
is therefore false for independent causal, morphology, persistence, and
refinement reasons.

## DRX processing and interval heat balance

The fresh physical member records 2.285e-21 m3 fresh sweep and 2.705e-22 m3
revisit; the frozen member records 2.191e-21 m3 fresh sweep and 2.598e-22 m3
revisit. The largest physical grain gains 1.950e-21 m3 versus 1.876e-21 m3 in
the frozen control. These are conservative existing-boundary transformations,
but the endpoint classifier does not mark them substantial. Grain count stays
four and no spontaneous grain is allocated.

The fresh n32 spatial partner completes 320 caller intervals and 548 accepted
internal intervals with zero rejection. Maximum complete-energy closure is
0.5316%, and relative owner-Nye mismatch is 4.46e-15. It records 2.245e-21 m3
fresh sweep, 2.686e-22 m3 revisit, and 1.895e-21 m3 largest-grain growth. Its
existing-boundary transformation is likewise conservative and below the
substantial endpoint threshold; grain count remains four.

For physical step 268 to 269, independent direct thermal storage is
3.1569901e-14 J. Mechanical/reaction heat contributes 8.6035442e-15 J and
front heat 2.2966357e-14 J; global conduction integrates to numerical zero.
The fixed 46-cell candidate stores 1.8924336e-15 J from 1.3272107e-15 J
mechanical heat plus 8.7580351e-16 J front heat minus 3.1058057e-16 J
conduction. Whole-domain and candidate relative residuals are 4.04e-12 and
3.89e-12, and saved front heat exactly matches pressure-times-realized-volume
event heat.

## Claim boundary and next decision

V62 establishes a usable adaptive physical-time front operator, a valid fresh
n48 causal pair, causally shifted short localization, complete interval heat
closure, conservative prepared-boundary processing, and repaired-clock
agreement of the selected bulk n32/n48 response. It does not establish a
mesh-converged ASB: signed spatial content and localization morphology remain
grid-sensitive, and no one-microsecond strict episode forms. It also does not
establish substantial DRX at this horizon, spontaneous nucleation, a
rate/temperature boundary, or material calibration.

The next scientific decision should not tune width or temperature thresholds.
It should first add or justify an energy-conjugate nonlocal regularization of
the owner stress-to-rate pathway (or demonstrate resolution of the existing
one) and then repeat a bounded common-origin spatial refinement. The unchanged
EXP-floor reference must be preserved. A longer pair is secondary: the present
strict failure is already controlled by matched excess and morphology, not
only by insufficient observation time.

Exact continuation commands remain in `full_model/docs/v62_restart_commands.md`.
Machine-readable hashes, runtime provenance, classifications, and regression
status are in `full_model/verification/v62_campaign_manifest.json`.
