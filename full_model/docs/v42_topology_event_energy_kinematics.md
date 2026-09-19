# V42 topology event: energy and kinematic ownership

## Failure localized from the retained 5% state

The first legacy topology-on increment starts from the checksum-retained
`56151298` endpoint.  Its topology-off twin closes the authoritative
source-offset and line-continuity measures at `1.09e-14` and `9.05e-15`.
The legacy topology-on increment raises them to `8.15e-2` and `8.07e-2` while
moving the wall-local ordered fraction from `8.27e-5` to `8.07e-2`.

The first offending operation is `finite_segment_kink_pair_reorientation`.
It rotates a local first moment and records scalar turning-node and curvature
counts, but the persistent state contains neither segment endpoints nor the
swept surface needed to update plastic distortion.  The following junction
operation is an independent violation for the same reason.  Production then
used the measured reservoir-Nye increment as a source added to
`family_nye_m1`; this made its internal stage comparison close while leaving
`alpha=-Curl(beta_p)` violated.  That reconstruction is removed.

The energy failure is larger.  In this single increment the legacy on/off
ordered-gradient energy difference is about `2.35958e-2 J/m`; the full legacy
5.01% comparator later reached `2.82549 J/m`.  The former topology acceptance
used scalar excess energy but did not include the actual discrete gradient
increment.

## Representable production event

The present persistent state supports one topology operation without adding a
new geometric model: a reservoir conversion that carries scalar line and its
existing first moment together.  For sign `s` and family `a`,

\[
 \rho_{t,s,a}'=\rho_{t,s,a}-\xi_{s,a},\qquad
 \rho_{o,s,a}'=\rho_{o,s,a}+\xi_{s,a},
\]

\[
 \boldsymbol\kappa_{t,s,a}'=\boldsymbol\kappa_{t,s,a}
 -|\xi_{s,a}|\boldsymbol\kappa_{d,s,a}/\rho_{d,s,a},\qquad
 \boldsymbol\kappa_{o,s,a}'=\boldsymbol\kappa_{o,s,a}
 +|\xi_{s,a}|\boldsymbol\kappa_{d,s,a}/\rho_{d,s,a},
\]

with the donor selected by the sign of `xi`.  Density and first moment have
units `m^-2`; their Nye contribution `b tensor kappa` has units `m^-1`.
Because the operation changes neither line geometry nor plastic distortion,

\[
 \Delta\alpha_{\rm total}=0,\qquad
 \operatorname{Curl}_h\Delta\beta^p=0,\qquad
 \Delta\alpha_{\rm declared\ source}=0.
\]

The event is constructed from immutable input.  Its full discrete extensive
energy contains rho-log-rho storage, disordered and ordered reservoir excess,
Nye mismatch (zero coefficient in this production case), ordered-gradient
energy, and junction topology energy.  Publication requires

\[
 \sum_h (F^{n+1}-F^n)\leq\epsilon_F,
\]

as well as scalar-line and total reservoir-Nye closure.  Released free energy
is deposited as nonnegative physical heat on accepted reaction support.  A
failed candidate returns the original inventory/alignment objects and zero
heat; no state or ledger is partly committed.

The legacy reorientation and junction maps remain isolated fixtures.  They
are not production-selectable until a persistent segment/node and swept-area
state can derive, rather than fit, their beta-p and Nye changes.

## Retained-state result

The repaired topology-on increment has a nonthermal state bitwise identical
to the valid control.  Its source-offset and line-continuity measures remain
`1.09e-14` and `9.05e-15`.  Complete-energy accounting adds a maximum local
temperature increment of `0.01204 K`; it does not create apparent order.
Classification is
`VALID_NEGATIVE_AT_TESTED_CONDITION; LEGACY_APPARENT_ORDER_REJECTED;
NO_QUALIFIED_BOUNDARY`.

## Front event normalization clarification

Front trial energy and elementary-event energy are distinct:

\[
 N_e=|\Delta V_{\rm trial}|/V_e,\qquad
 \Delta g_e=\Delta\mathcal A_{\rm trial}/N_e.
\]

`Delta A_trial` is the complete candidate Helmholtz change minus actual
external work plus declared material-sink export.  The per-event Boltzmann
factor consumes `Delta g_e`, not the whole trial energy.  Sink export is part
of the available-affinity definition and remains a separate first-law export;
it is not added twice to Helmholtz energy or heat.  Front event volume remains
separate from the bulk-glide activation volume.
