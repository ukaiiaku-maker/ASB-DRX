# V55 interpretation addendum from the V56 evidence audit

## Corrected claim

V55 demonstrates substantial generic prepared-boundary migration and defect
processing in the coupled production architecture. The retained controls,
complete-energy ledgers, and scoped restart audit are valid. It does not yet
establish complete kinetic convergence of the monolithic integrated path.

The former n32/n64/n128 terminal comparison used tracking-window exits at
different physical clocks: 0.343333, 0.370000, and 0.396667 microseconds and
strains 1.03%, 1.11%, and 1.19%. `PAIR_LEFT_ACTIVE_WINDOW` is a numerical
tracking limit, not physical arrest or grain consumption. The terminal
fractions remain useful endpoint observations but are not matched-time kinetic
refinement.

## Retained matched-time reduction

All three grids contain an authoritative step-100 checkpoint at the same
0.336667 microsecond clock and 1.01% applied strain. Increments use each
checkpoint's stored initial child-material fraction, rather than a hardcoded
0.5 baseline.

| Observable | n32 | n64 | n128 | n32→n64 error | n64→n128 error |
|---|---:|---:|---:|---:|---:|
| Net transformed material | 0.285986 | 0.288550 | 0.290128 | 0.889% | 0.544% |
| Gross swept volume (m3) | 1.41849e-20 | 1.43121e-20 | 1.43904e-20 | 0.889% | 0.544% |
| Processed line (m) | 5.67396e-3 | 5.72483e-3 | 5.75614e-3 | 0.889% | 0.544% |
| Stress (Pa) | 2.51018e9 | 2.50717e9 | 2.50707e9 | 0.120% | 0.004% |
| Mean temperature (K) | 941.972 | 942.293 | 942.392 | 0.034% | 0.010% |
| Peak temperature (K) | 944.241 | 944.576 | 944.683 | 0.035% | 0.011% |
| Mean absolute slip | 5.78300e-4 | 5.90595e-4 | 5.91070e-4 | 2.082% | 0.080% |

Every retained supporting prefix, enabled/disabled/frozen control, and coupled
trajectory passes the declared hard checks. The step-60 segmented and
continuous audited state remains bitwise identical. Historical first-step
energy residuals are accepted only when either the relative residual is below
1e-8 or the absolute residual is below 1e-6 J/m3; the largest exceptional
first-step value is about 5.64e-8 J/m3, while subsequent relative closures pass.

The monolithic integrated path has no evolved-state full/half-step temporal
fork. Therefore the corrected outcome-neutral top-level result is
`INCOMPLETE_EVIDENCE`, with temporal refinement missing and no failed hard
condition. This does not erase the valid substantial prepared-boundary motion.

## Claim boundaries

- The 1e12 and 1e10 s-1 attempt frequencies are generic constitutive
  hypotheses, not numerical acceleration or material calibration.
- The dedicated V54 recurrent and fresh V55 monolithic calculations differ in
  initial state, loading, temperature, domain, and bulk operator. Their
  difference is not a frequency-only comparison.
- The eight-grain legacy result is a hard-invalid diagnostic because its
  physical energy ledger is incomplete. It is not polycrystal ASB evidence.
- Strict ASB and spontaneous intragranular grain birth remain unqualified.

The immutable corrected decision and figures are indexed by the V56 manifest.
The frozen V56 source passes 895 canonical tests in 433.16 s.
