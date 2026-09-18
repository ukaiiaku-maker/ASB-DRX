# V40 numerical-completion and physical-response decision report

## Evidence boundary

V40 continues the pushed V39 checkpoint `0632526` on
`exp/full-v34-recovery-v1`. Every long calculation uses an immutable detached
source: the fine-grid baseline remains attributable to `0632526`, the n256
discriminator to `b627887`, the current thermal matrix to `62e424f`, the first
promoted one-grain continuation to `e4e27ce`, and its compact-restart retry to
`a23f8c7`. Historical V37 Mura evidence is
retained as a quarantined comparator and is not relabelled as current-source
formation. The unrelated Slurm job `56132213` was observed and not touched.

## Comparison-layer repair and stiff kinetics

The V39 raw `*_J_m3_cells` values were sums of energy density over cells, not
physical cross-grid integrals. After division by cell count, the n64/n128 first
interval differs by 0.03248% in deposited heat and 0.008593% in plastic work.
The corresponding per-unit-thickness quadrature gives the same result. A
uniform 325 kJ/m3 fixture integrates to exactly 3.328e-6 J/m on n64, n128, and
n192. This correction does not erase the real 12.0% front, 43.7% Nye, and
near-extinction ordered-reservoir discrepancies.

`ordering_endpoint_remainder_relative` is now documented as an endpoint drift
diagnostic, not a finite-time error bound. Production uses a bounded BDF solve
through Arrhenius exposure 50 and the asymptotic endpoint above that switch.
The switch overlap, exposure-100 finite/asymptotic overlap, and complete-rate
rescaling test all pass. The certificate is fixed-field; common-macro
refinement remains responsible for coefficient-freezing error.

The actual archive/finalizer/retrieval path was exercised by Slurm job
`56145703`. Its 178-field checkpoint was checksum verified and advanced from
10 to 20 ns after retrieval, establishing a loadable restart rather than a
provenance-only archive.

## Common-state numerical decision

Both n128 and n192 pass their final temporal pairs at a 7.8125 us macro
interval. At that same interval the n128/n192 front displacement differs by
4.18% and beta-p RMS by 0.0167%, but Nye RMS differs by 38.1%, ordered line by
240%, and orientation span by 10.1%. The selected pair therefore has a
front-qualified but spatially unresolved gradient state. The longer n128
trajectory is exploratory and cannot promote fine-grid DRX physics.

The bounded n256 discriminator preserves that conclusion: relative to n192 at
the same 7.8125 us interval, front displacement improves to 2.15% and beta-p
RMS to 0.00893%, while Nye RMS remains 27.7%, ordered line 138%, and
orientation span 5.95%. Because only one n256 timestep was run it is not a
temporally qualified promotion pair; it nevertheless rules out interpreting
the n128/n192 discrepancy as an isolated coarse-grid anomaly.

Stage-localized instrumentation shows physical energy, interface measure, and
curl/reservoir Nye agreement through the first Mura half-step. The front keeps
curl Nye grid-consistent; the following Mura half-step is the first greater
than 5% Nye-state divergence. Trace ordered-line seeding is independently
grid-sensitive in the first half-step. This evidence does not support a guessed
cell-factor repair or kinetic retuning.

## Common-state physical response

The zero-applied-front-pressure n128 continuation reaches a maximum accepted
advance of about 0.277 nm, only about 0.277% of the 100 nm quarter-interface-
width observation target. The accepted forward motion ends when the computed
front rate changes sign. From that state a forward proposal is rejected by its
rate direction, whereas the reverse-direction proposal is kinetically allowed
but rejected because the complete physical free-energy change is uphill. The
front is therefore pinned at a critical state rather than stopped by a solver
failure or unsupported topology. Nye, orientation, and temperature continue
to evolve after migration stops, so stationary migration is not a stationary
internal state.

The final 1 ms history is followed by a same-horizon 31.25/15.625 us paired
macro comparison. It gives zero incremental-front difference and relative
differences of 0.0118% in Nye RMS, 0.00340% in orientation, 0.00475% in beta-p,
and less than 0.0003% in total line and maximum temperature. No 5--30 ms
extension is warranted when both directions are inadmissible, the post-arrest
state is temporally resolved, and the observation target is missed by more
than two orders of magnitude.

## Thermal and one-grain response

The current-source n128 finite-conduction matrix evaluates `rate_low`,
`heterogeneity_short`, and `heterogeneity_long` through the common 0.2501
nominal-strain horizon. Plastic power and irreversible heat are stored as
independent fields; the reduced thermal ledger's zero elastic-storage scope is
not copied into the integrated common-state energy ledger. Component-local
temperature-rise, activity width, overlap, and persistence metrics retain the
predeclared strict-ASB thresholds.

All three terminal cases pass their hard invariants and remain broad or
nonpersistent. `rate_low` has a 9.875 K peak-minus-mean rise and 6.731 um
activity width; `heterogeneity_short` has 29.770 K and 6.707 um; and
`heterogeneity_long` reaches 58.253 K but retains an activity IPR fraction of
0.903 and a 6.457 um width. The strongest thermal contrast is therefore not a
localized persistent band. No strict ASB result is claimed.

The current-source one-grain control and mechanical-heterogeneity path both
close their hard invariants through 2% strain without phase or grain-label
allocation. The homogeneous state stays at zero Nye and zero orientation span.
The heterogeneous state reaches 0.05275 degrees global span and 2132 1/m Nye
RMS, but its independent Frank--Bilby test is negative with a relative residual
of 59.24. Ordered fraction or global span is not accepted as a boundary. The
promotion classifier now also requires the independent Frank--Bilby residual to
be at most 20%; angle alone cannot trigger unloading.

The first promoted continuation remained invariant-clean through an observed
4.198% strain and developed a recurrent, domain-scale response, but
free-partition job `56147765` was preempted before its full archive finalized.
Same-node rescue job `56149239` confirmed that the scheduler epilog had already
removed its scratch checkpoint, so the 4.198% partial is not claimed as archived
evidence. Retry job `56149258` is advancing from the
checksum-verified 2% archive with atomic compact restarts every 10 minutes.
Durable local fetch and closure-aware postprocessing are active. This is an
infrastructure continuation, not a scientific promotion or a request for user
action. A bounded chain controller may resubmit once from the latest verified
compact checkpoint and then retains the terminal exposure classification.

## Decisions and claim boundary

- Physical normalization, archive/restart canary, and finite-time stiff
  ordering certificate: **passed**.
- n128 and n192 timestep refinement: **passed**.
- Fine-grid gradient-state spatial promotion: **failed numerical
  qualification**; the n128 long history is exploratory.
- Zero-pressure existing-boundary migration: **pinned critical arrest after a
  small accepted advance**, with continuing internal-state evolution.
- Current-source one-grain organization at 2% strain: **no physical LAGB
  precursor**; a preemption-safe 5% exposure continuation remains active.
- Current-source finite-conduction family: **valid broad/nonpersistent heating
  in all three selected cases; no strict ASB**.
- Full DRX, a current-source persistent LAGB, strict ASB, and material
  calibration are **not claimed**.

The executable controller and case manifest contain final process/job states,
source and archive hashes, physical progress, regression results, and the
specific claim boundary for every branch.

The configured regression boundary passes 742 tests in 144.53 seconds.
