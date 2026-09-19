# V42 topology-energy, organization, and physical-response decision

## Current decision

V42 repairs the production topology publication defect and rejects the V41
apparent-order result.  The old event rotated local line moments, injected the
measured reservoir-Nye residual into the plastic-curl field, and omitted the
dominant ordered-gradient increment.  Production now permits only the
geometry-neutral conversion representable by its persistent state, evaluates
the complete discrete extensive energy before publication, deposits released
physical energy as heat, and leaves every state and ledger unchanged on
rejection.  Reorientation and junction creation fail closed until persistent
endpoint and swept-surface geometry is implemented.

At the first retained-state increment, the legacy source-offset and
line-continuity errors are 0.08146 and 0.08072.  The repair closes them to
`1.09e-14` and `9.05e-15`.  The repaired nonthermal state is bitwise equal to
the valid control; its complete heat ownership produces at most `0.01204 K`
additional local temperature.  It does not create apparent order and remains
`NO_QUALIFIED_BOUNDARY`.

## Spatial and organization decisions

The original matched 62.5-microsecond n128/n192 test is complete.  Curl-Nye
RMS differs by 38.06%, maximum tensor norm by 50.22%, the L1 tensor-norm
integral by 29.53%, and ordered line by 70.60%.  The selected mode-24 band
differs by 40.89%; the nearly complete common coefficient square differs by
132.31%.  V41's 7.31% one-macro result was a valid short-window observation,
not evidence of convergence at the original horizon.  Classification remains
`UNRESOLVED_SPATIAL_SCALE`; no output filtering, diffusivity tuning, or changed
threshold is used.

The retained 5% orientation span is localized to two adjacent cells at
`+5.664 deg` and `-4.116 deg`.  Its p95-p05 spread is only `0.1132 deg`, and
unwrapping does not change the span.  This is an underresolved dipolar
extremum, not resolved plateaus or a LAGB.  DRX and persistent-boundary claims
remain false.

## Physical-response alternatives

At the archived stationary front state, eight of sixteen signed proposals in
the retained 0.01 mean-shear hold are downhill as raw finite Helmholtz
comparisons, but none publishes under the complete directional law.  A prepared
zero-mean-shear counterfactual publishes four of sixteen signed trials.  This
supports the V41 energy decomposition: elastic mismatch controls the later
stationarity.  The alternative is explicitly a prepared-state comparison,
not a free plastic-history reset or a work-accounted unloading trajectory.

The V41 matched finite-conduction full/frozen-flow result is retained rather
than replaying a broad thermal matrix.  It already establishes stronger heat
concentration from temperature-sensitive flow while both trajectories remain
broad/nonpersistent.  Strict ASB is false.

## Verification and active continuation

The unchanged configured suite passes 747 tests in 252.37 s; focused topology
tests pass 41/41.  The 5.01% topology-off control is complete.  The repaired
topology-on continuation was profiled locally to step 22,620 and then moved to
HPC3 because its projected runtime exceeded the local 30-minute placement
limit.  Job `56166842` runs from pushed source `bb67429` and a checksum-verified
step-22,620 seed.  A first attempt, job `56166765`, is quarantined because a
renamed seed caused a fresh trajectory; it contributes no scientific evidence.

`tmux:v42_topology_manager` is polling the live job, and will checksum and
fetch its archive after scheduler completion.  The executable closure is:
postprocess the matched 5.01% pair, update the topology decision, assemble the
final controller/manifest, then commit and push a versioned addendum.  This
report is therefore a decision-grade active-job checkpoint, not a claim that
the long continuation has already completed.

No material calibration, physical LAGB, DRX, or strict ASB is claimed.
