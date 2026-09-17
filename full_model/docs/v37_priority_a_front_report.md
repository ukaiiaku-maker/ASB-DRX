# V37 Priority A: current-source front controls

## Scope and provenance

These controls use the current n128 production front from source checkpoint
`0c34e6f2b85b7301967c9db023fdb3281a14e804`, based on canonical V36
checkpoint `e1795173471ea3deea1f44fe68e4be2de384fab0`.  They are short,
zero-applied-pressure discriminators, not a material calibration or a claim of
finite-amplitude migration.  The registered screen contains 17 hypotheses and
was frozen before response evaluation.

## Decision

The old scalar-equality control was not equality of the recurrent state: the
inactive wake retained a different history.  Explicit owner equality fixes
that initial condition.  One Mura interval subsequently separates active
parent/child history from the unsupported wake.  At the finite periodic slab,
the forward and reverse defect and boundary increments then agree, while the
phase-gradient and phase-local endpoint increments do not.  The residual
equal-state response is therefore attributable to finite diffuse-interface
interaction and recurrent history, not broken label symmetry.

Complete material exchange is the decisive symmetry control.  It reverses the
velocity and accepted contour displacement exactly while preserving processed
line magnitude.  Reversing only the geometry proposal does not exchange the
materials: its proposed volume opposes the positive constitutive rate, so it
is rejected before publication.  Its accepted displacement, processed line,
and common-state energy change are all zero.  A positive diagnostic rate in
that rejected record is not physical migration.

The baseline accepted displacement in one interval is
`3.69684016154892e-13 m`; processed line is `1.1261320234643155e-12 m`.
The preregistered rate-only screen spans `2.2091978635787967e-09` to
`2.268291575616896e-06 m/s`; it identifies bounded trajectory candidates but
does not qualify any parameter set.

## Equivalence and cost

The archived local n128 and HPC n192 production modules are bitwise identical.
The HPC runner differs only by its source-SHA environment override.  V37 adds
endpoint component diagnostics to `complete_front_energy.py`; accepted state
and kinetics are unchanged.

Six n128 intervals required 461.6 s under profiling (about 76.9 s/interval).
The bounded moment projection consumed 334.3 s, or 72.4% of profiled time.
This makes that projection the first performance target before a broad
finite-amplitude campaign.

Machine-readable decisions are in
`v37_current_source_controls.json` and `v37_front_acceptance_audit.json`.
Raw outputs and the cProfile file remain under
`/Users/sdillon/HPC3/local-results/asb-drx-v37-priority-a-0c34e6f/` with their
SHA-256 hashes recorded in the decision artifact.
