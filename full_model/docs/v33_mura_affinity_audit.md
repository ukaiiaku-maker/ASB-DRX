# V33 Mura limiter and discrete-affinity audit

This audit is a read-only fork of the preserved 64-grid checkpoints at step
1451 (3.841278% strain) and step 2971 (4.200103% strain). The live `e7aa16e`
continuation was not modified.

## Limiter semantics

The scalar event extent multiplies the signed Mura velocities and family
plastic-flow rate. Consequently it scales transport, slip, plastic distortion,
Nye, and orientation increments. It does **not** scale `accepted_dt_s`.
The case runner advances physical time by `accepted_dt_s` and computes imposed
strain from that time. Locking/unlocking and ordering also use the full
`accepted_dt_s`. Thus the present implementation limits constitutive amplitude
while the prescribed-loading clock continues; it is not timestep subdivision.

## Discrete affinity

For an isolated family finite event, the audited fixed-total-strain affinity is

`A_f = (E_elastic,before - E_elastic,after,f) - (F_defect,after,f - F_defect,before)`.

Positive `A_f` is admissible heat. The frozen V32 family complementarity used
only conjugate mechanical work `W_f > tolerance` before the complete discrete
event was evaluated.

At step 1451, family 2 has positive mechanical work (`1.1717e5` raw
J m^-3-cells) but negative full-event affinity (`-2.6556e6`). Its near-zero
`2^-20` event is also negative (`-8.73e-2`). Family 3 has negative mechanical
work and is correctly stalled by the old complementarity. Families 0 and 1
have positive full-event affinities; their near-zero elastic differences fall
below double-precision subtraction resolution, so those tiny-event signs are
not decision-grade.

At step 2971 all four mechanical works are positive, but families 2 and 3 still
have negative full-event affinities (`-1.1518e7` and `-4.1562e7`). Both become
positive at `2^-20`, explaining why scalar backtracking accepts a finite small
extent. Therefore negative mechanical work is sufficient to stall a family,
but positive mechanical work is not sufficient to authorize its full discrete
event.

## Checkpoint-fork controls

At step 1451, reducing requested timestep below the CFL-limited interval to
`2.5e-10` and `1.25e-10 s` still produces an exact zero Mura event. This rejects
a simple solver/timestep-resolution explanation. Continuing the prescribed load
by one requested increment admits `s=2^-20`; holding total strain fixed remains
at exact stall. Removing the fixed mechanical heterogeneity admits the full
event, while disabling recovery/exchange reactions does not change the Mura
stall and only removes the later exchange extent. A `+25 K` perturbation admits
`s=2^-8` and activates all families.

At step 2971, the hold accepts `s=2^-12`; one requested loading increment raises
it to `2^-6`, removal of the fixed constraint admits `s=1`, and `+25 K` admits
`s=1/4`. The below-CFL `1.25e-10 s` refinement accepts `s=2^-11`.

Every accepted fork passes the Nye hard invariant, uses no projection, has
nonnegative deposited heat, and closes its first-law ledger. The machine-readable
artifact reports each energy as a raw grid sum, volume average, and
unit-thickness total, together with corresponding rates, active family masks,
reaction extents, input hashes, and source provenance.

## Decision

The long trajectory is not invalidated by another numerical zero-flux defect.
It is traversing a load-conditioned, strongly extent-limited regime. However,
the current family gate is only a first-order mechanical sign test, not the
complete discrete thermodynamic complementarity. Any V33 repair should decide
family activation from the complete discrete affinity and must separately state
whether an unreacted event remainder is carried, discarded, or represented as
an internal clock. Broad B2 remains unauthorized by this audit.
