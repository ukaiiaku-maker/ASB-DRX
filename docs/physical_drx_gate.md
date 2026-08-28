# Physical DRX embryo and orientation gate

The DRX gate is now separate from EXP-floor flow, recovery, collective-event
diagnostics, and phase-label allocation. A candidate is a checkpointed physical
object with a unique ID, position, circular radius, trial orientation,
crystallographic misorientation, parent and lineage, birth time/strain, RNG
lineage, thermal-attempt record, integrated positive driving force, phase
support history, and active/promoted/retired/rejected lifecycle.

## Thermodynamic evolution

For represented thickness `t`, boundary energy `gamma`, stored-energy relief
`Delta f`, and radius `R`, the isolated excess energy is

\[
\Delta F=t(2\pi R\gamma-\pi R^2\Delta f).
\]

The critical radius is `gamma/Delta f`, the zero-excess escape radius is twice
that value, and radial motion follows

\[
\dot R=M_R(\Delta f-\gamma/R).
\]

Every accepted step must decrease the frozen-driving excess energy. The
decrease is recorded as released heat with an exact closure ledger. A
subcritical embryo shrinks and retires below the resolved radius; a
supercritical embryo grows. No density is reset and no interface is created
without its energy being represented.

## Promotion

An accepted thermal attempt alone creates only an active embryo. Promotion
requires all of the following:

- crystallographically distinct orientation under the declared symmetry;
- growth beyond the zero-excess escape radius;
- minimum physical survival time;
- positive integrated growth driving;
- persistent resolved phase-field support for the required number of updates;
- phase purity above the declared threshold.

Only a promoted embryo can generate a child `GrainRecord` with embryo
provenance. The grain tracker now rejects even a large, pure, persistent child
phase label if that record lacks a passed embryo gate. This closes the prior
loophole in which field support and lineage text alone could be classified as
DRX.

## Verification and remaining integration

Tests cover subcritical shrinkage/retirement, supercritical growth without
premature promotion, persistent-support promotion, crystallographic-equivalence
rejection, unique IDs, exact checkpoint/restart, energy/heat closure, and the
phase-label negative control. The embryo population is not yet coupled to an
automatic stochastic sampler or dynamic order-parameter allocation. That is
intentional: an attempt prefactor and orientation distribution have not been
calibrated, and inventing them would turn a verified gate into a forced DRX
source.
