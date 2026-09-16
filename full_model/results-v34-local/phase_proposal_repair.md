# V34 phase-proposal repair

## Decision

The existing-boundary production phase proposal is locally qualified.  The
production path now advances all labels simultaneously on the simplex with an
embedded IMEX error estimate, implicit spectral capillarity, a substep
increment bound, and a frozen-state energy-descent check.  The historical
sequential clipped Euler path remains available only as the explicit
`legacy_sequential_euler` reproduction mode.

This is not merely a smaller explicit timestep.  A stable unconstrained IMEX
replay integrated the full physical step in 19 accepted substeps and lowered
the declared phase energy, but still formed the same forbidden zero-level
filament.  The actual missing contract was topological: an existing boundary
with nucleation disabled cannot create a disconnected sign component or split
an existing periodic phase component.  The production proposal therefore
uses a periodic active-set projection that allows attached front translation
while projecting those two inadmissible events.  The downstream atomic
adapter remains fail-closed and independently rechecks the result.

## Archived terminal replay

Both n128 `+1e-6` and `-1e-6` checkpoints pass steps 138 and 139 without a
topology terminal.  Every step integrates the requested `1e-7 s`, decreases
the declared energy, stays within `1.12e-16` of the simplex, and retains the
existing component identities.  The current complete directional-rate
operator rejects these near-equal proposed sweeps; that is an independent
kinetic decision, not a phase-topology failure.

The exact-off legacy replay still produces `UNAUTHORIZED_PHASE_ISLAND` at the
historical positive checkpoint.  This preserves the regression boundary and
demonstrates that the repaired result is not caused by changed capillarity,
barrier, mobility, pressure, or timestep.

## Finite response and grid convergence

At the unchanged physical coefficients and a signed 200 MPa phase-energy
difference, the compatible n128 and n192 fixtures give the same physical mean
two-front displacement:

| grid | forward displacement | reverse displacement | width |
|---|---:|---:|---:|
| 128 | `+2.701810619e-8 m` | `-2.701810643e-8 m` | `8.005 cells` |
| 192 | `+2.701810621e-8 m` | `-2.701810645e-8 m` | `11.745 cells` |

The n128/n192 forward displacement differs by `7.8e-10` relatively.  Forward
and reverse energy drops agree to roundoff, simplex error is `1.12e-16`, and
the high-wavenumber fraction does not increase materially.  No topology
projection is invoked in these smooth finite translations.  The corresponding
NPZ files contain `initial_eta`, `forward_eta`, and `reverse_eta` and are
directly consumable by `run_v34_finite_coupled_response.py`.

The n64 version is intentionally not called a resolved fixture: the unchanged
`0.30 micrometre` interface is less than two n64 cells.  The second resolved
grid is n192.

## Scope and limitations

The active set applies only to the declared existing-boundary, no-nucleation,
no-label-allocation production mode.  It must not be reused to suppress
physical nucleation or general multiphase topology changes.  Long-horizon
front evolution and the same-state Mura/front/thermal response matrix remain
downstream integration questions; this checkpoint qualifies the phase
proposal supplied to those operators.
