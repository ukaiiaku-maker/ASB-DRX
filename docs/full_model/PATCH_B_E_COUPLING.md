# Patch B--E coupling contract

Status: staged component design; not yet an integrated scientific claim.

## Separation of roles

The full v34 fields remain authoritative for elasticity, slip, temperature,
dislocation state, stored energy, orientation, and phase evolution. A hazard
event may create an `EmbryoRecord`, but it may not create a grain label. The
record owns identity and history; the full fields supply its local environment.

The legacy classical circular free energy is retained as a thermodynamic
feasibility diagnostic, not used as an undeclared substitute for the required
EXP-floor kinetic activation enthalpy. Embryo creation and radial mobility will
each have their own bounded EXP-floor enthalpy, signed activation entropy, and
attempt/mobility prefactor.

## Embryo free energy and kinetics

For represented thickness `h`, radius `R`, effective interface energy `gamma`,
stored-energy relief `Delta psi`, orientation penalty `P_ori`, and compatibility
penalty `P_comp`,

```text
F_emb = h [2 pi R gamma + pi R^2 (P_ori + P_comp - Delta psi)]
P_net = Delta psi - P_ori - P_comp
R_critical = gamma / P_net,                         P_net > 0
R_dot = M_R(T, z, |P_net - gamma/R|)
        [P_net - gamma/R]
```

With frozen local fields this gives

```text
dF_emb/dt = -2 pi h R M_R [P_net - gamma/R]^2 <= 0.
```

The mobility factor is

```text
M_R = M_0 exp[-(Delta H_R_EXP-floor - T Delta S_R)/(k_B T)]
```

with an explicit drag/rejection branch for a negative free barrier. Forward and
reverse radius motion use the same positive mobility; the thermodynamic force
sets the sign. Event rate and radial velocity remain dimensionally distinct.

## Stateful candidate lifecycle

1. Integrate local hazard and explicitly count every raw `H >= E` trigger.
2. Evaluate thermodynamic feasibility, residual-Burgers support, and trial
   crystallographic orientation without allocating a label.
3. On acceptance, allocate a unique embryo ID and record parent/lineage,
   position, orientation, radius, physical time/strain, cumulative hazard, RNG
   stream/state, barrier, rate, and contact fields.
4. Evolve radius continuously at accepted physical timesteps. Unfavorable
   embryos shrink and retire; a transient barrier excursion does not erase the
   record.
5. Couple a candidate-support order field to the same PF interface/stored-energy
   functional. This support is not a hard grain label.
6. Mark an embryo promotable only when it is supercritical and growing, survives
   in physical time, and has resolved/pure phase support.
7. Only then allocate the production grain/order field and preserve the embryo
   ID as its provenance.

## Exact restart state

Atomic checkpoints must contain the complete embryo population JSON, every
candidate-support field and slot mapping, next unique ID, event histories,
cumulative raw/viable trigger counters, all RNG streams, physical/global time,
and the existing full-model fields. Continuous and segmented trajectories must
agree for identities, free-energy/radius histories, promotion time, phase
fields, thermal/dislocation fields, and RNG state.

## Physical grain recognition

An allocated label is a physical DRX grain only when all of the following hold:

- one resolved connected component with area scaled to the declared interface
  width;
- sufficient order-parameter purity and physical-time persistence;
- coherent parent lineage and a promoted source embryo;
- crystallographic distinction modulo the declared symmetry;
- lower stored energy than the parent support by a declared margin;
- demonstrated growth or stable support;
- survival of the retirement grace time.

The reporting ledger keeps allocated labels, topology components, candidates,
promotable/promoted embryos, physical DRX grains, matrix grains, and
recrystallized area fraction separate.

## Baseline invariance

When stateful embryos are disabled, the added modules and diagnostics must not
alter any authoritative v34 field. Diagnostic counters may add checkpoint/CSV
columns but may not consume RNG draws or enter evolution. `asb_only` must remain
bitwise matched to the immutable v32 response at zero activation entropy.
