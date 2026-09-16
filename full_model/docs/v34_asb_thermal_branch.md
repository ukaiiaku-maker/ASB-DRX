# V34 ASB thermal causal branch

This branch separates five thermal hypotheses from one checksum-verified step-100
state at 900 K, 30,000 s^-1, grid 128x128, and seed 43. No ASB classification
threshold is changed, and the causal matrix is not itself a strict-ASB claim.

## Declared operators

- `full_law_local_adiabatic`: local heat evolves with zero conduction and zero bath.
- `frozen_flow_T_local_adiabatic`: heat evolves identically, while the EXP-floor
  flow law and its refreshed potential see 900 K.
- `frozen_recovery_T_local_adiabatic`: heat evolves identically, while KM and
  lattice recovery see 900 K.
- `exact_prescribed_T_thermostat`: every accepted step enforces T=900 K. The
  removed thermal energy is independently computed as
  `rho_cp * mean(T_unconstrained - 900 K)` and enters both exported heat and the
  declared-sink channel. It is not a large finite bath approximation.
- `finite_conduction_periodic_insulated`: Fourier redistribution with
  k=0.15 W m^-1 K^-1, periodic boundaries, and zero bath. Its net heat export is
  zero.

The diagnostic stream records the physical temperature, temperature supplied to
the flow operator, temperature supplied to recovery, deposited heat, thermostat
export, total export, and all common-Mura invariants. Thermal/model validity
stops remain enabled and are terminal scientific outcomes.

## Launch policy

The shared prefix must be generated and verified before branching. Each branch
verifies the prefix SHA-256 and source commit. The common comparison horizon is
step 2500, immediately before the prior 900 K local-adiabatic validity stop.
Concurrency is capped at two and unrelated source-frozen runs remain untouched.
