# V35 phase-control and physical-event decision

## Decision

The phase integrator and topology active set are now independent production
controls.  With the projector inactive, switching it on or off is bitwise
neutral in `eta`, `rho`, and `T`.  At the archived n128 preterminal state the
projector activates and prevents the historical disconnected phase island,
while the identical IMEX proposal with the projector disabled reproduces the
`UNAUTHORIZED_PHASE_ISLAND` terminal.  This preserves the narrow scientific
interpretation: the active set enforces an existing-boundary/no-nucleation
contract; it is not a general constraint on physical grain topology.

The atomic front measure is now expressed in physical sites.  For the declared
`b^3,b` baseline,

```
event volume     V_e = b^3
event length     l_e = b
site area        A_site = V_e/l_e = b^2
site count       N_site = A_front/b^2
event counts     N_+/- = N_site r_+/- dt
signed sweep     Delta V = (N_+ - N_-) b^3
normal velocity  v_n = b (r_+ - r_-)
```

Partitioning an interface into patches changes neither total site count nor
total swept volume.  Changing mesh resolution for fixed physical geometry is
also neutral.  Raw count and swept volume correctly remain extensive in real
interface area (and therefore in represented thickness); their areal measures
and normal velocity are thickness independent.

## Phase switch evidence

The inactive n64 favorable control used one production physical step.  The
on/off final `eta`, `rho`, and `T` arrays are bitwise identical; both lower the
declared phase energy by `9.3464008125e-4 J/m`, move signed child support by
`6.9356513405e-22 m3`, and propose `6.6421016310e-22 m3` of contour sweep.  The
enabled projector records zero activation calls and zero changed pixels.

At the archived n128 `+1e-6` preterminal checkpoint, the enabled projector
records 48 activating calls and 596 projected pixel updates.  It leaves one
attached sign flip, lowers phase energy by `2.2691795527e-5 J/m`, and passes
the topology adapter.  Disabled, the same IMEX path lowers energy by
`2.2943204082e-5 J/m` but produces 104 sign flips and terminates as
`UNAUTHORIZED_PHASE_ISLAND`.  Thus energy descent alone does not enforce the
declared topology contract, and the topology intervention is now directly
measured rather than hidden inside the integrator choice.

## Bounded kinetic screen

The 15-case screen uses three temperatures (900, 1100, and 1300 K) and five
stored-energy pressures (-600, -200, 0, +200, and +600 MPa), with applied
pressure exactly zero, equal defect states, and unit transmission.  It is a
hypothesis screen, not a calibration.

- Zero stored-energy pressure gives exactly equal directional rates and zero
  velocity at every temperature.
- Reversing stored-energy pressure swaps the directional rates and reverses
  velocity exactly.
- Maximum detailed-balance log residual is `1.12e-16`.
- The predicted rate range is about `1.10e6` to `2.41e7 site^-1 s^-1`; maximum
  predicted speed is `2.389e-3 m/s`.
- For a 10 micrometre interface represented through thickness `2b`, the
  physical count is `80645.16` sites and the areal density is
  `1.62591e19 m^-2`.

The exact-source records and hashes are in
`full_model/verification/v35_phase_kinetic_decision.json`.  No long campaign
was launched, and the result makes no production-material calibration claim.

## Limitations

The topology active set must remain restricted to existing-boundary evolution
with nucleation and label allocation disabled.  A topology-valid phase
proposal may still be rejected by the independent complete common-state
energy transaction.  The kinetic numbers are bounded model hypotheses using
the declared EXP-floor baseline; DD data have not calibrated a production
collective transfer law.
