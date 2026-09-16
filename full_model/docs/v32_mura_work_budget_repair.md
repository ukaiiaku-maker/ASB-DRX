# V32 Mura work-budget diagnosis and repair

## Outcome

The local production repair passes its conservation, energy, restart, and
timestep checks. The broad B2 temperature/rate/seed matrix is not yet
authorized because V31 spatial convergence remains unresolved. The only
justified continuation is the prepared 64/128 matched-horizon pair from the
frozen 3% checkpoints to 5% strain.

No V30 or V31 artifact was modified, and no continuation was launched.

## First rejected events

The V31 terminal checkpoints reproduce the old guard exactly.

At 64 square, the full event has:

- conjugate plastic work: -2.037e5 J m^-3-cells;
- recoverable elastic-energy release: -9.329e5 J m^-3-cells;
- defect free-energy increase: 3.557e6 J m^-3-cells;
- line-creation contribution: 3.120e6 J m^-3-cells;
- physical heat/dissipation balance: -4.490e6 J m^-3-cells.

At 128 square, it has:

- conjugate plastic work: 2.960e6 J m^-3-cells;
- recoverable elastic-energy release: 2.400e6 J m^-3-cells;
- defect free-energy increase: 4.407e6 J m^-3-cells;
- line-creation contribution: 3.489e6 J m^-3-cells;
- physical heat/dissipation balance: -2.008e6 J m^-3-cells.

Thus the original unpartitioned event is thermodynamically inadmissible at
both grids. The 64 event would increase recoverable elastic energy as well as
defect energy. The 128 event releases elastic energy, but not enough to pay the
complete defect free-energy increase.

The failure is family-resolved. At 64, family 2 contributes -1.256e7 J m^-3-
cells of plastic work while family 3 contributes +1.236e7. At 128, family 3
contributes -6.322e6 while family 2 contributes +9.282e6. Stalling only the
negative-work channel also removes that channel's artificial CFL restriction;
the requested 2 ns event then leaves respectively 1.166e8 and 1.929e8
J m^-3-cells of positive physical dissipation.

## Hypothesis decisions

- **B1, scalar event extent or timestep:** rejected as the explanation. The
  legacy event remains inadmissible throughout the requested timestep series;
  CFL fixes the first four trials, and the smaller 64 trial remains uphill.
  A constrained family extent is still required.
- **B2, omitted energy source:** rejected at this event. Independently
  recomputed recoverable elastic release and the complete local defect free
  energy make the deficit larger, not smaller. Plastic work is the conjugate
  diagnostic and is not counted a second time.
- **B3, missing reverse/saturation path:** not required to repair this first
  event. Existing reversible exchanges cannot be spent prospectively to pay
  an uphill flux. Longer exposure may still motivate a separately activated
  release mechanism.
- **B4, physical stall:** supported at Burgers-family resolution. A family
  with nonpositive conjugate work stalls while dissipative families continue.

## Production change

The accepted transaction now has two nested physical constraints:

1. family-wise dissipation complementarity sets a negative-work Burgers
   family extent to zero;
2. the remaining common Mura event is backtracked atomically until the exact
   fixed-total-strain recoverable elastic release pays the full defect
   free-energy change and leaves nonnegative heat.

Population transport, alignment, plastic distortion, family Nye, slip, work,
heat, and reaction extent all use the same accepted family and scalar event
scales. Unaccepted extent remains unreacted. If no positive scalar extent is
admissible, a zero-flux physical stall is accepted while time and external
loading continue. No state is clipped or projected.

The exact off comparator is
`mura_work_budget_mode="legacy_reject"`. It retains all family extents and the
old line-creation-only work guard and heat partition.

## Units and first law

The spatial fields are energy densities in J m^-3. Their array sums are
reported as J m^-3-cells, matching the pre-existing ledger convention.
At fixed total strain,

`elastic release = defect free-energy increase + physical heat`.

Plastic work is recorded independently to check work conjugacy; it is not an
additional energy source. External work during the internal fixed-strain
transaction is zero. Numerical compatibility terms remain absent from the
physical energy and heat ledger.

## Verification and continuation decision

The compact 64/128 screen advances four CFL-limited steps and the identical
physical horizon with roughly half-sized steps. All Nye, population, and
first-law invariants pass; no projection occurs; no whole-state stall occurs.
The largest timestep difference among declared metrics is 5.02e-5 at 64 and
1.79e-5 at 128, far below 5%. First-law relative residuals remain below
3.6e-16.

Machine-readable evidence is in:

- `full_model/verification/v32_mura_first_rejection.json`;
- `full_model/verification/v32_mura_repair_screen.json`;
- `full_model/verification/v32_mura_work_budget.png`;
- `full_model/reference_configs/v32_mura_matched_continuation.json`.
