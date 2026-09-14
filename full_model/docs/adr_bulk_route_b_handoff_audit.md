# ADR: bulk Route-B rejected pending embryo-to-phase handoff audit

Status at v10 restart: `BULK_ROUTE_B_REJECTED_PENDING_EMBRYO_PHASE_HANDOFF_AUDIT`.

The v9 phase transaction rejected the frozen precursor without mutating state,
but it compared only the parent and trial phase-field energies.  The stateful
embryo history explicitly reports and evolves a classical excess free energy;
promotion did not debit that owned energy when the embryo representation was
retired.  This is an energy-accounting omission and is the sole Route-B repair
permitted by Directive v10.

The corrected handoff treats promotion as a representation map.  Its energy
change is the PF field change plus the change in embryo-owned excess energy.
No line, signed Burgers content, boundary reservoir, orientation, RNG state, or
lineage may change during zero-front initialization.  Subsequent front advance
alone may process defects through the v9 moving-front ledger.

Physical compatibility is split into long-range signed-GND, Frank--Bilby GB,
and orientation channels.  Quadratic alpha/GB penalties are numerical
constraint energies, remain diagnostic, never become heat, and never decide
promotion.

## Final decision

The audit found one real ownership defect: the embryo's explicitly evolved
classical excess energy was omitted when its representation was retired.  The
one authorized repair transfers that energy and initializes the diffuse phase
without front processing.  It does not change labels, line content, signed
Burgers content, boundary reservoirs, orientation, lineage, or RNG state.

The corrected frozen replay still rejected all five attempts.  On the first
attempt, the bulk and line changes were zero, physical compatibility increased
by `5.051614232e-15 J`, interface/order energy increased by
`2.833907345e-16 J`, and removal of the negative embryo-owned excess increased
free energy by `2.963666309e-13 J`.  The total map residual was therefore
`+3.017016359e-13 J`.  The physical compatibility split was
`1.59058e-15 J` long-range signed-GND, `3.46103e-15 J` Frank--Bilby, and zero
orientation energy.  The approximately `68.47 J` quadratic penalty change was
reported separately and excluded from the decision and heat.

The physical terms and map residual converge on 64, 128, 256, and 512 grids,
two interface widths, and two penalty multipliers.  Changing the penalty
multiplier changes only the diagnostic numerical energy.  Consequently the
tested state is classified exactly as:

`BULK_ROUTE_B_PRECURSOR_REJECTED_AFTER_ENERGY_HANDOFF_AND_COMPATIBILITY_AUDIT`

This is a state-specific Route-B rejection, not a universal materials claim.
The complete machine record is
`full_model/verification/v10_embryo_phase_handoff_audit.json`.
