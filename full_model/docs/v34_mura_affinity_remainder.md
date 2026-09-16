# V34 complete-affinity family selection and remainder decision

## Production rule

The production `energy_limited` Mura path now tests the complete finite-event
affinity of each Burgers family at the conservative full-rate CFL/spin
timestep. A family is selected only when its isolated event satisfies

`elastic release - exact defect free-energy change >= -tolerance`.

The selected families are then evaluated together. If coupling makes the joint
candidate inadmissible, the existing dyadic scalar backtracking acts on that
joint event. Positivity, Mura/Nye consistency, and the first-law guard remain
hard constraints. `legacy_reject` remains the exact off comparator.

The unreacted fraction is a constrained instantaneous rate, not deferred work:
it creates no hidden state or debt and is recomputed from the next accepted
state and imposed load. The accepted physical timestep and loading clock are
not rescaled by the event extent. The ledger reports per-family remainder as
`1 - joint_scale * family_selection_scale`.

## 64-grid checkpoint forks

At the preserved step-1451 checkpoint (3.841278% strain), the old mechanical
gate admitted families 0, 1, and 2 and then reached a zero event. The complete
rule finds isolated raw affinities of `+0.0788`, `+0.0828`, `-1.696e6`, and
`-4.992e6 J m^-3-cells`. It selects families 0 and 1; their joint full event is
admissible with raw heat `0.1616 J m^-3-cells`, zero first-law residual, and no
projection. Below-CFL timestep refinement preserves the same selected set and
full joint extent.

At step 2971 (4.200103% strain), families 0 and 1 remain the full-event
admissible set. Their joint event is accepted at full extent with raw heat
`0.6017 J m^-3-cells`. Families 2 and 3 retain negative full-event affinities,
so the no-debt rule does not carry their formerly admissible small fractions
forward. This is a deliberate constrained-rate choice, not a claim that those
families can never react under a later state or load.

A +25 K perturbation at step 1451 activates family 2 and yields raw heat
`1.002e5 J m^-3-cells`; at step 2971 it does not make families 2 or 3 admissible
at full extent. One requested load increment does not change the selected set
at either checkpoint.

## Constraint-release accounting

Removing the fixed mechanical heterogeneity changes the pre-event elastic
energy and is an external parameter switch. That contribution is therefore
reported separately and is never deposited as Mura heat. The raw external
switch energies are `3.436e9` and `3.774e9 J m^-3-cells` at steps 1451 and
2971. After the switch all four families have positive complete affinities and
the joint full Mura events close independently, with raw heat `2.067e8` and
`1.401e8 J m^-3-cells`. Their relative first-law residuals are at roundoff.

## Selected 128-grid check

The frozen 128-grid 3% checkpoint was feasible locally. At 1100 K and +25 K,
all four isolated families and their joint full events are admissible. Raw joint
heat is `1.863e8` and `2.302e8 J m^-3-cells`, respectively; hard Nye/Mura
invariants pass, heat is nonnegative, and no projection is used.

## Decision and scope

The V34 rule removes the known false authorization in which positive mechanical
work allowed a family with negative complete discrete affinity. It also removes
the artificial all-family scalar bottleneck at both preserved 64-grid states.
The bounded forks qualify the operator and remainder semantics; they do not
qualify broad B2, spatial convergence, or a new long trajectory. Those remain
separate decisions after canonical integration and restart comparison.
