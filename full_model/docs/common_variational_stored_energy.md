# Common variational stored-energy review

## Functional

The qualified common coupling uses one normalized interpolation for every
active phase,

\[
f_s(\boldsymbol\eta)=
\frac{\sum_i h(\eta_i)\psi_i}{\sum_i h(\eta_i)},
\qquad h(\eta)=\eta^2(3-2\eta),
\]

with exact partial derivative

\[
\frac{\partial f_s}{\partial\eta_i}=
\frac{h'(\eta_i)}{\sum_j h(\eta_j)}(\psi_i-f_s).
\]

This gives the required child-growth sign when the child state has lower
stored energy. It is stationary for pure phases, cannot create absent phase
support because `h'(0)=0`, and is invariant under label permutation to
floating-point summation error.

Unrecrystallized orientation labels share the current local deformed-matrix
energy. A promoted child carries the current mean stored energy measured in its
dominant pure core. This is a material-state distinction, not a different
evolution equation. Continued deformation therefore rehardens the child and
updates its driving force.

## Deterministic comparison

Four 449-step local continuations started from the frozen HPC3 step-6750
checkpoint. New stochastic creation was disabled; the same existing precursor,
interface energy, mobility, mechanics, and temperature state were retained.

The lineage-scoped reference again produced a physical grain and grew to an
equivalent radius of 3.614 micrometers. Its largest checkpoint-sampled radius
increment corresponds to 5.39 m/s, reinforcing the warning that growth is
numerically abrupt and domain consuming.

The common variational child promoted but reached only 0.534 micrometers before
contracting to zero resolved area. Its mean core density rose from
`2.81e16 m^-2` to `1.67e17 m^-2` in five steps and approximately
`3.29e17 m^-2` near loss of resolved support. The evolving state therefore
removes the stored-energy advantage that the lineage-scoped law holds fixed.

With stored-energy coupling disabled, the child never grew and remains only an
allocated label under the strengthened recognizer. Reversing the common sign
also gives no physical grain. These controls have the expected qualitative
ordering, but the thermodynamically correct common route does not reproduce the
verified lifecycle.

## Decision

The accepted claim remains `SINGLE_FULL_MODEL_DRX_LIFECYCLE_PATH_VERIFIED` for
the frozen `b9afe1f` trajectory. The stronger claim
`FULL_MODEL_DRX_GROWTH_MECHANISM_SUPPORTED` is rejected: the successful
domain-scale growth depends on a lineage-scoped governing term.

Mobility, grid, and domain sweeps would only quantify an equation that failed
the prerequisite thermodynamic comparison, so they were not launched. The
next admissible development is a common state-evolution model coupled to
continuous moving-front line processing. Hazard parameters must remain frozen.

## Energy-channel clarification

The historical promotion ledger reported zero separate line-energy change
because the physical Taylor line energy was embedded in `F_bulk`; it was not
omitted, but the channel name was ambiguous. New-source promotion accounting
subtracts that same physical line term from `bulk_stored` and reports it as
`line`, leaving the total event energy unchanged. Quadratic compatibility
penalties remain diagnostic numerical constraint energies and are not converted
to heat.
