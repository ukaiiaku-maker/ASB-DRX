# V24 topological line-supply model

## State, units, and sign convention

For every BCC Burgers family `a`, sign `s`, and mobile, forest, tangle, and
ordered reservoir `r`, the state carries line density
`rho[r,s,a]` in m^-2 and first line-direction moment `kappa[r,s,a]` in m^-2,
with `|kappa| <= rho`. Junction extent is in m^-2 and its product-line moment
is also in m^-2. The reconstructed Nye tensor is

`alpha = sum(r,a) b[a] tensor (kappa[r,+,a]-kappa[r,-,a])
       + sum(j) b_product[j] tensor kappa_j`,

in m^-1. Burgers and line vectors are both expressed in the laboratory frame.
The plastic convention remains `alpha = -Curl(beta_p)`.

## Transport-and-capture route

Signed mobile density and its alignment moment use the same periodic
donor-cell face event. A crossing is captured only when its donor lies outside
and its receiver lies inside a caller-declared physical trap. The trap API has
no orientation, Frank--Bilby target, phase, or grain-label argument. The
multidimensional Courant sum is required to be at most one.

The ledger reports signed captured line, captured alignment, global scalar and
alignment closure, local Nye change, and the integrated Nye residual. Captured
tangle can mature into ordered content only by moving the same scalar line and
alignment; an unpolarized tangle therefore cannot become polarized merely by
changing its reservoir name.

## Explicit topology route

A junction event consumes equal signed tangle extents from two declared
parents and creates its declared product. Each topology records parent signs,
product Burgers vector, parent and product line directions, product-line
multiplicity, and line free-energy change. Frank's rule and node-line closure
are checked independently. The balance is

`d(alpha)/dt + Curl(J_alpha) = R_topology`.

The implementation returns `R_topology` explicitly; it never calls a change
in tensorial content “conservative” merely because scalar density is balanced.
The scalar line source `(m_product-2)*extent`, vector Burgers residual, free
energy change, and irreversible heat are separate fields.

The single-line cross-slip/climb comparator reorients only a finite segment of
declared event length. A forward event creates a pair of turning nodes and
records the integrated turning curvature in m^-3; its reverse consumes the
same stored node and curvature inventories and restores the declared glide-line
direction. Scalar line and signed Burgers family are unchanged, while the
resulting tensorial change is exposed as a separate `R_topology`. These
node/curvature fields are checkpointed state, not transient diagnostics.

## Kinetics

The extensive ordering attempt rate now calls the campaign-wide EXP-floor
enthalpy and signed-entropy kernel. Activation entropy is applied once in
`Delta G* = Delta H*_EXP-floor - T Delta S*`. Every process declares either a
drag-limited or rejection policy for a nonpositive free barrier.

Finite-segment forward and reverse rates share that same attempt rate and use
stable logistic affinity factors. Consequently
`k_forward/k_reverse = exp(-Delta F_event/(k_B T))` pointwise; the free-energy
affinity selects direction while the EXP-floor barrier controls timescale.

## Independent qualification

Wall qualification is postprocessing-only. Connected orientation-gradient
segments receive local normals/tangents, two-sided plateau orientations,
local misorientation, fixed-physical-window Nye circuit integrals,
Frank--Bilby closure, supply ratio, ordered-line overlap/outside support,
length, and release persistence. In this 2.5-D convention the relevant circuit
integrates the retained out-of-plane line column of `alpha` through the wall
normal. The global orientation span is diagnostic only.

The manufactured target helper remains a representation fixture. It is not an
evolution source and cannot establish independent mechanism validation.

## Current scientific status

The representation, transport capture, explicit junction ledger, restart, and
local circuit diagnostics are implemented and locally tested. A persistent
mechanically generated LAGB has not yet been demonstrated. Phase and grain
allocation remain disabled for this pathway.
