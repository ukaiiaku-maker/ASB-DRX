# Burgers-resolved state and balance

Status: convention and conservation contract for the reduced plane-strain
research architecture. Reaction rates are not yet calibrated.

## State

The proposed state is

\[
\{F^p,R,\rho^a_{m+},\rho^a_{m-},\rho^r_j,\alpha,
\varrho^a_{GB},\eta,z_{mh},T\}.
\]

For the BCC demonstration, the four unoriented Burgers families are represented
by normalized directions `[111]`, `[1 -1 1]`, `[-1 1 1]`, and `[1 1 -1]`,
with magnitude `b/2 * sqrt(3)`. Sign is stored separately. Each family uses an
instantaneous maximum-resolved-shear plane in the material-point verification.
This is a reduced projection, not a complete 3-D BCC junction topology.

| State | Units | Type | Admissibility |
|---|---:|---|---|
| `F^p` | 1 | kinematic history | positive determinant |
| `R` | 1 | orientation | `R^T R=I`, `det R=1` |
| `rho^a_m+`, `rho^a_m-` | m^-2 | transported signed line populations | nonnegative |
| `rho^r_j` | m^-2 | reaction-resolved junction/forest population | nonnegative |
| `alpha` | m^-1 | derived Nye tensor | curl-compatible with selected `F^p` convention |
| `varrho^a_GB` | m^-1 on interface (or m^-2 diffuse volumetric equivalent) | transported/reacting boundary line content | signed; unit convention recorded |
| `eta` | 1 | nonconserved crystallinity/interface field | `[0,1]` |
| `z_mh` | closure dependent | DD memory state | closure validity envelope |
| `T` | K | energy balance | positive |

## Transport and reactions

For mobile sign `s in {+1,-1}`,

\[
\partial_t\rho^a_{m,s}+\nabla\cdot(\rho^a_{m,s}v^a_s)
=S^a_{mult,s}-S^a_{ann,s}-S^a_{lock,s}
+S^a_{unlock,s}-S^a_{GB,s}.
\]

The boundary and junction equations receive the equal and opposite capture,
locking, and unlocking transfers. Pair annihilation removes equal line content
from opposite signs of the same compatible family. Multiplication and true
annihilation are the only terms permitted to change total line content; all
other reactions must close a line-content ledger.

A reaction `r` uses integer stoichiometry `nu_{ir}` over oriented Burgers
vectors `b_i` and is admissible only if

\[
\sum_i \nu_{ir} b_i=0.
\]

No 2-D reaction table is authorized yet. It must be projected from a documented
3-D BCC reaction catalog, and every row must pass the vector residual test.

## Nye convention

The first implementation will use small strain with

\[
\beta^p=\sum_a \gamma^a s^a\otimes m^a,
\qquad \alpha=-\operatorname{Curl}\beta^p,
\qquad
\alpha_{ij}=-\epsilon_{jkl}\partial_k\beta^p_{il}.
\]

At finite strain this equation must be replaced consistently throughout; a
mixture of reference- and current-configuration curls is prohibited. The
discrete curl operator must annihilate compatible gradient fields and its
periodic integral must close to numerical tolerance.

## Frank--Bilby convention

For an interface tangent/probe vector `p` and relative rotation
`R_ij = R_j R_i^T`, define

\[
B(p)=(I-R_{ij}^{-1})p.
\]

The resolved boundary inventory satisfies

\[
B(p)=\sum_a b^a N^a(p),
\quad
r_{FB}=\frac{\|B-\sum_a b^aN^a\|}
{\max(\|B\|,b/L_{probe})}.
\]

The sign/reference convention, probe length, active Burgers basis, and
regularization in the denominator are checkpointed. A boundary cannot be
accepted as a LAGB unless `r_FB` is below a resolution-verified tolerance.

## Required ledgers

1. **Burgers vector:** transport changes boundary flux only; each internal
   reaction closes its vector stoichiometry.
2. **Line content:** mobile/junction/boundary transfers close exactly; named
   multiplication and annihilation are reported separately.
3. **Energy:** line, correlation, and boundary energy changes enter the same
   mechanical-work/heat ledger.
4. **Orientation compatibility:** `alpha`, `grad R`, and boundary inventory are
   compared rather than assumed equivalent.

Same total density with a different signed/family mixture is therefore a
different state. Conversely, balanced density clustering with zero curvature
does not create a crystallographic boundary.
