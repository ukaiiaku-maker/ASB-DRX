# Gate A: Bertin BCC material-point reference

## Status and purpose

This is a quarantined reproduction of the constitutive architecture in
Bertin, Carson, Bulatov, Lind, and Nelms, *Acta Materialia* 260 (2023) 119336,
[DOI 10.1016/j.actamat.2023.119336](https://doi.org/10.1016/j.actamat.2023.119336).
The open LLNL manuscript is archived by
[OSTI](https://www.osti.gov/servlets/purl/2005100); the PDF retrieved on
2026-09-08 has SHA-256
`7e8190b6c929e81ff39e4ca6577c3c587ebc8945b7ac43947daee2f5b8aba44d`.

The implementation verifies lattice-rotation physics required by campaign
Gate A. Its BCC-Ta fit is not a parameter source for the production DRX/ASB
model and is never mixed with the generic EXP-floor baseline.

## Implemented published equations

The total deformation gradient is decomposed multiplicatively,

\[
F=F^eF^p,\qquad L^p=\dot F^pF^{p-1}
=\sum_{i=1}^4\dot\gamma_i\hat b_i\otimes\bar n_i(\chi_i).
\]

The four modes are the four unoriented `1/2<111>` Burgers families. Their slip
normals are recomputed from the maximum-resolved-shear plane rather than chosen
from a fixed slip-plane list. Elastic rotation is obtained by polar
decomposition of `F^e`, so the physical lattice orientation follows directly
from the multiplicative kinematics.

The flow rule implements the paper's Eqs. 5--11:

- Orowan slip `dot gamma_i = rho_i b v_i`;
- power-law velocity with exponent 25;
- MRSSP-angle-dependent activation and T/AT velocity prefactor;
- scalar Taylor resistance `0.3 mu b sqrt(sum rho_i)`;
- phonon-drag velocity cap.

The density balance implements Eqs. 12 and 14--16:

\[
\dot\rho_i=(k_1(\chi_i)\sqrt{\rho_i}-k_2(\dot\gamma,T)\rho_i)
|\dot\gamma_i|-f_i k_{relax}\rho_i.
\]

The frozen-coefficient density substep is integrated analytically in
`sqrt(rho_i)`, preserving positivity without clipping. Plastic deformation is
advanced by a matrix exponential. The prescribed reference path is isochoric
uniaxial true strain along the laboratory z axis.

## Reference envelope

The published Ta constants from Tables 2--3 are represented explicitly. Cubic
elastic constants are linearly interpolated only over 50--1000 K, and axial
rate requests outside `1e6`--`3e8 s^-1` stop. All Gate A cases use 300 K and
`2e8 s^-1`. This validity enforcement is part of the model, not a warning
buried in post-processing.

The initial per-family densities are close to the published
`4.8e14 m^-2`; small declared family differences and orientation perturbations
break exact crystal symmetry. They are numerical perturbations, not fitted
microstructures.

## Acceptance tests

Gate A requires all of the following with the published equation set:

1. `[001]` and `[111]` compression and `[101]` tension remain within one
   degree of their stable cubic families through unit true strain.
2. `[419]` compression moves toward `<111>` and `[419]` tension toward
   `<101>`.
3. `[111]` tension moves toward `<101>`; suppressing plastic spin removes this
   attractor motion.
4. An inactive Burgers family loses density, while active families grow; the
   inactive loss disappears when its distinct relaxation term is disabled.
5. Plastic-spin and inactive-relaxation ablations each change axial stress, so
   orientation and density effects are separable.
6. The full state restarts bitwise exactly.
7. Final stress, total density, and attractor angle change by less than 5%
   between strain increments `1e-3` and `5e-4`.

The local verification passes all nine targeted tests and the complete suite
passes 160 tests. The final refinement changes are 0.1243% in stress, 0.1621%
in total density, and 0.1723% in attractor angle.

## Interpretation boundary

This passes the material-point rotation gate qualitatively against published
stable/unstable directions and mechanistic ablations. It does not reproduce
digitized MD curves, because no immutable numerical trajectory table was
provided by the paper. It also does not establish Fe behavior, spatial GND
content, a wall, a grain boundary, DRX, or ASB. Those claims remain gated by
the signed spatial transport and Frank--Bilby tests.
