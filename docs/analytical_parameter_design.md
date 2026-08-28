# Governing-equation parameter design

The EXP-floor parameters remain the authorized single-glider DDD
parameterization. They are not retuned by the continuum model. The first full
stability calculation exposed a structural omission instead: storage alone has
no competing sink. On the post-peak density branch, a compatible isothermal
mode has growth rate

\[
\lambda_\rho=K\,\partial r/\partial\rho.
\]

Its sign is independent of thermal conductivity and phase mobility, and its
magnitude merely scales with the storage coefficient. Consequently no honest
optimization of the existing parameters can create a temperature/rate regime
boundary; it can only speed up or slow down the same instability. At the exact
strength peak this mode is neutral. This also explains why a visually chosen
parameter sweep would be non-identifiable.

The minimal missing competition is dynamic recovery. For

\[
\dot\rho=K r-(\rho-\rho_{eq})/\tau_{rec}(T),
\qquad
\tau_{rec}^{-1}=\tau_0^{-1}\exp[-Q_{rec}/k_B(1/T-1/T_0)],
\]

the compatible isothermal density eigenvalue becomes

\[
\lambda_\rho=K\,\partial r/\partial\rho-\tau_{rec}^{-1}(T).
\]

Two declared neutral boundary points determine `Q_rec` and `tau_0` exactly.
This is an inverse solution of the governing equation, not a fit to the invalid
legacy data and not a modification of the DDD flow law.

Because the user permits an arbitrary generic boundary, the provisional design
anchors are `(850 K, 450 s^-1)` and `(1050 K, 45000 s^-1)`, both at density
ratio 2 relative to the net EXP-floor peak. With the inherited generic storage
coefficient and reference temperature 950 K, the exact solution is
approximately `Q_rec=1.138 eV` and `tau_rec(950 K)=1.856 s`. This single law
makes the same post-peak state stable at 950 K and 450 s^-1 and unstable at
950 K and 45000 s^-1. These values define a generic numerical hypothesis, not
a material calibration. The anchors remain explicit inputs and can later be
replaced by experimental or literature constraints without changing the
derivation.

The next numerical gate must integrate recovery with an exact stored-energy
release/heat ledger and then test whether nonlinear outcomes converge. A
finite wavelength still cannot be selected by local recovery alone; thermal
diffusion, antiplane compatibility, and any later evidence-based nonlocal
collective length must be assessed separately.
