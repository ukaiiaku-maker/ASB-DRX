# V67 multiplication normalization and accepted-throughput repair

## Scope

This repair preserves the common Arrhenius EXP-floor Peierls--Taylor--Hall--
Petch mechanics and the Hall--Petch event measure. It changes neither a rate
switch nor an ASB-specific parameter. The strict no-multiplication model is now
a comparator rather than the primary GB-origin model.

## Recovered law

The executable v34 reference, not a dimensional guess, defines the unsigned
Kocks--Mecking storage rate on family \(s\) as

\[
  \dot\rho^{\rm raw}_{\rm store,s}
  = k_1\sqrt{\rho_{{\rm eff},s}}\,|\dot\gamma_s|,
  \qquad k_1=5.0\times10^8\ {\rm m}^{-1}.
\]

The production signed representation adds half of that rate to each forest
sign. Thus the Burgers-vector sum is unchanged while unsigned stored line rises
by the reference rate. There is no additional \(1/b\): \(k_1\),
\(\sqrt\rho\), and \(|\dot\gamma|\) already give units of
\({\rm m}^{-2}{\rm s}^{-1}\). The saved value also gives the useful
dimensionless check \(k_1b=0.124\) for \(b=2.48\times10^{-10}\) m.

The recovered obstacle argument is

\[
 \rho_{{\rm eff},s} = 0.65\rho_{{\rm forest},s}
 +0.35\sum_r\rho_{{\rm forest},r}
 +0.10\rho_{{\rm mobile},s}
 +0.35\sum_r\rho_{{\rm wall},r}
 +0.10\left|\sum_r(\rho^+_{{\rm mobile},r}
                         -\rho^-_{{\rm mobile},r})\right|.
\]

The archived diffuse GB-shell contribution is intentionally not copied: the
common-owner model already carries stationary interface line explicitly, so
including the shell again would double count boundary content as bulk line.

The accepted rate is bounded *before* integration by the available stored-work
fraction,

\[
  (\mu^+_{\rm forest,s}+\mu^-_{\rm forest,s})
  \dot\rho^+_{\rm store,s}
  \le 0.1\sum_r |\tau_r|\,|\dot\gamma_r|.
\]

Raw and accepted line increments, the active budget fraction, and signed and
absolute slip increments are accumulated over every accepted internal solver
step. Restart files serialize the law, coefficient, work fraction, obstacle
weights, and destination reservoir.

## Reference and regression evidence

- `full_model/reference_sources/v34/drx_var_v34_candidate_drx_asb_sweep.py`
  contains `KM_k1=5.0e8`, `stor_raw=KM_k1*sqrt(rf)*abs(gdot)`, the 0.1 stored
  work fraction, and `rho_state_store_to_forest=True`.
- Its saved run configuration does not override those defaults.
- Git history shows that coefficient 10 entered V21 as a generic verification
  default, without a material normalization derivation.
- Manufactured scalar integration converges to
  \([\sqrt{\rho_0}+k_1\gamma/2]^2\).
- Tests verify sign symmetry, exact obstacle weights, forest routing, work-cap
  activation, exact zero behavior, restart schema inheritance, and inherited
  front transmission.

The state audit is machine-readable in
`v67_multiplication_normalization_routing_audit.json`. On the saved ad400,
ad500, and ad600 states, the recovered-to-legacy accepted-line ratios are
\(6.06\times10^7\), \(6.35\times10^7\), and \(6.62\times10^7\). The
strict comparator accepts exactly zero line, and every probe consumes the full
physical interval.

## Production interpretation

The repaired model stores line in the forest reservoir; it does not create a
glissile mobile reservoir by relabeling stored content. Consequently, a
positive multiplication result can preserve or redistribute accepted flow only
through the coupled stress, slip, reaction, and boundary-transport trajectory.
It cannot manufacture carrier throughput directly.

At the matched 0.13064-shear endpoint, the repaired trajectory has accepted
approximately \(5.01\times10^{-6}\) m of new forest line and retains 0.1813%
accepted plastic/imposed power. It remains unbanded at the endpoint and has no
post-peak softening. Final quantitative classification awaits completion of the
strict zero comparator through the identical interval.

