# Common-state and moving-front formulation (v9)

## State and common free energy

The production opt-in state for one promoted pair is sparse: virgin parent
`p`, active child `c`, recovered wake `w`, current child fraction `chi`, and
maximum historically swept fraction `m`. The physical reservoir is

`rho_bar = (1-m) rho_p + chi rho_c + (m-chi) rho_w`.

Thus retreat replaces child material by recovered wake, not virgin parent.
No label lineage changes a constitutive sign. With
`h(eta)=eta^2(3-2 eta)` and `chi_i=h_i/sum(h_j)`, the common stored term is

`f_s = sum_i h_i psi(z_i,T) / sum_i h_i`,

and its exact derivative is

`df_s/deta_i = h'_i (psi_i-f_s) / sum_j h_j`.

The full declared functional is

`F = integral [f_phase + f_interface + f_elastic + f_stored
               + f_compatibility + f_boundary] dV`.

Physical Taylor line energy is `0.5 mu(T)b^2 rho`. Forest/junction and wall
reservoirs use that line coefficient in the current first implementation;
their separate kinetic roles remain explicit. Long-range incompatibility is
the signed-GND/orientation term. GB residual/disconnection energy is the
Frank--Bilby boundary term. Stiff numerical compatibility penalties are
reported diagnostically and are never converted to heat.

## Front balance

For newly swept fraction `dchi=max(chi_new-m,0)`, parent line is partitioned
as

`dL_p = dL_c + dL_boundary + dL_pair_annihilation + dL_sink`.

The child reaction uses an EXP-floor free barrier

`DeltaG = H0[f+(1-f)exp(-a(sigma/sigmac)^n)] - T DeltaS`.

Entropy occurs once. The finite-step reaction probability is
`1-exp(-rate dt)` and is also bounded by resolved interface traversal
`v_n dt/w_interface`. Signed parent/child mismatch is stored in an explicit
per-slip boundary reservoir; it is never pair-annihilated. A neutral fraction
may be stored at the boundary, annihilated, or sent to the declared sink.
Line energy released by neutral annihilation is heat. External work obeys
`W_ext = DeltaF + Q + D_other`; numerical penalties are excluded from `Q`.

The maximum swept fraction prevents repeated processing during
advance/retreat cycles. Checkpoints serialize all three material states,
current and historical fractions, scalar boundary line, signed boundary
content, and the cumulative line/energy/heat ledger.

## Qualification disposition

The kinematically bounded replay from the frozen precursor rejects phase
initialization. At steps 6904--6906 the common transaction increases physical
free energy by about `4.8e-15 J`; the approximately `5.0e-15 J` compatibility
cost exceeds the finite one-step line/bulk relief. No label or cleanup commits.
The earlier unbounded ablation was rejected because it cashed the embryo age
into disk-wide cleanup and produced domain-scale evolution within a few steps.

Therefore the Route-B bulk precursor is not self-sustaining under the common
thermodynamics and bounded moving-front kinetics. The next admissible pathways
are HAGB bulging/SIBM, pre-existing subgrain/CDRX promotion, and
orientation-gradient-driven support, all using this common state and front
ledger.
