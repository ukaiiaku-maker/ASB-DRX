# Rotation mechanism separation

## Scope

Three rotations act on different state and must never be represented by one
scalar relaxation coefficient.

## 1. Deformation-induced lattice rotation

For slip modes `a`,

\[
L^p=\sum_a \dot\gamma^a s^a\otimes m^a,\qquad
W^p=\operatorname{skw}L^p,
\]

and the selected mechanical kinematics determine lattice spin. In the
small-elastic-strain verification limit,

\[
\dot R R^T=W-W^p.
\]

This term acts throughout crystalline bulk. Its inputs are the slip rates and
current orientation; it cannot read an embryo age, phase label, or boundary
curvature. The BCC reference uses four `1/2<111>` Burgers families and an MRSSP
plane per family, following the pencil-glide motivation of
[Bertin et al.](https://doi.org/10.1016/j.actamat.2023.119336).

**Ablation proof.** Setting every `dot gamma^a=0` gives `L^p=W^p=0`; for a
nonrotating frame, `dot R=0`. Setting the plastic-spin contribution to zero
removes attractor selection driven by differential slip while leaving the
other two boundary-localized mechanisms unchanged.

Inactive-family density relaxation is a separate kinetic term. It may respond
to loss of relative slip activity, but it is not climb and it requires an
atomistic/DD calibration or a declared sensitivity bound.

## 2. Progressive CDRX rotation

This is not an extra local spin law. It is the spatial consequence of
heterogeneous deformation-induced rotation together with signed transport and
selective recovery:

\[
\alpha=-\operatorname{Curl}\beta^p,
\qquad \kappa_R\sim \nabla R.
\]

Different slip histories create orientation curvature. Recovery may annihilate
redundant `+/-` content while preserving the excess required by `alpha`, and
wall capture may localize that excess. A persistent LAGB is recognized only
after its interface, orientation jump, boundary Burgers inventory, and
Frank--Bilby residual are resolved.

**Ablation proof.** Uniform slip activity gives spatially uniform `R` and hence
no orientation-gradient wall. If `W^p` is disabled, the progressive
orientation gradient disappears even if a scalar density pattern survives. If
signed transport is disabled, the model has no content with which to satisfy
Frank--Bilby compatibility and cannot promote a LAGB.

## 3. Boundary-mediated rotation

This acts only on an already recognized physical interface. Its state contains
boundary line defects/disconnections, misorientation, normal and tangential
velocity, and elastic stress. A dynamic Frank--Bilby constrained Onsager law
couples defect reactions, normal migration, tangential translation, and
misorientation. Relevant references are
[Qiu et al. (2024)](https://doi.org/10.1073/pnas.2310302121) and
[Zhang, Qin & Xiang (2026)](https://doi.org/10.1137/24M1712618).

**Ablation proof.** Multiplying the shear-coupling block by `c_sc` makes both
tangential coupling and its boundary-rotation contribution vanish exactly at
`c_sc=0`, while bulk plastic-spin rotation remains. Setting boundary mobility
to zero suppresses migration-coupled rotation without suppressing a
deformation-induced orientation gradient.

## Non-overlap contract

| Mechanism | Spatial support | Creates orientation through | Required precursor | Must vanish when |
|---|---|---|---|---|
| Plastic-spin rotation | crystalline bulk | slip kinematics | active slip | plastic spin is disabled |
| Progressive CDRX | heterogeneous bulk/wall | spatially different plastic-spin histories plus compatible signed content | slip heterogeneity | plastic spin or signed compatibility is disabled |
| Boundary-mediated rotation | accepted interface | disconnection/boundary-defect kinetics | physical boundary | shear coupling or boundary mobility is disabled |

The production code must expose three separate diagnostic rates and three
separate ablations. Adding them into one undifferentiated `orientation_rate`
would make the falsification tests impossible.
