# Literature mechanism matrix

Status: design review for the v2 physics addendum. This document constrains a
future Burgers-resolved production architecture; it does not calibrate a
material model.

## Evidence rules

- Project DD trajectories constrain only observables they actually record.
- Literature supplies mechanism form and possible calibration routes, not
  transferable constants unless the material and regime match.
- The existing net EXP-floor antiplane model is a verified scalar baseline. It
  has no crystallographic wall, lattice-rotation, or Frank--Bilby claim.
- A full constitutive term remains disabled until its state, force, kinetic
  coefficient, validity envelope, balance, and ablation are independently
  specified.

## Matrix

| Mechanism | Required state | Thermodynamic force | Kinetic mechanism | Temperature dependence | Rate dependence | Supporting source | Parameterization route | Current representation | Deficiency | Proposed correction |
|---|---|---|---|---|---|---|---|---|---|---|
| Trapping | signed mobile density by Burgers family; obstacle/junction inventory | Peach--Koehler force and line-energy change | glide flux into persistent contacts | glide mobility and obstacle activation | flux and residence-time dependent | Humphreys--Rohrer--Rollett (2017); Bertin et al. (2024), DOI [10.1016/j.actamat.2024.119884](https://doi.org/10.1016/j.actamat.2024.119884) | targeted DDD/MD contact creation and residence histories | scalar forest storage proportional to plastic increment | no sign, family, topology, or conservative transfer | conservative mobile-to-junction reaction channels |
| Annihilation | opposite signed mobile/boundary populations | line/correlation-energy decrease | coplanar glide, cross-slip, and climb reactions | glide weak-to-moderate; cross-slip material dependent; climb diffusion activated | encounter-flux dependent | Humphreys--Rohrer--Rollett (2017) | separate glide/cross-slip/climb datasets or bounded sensitivity | one Arrhenius scalar recovery time | merges distinct mechanisms and cannot preserve GND | stoichiometric pair reactions with separate mobilities |
| Junctioning | mobile families and junction population by reaction type | line plus junction energy and resolved force | binary reactions satisfying Burgers conservation | mobility and reaction-barrier dependent | collision-flux dependent | Bertin et al. (2024), DOI [10.1016/j.actamat.2024.119884](https://doi.org/10.1016/j.actamat.2024.119884) | 3-D DDD/MD reaction catalog and held-out rates | scalar forest reservoir | no reaction type or Burgers inventory | projected 3-D reaction matrix; reject invented 2-D reactions |
| Wall formation | signed density, Nye tensor, plastic distortion, nonlocal stress | variation of elastic/correlation energy plus frictional transport force | stress-driven dislocation transport and screening | through mobility/recovery, not an ad hoc ordering temperature | loading controls flux instability | Wu et al. (2018), DOI [10.1103/PhysRevB.98.054110](https://doi.org/10.1103/PhysRevB.98.054110) | signed-density structure factors and finite-wavelength stability | no wall state; scalar density perturbations only | cannot distinguish a density band from a compatible wall | Burgers-resolved transport with nonlocal or derived gradient correlation energy |
| Polygonization | boundary/mobile signed content and orientation gradient | line/correlation-energy reduction subject to Burgers compatibility | glide redistribution, cross-slip, climb, boundary reactions | strongly route dependent; climb diffusion activated | competes with loading time | Humphreys--Rohrer--Rollett (2017) | recovery experiments/DDD with signed spatial fields | scalar recovery plus phase relaxation | may lower density but cannot sharpen a LAGB | conservative wall capture plus separate recovery channels |
| Plastic spin | plastic distortion and lattice orientation | crystal resolved stress through slip power | `W^p = skw(L^p)` from active slip modes | inherited from slip mobilities | instantaneous slip-activity competition | Bertin et al. (2023), DOI [10.1016/j.actamat.2023.119336](https://doi.org/10.1016/j.actamat.2023.119336) | BCC orientation trajectories and family-resolved slip | two scalar order parameters; no crystal kinematics | orientation can evolve independently of slip | four-family BCC pencil-glide material point followed by spatial CP |
| Progressive lattice rotation | orientation, plastic spin, GND/Nye tensor | incompatibility and resolved slip forces | heterogeneous slip paths and recovery preserving boundary excess | recovery and slip-family dependent | accumulated-strain and loading-path dependent | Humphreys--Rohrer--Rollett (2017); Ask et al. (2018), DOI [10.1016/j.jmps.2018.03.006](https://doi.org/10.1016/j.jmps.2018.03.006) | orientation/GND maps during CDRX | absent | scalar phase support cannot create compatible misorientation | couple lattice rotation to plastic spin and recognize persistent compatible walls |
| LAGB migration | interface, misorientation, boundary Burgers content, curvature | stored-energy jump, capillarity, stress, boundary self-energy | boundary-dislocation glide/climb/reaction | mobility mechanism dependent | competes with deformation and recovery | Zhang, Qin & Xiang (2026), DOI [10.1137/24M1712618](https://doi.org/10.1137/24M1712618) | bicrystal mobility and dynamic Frank--Bilby benchmarks | isotropic phase mobility | no boundary defect content or compatibility | dynamic Frank--Bilby constrained boundary law |
| Shear-coupled rotation | physical boundary, disconnections, orientation, elastic stress | Onsager forces for normal/tangential motion and defect reactions | disconnection flow | mobility and mode dependent | stress and migration-rate dependent | Qiu et al. (2024), DOI [10.1073/pnas.2310302121](https://doi.org/10.1073/pnas.2310302121); Qiu et al. (2025), DOI [10.1073/pnas.2500707122](https://doi.org/10.1073/pnas.2500707122) | bicrystal shear-coupling factors and rotation histories | absent | curvature-only phase motion misses internal stress and rotation | add only after Gate D validates static and dynamic compatibility |
| DDRX growth | closed physical interface, orientation, stored-energy contrast | stored-energy pressure minus capillarity plus mechanical terms | boundary migration/bulging | GB mobility and recovery dependent | growth must outrun loading/localization | Tandogan, Budnitzki & Sandfeld (2026), DOI [10.1016/j.jmps.2025.106325](https://doi.org/10.1016/j.jmps.2025.106325) | boundary bulging/subgrain benchmarks; material GB data | stateful circular embryo gate and binary phase growth | orientation is assigned, not produced by crystal kinematics | retain embryo fixture only as an ablation; prefer boundary-origin growth tests |
| ASB | stress/strain, local slip, storage reservoirs, temperature, conduction | mechanical power and thermal constitutive feedback | local stress redistribution plus heat/storage/recovery competition | all properties and kinetics as functions of temperature | instability occurs when localization outruns diffusion/recovery | established ASB literature plus project v32 regression | matched mechanical/thermal data and finite-wavelength analysis | verified antiplane scalar no-localization baseline | couple resolved crystal transport and thermal mechanics; require converged band width/onset |

## Architecture decision

The next minimal research model is a reduced plane-strain, four-Burgers-family
BCC transport/orientation model. It must retain plastic distortion, physical
lattice orientation, signed mobile densities, reaction-resolved junction
content, Nye/GND content, boundary Burgers content, a crystallinity/interface
field, temperature, and (only when identifiable) a DD memory state. The
existing scalar density and multi-order modules remain regression fixtures.

No production parameter set is authorized by this matrix. In particular,
Bertin's Ta high-rate calibration cannot be silently relabeled as Fe, and the
single-glider DD data cannot calibrate wall formation.
