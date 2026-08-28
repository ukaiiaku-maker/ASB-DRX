# Validation and verification plan

## Evidence separation

- DD calibration/validation selects and freezes the collective closure independently.
- Material/GB/thermal/mechanical datasets have versioned provenance and are separate from PF calibration targets.
- PF calibration and validation conditions are disjoint. One resolved equation/parameter set applies everywhere.
- Material selection is gated by `material_target_audit.md`; no mixed Fe/Cr/alloy parameter set is allowed.

## Gate 0 regressions

On HPC3, reproduce from exact source/configuration: one v32 ASB-like case, one v33 label-explosion negative control, and one v34 candidate-without-promotion case. Fetch and checksum all data. Build a common post-processor that does not recognize a hazard, candidate, or allocated label as a grain.

## Closure verification

At material points, held-out conditions must reproduce mean rate plus survival/hazard, CV/Fano or renewal shape, amplitude distribution, temporal/spatial correlations available in DD, seed uncertainty, and censoring. Lock artifact hash before PF calibration. Tests prevent mutation of DD parameters.

## Thermodynamic/numerical verification

- dimensions and SI schema;
- unloaded isothermal nonincreasing discrete free energy;
- dislocation/Burgers reservoir ledgers with known conserved cases;
- mechanical work partition closure to about 1% or tighter justified tolerance;
- subcritical shrinkage and supercritical growth/rate in the circular-nucleus limit if embryos are used;
- zero-coupling and homogeneous analytical/material-point limits;
- continuous versus segmented restart including collective/embryo/RNG/controller states;
- timestep and grid refinement, provisionally less than 5% final-step changes in primary observables;
- output/checkpoint schema and provenance tests;
- label-only events never alter physical-grain count;
- no-nucleation controls keep physical-grain count fixed.

## Common-equation mechanism ladder

1. unloaded isothermal relaxation;
2. isothermal deformation with grain evolution disabled;
3. isothermal DRX with thermal localization suppressed by a physical high-conductance boundary;
4. thermal high-rate with nucleation disabled;
5. coupled intermediate-rate;
6. coupled high-rate.

## Production staging

Only after Gates 0--5, run a sparse matrix selected from actual validity envelopes (provisionally three temperatures, five log rates, one microstructure/stochastic seed). Locate boundaries, then use at least three independent seeds near boundaries and representative regimes. Keep microstructure, closure, embryo, and perturbation RNG streams distinguishable.

## Classification

- Recovery/GB relaxation: no persistent new physical grain and no sustained thermomechanical localization.
- DRX: finite phase-field support relative to interface width, purity, orientation/provenance, persistence, reduced stored energy versus parent, and growth/stable survival.
- ASB: sustained plastic-rate/work concentration, significant local temperature rise above controls/noise, associated softening, and finite width exceeding numerical interfaces and converging with mesh.
- Coupled: both criteria, with temporal/spatial ordering reported rather than forced exclusivity.

Raw observables and uncertainty are retained; no single OR condition assigns a regime.
