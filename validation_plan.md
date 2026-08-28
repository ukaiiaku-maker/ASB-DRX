# Validation and verification plan

## Evidence separation

- The analytical EXP-floor law is verified first and calibrated only to an authoritative material strength/rate/temperature dataset.
- A collective extension is tested separately and only if discriminating observations reject the independent baseline; DD is not a parameter source.
- Material/GB/thermal/mechanical datasets have versioned provenance and are separate from PF calibration targets.
- PF calibration and validation conditions are disjoint. One resolved equation/parameter set applies everywhere.
- Material selection is gated by `material_target_audit.md`; no mixed Fe/Cr/alloy parameter set is allowed.

## Context audit already completed

Historical v32--v34 controls were reproduced and archived, but the clarified campaign treats them as context only. They impose no regression requirement on the new model.

## Analytical and parameter verification

Verify barrier limits, dimensions, activation-volume derivative, forward/inverse closure, Lambert-W branch and stationarity, peak-existence criterion, and exact fixed-temperature rate scalings on HPC3. Then use synthetic recovery to test optimizer identifiability without claiming physical validity. Physical fitting uses condition-grouped held-out strength data and reports parameter correlations and validity limits.

If a collective extension is attempted, it must additionally predict relevant transient/burst/correlation observations on held-out conditions, satisfy the independent limit, and expose the branching spectral radius and memory timescale.

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
- sub-resolution or disconnected support never becomes one physical grain;
- promotion requires persistent pure support, valid lineage, and symmetry-reduced misorientation;
- loss of support removes a grain from current counts before provenance-preserving retirement;
- tracker checkpoint/restart preserves every lifecycle field exactly.

## Common-equation mechanism ladder

1. unloaded isothermal relaxation;
2. isothermal deformation with grain evolution disabled;
3. isothermal DRX with thermal localization suppressed by a physical high-conductance boundary;
4. thermal high-rate with nucleation disabled;
5. coupled intermediate-rate;
6. coupled high-rate.

## Production staging

Only after analytical verification, target selection, parameter validation, and thermodynamic/numerical gates, run a sparse matrix selected from actual validity envelopes. Locate boundaries, then use at least three independent seeds near boundaries and representative regimes. Keep microstructure, collective, embryo, and perturbation RNG streams distinguishable.

## Classification

- Recovery/GB relaxation: no persistent new physical grain and no sustained thermomechanical localization.
- DRX: finite phase-field support relative to interface width, purity, orientation/provenance, persistence, reduced stored energy versus parent, and growth/stable survival.
- ASB: sustained plastic-rate/work concentration, significant local temperature rise above controls/noise, associated softening, and finite width exceeding numerical interfaces and converging with mesh.
- Coupled: both criteria, with temporal/spatial ordering reported rather than forced exclusivity.

Raw observables and uncertainty are retained; no single OR condition assigns a regime.
