# Independent DD-constrained PF DRX/ASB campaign

Campaign ID: `asb-drx-independent-20260827`

This branch is a scientific redevelopment. Legacy v32--v34 sources and outputs are evidence and regression references; they are not the production architecture. The new implementation will live under `src/asb_drx/` and will use one equation set and one parameter set across temperature and strain rate.

## Immutable interpretation

- The Arrhenius--Taylor peak and negative slope are kinetic diagnostics, not a DRX trigger or a free-energy term.
- The reported DD Poisson-to-coordinated transition is an external constitutive constraint. It cannot be tuned against phase-field outcomes.
- Collective-event state may alter kinetics, storage, correlation, intermittency, and finite-fluctuation sampling, but cannot directly create a grain.
- A grain requires finite order-parameter support, orientation/provenance, persistence, and growth or stable survival.
- ASB requires converged thermomechanical localization, not a weak scalar OR test.

## Gate sequence

0. Repository isolation, evidence inventory, exact DD dataset identification, legacy reproductions.
1. Standalone DD closure with held-out first- and higher-moment validation; freeze and hash the artifact.
2. Thermodynamic, dimensional, relaxation, content-ledger, work/energy, and nucleus-limit verification.
3. Time/grid convergence, restart equivalence, schema tests, and physical-grain invariants.
4. Common-equation mechanism ladder.
5. Sparse one-seed temperature/rate map.
6. Boundary/representative ensembles, uncertainty propagation, and validation.

No later gate is authorized scientifically by a failed earlier gate. Failed conservation, thermodynamic, restart, convergence, or validation tests will be corrected rather than tuned around.

## Repository layout

- `docs/`: audit, derivation, validation, and decision records.
- `evidence/`: machine-generated inventory metadata; raw project evidence remains immutable at its recorded source paths.
- `src/asb_drx/`: new modular implementation (not yet started at Gate 0).
- `tests/`: tests whose numerical execution is submitted to HPC3.
- `.hpc3/jobs/`: versioned job specifications.
- `hpc3-results/asb-drx-independent/`: fetched run data, excluded from Git where large.

## Reproducibility rule

Every numerical run must resolve SI configuration into a manifest, identify the Git/source/DD hashes and independent RNG streams, record Slurm/environment/resource provenance, inventory outputs, and verify fetched checksums. Numerical simulations and numerical tests run only on UCI HPC3.
