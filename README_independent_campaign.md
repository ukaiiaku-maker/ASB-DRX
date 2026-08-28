# Analytical EXP-floor PF DRX/ASB campaign

Campaign ID: `asb-drx-independent-20260827`

This branch is a scientific redevelopment. Legacy programs, parameters, and outputs are context only: they do not calibrate, validate, or regression-gate the new model. The implementation lives under `src/asb_drx/` and will use one equation set and one parameter set across temperature and strain rate.

## Immutable interpretation

- The Arrhenius--Taylor peak and negative slope are kinetic diagnostics, not a DRX trigger or a free-energy term.
- Dislocation dynamics does not parameterize the model.
- The independent-node EXP-floor barrier and its closed-form rate--temperature strength peak are the constitutive baseline.
- A collective transparent-node extension is admitted only if a stress-transfer/memory derivation and discriminating observations justify it.
- Collective-event state may alter kinetics, storage, correlation, intermittency, and finite-fluctuation sampling, but cannot directly create a grain.
- A grain requires finite order-parameter support, orientation/provenance, persistence, and growth or stable survival.
- ASB requires converged thermomechanical localization, not a weak scalar OR test.

## Gate sequence

0. Derive and independently verify the EXP-floor inverse, activation volume, and analytical peak.
1. Select an authoritative material and strength/rate/temperature dataset; optimize the analytical parameters with identifiability and held-out tests.
2. Test whether collective observables reject the independent-node baseline; if so, derive and validate the smallest stress-transfer/memory extension.
3. Verify thermodynamics, dimensions, content ledgers, work/energy closure, and nucleus limits.
4. Verify time/grid convergence, restart equivalence, schema, and physical-grain invariants.
5. Run the common-equation mechanism ladder and sparse temperature/rate map.
6. Run boundary/representative ensembles, uncertainty propagation, and external validation.

No later gate is authorized scientifically by a failed earlier gate. Failed conservation, thermodynamic, restart, convergence, or validation tests will be corrected rather than tuned around.

## Repository layout

- `docs/`: audit, derivation, validation, and decision records.
- `evidence/`: machine-generated inventory metadata; raw project evidence remains immutable at its recorded source paths.
- `src/asb_drx/`: new modular analytical and, later, continuum implementation.
- `tests/`: tests whose numerical execution is submitted to HPC3.
- `.hpc3/jobs/`: versioned job specifications.
- `hpc3-results/asb-drx-independent/`: fetched run data, excluded from Git where large.

## Reproducibility rule

Every numerical run must resolve SI configuration into a manifest, identify the Git/source/DD hashes and independent RNG streams, record Slurm/environment/resource provenance, inventory outputs, and verify fetched checksums. Numerical simulations and numerical tests run only on UCI HPC3.
