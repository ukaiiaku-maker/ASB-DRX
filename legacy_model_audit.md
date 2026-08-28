# First-pass legacy model audit (v32--v34)

Disposition updated 2026-08-28: context only. No legacy equation, parameter, output, or reproduced trajectory calibrates, validates, or regression-gates the new model.

Scope: static inspection of the supplied top-level v34 files and canonical sources/results under `/Users/sdillon/DRX-ASB/recrysyallization_PF-2D/shear_banding`. This is an evidence audit, not endorsement. Exact hashes are in `evidence_manifest.json`.

## Represented equations and state

The legacy Python driver is a monolithic 2-D periodic-grid model combining:

- multiple nonconserved grain order parameters `eta_i` and grain-slaved orientation fields;
- signed mobile slip densities `rp`, `rm`, optional forest and wall reservoirs, scalar total density `rho`, signed GND-derived fields, and GB residual density `rho_GB`;
- strain/plastic-slip histories and a scalar finite-loading macroscopic stress update;
- an explicit temperature field with conduction/bath loss and local plastic-work heating;
- local Arrhenius slip/event kinetics modified by Taylor barriers, collective-domain or Poisson-tail activity, GB transmission, stress redistribution, and caps;
- Cahn--Hilliard-like density evolution and Allen--Cahn/KWC-like grain/orientation updates;
- cumulative nucleation hazard and, in v34, cell-indexed candidate incubation before label allocation.

The effective mechanical control is a reduced constitutive/servo system, not a full equilibrium or dynamic momentum solve. Local stress heterogeneity is constructed and plastic rates are recomputed; the finite-loading branch updates macroscopic stress approximately as `sigma_dot = E_eff (edot_target - <edot_p>)` with damping and validity stops.

## Variational versus explicit pieces

| Sector | Legacy character | Audit finding |
|---|---|---|
| Grain/order parameters | Allen--Cahn/KWC-like variations of stitched energy terms | Partly variational, but grain insertion, wake resets, label allocation, grain-slaved orientation, caps, and periodic refreshes are explicit interventions. |
| Density patterning | CH-like update from a constructed potential plus gradient regularization | The potential mixes kinetic Arrhenius--Taylor constructions and stored-density ideas without a demonstrated work-conjugate thermodynamic derivation. A density instability does not create orientation. |
| Plastic slip | Explicit Arrhenius constitutive kinetics | Legitimately can sit outside free energy, but dissipation and work partition require a complete audit. Multiple rate/activity closures and caps change behavior. |
| Storage/recovery | Explicit Kocks--Mecking-like sources/sinks and optional reservoir partition | Useful mechanism candidates, but provenance, conservation across reservoirs, and dimensional consistency are incomplete. |
| Mechanics | Algebraic/servo finite loading plus heuristic heterogeneous redistribution | Not derived from force balance; quasi-static/dynamic validity and local energy consistency are unproved. |
| Thermal | Explicit diffusion/bath update with local work heating | Useful framework, but timestep caps, process-zone smoothing, and work partition determine localization. |
| Nucleation | Explicit hazard and insertion | Nonvariational finite event. v34 incubation is bookkeeping, not a stateful physical embryo. |

## Hard gates, floors, caps, kernels, and overrides

The source contains a large parameter dictionary with JSON overrides and permissive unknown-key behavior. Important interventions include:

- density floors/upper physical bounds and initial-density clipping;
- Arrhenius exponent clipping, rate floors/caps, stress/servo validity bounds, and thermal validity stops;
- branch/mode switches for collective law, rate closure, density partition, mechanics, heating, KWC, nucleation, and topology;
- nucleation viability tests on barrier, rate, and free-energy-density relief;
- candidate hold/decay counters and promotion selection;
- Gaussian filters for activity/nonlocal fields and a periodic heat process-zone kernel;
- minimum kernel width in pixels (`heat_process_zone_min_sigma_px=2`) in addition to the nominal physical width;
- smoothing during approximate label-to-order-parameter restart reconstruction;
- timestep selection/capping tied to strain increment and explicit stability estimates;
- grain/wake density resets and residual-content redistribution after insertion/migration;
- production defaults that disable temperature-dependent GB mobility.

Every such mechanism must become typed, unit-checked, provenance-tagged, visible in manifests, and covered by a limiting/ablation test if retained.

## Known concern disposition

1. **Collective law:** default `collective_rate_closure='domain_count'`; the Poisson-tail completion is primarily diagnostic/activity. Neither is locked to qualifying raw DD evidence. Reject as production closure until Gate 1.
2. **Candidate state:** v34 arrays store only active flag, age, best barrier, and birth step on grid cells. Radius, shape, trial orientation, parent, integrated driving force, spatial object identity, and RNG lineage are absent.
3. **Restart:** `_save_restart_checkpoint` does not serialize the four candidate arrays; candidate continuation cannot be exact.
4. **Decay:** for positive `nuc_candidate_decay_evals`, the implementation decrements age by one per failed evaluation. The parameter does not define or store a separate number of consecutive failures, so values above zero do not implement the stated failure-count semantics.
5. **Promotion:** promotion recomputes current local radius/orientation fields. There is no persistent embryo geometry/orientation to promote.
6. **Grain count:** diagnostics use `Ng`, the allocated order-parameter field count. There is no finite-support/purity/persistence/provenance physical-grain count or demonstrated label retirement.
7. **Classification:** the supplied summarizer applies weak threshold logic and treats legacy births/candidates too permissively. It cannot establish DRX or ASB.
8. **Sweep:** the supplied shell loop is local, rate-only, single-seed by default, has branch-specific runs, and has no temperature design or Slurm dependencies.
9. **GB mobility:** a temperature-dependent option exists but `use_temperature_dependent_gb_mobility=False` by default.
10. **Localization length:** `heat_process_zone_sigma_um=0.30` and a two-pixel minimum feed a Gaussian kernel. No source/measurement/DD link or grid-convergence evidence was located. It is provisional numerical regularization.

## Parameters and provenance

The legacy dictionary mixes SI dimensional quantities, reduced/numerical factors, calibration parameters, mode choices, and diagnostics. Provenance is mainly comments/version history rather than a machine-readable source record. Several parameters retain old `v25` names in v34 outputs. A complete line-by-line parameter table will be generated for the regression snapshot, but the first-pass disposition is:

- **physical/material candidates:** Burgers vector, elastic/shear modulus law, density, heat capacity, conductivity, GB energy/mobility, activation energies/volumes;
- **calibrated candidates:** slip prefactors/barriers, storage/recovery coefficients, GB transmission/accommodation parameters;
- **DD-claimed but not locked:** collective hit order/domain size/rate/correlation quantities;
- **numerical:** grid, `dt`, solver tolerances, smoothing, pixel minima, limiter/cap values;
- **legacy hypothesis parameters:** stitched Taylor potential, hazard multipliers, candidate hold/decay, branch toggles.

No production parameter may enter the new schema without units, an admissible range, and one of these provenance classes.

## Restart coverage

The exact checkpoint path includes core `eta`, orientation/grain labels, slip/density fields, temperature, strain/plastic histories, selected energy/hazard histories, and RNG state. Approximate diagnostic restarts reconstruct `eta` from labels with smoothing. Candidate arrays are allocated after load and omitted from checkpoint serialization. Full schema/version compatibility and segmented equivalence are absent. The new checkpoint must include mechanics, thermal state, every density reservoir, collective internal state/distribution, embryo objects, all independent RNG bit-generator states, numerical controller state, and provenance hashes.

## Energy and content bookkeeping

Positive density storage is locally capped by available work; plastic work is divided among heat, dislocation storage, residual GB storage, and accommodation. This is a useful v32 regression concept. However, insertion/wake operations redistribute or reset density through explicit rules, the stitched energy is not fully thermodynamically derived, and no demonstrated global approximately 1% work closure spans all mechanisms. Signed Burgers and reservoir transfers need separate conserved/nonconserved ledgers with boundary fluxes and annihilation heat.

## Labels and classification

New `eta` fields are allocated on promotion and `Ng` grows. Label support can vanish without retiring the allocation. Candidate and hazard events are not physical grains. A replacement metric must separately report allocated labels, connected components, active embryos, promoted births, physical recrystallized grains, and recrystallized fraction. ASB classification must require sustained strain-rate/work concentration, significant temperature rise, associated softening, and a converged finite width above interface/grid scales.

## Boundary conditions and intrinsic scales

Fields commonly use periodic finite differences/filtering; temperature also uses a bath-loss term. Mechanical boundary conditions are reduced to prescribed average strain rate/finite elastic loading rather than spatial displacement/traction conditions. There is no demonstrated intrinsic localization length independent of the fixed/nonlocal kernels and diffuse-interface gradients. Timescale comparisons to elastic waves, event correlation, diffusion, loading, recovery, embryo growth, and GB migration are missing.

## Retain / modify / reject summary

| Mechanism | Disposition | Condition |
|---|---|---|
| Finite elastic loading | Retain/derive | Replace reduced servo with audited mechanics; verify work and wave/quasi-static times. |
| Local plastic power and partition ledger | Retain/strengthen | Close all channels globally and locally to target tolerance. |
| Thermal conduction and bath boundaries | Retain/modify | Use material data, physical boundaries, and timestep convergence. |
| GB transmission/residual Burgers concepts | Test/modify | Require units, conservation, literature/DD support, and ablation. |
| Kocks--Mecking storage/recovery reservoirs | Test/modify | Separate mobile/forest/wall/GND roles and calibrate once. |
| Fixed Gaussian process zone | Reject by default | Retain only if physical length is independently established and mesh convergence passes. |
| Taylor-peak/nonconvex work potential as DRX driver | Reject | Kinetic diagnostic only. |
| Direct collective-to-grain hazard | Reject | Collective state may affect testable kinetic/storage pathways only. |
| v34 cell candidate counters | Reject for production | Preserve only as a negative bookkeeping regression. |
| `Ng` as physical grain count | Reject | Replace with finite-support physical metric and retirement/exclusion. |
| Branch-specific production physics/parameters | Reject | Modes remain verification ablations of one common model. |
