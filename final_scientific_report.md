# Final scientific report

Status: rebaselined on the analytical EXP-floor law; isolated thermodynamic, grain, coupled spatial, and localization-metric gates executed on HPC3.

The final report will separately identify historical context, analytical consequences, new-model assumptions, numerical regularizations, calibrated quantities, validation results, failed/contradictory results, validity envelopes, and unresolved questions. No predictive-realism claim is currently made.

## Current evidence-backed outcome

- The isolated campaign is on `exp/independent-dd-pf-drx-asb-20260827`, based on remote `main` commit `a5dd798096e3896f319d314e8e4c60f5b277e589`.
- The inventory contains 33,358 files (21,536,785,369 bytes) with SHA-256 hashes and no read/hash failures.
- By explicit user authorization, the single-glider DDD constants are reused as a generic, non-material fixture; DDD trajectories are not a calibration target.
- The EXP-floor barrier now has a closed-form inverse and Lambert-W prediction for peak density, local stress, and macroscopic strength as functions of rate and temperature.
- HPC3 run `20260828T120911Z-9d9e7c4-1bdf8a` / job `55637582` passed all five analytical kernel tests and was fetched with verified checksums.
- HPC3 run `20260828T131741Z-9caa154-ffbe2c` / job `55637767` shows that strength peaks alone are scale-nonidentifiable, while independent peak density restores the tested five-parameter local rank.
- HPC3 run `20260828T132308Z-07d589a-1e7eac` / job `55637784` passed the first executable thermodynamic gate: variational consistency, nonincreasing free-energy relaxation, conservative density transfer, exact incremental work closure, and the circular-nucleus sign limit.
- HPC3 run `20260828T132810Z-a4a0bf0-5c7bfe` / job `55637801` passed the diffuse 2-D nucleus gate, final grid/timestep refinement targets, and bitwise restart for the kernel's complete limited state.
- HPC3 run `20260828T133324Z-6920914-866ee2` / job `55637814` passed the homogeneous finite-loading thermomechanical ledger, impossible-partition rejection, and complete current material-point restart.
- HPC3 run `20260828T133943Z-ae2fdf9-d7f5b0` / job `55637821` passed the periodic common-stress shear-layer controls after an earlier failed run exposed and corrected a missing explicit-diffusion stability bound.
- HPC3 run `20260828T134840Z-5e8fabc-192eaf` / job `55637844` passed all ten physical-grain metric tests: label-only/sub-resolution/disconnected support cannot become a physical grain; promotion requires persistence, valid lineage, and symmetry-reduced misorientation; loss and retirement preserve provenance; and tracker restart is exact.
- HPC3 run `20260828T135602Z-d1de48a-eb7ef1` / job `55637866` passed the constrained multi-order gate and all grain regressions. It verifies the variational derivative, energy decrease, pure-parent invariance, pointwise simplex constraint, label symmetry, sharp-interface radius-change signs, exact limited-state restart, and tracker coupling at fixed allocated-label count.
- HPC3 run `20260828T140503Z-8fb6236-7dab82` / job `55637879` passed the explicit stored-dislocation-energy coupling gate. The generic relaxation converts a `5.5970e-9 J m^-1` stored-energy decrease into `2.4351e-9 J m^-1` interface/order-energy increase and `3.1619e-9 J m^-1` heat with zero ledger error.
- HPC3 run `20260828T141129Z-e01a0ce-969f27` / job `55637888` passed the auditable candidate-decision gate. It verifies the classical cylindrical barrier/critical-radius identities, bounded thermal Poisson probability, expected monotonicities, external-draw determinism, and distinct rejection reasons without allocating a grain label.
- HPC3 run `20260828T141957Z-c9708bb-f4aa93` / job `55637907` passed the first combined thermomechanical/phase gate and all upstream regressions. External work `4.5781612e6 J m^-3` closes across elastic, total stored, interface/order, mechanical-heat, and phase-heat changes with cumulative global error `1.45e-9 J m^-3`.
- HPC3 run `20260828T142706Z-3224ef1-bdf10f` / job `55638019` passed the periodic 2-D spatial coupling gate. Its temperature perturbation damps from `0.1767767` to `0.0746919 K` standard deviation while the global ledger closes to `1.07e-11 J m^-3`; this is explicitly a non-localizing control.
- HPC3 run `20260828T143238Z-ca7ab69-74cc1d` / job `55638898` passed six localization-metric tests. The ASB gate is conjunctive and requires persistent plastic concentration, temperature excess over a matched control, post-peak softening, a band wider than the numerical interface, and converged onset and width. Its thresholds are generic fixtures, and no coupled trajectory has yet been classified as ASB.
- HPC3 run `20260828T144000Z-c9dc4ab-7b33f9` / job `55640278` passed the first six-case common-equation mechanism ladder and upstream regressions. Isothermal variants account for generated heat through an explicit bath, phase-disabled variants preserve fields exactly, and thermal cases use same-rate/same-phase isothermal twins. Every generic case is a negative localization control (`f_q >= 0.907`, `Delta T_control <= 0.125 K`, zero softening).
- HPC3 run `20260828T144457Z-535a0ff-662a0e` / marker job `55640502` passed ten analytical stability tests. The evaluated generic state has conduction-damped thermal response and a positive `4.807 s^-1` forest-storage mode; this separates a provisional structural amplification from thermal ASB. Duplicate job `55640458`, created after a silent staging receipt, is retained in the audit record and makes no independent scientific claim.
- HPC3 run `20260828T151810Z-d31c6a4-d3af25` / job `55641106` passed fourteen tests for the exact single-glider DDD fixture mapping. The governing equations predict `rho_peak=4.4117e15` to `8.5383e15 m^-2` over 1050 to 850 K at `4.5 s^-1`; the source driver's hard-coded `1e18 m^-2` field is excluded. The explicit DDD response remains monotone beyond the analytical peak, which is retained as evidence against identifying the independent falling branch with actual transparency.
- HPC3 run `20260828T152809Z-db81077-68e54d` / job `55641308` passed fifteen tests for the preregistered analytical boundary surface. At the source rate, the DDD campaign upper density is 3.514--6.800 times the analytical peak across 850--1050 K, yet its strength remains rising. The post-peak side is therefore a collective-candidate ablation region, not an asserted transparent-node transition.
- Corrected HPC3 single-job boundary smoke `20260828T154200Z-9d7ed90-e57dd1` / job `55642217` passed 21 tests after a retained packaging-only preflight failure. The 950 K, `45000 s^-1` analytical-peak case converged strongly between 16² and 32² grids and heated `26.74 K` above its matched control, but plastic flow remained essentially uniform and failed the ASB classifier. This is numerical compatibility evidence and a no-go for a regime array until local stress redistribution replaces frozen common stress.
- The target material is unresolved because legacy Fe/BCC descriptions conflict with embedded Cr provenance and mixed-alloy validation data.
- The thermodynamic document defines candidate state/balance/dissipation structures, rejects flow stress as a free-energy density, and keeps both the closure and DRX representation decisions open pending evidence.
- HPC3 environment/staging smoke job `55633650` passed and fetched. Exact legacy jobs `55633674`, `55633691`, and `55633694` all completed and fetched with verified checksums.

## Legacy observations, retained only as context

| Version | Observation | Interpretation |
|---|---|---|
| v32 | The reproduced 30,000 s^-1 case reaches 1245.50 K maximum temperature and 29.57 K maximum spatial temperature standard deviation; 12 labels and 12 topology components remain | Useful ASB-like regression only; persistence and mesh-converged band width are absent |
| v33 | 165 hazard births increase allocated labels from 12 to 177 while topology components remain 12 | Strong false-DRX negative control; zero accepted physical births |
| v34 | Candidate active/new/promotable/age maxima and births are all zero | Contradicts the candidate-without-promotion premise; it is a zero-candidate failure control |

v33 and v34 detailed trajectories do not reproduce the stored references from identical recorded parameters. Their old output directories lack immutable runtime/source provenance, so this is reported as unresolved legacy nondeterminism rather than tuned away.

## Current boundary

No physical parameter has been calibrated and no production solver or array has been launched. The thermodynamic, grain-classification, coupled-ledger, and localization-metric results are generic verification fixtures, not evidence of realistic DRX or ASB. The arbitrary boundary is the analytical EXP-floor peak surface, not a material phase boundary. A collective transparent-node extension remains optional research until a derived transfer/memory model and discriminating observations justify it. Crystallographic orientation evolution, physical nucleation/field allocation, displacement-resolved spatial mechanics, production-state restart, boundary-map simulations, localization convergence, uncertainty propagation, and external validation remain required.
