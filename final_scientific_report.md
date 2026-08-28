# Final scientific report

Status: rebaselined on the analytical EXP-floor law; analytical, identifiability, collective-context, thermodynamic, finite-loading, shear-layer, and physical-grain metric gates executed on HPC3.

The final report will separately identify historical context, analytical consequences, new-model assumptions, numerical regularizations, calibrated quantities, validation results, failed/contradictory results, validity envelopes, and unresolved questions. No predictive-realism claim is currently made.

## Current evidence-backed outcome

- The isolated campaign is on `exp/independent-dd-pf-drx-asb-20260827`, based on remote `main` commit `a5dd798096e3896f319d314e8e4c60f5b277e589`.
- The inventory contains 33,358 files (21,536,785,369 bytes) with SHA-256 hashes and no read/hash failures.
- DD is not a parameterization route. Missing DD transition data no longer block the campaign.
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

No physical parameter has been calibrated and no production solver or array has been launched. The thermodynamic and grain-classification results are generic verification fixtures, not evidence of realistic DRX or ASB. Physical optimization requires one authoritative material/validation target. A collective transparent-node extension remains optional research until a derived transfer/memory model and discriminating observations justify it. Multi-order-parameter orientation evolution, energetic nucleation, DRX coupling, production-state restart, the common-equation mechanism ladder, uncertainty propagation, and external validation remain required.
