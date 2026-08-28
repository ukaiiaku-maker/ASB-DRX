# Final scientific report

Status: rebaselined on the analytical EXP-floor law; analytical verification in progress.

The final report will separately identify historical context, analytical consequences, new-model assumptions, numerical regularizations, calibrated quantities, validation results, failed/contradictory results, validity envelopes, and unresolved questions. No predictive-realism claim is currently made.

## Current evidence-backed outcome

- The isolated campaign is on `exp/independent-dd-pf-drx-asb-20260827`, based on remote `main` commit `a5dd798096e3896f319d314e8e4c60f5b277e589`.
- The inventory contains 33,358 files (21,536,785,369 bytes) with SHA-256 hashes and no read/hash failures.
- DD is not a parameterization route. Missing DD transition data no longer block the campaign.
- The EXP-floor barrier now has a closed-form inverse and Lambert-W prediction for peak density, local stress, and macroscopic strength as functions of rate and temperature.
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

No physical parameter has been calibrated and no production solver or array has been launched. Analytical implementation and verification proceed now. Physical optimization requires one authoritative material/validation target. A collective transparent-node extension remains optional research until a derived transfer/memory model and discriminating observations justify it.
