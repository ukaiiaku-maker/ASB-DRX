# ASB-DRX technical status handoff

> Historical legacy note: the v32-v34 discussion below is retained only as
> context and is not the active scientific architecture or valid calibration
> data. The current independent implementation and its first sparse boundary
> result are documented in
> [local_antiplane_boundary_campaign.md](local_antiplane_boundary_campaign.md).
> DDD does not parameterize the present model.

## Active baseline

The active local baseline is `drx_var_v34_candidate_drx_asb_sweep.py`. It builds on the v32 GB-transmission/process-zone ASB model and adds candidate-nucleus bookkeeping for DRX.

The v32 branch should be treated as the current ASB reference. The v33 branch should be treated as a negative-control failure case because aggressive hazard nucleation produced many grain IDs without stable physical DRX. The v34 branch fixed that bookkeeping problem by separating candidate embryos from promoted persistent grains.

## Scientific target

The code should distinguish three mechanisms:

1. Recovery and grain-boundary relaxation.
2. ASB-like thermoplastic localization.
3. Physical DRX from stable finite-amplitude nuclei.

A hazard event should not automatically create a permanent grain ID. Physical DRX requires a finite-amplitude embryo that survives, grows or remains viable, and develops persistent phase-field support.

## What currently works

The ASB side is the most credible part of the model. The v32/v34 framework includes finite elastic loading, local plastic-work heating, a finite-loading work budget, a process-zone heat kernel, GB slip-transmission barriers, residual-Burgers storage, and blocked-GB work partition. This can produce finite-width GB-assisted thermoplastic localization with scalar ASB-like signatures such as increasing thermal heterogeneity, density depletion in hot regions, negative temperature-density correlation, and stress softening after peak.

The grain-ID bookkeeping is improved in v34. Hazard nucleation now creates candidate embryos rather than immediate permanent grain labels. This prevents the v33 label explosion from masquerading as DRX.

## What does not yet work

Physical DRX is not solved. v34 currently produces the opposite failure mode from v33. v33 produced too many permanent labels. v34 suppresses fake labels, but candidate embryos generally do not promote to stable grains. The DRX branch needs stateful embryo growth, survival, and promotion logic.

Low-rate DRX is still ambiguous. Earlier low-rate sweeps either disabled nucleation, failed from mechanical validity due to timestep/servo choices, or produced only candidate embryos. We cannot yet conclude that low-rate DRX is absent physically.

The ASB classifier is too permissive in some summaries. Mild recovery or weak thermal heterogeneity should not be classified as ASB. Classification should require meaningful thermal heterogeneity, density depletion in hot regions, negative temperature-density correlation, and stress softening.

## Recommended Codex tasks

1. Audit candidate-nucleus promotion logic and report which promotion criterion fails for each candidate.
2. Convert candidates into stateful embryos with position, radius, orientation, parent grain, age, integrated viability score, barrier history, and local driving-force history.
3. Add embryo growth or shrinkage using stored-energy relief minus interfacial penalty.
4. Separate `label_count`, `candidate_count`, `promoted_birth_count`, and `physical_grain_count`.
5. Tighten ASB classification thresholds.
6. Preserve the `asb_only`, `drx_isothermal`, and `coupled` branch separation.
7. Add overlay diagnostics for temperature, heat generation, GB transmission factor, residual Burgers vector, GB transmission barrier, GND, rho_GB, and candidate embryo positions.
8. Add regression tests so hazard events cannot increment physical grain count before candidate promotion.

## Suggested branch semantics

- `BRANCH=asb_only`: hazard nucleation off and topology relabel off. Used to test whether hotspots are intrinsic to the ASB/GB-transmission model.
- `BRANCH=drx_isothermal`: candidate DRX on with strong thermal bath. Used to test candidate embryo formation without thermal runaway.
- `BRANCH=coupled`: full ASB plus candidate DRX interaction.

## Immediate caution

Do not tune the nucleation attempt frequency upward to force DRX. That reproduces the v33 failure mode. The next improvement should be embryo physics and diagnostics, not raw hazard amplification.
