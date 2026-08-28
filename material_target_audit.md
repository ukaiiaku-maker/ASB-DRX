# Material and validation-target audit

Status: unresolved; no production material dataset selected.

## Direct project evidence

All items in this section are historical context and are not accepted material data.

- Legacy v32/v34 source describes a “BCC iron” model and uses a Burgers vector of 2.48e-10 m, temperature-dependent Fe-like shear-modulus code, a nominal high-angle GB energy of 0.50 J m^-2, and an 1811 K thermal validity ceiling.
- The manuscript draft explicitly calls a Potts demonstration an “Fe parameterization” at 1300 K and 1 s^-1.
- Files named `Poliak-Jonas/Fe.json`, `Prasad_map/Fe.json`, and `recrystallization/Fe.json` instead contain the field `"material": "Cr"` and provenance `Cr_NelderMeadDislocationOnlyStressDependent2.m`. File naming therefore contradicts embedded metadata.
- `Garofalo_data` contains multiple possible targets rather than one resolved material: 42CrMo, 30CrNiMoV, stainless-steel compression, Ti, AA6005/7075, AZ31/61, and several HEAs.
- The legacy regression case uses 1100 K and strain rates up to 30,000 s^-1, far outside several low-rate hot-deformation tables. A common validity envelope has not been established.

## Consequences

1. “Fe” cannot be accepted as a versioned material dataset solely from filenames/comments.
2. Chromium-fit Arrhenius parameters cannot be combined silently with iron thermal/GB properties.
3. No temperature/rate production matrix or Zener--Hollomon comparison will be chosen until composition, phase/crystal structure, source experiments, and property provenance agree.
4. Legacy cases are context only and do not regression-gate or validate the new model.

## Resolution required

Select one actual alloy/purity/phase target and construct a versioned SI dataset containing elastic constants, density, heat capacity, conductivity, thermal expansion, slip systems, Burgers vector, mobility/activation laws, dislocation storage/recovery data, GB energy/mobility versus misorientation/temperature, and matching flow/DRX/ASB validation data. Each source needs DOI/table/figure/digitization provenance and uncertainty. Conflicting datasets remain visible rather than condition-specifically retuned.
