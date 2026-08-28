# Material and validation-target audit

Status: unresolved; no production material dataset selected.

The 2026-08-28 clarification makes all local legacy values and derived tables context only. Primary publications may identify candidates, but a local digitization is not accepted merely because it is paired with a paper PDF.

## Direct project evidence

All items in this section are historical context and are not accepted material data.

- Legacy v32/v34 source describes a “BCC iron” model and uses a Burgers vector of 2.48e-10 m, temperature-dependent Fe-like shear-modulus code, a nominal high-angle GB energy of 0.50 J m^-2, and an 1811 K thermal validity ceiling.
- The manuscript draft explicitly calls a Potts demonstration an “Fe parameterization” at 1300 K and 1 s^-1.
- Files named `Poliak-Jonas/Fe.json`, `Prasad_map/Fe.json`, and `recrystallization/Fe.json` instead contain the field `"material": "Cr"` and provenance `Cr_NelderMeadDislocationOnlyStressDependent2.m`. File naming therefore contradicts embedded metadata.
- `Garofalo_data` contains multiple possible targets rather than one resolved material: 42CrMo, 30CrNiMoV, stainless-steel compression, Ti, AA6005/7075, AZ31/61, and several HEAs.
- The legacy regression case uses 1100 K and strain rates up to 30,000 s^-1, far outside several low-rate hot-deformation tables. A common validity envelope has not been established.
- A primary hot-compression candidate is 42CrMo/AISI 4140 steel: Lin, Chen, and Zhong, *Journal of Materials Processing Technology* 205 (2008) 308--315, DOI `10.1016/j.jmatprotec.2007.11.113`, covering 850--1150 degC and 0.01--50 s^-1. The local `42CrMo_peaks_from_paper.csv` is an unverified project digitization and remains context only.
- Primary high-rate 42CrMo4 studies exist to approximately `10^3--4.5e3 s^-1`, including DOI `10.1016/j.msea.2021.141953` and DOI `10.1016/j.matdes.2017.01.066`, but their quenched/tempered microstructures are not automatically the same material state as the hot-compression austenite. Combining them would violate the one-material-state requirement unless the phase/heat-treatment pathway is explicitly modeled and sourced.

## Consequences

1. “Fe” cannot be accepted as a versioned material dataset solely from filenames/comments.
2. Chromium-fit Arrhenius parameters cannot be combined silently with iron thermal/GB properties.
3. No temperature/rate production matrix or Zener--Hollomon comparison will be chosen until composition, phase/crystal structure, source experiments, and property provenance agree.
4. Legacy cases are context only and do not regression-gate or validate the new model.
5. 42CrMo is the best-documented candidate family currently visible, but it is not provenance-locked because the hot-DRX and high-rate datasets do not yet share a demonstrated initial material state.

## Resolution required

Select one actual alloy/purity/phase target and construct a versioned SI dataset containing elastic constants, density, heat capacity, conductivity, thermal expansion, slip systems, Burgers vector, mobility/activation laws, dislocation storage/recovery data, GB energy/mobility versus misorientation/temperature, and matching flow/DRX/ASB validation data. Each source needs DOI/table/figure/digitization provenance and uncertainty. Conflicting datasets remain visible rather than condition-specifically retuned.
