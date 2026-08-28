# Legacy DD-adjacent inventory (context only)

## Disposition

This inventory is retained only to document what was examined. None of the artifacts below may calibrate or validate the new model, and missing DD data no longer blocks the analytical campaign.

## Located DD-adjacent artifacts

| Artifact | What it contains | Evidence status |
|---|---|---|
| `recrysyallization_PF-2D/shear_banding/arrhenius_nanopillar_first_avalanche_ddd_stress.py` | A stochastic nanopillar first-avalanche driver with optional OpenDiS stress sampling, independent root hazards, cascade facilitation, and synthetic event histories. | Code/model hypothesis, not raw DD observation. |
| `.../run_first_avalanche_campaign_ddd_stress_standalone_v6.sh` | Default launcher: 50 seeds, 300 K, 2e4 s^-1, radius 200 and height 500 reduced units, 256 sites, 25,000 steps, imposed strain increments, configurable cascade model. | Campaign specification only. |
| `.../README_arrhenius_avalanche_ddd_v6.md` | Notes removal of stress ceilings/caps and recommends checking first-event/cascade summaries. | Prior interpretation and code provenance. |
| `.../summarize_first_avalanche_campaign.py` | Summarizes avalanche size and realized branching ratio; produces a CCDF. | Derived-analysis code. |
| `DRX_and _ASB/multihit.m` and `results/.../strength_vs_density_multihit.csv` | Prescribed multi-hit/Taylor calculations and derived strength table. | Hypothesis/derived data; not DD transition evidence. |

No `avalanche_summary.json`, campaign event table, waiting-time table, or raw event trajectory was found under `/Users/sdillon/DRX-ASB` by the first-pass search.

## Metadata that would be required for an optional collective comparison

For each condition, obtain without inference:

- raw trajectory/event file and immutable hash;
- DD code/version, force/mobility laws, boundary conditions, loading protocol;
- event definition, detection threshold, dead time, merging/splitting rule;
- time and stress units and sampling cadence;
- temperature, total/mobile/forest/GND density definitions and units;
- applied/resolved stress, strain rate, strain, and control mode;
- seed, number of realizations, cell dimensions/volume, periodicity, and initial microstructure;
- censored runs and observation windows;
- event time, location, amplitude, slip system/Burgers character where available;
- uncertainty and dependencies on density, temperature, stress, rate, correlation time, and length.

## Optional characterization if qualifying observations become available

Use condition/seed-grouped calibration and held-out validation. Estimate waiting/inter-event distributions, survival/hazard, CV, Fano factor versus window, renewal shape/effective hit order, amplitude law, temporal correlation, spatial correlation/domain length, censoring likelihood, and seed uncertainty. Test dependencies jointly before any dimensional reduction. Out-of-envelope queries will warn or stop according to configuration and will never extrapolate silently.
