# V58 autonomous DRX/ASB campaign decision report

## Decision

V58 establishes a single, energy-qualified three-grain production architecture
that produces substantial prepared-boundary DRX and a persistent ASB trajectory
candidate under continued deformation with one fixed physical parameter set.
The response is rate-selected: at 2e3/s and 900 K the repaired model produces
substantial existing-boundary DRX without an ASB candidate, whereas at 2e4/s
and 900 K it produces substantial boundary migration together with persistent
connected plastic-power and irreversible-heat bands, temperature contrast above
100 K, and post-peak softening.

This is a prepared three-grain existing-boundary result. It is not evidence of
spontaneous intragranular grain birth, a calibrated material, or a general
polycrystal prediction. Within that declared scope, strict coupled-response
qualification passes: temporal refinement, successive n48-to-n64 spatial
refinement, causal controls, persistence, and the physical ledgers pass.

## Production physics completed in V58

The production caller now advances one authoritative multi-grain common state.
Each physical interval couples synchronized common-stress owner mechanics,
signed spatial dislocation transport, Mura/Nye reconstruction, recovery and
storage, common heat, and the bidirectional multi-interface front transaction.
The front transaction prices the actually realized direction and sweep after
donor competition, subcycles in physical time, transports supported state
conservatively, retains independent dissipation channels, and publishes only
after the complete-energy decision. Failed trials roll back atomically.

The final initialization repair gives `interface_width_m` its declared physical
meaning. The superseded initializer applied a Gaussian to squared distance, so
the bisector thickness scaled as width squared divided by center separation and
became subcell and grid-dependent. The repaired radial-distance soft Voronoi
construction gives a fixed physical transition width and doubles transition
cells when resolution doubles. Results produced before source `ce3d101` are
retained only as diagnosis or historical context, not as final V58 evidence.

## Fixed-parameter response family

All selected cases use source `ce3d101`, a 5 um periodic square, 312.5 nm
physical interface width, 900 K initialization, zero imposed migration pressure,
the same three prepared orientations and defect contrast, and the same kinetic,
thermal, elastic, storage, and EXP-floor parameters.

### Low-rate DRX-dominant trajectory

The n32 calculation at 2e3/s reaches 5.0 us and 1% additional applied shear.
It completes 500 accepted interface events, sweeps 9.7303e-22 m3, and has a
10.17 K terminal temperature contrast. It passes owner/Nye consistency,
nonnegative local dissipation, and complete-energy closure (maximum relative
closure 1.5298e-4). The outcome-neutral classifier reports substantial
existing-boundary DRX true and persistent ASB false.

### High-rate coupled trajectory

The n32 calculation at 2e4/s reaches 3.15 us and 6.3% additional applied
shear. At the common endpoint it has 4.4467 GPa shear stress, 994.95 K mean
temperature, 145.02 K contrast, 8.7058e-22 m3 cumulative physical sweep, and
8.1887e-22 m3 favored-grain growth. A restart from the exact step-110 state at
dense output cadence records eight consecutive ASB-candidate states within the
same horizon. The sequence contains connected plastic-power and independently
computed irreversible-heat bands, exceeds 100 K contrast, and softens after the
stress maximum. No grain label is born.

The n48 calculation reaches the same 3.15 us horizon with all 378 front events
accepted, 4.4637 GPa terminal stress, 995.43 K mean temperature, 156.75 K
contrast, and 8.9951e-22 m3 cumulative sweep. Maximum complete-energy closure
is 7.8176e-4 and owner/Nye mismatch remains negligible relative to the signal.

The n64 calculation also reaches 3.15 us with all 378 front events accepted.
It ends at 4.4018 GPa, 996.26 K mean temperature, 162.98 K contrast,
9.0975e-22 m3 cumulative sweep, and 8.6221e-22 m3 favored-grain growth. It
contains 18 consecutive ASB-candidate records, connected terminal plastic and
heat-rate bands, and a 3.4853e-4 maximum relative energy-closure error.

## Matched front/flow-temperature factorial

The n32 front on/off by physical/frozen-flow factorial uses a common initial
state, loading history, physical clock, and source. `freeze_flow` changes only
the temperature input to the flow law; recovery, front kinetics, heat,
conduction, and their recorded routing remain live.

At 3.15 us, the endpoint observables are:

| case | stress (GPa) | mean T (K) | contrast (K) | favored growth (m3) |
|---|---:|---:|---:|---:|
| front / full flow | 4.4467 | 994.95 | 145.02 | 8.1887e-22 |
| front / frozen flow | 4.6004 | 1002.39 | 133.07 | 8.8909e-22 |
| no front / full flow | 4.3136 | 950.67 | 155.50 | 0 |
| no front / frozen flow | 4.3786 | 952.75 | 152.75 | 0 |

Front motion at full feedback raises endpoint stress by 133.0 MPa and mean
temperature by 44.29 K while reducing contrast by 10.48 K. Physical versus
frozen flow lowers stress by 153.8 MPa with a front and by 64.9 MPa without a
front. The stress difference-of-differences is -88.84 MPa; the contrast
difference-of-differences is +9.20 K. Thus DRX/front evolution and thermal flow
feedback interact nonlinearly. Localization also occurs without front motion,
so the model does not incorrectly require DRX to create ASB; front motion
changes the localization topology and heat distribution.

## Numerical evidence

The n32 half-timestep calculation (`dt=12.5 ns`, 252 steps) reaches the same
3.15 us horizon. Relative to the 25 ns n32 reference, endpoint errors are
0.1205% stress, 0.00873% mean temperature, 0.0777% temperature contrast, and
0.3372% favored-grain growth. The selected temporal screen passes 5%.

The fixed-width n32-to-n48 endpoint errors are 0.3838% stress, 0.0474% mean
temperature, 8.0887% contrast, and 3.8071% favored-grain growth. The contrast
therefore does not pass on the first pair. The completed successive n48-to-n64
errors are 1.4075% stress, 0.0838% mean temperature, 3.8227% contrast, 1.4108%
favored-grain growth, and 1.1255% fresh sweep. Thus every selected observable
passes the provisional 5% screen on the finer successive pair, without changing
parameters or physical horizon. At the earlier fixed 2.5 us horizon all three
grids also agree closely; n64 gives 46.66 K contrast versus 47.37 K and 47.25 K
on n32 and n48.

## Claims and limits

- Energy-qualified multi-grain production: passed.
- Generic prepared-boundary DRX: passed.
- Persistent ASB response: passed in the declared prepared-boundary scope;
  n64 contains 18 consecutive candidate records at fine cadence.
- Matched DRX/thermal-feedback interaction controls: passed.
- Selected temporal refinement: passed.
- Selected spatial refinement: passed on the successive n48-to-n64 pair; the
  coarser n32-to-n48 contrast remains outside 5% and is retained transparently.
- Strict coupled response: passed for the generic prepared-boundary three-grain
  model and its fixed parameter set.
- Spontaneous intragranular grain birth: not implemented or claimed.
- Material calibration and quantitative material-class prediction: not claimed.

Machine-readable decisions, exact paths, source identities, hashes, and
reproduction commands are recorded in `full_model/verification`.
The frozen campaign source and evidence records pass 936 canonical tests in
326.17 s (`PYTHONPATH=src:. pytest -q`).

## Reproduction templates

Run from the V58 worktree with source `ce3d101`. The common suffix is:

```text
--length 5e-6 --interface-width 3.125e-7
--front-attempt-frequency 1e10 --front-maximum-fraction .015
--front-maximum-substep 1e-8 --mechanical-maximum-fraction .10
--maximum-mechanical-subdivisions 15
--thermal-diffusivity 3.9473684210526316e-8
--mechanics-mode physical --source-commit ce3d101
```

The principal high-rate baseline command is:

```text
python -m full_model.analysis.run_v58_three_grain_production
  --out <output> --n 32 --steps 126 --dt 2.5e-8
  --shear-rate 2e4 --temperature 900 --case baseline
  --checkpoint-every 10 --flow-temperature-mode physical <common suffix>
```

Use `--case no_front` to disable front motion and
`--flow-temperature-mode frozen` for the narrow flow-temperature intervention.
The low-rate member uses `--n 32 --steps 50 --dt 1e-7 --shear-rate 2e3`.
The temporal member uses `--n 32 --steps 252 --dt 1.25e-8`. Spatial members
use n48 and n64 with the baseline physical clock. Exact restart provenance and
SHA-256 values are stored in each result directory and the V58 manifest.
