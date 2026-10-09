# V62 repaired-clock continuation commands

Numerical calculations in this file use the immutable detached source
`1c0cb4c` at
`/Users/sdillon/HPC3/worktrees/asb-drx-v62-numerical-1c0cb4c`.
The physical coefficients and 5 micrometre geometry are unchanged from V61.
V62 changes the numerical front clock: a clipped contour request is rolled
back and subdivided until every accepted event is below the unchanged 0.015
contour-CFL limit. The copied-state audit selected a 6.25 ns caller cadence.

## Repaired physical n48 continuation from the V61 state

The parent is V61 checkpoint 67 at 1.675 microseconds:

`e7f584ffb40aa859546b6f8cf78b53d7d0aad24fcdbb99d99e7bd9bf32ba9c59`

```sh
PYTHONPATH=src:. python -m full_model.analysis.run_v58_three_grain_production \
  --out /Users/sdillon/HPC3/campaigns/asb-drx-v62-20261008/n48_physical_adaptive_quarter_to2p0us_1c0cb4c \
  --n 48 --grain-count 4 --steps 119 --dt 6.25e-9 --shear-rate 4e4 \
  --temperature 900 --case baseline --checkpoint-every 1 --length 5e-6 \
  --interface-width 3.125e-7 --front-attempt-frequency 1e10 \
  --front-maximum-fraction .015 --front-maximum-substep 1e-8 \
  --mechanical-maximum-fraction .1 \
  --thermal-diffusivity 3.947368421052632e-8 \
  --flow-temperature-mode physical --recovery-temperature-mode physical \
  --front-temperature-mode physical \
  --front-temperature-resolution local_interface \
  --front-heat-deposition local_realized_event --mechanics-mode physical \
  --resume /Users/sdillon/HPC3/campaigns/asb-drx-v61-20261001/local_interface_physical_n48_to3p5us/checkpoint_000067.npz \
  --expected-resume-sha256 e7f584ffb40aa859546b6f8cf78b53d7d0aad24fcdbb99d99e7bd9bf32ba9c59 \
  --resume-transition v62_adaptive_front_cfl_quarter_from_b457eaf \
  --source-commit 1c0cb4c
```

## Fresh analytic-origin all-Arrhenius-frozen n48 control

This is not initialized from an evolved physical checkpoint. Temperature and
the heat equation remain physical; only the declared flow, recovery, and
front Arrhenius temperature arguments are held at 900 K.

```sh
PYTHONPATH=src:. python -m full_model.analysis.run_v58_three_grain_production \
  --out /Users/sdillon/HPC3/campaigns/asb-drx-v62-20261008/n48_allfrozen_adaptive_quarter_to2p0us_1c0cb4c \
  --n 48 --grain-count 4 --steps 320 --dt 6.25e-9 --shear-rate 4e4 \
  --temperature 900 --case baseline --checkpoint-every 1 --length 5e-6 \
  --interface-width 3.125e-7 --front-attempt-frequency 1e10 \
  --front-maximum-fraction .015 --front-maximum-substep 1e-8 \
  --mechanical-maximum-fraction .1 \
  --thermal-diffusivity 3.947368421052632e-8 \
  --flow-temperature-mode frozen --recovery-temperature-mode frozen \
  --front-temperature-mode frozen \
  --front-temperature-resolution local_interface \
  --front-heat-deposition local_realized_event --mechanics-mode physical \
  --source-commit 1c0cb4c
```

For either branch, resume only from the last checksum-verified committed
checkpoint, retain the same configuration and source, set `--steps` to the
new absolute runner step target, and supply both `--resume` and
`--expected-resume-sha256`. Exact same-source restart provenance is then
recorded automatically.

## Fresh analytic-origin physical n48 reference

This is the causal partner of the all-frozen control. It must be generated
from the same analytic origin because the repaired clock changes accumulated
front history; the transitioned V61 continuation above is informative but is
not a clean causal reference.

```sh
PYTHONPATH=src:. python -m full_model.analysis.run_v58_three_grain_production \
  --out /Users/sdillon/HPC3/campaigns/asb-drx-v62-20261008/n48_physical_fresh_adaptive_quarter_to2p0us_1c0cb4c \
  --n 48 --grain-count 4 --steps 320 --dt 6.25e-9 --shear-rate 4e4 \
  --temperature 900 --case baseline --checkpoint-every 1 --length 5e-6 \
  --interface-width 3.125e-7 --front-attempt-frequency 1e10 \
  --front-maximum-fraction .015 --front-maximum-substep 1e-8 \
  --mechanical-maximum-fraction .1 \
  --thermal-diffusivity 3.947368421052632e-8 \
  --flow-temperature-mode physical --recovery-temperature-mode physical \
  --front-temperature-mode physical \
  --front-temperature-resolution local_interface \
  --front-heat-deposition local_realized_event --mechanics-mode physical \
  --source-commit 1c0cb4c
```

## Fresh analytic-origin physical n32 spatial partner

This calculation uses the same repaired front clock, physical coefficients,
physical dimensions, caller cadence, and analytic four-grain origin as the
n48 physical reference.  It is a physical-only spatial partner, not a causal
Arrhenius control and not by itself a spatial-convergence certificate.

```sh
PYTHONPATH=src:. python -m full_model.analysis.run_v58_three_grain_production \
  --out /Users/sdillon/HPC3/campaigns/asb-drx-v62-20261008/n32_physical_fresh_adaptive_quarter_to2p0us_1c0cb4c \
  --n 32 --grain-count 4 --steps 320 --dt 6.25e-9 --shear-rate 4e4 \
  --temperature 900 --case baseline --checkpoint-every 1 --length 5e-6 \
  --interface-width 3.125e-7 --front-attempt-frequency 1e10 \
  --front-maximum-fraction .015 --front-maximum-substep 1e-8 \
  --mechanical-maximum-fraction .1 \
  --thermal-diffusivity 3.947368421052632e-8 \
  --flow-temperature-mode physical --recovery-temperature-mode physical \
  --front-temperature-mode physical \
  --front-temperature-resolution local_interface \
  --front-heat-deposition local_realized_event --mechanics-mode physical \
  --source-commit 1c0cb4c
```

## Sparse n64 physical prefix and continuation

The fresh n64 member uses the identical physical protocol but saves every
fifth caller state. A bounded continuation completed step 210 at 1.3125
microseconds. Continue from that checksum-bound endpoint without changing its
cadence or coefficients:

```sh
PYTHONPATH=src:. python -m full_model.analysis.run_v58_three_grain_production \
  --out /Users/sdillon/HPC3/campaigns/asb-drx-v62-20261008/n64_physical_sparse_continuation_from1p3125us_1c0cb4c \
  --n 64 --grain-count 4 --steps 320 --dt 6.25e-9 --shear-rate 4e4 \
  --temperature 900 --case baseline --checkpoint-every 5 --length 5e-6 \
  --interface-width 3.125e-7 --front-attempt-frequency 1e10 \
  --front-maximum-fraction .015 --front-maximum-substep 1e-8 \
  --mechanical-maximum-fraction .1 \
  --thermal-diffusivity 3.947368421052632e-8 \
  --flow-temperature-mode physical --recovery-temperature-mode physical \
  --front-temperature-mode physical \
  --front-temperature-resolution local_interface \
  --front-heat-deposition local_realized_event --mechanics-mode physical \
  --resume /Users/sdillon/HPC3/campaigns/asb-drx-v62-20261008/n64_physical_sparse_continuation_from1p25us_1c0cb4c/checkpoint_000210.npz \
  --expected-resume-sha256 f44f2e437845688a70e6eb2a47edf972ead192d580c65e9a6602711928175ff0 \
  --source-commit 1c0cb4c
```

The intended next discriminator remains the common n48/n64 interval from 1.65
to 2.0 microseconds. The existing n64 partial continuation is not a
late-morphology certificate.
