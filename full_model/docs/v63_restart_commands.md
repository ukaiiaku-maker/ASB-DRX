# V63 exact continuation commands

V63 preserves two numerical-source identities.  The physical n64 continuation
uses the unchanged V62 production source
`1c0cb4ca0e8a9a30b0c1a320817a0b35b4d14499`.  New temporal diagnostics and
the 80,000/s response experiment use diagnostic-only source
`ad6171b87a0ec5e3ef588be2484934356514443b`; that commit does not alter the
production constitutive kernels.

## Continue the unchanged-source n64 trajectory

The committed step-320 checkpoint is
`/Users/sdillon/HPC3/campaigns/asb-drx-v63-20261009/n64_physical_same_source_from_step210_to320_1c0cb4c/checkpoint_000320.npz`
with SHA-256
`ac4ac1ee6e7b01cd2d95fc4d5e35d694604c4e886e15111c1e01e297ce21a3ea`.
To extend it, retain every physical and numerical argument and increase only
the absolute target step:

```sh
cd /Users/sdillon/HPC3/worktrees/asb-drx-v62-numerical-1c0cb4c
PYTHONPATH=src:. python -m full_model.analysis.run_v58_three_grain_production \
  --out /Users/sdillon/HPC3/campaigns/asb-drx-v63-20261009/n64_physical_after_step320_1c0cb4c \
  --n 64 --grain-count 4 --steps TARGET_STEP --dt 6.25e-9 \
  --shear-rate 4e4 --temperature 900 --case baseline --checkpoint-every 5 \
  --length 5e-6 --interface-width 3.125e-7 \
  --front-attempt-frequency 1e10 --front-maximum-fraction .015 \
  --front-maximum-substep 1e-8 --mechanical-maximum-fraction .1 \
  --thermal-diffusivity 3.947368421052632e-8 \
  --flow-temperature-mode physical --recovery-temperature-mode physical \
  --front-temperature-mode physical \
  --front-temperature-resolution local_interface \
  --front-heat-deposition local_realized_event --mechanics-mode physical \
  --resume /Users/sdillon/HPC3/campaigns/asb-drx-v63-20261009/n64_physical_same_source_from_step210_to320_1c0cb4c/checkpoint_000320.npz \
  --expected-resume-sha256 ac4ac1ee6e7b01cd2d95fc4d5e35d694604c4e886e15111c1e01e297ce21a3ea \
  --source-commit 1c0cb4ca0e8a9a30b0c1a320817a0b35b4d14499
```

## Reproduce the 80,000/s diagnostic pair

Both members begin from the same analytic four-grain state.  The physical
member sets all three temperature modes to `physical`; the matched control
sets all three to `frozen`.  Temperature and the heat equation remain live in
the frozen member.

```sh
cd /Users/sdillon/HPC3/worktrees/asb-drx-v63-numerical-ad6171b
PYTHONPATH=src:. python -m full_model.analysis.run_v58_three_grain_production \
  --out OUTPUT_DIRECTORY --n 48 --grain-count 4 --steps 160 --dt 6.25e-9 \
  --shear-rate 8e4 --temperature 900 --case baseline --checkpoint-every 5 \
  --length 5e-6 --interface-width 3.125e-7 \
  --front-attempt-frequency 1e10 --front-maximum-fraction .015 \
  --front-maximum-substep 1e-8 --mechanical-maximum-fraction .1 \
  --thermal-diffusivity 3.947368421052632e-8 \
  --flow-temperature-mode MODE --recovery-temperature-mode MODE \
  --front-temperature-mode MODE --front-temperature-resolution local_interface \
  --front-heat-deposition local_realized_event --mechanics-mode physical \
  --source-commit ad6171b87a0ec5e3ef588be2484934356514443b
```

Use `MODE=physical` and output directory
`n48_rate80000_physical_to_step160_ad6171b`, or `MODE=frozen` and directory
`n48_rate80000_frozen_arrhenius_to_step160_ad6171b`.  Never resume one member
from the other member's checkpoint.

## Late-state temporal fork

The step-275 n64 quarter/eighth comparison is an immutable copied-state
diagnostic.  Its production reference state remains the unchanged-source
checkpoint; the diagnostic transition name is
`v63_eighth_step_copied_state_diagnostic_from_1c0cb4c`.  Do not continue a
physical trajectory from a fork endpoint.
