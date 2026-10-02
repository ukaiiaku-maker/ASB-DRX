# V61 exact continuation commands

All commands are run from
`/Users/sdillon/HPC3/worktrees/asb-drx-full-v61-20261001` with source
`b457eaf1ba7436431e95e75ec0da5448cb4cd950` checked out. The checkpoint hash
must be verified before restarting. The n48 prefix is intentionally unmatched
and nonconverged; continuation does not itself qualify ASB.

## Continue the fresh n48 physical prefix

Checkpoint 67:
`e7f584ffb40aa859546b6f8cf78b53d7d0aad24fcdbb99d99e7bd9bf32ba9c59`

```sh
PYTHONPATH=src:. python -m full_model.analysis.run_v58_three_grain_production \
  --out /Users/sdillon/HPC3/campaigns/asb-drx-v61-20261001/local_interface_physical_n48_from1p675us \
  --n 48 --grain-count 4 --steps 140 --dt 2.5e-8 --shear-rate 4e4 \
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
  --source-commit b457eaf1ba7436431e95e75ec0da5448cb4cd950
```

Do not weaken mechanical tolerances to bypass the observed step-68 cost.
Profile and optimize the common wall residual/rotated-system evaluation while
preserving identical accepted trajectories before extending this run.

## Reproduce the clean matched n32 900 K control

The physical member differs only by changing all three `frozen` values below
to `physical`. Both existing endpoint checkpoints contain 201 accepted events
and zero rejections.

```sh
PYTHONPATH=src:. python -m full_model.analysis.run_v58_three_grain_production \
  --out /Users/sdillon/HPC3/campaigns/asb-drx-v61-20261001/local_interface_allfrozen_n32_to1p675us \
  --n 32 --grain-count 4 --steps 67 --dt 2.5e-8 --shear-rate 4e4 \
  --temperature 900 --case baseline --checkpoint-every 1 --length 5e-6 \
  --interface-width 3.125e-7 --front-attempt-frequency 1e10 \
  --front-maximum-fraction .015 --front-maximum-substep 1e-8 \
  --mechanical-maximum-fraction .1 \
  --thermal-diffusivity 3.947368421052632e-8 \
  --flow-temperature-mode frozen --recovery-temperature-mode frozen \
  --front-temperature-mode frozen \
  --front-temperature-resolution local_interface \
  --front-heat-deposition local_realized_event --mechanics-mode physical \
  --source-commit b457eaf1ba7436431e95e75ec0da5448cb4cd950
```

## Source-bound reduction

```sh
PYTHONPATH=src:. python -m full_model.analysis.postprocess_v59_physical_asb \
  --baseline-dir /Users/sdillon/HPC3/campaigns/asb-drx-v61-20261001/local_interface_physical_n32_to1p675us \
  --control-dir /Users/sdillon/HPC3/campaigns/asb-drx-v61-20261001/local_interface_allfrozen_n32_to1p675us \
  --output /Users/sdillon/HPC3/campaigns/asb-drx-v61-20261001/local_interface_clean_n32_to1p675us_matched_classification.json
```
