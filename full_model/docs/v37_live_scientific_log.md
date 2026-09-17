# V37 live scientific log

## Opening audit — 2026-09-17

- Canonical branch: `exp/full-v34-recovery-v1`.
- Local and remote start identity:
  `e1795173471ea3deea1f44fe68e4be2de384fab0`.
- V36 recurrent physics source remains
  `890cb8906a9772d8bd5c5eb43164ecd44ad2720f`; the n192 submission/configuration
  wrapper remains `e625a60dc01a119a7c041af3efdb771a30d01246` and Slurm job
  `56099919`. These provenance roles are not merged.
- V36 reports, evidence JSON, figures, and raw local/HPC outputs are frozen.
- No V36 worker was live. HPC3 was reachable through `uci-hpc3`. Unrelated job
  `56126061` (`pfgg-qiu-pristine-preflight-v3`) was live and was not modified.
- The V37 front/recurrent, Mura/organization, and conduction/localization work
  streams began in parallel from the same checkpoint.

## Thermal transport selection

The selected V36 finite-conduction state uses `k=0.15 W m^-1 K^-1`, volumetric
heat capacity `3.8e6 J m^-3 K^-1`, a 10 micrometre periodic domain, a 0.75
micrometre particle radius, and a 0.30 micrometre heat-process-zone scale. The
corresponding diffusion times are 2.28 microseconds across the process-zone
scale, 14.25 microseconds across the particle radius, and 2.533 milliseconds
across the domain. The matched loading horizon is 8.337 microseconds.

This places process-zone diffusion below the loading time, heterogeneity-scale
diffusion near it, and domain-scale diffusion far above it. The V37 screen
therefore varies rate, initial temperature, and one physical heterogeneity
length while keeping positive finite conductivity fixed. Zero conductivity is
retained only as the frozen V36 limiting ablation.

Manufactured broad and 0.25-micrometre-sigma narrow fields verify that the new
participation, connectivity, and second-moment metrics distinguish broad support
from a narrow band without changing the frozen strict-ASB thresholds. The actual
V36 finite-conduction temperature field is broad: its peak-minus-mean is 37.02 K
and its temperature inverse-participation fraction is 0.99987. These temperature
metrics are distinct from the shear-rate participation reported by V36.
