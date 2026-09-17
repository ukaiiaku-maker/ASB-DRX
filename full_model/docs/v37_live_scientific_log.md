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

## Current-source front reconciliation

The n128 current-source controls separate complete material exchange from scalar
or proposal reversal. Complete material exchange is exactly odd: the two raw
velocities are `+/-7.39397845456265e-8 m/s`, and accepted displacements are
opposite. Reversing only the proposal produces no publication, processed line,
heat, or displacement. It is a direction mismatch, not physical motion or an
energy-guard arrest.

The historical case named `equal_state` equalized only a scalar and retained
different wake history. Explicit owner equality before Mura does not remain a
complete equality after the recurrent Mura update. Its residual accepted
displacement is `5.409715679860483e-14 m`; endpoint defect and boundary terms
match, while diffuse-slab phase-gradient/local and thermal-history terms do not.
This is finite-interface/history response, not broken material-label symmetry.

The ordinary accepted current-source interval moves the contour
`3.69684016154892e-13 m` (`9.242100403872301e-7` interface widths), transforms
`1.1735249408820892e-27 m3`, and processes `1.1261320234643155e-12 m` of line.
This is direct accepted geometry but is not finite-amplitude migration or DRX.
The preregistered 17-case generic rate screen selects the unchanged baseline,
`availability_mid`, and `shape_n_low` for later full-solver trajectories. Its
minimum estimated quarter-width time is 0.04409 s; no rate-screen row is promoted
to a trajectory claim.

## Mura organization selection

The ordered-reservoir audit finds a large relative timestep difference only in
small reservoirs: absolute ordered-line difference `1.675 m/m`, ordered-moment
difference `0.714 m/m`, wall-local line error `0.1504%`, and energy difference
`1.63e-10 J/m` (`1.04e-6` of defect energy). This does not block bulk or thermal
evolution, but it cannot support a resolved wall claim.

Matched short topology controls preserve all hard invariants and allocate no
phase or grain. Homogeneous and broadband-noise cases capture no wall line.
Mechanical heterogeneity captures ordered line, but its orientation span is only
`6.57e-7 deg`; it is therefore classified as captured ordered line without an
orientation-compatible wall. The topology-off one-grain route is selected for a
long n64 calculation to 5% strain; topology-on remains an existing-boundary
comparator rather than a global organization veto.

## Long-run execution

- Finite-conduction n128 six-case array: run
  `20260917T191904Z-cf6c444-6d82ab`, Slurm `56126299`, at most two concurrent
  array tasks. Two tasks were running and four were queued at the first manager
  record.
- Mura mechanical-heterogeneity n64 continuation: run
  `20260917T192050Z-50dbde9-bf00af`, Slurm `56126506`, running toward 5% strain.
- A durable scoped manager records scheduler state every 300 s and fetches after
  each scoped bundle leaves active scheduler state. It runs in detached tmux
  session `v37_manager`; its state is external at
  `/Users/sdillon/HPC3/local-results/asb-drx-v37-manager/v37_manager_state.json`.
  When the conduction array releases its two slots, the same controller submits
  the preregistered n128 continued-loading front array (baseline,
  `availability_mid`, and `shape_n_low`) with one case active at a time. This
  follow-up does not wait for the independent Mura continuation.
  Retrieved conduction, Mura, and front bundles each invoke their scoped
  classifier automatically and write machine-readable decisions and plots
  under the manager's external `postprocessed` directory.

Earlier conduction submissions that failed before physics are retained in the
case manifest. Invalid QoS, shared array extraction, and missing archive Git
metadata were repaired without changing the registered scientific cases. They
are infrastructure evidence, not negative physics.
