# V37 finite-amplitude coupling and response families

## Decision status

V37 has completed its current-source front reconciliation, bounded kinetic
screen, ordered-reservoir audit, short Mura organization controls, and thermal
transport design. Two production continuations are live: a six-case n128
finite-conduction array and an n64 one-grain Mura continuation to 5% strain.
Consequently this is an attributable **active checkpoint**, not the final V37
scientific classification.

No current evidence supports independently recognized DRX, a persistent
orientation-compatible LAGB, strict ASB, coupled DRX/ASB, or material
calibration. V36 evidence remains frozen.

## Front and recurrent-state decision

The scalar-named historical equality was not complete state equality. It retained
different inactive wake history and came from a different source. In the current
n128 source, full material exchange gives exactly antisymmetric kinetic velocity
and accepted motion. Proposal-only reversal is rejected before publication and
leaves geometry, processed line, heat, and energy unchanged.

One accepted current-source interval has direct contour displacement
`3.69684016154892e-13 m`, or `1.4787360646195681e-5` cells and
`9.242100403872301e-7` interface widths. It transforms
`1.1735249408820892e-27 m3` and processes `1.1261320234643155e-12 m` of line.
The complete-energy first-law residual is `-1.14e-27 J`. This is existing-boundary
stored-energy migration at sub-finite amplitude—not DRX.

Explicit owner equality before Mura leaves a smaller residual response after the
recurrent update: `5.409715679860483e-14 m`. Defect and boundary endpoint
increments match; finite diffuse-slab phase-gradient/local terms and recurrent
thermal history differ. The residual is therefore named finite-interface/history
response rather than a symmetry failure.

The preregistered 17-case rate family remains a rate screen, not a trajectory or
calibration. It estimates quarter-width times from 1.352 s for the unchanged
baseline to 0.04409 s for the fastest case. The next full-solver neighborhood is
baseline, `availability_mid`, and `shape_n_low`; it will run when an authorized
compute slot becomes free. Bounded moment projection consumes 72.4% of the
profiled control time and is the main acceleration target.

## Mura organization decision

The timestep-sensitive ordered reservoirs are absolutely small: the wall-local
line error is 0.1504%, and their energy difference is 1.63e-10 J/m, about
1.04e-6 of total defect energy. This permits bulk/thermal progress but does not
qualify a wall claim.

At 0.02% additional strain, homogeneous and broadband-noise controls capture no
wall line. Mechanical heterogeneity captures ordered content, with topology-off
ordered fraction 0.1404 and polarization 0.7457; topology-on gives 0.4081 and
0.5783. The orientation span is only 6.57e-7 degrees. Classification is
`CAPTURED_ORDERED_LINE_WITHOUT_ORIENTATION_WALL`. All hard invariants pass, and
no phase or grain state exists. The running n64 topology-off continuation is the
selected discriminator; no LAGB, phase support, nucleation, or DRX is claimed.

## Thermal/localization decision

The physical branch fixes positive conductivity at `0.15 W m^-1 K^-1` and
volumetric heat capacity at `3.8e6 J m^-3 K^-1`. Diffusion times are 2.28 us
across the 0.30 um process zone, 14.25 us across the 0.75 um heterogeneity, and
2.533 ms across the 10 um domain, compared with an 8.337 us loading horizon.

Manufactured profiles qualify the new band-normal metric: it recovers the
0.25 um narrow-profile scale and distinguishes broad support. The frozen V36
finite-conduction temperature field is broad (`Tmax-Tmean = 37.02 K`, temperature
IPR fraction 0.99987); zero conductivity remains only a frozen limiting ablation.
The running array varies rate, initial temperature, and heterogeneity radius one
group at a time. Strict ASB thresholds are unchanged.

## Integration and next automatic decisions

An integrated continued-loading common-state trajectory is deliberately not yet
published: the selected Mura and thermal production results are still active.
The controller will fetch and classify both bundles. Valid conduction results
will select strongest concentration, broad-heating negative, and intermediate
cases for refinement. The Mura result will be judged at physical strain exposure
using signed populations, Nye consistency, orientation, topology, and checkpoint
invariants. A freed compute slot then triggers the three-case front trajectory
neighborhood. Only after those states are attributable will one controlled
continued-loading integrated example be eligible.

## Provenance and regressions

- V36 completion: `e1795173471ea3deea1f44fe68e4be2de384fab0`.
- V36 recurrent physics: `890cb8906a9772d8bd5c5eb43164ecd44ad2720f`.
- V36 n192 submission/configuration: `e625a60dc01a119a7c041af3efdb771a30d01246`, job `56099919`.
- V37 live conduction: run `20260917T191904Z-cf6c444-6d82ab`, job `56126299`.
- V37 live Mura: run `20260917T192050Z-50dbde9-bf00af`, job `56126506`.
- Current execution and artifact identities are recorded in
  `v37_case_manifest.json`; invalid infrastructure attempts remain visible.

The merged repository regression suite is run after integration. Long-run
completion and postprocessing will update this report rather than rewriting
frozen V36 records.
