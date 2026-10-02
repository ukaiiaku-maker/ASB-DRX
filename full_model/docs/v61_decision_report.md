# V61 decision report: clean local-interface history and fixed-scale audit

## Decision

V61 advances the common production model, but does **not** qualify strict ASB.
The strongest positive result is a causal thermomechanical-localization response
in the n32 transition-history physical/control pair. It persists as a coupled
model effect, but its power structure collapses toward grid scale and fails the
registered width/persistence conjunction. A fresh, identical-history n32/n48
physical pair then demonstrates material mesh sensitivity at 1.675 microseconds.
The apparent thin band is therefore grid-selected at the tested resolution.

Prepared-boundary DRX remains demonstrated in its prior qualified scope. V61
does not qualify spontaneous grain birth, a converged ASB morphology, a
rate/temperature regime boundary, or material calibration.

## Production and evidence repairs

The solver source at `b457eaf1ba7436431e95e75ec0da5448cb4cd950`
contains the V61 repairs accumulated from the frozen V60 lineage:

- whole-history and operator-segment stress peaks are separately carried across
  restarts, with source and checkpoint provenance;
- sampling gaps use declared segment cadence and source-seam checks rather than
  the minimum separation anywhere in a concatenated record;
- candidate-associated source integrals are retained separately from the old
  independently-largest-component overlap;
- accepted-interval mechanical, front, conduction, bath, and storage terms are
  independently closed on the whole domain and a fixed Eulerian candidate;
- signed family slip rate is exposed, and spatial evidence uses physical axes;
- front mobility uses a resolved interface-weighted temperature with one joint
  donor-capacity transaction and complete scalar energy force; and
- incompatible material directions are rejected rather than silently summed.

The local-interface fixed point is damped and uniform temperature exactly
recovers the global operator. The pre-campaign V60 source remains immutable.

## Clean local-interface n32 physical/control history

The local-interface physical and all-Arrhenius-frozen n32 members were each
continued from their own exact 2.5 microsecond parents through 3.5
microseconds. Both have 40 unique 25 ns records under the V61 operator and all
three source seams are verified. The physical member has 420 accepted events
and zero rejections.

The pair demonstrates strong causal thermomechanical response: maximum local
matched excess is `54.184 K`, peak-minus-mean temperature is `283.233 K`,
same-operator softening reaches `33.935%`, and heat/power overlap reaches one at
some states. It nevertheless has no conjunctive episode: resolved width passes
at `0/40` states, the maximum width is only `0.6165` interface widths, and the
late component degenerates toward a one-cell structure. Its honest
classification is **causal but unresolved/nonpersistent thermomechanical
localization**, not strict ASB.

## Fresh fixed-scale n32/n48 result

Both physical calculations start from the same analytic four-grain
initialization and use a 5 micrometer periodic box, 312.5 nm interface width,
25 ns macro interval, 900 K initial temperature, 4e4/s loading, identical
thermodynamics and local-interface/local-realized-event operators. Each reaches
1.675 microseconds with 201 accepted events and zero rejections.

| quantity at the common endpoint | n32 | n48 |
| --- | ---: | ---: |
| post-front stress | 3.90106 GPa | 4.05033 GPa |
| operator peak time | 1.225 us | 1.525 us |
| final operator softening | 2.623% | 0.649% |
| mean temperature | 991.290 K | 985.882 K |
| temperature contrast | 207.858 K | 179.540 K |
| power minor width | 264.294 nm | 153.410 nm |
| width in cells | 1.691 | 1.473 |
| width/interface width | 0.846 | 0.491 |
| aspect ratio | 2.121 | 6.526 |
| heat/power overlap | 0.750 | 0.000 |

The n48 run was terminated only after checkpoint 67 when the next uncommitted
step entered prolonged constitutive substepping. The retained prefix and its
checkpoint are valid; no n48 control or late softening horizon exists. The
large changes in peak time, temperature, width, aspect and source association
reject a spatial-convergence claim. V61 therefore issues no refinement
certificate and does not launch n64.

The clean n32 all-Arrhenius-frozen member also reaches the same horizon from
the same analytic origin. Its intervention certificate passes at all 67 common
states. The physical member's matched component excess grows to `24.194 K`,
but its softening is only `2.623%`; the early clean history is therefore causal
but episode-negative. The much stronger transition-history response develops
later and cannot repair the early mesh sensitivity.

## Bounded temperature discriminator

An otherwise identical fresh n32 physical/frozen pair at 1000 K also reaches
1.675 microseconds with 201 accepted events and no rejection. Its intervention
certificate passes at every common state. Relative to 900 K at equal time and
strain, the physical peak falls from `4.0061` to `3.9191 GPa`, peak time moves
from `1.225` to `1.275 us`, endpoint softening rises from `2.623%` to `3.837%`,
and maximum matched component excess rises from `24.194` to `28.824 K`.
Neither condition forms a registered episode and both endpoint widths remain
underresolved. This is a resolved temperature trend in the fixed model, not a
demonstrated rate/temperature-selected ASB boundary.

## Independent interval heat balance

For n48 step 66 to 67, direct whole-domain thermal storage is
`1.1751625160488e-13 J`. Independent mechanical/reaction heat is
`5.1683519028786e-14 J` and front heat is `6.5832732577124e-14 J`;
conduction integrates to numerical zero globally. The whole-domain relative
residual is `7.97e-13`, and the fixed 26-cell candidate residual is `7.75e-13`.
The saved front-source integral exactly equals independently computed
pressure-times-realized-volume heat.

The candidate receives `1.5715e-15 J` mechanical heat and `2.0496e-15 J`
front heat while losing `6.1603e-16 J` by conduction, closing to
`3.0051e-15 J` direct storage. Thus the low independently-largest overlap is
not absence of candidate heating: the candidate has nonzero direct source
support, while larger remote components dominate the historical overlap
diagnostic. The corresponding n32 interval closes to `1.67e-13` globally and
`1.45e-12` on its fixed candidate.

## Status by scientific responsibility

| responsibility | V61 status |
| --- | --- |
| production correctness | passed at final regression boundary |
| conservative front heat | passed; pre-normalization mismatch zero |
| physical spatial-source adequacy | active and independently balanced |
| prepared-boundary DRX | preserved from qualified prior scope |
| causal thermal feedback | demonstrated on matched n32 pair |
| strict localization/ASB | not established |
| morphology | grid-selected and underresolved |
| one-microsecond physical persistence | failed registered conjunction |
| spatial refinement | failed / nonconverged at common horizon |
| temporal refinement | 25 ns accepted cadence; no separate dt certificate |
| rate/temperature regime comparison | bounded 900/1000 K trend; no selected ASB regime |
| spontaneous grain birth | unqualified |

## Evidence and continuation

The machine-readable manifest records every source, checkpoint and artifact
hash. Both grids have 12-frame physical-axis GIFs, original-frame ZIP archives
and selected endpoint PNGs. Exact continuation and reproduction commands are
in `full_model/docs/v61_restart_commands.md`. The next calculation should not extend the same
underresolved n32 hotspot. It should first make the common-history n48
mechanics affordable without changing constitutive exposure, then obtain a
matched n48 frozen control and a wider resolved physical structure (or reject
this parameterization) before any n64 or regime claim.
