# V64 decision report

V64 delivered a production error-controlled coupled interval map and advanced
the four-grain analytic-origin trajectory to `1.0300000000000003 us` at
engineering shear `0.05320000000000526`.  The terminal checkpoint is adaptive
interval 2585, SHA-256
`2bbf987b6e6dbbf603c518686b90812a91c821715874bd580a0367dbeb1b1667`,
under physical/numerical source `b579a1623361a93e68ae26aec27edbefc9a91a3d`.
The final two intervals were replayed from checksum-bound checkpoints after the
long worker was interrupted for closure; every recorded scalar state, load,
controller value, stress, and temperature matched the already observed
accepted intervals exactly.

The final stress is `3.97103031704408 GPa` and remains the whole-history peak,
so the trajectory is still hardening.  Mean temperature is `928.7949 K`,
temperature range is `75.1111 K`, and peak-minus-mean is `48.3403 K`.  The
plastic-power component has minor width `0.972066 um` and aspect ratio `1.148`.
There is no strict snapshot or persistent episode.  Matched accepted-time
temperature control and localization refinement certificates are also absent.
Strict ASB is therefore false at the attained horizon; this does not reject
ASB at longer exposure or another rate/temperature.

Prepared-boundary redistribution is active.  The initially lowest-density
grain 20 grows by `6.52058%` and fresh sweep is `3.52083e-22 m3`.  Its
equivalent contour displacement is `0.441198` of the n48 grid spacing.  The registered V58
existing-boundary criterion requires at least half a grid spacing as well as
one-percent growth and fresh sweep, so substantial existing-boundary DRX is
false by a narrow geometric margin.  Grain count remains four.  This test does
not exercise spontaneous grain birth or establish a recrystallized material
fraction.

The production controller recorded seven accuracy rejections and one
admissibility-failed trial over the complete lineage, with no rejected physical
front event.  Two widely separated production accuracy rejections on the final
source demonstrated private rollback, exact first-half reuse, and bounded
regrowth.  On the final long source segment, median accepted-interval wall time
was `11.583 s`; instrumented complete-map cost was 79.17% front, 10.47%
mechanics, and 10.36% endpoint observables.  No physical coefficient was
changed in V64.

The final configured regression passes `1009` tests in `280.67 s`.  Native-axis
frames, classification JSON, exact restart lineage, histories, selected
checkpoints, source, logs, and checksums are retained in the consolidated V64
delivery.  The post-one-microsecond extensions used the remaining compute
allocation without changing source, physics, or tolerances and terminated
normally.  An independent checkpoint-2577 replay reproduces the terminal
physical state, load, clocks, and ledgers bitwise.  Delivery is complete at the attained experimental scope; the full
scientific objective of a jointly qualified DRX/ASB response family remains
incomplete.

The fixed allocation ran from `2026-10-12T19:57:02Z` through its deadline at
`2026-10-13T07:57:02Z`; monitored closure ended at `2026-10-13T07:57:30Z` with
execution status `DEADLINE_REACHED`.  The definitive 85-file delivery is
`v64_consolidated_delivery_v5_definitive.zip`, SHA-256
`cedefcad33197f7bd709473a973754e43df2f314329f9eed6d5b6f91968698b7`.
