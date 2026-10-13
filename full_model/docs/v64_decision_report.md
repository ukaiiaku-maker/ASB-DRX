# V64 decision report

V64 delivered a production error-controlled coupled interval map and advanced
the four-grain analytic-origin trajectory to `1.020513582229403 us` at
engineering shear `0.052820543289181024`.  The terminal checkpoint is adaptive
interval 2457, SHA-256
`1a66604c813bff2d044910c12daa66bf3fbfb8dc7da3fd0b33ae4c3d166cf165`,
under physical/numerical source `b579a1623361a93e68ae26aec27edbefc9a91a3d`.
The final two intervals were replayed from checksum-bound checkpoints after the
long worker was interrupted for closure; every recorded scalar state, load,
controller value, stress, and temperature matched the already observed
accepted intervals exactly.

The final stress is `3.9681768194182334 GPa` and remains the whole-history peak,
so the trajectory is still hardening.  Mean temperature is `927.9781 K`,
temperature range is `73.1575 K`, and peak-minus-mean is `47.1811 K`.  The
plastic-power component has minor width `0.940770 um` and aspect ratio `1.173`.
There is no strict snapshot or persistent episode.  Matched accepted-time
temperature control and localization refinement certificates are also absent.
Strict ASB is therefore false at the attained horizon; this does not reject
ASB at longer exposure or another rate/temperature.

Prepared-boundary redistribution is active.  The initially lowest-density
grain 20 grows by `6.70537%` and fresh sweep is `3.48307e-22 m3`.  Its
equivalent contour displacement is `0.446633` of the n48 grid spacing.  The registered V58
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
delivery.  The post-one-microsecond extension used the remaining compute
allocation without changing source, physics, or tolerances and terminated
normally.  Delivery is complete at the attained experimental scope; the full
scientific objective of a jointly qualified DRX/ASB response family remains
incomplete.
