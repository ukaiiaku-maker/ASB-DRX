# V63 execution record

V63 was adopted at `2026-10-09T17:26:51Z`; its fixed local deadline is
`2026-10-10T05:26:51Z`. The controlling directive and assessment have SHA-256
hashes `2666699c8453ab14aa6f2be9985540dbe748d194a9b744246c99663c3a1e3852`
and `55659a129ffe89c512a29b94ec673298d7c375760b695b14d7b1f276ef550e6b`.

The startup audit found the pushed V62 documentation head
`58ee7de5c9c52c65990250c72f7e3e75d9a25241`, clean immutable numerical source
`1c0cb4ca0e8a9a30b0c1a320817a0b35b4d14499`, and checksum-valid n64 step-210
parent `f44f2e437845688a70e6eb2a47edf972ead192d580c65e9a6602711928175ff0`.
No related campaign solver was active. The outer V62 package and source archive
also matched their reported hashes. Available disk was 7.2 GiB; sparse
five-step n64 restart states fit safely without deleting historical evidence.

The unchanged-source n64 continuation started in
`/Users/sdillon/HPC3/campaigns/asb-drx-v63-20261009/n64_physical_same_source_from_step210_to320_1c0cb4c`.
Its requested milestones are steps 240, 264, and 320. Only committed
checkpoints are authoritative.

The continuation reached checksum-readable step 215 at 1.34375 microseconds
with zero rejected front subintervals and no increase in the inherited maximum
energy-closure bound. Phase-sensitive common-band comparisons now cover steps
185, 200, 210, and 215. On band 15 the plastic-power discrepancy rises from
10.03% to 56.83%, 78.24%, and 89.63%; the corresponding temperature errors are
1.04%, 1.98%, 2.25%, and 2.50%. At step 215 the exact reconstructed-Nye error
is 39.51%. This rejects the earlier interpretation of radial spectral-shape
agreement as local-field convergence.

A fixed physical candidate, tracked by component overlap rather than by
reselecting the largest component, has only 0.31 n48/n64 mask overlap at step
210 and 0.46 at step 215. Fixed-axis transverse-profile errors are 83% and 90%.
The step-215 n64 candidate FWHM is only 1.5 native cells and is explicitly
unresolved.

Matched-coordinate reevaluation of the unchanged EXP-floor map attributes
about 88% of the recomputed step-215 rate discrepancy to replacement of the
raw resolved stress. Resistance removes about 4.8%, temperature 0.5%, and
chemical backstress is negligible. The raw-stress directional linearization
has a 60% delta-prediction error, and the joint linearization has a 63% error,
so the finite discrepancy is already strongly nonlinear. Substituting common
stress or orientation separately changes the rate strongly but does not
reproduce the refined member; their spatial correlation matters.

Diagnostic source `dbd23bf78fe5103335d1bdcc24523286ae13a8ae` is frozen in
`/Users/sdillon/HPC3/worktrees/asb-drx-v63-numerical-dbd23bf`. It adds exact
temporal-fork provenance, phase-sensitive and EXP-floor audits, exact DRX
event-energy replay, fixed-candidate profiles, and atomic physical-cadence
scalar diagnostics. It does not change the physical kernels used by the live
`1c0cb4c` continuation.

The exact n64 step-240 milestone reached 1.500 microseconds with SHA-256
`90b2c19c3366e744634f84709a363299161e2852a243c30dadea269f4999186c`.
All physical events remained accepted. Its band-15 discrepancies are 49.95%
for instantaneous power, 92.50% for owner speed, 48.28% for signed density,
23.82% for plastic distortion, 49.08% for exact Nye, and 3.93% for
temperature. Support and orientation differ by only about 0.5%. The tracked
candidate has 0.32 cross-grid mask overlap and 97.78% profile error. Thus a
temporary reduction in the instantaneous power norm does not constitute local
field convergence.

The same-source continuation subsequently completed exact step 320 at 2.0
microseconds.  Its terminal checkpoint SHA-256 is
`ac4ac1ee6e7b01cd2d95fc4d5e35d694604c4e886e15111c1e01e297ce21a3ea`.
All 832 cumulative front intervals were accepted; maximum complete-energy
closure is 0.1903%.  Endpoint stress is 2.35254 GPa, mean temperature is
1034.07 K, and temperature range is 283.35 K.  The final five-step block cost
838.6 seconds, or 167.7 seconds per caller interval.

The continuously tracked late candidate spans saved steps 265--320, only
343.75 ns.  At step 320 its cross-grid mask Jaccard index is 0.668 and its
fixed-axis profile error is 45.7%.  The n48/n64 FWHM values are 1.016/0.742
micrometres despite both occupying about 9.5 native cells.  Phase-sensitive
band-15 discrepancies are 8.24% for temperature, 61.51% for exact Nye, and
73.66% for instantaneous power.  Late morphology is not converged.

Fresh-state caller splitting was tested at parent steps 185, 210, and 275.
At 185 and 210, 3.125 versus 1.5625 ns is below 5% for signed density, exact
Nye, temperature, and power.  At 275 that comparison gives 5.56% signed
density error; the added 1.5625 versus 0.78125 ns fork reduces signed-density,
Nye, temperature, and power increment errors to 2.57%, 2.34%, 0.75%, and
2.40%.  Every requested mechanics and front clock closes.  Diagnostic commit
`ad6171b87a0ec5e3ef588be2484934356514443b` adds this eighth-step transition
without changing a production physical kernel.

Exact replay of the accepted n48 step-268--269 front transaction gives a
2.2966e-14 J cold Helmholtz decrease: defect storage decreases by 1.5641e-14 J,
recoverable elastic energy by 7.4734e-15 J, and boundary excess increases by
1.4783e-16 J.  Grain 40 grows even though its own assigned defect storage
increases, because the constrained multigrain transaction is downhill as a
whole.  A separate n64 step-264--265 heat audit closes to 2.62e-12 relative
over the domain and 1.69e-12 over the fixed candidate; saved front heat exactly
matches independent pressure-times-realized-volume heat.

After the n64 decision, a fixed-parameter 80,000/s experiment was selected.
The n32 physical/frozen pair completed to 1.0 microsecond with zero rejection.
The physical member is slightly stronger and hotter on average but has lower
temperature contrast and lower power concentration than the frozen member.
No strict episode occurs.  The n48 physical member then completed the same
protocol with zero rejection and 0.3744% maximum energy closure; its matched
frozen control and outcome-neutral pair postprocessing are recorded in the
final V63 manifest.

The original n32 pair was finally continued in checksum-bound segments through
step 320 (2.0 microseconds).  Physical and frozen checkpoint hashes are
`5da317ca04f697179ebdb9c3bc03f04acab0947dc7926588694859fe208eee4f` and
`063fdb83a666796ae6fd681372f764d7373b13d19cdb1b637b47470bbdf8460d`.
Both have zero rejected events.  Full three-segment matched reduction passes
the intervention certificate and finds 67.174 K maximum matched component
excess and 70.91% maximum softening, but zero conjunctive snapshots and zero
strict episodes.  The classifier artifact SHA-256 is
`0ce01d78b2a19037da44cbe26d2d5c61d5246ff60225e0a6502de2e41388ff92`.
Because n32 is already a demonstrated spatial outlier, this extension cannot
override the n48/n64 nonconvergence finding.

All campaign solvers were stopped naturally after committed endpoints.  The
canonical regression boundary is 987 passing tests in 673.72 seconds.
