# V32 live scientific log

## 2026-09-16 restart audit

- Local and remote canonical branch both resolve to V31 evidence checkpoint `8df28e8e3b47eea9f04c154db13f2df468c2c096`; the canonical worktree is clean.
- Immutable V31 calculation source remains `c643afe00b4ea6f7f30e021675d7648ecdb6422c`.
- V30/V31 raw evidence and decision records are frozen.
- Unrelated Slurm job `55950433` is running and remains untouched; no V32 Slurm job exists at restart.
- Front continuation audit: six normal completions, one conservative `FRONT_COMPONENT_SPLIT` terminal, two running 128-square cases, and two near-equal 128-square cases not yet started. Exact-equal and mobility-off controls will not be replayed.
- The 64-square positive near-equal terminal occurs at step 135 after 135 accepted transactions; its cumulative maximum line-closure magnitude is `1.21e-19 m` and signed-Burgers closure is exact. The rejected topology event itself commits no heat, line, or signed content.
- ASB anchor audit: all four 128-square cases are live at step 2600 of 5000 with 27 restart files per case after about 2.4 hours. The anchor is therefore adequately checkpointed and is projected to close within the local campaign horizon.
- Three independent scientific loops are active: topology-event/front kinetics, Mura work-budget repair, and physical-ASB classification.

## 2026-09-16 autonomous branch decisions

- Front: all ten frozen-source V31 continuation cases closed. The 64-square
  positive near-equal split was a `0.0166`-cell-squared subcell island and is
  filtered conservatively by the V32 topology repair. Continued replay then
  reaches unresolved pair-identity loss. Both 128-square near-equal signs stop
  symmetrically at step 137 with resolved forbidden islands, while favorable,
  reversed, and label-swapped controls complete. Front fragmentation is a
  mechanistic negative and no long-front promotion is authorized.
- Mura: the first-rejection audit identified one negative-work Burgers family.
  Per-family complementarity plus fixed-strain elastic/defect-energy
  backtracking repaired that hard failure without projection. The first 64
  continuation later exposed a separate exact-zero-velocity roundoff defect:
  an event scale of zero manufactured `8.5 m^-2-cells` of line. That trajectory
  is quarantined as numerical, the exact identity endpoint is repaired with
  531 regression tests passing, and a provenance-separated continuation from
  the preserved 3.841278% checkpoint is running from pushed source `e7aa16e`.
- ASB: the heterogeneous adiabatic anchor reached the declared thermal-validity
  terminal at step 3272 with strong heating but a broad response, so it is not
  strict ASB. The other three 128-square controls continue. A two-step and
  exact-restart adaptive preflight passed all channel, first-law, source, and
  checksum checks. The eight-case Tier-1 900/1300 K and 1e4/1e5 s-1 screen is
  running locally with maximum concurrency two from immutable source `fb21dbc`;
  Tier-2 seeds remain conditional on a strict raw candidate.
- Integration: a compact 32-square positive-drive combined-mode reproduction
  reported 608 MPa front pressure and 20.7 m/s net kinetic velocity but zero
  proposed sweep because common-Mura mode silently froze KWC phase evolution.
  The later front transaction also lacks ownership of signed forest/wall,
  beta/Nye/alignment, and junction state. The nominal flag combination is not
  an integrated model; production is being changed to fail fast until one
  atomic common-state adapter exists.

## 2026-09-16 V33 reconciliation and decisions

- Restart audit resolved local and remote canonical source to `e5f6232` at
  campaign start. V31/V32 source identities remain immutable. HPC3 queue audit
  failed at DNS resolution and is recorded as unverified rather than inferred.
- Common state: source `5b3b5b6`, evidence `327c730`. I0 and compact I1 pass in
  the actual driver; 95 authoritative fields are bitwise restart-identical.
  I2 complete front-energy acceptance and I3 continuation remain open.
- Mura: full-affinity audit commit `e8d9331` classifies the limiter as a
  load-conditioned physical constraint, not a second loading-clock scale or
  demonstrated saturation. The immutable `e7aa16e` process remains live.
- Front: source/tools `e2fcd65`, evidence `de46924`. The rejected near-equal
  trial is an unresolved Allen-Cahn zero-level filament; the accepted physical
  state remains bitwise pretrial. Classification:
  `GRID_SCALE_PHASE_REPRESENTATION_BREAKDOWN`.
- ASB: source/tools `833ad72`, reconciliation `5ec3eb5`. Anchor and 900 K
  no-conduction cases are validity-limited negatives. The finite-bath Tier-1
  partner remains live. Prepared causal ablations are not reported as run.
