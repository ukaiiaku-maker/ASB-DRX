# Taylor DDD single-glider context audit

Status: located and reviewed 2026-08-28. These simulations constrain mechanism interpretation and test mathematical reductions; they do not parameterize the new model.

## Source discovery

Prior Codex attachments identify the source repository as `https://github.com/ukaiiaku-maker/Taylor_DDD.git`, with local worktrees `/Users/sdillon/Taylor_DDD` and `/Users/sdillon/Taylor_DDD_arrhenius_native`. Both contain user work, so they were inspected read-only and left unchanged.

The reduced-model worktree is on `codex/poisson-peak-diagnostic-campaign` at `dc9007c` with existing modifications and untracked analysis files. The native worktree is on `arrhenius-exadis-strain-hardening` at `78e1535`, eleven commits ahead of its remote and with unrelated existing changes. Scientific provenance is therefore assigned to immutable commits and hashed result files, not merely to current worktree state.

## Native ExaDiS controlled single-glider gate

Commit `fb7610bdcfd3f32b4290983d907a06a581329f64` is titled `pass native ExaDiS single-glider Taylor gate`. It adds the controlled driver, strict validator, native constant-line-tension patch, launcher, and report. The committed driver SHA-256 is `89d4f7ea45915cef0aa68f1e8a67a6a996e01686f11dea1685e46526dc600468`; validator SHA-256 is `22a96dc95c7b469b0d54ca46978af5257759f265d1c3bbc35c1c6a57ac493163`.

The geometry has one periodic mobile glider and an independently varied square grid of persistent fixed-forest contacts. Contacts have stable identity, capture/release generations, accumulated hazard, neighboring load-bearing arm lengths, native force work, and actual swept plastic strain. The high-barrier test was declared in advance and not fitted to the result.

The completed four-condition validation reports:

| Forest density, m^-2 | Tail stress, MPa | Captures/releases | Median `L_eff`, m |
|---:|---:|---:|---:|
| `1.3891e13` | 11.263 | 10 / 10 | `1.8972e-7` |
| `5.5565e13` | 37.137 | 28 / 24 | `9.4860e-8` |
| `2.2226e14` | 70.800 | 80 / 78 | `4.7430e-8` |
| `8.8905e14` | 130.364 | 201 / 192 | `2.3715e-8` |

The full-range log stress--log forest-density exponent is `0.576481`; the final decade-plus subset gives `0.452905`. The 64-fold density range therefore passes the declared 0.4--0.6 Taylor interval. Mobile density is invariant, all networks are sane, all cases contain accepted events, and the transparency peak lies outside the tested range. The validation JSON SHA-256 is `35a739315822c6310443fda6374233fff3357c757f4a590750fc0329dea57ca5`.

A separate 1100 K regression records 11 captures and 11 releases with zero reported divergence from the reduced v17 force, effective-stress, arm-length, barrier, probability, and event fields. Its summary SHA-256 is `b41bfa0a93ef224a592f774d9ca4b2c5228a3acd8abb5c2a146e5e1b9b3bfd33`.

This is strong evidence that persistent load-bearing contact spacing and force work can recover Taylor scaling in a controlled geometry. It is not a material calibration: the named barrier is a test fixture, the geometry is deliberately reduced, and the tested range does not contain transparency.

## Complete reduced full-glider campaigns

`/Users/sdillon/Taylor_DDD/results` contains complete continuous-contact single-glider campaigns with per-run parameters, history, contact crossing events, depinning counts, and summaries. Key context artifacts are:

| Artifact | SHA-256 | Use |
|---|---|---|
| `full_glider_taylor_fix_20260814/FINAL_FULL_GLIDER_TAYLOR_FIX_CONCLUSION.md` | `9337b81df749c6e4b59b3e065bb2ae327cb5bb41044a8e945d83b0cb98ebb4ff` | Records corrected Taylor exponent 0.562, `L_eff` exponent -0.514, and failure of the shifted analytical peak because geometry/capacity, multi-hit, transparent, or Peierls regimes preempt it. |
| `full_glider_temperature_transition_screen_20260817/temperature_sweep_mechanics_plus_poisson.csv` | `6120ef7f4e5bc76f97676ef608baf85ddd16d427eb8e5912512e61ae01a0510b` | Joint mechanics/event diagnostics versus density and temperature. |
| `full_glider_temperature_transition_refine_20260817/temperature_sweep_mechanical_summary_by_T.csv` | `64753021e8344a1c66665f16a3fdaeddb9525a9db7d45f1d92ee6aa7087307d2` | Refined load-bearing-to-transparent screening. |
| `full_glider_expfloor_HT050_F020_overnight_peakmap_20260817/poisson_likeness_temperature_summary.json` | `a99088101a5ceabd67d5cab379f44afc1e4eacedbecdb04ca5d29022c797cb4c` | Event-level EXP-floor temperature/density summaries over multiple seeds. |
| `.../analysis_outputs_v2/peak_report.csv` | `c18fdd9fcfa8f633ebc55e3bbfdd5fe04f56e0f4ad9be6adbe5b51967d31089e` | Reports no interior DDD peak for 850--1050 K; every curve remains increasing through `3e16 m^-2`. |
| `.../analysis_outputs_v2/by_condition_summary.csv` | `4b1e0979259b94a845cc9df297330727e5306a00341acaa01ba7c27a8db47a1e` | Four-seed condition aggregates and analytical comparisons. |

Examples in the temperature screen show approximately Poisson-like low-density histories and multi-hit/correlated higher-density histories under the campaign's diagnostics. The later peak map also shows that simultaneous/multi-hit fractions and aggregate event intensity grow while the DDD strength remains monotone, so the analytical independent-site peak can be preempted by contact capacity and collective event structure.

These classifications require caution. Several summaries use a one-step correlation window, simultaneous depinning counts, fixed Fano windows, and model-internal hazards. They do not by themselves demonstrate a universal physical transition or determine continuum parameters. They are nevertheless much stronger structural evidence than the previously located scaffold avalanche code because the event histories arise from a completed explicit single-glider/contact simulation.

## Consequence for the collective mathematics

The event files make the proposed collective hypothesis falsifiable without using DDD as a parameter source:

1. A node is a persistent contact with a residence generation, neighboring arm lengths, force-work barrier, accumulated hazard, and release/reset event.
2. An event changes the neighboring load-bearing geometry and forces. The corresponding before/after resolved-force increment defines the sign and spatial support of `Delta tau_ij(t)` in the branching matrix.
3. Multiple releases within a relaxation/contact-rearm interval define observed multi-hit clusters. Their distribution can test, but not fit production values for, the shot-noise first-passage reduction.
4. Density enters through the evolving contact graph, `L_eff` distribution, coordination, occupancy, residence/rearm time, and elastic stress-transfer kernel. It is not an independent switch variable.
5. The independent EXP-floor model remains the zero-transfer limit. The collective extension is retained only if it explains event clustering and macroscopic response that the independent limit cannot.

The next analysis should reconstruct contact-event parentage from force changes and persistent contact IDs, compare the observed triggered-cluster distribution with the branching prediction, and test whether a scalar spectral radius is adequate or a state-dependent operator is required. Any such analysis is structural validation only; physical parameters still come from governing equations and the selected material dataset.
