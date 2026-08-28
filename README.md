# ASB-DRX

Reduced 2D variational simulation code for adiabatic shear banding (ASB) and dynamic recrystallization (DRX) model development.

Current working baseline is v34, which builds on the v32 GB-transmission/process-zone ASB branch and adds candidate-nucleus bookkeeping for DRX.

## Current status

- `drx_var_v34_candidate_drx_asb_sweep.py` is the active driver.
- `run_v34_candidate_drx_asb.sh` runs one case and supports `BRANCH=asb_only`, `BRANCH=drx_isothermal`, and `BRANCH=coupled`.
- `run_v34_rate_sweep_equivstrain.sh` runs equivalent-strain rate/seed sweeps.
- `summarize_v34_drx_asb_rate_sweep.py` summarizes branch/rate sweeps.
- `plot_v32_gb_transmission_overlay.py` provides GB transmission and residual-Burgers overlay diagnostics for v32/v34-style output.

## Scientific interpretation

v32/v34 can produce energy-budgeted, finite-width, GB-assisted thermoplastic localization with ASB-like scalar signatures. The DRX branch is not yet validated. v34 prevents fake grain-ID explosions by separating candidate embryos from promoted persistent grains, but candidate promotion/growth physics still needs work.

See `docs/status_handoff.md` for the current technical handoff and Codex plan.
